/* The run loop, and the status bar it keeps up to date. */

import { pumpAudio } from './audio';
import { drainDiagnostics, drainTrace, logTrace, traceEnabled } from './debug';
import { countEmulation, presentIfNewFrame, renderFb } from './display';
import { $ } from './dom';
import { formatBytes } from './format';
import { bootDetail, endLoad } from './loading';
import { log, logBlock, type LogClass } from './log';
import { call, readLastError } from './rpc';
import { saveFlush } from './saves';
import { sdFlush } from './sdcard';
import { loaded, panelOpen, setPanel, setState } from './shell';
import { holdWakeLock, releaseWakeLock } from './wakelock';

// Run in worker slices so the page can paint and input can reach the emulator
// between them. There is no overall step budget - hbmenu never halts - so the
// loop is driven by the pause flag and by faults.
//
// The slice length *is* the input sampling period: the worker is single
// threaded, so a `set_input` posted mid-slice sits in its queue until
// `switch_run` returns. At the ~23M steps/s the wasm build manages, the old
// 5,000,000 was a 240ms slice, and every keypress waited that long before the
// guest could possibly see it. Slice size costs the interpreter nothing
// (`Cpu::run` is a bare loop with no per-call setup - measured flat from 100k
// to 5M steps), only the round trips below, so this buys ~5x lower input
// latency for ~6% of throughput.
const RUN_SLICE = 1_000_000;
// Slices between panel refreshes. `updatePc`/`drainOutput`/`drainDiagnostics`/
// `sdFlush` are eight postMessage round trips of debug-panel text that nothing
// time-critical reads, so running them once per slice would spend more of the
// budget on chatter than the shorter slice saves.
const HOUSEKEEPING_EVERY = 8;
// Tracing prints a line per instruction, so a full slice would be a megabyte
// of log nobody can read.
const TRACE_SLICE = 5000;

let running = false;
let pauseRequested = false;
// Set when the session itself is going away rather than merely stopping. A
// pause is polite - it lets the slice finish and tidies up after it - and this
// is not, because everything the tidying would read is about to be freed.
let aborted = false;

const PLAY_GLYPH = '▶';
const PAUSE_GLYPH = '❙❙';

function setRunButton(isRunning: boolean): void {
  $('run-glyph').textContent = isRunning ? PAUSE_GLYPH : PLAY_GLYPH;
  $('run-label').textContent = isRunning ? 'Pause' : 'Run';
}

// Both transport actions are reachable from the keyboard as well as from the
// two buttons `setState` disables, so the refusal lives here rather than on
// the buttons alone.
function nothingLoaded(): boolean {
  if (loaded()) return false;
  log('Nothing is loaded - open a .nro, .elf, .nsp, .xci or .nca to boot one.', 'dim');
  return true;
}

export async function run(): Promise<void> {
  if (running) { pauseRequested = true; return; }
  if (nothingLoaded()) return;
  running = true;
  pauseRequested = false;
  aborted = false;
  setRunButton(true);
  setState('running');
  // A guest that runs for minutes without a keypress is a page the browser
  // would otherwise let the screen sleep on.
  holdWakeLock();
  const slice = traceEnabled() ? TRACE_SLICE : RUN_SLICE;
  let steps = 0;
  let tick = 0;
  try {
    for (;;) {
      const sliceAt = performance.now();
      steps = await call('run', slice);
      countEmulation(performance.now() - sliceAt);
      // Reset does not wait for the slice already in flight - it is about to
      // be thrown away - so by the time one returns the session may be gone.
      // Every call below reads it, so the loop leaves rather than asking.
      if (aborted) return;
      // Yield so the UI repaints and any queued input is processed.
      await new Promise((r) => setTimeout(r, 0));
      if (aborted) return;
      // `Cpu::run` only stops short of its budget when the machine halted, so
      // a short slice means this run is over - no separate `halted` round trip.
      const done = steps < 0 || steps < slice;
      // Audio has to track the guest or the stream gaps; the panel does not.
      await pumpAudio();
      if (done || ++tick % HOUSEKEEPING_EVERY === 0) {
        await updatePc();
        await drainOutput();
        await drainDiagnostics();
        await sdFlush();
        await saveFlush();
      }
      await presentIfNewFrame();
      if (done) break;
      if (pauseRequested) {
        // Whatever the guest was about to present, it is not going to now: a
        // paused machine is not a loading one.
        endLoad();
        setState('paused');
        await renderFb();
        return;
      }
    }
  } catch (err) {
    // A reset landing between two of the calls above is the expected way for
    // one of them to fail, and it has already put the page where it wants it.
    // Anything else is a real failure of the loop and has to be said out loud.
    if (!aborted) {
      setState('fault');
      // `RuntimeError: unreachable` is what a Rust panic looks like from here:
      // the release profile aborts on panic, and an abort on wasm is a trap
      // with nothing in it. The panic hook caught the real message on the way
      // down, and `switch_last_error` hands it back without needing a live
      // session -- so ask, rather than report the trap and lose the one line
      // that says what happened.
      let why = (err as Error).message;
      try {
        const captured = await readLastError();
        if (captured) why += ' - ' + captured;
      } catch {
        // The module may be too far gone to answer. The trap is still worth
        // saying on its own.
      }
      log('The run loop stopped: ' + why, 'err');
      // A trap does not destroy linear memory -- reading the message back is
      // already proof of that -- so the context is still there to be had, and
      // this is the last moment anyone can have it. A panic reported as one
      // line is a panic that has to be reproduced before it can be looked at.
      await reportPanicContext();
    }
    return;
  } finally {
    running = false;
    setRunButton(false);
    releaseWakeLock();
  }
  await finishRun(steps);
}

/** After a trap: everything about where the emulator was when it stopped.
 *
 *  Each of these can fail on its own -- the module is, by definition, in a
 *  state nobody designed -- so each is asked for separately and a refusal
 *  costs only that one piece. */
async function reportPanicContext(): Promise<void> {
  const parts: [string, () => Promise<string>][] = [
    ['registers', () => call('dump_regs')],
    ['threads', () => call('thread_dump')],
    ['trace', () => drainTrace()],
  ];
  for (const [what, ask] of parts) {
    try {
      const text = await ask();
      if (!text) continue;
      if (what === 'trace') logTrace(text);
      else logBlock(text, 'dim');
    } catch {
      log(`The module could not be asked for its ${what}.`, 'dim');
    }
  }
  log('Take a crash report from the debug panel before resetting - a reset is what loses this.',
    'warn');
}

/** Stop the loop because the session itself is going away (Reset). Pausing
 *  politely would be waiting for a slice that is about to be freed. */
export function abortRun(): void {
  aborted = true;
  pauseRequested = true;
  running = false;
  setRunButton(false);
}

$('btn-run').addEventListener('click', run);

$('btn-step').addEventListener('click', async () => {
  if (running) { pauseRequested = true; return; }
  if (nothingLoaded()) return;
  const r = await call('run', 1);
  await finishRun(r, true);
  if (traceEnabled() && r >= 0) {
    const t = await drainTrace();
    if (t) log(t.replace(/\n$/, ''), 'dim');
  }
});

// Space toggles run/pause and backtick toggles the panel - but not while the
// user is typing into one of the panel's inputs.
window.addEventListener('keydown', (e) => {
  const focused = document.activeElement;
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(focused?.tagName || '')) return;
  if (e.code === 'Space') {
    // Space is also how a keyboard presses whatever has the focus - a button,
    // a section's <summary>, a NAND Launch row - so the transport only gets it
    // when nothing else holds it. Taking it unconditionally meant tabbing to
    // Reset and pressing Space ran the console instead of resetting it.
    if (focused && focused !== document.body && focused !== document.documentElement) return;
    e.preventDefault();
    run();
  } else if (e.code === 'Backquote') {
    e.preventDefault();
    setPanel(!panelOpen());
  }
});

/* What the guest itself printed, at the severity the guest gave it.

   `cpu/log.rs` decodes the severity out of every `lm` packet and writes it
   into the line as `[lm/ERROR]`. Logging the whole drain as one unclassified
   block threw that away again: a title reporting an error looked exactly like
   the same title printing a frame counter. */
const GUEST_LEVELS: Record<string, LogClass> = {
  FATAL: 'err',
  ERROR: 'err',
  WARN: 'warn',
  INFO: 'dim',
  TRACE: 'dim',
};

export async function drainOutput(): Promise<void> {
  const bytes = await call('drain_output');
  if (!bytes || !bytes.length) return;
  for (const line of new TextDecoder().decode(bytes).replace(/\n$/, '').split('\n')) {
    log(line, GUEST_LEVELS[/^\[lm\/([A-Z]+)/.exec(line)?.[1] ?? '']);
  }
}

async function finishRun(steps: number, stepped?: boolean): Promise<void> {
  // The run is over however it ended, so a boot that was still being waited on
  // is over too - including the common case of a program that prints and exits
  // without ever presenting a frame.
  endLoad();
  const err = await readLastError();
  if (steps < 0) {
    setState('fault');
    // Not necessarily the CPU's: the error names its own kind, and a
    // renderer that refused a frame used to be reported as `CPU fault: GPU:`.
    log('Fault: ' + err, 'err');
    // The fault trace already carries the register snapshot from the CPU, and
    // carries its own levels: the block is an error, the lines that led up to
    // it are not.
    logTrace(await drainTrace());
  } else if (await call('halted')) {
    setState('halted');
    log('Halted (ExitProcess)', 'ok');
    await drainDiagnostics();
  } else if (!stepped) {
    setState('fault');
    log('Stopped unexpectedly.', 'err');
  }
  await drainOutput();
  await sdFlush();
  await saveFlush();
  await renderFb();
  await updatePc();
}

export async function updatePc(): Promise<void> {
  const pc = await call('get_pc');
  // Instructions retired, not the clock. The clock idles forward to the
  // earliest sleeper whenever every thread is blocked, so reading it here
  // made a parked Home Menu jump from 24M to 313M with nothing executed.
  const steps = await call('get_steps');
  $('pc').textContent = '0x' + pc.toString(16).padStart(8, '0');
  $('steps').textContent = steps.toLocaleString();
  // The same two figures on the loading screen, where they are the only sign
  // that a title still working towards its first frame is working at all.
  bootDetail(pc, steps);
  await updateRam();
}

// Guest RAM is the emulated console's own memory use (pages the guest has
// actually touched); the wasm figure is what the worker's linear memory costs
// the browser, which is the number that matters when a load fails to allocate.
async function updateRam(): Promise<void> {
  const ram = await call('ram');
  if (!ram) return;
  $('ram').textContent = `${formatBytes(ram.guest)} (${formatBytes(ram.wasm)})`;
}
