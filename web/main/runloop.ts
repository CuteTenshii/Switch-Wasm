// The run loop, and the status bar it keeps up to date.

import { pumpAudio } from './audio';
import { showEmulatorCrash, showGuestCrash } from './crash';
import { drainDiagnostics, drainTrace, logTrace, traceEnabled } from './debug';
import { abortDisplay, countEmulation, flushDisplay, schedulePresentIfNewFrame } from './display';
import { recordRunSlice } from './display-metrics';
import { $ } from './dom';
import { fmtCount } from '../shared/format';
import { formatBytes } from './format';
import { dropKeyboard, pollKeyboard } from './keyboard';
import { endLoad } from './loading';
import { log, logBlock, type LogClass } from './log';
import { call, readLastError } from './rpc';
import { saveFlush } from './saves';
import { sdFlush } from './sdcard';
import { pullProfileEdits } from './users';
import { loaded, panelOpen, setPanel, setState } from './shell';
import { holdWakeLock, releaseWakeLock } from './wakelock';

// Run in worker slices sized to about one display frame, so input and painting interleave.
const SLICE_TARGET_MS = 16;
const FIRST_SLICE = 1_000_000;
const MIN_SLICE = 100_000;
const MAX_SLICE = 50_000_000;
// Slices between debug panel refreshes.
const HOUSEKEEPING_EVERY = 8;
// Tracing prints a line per instruction.
const TRACE_SLICE = 5000;

let running = false;
let pauseRequested = false;
// Set when the session is being freed, so the loop must not touch it again.
let aborted = false;

const PLAY_GLYPH = '▶';
const PAUSE_GLYPH = '❙❙';

function setRunButton(isRunning: boolean): void {
  $('run-glyph').textContent = isRunning ? PAUSE_GLYPH : PLAY_GLYPH;
  $('run-label').textContent = isRunning ? 'Pause' : 'Run';
}

function nothingLoaded(): boolean {
  if (loaded()) return false;
  log('Nothing is loaded - open a .nro, .elf, .nsp, .xci or .nca to boot one.');
  return true;
}

export async function run(): Promise<void> {
  if (running) {
    pauseRequested = true;
    return;
  }
  if (nothingLoaded()) return;
  running = true;
  pauseRequested = false;
  aborted = false;
  countedAt = 0;
  setRunButton(true);
  setState('running');
  holdWakeLock();
  const tracing = traceEnabled();
  let slice = tracing ? TRACE_SLICE : FIRST_SLICE;
  let steps: number;
  let tick = 0;
  try {
    for (;;) {
      const sliceAt = performance.now();
      steps = await call('run', slice);
      const sliceMs = performance.now() - sliceAt;
      countEmulation(sliceMs);
      recordRunSlice(sliceMs);
      // Reset does not wait for the in-flight slice, so the session may be gone.
      if (aborted) return;
      // Yield to the page; display snapshots are paced separately.
      await new Promise((resolve) => setTimeout(resolve, 0));
      if (aborted) return;
      // A short slice means the machine halted.
      const done = steps < 0 || steps < slice;
      if (!tracing) {
        const scale = Math.min(2, Math.max(0.5, SLICE_TARGET_MS / Math.max(sliceMs, 0.1)));
        slice = Math.min(MAX_SLICE, Math.max(MIN_SLICE, Math.round(slice * scale)));
      }
      await pumpAudio();
      if (done || ++tick % HOUSEKEEPING_EVERY === 0) {
        await Promise.all([
          updatePc(), drainOutput(), drainDiagnostics(), sdFlush(), saveFlush(), pullProfileEdits(),
          pollKeyboard(),
        ]);
      }
      schedulePresentIfNewFrame();
      if (done) break;
      if (pauseRequested) {
        endLoad();
        setState('paused');
        await flushDisplay();
        return;
      }
    }
  } catch (err) {
    // A failure after a reset is expected; anything else is reported.
    if (!aborted) {
      setState('fault');
      // A panic traps as `unreachable`; `switch_last_error` recovers its message.
      let why = (err as Error).message;
      try {
        const captured = await readLastError();
        if (captured) why += ' - ' + captured;
      } catch {
        // The module may be too far gone to answer.
      }
      log('The run loop stopped: ' + why, 'err');
      // Linear memory survives a trap, so the context can still be read.
      await reportPanicContext();
      showEmulatorCrash(why);
    }
    return;
  } finally {
    running = false;
    setRunButton(false);
    releaseWakeLock();
  }
  await finishRun(steps);
}

// Each piece is asked for separately so one failure costs only that piece.
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
      else logBlock(text);
    } catch {
      log(`The module could not be asked for its ${what}.`);
    }
  }
  log('Take a crash report from the debug panel before resetting - a reset is what loses this.',
    'warn');
}

// Stops the loop for Reset without waiting for the slice.
export function abortRun(): void {
  aborted = true;
  pauseRequested = true;
  running = false;
  abortDisplay();
  dropKeyboard();
  setRunButton(false);
}

$('btn-run').addEventListener('click', run);

$('btn-step').addEventListener('click', async () => {
  if (running) {
    pauseRequested = true;
    return;
  }
  if (nothingLoaded()) return;
  const r = await call('run', 1);
  await finishRun(r, true);
  if (traceEnabled() && r >= 0) {
    const t = await drainTrace();
    if (t) log(t.replace(/\n$/, ''), 'dim');
  }
});

// Space toggles run/pause and backtick toggles the panel, except while typing.
window.addEventListener('keydown', (e) => {
  const focused = document.activeElement;
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(focused?.tagName || '')) return;
  if (document.querySelector('dialog[open]')) return;
  if (e.code === 'Space') {
    // Space only reaches the transport when nothing else has focus.
    if (focused && focused !== document.body && focused !== document.documentElement) return;
    e.preventDefault();
    void run();
  } else if (e.code === 'Backquote') {
    e.preventDefault();
    setPanel(!panelOpen());
  }
});

// Maps the `[lm/LEVEL]` tags `cpu/log.rs` writes to log levels.
const GUEST_LEVELS: Record<string, LogClass> = {
  FATAL: 'err',
  ERROR: 'err',
  WARN: 'warn',
  INFO: 'ok',
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
  endLoad();
  const err = await readLastError();
  if (steps < 0) {
    setState('fault');
    log('Fault: ' + err, 'err');
    showGuestCrash(err || 'The guest returned a fault without a message.');
    logTrace(await drainTrace());
  } else if (await call('halted')) {
    const fatal = await call('guest_fatal');
    if (fatal) {
      setState('fault');
      showGuestCrash(fatal);
    } else {
      setState('halted');
      log('Halted (ExitProcess)', 'ok');
    }
    await drainDiagnostics();
  } else if (!stepped) {
    setState('fault');
    log('Stopped unexpectedly.', 'err');
    showGuestCrash('The guest stopped unexpectedly.');
  }
  await Promise.all([
    drainOutput(), sdFlush(), saveFlush(), pullProfileEdits(), flushDisplay(), updatePc(),
  ]);
}

let countedInstructions = 0;
let countedAt = 0;

export async function updatePc(): Promise<void> {
  const pc = await call('get_pc');
  // Instructions retired, not the clock, which jumps while threads idle.
  const instructions = await call('get_steps');
  const at = performance.now();
  $('pc').textContent = '0x' + pc.toString(16).padStart(8, '0');
  // Only while running; a count that went backwards is a new session.
  const seconds = (at - countedAt) / 1000;
  const ran = instructions - countedInstructions;
  const rate = running && countedAt && seconds > 0 && ran >= 0
    ? ` (${fmtCount(ran / seconds)}/s)`
    : '';
  $('instructions').textContent = fmtCount(instructions) + rate;
  countedInstructions = instructions;
  countedAt = at;
  await updateRam();
}

// Guest RAM is pages the guest touched; the wasm figure is the worker's linear memory.
async function updateRam(): Promise<void> {
  const ram = await call('ram');
  if (!ram) return;
  $('ram').textContent = `${formatBytes(ram.guest)} (${formatBytes(ram.wasm)})`;
}
