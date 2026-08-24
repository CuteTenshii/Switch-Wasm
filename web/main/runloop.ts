/* The run loop, and the status bar it keeps up to date. */

import { pumpAudio } from './audio';
import { drainDiagnostics, drainTrace, traceEnabled } from './debug';
import { presentIfNewFrame, renderFb } from './display';
import { $ } from './dom';
import { formatBytes } from './format';
import { bootDetail, endLoad } from './loading';
import { log } from './log';
import { call, readLastError } from './rpc';
import { saveFlush } from './saves';
import { sdFlush } from './sdcard';
import { panelOpen, setPanel, setState } from './shell';

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

const PLAY_GLYPH = '▶';
const PAUSE_GLYPH = '❙❙';

function setRunButton(isRunning: boolean): void {
  $('run-glyph').textContent = isRunning ? PAUSE_GLYPH : PLAY_GLYPH;
  $('run-label').textContent = isRunning ? 'Pause' : 'Run';
}

export async function run(): Promise<void> {
  if (running) { pauseRequested = true; return; }
  running = true;
  pauseRequested = false;
  setRunButton(true);
  setState('running');
  const slice = traceEnabled() ? TRACE_SLICE : RUN_SLICE;
  let steps = 0;
  let tick = 0;
  for (;;) {
    steps = await call('run', slice);
    // Yield so the UI repaints and any queued input is processed.
    await new Promise((r) => setTimeout(r, 0));
    // `Cpu::run` only stops short of its budget when the machine halted, so a
    // short slice means this run is over - no separate `halted` round trip.
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
      running = false;
      setRunButton(false);
      // Whatever the guest was about to present, it is not going to now: a
      // paused machine is not a loading one.
      endLoad();
      setState('paused');
      await renderFb();
      return;
    }
  }
  running = false;
  setRunButton(false);
  await finishRun(steps);
}

/** Stop the loop because the session itself is going away (Reset). Pausing
 *  politely would be waiting for a slice that is about to be freed. */
export function abortRun(): void {
  pauseRequested = true;
  running = false;
  setRunButton(false);
}

$('btn-run').addEventListener('click', run);

$('btn-step').addEventListener('click', async () => {
  if (running) { pauseRequested = true; return; }
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
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName || '')) return;
  if (e.code === 'Space') { e.preventDefault(); run(); }
  else if (e.code === 'Backquote') { e.preventDefault(); setPanel(!panelOpen()); }
});

export async function drainOutput(): Promise<void> {
  const bytes = await call('drain_output');
  if (bytes && bytes.length) {
    log(new TextDecoder().decode(bytes));
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
    log('CPU fault: ' + err, 'err');
    // The fault trace already carries the register snapshot from the CPU.
    const t = await drainTrace();
    if (t) log(t.replace(/\n$/, ''), 'err');
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
  const steps = await call('get_cycles');
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
