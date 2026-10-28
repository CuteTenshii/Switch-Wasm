// The run loop, and the status bar it keeps up to date.

import { Pause, Play } from 'lucide';
import type { Housekeeping, SliceReport } from '../shared/protocol';
import { noteAudio } from './audio';
import { showEmulatorCrash, showGuestCrash } from './crash';
import { traceEnabled } from './debug';
import { countEmulation, noteFrame } from './display';
import { recordRunSlice } from './display-metrics';
import { $ } from './dom';
import { fmtCount } from '../shared/format';
import { formatBytes } from './format';
import { icon } from './icons';
import { dropErrors, offerErrors } from './errors';
import { dropKeyboard, offerKeyboard } from './keyboard';
import { endLoad } from './loading';
import { log } from './log';
import { noteRam } from './memory';
import { call, explainFailure, onLoopEvents, readLastError } from './rpc';
import { pullProfileEdits, storeProfileEdits } from './users';
import { loaded, setState } from './shell';
import { holdWakeLock, releaseWakeLock } from './wakelock';

let running = false;
// Set when the session is being freed, so the loop must not touch it again.
let aborted = false;

function setRunButton(isRunning: boolean): void {
  $('run-glyph').replaceChildren(icon(isRunning ? Pause : Play));
  $('run-label').textContent = isRunning ? 'Pause' : 'Run';
}

setRunButton(false);

function nothingLoaded(): boolean {
  if (loaded()) return false;
  log('Nothing is loaded - open a .nro, .elf, .nsp, .xci or .nca to boot one.');
  return true;
}

// Start the worker's loop, or ask it to pause after the slice in flight.
export async function run(): Promise<void> {
  if (running) {
    await call('pause_loop');
    return;
  }
  if (nothingLoaded()) return;
  running = true;
  aborted = false;
  countedAt = 0;
  setRunButton(true);
  setState('running');
  holdWakeLock();
  await call('start_loop', traceEnabled());
}

function loopEnded(): void {
  running = false;
  setRunButton(false);
  releaseWakeLock();
}

onLoopEvents({
  slice: (report) => void onSlice(report),
  failed: (error) => void onFailed(error),
});

async function onSlice(report: SliceReport): Promise<void> {
  countEmulation(report.ms);
  recordRunSlice(report.ms);
  // Stop does not wait for the slice in flight, so the session may be gone.
  if (aborted) return;
  noteAudio(report.audioFormat);
  if (report.housekeeping) applyHousekeeping(report.housekeeping);
  noteFrame(report);
  if (!report.stopped) return;
  loopEnded();
  if (report.steps >= 0 && report.steps >= report.budget) {
    endLoad();
    setState('paused');
    return;
  }
  await finishRun(report.steps);
}

function applyHousekeeping(h: Housekeeping): void {
  showReadouts(h.steps, h.ram);
  if (h.profilesEdited) void storeProfileEdits();
  offerKeyboard(h.keyboard);
  offerErrors(h.errors);
}

async function onFailed(error: string): Promise<void> {
  loopEnded();
  // A failure after Stop is expected; anything else is reported.
  if (aborted) return;
  setState('fault');
  const why = await explainFailure(new Error(error));
  log('The run loop stopped: ' + why, 'error');
  // Linear memory survives a trap, so the context can still be read.
  await reportPanicContext();
  showEmulatorCrash(why);
}

// Each piece is asked for separately so one failure costs only that piece.
async function reportPanicContext(): Promise<void> {
  const parts = [
    ['registers', () => call('log_regs')],
    ['threads', () => call('log_threads')],
    ['trace', () => call('flush_log')],
  ] as const;
  for (const [what, ask] of parts) {
    try {
      await ask();
    } catch {
      log(`The module could not be asked for its ${what}.`);
    }
  }
}

// Stops the loop for Stop without waiting for the slice; freeing the session ends it.
export function abortRun(): void {
  aborted = true;
  running = false;
  dropKeyboard();
  dropErrors();
  setRunButton(false);
}

$('btn-run').addEventListener('click', run);

$('btn-step').addEventListener('click', async () => {
  if (running) {
    await call('pause_loop');
    return;
  }
  if (nothingLoaded()) return;
  await finishRun(await call('run', 1), true);
});

async function finishRun(steps: number, stepped?: boolean): Promise<void> {
  endLoad();
  const err = await readLastError();
  if (steps < 0) {
    setState('fault');
    log('Fault: ' + err, 'error');
    showGuestCrash(err || 'The guest returned a fault without a message.');
  } else if (await call('halted')) {
    const fatal = await call('guest_fatal');
    if (fatal) {
      setState('fault');
      showGuestCrash(fatal);
    } else {
      setState('halted');
      log('Halted (ExitProcess)', 'info');
    }
  } else if (!stepped) {
    setState('fault');
    log('Stopped unexpectedly.', 'error');
    showGuestCrash('The guest stopped unexpectedly.');
  }
  await Promise.all([call('flush_storage'), pullProfileEdits(), updateReadouts()]);
}

let countedInstructions = 0;
let countedAt = 0;

export async function updateReadouts(): Promise<void> {
  showReadouts(await call('get_steps'), await call('ram'));
}

// Instructions retired, not the clock, which jumps while threads idle.
function showReadouts(instructions: number, ram: Housekeeping['ram']): void {
  const at = performance.now();
  // Only while running; a count that went backwards is a new session.
  const seconds = (at - countedAt) / 1000;
  const ran = instructions - countedInstructions;
  const rate = running && countedAt && seconds > 0 && ran >= 0
    ? ` (${fmtCount(ran / seconds)}/s)`
    : '';
  $('instructions').textContent = fmtCount(instructions) + rate;
  countedInstructions = instructions;
  countedAt = at;
  showRam(ram);
}

// Guest RAM is pages the guest touched; the wasm figure is the worker's linear memory.
export async function updateRam(): Promise<void> {
  showRam(await call('ram'));
}

function showRam(ram: Housekeeping['ram']): void {
  noteRam(ram);
  $('ram').textContent = `${formatBytes(ram.guest)} (${formatBytes(ram.wasm)})`;
}
