// The run loop: the worker runs slices back to back and reports each to the page.

import type { ErrorDialog, Housekeeping, SliceReport, WorkerMessage } from '../shared/protocol';
import { reportActivity } from './activity';
import { CMD } from './commands';
import { tryGpu } from './gpu';
import { flushLog } from './log';
import { flushStorage } from './persist';
import { switch_present } from '@core/switch_wasm.js';
import { core, state } from './wasm';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

// Slices sized to about one display frame, so input and commands interleave.
const SLICE_TARGET_MS = 16;
const FIRST_SLICE = 1_000_000;
const MIN_SLICE = 100_000;
const MAX_SLICE = 50_000_000;
// Tracing prints a line per instruction.
const TRACE_SLICE = 5000;
// The guest's clock rate, which `Cpu::pace_to_real_time` counts in.
const CYCLES_PER_MS = 1_020_000;
// Real time a slice can credit the guest with: past this a stall is lost rather than
// jumped over, or one long load would run every timeout in the title out at once.
const MAX_CREDIT_MS = 100;
// Waits are at most this long, so housekeeping and presents keep going through a long idle.
const MAX_WAIT_MS = 16;
// Closer to real time than this is not worth a timer.
const MIN_WAIT_MS = 1;
// Slices between housekeeping reports.
const HOUSEKEEPING_EVERY = 8;

let running = false;
let pauseRequested = false;
let tracing = false;
let budget = FIRST_SLICE;
let tick = 0;
let lastPace = 0;

// A zero-delay task: a `setTimeout` chain is clamped to 4 ms after a few calls.
const wake = new MessageChannel();
wake.port1.onmessage = () => slice();

export function startLoop(trace: boolean): void {
  if (running) return;
  running = true;
  pauseRequested = false;
  tracing = trace;
  budget = trace ? TRACE_SLICE : FIRST_SLICE;
  tick = 0;
  lastPace = performance.now();
  // Every title draws: open the device as it starts rather than once it asks.
  tryGpu();
  wake.port2.postMessage(null);
}

// Ends the loop after the slice in flight.
export function pauseLoop(): void {
  if (running) pauseRequested = true;
}

// For a session being freed: nothing more runs on it.
export function stopLoop(): void {
  running = false;
}

function slice(): void {
  if (!running || state.handle < 0) {
    running = false;
    return;
  }
  const at = performance.now();
  // A trace replays an exact run, which real time would change.
  if (!tracing) {
    const credit = Math.min(at - lastPace, MAX_CREDIT_MS);
    lastPace = at;
    const elapsed = BigInt(Math.round(credit * CYCLES_PER_MS));
    const aheadMs = Number(core.switch_pace(state.handle, elapsed)) / CYCLES_PER_MS;
    if (aheadMs >= MIN_WAIT_MS && !pauseRequested) {
      setTimeout(slice, Math.min(aheadMs, MAX_WAIT_MS));
      return;
    }
  }
  let steps: number;
  try {
    steps = CMD.run(budget) as number;
  } catch (err) {
    running = false;
    ctx.postMessage({ type: 'loop_failed', error: String(err) } satisfies WorkerMessage);
    return;
  }
  const ms = performance.now() - at;
  // A slice also ends short when the clock idles past real time.
  const done = steps < 0 || core.switch_halted(state.handle) !== 0;
  const stopped = done || pauseRequested;
  tryGpu();
  reportActivity(done);
  // Before the report, so a halt's last lines are on the page when it handles the halt.
  flushLog();
  const paintMs = paint();

  core.switch_audio_feed(state.handle);
  const report: SliceReport = {
    ended: done,
    steps,
    budget,
    ms,
    stopped,
    frame: core.switch_frame_count(state.handle),
    width: core.switch_fb_width(state.handle),
    height: core.switch_fb_height(state.handle),
    paintMs,
    audioFormat: core.switch_audio_format(state.handle),
    housekeeping: stopped || ++tick % HOUSEKEEPING_EVERY === 0 ? housekeeping() : null,
  };
  ctx.postMessage({ type: 'slice', report } satisfies WorkerMessage);

  if (!tracing) {
    const scale = Math.min(2, Math.max(0.5, SLICE_TARGET_MS / Math.max(ms, 0.1)));
    budget = Math.min(MAX_SLICE, Math.max(MIN_SLICE, Math.round(budget * scale)));
  }
  if (stopped) {
    running = false;
    return;
  }
  wake.port2.postMessage(null);
}

// Paint a new frame on the page's canvas; how long it took, or null if there was none.
export function paint(): number | null {
  const at = performance.now();
  return switch_present(state.handle) ? performance.now() - at : null;
}

function housekeeping(): Housekeeping {
  void flushStorage();
  const errors: ErrorDialog[] = [];
  for (let error = CMD.error_dialog(); error; error = CMD.error_dialog()) {
    errors.push(error as ErrorDialog);
  }
  return {
    steps: CMD.get_steps() as number,
    ram: CMD.ram() as Housekeeping['ram'],
    profilesEdited: CMD.users_take_edits() as boolean,
    keyboard: CMD.keyboard_request() as Housekeeping['keyboard'],
    errors,
  };
}
