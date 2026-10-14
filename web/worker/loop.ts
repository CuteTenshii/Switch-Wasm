// The run loop: the worker runs slices back to back and reports each to the page.

import type { ErrorDialog, Housekeeping, SliceReport, WorkerMessage } from '../shared/protocol';
import { reportActivity } from './activity';
import { CMD } from './commands';
import { tryGpu } from './gpu';
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
// Slices between housekeeping reports.
const HOUSEKEEPING_EVERY = 8;
// A second of 48 kHz stereo, past what one slice produces.
const AUDIO_MAX_PULL = 96000;

let running = false;
let pauseRequested = false;
let tracing = false;
let budget = FIRST_SLICE;
let tick = 0;

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

function post(message: WorkerMessage, transfer: Transferable[] = []): void {
  ctx.postMessage(message, transfer);
}

function slice(): void {
  if (!running || state.handle < 0) {
    running = false;
    return;
  }
  const at = performance.now();
  let steps: number;
  try {
    steps = CMD.run(budget) as number;
  } catch (err) {
    running = false;
    post({ type: 'loop_failed', error: String(err) });
    return;
  }
  const ms = performance.now() - at;
  // A short slice means the machine halted.
  const done = steps < 0 || steps < budget;
  const stopped = done || pauseRequested;
  tryGpu();
  reportActivity(done);
  const paintMs = paint();

  const audio = pullAudio();
  const report: SliceReport = {
    steps,
    budget,
    ms,
    stopped,
    frame: core.switch_frame_count(state.handle),
    width: core.switch_fb_width(state.handle),
    height: core.switch_fb_height(state.handle),
    paintMs,
    audio,
    housekeeping: stopped || ++tick % HOUSEKEEPING_EVERY === 0 ? housekeeping() : null,
  };
  post({ type: 'slice', report }, audio ? [audio.pcm.buffer] : []);

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

function pullAudio(): SliceReport['audio'] {
  const packed = core.switch_audio_format(state.handle);
  const rate = packed & 0x00ffffff;
  const channels = packed >>> 24;
  if (!rate || !channels) return null;
  const pcm = CMD.audio_pull(AUDIO_MAX_PULL) as Uint8Array<ArrayBuffer> | null;
  return pcm && pcm.length >= channels * 2 ? { rate, channels, pcm } : null;
}

const text = new TextDecoder();

function housekeeping(): Housekeeping {
  const decode = (bytes: unknown) => (bytes instanceof Uint8Array && bytes.length
    ? text.decode(bytes)
    : '');
  const h = state.handle;
  const errors: ErrorDialog[] = [];
  for (let error = CMD.error_dialog(); error; error = CMD.error_dialog()) {
    errors.push(error as ErrorDialog);
  }
  return {
    steps: CMD.get_steps() as number,
    ram: CMD.ram() as Housekeeping['ram'],
    output: decode(CMD.drain_output()),
    trace: decode(CMD.drain_trace()),
    sdChanged: core.switch_sd_pending_changes(h) !== 0,
    savesChanged: CMD.saves_pending() as boolean,
    profilesEdited: CMD.users_take_edits() as boolean,
    keyboard: CMD.keyboard_request() as Housekeeping['keyboard'],
    errors,
  };
}
