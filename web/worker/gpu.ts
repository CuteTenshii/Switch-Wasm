// Installing the WebGPU backend between run slices, and replacing a lost device.

import { switch_gpu_open } from '@core/switch_wasm.js';
import { workerLog } from './log';
import { core, readWholeJson, state } from './wasm';

// GPU install state. Installed as the loop starts, between slices: opening a device
// needs promises a slice cannot await.
let gpu: 'no' | 'trying' | 'done' | 'never' = 'no';

// The session the backend was installed on; a new session needs its own.
let gpuSession = -1;

const INSTALLED = 'installed';

const GPU_BACKEND_READY = true;

// Let the device multisample instead of rendering the expanded surface. Off:
// WebGPU's sample positions differ from Maxwell's and the software reference.
const GPU_DEVICE_MSAA = false;

// Allow software fallback draws inside a device frame. Off: they race the
// readback and corrupt pixels.
const GPU_INTERLEAVE = false;

// Lost-device replacements before the rasterizer keeps the session.
const GPU_REOPENS = 3;
let gpuReopens = 0;

// Cheaper than parsing the JSON report after every slice.
function gpuLost(): boolean {
  if (state.handle < 0) return false;
  return core.switch_gpu_lost(state.handle) !== 0;
}

export function tryGpu(): void {
  if (!GPU_BACKEND_READY) {
    if (gpu === 'no') {
      gpu = 'never';
      workerLog('[gpu] software rasterizer: turned off at GPU_BACKEND_READY');
    }
    return;
  }
  if (gpu === 'done' && gpuLost()) {
    if (gpuReopens >= GPU_REOPENS) {
      gpu = 'never';
      workerLog(
        `[gpu] software rasterizer: the device was lost ${gpuReopens} times; not asking again`,
      );
      return;
    }
    gpuReopens++;
    gpu = 'no';
    const report = readWholeJson<{ lostBecause?: string | null }>(
      2048,
      (buf, cap) => core.switch_gpu_report_json(state.handle, buf, cap),
      {},
    );
    const why = report.lostBecause ? ` (${report.lostBecause})` : '';
    workerLog(`[gpu] the device was lost${why} - opening another (attempt ${gpuReopens})`, 'err');
  }
  // `never` survives new sessions: this browser has no device to give.
  if (gpu === 'done' && state.handle !== gpuSession) gpu = 'no';
  if (gpu !== 'no' || state.handle < 0) return;
  gpu = 'trying';
  switch_gpu_open(state.handle, GPU_DEVICE_MSAA, GPU_INTERLEAVE).then((what) => {
    if (what === INSTALLED) {
      // The backend logs once its first draw has rendered on the device.
      gpu = 'done';
      gpuSession = state.handle;
    } else {
      // No adapter or device: stop asking, which costs an instance and console
      // warnings per slice.
      gpu = 'never';
      workerLog('[gpu] software rasterizer: ' + what);
    }
  }).catch((e) => {
    gpu = 'never';
    workerLog('[gpu] software rasterizer: ' + String(e));
  });
}
