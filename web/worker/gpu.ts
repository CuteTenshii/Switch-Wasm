// Installing the WebGPU backend between run slices, and replacing a lost device.

import { switch_gpu_lost, switch_gpu_open } from '@core/switch_wasm.js';
import { workerLog } from './log';

// GPU install state. Installed as the loop starts, between slices: opening a device
// needs promises a slice cannot await.
let gpu: 'no' | 'trying' | 'done' | 'never' = 'no';

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

export function tryGpu(): void {
  if (!GPU_BACKEND_READY) {
    if (gpu === 'no') {
      gpu = 'never';
      workerLog('[gpu] software rasterizer: turned off at GPU_BACKEND_READY');
    }
    return;
  }
  // Cheaper than parsing the JSON report after every slice.
  if (gpu === 'done' && switch_gpu_lost()) {
    if (gpuReopens >= GPU_REOPENS) {
      gpu = 'never';
      workerLog(
        `[gpu] software rasterizer: the device was lost ${gpuReopens} times; not asking again`,
      );
      return;
    }
    gpuReopens++;
    gpu = 'no';
    // The core has logged why the device was lost.
    workerLog(`[gpu] opening another device (attempt ${gpuReopens})`);
  }
  if (gpu !== 'no') return;
  gpu = 'trying';
  switch_gpu_open(GPU_DEVICE_MSAA, GPU_INTERLEAVE).then((what) => {
    if (what === INSTALLED) {
      // The backend logs once its first draw has rendered on the device.
      gpu = 'done';
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
