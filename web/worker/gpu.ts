// Installing the WebGPU backend between run slices, and replacing a lost device.

import { workerLog } from './log';
import { api, readWholeJson, state } from './wasm';

// GPU install state. Installed between run slices, once the title has opened a
// channel: opening a device needs promises a slice cannot await.
let gpu: 'no' | 'trying' | 'done' | 'never' = 'no';

// The session the backend was installed on; a new session needs its own.
let gpuSession = -1;

const NO_CHANNEL_YET = 'the title has not opened a channel yet';

const RENDERING_ON = 'rendering on';

const GPU_BACKEND_READY = true;

// Let the device multisample instead of rendering the expanded surface. Off:
// WebGPU's sample positions differ from Maxwell's and the software reference.
const GPU_DEVICE_MSAA = false;

// Allow software fallback draws inside a device frame. Off: they race the
// readback and corrupt pixels.
const GPU_INTERLEAVE = false;

// Fallback adapter name: Chrome leaves `description` empty on macOS, and
// Firefox blanks every field against fingerprinting.
type AdapterInfo = {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
};

type WebGpu = { requestAdapter(): Promise<{ info?: AdapterInfo } | null> };

const UNNAMED_ADAPTER = 'an unnamed adapter';

async function adapterName(): Promise<string> {
  try {
    const webgpu = (navigator as unknown as { gpu?: WebGpu }).gpu;
    if (!webgpu) return UNNAMED_ADAPTER;
    const info = (await webgpu.requestAdapter())?.info;
    if (!info) return UNNAMED_ADAPTER;
    const named = info.description || info.device;
    if (named) return named;
    return [info.vendor, info.architecture].filter(Boolean).join(' ') || UNNAMED_ADAPTER;
  } catch {
    // Failing to label a running device is not worth reporting.
    return UNNAMED_ADAPTER;
  }
}

// Lost-device replacements before the rasterizer keeps the session.
const GPU_REOPENS = 3;
let gpuReopens = 0;

// Cheaper than parsing the JSON report after every slice.
function gpuLost(): boolean {
  if (state.handle < 0) return false;
  const lost = (state.exports as unknown as {
    switch_gpu_lost?: (handle: number) => number;
  }).switch_gpu_lost;
  return !!lost && lost(state.handle) !== 0;
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
      (buf, cap) => api().switch_gpu_report_json(state.handle, buf, cap),
      {},
    );
    const why = report.lostBecause ? ` (${report.lostBecause})` : '';
    workerLog(`[gpu] the device was lost${why} - opening another (attempt ${gpuReopens})`, 'err');
  }
  // `never` survives new sessions: this browser has no device to give.
  if (gpu === 'done' && state.handle !== gpuSession) gpu = 'no';
  if (gpu !== 'no' || state.handle < 0) return;
  gpu = 'trying';
  const open = (state.exports as unknown as {
    switch_gpu_open: (handle: number, deviceMsaa: boolean, interleave: boolean) => Promise<string>;
  }).switch_gpu_open;
  open(state.handle, GPU_DEVICE_MSAA, GPU_INTERLEAVE).then(async (what) => {
    if (what.startsWith(RENDERING_ON)) {
      gpu = 'done';
      gpuSession = state.handle;
      const named = what.slice(RENDERING_ON.length).trim();
      workerLog('[gpu] ' + RENDERING_ON + ' ' + (named || await adapterName()), 'ok');
    } else if (what === NO_CHANNEL_YET) {
      gpu = 'no';
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
