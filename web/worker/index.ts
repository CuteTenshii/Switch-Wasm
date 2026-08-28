/* switch-wasm worker host. Owns the wasm instance + session so the emulator
   runs off the main thread and long runs don't freeze the page. The main
   thread talks to this via postMessage: { id, cmd, args } -> { id, ok, result }.
   Byte buffers (file uploads, framebuffer, console/trace output) cross the
   boundary as transferred ArrayBuffers. */

import type { CallRequest, WorkerMessage } from '../shared/protocol';
import { CMD } from './commands';
import init from '@core/switch_wasm.js';
import wasmUrl from '@core/switch_wasm_bg.wasm?url';
import { state, type WasmExports } from './wasm';

// The WebWorker lib types `self` as the shared WorkerGlobalScope, which has no
// postMessage; this worker is a dedicated one, and that is where the reply
// half of the RPC lives.
const ctx = self as unknown as DedicatedWorkerGlobalScope;

function reply(message: WorkerMessage, transfer?: Transferable[]): void {
  if (transfer) ctx.postMessage(message, transfer);
  else ctx.postMessage(message);
}

// The two commands that mean something with no session open: `new` is what
// creates one, and `set_battery` caches its reading for whichever session
// comes next. Everything else works on the session this worker is holding.
// `last_error` belongs here because the case it exists for is the one where
// the session is gone or the module has trapped: `switch_last_error` returns a
// captured panic without consulting a handle at all.
const SESSIONLESS = new Set(['new', 'set_battery', 'last_error']);

/* Installing the GPU backend, once the guest has a channel to install it on.

   It cannot happen at startup: the channel a title draws through is opened by
   the title, so there is nothing to attach to until it has run. And it cannot
   happen from inside a run slice either, because opening a device is two
   promises and a slice has nowhere to await one. So it is tried after run
   slices, between them, which is exactly where a worker is free to await. */
let gpu: 'no' | 'trying' | 'done' | 'never' = 'no';

// The one answer worth asking again about. A title opens its channel a moment
// after it starts running, so until it has, there is nothing to attach a
// backend to and the right response is to try the next slice.
const NO_CHANNEL_YET = 'the title has not opened a channel yet';

/* Whether the GPU backend may be installed at all.

   On again: the readback that kept it off no longer waits. `Renderer::flush`
   answers "pending" instead of blocking, and the *present* is what waits --
   `Cpu::complete_pending_present` puts the frame up from a later slice, which
   in a worker is a later message, which is after the event loop has run and
   the map has completed.

   Left as a named constant because this is the switch to reach for if the
   browser disagrees with the CLI about it: the mechanism is verified natively
   (20 deferred presents over 300 frames, all 300 still presented), and a
   browser is where its no-op `Device::poll` lives. */
const GPU_BACKEND_READY = true;

/* Whether the device does the multisampling, instead of the backend rendering
   the expanded multisample surface a texel at a time.

   It is off. Maxwell's samples sit at the centres of the texels they are
   stored in and WebGPU's sit on a rotated grid the spec fixes, so the device's
   own multisampling anti-aliases every edge correctly and *differently* from
   the software rasterizer -- which is the reference the frame is checked
   against. What it buys is shading once per pixel rather than once per sample.

   It is a named constant because turning it on is a judgement about a title:
   at 4x that is four times less fragment work, and whether the difference
   shows is a question only a comparison against the reference answers. WebGPU
   guarantees four samples and no more, so 2x1, 4x2 and 4x4 render the expanded
   way here whatever this says. */
const GPU_DEVICE_MSAA = false;

function tryGpu(): void {
  if (!GPU_BACKEND_READY) {
    if (gpu === 'no') {
      gpu = 'never';
      console.info('[gpu] software rasterizer: turned off at GPU_BACKEND_READY');
    }
    return;
  }
  if (gpu !== 'no' || state.handle < 0) return;
  gpu = 'trying';
  const open = (state.exports as unknown as {
    switch_gpu_open(handle: number, deviceMsaa: boolean): Promise<string>;
  }).switch_gpu_open;
  open(state.handle, GPU_DEVICE_MSAA).then((what) => {
    if (what.startsWith('rendering on')) {
      gpu = 'done';
      console.info('[gpu] ' + what);
    } else if (what === NO_CHANNEL_YET) {
      gpu = 'no';
    } else {
      // A browser with no adapter and no device will not grow one. Asking
      // anyway costs a `wgpu::Instance` and a `requestAdapter` per slice, and
      // Chrome files two "Failed to create WebGPU Context Provider" warnings
      // with each of them -- two thousand of them in one sitting, on a build
      // whose answer was decided before the first. The software rasterizer is
      // what runs here, and it is what runs from now on.
      gpu = 'never';
      console.info('[gpu] software rasterizer: ' + what);
    }
  }).catch((e) => {
    gpu = 'never';
    console.info('[gpu] software rasterizer: ' + String(e));
  });
}

ctx.onmessage = (e: MessageEvent<CallRequest>) => {
  const { id, cmd, args } = e.data;
  try {
    const handler = CMD[cmd] as ((...a: unknown[]) => unknown) | undefined;
    if (!handler) throw new Error('unknown command ' + cmd);
    // A session handle is an index into the module's own session table, and a
    // miss there is a Rust panic - which on wasm is `unreachable`, taking the
    // whole core down rather than returning an error. The page cannot order
    // its way out of this on its own: Reset frees the session without waiting
    // for the run slice already in flight, so that slice's own follow-up calls
    // land just after the free. Refusing them here is what keeps a reset from
    // trapping the module.
    if (state.handle < 0 && !SESSIONLESS.has(cmd)) {
      reply({ id, ok: false, error: 'there is no session (it has been freed)' });
      return;
    }
    const result = handler(...args);
    if (result instanceof Uint8Array) {
      reply({ id, ok: true, result }, [result.buffer as ArrayBuffer]);
    } else if (result && typeof result === 'object' && 'error' in result) {
      reply({ id, ok: false, error: String((result as { error: unknown }).error) });
    } else {
      reply({ id, ok: true, result });
    }
    if (cmd === 'run') tryGpu();
  } catch (err) {
    reply({ id, ok: false, error: String(err) });
  }
};

(async () => {
  try {
    // The core is a wasm-bindgen module, because the GPU backend inside it
    // reaches WebGPU through wasm-bindgen's glue. `init` builds the import
    // object itself -- including `hostRead`, which it imports from
    // `@host/files` -- so there is nothing to pass in here, and the raw
    // `switch_*` exports are on the object it returns exactly as they were
    // when the worker instantiated the module itself.
    state.exports = await init({ module_or_path: wasmUrl }) as unknown as WasmExports;
    reply({ type: 'ready' });
  } catch (err) {
    reply({ type: 'ready', error: String(err) });
  }
})();
