/* switch-wasm worker host. Owns the wasm instance + session so the emulator
   runs off the main thread and long runs don't freeze the page. The main
   thread talks to this via postMessage: { id, cmd, args } -> { id, ok, result }.
   Byte buffers (file uploads, framebuffer, console/trace output) cross the
   boundary as transferred ArrayBuffers. */

import type { CallRequest, WorkerMessage } from '../shared/protocol';
import { CMD } from './commands';
import { hostRead } from './hostfiles';
import wasmUrl from '@core/switch_wasm.wasm?url';
import { state, type WasmExports } from './wasm';

// The WebWorker lib types `self` as the shared WorkerGlobalScope, which has no
// postMessage; this worker is a dedicated one, and that is where the reply
// half of the RPC lives.
const ctx = self as unknown as DedicatedWorkerGlobalScope;

function reply(message: WorkerMessage, transfer?: Transferable[]): void {
  if (transfer) ctx.postMessage(message, transfer);
  else ctx.postMessage(message);
}

ctx.onmessage = (e: MessageEvent<CallRequest>) => {
  const { id, cmd, args } = e.data;
  try {
    const handler = CMD[cmd] as ((...a: unknown[]) => unknown) | undefined;
    if (!handler) throw new Error('unknown command ' + cmd);
    const result = handler(...args);
    if (result instanceof Uint8Array) {
      reply({ id, ok: true, result }, [result.buffer as ArrayBuffer]);
    } else if (result && typeof result === 'object' && 'error' in result) {
      reply({ id, ok: false, error: String((result as { error: unknown }).error) });
    } else {
      reply({ id, ok: true, result });
    }
  } catch (err) {
    reply({ id, ok: false, error: String(err) });
  }
};

(async () => {
  try {
    // instantiateStreaming fetches + compiles in one pass (works in workers).
    // The one import is how the module reads the open container; see
    // `hostRead`.
    const { instance } = await WebAssembly.instantiateStreaming(
      fetch(wasmUrl), { env: { host_read: hostRead } });
    state.exports = instance.exports as unknown as WasmExports;
    reply({ type: 'ready' });
  } catch (err) {
    reply({ type: 'ready', error: String(err) });
  }
})();
