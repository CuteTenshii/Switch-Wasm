// A guest core: the core's module instantiated on the system worker's shared
// memory, so guest RAM and the emulator's heap are the same bytes on both.
// It then serves one core of each session, blocking until the session stops it.

import { initSync } from '@core/switch_wasm.js';

// From the system worker: the compiled core and the memory to run it on, then
// a core slot per session.
export type CoreMessage =
  | { module: WebAssembly.Module; memory: WebAssembly.Memory }
  | { slot: number };

export type CoreStarted = { started: true } | { started: false; error: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

// The raw export; the glue only wraps the `#[wasm_bindgen]` ones.
let serve: ((slot: number) => void) | null = null;

ctx.onmessage = (e: MessageEvent<CoreMessage>) => {
  if ('slot' in e.data) {
    serve?.(e.data.slot);
    return;
  }
  let reply: CoreStarted;
  try {
    const exports = initSync({ module: e.data.module, memory: e.data.memory }) as unknown as {
      switch_serve_core: (slot: number) => void;
    };
    serve = exports.switch_serve_core;
    reply = { started: true };
  } catch (err) {
    reply = { started: false, error: String(err) };
  }
  ctx.postMessage(reply);
};
