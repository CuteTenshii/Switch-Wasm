// A guest core: the core's module instantiated on the system worker's shared
// memory, so guest RAM and the emulator's heap are the same bytes on both.

import { initSync } from '@core/switch_wasm.js';

// From the system worker: the compiled core and the memory to run it on.
export interface CoreStart {
  module: WebAssembly.Module;
  memory: WebAssembly.Memory;
}

export type CoreStarted = { started: true } | { started: false; error: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<CoreStart>) => {
  let reply: CoreStarted;
  try {
    initSync({ module: e.data.module, memory: e.data.memory });
    reply = { started: true };
  } catch (err) {
    reply = { started: false, error: String(err) };
  }
  ctx.postMessage(reply);
};
