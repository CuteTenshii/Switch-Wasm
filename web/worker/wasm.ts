// The wasm module's own exports; every callable one is the glue's (`@core/switch_wasm.js`).

import { switch_panic_message } from '@core/switch_wasm.js';

export interface WasmExports {
  memory: WebAssembly.Memory;
}

// The module's exports, set once at load. A live binding: importers see the assignment.
export let core = null as unknown as WasmExports;

export function setCore(exports: WasmExports): void {
  core = exports;
}

// IndexedDB commands can arrive while the module is still loading.
export function coreLoaded(): boolean {
  return (core as WasmExports | null) !== null;
}

// An empty answer is an empty report, so fall back instead of throwing.
export function parseOr<T>(text: string, fallback: T): T {
  try {
    return JSON.parse(text) as T;
  } catch {
    return fallback;
  }
}

// What a thrown `err` says. A Rust panic traps as a bare `unreachable`, so the hook's message
// goes with it.
export function explainFailure(err: unknown): string {
  const why = String(err);
  try {
    const panic = coreLoaded() ? switch_panic_message() : '';
    return panic ? why + ' - ' + panic : why;
  } catch {
    // The module may be too far gone to answer.
    return why;
  }
}
