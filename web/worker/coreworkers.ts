// The core workers beyond the one this worker runs: the console gives a title three.

import type { CoreMessage, CoreStarted } from './core';
import { workerLog } from './log';
import { switch_add_core } from '@core/switch_wasm.js';

const CORE_WORKERS = 2;

const started: Worker[] = [];

// Start the core workers on this worker's module and shared memory.
export function startCores(module: WebAssembly.Module, memory: WebAssembly.Memory): void {
  for (let i = 0; i < CORE_WORKERS; i++) {
    const worker = new Worker(new URL('./core.ts', import.meta.url), {
      type: 'module',
      name: `Guest core ${i + 1}`,
    });
    worker.onmessage = (e: MessageEvent<CoreStarted>) => {
      if (!e.data.started) {
        workerLog(`[core] core worker ${i + 1} did not start: ${e.data.error}`, 'error');
        return;
      }
      started.push(worker);
      worker.postMessage({ slot: switch_add_core() } satisfies CoreMessage);
      if (started.length === CORE_WORKERS) {
        workerLog(`[core] ${CORE_WORKERS} core workers started on shared memory`);
      }
    };
    worker.onerror = (e) => workerLog(`[core] core worker ${i + 1} failed: ${e.message}`, 'error');
    worker.postMessage({ module, memory } satisfies CoreMessage);
  }
}
