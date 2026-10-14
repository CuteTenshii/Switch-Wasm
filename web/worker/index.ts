// Worker host: owns the wasm instance and session, answering `{ id, cmd, args }`
// RPCs from the main thread.

import type { CallRequest, WorkerMessage } from '../shared/protocol';
import { reportActivity } from './activity';
import { CMD } from './commands';
import { paint } from './loop';
import { tryGpu } from './gpu';
import { workerLog } from './log';
import init, { switch_attach_screen } from '@core/switch_wasm.js';
import wasmUrl from '@core/switch_wasm_bg.wasm?url';
import { core, setCore, state, type WasmExports } from './wasm';
import type { CoreStart, CoreStarted } from './core';

// `self` is typed as the shared WorkerGlobalScope, which lacks postMessage.
const ctx = self as unknown as DedicatedWorkerGlobalScope;

function reply(message: WorkerMessage, transfer?: Transferable[]): void {
  if (transfer) ctx.postMessage(message, transfer);
  else ctx.postMessage(message);
}

// The page's canvas, handed over once at startup.
interface ScreenHandoff {
  type: 'screen';
  canvas: OffscreenCanvas;
}

// Commands valid with no session open.
const SESSIONLESS = new Set([
  'new', 'set_battery', 'last_error', 'clear_screen',
]);

ctx.onmessage = (e: MessageEvent<CallRequest | ScreenHandoff>) => {
  if ('type' in e.data) {
    try {
      switch_attach_screen(e.data.canvas);
    } catch (err) {
      workerLog('[display] the screen could not be attached: ' + String(err), 'err');
    }
    return;
  }
  const { id, cmd, args } = e.data;
  try {
    const handler = CMD[cmd] as ((...a: unknown[]) => unknown) | undefined;
    if (!handler) throw new Error('unknown command ' + cmd);
    // A stale handle would panic (trap) the module; Reset frees the session
    // while a slice's follow-up calls are still in flight.
    if (state.handle < 0 && !SESSIONLESS.has(cmd)) {
      reply({ id, ok: false, error: 'there is no session (it has been freed)' });
      return;
    }
    const result = handler(...args);
    const stopped = cmd === 'run' && typeof result === 'number' && result < Number(args[0]);
    if (result instanceof Uint8Array) {
      reply({ id, ok: true, result }, [result.buffer as ArrayBuffer]);
    } else if (result && typeof result === 'object' && 'error' in result) {
      reply({ id, ok: false, error: String((result).error) });
    } else {
      reply({ id, ok: true, result });
    }
    if (cmd === 'run') {
      paint();
      tryGpu();
      reportActivity(stopped);
    }
  } catch (err) {
    reply({ id, ok: false, error: String(err) });
  }
};

// Guest cores beyond the one this worker runs: the console gives a title three.
const CORE_WORKERS = 2;

// Start the core workers on this worker's module and shared memory.
function startCores(module: WebAssembly.Module, memory: WebAssembly.Memory): void {
  let started = 0;
  for (let i = 0; i < CORE_WORKERS; i++) {
    const worker = new Worker(new URL('./core.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e: MessageEvent<CoreStarted>) => {
      if (!e.data.started) {
        workerLog(`[core] core worker ${i + 1} did not start: ${e.data.error}`, 'err');
      } else if (++started === CORE_WORKERS) {
        workerLog(`[core] ${CORE_WORKERS} core workers started on shared memory`);
      }
    };
    worker.onerror = (e) => workerLog(`[core] core worker ${i + 1} failed: ${e.message}`, 'err');
    worker.postMessage({ module, memory } satisfies CoreStart);
  }
}

void (async () => {
  try {
    // Compiled here rather than by the glue, which keeps the module to itself.
    const module = await WebAssembly.compileStreaming(fetch(wasmUrl));
    setCore(await init({ module_or_path: module }) as unknown as WasmExports);
    // Install the panic hook before anything else can panic.
    core.switch_init();
    startCores(module, core.memory);
    reply({
      type: 'audio_ring',
      buffer: core.memory.buffer as unknown as SharedArrayBuffer,
      at: core.switch_audio_ring(),
      samples: core.switch_audio_ring_samples(),
    });
    reply({ type: 'ready' });
  } catch (err) {
    reply({ type: 'ready', error: String(err) });
  }
})();
