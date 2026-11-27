// Worker host: owns the wasm instance and session, answering `{ id, cmd, args }`
// RPCs from the main thread.

import type { CallRequest, CommandHandlers, WorkerMessage } from '../shared/protocol';
import { CMD } from './commands';
import { flushLog, workerLog } from './log';
import init, { switch_attach_screen, switch_init, switch_set_reader } from '@core/switch_wasm.js';
import wasmUrl from '@core/switch_wasm_bg.wasm?url';
import { core, explainFailure, setCore, type WasmExports } from './wasm';
import { startCores } from './coreworkers';
import { onPageMessage, postToPage } from './post';
import { spawnWorker } from '../shared/spawn';
import readerUrl from './reader.ts?worker&url';

onPageMessage((message) => {
  if ('type' in message) {
    try {
      switch_attach_screen(message.canvas);
    } catch (err) {
      workerLog('[display] the screen could not be attached: ' + String(err), 'error');
    }
    return;
  }
  void answer(message);
});

// Sync and async handlers alike: IndexedDB work answers when it is done.
async function answer({ id, cmd, args }: CallRequest): Promise<void> {
  let reply: WorkerMessage;
  try {
    const handler = (CMD as CommandHandlers)[cmd] as ((...a: unknown[]) => unknown) | undefined;
    if (!handler) throw new Error('unknown command ' + cmd);
    reply = { type: 'reply', id, ok: true, result: await handler(...args) };
  } catch (err) {
    reply = { type: 'reply', id, ok: false, error: explainFailure(err) };
  }
  // Before the reply, so what the command logged is on the page when the caller resumes.
  flushLog();
  postToPage(reply);
}

void (async () => {
  try {
    // Compiled here rather than by the glue, which keeps the module to itself.
    const module = await WebAssembly.compileStreaming(fetch(wasmUrl));
    setCore(await init({ module_or_path: module }) as unknown as WasmExports);
    // Install the panic hook before anything else can panic.
    switch_init();
    let reader!: Worker;
    // This thread starts the reader, so a read that blocks it before the reader runs times out.
    const readerRunning = new Promise<void>((resolve, reject) => {
      reader = spawnWorker(readerUrl, 'File reader', (why) => reject(new Error(why)));
      reader.addEventListener('message', () => resolve(), { once: true });
    });
    switch_set_reader(reader);
    startCores(module, core.memory);
    await readerRunning;
    postToPage({ type: 'ready' });
  } catch (err) {
    postToPage({ type: 'ready', error: String(err) });
  }
})();
