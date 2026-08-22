/* Promise-based RPC over postMessage.
 *
 * The emulator runs in a web worker (see web/worker) so long executions
 * don't freeze the page. Buffers (files, framebuffer, console/trace output)
 * are transferred across the boundary rather than copied. */

import type { CallRequest, Commands, WorkerMessage } from '../shared/protocol';
import { log } from './log';

let worker: Worker | null = null;
let ready = false;
let readyResolve!: () => void;
const readyPromise = new Promise<void>((r) => { readyResolve = r; });

// The session the worker is holding, mirrored here so the persistence flushes
// can tell whether there is anything to flush. Display only otherwise.
let session = -1;

let msgId = 0;
const pending = new Map<number, {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
}>();

export function isReady(): boolean {
  return ready;
}

export function whenReady(): Promise<void> {
  return readyPromise;
}

export function hasSession(): boolean {
  return session >= 0;
}

export function setSession(handle: number): void {
  session = handle;
}

export function call<K extends keyof Commands>(
  cmd: K,
  ...args: Parameters<Commands[K]>
): Promise<ReturnType<Commands[K]>> {
  return new Promise<ReturnType<Commands[K]>>((resolve, reject) => {
    if (!worker) {
      reject(new Error('the emulator worker has not been started'));
      return;
    }
    const id = ++msgId;
    pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
    const request: CallRequest = { id, cmd, args };
    worker.postMessage(request);
  });
}

// The worker is named by its *source*, which the bundler follows: it emits it
// as its own hashed chunk and rewrites this URL to match. That is what retires
// the old path trap - there is no longer a built path written down here that
// has to agree with wherever the build actually put the file.
//
// `{ type: 'module' }` is required, not incidental: `worker.format` is 'es', in
// dev and in the build alike. See the note in vite.config.ts before changing
// either half.
export function initWorker(): void {
  worker = new Worker(new URL('../worker/index.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (e: MessageEvent<WorkerMessage>) => {
    const d = e.data;
    if ('type' in d) {
      ready = true;
      // A core that failed to instantiate still reports ready, or the page
      // would sit on "starting core..." for ever; the reason it failed is the
      // only useful thing left to say.
      if (d.error) log('core failed to load: ' + d.error, 'err');
      readyResolve();
      return;
    }
    const p = pending.get(d.id);
    if (!p) return;
    pending.delete(d.id);
    if (d.ok) p.resolve(d.result);
    else p.reject(new Error(d.error || 'unknown error'));
  };
  worker.onerror = (e) => {
    readyResolve();
    log('worker error: ' + e.message, 'err');
  };
}

export async function readLastError(): Promise<string> {
  return await call('last_error');
}
