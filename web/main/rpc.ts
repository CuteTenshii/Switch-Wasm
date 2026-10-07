// Promise-based RPC to the emulator worker over postMessage.

import type { CallRequest, Commands, WorkerMessage } from '../shared/protocol';
import { log } from './log';

let worker: Worker | null = null;
let ready = false;
let readyResolve!: () => void;
const readyPromise = new Promise<void>((r) => {
  readyResolve = r;
});

// Mirrors the worker's session handle.
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

// `{ type: 'module' }` must match Vite's `worker.format: 'es'`.
export function initWorker(): void {
  worker = new Worker(new URL('../worker/index.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (e: MessageEvent<WorkerMessage>) => {
    const d = e.data;
    if ('type' in d && d.type === 'log') {
      log(d.text, d.cls);
      return;
    }
    if ('type' in d) {
      ready = true;
      // A core that failed to load still reports ready, with the error.
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
