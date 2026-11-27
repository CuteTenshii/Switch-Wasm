// Promise-based RPC to the emulator worker over postMessage.

import type { CallRequest, Commands, PageMessage, WorkerMessage } from '../shared/protocol';
import { spawnWorker } from '../shared/spawn';
import workerUrl from '../worker/index.ts?worker&url';
import { log } from './log';

let worker: Worker | null = null;
let startResolve!: () => void;
let startReject!: (err: Error) => void;
const started = new Promise<void>((resolve, reject) => {
  startResolve = resolve;
  startReject = reject;
});
// Never settles when the start fails, so work queued behind it is dropped.
const readyPromise = started.catch(() => new Promise<void>(() => undefined));

// Why the worker could not start, once it has failed.
let startFailure: string | null = null;

// Whether the worker has loaded the core, which builds its session.
let loaded = false;

/** A message the worker sends unprompted; `rpc.ts` handles replies, readiness and log lines itself. */
export type WorkerEvent = Exclude<WorkerMessage['type'], 'reply' | 'ready' | 'log'>;

type EventOf<T extends WorkerEvent> = Extract<WorkerMessage, { type: T }>;

const listeners: { [T in WorkerEvent]?: ((message: EventOf<T>) => void)[] } = {};

/**
 * Call `handler` with every `type` message from the worker. Messages sent once at startup are
 * seen only by modules imported before `initWorker` runs.
 */
export function onWorker<T extends WorkerEvent>(type: T, handler: (message: EventOf<T>) => void): void {
  const list: ((message: EventOf<T>) => void)[] = listeners[type] ??= [];
  list.push(handler);
}

let msgId = 0;
const pending = new Map<number, {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
}>();

export function whenReady(): Promise<void> {
  return readyPromise;
}

// Settles once the worker has loaded the core, rejecting with the reason when it could not.
export function whenStarted(): Promise<void> {
  return started;
}

export function hasSession(): boolean {
  return loaded;
}

export function call<K extends keyof Commands>(
  cmd: K,
  ...args: Parameters<Commands[K]>
): Promise<ReturnType<Commands[K]>> {
  return new Promise<ReturnType<Commands[K]>>((resolve, reject) => {
    if (!worker) {
      reject(new Error(startFailure ?? 'the emulator worker has not been started'));
      return;
    }
    const id = ++msgId;
    pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
    const request: CallRequest = { id, cmd, args };
    worker.postMessage(request);
  });
}

export function initWorker(): void {
  worker = spawnWorker(workerUrl, 'Emulator', (why) => {
    if (loaded) log(why, 'error');
    else failStart(why);
  });
  worker.onmessage = (e: MessageEvent<WorkerMessage>) => {
    const d = e.data;
    switch (d.type) {
      case 'reply': {
        const p = pending.get(d.id);
        if (!p) break;
        pending.delete(d.id);
        if (d.ok) p.resolve(d.result);
        else p.reject(new Error(d.error || 'unknown error'));
        break;
      }
      case 'log':
        log(d.text, d.level);
        break;
      case 'ready':
        // A core that failed to load still reports ready, with the error.
        if (d.error) {
          failStart(`The WebAssembly core did not load (${d.error})`);
          break;
        }
        loaded = true;
        startResolve();
        break;
      default:
        for (const handler of (listeners[d.type] ?? []) as ((message: WorkerMessage) => void)[]) handler(d);
    }
  };
}

// A worker that never started answers nothing, so its calls fail now rather than hang.
function failStart(why: string): void {
  startFailure = why;
  worker?.terminate();
  worker = null;
  for (const p of pending.values()) p.reject(new Error(why));
  pending.clear();
  startReject(new Error(why));
}

// Give the worker the canvas; it paints every frame from then on.
export function attachScreen(canvas: OffscreenCanvas): void {
  worker?.postMessage({ type: 'screen', canvas } satisfies PageMessage, [canvas]);
}
