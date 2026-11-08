// Promise-based RPC to the emulator worker over postMessage.

import type {
  CallRequest, Commands, ErrorDialog, KeyboardRequest, NetState, SliceReport, WorkerMessage,
} from '../shared/protocol';
import { workerErrorText } from '../shared/workererror';
import { setAudioFormat, setAudioRing } from './audio';
import { log } from './log';
import { setMailbox } from './mailbox';

let worker: Worker | null = null;
let ready = false;
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

// The run loop's handlers for what the worker's loop reports; set by `runloop.ts`.
let loopEvents: {
  slice: (report: SliceReport) => void;
  failed: (error: string) => void;
  keyboard: (request: KeyboardRequest) => void;
  errorDialog: (dialog: ErrorDialog) => void;
  profilesEdited: () => void;
} | null = null;

let netState: ((state: NetState) => void) | null = null;

export function onNetState(handler: NonNullable<typeof netState>): void {
  netState = handler;
}

let nandProgress: ((done: number, total: number) => void) | null = null;

export function onNandProgress(handler: NonNullable<typeof nandProgress>): void {
  nandProgress = handler;
}

export function onLoopEvents(handlers: NonNullable<typeof loopEvents>): void {
  loopEvents = handlers;
}

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

// `{ type: 'module' }` must match Vite's `worker.format: 'es'`.
export function initWorker(): void {
  worker = new Worker(new URL('../worker/index.ts', import.meta.url), {
    type: 'module',
    name: 'Emulator',
  });
  worker.onmessage = (e: MessageEvent<WorkerMessage>) => {
    const d = e.data;
    if (!('type' in d)) {
      const p = pending.get(d.id);
      if (!p) return;
      pending.delete(d.id);
      if (d.ok) p.resolve(d.result);
      else p.reject(new Error(d.error || 'unknown error'));
      return;
    }
    switch (d.type) {
      case 'log':
        log(d.text, d.level);
        break;
      case 'slice':
        loopEvents?.slice(d.report);
        break;
      case 'loop_failed':
        loopEvents?.failed(d.error);
        break;
      case 'keyboard':
        loopEvents?.keyboard(d);
        break;
      case 'error_dialog':
        loopEvents?.errorDialog(d);
        break;
      case 'profiles_edited':
        loopEvents?.profilesEdited();
        break;
      case 'input_mailbox':
        setMailbox(d.buffer, d.at);
        break;
      case 'audio_ring':
        setAudioRing(d);
        break;
      case 'audio_format':
        setAudioFormat(d.format);
        break;
      case 'net_state':
        netState?.(d.state);
        break;
      case 'nand_progress':
        nandProgress?.(d.done, d.total);
        break;
      case 'ready':
        // A core that failed to load still reports ready, with the error.
        if (d.error) {
          failStart(`The WebAssembly core did not load (${d.error})`);
          break;
        }
        ready = true;
        loaded = true;
        startResolve();
        break;
    }
  };
  worker.onerror = (e) => {
    if (ready) log('The emulator worker ' + workerErrorText(e), 'error');
    else failStart('The emulator worker ' + workerErrorText(e));
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
  worker?.postMessage({ type: 'screen', canvas }, [canvas]);
}
