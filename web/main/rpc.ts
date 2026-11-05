// Promise-based RPC to the emulator worker over postMessage.

import type {
  CallRequest, Commands, ErrorDialog, KeyboardRequest, SliceReport, WorkerMessage,
} from '../shared/protocol';
import { setAudioFormat, setAudioRing } from './audio';
import { log } from './log';

let worker: Worker | null = null;
let ready = false;
let readyResolve!: () => void;
const readyPromise = new Promise<void>((r) => {
  readyResolve = r;
});

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

export function hasSession(): boolean {
  return loaded;
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
      case 'audio_ring':
        setAudioRing(d);
        break;
      case 'audio_format':
        setAudioFormat(d.format);
        break;
      case 'nand_progress':
        nandProgress?.(d.done, d.total);
        break;
      case 'ready':
        ready = true;
        // A core that failed to load still reports ready, with the error.
        if (d.error) log('core failed to load: ' + d.error, 'error');
        else loaded = true;
        readyResolve();
        break;
    }
  };
  worker.onerror = (e) => {
    readyResolve();
    log('worker error: ' + e.message, 'error');
  };
}

// Give the worker the canvas; it paints every frame from then on.
export function attachScreen(canvas: OffscreenCanvas): void {
  worker?.postMessage({ type: 'screen', canvas }, [canvas]);
}

// A panic traps as `unreachable`; `switch_last_error` recovers the message the hook kept.
export async function explainFailure(err: unknown): Promise<string> {
  let why = err instanceof Error ? err.message : String(err);
  try {
    const captured = await call('last_error');
    if (captured && !why.includes(captured)) why += ' - ' + captured;
  } catch {
    // The module may be too far gone to answer.
  }
  return why;
}
