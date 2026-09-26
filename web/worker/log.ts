/* The worker's half of the page's log.

   Anything the worker has to say goes to the page, which logs it on its own
   console and mirrors it into DevTools in one place. A `console` call made
   here would reach DevTools alone, and the two consoles would disagree about
   what happened. */

import type { LogClass, WorkerMessage } from '../shared/protocol';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

export function workerLog(text: string, cls?: LogClass): void {
  const message: WorkerMessage = { type: 'log', text, cls };
  ctx.postMessage(message);
}
