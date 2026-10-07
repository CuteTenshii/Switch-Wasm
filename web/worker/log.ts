// Worker logging, forwarded to the page's log instead of `console`.

import type { LogClass, WorkerMessage } from '../shared/protocol';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

export function workerLog(text: string, cls?: LogClass): void {
  const message: WorkerMessage = { type: 'log', text, cls };
  ctx.postMessage(message);
}
