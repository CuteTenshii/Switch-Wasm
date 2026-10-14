// Worker logging, forwarded to the page's log instead of `console`.

import type { LogLevel, WorkerMessage } from '../shared/protocol';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

export function workerLog(text: string, level: LogLevel = 'log'): void {
  const message: WorkerMessage = { type: 'log', text, level };
  ctx.postMessage(message);
}
