// Worker logging, forwarded to the page's log instead of `console`.

import type { LogLevel, WorkerMessage } from '../shared/protocol';
import { switch_log_drain } from '@core/switch_wasm.js';
import { coreLoaded } from './wasm';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

export function workerLog(text: string, level: LogLevel = 'log'): void {
  const message: WorkerMessage = { type: 'log', text, level };
  ctx.postMessage(message);
}

// Indexed by the digit that opens each line `switch_log_drain` returns.
const LEVELS: LogLevel[] = ['error', 'warn', 'info', 'log', 'debug'];

// Pass on every line the core has queued, as it wrote them. Returns how many there were.
export function flushLog(): number {
  if (!coreLoaded()) return 0;
  const lines = switch_log_drain();
  for (const line of lines) workerLog(line.slice(1), LEVELS[Number(line[0]) - 1]);
  return lines.length;
}
