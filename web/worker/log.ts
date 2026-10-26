// Worker logging, forwarded to the page's log instead of `console`.

import type { LogLevel, WorkerMessage } from '../shared/protocol';
import { core, coreLoaded, readString, state } from './wasm';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

export function workerLog(text: string, level: LogLevel = 'log'): void {
  const message: WorkerMessage = { type: 'log', text, level };
  ctx.postMessage(message);
}

// Indexed by the level `switch_log_next` returns; 0 means no line.
const LEVELS: (LogLevel | null)[] = [null, 'error', 'warn', 'info', 'log', 'debug'];

// Pass on every line the core has queued, as it wrote them. Returns how many there were.
export function flushLog(): number {
  if (!coreLoaded()) return 0;
  let lines = 0;
  for (let level = LEVELS[core.switch_log_next(state.handle)]; level;
    level = LEVELS[core.switch_log_next(state.handle)]) {
    const length = core.switch_log_line(0, 0);
    workerLog(readString(length || 1, (buf, cap) => core.switch_log_line(buf, cap)), level);
    lines++;
  }
  return lines;
}
