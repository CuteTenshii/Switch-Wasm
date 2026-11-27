// Worker logging, forwarded to the page's log instead of `console`.

import type { LogLevel } from '../shared/protocol';
import { switch_log_drain } from '@core/switch_wasm.js';
import { postToPage } from './post';
import { coreLoaded } from './wasm';

export function workerLog(text: string, level: LogLevel = 'log'): void {
  postToPage({ type: 'log', text, level });
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
