// The once-a-second activity report. The core writes its own part into the log; the
// worker adds what only it sees: archive registrations.

import { fmtSize } from '../shared/format';
import { flushLog, workerLog } from './log';
import { switch_log_activity } from '@core/switch_wasm.js';

const REPORT_EVERY_MS = 1000;

let reportedAt = 0;

// Host file registrations, summed rather than listed (NAND boot registers hundreds).
const registered = new Map<string, { count: number; bytes: number }>();

export function noteRegistered(what: string, bytes: number): void {
  const entry = registered.get(what) ?? { count: 0, bytes: 0 };
  entry.count++;
  entry.bytes += bytes;
  registered.set(what, entry);
}

// Log what changed if a second has passed, or at once when `now` (the run just
// stopped). Never throws.
export function reportActivity(now = false): void {
  const at = performance.now();
  const elapsed = at - reportedAt;
  if (!now && reportedAt && elapsed < REPORT_EVERY_MS) return;
  try {
    logRegistered();
    switch_log_activity(reportedAt ? elapsed / 1000 : 0);
    flushLog();
    reportedAt = at;
  } catch (e) {
    workerLog('[switch-wasm] activity report failed: ' + String(e), 'warn');
  }
}

function logRegistered(): void {
  for (const [what, { count, bytes }] of registered) {
    workerLog(`[io] registered ${count} ${what} (${fmtSize(bytes)})`, 'debug');
  }
  registered.clear();
}
