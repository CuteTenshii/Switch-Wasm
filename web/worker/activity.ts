// The once-a-second activity report. The core writes its own part into the log; the
// worker adds what only it sees: host file reads and archive registrations.

import { fmtSize } from '../shared/format';
import { takeHostIo, type HostIo } from './hostfiles';
import { flushLog, workerLog } from './log';
import { core, state } from './wasm';

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
  if (state.handle < 0) return;
  const at = performance.now();
  const elapsed = at - reportedAt;
  if (!now && reportedAt && elapsed < REPORT_EVERY_MS) return;
  try {
    logRegistered();
    core.switch_log_activity(state.handle, reportedAt ? elapsed / 1000 : 0);
    flushLog();
    logHostFiles(takeHostIo());
    reportedAt = at;
  } catch (e) {
    workerLog('[switch-wasm] activity report failed: ' + String(e), 'warn');
  }
}

function count(n: number, noun: string, plural = noun + 's'): string {
  return `${n} ${n === 1 ? noun : plural}`;
}

// Only non-zero parts, so a quiet second prints nothing.
function joined(parts: (string | false)[]): string {
  return parts.filter(Boolean).join(', ');
}

function logRegistered(): void {
  for (const [what, { count, bytes }] of registered) {
    workerLog(`[io] registered ${count} ${what} (${fmtSize(bytes)})`, 'debug');
  }
  registered.clear();
}

// Up to this much read per file per second, with no failures, is a header read.
const SMALL_IO = 4 * 1024;

function logHostFiles(files: HostIo[]): void {
  const small = files.filter((io) => io.bytes <= SMALL_IO && !io.failures);
  for (const io of files) {
    if (!small.includes(io)) logHostIo(io);
  }
  if (small.length === 1) logHostIo(small[0]);
  if (small.length < 2) return;
  const sum = (key: 'reads' | 'bytes' | 'diskBytes' | 'chunkMisses') =>
    small.reduce((total, io) => total + io[key], 0);
  const disk = sum('diskBytes');
  workerLog(`[io] ${small.length} files with small reads: ` + joined([
    `${count(sum('reads'), 'read')} (${fmtSize(sum('bytes'))})`,
    disk > 0 && `${fmtSize(disk)} read from disk (${count(sum('chunkMisses'), 'chunk miss', 'chunk misses')})`,
    disk === 0 && 'all from cache',
  ]), 'debug');
}

function logHostIo(io: HostIo): void {
  const line = joined([
    `${count(io.reads, 'read')} (${fmtSize(io.bytes)})`,
    io.diskBytes > 0 && `${fmtSize(io.diskBytes)} read from disk`
    + (io.chunkMisses ? ` (${io.chunkMisses} chunk misses)` : ''),
    io.diskBytes === 0 && 'all from cache',
    io.failures > 0 && `${io.failures} failed`,
  ]);
  workerLog(`[io] ${io.file}: ${line}`, io.failures > 0 ? 'warn' : 'debug');
}
