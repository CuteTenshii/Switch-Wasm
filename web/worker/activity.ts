/* What the emulator has been doing, in the page's console (and so in
   DevTools, which it mirrors to).

   Once a second at most, and only for what changed: the frames and the draws,
   clears and copies behind them, and then each surface on its own line, what
   was drawn into it, cleared, copied or blitted between which two, and which
   one each frame presented; every guest thread created, started, paused or
   ended, and each thread's share of the instructions and what it is waiting
   on; each file the guest read or wrote, by name,
   and how much; the guest's path operations (opens, creates, deletes, and the
   lookups that found nothing), one per line, because "the title looked for a
   file and it was not there" is the most common silent failure there is; and
   each file on the host's disk those reads came out of. */

import { fmtSize } from '../shared/format';
import { takeHostIo, type HostIo } from './hostfiles';
import { workerLog } from './log';
import { api, readJson, state } from './wasm';

/** Reads and writes through one guest file or storage since the last reading. */
interface FileActivity {
  name: string;
  reads: number;
  readBytes: number;
  writes: number;
  writeBytes: number;
}

/** One surface's share of what the GPU did since the last reading. `amount`
 *  is vertices for a draw, bytes for a copy or upload, destination pixels for
 *  a blit, and unused for clears and presents. */
interface GpuEntry {
  kind: 'draw' | 'clear' | 'copy' | 'upload' | 'blit' | 'present';
  label: string;
  count: number;
  amount: number;
  failed: number;
}

/** One guest thread, as of the reading. `ran` and `switches` cover the time
 *  since the last one; `state` is what it is doing now, in words. */
interface ThreadActivity {
  index: number;
  handle: number;
  /** 0 is the most urgent, 63 the least. */
  priority: number;
  running: boolean;
  ran: number;
  switches: number;
  entry: string;
  /** Its pc and the frames above it, innermost first. */
  at: string;
  state: string;
  /** Its `nn::os` name, or for a thread never named, the function it runs. */
  name: string | null;
}

/** `switch_activity_json`. The GPU counts and `failures` run from the start
 *  of the session; `gpu`, `files` and `journal` cover the time since the last
 *  call. */
interface Activity {
  frames: number;
  submissions: number;
  draws: number;
  drawsSkipped: number;
  clears: number;
  clearsElided: number;
  copies: number;
  dispatches: number;
  failures: number;
  gpu: GpuEntry[];
  gpuDropped: number;
  files: FileActivity[];
  filesDropped: number;
  threads: ThreadActivity[];
  threadLog: string[];
  threadLogDropped: number;
  journal: string[];
  dropped: number;
}

const REPORT_EVERY_MS = 1000;

/** Room for everything the core holds between two readings: 256 files and 256
 *  path operations of a few hundred bytes each. */
const ACTIVITY_CAP = 256 * 1024;

const ZERO: Activity = {
  frames: 0,
  submissions: 0,
  draws: 0,
  drawsSkipped: 0,
  clears: 0,
  clearsElided: 0,
  copies: 0,
  dispatches: 0,
  failures: 0,
  gpu: [],
  gpuDropped: 0,
  files: [],
  filesDropped: 0,
  threads: [],
  threadLog: [],
  threadLogDropped: 0,
  journal: [],
  dropped: 0,
};

let previous: Activity = ZERO;
let reportedAt = 0;

/** Each thread's state at the last report, by index, so a thread that sat
 *  still and did not change is not repeated every second. */
const lastThreadState = new Map<number, string>();

/** Host files registered since the last report, summed rather than listed:
 *  booting from the NAND registers a couple of hundred system archives at
 *  once, and a line each buries everything else. */
const registered = new Map<string, { count: number; bytes: number }>();

/** Count one host file of kind `what` towards the next report. */
export function noteRegistered(what: string, bytes: number): void {
  const entry = registered.get(what) ?? { count: 0, bytes: 0 };
  entry.count++;
  entry.bytes += bytes;
  registered.set(what, entry);
}

/** Start counting from zero: the session the counts were about is gone. */
export function resetActivity(): void {
  previous = ZERO;
  reportedAt = 0;
  lastThreadState.clear();
  registered.clear();
  takeHostIo();
}

/** Log what changed since the last report, if a second has passed, or at
 *  once when `now` says so. Called after every run slice; never throws,
 *  because a report is not worth a failed slice.
 *
 *  `now` is for a run that has just stopped, halted or faulted: the second
 *  before that is the one worth reading, and waiting out the interval would
 *  lose it, since no slice follows to report it. */
export function reportActivity(now = false): void {
  if (state.handle < 0) return;
  const at = performance.now();
  const elapsed = at - reportedAt;
  if (!now && reportedAt && elapsed < REPORT_EVERY_MS) return;
  try {
    const current = readJson<Activity | null>(
      ACTIVITY_CAP,
      (buf, cap) => api().switch_activity_json(state.handle, buf, cap),
      null,
    );
    if (!current) return;
    logRegistered();
    logGpu(previous, current, reportedAt ? elapsed / 1000 : 0);
    logThreads(current);
    logFs(previous, current);
    for (const io of takeHostIo()) logHostIo(io);
    previous = current;
    reportedAt = at;
  } catch (e) {
    workerLog('[switch-wasm] activity report failed: ' + String(e), 'warn');
  }
}

/** `n` and the noun, singular when there is one. */
function count(n: number, noun: string, plural = noun + 's'): string {
  return `${n} ${n === 1 ? noun : plural}`;
}

/** Only the parts that are not zero, so a quiet second prints nothing. */
function joined(parts: (string | false)[]): string {
  return parts.filter(Boolean).join(', ');
}

function logRegistered(): void {
  for (const [what, { count, bytes }] of registered) {
    workerLog(`[io] registered ${count} ${what} (${fmtSize(bytes)})`);
  }
  registered.clear();
}

function logGpu(before: Activity, now: Activity, seconds: number): void {
  const frames = now.frames - before.frames;
  const draws = now.draws - before.draws;
  const skipped = now.drawsSkipped - before.drawsSkipped;
  const clears = now.clears - before.clears;
  const elided = now.clearsElided - before.clearsElided;
  const copies = now.copies - before.copies;
  const dispatches = now.dispatches - before.dispatches;
  const line = joined([
    frames > 0 && count(frames, 'frame')
      + (seconds ? ` (${(frames / seconds).toFixed(1)}/s)` : ''),
    draws > 0 && count(draws, 'draw'),
    // Called out on its own: a skipped draw is a hole in the frame.
    skipped > 0 && count(skipped, 'draw') + ' skipped',
    clears > 0 && count(clears, 'clear') + (elided ? ` (${elided} elided)` : ''),
    copies > 0 && count(copies, 'copy', 'copies'),
    dispatches > 0 && count(dispatches, 'compute dispatch', 'compute dispatches'),
  ]);
  if (line) workerLog('[gpu] ' + line, skipped > 0 ? 'warn' : undefined);
  for (const entry of now.gpu) logSurface(entry);
  if (now.gpuDropped > 0) {
    workerLog(`[gpu] ...and ${now.gpuDropped} more surfaces than could be listed`);
  }
}

/** One surface: what was done to it, how often, and how much. */
function logSurface(entry: GpuEntry): void {
  const { kind, label, count: n, amount, failed } = entry;
  let line: string;
  switch (kind) {
    case 'draw':
      line = `draws into ${label}: ${count(n, 'draw')}, ${count(amount, 'vertex', 'vertices')}`
        + (failed ? `, ${failed} skipped` : '');
      break;
    case 'clear':
      line = `clears of ${label}: ${n}`;
      break;
    case 'copy':
      line = `copy ${label}: ${count(n, 'copy', 'copies')} (${fmtSize(amount)})`;
      break;
    case 'upload':
      line = `inline uploads to ${label}: ${n} (${fmtSize(amount)})`;
      break;
    case 'blit':
      line = `2D blit ${label}: ${count(n, 'blit')} (${count(amount, 'pixel')})`;
      break;
    case 'present':
      line = `presented ${label}: ${count(n, 'frame')}`;
      break;
  }
  workerLog('[gpu] ' + line, failed ? 'warn' : undefined);
}

/** An instruction count the way a person reads one: 12.3M rather than
 *  12345678. */
function amount(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'G';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

/* The threads: every one created, started, paused or ended since the last
   report, then each thread that ran or changed what it is doing, with its
   share of the instructions. A thread at 99% is the one a stalled title is
   spinning in; one that has been "waiting on" the same thing report after
   report is the one waiting for something that never comes. */
function logThreads(now: Activity): void {
  for (const line of now.threadLog) workerLog('[thread] ' + line);
  if (now.threadLogDropped > 0) {
    workerLog(`[thread] ...and ${now.threadLogDropped} more thread events than could be listed`);
  }
  const total = now.threads.reduce((sum, t) => sum + t.ran, 0);
  for (const thread of now.threads) {
    const changed = lastThreadState.get(thread.index) !== thread.state;
    lastThreadState.set(thread.index, thread.state);
    if (!thread.ran && !changed) continue;
    const share = total ? ` (${Math.round((thread.ran / total) * 100)}%)` : '';
    const ran = thread.ran
      ? `${amount(thread.ran)} instructions${share}, ${count(thread.switches, 'switch', 'switches')}`
      : 'did not run';
    const who = thread.name ? `${thread.name}, via ${thread.entry}` : thread.entry;
    workerLog(
      `[thread] ${thread.index}${thread.running ? '*' : ''} (${who}, handle `
        + `${thread.handle.toString(16)}, priority ${thread.priority}): ${ran}; ${thread.state}; `
        + thread.at,
    );
  }
}

function logFs(before: Activity, now: Activity): void {
  for (const file of now.files) {
    const line = joined([
      file.reads > 0 && `${count(file.reads, 'read')} (${fmtSize(file.readBytes)})`,
      file.writes > 0 && `${count(file.writes, 'write')} (${fmtSize(file.writeBytes)})`,
    ]);
    if (line) workerLog(`[fs] ${file.name}: ${line}`);
  }
  if (now.filesDropped > 0) {
    workerLog(`[fs] ...and ${now.filesDropped} more files than could be listed`);
  }
  for (const entry of now.journal) workerLog('[fs] ' + entry);
  if (now.dropped > 0) {
    workerLog(`[fs] ...and ${now.dropped} more path operations than could be listed`);
  }
  const failures = now.failures - before.failures;
  if (failures > 0) workerLog(`[fs] ${failures} requests failed`);
}

function logHostIo(io: HostIo): void {
  const line = joined([
    `${count(io.reads, 'read')} (${fmtSize(io.bytes)})`,
    io.diskBytes > 0 && `${fmtSize(io.diskBytes)} read from disk`
      + (io.chunkMisses ? ` (${io.chunkMisses} chunk misses)` : ''),
    io.diskBytes === 0 && 'all from cache',
    io.failures > 0 && `${io.failures} failed`,
  ]);
  workerLog(`[io] ${io.file}: ${line}`, io.failures > 0 ? 'warn' : undefined);
}
