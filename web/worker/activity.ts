// Once-a-second activity report in the page console: frames, surfaces, threads,
// files, path operations, audio, and problems (unanswered requests, refused draws,
// failed ioctls, idle threads, controller styles, memory growth).

import { fmtCount, fmtSize } from '../shared/format';
import { takeHostIo, type HostIo } from './hostfiles';
import { workerLog } from './log';
import { api, readJson, state } from './wasm';

// Reads and writes through one guest file or storage since the last reading.
interface FileActivity {
  name: string;
  reads: number;
  readBytes: number;
  writes: number;
  writeBytes: number;
}

// One surface's GPU work since the last reading. `amount` is vertices for a draw,
// bytes for a copy or upload, destination pixels for a blit.
interface GpuEntry {
  kind: 'draw' | 'clear' | 'copy' | 'upload' | 'blit' | 'present';
  label: string;
  count: number;
  amount: number;
  failed: number;
}

// One guest thread. `ran` and `switches` cover the time since the last reading.
interface ThreadActivity {
  index: number;
  handle: number;
  // 0 is the most urgent, 63 the least.
  priority: number;
  running: boolean;
  ran: number;
  switches: number;
  entry: string;
  // Its pc and the frames above it, innermost first.
  at: string;
  state: string;
  // Its `nn::os` name, or the function an unnamed thread runs.
  name: string | null;
  // Guest milliseconds since it last did real work.
  idleMs: number;
}

interface ServiceGap {
  kind: 'refused' | 'missing' | 'stub' | 'ioctl';
  name: string;
  command: number | null;
  calls: number;
}

// Draws or dispatches the GPU refused for one reason.
interface Refusal {
  kind: 'draw' | 'dispatch';
  reason: string;
  count: number;
}

interface NvError {
  node: string;
  request: number;
  error: number;
  calls: number;
}

// One open `audout` device; counts run from when it opened.
interface AudioOutput {
  handle: number;
  sampleRate: number;
  channels: number;
  started: boolean;
  volume: number;
  appendedBuffers: number;
  appendedFrames: number;
  releasedBuffers: number;
  pendingBuffers: number;
  // Appended while stopped: never played.
  discardedFrames: number;
  // Descriptors outside their own buffer: never played.
  unplayableBuffers: number;
}

// One open audio renderer; counts run from when it opened.
interface AudioRenderer {
  handle: number;
  sampleRate: number;
  started: boolean;
  updates: number;
  renderedFrames: number;
  voices: number;
  voicesPlaying: number;
  // 0 when no playable sink is configured.
  sinkChannels: number;
}

// `Cpu::audio_activity`. Sample counts are interleaved, from session start;
// `backlog` is what is queued for the page now.
interface Audio {
  sampleRate: number;
  channels: number;
  samplesProduced: number;
  samplesTaken: number;
  samplesDropped: number;
  backlog: number;
  outputs: AudioOutput[];
  renderers: AudioRenderer[];
}

// `switch_activity_json`. GPU counts, `failures` and `audio` run from session
// start; `gpu`, `files` and `journal` cover the time since the last call.
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
  audio: Audio;
  // `HidNpadStyleTag` bits: what the title accepts (0 before it says) and the pad's style.
  input: { supported: number; presented: number };
  refusals: Refusal[];
  gaps: ServiceGap[];
  nvErrors: NvError[];
  problemsDropped: number;
}

const REPORT_EVERY_MS = 1000;

// Room for 256 files and 256 path operations between readings.
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
  audio: {
    sampleRate: 0,
    channels: 0,
    samplesProduced: 0,
    samplesTaken: 0,
    samplesDropped: 0,
    backlog: 0,
    outputs: [],
    renderers: [],
  },
  input: { supported: 0, presented: 0 },
  refusals: [],
  gaps: [],
  nvErrors: [],
  problemsDropped: 0,
};

let previous: Activity = ZERO;
let reportedAt = 0;

// Each thread's last reported state, so unchanged threads aren't repeated.
const lastThreadState = new Map<number, string>();

// Idle time at each thread's last warning: named at 5 s, 30 s, then every minute.
const idleWarned = new Map<number, number>();

let memoryLogged = { guest: 0, wasm: 0 };

// Host file registrations, summed rather than listed (NAND boot registers hundreds).
const registered = new Map<string, { count: number; bytes: number }>();

export function noteRegistered(what: string, bytes: number): void {
  const entry = registered.get(what) ?? { count: 0, bytes: 0 };
  entry.count++;
  entry.bytes += bytes;
  registered.set(what, entry);
}

// Reset: the session the counts were about is gone.
export function resetActivity(): void {
  previous = ZERO;
  reportedAt = 0;
  lastThreadState.clear();
  idleWarned.clear();
  memoryLogged = { guest: 0, wasm: 0 };
  registered.clear();
  takeHostIo();
}

// Log what changed if a second has passed, or at once when `now` (the run just
// stopped). Never throws.
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
    logAudio(previous.audio, current.audio);
    logProblems(current);
    logIdleThreads(current);
    logInput(previous.input, current.input);
    logMemory();
    logHostFiles(takeHostIo());
    previous = current;
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
    // A skipped draw is a hole in the frame.
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

// Thread lifecycle events, then each thread that ran or changed state.
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
      ? `${fmtCount(thread.ran)} instructions${share}, ${count(thread.switches, 'switch', 'switches')}`
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

// A frame count as time at `rate`.
function duration(frames: number, rate: number): string {
  if (!rate) return count(frames, 'frame');
  const ms = (frames / rate) * 1000;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function layout(channels: number): string {
  return channels === 1 ? 'mono' : channels === 2 ? 'stereo' : `${channels} channels`;
}

// Audio devices and renderers that changed, and samples between guest and page.
function logAudio(before: Audio, now: Audio): void {
  const outputsBefore = new Map(before.outputs.map((o) => [o.handle, o]));
  for (const o of now.outputs) {
    const was = outputsBefore.get(o.handle);
    outputsBefore.delete(o.handle);
    const name = `audout ${o.handle.toString(16)}`;
    if (!was) {
      workerLog(`[audio] ${name} opened: ${o.sampleRate} Hz ${layout(o.channels)}`);
    }
    const buffers = o.appendedBuffers - (was?.appendedBuffers ?? 0);
    const frames = o.appendedFrames - (was?.appendedFrames ?? 0);
    const released = o.releasedBuffers - (was?.releasedBuffers ?? 0);
    const discarded = o.discardedFrames - (was?.discardedFrames ?? 0);
    const unplayable = o.unplayableBuffers - (was?.unplayableBuffers ?? 0);
    const stateChanged = !was || was.started !== o.started || was.volume !== o.volume;
    const line = joined([
      buffers > 0 && `${count(buffers, 'buffer')} (${duration(frames, o.sampleRate)}) appended`,
      released > 0 && `${released} released`,
      o.pendingBuffers > 0 && buffers > 0 && `${o.pendingBuffers} pending`,
    ]);
    if (line || stateChanged) {
      const state = `${o.started ? 'started' : 'stopped'}`
        + (o.volume !== 1 ? `, volume ${Math.round(o.volume * 100)}%` : '');
      workerLog(`[audio] ${name} ${state}` + (line ? `: ${line}` : ''));
    }
    if (discarded > 0) {
      workerLog(
        `[audio] ${name}: ${duration(discarded, o.sampleRate)} appended while stopped, never played`,
        'warn',
      );
    }
    if (unplayable > 0) {
      workerLog(
        `[audio] ${name}: ${count(unplayable, 'buffer')} described outside itself, not played`,
        'warn',
      );
    }
  }
  for (const gone of outputsBefore.keys()) {
    workerLog(`[audio] audout ${gone.toString(16)} closed`);
  }

  const renderersBefore = new Map(before.renderers.map((r) => [r.handle, r]));
  for (const r of now.renderers) {
    const was = renderersBefore.get(r.handle);
    renderersBefore.delete(r.handle);
    const name = `renderer ${r.handle.toString(16)}`;
    if (!was) {
      workerLog(`[audio] ${name} opened: ${r.sampleRate} Hz, ${count(r.voices, 'voice')}`);
    }
    const frames = r.renderedFrames - (was?.renderedFrames ?? 0);
    const updates = r.updates - (was?.updates ?? 0);
    const stateChanged = !was || was.started !== r.started || was.sinkChannels !== r.sinkChannels
      || was.voicesPlaying !== r.voicesPlaying;
    if (frames > 0 || updates > 0 || stateChanged) {
      const sink = r.sinkChannels ? layout(r.sinkChannels) : 'no playable sink';
      const line = joined([
        frames > 0 && `${count(frames, 'frame')} rendered`,
        updates > 0 && count(updates, 'update'),
        `${r.voicesPlaying} of ${count(r.voices, 'voice')} playing`,
        sink,
      ]);
      // Frames mixed into nothing.
      const silent = r.started && frames > 0 && !r.sinkChannels;
      workerLog(
        `[audio] ${name} ${r.started ? 'started' : 'stopped'}: ${line}`,
        silent ? 'warn' : undefined,
      );
    }
  }
  for (const gone of renderersBefore.keys()) {
    workerLog(`[audio] renderer ${gone.toString(16)} closed`);
  }

  const produced = now.samplesProduced - before.samplesProduced;
  const taken = now.samplesTaken - before.samplesTaken;
  const dropped = now.samplesDropped - before.samplesDropped;
  if (produced > 0 || taken > 0 || dropped > 0) {
    const perFrame = Math.max(now.channels, 1);
    const line = joined([
      `${duration(produced / perFrame, now.sampleRate)} produced`,
      `${duration(taken / perFrame, now.sampleRate)} taken by the page`,
      now.backlog > 0 && `${duration(now.backlog / perFrame, now.sampleRate)} queued`,
    ]);
    workerLog(`[audio] ${now.sampleRate} Hz ${layout(now.channels)}: ${line}`);
  }
  if (dropped > 0) {
    const perFrame = Math.max(now.channels, 1);
    workerLog(
      `[audio] ${duration(dropped / perFrame, now.sampleRate)} dropped: the page fell a second behind`,
      'warn',
    );
  }
}

const GAP_KIND: Record<ServiceGap['kind'], string> = {
  refused: 'refused',
  missing: 'no such service',
  stub: 'stubbed',
  ioctl: 'no handler',
};

// nvdrv's error codes, as `nvdrv.rs` names them.
const NV_ERROR: Record<number, string> = {
  1: 'not implemented',
  2: 'not supported',
  4: 'bad parameter',
  6: 'insufficient memory',
  8: 'invalid state',
  0x30006: 'config variable not found',
};

function hex(n: number): string {
  return '0x' + (n >>> 0).toString(16);
}

function logProblems(now: Activity): void {
  // Stubs answering every frame are ordinary; refusals are what stall a title.
  const asked = (kinds: ServiceGap['kind'][]) => now.gaps
    .filter((gap) => kinds.includes(gap.kind))
    .map((gap) => `${gap.name}${gap.command === null ? '' : ` cmd ${gap.command}`} `
      + `(${GAP_KIND[gap.kind]}) x${gap.calls}`);
  const failing = asked(['refused', 'missing', 'ioctl']);
  if (failing.length) workerLog('[ipc] unanswered: ' + failing.join(', '), 'warn');
  const stubbed = asked(['stub']);
  if (stubbed.length) workerLog('[ipc] answered by stubs: ' + stubbed.join(', '));
  for (const refusal of now.refusals) {
    const what = refusal.kind === 'draw'
      ? count(refusal.count, 'draw')
      : count(refusal.count, 'compute dispatch', 'compute dispatches');
    workerLog(`[gpu] ${what} refused: ${refusal.reason}`, 'warn');
  }
  for (const e of now.nvErrors) {
    const why = NV_ERROR[e.error] ?? `error ${hex(e.error)}`;
    workerLog(
      `[nv] ${e.node} ioctl ${hex(e.request)} (nr ${hex(e.request & 0xff)}) failed: ${why} x${e.calls}`,
      'warn',
    );
  }
  if (now.problemsDropped > 0) {
    workerLog(`[ipc] ...and ${now.problemsDropped} more problems than could be listed`);
  }
}

// Idle seconds a thread is named at, then every minute.
const IDLE_WARNINGS_S = [5, 30];

function nextIdleWarning(after: number): number {
  const fixed = IDLE_WARNINGS_S.find((s) => s > after);
  return fixed ?? (Math.floor(after / 60) + 1) * 60;
}

// Name threads waiting a long time without work, and what they wait on.
function logIdleThreads(now: Activity): void {
  for (const thread of now.threads) {
    const idle = Math.floor(thread.idleMs / 1000);
    const warned = idleWarned.get(thread.index) ?? 0;
    if (idle < warned) idleWarned.delete(thread.index);
    if (!thread.state.startsWith('waiting') || idle < nextIdleWarning(idleWarned.get(thread.index) ?? 0)) {
      continue;
    }
    idleWarned.set(thread.index, idle);
    const who = thread.name ? `${thread.index} (${thread.name})` : `${thread.index} (${thread.entry})`;
    workerLog(`[thread] ${who} has done no work for ${idle} s of guest time; ${thread.state}`);
  }
}

// `HidNpadStyleTag` bits, as a player would name the controller.
const NPAD_STYLES: [number, string][] = [
  [1 << 0, 'Pro Controller'],
  [1 << 1, 'handheld'],
  [1 << 2, 'Joy-Con pair'],
  [1 << 3, 'left Joy-Con'],
  [1 << 4, 'right Joy-Con'],
  [1 << 5, 'GameCube controller'],
  [1 << 6, 'Poke Ball Plus'],
  [1 << 7, 'NES controller'],
  [1 << 8, 'handheld NES controllers'],
  [1 << 9, 'SNES controller'],
  [1 << 10, 'N64 controller'],
  [1 << 11, 'Sega Genesis controller'],
];

function styleNames(bits: number): string {
  const names = NPAD_STYLES.filter(([bit]) => bits & bit).map(([, name]) => name);
  return names.length ? names.join(', ') : 'nothing this console can present';
}

function logInput(before: Activity['input'], now: Activity['input']): void {
  if (now.supported === before.supported && now.presented === before.presented) return;
  if (!now.supported) return;
  // The handheld slot carries every button, so accepting it means full input.
  const handheld = now.supported & (1 << 1) ? ', and the handheld slot has every button' : '';
  workerLog(
    `[input] the title accepts ${styleNames(now.supported)}; player 1 is presented as `
    + styleNames(now.presented) + handheld,
  );
}

// Memory growth worth a line, and the warning point (wasm32 caps at 4 GiB).
const MEMORY_STEP = 64 * 1024 * 1024;
const WASM_WARN = 3.5 * 1024 * 1024 * 1024;

function logMemory(): void {
  if (state.handle < 0) return;
  const guest = Number(api().switch_guest_ram(state.handle));
  const wasm = api().memory.buffer.byteLength;
  if (Math.abs(guest - memoryLogged.guest) < MEMORY_STEP && Math.abs(wasm - memoryLogged.wasm) < MEMORY_STEP) {
    return;
  }
  memoryLogged = { guest, wasm };
  const near = wasm >= WASM_WARN;
  workerLog(
    `[memory] guest RAM ${fmtSize(guest)}, WebAssembly heap ${fmtSize(wasm)} of 4 GiB`
    + (near ? ': close to the limit, the next large allocation may fail' : ''),
    near ? 'warn' : undefined,
  );
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
  ]));
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
