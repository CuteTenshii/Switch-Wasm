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
   each file on the host's disk those reads came out of; and the audio, each
   device and renderer the guest has open and the samples between them and
   the page, so a silent title says where its sound stopped.

   Then the problems, each summed over the second: service requests nothing
   answers, draws and compute kernels the GPU refused and why, nvdrv ioctls
   that failed, threads that have done no work for a long time, the
   controller styles the title accepts, and memory as it grows. */

import { fmtCount, fmtSize } from '../shared/format';
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
  /** Guest milliseconds since it last did real work. */
  idleMs: number;
}

/** A request the core answered without an implementation, and how often. */
interface ServiceGap {
  kind: 'refused' | 'missing' | 'stub' | 'ioctl';
  name: string;
  command: number | null;
  calls: number;
}

/** Draws or compute dispatches the GPU refused for one reason. */
interface Refusal {
  kind: 'draw' | 'dispatch';
  reason: string;
  count: number;
}

/** An nvdrv ioctl that failed, and how often. */
interface NvError {
  node: string;
  request: number;
  error: number;
  calls: number;
}

/** One open `audout` device. The counts run from when it was opened. */
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
  /** Appended while the device was stopped: never played. */
  discardedFrames: number;
  /** Descriptors pointing outside their own buffer: never played. */
  unplayableBuffers: number;
}

/** One open audio renderer. The counts run from when it was opened. */
interface AudioRenderer {
  handle: number;
  sampleRate: number;
  started: boolean;
  updates: number;
  renderedFrames: number;
  voices: number;
  voicesPlaying: number;
  /** 0 when no sink the renderer can play has been configured. */
  sinkChannels: number;
}

/** The guest's audio, as `Cpu::audio_activity` reports it. Sample counts are
 *  interleaved samples from the start of the session; `backlog` is what is
 *  queued for the page now. */
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

/** `switch_activity_json`. The GPU counts, `failures` and `audio` run from
 *  the start of the session; `gpu`, `files` and `journal` cover the time
 *  since the last call. */
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
  /** `HidNpadStyleTag` bits: what the title accepts (0 before it says), and
   *  the one style the pad is presented as. */
  input: { supported: number; presented: number };
  refusals: Refusal[];
  gaps: ServiceGap[];
  nvErrors: NvError[];
  problemsDropped: number;
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

/** Each thread's state at the last report, by index, so a thread that sat
 *  still and did not change is not repeated every second. */
const lastThreadState = new Map<number, string>();

/** How long each thread had been idle at the last warning about it, by
 *  index, so a stuck thread is named at 5 s, 30 s and then once a minute
 *  rather than every second. */
const idleWarned = new Map<number, number>();

/** Memory at the last line about it. */
let memoryLogged = { guest: 0, wasm: 0 };

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
  idleWarned.clear();
  memoryLogged = { guest: 0, wasm: 0 };
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
    logAudio(previous.audio, current.audio);
    logProblems(current);
    logIdleThreads(current);
    logInput(previous.input, current.input);
    logMemory();
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

/** A frame count as time, at `rate`: what a person can compare with the
 *  second the report covers. */
function duration(frames: number, rate: number): string {
  if (!rate) return count(frames, 'frame');
  const ms = (frames / rate) * 1000;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function layout(channels: number): string {
  return channels === 1 ? 'mono' : channels === 2 ? 'stereo' : `${channels} channels`;
}

/* The audio: each device and renderer that opened, closed, changed state or
   did anything, and the samples between the guest and the page. Silence has
   a place it happens: no device, a device never started or never fed,
   samples produced and never taken, or taken faster than they come; each of
   those reads differently here. */
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
      // Frames mixed into nothing: the guest's sound goes nowhere.
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

/** What each kind of unanswered request is, in the words a line uses. */
const GAP_KIND: Record<ServiceGap['kind'], string> = {
  refused: 'refused',
  missing: 'no such service',
  stub: 'stubbed',
  ioctl: 'no handler',
};

/** nvdrv's error codes, as `nvdrv.rs` names them. */
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
  // Stubs answer, so a title asking one every frame is ordinary; refusals
  // and missing services are what a stalled title is waiting behind.
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

/** Seconds of idleness a thread is named at: these, then every minute. */
const IDLE_WARNINGS_S = [5, 30];

function nextIdleWarning(after: number): number {
  const fixed = IDLE_WARNINGS_S.find((s) => s > after);
  return fixed ?? (Math.floor(after / 60) + 1) * 60;
}

/* A thread waiting on something for a long time without doing any work is
   either a worker with nothing to do or the reason a title has stalled.
   Which one is up to the reader; this names it and what it waits on. */
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

/** `HidNpadStyleTag` bits, as a player would name the controller. */
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
  workerLog(
    `[input] the title accepts ${styleNames(now.supported)}; the pad is presented as `
    + styleNames(now.presented),
  );
}

/** Memory growth worth a line, and the point it becomes a warning: wasm32
 *  cannot address more than 4 GiB. */
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
