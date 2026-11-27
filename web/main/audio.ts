// Sound: an AudioWorklet plays the worker's audio ring straight out of shared memory.

import type { RingOptions, WorkletReport } from './pcm-worklet';
import workletUrl from './pcm-worklet.ts?worker&url';
import { log } from './log';
import { onWorker } from './rpc';

let ring: RingOptions | null = null;
let audioCtx: AudioContext | null = null;
// Opening is tried once per page: a failure is reported and sound stays off.
let opening = false;
let unavailable = false;
let format = 0;
let lastFormat = 0;
// Warnings already given, so a persisting state is reported once.
let saidSuspended = false;
// The worklet's reports summed since the output opened, for the Audio debug section.
const played = { seconds: 0, gaps: 0, silence: 0, trimmed: 0, queued: 0 };

// The page's half of the Audio debug section: the browser output and what it has played.
export interface AudioOutput {
  // Why there is no output: not opened yet, or the browser cannot open one.
  missing: 'waiting' | 'unavailable' | null;
  state: AudioContextState | null;
  sampleRate: number;
  // Seconds; `outputLatency` is 0 where the browser does not report it.
  baseLatency: number;
  outputLatency: number;
  played: number;
  gaps: number;
  silence: number;
  trimmed: number;
  queued: number;
}

export function audioOutput(): AudioOutput {
  return {
    missing: audioCtx ? null : unavailable ? 'unavailable' : 'waiting',
    state: audioCtx?.state ?? null,
    sampleRate: audioCtx?.sampleRate ?? 0,
    baseLatency: audioCtx?.baseLatency ?? 0,
    outputLatency: audioCtx?.outputLatency ?? 0,
    played: played.seconds,
    gaps: played.gaps,
    silence: played.silence,
    trimmed: played.trimmed,
    queued: played.queued,
  };
}

onWorker('audio_ring', ({ buffer, at, samples }) => {
  ring = { buffer, at, samples };
});
onWorker('audio_format', ({ format: next }) => {
  format = next;
});

// Each slice: the first sound opens the output, and a held-back one is retried.
export function noteAudio(): void {
  if (!format || !ring) return;
  if (!audioCtx) {
    if (!opening && !unavailable) void open(ring);
    return;
  }
  if (format !== lastFormat) {
    const rate = format & 0xffffff;
    const channels = format >>> 24;
    const layout = channels === 2 ? 'stereo' : channels === 1 ? 'mono' : `${channels} channels`;
    log(`[audio] page: playing ${rate} Hz, ${layout} through a ${audioCtx.sampleRate} Hz output`, 'debug');
    lastFormat = format;
  }
  if (audioCtx.state === 'suspended') void resume(audioCtx);
}

async function open(options: RingOptions): Promise<void> {
  opening = true;
  try {
    const Ctx = window.AudioContext
      || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx || !('audioWorklet' in Ctx.prototype)) {
      unavailable = true;
      log('[audio] this browser has no audio worklets: sound is off', 'warn');
      return;
    }
    const ctx = new Ctx();
    await ctx.audioWorklet.addModule(workletUrl);
    const node = new AudioWorkletNode(ctx, 'ring-player', {
      numberOfInputs: 0,
      outputChannelCount: [2],
      processorOptions: options,
    });
    node.port.onmessage = (e: MessageEvent<WorkletReport>) => {
      const { played: seconds, gaps, silence, trimmed, queued } = e.data;
      played.seconds += seconds;
      played.gaps += gaps;
      played.silence += silence;
      played.trimmed += trimmed;
      played.queued = queued;
      if (!seconds && !gaps) return;
      const parts = [`${seconds.toFixed(2)} s played`];
      if (gaps) {
        parts.push(`${gaps} ${gaps === 1 ? 'gap' : 'gaps'} (${Math.round(silence * 1000)} ms of silence)`);
      }
      log(`[audio] page: ${parts.join(', ')}`, 'debug');
    };
    node.connect(ctx.destination);
    audioCtx = ctx;
  } catch (err) {
    unavailable = true;
    log('[audio] sound is off: the output could not be opened (' + (err as Error).message + ')', 'error');
  } finally {
    opening = false;
  }
}

// Autoplay policy: a context created before a gesture starts suspended.
async function resume(ctx: AudioContext): Promise<void> {
  await ctx.resume();
  if (ctx.state === 'suspended') {
    if (!saidSuspended) {
      log('[audio] page: the browser is holding sound back until you click or press a key', 'warn');
    }
    saidSuspended = true;
  } else if (saidSuspended) {
    log('[audio] page: sound released');
    saidSuspended = false;
  }
}

// For a session being replaced; the worker drops what its guest queued.
export function resetAudio(): void {
  lastFormat = 0;
  saidSuspended = false;
}
