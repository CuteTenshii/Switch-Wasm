// Sound: an AudioWorklet plays the worker's audio ring straight out of shared memory.

import type { RingOptions } from './pcm-worklet';
import workletUrl from './pcm-worklet.ts?worker&url';
import { log } from './log';

let ring: RingOptions | null = null;
let audioCtx: AudioContext | null = null;
// Opening is tried once per page: a failure is reported and sound stays off.
let opening = false;
let unavailable = false;
let lastFormat = 0;
// Warnings already given, so a persisting state is reported once.
let saidSuspended = false;

export function setAudioRing(options: RingOptions): void {
  ring = options;
}

// Each slice's audio format; the first sound opens the output.
export function noteAudio(format: number): void {
  if (!format || !ring) return;
  if (!audioCtx) {
    if (!opening && !unavailable) void open(ring);
    return;
  }
  if (format !== lastFormat) {
    const rate = format & 0xffffff;
    const channels = format >>> 24;
    const layout = channels === 2 ? 'stereo' : channels === 1 ? 'mono' : `${channels} channels`;
    log(`[audio] page: playing ${rate} Hz, ${layout} through a ${audioCtx.sampleRate} Hz output`);
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
    node.port.onmessage = (e: MessageEvent<{ played: number; gaps: number; silence: number }>) => {
      const { played, gaps, silence } = e.data;
      if (!played && !gaps) return;
      const parts = [`${played.toFixed(2)} s played`];
      if (gaps) {
        parts.push(`${gaps} ${gaps === 1 ? 'gap' : 'gaps'} (${Math.round(silence * 1000)} ms of silence)`);
      }
      log(`[audio] page: ${parts.join(', ')}`);
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

// For a session being replaced: drop what its guest queued.
export function resetAudio(): void {
  lastFormat = 0;
  saidSuspended = false;
  if (!ring) return;
  const head = new Uint32Array(ring.buffer, ring.at, 2);
  Atomics.store(head, 1, Atomics.load(head, 0));
}
