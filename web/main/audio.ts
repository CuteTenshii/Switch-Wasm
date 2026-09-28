/* audio

   `audout` hands the guest's PCM over interleaved, at whatever rate and
   channel count it opened the device with. Each pump takes everything that has
   queued up since the last one and schedules it as a single buffer, butted up
   against the end of the previous one, so a continuous stream stays
   continuous. The emulator rarely runs a retail title in real time, so
   underruns are the normal case: the cursor simply restarts a little ahead of
   `currentTime` rather than trying to stretch anything to cover the gap.

   What the page did with the samples goes to the log, once a second at most:
   how much it played and how much silence it left between buffers, and the
   two ways it can play nothing at all, a browser without Web Audio and a
   context the autoplay policy keeps suspended until the user interacts. */

import { log } from './log';
import { call } from './rpc';

let audioCtx: AudioContext | null = null;
let audioCursor = 0;
// One second of 48 kHz stereo, matching the cap the core queues.
const AUDIO_MAX_PULL = 96000;

const REPORT_EVERY_MS = 1000;

/** What was played since the last report, in seconds, and the gaps between
 *  buffers: each one a stretch of silence the page left because the next
 *  samples arrived after the previous ones had finished. */
let played = 0;
let gaps = 0;
let silence = 0;
let reportedAt = 0;
let lastFormat = '';
/** Warnings already given, so a state that persists is said once. */
let saidSuspended = false;
let saidUnsupported = false;

export function resetAudio(): void {
  audioCursor = 0;
  played = 0;
  gaps = 0;
  silence = 0;
  reportedAt = 0;
  lastFormat = '';
  saidSuspended = false;
}

function reportPlayback(): void {
  const now = performance.now();
  if (reportedAt && now - reportedAt < REPORT_EVERY_MS) return;
  reportedAt = now;
  if (!played && !gaps) return;
  const parts = [`${played.toFixed(2)} s played`];
  if (gaps) parts.push(`${gaps} ${gaps === 1 ? 'gap' : 'gaps'} (${Math.round(silence * 1000)} ms of silence)`);
  log(`[audio] page: ${parts.join(', ')}`);
  played = 0;
  gaps = 0;
  silence = 0;
}

export async function pumpAudio(): Promise<void> {
  const packed = await call('audio_format');
  if (!packed) return; // nothing has opened an audio device yet
  const rate = packed & 0x00ffffff;
  const channels = packed >>> 24;
  if (!rate || !channels) return;
  const bytes = await call('audio_pull', AUDIO_MAX_PULL);
  if (!bytes || bytes.length < channels * 2) return;
  if (!audioCtx) {
    const Ctx = window.AudioContext
      || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) {
      if (!saidUnsupported) log('[audio] this browser has no Web Audio: sound is off', 'warn');
      saidUnsupported = true;
      return;
    }
    audioCtx = new Ctx();
  }
  const format = `${rate} Hz, ${channels === 2 ? 'stereo' : channels === 1 ? 'mono' : `${channels} channels`}`;
  if (format !== lastFormat) {
    log(`[audio] page: playing ${format} through a ${audioCtx.sampleRate} Hz output`);
    lastFormat = format;
  }
  // Autoplay policy: a context created before the first gesture starts
  // suspended and stays silent until resumed.
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  if (audioCtx.state === 'suspended') {
    if (!saidSuspended) {
      log('[audio] page: the browser is holding sound back until you click or press a key', 'warn');
    }
    saidSuspended = true;
  } else if (saidSuspended) {
    log('[audio] page: sound released');
    saidSuspended = false;
  }
  const pcm = bytes.byteOffset % 2
    ? new Int16Array(bytes.slice().buffer)
    : new Int16Array(bytes.buffer, bytes.byteOffset, bytes.length >> 1);
  const frames = Math.floor(pcm.length / channels);
  if (!frames) return;
  const buffer = audioCtx.createBuffer(channels, frames, rate);
  for (let c = 0; c < channels; c++) {
    const out = buffer.getChannelData(c);
    for (let i = 0; i < frames; i++) out[i] = pcm[i * channels + c] / 32768;
  }
  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(audioCtx.destination);
  // Schedule a little ahead of now so a late buffer is not clipped, then keep
  // every later one flush against its predecessor.
  const start = Math.max(audioCtx.currentTime + 0.05, audioCursor);
  if (audioCursor && start > audioCursor) {
    gaps++;
    silence += start - audioCursor;
  }
  src.start(start);
  audioCursor = start + buffer.duration;
  played += buffer.duration;
  reportPlayback();
}
