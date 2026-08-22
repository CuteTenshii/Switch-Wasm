/* ---------- audio ----------

   `audout` hands the guest's PCM over interleaved, at whatever rate and
   channel count it opened the device with. Each pump takes everything that has
   queued up since the last one and schedules it as a single buffer, butted up
   against the end of the previous one, so a continuous stream stays
   continuous. The emulator rarely runs a retail title in real time, so
   underruns are the normal case: the cursor simply restarts a little ahead of
   `currentTime` rather than trying to stretch anything to cover the gap. */

import { call } from './rpc';

let audioCtx: AudioContext | null = null;
let audioCursor = 0;
// One second of 48 kHz stereo, matching the cap the core queues.
const AUDIO_MAX_PULL = 96000;

export function resetAudio(): void {
  audioCursor = 0;
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
    if (!Ctx) return;
    audioCtx = new Ctx();
  }
  // Autoplay policy: a context created before the first gesture starts
  // suspended and stays silent until resumed.
  if (audioCtx.state === 'suspended') await audioCtx.resume();
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
  src.start(start);
  audioCursor = start + buffer.duration;
}
