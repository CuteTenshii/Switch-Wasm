// Plays the worker's audio ring (`crates/switch-wasm/src/audio.rs`) straight out of its
// shared memory, resampled to the output rate.

// The worklet scope's globals, which the DOM types do not describe.
declare const sampleRate: number;
declare class AudioWorkletProcessor {
  readonly port: MessagePort;
}
declare function registerProcessor(
  name: string,
  processor: new (options: { processorOptions: RingOptions }) => AudioWorkletProcessor,
): void;

export interface RingOptions {
  buffer: SharedArrayBuffer;
  at: number;
  samples: number;
}

// Header words: samples written, samples read, then the format.
const WRITTEN = 0;
const READ = 1;
const FORMAT = 2;
const HEADER_BYTES = 12;
// More queued than this is dropped from the front, so latency stays low.
const MAX_QUEUED_SECONDS = 0.2;
const REPORT_EVERY_SECONDS = 1;
// The speaker downmix of 5.1 to stereo.
const SURROUND = Math.SQRT1_2;

class RingPlayer extends AudioWorkletProcessor {
  private readonly head: Uint32Array;
  private readonly pcm: Int16Array;
  private readonly mask: number;
  // Position between the next two source frames.
  private phase = 0;
  private played = 0;
  private gaps = 0;
  private silent = 0;
  private sinceReport = 0;
  private wasPlaying = false;

  constructor({ processorOptions: { buffer, at, samples } }: { processorOptions: RingOptions }) {
    super();
    this.head = new Uint32Array(buffer, at, 3);
    this.pcm = new Int16Array(buffer, at + HEADER_BYTES, samples);
    this.mask = samples - 1;
  }

  // One interleaved frame as stereo.
  private frame(at: number, channels: number): [number, number] {
    const sample = (c: number) => this.pcm[(at + c) & this.mask] / 32768;
    if (channels === 1) return [sample(0), sample(0)];
    if (channels < 6) return [sample(0), sample(1)];
    return [
      sample(0) + SURROUND * (sample(2) + sample(4)),
      sample(1) + SURROUND * (sample(2) + sample(5)),
    ];
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const [left, right] = outputs[0];
    const format = Atomics.load(this.head, FORMAT);
    const rate = format & 0xffffff;
    const channels = format >>> 24;
    let read = Atomics.load(this.head, READ);
    const written = Atomics.load(this.head, WRITTEN);
    if (!rate || !channels) {
      this.account(0, left.length);
      return true;
    }
    let queued = Math.floor(((written - read) >>> 0) / channels);
    const most = Math.ceil(rate * MAX_QUEUED_SECONDS);
    if (queued > most) {
      read = (read + (queued - most) * channels) >>> 0;
      queued = most;
    }
    const step = rate / sampleRate;
    let filled = 0;
    for (; filled < left.length; filled++) {
      // Two frames are needed to interpolate between them.
      if (queued < 2) break;
      const [l0, r0] = this.frame(read, channels);
      const [l1, r1] = this.frame(read + channels, channels);
      left[filled] = l0 + (l1 - l0) * this.phase;
      right[filled] = r0 + (r1 - r0) * this.phase;
      this.phase += step;
      const whole = Math.floor(this.phase);
      this.phase -= whole;
      read = (read + whole * channels) >>> 0;
      queued -= whole;
    }
    left.fill(0, filled);
    right.fill(0, filled);
    Atomics.store(this.head, READ, read);
    this.account(filled, left.length - filled);
    return true;
  }

  // Count what was played and missed, and say so to the page once a second.
  private account(played: number, missed: number): void {
    const playing = played > 0;
    if (missed && this.wasPlaying) this.gaps++;
    if (playing || this.wasPlaying) this.silent += missed / sampleRate;
    this.wasPlaying = playing;
    this.played += played / sampleRate;
    this.sinceReport += (played + missed) / sampleRate;
    if (this.sinceReport < REPORT_EVERY_SECONDS) return;
    this.port.postMessage({ played: this.played, gaps: this.gaps, silence: this.silent });
    this.played = 0;
    this.gaps = 0;
    this.silent = 0;
    this.sinceReport = 0;
  }
}

registerProcessor('ring-player', RingPlayer);
