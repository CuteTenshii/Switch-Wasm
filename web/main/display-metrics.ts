// Timing accumulated by one repeated stage of the display path.
export interface DisplayTiming {
  count: number;
  last: number;
  mean: number;
  max: number;
}

interface MutableTiming {
  count: number;
  last: number;
  total: number;
  max: number;
}

export interface DisplayMetrics {
  guestFrames: number;
  canvasUpdates: number;
  skippedFrames: number;
  runSlice: DisplayTiming;
  canvasWrite: DisplayTiming;
}

const timing = (): MutableTiming => ({ count: 0, last: 0, total: 0, max: 0 });

let guestFrames = 0;
let canvasUpdates = 0;
let skippedFrames = 0;
let runSlice = timing();
let canvasWrite = timing();

function record(sample: MutableTiming, ms: number): void {
  sample.count++;
  sample.last = ms;
  sample.total += ms;
  sample.max = Math.max(sample.max, ms);
}

function read(sample: MutableTiming): DisplayTiming {
  return {
    count: sample.count,
    last: sample.last,
    mean: sample.count ? sample.total / sample.count : 0,
    max: sample.max,
  };
}

export function recordRunSlice(ms: number): void {
  record(runSlice, ms);
}

export function recordGuestFrames(count: number): void {
  guestFrames += count;
  if (count > 1) skippedFrames += count - 1;
}

export function recordCanvasWrite(ms: number): void {
  record(canvasWrite, ms);
  canvasUpdates++;
}

export function displayMetrics(): DisplayMetrics {
  return {
    guestFrames,
    canvasUpdates,
    skippedFrames,
    runSlice: read(runSlice),
    canvasWrite: read(canvasWrite),
  };
}

export function resetDisplayMetrics(): void {
  guestFrames = 0;
  canvasUpdates = 0;
  skippedFrames = 0;
  runSlice = timing();
  canvasWrite = timing();
}
