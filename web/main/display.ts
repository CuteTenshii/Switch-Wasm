// The emulated screen and the frame counter behind the fps readout.

import { $ } from './dom';
import {
  recordCanvasWrite, recordFrameCounter, recordGuestFrames, recordMergedDisplayRequest,
  recordPaintWait, recordSnapshot, resetDisplayMetrics,
} from './display-metrics';
import { endLoad } from './loading';
import { log } from './log';
import { call, hasSession } from './rpc';
import { screenCtx, screenEl, showOverlay, showScreen } from './shell';

let fbW = 0;
let fbH = 0;
let fbBytes = 0;
let lastFrame = 0;
let scheduledFrame = 0;
let presentInFlight: Promise<void> | null = null;
// Reset can replace the session between worker messages; check the generation before each step.
let displayGeneration = 0;

async function frameCount(): Promise<number> {
  const started = performance.now();
  const frames = await call('frame_count');
  recordFrameCounter(performance.now() - started);
  return frames;
}

// Size the canvas from the fresh session before the first frame.
export async function initFbSize(): Promise<void> {
  fbW = await call('fb_width');
  fbH = await call('fb_height');
  fbBytes = fbW * fbH * 4;
  screenEl.width = fbW;
  screenEl.height = fbH;
}

export async function renderFb(expectedGeneration = displayGeneration): Promise<void> {
  const w = await call('fb_width');
  if (expectedGeneration !== displayGeneration) return;
  const h = await call('fb_height');
  if (expectedGeneration !== displayGeneration) return;
  if (!w || !h) return;
  if (w !== fbW || h !== fbH) {
    fbW = w;
    fbH = h;
    fbBytes = w * h * 4;
    screenEl.width = w;
    screenEl.height = h;
  }
  // Before the first present, `fb_width`/`fb_height` report an unused fallback framebuffer.
  if (lastFrame === 0) {
    const frames = await frameCount();
    if (expectedGeneration !== displayGeneration) return;
    lastFrame = frames;
  }
  $('res').textContent = lastFrame > 0 ? w + '×' + h : '—';
  if (lastFrame === 0) {
    showScreen();
    return;
  }
  const snapshotAt = performance.now();
  const pixels = await call('fb_snapshot', fbBytes);
  recordSnapshot(performance.now() - snapshotAt, pixels?.length ?? 0);
  if (expectedGeneration !== displayGeneration) return;
  if (pixels && pixels.length >= fbBytes) {
    const arr = new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, fbBytes);
    const writeAt = performance.now();
    screenCtx.putImageData(new ImageData(arr, fbW, fbH), 0, 0);
    recordCanvasWrite(performance.now() - writeAt);
    showOverlay(false);
    endLoad();
  }
}

let fpsFrames = 0;
let fpsSince = performance.now();
// Worker run time since the last readout, charged to the frames it produced.
let emulatedMs = 0;

export function countEmulation(ms: number): void {
  emulatedMs += ms;
}

function countFrames(delta: number): void {
  fpsFrames += delta;
  const now = performance.now();
  const elapsed = now - fpsSince;
  if (elapsed >= 500) {
    $('fps').textContent = (fpsFrames * 1000 / elapsed).toFixed(1) + ' fps';
    $('frame-ms').textContent = (emulatedMs / fpsFrames).toFixed(1) + ' ms';
    fpsFrames = 0;
    emulatedMs = 0;
    fpsSince = now;
  }
}

// Repaint only on a new present; the snapshot is several megabytes.
export async function presentIfNewFrame(): Promise<void> {
  const expectedGeneration = displayGeneration;
  const frames = await frameCount();
  if (expectedGeneration !== displayGeneration) return;
  if (frames === lastFrame) return;
  const delta = frames - lastFrame;
  countFrames(delta);
  recordGuestFrames(delta);
  lastFrame = frames;
  await renderFb(expectedGeneration);
}

// Repeated calls collapse into the pending paint, so guest execution is never paced by the browser.
export function schedulePresentIfNewFrame(): void {
  if (scheduledFrame || presentInFlight) {
    recordMergedDisplayRequest();
    return;
  }
  const scheduledAt = performance.now();
  scheduledFrame = requestAnimationFrame(() => {
    scheduledFrame = 0;
    recordPaintWait(performance.now() - scheduledAt);
    presentInFlight = presentIfNewFrame()
      .catch((err: unknown) => {
        if (hasSession()) log('Display update failed: ' + (err as Error).message, 'err');
      })
      .finally(() => {
        presentInFlight = null;
      });
  });
}

// Copy the newest frame before a run stops so it is not left in the worker.
export async function flushDisplay(): Promise<void> {
  if (scheduledFrame) {
    cancelAnimationFrame(scheduledFrame);
    scheduledFrame = 0;
  }
  if (presentInFlight) await presentInFlight;
  await presentIfNewFrame();
}

export function abortDisplay(): void {
  displayGeneration++;
  if (scheduledFrame) cancelAnimationFrame(scheduledFrame);
  scheduledFrame = 0;
}

export function resetDisplay(): void {
  abortDisplay();
  presentInFlight = null;
  lastFrame = 0;
  fbW = 0;
  fbH = 0;
  fbBytes = 0;
  fpsFrames = 0;
  emulatedMs = 0;
  fpsSince = performance.now();
  resetDisplayMetrics();
  $('res').textContent = '—';
  $('fps').textContent = '- fps';
  $('frame-ms').textContent = '- ms';
}
