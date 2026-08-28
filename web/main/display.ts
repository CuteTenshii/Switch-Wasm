/* The emulated screen, and the frame counter the fps readout is derived from. */

import { $ } from './dom';
import { endLoad } from './loading';
import { call } from './rpc';
import { screenCtx, screenEl, showOverlay, showScreen } from './shell';

let fbW = 0;
let fbH = 0;
let fbBytes = 0;
let lastFrame = 0;

/** Size the canvas to whatever the fresh session reports before anything has
 *  run, so the page is not a zero-sized canvas until the first frame. */
export async function initFbSize(): Promise<void> {
  fbW = await call('fb_width');
  fbH = await call('fb_height');
  fbBytes = fbW * fbH * 4;
  screenEl.width = fbW;
  screenEl.height = fbH;
}

// Copy the emulated screen into the canvas, resizing it to whatever resolution
// the guest presented (1280x720 for most homebrew). Before the guest hands the
// display its first frame there is nothing to copy, so the canvas stays a blank
// screen - visible, but empty.
export async function renderFb(): Promise<void> {
  const w = await call('fb_width');
  const h = await call('fb_height');
  if (!w || !h) return;
  if (w !== fbW || h !== fbH) {
    fbW = w;
    fbH = h;
    fbBytes = w * h * 4;
    screenEl.width = w;
    screenEl.height = h;
  }
  // Until the guest hands the display a frame there is no resolution to
  // report - `fb_width`/`fb_height` fall back to the memory-mapped
  // framebuffer's size, which real homebrew never uses.
  if (lastFrame === 0) lastFrame = await call('frame_count');
  $('res').textContent = lastFrame > 0 ? w + '×' + h : '—';
  if (lastFrame === 0) {
    // Nothing has been presented, so there is no screen content to copy: the
    // fallback framebuffer region is just guest memory that Horizon homebrew
    // never writes. Show it as a blank screen instead of that memory's
    // contents.
    showScreen();
    return;
  }
  const pixels = await call('fb_snapshot', fbBytes);
  if (pixels && pixels.length >= fbBytes) {
    const arr = new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, fbBytes);
    screenCtx.putImageData(new ImageData(arr, fbW, fbH), 0, 0);
    showOverlay(false);
    // There is now a frame under the loading screen, which is the one thing it
    // was waiting for.
    endLoad();
  }
}

// Frames per second, measured from the guest's own present count.
let fpsFrames = 0;
let fpsSince = performance.now();
// Wall clock the run loop has spent inside the worker since the last readout.
// A slice carries whatever presents happened during it, so charging the run
// time to the frames it produced is the only division that means anything -
// the same reasoning as `FRAME_TIMES` in examples/common/mod.rs.
let emulatedMs = 0;

/** Wall clock one run slice cost, for the ms/frame readout. */
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

/** Repaint only when the guest has actually presented a new frame - the
 *  snapshot is several megabytes at 1280x720. */
export async function presentIfNewFrame(): Promise<void> {
  const frames = await call('frame_count');
  if (frames === lastFrame) return;
  countFrames(frames - lastFrame);
  lastFrame = frames;
  await renderFb();
}

/** A new session presents its own first frame, so nothing about the last
 *  one's should be believed. */
export function resetDisplay(): void {
  lastFrame = 0;
  fbW = 0;
  fbH = 0;
  fbBytes = 0;
  fpsFrames = 0;
  emulatedMs = 0;
  fpsSince = performance.now();
  $('res').textContent = '—';
  $('fps').textContent = '- fps';
  $('frame-ms').textContent = '- ms';
}
