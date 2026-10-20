// The emulated screen's readouts: the worker paints the canvas, the page counts frames.

import type { SliceReport } from '../shared/protocol';
import { $ } from './dom';
import { recordCanvasWrite, recordGuestFrames, resetDisplayMetrics } from './display-metrics';
import { endLoad } from './loading';
import { showOverlay } from './shell';

let lastFrame = 0;
// The painted frame's size, for mapping touches onto it.
let width = 0;
let height = 0;

export function screenSize(): { width: number; height: number } {
  return { width, height };
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
    $('frame-ms').textContent = fpsFrames > 0 ? (emulatedMs / fpsFrames).toFixed(1) + ' ms' : '- ms';
    fpsFrames = 0;
    emulatedMs = 0;
    fpsSince = now;
  }
}

// What a slice did to the screen.
export function noteFrame(report: SliceReport): void {
  if (report.paintMs !== null) recordCanvasWrite(report.paintMs);
  const delta = report.frame - lastFrame;
  countFrames(delta);
  if (delta === 0) return;
  recordGuestFrames(delta);
  // The first real frame uncovers the screen.
  if (lastFrame === 0) {
    showOverlay(false);
    endLoad();
  }
  lastFrame = report.frame;
  width = report.width;
  height = report.height;
  $('res').textContent = width + '×' + height;
}

export function resetDisplay(): void {
  lastFrame = 0;
  width = 0;
  height = 0;
  fpsFrames = 0;
  emulatedMs = 0;
  fpsSince = performance.now();
  resetDisplayMetrics();
  $('res').textContent = '-';
  $('fps').textContent = '- fps';
  $('frame-ms').textContent = '- ms';
}
