import { beforeEach, describe, expect, test } from 'bun:test';
import {
  displayMetrics, recordCanvasWrite, recordGuestFrames, recordRunSlice, resetDisplayMetrics,
} from '../../web/main/display-metrics';

beforeEach(resetDisplayMetrics);

describe('display metrics', () => {
  test('start at zero with no mean to divide by', () => {
    const metrics = displayMetrics();
    expect(metrics.guestFrames).toBe(0);
    expect(metrics.runSlice).toEqual({ count: 0, last: 0, mean: 0, max: 0 });
  });

  test('keep the count, last, mean and max of each stage', () => {
    for (const ms of [4, 10, 1]) recordRunSlice(ms);
    expect(displayMetrics().runSlice).toEqual({ count: 3, last: 1, mean: 5, max: 10 });
  });

  test('count frames presented between two canvas writes as skipped', () => {
    recordGuestFrames(1);
    recordGuestFrames(3);
    recordCanvasWrite(2);
    const metrics = displayMetrics();
    expect(metrics.guestFrames).toBe(4);
    expect(metrics.skippedFrames).toBe(2);
    expect(metrics.canvasUpdates).toBe(1);
  });

  test('reset forgets everything', () => {
    recordRunSlice(5);
    recordGuestFrames(2);
    recordCanvasWrite(1);
    resetDisplayMetrics();
    const metrics = displayMetrics();
    expect([metrics.guestFrames, metrics.skippedFrames, metrics.canvasUpdates]).toEqual([0, 0, 0]);
    expect(metrics.canvasWrite.count).toBe(0);
  });
});
