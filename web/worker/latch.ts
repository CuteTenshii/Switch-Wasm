// Input latch: a press is held until the guest frame counter advances twice,
// so taps between run slices are not lost. Keys still down publish live.

import { alloc, api, handle } from './wasm';

let heldButtons = 0n; // what the host says is physically down right now
let latchedButtons = 0n; // pressed, but not yet guaranteed seen by the guest
let sticks = [0, 0, 0, 0]; // newest analog values
let latchedSticks: number[] | null = null; // a deflection held like a press

// Touch uses the same latch. Contacts are flat {finger_id, x, y} triples.
const TOUCH_MAX = 16;
const NO_TOUCHES = new Uint32Array(0);
let touches = NO_TOUCHES; // newest host contacts
let latchedTouches: Uint32Array | null = null; // a tap held until a frame passes
let touchIds = new Set<number>(); // finger ids down at the last sample
let touchScratch = 0; // wasm-side staging buffer, allocated once
let publishedTouches = 0; // contacts the guest was last told about

// Frame the latch waits on, plus a slice cap for programs that never present.
let latchFrame = -1;
let latchSlices = 0;
const LATCH_FRAMES = 2;
const MAX_LATCH_SLICES = 64;

// Matches HID_STICK_THRESHOLD in cpu/mod.rs; a flick latches like a press.
const STICK_THRESHOLD = 0x4000;
const deflected = (s: number[]) => s.some((v) => Math.abs(v) > STICK_THRESHOLD);

function publishInput(): void {
  if (handle() < 0) return;
  // Live deflection wins; a latched flick stands in once the stick recentres.
  const s = latchedSticks && !deflected(sticks) ? latchedSticks : sticks;
  api().switch_set_input(handle(), heldButtons | latchedButtons, s[0], s[1], s[2], s[3]);
  publishTouch();
}

// Live contacts win; a latched tap stands in once the finger is up.
// Rebuild the view every time: heap growth detaches it.
function publishTouch(): void {
  const src = touches.length ? touches : latchedTouches || NO_TOUCHES;
  const count = Math.min(TOUCH_MAX, src.length / 3);
  if (count === 0 && publishedTouches === 0) return;
  if (!touchScratch) touchScratch = alloc(TOUCH_MAX * 3 * 4);
  if (count > 0) {
    new Uint32Array(api().memory.buffer, touchScratch, count * 3)
      .set(src.subarray(0, count * 3));
  }
  api().switch_set_touch(handle(), touchScratch, count);
  publishedTouches = count;
}

// A reset clears the latch at both ends of the session lifecycle.
export function resetInput(): void {
  heldButtons = 0n;
  latchedButtons = 0n;
  sticks = [0, 0, 0, 0];
  latchedSticks = null;
  touches = NO_TOUCHES;
  latchedTouches = null;
  touchIds = new Set();
  publishedTouches = 0;
  latchFrame = -1;
  latchSlices = 0;
}

// Start or restart the wait; restarting extends it for earlier presses too.
function armLatch(): void {
  latchFrame = handle() < 0 ? -1 : api().switch_frame_count(handle());
  latchSlices = 0;
}

// Called once per run slice; drops the latch after a full visible frame.
export function releaseLatchIfSeen(): void {
  if (handle() < 0) return;
  if (latchedButtons === 0n && !latchedSticks && !latchedTouches) return;
  const frames = api().switch_frame_count(handle());
  if (frames - latchFrame < LATCH_FRAMES && ++latchSlices < MAX_LATCH_SLICES) return;
  latchedButtons = 0n;
  latchedSticks = null;
  latchedTouches = null;
  latchFrame = -1;
  latchSlices = 0;
  publishInput();
}

export function setGamepad(
  mask: number,
  slx: number,
  sly: number,
  srx: number,
  sry: number,
): void {
  const next = BigInt(mask);
  const pressed = next & ~heldButtons; // edges, not level
  heldButtons = next;
  sticks = [slx, sly, srx, sry];
  const flicked = deflected(sticks);
  if (flicked) latchedSticks = sticks;
  if (pressed !== 0n || flicked) {
    latchedButtons |= pressed;
    armLatch();
  }
  publishInput();
}

// Contacts in 1280x720 digitizer space; a new finger id arms the latch.
export function setTouch(points: Uint32Array): void {
  const next = points && points.length ? new Uint32Array(points) : NO_TOUCHES;
  const ids = new Set<number>();
  let fresh = false;
  for (let i = 0; i < next.length; i += 3) {
    ids.add(next[i]);
    if (!touchIds.has(next[i])) fresh = true;
  }
  touches = next;
  touchIds = ids;
  if (fresh) {
    latchedTouches = next;
    armLatch();
  }
  publishInput();
}
