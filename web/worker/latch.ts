/* Input, held until the guest has had a chance to see it.

   Gamepad state arrives from the main thread far more often than the emulator
   gets to look at it: the worker is blocked inside `switch_run` while the input
   messages pile up in its queue, so a whole slice's worth of them lands at once
   at the slice boundary and a quick tap can be pressed and released without the
   guest ever having been running to see it.

   The unit a press has to survive is a *guest frame*, not a run slice. The
   guest polls hid once per iteration of its own loop and presents once per
   iteration, and one of those spans many slices, so holding a tap for a single
   slice still let most taps fall between two polls. A press is therefore held
   until the frame counter has advanced twice: the poll sits somewhere inside
   the guest's loop, so only a complete present-to-present interval is
   guaranteed to contain one.

   Only bits the guest may not have seen are held. A key the host still reports
   as down is published from `heldButtons` on its own, so releasing it takes
   effect at the very next slice instead of a slice later - that extra slice of
   stickiness was making one d-pad tap step two menu entries. */

import { alloc, api, handle } from './wasm';

let heldButtons = 0n;    // what the host says is physically down right now
let latchedButtons = 0n; // pressed, but not yet guaranteed seen by the guest
let sticks = [0, 0, 0, 0];        // newest analog values
let latchedSticks: number[] | null = null; // a deflection held like a press

// Touch rides the same latch, for the same reason: a tap that goes down and up
// inside one run slice would otherwise happen entirely while the guest was not
// running to see it. Contacts are flat {finger_id, x, y} triples.
const TOUCH_MAX = 16;
const NO_TOUCHES = new Uint32Array(0);
let touches = NO_TOUCHES;                    // newest host contacts
let latchedTouches: Uint32Array | null = null; // a tap held until a frame passes
let touchIds = new Set<number>();            // finger ids down at the last sample
let touchScratch = 0;                        // wasm-side staging buffer, allocated once
let publishedTouches = 0;                    // contacts the guest was last told about

// Frame the latch is waiting on, plus a slice cap so that a program which never
// presents - or has stopped, mid-load - still releases instead of holding a
// phantom press until it draws again. A couple of seconds' worth of slices.
let latchFrame = -1;
let latchSlices = 0;
const LATCH_FRAMES = 2;
const MAX_LATCH_SLICES = 64;

// Matches HID_STICK_THRESHOLD in cpu/mod.rs: past this the core reports the
// HidNpadButton_StickL*/StickR* pseudo-buttons, which is what menus navigate
// with, so a flick has to be latched exactly like a button press.
const STICK_THRESHOLD = 0x4000;
const deflected = (s: number[]) => s.some((v) => Math.abs(v) > STICK_THRESHOLD);

function publishInput(): void {
  if (handle() < 0) return;
  // A stick the host is still deflecting reports its live value; a latched
  // flick only stands in once the stick has sprung back to centre.
  const s = latchedSticks && !deflected(sticks) ? latchedSticks : sticks;
  api().switch_set_input(handle(), heldButtons | latchedButtons, s[0], s[1], s[2], s[3]);
  publishTouch();
}

// Same rule as the sticks: live contacts win, a latched tap only stands in once
// the finger is up. `switch_set_touch` reads {finger_id, x, y} triples out of
// wasm memory, so they go through a buffer allocated once and reused - the view
// is rebuilt every time because growing the heap detaches the old one.
function publishTouch(): void {
  const src = touches.length ? touches : latchedTouches || NO_TOUCHES;
  const count = Math.min(TOUCH_MAX, src.length / 3);
  // Nothing down and nothing to retract: the guest already knows.
  if (count === 0 && publishedTouches === 0) return;
  if (!touchScratch) touchScratch = alloc(TOUCH_MAX * 3 * 4);
  if (count > 0) {
    new Uint32Array(api().memory.buffer, touchScratch, count * 3)
      .set(src.subarray(0, count * 3));
  }
  api().switch_set_touch(handle(), touchScratch, count);
  publishedTouches = count;
}

// A latch armed against the old session's frame counter would outlive a reset,
// so both ends of the session lifecycle clear it.
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

// Start (or restart) the wait for the guest to run a frame with the latch
// visible. Restarting on a later press extends the window for earlier ones too,
// which is what we want: they are all still unseen.
function armLatch(): void {
  latchFrame = handle() < 0 ? -1 : api().switch_frame_count(handle());
  latchSlices = 0;
}

// Called once per run slice: drop the latch as soon as the guest has had a
// whole frame to poll with it visible.
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
  const pressed = next & ~heldButtons; // edges, not level: what just went down
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

// Contacts as flat {finger_id, x, y} triples, already in the console's
// 1280x720 digitizer space. A finger id the previous sample did not carry is
// a new contact, which is what arms the latch.
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
