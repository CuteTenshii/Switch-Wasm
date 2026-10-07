// Keyboard, gamepad and touch input, pushed to the emulator as one state.

import { $ } from './dom';
import { log } from './log';
import { call, hasSession, isReady } from './rpc';
import { pullVibration } from './rumble';
import { screenEl } from './shell';

// HidNpadButton bits, in Horizon's order.
const BTN = {
  A: 1 << 0, B: 1 << 1, X: 1 << 2, Y: 1 << 3,
  STICK_L: 1 << 4, STICK_R: 1 << 5,
  L: 1 << 6, R: 1 << 7, ZL: 1 << 8, ZR: 1 << 9,
  PLUS: 1 << 10, MINUS: 1 << 11,
  LEFT: 1 << 12, UP: 1 << 13, RIGHT: 1 << 14, DOWN: 1 << 15,
};

function inputStatus(text: string, detail = ''): void {
  const el = $('input-state');
  el.textContent = text;
  el.title = detail;
}

// A controller's name without the vendor/product decoration browsers add.
function padName(pad: Gamepad): string {
  const name = pad.id
    .replace(/\s*\([^)]*\)\s*$/, '')
    .replace(/^[0-9a-f]{1,4}-[0-9a-f]{1,4}-/i, '')
    .trim();
  return name || 'gamepad';
}

function showPad(pad: Gamepad): void {
  inputStatus(padName(pad), pad.id);
}

// Keyboard fallback: dpad + A/B/X/Y + start/select.
const KEY_MAP: Record<string, number> = {
  arrowleft: BTN.LEFT, arrowup: BTN.UP, arrowright: BTN.RIGHT, arrowdown: BTN.DOWN,
  enter: BTN.PLUS, shift: BTN.MINUS,
  z: BTN.A, x: BTN.B, a: BTN.X, s: BTN.Y,
  q: BTN.L, e: BTN.R,
};

const keysDown = new Set<string>();

// Pushed on key events as well as on the poll, to avoid a tick of latency.
// Keys belong to the page while a field or dialog has focus.
function pageHasKeyboard(): boolean {
  const focused = document.activeElement;
  return /^(INPUT|SELECT|TEXTAREA)$/.test(focused?.tagName || '')
    || document.querySelector('dialog[open]') !== null;
}

window.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();
  if (!KEY_MAP[key] || pageHasKeyboard()) return;
  e.preventDefault();
  // Go by the set, not `e.repeat`: `blur` clears the set while keys may still be down.
  if (keysDown.has(key)) return;
  keysDown.add(key);
  pushInput();
});
window.addEventListener('keyup', (e) => {
  if (keysDown.delete(e.key.toLowerCase())) pushInput();
});
window.addEventListener('blur', () => {
  if (!keysDown.size && !touchPoints.size) return;
  keysDown.clear();
  touchPoints.clear();
  pushInput();
});

function keyboardMask(): number {
  let m = 0;
  for (const code of keysDown) m |= KEY_MAP[code] || 0;
  return m;
}

// Touch, mapped into the console's 1280x720 digitizer space.
const TOUCH_W = 1280;
const TOUCH_H = 720;
const TOUCH_MAX = 16;

interface Contact {
  slot: number;
  x: number;
  y: number;
}

// pointerId -> { slot, x, y }; `slot` is the guest finger id, the lowest free one.
const touchPoints = new Map<number, Contact>();
let touchWasDown = false;

function claimTouchSlot(): number {
  const taken = new Set([...touchPoints.values()].map((t) => t.slot));
  for (let i = 0; i < TOUCH_MAX; i++) if (!taken.has(i)) return i;
  return -1;
}

// Map through the `object-fit: contain` rect; null for a tap on the letterbox.
function touchAt(e: PointerEvent): { x: number; y: number } | null {
  const rect = screenEl.getBoundingClientRect();
  const iw = screenEl.width, ih = screenEl.height;
  if (!iw || !ih || !rect.width || !rect.height) return null;
  const scale = Math.min(rect.width / iw, rect.height / ih);
  const dw = iw * scale, dh = ih * scale;
  const x = (e.clientX - rect.left - (rect.width - dw) / 2) / dw;
  const y = (e.clientY - rect.top - (rect.height - dh) / 2) / dh;
  if (x < 0 || x >= 1 || y < 0 || y >= 1) return null;
  return {
    x: Math.min(TOUCH_W - 1, Math.floor(x * TOUCH_W)),
    y: Math.min(TOUCH_H - 1, Math.floor(y * TOUCH_H)),
  };
}

function touchTriples(): Uint32Array {
  const out = new Uint32Array(touchPoints.size * 3);
  let i = 0;
  for (const t of touchPoints.values()) {
    out[i++] = t.slot;
    out[i++] = t.x;
    out[i++] = t.y;
  }
  return out;
}

screenEl.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  const p = touchAt(e);
  if (!p) return;
  const slot = claimTouchSlot();
  if (slot < 0) return; // all sixteen contacts are down
  touchPoints.set(e.pointerId, { slot, x: p.x, y: p.y });
  // Capture so a finger sliding off the canvas still reports its lift.
  try {
    screenEl.setPointerCapture(e.pointerId);
  } catch {
    // not capturable
  }
  e.preventDefault();
  pushInput();
});

screenEl.addEventListener('pointermove', (e) => {
  const t = touchPoints.get(e.pointerId);
  if (!t) return;
  const p = touchAt(e);
  // A finger dragged into the letterbox holds its last position.
  if (p) {
    t.x = p.x;
    t.y = p.y;
  }
  e.preventDefault();
});

function liftTouch(e: PointerEvent): void {
  if (touchPoints.delete(e.pointerId)) pushInput();
}
screenEl.addEventListener('pointerup', liftTouch);
screenEl.addEventListener('pointercancel', liftTouch);

function pushInput(): void {
  // Reset frees the session first; swallow rejections from these fire-and-forget calls.
  if (!isReady() || !hasSession()) return;
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  const pad = pads.find((p) => p && p.connected);
  let mask = keyboardMask();
  let slx = 0, sly = 0, srx = 0, sry = 0;
  if (pad) {
    // Standard order: 0-3 B/A/Y/X, 4-7 L/R/ZL/ZR, 8-9 select/start, 10-11 sticks, 12-17 dpad.
    if (pad.buttons[0]?.pressed) mask |= BTN.B;
    if (pad.buttons[1]?.pressed) mask |= BTN.A;
    if (pad.buttons[2]?.pressed) mask |= BTN.Y;
    if (pad.buttons[3]?.pressed) mask |= BTN.X;
    if (pad.buttons[4]?.pressed) mask |= BTN.L;
    if (pad.buttons[5]?.pressed) mask |= BTN.R;
    if (pad.buttons[6]?.pressed) mask |= BTN.ZL;
    if (pad.buttons[7]?.pressed) mask |= BTN.ZR;
    if (pad.buttons[8]?.pressed) mask |= BTN.MINUS;
    if (pad.buttons[9]?.pressed) mask |= BTN.PLUS;
    if (pad.buttons[10]?.pressed) mask |= BTN.STICK_L;
    if (pad.buttons[11]?.pressed) mask |= BTN.STICK_R;
    if (pad.buttons[12]?.pressed) mask |= BTN.UP;
    if (pad.buttons[13]?.pressed) mask |= BTN.DOWN;
    if (pad.buttons[14]?.pressed) mask |= BTN.LEFT;
    if (pad.buttons[15]?.pressed) mask |= BTN.RIGHT;
    // -32768..32767, ~15% deadzone; Y negated because Horizon's axis points up.
    const dz = 0.15;
    const axes = pad.axes || [];
    const axis = (i: number) => (Math.abs(axes[i] || 0) > dz ? axes[i] : 0);
    slx = Math.round(axis(0) * 32767);
    sly = Math.round(-axis(1) * 32767);
    srx = Math.round(axis(2) * 32767);
    sry = Math.round(-axis(3) * 32767);
    showPad(pad);
  } else if (mask) {
    inputStatus('keyboard');
  }
  void call('set_input', mask, slx, sly, srx, sry);
  // Only while something is down, plus one push for the lift.
  if (touchPoints.size || touchWasDown) {
    void call('set_touch', touchTriples());
    touchWasDown = touchPoints.size > 0;
  }
  if (touchPoints.size) inputStatus('touch');
  void pullVibration(pad || undefined);
}

setInterval(pushInput, 16);
window.addEventListener('gamepadconnected', (e) => {
  const pad = e.gamepad;
  showPad(pad);
  // Buttons are read by W3C standard layout position.
  const layout = pad.mapping === 'standard'
    ? 'standard layout'
    : 'no standard layout, so its buttons may be mismatched';
  log(`[input] controller connected: ${padName(pad)} (${layout})`, pad.mapping === 'standard' ? undefined : 'warn');
});
window.addEventListener('gamepaddisconnected', (e) => {
  inputStatus('none');
  log(`[input] controller disconnected: ${padName(e.gamepad)}`);
});
