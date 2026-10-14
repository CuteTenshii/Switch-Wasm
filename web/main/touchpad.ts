// The on-screen controller, for touch devices without a gamepad, and the
// toolbar toggle of landscape phones.

import { ChevronDown, ChevronLeft, ChevronRight, ChevronUp } from 'lucide';
import { $ } from './dom';
import { icon } from './icons';

const padEl = $('touchpad');
const buttonEls = [...padEl.querySelectorAll<HTMLElement>('[data-button]')];

for (const [name, glyph] of [['UP', ChevronUp], ['DOWN', ChevronDown], ['LEFT', ChevronLeft], ['RIGHT', ChevronRight]] as const) {
  padEl.querySelector(`[data-button="${name}"]`)!.append(icon(glyph));
}

// The button each finger is on; a finger slides from one to the next.
const held = new Map<number, string>();

interface Stick {
  el: HTMLElement;
  knob: HTMLElement;
  x: number;
  y: number;
}

const sticks = new Map<string, Stick>();
for (const el of padEl.querySelectorAll<HTMLElement>('[data-stick]')) {
  sticks.set(el.dataset.stick!, { el, knob: el.querySelector<HTMLElement>('.tp-knob')!, x: 0, y: 0 });
}

// The stick each finger drags.
const dragging = new Map<number, Stick>();

// Names from `data-button`, as `input.ts` maps them.
export function touchpadButtons(): Iterable<string> {
  return held.values();
}

// -32768..32767, up positive as Horizon reads it.
export function touchpadStick(side: 'left' | 'right'): { x: number; y: number } {
  const stick = sticks.get(side)!;
  return { x: stick.x, y: stick.y };
}

export function touchpadActive(): boolean {
  return held.size > 0 || dragging.size > 0;
}

function showHeld(): void {
  const down = new Set(held.values());
  for (const el of buttonEls) el.classList.toggle('is-down', down.has(el.dataset.button!));
}

function holdAt(e: PointerEvent): void {
  const name = document.elementFromPoint(e.clientX, e.clientY)
    ?.closest<HTMLElement>('#touchpad [data-button]')?.dataset.button;
  if (name) held.set(e.pointerId, name);
  else held.delete(e.pointerId);
  showHeld();
}

function dragTo(stick: Stick, e: PointerEvent): void {
  const rect = stick.el.getBoundingClientRect();
  const radius = rect.width / 2;
  let dx = (e.clientX - rect.left - radius) / radius;
  let dy = (e.clientY - rect.top - radius) / radius;
  const length = Math.hypot(dx, dy);
  if (length > 1) {
    dx /= length;
    dy /= length;
  }
  // The knob stays inside the base.
  const travel = radius - stick.knob.offsetWidth / 2;
  stick.knob.style.transform = `translate(${dx * travel}px, ${dy * travel}px)`;
  stick.x = Math.round(dx * 32767);
  stick.y = Math.round(-dy * 32767);
}

padEl.addEventListener('pointerdown', (e) => {
  const stickEl = (e.target as Element).closest<HTMLElement>('[data-stick]');
  const stick = stickEl && sticks.get(stickEl.dataset.stick!);
  if (stick) {
    dragging.set(e.pointerId, stick);
    dragTo(stick, e);
  } else {
    holdAt(e);
  }
  e.preventDefault();
});

padEl.addEventListener('pointermove', (e) => {
  const stick = dragging.get(e.pointerId);
  if (stick) dragTo(stick, e);
  else if (held.has(e.pointerId)) holdAt(e);
});

function lift(e: PointerEvent): void {
  const stick = dragging.get(e.pointerId);
  if (stick) {
    dragging.delete(e.pointerId);
    stick.x = 0;
    stick.y = 0;
    stick.knob.style.transform = '';
  }
  if (held.delete(e.pointerId)) showHeld();
}

padEl.addEventListener('pointerup', lift);
padEl.addEventListener('pointercancel', lift);

// Shown on touchscreens until a gamepad connects; CSS adds the run state.
const coarse = window.matchMedia('(pointer: coarse)');

function updateTouchControls(): void {
  const gamepad = (navigator.getGamepads?.() ?? []).some((p) => p?.connected);
  document.body.classList.toggle('touch-controls', coarse.matches && !gamepad);
}

updateTouchControls();
coarse.addEventListener('change', updateTouchControls);
window.addEventListener('gamepadconnected', updateTouchControls);
window.addEventListener('gamepaddisconnected', updateTouchControls);

// Landscape phones hide the bars while a program runs; this brings them back.
const chromeEl = $('btn-chrome');
chromeEl.append(icon(ChevronDown));
chromeEl.addEventListener('click', () => {
  const shown = document.body.classList.toggle('chrome-shown');
  chromeEl.setAttribute('aria-expanded', String(shown));
  chromeEl.setAttribute('aria-label', shown ? 'Hide the toolbar' : 'Show the toolbar');
});
