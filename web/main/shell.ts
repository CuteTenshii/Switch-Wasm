/* The application shell: the stage the emulated screen sits on, the state
   chip, and the side panel around them. */

import { $ } from './dom';

export const stageEl = $('stage');
export const screenEl = $<HTMLCanvasElement>('screen');
export const overlayEl = $('overlay');
export const dropveilEl = $('dropveil');

const ctx = screenEl.getContext('2d', { alpha: false });
if (!ctx) throw new Error('this browser has no 2d canvas context');
export const screenCtx = ctx;

const stateEl = $('state');
const runEl = $<HTMLButtonElement>('btn-run');
const stepEl = $<HTMLButtonElement>('btn-step');

/** What the chip in the top bar says, and what `[data-state]` styles. */
export type EmuState = 'idle' | 'loading' | 'loaded' | 'running' | 'paused' | 'halted' | 'fault';

let state: EmuState = 'idle';

/** Whether there is a program in the session for the transport to act on.
 *
 * `idle` is a console with nothing loaded and `loading` one still being
 * handed a title. In both the guest's pc is 0 and guest memory there reads
 * zeroes, so Run and Step would execute those zeroes and report the fault as
 * if a title had crashed. */
export function loaded(): boolean {
  return state !== 'idle' && state !== 'loading';
}

export function setState(text: EmuState): void {
  state = text;
  stateEl.textContent = text;
  stateEl.dataset.state = text;
  runEl.disabled = !loaded();
  stepEl.disabled = !loaded();
}

// index.html draws the transport live, and the machine it acts on is empty
// until something is loaded into it.
setState('idle');

export function showOverlay(show: boolean): void {
  overlayEl.classList.toggle('hidden', !show);
}

// Uncover the canvas and blank it. The context is alpha-less, so clearRect
// paints black - the same "powered on, nothing presented yet" state a real
// console shows.
export function showScreen(): void {
  screenCtx.clearRect(0, 0, screenEl.width, screenEl.height);
  showOverlay(false);
}

// Side panel (Console / Debug / Files). Closed by default: the screen is the
// point of the page, not the tooling around it.
export function panelOpen(): boolean {
  return document.body.classList.contains('panel-open');
}

export function setPanel(open: boolean): void {
  document.body.classList.toggle('panel-open', open);
  $('btn-panel').setAttribute('aria-expanded', String(open));
}

export function openPanel(tab?: string): void {
  setPanel(true);
  if (tab) selectTab(tab);
}

export function selectTab(name: string): void {
  document.querySelectorAll<HTMLElement>('.tab').forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle('is-active', on);
    t.setAttribute('aria-selected', String(on));
  });
  document.querySelectorAll<HTMLElement>('.tabpanel').forEach((p) => {
    p.classList.toggle('is-active', p.dataset.panel === name);
  });
}

document.querySelectorAll<HTMLElement>('.tab').forEach((t) => {
  t.addEventListener('click', () => selectTab(t.dataset.tab || ''));
});
$('btn-panel').addEventListener('click', () => setPanel(!panelOpen()));
$('btn-panel-close').addEventListener('click', () => setPanel(false));

/** The short status a collapsible section shows on its own header, so what a
 *  section holds is readable with the section shut. */
export function setNote(id: string, text: string, on?: boolean): void {
  const node = $(id);
  node.textContent = text;
  node.classList.toggle('on', Boolean(on));
}

// ---------- panel width ----------
//
// A fixed 380px is right for the status readouts and far too narrow for a
// register dump or a container's file names, so the seam between the stage and
// the panel is draggable and the width is remembered.

const PANEL_W_KEY = 'switch-wasm-panel-width';
const PANEL_W_MIN = 300;
const PANEL_W_MAX = 760;
const gripEl = $('panel-grip');
let panelWidth = 380;

function setPanelWidth(px: number, persist: boolean): void {
  // Never more than three fifths of the window: the screen is the point of the
  // page, and a panel that has eaten it is not a panel any more.
  const limit = Math.min(PANEL_W_MAX, Math.max(PANEL_W_MIN, window.innerWidth * 0.6));
  panelWidth = Math.round(Math.min(limit, Math.max(PANEL_W_MIN, px)));
  document.documentElement.style.setProperty('--panel-w', panelWidth + 'px');
  if (persist) localStorage.setItem(PANEL_W_KEY, String(panelWidth));
}

const storedPanelWidth = parseInt(localStorage.getItem(PANEL_W_KEY) || '', 10);
if (Number.isFinite(storedPanelWidth)) setPanelWidth(storedPanelWidth, false);

let gripPointer = -1;

gripEl.addEventListener('pointerdown', (e) => {
  // Below the breakpoint the panel is a bottom sheet whose height the media
  // query owns; there is no vertical seam to drag.
  if (!window.matchMedia('(min-width: 821px)').matches) return;
  e.preventDefault();
  gripPointer = e.pointerId;
  gripEl.classList.add('dragging');
  document.body.classList.add('resizing');
  // Capture keeps the pointer events coming while the cursor is off the 9px
  // grip, which it is for all but the first pixel of any real drag. The move
  // and up listeners are on the window rather than the grip so a browser that
  // refuses the capture still gets a working drag out of it.
  try { gripEl.setPointerCapture(e.pointerId); } catch { /* enhancement only */ }
});
window.addEventListener('pointermove', (e) => {
  if (gripPointer < 0) return;
  setPanelWidth(window.innerWidth - e.clientX, false);
});
function endPanelResize(): void {
  if (gripPointer < 0) return;
  try { gripEl.releasePointerCapture(gripPointer); } catch { /* never captured */ }
  gripPointer = -1;
  gripEl.classList.remove('dragging');
  document.body.classList.remove('resizing');
  setPanelWidth(panelWidth, true);
}
window.addEventListener('pointerup', endPanelResize);
window.addEventListener('pointercancel', endPanelResize);
// The arrow keys are also the emulated d-pad, so a focused seam has to keep
// them from reaching the guest as well as from scrolling the page.
gripEl.addEventListener('keydown', (e) => {
  const step = e.shiftKey ? 40 : 12;
  if (e.key === 'ArrowLeft') setPanelWidth(panelWidth + step, true);
  else if (e.key === 'ArrowRight') setPanelWidth(panelWidth - step, true);
  else return;
  e.preventDefault();
  e.stopPropagation();
});

$('btn-fullscreen').addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else stageEl.requestFullscreen?.();
});
