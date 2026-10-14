// The application shell: stage and side panel.

import { Maximize, PanelRight, X } from 'lucide';
import { $ } from './dom';
import { icon } from './icons';
import { call } from './rpc';

export const stageEl = $('stage');
export const screenEl = $<HTMLCanvasElement>('screen');
export const overlayEl = $('overlay');
export const dropveilEl = $('dropveil');

const runEl = $<HTMLButtonElement>('btn-run');
// Readouts that mean nothing without a program.
const programItems = ['instructions-item', 'res-item'].map((id) => $(id));
const stepEl = $<HTMLButtonElement>('btn-step');
const stopEl = $<HTMLButtonElement>('btn-stop');

// The run state, also on `<body data-state>` for the page tests.
export type EmuState = 'idle' | 'loading' | 'loaded' | 'running' | 'paused' | 'halted' | 'fault';

let state: EmuState = 'idle';

// Whether there is a program for Run and Step to act on.
export function loaded(): boolean {
  return state !== 'idle' && state !== 'loading';
}

export function setState(text: EmuState): void {
  state = text;
  document.body.dataset.state = text;
  for (const item of programItems) item.hidden = !loaded();
  runEl.disabled = !loaded();
  stepEl.disabled = !loaded();
  stopEl.disabled = !loaded();
}

setState('idle');

export function showOverlay(show: boolean): void {
  overlayEl.classList.toggle('hidden', !show);
}

// Uncover the canvas and blank it; the worker paints it.
export function showScreen(): void {
  void call('clear_screen');
  showOverlay(false);
}

// Side panel (Console / Debug / Files).
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
  document.querySelectorAll<HTMLElement>('.panel .tab').forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle('is-active', on);
    t.setAttribute('aria-selected', String(on));
  });
  document.querySelectorAll<HTMLElement>('.panel .tabpanel').forEach((p) => {
    p.classList.toggle('is-active', p.dataset.panel === name);
  });
}

document.querySelectorAll<HTMLElement>('.panel .tab').forEach((t) => {
  t.addEventListener('click', () => selectTab(t.dataset.tab || ''));
});
$('btn-panel').addEventListener('click', () => setPanel(!panelOpen()));
$('btn-panel-close').addEventListener('click', () => setPanel(false));

// The short status a collapsible section shows on its header.
export function setNote(id: string, text: string, on?: boolean): void {
  const node = $(id);
  node.textContent = text;
  node.classList.toggle('on', Boolean(on));
}

// Draggable, persisted panel width.

const PANEL_W_KEY = 'switch-wasm-panel-width';
const PANEL_W_MIN = 300;
const PANEL_W_MAX = 760;
const gripEl = $('panel-grip');
let panelWidth = 380;

function setPanelWidth(px: number, persist: boolean): void {
  // At most three fifths of the window.
  const limit = Math.min(PANEL_W_MAX, Math.max(PANEL_W_MIN, window.innerWidth * 0.6));
  panelWidth = Math.round(Math.min(limit, Math.max(PANEL_W_MIN, px)));
  document.documentElement.style.setProperty('--panel-w', panelWidth + 'px');
  if (persist) localStorage.setItem(PANEL_W_KEY, String(panelWidth));
}

const storedPanelWidth = parseInt(localStorage.getItem(PANEL_W_KEY) || '', 10);
if (Number.isFinite(storedPanelWidth)) setPanelWidth(storedPanelWidth, false);

let gripPointer = -1;

gripEl.addEventListener('pointerdown', (e) => {
  // Below the breakpoint the panel is a bottom sheet with no seam to drag.
  if (!window.matchMedia('(min-width: 821px)').matches) return;
  e.preventDefault();
  gripPointer = e.pointerId;
  gripEl.classList.add('dragging');
  document.body.classList.add('resizing');
  try {
    gripEl.setPointerCapture(e.pointerId);
  } catch {
    // enhancement only
  }
});
window.addEventListener('pointermove', (e) => {
  if (gripPointer < 0) return;
  setPanelWidth(window.innerWidth - e.clientX, false);
});
function endPanelResize(): void {
  if (gripPointer < 0) return;
  try {
    gripEl.releasePointerCapture(gripPointer);
  } catch {
    // never captured
  }
  gripPointer = -1;
  gripEl.classList.remove('dragging');
  document.body.classList.remove('resizing');
  setPanelWidth(panelWidth, true);
}
window.addEventListener('pointerup', endPanelResize);
window.addEventListener('pointercancel', endPanelResize);
// Arrow keys are also the d-pad; keep them from reaching the guest.
gripEl.addEventListener('keydown', (e) => {
  const step = e.shiftKey ? 40 : 12;
  if (e.key === 'ArrowLeft') setPanelWidth(panelWidth + step, true);
  else if (e.key === 'ArrowRight') setPanelWidth(panelWidth - step, true);
  else return;
  e.preventDefault();
  e.stopPropagation();
});

for (const [id, glyph] of [
  ['btn-fullscreen', Maximize], ['btn-panel', PanelRight], ['btn-panel-close', X],
  ['btn-profiles-close', X], ['btn-files-close', X],
] as const) {
  $(id).append(icon(glyph));
}

$('btn-fullscreen').addEventListener('click', () => {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void stageEl.requestFullscreen?.();
});
