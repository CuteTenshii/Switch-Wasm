// The application shell: stage and side panel.

import {
  Bug, ChevronDown, ChevronLeft, ChevronRight, ChevronUp, Cpu, Download, Ellipsis, Eraser, FileText, FileUp,
  Folder, FolderOpen, FolderSearch, FolderUp, HardDrive, History, KeyRound, Keyboard, Maximize, MessagesSquare,
  Network, PackagePlus, PanelRight, Square, SquareTerminal, StepForward, Trash2, UserPlus, X, type IconNode,
} from 'lucide';
import { $ } from './dom';
import { icon } from './icons';
import { call } from './rpc';

export const stageEl = $('stage');
export const screenEl = $<HTMLCanvasElement>('screen');
export const overlayEl = $('overlay');
export const dropveilEl = $('dropveil');

const runEl = $<HTMLButtonElement>('btn-run');
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
  // Hidden with nothing to act on, and disabled so a stray click or Enter can't reach them either.
  for (const button of [runEl, stepEl, stopEl]) {
    button.hidden = !loaded();
    button.disabled = !loaded();
  }
}

setState('idle');

export function showOverlay(show: boolean): void {
  overlayEl.classList.toggle('hidden', !show);
}

// Uncover the canvas and blank it; the worker paints it.
export function showScreen(): void {
  void call('clearScreen');
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

const storedPanelWidth = Number.parseInt(localStorage.getItem(PANEL_W_KEY) || '', 10);
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

const staticIcons: [string, IconNode][] = [
  ['#btn-discord', MessagesSquare], ['#btn-fullscreen', Maximize], ['#btn-panel', PanelRight], ['#btn-more', Ellipsis], ['#btn-panel-close', X],
  ['#btn-profiles-close', X], ['#btn-files-close', X],
  ['#btn-step', StepForward], ['#btn-stop', Square], ['#btn-open', FolderOpen],
  ['#overlay-mark', FileUp], ['#overlay-open', FolderOpen], ['#overlay-open-folder', Folder],
  ['#keymap-title', Keyboard],
  ['.keymap-button[data-glyph="up"]', ChevronUp], ['.keymap-button[data-glyph="down"]', ChevronDown],
  ['.keymap-button[data-glyph="left"]', ChevronLeft], ['.keymap-button[data-glyph="right"]', ChevronRight],
  ['#crash-save', Download], ['#crash-console', SquareTerminal], ['#crash-stop', Square],
  ['.tab[data-tab="console"]', SquareTerminal], ['.tab[data-tab="debug"]', Bug],
  ['.tab[data-tab="files"]', Folder], ['.tab[data-tab="nand"]', HardDrive], ['.tab[data-tab="network"]', Network],
  ['#btn-save-console', Download], ['#btn-save-previous', History], ['#btn-clear-console', Eraser],
  ['#btn-dumptrace', FileText], ['#btn-dumpregs', Cpu],
  ['#btn-files', FolderSearch], ['label[for="prod-keys"]', KeyRound], ['label[for="title-keys"]', KeyRound],
  ['#btn-clear-keys', Eraser], ['.tabpanel label[for="firmware-dir"]', FolderUp],
  ['.tabpanel label[for="firmware-zip"]', PackagePlus], ['#btn-erase-nand', Trash2],
  ['#profile-add-submit', UserPlus],
];
for (const [selector, glyph] of staticIcons) {
  for (const target of document.querySelectorAll(selector)) target.prepend(icon(glyph));
}

// Below the breakpoint, the bar's secondary controls live in the More menu.
const moreMenu = $('more-menu');
const narrowBar = window.matchMedia('(width <= 900px)');
// Each with a marker where it sits in the bar, to go back to.
const moreItems = ['btn-profile', 'btn-dock', 'btn-discord', 'btn-fullscreen', 'btn-step'].map((id) => {
  const item = $(id);
  const home = document.createComment(id);
  item.before(home);
  return { item, home };
});

function placeMoreItems(): void {
  for (const { item, home } of moreItems) {
    if (narrowBar.matches) moreMenu.append(item);
    else home.after(item);
  }
  if (!narrowBar.matches) moreMenu.hidePopover();
}

placeMoreItems();
narrowBar.addEventListener('change', placeMoreItems);
// The path, not `target.closest`: a button that re-renders its contents on click
// has detached the target by the time the click bubbles here.
moreMenu.addEventListener('click', (e) => {
  if (e.composedPath().some((node) => node instanceof HTMLButtonElement)) moreMenu.hidePopover();
});

$('btn-fullscreen').addEventListener('click', () => {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void stageEl.requestFullscreen?.();
});
