// The stage's loading screen: core boot, program load, and waiting for a first frame.

import { $ } from './dom';
import { hideCrash } from './crash';

const rootEl = $('loading');
const iconEl = $<HTMLImageElement>('loading-icon');
const titleEl = $('loading-title');
const fillEl = $('loading-fill');
const phaseEl = $('loading-phase');
const detailEl = $('loading-detail');
const dismissEl = $('loading-dismiss');

function setBar(fraction: number | null): void {
  rootEl.classList.toggle('is-determinate', fraction !== null);
  fillEl.style.width = fraction === null ? '' : (fraction * 100).toFixed(1) + '%';
}

export function beginLoad(title: string, phase: string, iconUrl?: string | null): void {
  hideCrash();
  rootEl.classList.remove('hidden', 'is-error');
  setBar(null);
  phaseEl.textContent = phase;
  detailEl.textContent = '';
  dismissEl.hidden = true;
  loadIdentity(title, iconUrl ?? null);
}

// Name the load once the file has been read (homebrew carries its own name and icon).
export function loadIdentity(title: string, iconUrl: string | null): void {
  titleEl.textContent = title;
  iconEl.hidden = !iconUrl;
  if (iconUrl) iconEl.src = iconUrl;
  else iconEl.removeAttribute('src');
}

export function loadPhase(phase: string, detail?: string): void {
  setBar(null);
  phaseEl.textContent = phase;
  detailEl.textContent = detail || '';
}

export function loadProgress(done: number, total: number): void {
  setBar(total > 0 ? Math.min(1, done / total) : null);
}

// The guest is running but has not presented a frame; offers a way out.
export function awaitFirstFrame(): void {
  loadPhase('booting', 'waiting for the first frame');
  dismissEl.hidden = false;
}

export function endLoad(): void {
  rootEl.classList.add('hidden');
  dismissEl.hidden = true;
}

// The load failed; the screen stays up showing why.
export function failLoad(message: string): void {
  rootEl.classList.remove('hidden');
  rootEl.classList.add('is-error');
  setBar(1);
  phaseEl.textContent = message;
  detailEl.textContent = '';
  dismissEl.hidden = false;
}

dismissEl.addEventListener('click', endLoad);
