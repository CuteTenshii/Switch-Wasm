// The stage's loading screen: core boot, program load, and waiting for a first frame.

import { $ } from './dom';
import { hideCrash } from './crash';

const rootEl = $('loading');
const iconEl = $<HTMLImageElement>('loading-icon');
const titleEl = $('loading-title');
const phaseEl = $('loading-phase');
const detailEl = $('loading-detail');
const dismissEl = $('loading-dismiss');

export function beginLoad(title: string, phase: string, iconUrl?: string | null): void {
  hideCrash();
  rootEl.classList.remove('hidden', 'is-error');
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
  phaseEl.textContent = phase;
  detailEl.textContent = detail || '';
}

// The guest is running but has not presented a frame; offers a way out.
export function awaitFirstFrame(): void {
  loadPhase('Booting', 'Waiting for the first frame');
  dismissEl.hidden = false;
}

export function endLoad(): void {
  rootEl.classList.add('hidden');
  dismissEl.hidden = true;
}

// The load failed; the screen stays up showing why, retitled when `title` is given.
export function failLoad(message: string, title?: string): void {
  rootEl.classList.remove('hidden');
  rootEl.classList.add('is-error');
  if (title) loadIdentity(title, null);
  phaseEl.textContent = message;
  detailEl.textContent = '';
  dismissEl.hidden = false;
}

dismissEl.addEventListener('click', endLoad);
