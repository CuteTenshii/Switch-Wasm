/* The stage's loading screen.
 *
 * The state chip says *that* something is loading; this says what, on the
 * surface the eye is already on. It covers three kinds of wait that the page
 * used to spend blank: bringing the core up, loading a program, and - the long
 * one - a title that has been loaded and is running but has not presented a
 * frame yet, which for a real title is minutes of a black rectangle.
 *
 * The screen owns nothing but its own markup: every caller opens it, names its
 * phases and closes it, and `awaitFirstFrame` is the one state it leaves open
 * for someone else to end (`display.renderFb` on the first painted frame, the
 * run loop on a fault, a halt or a pause).
 */

import { $ } from './dom';

const rootEl = $('loading');
const iconEl = $<HTMLImageElement>('loading-icon');
const titleEl = $('loading-title');
const fillEl = $('loading-fill');
const phaseEl = $('loading-phase');
const detailEl = $('loading-detail');
const dismissEl = $('loading-dismiss');

// Whether the guest has been started and the screen is now waiting on its
// first frame. Only in that state does the run loop's pc/step readout belong
// on the screen - during the fixed phases above it the detail line is saying
// something the caller chose.
let awaitingFrame = false;

function setBar(fraction: number | null): void {
  rootEl.classList.toggle('is-determinate', fraction !== null);
  fillEl.style.width = fraction === null ? '' : (fraction * 100).toFixed(1) + '%';
}

/** Show the screen for a new piece of work. `iconUrl` is the title's own icon
 *  where the container gave us one; a load with no identity of its own leaves
 *  the slot empty rather than filling it with a placeholder. */
export function beginLoad(title: string, phase: string, iconUrl?: string | null): void {
  awaitingFrame = false;
  rootEl.classList.remove('hidden', 'is-error');
  setBar(null);
  phaseEl.textContent = phase;
  detailEl.textContent = '';
  dismissEl.hidden = true;
  loadIdentity(title, iconUrl ?? null);
}

/** Say what is loading, for a load that only learns it once the file has been
 *  read: homebrew carries its own name and icon inside the NRO, and the screen
 *  stands over the whole of its boot. */
export function loadIdentity(title: string, iconUrl: string | null): void {
  titleEl.textContent = title;
  iconEl.hidden = !iconUrl;
  if (iconUrl) iconEl.src = iconUrl;
  else iconEl.removeAttribute('src');
}

/** Move to the next step of the current load. */
export function loadPhase(phase: string, detail?: string): void {
  awaitingFrame = false;
  setBar(null);
  phaseEl.textContent = phase;
  detailEl.textContent = detail || '';
}

/** A step whose size is known, so the bar can stop guessing. */
export function loadProgress(done: number, total: number): void {
  setBar(total > 0 ? Math.min(1, done / total) : null);
}

/** The guest is running and has not presented anything yet. This is the one
 *  open-ended phase, so it is also the only one that offers a way out: a title
 *  that never presents would otherwise hold the stage for ever. */
export function awaitFirstFrame(): void {
  loadPhase('booting', 'waiting for the first frame');
  awaitingFrame = true;
  dismissEl.hidden = false;
}

/** Mirror the run loop's own readout while the boot is being waited on, so a
 *  title that takes minutes to present is visibly still executing. */
export function bootDetail(pc: number, steps: number): void {
  if (!awaitingFrame) return;
  detailEl.textContent = 'pc 0x' + pc.toString(16).padStart(8, '0')
    + ' · ' + steps.toLocaleString() + ' steps';
}

export function endLoad(): void {
  awaitingFrame = false;
  rootEl.classList.add('hidden');
  dismissEl.hidden = true;
}

/** The load did not finish. The screen stays up saying why, because the
 *  alternative is uncovering a black stage and leaving the reason in a panel
 *  that is closed by default. */
export function failLoad(message: string): void {
  awaitingFrame = false;
  rootEl.classList.remove('hidden');
  rootEl.classList.add('is-error');
  setBar(1);
  phaseEl.textContent = message;
  detailEl.textContent = '';
  dismissEl.hidden = false;
}

dismissEl.addEventListener('click', endLoad);
