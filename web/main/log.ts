/* The on-page console: everything the emulator has to say, mirrored into
   DevTools so it can be filtered by severity there too. */

import { $, el } from './dom';
import { openPanel } from './shell';

export type LogClass = 'err' | 'ok' | 'dim';

const consoleEl = $('console');
const autoscrollCb = $<HTMLInputElement>('autoscroll-cb');
const TAG = '[switch-wasm]';

export function log(msg: string, cls?: LogClass): void {
  // Real browser console (DevTools): route by severity for filterability.
  if (cls === 'err') console.error(TAG, msg);
  else if (cls === 'ok') console.info(TAG, msg);
  else if (cls === 'dim') console.debug(TAG, msg);
  else console.log(TAG, msg);
  // On-page console mirror.
  consoleEl.appendChild(el('div', cls, msg));
  if (autoscrollCb.checked) consoleEl.scrollTop = consoleEl.scrollHeight;
  // Anything that went wrong is worth surfacing even with the panel closed.
  if (cls === 'err') openPanel('console');
}

export function clearConsole(): void {
  consoleEl.textContent = '';
}

$('btn-clear-console').addEventListener('click', clearConsole);

/** The whole on-page log as text, one line per entry.
 *
 *  Not `consoleEl.textContent`: every entry is its own `<div>`, and that
 *  property concatenates their text with nothing in between, so a register
 *  dump and the trace after it would arrive as one unbroken line. */
function consoleText(): string {
  return Array.from(consoleEl.children).map((node) => node.textContent).join('\n');
}

const copyBtn = $('btn-copy-console');

/** Say what happened on the button itself and put its label back. A log copy
 *  is worth confirming -- there is no other sign it worked -- but not worth a
 *  line in the log it just copied. */
let copyLabelTimer = 0;
function flashCopyLabel(text: string): void {
  clearTimeout(copyLabelTimer);
  copyBtn.textContent = text;
  copyLabelTimer = setTimeout(() => { copyBtn.textContent = 'Copy all'; }, 1400);
}

/** `navigator.clipboard` needs a secure context, which a page served over
 *  plain http from another machine is not -- and that is exactly how this gets
 *  opened when someone is testing on a phone. Fall back to a selection copy,
 *  which has no such requirement. */
function copyViaSelection(text: string): boolean {
  const area = el('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  // Off-screen rather than hidden: a display:none textarea cannot be selected.
  area.style.cssText = 'position:fixed;top:-1000px;left:-1000px;opacity:0';
  document.body.appendChild(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } catch {
    ok = false;
  }
  area.remove();
  return ok;
}

async function copyConsole(): Promise<void> {
  const text = consoleText();
  if (!text) {
    flashCopyLabel('Log is empty');
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    flashCopyLabel('Copied');
    return;
  } catch {
    // Fall through: no clipboard API, or the permission was refused.
  }
  flashCopyLabel(copyViaSelection(text) ? 'Copied' : 'Copy failed');
}

copyBtn.addEventListener('click', copyConsole);
