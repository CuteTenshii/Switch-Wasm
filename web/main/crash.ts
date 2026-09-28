/* The stage-level end state for a run that cannot continue. The detailed
   evidence remains in the console and crash report; this surface says what
   happened and puts the recovery paths where the guest screen used to be. */

import { saveCrashReport } from './debug';
import { $ } from './dom';
import { log } from './log';
import { openPanel } from './shell';

const rootEl = $('crash');
const titleEl = $('crash-title');
const messageEl = $('crash-message');
const saveEl = $<HTMLButtonElement>('crash-save');

function showCrash(title: string, message: string): void {
  titleEl.textContent = title;
  messageEl.textContent = message;
  rootEl.hidden = false;
  saveEl.focus();
}

export function showGuestCrash(message: string): void {
  showCrash('Guest crashed', message);
}

export function showEmulatorCrash(message: string): void {
  showCrash('Emulator crashed', message);
}

export function hideCrash(): void {
  rootEl.hidden = true;
}

saveEl.addEventListener('click', async () => {
  try {
    await saveCrashReport();
  } catch (err) {
    openPanel('console');
    log('Could not build the crash report: ' + (err as Error).message, 'err');
  }
});

$('crash-console').addEventListener('click', () => openPanel('console'));
$('crash-reset').addEventListener('click', () => {
  const resetEl = $<HTMLButtonElement>('btn-reset');
  hideCrash();
  resetEl.focus();
  resetEl.click();
});
