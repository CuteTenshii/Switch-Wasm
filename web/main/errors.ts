// Errors the running title asked the console's error applet to show.

import type { ErrorDialog } from '../shared/protocol';
import { $ } from './dom';
import { hasSession } from './rpc';

const dialogEl = $<HTMLDialogElement>('error-dialog');
const messageEl = $('error-dialog-message');
const codeEl = $('error-dialog-code');

function show(error: ErrorDialog): void {
  messageEl.textContent = error.message || 'An error occurred.';
  codeEl.textContent = error.code ? 'Error code ' + error.code : '';
  codeEl.hidden = !error.code;
  dialogEl.showModal();
}

// Errors not shown yet; the next appears once the last is dismissed.
const queued: ErrorDialog[] = [];

function showNext(): void {
  const next = queued.shift();
  if (next && hasSession()) show(next);
}

export function offerErrors(errors: ErrorDialog[]): void {
  queued.push(...errors);
  if (!dialogEl.open) showNext();
}

dialogEl.addEventListener('close', showNext);

// For a session being replaced: its errors go with it.
export function dropErrors(): void {
  queued.length = 0;
  dialogEl.close();
}
