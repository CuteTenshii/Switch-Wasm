// Errors the running title asked the console's error applet to show.

import type { ErrorDialog } from '../shared/protocol';
import { $ } from './dom';
import { call, hasSession } from './rpc';

const dialogEl = $<HTMLDialogElement>('error-dialog');
const messageEl = $('error-dialog-message');
const codeEl = $('error-dialog-code');

function show(error: ErrorDialog): void {
  messageEl.textContent = error.message || 'An error occurred.';
  codeEl.textContent = error.code ? 'Error code ' + error.code : '';
  codeEl.hidden = !error.code;
  dialogEl.showModal();
}

// Show the next error once the last one is dismissed.
export async function pollErrors(): Promise<void> {
  if (dialogEl.open || !hasSession()) return;
  const error = await call('error_dialog');
  if (error && !dialogEl.open) show(error);
}

// For a session being replaced: its errors go with it.
export function dropErrors(): void {
  dialogEl.close();
}
