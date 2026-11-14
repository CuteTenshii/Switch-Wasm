// The dialog shown when another tab holds the stored card, saves and NAND, so this one keeps no changes.

import { $ } from './dom';

const dialogEl = $<HTMLDialogElement>('read-only');

$('btn-read-only-reload').addEventListener('click', () => location.reload());

export function showReadOnly(): void {
  if (!dialogEl.open) dialogEl.showModal();
}
