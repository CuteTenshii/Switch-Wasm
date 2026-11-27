// The dialog shown when another tab holds the stored card, saves and NAND, so this one keeps no changes.

import { $ } from './dom';
import { onWorker } from './rpc';

const dialogEl = $<HTMLDialogElement>('read-only');

$('btn-read-only-reload').addEventListener('click', () => location.reload());

onWorker('storage_read_only', () => {
  if (!dialogEl.open) dialogEl.showModal();
});
