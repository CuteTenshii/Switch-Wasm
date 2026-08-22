/* ---------- keys ----------

   Keys are persisted in localStorage so they survive page reloads (they're
   just text; they never leave the browser). */

import { $, pickedFile } from './dom';
import { log } from './log';
import { call, isReady } from './rpc';
import { setNote } from './shell';

const KEYS_STORE = { prod: 'switch-prod-keys', title: 'switch-title-keys' };

let prodKeysText = localStorage.getItem(KEYS_STORE.prod) || '';
let titleKeysText = localStorage.getItem(KEYS_STORE.title) || '';
let restoredKeys = Boolean(prodKeysText || titleKeysText);

/** Whether there is anything to hand a new session. */
export function hasKeys(): boolean {
  return Boolean(prodKeysText || titleKeysText);
}

export async function stageKeys(): Promise<number | undefined> {
  if (!isReady()) return;
  const prod = prodKeysText ? new TextEncoder().encode(prodKeysText) : null;
  const title = titleKeysText ? new TextEncoder().encode(titleKeysText) : null;
  const rc = await call('load_keys', prod, title);
  updateKeysState();
  return rc;
}

export function updateKeysState(): void {
  const parts = [];
  if (prodKeysText) parts.push('prod.keys');
  if (titleKeysText) parts.push('title.keys');
  $('keys-state').textContent = parts.length === 0
    ? 'no keys loaded - encrypted NCA headers can\'t be inspected'
    : 'loaded: ' + parts.join(' + ') + (restoredKeys ? ' (from storage)' : '');
  setNote('keys-badge', parts.length ? parts.length + ' loaded' : 'none', parts.length > 0);
}

$('prod-keys').addEventListener('change', async (e) => {
  const f = pickedFile(e);
  if (!f) return;
  prodKeysText = await f.text();
  restoredKeys = false;
  localStorage.setItem(KEYS_STORE.prod, prodKeysText);
  await stageKeys();
  log('prod.keys loaded - NCA header decryption enabled.', 'ok');
});

$('title-keys').addEventListener('change', async (e) => {
  const f = pickedFile(e);
  if (!f) return;
  titleKeysText = await f.text();
  restoredKeys = false;
  localStorage.setItem(KEYS_STORE.title, titleKeysText);
  await stageKeys();
  log('title.keys loaded.', 'ok');
});

$('btn-clear-keys').addEventListener('click', () => {
  prodKeysText = '';
  titleKeysText = '';
  localStorage.removeItem(KEYS_STORE.prod);
  localStorage.removeItem(KEYS_STORE.title);
  restoredKeys = false;
  stageKeys();
  log('Keys cleared.', 'dim');
});
