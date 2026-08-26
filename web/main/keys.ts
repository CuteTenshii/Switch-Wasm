/* ---------- keys ----------

   Keys are persisted in localStorage so they survive page reloads (they're
   just text; they never leave the browser). */

import { $, pickedFile } from './dom';
import { readKeysFile } from './filetype';
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

// Anything picked here is persisted and used to decrypt with, so a file that
// is not keys is refused rather than stored: the alternative is every later
// NCA failing to open with nothing pointing back at this.
async function acceptKeys(e: Event, which: 'prod' | 'title'): Promise<void> {
  const file = pickedFile(e);
  (e.target as HTMLInputElement).value = '';
  if (!file) return;
  let text: string | null;
  try {
    text = await readKeysFile(file);
  } catch (err) {
    refuseKeys('Could not read ' + file.name + ': ' + (err as Error).message);
    return;
  }
  if (text === null) {
    refuseKeys(file.name + ' is not a keys file - expected "name = hex" lines.');
    return;
  }
  if (which === 'prod') prodKeysText = text;
  else titleKeysText = text;
  restoredKeys = false;
  localStorage.setItem(KEYS_STORE[which], text);
  await stageKeys();
  log(which === 'prod'
    ? 'prod.keys loaded - NCA header decryption enabled.'
    : 'title.keys loaded.', 'ok');
}

function refuseKeys(why: string): void {
  $('keys-state').textContent = why;
  log(why, 'err');
}

$('prod-keys').addEventListener('change', (e) => acceptKeys(e, 'prod'));
$('title-keys').addEventListener('change', (e) => acceptKeys(e, 'title'));

$('btn-clear-keys').addEventListener('click', () => {
  prodKeysText = '';
  titleKeysText = '';
  localStorage.removeItem(KEYS_STORE.prod);
  localStorage.removeItem(KEYS_STORE.title);
  restoredKeys = false;
  stageKeys();
  log('Keys cleared.', 'dim');
});
