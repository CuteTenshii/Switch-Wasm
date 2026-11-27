// Key files, persisted in localStorage.
import { $, pickedFile } from './dom';
import { readKeysFile } from './filetype';
import { log } from './log';
import { call, hasSession } from './rpc';
import { openPanel, setNote } from './shell';

const KEYS_STORE = { prod: 'switch-prod-keys', title: 'switch-title-keys' };

let prodKeysText = localStorage.getItem(KEYS_STORE.prod) || '';
let titleKeysText = localStorage.getItem(KEYS_STORE.title) || '';

export function hasKeys(): boolean {
  return Boolean(prodKeysText || titleKeysText);
}

export function hasProdKeys(): boolean {
  return Boolean(prodKeysText);
}

// Open the side panel on the Keys section, with focus on it for the keyboard.
export function revealKeys(): void {
  openPanel('nand');
  const section = $<HTMLDetailsElement>('keys-section');
  section.open = true;
  section.scrollIntoView({ block: 'nearest' });
  section.querySelector('summary')?.focus();
}

const keysListeners: (() => void)[] = [];

// Call `handler` whenever the keys are loaded, restored or cleared.
export function onKeysChanged(handler: () => void): void {
  keysListeners.push(handler);
}

export async function stageKeys(): Promise<void> {
  if (!hasSession()) return;
  const prod = prodKeysText ? new TextEncoder().encode(prodKeysText) : null;
  const title = titleKeysText ? new TextEncoder().encode(titleKeysText) : null;
  await call('loadKeys', prod, title);
  updateKeysState();
}

export function updateKeysState(): void {
  const parts = [];
  if (prodKeysText) parts.push('prod.keys');
  if (titleKeysText) parts.push('title.keys');
  setNote('keys-badge', parts.length ? parts.length + ' loaded' : 'none', parts.length > 0);
  for (const handler of keysListeners) handler();
}

// Refuse files that are not keys rather than persisting them.
async function acceptKeys(e: Event, which: 'prod' | 'title'): Promise<void> {
  const file = pickedFile(e);
  (e.target as HTMLInputElement).value = '';
  if (!file) return;
  // A file picker filters by extension only, so the other key file would get through.
  const expected = which + '.keys';
  if (file.name.toLowerCase() !== expected) {
    log(file.name + ' is not ' + expected + '. Pick the file named ' + expected + '.', 'error');
    return;
  }
  let text: string | null;
  try {
    text = await readKeysFile(file);
  } catch (err) {
    log('Could not read ' + file.name + ': ' + (err as Error).message, 'error');
    return;
  }
  if (text === null) {
    log(file.name + ' is not a keys file - expected "name = hex" lines.', 'error');
    return;
  }
  if (which === 'prod') prodKeysText = text;
  else titleKeysText = text;
  localStorage.setItem(KEYS_STORE[which], text);
  await stageKeys();
  log(which === 'prod'
    ? 'prod.keys loaded - NCA header decryption enabled.'
    : 'title.keys loaded.', 'info');
}

$('prod-keys').addEventListener('change', (e) => acceptKeys(e, 'prod'));
$('title-keys').addEventListener('change', (e) => acceptKeys(e, 'title'));

$('btn-clear-keys').addEventListener('click', () => {
  prodKeysText = '';
  titleKeysText = '';
  localStorage.removeItem(KEYS_STORE.prod);
  localStorage.removeItem(KEYS_STORE.title);
  void stageKeys();
  log('Keys cleared.');
});
