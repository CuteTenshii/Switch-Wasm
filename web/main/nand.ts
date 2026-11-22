// The NAND: NCA content by file name (OPFS, or IndexedDB without it) and an index of titles by id.

import type { NandEntry } from '../shared/db';
import { $, el } from './dom';
import { beginLoad } from './loading';
import { hasProdKeys, onKeysChanged } from './keys';
import { log } from './log';
import { nandBusy, nandOperation, onNandBusy } from './nandbusy';
import { call, onNandProgress } from './rpc';
import { refuseBootWhileNandBusy } from './session';
import { setNote } from './shell';
import { doLaunchNca } from './container';
import { zipExtract, zipMembers, type ZipMember } from './unzip';

// Shared data archives (applet assets, Mii and amiibo models) registered by title id.
let archiveCount = 0;

const SYSTEM_TITLES: Record<string, string> = {
  '0100000000001000': 'Home Menu',
  '0100000000001001': 'Auth',
  '0100000000001002': 'Cabinet (amiibo)',
  '0100000000001003': 'Controller',
  '0100000000001004': 'Data Erase',
  '0100000000001005': 'Error',
  '0100000000001006': 'Net Connect',
  '0100000000001007': 'User Select',
  '0100000000001008': 'Software Keyboard',
  '0100000000001009': 'Mii Editor',
  '010000000000100a': 'Web',
  '010000000000100b': 'Shop',
  '010000000000100c': 'Overlay',
  '010000000000100d': 'Album',
  '010000000000100f': 'Offline Web',
  '0100000000001010': 'Login Share',
  '0100000000001011': 'Wi-Fi Web Auth',
  '0100000000001012': 'Starter',
  '0100000000001013': 'My Page',
};

// What Eden offers to launch on its own. The rest expect a caller's launch arguments, which a launch
// from here could only make up, so they panic for reasons no title would ever hit.
const STANDALONE_TITLES = new Set([
  '0100000000001000', '0100000000001003', '0100000000001009', '010000000000100d',
]);

// In-memory copy of the title index, refreshed on install and erase.
let nandTitles: [string, NandEntry][] = [];

// Registrations run one after another, each handing the stored firmware to the worker.
let restoring: Promise<unknown> = Promise.resolve();

let registering = false;

let restoreProgress: { done: number; total: number } | null = null;

onNandProgress((done, total) => {
  restoreProgress = { done, total };
  updateFirmwareState();
});

// Resolves to the number of archives registered.
function beginArchiveRestore(): Promise<number> {
  const run = restoring.catch(() => {}).then(() => nandOperation('Registering system data archives and applets', async () => {
    registering = true;
    restoreProgress = null;
    updateFirmwareState();
    try {
      archiveCount = await call('restoreNand');
      return archiveCount;
    } finally {
      registering = false;
      updateFirmwareState();
    }
  }));
  restoring = run.catch(() => null);
  return run;
}

// Read the index now; register archives in the background.
export async function initNand(): Promise<void> {
  try {
    nandTitles = await call('nandTitles');
  } catch (err) {
    log('NAND: could not be read (' + (err as Error).message + ')', 'error');
    nandTitles = [];
  }
  if (nandTitles.length) {
    log('NAND: ' + nandTitles.length + ' title(s) installed.', 'debug');
  }
  renderNandTitles();
  updateFirmwareState();
  // The core logs how many archives it restored.
  void beginArchiveRestore().catch((err) => {
    log('NAND: the archives could not be registered (' + (err as Error).message + ')', 'error');
  });
}

// Launching waits for NAND operations, so the rows say so while one runs.
onNandBusy(() => renderNandTitles());

export function titleLabel(id: string): string {
  return SYSTEM_TITLES[id] || id;
}

// One row for each installed program that can run on its own.
function renderNandTitles(): void {
  const host = $('nand-titles');
  host.textContent = '';
  const programs = nandTitles.filter(([id, entry]) => entry.kind === 0 && STANDALONE_TITLES.has(id));
  programs.sort((a, b) => titleLabel(a[0]).localeCompare(titleLabel(b[0])));
  $('nand-tools').hidden = nandTitles.length === 0;
  if (!programs.length) {
    host.appendChild(el('p', 'muted tiny empty-note',
      'Nothing installed. Install a firmware dump from a folder or a .zip to put the system applets here.'));
    return;
  }
  for (const [id, entry] of programs) host.appendChild(nandRow(id, entry));
}

function nandRow(id: string, entry: NandEntry): HTMLButtonElement {
  const row = el('button', 'row-item');
  row.type = 'button';
  row.title = entry.name;
  const main = el('div', 'row-main');
  main.appendChild(el('span', 'row-name', titleLabel(id)));
  main.appendChild(el('span', 'row-sub', id));
  row.appendChild(main);
  row.appendChild(el('span', 'row-action', '\u25B6 Launch'));
  row.disabled = nandBusy() !== null;
  row.addEventListener('click', () => launchInstalled(id, entry));
  return row;
}

function launchInstalled(id: string, entry: NandEntry): Promise<void> {
  const name = titleLabel(id);
  if (refuseBootWhileNandBusy(name)) return Promise.resolve();
  beginLoad(name, 'Reading ' + entry.name + ' from the NAND');
  return doLaunchNca(name, () => call('nandLaunch', entry.name));
}

// Content and index entry go together; a registered archive stays until the session is replaced.
export async function removeNandTitle(id: string): Promise<void> {
  await nandOperation('Removing a title', () => call('nandRemove', id));
  nandTitles = nandTitles.filter(([titleId]) => titleId !== id);
  renderNandTitles();
  updateFirmwareState();
  document.dispatchEvent(new Event('nand-changed'));
}

function updateFirmwareState(): void {
  const held = nandTitles.length ? ', ' + nandTitles.length + ' on the NAND' : '';
  let state: string;
  if (registering) {
    state = restoreProgress
      ? 'Registering system data archives: ' + restoreProgress.done + ' of '
      + restoreProgress.total + ' \u2026'
      : 'Registering system data archives \u2026';
  } else if (archiveCount === 0) {
    state = 'No system data archives. A title that mounts one - an applet\'s'
      + ' shared assets, the Mii and amiibo models - will not find it.';
  } else {
    state = archiveCount + ' system data archive(s) registered' + held;
  }
  $('firmware-state').textContent = state;
  setNote('nand-badge', nandTitles.length ? nandTitles.length + ' held' : 'empty',
    nandTitles.length > 0);
}

let installListener: ((text: string, error: boolean) => void) | null = null;

// The Files tab shows an install's progress and outcome too, since it offers the same buttons.
export function onInstallStatus(handler: NonNullable<typeof installListener>): void {
  installListener = handler;
}

function installStatus(text: string, error = false): void {
  const status = $('firmware-status');
  status.textContent = text;
  status.classList.toggle('is-error', error);
  installListener?.(text, error);
}

// An NCA to install, read only when its turn comes.
interface FirmwareNca {
  name: string;
  open: () => Promise<Blob>;
}

// Plain `x.nca` files, and the `x.nca/00`, `x.nca/01`… pieces the console's own Contents folder splits them into.
function gatherNcas<T>(
  items: { path: string; item: T }[],
  open: (pieces: T[]) => Promise<Blob>,
): { ncas: FirmwareNca[]; ignored: number; failures: string[] } {
  const found = new Map<string, { name: string; pieces: { order: number; item: T }[] }>();
  let ignored = 0;
  for (const { path, item } of items) {
    const parts = path.split('/');
    const base = parts[parts.length - 1];
    const parent = parts.length > 1 ? parts[parts.length - 2] : '';
    let key: string;
    let name: string;
    let order = 0;
    if (/\.nca$/i.test(base)) {
      key = path;
      name = base;
    } else if (/\.nca$/i.test(parent) && /^\d{2}$/.test(base)) {
      key = parts.slice(0, -1).join('/');
      name = parent;
      order = Number(base);
    } else {
      ignored++;
      continue;
    }
    const nca = found.get(key) ?? { name, pieces: [] };
    nca.pieces.push({ order, item });
    found.set(key, nca);
  }
  const ncas: FirmwareNca[] = [];
  const failures: string[] = [];
  for (const { name, pieces } of found.values()) {
    pieces.sort((a, b) => a.order - b.order);
    if (pieces.some((piece, i) => piece.order !== i)) {
      failures.push(name + ' (some of its pieces are missing)');
      continue;
    }
    ncas.push({ name, open: () => open(pieces.map((piece) => piece.item)) });
  }
  return { ncas, ignored, failures };
}

function installFirmware(
  source: string,
  found: { ncas: FirmwareNca[]; ignored: number; failures: string[] },
): Promise<void> {
  if (refuseInstallWithoutKeys()) return Promise.resolve();
  return nandOperation('Installing firmware', () => installFirmwareNcas(source, found));
}

// Each NCA is identified from its encrypted header before it is stored, which needs prod.keys.
const NEEDS_KEYS = 'Load prod.keys to install firmware: each NCA is identified from its encrypted'
  + ' header before it is stored.';

function refuseInstallWithoutKeys(): boolean {
  if (hasProdKeys()) return false;
  installStatus(NEEDS_KEYS, true);
  return true;
}

// The install buttons are labels for the two inputs; a disabled input opens no picker.
function updateInstallButtons(): void {
  const ready = hasProdKeys();
  for (const id of ['firmware-dir', 'firmware-zip']) {
    $<HTMLInputElement>(id).disabled = !ready;
    for (const label of document.querySelectorAll<HTMLLabelElement>('label[for="' + id + '"]')) {
      label.setAttribute('aria-disabled', String(!ready));
      label.title = ready ? '' : NEEDS_KEYS;
    }
  }
  const status = $('firmware-status');
  if (!ready && !status.textContent) installStatus(NEEDS_KEYS);
  else if (ready && status.textContent === NEEDS_KEYS) installStatus('');
}

onKeysChanged(updateInstallButtons);
updateInstallButtons();

async function installFirmwareNcas(
  source: string,
  { ncas, ignored, failures }: { ncas: FirmwareNca[]; ignored: number; failures: string[] },
): Promise<void> {
  if (!ncas.length && !failures.length) {
    installStatus('Nothing in ' + source + ' is an NCA.', true);
    return;
  }
  let installed = 0;
  for (const [index, nca] of ncas.entries()) {
    installStatus('Reading ' + (index + 1) + ' of ' + ncas.length + ' from ' + source + ' (' + nca.name + ')');
    setNote('nand-badge', (index + 1) + '/' + ncas.length, false);
    try {
      const blob = await nca.open();
      // Identify from the header without reading the rest.
      const what = await call('nandIdentify', blob);
      if (what.kind === 2) continue;
      // Store the Blob itself so the browser owns the bytes.
      await call('nandInstall', blob, nca.name, what.id, what.kind);
      installed++;
    } catch (err) {
      failures.push(nca.name + ' (' + (err as Error).message + ')');
    }
  }
  try {
    nandTitles = await call('nandTitles');
  } catch { /* the panel just stays as it was */ }
  renderNandTitles();
  // Registered from the NAND so an archive counts once across repeated installs.
  const registered = await beginArchiveRestore();
  updateFirmwareState();
  let outcome = 'Installed ' + installed + ' title(s) from ' + source + '; '
    + registered + ' system data archive(s) registered.';
  if (ignored) outcome += ' Skipped ' + ignored + ' file(s) that are not NCAs.';
  if (failures.length) outcome += ' ' + failures.length + ' could not be installed: ' + failures.join(', ') + '.';
  installStatus(outcome, failures.length > 0);
  document.dispatchEvent(new Event('nand-changed'));
}

function takeFiles(e: Event): File[] {
  const input = e.target as HTMLInputElement;
  const picked = Array.from(input.files || []);
  input.value = '';
  return picked;
}

$('firmware-dir').addEventListener('change', (e) => {
  const files = takeFiles(e);
  if (!files.length) return;
  const folder = files[0].webkitRelativePath.split('/')[0] || 'the folder';
  void installFirmware(folder, gatherNcas(
    files.map((file) => ({ path: file.webkitRelativePath || file.name, item: file })),
    (pieces) => Promise.resolve(pieces.length === 1 ? pieces[0] : new Blob(pieces)),
  ));
});

$('firmware-zip').addEventListener('change', async (e) => {
  const [zip] = takeFiles(e);
  if (!zip || refuseInstallWithoutKeys()) return;
  let members: ZipMember[];
  try {
    members = await zipMembers(zip);
  } catch (err) {
    installStatus(zip.name + ' could not be read (' + (err as Error).message + ').', true);
    return;
  }
  await installFirmware(zip.name, gatherNcas(
    members.filter((member) => !member.path.endsWith('/')).map((member) => ({ path: member.path, item: member })),
    async (pieces) => {
      const blobs: Blob[] = [];
      for (const piece of pieces) blobs.push(await zipExtract(zip, piece));
      return blobs.length === 1 ? blobs[0] : new Blob(blobs);
    },
  ));
});

$('btn-erase-nand').addEventListener('click', async () => {
// Already registered content stays registered until the session is replaced.
  try {
    await nandOperation('Erasing the NAND', () => call('nandErase'));
  } catch (err) {
    log('NAND: could not be erased (' + (err as Error).message + ')', 'error');
    return;
  }
  nandTitles = [];
  renderNandTitles();
  updateFirmwareState();
  log('NAND erased.', 'info');
  document.dispatchEvent(new Event('nand-changed'));
});
