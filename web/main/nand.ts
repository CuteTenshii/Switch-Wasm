// The NAND in IndexedDB: NCA content as Blobs by file name, and an index of titles by id.

import type { NandEntry } from '../shared/db';
import { $, el } from './dom';
import { beginLoad } from './loading';
import { log } from './log';
import { call, onNandProgress } from './rpc';
import { setNote } from './shell';
import { doLaunchNca } from './container';

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
  const run = restoring.catch(() => {}).then(async () => {
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
  });
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
  void beginArchiveRestore().then((registered) => {
    if (registered) log('NAND: ' + registered + ' system data archive(s) restored.', 'info');
  }, (err) => {
    log('NAND: the archives could not be registered (' + (err as Error).message + ')', 'error');
  });
}

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
      'Nothing installed - point "Install firmware" at a dump to put the system applets here.'));
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
  row.addEventListener('click', () => launchInstalled(id, entry));
  return row;
}

function launchInstalled(id: string, entry: NandEntry): Promise<void> {
  const name = titleLabel(id);
  beginLoad(name, 'reading ' + entry.name + ' from the NAND');
  return doLaunchNca(name, () => call('nandLaunch', entry.name));
}

// Content and index entry go together; a registered archive stays until the session is replaced.
export async function removeNandTitle(id: string): Promise<void> {
  await call('nandRemove', id);
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

$('firmware-ncas').addEventListener('change', async (e) => {
  const input = e.target as HTMLInputElement;
  const picked = Array.from(input.files || []);
  input.value = '';
  if (!picked.length) return;
  const files = picked.filter((f) => /\.nca$/i.test(f.name));
  if (picked.length !== files.length) {
    log('Ignoring ' + (picked.length - files.length) + ' file(s) that are not .nca.');
  }
  if (!files.length) {
    log('Nothing in that selection is an .nca.', 'error');
    return;
  }
  log('Reading ' + files.length + ' firmware file(s) ...');
  let installed = 0;
  const stateEl = $('firmware-state');
  for (const [index, f] of files.entries()) {
    stateEl.textContent = 'Reading ' + (index + 1) + ' of ' + files.length + ': ' + f.name;
    setNote('nand-badge', (index + 1) + '/' + files.length, false);
    try {
      // Identify from the header without reading the file.
      const what = await call('nandIdentify', f);
      if (!what || what.kind === 2) continue;
      // Store the File itself so the browser owns the bytes.
      await call('nandInstall', f, what.id, what.kind);
      installed++;
    } catch (err) {
      log('Could not read ' + f.name + ': ' + (err as Error).message, 'error');
    }
  }
  try {
    nandTitles = await call('nandTitles');
  } catch { /* the panel just stays as it was */ }
  renderNandTitles();
  // Registered from the NAND so an archive counts once across repeated installs.
  const registered = await beginArchiveRestore();
  updateFirmwareState();
  log('Installed ' + installed + ' title(s) of ' + files.length + ' file(s); '
    + registered + ' registered as system data archives.',
  installed ? 'info' : 'log');
  document.dispatchEvent(new Event('nand-changed'));
});

$('btn-erase-nand').addEventListener('click', async () => {
// Already registered content stays registered until the session is replaced.
  try {
    await call('nandErase');
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
