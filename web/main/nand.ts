// The NAND in IndexedDB: NCA content as Blobs by file name, and an index of titles by id.

import {
  idbApply,
  idbGet,
  idbGetAll,
  NAND_CONTENT,
  NAND_TITLES,
  nandIdb,
  type NandEntry,
} from './db';
import { $, el } from './dom';
import { beginLoad, failLoad } from './loading';
import { log } from './log';
import { call } from './rpc';
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

// Content and index entry go in one transaction so the index never names missing content.
function nandInstall(name: string, content: Blob, titleId: string, kind: number) {
  return nandIdb().then((db) => new Promise<void>((resolve, reject) => {
    const tx = db.transaction([NAND_CONTENT, NAND_TITLES], 'readwrite');
    tx.objectStore(NAND_CONTENT).put(content, name);
    tx.objectStore(NAND_TITLES).put({ name, kind }, titleId);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB request failed'));
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB request failed'));
  }));
}

function nandErase() {
  return nandIdb().then((db) => new Promise<void>((resolve, reject) => {
    const tx = db.transaction([NAND_CONTENT, NAND_TITLES], 'readwrite');
    tx.objectStore(NAND_CONTENT).clear();
    tx.objectStore(NAND_TITLES).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB request failed'));
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB request failed'));
  }));
}

// Content stored by older builds as an ArrayBuffer is rewritten as a Blob on read.
async function nandContent(name: string): Promise<Blob | undefined> {
  const stored = await idbGet<Blob | ArrayBuffer>(await nandIdb(), NAND_CONTENT, name);
  if (!stored) return undefined;
  if (stored instanceof Blob) return stored;
  const blob = new Blob([stored]);
  try {
    await idbApply(await nandIdb(), NAND_CONTENT, [[name, blob]]);
  } catch { /* it is readable either way; the next session pays the same again */ }
  return blob;
}

// In-memory copy of the title index, refreshed on install and erase.
let nandTitles: [string, NandEntry][] = [];

// Registration is per session; a newer restore supersedes older ones at their next check.
let restoreGen = 0;
let restoring: Promise<unknown> = Promise.resolve();

let restoreProgress: { done: number; total: number } | null = null;

// Hand the stored data archives to the current session; null if superseded.
async function registerArchives(gen: number): Promise<number | null> {
  const archives = nandTitles.filter(([, entry]) => entry.kind === 1);
  let registered = 0;
  let done = 0;
  if (archives.length) {
    restoreProgress = { done: 0, total: archives.length };
    updateFirmwareState();
  }
  for (const [, entry] of archives) {
    if (gen !== restoreGen) return null;
    try {
      const content = await nandContent(entry.name);
      if (gen !== restoreGen) return null;
      // Named so the worker's I/O lines can say which archive they read.
      const named = content instanceof File ? content : content && new File([content], entry.name);
      if (named && await call('add_archive', named) === 0) registered++;
    } catch { /* one unreadable archive should not cost the rest */ }
    restoreProgress = { done: ++done, total: archives.length };
    archiveCount = registered;
    updateFirmwareState();
  }
  restoreProgress = null;
  archiveCount = registered;
  updateFirmwareState();
  return registered;
}

// Resolves to the number of archives registered, or null if superseded.
function beginArchiveRestore(): Promise<number | null> {
  const gen = ++restoreGen;
  const earlier = restoring;
  const run = (async () => {
    await earlier.catch(() => {});
    return gen === restoreGen ? registerArchives(gen) : null;
  })();
  restoring = run.catch(() => null);
  return run;
}

// Read the index now; register archives in the background.
export async function initNand(): Promise<void> {
  try {
    nandTitles = await idbGetAll<NandEntry>(await nandIdb(), NAND_TITLES);
  } catch (err) {
    log('NAND: could not be read (' + (err as Error).message + ')', 'err');
    nandTitles = [];
  }
  if (nandTitles.length) {
    log('NAND: ' + nandTitles.length + ' title(s) installed.', 'dim');
  }
  renderNandTitles();
  updateFirmwareState();
  void beginArchiveRestore().then((registered) => {
    if (registered) log('NAND: ' + registered + ' system data archive(s) restored.', 'ok');
  }, (err) => {
    log('NAND: the archives could not be registered (' + (err as Error).message + ')', 'err');
  });
}

function titleLabel(id: string): string {
  return SYSTEM_TITLES[id] || id;
}

// Installed programs, one launchable row each.
function renderNandTitles(): void {
  const host = $('nand-titles');
  host.textContent = '';
  const programs = nandTitles.filter(([, entry]) => entry.kind === 0);
  // Named applets first; unknown ids sort last.
  programs.sort((a, b) => {
    const named = Number(a[0] in SYSTEM_TITLES) - Number(b[0] in SYSTEM_TITLES);
    return named !== 0 ? -named : titleLabel(a[0]).localeCompare(titleLabel(b[0]));
  });
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

async function launchInstalled(id: string, entry: NandEntry): Promise<void> {
  const name = titleLabel(id);
  beginLoad(name, 'reading ' + entry.name + ' from the NAND');
  let content: Blob | undefined;
  try {
    content = await nandContent(entry.name);
  } catch (err) {
    failLoad('NAND: ' + (err as Error).message);
    log('NAND: ' + (err as Error).message, 'err');
    return;
  }
  if (!content) {
    failLoad(entry.name + ' is indexed but its content is missing.');
    log('NAND: ' + entry.name + ' is indexed but its content is missing.', 'err');
    return;
  }
  // The loader maps the whole image, so programs are booted from bytes.
  const bytes = new Uint8Array(await content.arrayBuffer());
  return doLaunchNca(name, () => call('nand_launch', bytes));
}

// Awaited: a title cannot find an archive that is still being registered.
export async function restoreArchives(): Promise<void> {
  await beginArchiveRestore();
}

function updateFirmwareState(): void {
  const held = nandTitles.length ? ', ' + nandTitles.length + ' on the NAND' : '';
  let state: string;
  if (restoreProgress) {
    state = 'Registering system data archives \u2014 ' + restoreProgress.done
      + ' of ' + restoreProgress.total + ' \u2026';
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
    log('Nothing in that selection is an .nca.', 'err');
    return;
  }
  log('Reading ' + files.length + ' firmware file(s) ...');
  let installed = 0;
  const stateEl = $('firmware-state');
  for (const [index, f] of files.entries()) {
    stateEl.textContent = 'Reading ' + (index + 1) + ' of ' + files.length + ' \u2014 ' + f.name;
    setNote('nand-badge', (index + 1) + '/' + files.length, false);
    try {
      // Identify from the header without reading the file.
      const what = await call('nand_identify', f);
      if (!what || what.kind === 2) continue;
      // Store the File itself so the browser owns the bytes.
      await nandInstall(f.name, f, what.id, what.kind);
      installed++;
    } catch (err) {
      log('Could not read ' + f.name + ': ' + (err as Error).message, 'err');
    }
  }
  try {
    nandTitles = await idbGetAll<NandEntry>(await nandIdb(), NAND_TITLES);
  } catch { /* the panel just stays as it was */ }
  renderNandTitles();
  // Registered from the NAND so an archive counts once across repeated installs.
  const registered = await beginArchiveRestore();
  updateFirmwareState();
  log('Installed ' + installed + ' title(s) of ' + files.length + ' file(s); '
    + (registered ?? archiveCount) + ' registered as system data archives.',
  installed ? 'ok' : undefined);
});

$('btn-erase-nand').addEventListener('click', async () => {
// Already registered content stays registered until the session is replaced.
  try {
    await nandErase();
  } catch (err) {
    log('NAND: could not be erased (' + (err as Error).message + ')', 'err');
    return;
  }
  nandTitles = [];
  renderNandTitles();
  updateFirmwareState();
  log('NAND erased.', 'ok');
});
