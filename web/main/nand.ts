/* ---------- the NAND ----------

   A console keeps its system content on internal storage and finds it again
   every boot. Here the equivalent is IndexedDB, in two stores: `content` holds
   the NCA bytes by file name, and `titles` holds what each one *is* by title
   id. Splitting them is what lets the panel list what is installed without
   reading hundreds of megabytes to find out.

   The bytes are what gets stored, not the File - a page cannot reopen a file
   it was not handed, which is the whole reason a firmware dump had to be
   re-picked every session before this existed. */

import {
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

/* Content a title mounts that is not its own: an applet's shared assets, the
   system's Mii and amiibo models. Each is a separate NCA on a console's NAND,
   so there is nothing to find here unless someone hands them over - point this
   at a firmware dump and every data archive in it is registered by title id.

   Only the File references cross over; nothing is read until a title asks for
   one, so selecting a few hundred NCAs costs nothing. They cannot be
   persisted the way keys are - the browser will not hand a page a file again
   without being asked - so this is per session. */
let archiveCount = 0;
// The archives themselves, so a new session can be given them again. The
// browser will not hand a page a file it was not asked for, so losing these
// on reset would mean re-picking a whole firmware dump.
let firmwareFiles: File[] = [];

// What each system title is, for a panel that would otherwise list bare hex.
// Everything else installed is shown by its id.
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

// Install one piece of content: its bytes under its file name, and what it is
// under its title id. Both in one transaction, so the index can never end up
// naming content that is not there.
function nandInstall(name: string, bytes: ArrayBuffer, titleId: string, kind: number) {
  return nandIdb().then((db) => new Promise<void>((resolve, reject) => {
    const tx = db.transaction([NAND_CONTENT, NAND_TITLES], 'readwrite');
    tx.objectStore(NAND_CONTENT).put(bytes, name);
    tx.objectStore(NAND_TITLES).put({ name, kind }, titleId);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  }));
}

function nandErase() {
  return nandIdb().then((db) => new Promise<void>((resolve, reject) => {
    const tx = db.transaction([NAND_CONTENT, NAND_TITLES], 'readwrite');
    tx.objectStore(NAND_CONTENT).clear();
    tx.objectStore(NAND_TITLES).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  }));
}

// What the NAND holds, as [title id, {name, kind}] pairs.
let nandTitles: [string, NandEntry][] = [];

// Hand the stored *data archives* to a fresh session. Programs are not loaded
// here — an installed title is launched on request, not at boot.
async function nandRestore(): Promise<number> {
  try {
    nandTitles = await idbGetAll<NandEntry>(await nandIdb(), NAND_TITLES);
  } catch (err) {
    log('NAND: could not be read (' + (err as Error).message + ')', 'err');
    nandTitles = [];
    return 0;
  }
  let restored = 0;
  for (const [, entry] of nandTitles) {
    if (entry.kind !== 1) continue;
    try {
      const bytes = await idbGet<ArrayBuffer>(await nandIdb(), NAND_CONTENT, entry.name);
      if (bytes && await call('nand_add_archive', new Uint8Array(bytes))) restored++;
    } catch { /* one unreadable archive should not cost the rest */ }
  }
  renderNandTitles();
  return restored;
}

/** The NAND as a fresh page finds it. Nothing was restored here before,
 *  because nothing survived a reload to restore. */
export async function initNand(): Promise<void> {
  const held = await nandRestore();
  if (held) {
    archiveCount = held;
    log('NAND: ' + held + ' system data archive(s) restored.', 'ok');
  }
  if (nandTitles.length) {
    log('NAND: ' + nandTitles.length + ' title(s) installed.', 'dim');
  }
  updateFirmwareState();
}

// A title id nothing is known about is still launchable, so it takes the id's
// place as the row's name rather than leaving the row nameless.
function titleLabel(id: string): string {
  return SYSTEM_TITLES[id] || id;
}

/* The installed programs, one launchable row each. A system applet ships as a
   bare NCA inside firmware with no container around it, so these rows are the
   only place on the page one can be started from - which is why each says what
   it launches instead of being one more small button in a wrap of identical
   small buttons. */
function renderNandTitles(): void {
  const host = $('nand-titles');
  host.textContent = '';
  const programs = nandTitles.filter(([, entry]) => entry.kind === 0);
  // Named applets first, alphabetically; a title id nothing is known about
  // sorts to the bottom rather than to wherever its leading zeroes put it.
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
  // An applet is a few hundred megabytes coming back out of IndexedDB before
  // the launch proper even starts, so the screen goes up here rather than in
  // `doLaunchNca`, which re-titles it with the same name a moment later.
  beginLoad(name, 'reading ' + entry.name + ' from the NAND');
  let bytes: ArrayBuffer | undefined;
  try {
    bytes = await idbGet<ArrayBuffer>(await nandIdb(), NAND_CONTENT, entry.name);
  } catch (err) {
    failLoad('NAND: ' + (err as Error).message);
    log('NAND: ' + (err as Error).message, 'err');
    return;
  }
  if (!bytes) {
    failLoad(entry.name + ' is indexed but its content is missing.');
    log('NAND: ' + entry.name + ' is indexed but its content is missing.', 'err');
    return;
  }
  // Same path a title picked out of a container takes, so a launch from the
  // NAND behaves the same and reports the same way.
  return doLaunchNca(name, () => call('nand_launch', new Uint8Array(bytes)));
}

// Re-register every archive the page still claims to have. Runs before the
// container is re-opened and after the keys are re-staged, because parsing an
// NCA header needs them.
export async function restoreArchives(): Promise<void> {
  // The NAND first: content stored in a previous session, which is the only
  // kind that survives a reload.
  const fromNand = await nandRestore();
  // Then anything picked in *this* session that the NAND does not already
  // hold - the File references are still good until the page goes away.
  const kept = [];
  for (const f of firmwareFiles) {
    if (await call('add_archive', f).catch(() => -1) === 0) kept.push(f);
  }
  firmwareFiles = kept;
  archiveCount = kept.length + fromNand;
  updateFirmwareState();
}

export function updateFirmwareState(): void {
  const held = nandTitles.length ? ', ' + nandTitles.length + ' on the NAND' : '';
  $('firmware-state').textContent = archiveCount === 0
    ? 'No system data archives. A title that mounts one - an applet\'s shared assets, the Mii and amiibo models - will not find it.'
    : archiveCount + ' system data archive(s) registered' + held;
  setNote('nand-badge', nandTitles.length ? nandTitles.length + ' held' : 'empty',
    nandTitles.length > 0);
}

$('firmware-ncas').addEventListener('change', async (e) => {
  const files = Array.from((e.target as HTMLInputElement).files || []);
  if (!files.length) return;
  log('Reading ' + files.length + ' firmware file(s) ...');
  let added = 0;
  let installed = 0;
  // A firmware dump is hundreds of files and several gigabytes through
  // IndexedDB. It is the panel's work rather than the stage's, so it counts
  // itself off where it lives instead of behind a loading screen.
  const stateEl = $('firmware-state');
  for (const [index, f] of files.entries()) {
    stateEl.textContent = 'Reading ' + (index + 1) + ' of ' + files.length + ' \u2014 ' + f.name;
    setNote('nand-badge', (index + 1) + '/' + files.length, false);
    try {
      // Ask what it is first. That reads a header, not a file: a firmware dump
      // runs to gigabytes and most of it is metadata neither worth keeping nor
      // worth pulling through the page to find out.
      const what = await call('nand_identify', f);
      if (!what || what.kind === 2) continue;
      // Now it is worth reading the whole thing, because keeping it is the
      // point - a browser will not hand the page this file again unprompted,
      // and an applet that is not on the NAND cannot be launched at all.
      await nandInstall(f.name, await f.arrayBuffer(), what.id, what.kind);
      installed++;
      if (what.kind === 1) {
        if (await call('add_archive', f) === 0) { added++; firmwareFiles.push(f); }
      }
    } catch (err) {
      log('Could not read ' + f.name + ': ' + (err as Error).message, 'err');
    }
  }
  try {
    nandTitles = await idbGetAll<NandEntry>(await nandIdb(), NAND_TITLES);
  } catch { /* the panel just stays as it was */ }
  renderNandTitles();
  archiveCount = firmwareFiles.length;
  updateFirmwareState();
  // Most of a firmware dump is programs and metadata, not data archives; only
  // the ones that are get registered, so the skipped count is expected.
  log('Installed ' + installed + ' title(s) of ' + files.length + ' file(s); '
    + added + ' registered as system data archives.', installed ? 'ok' : 'dim');
});

$('btn-erase-nand').addEventListener('click', async () => {
  // Erasing the NAND does not disturb the running session: content already
  // registered stays registered until it is replaced, exactly as formatting a
  // console's storage does not unload what is already running.
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
