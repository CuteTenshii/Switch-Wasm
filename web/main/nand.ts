/* the NAND

   A console keeps its system content on internal storage and finds it again
   every boot. Here the equivalent is IndexedDB, in two stores: `content` holds
   the NCA bytes by file name, and `titles` holds what each one *is* by title
   id. Splitting them is what lets the panel list what is installed without
   reading hundreds of megabytes to find out.

   The content itself is what gets stored, not a reference to where the user
   found it - a page cannot reopen a file it was not handed, which is the whole
   reason a firmware dump had to be re-picked every session before this
   existed. It is stored as a `Blob`, so the browser goes on owning the bytes
   and hands back a handle: the worker reads ranges out of one exactly as it
   reads a container the user just picked, and a NAND full of firmware is
   registered for the cost of its headers rather than every byte through the
   page, the postMessage boundary and the wasm heap. */

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

/* Content a title mounts that is not its own: an applet's shared assets, the
   system's Mii and amiibo models. Each is a separate NCA on a console's NAND,
   so there is nothing to find here unless someone hands them over - point this
   at a firmware dump and every data archive in it is registered by title id.

   Nothing is read but each one's header until a title asks for it, so handing
   a session a whole dump's worth costs a round trip apiece. */
let archiveCount = 0;

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
function nandInstall(name: string, content: Blob, titleId: string, kind: number) {
  return nandIdb().then((db) => new Promise<void>((resolve, reject) => {
    const tx = db.transaction([NAND_CONTENT, NAND_TITLES], 'readwrite');
    tx.objectStore(NAND_CONTENT).put(content, name);
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

/** One piece of content, as something that can be read without pulling all of
 *  it into memory first.
 *
 *  Content written before this was a Blob is an ArrayBuffer, and reading one
 *  costs exactly what it always did. It is written back as a Blob on the way
 *  past, so a NAND filled by an older build pays that once and every session
 *  after it gets a handle. */
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

// What the NAND holds, as [title id, {name, kind}] pairs. The one copy: it is
// read at start-up and rewritten by an install or an erase, so nothing else
// has to go back to the store to find out what is there.
let nandTitles: [string, NandEntry][] = [];

/* Registering the archives is per session - a session is handed them when it
   is built and loses them when it is thrown away - so only the newest
   registration counts. A run takes a ticket and stops at its next check once a
   later one has been started, because every call it has left to make would
   land in a session that no longer exists. */
let restoreGen = 0;
let restoring: Promise<unknown> = Promise.resolve();

// How far the registration in flight has got, so the panel can say so instead
// of reporting the count it had before it started.
let restoreProgress: { done: number; total: number } | null = null;

/** Hand the stored *data archives* to the session running now. Programs are
 *  not loaded here - an installed title is launched on request, not at boot.
 *  Returns how many the session took, or null if a later restore replaced
 *  this one part-way through. */
async function registerArchives(gen: number): Promise<number | null> {
  const archives = nandTitles.filter(([, entry]) => entry.kind === 1);
  let registered = 0;
  let done = 0;
  // Said before the first one is read rather than after, or the panel spends
  // the whole of a first pass over content an older build stored - which is
  // the one pass that reads every byte - insisting there is nothing there.
  if (archives.length) {
    restoreProgress = { done: 0, total: archives.length };
    updateFirmwareState();
  }
  for (const [, entry] of archives) {
    if (gen !== restoreGen) return null;
    try {
      const content = await nandContent(entry.name);
      if (gen !== restoreGen) return null;
      if (content && await call('add_archive', content) === 0) registered++;
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

/** Start registering what the NAND holds, superseding whatever registration
 *  was still in flight. Resolves to how many archives the session took, or
 *  null if it was itself superseded. */
function beginArchiveRestore(): Promise<number | null> {
  const gen = ++restoreGen;
  const earlier = restoring;
  const run = (async () => {
    // The superseded run stops at its next check, which is one call away.
    await earlier.catch(() => {});
    return gen === restoreGen ? registerArchives(gen) : null;
  })();
  restoring = run.catch(() => null);
  return run;
}

/** The NAND as a fresh page finds it: the index, which is small, and then the
 *  archives, which are not the page's to wait for.
 *
 *  Reading the index is what fills the panel, and it is the only part a
 *  start-up has to finish. Registering the archives happens behind the page -
 *  a file dropped on the stage while it runs rebuilds the session and starts
 *  the registration again anyway, so waiting for it would be waiting for work
 *  that is about to be redone. */
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
    // Nothing is awaiting this one, so it has to say so itself or fail in
    // silence with the page looking ready.
    log('NAND: the archives could not be registered (' + (err as Error).message + ')', 'err');
  });
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
  // A program is booted from bytes, not read a range at a time: the loader
  // maps the whole image, so there is nothing for a handle to defer.
  const bytes = new Uint8Array(await content.arrayBuffer());
  // Same path a title picked out of a container takes, so a launch from the
  // NAND behaves the same and reports the same way.
  return doLaunchNca(name, () => call('nand_launch', bytes));
}

/** Give a rebuilt session the archives the NAND holds.
 *
 *  Awaited by the paths that rebuild a session, unlike the start-up one: a
 *  title mounts what it needs early, and an archive still being registered
 *  when it does is one it does not find. */
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
  // A dump is selected whole, so the metadata and stray files sitting beside
  // the NCAs are turned away by name here rather than each costing a header
  // read through the worker to find out the same thing.
  const files = picked.filter((f) => /\.nca$/i.test(f.name));
  if (picked.length !== files.length) {
    log('Ignoring ' + (picked.length - files.length) + ' file(s) that are not .nca.', 'dim');
  }
  if (!files.length) {
    log('Nothing in that selection is an .nca.', 'err');
    return;
  }
  log('Reading ' + files.length + ' firmware file(s) ...');
  let installed = 0;
  // A firmware dump is hundreds of files and several gigabytes. It is the
  // panel's work rather than the stage's, so it counts itself off where it
  // lives instead of behind a loading screen.
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
      // The File itself is what is stored, not its bytes. The browser copies
      // it into its own storage and the page never holds it: a dump that used
      // to pass through the JS heap a file at a time now does not pass through
      // it at all.
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
  // Registered from the NAND rather than from what was just picked, so an
  // archive counts once whether this is the first install or the third of the
  // same dump.
  const registered = await beginArchiveRestore();
  updateFirmwareState();
  // Most of a firmware dump is programs and metadata, not data archives; only
  // the ones that are get registered, so the skipped count is expected.
  log('Installed ' + installed + ' title(s) of ' + files.length + ' file(s); '
    + (registered ?? archiveCount) + ' registered as system data archives.',
    installed ? 'ok' : 'dim');
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
