// The start screen's game list: the games in a folder the player picked, each launched with
// the update and add-on content found beside it. Chromium keeps the folder for the next visit.

import type { PeekedTitle } from '../shared/protocol';
import { LIBRARY_KEY, LIBRARY_STORE, idbApply, idbGet, libraryIdb } from '../shared/db';
import { bootFile } from './boot';
import { holdForLaunch } from './container';
import { $, el } from './dom';
import { hasProdKeys, onKeysChanged, revealKeys } from './keys';
import { log } from './log';
import { call } from './rpc';
import { knownTitle } from './titles';

const LISTED = /\.(nsp|nsz|xci|nro)$/i;

interface Found {
  file: File;
  // Inside the folder: two files of one name in different sub-folders stay apart.
  path: string;
  peek: PeekedTitle | null;
  iconUrl: string | null;
  // Peeked, whether or not that found anything.
  read: boolean;
  // Peeked with keys loaded and found nothing; read again when the keys change.
  failed: boolean;
}

interface Folder {
  name: string;
  // Null for a folder picked through the file input, which cannot be kept.
  handle: FileSystemDirectoryHandle | null;
  state: 'locked' | 'denied' | 'scanning' | 'listed';
  files: Found[];
}

// One game or homebrew, with what the folder holds for it.
interface Row {
  found: Found;
  update: Found | null;
  dlc: Found[];
}

const overlayEl = $('overlay');
const libraryEl = $('library');
const titleEl = $('library-title');
const statusEl = $('library-status');
const listEl = $('library-list');
const openButton = $<HTMLButtonElement>('overlay-open-folder');
const dirInput = $<HTMLInputElement>('library-dir');

let folder: Folder | null = null;
// The folder whose files are being peeked, so a second pass does not start beside it.
let peeking: Folder | null = null;

const canPickFolder = 'showDirectoryPicker' in window || 'webkitdirectory' in dirInput;
openButton.hidden = !canPickFolder;

function isNro(found: Found): boolean {
  return /\.nro$/i.test(found.file.name);
}

function format(found: Found): string {
  return (found.file.name.match(/\.(\w+)$/)?.[1] ?? '').toUpperCase();
}

function bareName(found: Found): string {
  return found.file.name.replace(LISTED, '');
}

function label(found: Found): string {
  const peek = found.peek;
  return (peek && (peek.kind === 'game' || peek.kind === 'homebrew') && peek.name) || bareName(found);
}

// Numeric parts compared in turn; "260945.283290" and "1.0.1" both order as people expect.
function newerVersion(a: string, b: string): boolean {
  const parts = (v: string) => v.split(/[.-]/).map((n) => Number.parseInt(n, 10) || 0);
  const pa = parts(a);
  const pb = parts(b);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    if ((pa[i] ?? 0) !== (pb[i] ?? 0)) return (pa[i] ?? 0) > (pb[i] ?? 0);
  }
  return false;
}

// Games and homebrew as rows, each update and DLC on its game's row; the ones whose game is
// not in the folder are counted.
function arrange(files: Found[]): { rows: Row[]; strays: number } {
  const updates = new Map<string, Found>();
  const dlc = new Map<string, Found[]>();
  const rows: Row[] = [];
  for (const found of files) {
    const peek = found.peek;
    if (peek?.kind === 'update') {
      const held = updates.get(peek.title_id)?.peek;
      if (!held || (held.kind === 'update' && newerVersion(peek.version, held.version))) {
        updates.set(peek.title_id, found);
      }
    } else if (peek?.kind === 'dlc') {
      dlc.set(peek.title_id, [...(dlc.get(peek.title_id) ?? []), found]);
    } else {
      rows.push({ found, update: null, dlc: [] });
    }
  }
  const games = new Set<string>();
  for (const row of rows) {
    const id = row.found.peek?.title_id;
    if (!id) continue;
    games.add(id);
    row.update = updates.get(id) ?? null;
    row.dlc = dlc.get(id) ?? [];
  }
  let strays = [...updates.keys()].filter((id) => !games.has(id)).length;
  for (const [id, packs] of dlc) if (!games.has(id)) strays += packs.length;
  rows.sort((a, b) => label(a.found).localeCompare(label(b.found)));
  return { rows, strays };
}

function describe(row: Row): string {
  const peek = row.found.peek;
  const parts: string[] = [];
  if (peek && (peek.kind === 'game' || peek.kind === 'homebrew')) {
    if (peek.publisher) parts.push(peek.publisher);
    if (peek.version) parts.push('v' + peek.version);
  }
  const update = row.update?.peek;
  if (update?.kind === 'update') parts.push('update ' + (update.version ? 'v' + update.version : 'included'));
  const packs = row.dlc.reduce((n, f) => n + (f.peek?.kind === 'dlc' ? f.peek.packs.length : 0), 0);
  if (packs) parts.push(packs + ' DLC');
  // A file not read yet is known by its name only, which the row shows; where it sits is new.
  const folderPath = row.found.path.split('/').slice(0, -1).join('/');
  if (!row.found.peek && folderPath) parts.push('in ' + folderPath);
  return parts.join(' · ');
}

function renderRow(row: Row): HTMLElement {
  const item = el('li');
  const button = el('button', 'library-row');
  button.type = 'button';
  button.dataset.path = row.found.path;
  if (row.found.iconUrl) {
    const img = el('img', 'library-icon');
    img.alt = '';
    img.src = row.found.iconUrl;
    button.appendChild(img);
  } else {
    button.appendChild(el('span', 'library-icon blank'));
  }
  const meta = el('span', 'library-meta');
  meta.appendChild(el('span', 'library-name', label(row.found)));
  const detail = describe(row);
  if (detail) meta.appendChild(el('span', 'library-detail', detail));
  button.appendChild(meta);
  button.appendChild(el('span', 'library-format', format(row.found)));
  button.addEventListener('click', () => void launch(row));
  item.appendChild(button);
  return item;
}

function statusLine(text: string, action?: { label: string; run: () => void }): void {
  statusEl.replaceChildren(el('span', null, text));
  if (action) {
    const button = el('button', 'btn small', action.label);
    button.type = 'button';
    button.addEventListener('click', action.run);
    statusEl.appendChild(button);
  }
}

function render(): void {
  const focused = (document.activeElement as HTMLElement | null)?.dataset?.path;
  listEl.replaceChildren();
  statusEl.replaceChildren();
  overlayEl.classList.toggle('has-library', folder !== null);
  libraryEl.hidden = folder === null;
  if (!folder) return;
  titleEl.textContent = folder.name;

  if (folder.state === 'locked') {
    statusLine('The browser needs your permission to read this folder again.',
      { label: 'Allow access', run: () => void unlock() });
    return;
  }
  if (folder.state === 'denied') {
    statusLine('The browser did not allow access. Pick the folder again to list its games.');
    return;
  }
  if (folder.state === 'scanning') {
    statusLine('Looking for games…');
    return;
  }
  if (!folder.files.length) {
    statusLine('No .nsp, .nsz, .xci or .nro files in this folder or its sub-folders.');
    return;
  }

  const { rows, strays } = arrange(folder.files);
  for (const row of rows) listEl.appendChild(renderRow(row));

  const unread = folder.files.filter((f) => !f.read).length;
  const locked = folder.files.some((f) => !f.read && !isNro(f));
  if (peeking === folder && unread) {
    statusLine('Reading game details: ' + (folder.files.length - unread) + ' of ' + folder.files.length);
  } else if (locked && !hasProdKeys()) {
    statusLine('Load prod.keys to see game names and icons, and to put each update and DLC with its game.',
      { label: 'Load prod.keys', run: revealKeys });
  } else if (strays) {
    statusLine(strays === 1
      ? '1 update or DLC file is for a game that is not in this folder.'
      : strays + ' update and DLC files are for games that are not in this folder.');
  }

  if (focused) listEl.querySelector<HTMLElement>(`[data-path="${CSS.escape(focused)}"]`)?.focus();
}

async function launch(row: Row): Promise<void> {
  const extras = [row.update, ...row.dlc].filter((f): f is Found => f !== null).map((f) => f.file);
  await holdForLaunch(extras);
  await bootFile(row.found.file);
}

function found(file: File, path: string): Found {
  return { file, path, peek: null, iconUrl: null, read: false, failed: false };
}

// Dot folders (`.Trash-1000`) and ones the browser refuses (`System Volume Information`) are
// skipped rather than failing the whole folder.
async function scanHandle(dir: FileSystemDirectoryHandle, prefix: string, out: Found[]): Promise<void> {
  try {
    for await (const entry of dir.values()) {
      if (entry.name.startsWith('.')) continue;
      if (entry.kind === 'directory') {
        await scanHandle(entry, prefix + entry.name + '/', out);
      } else if (LISTED.test(entry.name)) {
        out.push(found(await entry.getFile(), prefix + entry.name));
      }
    }
  } catch (err) {
    log('Skipped ' + (prefix || dir.name) + ': ' + (err as Error).message, 'debug');
  }
}

function sortByPath(files: Found[]): Found[] {
  return files.sort((a, b) => a.path.localeCompare(b.path));
}

async function openHandle(handle: FileSystemDirectoryHandle): Promise<void> {
  const mine: Folder = { name: handle.name, handle, state: 'scanning', files: [] };
  folder = mine;
  render();
  const files: Found[] = [];
  await scanHandle(handle, '', files);
  if (folder !== mine) return;
  mine.files = sortByPath(files);
  mine.state = 'listed';
  render();
  await peekAll();
}

// The file input lists every file below the folder, each path starting with the folder's name.
function openPicked(files: FileList): void {
  const all = [...files];
  const name = all[0]?.webkitRelativePath.split('/')[0] ?? '';
  const listed = all
    .map((file) => ({ file, path: file.webkitRelativePath.split('/').slice(1).join('/') }))
    .filter(({ file, path }) => LISTED.test(file.name) && !path.split('/').some((part) => part.startsWith('.')))
    .map(({ file, path }) => found(file, path));
  folder = { name: name || 'Folder', handle: null, state: 'listed', files: sortByPath(listed) };
  render();
  void peekAll();
}

async function peekAll(): Promise<void> {
  const mine = folder;
  if (!mine || peeking === mine) return;
  peeking = mine;
  try {
    for (;;) {
      if (folder !== mine) return;
      const next = mine.files.find((f) => !f.read && (isNro(f) || hasProdKeys()));
      if (!next) return;
      render();
      try {
        const { title, icon } = await call('peekTitle', next.file);
        next.peek = title;
        next.iconUrl = title.kind === 'game' || title.kind === 'homebrew'
          ? knownTitle(title, icon).iconUrl
          : null;
      } catch (err) {
        next.failed = !isNro(next);
        log('No details for ' + next.path + ': ' + (err as Error).message, 'debug');
      }
      next.read = true;
    }
  } finally {
    if (peeking === mine) peeking = null;
    render();
  }
}

async function keep(handle: FileSystemDirectoryHandle | null): Promise<void> {
  try {
    await idbApply(await libraryIdb(), LIBRARY_STORE, [[LIBRARY_KEY, handle]]);
  } catch (err) {
    log('The folder will not be remembered: ' + (err as Error).message, 'debug');
  }
}

async function pick(): Promise<void> {
  if (!window.showDirectoryPicker) {
    dirInput.click();
    return;
  }
  let handle: FileSystemDirectoryHandle;
  try {
    handle = await window.showDirectoryPicker({ id: 'games', mode: 'read' });
  } catch (err) {
    // Closing the picker is not a failure.
    if ((err as DOMException).name !== 'AbortError') {
      log('Could not open the folder: ' + (err as Error).message, 'error');
    }
    return;
  }
  await keep(handle);
  await openHandle(handle);
}

// Reading a kept folder again needs a click, which is what calls this.
async function unlock(): Promise<void> {
  const handle = folder?.handle;
  if (!folder || !handle) return;
  if (await handle.requestPermission({ mode: 'read' }) === 'granted') {
    await openHandle(handle);
    return;
  }
  folder.state = 'denied';
  render();
}

async function close(): Promise<void> {
  folder = null;
  render();
  await keep(null);
}

/** Open the folder kept from the last visit. Called once the core has started and restored its
 * storage: a peek sent while the worker is still restoring would run inside that restore. */
export async function restoreLibrary(): Promise<void> {
  if (!window.showDirectoryPicker) return;
  let handle: FileSystemDirectoryHandle | undefined;
  try {
    handle = await idbGet<FileSystemDirectoryHandle>(await libraryIdb(), LIBRARY_STORE, LIBRARY_KEY);
  } catch (err) {
    log('The kept folder could not be read: ' + (err as Error).message, 'debug');
    return;
  }
  // A folder picked since the page loaded wins.
  if (!handle || folder) return;
  if (await handle.queryPermission({ mode: 'read' }) === 'granted') {
    await openHandle(handle);
  } else {
    folder = { name: handle.name, handle, state: 'locked', files: [] };
    render();
  }
}

openButton.addEventListener('click', () => void pick());
$('library-change').addEventListener('click', () => void pick());
$('library-close').addEventListener('click', () => void close());
dirInput.addEventListener('change', () => {
  const files = dirInput.files;
  if (files?.length) openPicked(files);
  dirInput.value = '';
});

onKeysChanged(() => {
  if (!folder) return;
  // New keys may read what the old ones could not.
  for (const f of folder.files) {
    if (f.failed) {
      f.read = false;
      f.failed = false;
    }
  }
  if (hasProdKeys()) void peekAll();
  else render();
});
