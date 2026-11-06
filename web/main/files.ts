// The file manager: the SD card, save data and NAND titles as stored in this browser.

import type { StorageInfo, StorageVolume } from '../shared/protocol';
import {
  Archive, Database, Download, File, FileArchive, FilePen, FileUp, Folder, FolderPlus, FolderUp,
  MemoryStick, Package, PackagePlus, Pencil, Play, Save, Trash2, type IconNode,
} from 'lucide';
import { launchFromSd } from './boot';
import { $, el } from './dom';
import { showView } from './fileview';
import { fmtSize } from './format';
import { icon } from './icons';
import { removeNandTitle, titleLabel } from './nand';
import { call, hasSession } from './rpc';

type Place =
  | { kind: 'sd'; dir: string }
  | { kind: 'saves' }
  | { kind: 'save'; id: string; dir: string }
  | { kind: 'nand' };

const sdVolume: StorageVolume = { kind: 'sd' };

function saveVolume(id: string): StorageVolume {
  return { kind: 'save', id };
}

const dialog = $<HTMLDialogElement>('files');
const list = $('files-list');
const crumbs = $('files-crumbs');
const status = $('files-status');
const note = $('files-note');
const view = $('files-view');
const viewTools = $('files-view-tools');
const tools = $('files-tools');
const uploadInput = $<HTMLInputElement>('files-upload');
const uploadDirInput = $<HTMLInputElement>('files-upload-dir');

let place: Place = { kind: 'sd', dir: '/' };
// The open file, and whether it has unsaved edits.
let viewing: { name: string; dirty: () => boolean } | null = null;
// Bumped on every render so a slow read cannot paint over a newer one.
let generation = 0;

function join(dir: string, name: string): string {
  return dir === '/' ? '/' + name : dir + '/' + name;
}

function basename(path: string): string {
  return path.slice(path.lastIndexOf('/') + 1);
}

function setStatus(text: string, error = false): void {
  status.textContent = text;
  status.classList.toggle('is-error', error);
}

// A directory's children, counting directories only implied by a deeper path.
function children(all: Map<string, StorageInfo>, dir: string): Map<string, StorageInfo> {
  const prefix = dir === '/' ? '/' : dir + '/';
  const out = new Map<string, StorageInfo>();
  for (const [path, info] of all) {
    if (!path.startsWith(prefix) || path === dir) continue;
    const rest = path.slice(prefix.length);
    const cut = rest.indexOf('/');
    if (cut < 0) out.set(rest, info);
    else if (!out.has(rest.slice(0, cut))) out.set(rest.slice(0, cut), { kind: 'dir', size: 0 });
  }
  return out;
}

function saveBlob(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = el('a');
  link.href = url;
  link.download = name;
  link.click();
  // Not before the click: revoking the URL cancels the download.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

async function exportZip(volume: StorageVolume, root: string, name: string): Promise<void> {
  const { blob, entries } = await call('storageExport', volume, root);
  saveBlob(name + '.zip', blob);
  setStatus('Exported ' + entries + ' entries to ' + name + '.zip.');
}

// Upload `files` (paths relative to `dir`) one at a time, so memory holds one file.
async function upload(volume: StorageVolume, dir: string, files: [string, File][]): Promise<void> {
  let done = 0;
  let bytes = 0;
  for (const [rel, file] of files) {
    setStatus('Uploading ' + (done + 1) + ' of ' + files.length + ': ' + rel);
    bytes += await call('storageUpload', volume, join(dir, rel), file);
    done++;
  }
  setStatus('Uploaded ' + done + ' file(s), ' + fmtSize(bytes) + '.');
}

// Files from a drop, keeping each one's path inside a dropped folder.
async function droppedFiles(transfer: DataTransfer): Promise<[string, File][]> {
  const out: [string, File][] = [];
  const walk = async (entry: FileSystemEntry, at: string): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) =>
        (entry as FileSystemFileEntry).file(resolve, reject));
      out.push([at + entry.name, file]);
      return;
    }
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    // `readEntries` hands back a batch at a time until it returns none.
    for (;;) {
      const batch = await new Promise<FileSystemEntry[]>((resolve, reject) =>
        reader.readEntries(resolve, reject));
      if (!batch.length) break;
      for (const child of batch) await walk(child, at + entry.name + '/');
    }
  };
  const entries = [...transfer.items]
    .map((item) => item.webkitGetAsEntry())
    .filter((entry): entry is FileSystemEntry => entry !== null);
  if (!entries.length) return [...transfer.files].map((f) => [f.name, f]);
  for (const entry of entries) await walk(entry, '');
  return out;
}

function volumeOf(at: Place): StorageVolume | null {
  if (at.kind === 'sd') return sdVolume;
  if (at.kind === 'save') return saveVolume(at.id);
  return null;
}

function dirOf(at: Place): string {
  return at.kind === 'sd' || at.kind === 'save' ? at.dir : '/';
}

// Run `then` now, or after the user agrees to drop unsaved edits.
function leaveView(then: () => void): void {
  if (!viewing?.dirty()) {
    viewing = null;
    then();
    return;
  }
  const keep = [...viewTools.childNodes];
  viewTools.textContent = '';
  viewTools.appendChild(el('span', 'files-confirm', 'Discard your edits?'));
  const discard = el('button', 'btn small ghost danger', 'Discard');
  discard.type = 'button';
  discard.addEventListener('click', () => {
    viewing = null;
    then();
  });
  const back = el('button', 'btn small ghost', 'Keep editing');
  back.type = 'button';
  back.addEventListener('click', () => viewTools.replaceChildren(...keep));
  viewTools.append(discard, back);
  discard.focus();
}

function go(next: Place): void {
  leaveView(() => {
    place = next;
    void render();
  });
}

function renderCrumbs(): void {
  crumbs.textContent = '';
  const crumb = (label: string, target: Place | null) => {
    if (crumbs.childNodes.length) crumbs.appendChild(el('span', 'files-sep', '/'));
    if (!target) {
      crumbs.appendChild(el('span', 'files-crumb is-here', label));
      return;
    }
    const button = el('button', 'files-crumb', label);
    button.type = 'button';
    button.addEventListener('click', () => go(target));
    crumbs.appendChild(button);
  };
  const path = (root: string, dir: string, make: (d: string) => Place) => {
    const parts = dir.split('/').filter(Boolean);
    crumb(root, parts.length ? make('/') : null);
    parts.forEach((part, i) => {
      const target = '/' + parts.slice(0, i + 1).join('/');
      crumb(part, i === parts.length - 1 ? null : make(target));
    });
  };
  if (place.kind === 'sd') path('sdmc:', place.dir, (dir) => ({ kind: 'sd', dir }));
  else if (place.kind === 'nand') crumb('System titles', null);
  else if (place.kind === 'saves') crumb('Saves', null);
  else {
    const id = place.id;
    crumb('Saves', { kind: 'saves' });
    path(id, place.dir, (dir) => ({ kind: 'save', id, dir }));
  }
  if (viewing) {
    // The folder crumb was the current one; make it lead back to the listing.
    const here = crumbs.querySelector('.files-crumb.is-here');
    if (here) {
      const back = el('button', 'files-crumb', here.textContent ?? '');
      back.type = 'button';
      back.addEventListener('click', () => go(place));
      here.replaceWith(back);
    }
    crumbs.appendChild(el('span', 'files-sep', '/'));
    crumbs.appendChild(el('span', 'files-crumb is-here', viewing.name));
  }
}

function renderTools(): void {
  tools.hidden = viewing !== null;
  viewTools.hidden = viewing === null;
  list.hidden = viewing !== null;
  view.hidden = viewing === null;
  const files = place.kind === 'sd' || place.kind === 'save';
  $('btn-files-mkdir').hidden = !files;
  $('files-upload-label').hidden = !files;
  $('files-upload-dir-label').hidden = !files;
  $('btn-files-export').hidden = !files;
  $('files-install-label').hidden = place.kind !== 'nand';
  document.querySelectorAll<HTMLElement>('.files-tab').forEach((tab) => {
    const on = tab.dataset.volume === (place.kind === 'save' ? 'saves' : place.kind);
    tab.classList.toggle('is-active', on);
    tab.setAttribute('aria-selected', String(on));
  });
  note.textContent = place.kind === 'nand'
    ? 'Removing a data archive takes effect once the running title is replaced.'
    : hasSession()
      ? 'Changes reach the running title straight away.'
      : 'Changes are kept in this browser and load with the next title.';
}

function message(text: string): void {
  list.textContent = '';
  list.appendChild(el('li', 'files-empty muted', text));
}

function row(kind: IconNode, name: string, sub: string, open: (() => void) | null): {
  item: HTMLLIElement; actions: HTMLElement;
} {
  const item = el('li', 'files-row');
  const main = el(open ? 'button' : 'div', 'files-main');
  if (main instanceof HTMLButtonElement) {
    main.type = 'button';
    main.addEventListener('click', () => open?.());
  }
  const text = el('span', 'files-text');
  text.appendChild(el('span', 'files-name', name));
  if (sub) text.appendChild(el('span', 'files-sub', sub));
  const glyph = icon(kind);
  glyph.classList.add('files-kind');
  main.append(glyph, text);
  const actions = el('div', 'files-row-actions');
  item.append(main, actions);
  list.appendChild(item);
  return { item, actions };
}

// An icon button when `glyph` is given, its label then the accessible name and tooltip.
function action(actions: HTMLElement, label: string, run: () => void, danger = false, glyph?: IconNode): void {
  const button = el('button', 'btn small ghost' + (danger ? ' danger' : '') + (glyph ? ' files-act' : ''));
  button.type = 'button';
  if (glyph) {
    button.setAttribute('aria-label', label);
    button.title = label;
    button.appendChild(icon(glyph));
  } else {
    button.textContent = label;
  }
  button.addEventListener('click', run);
  actions.appendChild(button);
}

// Swap a row's actions for a yes/no question; `run` happens on yes.
function confirmIn(actions: HTMLElement, question: string, yes: string, run: () => Promise<void>): void {
  const before = [...actions.childNodes];
  actions.textContent = '';
  actions.appendChild(el('span', 'files-confirm', question));
  action(actions, yes, () => void guarded(run), true);
  action(actions, 'Cancel', () => actions.replaceChildren(...before));
  actions.querySelector('button')?.focus();
}

// A name field in place of a row; resolves to the name, or null when cancelled.
function askName(item: HTMLElement, initial: string, verb: string, taken: Set<string>): Promise<string | null> {
  return new Promise((resolve) => {
    const before = [...item.childNodes];
    const form = el('form', 'files-name-form');
    const input = el('input', 'text');
    input.value = initial;
    input.setAttribute('aria-label', 'Name');
    const submit = el('button', 'btn small', verb);
    submit.type = 'submit';
    const cancel = el('button', 'btn small ghost', 'Cancel');
    cancel.type = 'button';
    const done = (name: string | null) => {
      item.replaceChildren(...before);
      resolve(name);
    };
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const name = input.value.trim();
      if (!name || name.includes('/') || name === '.' || name === '..') {
        setStatus('A name cannot be empty or contain "/".', true);
        return;
      }
      if (name !== initial && taken.has(name)) {
        setStatus('"' + name + '" already exists here.', true);
        return;
      }
      done(name === initial ? null : name);
    });
    // Escape cancels the field instead of closing the dialog.
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        done(null);
      }
    });
    cancel.addEventListener('click', () => done(null));
    form.append(input, submit, cancel);
    item.replaceChildren(form);
    input.focus();
    input.select();
  });
}

async function guarded(run: () => Promise<void>): Promise<void> {
  try {
    await run();
  } catch (err) {
    setStatus(String((err as Error).message || err), true);
  }
  await render();
}

async function renderFiles(volume: StorageVolume, dir: string, mine: number): Promise<void> {
  const all = await call('storageScan', volume);
  if (mine !== generation) return;
  const entries = [...children(all, dir)].sort(([a, x], [b, y]) =>
    x.kind === y.kind ? a.localeCompare(b) : x.kind === 'dir' ? -1 : 1);
  list.textContent = '';
  $<HTMLButtonElement>('btn-files-export').disabled = !entries.length;
  if (!entries.length) {
    message('Empty. Drop files or a folder here, or upload them.');
    return;
  }
  const names = new Set(entries.map(([name]) => name));
  for (const [name, info] of entries) {
    const path = join(dir, name);
    const into = (d: string): Place => place.kind === 'save'
      ? { kind: 'save', id: place.id, dir: d }
      : { kind: 'sd', dir: d };
    const { item, actions } = row(
      info.kind === 'dir' ? Folder : File,
      info.kind === 'dir' ? name + '/' : name,
      info.kind === 'dir' ? 'folder' : fmtSize(info.size),
      info.kind === 'dir' ? () => go(into(path)) : () => void openFile(volume, path, name),
    );
    if (info.kind === 'file') {
      if (place.kind === 'sd' && name.toLowerCase().endsWith('.nro')) {
        action(actions, 'Launch ' + name, () => {
          dialog.close();
          void launchFromSd(path);
        }, false, Play);
      }
      action(actions, 'Open ' + name, () => void openFile(volume, path, name), false, FilePen);
      action(actions, 'Download ' + name, () => void guarded(async () => {
        const value = await call('storageRead', volume, path);
        saveBlob(name, new Blob([value?.data ?? new Uint8Array(0)]));
      }), false, Download);
    } else {
      action(actions, 'Export ' + name + ' as .zip', () => void guarded(() => exportZip(volume, path, name)), false, Archive);
    }
    action(actions, 'Rename ' + name, () => void (async () => {
      const next = await askName(item, name, 'Rename', names);
      if (next) {
        await guarded(async () => {
          await call('storageRename', volume, path, join(dir, next));
          setStatus('Renamed to ' + next + '.');
        });
      }
    })(), false, Pencil);
    action(actions, 'Delete ' + name, () => confirmIn(actions, 'Delete ' + name + '?', 'Delete', async () => {
      await call('storageRemove', volume, path);
      setStatus('Deleted ' + name + '.');
    }), true, Trash2);
  }
}

async function openFile(volume: StorageVolume, path: string, name: string): Promise<void> {
  const value = await call('storageRead', volume, path).catch((err: unknown) => {
    setStatus(String((err as Error).message || err), true);
    return undefined;
  });
  if (!value || value.kind !== 'file') return;
  const data = value.data ?? new Uint8Array(0);
  setStatus('');
  const dirty = showView(view, {
    name,
    data,
    save: async (next) => {
      await call('storageApply', volume, [[path, { kind: 'file', data: next }]]);
      setStatus('Saved ' + name + '.');
    },
    download: () => saveBlob(name, new Blob([data])),
    close: () => go(place),
  }, viewTools);
  viewing = { name, dirty };
  renderCrumbs();
  renderTools();
}

async function renderSaves(mine: number): Promise<void> {
  const { ids, names: nicknames } = await call('savesList');
  if (mine !== generation) return;
  const names = new Map(nicknames);
  list.textContent = '';
  if (!ids.length) {
    message('No save data yet. A title creates its save the first time it writes one.');
    return;
  }
  for (const id of ids) {
    const [save, uid] = id.split('@');
    const owner = uid ? names.get(uid) ?? 'profile ' + uid.slice(0, 8) : 'shared';
    const label = titleLabel(save);
    const { actions } = row(Save, label, (label === save ? '' : save + ' · ') + owner,
      () => go({ kind: 'save', id, dir: '/' }));
    action(actions, 'Export ' + label + ' as .zip', () => void guarded(() => exportZip(saveVolume(id), '/', id)), false, Archive);
    action(actions, 'Delete ' + label, () => confirmIn(actions, 'Delete this save?', 'Delete', async () => {
      await call('storageRemove', saveVolume(id), '/');
      setStatus('Deleted the save ' + id + '.');
    }), true, Trash2);
  }
}

async function renderNand(mine: number): Promise<void> {
  const titles = await call('nandList');
  if (mine !== generation) return;
  list.textContent = '';
  if (!titles.length) {
    message('Nothing installed. Install firmware to add the system applets and data archives.');
    return;
  }
  titles.sort((a, b) => a.kind - b.kind || titleLabel(a.id).localeCompare(titleLabel(b.id)));
  for (const title of titles) {
    const kind = title.kind === 0 ? 'program' : 'data archive';
    const { actions } = row(title.kind === 0 ? Package : Database, titleLabel(title.id),
      title.id + ' · ' + kind + ' · ' + fmtSize(title.size), null);
    action(actions, 'Download ' + title.name, () => void guarded(async () => {
      const content = await call('nandContent', title.name);
      if (!content) throw new Error(title.name + ' is indexed but its content is missing.');
      saveBlob(title.name, content);
    }), false, Download);
    action(actions, 'Remove ' + titleLabel(title.id), () => confirmIn(actions, 'Remove ' + titleLabel(title.id) + '?', 'Remove', async () => {
      await removeNandTitle(title.id);
      setStatus('Removed ' + titleLabel(title.id) + '.');
    }), true, Trash2);
  }
}

async function render(): Promise<void> {
  const mine = ++generation;
  viewing = null;
  view.textContent = '';
  viewTools.textContent = '';
  $<HTMLButtonElement>('btn-files-export').disabled = true;
  renderCrumbs();
  renderTools();
  message('Reading…');
  try {
    const volume = volumeOf(place);
    if (volume) await renderFiles(volume, dirOf(place), mine);
    else if (place.kind === 'saves') await renderSaves(mine);
    else await renderNand(mine);
  } catch (err) {
    if (mine !== generation) return;
    message('Could not read this storage.');
    setStatus(String((err as Error).message || err), true);
  }
}

export function openFiles(): void {
  setStatus('');
  if (!dialog.open) dialog.showModal();
  void render();
}

$('btn-files').addEventListener('click', openFiles);
$('btn-files-close').addEventListener('click', () => leaveView(() => dialog.close()));
dialog.addEventListener('cancel', (e) => {
  if (!viewing?.dirty()) return;
  e.preventDefault();
  leaveView(() => dialog.close());
});

const tabIcons: Record<string, IconNode> = { sd: MemoryStick, saves: Save, nand: Package };
const toolIcons: [string, IconNode][] = [
  ['btn-files-mkdir', FolderPlus], ['files-upload-label', FileUp], ['files-upload-dir-label', FolderUp],
  ['btn-files-export', FileArchive], ['files-install-label', PackagePlus],
];
for (const [id, glyph] of toolIcons) $(id).prepend(icon(glyph));

document.querySelectorAll<HTMLElement>('.files-tab').forEach((tab) => {
  tab.prepend(icon(tabIcons[tab.dataset.volume ?? ''] ?? File));
  tab.addEventListener('click', () => {
    setStatus('');
    const volume = tab.dataset.volume;
    go(volume === 'saves' ? { kind: 'saves' } : volume === 'nand' ? { kind: 'nand' } : { kind: 'sd', dir: '/' });
  });
});

$('btn-files-mkdir').addEventListener('click', () => {
  const volume = volumeOf(place);
  if (!volume) return;
  const dir = dirOf(place);
  const item = el('li', 'files-row');
  if (list.querySelector('.files-empty')) list.textContent = '';
  list.prepend(item);
  const taken = new Set([...list.querySelectorAll('.files-name')].map((n) => (n.textContent ?? '').replace(/\/$/, '')));
  void askName(item, '', 'Create', taken).then((name) => {
    item.remove();
    if (name) {
      void guarded(async () => {
        await call('storageApply', volume, [[join(dir, name), { kind: 'dir' }]]);
        setStatus('Created ' + name + '.');
      });
    }
  });
});

$('btn-files-export').addEventListener('click', () => {
  const volume = volumeOf(place);
  if (!volume) return;
  const dir = dirOf(place);
  const name = dir === '/' ? (place.kind === 'save' ? place.id : 'sdmc') : basename(dir);
  void guarded(() => exportZip(volume, dir, name));
});

function uploadFrom(input: HTMLInputElement, relative: boolean): void {
  const volume = volumeOf(place);
  const picked = [...(input.files ?? [])];
  input.value = '';
  if (!volume || !picked.length) return;
  const files: [string, File][] = picked.map((f) => [relative && f.webkitRelativePath ? f.webkitRelativePath : f.name, f]);
  void guarded(() => upload(volume, dirOf(place), files));
}
uploadInput.addEventListener('change', () => uploadFrom(uploadInput, false));
uploadDirInput.addEventListener('change', () => uploadFrom(uploadDirInput, true));

list.addEventListener('dragover', (e) => {
  if (!volumeOf(place)) return;
  e.preventDefault();
  list.classList.add('is-drop');
});
list.addEventListener('dragleave', () => list.classList.remove('is-drop'));
list.addEventListener('drop', (e) => {
  list.classList.remove('is-drop');
  const volume = volumeOf(place);
  if (!volume || !e.dataTransfer) return;
  e.preventDefault();
  const transfer = e.dataTransfer;
  void guarded(async () => upload(volume, dirOf(place), await droppedFiles(transfer)));
});

document.addEventListener('nand-changed', () => {
  if (dialog.open && place.kind === 'nand') void render();
});
