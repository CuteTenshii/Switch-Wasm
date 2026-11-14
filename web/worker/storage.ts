// The SD card, saves, profiles and NAND as stored, read and written here so the page stays free.

import type { CommandHandlers, NandTitle, StorageInfo, StorageVolume } from '../shared/protocol';
import {
  idbApply,
  idbGetAll,
  NAND_TITLES,
  NAND_USERS,
  nandIdb,
  type NandEntry,
  type StoredEntry,
} from '../shared/db';
import {
  switch_flush_storage, switch_fs_create_dir, switch_fs_entries, switch_fs_read, switch_fs_remove,
  switch_fs_upload, switch_fs_write, switch_nand_content, switch_nand_content_size, switch_nand_erase,
  switch_nand_install, switch_nand_remove, switch_saves, switch_storage_writable,
} from '@core/switch_wasm.js';
import { workerErrorText } from '../shared/workererror';
import { coreLoaded } from './wasm';
import type { ZipEntry } from './zip';
import type { ZipReply } from './zipper';

// The zip of `entries`, built in a worker of its own that takes their bytes.
function zipOffThread(entries: ZipEntry[]): Promise<Blob> {
  const worker = new Worker(new URL('./zipper.ts', import.meta.url), { type: 'module', name: 'Zip export' });
  return new Promise((resolve, reject) => {
    worker.onmessage = (e: MessageEvent<ZipReply>) => {
      worker.terminate();
      if ('error' in e.data) reject(new Error(e.data.error));
      else resolve(e.data.blob);
    };
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error('the zip worker ' + workerErrorText(e)));
    };
    worker.postMessage(entries, [...new Set(entries.map((entry) => entry.data.buffer))]);
  });
}

// A volume as the core names it: `sd`, or a save's id.
function nameOf(volume: StorageVolume): string {
  return volume.kind === 'sd' ? 'sd' : volume.id;
}

// The session holds the card and every save; the core stores what changes.
function needSession(): void {
  if (!coreLoaded()) throw new Error('the emulator is still loading');
}

function scan(volume: StorageVolume): Map<string, StorageInfo> {
  needSession();
  const out = new Map<string, StorageInfo>();
  for (const [path, kind, size] of switch_fs_entries(nameOf(volume))) {
    out.set(path, { kind: kind === 0 ? 'dir' : 'file', size });
  }
  return out;
}

function under(all: Map<string, StorageInfo>, root: string): string[] {
  return [...all.keys()].filter((p) => root === '/' || p === root || p.startsWith(root + '/'));
}

function readEntry(volume: StorageVolume, path: string, info?: StorageInfo): StoredEntry | undefined {
  if (info?.kind === 'dir') return { kind: 'dir' };
  const data = switch_fs_read(nameOf(volume), path);
  return data ? { kind: 'file', data } : undefined;
}

// Deletions go first, deepest first, so a name being replaced is free when it is written.
async function apply(volume: StorageVolume, changes: [string, StoredEntry | null][]): Promise<void> {
  needSession();
  const name = nameOf(volume);
  const removals = changes.filter(([, v]) => v === null).map(([p]) => p)
    .sort((a, b) => b.length - a.length);
  for (const path of removals) switch_fs_remove(name, path);
  for (const [path, value] of changes) {
    if (value?.kind === 'dir') switch_fs_create_dir(name, path);
    else if (value) switch_fs_write(name, path, value.data ?? new Uint8Array(0));
  }
  await switch_flush_storage();
}

async function nandTitles(): Promise<[string, NandEntry][]> {
  return idbGetAll<NandEntry>(await nandIdb(), NAND_TITLES);
}

export const STORAGE = {
  storageScan: scan,
  storageRead(volume, path) {
    needSession();
    return readEntry(volume, path, scan(volume).get(path));
  },
  async storageApply(volume, changes) {
    await apply(volume, changes);
  },
  // Copied into storage by range, never staged whole in memory where OPFS holds the card.
  async storageUpload(volume, path, file) {
    needSession();
    await switch_fs_upload(nameOf(volume), path, file);
    await switch_flush_storage();
    return file.size;
  },
  async storageRemove(volume, path) {
    await apply(volume, under(scan(volume), path).map((p) => [p, null]));
  },
  async storageRename(volume, from, to) {
    const all = scan(volume);
    const changes: [string, StoredEntry | null][] = [];
    for (const path of under(all, from)) {
      const value = readEntry(volume, path, all.get(path));
      if (value) changes.push([to + path.slice(from.length), value]);
      changes.push([path, null]);
    }
    await apply(volume, changes);
  },
  async storageExport(volume, root) {
    const all = scan(volume);
    const entries: ZipEntry[] = [];
    for (const path of under(all, root).sort()) {
      const rel = root === '/' ? path.slice(1) : path.slice(root.length + 1);
      if (!rel) continue;
      const value = readEntry(volume, path, all.get(path));
      if (value?.kind === 'dir') entries.push({ name: rel + '/', data: new Uint8Array(0) });
      else if (value) entries.push({ name: rel, data: value.data ?? new Uint8Array(0) });
    }
    return { blob: await zipOffThread(entries), entries: entries.length };
  },

  async savesList() {
    needSession();
    const names = new Map<string, string>();
    for (const [, value] of await idbGetAll<{ uid?: string; nickname?: string }>(await nandIdb(), NAND_USERS)) {
      if (value.uid && value.nickname) names.set(value.uid, value.nickname);
    }
    return { ids: switch_saves(), names: [...names] };
  },
  // Saves are named "<save id>@<uid>"; returns how many entries went.
  async savesDeleteUser(uid) {
    let removed = 0;
    for (const id of switch_saves().filter((save) => save.endsWith('@' + uid))) {
      const volume: StorageVolume = { kind: 'save', id };
      const paths = [...scan(volume).keys()];
      await apply(volume, paths.map((path) => [path, null]));
      removed += paths.length;
    }
    return removed;
  },

  // New copies are written before the old ones go, so an interrupted move loses nothing.
  async savesMoveUser(from, to) {
    // A save both UIDs own would merge two titles' saves.
    const ids = switch_saves();
    const moving = ids.filter((id) => id.endsWith('@' + from)).map((id) => id.slice(0, id.indexOf('@')));
    const taken = new Set(ids.filter((id) => id.endsWith('@' + to)).map((id) => id.slice(0, id.indexOf('@'))));
    const clashes = moving.filter((save) => taken.has(save));
    if (clashes.length) {
      throw new Error(`${to} already has saves for ${clashes.length === 1 ? 'title' : 'titles'} ${clashes.join(', ')}`);
    }
    let moved = 0;
    for (const save of moving) {
      const source: StorageVolume = { kind: 'save', id: `${save}@${from}` };
      const all = scan(source);
      const changes: [string, StoredEntry][] = [];
      for (const [path, info] of all) {
        const value = readEntry(source, path, info);
        if (value) changes.push([path, value]);
      }
      await apply({ kind: 'save', id: `${save}@${to}` }, changes);
      await apply(source, [...all.keys()].map((path) => [path, null]));
      moved += changes.length;
    }
    return moved;
  },

  async usersRead() {
    return idbGetAll<unknown>(await nandIdb(), NAND_USERS);
  },
  async usersStore(entries) {
    if (!await switch_storage_writable()) {
      throw new Error('This tab can\'t change anything stored: the emulator is open in another tab.');
    }
    await idbApply(await nandIdb(), NAND_USERS, entries);
  },

  nandTitles,
  async nandList() {
    const out: NandTitle[] = [];
    for (const [id, entry] of await nandTitles()) {
      const size = await switch_nand_content_size(entry.name).catch(() => undefined);
      out.push({ id, name: entry.name, kind: entry.kind, size: size ?? 0 });
    }
    return out;
  },
  // Content lives in OPFS, or IndexedDB without it; `crates/switch-wasm/src/idb/content.rs` owns both.
  nandContent: switch_nand_content,
  nandInstall: switch_nand_install,
  nandRemove: switch_nand_remove,
  nandErase: switch_nand_erase,
} satisfies Partial<CommandHandlers>;
