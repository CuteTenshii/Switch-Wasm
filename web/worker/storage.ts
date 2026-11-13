// The SD card, saves, profiles and NAND as stored, read and written here so the page stays free.

import type { Bytes, CommandHandlers, NandTitle, StorageInfo, StorageVolume } from '../shared/protocol';
import {
  idbApply,
  idbGet,
  idbGetAll,
  NAND_SAVES,
  NAND_TITLES,
  NAND_USERS,
  nandIdb,
  SD_STORE,
  sdIdb,
  type NandEntry,
  type StoredEntry,
} from '../shared/db';
import {
  switch_flush_storage, switch_fs_create_dir, switch_fs_remove, switch_fs_write, switch_nand_content,
  switch_nand_content_size, switch_nand_erase, switch_nand_install, switch_nand_remove,
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

// A volume's store, and how a change reaches the running session.
interface Place {
  db(): Promise<IDBDatabase>;
  store: string;
  // Key prefix in front of the path: a save's id, nothing for the SD card.
  prefix: string;
  write(path: string, data: Bytes): void;
  mkdir(path: string): void;
  remove(path: string): void;
}

function placeOf(volume: StorageVolume): Place {
  const name = volume.kind === 'sd' ? 'sd' : volume.id;
  const session = {
    write: (path: string, data: Bytes) => switch_fs_write(name, path, data),
    mkdir: (path: string) => switch_fs_create_dir(name, path),
    remove: (path: string) => switch_fs_remove(name, path),
  };
  return volume.kind === 'sd'
    ? { db: sdIdb, store: SD_STORE, prefix: '', ...session }
    : { db: nandIdb, store: NAND_SAVES, prefix: volume.id, ...session };
}

function committed(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = tx.onabort = () => reject(tx.error ?? new Error('IndexedDB request failed'));
  });
}

// The core holds changes the guest made that IndexedDB has not seen yet.
async function flush(): Promise<void> {
  if (coreLoaded()) await switch_flush_storage();
}

async function scan(volume: StorageVolume): Promise<Map<string, StorageInfo>> {
  await flush();
  const place = placeOf(volume);
  const db = await place.db();
  const range = place.prefix
    ? IDBKeyRange.bound(place.prefix + '/', place.prefix + '/\uffff')
    : undefined;
  return new Promise((resolve, reject) => {
    const out = new Map<string, StorageInfo>();
    const tx = db.transaction(place.store, 'readonly');
    const req = tx.objectStore(place.store).openCursor(range);
    req.onsuccess = () => {
      const cursor = req.result;
      if (!cursor) return;
      const value = cursor.value as StoredEntry;
      const path = (cursor.key as string).slice(place.prefix.length);
      out.set(path, { kind: value.kind, size: value.data?.length ?? 0 });
      cursor.continue();
    };
    tx.oncomplete = () => resolve(out);
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB request failed'));
  });
}

async function saveKeys(db: IDBDatabase): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(NAND_SAVES, 'readonly');
    const req = tx.objectStore(NAND_SAVES).getAllKeys();
    tx.oncomplete = () => resolve(req.result as string[]);
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB request failed'));
  });
}

function under(all: Map<string, StorageInfo>, root: string): string[] {
  return [...all.keys()].filter((p) => root === '/' || p === root || p.startsWith(root + '/'));
}

async function readEntry(place: Place, path: string): Promise<StoredEntry | undefined> {
  return idbGet<StoredEntry>(await place.db(), place.store, place.prefix + path);
}

// Store first, then the running session; deletions go deepest first.
async function apply(place: Place, changes: [string, StoredEntry | null][]): Promise<void> {
  await idbApply(await place.db(), place.store,
    changes.map(([path, value]) => [place.prefix + path, value]));
  if (!coreLoaded()) return;
  const removals = changes.filter(([, v]) => v === null).map(([p]) => p)
    .sort((a, b) => b.length - a.length);
  for (const path of removals) place.remove(path);
  for (const [path, value] of changes) {
    if (value?.kind === 'dir') place.mkdir(path);
    else if (value) place.write(path, value.data ?? new Uint8Array(0));
  }
}

async function nandTitles(): Promise<[string, NandEntry][]> {
  return idbGetAll<NandEntry>(await nandIdb(), NAND_TITLES);
}

export const STORAGE = {
  storageScan: scan,
  async storageRead(volume, path) {
    return readEntry(placeOf(volume), path);
  },
  async storageApply(volume, changes) {
    await apply(placeOf(volume), changes);
  },
  async storageUpload(volume, path, file) {
    const data = new Uint8Array(await file.arrayBuffer());
    await apply(placeOf(volume), [[path, { kind: 'file', data }]]);
    return data.length;
  },
  async storageRemove(volume, path) {
    await apply(placeOf(volume), under(await scan(volume), path).map((p) => [p, null]));
  },
  async storageRename(volume, from, to) {
    const place = placeOf(volume);
    const changes: [string, StoredEntry | null][] = [];
    for (const path of under(await scan(volume), from)) {
      const value = await readEntry(place, path);
      if (value) changes.push([to + path.slice(from.length), value]);
      changes.push([path, null]);
    }
    await apply(place, changes);
  },
  async storageExport(volume, root) {
    const place = placeOf(volume);
    const entries: ZipEntry[] = [];
    for (const path of under(await scan(volume), root).sort()) {
      const rel = root === '/' ? path.slice(1) : path.slice(root.length + 1);
      if (!rel) continue;
      const value = await readEntry(place, path);
      if (value?.kind === 'dir') entries.push({ name: rel + '/', data: new Uint8Array(0) });
      else if (value) entries.push({ name: rel, data: value.data ?? new Uint8Array(0) });
    }
    return { blob: await zipOffThread(entries), entries: entries.length };
  },

  async savesList() {
    await flush();
    const db = await nandIdb();
    const ids = [...new Set((await saveKeys(db)).map((k) => k.slice(0, k.indexOf('/'))))].sort();
    const names = new Map<string, string>();
    for (const [, value] of await idbGetAll<{ uid?: string; nickname?: string }>(db, NAND_USERS)) {
      if (value.uid && value.nickname) names.set(value.uid, value.nickname);
    }
    return { ids, names: [...names] };
  },
  // Deletes keys of the form "<save id>@<uid>/<path>".
  async savesDeleteUser(uid) {
    const db = await nandIdb();
    const tx = db.transaction(NAND_SAVES, 'readwrite');
    const store = tx.objectStore(NAND_SAVES);
    const cursor = store.openKeyCursor();
    let removed = 0;
    cursor.onsuccess = () => {
      const at = cursor.result;
      if (!at) return;
      const key = at.key as string;
      const cut = key.indexOf('/');
      const id = cut < 0 ? key : key.slice(0, cut);
      if (id.endsWith('@' + uid)) {
        store.delete(at.key);
        removed++;
      }
      at.continue();
    };
    await committed(tx);
    return removed;
  },

  // New copies are written before the old ones go, so an interrupted move loses nothing.
  async savesMoveUser(from, to) {
    await flush();
    // Keys are "<save id>[@<uid>]/<path>"; a save both UIDs own would merge two titles' saves.
    const moves = new Map<string, string[]>();
    const taken = new Set<string>();
    for (const key of await saveKeys(await nandIdb())) {
      const cut = key.indexOf('/');
      if (cut < 0) continue;
      const [save, owner] = key.slice(0, cut).split('@');
      if (owner === to) taken.add(save);
      if (owner !== from) continue;
      const paths = moves.get(save) ?? [];
      paths.push(key.slice(cut));
      moves.set(save, paths);
    }
    const clashes = [...moves.keys()].filter((save) => taken.has(save));
    if (clashes.length) {
      throw new Error(`${to} already has saves for ${clashes.length === 1 ? 'title' : 'titles'} ${clashes.join(', ')}`);
    }
    let moved = 0;
    for (const [save, paths] of moves) {
      const source = placeOf({ kind: 'save', id: `${save}@${from}` });
      const changes: [string, StoredEntry][] = [];
      for (const path of paths) {
        const value = await readEntry(source, path);
        if (value) changes.push([path, value]);
      }
      await apply(placeOf({ kind: 'save', id: `${save}@${to}` }), changes);
      await apply(source, paths.map((path) => [path, null]));
      moved += changes.length;
    }
    return moved;
  },

  async usersRead() {
    return idbGetAll<unknown>(await nandIdb(), NAND_USERS);
  },
  async usersStore(entries) {
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
