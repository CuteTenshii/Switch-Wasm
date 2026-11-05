// The SD card, saves, profiles and NAND as stored in IndexedDB, read and written here so the page stays free.

import type { Bytes, CommandHandlers, NandTitle, StorageInfo, StorageVolume } from '../shared/protocol';
import {
  idbApply,
  idbGet,
  idbGetAll,
  NAND_CONTENT,
  NAND_SAVES,
  NAND_TITLES,
  NAND_USERS,
  nandIdb,
  SD_STORE,
  sdIdb,
  type NandEntry,
  type StoredEntry,
} from '../shared/db';
import { switch_flush_storage } from '@core/switch_wasm.js';
import { core, coreLoaded, withBytes, withPath } from './wasm';
import { zip, type ZipEntry } from './zip';

// A uid (32 hex digits in memory order) as two little-endian halves.
export function uidHalves(hex: string): [bigint, bigint] {
  const half = (from: number) => {
    let value = 0n;
    for (let i = 7; i >= 0; i--) {
      value = (value << 8n) | BigInt(parseInt(hex.slice(from + i * 2, from + i * 2 + 2), 16));
    }
    return value;
  };
  return [half(0), half(16)];
}

// `<hex save id>[@<user uid>]`; no uid means no owning user.
export function saveKey(id: string): [bigint, bigint, bigint] {
  const [save, user] = id.split('@');
  const [lo, hi] = user ? uidHalves(user) : [0n, 0n];
  return [BigInt('0x' + save), lo, hi];
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
  if (volume.kind === 'sd') {
    return {
      db: sdIdb,
      store: SD_STORE,
      prefix: '',
      write: (path, data) => void withPath(path, (pptr, plen) =>
        withBytes(data, (dptr, dlen) => core.switch_sd_write_file(pptr, plen, dptr, dlen))),
      mkdir: (path) => void withPath(path, (ptr, len) => core.switch_sd_create_dir(ptr, len)),
      remove: (path) => void withPath(path, (ptr, len) => core.switch_sd_remove(ptr, len)),
    };
  }
  const save = saveKey(volume.id);
  return {
    db: nandIdb,
    store: NAND_SAVES,
    prefix: volume.id,
    write: (path, data) => void withPath(path, (pptr, plen) =>
      withBytes(data, (dptr, dlen) => core.switch_save_write_file(...save, pptr, plen, dptr, dlen))),
    mkdir: (path) => void withPath(path, (ptr, len) => core.switch_save_create_dir(...save, ptr, len)),
    remove: (path) => void withPath(path, (ptr, len) => core.switch_save_remove(...save, ptr, len)),
  };
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

// Content stored by older builds as an ArrayBuffer is rewritten as a Blob on read.
export async function nandContent(name: string): Promise<Blob | undefined> {
  const stored = await idbGet<Blob | ArrayBuffer>(await nandIdb(), NAND_CONTENT, name);
  if (!stored) return undefined;
  if (stored instanceof Blob) return stored;
  const blob = new Blob([stored]);
  try {
    await idbApply(await nandIdb(), NAND_CONTENT, [[name, blob]]);
  } catch { /* it is readable either way; the next session pays the same again */ }
  return blob;
}

async function nandTitles(): Promise<[string, NandEntry][]> {
  return idbGetAll<NandEntry>(await nandIdb(), NAND_TITLES);
}

export const STORAGE = {
  storage_scan: scan,
  async storage_read(volume, path) {
    return readEntry(placeOf(volume), path);
  },
  async storage_apply(volume, changes) {
    await apply(placeOf(volume), changes);
  },
  async storage_upload(volume, path, file) {
    const data = new Uint8Array(await file.arrayBuffer());
    await apply(placeOf(volume), [[path, { kind: 'file', data }]]);
    return data.length;
  },
  async storage_remove(volume, path) {
    await apply(placeOf(volume), under(await scan(volume), path).map((p) => [p, null]));
  },
  async storage_rename(volume, from, to) {
    const place = placeOf(volume);
    const changes: [string, StoredEntry | null][] = [];
    for (const path of under(await scan(volume), from)) {
      const value = await readEntry(place, path);
      if (value) changes.push([to + path.slice(from.length), value]);
      changes.push([path, null]);
    }
    await apply(place, changes);
  },
  async storage_export(volume, root) {
    const place = placeOf(volume);
    const entries: ZipEntry[] = [];
    for (const path of under(await scan(volume), root).sort()) {
      const rel = root === '/' ? path.slice(1) : path.slice(root.length + 1);
      if (!rel) continue;
      const value = await readEntry(place, path);
      if (value?.kind === 'dir') entries.push({ name: rel + '/', data: new Uint8Array(0) });
      else if (value) entries.push({ name: rel, data: value.data ?? new Uint8Array(0) });
    }
    return { blob: zip(entries), entries: entries.length };
  },

  async saves_list() {
    await flush();
    const db = await nandIdb();
    const ids = await new Promise<string[]>((resolve, reject) => {
      const tx = db.transaction(NAND_SAVES, 'readonly');
      const req = tx.objectStore(NAND_SAVES).getAllKeys();
      tx.oncomplete = () => resolve(
        [...new Set((req.result as string[]).map((k) => k.slice(0, k.indexOf('/'))))].sort());
      tx.onerror = () => reject(tx.error ?? new Error('IndexedDB request failed'));
    });
    const names = new Map<string, string>();
    for (const [, value] of await idbGetAll<{ uid?: string; nickname?: string }>(db, NAND_USERS)) {
      if (value.uid && value.nickname) names.set(value.uid, value.nickname);
    }
    return { ids, names: [...names] };
  },
  // Deletes keys of the form "<save id>@<uid>/<path>".
  async saves_delete_user(uid) {
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

  async users_read() {
    return idbGetAll<unknown>(await nandIdb(), NAND_USERS);
  },
  async users_store(entries) {
    await idbApply(await nandIdb(), NAND_USERS, entries);
  },

  nand_titles: nandTitles,
  async nand_list() {
    const out: NandTitle[] = [];
    for (const [id, entry] of await nandTitles()) {
      const content = await nandContent(entry.name).catch(() => undefined);
      out.push({ id, name: entry.name, kind: entry.kind, size: content?.size ?? 0 });
    }
    return out;
  },
  nand_content: nandContent,
  // Content and index entry go in one transaction so the index never names missing content.
  async nand_install(file, id, kind) {
    const tx = (await nandIdb()).transaction([NAND_CONTENT, NAND_TITLES], 'readwrite');
    tx.objectStore(NAND_CONTENT).put(file, file.name);
    tx.objectStore(NAND_TITLES).put({ name: file.name, kind }, id);
    await committed(tx);
  },
  // Content and index entry go together; a registered archive stays until the session is replaced.
  async nand_remove(id) {
    const db = await nandIdb();
    const entry = await idbGet<NandEntry>(db, NAND_TITLES, id);
    if (!entry) return;
    const tx = db.transaction([NAND_CONTENT, NAND_TITLES], 'readwrite');
    tx.objectStore(NAND_CONTENT).delete(entry.name);
    tx.objectStore(NAND_TITLES).delete(id);
    await committed(tx);
  },
  async nand_erase() {
    const tx = (await nandIdb()).transaction([NAND_CONTENT, NAND_TITLES], 'readwrite');
    tx.objectStore(NAND_CONTENT).clear();
    tx.objectStore(NAND_TITLES).clear();
    await committed(tx);
  },
} satisfies Partial<CommandHandlers>;
