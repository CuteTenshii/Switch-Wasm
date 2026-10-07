// The page's IndexedDB databases and shared transaction helpers.

import type { Bytes } from '../shared/protocol';

export const SD_DB_NAME = 'switch-wasm-sd';
export const SD_STORE = 'entries';

// The page's own log, so it survives the tab being killed.
export const LOG_DB_NAME = 'switch-wasm-log';
export const LOG_STORE = 'log';
export const LOG_KEY = 'previous';

export const NAND_DB_NAME = 'switch-wasm-nand';
export const NAND_CONTENT = 'content';
export const NAND_TITLES = 'titles';
export const NAND_SAVES = 'saves';
export const NAND_USERS = 'users';

export interface StoredEntry {
  kind: 'dir' | 'file';
  data?: Bytes;
}

// `kind` is 0 for a program and 1 for a data archive.
export interface NandEntry {
  name: string;
  kind: number;
}

let sdDb: IDBDatabase | null = null;

export function sdIdb(): Promise<IDBDatabase> {
  if (sdDb) return Promise.resolve(sdDb);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(SD_DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(SD_STORE)) req.result.createObjectStore(SD_STORE);
    };
    req.onsuccess = () => {
      sdDb = req.result;
      resolve(sdDb);
    };
    req.onerror = () => reject(req.error ?? new Error('IndexedDB request failed'));
  });
}

let logDb: IDBDatabase | null = null;

export function logIdb(): Promise<IDBDatabase> {
  if (logDb) return Promise.resolve(logDb);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(LOG_DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(LOG_STORE)) req.result.createObjectStore(LOG_STORE);
    };
    req.onsuccess = () => {
      logDb = req.result;
      resolve(logDb);
    };
    req.onerror = () => reject(req.error ?? new Error('IndexedDB request failed'));
  });
}

let nandDb: IDBDatabase | null = null;

export function nandIdb(): Promise<IDBDatabase> {
  if (nandDb) return Promise.resolve(nandDb);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(NAND_DB_NAME, 3);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(NAND_CONTENT)) db.createObjectStore(NAND_CONTENT);
      if (!db.objectStoreNames.contains(NAND_TITLES)) db.createObjectStore(NAND_TITLES);
      if (!db.objectStoreNames.contains(NAND_SAVES)) db.createObjectStore(NAND_SAVES);
      if (!db.objectStoreNames.contains(NAND_USERS)) db.createObjectStore(NAND_USERS);
    };
    req.onsuccess = () => {
      nandDb = req.result;
      resolve(nandDb);
    };
    req.onerror = () => reject(req.error ?? new Error('IndexedDB request failed'));
  });
}

// Keys and values are read in one transaction so they stay in step.
export function idbGetAll<T>(db: IDBDatabase, store: string): Promise<[string, T][]> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readonly');
    const s = tx.objectStore(store);
    const keys = s.getAllKeys();
    const values = s.getAll();
    tx.oncomplete = () => resolve(
      (keys.result as string[]).map((k, i) => [k, values.result[i] as T]));
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB request failed'));
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB request failed'));
  });
}

export function idbGet<T>(db: IDBDatabase, store: string, key: string): Promise<T | undefined> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readonly');
    const req = tx.objectStore(store).get(key);
    tx.oncomplete = () => resolve(req.result as T | undefined);
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB request failed'));
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB request failed'));
  });
}

// A null value deletes the key.
export function idbApply(
  db: IDBDatabase,
  store: string,
  entries: [string, unknown][],
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    const s = tx.objectStore(store);
    for (const [key, value] of entries) {
      if (value === null) s.delete(key);
      else s.put(value, key);
    }
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB request failed'));
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB request failed'));
  });
}
