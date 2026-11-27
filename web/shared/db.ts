// The IndexedDB databases and transaction helpers, shared by the page and the worker.

// The page's own log, so it survives the tab being killed.
export const LOG_DB_NAME = 'switch-wasm-log';
export const LOG_STORE = 'log';
export const LOG_KEY = 'previous';

export const NAND_DB_NAME = 'switch-wasm-nand';
export const NAND_CONTENT = 'content';
export const NAND_TITLES = 'titles';
export const NAND_SAVES = 'saves';
export const NAND_USERS = 'users';
export const NAND_CONTROLS = 'controls';
// Must match the version and stores `crates/wasm/src/idb/mod.rs` opens the NAND with.
const NAND_VERSION = 4;
const NAND_STORES = [NAND_CONTENT, NAND_TITLES, NAND_SAVES, NAND_USERS, NAND_CONTROLS];

// The start screen's game folder, a `FileSystemDirectoryHandle` where the browser keeps one.
export const LIBRARY_DB_NAME = 'switch-wasm-library';
export const LIBRARY_STORE = 'folder';
export const LIBRARY_KEY = 'handle';

// Opens `name` once and hands out that connection, creating any of `stores` it lacks.
function opener(name: string, version: number, stores: string[]): () => Promise<IDBDatabase> {
  let open: Promise<IDBDatabase> | null = null;
  return () => {
    open ??= new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open(name, version);
      req.onupgradeneeded = () => {
        for (const store of stores) {
          if (!req.result.objectStoreNames.contains(store)) req.result.createObjectStore(store);
        }
      };
      req.onsuccess = () => {
        const db = req.result;
        // A newer schema, from another tab or the core, waits until every older connection closes.
        db.onversionchange = () => {
          db.close();
          open = null;
        };
        resolve(db);
      };
      req.onerror = () => {
        open = null;
        reject(req.error ?? new Error('IndexedDB request failed'));
      };
    });
    return open;
  };
}

export const logIdb = opener(LOG_DB_NAME, 1, [LOG_STORE]);
export const nandIdb = opener(NAND_DB_NAME, NAND_VERSION, NAND_STORES);
export const libraryIdb = opener(LIBRARY_DB_NAME, 1, [LIBRARY_STORE]);

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
