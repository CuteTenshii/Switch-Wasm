// The SD card and saves in IndexedDB: restored into the core when a session starts,
// and written as the guest changes them, without a trip through the page.

import {
  idbApply, idbGetAll, NAND_SAVES, nandIdb, SD_STORE, sdIdb, type StoredEntry,
} from '../shared/db';
import { fmtSize } from '../shared/format';
import type { Bytes } from '../shared/protocol';
import { CMD } from './commands';
import { workerLog } from './log';

// Taken from the core but not yet stored, by key; IndexedDB refusals wait for the next flush.
const sdBacklog = new Map<string, StoredEntry | null>();
const saveBacklog = new Map<string, StoredEntry | null>();
// Flushes run one after another, so each one's changes are stored when it resolves.
let flushing: Promise<void> = Promise.resolve();

// Directories first, so empty ones survive.
function dirsFirst(entries: [string, StoredEntry][]): [string, StoredEntry][] {
  return entries.sort((a, b) => (a[1].kind === b[1].kind ? 0 : a[1].kind === 'dir' ? -1 : 1));
}

// Load the stored card and saves through host entry points, which are not guest changes.
export async function restoreStorage(): Promise<number> {
  try {
    const sd = dirsFirst(await idbGetAll<StoredEntry>(await sdIdb(), SD_STORE));
    for (const [path, value] of sd) {
      if (value.kind === 'dir') void CMD.sd_create_dir(path);
      else void CMD.sd_write_file(path, value.data ?? new Uint8Array(0));
    }
    if (sd.length) workerLog('SD card: restored ' + sd.length + ' entries', 'debug');
  } catch (err) {
    workerLog('SD card: could not be read (' + String(err) + ')', 'error');
  }
  try {
    const saves = dirsFirst(await idbGetAll<StoredEntry>(await nandIdb(), NAND_SAVES));
    for (const [key, value] of saves) {
      const cut = key.indexOf('/');
      if (cut < 0) continue;
      const [id, path] = [key.slice(0, cut), key.slice(cut)];
      if (value.kind === 'dir') void CMD.save_create_dir(id, path);
      else void CMD.save_write_file(id, path, value.data ?? new Uint8Array(0));
    }
    if (saves.length) workerLog('Saves: restored ' + saves.length + ' entries', 'debug');
  } catch (err) {
    workerLog('Saves: could not be read (' + String(err) + ')', 'error');
  }
  return 0;
}

// Take what the guest changed and store it.
export function flushStorage(): Promise<void> {
  takeChanges();
  flushing = flushing.then(store);
  return flushing;
}

function takeChanges(): void {
  for (const change of CMD.sd_take_changes() as { kind: string; path: string }[]) {
    sdBacklog.set(change.path, entryFor(change.kind, () => CMD.sd_read_file(change.path)));
  }
  if (!CMD.saves_pending()) return;
  for (const id of CMD.save_ids() as string[]) {
    for (const change of CMD.save_take_changes(id) as { kind: string; path: string }[]) {
      saveBacklog.set(id + change.path,
        entryFor(change.kind, () => CMD.save_read_file(id, change.path)));
    }
  }
}

function entryFor(kind: string, read: () => unknown): StoredEntry | null {
  if (kind === 'deleted') return null;
  if (kind === 'dir') return { kind: 'dir' };
  const data = read();
  // Read out of linear memory into a buffer of its own (`fromWasm`).
  return { kind: 'file', data: data instanceof Uint8Array ? data as Bytes : new Uint8Array(0) };
}

async function store(): Promise<void> {
  for (const [what, backlog, db, storeName] of [
    ['SD card', sdBacklog, sdIdb, SD_STORE],
    ['saves', saveBacklog, nandIdb, NAND_SAVES],
  ] as const) {
    if (!backlog.size) continue;
    const changes = [...backlog];
    backlog.clear();
    try {
      await idbApply(await db(), storeName, changes);
      logStored(what, changes);
    } catch (err) {
      // Put them back under anything newer that arrived meanwhile.
      for (const [key, value] of changes) if (!backlog.has(key)) backlog.set(key, value);
      workerLog(`${what}: could not be written (${String(err)}) - retrying on the next flush.`, 'error');
    }
  }
}

// Paths shown in a store line before the rest is summarized.
const STORED_NAMED_MAX = 8;

function logStored(what: string, changes: [string, StoredEntry | null][]): void {
  let bytes = 0;
  const named: string[] = [];
  for (const [path, entry] of changes) {
    if (entry?.kind === 'file') bytes += entry.data?.length ?? 0;
    if (named.length < STORED_NAMED_MAX) named.push(entry ? path : path + ' (deleted)');
  }
  const more = changes.length > named.length ? `, and ${changes.length - named.length} more` : '';
  workerLog(`[io] ${what}: stored ${changes.length} changes (${fmtSize(bytes)}) in IndexedDB: `
    + named.join(', ') + more, 'debug');
}
