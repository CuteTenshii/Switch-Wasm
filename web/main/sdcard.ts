/* ---------- persistent SD card ----------

   The emulated SD card lives in the session's memory, so without this nothing
   the guest writes survives a reload - and a save manager that cannot keep a
   save is not much of one. IndexedDB fits the shape the core exposes: the card
   is a path -> bytes map, and the core reports which paths the *guest*
   changed, so a flush writes back only those instead of the whole card. */

import { idbApply, idbGetAll, SD_STORE, sdIdb, type StoredEntry } from './db';
import { log } from './log';
import { call, hasSession } from './rpc';

// Entries drained from the core but not yet stored - keyed by path, so a file
// written repeatedly between two successful flushes only costs one slot. The
// core cannot be handed a change back once drained, so anything IndexedDB
// refuses (a quota, most likely) waits here for the next flush rather than
// being lost.
const sdBacklog = new Map<string, StoredEntry | null>();
let sdFlushing = false;

// Ask the browser not to evict the card under storage pressure. Without this
// IndexedDB is best-effort and a save can quietly disappear.
export async function sdRequestPersistence(): Promise<void> {
  if (!navigator.storage || !navigator.storage.persist) return;
  try {
    if (await navigator.storage.persisted()) return;
    if (!(await navigator.storage.persist())) {
      log('SD card: storage is not marked persistent - the browser may evict it.', 'dim');
    }
  } catch { /* not fatal: the card still works for this session */ }
}

// Put the stored card back into a fresh session. Restores through the host
// entry points, which do not count as guest changes, so this does not
// immediately queue everything to be written straight back.
export async function sdRestore(): Promise<void> {
  let entries: [string, StoredEntry][];
  try {
    entries = await idbGetAll<StoredEntry>(await sdIdb(), SD_STORE);
  } catch (err) {
    log('SD card: could not be read (' + err + ')', 'err');
    return;
  }
  if (!entries.length) return;
  // Directories first, so one the guest left empty survives on its own.
  entries.sort((a, b) => (a[1].kind === b[1].kind ? 0 : a[1].kind === 'dir' ? -1 : 1));
  for (const [path, value] of entries) {
    if (value.kind === 'dir') await call('sd_create_dir', path);
    else await call('sd_write_file', path, value.data || new Uint8Array(0));
  }
  log('SD card: restored ' + entries.length + ' entries', 'dim');
}

// Write back what the guest changed. Cheap when it changed nothing, which is
// almost every slice.
export async function sdFlush(): Promise<void> {
  if (sdFlushing || !hasSession()) return;
  sdFlushing = true;
  try {
    for (const change of await call('sd_take_changes')) {
      if (change.kind === 'deleted') sdBacklog.set(change.path, null);
      else if (change.kind === 'dir') sdBacklog.set(change.path, { kind: 'dir' });
      else {
        const data = await call('sd_read_file', change.path);
        sdBacklog.set(change.path, { kind: 'file', data: data || new Uint8Array(0) });
      }
    }
    if (sdBacklog.size) {
      await idbApply(await sdIdb(), SD_STORE, [...sdBacklog]);
      sdBacklog.clear();
    }
  } catch (err) {
    log('SD card: could not be written (' + err + ') - retrying on the next flush.', 'err');
  } finally {
    sdFlushing = false;
  }
}
