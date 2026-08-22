/* ---------- save data ----------

   Same shape as the SD card's persistence, one store further in: entries are
   keyed by "<save id>/<path>", so everything a title saved can be found again
   by prefix and nothing else can see it. A console keeps saves on its NAND
   rather than its card, which is why they live in that database. */

import { idbApply, idbGetAll, NAND_SAVES, nandIdb, type StoredEntry } from './db';
import { log } from './log';
import { call, hasSession } from './rpc';

// Drained but not yet stored, for the same reason the card keeps a backlog:
// the core cannot be handed a change back, so anything IndexedDB refuses waits
// here for the next flush rather than being lost.
const saveBacklog = new Map<string, StoredEntry | null>();
let saveFlushing = false;

// Put every stored save back into a fresh session, through the host entry
// points - which do not count as guest changes, so this does not immediately
// queue everything to be written straight back.
export async function saveRestore(): Promise<void> {
  let entries: [string, StoredEntry][];
  try {
    entries = await idbGetAll<StoredEntry>(await nandIdb(), NAND_SAVES);
  } catch (err) {
    log('Saves: could not be read (' + (err as Error).message + ')', 'err');
    return;
  }
  if (!entries.length) return;
  // Directories first, so one a title left empty survives on its own.
  entries.sort((a, b) => (a[1].kind === b[1].kind ? 0 : a[1].kind === 'dir' ? -1 : 1));
  for (const [key, value] of entries) {
    const cut = key.indexOf('/');
    if (cut < 0) continue;
    const id = key.slice(0, cut);
    const path = key.slice(cut);
    if (value.kind === 'dir') await call('save_create_dir', id, path);
    else await call('save_write_file', id, path, value.data || new Uint8Array(0));
  }
  log('Saves: restored ' + entries.length + ' entries', 'dim');
}

// Write back what the guest changed in any save it has open. Cheap when it
// changed nothing, which is almost every slice.
export async function saveFlush(): Promise<void> {
  if (saveFlushing || !hasSession()) return;
  saveFlushing = true;
  try {
    for (const id of await call('save_ids')) {
      for (const change of await call('save_take_changes', id)) {
        const key = id + change.path;
        if (change.kind === 'deleted') saveBacklog.set(key, null);
        else if (change.kind === 'dir') saveBacklog.set(key, { kind: 'dir' });
        else {
          const data = await call('save_read_file', id, change.path);
          saveBacklog.set(key, { kind: 'file', data: data || new Uint8Array(0) });
        }
      }
    }
    if (saveBacklog.size) {
      await idbApply(await nandIdb(), NAND_SAVES, [...saveBacklog]);
      saveBacklog.clear();
    }
  } catch (err) {
    log('Saves: could not be written (' + (err as Error).message + ') - retrying on the next flush.', 'err');
  } finally {
    saveFlushing = false;
  }
}
