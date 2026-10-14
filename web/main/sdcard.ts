// Asks the browser to keep the SD card and saves; the worker stores them.

import { log } from './log';

// Ask the browser not to evict the card under storage pressure.
export async function sdRequestPersistence(): Promise<void> {
  if (!navigator.storage || !navigator.storage.persist) return;
  try {
    if (await navigator.storage.persisted()) return;
    if (!(await navigator.storage.persist())) {
      log('SD card: storage is not marked persistent - the browser may evict it.', 'debug');
    }
  } catch { /* not fatal: the card still works for this session */ }
}
