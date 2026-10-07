// Session bring-up and replacement. A session is a whole console, so a new
// title needs a new session rather than loading over the old one.

import fontUrl from '../font.ttf?url';
import type { Bytes } from '../shared/protocol';
import { hasKeys, stageKeys } from './keys';
import { loadPhase } from './loading';
import { log } from './log';
import { call, setSession } from './rpc';
import { abortRun } from './runloop';
import { restoreArchives } from './nand';
import { saveRestore } from './saves';
import { sdRestore } from './sdcard';
import { stageUsers } from './users';

// Held across sessions; only the staging repeats.
let fontBytes: Bytes | null = null;

export async function stageFont(): Promise<void> {
  if (!fontBytes) {
    try {
      const res = await fetch(fontUrl);
      if (!res.ok) throw new Error(res.status + ' ' + res.statusText);
      fontBytes = new Uint8Array(await res.arrayBuffer());
    } catch (err) {
      log('No system font (' + fontUrl + '): ' + (err as Error).message
        + ' - text will not render.', 'err');
      return;
    }
  }
  await call('load_font', fontBytes);
}

// A session nothing has booted into is already fresh.
let booted = false;

// Record that a title was loaded into the running session.
export function noteBooted(): void {
  booted = true;
}

export function titleBooted(): boolean {
  return booted;
}

// What a rebuilt session needs before it can boot.
export interface Recycle {
  // Re-open the container the Files panel shows; supplied by the caller to avoid an import cycle.
  reopen?: () => Promise<void>;
  // Rebuild even if nothing has been booted, as Reset wants.
  force?: boolean;
}

// Replace the running session, restaging everything it was given. Throws on failure.
export async function recycleSession({ reopen, force = false }: Recycle = {}): Promise<void> {
  if (!booted && !force) return;
  booted = false;
  abortRun();
  // Before the free, so timers stop pushing at the session now.
  setSession(-1);
  loadPhase('freeing the session');
  await call('free_session');
  setSession(await call('new'));
  loadPhase('restoring the SD card, saves, keys and system data');
  // Independent, except the archives, which need the keys.
  await Promise.all([
    stageFont(),
    sdRestore(),
    saveRestore(),
    stageUsers(),
    (hasKeys() ? stageKeys() : Promise.resolve()).then(restoreArchives),
  ]);
  if (reopen) {
    loadPhase('re-opening the container');
    await reopen();
  }
}
