/* Bringing a session up, and replacing the one that is running.

   A session owns the whole emulated console: its guest RAM, its threads, its
   filesystem handles. Nothing inside one is cleared per title, because on real
   hardware nothing is - a console loads a title into a machine that was just
   powered on. So the only way to start a second title cleanly is to throw the
   machine away and build another, which is what `recycleSession` does.

   Reset did this and booting did not, so opening a game while one was running
   loaded it on top of the last one: the outgoing title's pages stayed mapped
   (a retail game is a couple of hundred MiB of them), its globals stayed
   where the new title's would go, and guest RAM only ever went up. A few
   swaps reached the RAM cap and the next allocation failed inside whatever
   the title happened to be doing - a `stp` loop with no allocation and no
   service anywhere near it. */

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

// `fontUrl` is the built file's hashed URL, so a replaced font is a fetch the
// browser cannot answer from its cache. Held across sessions: the bytes are
// the same every time, and only the staging into the session has to repeat.
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

// Whether a title has been loaded into the session that is running now. A
// session nothing has booted into is already the fresh one a boot wants, and
// throwing away the one `init` just built to make another costs a visible
// moment for no difference.
let booted = false;

/** Record that a title was loaded into the running session, so the next boot
 *  knows it is replacing something. Every path that loads one calls this. */
export function noteBooted(): void {
  booted = true;
}

/** What a rebuilt session has to be given back before it can boot anything. */
export interface Recycle {
  /** Re-open the container the Files panel is still showing, if the caller
   *  needs one in the new session.
   *
   * Supplied by the caller rather than reached for here, so that rebuilding a
   * session does not have to know what a container is — `container.ts` is
   * what would then import this back. Reset and a panel Launch both pass it:
   * the page goes on showing an open container, and a launch reads the title
   * out of one. A boot from the stage does not, since it is about to open a
   * different container anyway. */
  reopen?: () => Promise<void>;
  /** Rebuild even if nothing has been booted into this session. Reset means
   *  "give me a new console" whether or not the current one ever ran. */
  force?: boolean;
}

/** Throw the running session away and build a fresh one in its place.
 *
 * Everything a session was *given* is staged again — the font, the SD card,
 * save data, keys, the system data archives — because a new session starts
 * with none of them while the page goes on reporting all of them. Throws if
 * the session could not be rebuilt; the caller owns how that is reported,
 * since Reset and a failed boot say different things about it. */
export async function recycleSession({ reopen, force = false }: Recycle = {}): Promise<void> {
  if (!booted && !force) return;
  booted = false;
  abortRun();
  // Said before the free is even posted, so that everything which pushes at
  // the session on a timer stops now rather than one round trip from now.
  setSession(-1);
  loadPhase('freeing the session');
  await call('free_session');
  setSession(await call('new'));
  loadPhase('loading the system font');
  await stageFont();
  loadPhase('restoring the SD card');
  await sdRestore();
  loadPhase('restoring save data');
  await saveRestore();
  if (hasKeys()) {
    loadPhase('staging keys');
    await stageKeys();
  }
  loadPhase('restoring system data archives');
  await restoreArchives();
  if (reopen) {
    loadPhase('re-opening the container');
    await reopen();
  }
}
