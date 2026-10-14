// Session bring-up and ending a title. The session is the console, staged once; each title
// is a process in it.

import fontUrl from '../font.ttf?url';
import type { Bytes } from '../shared/protocol';
import { log } from './log';
import { call } from './rpc';
import { abortRun } from './runloop';
import { stageUsers } from './users';

export async function stageFont(): Promise<void> {
  let font: Bytes;
  try {
    const res = await fetch(fontUrl);
    if (!res.ok) throw new Error(res.status + ' ' + res.statusText);
    font = new Uint8Array(await res.arrayBuffer());
  } catch (err) {
    log('No system font (' + fontUrl + '): ' + (err as Error).message
      + ' - text will not render.', 'error');
    return;
  }
  await call('load_font', font);
}

let booted = false;

// Record that a title was loaded.
export function noteBooted(): void {
  booted = true;
}

export function titleBooted(): boolean {
  return booted;
}

// End the running title; the console, with its storage, keys and system data, stays.
export async function endTitle(): Promise<void> {
  abortRun();
  await call('end_title');
  booted = false;
  // A running title keeps the player it started with; the next one gets the current list.
  await stageUsers();
}
