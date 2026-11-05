// Session bring-up and ending a title. The session is the console, staged once; each title
// is a process in it.

import { resetMemoryHistory } from './memory';
import { call } from './rpc';
import { abortRun } from './runloop';
import { stageUsers } from './users';

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
  resetMemoryHistory();
  // A running title keeps the player it started with; the next one gets the current list.
  await stageUsers();
}
