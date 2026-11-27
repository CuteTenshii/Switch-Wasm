// Session bring-up and ending a title. The session is the console, staged once; each title
// is a process in it.

import { resetAudio } from './audio';
import { resetDisplay } from './display';
import { log } from './log';
import { resetMemoryHistory } from './memory';
import { bootRefusal } from './nandbusy';
import { call } from './rpc';
import { abortRun } from './runloop';
import { stageUsers } from './users';

let booted = false;
// Nothing has been loaded since the title was last ended, so ending it again would do nothing.
let ended = true;

// Record that a load into the session begins; whether it succeeds or fails, it leaves a title to end.
export function noteLoading(): void {
  ended = false;
}

// Record that a title was loaded.
export function noteBooted(): void {
  booted = true;
}

export function titleEnded(): boolean {
  return ended;
}

export function titleBooted(): boolean {
  return booted;
}

// Whether `name` may not boot now, while a NAND operation runs; says why in the log.
export function refuseBootWhileNandBusy(name: string): boolean {
  const why = bootRefusal(name);
  if (why) log(why, 'warn');
  return why !== null;
}

// Leaving the page drops a loaded title's unsaved progress; the idle console loses nothing.
// Development skips it so reloads stay one keystroke.
if (!import.meta.env.DEV) {
  window.addEventListener('beforeunload', (event) => {
    if (booted) event.preventDefault();
  });
}

// End the running title; the console, with its storage, keys and system data, stays.
export async function endTitle(): Promise<void> {
  abortRun();
  // The next title counts its frames and plays its sound from zero.
  resetAudio();
  resetDisplay();
  await call('endTitle');
  ended = true;
  booted = false;
  resetMemoryHistory();
  // A running title keeps the player it started with; the next one gets the current list.
  await stageUsers();
}
