// Starting and stopping titles. Every boot, whether a file, an SD homebrew, a container or a NAND
// title, ends what runs, loads, and shows the result the same way.

import { endLoad, awaitFirstFrame, beginLoad, failLoad, loadIdentity, loadPhase } from './loading';
import { log } from './log';
import { call } from './rpc';
import { run, updateReadouts } from './runloop';
import { endTitle, noteBooted, noteLoading, refuseBootWhileNandBusy, titleEnded } from './session';
import { setState, showOverlay, showScreen } from './shell';
import { runningIconUrl, setRunning } from './title';
import type { KnownTitle } from './titles';

/** A boot that could not go on: the loading screen and the log say why. */
export function failBoot(message: string): void {
  setState('fault');
  failLoad(message);
  log(message, 'error', { openConsole: false });
}

/** End whatever runs before booting `name`; false, with the failure shown, when it can't be. */
export async function replaceTitle(name: string, iconUrl?: string | null): Promise<boolean> {
  beginLoad(name, 'Ending the running title', iconUrl);
  setRunning(null);
  if (titleEnded()) return true;
  try {
    await endTitle();
    return true;
  } catch (err) {
    failBoot('The running title could not be ended: ' + (err as Error).message);
    return false;
  }
}

export interface TitleStart {
  // The file or label the boot is known by until the title names itself.
  name: string;
  // What is known about the title before it loads; homebrew learns it from `loaded`.
  identity?: KnownTitle | null;
  phase: string;
  detail?: string;
  // Logged once the boot is allowed to go ahead.
  announce?: string[];
  // Throws why the title can't load.
  load: () => Promise<number>;
  // The title as it runs, read once it has loaded.
  loaded?: () => Promise<KnownTitle>;
  // Shows why the load failed; a plain "Load failed" without it.
  failed?: (why: string) => void;
}

/** Boot `start` in place of whatever runs, and run it; false when it did not boot. */
export async function startTitle(start: TitleStart): Promise<boolean> {
  const label = start.identity?.name || start.name;
  if (refuseBootWhileNandBusy(label)) return false;
  for (const line of start.announce ?? []) log(line, 'info');
  setState('loading');
  if (!(await replaceTitle(label, start.identity?.iconUrl))) return false;
  loadPhase(start.phase, start.detail);
  noteLoading();
  try {
    await start.load();
  } catch (err) {
    const why = (err as Error).message;
    if (start.failed) start.failed(why);
    else failBoot('Load failed: ' + why);
    return false;
  }
  const title: KnownTitle = await start.loaded?.() ?? {
    id: start.identity?.id || '',
    name: label,
    publisher: start.identity?.publisher || '',
    version: start.identity?.version || '',
    iconUrl: start.identity?.iconUrl ?? null,
  };
  setRunning(title);
  loadIdentity(title.name, runningIconUrl());
  noteBooted();
  setState('loaded');
  // The loading screen stays over it until the first real frame (`display.noteFrame`).
  showScreen();
  awaitFirstFrame();
  await updateReadouts();
  await run();
  return true;
}

/** End the running title and go back to the idle console. */
export async function stopTitle(): Promise<void> {
  // Stops the run before any slice report is handled.
  const ended = endTitle();
  endLoad();
  setRunning(null);
  setState('idle');
  showOverlay(true);
  try {
    await ended;
  } catch (err) {
    log('Stop failed: ' + (err as Error).message, 'error');
    return;
  }
  await call('clearScreen');
  await updateReadouts();
}
