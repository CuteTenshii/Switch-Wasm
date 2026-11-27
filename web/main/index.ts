// switch-wasm browser frontend entry: starts the worker, restores a session
// and wires Stop.

import { resetAudio } from './audio';
import { watchBattery } from './battery';
import { resetDisplay } from './display';
import { $ } from './dom';
import { hasKeys, stageKeys, updateKeysState } from './keys';
import { log, logLevel, offerPreviousLog, onLogLevel } from './log';
import { initNand } from './nand';
import { attachScreen, call, initWorker, whenStarted } from './rpc';
import { updateReadouts, updateRam } from './runloop';
import { loadProfiles, stageUsers } from './users';
import { sdRequestPersistence } from './sdcard';
import { screenEl, setState, showOverlay } from './shell';
import { endTitle } from './session';
import { setRunning } from './title';
import { restoreLibrary } from './library';
import { endLoad, failLoad, loadPhase } from './loading';

// Imported for their side effects: each binds its own part of the page.
import './boot';
import './dock';
import './elements/info-tip';
import './network';
import './readonly';
import './relay-settings';
import './files';
import './input';

// iOS greys out every file whose extension it has no type for, which is all of ours.
// iPadOS reports itself as a Mac, told apart by its touchscreen.
const ios = /iPhone|iPad|iPod/.test(navigator.userAgent)
  || (navigator.userAgent.includes('Macintosh') && navigator.maxTouchPoints > 1);
if (ios) {
  for (const input of document.querySelectorAll('input[type=file][accept^="."]')) input.removeAttribute('accept');
}

// Diagnostic channels record only at trace log level.
onLogLevel((level) => void call('setTraceChannels', level === 'trace'));

// The loading screen is up from first paint (see index.html); this names each step.
async function init(): Promise<void> {
  try {
    initWorker();
    await whenStarted();
    // Firefox settles this only once its prompt is answered, so boot must not wait on it.
    void sdRequestPersistence();
    // The worker paints the screen from here on.
    attachScreen(screenEl.transferControlToOffscreen());
    loadPhase('Restoring the SD card, saves, profiles and keys');
    // Independent of each other; the worker built the session when it loaded the core.
    await Promise.all([
      call('restoreStorage'),
      loadProfiles().then(stageUsers),
      offerPreviousLog(),
      call('setTraceChannels', logLevel() === 'trace'),
      hasKeys() ? stageKeys() : undefined,
    ]);
    updateKeysState();
    // The NAND needs the keys. Only its index is awaited; archives register later.
    loadPhase('Reading the NAND');
    await initNand();
    await updateRam();
  } catch (err) {
    const why = (err as Error).message;
    failLoad(why, 'The core could not start');
    log('The core could not start: ' + why, 'error', { openConsole: false });
    return;
  }
  endLoad();
  await restoreLibrary();
}

$('btn-stop').addEventListener('click', async () => {
  // Stops the run before any slice report is handled.
  const ended = endTitle();
  endLoad();
  resetAudio();
  resetDisplay();
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
});

watchBattery();
void init();
