// switch-wasm browser frontend entry: starts the worker, restores a session
// and wires Reset.

import { resetAudio } from './audio';
import { watchBattery } from './battery';
import { controlRestore } from './controls';
import { initFbSize, resetDisplay } from './display';
import { $ } from './dom';
import { hasKeys, stageKeys, updateKeysState } from './keys';
import { clearConsole, log, logLevel, offerPreviousLog, onLogLevel } from './log';
import { initNand } from './nand';
import { call, initWorker, setSession, whenReady } from './rpc';
import { updatePc } from './runloop';
import { saveRestore } from './saves';
import { loadProfiles, stageUsers } from './users';
import { sdRequestPersistence, sdRestore } from './sdcard';
import { screenCtx, screenEl, setState, showOverlay } from './shell';
import { reopenContainer } from './container';
import { recycleSession, stageFont } from './session';
import { setRunning } from './title';
import { beginLoad, endLoad, failLoad, loadPhase } from './loading';

// Imported for their side effects: each binds its own part of the page.
import './boot';
import './dock';
import './files';
import './input';

// Diagnostic channels record only at debug log level.
onLogLevel((level) => void call('set_trace_channels', level === 'debug'));

// The loading screen is up from first paint (see index.html); this names each step.
async function init(): Promise<void> {
  try {
    initWorker();
    await whenReady();
    loadPhase('creating a session');
    setSession(await call('new'));
    loadPhase('restoring the SD card, saves, profiles and keys');
    // Independent of each other once the session exists.
    const [version] = await Promise.all([
      call('version'),
      stageFont(),
      sdRequestPersistence().then(sdRestore),
      saveRestore(),
      controlRestore(),
      loadProfiles().then(stageUsers),
      initFbSize(),
      offerPreviousLog(),
      call('set_trace_channels', logLevel() === 'debug'),
      hasKeys() ? stageKeys() : undefined,
    ]);
    $('wasm-ver').textContent = 'core ' + version;
    log('core ready - build ' + version, 'ok');
    updateKeysState();
    // The NAND needs the keys. Only its index is awaited; archives register later.
    loadPhase('reading the NAND');
    await initNand();
  } catch (err) {
    failLoad('The core could not be started: ' + (err as Error).message);
    log('core failed to start: ' + (err as Error).message, 'err');
    return;
  }
  endLoad();
}

$('btn-reset').addEventListener('click', async () => {
  // `recycleSession` stops the run and disowns the session first.
  beginLoad('resetting', 'freeing the session');
  try {
    // `force`: a new console even if nothing booted.
    await recycleSession({ reopen: reopenContainer, force: true });
  } catch (err) {
    failLoad('The session could not be rebuilt: ' + (err as Error).message);
    log('Reset failed: ' + (err as Error).message, 'err');
    return;
  }
  resetAudio();
  clearConsole();
  resetDisplay();
  setRunning(null);
  setState('idle');
  showOverlay(true);
  screenCtx.clearRect(0, 0, screenEl.width, screenEl.height);
  await updatePc();
  endLoad();
});

watchBattery();
void init();
