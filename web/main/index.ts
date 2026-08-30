/* switch-wasm browser frontend.

   The emulator runs in a web worker (web/worker) so long executions don't
   freeze the page; this module starts it, brings a session up with everything
   the last one persisted, and wires the one action that touches every part of
   the page at once - Reset. */

import { resetAudio } from './audio';
import { watchBattery } from './battery';
import { initFbSize, resetDisplay } from './display';
import { $ } from './dom';
import { hasKeys, stageKeys, updateKeysState } from './keys';
import { clearConsole, log } from './log';
import { initNand } from './nand';
import { call, initWorker, setSession, whenReady } from './rpc';
import { updatePc } from './runloop';
import { saveRestore } from './saves';
import { sdRequestPersistence, sdRestore } from './sdcard';
import { screenCtx, screenEl, setState, showOverlay } from './shell';
import { reopenContainer } from './container';
import { recycleSession, stageFont } from './session';
import { setRunning } from './title';
import { beginLoad, endLoad, failLoad, loadPhase } from './loading';

// Registered for their side effects: each of these owns a part of the page and
// binds its own controls when it is loaded.
import './boot';
import './debug';
import './dock';
import './input';

// Bringing the core up is the page's own load, and every step of it is
// something that can take a visible moment on a cold cache or a full SD card.
// The loading screen is up from first paint (see index.html) so this only ever
// names the step it has reached.
async function init(): Promise<void> {
  try {
    initWorker();
    await whenReady();
    loadPhase('creating a session');
    setSession(await call('new'));
    loadPhase('loading the system font');
    await stageFont();
    loadPhase('restoring the SD card');
    await sdRequestPersistence();
    await sdRestore();
    loadPhase('restoring save data');
    await saveRestore();
    await initFbSize();
    $('wasm-ver').textContent = 'core ready';
    log('core ready', 'dim');
    // Restore persisted keys into the session.
    if (hasKeys()) {
      loadPhase('staging keys');
      await stageKeys();
    }
    updateKeysState();
    // And then the NAND, which needs those keys to parse a header. Only its
    // index is waited for here; the archives it holds register behind the
    // page, so a firmware dump does not stand between a cold load and the
    // first file the user drops on the stage.
    loadPhase('reading the NAND');
    await initNand();
  } catch (err) {
    // A core that never came up leaves nothing behind it worth uncovering, so
    // the screen stays and says so rather than handing over to an idle splash
    // that would invite a boot which cannot work.
    failLoad('The core could not be started: ' + (err as Error).message);
    log('core failed to start: ' + (err as Error).message, 'err');
    return;
  }
  endLoad();
}

$('btn-reset').addEventListener('click', async () => {
  // A reset rebuilds everything a boot built, so it reports itself the same
  // way a boot does instead of freezing the stage on the last frame it drew.
  // Stopping the run and disowning the session are `recycleSession`'s first
  // two acts, so they are not repeated here.
  beginLoad('resetting', 'freeing the session');
  try {
    // `force`, because Reset means "give me a new console" whether or not this
    // one ever booted anything, and `reopen` because the page is left showing
    // exactly what it was showing before.
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
init();
