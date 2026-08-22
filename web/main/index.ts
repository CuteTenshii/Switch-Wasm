/* switch-wasm browser frontend.

   The emulator runs in a web worker (web/worker) so long executions don't
   freeze the page; this module starts it, brings a session up with everything
   the last one persisted, and wires the one action that touches every part of
   the page at once - Reset. */

import fontUrl from '../font.ttf?url';
import type { Bytes } from '../shared/protocol';
import { resetAudio } from './audio';
import { watchBattery } from './battery';
import { initFbSize, resetDisplay } from './display';
import { $ } from './dom';
import { hasKeys, stageKeys, updateKeysState } from './keys';
import { clearConsole, log } from './log';
import { initNand, restoreArchives } from './nand';
import { call, initWorker, setSession, whenReady } from './rpc';
import { abortRun, updatePc } from './runloop';
import { saveRestore } from './saves';
import { sdRequestPersistence, sdRestore } from './sdcard';
import { screenCtx, screenEl, setState, showOverlay } from './shell';
import { reopenContainer } from './container';

// Registered for their side effects: each of these owns a part of the page and
// binds its own controls when it is loaded.
import './boot';
import './debug';
import './input';

// The font the emulator serves as the console's shared system font. Homebrew
// reads it out of pl:u's shared memory and renders it with its own copy of
// FreeType, so without it nothing but pre-rendered bitmaps appears on screen.
// `fontUrl` is the built file's hashed URL, so a replaced font is a fetch the
// browser cannot answer from its cache.
let fontBytes: Bytes | null = null;

async function stageFont(): Promise<void> {
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

async function init(): Promise<void> {
  initWorker();
  await whenReady();
  setSession(await call('new'));
  await stageFont();
  await sdRequestPersistence();
  await sdRestore();
  await saveRestore();
  await initFbSize();
  $('wasm-ver').textContent = 'core ready';
  log('core ready', 'dim');
  // Restore persisted keys into the session.
  if (hasKeys()) await stageKeys();
  updateKeysState();
  // And then the NAND, which needs those keys to parse a header.
  await initNand();
}

$('btn-reset').addEventListener('click', async () => {
  abortRun();
  await call('free_session');
  setSession(await call('new'));
  await stageFont();
  await sdRestore();
  await saveRestore();
  // Everything else the page is still showing. A new session starts with no
  // keys, no container and no data archives, while the panel above goes on
  // reporting all three -- so Launch failed with "no container is open" on a
  // card that was still sitting on screen.
  await stageKeys();
  await restoreArchives();
  await reopenContainer();
  resetAudio();
  clearConsole();
  resetDisplay();
  setState('idle');
  showOverlay(true);
  screenCtx.clearRect(0, 0, screenEl.width, screenEl.height);
  await updatePc();
});

watchBattery();
init();
