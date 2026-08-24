/* Loading a homebrew program: the Open buttons, and dropping one on the
   stage. */

import { $, pickedFile } from './dom';
import { fmtSize } from './format';
import { awaitFirstFrame, beginLoad, failLoad, loadPhase } from './loading';
import { clearConsole, log } from './log';
import { call, readLastError } from './rpc';
import { run, updatePc } from './runloop';
import { dropveilEl, setState, showScreen, stageEl } from './shell';

export async function loadProgram(file: File, kind: 'nro' | 'elf'): Promise<boolean> {
  clearConsole();
  setState('loading');
  beginLoad(file.name, 'reading ' + fmtSize(file.size));
  const data = new Uint8Array(await file.arrayBuffer());
  loadPhase(kind === 'nro' ? 'loading the NRO' : 'loading the ELF',
    'mapping the image and seeding the guest');
  let entry: number;
  try {
    entry = kind === 'nro'
      ? await call('load_nro', data)
      : await call('load_elf', data);
  } catch (err) {
    setState('fault');
    failLoad('Load failed: ' + (err as Error).message);
    log('Load failed: ' + (err as Error).message, 'err');
    return false;
  }
  if (entry < 0) {
    const why = await readLastError();
    setState('fault');
    failLoad('Load failed: ' + why);
    log('Load failed: ' + why, 'err');
    return false;
  }
  log('Loaded ' + file.name + ' - entry 0x' + entry.toString(16).padStart(8, '0'), 'ok');
  setState('loaded');
  // Uncover the emulated screen now, but keep the loading screen over it:
  // homebrew can run for a long time (or fault) before it presents anything,
  // and a blank stage with a live step counter on it is the difference between
  // "still booting" and "dead". `display.renderFb` takes the screen down as
  // soon as there is a real frame under it.
  showScreen();
  awaitFirstFrame();
  await updatePc();
  return true;
}

async function bootFile(file: File): Promise<void> {
  const kind = /\.nro$/i.test(file.name) ? 'nro' : 'elf';
  if (await loadProgram(file, kind)) await run();
}

for (const id of ['nro-file', 'nro-file-2']) {
  $(id).addEventListener('change', async (e) => {
    const f = pickedFile(e);
    (e.target as HTMLInputElement).value = '';
    if (f) await bootFile(f);
  });
}

// Drop an NRO anywhere on the stage to boot it.
let dragDepth = 0;
stageEl.addEventListener('dragenter', (e) => {
  e.preventDefault();
  if (++dragDepth === 1) dropveilEl.classList.add('on');
});
stageEl.addEventListener('dragover', (e) => e.preventDefault());
stageEl.addEventListener('dragleave', () => {
  if (--dragDepth <= 0) { dragDepth = 0; dropveilEl.classList.remove('on'); }
});
stageEl.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropveilEl.classList.remove('on');
  const file = e.dataTransfer?.files[0];
  if (file) await bootFile(file);
});
