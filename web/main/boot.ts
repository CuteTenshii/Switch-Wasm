// Booting from the stage: the Open buttons and file drops. Executables load
// directly; containers go to `container.ts`. The format comes from the header.

import type { ControlInfo } from '../shared/protocol';
import { bootContainer } from './container';
import { $, pickedFile } from './dom';
import { classify } from './filetype';
import { fmtSize } from './format';
import { awaitFirstFrame, beginLoad, failLoad, loadIdentity, loadPhase } from './loading';
import { clearConsole, log } from './log';
import { call, readLastError } from './rpc';
import { run, updatePc } from './runloop';
import { noteBooted, recycleSession } from './session';
import { dropveilEl, setState, showScreen, stageEl } from './shell';
import { runningIconUrl, setRunning } from './title';
import type { RunningTitle } from './title';

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
  const title = await homebrewTitle(file.name);
  setRunning(title);
  loadIdentity(title.name, runningIconUrl());
  noteBooted();
  setState('loaded');
  // The loading screen stays over it until `display.renderFb` has a real frame.
  showScreen();
  awaitFirstFrame();
  await updatePc();
  return true;
}

// An NRO's icon and name from its asset section, falling back to the file name.
async function homebrewTitle(filename: string): Promise<RunningTitle> {
  let info: ControlInfo;
  try {
    info = JSON.parse(await call('control_json')) as ControlInfo;
  } catch (err) {
    log('No homebrew details: ' + (err as Error).message, 'dim');
    return { name: filename, publisher: '', icon: null, version: '' };
  }
  const icon = info.icon_size > 0 ? await call('control_icon', info.icon_size) : null;
  if (info.name) {
    log('Homebrew: ' + info.name + (info.publisher ? ' - ' + info.publisher : ''), 'ok');
  }
  return {
    name: info.name || filename,
    publisher: info.publisher || '',
    icon: icon && icon.length ? new Blob([icon], { type: info.icon_mime }) : null,
    version: info.version || '',
  };
}

async function bootFile(file: File): Promise<void> {
  const verdict = await classify(file, ['nro', 'elf', 'pfs0', 'xci', 'nca']);
  if (!verdict.ok) {
    // Not a fault: whatever is running keeps running.
    failLoad(verdict.why);
    log(verdict.why, 'err');
    return;
  }
  // A title gets a fresh console. See `recycleSession`.
  try {
    beginLoad(file.name, 'replacing the running session');
    setRunning(null);
    await recycleSession();
  } catch (err) {
    setState('fault');
    failLoad('The session could not be replaced: ' + (err as Error).message);
    log('Reset before boot failed: ' + (err as Error).message, 'err');
    return;
  }
  if (verdict.format === 'pfs0' || verdict.format === 'xci' || verdict.format === 'nca') {
    await bootContainer(file, verdict.format);
    return;
  }
  if (await loadProgram(file, verdict.format)) await run();
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
  if (--dragDepth <= 0) {
    dragDepth = 0;
    dropveilEl.classList.remove('on');
  }
});
stageEl.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropveilEl.classList.remove('on');
  const file = e.dataTransfer?.files[0];
  if (file) await bootFile(file);
});
