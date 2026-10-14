// Booting from the stage: the Open buttons and file drops. Executables load
// directly; containers go to `container.ts`. The format comes from the header.

import type { Bytes, ControlInfo } from '../shared/protocol';
import { bootContainer } from './container';
import { $, pickedFile } from './dom';
import { classify } from './filetype';
import { fmtSize } from './format';
import { awaitFirstFrame, beginLoad, failLoad, loadIdentity, loadPhase } from './loading';
import { log } from './log';
import { call, readLastError } from './rpc';
import { run, updateReadouts } from './runloop';
import { noteBooted, recycleSession } from './session';
import { dropveilEl, setState, showScreen, stageEl } from './shell';
import { runningIconUrl, setRunning } from './title';
import type { RunningTitle } from './title';

export async function loadProgram(file: File, kind: 'nro' | 'elf'): Promise<boolean> {
  setState('loading');
  beginLoad(file.name, 'reading ' + fmtSize(file.size));
  const data = new Uint8Array(await file.arrayBuffer());
  loadPhase(kind === 'nro' ? 'loading the NRO' : 'loading the ELF',
    'mapping the image and seeding the guest');
  return loadWith(file.name, () => kind === 'nro' ? call('load_nro', data) : call('load_elf', data));
}

// Launch `data`, the NRO at `path` on the SD card, on a fresh console as hbmenu does.
export async function launchFromSd(path: string, data: Bytes): Promise<void> {
  const name = path.slice(path.lastIndexOf('/') + 1);
  if (!(await replaceSession(name))) return;
  setState('loading');
  loadPhase('loading the NRO', 'mapping the image and seeding the guest');
  if (await loadWith(name, () => call('load_nro', data, path))) await run();
}

// Load with `load`, which answers the entry point, then show the program as booted.
async function loadWith(name: string, load: () => Promise<number>): Promise<boolean> {
  let entry: number;
  try {
    entry = await load();
  } catch (err) {
    setState('fault');
    failLoad('Load failed: ' + (err as Error).message);
    log('Load failed: ' + (err as Error).message, 'error');
    return false;
  }
  if (entry < 0) {
    const why = await readLastError();
    setState('fault');
    failLoad('Load failed: ' + why);
    log('Load failed: ' + why, 'error');
    return false;
  }
  log('Loaded ' + name + ' - entry 0x' + entry.toString(16).padStart(8, '0'), 'info');
  const title = await homebrewTitle(name);
  setRunning(title);
  loadIdentity(title.name, runningIconUrl());
  noteBooted();
  setState('loaded');
  // The loading screen stays over it until the first real frame (`display.noteFrame`).
  showScreen();
  awaitFirstFrame();
  await updateReadouts();
  return true;
}

// An NRO's icon and name from its asset section, falling back to the file name.
async function homebrewTitle(filename: string): Promise<RunningTitle> {
  let info: ControlInfo;
  try {
    info = JSON.parse(await call('control_json')) as ControlInfo;
  } catch (err) {
    log('No homebrew details: ' + (err as Error).message, 'debug');
    return { name: filename, publisher: '', icon: null, version: '' };
  }
  const icon = info.icon_size > 0 ? await call('control_icon', info.icon_size) : null;
  if (info.name) {
    log('Homebrew: ' + info.name + (info.publisher ? ' - ' + info.publisher : ''), 'info');
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
    log(verdict.why, 'error');
    return;
  }
  if (!(await replaceSession(file.name))) return;
  if (verdict.format === 'pfs0' || verdict.format === 'xci' || verdict.format === 'nca') {
    await bootContainer(file, verdict.format);
    return;
  }
  if (await loadProgram(file, verdict.format)) await run();
}

// A title gets a fresh console. See `recycleSession`.
async function replaceSession(name: string): Promise<boolean> {
  try {
    beginLoad(name, 'replacing the running session');
    setRunning(null);
    await recycleSession();
    return true;
  } catch (err) {
    setState('fault');
    failLoad('The session could not be replaced: ' + (err as Error).message);
    log('A new console could not be prepared for the boot: ' + (err as Error).message, 'error');
    return false;
  }
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
