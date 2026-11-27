// Booting from the stage: the Open buttons and file drops. Executables load
// directly; containers go to `container.ts`. The format comes from the header.

import type { ControlData } from '../shared/protocol';
import { bootContainer } from './container';
import { $, pickedFile } from './dom';
import { classify } from './filetype';
import { fmtSize } from '../shared/format';
import { awaitFirstFrame, beginLoad, failLoad, loadIdentity, loadPhase } from './loading';
import { log } from './log';
import { nandBusy, onNandBusy } from './nandbusy';
import { call } from './rpc';
import { onChainLoad, run, updateReadouts } from './runloop';
import { endTitle, noteBooted, refuseBootWhileNandBusy } from './session';
import { dropveilEl, setState, showScreen, stageEl } from './shell';
import { runningIconUrl, setRunning } from './title';
import { knownTitle, type KnownTitle } from './titles';

export async function loadProgram(file: File, kind: 'nro' | 'elf'): Promise<boolean> {
  if (refuseBootWhileNandBusy(file.name)) return false;
  setState('loading');
  beginLoad(file.name, 'Reading ' + fmtSize(file.size));
  loadPhase(kind === 'nro' ? 'Loading the NRO' : 'Loading the ELF',
    'Reading, mapping the image and seeding the guest');
  return loadWith(file.name, () => call('loadProgram', file));
}

// Launch the NRO at `path` on the SD card in place of the running title, as hbmenu does.
export async function launchFromSd(path: string): Promise<void> {
  const name = path.slice(path.lastIndexOf('/') + 1);
  if (refuseBootWhileNandBusy(name)) return;
  if (!(await replaceTitle(name))) return;
  setState('loading');
  loadPhase('Loading the NRO', 'Mapping the image and seeding the guest');
  if (await loadWith(name, () => call('loadSd', path))) await run();
}

onChainLoad((path) => void launchFromSd(path));

// Load with `load`, which throws why it cannot, then show the program as booted.
async function loadWith(name: string, load: () => Promise<number>): Promise<boolean> {
  try {
    await load();
  } catch (err) {
    const why = (err as Error).message;
    setState('fault');
    failLoad('Load failed: ' + why);
    log('Load failed: ' + why, 'error', { openConsole: false });
    return false;
  }
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
async function homebrewTitle(filename: string): Promise<KnownTitle> {
  let control: ControlData;
  try {
    control = await call('control');
  } catch (err) {
    log('No homebrew details: ' + (err as Error).message, 'debug');
    return { id: '', name: filename, publisher: '', version: '', iconUrl: null };
  }
  const { info, icon } = control;
  const title = knownTitle(
    { id: info.title_id, name: info.name, publisher: info.publisher, version: info.version },
    icon, info.icon_mime);
  return { ...title, name: title.name || filename };
}

export async function bootFile(file: File): Promise<void> {
  const verdict = await classify(file, ['nro', 'elf', 'pfs0', 'xci', 'nca']);
  if (!verdict.ok) {
    // Not a fault: whatever is running keeps running.
    failLoad(verdict.why);
    log(verdict.why, 'error', { openConsole: false });
    return;
  }
  // Before the running title is ended for it.
  if (refuseBootWhileNandBusy(file.name)) return;
  if (!(await replaceTitle(file.name))) return;
  if (verdict.format === 'pfs0' || verdict.format === 'xci' || verdict.format === 'nca') {
    await bootContainer(file, verdict.format);
    return;
  }
  if (await loadProgram(file, verdict.format)) await run();
}

// End whatever runs before booting `name`.
async function replaceTitle(name: string): Promise<boolean> {
  try {
    beginLoad(name, 'Ending the running title');
    setRunning(null);
    await endTitle();
    return true;
  } catch (err) {
    const why = (err as Error).message;
    setState('fault');
    failLoad('The running title could not be ended: ' + why);
    log('The running title could not be ended: ' + why, 'error', { openConsole: false });
    return false;
  }
}

// The Open buttons are labels for the file input, so disabling the input stops the picker.
const openTitles = new Map<string, string>();
onNandBusy(() => {
  const busy = nandBusy();
  $<HTMLInputElement>('nro-file').disabled = busy !== null;
  for (const id of ['btn-open', 'overlay-open']) {
    const label = $(id);
    if (!openTitles.has(id)) openTitles.set(id, label.title);
    label.setAttribute('aria-disabled', String(busy !== null));
    label.title = busy ? busy + ' on the NAND; opening a file waits for it.' : openTitles.get(id) ?? '';
  }
});

$('nro-file').addEventListener('change', async (e) => {
  const f = pickedFile(e);
  (e.target as HTMLInputElement).value = '';
  if (f) await bootFile(f);
});

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
