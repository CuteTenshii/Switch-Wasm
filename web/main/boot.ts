// Booting from the stage: the Open buttons and file drops. Executables load
// directly; containers go to `container.ts`. The format comes from the header.

import type { ControlData } from '../shared/protocol';
import { bootContainer } from './container';
import { $, pickedFile } from './dom';
import { classify } from './filetype';
import { failLoad } from './loading';
import { replaceTitle, startTitle } from './launch';
import { log } from './log';
import { nandBusy, onNandBusy } from './nandbusy';
import { call } from './rpc';
import { onChainLoad } from './runloop';
import { refuseBootWhileNandBusy } from './session';
import { dropveilEl, stageEl } from './shell';
import { knownTitle, type KnownTitle } from './titles';

function loadProgram(file: File, kind: 'nro' | 'elf'): Promise<boolean> {
  return startTitle({
    name: file.name,
    phase: kind === 'nro' ? 'Loading the NRO' : 'Loading the ELF',
    detail: 'Reading, mapping the image and seeding the guest',
    load: () => call('loadProgram', file),
    loaded: () => homebrewTitle(file.name),
  });
}

// Launch the NRO at `path` on the SD card in place of the running title, as hbmenu does.
export async function launchFromSd(path: string): Promise<void> {
  const name = path.slice(path.lastIndexOf('/') + 1);
  await startTitle({
    name,
    phase: 'Loading the NRO',
    detail: 'Mapping the image and seeding the guest',
    load: () => call('loadSd', path),
    loaded: () => homebrewTitle(name),
  });
}

onChainLoad((path) => void launchFromSd(path));

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
  const title = knownTitle(info, icon);
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
  await loadProgram(file, verdict.format);
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
