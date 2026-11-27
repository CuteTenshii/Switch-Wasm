// NSP / NCA / XCI containers: inspecting, showing the title, launching.

import type {
  Bytes, ControlData, ControlInfo, DlcEntry, NcaInfo, NspFile, OpenedContainer, UpdateInfo,
} from '../shared/protocol';
import { $, el, pickedFile } from './dom';
import { classify } from './filetype';
import { fmtSize } from '../shared/format';
import { hasProdKeys, onKeysChanged, revealKeys } from './keys';
import {
  awaitFirstFrame, beginLoad, currentAction, failLoad, loadPhase, type LoadAction,
} from './loading';
import { log } from './log';
import {
  describeDlc, describeUpdate, extrasFor, holdDlc, holdUpdate, pairedNotes, type HeldDlc,
} from './pairing';
import { call } from './rpc';
import { run, updateReadouts } from './runloop';
import { endTitle, noteBooted, refuseBootWhileNandBusy } from './session';
import { openPanel, setNote, setState, showScreen } from './shell';
import { setRunning } from './title';
import { appendRows, renderTitleCard } from './titlecard';
import { knownTitle, type KnownTitle } from './titles';

const nspDrop = $('nsp-drop');
nspDrop.addEventListener('dragover', (e) => {
  e.preventDefault();
  nspDrop.classList.add('drag');
});
nspDrop.addEventListener('dragleave', () => nspDrop.classList.remove('drag'));
nspDrop.addEventListener('drop', (e) => {
  e.preventDefault();
  nspDrop.classList.remove('drag');
  const file = e.dataTransfer?.files[0];
  if (file) void handleContainerFile(file);
});
$('nsp-file').addEventListener('change', (e) => {
  const file = pickedFile(e);
  (e.target as HTMLInputElement).value = '';
  if (file) void handleContainerFile(file);
});

async function handleContainerFile(file: File): Promise<void> {
  const verdict = await classify(file, ['pfs0', 'xci', 'nca'], 'Drop it on the screen to boot it.');
  if (!verdict.ok) {
    showInspectNote(verdict.why);
    log(verdict.why, 'error');
    return;
  }
  if (verdict.format === 'nca') {
    await handleStandaloneNca(file);
    return;
  }
  const opened = await openFile(file);
  if (opened?.kind === 'update') handleUpdateFile(file, opened);
  else if (opened?.kind === 'dlc') handleDlcFile(file, opened.packs);
  else if (opened) await showContainer(file, opened.files);
}

// The core reads `file` once and says what it is: null, with the reason shown, if it is no
// container.
async function openFile(file: File): Promise<OpenedContainer | null> {
  log('Opening ' + file.name + ' (' + fmtSize(file.size) + ') ...');
  try {
    return await call('openContainer', file);
  } catch (err) {
    const why = 'Could not open ' + file.name + ': ' + (err as Error).message;
    showInspectNote(why);
    log(why, 'error');
    return null;
  }
}

// In place of the last inspect note, without clearing: an open container should not vanish
// over a bad drop. Matched on `.nca-inspect`: the title card has an `.nca-info` block too.
function showInspectNote(text: string): HTMLElement {
  $('nsp-result').querySelectorAll('.nca-inspect').forEach((node) => node.remove());
  const note = el('div', 'nca-info nca-inspect', text);
  $('nsp-result').appendChild(note);
  return note;
}

// Empty until the open container's Control NCA has been read.
let openTitleId = '';

// `explain` says what happens to the file next; a launch that holds it says that itself.
function handleDlcFile(file: File, packs: DlcEntry[], explain = true): HeldDlc {
  const held = holdDlc(file, packs);
  // The core has logged what the file holds; this is what happens to it.
  if (explain) log(held.titleId === openTitleId
    ? 'The add-on content is mounted when this title launches.'
    : 'Open that game to launch it with the add-on content.', 'info');
  showPairedNotes();
  return held;
}

// Hold each of `files`, updates and add-on content the caller has checked, for the next launch.
export async function holdForLaunch(files: File[]): Promise<void> {
  for (const file of files) {
    const opened = await openFile(file);
    if (opened?.kind === 'update') handleUpdateFile(file, opened, false);
    else if (opened?.kind === 'dlc') handleDlcFile(file, opened.packs, false);
  }
}

function handleUpdateFile(file: File, update: UpdateInfo, explain = true): void {
  holdUpdate(file, update);
  // The core has logged what the file holds; this is what happens to it.
  if (explain) log(update.title === openTitleId
    ? 'The update is applied when this title launches.'
    : 'Open that game to launch it patched.', 'info');
  showPairedNotes();
}

// Rendered under the card, not inside it, so rebuilding the card clears them.
function showPairedNotes(): void {
  $('nsp-result').querySelectorAll('.paired-note').forEach((node) => node.remove());
  let after: Element | null = $('nsp-result').querySelector('.title-card');
  for (const line of pairedNotes(openTitleId)) {
    const note = el('div', 'nca-info paired-note', line);
    if (after) after.after(note);
    else $('nsp-result').prepend(note);
    after = note;
  }
}

// Boot from the stage: open the container and launch its Program NCA.
export async function bootContainer(file: File, format: 'pfs0' | 'xci' | 'nca'): Promise<void> {
  if (refuseBootWhileNandBusy(file.name)) return;
  setState('loading');
  beginLoad(file.name, 'Opening the container (' + fmtSize(file.size) + ')');
  if (format === 'nca') {
    const info = await handleStandaloneNca(file);
    if (!info) {
      setState('fault');
      failLoad('Could not read ' + file.name + '. The Files panel has its header.');
      // The reason is in the panel's output, so open it.
      openPanel('files');
      return;
    }
    if (info.content_type !== 'Program') {
      const why = file.name + ' is a ' + info.content_type
        + ' NCA - only a Program NCA holds an executable.';
      setState('fault');
      failLoad(why);
      log(why, 'error', { openConsole: false });
      return;
    }
    await launchStandaloneNca(file);
    return;
  }
  const opened = await openFile(file);
  if (!opened) {
    setState('fault');
    failLoad('Could not open ' + file.name + '. The Files panel has the details.');
    openPanel('files');
    return;
  }
  // An update or add-on content means: pair it with the game and launch that.
  if (opened.kind === 'update') {
    handleUpdateFile(file, opened);
    setState('idle');
    failLoad(opened.title === openTitleId
      ? 'Update applied - launch ' + (heldTitle?.name || 'the game') + ' to run it patched.'
      : 'That is an update for ' + opened.title + ', not a game. Open that title to launch it patched.');
    openPanel('files');
    return;
  }
  if (opened.kind === 'dlc') {
    const held = handleDlcFile(file, opened.packs);
    setState('idle');
    failLoad(held.titleId === openTitleId
      ? describeDlc(opened.packs.length) + ' added - launch ' + (heldTitle?.name || 'the game')
      + ' to run with them.'
      : 'That is add-on content for ' + held.titleId + ', not a game. Open that title'
        + ' to launch it with them.');
    openPanel('files');
    return;
  }
  await showContainer(file, opened.files);
  await launchNca(file.name);
}

let openContainer: { file: File; kind: 'nsp' | 'nca' } | null = null;

// The container the core has just opened, with its file table: listed, with its title card.
async function showContainer(file: File, files: NspFile[]): Promise<void> {
  clearNsp();
  openContainer = { file, kind: 'nsp' };
  const ul = el('ul', 'nsp-list');
  files.forEach((f, index) => {
    const li = el('li');
    li.appendChild(el('span', 'name', f.name));
    li.appendChild(el('span', 'size', fmtSize(f.size)));
    if (/\.nca$/i.test(f.name)) {
      li.classList.add('clickable');
      li.addEventListener('click', () => inspectNca(f, index));
    }
    ul.appendChild(li);
  });
  $('nsp-result').appendChild(ul);
  setNote('container-badge', 'reading title details\u2026', false);
  await showTitleCard();
  setNote('container-badge', file.name, true);
}

function clearNsp(): void {
  $('nsp-result').textContent = '';
  setNote('container-badge', 'none open', false);
  // The held update outlives the container; the pairing does not.
  openTitleId = '';
  holdTitle(null, null);
}

// The open container's title, which its launch puts on the loading screen.
let heldTitle: KnownTitle | null = null;

function holdTitle(info: ControlInfo | null, icon: Bytes | null): KnownTitle | null {
  heldTitle = info
    ? knownTitle(info, icon)
    : null;
  return heldTitle;
}

// Title card: icon, name, publisher and NACP details from the Control NCA.
// Needs prod.keys, since the content type is in the encrypted header.
async function showTitleCard(): Promise<ControlInfo | null> {
  let control: ControlData;
  try {
    control = await call('loadControl');
  } catch (err) {
    log('No title details: ' + (err as Error).message, 'debug');
    return null;
  }
  const { info, icon } = control;
  if (!info.name) return null;
  const card = renderTitleCard(info, holdTitle(info, icon)?.iconUrl ?? null);
  $('nsp-result').prepend(card);
  openTitleId = info.title_id;
  showPairedNotes();
  return info;
}

async function inspectNca(f: NspFile, index: number): Promise<void> {
  const out = showInspectNote('Parsing ' + f.name + ' ...');
  await parseAndRenderNca(out, index, () => launchNca(f.name, index));
}

// Drop/browse a standalone .nca, which becomes the open container.
async function handleStandaloneNca(file: File): Promise<NcaInfo | null> {
  clearNsp();
  setNote('container-badge', 'opening ' + file.name, false);
  const out = showInspectNote('Parsing ' + file.name + ' ...');
  try {
    await call('openNca', file);
  } catch (e) {
    out.textContent = 'Could not open ' + file.name + ': ' + (e as Error).message;
    setNote('container-badge', 'none open', false);
    return null;
  }
  openContainer = { file, kind: 'nca' };
  setNote('container-badge', file.name, true);
  const info = await parseAndRenderNca(out, undefined, () => launchStandaloneNca(file));
  if (info?.content_type === 'Control') {
    setNote('container-badge', 'reading title details\u2026', false);
    await showTitleCard();
    setNote('container-badge', file.name, true);
  }
  return info;
}

// The header of NSP file `index`, or of the container when it is a standalone NCA.
async function parseAndRenderNca(
  out: HTMLElement,
  index: number | undefined,
  onLaunch: () => void,
): Promise<NcaInfo | null> {
  let info: NcaInfo;
  try {
    info = await call('inspectNca', index);
  } catch (err) {
    out.textContent = 'parse failed: ' + (err as Error).message;
    return null;
  }
  if (info.error) {
    // A CDN NCA's header is encrypted, so the magic is hidden without keys.
    out.textContent = /bad magic/.test(info.error)
      ? 'NCA header is encrypted - load prod.keys in the NAND tab to decrypt and inspect. (' + info.error + ')'
      : 'NCA: ' + info.error;
    return null;
  }
  out.textContent = '';
  const rows: [string, string][] = [
    ['Title ID', info.title_id],
    ['Content type', info.content_type],
    ['SDK version', info.sdk_version],
    ['Crypto', 'type ' + info.crypto_type + (info.encrypted ? ' (encrypted)' : ' (cleartext)')],
    ['File size', fmtSize(info.file_size)],
    ['Sections', info.sections.map((s, i) =>
      '#' + i + ' ' + s.fs_type + ' @' + s.offset + ' (' + fmtSize(s.size) + ')').join(', ')],
  ];
  appendRows(out, rows);
  if (info.content_type === 'Program') {
    const actions = el('div', 'nca-actions');
    const btn = el('button', 'btn small primary', 'Launch');
    btn.addEventListener('click', onLaunch);
    actions.appendChild(btn);
    out.appendChild(actions);
  }
  return info;
}

// Boot NSP file `index`, or without one the program the core finds in the open container.
function launchNca(name: string, index?: number): Promise<void> {
  // The core's trace buffer is cleared by booting, so the page reports the version.
  const notes = [];
  let identity = heldTitle;
  const { update, packs } = extrasFor(openTitleId);
  if (update) {
    notes.push('Update ' + describeUpdate({ version: update.version, name: update.file.name })
      + ' applied: its modules, over the base game\'s data.');
    // The update's version is what runs.
    if (identity && update.version) identity = { ...identity, version: update.version };
  }
  if (packs) notes.push(describeDlc(packs) + ' mounted.');
  // The title card also comes from an encrypted header, so a rerun once keys are in reads it again.
  const relaunch = async () => {
    await showTitleCard();
    await launchNca(name, index);
  };
  return doLaunchNca(name, () => call('launch', index), identity, notes.join(' '), relaunch);
}

// Same as `launchNca`, for a standalone .nca that is already the open container.
function launchStandaloneNca(file: File): Promise<void> {
  return doLaunchNca(file.name, () => call('launch'), heldTitle);
}

// The log's first line: title, publisher, id, version and file.
function describeLaunch(file: string, identity?: KnownTitle | null): string {
  if (!identity) return 'Launching ' + file;
  const parts = [identity.name + (identity.publisher ? ' by ' + identity.publisher : '')];
  if (identity.id) parts.push(identity.id.toUpperCase());
  if (identity.version) parts.push('version ' + identity.version);
  const container = openContainer?.file.name;
  parts.push('from ' + (container && container !== file ? container + ', ' + file : file));
  return 'Launching ' + parts.join(', ');
}

// A launch that failed for want of prod.keys, rerun as soon as they are loaded.
let launchAwaitingKeys: { action: LoadAction; relaunch: () => Promise<void> } | null = null;

onKeysChanged(() => {
  const pending = launchAwaitingKeys;
  if (!pending || !hasProdKeys()) return;
  launchAwaitingKeys = null;
  // Not when the failure was dismissed, or another load has replaced it.
  if (currentAction() === pending.action) void pending.relaunch();
});

export async function doLaunchNca(
  name: string,
  loadFn: () => Promise<number>,
  identity?: KnownTitle | null,
  note?: string,
  relaunch?: () => Promise<void>,
): Promise<void> {
  if (refuseBootWhileNandBusy(identity?.name || name)) return;
  log(describeLaunch(name, identity), 'info');
  if (note) log(note, 'info');
  setState('loading');
  setRunning(null);
  beginLoad(identity?.name || name, 'Decrypting the program and reading its ExeFS',
    identity?.iconUrl);
  try {
    await endTitle();
  } catch (err) {
    const why = (err as Error).message;
    setState('fault');
    failLoad('The running title could not be ended: ' + why);
    log('The running title could not be ended: ' + why, 'error', { openConsole: false });
    return;
  }
  try {
    await loadFn();
  } catch (err) {
    const why = (err as Error).message;
    setState('fault');
    // Without prod.keys no NCA header decrypts, so that is the cause whatever the core reports.
    if (hasProdKeys()) failLoad('Launch failed: ' + why);
    else {
      const action = { label: 'Load prod.keys', run: revealKeys };
      launchAwaitingKeys = {
        action,
        relaunch: relaunch ?? (() => doLaunchNca(name, loadFn, identity, note)),
      };
      failLoad('This game is encrypted. Load the prod.keys file from your Switch to play it.',
        undefined, action);
    }
    log('Launch failed: ' + why, 'error', { openConsole: false });
    return;
  }
  setRunning({
    id: identity?.id || '',
    name: identity?.name || name,
    publisher: identity?.publisher || '',
    version: identity?.version || '',
    iconUrl: identity?.iconUrl ?? null,
  });
  noteBooted();
  setState('loaded');
  loadPhase('Starting the process');
  showScreen();
  awaitFirstFrame();
  await updateReadouts();
  await run();
}
