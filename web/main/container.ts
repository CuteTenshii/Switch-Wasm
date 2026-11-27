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
import { call } from './rpc';
import { run, updateReadouts } from './runloop';
import { endTitle, noteBooted, refuseBootWhileNandBusy } from './session';
import { openPanel, setNote, setState, showScreen } from './shell';
import { setRunning } from './title';
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
    // Without clearing: an open container should not vanish over a bad drop.
    $('nsp-result').querySelectorAll('.nca-inspect').forEach((node) => node.remove());
    $('nsp-result').appendChild(el('div', 'nca-info nca-inspect', verdict.why));
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
    // Without clearing: an open container should not vanish over a bad drop.
    $('nsp-result').querySelectorAll('.nca-inspect').forEach((node) => node.remove());
    $('nsp-result').appendChild(el('div', 'nca-info nca-inspect', why));
    log(why, 'error');
    return null;
  }
}

// Updates and add-on content are paired with their title rather than opened.
interface HeldUpdate {
  file: File;
  titleId: string;
  version: string;
}

let heldUpdate: HeldUpdate | null = null;

interface HeldDlc {
  file: File;
  titleId: string;
  indices: number[];
}

let heldDlc: HeldDlc[] = [];

// Empty until the open container's Control NCA has been read.
let openTitleId = '';

// What each title was last launched with; the files themselves do not survive a reload.
const UPDATE_MEMORY = 'switch-wasm:updates';
const DLC_MEMORY = 'switch-wasm:dlc';

interface RememberedUpdate { name: string; version: string }

function rememberedUpdates(): Record<string, RememberedUpdate> {
  try {
    return JSON.parse(localStorage.getItem(UPDATE_MEMORY) || '{}') as Record<string, RememberedUpdate>;
  } catch {
    return {};
  }
}

function rememberUpdate(u: HeldUpdate): void {
  const all = rememberedUpdates();
  all[u.titleId] = { name: u.file.name, version: u.version };
  try {
    localStorage.setItem(UPDATE_MEMORY, JSON.stringify(all));
  } catch {
    // A full or blocked store only loses the reminder.
  }
}

function rememberedDlc(): Record<string, string[]> {
  try {
    return JSON.parse(localStorage.getItem(DLC_MEMORY) || '{}') as Record<string, string[]>;
  } catch {
    return {};
  }
}

function rememberDlc(titleId: string, name: string): void {
  const all = rememberedDlc();
  const names = new Set(all[titleId] || []);
  names.add(name);
  all[titleId] = [...names];
  try {
    localStorage.setItem(DLC_MEMORY, JSON.stringify(all));
  } catch {
    // A full or blocked store only loses the reminder.
  }
}

// `explain` says what happens to the file next; a launch that holds it says that itself.
function handleDlcFile(file: File, packs: DlcEntry[], explain = true): void {
  const titleId = packs[0]?.title_id ?? '';
  heldDlc = heldDlc.filter((held) => held.file.name !== file.name);
  heldDlc.push({ file, titleId, indices: packs.map((e) => e.index) });
  rememberDlc(titleId, file.name);
  // The core has logged what the file holds; this is what happens to it.
  if (explain) log(titleId === openTitleId
    ? 'The add-on content is mounted when this title launches.'
    : 'Open that game to launch it with the add-on content.', 'info');
  showPairedNotes();
}

// Hold each of `files`, updates and add-on content the caller has checked, for the next launch.
export async function holdForLaunch(files: File[]): Promise<void> {
  for (const file of files) {
    const opened = await openFile(file);
    if (opened?.kind === 'update') handleUpdateFile(file, opened, false);
    else if (opened?.kind === 'dlc') handleDlcFile(file, opened.packs, false);
  }
}

function describeDlc(packs: number): string {
  return packs === 1 ? '1 add-on content pack' : packs + ' add-on content packs';
}

function handleUpdateFile(file: File, { title: titleId, version }: UpdateInfo, explain = true): void {
  heldUpdate = { file, titleId, version };
  rememberUpdate(heldUpdate);
  // The core has logged what the file holds; this is what happens to it.
  if (explain) log(titleId === openTitleId
    ? 'The update is applied when this title launches.'
    : 'Open that game to launch it patched.', 'info');
  showPairedNotes();
}

function describeUpdate(u: HeldUpdate | RememberedUpdate & { file?: File }): string {
  const version = 'version' in u && u.version ? 'v' + u.version : '';
  const name = 'file' in u && u.file ? u.file.name : (u as RememberedUpdate).name;
  return version || name;
}

// Rendered under the card, not inside it, so rebuilding the card clears them.
function showPairedNotes(): void {
  $('nsp-result').querySelectorAll('.paired-note').forEach((node) => node.remove());
  let after: Element | null = $('nsp-result').querySelector('.title-card');
  for (const line of pairedNotes()) {
    const note = el('div', 'nca-info paired-note', line);
    if (after) after.after(note);
    else $('nsp-result').prepend(note);
    after = note;
  }
}

function pairedNotes(): string[] {
  const lines: string[] = [];
  if (heldUpdate) {
    lines.push(heldUpdate.titleId === openTitleId
      ? 'Update ' + describeUpdate(heldUpdate) + ' - applied when this title launches.'
      : 'Update ' + describeUpdate(heldUpdate) + ' held for ' + heldUpdate.titleId
        + ' - open that game to launch it patched.');
  } else {
    const known = openTitleId ? rememberedUpdates()[openTitleId] : undefined;
    if (known) {
      lines.push('This title was last launched with update ' + describeUpdate(known)
        + ' (' + known.name + '). Drop that file again to launch it patched.');
    }
  }

  const mine = heldDlc.filter((held) => held.titleId === openTitleId);
  const packs = mine.reduce((n, held) => n + held.indices.length, 0);
  if (packs) {
    lines.push(describeDlc(packs) + ' - mounted when this title launches ('
      + mine.map((held) => held.file.name).join(', ') + ').');
  }
  for (const held of heldDlc.filter((h) => h.titleId !== openTitleId)) {
    lines.push(describeDlc(held.indices.length) + ' held for ' + held.titleId
      + ' - open that game to launch it with them.');
  }
  if (!packs && openTitleId) {
    const known = rememberedDlc()[openTitleId] || [];
    const missing = known.filter((name) => !heldDlc.some((h) => h.file.name === name));
    if (missing.length) {
      lines.push('This title was last launched with add-on content ('
        + missing.join(', ') + '). Drop those files again to launch it with them.');
    }
  }
  return lines;
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
    handleDlcFile(file, opened.packs);
    const held = heldDlc[heldDlc.length - 1];
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

function renderTitleCard(info: ControlInfo, iconUrl: string | null): HTMLElement {
  const card = el('div', 'title-card');
  if (iconUrl) {
    const img = el('img', 'title-icon');
    img.alt = info.name;
    img.src = iconUrl;
    card.appendChild(img);
  }
  const meta = el('div', 'title-meta');
  meta.appendChild(el('div', 'title-name', info.name));
  if (info.publisher) meta.appendChild(el('div', 'title-publisher', info.publisher));
  const tags = [];
  if (info.version) tags.push('v' + info.version);
  if (info.demo) tags.push('demo');
  tags.push(info.title_id);
  meta.appendChild(el('div', 'title-tags', tags.join(' · ')));
  card.appendChild(meta);

  const details = el('div', 'nca-info');
  appendRows(details, titleRows(info));
  card.appendChild(details);
  return card;
}

function titleRows(info: ControlInfo): [string, string][] {
  const rows: [string, string][] = [];
  const push = (k: string, v: string | undefined) => {
    if (v) rows.push([k, v]);
  };
  push('Language', info.language);
  push('Localized', (info.languages || []).join(', '));
  push('Age rating', (info.ratings || []).map((r) => r.organisation + ' ' + r.age).join(', '));
  push('User account', info.startup_user_account);
  push('Screenshots', info.screenshot);
  push('Video capture', info.video_capture);
  push('Save data', saveDataSummary(info));
  if (info.add_on_content_base_id && !/^0+$/.test(info.add_on_content_base_id)) {
    push('DLC base id', info.add_on_content_base_id);
  }
  if (info.save_data_owner_id && info.save_data_owner_id !== info.title_id
    && !/^0+$/.test(info.save_data_owner_id)) {
    push('Save data owner', info.save_data_owner_id);
  }
  push('Error codes', info.error_code_category);
  push('ISBN', info.isbn);
  return rows;
}

function saveDataSummary(info: ControlInfo): string {
  const part = (label: string, size = 0, journal = 0) => {
    if (!size && !journal) return null;
    const journalNote = journal ? ' (+' + fmtSize(journal) + ' journal)' : '';
    return label + ' ' + fmtSize(size) + journalNote;
  };
  return [
    part('user', info.user_save_size, info.user_save_journal_size),
    part('device', info.device_save_size, info.device_save_journal_size),
    part('BCAT', info.bcat_storage_size, 0),
  ].filter(Boolean).join(', ');
}

function appendRows(out: HTMLElement, rows: [string, string][]): void {
  for (const [k, v] of rows) {
    const row = el('div');
    row.appendChild(el('span', 'k', k + ':'));
    row.append(' ' + v);
    out.appendChild(row);
  }
}

async function inspectNca(f: NspFile, index: number): Promise<void> {
  // Matched on `.nca-inspect`: the title card has an `.nca-info` block too.
  $('nsp-result').querySelectorAll('.nca-inspect').forEach((node) => node.remove());
  const out = el('div', 'nca-info nca-inspect', 'Parsing ' + f.name + ' ...');
  $('nsp-result').appendChild(out);
  await parseAndRenderNca(out, index, () => launchNca(f.name, index));
}

// Drop/browse a standalone .nca, which becomes the open container.
async function handleStandaloneNca(file: File): Promise<NcaInfo | null> {
  clearNsp();
  setNote('container-badge', 'opening ' + file.name, false);
  const out = el('div', 'nca-info nca-inspect', 'Parsing ' + file.name + ' ...');
  $('nsp-result').appendChild(out);
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
  if (heldUpdate?.titleId === openTitleId) {
    notes.push('Update ' + describeUpdate(heldUpdate)
      + ' applied: its modules, over the base game\'s data.');
    // The update's version is what runs.
    if (identity && heldUpdate.version) identity = { ...identity, version: heldUpdate.version };
  }
  const packs = heldDlc
    .filter((held) => held.titleId === openTitleId)
    .reduce((n, held) => n + held.indices.length, 0);
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
