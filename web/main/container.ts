// NSP / NCA / XCI containers: inspecting, showing the title, launching.

import type { Bytes, ControlInfo, DlcEntry, NcaInfo, NspFile } from '../shared/protocol';
import { controlRemember } from './controls';
import { $, el, pickedFile } from './dom';
import { classify } from './filetype';
import { fmtSize } from './format';
import { awaitFirstFrame, beginLoad, failLoad, loadPhase } from './loading';
import { clearConsole, log } from './log';
import { call, readLastError } from './rpc';
import { run, updatePc } from './runloop';
import { noteBooted, recycleSession } from './session';
import { openPanel, setNote, setState, showScreen } from './shell';
import { setRunning } from './title';

// What a launch puts on the loading screen.
export interface LaunchIdentity {
  name: string;
  publisher: string;
  titleId: string;
  iconUrl: string | null;
  // Kept so the top bar can make its own URL after this one is revoked.
  icon: Blob | null;
  // The NACP display version ("1.0.1"); may be empty.
  version: string;
}

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
    log(verdict.why, 'err');
    return;
  }
  if (verdict.format === 'nca') {
    await handleStandaloneNca(file);
    return;
  }
  // An update holds no game; inspecting it also names the title it patches.
  const patches = await call('add_update', file).catch(() => '');
  if (patches) {
    await handleUpdateFile(file, patches);
    return;
  }
  const packs = await call('add_dlc', file).catch(() => 0);
  if (packs > 0) {
    await handleDlcFile(file, packs);
    return;
  }
  await handleNspFile(file);
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

async function handleDlcFile(file: File, packs: number): Promise<void> {
  const entries = JSON.parse(await call('dlc_json').catch(() => '[]')) as DlcEntry[];
  // This container's pieces are the newest ones the session holds.
  const mine = entries.slice(-packs);
  const titleId = mine[0]?.title_id ?? '';
  heldDlc = heldDlc.filter((held) => held.file.name !== file.name);
  heldDlc.push({ file, titleId, indices: mine.map((e) => e.index) });
  rememberDlc(titleId, file.name);
  log(describeDlc(packs) + ' for ' + titleId + ' - '
    + (titleId === openTitleId
      ? 'mounted when this title launches.'
      : 'open that game to launch it with them.'), 'ok');
  showPairedNotes();
}

function describeDlc(packs: number): string {
  return packs === 1 ? '1 add-on content pack' : packs + ' add-on content packs';
}

async function handleUpdateFile(file: File, titleId: string): Promise<void> {
  const version = await call('update_version').catch(() => '');
  heldUpdate = { file, titleId, version };
  rememberUpdate(heldUpdate);
  log('Update ' + describeUpdate(heldUpdate) + ' for ' + titleId + ' - '
    + (titleId === openTitleId
      ? 'applied when this title launches.'
      : 'open that game to launch it patched.'), 'ok');
  await syncPaired();
}

function describeUpdate(u: HeldUpdate | RememberedUpdate & { file?: File }): string {
  const version = 'version' in u && u.version ? 'v' + u.version : '';
  const name = 'file' in u && u.file ? u.file.name : (u as RememberedUpdate).name;
  return version || name;
}

// The session refuses an update for another title, so it is handed over only
// while its title is open. Add-on content is always handed over.
async function syncPaired(): Promise<void> {
  if (heldUpdate && heldUpdate.titleId === openTitleId) {
    await call('add_update', heldUpdate.file).catch(() => '');
  } else {
    await call('clear_update').catch(() => 0);
  }
  await call('clear_dlc').catch(() => 0);
  for (const held of heldDlc) {
    await call('add_dlc', held.file).catch(() => 0);
  }
  showPairedNotes();
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
  setState('loading');
  beginLoad(file.name, 'opening the container (' + fmtSize(file.size) + ')');
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
      log(why, 'err');
      return;
    }
    await launchStandaloneNca(file);
    return;
  }
  // An update or add-on content means: pair it with the game and launch that.
  const patches = await call('add_update', file).catch(() => '');
  if (patches) {
    await handleUpdateFile(file, patches);
    setState('idle');
    failLoad(patches === openTitleId
      ? 'Update applied - launch ' + (heldTitle?.name || 'the game') + ' to run it patched.'
      : 'That is an update for ' + patches + ', not a game. Open that title to launch it patched.');
    openPanel('files');
    return;
  }
  const packs = await call('add_dlc', file).catch(() => 0);
  if (packs > 0) {
    await handleDlcFile(file, packs);
    const held = heldDlc[heldDlc.length - 1];
    setState('idle');
    failLoad(held.titleId === openTitleId
      ? describeDlc(packs) + ' added - launch ' + (heldTitle?.name || 'the game')
      + ' to run with them.'
      : 'That is add-on content for ' + held.titleId + ', not a game. Open that title'
        + ' to launch it with them.');
    openPanel('files');
    return;
  }
  await handleNspFile(file);
  // A failed open leaves `openContainer` unchanged, so compare by identity.
  if (openContainer?.file !== file) {
    setState('fault');
    failLoad('Could not open ' + file.name + '. The Files panel has the details.');
    openPanel('files');
    return;
  }
  loadPhase('looking for the title\'s program');
  const index = await call('program_nca_index');
  if (index < 0 || !nspFiles[index]) {
    const why = await readLastError();
    setState('fault');
    failLoad(why);
    log('Nothing to boot in ' + file.name + ': ' + why, 'err');
    return;
  }
  await launchNca(nspFiles[index], index);
}

let openContainer: { file: File; kind: 'nsp' | 'nca' } | null = null;

let nspFiles: NspFile[] = [];

// Give a fresh session the container the page is still showing.
export async function reopenContainer(): Promise<void> {
  if (!openContainer) return;
  const { file, kind } = openContainer;
  const ok = await call(kind === 'nca' ? 'open_nca' : 'open_nsp', file).catch(() => -1);
  if (ok !== 0) {
    log('Could not re-open ' + file.name + ' - load it again to launch it.', 'err');
    openContainer = null;
    clearNsp();
    return;
  }
  await syncPaired();
}

// The File goes to the worker, which reads ranges from it; only its header is read.
async function handleNspFile(file: File): Promise<void> {
  clearNsp();
  setNote('container-badge', 'opening ' + file.name, false);
  const status = el('div', 'nca-info', 'Reading the container header \u2026');
  $('nsp-result').appendChild(status);
  log('Opening ' + file.name + ' (' + fmtSize(file.size) + ') ...');
  try {
    const ok = await call('open_nsp', file);
    if (ok !== 0) {
      const why = await readLastError();
      status.textContent = 'NSP error: ' + why;
      setNote('container-badge', 'none open', false);
      log('NSP error: ' + why, 'err');
      return;
    }
    openContainer = { file, kind: 'nsp' };
    setNote('container-badge', file.name, true);
  } catch (e) {
    status.textContent = 'Could not open ' + file.name + ': ' + (e as Error).message;
    setNote('container-badge', 'none open', false);
    log('Could not open ' + file.name + ': ' + (e as Error).message, 'err');
    return;
  }
  status.remove();
  nspFiles = JSON.parse(await call('nsp_files_json')) as NspFile[];
  log('Parsed ' + nspFiles.length + ' file(s). Click an .nca to inspect it.', 'ok');

  const ul = el('ul', 'nsp-list');
  nspFiles.forEach((f, index) => {
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
  await showTitleCard(() => call('load_control_from_nsp'));
  setNote('container-badge', file.name, true);
}

export function clearNsp(): void {
  $('nsp-result').textContent = '';
  setNote('container-badge', 'none open', false);
  nspFiles = [];
  // The held update outlives the container; the pairing does not.
  openTitleId = '';
  holdTitle(null, null);
}

// Exactly one icon URL is alive at a time; replacing the identity revokes it.
let heldTitle: LaunchIdentity | null = null;

function holdTitle(info: ControlInfo | null, icon: Bytes | null): LaunchIdentity | null {
  if (heldTitle?.iconUrl) URL.revokeObjectURL(heldTitle.iconUrl);
  const blob = info && icon && icon.length
    ? new Blob([icon], { type: info.icon_mime })
    : null;
  heldTitle = info
    ? {
        name: info.name,
        publisher: info.publisher || '',
        titleId: info.title_id,
        iconUrl: blob ? URL.createObjectURL(blob) : null,
        icon: blob,
        version: info.version || '',
      }
    : null;
  return heldTitle;
}

// Title card: icon, name, publisher and NACP details from the Control NCA.
// Needs prod.keys, since the content type is in the encrypted header.
async function showTitleCard(loader: () => Promise<number>): Promise<ControlInfo | null> {
  let info: ControlInfo;
  try {
    if (await loader() !== 0) {
      log('No title details: ' + await readLastError(), 'dim');
      return null;
    }
    info = JSON.parse(await call('control_json')) as ControlInfo;
  } catch (err) {
    log('No title details: ' + (err as Error).message, 'dim');
    return null;
  }
  if (!info.name) return null;
  const icon = info.icon_size > 0 ? await call('control_icon', info.icon_size) : null;
  if (!/^0*$/.test(info.title_id)) await controlRemember(info.title_id, icon);
  const card = renderTitleCard(info, holdTitle(info, icon)?.iconUrl ?? null);
  $('nsp-result').prepend(card);
  log('Title: ' + info.name + (info.publisher ? ' - ' + info.publisher : ''), 'ok');
  openTitleId = info.title_id;
  await syncPaired();
  return info;
}

function renderTitleCard(info: ControlInfo, iconUrl: string | null): HTMLElement {
  const card = el('div', 'title-card');
  if (iconUrl) {
    const img = el('img', 'title-icon');
    img.alt = info.name;
    // The URL belongs to `heldTitle`, which revokes it.
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

  // 0xC00 covers the base header and all four FS headers.
  const headerLen = Math.min(f.size, 0xC00);
  let header: Bytes;
  try {
    header = await call('read_file', index, 0, headerLen);
  } catch (err) {
    out.textContent = 'read failed: ' + (err as Error).message;
    return;
  }
  await parseAndRenderNca(out, header, () => launchNca(f, index));
}

// Drop/browse a standalone .nca, which becomes the open container.
async function handleStandaloneNca(file: File): Promise<NcaInfo | null> {
  clearNsp();
  setNote('container-badge', 'opening ' + file.name, false);
  const out = el('div', 'nca-info nca-inspect', 'Parsing ' + file.name + ' ...');
  $('nsp-result').appendChild(out);
  try {
    await call('open_nca', file);
  } catch (e) {
    out.textContent = 'Could not open ' + file.name + ': ' + (e as Error).message;
    setNote('container-badge', 'none open', false);
    return null;
  }
  openContainer = { file, kind: 'nca' };
  setNote('container-badge', file.name, true);
  const headerLen = Math.min(file.size, 0xC00);
  const header = new Uint8Array(await file.slice(0, headerLen).arrayBuffer());
  const info = await parseAndRenderNca(out, header, () => launchStandaloneNca(file));
  if (info && info.content_type === 'Control') {
    setNote('container-badge', 'reading title details\u2026', false);
    await showTitleCard(() => call('load_control_from_nca'));
    setNote('container-badge', file.name, true);
  }
  return info;
}

async function parseAndRenderNca(
  out: HTMLElement,
  header: Bytes,
  onLaunch: () => void,
): Promise<NcaInfo | null> {
  let info: NcaInfo;
  try {
    info = JSON.parse(await call('parse_nca', header)) as NcaInfo;
  } catch (err) {
    out.textContent = 'parse failed: ' + (err as Error).message;
    return null;
  }
  if (info.error) {
    // A CDN NCA's header is encrypted, so the magic is hidden without keys.
    out.textContent = /bad magic/.test(info.error)
      ? 'NCA header is encrypted - load prod.keys to decrypt and inspect. (' + info.error + ')'
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

// Boot NSP file `index` as a Program NCA.
function launchNca(f: NspFile, index: number): Promise<void> {
  // The core's trace buffer is cleared by booting, so the page reports the version.
  const notes = [];
  let identity = heldTitle;
  if (heldUpdate && heldUpdate.titleId === openTitleId) {
    notes.push('Update ' + describeUpdate(heldUpdate)
      + ' applied: its modules, over the base game\'s data.');
    // The update's version is what runs.
    if (identity && heldUpdate.version) identity = { ...identity, version: heldUpdate.version };
  }
  const packs = heldDlc
    .filter((held) => held.titleId === openTitleId)
    .reduce((n, held) => n + held.indices.length, 0);
  if (packs) notes.push(describeDlc(packs) + ' mounted.');
  return doLaunchNca(f.name, () => call('load_nca_from_nsp', index), identity,
    notes.join(' '));
}

// Same as `launchNca`, for a standalone .nca that is already the open container.
function launchStandaloneNca(file: File): Promise<void> {
  return doLaunchNca(file.name, () => call('load_nca'), heldTitle);
}

// The log's first line: title, publisher, id, version and file.
function describeLaunch(file: string, identity?: LaunchIdentity | null): string {
  if (!identity) return 'Launching ' + file;
  const parts = [identity.name + (identity.publisher ? ' by ' + identity.publisher : '')];
  if (identity.titleId) parts.push(identity.titleId.toUpperCase());
  if (identity.version) parts.push('version ' + identity.version);
  const container = openContainer?.file.name;
  parts.push('from ' + (container && container !== file ? container + ', ' + file : file));
  return 'Launching ' + parts.join(', ');
}

export async function doLaunchNca(
  name: string,
  loadFn: () => Promise<number>,
  identity?: LaunchIdentity | null,
  note?: string,
): Promise<void> {
  clearConsole();
  // After the clear, so the log opens by naming the game.
  log(describeLaunch(name, identity), 'ok');
  if (note) log(note, 'ok');
  setState('loading');
  setRunning(null);
  beginLoad(identity?.name || name, 'decrypting the program and reading its ExeFS',
    identity?.iconUrl);
  // Rebuild the session with the container still open; a no-op if the boot already did.
  try {
    await recycleSession({ reopen: reopenContainer });
  } catch (err) {
    setState('fault');
    failLoad('The session could not be replaced: ' + (err as Error).message);
    log('Reset before launch failed: ' + (err as Error).message, 'err');
    return;
  }
  let entry: number;
  try {
    entry = await loadFn();
  } catch (err) {
    setState('fault');
    failLoad('Launch failed: ' + (err as Error).message);
    log('Launch failed: ' + (err as Error).message, 'err');
    return;
  }
  if (entry < 0) {
    const why = await readLastError();
    setState('fault');
    failLoad('Launch failed: ' + why);
    log('Launch failed: ' + why, 'err');
    return;
  }
  log('Launched ' + name + ' - entry 0x' + entry.toString(16).padStart(8, '0'), 'ok');
  setRunning({
    name: identity?.name || name,
    publisher: identity?.publisher || '',
    icon: identity?.icon ?? null,
    version: identity?.version || '',
  });
  noteBooted();
  setState('loaded');
  loadPhase('starting the process');
  showScreen();
  awaitFirstFrame();
  await updatePc();
  await run();
}
