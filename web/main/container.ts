/* ---------- NSP / NCA containers ----------

   Inspecting what a container holds, showing the title it describes, and
   launching the program inside it. */

import type { Bytes, ControlInfo, NcaInfo, NspFile } from '../shared/protocol';
import { $, el, pickedFile } from './dom';
import { fmtSize } from './format';
import { awaitFirstFrame, beginLoad, failLoad, loadPhase } from './loading';
import { clearConsole, log } from './log';
import { call, readLastError } from './rpc';
import { run, updatePc } from './runloop';
import { openPanel, setNote, setState, showScreen } from './shell';

/** What a launch puts on the loading screen. A title picked out of a container
 *  has its own name and icon; one launched from the NAND is a bare NCA with
 *  nothing but a file name to go on. */
export interface LaunchIdentity {
  name: string;
  iconUrl: string | null;
}

const nspDrop = $('nsp-drop');
nspDrop.addEventListener('dragover', (e) => { e.preventDefault(); nspDrop.classList.add('drag'); });
nspDrop.addEventListener('dragleave', () => nspDrop.classList.remove('drag'));
nspDrop.addEventListener('drop', (e) => {
  e.preventDefault();
  nspDrop.classList.remove('drag');
  const file = e.dataTransfer?.files[0];
  if (file) handleContainerFile(file);
});
$('nsp-file').addEventListener('change', (e) => {
  const file = pickedFile(e);
  if (file) handleContainerFile(file);
});

async function handleContainerFile(file: File): Promise<void> {
  if (isStandaloneNca(file)) await handleStandaloneNca(file);
  else await handleNspFile(file);
}

// A `.nca` is one piece of content rather than a container of them, so it is
// opened as itself; everything else here is a PFS0 to look inside.
function isStandaloneNca(file: File): boolean {
  return /\.nca$/i.test(file.name);
}

/** Whether a file picked on the stage should go down the container path rather
 *  than be treated as a homebrew executable. Kept in step with the panel's own
 *  accept list, since both end up opening the same thing. */
export function isContainerFile(name: string): boolean {
  return /\.(nsp|nsz|xci|nca)$/i.test(name);
}

/* ---------- booting a container from the stage ----------

   The panel's flow is "look inside this, and maybe launch something out of
   it"; the stage's is "play this". Both open the container the same way and
   fill the same panel - what this adds is finding the title's own Program
   NCA, which someone would otherwise do by clicking down the file list
   looking for the one whose content type says Program. Every file in an NSP
   is named after its own hash, so the content type is the only thing that
   distinguishes it. */
export async function bootContainer(file: File): Promise<void> {
  setState('loading');
  beginLoad(file.name, 'opening the container (' + fmtSize(file.size) + ')');
  if (isStandaloneNca(file)) {
    const info = await handleStandaloneNca(file);
    if (!info) {
      setState('fault');
      failLoad('Could not read ' + file.name + '. The Files panel has its header.');
      // The reason is a line of the panel's own output - an encrypted header
      // with no prod.keys loaded, most often - so open it on the way past
      // rather than describing where to go and looking.
      openPanel('files');
      return;
    }
    // A Control or Meta NCA is a perfectly readable file with nothing
    // executable in it, and the panel has just shown what it does hold - so
    // say what is missing rather than letting the launch fail on an absent
    // ExeFS.
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
  await handleNspFile(file);
  // If the open failed, the panel and the log have already said why, and
  // `openContainer` is still whatever was open before this - so identity, not
  // nullness, is what tells the two apart.
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

// The container the wasm side has open, kept so a new session can be handed
// the same one. Only the File is held here; nothing is read from it.
let openContainer: { file: File; kind: 'nsp' | 'nca' } | null = null;

// The open container's file table, so booting one can name the NCA it picked
// without asking for the table a second time.
let nspFiles: NspFile[] = [];

// Give a fresh session the container the page is still showing. Reset means
// "run this again from the top", not "throw away the file I just picked" --
// the NSP/NCA card and its Launch button survive a reset either way, and a
// Launch that then reports "no container is open" is the page lying about its
// own state.
export async function reopenContainer(): Promise<void> {
  if (!openContainer) return;
  const { file, kind } = openContainer;
  const ok = await call(kind === 'nca' ? 'open_nca' : 'open_nsp', file).catch(() => -1);
  if (ok !== 0) {
    log('Could not re-open ' + file.name + ' - load it again to launch it.', 'err');
    openContainer = null;
    clearNsp();
  }
}

// The File itself is handed to the worker, not its bytes: a retail container
// is larger than anything the emulator can hold - larger, for a modern title,
// than a wasm32 module can address at all - so it stays on disk and is read a
// range at a time. Only its PFS0 header is touched here.
async function handleNspFile(file: File): Promise<void> {
  clearNsp();
  // Opening a container is the panel's work, not the stage's, so it reports
  // itself here rather than behind a loading screen over a screen that may
  // still be running something.
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
  nspFiles = JSON.parse(await call('nsp_files_json'));
  log('Parsed ' + nspFiles.length + ' file(s). Click an .nca to inspect it.', 'ok');

  const ul = el('ul', 'nsp-list');
  nspFiles.forEach((f, index) => {
    const li = el('li');
    li.appendChild(el('span', 'name', f.name));
    li.appendChild(el('span', 'size', fmtSize(f.size)));
    // Only an NCA has a header worth reading, so only those rows say - by
    // lighting up under the pointer - that clicking them does anything.
    if (/\.nca$/i.test(f.name)) {
      li.classList.add('clickable');
      li.addEventListener('click', () => inspectNca(f, index));
    }
    ul.appendChild(li);
  });
  $('nsp-result').appendChild(ul);
  // Reading the Control NCA means decrypting a section and mounting its RomFS
  // to pull an icon out of it, which is the slowest part of opening a
  // container and the one that used to pass in silence.
  setNote('container-badge', 'reading title details\u2026', false);
  await showTitleCard(() => call('load_control_from_nsp'));
  setNote('container-badge', file.name, true);
}

export function clearNsp(): void {
  $('nsp-result').textContent = '';
  setNote('container-badge', 'none open', false);
  nspFiles = [];
  holdTitle(null, null);
}

/* The title the open container describes, kept so that launching it can show
   its own name and icon rather than an NCA file name. Exactly one icon URL is
   alive at a time: replacing the identity revokes the last one, which is what
   the card's revoke-on-load did before the URL had a second reader. */
let heldTitle: LaunchIdentity | null = null;

function holdTitle(info: ControlInfo | null, icon: Bytes | null): LaunchIdentity | null {
  if (heldTitle?.iconUrl) URL.revokeObjectURL(heldTitle.iconUrl);
  heldTitle = info ? {
    name: info.name,
    iconUrl: icon && icon.length
      ? URL.createObjectURL(new Blob([icon], { type: info.icon_mime }))
      : null,
  } : null;
  return heldTitle;
}

/* ---------- title details ----------

   What a console's home menu shows for a title - its icon, name and publisher
   - plus the rest of what its NACP declares, read from the Control NCA that
   ships alongside the Program NCA in every container.

   Needs prod.keys, and not just for the RomFS: an NCA's content type lives in
   its encrypted header, so without the header key the Control NCA can't even
   be picked out of the container. A container that has none is unremarkable
   (an update or DLC package may ship without one), so this is a dim note
   rather than an error. */
async function showTitleCard(loader: () => Promise<number>): Promise<ControlInfo | null> {
  let info: ControlInfo;
  try {
    if (await loader() !== 0) {
      log('No title details: ' + await readLastError(), 'dim');
      return null;
    }
    info = JSON.parse(await call('control_json'));
  } catch (err) {
    log('No title details: ' + (err as Error).message, 'dim');
    return null;
  }
  if (!info.name) return null;
  const icon = info.icon_size > 0 ? await call('control_icon', info.icon_size) : null;
  const card = renderTitleCard(info, holdTitle(info, icon)?.iconUrl ?? null);
  $('nsp-result').prepend(card);
  log('Title: ' + info.name + (info.publisher ? ' - ' + info.publisher : ''), 'ok');
  return info;
}

function renderTitleCard(info: ControlInfo, iconUrl: string | null): HTMLElement {
  const card = el('div', 'title-card');
  if (iconUrl) {
    const img = el('img', 'title-icon');
    img.alt = info.name;
    // The URL belongs to `heldTitle`, which the loading screen reads too and
    // which revokes it when the next container replaces it.
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

/* The NACP fields worth showing, skipping the ones this title left unset -
   most titles set only a handful, and a column of zeroes says nothing. */
function titleRows(info: ControlInfo): [string, string][] {
  const rows: [string, string][] = [];
  const push = (k: string, v: string | undefined) => { if (v) rows.push([k, v]); };
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

/* The three save-data areas a title can reserve, each with a journal on top
   of it. Written as "user 16 MiB (+2 MiB journal)" so the journal doesn't
   read as a fourth, separate allocation. */
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
  // Replace any previous inspection result instead of stacking them up. Matched
  // on `.nca-inspect`, not on `.nca-info`: the title card renders its NACP rows
  // in an `.nca-info` block of its own, and clearing by that class took the
  // card's details down with the inspection above it.
  $('nsp-result').querySelectorAll('.nca-inspect').forEach((node) => node.remove());
  const out = el('div', 'nca-info nca-inspect', 'Parsing ' + f.name + ' ...');
  $('nsp-result').appendChild(out);

  // 0xC00 covers the base header plus all 4 per-section FS headers (needed
  // for an accurate fs_type in the display below) - still tiny next to the
  // (possibly hundreds-of-MB) payload, so no need to copy the whole file.
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

// Drop/browse a standalone .nca (not inside an NSP): same inspect-then-Launch
// flow, with the NCA itself as the open container instead of a file inside
// one. Opening it is what lets Launch - and the Control NCA card below - read
// from it later.
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
  // A standalone Control NCA is nothing but the title's icon and metadata, so
  // the same card the container path shows is the whole point of opening one.
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
    info = JSON.parse(await call('parse_nca', header));
  } catch (err) {
    out.textContent = 'parse failed: ' + (err as Error).message;
    return null;
  }
  if (info.error) {
    // A CDN NCA stores its header encrypted with the header key, so the NCA3
    // magic at 0x200 is invisible until it's decrypted - surface that clearly
    // instead of a bare "bad magic", and point at the keys files.
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
    // Below a rule rather than trailing the last field: launching is an action
    // taken on the header above it, not one more line of it.
    const actions = el('div', 'nca-actions');
    const btn = el('button', 'btn small primary', 'Launch');
    btn.addEventListener('click', onLaunch);
    actions.appendChild(btn);
    out.appendChild(actions);
  }
  return info;
}

// Decrypts NSP file `index` as a Program NCA and boots its ExeFS `main`
// executable. This gets a real title only as far as its own crt0 - there is
// no Horizon service surface for a full retail SDK program yet (that's a much
// larger undertaking than the homebrew this emulator otherwise runs), so
// expect it to run until the first missing service rather than reach a menu.
function launchNca(f: NspFile, index: number): Promise<void> {
  return doLaunchNca(f.name, () => call('load_nca_from_nsp', index), heldTitle);
}

// Same as `launchNca`, but for a standalone .nca file: it is already the open
// container, so there is nothing to read here that booting won't read itself.
function launchStandaloneNca(file: File): Promise<void> {
  return doLaunchNca(file.name, () => call('load_nca'), heldTitle);
}

export async function doLaunchNca(
  name: string,
  loadFn: () => Promise<number>,
  identity?: LaunchIdentity | null,
): Promise<void> {
  clearConsole();
  setState('loading');
  // A title the container named puts that name and its own icon on the screen;
  // a bare NCA off the NAND has only the file name to show.
  beginLoad(identity?.name || name, 'decrypting the program and reading its ExeFS',
    identity?.iconUrl);
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
  log('Decrypted and booted the title\'s own executable; there is no Horizon service support for retail games yet, so expect it to run until the first missing service rather than reach a menu.', 'dim');
  setState('loaded');
  loadPhase('starting the process');
  showScreen();
  awaitFirstFrame();
  await updatePc();
  await run();
}
