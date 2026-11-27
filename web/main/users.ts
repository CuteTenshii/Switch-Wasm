// User profiles and the picker the top bar opens.

import { Fingerprint, ImageOff, Pencil, Trash2, UserCheck, type IconNode } from 'lucide';
import type { Bytes, UserRecord } from '../shared/protocol';
import { $, el, pickedFile } from './dom';
import { icon } from './icons';
import { log } from './log';
import { call, hasSession } from './rpc';
import { titleBooted } from './session';

interface Profile extends UserRecord {
  created: number;
}

const MAX_PROFILES = 8;

const NICKNAME_MAX = 10;

const PICTURE_SIZE = 256;

// Holds the playing profile's uid, in the same store as the profiles.
const CURRENT_KEY = '@current';

// Must match `PROFILE_IMAGE_COLORS` in `acc.rs`.
const PICTURE_COLORS = [
  '#4b505a', '#2f6fb5', '#c04a3c', '#3e8e5a', '#b07a1e', '#7a4fa8', '#1f8a8c', '#b44c86',
];

function pictureColor(uid: string): string {
  let spread = 0;
  for (let i = 0; i < uid.length; i += 2) {
    spread = (Math.imul(spread, 31) + Number.parseInt(uid.slice(i, i + 2), 16)) >>> 0;
  }
  return PICTURE_COLORS[spread % PICTURE_COLORS.length];
}

let profiles: Profile[] = [];
let current = '';

const pictureUrls = new Map<string, { picture: Bytes; url: string }>();

function playing(): Profile {
  return profiles.find((p) => p.uid === current) ?? profiles[0];
}

// Never all zero: titles read that as "nobody".
function newUid(): string {
  const bytes = new Uint8Array(16);
  do crypto.getRandomValues(bytes); while (bytes.every((b) => b === 0));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function toRecord({ uid, nickname, editedAt, picture }: Profile): UserRecord {
  return { uid, nickname, editedAt, picture };
}

// At most `NICKNAME_MAX` characters, counted as the console does, by code point.
function clampNickname(name: string): string {
  return [...name.trim()].slice(0, NICKNAME_MAX).join('');
}

function newProfile(nickname: string, editedAt: number): Profile {
  return { uid: newUid(), nickname, editedAt, picture: null, created: Date.now() };
}

function firstProfile(): Profile {
  return newProfile('Player', 0);
}

async function readProfiles(): Promise<void> {
  profiles = [];
  for (const [key, value] of await call('usersRead')) {
    if (key === CURRENT_KEY) current = String(value);
    else profiles.push(value as Profile);
  }
  profiles.sort((a, b) => a.created - b.created);
  if (!profiles.length) {
    const first = firstProfile();
    profiles.push(first);
    await call('usersStore', [[first.uid, first]]);
  }
  if (!profiles.some((p) => p.uid === current)) current = profiles[0].uid;
}

export async function loadProfiles(): Promise<void> {
  try {
    await readProfiles();
  } catch (err) {
    log('Profiles: could not be read (' + (err as Error).message + ') - playing as "Player".', 'error');
    profiles = [firstProfile()];
    current = profiles[0].uid;
  }
  renderProfiles();
}

// Picks up what the worker stored for the guest's profile edits.
export async function reloadProfiles(): Promise<void> {
  try {
    await readProfiles();
  } catch (err) {
    log('Profiles: the guest\'s edits could not be read (' + (err as Error).message + ')', 'error');
  }
  renderProfiles();
}

export async function stageUsers(): Promise<void> {
  if (!hasSession()) return;
  // A refused list is logged by the core with its reason.
  await call('usersSet', profiles.map(toRecord), current);
}

// A running title keeps the user it started with; an idle session gets changes now.
async function stageIfIdle(): Promise<void> {
  if (!titleBooted()) await stageUsers();
}

// Reloads the profiles if the worker stored edits the guest made through `acc`.
export async function pullProfileEdits(): Promise<void> {
  if (hasSession() && await call('storeProfileEdits')) await reloadProfiles();
}

// Center-crops to the 256x256 JPEG titles expect.
async function toPicture(file: File): Promise<Bytes> {
  const bitmap = await createImageBitmap(file);
  const side = Math.min(bitmap.width, bitmap.height);
  const canvas = el('canvas');
  canvas.width = PICTURE_SIZE;
  canvas.height = PICTURE_SIZE;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('this browser has no 2d canvas context');
  ctx.drawImage(
    bitmap,
    (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side,
    0, 0, PICTURE_SIZE, PICTURE_SIZE,
  );
  bitmap.close();
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
  if (!blob) throw new Error('the picture could not be encoded');
  return new Uint8Array(await blob.arrayBuffer());
}

function now(): number {
  return Math.floor(Date.now() / 1000);
}

const buttonEl = $<HTMLButtonElement>('btn-profile');
const dialogEl = $<HTMLDialogElement>('profiles');
const listEl = $('profile-list');
const addForm = $<HTMLFormElement>('profile-add');
const addName = $<HTMLInputElement>('profile-add-name');
const addSubmit = $<HTMLButtonElement>('profile-add-submit');
const addNote = $('profile-add-note');
const runningNote = $('profile-running-note');
addName.maxLength = NICKNAME_MAX;

// The one row being renamed, given a new UID, or asked to confirm its deletion.
type Mode = { kind: 'rename' | 'uid' | 'confirm'; uid: string } | null;
let mode: Mode = null;

function setMode(next: Mode): void {
  mode = next;
  renderProfiles();
}

function button(label: string, className: string, glyph?: IconNode): HTMLButtonElement {
  const b = el('button', className);
  b.type = 'button';
  if (glyph) b.append(icon(glyph));
  b.append(label);
  return b;
}

function avatar(profile: Profile, className: string): HTMLElement {
  if (profile.picture) {
    let entry = pictureUrls.get(profile.uid);
    if (!entry || entry.picture !== profile.picture) {
      if (entry) URL.revokeObjectURL(entry.url);
      entry = { picture: profile.picture, url: URL.createObjectURL(new Blob([profile.picture], { type: 'image/jpeg' })) };
      pictureUrls.set(profile.uid, entry);
    }
    const img = el('img', className);
    img.src = entry.url;
    img.alt = '';
    return img;
  }
  const blank = el('span', className, [...profile.nickname][0]?.toUpperCase() ?? '');
  blank.style.background = pictureColor(profile.uid);
  blank.setAttribute('aria-hidden', 'true');
  return blank;
}

function renderButton(): void {
  const who = playing();
  buttonEl.replaceChildren(avatar(who, 'profile-avatar small'), el('span', 'profile-button-name', who.nickname));
  buttonEl.title = `Playing as ${who.nickname}. Choose the profile`;
}

function renderProfiles(): void {
  if (!profiles.length) return;
  renderButton();
  const rows = profiles.map(renderRow);
  listEl.replaceChildren(...rows);
  const full = profiles.length >= MAX_PROFILES;
  addName.disabled = full;
  addSubmit.disabled = full;
  addNote.hidden = !full;
  runningNote.hidden = !titleBooted();
}

function renderRow(profile: Profile): HTMLElement {
  const row = el('li', 'profile-row');
  const isPlaying = profile.uid === current;
  row.classList.toggle('is-playing', isPlaying);

  const picture = el('label', 'profile-picture');
  picture.title = `Choose a picture for ${profile.nickname}`;
  const input = el('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.hidden = true;
  input.addEventListener('change', (e) => {
    const file = pickedFile(e);
    if (file) void setPicture(profile, file);
  });
  picture.append(avatar(profile, 'profile-avatar'), input);
  picture.tabIndex = 0;
  picture.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      input.click();
    }
  });

  const own = mode?.uid === profile.uid ? mode.kind : null;
  const main = el('div', 'profile-main');
  const actions = el('div', 'profile-actions');
  if (own === 'rename') {
    main.append(renameForm(profile));
  } else if (own === 'uid') {
    main.append(uidForm(profile));
  } else if (own === 'confirm') {
    row.classList.add('is-confirming');
    main.append(el('span', 'profile-confirm', `Delete ${profile.nickname} and their saves?`));
    const yes = button('Delete', 'btn small danger-solid');
    yes.addEventListener('click', () => void removeProfile(profile));
    const no = button('Keep', 'btn small ghost');
    no.addEventListener('click', () => setMode(null));
    actions.append(yes, no);
    queueMicrotask(() => no.focus());
  } else {
    main.append(el('span', 'profile-name', profile.nickname));
    if (isPlaying) {
      main.append(el('span', 'profile-playing', 'Playing'));
    } else {
      const play = button('Play as', 'btn small', UserCheck);
      play.addEventListener('click', () => void choose(profile));
      actions.append(play);
    }
    const rename = button('Rename', 'btn small ghost', Pencil);
    rename.addEventListener('click', () => setMode({ kind: 'rename', uid: profile.uid }));
    actions.append(rename);
    // The running title committed the user list, and its saves sit in the core under these UIDs.
    const editUid = button('Edit UID', 'btn small ghost', Fingerprint);
    editUid.disabled = titleBooted();
    if (editUid.disabled) editUid.title = 'Stop the title to change a UID';
    editUid.addEventListener('click', () => setMode({ kind: 'uid', uid: profile.uid }));
    actions.append(editUid);
    if (profile.picture) {
      const clear = button('Remove picture', 'btn small ghost', ImageOff);
      clear.addEventListener('click', () => void clearPicture(profile));
      actions.append(clear);
    }
    // The last profile and the one a running title uses can't be deleted.
    const locked = isPlaying && titleBooted();
    if (profiles.length > 1) {
      const del = button('Delete', 'btn small ghost danger', Trash2);
      del.disabled = locked;
      if (locked) del.title = 'Stop the title to delete the profile the running title is playing as';
      del.addEventListener('click', () => setMode({ kind: 'confirm', uid: profile.uid }));
      actions.append(del);
    }
  }
  row.append(picture, main, actions);
  if (!own) {
    const uid = el('span', 'profile-uid', profile.uid);
    uid.title = 'UID';
    row.append(uid);
  }
  return row;
}

// A row's form of `input` with Save and Cancel; Escape cancels too, and the input takes the focus.
function inlineForm(
  className: string,
  input: HTMLInputElement,
  onSubmit: (save: HTMLButtonElement) => void,
): HTMLFormElement {
  const form = el('form', className);
  input.required = true;
  const save = el('button', 'btn small', 'Save');
  save.type = 'submit';
  const cancel = button('Cancel', 'btn small ghost');
  cancel.addEventListener('click', () => setMode(null));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      // The dialog would close on it too.
      e.preventDefault();
      e.stopPropagation();
      setMode(null);
    }
  });
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    onSubmit(save);
  });
  form.append(input, save, cancel);
  queueMicrotask(() => {
    input.focus();
    input.select();
  });
  return form;
}

function renameForm(profile: Profile): HTMLFormElement {
  const input = el('input', 'text');
  input.value = profile.nickname;
  input.maxLength = NICKNAME_MAX;
  input.setAttribute('aria-label', `New name for ${profile.nickname}`);
  return inlineForm('profile-rename', input, () => void rename(profile, input.value));
}

function uidForm(profile: Profile): HTMLFormElement {
  const input = el('input', 'text mono');
  input.value = profile.uid;
  input.maxLength = 32;
  input.spellcheck = false;
  input.autocomplete = 'off';
  input.setAttribute('aria-label', `UID for ${profile.nickname}`);
  const error = el('span', 'profile-error');
  error.id = `uid-error-${profile.uid}`;
  error.setAttribute('role', 'alert');
  error.hidden = true;
  input.setAttribute('aria-describedby', error.id);
  input.addEventListener('input', () => {
    error.hidden = true;
    input.removeAttribute('aria-invalid');
  });
  const form = inlineForm('profile-rename profile-uid-form', input, (save) => {
    save.disabled = true;
    void changeUid(profile, input.value).then((why) => {
      save.disabled = false;
      if (!why) return;
      error.textContent = why;
      error.hidden = false;
      input.setAttribute('aria-invalid', 'true');
      input.focus();
    });
  });
  form.append(error);
  return form;
}

async function update(profile: Profile): Promise<void> {
  profile.editedAt = now();
  try {
    await call('usersStore', [[profile.uid, profile]]);
  } catch (err) {
    log('Profiles: could not be saved (' + (err as Error).message + ')', 'error');
  }
  renderProfiles();
  await stageIfIdle();
}

async function choose(profile: Profile): Promise<void> {
  current = profile.uid;
  try {
    await call('usersStore', [[CURRENT_KEY, current]]);
  } catch (err) {
    log('Profiles: the choice could not be saved (' + (err as Error).message + ')', 'error');
  }
  log(titleBooted()
    ? `Profiles: ${profile.nickname} plays the next title; the running one keeps its player.`
    : `Profiles: playing as ${profile.nickname}.`);
  renderProfiles();
  await stageIfIdle();
}

async function rename(profile: Profile, name: string): Promise<void> {
  const nickname = clampNickname(name);
  if (!nickname) return;
  profile.nickname = nickname;
  mode = null;
  await update(profile);
}

// Moves the profile's saves along with it; resolves to why the UID was refused, or null.
async function changeUid(profile: Profile, text: string): Promise<string | null> {
  const uid = text.trim().toLowerCase();
  if (uid === profile.uid) {
    setMode(null);
    return null;
  }
  if (!/^[0-9a-f]{32}$/.test(uid)) return 'A UID is 32 hex digits, 0-9 and a-f.';
  if (/^0+$/.test(uid)) return 'A UID can\'t be all zeros: titles read that as no user.';
  const owner = profiles.find((p) => p.uid === uid);
  if (owner) return `${owner.nickname} already has this UID.`;
  if (titleBooted()) return 'Stop the title to change a UID.';

  let moved: number;
  try {
    moved = await call('savesMoveUser', profile.uid, uid);
  } catch (err) {
    return 'The saves could not be moved: ' + (err as Error).message;
  }
  const old = profile.uid;
  const picture = pictureUrls.get(old);
  if (picture) {
    pictureUrls.delete(old);
    pictureUrls.set(uid, picture);
  }
  profile.uid = uid;
  profile.editedAt = now();
  if (current === old) current = uid;
  mode = null;
  try {
    await call('usersStore', [[old, null], [uid, profile], [CURRENT_KEY, current]]);
  } catch (err) {
    log('Profiles: the new UID could not be saved (' + (err as Error).message + ')', 'error');
  }
  log(`Profiles: ${profile.nickname} now has UID ${uid}` + (moved ? `, with ${moved} saved entries` : '') + '.');
  renderProfiles();
  await stageIfIdle();
  return null;
}

async function setPicture(profile: Profile, file: File): Promise<void> {
  try {
    profile.picture = await toPicture(file);
  } catch (err) {
    log(`Profiles: ${file.name} could not be used as a picture (${(err as Error).message})`, 'error');
    return;
  }
  await update(profile);
}

async function clearPicture(profile: Profile): Promise<void> {
  profile.picture = null;
  await update(profile);
}

async function removeProfile(profile: Profile): Promise<void> {
  mode = null;
  profiles = profiles.filter((p) => p.uid !== profile.uid);
  const entry = pictureUrls.get(profile.uid);
  if (entry) URL.revokeObjectURL(entry.url);
  pictureUrls.delete(profile.uid);
  if (current === profile.uid) current = profiles[0].uid;
  try {
    await call('usersStore', [[profile.uid, null], [CURRENT_KEY, current]]);
    const removed = await call('savesDeleteUser', profile.uid);
    log(`Profiles: deleted ${profile.nickname}` + (removed ? ` and ${removed} saved entries` : ''));
  } catch (err) {
    log('Profiles: the deletion could not be finished (' + (err as Error).message + ')', 'error');
  }
  renderProfiles();
  await stageIfIdle();
  buttonEl.focus();
}

addForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const nickname = clampNickname(addName.value);
  if (!nickname || profiles.length >= MAX_PROFILES) return;
  const profile = newProfile(nickname, now());
  profiles.push(profile);
  addName.value = '';
  void update(profile);
});

buttonEl.addEventListener('click', () => {
  setMode(null);
  dialogEl.showModal();
});
$('btn-profiles-close').addEventListener('click', () => dialogEl.close());
// Backdrop clicks target the dialog element itself.
dialogEl.addEventListener('click', (e) => {
  if (e.target === dialogEl) dialogEl.close();
});
dialogEl.addEventListener('close', () => {
  mode = null;
});
