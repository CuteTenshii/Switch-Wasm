/* User profiles and the picker the top bar opens.

   A title asks which user is playing once, at startup, so a change here
   applies from the next title on. Profiles are stored in the NAND database
   next to the saves they own. */

import type { Bytes, UserRecord } from '../shared/protocol';
import { NAND_SAVES, NAND_USERS, nandIdb } from './db';
import { $, el, pickedFile } from './dom';
import { log } from './log';
import { call, hasSession } from './rpc';
import { titleBooted } from './session';

/** `created` orders the list. */
interface Profile extends UserRecord {
  created: number;
}

/** The console's limit. */
const MAX_PROFILES = 8;

/** The console's limit; titles lay names out for ten characters. */
const NICKNAME_MAX = 10;

const PICTURE_SIZE = 256;

/** Holds the playing profile's uid, in the same store as the profiles. */
const CURRENT_KEY = '@current';

/** Must match `PROFILE_IMAGE_COLORS` and `profile_image` in `acc.rs`, so a
 *  profile without a picture looks the same here as in a title. */
const PICTURE_COLORS = [
  '#4b505a', '#2f6fb5', '#c04a3c', '#3e8e5a', '#b07a1e', '#7a4fa8', '#1f8a8c', '#b44c86',
];

function pictureColor(uid: string): string {
  let spread = 0;
  for (let i = 0; i < uid.length; i += 2) {
    spread = (Math.imul(spread, 31) + parseInt(uid.slice(i, i + 2), 16)) >>> 0;
  }
  return PICTURE_COLORS[spread % PICTURE_COLORS.length];
}

let profiles: Profile[] = [];
let current = '';

/** Kept so re-renders reuse the URL and replaced pictures get revoked. */
const pictureUrls = new Map<string, { picture: Bytes; url: string }>();

function playing(): Profile {
  return profiles.find((p) => p.uid === current) ?? profiles[0];
}

/** Never all zero: titles read that as "nobody". */
function newUid(): string {
  const bytes = new Uint8Array(16);
  do crypto.getRandomValues(bytes); while (bytes.every((b) => b === 0));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function toRecord({ uid, nickname, editedAt, picture }: Profile): UserRecord {
  return { uid, nickname, editedAt, picture };
}

function idbRequest<T>(run: (store: IDBObjectStore) => IDBRequest<T>, mode: IDBTransactionMode): Promise<T> {
  return nandIdb().then((db) => new Promise<T>((resolve, reject) => {
    const tx = db.transaction(NAND_USERS, mode);
    const req = run(tx.objectStore(NAND_USERS));
    tx.oncomplete = () => resolve(req.result);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  }));
}

async function storeProfile(profile: Profile): Promise<void> {
  await idbRequest((s) => s.put(profile, profile.uid), 'readwrite');
}

async function storeCurrent(): Promise<void> {
  await idbRequest((s) => s.put(current, CURRENT_KEY), 'readwrite');
}

function firstProfile(): Profile {
  return { uid: newUid(), nickname: 'Player', editedAt: 0, picture: null, created: Date.now() };
}

/** Creates the first profile when there are none. */
export async function loadProfiles(): Promise<void> {
  try {
    const db = await nandIdb();
    const [keys, values] = await new Promise<[IDBValidKey[], unknown[]]>((resolve, reject) => {
      const tx = db.transaction(NAND_USERS, 'readonly');
      const store = tx.objectStore(NAND_USERS);
      const k = store.getAllKeys();
      const v = store.getAll();
      tx.oncomplete = () => resolve([k.result, v.result]);
      tx.onerror = () => reject(tx.error);
    });
    profiles = [];
    keys.forEach((key, i) => {
      if (key === CURRENT_KEY) current = String(values[i]);
      else profiles.push(values[i] as Profile);
    });
    profiles.sort((a, b) => a.created - b.created);
    if (!profiles.length) {
      const first = firstProfile();
      profiles.push(first);
      await storeProfile(first);
    }
    if (!profiles.some((p) => p.uid === current)) current = profiles[0].uid;
  } catch (err) {
    log('Profiles: could not be read (' + (err as Error).message + ') - playing as "Player".', 'err');
    profiles = [firstProfile()];
    current = profiles[0].uid;
  }
  renderProfiles();
}

/** Every new session needs this before a title starts. */
export async function stageUsers(): Promise<void> {
  if (!hasSession()) return;
  const refused = await call('users_set', profiles.map(toRecord), current);
  if (refused) log(`Profiles: the core refused the list (code ${refused}).`, 'err');
}

/** A session with nothing booted is where the next title starts, so it
 *  gets changes now. A running title keeps the user it started with. */
async function stageIfIdle(): Promise<void> {
  if (!titleBooted()) await stageUsers();
}

/** Stores profile edits made by system software through `acc`. */
export async function pullProfileEdits(): Promise<void> {
  if (!hasSession() || !(await call('users_take_edits'))) return;
  for (const edited of await call('users_read')) {
    const profile = profiles.find((p) => p.uid === edited.uid);
    if (!profile) continue;
    profile.nickname = edited.nickname;
    profile.editedAt = edited.editedAt;
    if (edited.picture) profile.picture = edited.picture;
    await storeProfile(profile);
  }
  renderProfiles();
}

/** Deletes keys of the form "<save id>@<uid>/<path>". */
async function deleteSavesOf(uid: string): Promise<number> {
  const db = await nandIdb();
  return new Promise<number>((resolve, reject) => {
    const tx = db.transaction(NAND_SAVES, 'readwrite');
    const store = tx.objectStore(NAND_SAVES);
    const cursor = store.openKeyCursor();
    let removed = 0;
    cursor.onsuccess = () => {
      const at = cursor.result;
      if (!at) return;
      const key = String(at.key);
      const cut = key.indexOf('/');
      const id = cut < 0 ? key : key.slice(0, cut);
      if (id.endsWith('@' + uid)) {
        store.delete(at.key);
        removed++;
      }
      at.continue();
    };
    tx.oncomplete = () => resolve(removed);
    tx.onerror = () => reject(tx.error);
  });
}

/** Center-crops to a 256x256 JPEG, the size and format titles expect. */
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
const addNote = $('profile-add-note');
const runningNote = $('profile-running-note');

/** Rows showing the delete confirmation or the rename field. */
let confirming: string | null = null;
let renaming: string | null = null;

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
  $<HTMLButtonElement>('profile-add-submit').disabled = full;
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

  const main = el('div', 'profile-main');
  if (renaming === profile.uid) {
    main.append(renameForm(profile));
  } else {
    main.append(el('span', 'profile-name', profile.nickname));
    if (isPlaying) main.append(el('span', 'profile-playing', 'Playing'));
  }

  const actions = el('div', 'profile-actions');
  if (confirming === profile.uid) {
    row.classList.add('is-confirming');
    main.replaceChildren(el('span', 'profile-confirm', `Delete ${profile.nickname} and their saves?`));
    const yes = el('button', 'btn small danger-solid', 'Delete');
    yes.type = 'button';
    yes.addEventListener('click', () => void removeProfile(profile));
    const no = el('button', 'btn small ghost', 'Keep');
    no.type = 'button';
    no.addEventListener('click', () => { confirming = null; renderProfiles(); });
    actions.append(yes, no);
    queueMicrotask(() => no.focus());
  } else if (renaming !== profile.uid) {
    if (!isPlaying) {
      const play = el('button', 'btn small', 'Play as');
      play.type = 'button';
      play.addEventListener('click', () => void choose(profile));
      actions.append(play);
    }
    const rename = el('button', 'btn small ghost', 'Rename');
    rename.type = 'button';
    rename.addEventListener('click', () => { renaming = profile.uid; confirming = null; renderProfiles(); });
    actions.append(rename);
    if (profile.picture) {
      const clear = el('button', 'btn small ghost', 'Remove picture');
      clear.type = 'button';
      clear.addEventListener('click', () => void clearPicture(profile));
      actions.append(clear);
    }
    // The last profile can't go: no title starts without a user. The one a
    // running title uses can't go either, or its saves would be orphaned.
    const locked = isPlaying && titleBooted();
    if (profiles.length > 1) {
      const del = el('button', 'btn small ghost danger', 'Delete');
      del.type = 'button';
      del.disabled = locked;
      if (locked) del.title = 'Reset the console to delete the profile the running title is playing as';
      del.addEventListener('click', () => { confirming = profile.uid; renaming = null; renderProfiles(); });
      actions.append(del);
    }
  }
  row.append(picture, main, actions);
  return row;
}

function renameForm(profile: Profile): HTMLFormElement {
  const form = el('form', 'profile-rename');
  const input = el('input', 'text');
  input.value = profile.nickname;
  input.maxLength = NICKNAME_MAX;
  input.required = true;
  input.setAttribute('aria-label', `New name for ${profile.nickname}`);
  const save = el('button', 'btn small', 'Save');
  save.type = 'submit';
  const cancel = el('button', 'btn small ghost', 'Cancel');
  cancel.type = 'button';
  cancel.addEventListener('click', () => { renaming = null; renderProfiles(); });
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    void rename(profile, input.value);
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      // Cancel the rename without closing the dialog.
      e.preventDefault();
      e.stopPropagation();
      renaming = null;
      renderProfiles();
    }
  });
  form.append(input, save, cancel);
  queueMicrotask(() => { input.focus(); input.select(); });
  return form;
}

async function update(profile: Profile): Promise<void> {
  profile.editedAt = now();
  try {
    await storeProfile(profile);
  } catch (err) {
    log('Profiles: could not be saved (' + (err as Error).message + ')', 'err');
  }
  renderProfiles();
  await stageIfIdle();
}

async function choose(profile: Profile): Promise<void> {
  current = profile.uid;
  try {
    await storeCurrent();
  } catch (err) {
    log('Profiles: the choice could not be saved (' + (err as Error).message + ')', 'err');
  }
  log(titleBooted()
    ? `Profiles: ${profile.nickname} plays the next title; the running one keeps its player.`
    : `Profiles: playing as ${profile.nickname}.`, 'dim');
  renderProfiles();
  await stageIfIdle();
}

async function rename(profile: Profile, name: string): Promise<void> {
  const nickname = name.trim();
  if (!nickname) return;
  profile.nickname = [...nickname].slice(0, NICKNAME_MAX).join('');
  renaming = null;
  await update(profile);
}

async function setPicture(profile: Profile, file: File): Promise<void> {
  try {
    profile.picture = await toPicture(file);
  } catch (err) {
    log(`Profiles: ${file.name} could not be used as a picture (${(err as Error).message})`, 'err');
    return;
  }
  await update(profile);
}

async function clearPicture(profile: Profile): Promise<void> {
  profile.picture = null;
  await update(profile);
}

async function removeProfile(profile: Profile): Promise<void> {
  confirming = null;
  profiles = profiles.filter((p) => p.uid !== profile.uid);
  const entry = pictureUrls.get(profile.uid);
  if (entry) URL.revokeObjectURL(entry.url);
  pictureUrls.delete(profile.uid);
  if (current === profile.uid) current = profiles[0].uid;
  try {
    await idbRequest((s) => s.delete(profile.uid), 'readwrite');
    await storeCurrent();
    const removed = await deleteSavesOf(profile.uid);
    log(`Profiles: deleted ${profile.nickname}` + (removed ? ` and ${removed} saved entries` : ''), 'dim');
  } catch (err) {
    log('Profiles: the deletion could not be finished (' + (err as Error).message + ')', 'err');
  }
  renderProfiles();
  await stageIfIdle();
  buttonEl.focus();
}

addForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const nickname = addName.value.trim();
  if (!nickname || profiles.length >= MAX_PROFILES) return;
  const profile: Profile = {
    uid: newUid(),
    nickname: [...nickname].slice(0, NICKNAME_MAX).join(''),
    editedAt: now(),
    picture: null,
    created: Date.now(),
  };
  profiles.push(profile);
  addName.value = '';
  void storeProfile(profile)
    .catch((err: Error) => log('Profiles: could not be saved (' + err.message + ')', 'err'))
    .then(() => {
      renderProfiles();
      return stageIfIdle();
    });
});

buttonEl.addEventListener('click', () => {
  confirming = null;
  renaming = null;
  renderProfiles();
  dialogEl.showModal();
});
$('btn-profiles-close').addEventListener('click', () => dialogEl.close());
// Backdrop clicks target the dialog element itself.
dialogEl.addEventListener('click', (e) => {
  if (e.target === dialogEl) dialogEl.close();
});
dialogEl.addEventListener('close', () => {
  confirming = null;
  renaming = null;
});
