// The on-page console.

import { $, el } from './dom';
import type { LogClass } from '../shared/protocol';
import { LOG_KEY, LOG_STORE, idbApply, idbGet, logIdb, type StoredEntry } from './db';
import { fmtSize } from './format';
import { openPanel } from './shell';

export type { LogClass } from '../shared/protocol';

const consoleEl = $('console');
const autoscrollCb = $<HTMLInputElement>('autoscroll-cb');
const levelSelect = $<HTMLSelectElement>('log-level');

export type LogLevel = 'error' | 'warn' | 'log' | 'debug';

const LEVELS: LogLevel[] = ['error', 'warn', 'log', 'debug'];
const LEVEL_KEY = 'switch-wasm-log-level';

function levelOf(cls?: LogClass): LogLevel {
  if (cls === 'err') return 'error';
  if (cls === 'warn') return 'warn';
  if (cls === 'dim') return 'debug';
  return 'log';
}

function storedLevel(): LogLevel {
  try {
    const stored = localStorage.getItem(LEVEL_KEY) as LogLevel | null;
    return stored && LEVELS.includes(stored) ? stored : 'log';
  } catch {
    return 'log';
  }
}

let level = storedLevel();
levelSelect.value = level;

function shown(cls?: LogClass): boolean {
  return LEVELS.indexOf(levelOf(cls)) <= LEVELS.indexOf(level);
}

export function logLevel(): LogLevel {
  return level;
}

const levelListeners: ((level: LogLevel) => void)[] = [];

export function onLogLevel(listener: (level: LogLevel) => void): void {
  levelListeners.push(listener);
}

levelSelect.addEventListener('change', () => {
  level = levelSelect.value as LogLevel;
  try {
    localStorage.setItem(LEVEL_KEY, level);
  } catch { /* the choice still holds for this session */ }
  renderConsole();
  for (const listener of levelListeners) listener(level);
});

// Entries kept on the page; `backlog` keeps more for copying and saving.
const SHOWN_MAX = 2000;

const KEPT_MAX = 50_000;

// Consecutive repeats collapse into one entry with a count.
interface Entry {
  text: string;
  cls?: LogClass;
  count: number;
}

// Every entry, whether or not it is still on the page.
const backlog: Entry[] = [];

// The newest entry's row, so a repeat can be counted in place.
let lastRow: HTMLElement | null = null;

export function log(msg: string, cls?: LogClass): void {
  mirrorDirty = true;
  const visible = shown(cls);

  // A row the view evicted is detached, so counting into it would hide the repeat.
  const last = backlog[backlog.length - 1];
  if (last && last.text === msg && last.cls === cls && !visible) {
    last.count += 1;
    return;
  }
  if (last && last.text === msg && last.cls === cls && lastRow?.isConnected) {
    last.count += 1;
    lastRow.dataset.repeat = String(last.count);
    if (autoscrollCb.checked) consoleEl.scrollTop = consoleEl.scrollHeight;
    return;
  }

  backlog.push({ text: msg, cls, count: 1 });
  if (backlog.length > KEPT_MAX) backlog.splice(0, backlog.length - KEPT_MAX);
  if (!visible) return;

  lastRow = row({ text: msg, cls, count: 1 });
  consoleEl.appendChild(lastRow);
  while (consoleEl.childElementCount > SHOWN_MAX) consoleEl.firstElementChild!.remove();
  if (autoscrollCb.checked) consoleEl.scrollTop = consoleEl.scrollHeight;
  if (cls === 'err') openPanel('console');
}

function row(entry: Entry): HTMLElement {
  const line = el('div', entry.cls, entry.text);
  if (entry.count > 1) line.dataset.repeat = String(entry.count);
  return line;
}

function renderConsole(): void {
  const rows: HTMLElement[] = [];
  lastRow = null;
  for (let i = backlog.length - 1; i >= 0 && rows.length < SHOWN_MAX; i--) {
    if (!shown(backlog[i].cls)) continue;
    const line = row(backlog[i]);
    if (i === backlog.length - 1) lastRow = line;
    rows.push(line);
  }
  consoleEl.replaceChildren(...rows.reverse());
  if (autoscrollCb.checked) consoleEl.scrollTop = consoleEl.scrollHeight;
}

const STORED_NAMED_MAX = 8;

// `null` in the map is a deletion.
export function logStored(what: string, changes: Map<string, StoredEntry | null>): void {
  let bytes = 0;
  const named: string[] = [];
  for (const [path, entry] of changes) {
    if (entry?.kind === 'file') bytes += entry.data?.length ?? 0;
    if (named.length < STORED_NAMED_MAX) {
      named.push(entry ? path : path + ' (deleted)');
    }
  }
  const more = changes.size > named.length ? `, and ${changes.size - named.length} more` : '';
  log(
    `[io] ${what}: stored ${changes.size} changes (${fmtSize(bytes)}) in IndexedDB: `
    + named.join(', ') + more,
  );
}

// One entry per line, at one level.
export function logBlock(text: string, cls?: LogClass): void {
  for (const line of text.replace(/\n$/, '').split('\n')) log(line, cls);
}

export function clearConsole(): void {
  consoleEl.textContent = '';
  backlog.length = 0;
  lastRow = null;
}

$('btn-clear-console').addEventListener('click', clearConsole);

// Includes entries the view has dropped.
export function consoleText(): string {
  return backlog.map(asLine).join('\n');
}

function asLine(entry: Entry): string {
  return entry.count > 1 ? `${entry.text}  (x${entry.count})` : entry.text;
}

const copyBtn = $('btn-copy-console');

let copyLabelTimer = 0;
function flashCopyLabel(text: string): void {
  clearTimeout(copyLabelTimer);
  copyBtn.textContent = text;
  copyLabelTimer = setTimeout(() => {
    copyBtn.textContent = 'Copy all';
  }, 1400);
}

// Fallback for insecure contexts, where `navigator.clipboard` is unavailable.
function copyViaSelection(text: string): boolean {
  const area = el('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  // Off-screen rather than hidden: a display:none textarea cannot be selected.
  area.style.cssText = 'position:fixed;top:-1000px;left:-1000px;opacity:0';
  document.body.appendChild(area);
  area.select();
  let ok: boolean;
  try {
    ok = document.execCommand('copy');
  } catch {
    ok = false;
  }
  area.remove();
  return ok;
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // No clipboard API, or the permission was refused.
    return copyViaSelection(text);
  }
}

async function copyConsole(): Promise<void> {
  const text = consoleText();
  if (!text) {
    flashCopyLabel('Log is empty');
    return;
  }
  flashCopyLabel(await copyText(text) ? 'Copied' : 'Copy failed');
}

copyBtn.addEventListener('click', copyConsole);

export function download(name: string, text: string, type = 'text/plain'): void {
  const url = URL.createObjectURL(new Blob([text], { type: `${type};charset=utf-8` }));
  const link = el('a');
  link.href = url;
  link.download = name;
  link.click();
  // Not before the click: revoking the URL cancels the download.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function stamp(): string {
  return new Date().toISOString().replace(/[:.]/g, '-').replace('Z', '');
}

// The log's tail is mirrored to IndexedDB on a timer, so it survives the browser killing the tab.
const MIRROR_EVERY_MS = 5000;
const MIRRORED_MAX = 4000;

let mirrorDirty = false;

setInterval(() => {
  if (!mirrorDirty) return;
  mirrorDirty = false;
  void mirrorLog();
}, MIRROR_EVERY_MS);

async function mirrorLog(): Promise<void> {
  try {
    const tail = backlog.slice(-MIRRORED_MAX).map(asLine).join('\n');
    await idbApply(await logIdb(), LOG_STORE, [[LOG_KEY, tail]]);
  } catch {
    // Private browsing, refused quota, or evicted origin: the mirror is lost silently.
  }
}

// Read once at startup and then cleared.
export async function takePreviousLog(): Promise<string> {
  try {
    const db = await logIdb();
    const previous = await idbGet<string>(db, LOG_STORE, LOG_KEY);
    if (!previous) return '';
    await idbApply(db, LOG_STORE, [[LOG_KEY, null]]);
    return previous;
  } catch {
    return '';
  }
}

// A `pagehide` mark in `localStorage` (synchronous) tells a clean exit from a killed tab.
const RUNNING_KEY = 'switch-wasm-running';

function endedCleanly(): boolean {
  try {
    return localStorage.getItem(RUNNING_KEY) === null;
  } catch {
    return true;
  }
}

function markRunning(running: boolean): void {
  try {
    if (running) localStorage.setItem(RUNNING_KEY, '1');
    else localStorage.removeItem(RUNNING_KEY);
  } catch {
    // No storage: see `endedCleanly`.
  }
}

window.addEventListener('pagehide', () => markRunning(false));

// The mark is per origin, so other live tabs are asked before reporting a dead session.
const TAB_CHANNEL = 'switch-wasm-tabs';
const TAB_ANSWER_MS = 250;

// A page receives its own `BroadcastChannel` pings, so they carry an id.
const TAB_ID = Math.random().toString(36).slice(2);

interface TabMessage {
  ask?: string;
  answer?: string;
}

function anotherTabIsLive(): Promise<boolean> {
  if (typeof BroadcastChannel === 'undefined') return Promise.resolve(false);
  return new Promise((resolve) => {
    const channel = new BroadcastChannel(TAB_CHANNEL);
    const done = (live: boolean) => {
      channel.close();
      resolve(live);
    };
    channel.onmessage = (e) => {
      if ((e.data as TabMessage)?.answer === TAB_ID) done(true);
    };
    channel.postMessage({ ask: TAB_ID } satisfies TabMessage);
    setTimeout(() => done(false), TAB_ANSWER_MS);
  });
}

if (typeof BroadcastChannel !== 'undefined') {
  const channel = new BroadcastChannel(TAB_CHANNEL);
  channel.onmessage = (e) => {
    const ask = (e.data as TabMessage)?.ask;
    if (ask && ask !== TAB_ID) channel.postMessage({ answer: ask } satisfies TabMessage);
  };
}

export async function offerPreviousLog(): Promise<void> {
  const clean = endedCleanly() || await anotherTabIsLive();
  markRunning(true);
  const previous = await takePreviousLog();
  if (clean || !previous) return;
  const button = $('btn-save-previous');
  button.hidden = false;
  button.addEventListener('click', () => {
    download(`switch-wasm-log-previous-${stamp()}.txt`, previous);
  });
  log('The previous session ended without closing its log - "Save previous" in the console bar '
    + 'has what it had said by then.', 'warn');
}

$('btn-save-console').addEventListener('click', () => {
  const text = consoleText();
  if (!text) {
    flashCopyLabel('Log is empty');
    return;
  }
  download(`switch-wasm-log-${stamp()}.txt`, text);
});
