// The on-page console.

import { $, el } from './dom';
import type { LogLevel } from '../shared/protocol';
import { LOG_KEY, LOG_STORE, idbApply, idbGet, logIdb } from '../shared/db';
import { openPanel } from './shell';

export type { LogLevel } from '../shared/protocol';

const consoleEl = $('console');
const autoscrollCb = $<HTMLInputElement>('autoscroll-cb');
const levelSelect = $<HTMLSelectElement>('log-level');

const LEVELS: LogLevel[] = ['error', 'warn', 'info', 'log', 'debug'];
const LEVEL_KEY = 'switch-wasm-log-level';

function storedLevel(): LogLevel {
  try {
    const stored = localStorage.getItem(LEVEL_KEY) as LogLevel | null;
    return stored && LEVELS.includes(stored) ? stored : 'log';
  } catch {
    return 'log';
  }
}

// The least severe level the console shows.
let threshold = storedLevel();
levelSelect.value = threshold;

function shown(level: LogLevel): boolean {
  return LEVELS.indexOf(level) <= LEVELS.indexOf(threshold);
}

export function logLevel(): LogLevel {
  return threshold;
}

const levelListeners: ((level: LogLevel) => void)[] = [];

export function onLogLevel(listener: (level: LogLevel) => void): void {
  levelListeners.push(listener);
}

levelSelect.addEventListener('change', () => {
  threshold = levelSelect.value as LogLevel;
  try {
    localStorage.setItem(LEVEL_KEY, threshold);
  } catch { /* the choice still holds for this session */ }
  renderConsole();
  for (const listener of levelListeners) listener(threshold);
});

// Entries kept on the page; `backlog` keeps every one for copying and saving.
const SHOWN_MAX = 2000;

// Consecutive repeats collapse into one entry with a count.
interface Entry {
  text: string;
  level: LogLevel;
  count: number;
}

// Every entry, whether or not it is still on the page. Only the Clear button empties it.
const backlog: Entry[] = [];

// The newest entry's row, so a repeat can be counted in place.
let lastRow: HTMLElement | null = null;

// One entry per line.
export function log(text: string, level: LogLevel = 'log'): void {
  for (const line of text.replace(/\n$/, '').split('\n')) logLine(line, level);
}

function logLine(msg: string, level: LogLevel): void {
  mirrorDirty = true;
  const visible = shown(level);

  // A row the view evicted is detached, so counting into it would hide the repeat.
  const last = backlog[backlog.length - 1];
  if (last && last.text === msg && last.level === level && !visible) {
    last.count += 1;
    return;
  }
  if (last && last.text === msg && last.level === level && lastRow?.isConnected) {
    last.count += 1;
    lastRow.dataset.repeat = String(last.count);
    if (autoscrollCb.checked) consoleEl.scrollTop = consoleEl.scrollHeight;
    return;
  }

  backlog.push({ text: msg, level, count: 1 });
  if (!visible) return;

  lastRow = row({ text: msg, level, count: 1 });
  consoleEl.appendChild(lastRow);
  while (consoleEl.childElementCount > SHOWN_MAX) consoleEl.firstElementChild!.remove();
  if (autoscrollCb.checked) consoleEl.scrollTop = consoleEl.scrollHeight;
  if (level === 'error') openPanel('console');
}

// `e>`, `w>`, `i>`, `l>` or `d>`, so the level survives copying and saving.
function prompted(entry: Entry): string {
  return `${entry.level[0]}> ${entry.text}`;
}

function row(entry: Entry): HTMLElement {
  const line = el('div', entry.level, prompted(entry));
  if (entry.count > 1) line.dataset.repeat = String(entry.count);
  return line;
}

function renderConsole(): void {
  const rows: HTMLElement[] = [];
  lastRow = null;
  for (let i = backlog.length - 1; i >= 0 && rows.length < SHOWN_MAX; i--) {
    if (!shown(backlog[i].level)) continue;
    const line = row(backlog[i]);
    if (i === backlog.length - 1) lastRow = line;
    rows.push(line);
  }
  consoleEl.replaceChildren(...rows.reverse());
  if (autoscrollCb.checked) consoleEl.scrollTop = consoleEl.scrollHeight;
}

function clearConsole(): void {
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
  const line = prompted(entry);
  return entry.count > 1 ? `${line}  (x${entry.count})` : line;
}

const saveBtn = $('btn-save-console');

let saveLabelTimer = 0;
function flashSaveLabel(text: string): void {
  clearTimeout(saveLabelTimer);
  saveBtn.textContent = text;
  saveLabelTimer = setTimeout(() => {
    saveBtn.textContent = 'Save';
  }, 1400);
}

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

// The log is mirrored to IndexedDB on a timer, so it survives the browser killing the tab.
const MIRROR_EVERY_MS = 5000;

let mirrorDirty = false;

setInterval(() => {
  if (!mirrorDirty) return;
  mirrorDirty = false;
  void mirrorLog();
}, MIRROR_EVERY_MS);

async function mirrorLog(): Promise<void> {
  try {
    await idbApply(await logIdb(), LOG_STORE, [[LOG_KEY, consoleText()]]);
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

// What a report needs that the log lines do not say.
function logHeader(): string {
  return [
    'Switch WASM build ' + __BUILD__,
    'Browser: ' + navigator.userAgent,
    'WebGPU: ' + ('gpu' in navigator ? 'available' : 'not available'),
  ].join('\n');
}

// Download the console as a text file, under a header naming the build and browser.
export function saveLog(): void {
  const text = consoleText();
  if (!text) {
    flashSaveLabel('Log is empty');
    return;
  }
  download(`switch-wasm-log-${stamp()}.txt`, logHeader() + '\n\n' + text);
}

saveBtn.addEventListener('click', saveLog);
