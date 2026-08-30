/* The on-page console: everything the emulator has to say, mirrored into
   DevTools so it can be filtered by severity there too. */

import { $, el } from './dom';
import { LOG_KEY, LOG_STORE, idbApply, idbGet, logIdb } from './db';
import { openPanel } from './shell';

export type LogClass = 'err' | 'warn' | 'ok' | 'dim';

const consoleEl = $('console');
const autoscrollCb = $<HTMLInputElement>('autoscroll-cb');
const TAG = '[switch-wasm]';

/** How many entries the console keeps on the page.
 *
 *  It kept all of them, and every entry is its own element: with the
 *  instruction trace on, a run puts tens of thousands into the document and
 *  the page slows to a crawl laying them out -- while `Copy all` walks every
 *  one of them. The text is kept in `backlog` either way, so what is dropped
 *  here is dropped from the *view*, not from what gets copied or saved. */
const SHOWN_MAX = 2000;

/** How many entries the log keeps at all.
 *
 *  Larger than the view by a lot, because this is what a bug report is cut
 *  from, and smaller than unbounded because a session left running overnight
 *  should not end as a tab the browser kills. */
const KEPT_MAX = 50_000;

/** One entry: what was said, how loudly, and how many times in a row.
 *
 *  A guest polling a service that is not there says the same thing on every
 *  frame, and a thousand identical rows hide everything else that happened.
 *  Consecutive repeats collapse into one row carrying a count, which is what a
 *  browser's own console does with the same problem. */
interface Entry {
  text: string;
  cls?: LogClass;
  count: number;
}

/** Every entry, in order, whether or not it is still on the page. */
const backlog: Entry[] = [];

/** The row the newest entry is on, so a repeat of it can be counted in place
 *  rather than appended. */
let lastRow: HTMLElement | null = null;

export function log(msg: string, cls?: LogClass): void {
  // Real browser console (DevTools): route by severity for filterability.
  // A repeat goes through here too -- DevTools keeps a count of its own, and
  // dropping repeats would leave its copy of the log disagreeing with this one
  // about what happened.
  if (cls === 'err') console.error(TAG, msg);
  else if (cls === 'warn') console.warn(TAG, msg);
  else if (cls === 'ok') console.info(TAG, msg);
  else if (cls === 'dim') console.debug(TAG, msg);
  else console.log(TAG, msg);

  mirrorDirty = true;

  // `isConnected` rather than a null check: a row the view has evicted is
  // still referenced here, and counting into a detached element would swallow
  // the repeat instead of showing it.
  const last = backlog[backlog.length - 1];
  if (last && last.text === msg && last.cls === cls && lastRow?.isConnected) {
    last.count += 1;
    lastRow.dataset.repeat = String(last.count);
    if (autoscrollCb.checked) consoleEl.scrollTop = consoleEl.scrollHeight;
    // Deliberately not `openPanel`: the first of these opened it already, and
    // re-opening on every repeat takes the panel back from someone who has
    // moved off it to look at something else.
    return;
  }

  backlog.push({ text: msg, cls, count: 1 });
  if (backlog.length > KEPT_MAX) backlog.splice(0, backlog.length - KEPT_MAX);

  // On-page console mirror.
  lastRow = el('div', cls, msg);
  consoleEl.appendChild(lastRow);
  while (consoleEl.childElementCount > SHOWN_MAX) consoleEl.firstElementChild!.remove();
  if (autoscrollCb.checked) consoleEl.scrollTop = consoleEl.scrollHeight;
  // Anything that went wrong is worth surfacing even with the panel closed.
  if (cls === 'err') openPanel('console');
}

/** Log a block of text one entry per line, at one level.
 *
 *  What arrives from the emulator is a stream, not a line: a fault is its
 *  message, a register dump and an instruction trail, and pushing all of that
 *  into a single element makes it one unbreakable row the console scrolls
 *  sideways for. */
export function logBlock(text: string, cls?: LogClass): void {
  for (const line of text.replace(/\n$/, '').split('\n')) log(line, cls);
}

export function clearConsole(): void {
  consoleEl.textContent = '';
  backlog.length = 0;
  lastRow = null;
}

$('btn-clear-console').addEventListener('click', clearConsole);

/** The whole log as text, one line per entry -- including the entries the
 *  view has since dropped. */
export function consoleText(): string {
  return backlog.map(asLine).join('\n');
}

/** One entry as a line, carrying its count. A log that collapsed a thousand
 *  identical lines has to say so: without this a copy of it reads as though
 *  the thing happened once. */
function asLine(entry: Entry): string {
  return entry.count > 1 ? `${entry.text}  (x${entry.count})` : entry.text;
}

const copyBtn = $('btn-copy-console');

/** Say what happened on the button itself and put its label back. A log copy
 *  is worth confirming -- there is no other sign it worked -- but not worth a
 *  line in the log it just copied. */
let copyLabelTimer = 0;
function flashCopyLabel(text: string): void {
  clearTimeout(copyLabelTimer);
  copyBtn.textContent = text;
  copyLabelTimer = setTimeout(() => { copyBtn.textContent = 'Copy all'; }, 1400);
}

/** `navigator.clipboard` needs a secure context, which a page served over
 *  plain http from another machine is not -- and that is exactly how this gets
 *  opened when someone is testing on a phone. Fall back to a selection copy,
 *  which has no such requirement. */
function copyViaSelection(text: string): boolean {
  const area = el('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  // Off-screen rather than hidden: a display:none textarea cannot be selected.
  area.style.cssText = 'position:fixed;top:-1000px;left:-1000px;opacity:0';
  document.body.appendChild(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } catch {
    ok = false;
  }
  area.remove();
  return ok;
}

/** Put `text` on the clipboard, however this context allows. */
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

/** Offer `text` as a file to save.
 *
 *  The clipboard is not enough on its own: a log worth reporting is tens of
 *  thousands of lines, a crash takes the tab that holds it, and a phone has
 *  nowhere to paste it. */
export function download(name: string, text: string, type = 'text/plain'): void {
  const url = URL.createObjectURL(new Blob([text], { type: `${type};charset=utf-8` }));
  const link = el('a');
  link.href = url;
  link.download = name;
  link.click();
  // Not before the click: revoking the URL is revoking the download.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** A filename stamp that sorts, and that names the moment rather than the
 *  locale: two reports from one session must not collide. */
export function stamp(): string {
  return new Date().toISOString().replace(/[:.]/g, '-').replace('Z', '');
}

/* Surviving the tab.

   Everything else that can go wrong here leaves the page alive: a wasm trap
   unwinds into the run loop, which then reads the panic message and dumps the
   context, and the log is still there to be saved. What is not survivable is
   the browser killing the tab -- which an emulator that maps gigabytes of
   guest memory is a candidate for -- and after that a reload has nothing at
   all to say about what was happening.

   So the log is mirrored, on a timer rather than per line: with the
   instruction trace on this would otherwise be a database write per
   instruction. What is kept is the tail, because a cap that can be reached is
   better than a store that grows until the browser evicts the whole origin --
   which would take the SD card and the NAND with it. */
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
    // Private browsing, a refused quota, an evicted origin. The log is still
    // on the page; only the copy that would outlive it is lost, and saying so
    // in the log would be a line per attempt.
  }
}

/** The log from the session before this one, if the browser kept it.
 *
 *  Read once at startup and then cleared, so that "previous" always means the
 *  run before this one rather than the oldest run that ever crashed. */
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

/* Whether the session before this one closed itself.

   The mirrored log is worth offering only when it is the account of a run that
   did not get to finish -- otherwise every ordinary reload nags about the last
   one. `pagehide` is the signal, and this is `localStorage` rather than the
   database the log itself lives in because a `pagehide` handler is not given
   time to await anything: a synchronous write is the only kind that reliably
   lands there. A tab the browser kills never runs the handler, which is
   exactly the case being detected. */
const RUNNING_KEY = 'switch-wasm-running';

function endedCleanly(): boolean {
  try {
    return localStorage.getItem(RUNNING_KEY) === null;
  } catch {
    // No storage at all: treat every run as clean rather than warning about
    // sessions this page cannot know anything about.
    return true;
  }
}

function markRunning(running: boolean): void {
  try {
    if (running) localStorage.setItem(RUNNING_KEY, '1');
    else localStorage.removeItem(RUNNING_KEY);
  } catch {
    // See `endedCleanly`.
  }
}

window.addEventListener('pagehide', () => markRunning(false));

/* The mark is per origin, not per tab, so a second tab opened beside a first
   sees it set and would report the *live* tab as a session that died. Asking
   is what tells the two apart: a tab that is still there answers, and a tab
   the browser killed cannot. */
const TAB_CHANNEL = 'switch-wasm-tabs';
const TAB_ANSWER_MS = 250;

/* Who is asking. A `BroadcastChannel` withholds a message only from the object
   that sent it, *not* from the rest of the page -- so the answering channel
   below is delivered this page's own ping and used to answer it, which made
   every session look as though it had a live sibling and suppressed the offer
   entirely. The id is what the two halves tell each other apart by. */
const TAB_ID = Math.random().toString(36).slice(2);

interface TabMessage {
  ask?: string;
  answer?: string;
}

function anotherTabIsLive(): Promise<boolean> {
  if (typeof BroadcastChannel === 'undefined') return Promise.resolve(false);
  return new Promise((resolve) => {
    const channel = new BroadcastChannel(TAB_CHANNEL);
    const done = (live: boolean) => { channel.close(); resolve(live); };
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

/** Offer the previous session's log, if there is one worth offering. */
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
