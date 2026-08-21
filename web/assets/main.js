/* switch-wasm browser frontend.

   The emulator runs in a web worker (worker.js) so long executions don't
   freeze the page; this file is a thin promise-based RPC client over
   postMessage plus the application shell. Buffers (files, framebuffer,
   console/trace output) are transferred across the worker boundary. */

const $ = (id) => document.getElementById(id);

/** Create an element with a class and text, avoiding innerHTML entirely. */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ---------- worker RPC ----------

let worker = null;
let handle = -1;   // client-side session id (display only)
let ready = false;
let readyResolve;
const readyPromise = new Promise((r) => { readyResolve = r; });

let msgId = 0;
const pending = new Map();

function call(cmd, ...args) {
  return new Promise((resolve, reject) => {
    const id = ++msgId;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, args });
  });
}

function initWorker() {
  worker = new Worker('assets/worker.js');
  worker.onmessage = (e) => {
    const d = e.data;
    if (d.type === 'ready') {
      ready = true;
      readyResolve();
      return;
    }
    const p = pending.get(d.id);
    if (!p) return;
    pending.delete(d.id);
    if (d.ok) p.resolve(d.result);
    else p.reject(new Error(d.error || 'unknown error'));
  };
  worker.onerror = (e) => {
    readyResolve();
    log('worker error: ' + e.message, 'err');
  };
}

// ---------- console ----------

const consoleEl = $('console');
const autoscrollCb = $('autoscroll-cb');
const TAG = '[switch-wasm]';

function log(msg, cls) {
  // Real browser console (DevTools): route by severity for filterability.
  if (cls === 'err') console.error(TAG, msg);
  else if (cls === 'ok') console.info(TAG, msg);
  else if (cls === 'dim') console.debug(TAG, msg);
  else console.log(TAG, msg);
  // On-page console mirror.
  consoleEl.appendChild(el('div', cls, msg));
  if (autoscrollCb.checked) consoleEl.scrollTop = consoleEl.scrollHeight;
  // Anything that went wrong is worth surfacing even with the panel closed.
  if (cls === 'err') openPanel('console');
}
function clearConsole() { consoleEl.textContent = ''; }
$('btn-clear-console').addEventListener('click', clearConsole);

/** The whole on-page log as text, one line per entry.
 *
 *  Not `consoleEl.textContent`: every entry is its own `<div>`, and that
 *  property concatenates their text with nothing in between, so a register
 *  dump and the trace after it would arrive as one unbroken line. */
function consoleText() {
  return Array.from(consoleEl.children).map((node) => node.textContent).join('\n');
}

const copyBtn = $('btn-copy-console');

/** Say what happened on the button itself and put its label back. A log copy
 *  is worth confirming -- there is no other sign it worked -- but not worth a
 *  line in the log it just copied. */
let copyLabelTimer = 0;
function flashCopyLabel(text) {
  clearTimeout(copyLabelTimer);
  copyBtn.textContent = text;
  copyLabelTimer = setTimeout(() => { copyBtn.textContent = 'Copy all'; }, 1400);
}

/** `navigator.clipboard` needs a secure context, which a page served over
 *  plain http from another machine is not -- and that is exactly how this gets
 *  opened when someone is testing on a phone. Fall back to a selection copy,
 *  which has no such requirement. */
function copyViaSelection(text) {
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

async function copyConsole() {
  const text = consoleText();
  if (!text) {
    flashCopyLabel('Log is empty');
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    flashCopyLabel('Copied');
    return;
  } catch {
    // Fall through: no clipboard API, or the permission was refused.
  }
  flashCopyLabel(copyViaSelection(text) ? 'Copied' : 'Copy failed');
}

copyBtn.addEventListener('click', copyConsole);

async function readLastError() { return await call('last_error'); }

function fmtSize(n) {
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(2) + ' GiB';
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(2) + ' MiB';
  if (n >= 1 << 10) return (n / (1 << 10)).toFixed(1) + ' KiB';
  return n + ' B';
}

// ---------- application shell ----------

const stageEl = $('stage');
const screenEl = $('screen');
const screenCtx = screenEl.getContext('2d', { alpha: false });
const overlayEl = $('overlay');
const dropveilEl = $('dropveil');
const stateEl = $('state');
let fbW = 0, fbH = 0, fbBytes = 0;

function setState(text) {
  stateEl.textContent = text;
  stateEl.dataset.state = text;
}

function showOverlay(show) {
  overlayEl.classList.toggle('hidden', !show);
}

// Uncover the canvas and blank it. The context is alpha-less, so clearRect
// paints black - the same "powered on, nothing presented yet" state a real
// console shows.
function showScreen() {
  screenCtx.clearRect(0, 0, screenEl.width, screenEl.height);
  showOverlay(false);
}

// Side panel (Console / Debug / Files). Closed by default: the screen is the
// point of the page, not the tooling around it.
function panelOpen() {
  return document.body.classList.contains('panel-open');
}
function setPanel(open) {
  document.body.classList.toggle('panel-open', open);
  $('btn-panel').setAttribute('aria-expanded', String(open));
}
function openPanel(tab) {
  setPanel(true);
  if (tab) selectTab(tab);
}
function selectTab(name) {
  document.querySelectorAll('.tab').forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle('is-active', on);
    t.setAttribute('aria-selected', String(on));
  });
  document.querySelectorAll('.tabpanel').forEach((p) => {
    p.classList.toggle('is-active', p.dataset.panel === name);
  });
}
document.querySelectorAll('.tab').forEach((t) => {
  t.addEventListener('click', () => selectTab(t.dataset.tab));
});
$('btn-panel').addEventListener('click', () => setPanel(!panelOpen()));
$('btn-panel-close').addEventListener('click', () => setPanel(false));

$('btn-fullscreen').addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else stageEl.requestFullscreen?.();
});

// ---------- display ----------

// Copy the emulated screen into the canvas, resizing it to whatever resolution
// the guest presented (1280x720 for most homebrew). Before the guest hands the
// display its first frame there is nothing to copy, so the canvas stays a blank
// screen - visible, but empty.
async function renderFb() {
  const w = await call('fb_width');
  const h = await call('fb_height');
  if (!w || !h) return;
  if (w !== fbW || h !== fbH) {
    fbW = w;
    fbH = h;
    fbBytes = w * h * 4;
    screenEl.width = w;
    screenEl.height = h;
  }
  // Until the guest hands the display a frame there is no resolution to
  // report - `fb_width`/`fb_height` fall back to the memory-mapped
  // framebuffer's size, which real homebrew never uses.
  if (lastFrame === 0) lastFrame = await call('frame_count');
  $('res').textContent = lastFrame > 0 ? w + '×' + h : '—';
  if (lastFrame === 0) {
    // Nothing has been presented, so there is no screen content to copy: the
    // fallback framebuffer region is just guest memory that Horizon homebrew
    // never writes. Show it as a blank screen instead of that memory's
    // contents.
    showScreen();
    return;
  }
  const pixels = await call('fb_snapshot', fbBytes);
  if (pixels && pixels.length >= fbBytes) {
    const arr = new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, fbBytes);
    screenCtx.putImageData(new ImageData(arr, fbW, fbH), 0, 0);
    showOverlay(false);
  }
}

// Frames per second, measured from the guest's own present count.
let fpsFrames = 0;
let fpsSince = performance.now();
function countFrames(delta) {
  fpsFrames += delta;
  const now = performance.now();
  const elapsed = now - fpsSince;
  if (elapsed >= 500) {
    $('fps').textContent = (fpsFrames * 1000 / elapsed).toFixed(1) + ' fps';
    fpsFrames = 0;
    fpsSince = now;
  }
}

// ---------- boot ----------

// The font the emulator serves as the console's shared system font. Homebrew
// reads it out of pl:u's shared memory and renders it with its own copy of
// FreeType, so without it nothing but pre-rendered bitmaps appears on screen.
const FONT_URL = 'assets/font.ttf';
let fontBytes = null;

async function stageFont() {
  if (!fontBytes) {
    try {
      const res = await fetch(FONT_URL);
      if (!res.ok) throw new Error(res.status + ' ' + res.statusText);
      fontBytes = new Uint8Array(await res.arrayBuffer());
    } catch (err) {
      log('No system font (' + FONT_URL + '): ' + err.message + ' - text will not render.', 'err');
      return;
    }
  }
  await call('load_font', fontBytes);
}

// ---------- persistent SD card ----------
//
// The emulated SD card lives in the session's memory, so without this nothing
// the guest writes survives a reload - and a save manager that cannot keep a
// save is not much of one. IndexedDB fits the shape the core exposes: the card
// is a path -> bytes map, and the core reports which paths the *guest*
// changed, so a flush writes back only those instead of the whole card.

const SD_DB_NAME = 'switch-wasm-sd';
const SD_STORE = 'entries';
let sdDb = null;
// Entries drained from the core but not yet stored - keyed by path, so a file
// written repeatedly between two successful flushes only costs one slot. The
// core cannot be handed a change back once drained, so anything IndexedDB
// refuses (a quota, most likely) waits here for the next flush rather than
// being lost.
const sdBacklog = new Map();
let sdFlushing = false;

function sdIdb() {
  if (sdDb) return Promise.resolve(sdDb);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(SD_DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(SD_STORE)) req.result.createObjectStore(SD_STORE);
    };
    req.onsuccess = () => { sdDb = req.result; resolve(sdDb); };
    req.onerror = () => reject(req.error);
  });
}

function sdReadAll() {
  return sdIdb().then((db) => new Promise((resolve, reject) => {
    const tx = db.transaction(SD_STORE, 'readonly');
    const store = tx.objectStore(SD_STORE);
    const keys = store.getAllKeys();
    const values = store.getAll();
    tx.oncomplete = () => resolve(keys.result.map((k, i) => [k, values.result[i]]));
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  }));
}

// `entries` is [path, value] pairs; a null value deletes the path.
function sdApply(entries) {
  return sdIdb().then((db) => new Promise((resolve, reject) => {
    const tx = db.transaction(SD_STORE, 'readwrite');
    const store = tx.objectStore(SD_STORE);
    for (const [path, value] of entries) {
      if (value === null) store.delete(path);
      else store.put(value, path);
    }
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  }));
}

// Ask the browser not to evict the card under storage pressure. Without this
// IndexedDB is best-effort and a save can quietly disappear.
async function sdRequestPersistence() {
  if (!navigator.storage || !navigator.storage.persist) return;
  try {
    if (await navigator.storage.persisted()) return;
    if (!(await navigator.storage.persist())) {
      log('SD card: storage is not marked persistent - the browser may evict it.', 'dim');
    }
  } catch { /* not fatal: the card still works for this session */ }
}

// Put the stored card back into a fresh session. Restores through the host
// entry points, which do not count as guest changes, so this does not
// immediately queue everything to be written straight back.
async function sdRestore() {
  let entries;
  try {
    entries = await sdReadAll();
  } catch (err) {
    log('SD card: could not be read (' + err + ')', 'err');
    return;
  }
  if (!entries.length) return;
  // Directories first, so one the guest left empty survives on its own.
  entries.sort((a, b) => (a[1].kind === b[1].kind ? 0 : a[1].kind === 'dir' ? -1 : 1));
  for (const [path, value] of entries) {
    if (value.kind === 'dir') await call('sd_create_dir', path);
    else await call('sd_write_file', path, value.data || new Uint8Array(0));
  }
  log('SD card: restored ' + entries.length + ' entries', 'dim');
}

// Write back what the guest changed. Cheap when it changed nothing, which is
// almost every slice.
async function sdFlush() {
  if (sdFlushing || handle < 0) return;
  sdFlushing = true;
  try {
    for (const change of await call('sd_take_changes')) {
      if (change.kind === 'deleted') sdBacklog.set(change.path, null);
      else if (change.kind === 'dir') sdBacklog.set(change.path, { kind: 'dir' });
      else {
        const data = await call('sd_read_file', change.path);
        sdBacklog.set(change.path, { kind: 'file', data: data || new Uint8Array(0) });
      }
    }
    if (sdBacklog.size) {
      await sdApply([...sdBacklog]);
      sdBacklog.clear();
    }
  } catch (err) {
    log('SD card: could not be written (' + err + ') - retrying on the next flush.', 'err');
  } finally {
    sdFlushing = false;
  }
}

async function init() {
  initWorker();
  await readyPromise;
  handle = await call('new');
  await stageFont();
  await sdRequestPersistence();
  await sdRestore();
  fbW = await call('fb_width');
  fbH = await call('fb_height');
  fbBytes = fbW * fbH * 4;
  screenEl.width = fbW;
  screenEl.height = fbH;
  $('wasm-ver').textContent = 'core ready';
  log('core ready', 'dim');
  // Restore persisted keys into the session.
  if (prodKeysText || titleKeysText) {
    await stageKeys();
  }
  updateKeysState();
}

// ---------- program loading ----------

async function loadProgram(file, kind) {
  clearConsole();
  setState('loading');
  const data = new Uint8Array(await file.arrayBuffer());
  let entry;
  try {
    entry = kind === 'nro'
      ? await call('load_nro', data)
      : await call('load_elf', data);
  } catch (err) {
    setState('fault');
    log('Load failed: ' + err.message, 'err');
    return false;
  }
  if (entry < 0) {
    setState('fault');
    log('Load failed: ' + await readLastError(), 'err');
    return false;
  }
  log('Loaded ' + file.name + ' - entry 0x' + entry.toString(16).padStart(8, '0'), 'ok');
  setState('loaded');
  // Hand the stage over to the emulated screen now, not when the first frame
  // arrives: homebrew can run for a long time (or fault) before it presents
  // anything, and leaving the boot splash up until then makes it look as
  // though nothing is happening.
  showScreen();
  await updatePc();
  return true;
}

async function bootFile(file) {
  const kind = /\.nro$/i.test(file.name) ? 'nro' : 'elf';
  if (await loadProgram(file, kind)) await run();
}

for (const id of ['nro-file', 'nro-file-2']) {
  $(id).addEventListener('change', async (e) => {
    const f = e.target.files[0];
    e.target.value = '';
    if (f) await bootFile(f);
  });
}

// Drop an NRO anywhere on the stage to boot it.
let dragDepth = 0;
stageEl.addEventListener('dragenter', (e) => {
  e.preventDefault();
  if (++dragDepth === 1) dropveilEl.classList.add('on');
});
stageEl.addEventListener('dragover', (e) => e.preventDefault());
stageEl.addEventListener('dragleave', () => {
  if (--dragDepth <= 0) { dragDepth = 0; dropveilEl.classList.remove('on'); }
});
stageEl.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropveilEl.classList.remove('on');
  const file = e.dataTransfer.files[0];
  if (file) await bootFile(file);
});

// ---------- run loop ----------

// Run in worker slices so the page can paint and input can reach the emulator
// between them. There is no overall step budget - hbmenu never halts - so the
// loop is driven by the pause flag and by faults.
//
// The slice length *is* the input sampling period: the worker is single
// threaded, so a `set_input` posted mid-slice sits in its queue until
// `switch_run` returns. At the ~23M steps/s the wasm build manages, the old
// 5,000,000 was a 240ms slice, and every keypress waited that long before the
// guest could possibly see it. Slice size costs the interpreter nothing
// (`Cpu::run` is a bare loop with no per-call setup - measured flat from 100k
// to 5M steps), only the round trips below, so this buys ~5x lower input
// latency for ~6% of throughput.
const RUN_SLICE = 1_000_000;
// Slices between panel refreshes. `updatePc`/`drainOutput`/`drainDiagnostics`/
// `sdFlush` are eight postMessage round trips of debug-panel text that nothing
// time-critical reads, so running them once per slice would spend more of the
// budget on chatter than the shorter slice saves.
const HOUSEKEEPING_EVERY = 8;
let running = false;
let pauseRequested = false;
let lastFrame = 0;

const PLAY_GLYPH = '▶';
const PAUSE_GLYPH = '❙❙';

function setRunButton(isRunning) {
  $('run-glyph').textContent = isRunning ? PAUSE_GLYPH : PLAY_GLYPH;
  $('run-label').textContent = isRunning ? 'Pause' : 'Run';
}

async function run() {
  if (running) { pauseRequested = true; return; }
  running = true;
  pauseRequested = false;
  setRunButton(true);
  setState('running');
  const slice = traceCb.checked ? 5000 : RUN_SLICE;
  let steps = 0;
  let tick = 0;
  for (;;) {
    steps = await call('run', slice);
    // Yield so the UI repaints and any queued input is processed.
    await new Promise((r) => setTimeout(r, 0));
    // `Cpu::run` only stops short of its budget when the machine halted, so a
    // short slice means this run is over - no separate `halted` round trip.
    const done = steps < 0 || steps < slice;
    // Audio has to track the guest or the stream gaps; the panel does not.
    await pumpAudio();
    if (done || ++tick % HOUSEKEEPING_EVERY === 0) {
      await updatePc();
      await drainOutput();
      await drainDiagnostics();
      await sdFlush();
    }
    // Repaint only when the guest has actually presented a new frame - the
    // snapshot is several megabytes at 1280x720.
    const frames = await call('frame_count');
    if (frames !== lastFrame) {
      countFrames(frames - lastFrame);
      lastFrame = frames;
      await renderFb();
    }
    if (done) break;
    if (pauseRequested) {
      running = false;
      setRunButton(false);
      setState('paused');
      await renderFb();
      return;
    }
  }
  running = false;
  setRunButton(false);
  await finishRun(steps);
}

$('btn-run').addEventListener('click', run);

$('btn-step').addEventListener('click', async () => {
  if (running) { pauseRequested = true; return; }
  const r = await call('run', 1);
  await finishRun(r, true);
  if (traceCb.checked && r >= 0) {
    const t = await drainTrace();
    if (t) log(t.replace(/\n$/, ''), 'dim');
  }
});

$('btn-reset').addEventListener('click', async () => {
  pauseRequested = true;
  running = false;
  setRunButton(false);
  await call('free_session');
  handle = await call('new');
  await stageFont();
  await sdRestore();
  // Everything else the page is still showing. A new session starts with no
  // keys, no container and no data archives, while the panel above goes on
  // reporting all three -- so Launch failed with "no container is open" on a
  // card that was still sitting on screen.
  await stageKeys();
  await restoreArchives();
  await reopenContainer();
  resetAudio();
  clearConsole();
  lastFrame = 0;
  fbW = fbH = 0;
  $('res').textContent = '—';
  setState('idle');
  showOverlay(true);
  screenCtx.clearRect(0, 0, screenEl.width, screenEl.height);
  await updatePc();
});

// Space toggles run/pause and backtick toggles the panel - but not while the
// user is typing into one of the panel's inputs.
window.addEventListener('keydown', (e) => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName || '')) return;
  if (e.code === 'Space') { e.preventDefault(); run(); }
  else if (e.code === 'Backquote') { e.preventDefault(); setPanel(!panelOpen()); }
});

async function drainOutput() {
  const bytes = await call('drain_output');
  if (bytes && bytes.length) {
    log(new TextDecoder().decode(bytes));
  }
}

async function finishRun(steps, stepped) {
  const err = await readLastError();
  if (steps < 0) {
    setState('fault');
    log('CPU fault: ' + err, 'err');
    // The fault trace already carries the register snapshot from the CPU.
    const t = await drainTrace();
    if (t) log(t.replace(/\n$/, ''), 'err');
  } else if (await call('halted')) {
    setState('halted');
    log('Halted (ExitProcess)', 'ok');
    await drainDiagnostics();
  } else if (!stepped) {
    setState('fault');
    log('Stopped unexpectedly.', 'err');
  }
  await drainOutput();
  await sdFlush();
  await renderFb();
  await updatePc();
}

async function updatePc() {
  $('pc').textContent = '0x' + (await call('get_pc')).toString(16).padStart(8, '0');
  $('steps').textContent = (await call('get_cycles')).toLocaleString();
  await updateRam();
}

function formatBytes(n) {
  if (n >= 1024 * 1024 * 1024) return (n / (1024 * 1024 * 1024)).toFixed(2) + ' GiB';
  if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + ' MiB';
  if (n >= 1024) return (n / 1024).toFixed(0) + ' KiB';
  return n + ' B';
}

// Guest RAM is the emulated console's own memory use (pages the guest has
// actually touched); the wasm figure is what the worker's linear memory costs
// the browser, which is the number that matters when a load fails to allocate.
async function updateRam() {
  const ram = await call('ram');
  if (!ram) return;
  $('ram').textContent = `${formatBytes(ram.guest)} (${formatBytes(ram.wasm)})`;
}

// ---------- debug tools ----------

const traceCb = $('trace-cb');
traceCb.addEventListener('change', () => {
  call('set_trace', traceCb.checked ? 1 : 0);
  if (traceCb.checked) log('Tracing enabled - run slices are capped for readability.', 'dim');
});

// The trace buffer carries more than the per-instruction disassembly: the
// emulator records diagnostics there whether or not tracing is enabled -
// services and applet commands a guest asked for that have no implementation
// behind them. There is no stderr in the browser, so this is the only way they
// reach anyone. Drained as the run goes rather than only at the end.
async function drainDiagnostics() {
  const t = await drainTrace();
  if (t) log(t.replace(/\n$/, ''), 'dim');
}

async function drainTrace() {
  const bytes = await call('drain_trace');
  if (bytes && bytes.length) return new TextDecoder().decode(bytes);
  return '';
}

$('btn-dumptrace').addEventListener('click', async () => {
  const t = await drainTrace();
  openPanel('console');
  log(t ? t.replace(/\n$/, '') : '(no trace)', 'dim');
});

$('btn-dumpregs').addEventListener('click', async () => {
  const s = await call('dump_regs');
  openPanel('console');
  if (s) log(s.replace(/\n$/, ''), 'dim');
});

const regIdx = $('reg-idx');
$('btn-readreg').addEventListener('click', async () => {
  const v = await call('get_reg', parseInt(regIdx.value, 10));
  $('reg-val').textContent = v;
});

// ---------- NSP container ----------

const nspDrop = $('nsp-drop');
nspDrop.addEventListener('dragover', (e) => { e.preventDefault(); nspDrop.classList.add('drag'); });
nspDrop.addEventListener('dragleave', () => nspDrop.classList.remove('drag'));
nspDrop.addEventListener('drop', (e) => {
  e.preventDefault();
  nspDrop.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file) handleContainerFile(file);
});
$('nsp-file').addEventListener('change', (e) => { if (e.target.files[0]) handleContainerFile(e.target.files[0]); });

function handleContainerFile(file) {
  if (/\.nca$/i.test(file.name)) return handleStandaloneNca(file);
  return handleNspFile(file);
}

// The container the wasm side has open, kept so a new session can be handed
// the same one. Only the File is held here; nothing is read from it.
let openContainer = null;

// Give a fresh session the container the page is still showing. Reset means
// "run this again from the top", not "throw away the file I just picked" --
// the NSP/NCA card and its Launch button survive a reset either way, and a
// Launch that then reports "no container is open" is the page lying about its
// own state.
async function reopenContainer() {
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
async function handleNspFile(file) {
  clearNsp();
  log('Opening ' + file.name + ' (' + fmtSize(file.size) + ') ...');
  try {
    const ok = await call('open_nsp', file);
    if (ok !== 0) {
      log('NSP error: ' + await readLastError(), 'err');
      return;
    }
    openContainer = { file, kind: 'nsp' };
  } catch (e) {
    log('Could not open ' + file.name + ': ' + e.message, 'err');
    return;
  }
  const files = JSON.parse(await call('nsp_files_json'));
  log('Parsed ' + files.length + ' file(s). Click an .nca to inspect it.', 'ok');

  const ul = el('ul', 'nsp-list');
  files.forEach((f, index) => {
    const li = el('li');
    li.appendChild(el('span', null, f.name));
    li.appendChild(el('span', 'size', fmtSize(f.size)));
    if (/\.nca$/i.test(f.name)) {
      li.style.cursor = 'pointer';
      li.addEventListener('click', () => inspectNca(f, index));
    }
    ul.appendChild(li);
  });
  $('nsp-result').appendChild(ul);
  await showTitleCard(() => call('load_control_from_nsp'));
}

function clearNsp() { $('nsp-result').textContent = ''; }

// ---------- title details ----------

/* What a console's home menu shows for a title - its icon, name and publisher
   - plus the rest of what its NACP declares, read from the Control NCA that
   ships alongside the Program NCA in every container.

   Needs prod.keys, and not just for the RomFS: an NCA's content type lives in
   its encrypted header, so without the header key the Control NCA can't even
   be picked out of the container. A container that has none is unremarkable
   (an update or DLC package may ship without one), so this is a dim note
   rather than an error. */
async function showTitleCard(loader) {
  let info;
  try {
    if (await loader() !== 0) {
      log('No title details: ' + await readLastError(), 'dim');
      return null;
    }
    info = JSON.parse(await call('control_json'));
  } catch (err) {
    log('No title details: ' + err.message, 'dim');
    return null;
  }
  if (!info.name) return null;
  const icon = info.icon_size > 0 ? await call('control_icon', info.icon_size) : null;
  const card = renderTitleCard(info, icon);
  $('nsp-result').prepend(card);
  log('Title: ' + info.name + (info.publisher ? ' - ' + info.publisher : ''), 'ok');
  return info;
}

function renderTitleCard(info, icon) {
  const card = el('div', 'title-card');
  if (icon && icon.length) {
    const img = el('img', 'title-icon');
    img.alt = info.name;
    const url = URL.createObjectURL(new Blob([icon], { type: info.icon_mime }));
    // The decoded image outlives the URL, so release it as soon as it has
    // been read rather than leaking a blob per inspected container.
    img.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
    img.src = url;
    card.appendChild(img);
  }
  const meta = el('div', 'title-meta');
  meta.appendChild(el('div', 'title-name', info.name));
  if (info.publisher) meta.appendChild(el('div', 'title-publisher', info.publisher));
  const tags = [];
  if (info.version) tags.push('v' + info.version);
  if (info.demo) tags.push('demo');
  tags.push(info.title_id);
  meta.appendChild(el('div', 'title-tags', tags.join(' \u00b7 ')));
  card.appendChild(meta);

  const details = el('div', 'nca-info');
  appendRows(details, titleRows(info));
  card.appendChild(details);
  return card;
}

/* The NACP fields worth showing, skipping the ones this title left unset -
   most titles set only a handful, and a column of zeroes says nothing. */
function titleRows(info) {
  const rows = [];
  const push = (k, v) => { if (v) rows.push([k, v]); };
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
function saveDataSummary(info) {
  const part = (label, size, journal) => {
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

function appendRows(out, rows) {
  for (const [k, v] of rows) {
    const row = el('div');
    row.appendChild(el('span', 'k', k + ':'));
    row.append(' ' + v);
    out.appendChild(row);
  }
}

async function inspectNca(f, index) {
  // Replace any previous inspection result instead of stacking them up.
  $('nsp-result').querySelectorAll('.nca-info').forEach((node) => node.remove());
  const out = el('div', 'nca-info', 'Parsing ' + f.name + ' ...');
  $('nsp-result').appendChild(out);

  // 0xC00 covers the base header plus all 4 per-section FS headers (needed
  // for an accurate fs_type in the display below) - still tiny next to the
  // (possibly hundreds-of-MB) payload, so no need to copy the whole file.
  const headerLen = Math.min(f.size, 0xC00);
  let header;
  try {
    header = await call('read_file', index, 0, headerLen);
  } catch (err) {
    out.textContent = 'read failed: ' + err.message;
    return;
  }
  await parseAndRenderNca(out, header, f.name, () => launchNca(f, index));
}

// Drop/browse a standalone .nca (not inside an NSP): same inspect-then-Launch
// flow, with the NCA itself as the open container instead of a file inside
// one. Opening it is what lets Launch - and the Control NCA card below - read
// from it later.
async function handleStandaloneNca(file) {
  clearNsp();
  const out = el('div', 'nca-info', 'Parsing ' + file.name + ' ...');
  $('nsp-result').appendChild(out);
  try {
    await call('open_nca', file);
  } catch (e) {
    out.textContent = 'Could not open ' + file.name + ': ' + e.message;
    return;
  }
  openContainer = { file, kind: 'nca' };
  const headerLen = Math.min(file.size, 0xC00);
  const header = new Uint8Array(await file.slice(0, headerLen).arrayBuffer());
  const info = await parseAndRenderNca(out, header, file.name, () => launchStandaloneNca(file));
  // A standalone Control NCA is nothing but the title's icon and metadata, so
  // the same card the container path shows is the whole point of opening one.
  if (info && info.content_type === 'Control') {
    await showTitleCard(() => call('load_control_from_nca'));
  }
}

async function parseAndRenderNca(out, header, name, onLaunch) {
  let info;
  try {
    info = JSON.parse(await call('parse_nca', header));
  } catch (err) {
    out.textContent = 'parse failed: ' + err.message;
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
  const rows = [
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
    const btn = el('button', 'btn small', 'Launch');
    btn.addEventListener('click', onLaunch);
    out.appendChild(btn);
  }
  return info;
}

// Decrypts NSP file `index` as a Program NCA and boots its ExeFS `main`
// executable. This gets a real title only as far as its own crt0 - there is
// no Horizon service surface for a full retail SDK program yet (that's a much
// larger undertaking than the homebrew this emulator otherwise runs), so
// expect it to run until the first missing service rather than reach a menu.
async function launchNca(f, index) {
  return doLaunchNca(f.name, () => call('load_nca_from_nsp', index));
}

// Same as `launchNca`, but for a standalone .nca file: it is already the open
// container, so there is nothing to read here that booting won't read itself.
async function launchStandaloneNca(file) {
  return doLaunchNca(file.name, () => call('load_nca'));
}

async function doLaunchNca(name, loadFn) {
  clearConsole();
  setState('loading');
  let entry;
  try {
    entry = await loadFn();
  } catch (err) {
    setState('fault');
    log('Launch failed: ' + err.message, 'err');
    return;
  }
  if (entry < 0) {
    setState('fault');
    log('Launch failed: ' + await readLastError(), 'err');
    return;
  }
  log('Launched ' + name + ' - entry 0x' + entry.toString(16).padStart(8, '0'), 'ok');
  log('Decrypted and booted the title\'s own executable; there is no Horizon service support for retail games yet, so expect it to run until the first missing service rather than reach a menu.', 'dim');
  setState('loaded');
  showScreen();
  await updatePc();
  await run();
}

// ---------- keys ----------

// Keys are persisted in localStorage so they survive page reloads (they're
// just text; they never leave the browser).
const KEYS_STORE = { prod: 'switch-prod-keys', title: 'switch-title-keys' };

let prodKeysText = localStorage.getItem(KEYS_STORE.prod) || '';
let titleKeysText = localStorage.getItem(KEYS_STORE.title) || '';
let restoredKeys = Boolean(prodKeysText || titleKeysText);

async function stageKeys() {
  if (!ready) return;
  const prod = prodKeysText ? new TextEncoder().encode(prodKeysText) : null;
  const title = titleKeysText ? new TextEncoder().encode(titleKeysText) : null;
  const rc = await call('load_keys', prod, title);
  updateKeysState();
  return rc;
}

function updateKeysState() {
  const parts = [];
  if (prodKeysText) parts.push('prod.keys');
  if (titleKeysText) parts.push('title.keys');
  $('keys-state').textContent = parts.length === 0
    ? 'no keys loaded - encrypted NCA headers can\'t be inspected'
    : 'loaded: ' + parts.join(' + ') + (restoredKeys ? ' (from storage)' : '');
}

$('prod-keys').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  prodKeysText = await f.text();
  restoredKeys = false;
  localStorage.setItem(KEYS_STORE.prod, prodKeysText);
  await stageKeys();
  log('prod.keys loaded - NCA header decryption enabled.', 'ok');
});
$('title-keys').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  titleKeysText = await f.text();
  restoredKeys = false;
  localStorage.setItem(KEYS_STORE.title, titleKeysText);
  await stageKeys();
  log('title.keys loaded.', 'ok');
});
// ---------- system data archives ----------

/* Content a title mounts that is not its own: an applet's shared assets, the
   system's Mii and amiibo models. Each is a separate NCA on a console's NAND,
   so there is nothing to find here unless someone hands them over - point this
   at a firmware dump and every data archive in it is registered by title id.

   Only the File references cross over; nothing is read until a title asks for
   one, so selecting a few hundred NCAs costs nothing. They cannot be
   persisted the way keys are - the browser will not hand a page a file again
   without being asked - so this is per session. */
let archiveCount = 0;
// The archives themselves, so a new session can be given them again. The
// browser will not hand a page a file it was not asked for, so losing these
// on reset would mean re-picking a whole firmware dump.
let firmwareFiles = [];

// Re-register every archive the page still claims to have. Runs before the
// container is re-opened and after the keys are re-staged, because parsing an
// NCA header needs them.
async function restoreArchives() {
  if (!firmwareFiles.length) return;
  const kept = [];
  for (const f of firmwareFiles) {
    if (await call('add_archive', f).catch(() => -1) === 0) kept.push(f);
  }
  firmwareFiles = kept;
  archiveCount = kept.length;
  updateFirmwareState();
}

function updateFirmwareState() {
  $('firmware-state').textContent = archiveCount === 0
    ? 'no system data archives - a title that mounts one (an applet\'s shared assets, the Mii and amiibo models) will not find it'
    : archiveCount + ' system data archive(s) registered';
}

$('firmware-ncas').addEventListener('change', async (e) => {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;
  log('Reading ' + files.length + ' firmware file(s) ...');
  let added = 0;
  for (const f of files) {
    try {
      if (await call('add_archive', f) === 0) { added++; firmwareFiles.push(f); }
    } catch (err) {
      log('Could not read ' + f.name + ': ' + err.message, 'err');
    }
  }
  archiveCount = firmwareFiles.length;
  updateFirmwareState();
  // Most of a firmware dump is programs and metadata, not data archives; only
  // the ones that are get registered, so the skipped count is expected.
  log('Registered ' + added + ' system data archive(s) of ' + files.length + ' file(s).',
    added ? 'ok' : 'dim');
});

$('btn-clear-keys').addEventListener('click', () => {
  prodKeysText = '';
  titleKeysText = '';
  localStorage.removeItem(KEYS_STORE.prod);
  localStorage.removeItem(KEYS_STORE.title);
  restoredKeys = false;
  stageKeys();
  log('Keys cleared.', 'dim');
});

// ---------- controller input ----------
// HidNpadButton bitfield, as the emulated program expects (switch_set_input).
// The order is Horizon's, not the browser's: face buttons, stick presses,
// shoulders, triggers, plus/minus, then the d-pad.
const BTN = {
  A: 1 << 0, B: 1 << 1, X: 1 << 2, Y: 1 << 3,
  STICK_L: 1 << 4, STICK_R: 1 << 5,
  L: 1 << 6, R: 1 << 7, ZL: 1 << 8, ZR: 1 << 9,
  PLUS: 1 << 10, MINUS: 1 << 11,
  LEFT: 1 << 12, UP: 1 << 13, RIGHT: 1 << 14, DOWN: 1 << 15,
};
function inputStatus(text) {
  $('input-state').textContent = text;
}

// Keyboard fallback: dpad + A/B/X/Y + start/select.
const KEY_MAP = {
  arrowleft: BTN.LEFT, arrowup: BTN.UP, arrowright: BTN.RIGHT, arrowdown: BTN.DOWN,
  enter: BTN.PLUS, shift: BTN.MINUS,
  z: BTN.A, x: BTN.B, a: BTN.X, s: BTN.Y,
  q: BTN.L, e: BTN.R,
};
const keysDown = new Set();
// Pushed on the edge as well as on the poll below: the 16ms tick is there for
// the gamepad, which can only be sampled, but a key press *is* an event and
// waiting up to a tick to forward it is latency for nothing. The worker
// coalesces whatever arrives before its next slice boundary.
window.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();
  if (!KEY_MAP[key]) return;
  e.preventDefault();
  // Auto-repeat is not a new press - but it is the only evidence a key is
  // still down after `blur` cleared the set, so go by the set, not `e.repeat`.
  if (keysDown.has(key)) return;
  keysDown.add(key);
  pushInput();
});
window.addEventListener('keyup', (e) => {
  if (keysDown.delete(e.key.toLowerCase())) pushInput();
});
window.addEventListener('blur', () => {
  if (!keysDown.size && !touchPoints.size) return;
  keysDown.clear();
  touchPoints.clear();
  pushInput();
});

function keyboardMask() {
  let m = 0;
  for (const code of keysDown) m |= KEY_MAP[code] || 0;
  return m;
}

// ---------- touchscreen ----------
//
// hid reports touches in the console's own 1280x720 digitizer space whatever
// resolution the guest is presenting at (TOUCH_SCREEN_WIDTH/HEIGHT in
// cpu/mod.rs), so the canvas is mapped onto that rather than the other way
// round. Touch is a handheld-only input on real hardware and this console
// always reports AppletOperationMode_Handheld, so it is always live.
const TOUCH_W = 1280;
const TOUCH_H = 720;
const TOUCH_MAX = 16;

// pointerId -> { slot, x, y }. `slot` is the finger id the guest sees: it has
// to stay put for the life of the contact so a title can follow a drag, which
// is why it is claimed from the lowest free one instead of being the pointer's
// position in the map.
const touchPoints = new Map();
let touchWasDown = false;

function claimTouchSlot() {
  const taken = new Set([...touchPoints.values()].map((t) => t.slot));
  for (let i = 0; i < TOUCH_MAX; i++) if (!taken.has(i)) return i;
  return -1;
}

// The canvas element fills the stage but `object-fit: contain` letterboxes the
// guest's frame inside it, so a tap has to be mapped through the *contained*
// rect - going by the element box offsets every tap by the size of the bars.
// Returns null for a tap that landed on a bar rather than on the screen.
function touchAt(e) {
  const rect = screenEl.getBoundingClientRect();
  const iw = screenEl.width, ih = screenEl.height;
  if (!iw || !ih || !rect.width || !rect.height) return null;
  const scale = Math.min(rect.width / iw, rect.height / ih);
  const dw = iw * scale, dh = ih * scale;
  const x = (e.clientX - rect.left - (rect.width - dw) / 2) / dw;
  const y = (e.clientY - rect.top - (rect.height - dh) / 2) / dh;
  if (x < 0 || x >= 1 || y < 0 || y >= 1) return null;
  return {
    x: Math.min(TOUCH_W - 1, Math.floor(x * TOUCH_W)),
    y: Math.min(TOUCH_H - 1, Math.floor(y * TOUCH_H)),
  };
}

function touchTriples() {
  const out = new Uint32Array(touchPoints.size * 3);
  let i = 0;
  for (const t of touchPoints.values()) {
    out[i++] = t.slot;
    out[i++] = t.x;
    out[i++] = t.y;
  }
  return out;
}

screenEl.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return; // a right-click is not a finger
  const p = touchAt(e);
  if (!p) return;
  const slot = claimTouchSlot();
  if (slot < 0) return; // all sixteen contacts are already down
  touchPoints.set(e.pointerId, { slot, x: p.x, y: p.y });
  // Capture so a finger that slides off the canvas still reports its lift here
  // rather than leaving a contact down forever.
  try { screenEl.setPointerCapture(e.pointerId); } catch {}
  e.preventDefault();
  pushInput();
});

screenEl.addEventListener('pointermove', (e) => {
  const t = touchPoints.get(e.pointerId);
  if (!t) return;
  const p = touchAt(e);
  // A finger dragged into the letterbox holds its last on-screen position
  // instead of lifting, which is what the bezel does on the console.
  if (p) { t.x = p.x; t.y = p.y; }
  e.preventDefault();
});

function liftTouch(e) {
  if (touchPoints.delete(e.pointerId)) pushInput();
}
screenEl.addEventListener('pointerup', liftTouch);
screenEl.addEventListener('pointercancel', liftTouch);

function pushInput() {
  if (!ready) return;
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  const pad = pads.find((p) => p && p.connected);
  let mask = keyboardMask();
  let slx = 0, sly = 0, srx = 0, sry = 0;
  if (pad) {
    // Standard button order: 0-3 = bottom/right/top/left (B/A/Y/X), 4-7 = L/R/ZL/ZR,
    // 8-9 = select/start, 10-11 = stick presses, 12-17 = dpad.
    if (pad.buttons[0]?.pressed) mask |= BTN.B;
    if (pad.buttons[1]?.pressed) mask |= BTN.A;
    if (pad.buttons[2]?.pressed) mask |= BTN.Y;
    if (pad.buttons[3]?.pressed) mask |= BTN.X;
    if (pad.buttons[4]?.pressed) mask |= BTN.L;
    if (pad.buttons[5]?.pressed) mask |= BTN.R;
    if (pad.buttons[6]?.pressed) mask |= BTN.ZL;
    if (pad.buttons[7]?.pressed) mask |= BTN.ZR;
    if (pad.buttons[8]?.pressed) mask |= BTN.MINUS;
    if (pad.buttons[9]?.pressed) mask |= BTN.PLUS;
    if (pad.buttons[10]?.pressed) mask |= BTN.STICK_L;
    if (pad.buttons[11]?.pressed) mask |= BTN.STICK_R;
    if (pad.buttons[12]?.pressed) mask |= BTN.UP;
    if (pad.buttons[13]?.pressed) mask |= BTN.DOWN;
    if (pad.buttons[14]?.pressed) mask |= BTN.LEFT;
    if (pad.buttons[15]?.pressed) mask |= BTN.RIGHT;
    // Analog sticks: -32768..32767, deadzone ~15%. Horizon's Y axis points up,
    // the browser's points down, so the vertical axes are negated. The emulator
    // derives the stick pseudo-buttons (which is what menus navigate with) from
    // these values, so they must arrive with the console's sign convention.
    const dz = 0.15;
    const axes = pad.axes || [];
    const axis = (i) => (Math.abs(axes[i] || 0) > dz ? axes[i] : 0);
    slx = Math.round(axis(0) * 32767); sly = Math.round(-axis(1) * 32767);
    srx = Math.round(axis(2) * 32767); sry = Math.round(-axis(3) * 32767);
    inputStatus('gamepad');
  } else if (mask) {
    inputStatus('keyboard');
  }
  call('set_input', mask, slx, sly, srx, sry);
  // Only while something is down, plus the single push that reports the lift -
  // an idle screen has nothing to say 60 times a second.
  if (touchPoints.size || touchWasDown) {
    call('set_touch', touchTriples());
    touchWasDown = touchPoints.size > 0;
  }
  if (touchPoints.size) inputStatus('touch');
  pullVibration(pad);
}

// ---------- audio ----------

// `audout` hands the guest's PCM over interleaved, at whatever rate and
// channel count it opened the device with. Each pump takes everything that has
// queued up since the last one and schedules it as a single buffer, butted up
// against the end of the previous one, so a continuous stream stays
// continuous. The emulator rarely runs a retail title in real time, so
// underruns are the normal case: the cursor simply restarts a little ahead of
// `currentTime` rather than trying to stretch anything to cover the gap.
let audioCtx = null;
let audioCursor = 0;
// One second of 48 kHz stereo, matching the cap the core queues.
const AUDIO_MAX_PULL = 96000;

function resetAudio() {
  audioCursor = 0;
}

async function pumpAudio() {
  const packed = await call('audio_format');
  if (!packed) return; // nothing has opened an audio device yet
  const rate = packed & 0x00ffffff;
  const channels = packed >>> 24;
  if (!rate || !channels) return;
  const bytes = await call('audio_pull', AUDIO_MAX_PULL);
  if (!bytes || bytes.length < channels * 2) return;
  if (!audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    audioCtx = new Ctx();
  }
  // Autoplay policy: a context created before the first gesture starts
  // suspended and stays silent until resumed.
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  const pcm = bytes.byteOffset % 2
    ? new Int16Array(bytes.slice().buffer)
    : new Int16Array(bytes.buffer, bytes.byteOffset, bytes.length >> 1);
  const frames = Math.floor(pcm.length / channels);
  if (!frames) return;
  const buffer = audioCtx.createBuffer(channels, frames, rate);
  for (let c = 0; c < channels; c++) {
    const out = buffer.getChannelData(c);
    for (let i = 0; i < frames; i++) out[i] = pcm[i * channels + c] / 32768;
  }
  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(audioCtx.destination);
  // Schedule a little ahead of now so a late buffer is not clipped, then keep
  // every later one flush against its predecessor.
  const start = Math.max(audioCtx.currentTime + 0.05, audioCursor);
  src.start(start);
  audioCursor = start + buffer.duration;
}

// ---------- rumble ----------

// Switch rumble drives two linear resonant actuators independently, and the
// Gamepad API's "dual-rumble" effect is the same shape: the guest's low band
// becomes strongMagnitude, its high band weakMagnitude. Only Chromium-family
// browsers implement vibrationActuator, so this is best-effort and silent
// where it is missing.
let lastRumble = -1;
async function pullVibration(pad) {
  const actuator = pad?.vibrationActuator;
  if (!actuator?.playEffect) return;
  const packed = await call('vibration');
  if (packed === lastRumble) return;   // re-issuing the same effect stutters it
  lastRumble = packed;
  const strong = (packed & 0xffff) / 1000;
  const weak = (packed >>> 16) / 1000;
  try {
    if (strong === 0 && weak === 0) {
      await actuator.reset?.();
    } else {
      // Outlive the poll interval so a held rumble is continuous rather than
      // a stutter, but stay short enough that it stops promptly when the
      // guest lets go.
      await actuator.playEffect('dual-rumble', {
        duration: 120,
        strongMagnitude: strong,
        weakMagnitude: weak,
      });
    }
  } catch {
    // A browser that advertises the actuator but refuses the effect.
  }
}

setInterval(pushInput, 16);
window.addEventListener('gamepadconnected', () => inputStatus('gamepad connected'));
window.addEventListener('gamepaddisconnected', () => inputStatus('none'));

// ---------- host battery ----------

// Feeds the Switch's psm (power management) service. Only Chromium exposes
// the Battery Status API — Firefox and Safari never shipped it, over privacy
// concerns — so elsewhere the emulated battery just stays at the wasm
// default (full, charging). Event-driven rather than polled: battery level
// changes far slower than the 16ms input tick, and the level/charging state
// is cached worker-side so a freshly created session (including after
// "reset") picks it up without this having to fire again.
if (navigator.getBattery) {
  navigator.getBattery().then((battery) => {
    const push = () => call('set_battery', Math.round(battery.level * 100), battery.charging ? 1 : 0);
    push();
    battery.addEventListener('levelchange', push);
    battery.addEventListener('chargingchange', push);
  });
}

init();
