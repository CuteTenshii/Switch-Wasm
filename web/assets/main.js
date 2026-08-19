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

async function init() {
  initWorker();
  await readyPromise;
  handle = await call('new');
  await call('set_syscall_mode', 2); // Horizon
  await stageFont();
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

// The frontend only runs real Horizon homebrew; the legacy UART demo ABI is
// removed from the UI so sdl-hello/hbmenu can't accidentally execute under it.
function applySyscallMode() {
  return call('set_syscall_mode', 2); // Horizon
}

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
  await applySyscallMode();
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

$('btn-demo').addEventListener('click', async () => {
  const name = $('asset-nro').value;
  const res = await fetch('assets/' + name);
  if (!res.ok) { log('Fetch failed: ' + name, 'err'); return; }
  const data = await res.arrayBuffer();
  await bootFile(new File([data], name));
});

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
const RUN_SLICE = 5_000_000;
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
  for (;;) {
    steps = await call('run', slice);
    // Yield so the UI repaints and any queued input is processed.
    await new Promise((r) => setTimeout(r, 0));
    await updatePc();
    await drainOutput();
    // Repaint only when the guest has actually presented a new frame - the
    // snapshot is several megabytes at 1280x720.
    const frames = await call('frame_count');
    if (frames !== lastFrame) {
      countFrames(frames - lastFrame);
      lastFrame = frames;
      await renderFb();
    }
    if (steps < 0) break;
    if (await call('halted')) break;
    if (steps < slice) break;
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
  await applySyscallMode();
  await stageFont();
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
    if (traceCb.checked) {
      const t = await drainTrace();
      if (t) log(t.replace(/\n$/, ''), 'dim');
    }
  } else if (!stepped) {
    setState('fault');
    log('Stopped unexpectedly.', 'err');
  }
  await drainOutput();
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

async function handleNspFile(file) {
  clearNsp();
  log('Reading ' + file.name + ' (' + fmtSize(file.size) + ') ...');
  const data = new Uint8Array(await file.arrayBuffer());
  try {
    const ok = await call('load_nsp', data);
    if (ok !== 0) {
      log('NSP error: ' + await readLastError(), 'err');
      return;
    }
  } catch (e) {
    log('Failed to stage ' + fmtSize(data.length) + ' in the emulator: ' + e.message, 'err');
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
}

function clearNsp() { $('nsp-result').textContent = ''; }

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
// flow, but the header slice comes straight off the browser File object
// instead of a staged NSP buffer.
async function handleStandaloneNca(file) {
  clearNsp();
  const out = el('div', 'nca-info', 'Parsing ' + file.name + ' ...');
  $('nsp-result').appendChild(out);
  const headerLen = Math.min(file.size, 0xC00);
  const header = new Uint8Array(await file.slice(0, headerLen).arrayBuffer());
  await parseAndRenderNca(out, header, file.name, () => launchStandaloneNca(file));
}

async function parseAndRenderNca(out, header, name, onLaunch) {
  let info;
  try {
    info = JSON.parse(await call('parse_nca', header));
  } catch (err) {
    out.textContent = 'parse failed: ' + err.message;
    return;
  }
  if (info.error) {
    // A CDN NCA stores its header encrypted with the header key, so the NCA3
    // magic at 0x200 is invisible until it's decrypted - surface that clearly
    // instead of a bare "bad magic", and point at the keys files.
    out.textContent = /bad magic/.test(info.error)
      ? 'NCA header is encrypted - load prod.keys to decrypt and inspect. (' + info.error + ')'
      : 'NCA: ' + info.error;
    return;
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
  for (const [k, v] of rows) {
    const row = el('div');
    row.appendChild(el('span', 'k', k + ':'));
    row.append(' ' + v);
    out.appendChild(row);
  }
  if (info.content_type === 'Program') {
    const btn = el('button', 'btn small', 'Launch');
    btn.addEventListener('click', onLaunch);
    out.appendChild(btn);
  }
}

// Decrypts NSP file `index` as a Program NCA and boots its ExeFS `main`
// executable. This gets a real title only as far as its own crt0 - there is
// no Horizon service surface for a full retail SDK program yet (that's a much
// larger undertaking than the homebrew this emulator otherwise runs), so
// expect it to run until the first missing service rather than reach a menu.
async function launchNca(f, index) {
  return doLaunchNca(f.name, () => call('load_nca_from_nsp', index));
}

// Same as `launchNca`, but for a standalone .nca file: the whole file has to
// be read and staged now (Launch is the first point a standalone NCA needs
// its full bytes, not just the header).
async function launchStandaloneNca(file) {
  return doLaunchNca(file.name, async () => {
    log('Reading ' + file.name + ' (' + fmtSize(file.size) + ') ...');
    const data = new Uint8Array(await file.arrayBuffer());
    return call('load_nca', data);
  });
}

async function doLaunchNca(name, loadFn) {
  clearConsole();
  setState('loading');
  await applySyscallMode();
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
  ArrowLeft: BTN.LEFT, ArrowUp: BTN.UP, ArrowRight: BTN.RIGHT, ArrowDown: BTN.DOWN,
  Enter: BTN.PLUS, ShiftLeft: BTN.MINUS, ShiftRight: BTN.MINUS,
  KeyZ: BTN.A, KeyX: BTN.B, KeyA: BTN.X, KeyS: BTN.Y,
  KeyQ: BTN.L, KeyE: BTN.R,
};
const keysDown = new Set();
window.addEventListener('keydown', (e) => {
  if (KEY_MAP[e.code]) { keysDown.add(e.code); e.preventDefault(); }
});
window.addEventListener('keyup', (e) => keysDown.delete(e.code));
window.addEventListener('blur', () => keysDown.clear());

function keyboardMask() {
  let m = 0;
  for (const code of keysDown) m |= KEY_MAP[code] || 0;
  return m;
}

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
