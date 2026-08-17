/* switch-wasm browser frontend. The emulator runs in a web worker
   (worker.js) so long executions don't freeze the page; this file is a thin
   promise-based RPC client over postMessage. Buffers (files, framebuffer,
   console/trace output) are transferred across the worker boundary. */

const $ = (id) => document.getElementById(id);

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
  worker = new Worker('worker.js');
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

const consoleEl = $('console');
const screenEl = $('screen');
const screenCtx = screenEl.getContext('2d');
let fbW = 0, fbH = 0, fbBytes = 0;
const TAG = '[switch-wasm]';
function log(msg, cls) {
  // Real browser console (DevTools): route by severity for filterability.
  if (cls === 'err') console.error(TAG, msg);
  else if (cls === 'ok') console.info(TAG, msg);
  else if (cls === 'dim') console.debug(TAG, msg);
  else console.log(TAG, msg);
  // On-page console mirror.
  const div = document.createElement('div');
  if (cls) div.className = cls;
  div.textContent = msg;
  consoleEl.appendChild(div);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}
function clearConsole() { consoleEl.textContent = ''; }

async function readLastError() { return await call('last_error'); }

function fmtSize(n) {
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(2) + ' GiB';
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(2) + ' MiB';
  if (n >= 1 << 10) return (n / (1 << 10)).toFixed(1) + ' KiB';
  return n + ' B';
}

function setState(text) {
  $('state').textContent = text;
}

// Copy the emulated screen into the canvas. Before anything is presented this
// is the memory-mapped demo framebuffer; once the guest hands a frame to the
// display it becomes the real console output, so the canvas is resized to
// whatever resolution the guest chose (1280x720 for most homebrew).
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
  const pixels = await call('fb_snapshot', fbBytes);
  if (pixels && pixels.length >= fbBytes) {
    const arr = new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, fbBytes);
    screenCtx.putImageData(new ImageData(arr, fbW, fbH), 0, 0);
  }
}

// ---------- boot ----------

async function init() {
  initWorker();
  await readyPromise;
  handle = await call('new');
  await call('set_syscall_mode', 2); // Horizon
  fbW = await call('fb_width');
  fbH = await call('fb_height');
  fbBytes = fbW * fbH * 4;
  screenEl.width = fbW;
  screenEl.height = fbH;
  log('core ready');
  $('wasm-ver').textContent = 'core ready (worker)';
  // Restore persisted keys into the session.
  if (prodKeysText || titleKeysText) {
    await stageKeys();
  }
}

// ---------- NSP container ----------

const nspDrop = $('nsp-drop');
nspDrop.addEventListener('dragover', (e) => { e.preventDefault(); nspDrop.classList.add('drag'); });
nspDrop.addEventListener('dragleave', () => nspDrop.classList.remove('drag'));
nspDrop.addEventListener('drop', (e) => {
  e.preventDefault();
  nspDrop.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file) handleNspFile(file);
});
$('nsp-file').addEventListener('change', (e) => { if (e.target.files[0]) handleNspFile(e.target.files[0]); });

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

  const ul = document.createElement('ul');
  ul.className = 'nsp-list';
  files.forEach((f, index) => {
    const li = document.createElement('li');
    li.innerHTML = '<span></span><span class="size"></span>';
    li.querySelector('span').textContent = f.name;
    li.querySelector('.size').textContent = fmtSize(f.size);
    if (/\.nca$/i.test(f.name)) {
      li.addEventListener('click', () => inspectNca(f, index));
    }
    ul.appendChild(li);
  });
  $('nsp-result').appendChild(ul);
}

function clearNsp() { $('nsp-result').textContent = ''; }

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
  const el = $('keys-state');
  const parts = [];
  if (prodKeysText) parts.push('prod.keys');
  if (titleKeysText) parts.push('title.keys');
  if (parts.length === 0) {
    el.textContent = 'no keys loaded — encrypted NCA headers can\'t be inspected';
  } else {
    el.textContent = 'loaded: ' + parts.join(' + ') + (restoredKeys ? ' (from storage)' : '') +
      ' — encrypted headers will be decrypted';
  }
}

$('prod-keys').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  prodKeysText = await f.text();
  restoredKeys = false;
  localStorage.setItem(KEYS_STORE.prod, prodKeysText);
  await stageKeys();
  log('prod.keys loaded — NCA header decryption enabled.', 'ok');
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

async function inspectNca(f, index) {
  // Replace any previous inspection result instead of stacking them up.
  $('nsp-result').querySelectorAll('.nca-info').forEach((el) => el.remove());
  const out = document.createElement('div');
  out.className = 'nca-info';
  out.textContent = 'Parsing ' + f.name + ' ...';
  $('nsp-result').appendChild(out);

  // Only the first 0x400 bytes (the header) are needed to inspect an NCA —
  // don't copy the whole (possibly hundreds-of-MB) payload to the worker.
  const headerLen = Math.min(f.size, 0x800);
  let header;
  try {
    header = await call('read_file', index, 0, headerLen);
  } catch (err) {
    out.textContent = 'read failed: ' + err.message;
    return;
  }
  let info;
  try {
    info = JSON.parse(await call('parse_nca', header));
  } catch (err) {
    out.textContent = 'parse failed: ' + err.message;
    return;
  }
  out.textContent = '';
  if (info.error) {
    // A CDN NCA stores its header encrypted with the header key, so the NCA3
    // magic at 0x200 is invisible until it's decrypted — surface that clearly
    // instead of a bare "bad magic", and point at the keys files.
    const encrypted = /bad magic/.test(info.error);
    out.textContent = encrypted
      ? 'NCA header is encrypted — load prod.keys above (or pass title keys) to decrypt and inspect. (' + info.error + ')'
      : 'NCA: ' + info.error;
    return;
  }
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
    const div = document.createElement('div');
    div.innerHTML = '<span class="k"></span> ';
    div.querySelector('.k').textContent = k + ':';
    div.append(v);
    out.appendChild(div);
  }
}

// ---------- homebrew runner ----------

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
    setState('error');
    log('Load failed: ' + err.message, 'err');
    return false;
  }
  if (entry < 0) {
    setState('error');
    log('Load failed: ' + await readLastError(), 'err');
    return false;
  }
  log('Loaded ' + file.name + ' — entry 0x' + entry.toString(16).padStart(8, '0'), 'ok');
  log('SVC ABI: Horizon stubs (svcOutputDebugString → console)', 'dim');
  setState('loaded');
  await updatePc();
  return true;
}

$('btn-demo').addEventListener('click', async () => {
  const name = $('asset-nro').value;
  await applySyscallMode();
  const res = await fetch('assets/' + name);
  if (!res.ok) { log('Fetch failed: ' + name, 'err'); return; }
  const data = await res.arrayBuffer();
  const file = new File([data], name);
  if (await loadProgram(file, 'nro')) await run();
});

$('nro-file').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  await applySyscallMode();
  const kind = /\.nro$/i.test(f.name) ? 'nro' : 'elf';
  if (await loadProgram(f, kind)) await run();
});

$('btn-run').addEventListener('click', run);
$('btn-step').addEventListener('click', async () => {
  const r = await call('run', 1);
  await finishRun(r, true);
  if (traceCb.checked && r >= 0) {
    const t = await drainTrace();
    if (t) log(t.replace(/\n$/, ''), 'dim');
  }
});
$('btn-reset').addEventListener('click', async () => {
  await call('free_session');
  handle = await call('new');
  await applySyscallMode();
  clearConsole();
  setState('idle');
});

// Run in worker slices until the app halts or faults. Each slice is short so
// the page can paint and input can reach the emulator between slices; there is
// no overall step budget (trace mode slices are small to keep the log usable).
const RUN_SLICE = 5_000_000;
let lastFrame = 0;
async function run() {
  setState('running');
  const slice = traceCb.checked ? 5000 : RUN_SLICE;
  let steps = 0;
  lastFrame = 0;
  for (;;) {
    steps = await call('run', slice);
    // Yield so the UI repaints and any queued input is processed.
    await new Promise((r) => setTimeout(r, 0));
    // Keep the display live during long runs: hbmenu never halts, so without
    // these the step counter would stay at the post-load value and console
    // output would only appear after the run "ends".
    await updatePc();
    await drainOutput();
    // Repaint only when the guest has actually presented a new frame — the
    // snapshot is several megabytes at 1280x720.
    const frames = await call('frame_count');
    if (frames !== lastFrame) {
      lastFrame = frames;
      await renderFb();
    }
    if (steps < 0) break;
    const halted = await call('halted');
    if (halted || steps < slice) break;
  }
  await finishRun(steps);
}

async function drainOutput() {
  const bytes = await call('drain_output');
  if (bytes && bytes.length) {
    log(new TextDecoder().decode(bytes));
  }
}

// Debug trace + register dumps.
const traceCb = $('trace-cb');
traceCb.addEventListener('change', () => {
  call('set_trace', traceCb.checked ? 1 : 0);
  if (traceCb.checked) log('Tracing enabled — run slices are capped for readability.', 'dim');
});

async function drainTrace() {
  const bytes = await call('drain_trace');
  if (bytes && bytes.length) return new TextDecoder().decode(bytes);
  return '';
}

$('btn-dumptrace').addEventListener('click', async () => {
  const t = await drainTrace();
  if (!t) { log('(no trace)', 'dim'); return; }
  log(t.replace(/\n$/, ''), 'dim');
});

$('btn-dumpregs').addEventListener('click', () => dumpRegs());

async function dumpRegs() {
  const s = await call('dump_regs');
  if (s) log(s.replace(/\n$/, ''), 'dim');
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
  $('steps').textContent = (await call('get_cycles')).toString();
}

// ---------- controller input ----------
// HidNpadButton bitfield, as the emulated program expects (switch_set_input).
const BTN = {
  A: 0x1, B: 0x2, X: 0x4, Y: 0x8, L: 0x10, R: 0x20, ZL: 0x40, ZR: 0x80,
  PLUS: 0x100, MINUS: 0x200, LEFT: 0x400, UP: 0x800, RIGHT: 0x1000, DOWN: 0x2000,
  STICK_L: 0x4000, STICK_R: 0x8000,
};
const inputEl = $('input-state');
function inputStatus(text) {
  if (inputEl) inputEl.textContent = text;
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
    // Analog sticks: -32768..32767, deadzone ~15%.
    const dz = 0.15;
    const axes = pad.axes || [];
    const dl = Math.abs(axes[0] || 0) > dz ? (axes[0] || 0) : 0;
    const dy = Math.abs(axes[1] || 0) > dz ? (axes[1] || 0) : 0;
    const rx = Math.abs(axes[2] || 0) > dz ? (axes[2] || 0) : 0;
    const ry = Math.abs(axes[3] || 0) > dz ? (axes[3] || 0) : 0;
    slx = Math.round(dl * 32767); sly = Math.round(dy * 32767);
    srx = Math.round(rx * 32767); sry = Math.round(ry * 32767);
    // DPAD can also come through as digital axes on some browsers.
    if (dl === 0) { if (axes[0] < -dz) mask |= BTN.LEFT; else if (axes[0] > dz) mask |= BTN.RIGHT; }
    inputStatus('gamepad: ' + (pad.id || 'connected'));
  } else if (mask) {
    inputStatus('keyboard');
  }
  call('set_input', mask, slx, sly, srx, sry);
}

setInterval(pushInput, 16);
window.addEventListener('gamepadconnected', () => inputStatus('gamepad connected'));
window.addEventListener('gamepaddisconnected', () => inputStatus('gamepad disconnected'));

// register inspector
const regIdx = $('reg-idx');
$('reg-idx-label').textContent = regIdx.value;
regIdx.addEventListener('input', () => { $('reg-idx-label').textContent = regIdx.value; });
$('btn-readreg').addEventListener('click', async () => {
  const v = await call('get_reg', parseInt(regIdx.value, 10));
  $('reg-val').textContent = v;
});

init();
