/* switch-wasm browser frontend. Thin glue over the exported wasm ABI:
   buffers are copied into wasm linear memory via switch_alloc + a DataView. */

const $ = (id) => document.getElementById(id);

let api = null;       // wasm exports
let memory = null;    // wasm linear memory
let handle = -1;      // session handle

function alloc(len) { return api.switch_alloc(len); }
function toWasm(jsbuf, ptr) {
  // Re-fetch the buffer every call: wasm memory can grow (detaching the old
  // ArrayBuffer) between allocations.
  const view = new Uint8Array(api.memory.buffer, ptr, jsbuf.length);
  view.set(jsbuf);
}
function fromWasm(ptr, len) {
  const b = new Uint8Array(api.memory.buffer, ptr, len);
  // copy out before any later call may grow memory
  return b.slice();
}
function strFromWasm(ptr, len) {
  const bytes = fromWasm(ptr, len);
  return new TextDecoder().decode(bytes).replace(/\u0000.*$/, '');
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

function readLastError() {
  const buf = alloc(512);
  const n = api.switch_last_error(handle, buf, 512);
  return strFromWasm(buf, n);
}

function fmtSize(n) {
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(2) + ' GiB';
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(2) + ' MiB';
  if (n >= 1 << 10) return (n / (1 << 10)).toFixed(1) + ' KiB';
  return n + ' B';
}

function setState(text) {
  $('state').textContent = text;
}

// Copy the emulated framebuffer (FB_BASE, 640x360 RGBA) into the canvas.
function renderFb() {
  if (!api || fbBytes === 0) return;
  const buf = alloc(fbBytes);
  const n = api.switch_fb_snapshot(handle, buf, fbBytes);
  if (n > 0) {
    // Copy out of wasm memory first so a later call can't invalidate the view.
    const pixels = new Uint8ClampedArray(fromWasm(buf, n));
    screenCtx.putImageData(new ImageData(pixels, fbW, fbH), 0, 0);
  }
  api.switch_free(buf, fbBytes);
}

// ---------- boot ----------

async function init() {
  const res = await fetch('assets/switch_wasm.wasm');
  const bytes = await res.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  api = instance.exports;
  memory = api.memory;
  handle = api.switch_new();
  applySyscallMode();
  fbW = api.switch_fb_width();
  fbH = api.switch_fb_height();
  fbBytes = fbW * fbH * 4;
  screenEl.width = fbW;
  screenEl.height = fbH;
  $('wasm-ver').textContent = 'core loaded: ' + bytes.byteLength + ' bytes of wasm';
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
  const data = await file.arrayBuffer();
  let ptr;
  try {
    ptr = alloc(data.byteLength);
    toWasm(new Uint8Array(data), ptr);
  } catch (e) {
    log('Failed to stage ' + fmtSize(data.byteLength) + ' in wasm memory: ' + e, 'err');
    return;
  }
  const ok = api.switch_load_nsp(handle, ptr, data.byteLength);
  // switch_load_nsp takes ownership of the staging buffer (it's now the
  // session's NSP image), so we must NOT free it here.
  if (ok !== 0) {
    log('NSP error: ' + readLastError(), 'err');
    return;
  }
  const jbuf = alloc(8192);
  const jlen = api.switch_nsp_files_json(handle, jbuf, 8192);
  const files = JSON.parse(strFromWasm(jbuf, jlen));
  api.switch_free(jbuf, 8192);
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
  // Send the staged key text (prod.keys / title.keys) to the wasm session so
  // NCA headers can be decrypted.
  if (!api) return;
  let pptr = 0, plen = 0, tptr = 0, tlen = 0;
  if (prodKeysText) {
    const b = new TextEncoder().encode(prodKeysText);
    pptr = alloc(b.length); toWasm(b, pptr); plen = b.length;
  }
  if (titleKeysText) {
    const b = new TextEncoder().encode(titleKeysText);
    tptr = alloc(b.length); toWasm(b, tptr); tlen = b.length;
  }
  const rc = api.switch_load_keys(handle, pptr, plen, tptr, tlen);
  if (pptr) api.switch_free(pptr, plen);
  if (tptr) api.switch_free(tptr, tlen);
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

function inspectNca(f, index) {
  // Replace any previous inspection result instead of stacking them up.
  $('nsp-result').querySelectorAll('.nca-info').forEach((el) => el.remove());
  const out = document.createElement('div');
  out.className = 'nca-info';
  out.textContent = 'Parsing ' + f.name + ' ...';
  $('nsp-result').appendChild(out);

  // Only the first 0x400 bytes (the header) are needed to inspect an NCA —
  // don't allocate the whole (possibly hundreds-of-MB) payload in wasm memory.
  const headerLen = Math.min(f.size, 0x800);
  const buf = alloc(headerLen);
  const got = api.switch_read_file(handle, index, 0n, buf, headerLen);
  if (got < 0) {
    out.textContent = 'read failed: ' + readLastError();
    api.switch_free(buf, headerLen);
    return;
  }
  const jbuf = alloc(4096);
  const jlen = api.switch_parse_nca(handle, buf, headerLen, jbuf, 4096);
  api.switch_free(buf, headerLen); // staging buffer no longer needed
  const info = JSON.parse(strFromWasm(jbuf, jlen));
  api.switch_free(jbuf, 4096);
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
  api.switch_set_syscall_mode(handle, 2); // Horizon
}

async function loadProgram(file, kind) {
  clearConsole();
  setState('loading');
  const data = await file.arrayBuffer();
  const ptr = alloc(data.byteLength);
  toWasm(new Uint8Array(data), ptr);
  const entry = kind === 'nro'
    ? api.switch_load_nro(handle, ptr, data.byteLength)
    : api.switch_load_elf(handle, ptr, data.byteLength);
  // switch_load_nro/elf copy the image into emulated memory — free the staging
  // buffer so repeated loads don't accumulate wasm memory.
  api.switch_free(ptr, data.byteLength);
  if (entry < 0) {
    setState('error');
    log('Load failed: ' + readLastError(), 'err');
    return false;
  }
  log('Loaded ' + file.name + ' — entry 0x' + entry.toString(16).padStart(8, '0'), 'ok');
  log('SVC ABI: Horizon stubs (svcOutputDebugString → console)', 'dim');
  setState('loaded');
  updatePc();
  return true;
}

$('btn-demo').addEventListener('click', async () => {
  const name = $('asset-nro').value;
  applySyscallMode();
  const res = await fetch('assets/' + name);
  if (!res.ok) { log('Fetch failed: ' + name, 'err'); return; }
  const data = await res.arrayBuffer();
  const file = new File([data], name);
  if (await loadProgram(file, 'nro')) run();
});

$('nro-file').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  applySyscallMode();
  const kind = /\.nro$/i.test(f.name) ? 'nro' : 'elf';
  if (await loadProgram(f, kind)) run();
});

$('btn-run').addEventListener('click', run);
$('btn-step').addEventListener('click', () => {
  const r = api.switch_run(handle, 1n);
  finishRun(r, 1);
  if (traceCb.checked && r >= 0) {
    const t = drainTrace();
    if (t) log(t.replace(/\n$/, ''), 'dim');
  }
});
$('btn-reset').addEventListener('click', () => {
  api.switch_free_session(handle);
  handle = api.switch_new();
  applySyscallMode();
  clearConsole();
  setState('idle');
});

function run() {
  // Real homebrew needs far more than 100k steps just to get through libnx
  // init, so the non-trace budget is large. Clicking Run again continues from
  // where the previous run stopped (the CPU state persists).
  const budget = traceCb.checked ? 5000n : 20_000_000n;
  setState('running');
  const r = api.switch_run(handle, budget);
  finishRun(r, Number(budget));
}

function drainOutput() {
  const outLen = 4096;
  const buf = alloc(outLen);
  let n = api.switch_drain_output(handle, buf, outLen);
  let total = '';
  while (n > 0) {
    total += strFromWasm(buf, n);
    n = api.switch_drain_output(handle, buf, outLen);
  }
  if (total) log(total);
}

// Debug trace + register dumps.
const traceCb = $('trace-cb');
traceCb.addEventListener('change', () => {
  api.switch_set_trace(handle, traceCb.checked ? 1 : 0);
  if (traceCb.checked) log('Tracing enabled — run budget is capped for readability.', 'dim');
});

function drainTrace() {
  const cap = 8192;
  const buf = alloc(cap);
  let n = api.switch_drain_trace(handle, buf, cap);
  let total = '';
  while (n > 0) {
    total += strFromWasm(buf, n);
    n = api.switch_drain_trace(handle, buf, cap);
  }
  return total;
}

$('btn-dumptrace').addEventListener('click', () => {
  const t = drainTrace();
  if (!t) { log('(no trace)', 'dim'); return; }
  log(t.replace(/\n$/, ''), 'dim');
});

$('btn-dumpregs').addEventListener('click', dumpRegs);

function dumpRegs() {
  const cap = 2048;
  const buf = alloc(cap);
  const n = api.switch_dump_regs(handle, buf, cap);
  log(strFromWasm(buf, n).replace(/\n$/, ''), 'dim');
}

function finishRun(steps, budget) {
  const err = readLastError();
  if (steps < 0) {
    setState('fault');
    log('CPU fault: ' + err, 'err');
    // The fault trace already carries the register snapshot from the CPU.
    const t = drainTrace();
    if (t) log(t.replace(/\n$/, ''), 'err');
  } else if (api.switch_halted(handle)) {
    setState('halted');
    log('Halted (ExitProcess)', 'ok');
    if (traceCb.checked) {
      const t = drainTrace();
      if (t) log(t.replace(/\n$/, ''), 'dim');
    }
  } else if (Number(steps) >= budget) {
    setState('timeout');
    log('Reached ' + budget + '-step budget; still running — click Run to continue.', 'dim');
  } else {
    setState('fault');
    log('Stopped unexpectedly.', 'err');
  }
  drainOutput();
  renderFb();
  updatePc();
}

function updatePc() {
  $('pc').textContent = '0x' + api.switch_get_pc(handle).toString(16).padStart(8, '0');
  $('steps').textContent = api.switch_get_cycles(handle).toString();
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
  if (!api) return;
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
  api.switch_set_input(handle, BigInt(mask), slx, sly, srx, sry);
}

setInterval(pushInput, 16);
window.addEventListener('gamepadconnected', () => inputStatus('gamepad connected'));
window.addEventListener('gamepaddisconnected', () => inputStatus('gamepad disconnected'));

// register inspector
const regIdx = $('reg-idx');
$('reg-idx-label').textContent = regIdx.value;
regIdx.addEventListener('input', () => { $('reg-idx-label').textContent = regIdx.value; });
$('btn-readreg').addEventListener('click', () => {
  const v = api.switch_get_reg(handle, parseInt(regIdx.value, 10));
  $('reg-val').textContent = '0x' + v.toString(16).padStart(16, '0');
});

init();
