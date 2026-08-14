/* switch-wasm browser frontend. Thin glue over the exported wasm ABI:
   buffers are copied into wasm linear memory via switch_alloc + a DataView. */

const $ = (id) => document.getElementById(id);

let api = null;       // wasm exports
let memory = null;    // wasm linear memory
let handle = -1;      // session handle

function alloc(len) { return api.switch_alloc(len); }
function toWasm(jsbuf, ptr) { new Uint8Array(memory.buffer, ptr, jsbuf.length).set(jsbuf); }
function fromWasm(ptr, len) {
  const b = new Uint8Array(memory.buffer, ptr, len);
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
  const ptr = alloc(data.byteLength);
  toWasm(new Uint8Array(data), ptr);
  const ok = api.switch_load_nsp(handle, ptr, data.byteLength);
  if (ok !== 0) {
    log('NSP error: ' + readLastError(), 'err');
    return;
  }
  const jbuf = alloc(8192);
  const jlen = api.switch_nsp_files_json(handle, jbuf, 8192);
  const files = JSON.parse(strFromWasm(jbuf, jlen));
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

function inspectNca(f, index) {
  const out = document.createElement('div');
  out.className = 'nca-info';
  out.textContent = 'Parsing ' + f.name + ' ...';
  $('nsp-result').appendChild(out);

  const buf = alloc(f.size);
  const got = api.switch_extract_file(handle, index, buf, f.size);
  if (got < 0) {
    out.textContent = 'extract failed: ' + readLastError();
    return;
  }
  const jbuf = alloc(4096);
  const jlen = api.switch_parse_nca(buf, f.size, jbuf, 4096);
  const info = JSON.parse(strFromWasm(jbuf, jlen));
  out.textContent = '';
  if (info.error) {
    out.textContent = 'NCA: ' + info.error;
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

const syscallMode = $('syscall-mode');
function applySyscallMode() {
  api.switch_set_syscall_mode(handle, parseInt(syscallMode.value, 10));
}
syscallMode.addEventListener('change', applySyscallMode);

async function loadProgram(file, kind) {
  clearConsole();
  setState('loading');
  const data = await file.arrayBuffer();
  const ptr = alloc(data.byteLength);
  toWasm(new Uint8Array(data), ptr);
  const entry = kind === 'nro'
    ? api.switch_load_nro(handle, ptr, data.byteLength)
    : api.switch_load_elf(handle, ptr, data.byteLength);
  if (entry < 0) {
    setState('error');
    log('Load failed: ' + readLastError(), 'err');
    return false;
  }
  log('Loaded ' + file.name + ' — entry 0x' + entry.toString(16).padStart(8, '0'), 'ok');
  const abi = syscallMode.value === '2'
    ? 'Horizon syscall stubs (svcOutputDebugString → console)'
    : 'UART demo ABI (#1 putchar, #2 putstr, #0 halt)';
  log('SVC ABI: ' + abi, 'dim');
  setState('loaded');
  updatePc();
  return true;
}

$('btn-demo').addEventListener('click', async () => {
  syscallMode.value = '1';
  applySyscallMode();
  const res = await fetch('assets/demo.nro');
  const data = await res.arrayBuffer();
  const file = new File([data], 'demo.nro');
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

function finishRun(steps, budget) {
  const err = readLastError();
  if (steps < 0) {
    setState('fault');
    log('CPU fault: ' + err, 'err');
  } else if (api.switch_halted(handle)) {
    setState('halted');
    log('Halted via SVC #0', 'ok');
  } else if (Number(steps) >= budget) {
    setState('timeout');
    log('Reached ' + budget + '-step budget; still running — click Run to continue.', 'dim');
  } else {
    setState('fault');
    log('Stopped unexpectedly.', 'err');
  }
  drainOutput();
  updatePc();
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
    log('Halted via SVC #0', 'ok');
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

// register inspector
const regIdx = $('reg-idx');
$('reg-idx-label').textContent = regIdx.value;
regIdx.addEventListener('input', () => { $('reg-idx-label').textContent = regIdx.value; });
$('btn-readreg').addEventListener('click', () => {
  const v = api.switch_get_reg(handle, parseInt(regIdx.value, 10));
  $('reg-val').textContent = '0x' + v.toString(16).padStart(16, '0');
});

init();
