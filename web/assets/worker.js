/* switch-wasm worker host. Owns the wasm instance + session so the emulator
   runs off the main thread and long runs don't freeze the page. The main
   thread talks to this via postMessage: { id, cmd, args } -> { id, ok, result }.
   Byte buffers (file uploads, framebuffer, console/trace output) cross the
   boundary as transferred ArrayBuffers. */

let api = null;      // wasm exports
let handle = -1;     // session handle

function alloc(len) { return api.switch_alloc(len); }
function toWasm(jsbuf, ptr) {
  const view = new Uint8Array(api.memory.buffer, ptr, jsbuf.length);
  view.set(jsbuf);
}
function fromWasm(ptr, len) {
  return new Uint8Array(api.memory.buffer, ptr, len).slice();
}

function lastError() {
  const buf = alloc(512);
  const n = api.switch_last_error(handle, buf, 512);
  const s = new TextDecoder().decode(fromWasm(buf, n)).replace(/\u0000.*$/, '');
  api.switch_free(buf, 512);
  return s;
}

// Drain a ring buffer (output/trace) to completion, concatenating the chunks.
function drain(fn, cap) {
  const chunks = [];
  for (;;) {
    const buf = alloc(cap);
    const n = fn(handle, buf, cap);
    if (n > 0) chunks.push(fromWasm(buf, n));
    api.switch_free(buf, cap);
    if (n < cap) break;
  }
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

// Every handler returns a plain value (Number/string/Uint8Array/object).
const CMD = {
  new() { handle = api.switch_new(); return handle; },
  free_session() { api.switch_free_session(handle); handle = -1; return 0; },
  set_syscall_mode(mode) { api.switch_set_syscall_mode(handle, mode); return 0; },
  set_trace(on) { api.switch_set_trace(handle, on ? 1 : 0); return 0; },
  set_input(mask, slx, sly, srx, sry) {
    api.switch_set_input(handle, BigInt(mask), slx, sly, srx, sry);
    return 0;
  },

  load_nro(bytes) {
    const ptr = alloc(bytes.length);
    toWasm(bytes, ptr);
    const entry = Number(api.switch_load_nro(handle, ptr, bytes.length));
    api.switch_free(ptr, bytes.length);
    return entry;
  },
  load_elf(bytes) {
    const ptr = alloc(bytes.length);
    toWasm(bytes, ptr);
    const entry = Number(api.switch_load_elf(handle, ptr, bytes.length));
    api.switch_free(ptr, bytes.length);
    return entry;
  },
  load_nsp(bytes) {
    const ptr = alloc(bytes.length);
    toWasm(bytes, ptr);
    const ok = api.switch_load_nsp(handle, ptr, bytes.length);
    // switch_load_nsp keeps the staging buffer; do not free it.
    return ok;
  },
  load_keys(prod, title) {
    let pptr = 0, plen = 0, tptr = 0, tlen = 0;
    if (prod && prod.length) { pptr = alloc(prod.length); toWasm(prod, pptr); plen = prod.length; }
    if (title && title.length) { tptr = alloc(title.length); toWasm(title, tptr); tlen = title.length; }
    const rc = api.switch_load_keys(handle, pptr, plen, tptr, tlen);
    if (pptr) api.switch_free(pptr, plen);
    if (tptr) api.switch_free(tptr, tlen);
    return rc;
  },
  nsp_files_json() {
    const buf = alloc(8192);
    const n = api.switch_nsp_files_json(handle, buf, 8192);
    const s = new TextDecoder().decode(fromWasm(buf, n));
    api.switch_free(buf, 8192);
    return s;
  },
  read_file(index, offset, len) {
    const buf = alloc(len);
    const got = api.switch_read_file(handle, index, offset, buf, len);
    if (got < 0) return { error: lastError() };
    const b = fromWasm(buf, got);
    api.switch_free(buf, len);
    return b;
  },
  parse_nca(header) {
    const buf = alloc(header.length);
    toWasm(header, buf);
    const jbuf = alloc(4096);
    const jlen = api.switch_parse_nca(handle, buf, header.length, jbuf, 4096);
    api.switch_free(buf, header.length);
    const s = new TextDecoder().decode(fromWasm(jbuf, jlen));
    api.switch_free(jbuf, 4096);
    return s;
  },

  run(budget) { return Number(api.switch_run(handle, BigInt(budget))); },
  halted() { return api.switch_halted(handle); },
  drain_output() { return drain((h, b, l) => api.switch_drain_output(h, b, l), 4096); },
  drain_trace() { return drain((h, b, l) => api.switch_drain_trace(h, b, l), 8192); },
  dump_regs() {
    const buf = alloc(2048);
    const n = api.switch_dump_regs(handle, buf, 2048);
    const s = new TextDecoder().decode(fromWasm(buf, n));
    api.switch_free(buf, 2048);
    return s;
  },
  get_pc() { return api.switch_get_pc(handle); },
  get_cycles() { return Number(api.switch_get_cycles(handle)); },
  // Guest RAM is what the emulated console has touched; wasm is what this
  // worker's linear memory costs the browser (the page table, the loaded
  // image and every staging buffer live there).
  ram() {
    return {
      guest: handle < 0 ? 0 : Number(api.switch_guest_ram(handle)),
      wasm: api.memory.buffer.byteLength,
    };
  },
  get_reg(i) { return '0x' + api.switch_get_reg(handle, i).toString(16).padStart(16, '0'); },
  last_error() { return lastError(); },
  fb_width() { return api.switch_fb_width(handle); },
  fb_height() { return api.switch_fb_height(handle); },
  frame_count() { return api.switch_frame_count(handle); },
  fb_snapshot(len) {
    const buf = alloc(len);
    const n = api.switch_fb_snapshot(handle, buf, len);
    if (n <= 0) { api.switch_free(buf, len); return null; }
    const b = fromWasm(buf, n);
    api.switch_free(buf, len);
    return b;
  },
};

self.onmessage = (e) => {
  const { id, cmd, args } = e.data;
  try {
    const result = CMD[cmd](...args);
    if (result instanceof Uint8Array) {
      self.postMessage({ id, ok: true, result }, [result.buffer]);
    } else if (result && typeof result === 'object' && 'error' in result) {
      self.postMessage({ id, ok: false, error: result.error });
    } else {
      self.postMessage({ id, ok: true, result });
    }
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err) });
  }
};

(async () => {
  try {
    // instantiateStreaming fetches + compiles in one pass (works in workers).
    const { instance } = await WebAssembly.instantiateStreaming(
      fetch('switch_wasm.wasm'), {});
    api = instance.exports;
    self.postMessage({ type: 'ready' });
  } catch (err) {
    self.postMessage({ type: 'ready', error: String(err) });
  }
})();
