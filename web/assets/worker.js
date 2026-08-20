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

// Gamepad state arrives from the main thread far more often than the emulator
// gets to look at it: a whole run slice executes between two message drains, so
// a quick tap could be pressed and released without the guest ever seeing it.
// Presses are therefore held until a slice has run with them visible; the
// sticks, being continuous, just take the newest value.
let heldButtons = 0n;
let pressedButtons = 0n;
let sticks = [0, 0, 0, 0];

function publishInput(mask) {
  if (handle < 0) return;
  api.switch_set_input(handle, mask, sticks[0], sticks[1], sticks[2], sticks[3]);
}

// wasm32-unknown-unknown has no OS clock, so the emulated RTC (time:u/time:s)
// only knows what we push into it. The worker (unlike the wasm guest) has a
// real Date, so it just samples it directly rather than round-tripping
// through the main thread the way gamepad input has to.
function pushTime() {
  if (handle < 0) return;
  api.switch_set_time(handle, BigInt(Math.floor(Date.now() / 1000)));
}

// The Battery Status API is Window-only (not exposed to Workers), so unlike
// time this arrives from the main thread rather than being sampled here.
// Cached so a freshly created session picks up the last known reading
// immediately instead of the wasm default (full, charging).
let lastBattery = { percent: 100, charging: true };
function pushBattery() {
  if (handle < 0) return;
  api.switch_set_battery(handle, lastBattery.percent, lastBattery.charging ? 1 : 0);
}

// Every handler returns a plain value (Number/string/Uint8Array/object).
const CMD = {
  new() { handle = api.switch_new(); pushTime(); pushBattery(); return handle; },
  free_session() { api.switch_free_session(handle); handle = -1; return 0; },
  set_trace(on) { api.switch_set_trace(handle, on ? 1 : 0); return 0; },
  vibration() { return api.switch_vibration(handle); },
  set_input(mask, slx, sly, srx, sry) {
    heldButtons = BigInt(mask);
    pressedButtons |= heldButtons;
    sticks = [slx, sly, srx, sry];
    publishInput(pressedButtons);
    return 0;
  },
  set_battery(percent, charging) {
    lastBattery = { percent, charging: !!charging };
    pushBattery();
    return 0;
  },

  load_font(bytes) {
    const ptr = alloc(bytes.length);
    toWasm(bytes, ptr);
    const taken = api.switch_load_font(handle, ptr, bytes.length);
    api.switch_free(ptr, bytes.length);
    return taken;
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
  // Decrypts NSP file `index` as a Program NCA (with whatever keys are
  // loaded) and boots its ExeFS `main` executable. Operates on the NSP bytes
  // already staged in the worker's wasm memory by load_nsp - no extra copy of
  // the (possibly hundreds-of-MB) NCA crosses the postMessage boundary.
  load_nca_from_nsp(index) {
    return Number(api.switch_load_nca_from_nsp(handle, index));
  },
  // Same, for a standalone .nca file (not inside an NSP): stages the whole
  // file into wasm memory and boots it.
  load_nca(bytes) {
    const ptr = alloc(bytes.length);
    toWasm(bytes, ptr);
    const entry = Number(api.switch_load_nca(handle, ptr, bytes.length));
    // switch_load_nca takes ownership of the staging buffer; do not free it.
    return entry;
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
    // file_offset is a wasm u64 (needs a BigInt going in) and the return is
    // an i64 (comes back as a BigInt too) - convert that back to a Number
    // before using it as a length.
    const got = Number(api.switch_read_file(handle, index, BigInt(offset), buf, len));
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

  run(budget) {
    pushTime();
    const steps = Number(api.switch_run(handle, BigInt(budget)));
    // The guest has now had a slice to see whatever was tapped; release it.
    if (pressedButtons !== heldButtons) {
      pressedButtons = heldButtons;
      publishInput(pressedButtons);
    }
    return steps;
  },
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
  audio_format() { return api.switch_audio_format(handle); },

  // Interleaved 16-bit PCM, as raw bytes. The main thread reinterprets them
  // as an Int16Array rather than paying for a second copy here.
  audio_pull(maxSamples) {
    const bytes = maxSamples * 2;
    const buf = alloc(bytes);
    const n = api.switch_audio_pull(handle, buf, maxSamples);
    if (n <= 0) { api.switch_free(buf, bytes); return null; }
    const b = fromWasm(buf, n * 2);
    api.switch_free(buf, bytes);
    return b;
  },

  // ---------- the emulated SD card ----------
  //
  // `Vfs` lives in the session, so on its own nothing the guest writes
  // survives a reload. The main thread mirrors it into IndexedDB using these:
  // `sd_write_file`/`sd_create_dir` restore the card before a boot, and
  // `sd_take_changes` reports what the guest touched so only that is written
  // back.

  sd_write_file(path, bytes) {
    const p = new TextEncoder().encode(path);
    const pptr = alloc(p.length);
    toWasm(p, pptr);
    const dptr = alloc(bytes.length || 1);
    if (bytes.length) toWasm(bytes, dptr);
    const rc = api.switch_sd_write_file(handle, pptr, p.length, dptr, bytes.length);
    api.switch_free(pptr, p.length);
    api.switch_free(dptr, bytes.length || 1);
    return rc;
  },

  sd_create_dir(path) {
    const p = new TextEncoder().encode(path);
    const ptr = alloc(p.length);
    toWasm(p, ptr);
    const rc = api.switch_sd_create_dir(handle, ptr, p.length);
    api.switch_free(ptr, p.length);
    return rc;
  },

  sd_remove(path) {
    const p = new TextEncoder().encode(path);
    const ptr = alloc(p.length);
    toWasm(p, ptr);
    const rc = api.switch_sd_remove(handle, ptr, p.length);
    api.switch_free(ptr, p.length);
    return rc;
  },

  // The whole file, or null when the path is not one. Read in slices so a
  // large save does not need a single allocation twice its size.
  sd_read_file(path) {
    const p = new TextEncoder().encode(path);
    const pptr = alloc(p.length);
    toWasm(p, pptr);
    const size = Number(api.switch_sd_file_size(handle, pptr, p.length));
    if (size < 0) { api.switch_free(pptr, p.length); return null; }
    const out = new Uint8Array(size);
    const CHUNK = 1 << 20;
    const cap = Math.min(Math.max(size, 1), CHUNK);
    const buf = alloc(cap);
    let off = 0;
    while (off < size) {
      const n = Number(api.switch_sd_read_file(handle, pptr, p.length, BigInt(off), buf, cap));
      if (n <= 0) break;
      out.set(fromWasm(buf, n), off);
      off += n;
    }
    api.switch_free(buf, cap);
    api.switch_free(pptr, p.length);
    return out;
  },

  sd_pending_changes() {
    return handle < 0 ? 0 : api.switch_sd_pending_changes(handle);
  },

  // Drains on the wasm side whether or not the JSON fits, so the buffer is
  // sized from the pending count first: a path is capped at 0x301 bytes by
  // the fs protocol, plus ~48 for the rest of the entry.
  sd_take_changes() {
    if (handle < 0) return [];
    const pending = api.switch_sd_pending_changes(handle);
    if (!pending) return [];
    const cap = 2 + pending * (0x301 * 2 + 64);
    const buf = alloc(cap);
    const n = api.switch_sd_take_changes_json(handle, buf, cap);
    const text = new TextDecoder().decode(fromWasm(buf, n));
    api.switch_free(buf, cap);
    try { return JSON.parse(text); } catch { return []; }
  },

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
