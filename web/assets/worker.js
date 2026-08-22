/* switch-wasm worker host. Owns the wasm instance + session so the emulator
   runs off the main thread and long runs don't freeze the page. The main
   thread talks to this via postMessage: { id, cmd, args } -> { id, ok, result }.
   Byte buffers (file uploads, framebuffer, console/trace output) cross the
   boundary as transferred ArrayBuffers. */

let api = null;      // wasm exports
let handle = -1;     // session handle

// Every buffer that crosses into wasm goes through here. A refused request
// comes back as null (switch_alloc will not trap on one), and writing at
// address 0 would corrupt the module's own data rather than fail, so this is
// where an impossible size has to stop.
function alloc(len) {
  const ptr = api.switch_alloc(len);
  if (!ptr) throw new Error('cannot allocate ' + len + ' bytes in the emulator');
  return ptr;
}
function toWasm(jsbuf, ptr) {
  const view = new Uint8Array(api.memory.buffer, ptr, jsbuf.length);
  view.set(jsbuf);
}
function fromWasm(ptr, len) {
  return new Uint8Array(api.memory.buffer, ptr, len).slice();
}

// ---------- the open container ----------
//
// A retail .nsp runs to several gigabytes: more than a wasm32 module can
// address, let alone allocate in one buffer, so it is never handed over as
// one. The File stays with the browser and the wasm side pulls ranges out of
// it through the `host_read` import below.
//
// That import has to answer *synchronously* - the emulator asks for RomFS
// ranges from inside `switch_run`, with nowhere to await a promise -
// and `FileReaderSync` exists only in a worker, which is the second reason
// the emulator lives in one.
// File 0 is the container being run; the rest are system data archives the
// page has added, which a title mounts by data id.
let hostFiles = [];
let hostReader = null;

// Reads land in bursts of a few hundred bytes as the guest walks its RomFS
// tables, so whole chunks are kept around; a Map iterates in insertion order,
// which makes it an LRU with no bookkeeping of its own.
const HOST_CHUNK = 1 << 20;
const HOST_CACHE_CHUNKS = 32;
const hostChunks = new Map();

// Opening a container replaces slot 0 and leaves the rest alone: the wasm
// side holds sources that address archives by index, so the table can only
// ever grow. The chunk cache is keyed by index too, hence the flush.
function openHostFile(file) {
  if (!hostReader) hostReader = new FileReaderSync();
  if (hostFiles.length === 0) hostFiles = [null];
  hostFiles[0] = file;
  hostChunks.clear();
  return BigInt(file.size);
}

// Add a file the wasm side can read, and return its index. Slot 0 stays
// reserved for the container even if nothing has been opened yet.
function addHostFile(file) {
  if (!hostReader) hostReader = new FileReaderSync();
  if (hostFiles.length === 0) hostFiles = [null];
  return hostFiles.push(file) - 1;
}

function readBlob(file, start, end) {
  return new Uint8Array(hostReader.readAsArrayBuffer(file.slice(start, end)));
}

function hostChunk(file, index, key) {
  const hit = hostChunks.get(key);
  if (hit) {
    hostChunks.delete(key);
    hostChunks.set(key, hit);
    return hit;
  }
  const start = index * HOST_CHUNK;
  const chunk = readBlob(file, start, Math.min(start + HOST_CHUNK, file.size));
  hostChunks.set(key, chunk);
  if (hostChunks.size > HOST_CACHE_CHUNKS) {
    hostChunks.delete(hostChunks.keys().next().value);
  }
  return chunk;
}

// The wasm import: fill `len` bytes at `ptr` from `offset` of the open file,
// and return how many were filled. `offset` arrives as a BigInt (it is an
// i64), and `ptr`/`len` as signed i32s, hence the `>>> 0`.
function hostRead(fileIndex, offset, ptr, len) {
  ptr >>>= 0;
  len >>>= 0;
  const file = hostFiles[fileIndex >>> 0];
  if (!file || !len) return 0;
  let at = Number(offset);
  const end = Math.min(at + len, file.size);
  if (at >= end) return 0;
  // The view has to be built here, not cached: growing the heap detaches it.
  const out = new Uint8Array(api.memory.buffer, ptr, end - at);
  let written = 0;
  try {
    // A read bigger than a chunk is the ExeFS being pulled in one go. Serve
    // it straight from the file: it would evict the whole cache on its way
    // through and never be asked for again.
    if (end - at > HOST_CHUNK) {
      out.set(readBlob(file, at, end));
      return end - at;
    }
    while (at < end) {
      const index = Math.floor(at / HOST_CHUNK);
      const chunk = hostChunk(file, index, `${fileIndex}:${index}`);
      const from = at - index * HOST_CHUNK;
      const take = Math.min(chunk.length - from, end - at);
      if (take <= 0) break;
      out.set(chunk.subarray(from, from + take), written);
      written += take;
      at += take;
    }
  } catch (e) {
    // The file was moved or replaced while it was open. Report the short
    // read; the wasm side turns that into an error with an offset on it.
    console.error('[switch-wasm] host read failed:', e);
  }
  return written;
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
// gets to look at it: the worker is blocked inside `switch_run` while the input
// messages pile up in its queue, so a whole slice's worth of them lands at once
// at the slice boundary and a quick tap can be pressed and released without the
// guest ever having been running to see it.
//
// The unit a press has to survive is a *guest frame*, not a run slice. The
// guest polls hid once per iteration of its own loop and presents once per
// iteration, and one of those spans many slices, so holding a tap for a single
// slice still let most taps fall between two polls. A press is therefore held
// until the frame counter has advanced twice: the poll sits somewhere inside
// the guest's loop, so only a complete present-to-present interval is
// guaranteed to contain one.
//
// Only bits the guest may not have seen are held. A key the host still reports
// as down is published from `heldButtons` on its own, so releasing it takes
// effect at the very next slice instead of a slice later - that extra slice of
// stickiness was making one d-pad tap step two menu entries.
let heldButtons = 0n;    // what the host says is physically down right now
let latchedButtons = 0n; // pressed, but not yet guaranteed seen by the guest
let sticks = [0, 0, 0, 0];  // newest analog values
let latchedSticks = null;   // a deflection held for the same reason as a press

// Touch rides the same latch, for the same reason: a tap that goes down and up
// inside one run slice would otherwise happen entirely while the guest was not
// running to see it. Contacts are flat {finger_id, x, y} triples.
const TOUCH_MAX = 16;
const NO_TOUCHES = new Uint32Array(0);
let touches = NO_TOUCHES;      // newest host contacts
let latchedTouches = null;     // a tap held until the guest has had a frame
let touchIds = new Set();      // finger ids down at the last host sample
let touchScratch = 0;          // wasm-side staging buffer, allocated once
let publishedTouches = 0;      // contacts the guest was last told about

// Frame the latch is waiting on, plus a slice cap so that a program which never
// presents - or has stopped, mid-load - still releases instead of holding a
// phantom press until it draws again. A couple of seconds' worth of slices.
let latchFrame = -1;
let latchSlices = 0;
const LATCH_FRAMES = 2;
const MAX_LATCH_SLICES = 64;

// Matches HID_STICK_THRESHOLD in cpu/mod.rs: past this the core reports the
// HidNpadButton_StickL*/StickR* pseudo-buttons, which is what menus navigate
// with, so a flick has to be latched exactly like a button press.
const STICK_THRESHOLD = 0x4000;
const deflected = (s) => s.some((v) => Math.abs(v) > STICK_THRESHOLD);

function publishInput() {
  if (handle < 0) return;
  // A stick the host is still deflecting reports its live value; a latched
  // flick only stands in once the stick has sprung back to centre.
  const s = latchedSticks && !deflected(sticks) ? latchedSticks : sticks;
  api.switch_set_input(handle, heldButtons | latchedButtons, s[0], s[1], s[2], s[3]);
  publishTouch();
}

// Same rule as the sticks: live contacts win, a latched tap only stands in once
// the finger is up. `switch_set_touch` reads {finger_id, x, y} triples out of
// wasm memory, so they go through a buffer allocated once and reused - the view
// is rebuilt every time because growing the heap detaches the old one.
function publishTouch() {
  const src = touches.length ? touches : latchedTouches || NO_TOUCHES;
  const count = Math.min(TOUCH_MAX, src.length / 3);
  // Nothing down and nothing to retract: the guest already knows.
  if (count === 0 && publishedTouches === 0) return;
  if (!touchScratch) touchScratch = alloc(TOUCH_MAX * 3 * 4);
  if (count > 0) {
    new Uint32Array(api.memory.buffer, touchScratch, count * 3)
      .set(src.subarray(0, count * 3));
  }
  api.switch_set_touch(handle, touchScratch, count);
  publishedTouches = count;
}

// A latch armed against the old session's frame counter would outlive a reset,
// so both ends of the session lifecycle clear it.
function resetInput() {
  heldButtons = 0n;
  latchedButtons = 0n;
  sticks = [0, 0, 0, 0];
  latchedSticks = null;
  touches = NO_TOUCHES;
  latchedTouches = null;
  touchIds = new Set();
  publishedTouches = 0;
  latchFrame = -1;
  latchSlices = 0;
}

// Start (or restart) the wait for the guest to run a frame with the latch
// visible. Restarting on a later press extends the window for earlier ones too,
// which is what we want: they are all still unseen.
function armLatch() {
  latchFrame = handle < 0 ? -1 : api.switch_frame_count(handle);
  latchSlices = 0;
}

// Called once per run slice: drop the latch as soon as the guest has had a
// whole frame to poll with it visible.
function releaseLatchIfSeen() {
  if (handle < 0) return;
  if (latchedButtons === 0n && !latchedSticks && !latchedTouches) return;
  const frames = api.switch_frame_count(handle);
  if (frames - latchFrame < LATCH_FRAMES && ++latchSlices < MAX_LATCH_SLICES) return;
  latchedButtons = 0n;
  latchedSticks = null;
  latchedTouches = null;
  latchFrame = -1;
  latchSlices = 0;
  publishInput();
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
  new() {
    handle = api.switch_new();
    resetInput(); // the new session's frame counter restarts at 0
    pushTime();
    pushBattery();
    return handle;
  },
  free_session() { api.switch_free_session(handle); handle = -1; resetInput(); return 0; },
  set_trace(on) { api.switch_set_trace(handle, on ? 1 : 0); return 0; },
  vibration() { return api.switch_vibration(handle); },
  set_input(mask, slx, sly, srx, sry) {
    const next = BigInt(mask);
    const pressed = next & ~heldButtons; // edges, not level: what just went down
    heldButtons = next;
    sticks = [slx, sly, srx, sry];
    const flicked = deflected(sticks);
    if (flicked) latchedSticks = sticks;
    if (pressed !== 0n || flicked) {
      latchedButtons |= pressed;
      armLatch();
    }
    publishInput();
    return 0;
  },
  // Contacts as flat {finger_id, x, y} triples, already in the console's
  // 1280x720 digitizer space. A finger id the previous sample did not carry is
  // a new contact, which is what arms the latch.
  set_touch(points) {
    const next = points && points.length ? new Uint32Array(points) : NO_TOUCHES;
    const ids = new Set();
    let fresh = false;
    for (let i = 0; i < next.length; i += 3) {
      ids.add(next[i]);
      if (!touchIds.has(next[i])) fresh = true;
    }
    touches = next;
    touchIds = ids;
    if (fresh) {
      latchedTouches = next;
      armLatch();
    }
    publishInput();
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
  // Open a container: the File is kept here and read range by range, so this
  // costs nothing but its PFS0 header no matter how large the file is.
  open_nsp(file) {
    return api.switch_open_nsp(handle, openHostFile(file));
  },
  // Same, for a standalone .nca - the container is the NCA, with no file
  // table in front of it.
  open_nca(file) {
    return api.switch_open_nca(handle, openHostFile(file));
  },
  // Register a firmware NCA as a system data archive. Costs nothing but the
  // File reference and its header until a title mounts it.
  add_archive(file) {
    const index = addHostFile(file);
    return api.switch_add_archive(handle, index, BigInt(file.size));
  },
  // The same, from bytes rather than a File reference - which is what makes
  // it keepable. A browser will not hand the page a file it was not asked
  // for, so an archive registered from a File is gone on reload; bytes can be
  // stored and handed back. Returns the archive's title id as hex, or '' if
  // the bytes are not one.
  // What a firmware NCA is, without reading it: a header read through the
  // File the page is still holding. Returns { id, kind } - kind 0 for a
  // program, 1 for a data archive, 2 for anything else - or null if it is not
  // an NCA this build can read. A firmware dump is mostly the third kind, and
  // this is what keeps the page from pulling all of it through memory to find
  // that out.
  nand_identify(file) {
    const index = addHostFile(file);
    const kindPtr = alloc(4);
    let id, kind;
    try {
      id = api.switch_nand_identify(handle, index, BigInt(file.size), kindPtr);
      kind = new DataView(api.memory.buffer).getUint32(kindPtr, true);
    } finally {
      api.switch_free(kindPtr, 4);
    }
    return id ? { id: id.toString(16).padStart(16, '0'), kind } : null;
  },
  // Boot a program the host has the bytes of: a title installed on the NAND
  // rather than one opened out of a container the user just picked. The
  // emulator keeps its own copy, so the staging buffer goes back immediately.
  nand_launch(bytes) {
    const ptr = alloc(bytes.length);
    toWasm(bytes, ptr);
    try {
      return Number(api.switch_nand_launch(handle, ptr, bytes.length));
    } finally {
      api.switch_free(ptr, bytes.length);
    }
  },
  nand_add_archive(bytes) {
    const ptr = alloc(bytes.length);
    toWasm(bytes, ptr);
    let id;
    try {
      id = api.switch_nand_add_archive(handle, ptr, bytes.length);
    } finally {
      api.switch_free(ptr, bytes.length);
    }
    return id ? id.toString(16).padStart(16, '0') : '';
  },
  // Decrypts NSP file `index` as a Program NCA (with whatever keys are
  // loaded) and boots its ExeFS `main` executable, reading both out of the
  // open container. Its RomFS is left where it is and decrypted on demand
  // while the title runs.
  load_nca_from_nsp(index) {
    return Number(api.switch_load_nca_from_nsp(handle, index));
  },
  // Same, for a container that is itself a single standalone .nca.
  load_nca() {
    return Number(api.switch_load_nca(handle));
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
  // The title's name, publisher, version and icon, out of the Control NCA in
  // the open container. Cheap next to the container itself: a Control NCA is
  // an icon and a metadata blob, not game data.
  load_control_from_nsp() { return api.switch_load_control_from_nsp(handle); },
  // Same, for a container that is itself a single standalone Control NCA.
  load_control_from_nca() { return api.switch_load_control_from_nca(handle); },
  control_json() {
    // Sized for the worst case rather than the usual one: the JSON carries a
    // 0x200-byte name and a 0x100-byte publisher straight out of the NACP,
    // and `switch_control_json` truncates silently rather than saying it
    // overflowed - which would surface as a JSON parse error, not a clue.
    const cap = 16384;
    const buf = alloc(cap);
    const n = api.switch_control_json(handle, buf, cap);
    const s = new TextDecoder().decode(fromWasm(buf, n));
    api.switch_free(buf, cap);
    return s;
  },
  // `size` comes from control_json's icon_size: the icon is a JPEG of
  // unpredictable length, so JS is told how big a buffer to hand over.
  control_icon(size) {
    if (!size) return new Uint8Array(0);
    const buf = alloc(size);
    const n = Number(api.switch_control_icon(handle, buf, size));
    const icon = n > 0 ? fromWasm(buf, n) : new Uint8Array(0);
    api.switch_free(buf, size);
    return icon;
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
    releaseLatchIfSeen();
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
    // The one import is how the module reads the open container; see
    // `hostRead`.
    const { instance } = await WebAssembly.instantiateStreaming(
      fetch('switch_wasm.wasm'), { env: { host_read: hostRead } });
    api = instance.exports;
    self.postMessage({ type: 'ready' });
  } catch (err) {
    self.postMessage({ type: 'ready', error: String(err) });
  }
})();
