/* Everything the page can ask the emulator to do.

   One entry per command, each returning a plain value (number/string/
   Uint8Array/object) or `{ error }`; the message loop in `index.ts` turns
   those into replies. Staging buffers are allocated and released around every
   call - the emulator's heap is the browser's memory too. */

import type {
  CommandHandlers, CrashReport, FsChange, GpuReport, IpcGaps, JitStats, TraceChannel,
} from '../shared/protocol';
import { addHostFile, openHostFile, resetHostFiles } from './hostfiles';
import { releaseLatchIfSeen, resetInput, setGamepad, setTouch } from './latch';
import {
  api,
  drain,
  fromWasm,
  handle,
  lastError,
  readJson,
  readString,
  state,
  withBuffer,
  withBytes,
  withPath,
} from './wasm';

// wasm32-unknown-unknown has no OS clock, so the emulated RTC (time:u/time:s)
// only knows what we push into it. The worker (unlike the wasm guest) has a
// real Date, so it just samples it directly rather than round-tripping
// through the main thread the way gamepad input has to.
function pushTime(): void {
  if (handle() < 0) return;
  api().switch_set_time(handle(), BigInt(Math.floor(Date.now() / 1000)));
}

// The Battery Status API is Window-only (not exposed to Workers), so unlike
// time this arrives from the main thread rather than being sampled here.
// Cached so a freshly created session picks up the last known reading
// immediately instead of the wasm default (full, charging).
let lastBattery = { percent: 100, charging: true };
function pushBattery(): void {
  if (handle() < 0) return;
  api().switch_set_battery(handle(), lastBattery.percent, lastBattery.charging ? 1 : 0);
}

// Whether the console is docked. Cached for the same reason the battery is:
// "reset" builds a fresh session, and a dock the user set before that is
// still a dock afterwards.
let docked = false;
function pushOperationMode(): void {
  if (handle() < 0) return;
  api().switch_set_operation_mode(handle(), docked ? 1 : 0);
}

// A save id travels as hex text: it is a u64, and JSON has no such number.
const saveId = (id: string) => BigInt('0x' + id);

// A path is capped at 0x301 bytes by the fs protocol, plus ~48 for the rest
// of an entry; the JSON drains on the wasm side whether or not it fits, so
// the buffer is sized from the pending count rather than guessed.
const changesCap = (pending: number) => 2 + pending * (0x301 * 2 + 64);

// A save large enough to want reading in slices, rather than one allocation
// twice its size.
const READ_CHUNK = 1 << 20;

export const CMD: CommandHandlers = {
  new() {
    state.handle = api().switch_new();
    resetInput(); // the new session's frame counter restarts at 0
    pushTime();
    pushOperationMode();
    pushBattery();
    return state.handle;
  },
  free_session() {
    api().switch_free_session(handle());
    state.handle = -1;
    resetInput();
    // Every host file the freed session was reading through went with it, and
    // the page re-registers what the next one needs.
    resetHostFiles();
    return 0;
  },

  set_trace(on) {
    api().switch_set_trace(handle(), on ? 1 : 0);
    return 0;
  },
  set_jit(on) {
    api().switch_set_jit(handle(), on ? 1 : 0);
    return 0;
  },
  vibration() {
    return api().switch_vibration(handle());
  },
  set_input(mask, slx, sly, srx, sry) {
    setGamepad(mask, slx, sly, srx, sry);
    return 0;
  },
  set_touch(points) {
    setTouch(points);
    return 0;
  },
  set_battery(percent, charging) {
    lastBattery = { percent, charging: !!charging };
    pushBattery();
    return 0;
  },
  set_operation_mode(next) {
    docked = !!next;
    pushOperationMode();
    return 0;
  },

  load_font(bytes) {
    return withBytes(bytes, (ptr, len) => api().switch_load_font(handle(), ptr, len));
  },
  load_nro(bytes) {
    return withBytes(bytes, (ptr, len) => Number(api().switch_load_nro(handle(), ptr, len)));
  },
  load_elf(bytes) {
    return withBytes(bytes, (ptr, len) => Number(api().switch_load_elf(handle(), ptr, len)));
  },

  // Open a container: the File is kept here and read range by range, so this
  // costs nothing but its PFS0 header no matter how large the file is.
  open_nsp(file) {
    return api().switch_open_nsp(handle(), openHostFile(file));
  },
  // Same, for a standalone .nca - the container is the NCA, with no file
  // table in front of it.
  open_nca(file) {
    return api().switch_open_nca(handle(), openHostFile(file));
  },
  // Register a firmware NCA as a system data archive. Costs nothing but the
  // reference and its header until a title mounts it - which is as true of a
  // Blob out of the page's NAND as of a File the user just picked, so this is
  // the one way in for both.
  add_archive(file) {
    const index = addHostFile(file);
    return api().switch_add_archive(handle(), index, BigInt(file.size));
  },
  // Register an update container for the title in the open container. Like
  // `add_archive` this keeps only the File reference, so an update costs its
  // header and its ticket and nothing else. Returns the title id it patches -
  // the base game's, which is what the page pairs the two containers by - or
  // '' if the file is not an update.
  add_update(file) {
    const index = addHostFile(file);
    const id = api().switch_add_update(handle(), index, BigInt(file.size));
    return id ? id.toString(16).padStart(16, '0') : '';
  },
  // The update's own version string, out of its Control NCA's NACP. Empty if
  // it ships without one.
  update_version() {
    return readString(256, (buf, cap) => api().switch_update_version(handle(), buf, cap));
  },
  // Register a container of add-on content. Like `add_archive` this keeps
  // only the File reference; which title the content belongs to is settled at
  // launch, against the id the title itself declares. Returns how many pieces
  // the container holds.
  add_dlc(file) {
    const index = addHostFile(file);
    return api().switch_add_dlc(handle(), index, BigInt(file.size));
  },
  // What the session holds: content id, base title id and index, per piece.
  dlc_json() {
    return readString(8192, (buf, cap) => api().switch_dlc_json(handle(), buf, cap));
  },
  clear_dlc() {
    api().switch_clear_dlc(handle());
    return 0;
  },
  // Drop it again: the next launch is the plain title.
  clear_update() {
    api().switch_clear_update(handle());
    return 0;
  },
  // What a firmware NCA is, without reading it: a header read through the
  // File the page is still holding. Returns { id, kind } - kind 0 for a
  // program, 1 for a data archive, 2 for anything else - or null if it is not
  // an NCA this build can read. A firmware dump is mostly the third kind, and
  // this is what keeps the page from pulling all of it through memory to find
  // that out.
  nand_identify(file) {
    const index = addHostFile(file);
    return withBuffer(4, (kindPtr) => {
      const id = api().switch_nand_identify(handle(), index, BigInt(file.size), kindPtr);
      const kind = new DataView(api().memory.buffer).getUint32(kindPtr, true);
      return id ? { id: id.toString(16).padStart(16, '0'), kind } : null;
    });
  },
  // Boot a program the host has the bytes of: a title installed on the NAND
  // rather than one opened out of a container the user just picked. The
  // emulator keeps its own copy, so the staging buffer goes back immediately.
  nand_launch(bytes) {
    return withBytes(bytes, (ptr, len) => Number(api().switch_nand_launch(handle(), ptr, len)));
  },
  // Decrypts NSP file `index` as a Program NCA (with whatever keys are
  // loaded) and boots its ExeFS `main` executable, reading both out of the
  // open container. Its RomFS is left where it is and decrypted on demand
  // while the title runs.
  load_nca_from_nsp(index) {
    return Number(api().switch_load_nca_from_nsp(handle(), index));
  },
  // Same, for a container that is itself a single standalone .nca.
  load_nca() {
    return Number(api().switch_load_nca(handle()));
  },
  // Which file in the open container holds the title's executable. Every file
  // in an NSP is named after its own hash, so this is the only way to boot one
  // without reading each header through the page to find out.
  program_nca_index() {
    return api().switch_program_nca_index(handle());
  },
  load_keys(prod, title) {
    const prodBytes = prod || new Uint8Array(0);
    const titleBytes = title || new Uint8Array(0);
    return withBytes(prodBytes, (pptr, plen) =>
      withBytes(titleBytes, (tptr, tlen) =>
        api().switch_load_keys(handle(), plen ? pptr : 0, plen, tlen ? tptr : 0, tlen)));
  },
  nsp_files_json() {
    return readString(8192, (buf, cap) => api().switch_nsp_files_json(handle(), buf, cap));
  },
  read_file(index, offset, len) {
    return withBuffer(len, (buf) => {
      // file_offset is a wasm u64 (needs a BigInt going in) and the return is
      // an i64 (comes back as a BigInt too) - convert that back to a Number
      // before using it as a length.
      const got = Number(api().switch_read_file(handle(), index, BigInt(offset), buf, len));
      if (got < 0) return { error: lastError() };
      return fromWasm(buf, got);
    });
  },

  // The title's name, publisher, version and icon, out of the Control NCA in
  // the open container. Cheap next to the container itself: a Control NCA is
  // an icon and a metadata blob, not game data.
  load_control_from_nsp() {
    return api().switch_load_control_from_nsp(handle());
  },
  // Same, for a container that is itself a single standalone Control NCA.
  load_control_from_nca() {
    return api().switch_load_control_from_nca(handle());
  },
  control_json() {
    // Sized for the worst case rather than the usual one: the JSON carries a
    // 0x200-byte name and a 0x100-byte publisher straight out of the NACP,
    // and `switch_control_json` truncates silently rather than saying it
    // overflowed - which would surface as a JSON parse error, not a clue.
    return readString(16384, (buf, cap) => api().switch_control_json(handle(), buf, cap));
  },
  // `size` comes from control_json's icon_size: the icon is a JPEG of
  // unpredictable length, so JS is told how big a buffer to hand over.
  control_icon(size) {
    if (!size) return new Uint8Array(0);
    return withBuffer(size, (buf) => {
      const n = Number(api().switch_control_icon(handle(), buf, size));
      return n > 0 ? fromWasm(buf, n) : new Uint8Array(0);
    });
  },
  parse_nca(header) {
    return withBytes(header, (ptr, len) =>
      readString(4096, (buf, cap) => api().switch_parse_nca(handle(), ptr, len, buf, cap)));
  },

  run(budget) {
    pushTime();
    const steps = Number(api().switch_run(handle(), BigInt(budget)));
    releaseLatchIfSeen();
    return steps;
  },
  halted() {
    return api().switch_halted(handle());
  },
  drain_output() {
    return drain((h, b, l) => api().switch_drain_output(h, b, l), 4096);
  },
  drain_trace() {
    return drain((h, b, l) => api().switch_drain_trace(h, b, l), 8192);
  },
  dump_regs() {
    return readString(2048, (buf, cap) => api().switch_dump_regs(handle(), buf, cap));
  },
  thread_dump() {
    return readString(8192, (buf, cap) => api().switch_thread_dump(handle(), buf, cap));
  },
  backtrace(depth) {
    return readJson<number[]>(
      1024,
      (buf, cap) => api().switch_backtrace_json(handle(), depth, buf, cap),
      [],
    );
  },
  wake_blocked() {
    return api().switch_wake_blocked(handle());
  },
  start_created_threads() {
    return api().switch_start_created_threads(handle());
  },
  ipc_gaps() {
    if (handle() < 0) return { unimplemented: [], stubbed: [] };
    return readJson<IpcGaps>(
      64 * 1024,
      (buf, cap) => api().switch_unimplemented_json(handle(), buf, cap),
      { unimplemented: [], stubbed: [] },
    );
  },
  // Big, because the trace is in it and the trace is the point: a report
  // truncated to a tidy size is one that leaves out the run-up to the fault.
  crash_report() {
    return readJson<CrashReport>(
      1024 * 1024,
      (buf, cap) => api().switch_crash_report_json(handle(), buf, cap),
      { version: 'unknown', panicked: false, traceMask: 0 },
    );
  },
  trace_channels() {
    return readJson<TraceChannel[]>(
      4096,
      (buf, cap) => api().switch_trace_channels_json(buf, cap),
      [],
    );
  },
  set_trace_mask(mask) {
    api().switch_set_trace_mask(mask);
  },
  version() {
    return readString(128, (buf, cap) => api().switch_version(buf, cap));
  },
  get_pc() {
    return api().switch_get_pc(handle());
  },
  get_cycles() {
    return Number(api().switch_get_cycles(handle()));
  },
  get_steps() {
    return Number(api().switch_get_steps(handle()));
  },
  get_reg(i) {
    return '0x' + api().switch_get_reg(handle(), i).toString(16).padStart(16, '0');
  },
  // Guest RAM is what the emulated console has touched; wasm is what this
  // worker's linear memory costs the browser (the page table, the loaded
  // image and every staging buffer live there).
  ram() {
    return {
      guest: handle() < 0 ? 0 : Number(api().switch_guest_ram(handle())),
      wasm: api().memory.buffer.byteLength,
    };
  },
  jit_stats() {
    if (handle() < 0) {
      return { enabled: false, blocks: 0, translated: 0, executed: 0, invalidated: 0 };
    }
    return readJson<JitStats>(
      256,
      (buf, cap) => api().switch_jit_stats_json(handle(), buf, cap),
      { enabled: false, blocks: 0, translated: 0, executed: 0, invalidated: 0 },
    );
  },
  gpu_report() {
    if (handle() < 0) return {};
    return readJson<GpuReport>(
      2048,
      (buf, cap) => api().switch_gpu_report_json(handle(), buf, cap),
      {},
    );
  },
  last_error() {
    return lastError();
  },

  fb_width() {
    return api().switch_fb_width(handle());
  },
  fb_height() {
    return api().switch_fb_height(handle());
  },
  frame_count() {
    return api().switch_frame_count(handle());
  },
  fb_snapshot(len) {
    return withBuffer(len, (buf) => {
      const n = api().switch_fb_snapshot(handle(), buf, len);
      return n > 0 ? fromWasm(buf, n) : null;
    });
  },

  audio_format() {
    return api().switch_audio_format(handle());
  },
  // Interleaved 16-bit PCM, as raw bytes. The main thread reinterprets them
  // as an Int16Array rather than paying for a second copy here.
  audio_pull(maxSamples) {
    return withBuffer(maxSamples * 2, (buf) => {
      const n = api().switch_audio_pull(handle(), buf, maxSamples);
      return n > 0 ? fromWasm(buf, n * 2) : null;
    });
  },

  // the emulated SD card
  //
  // `Vfs` lives in the session, so on its own nothing the guest writes
  // survives a reload. The main thread mirrors it into IndexedDB using these:
  // `sd_write_file`/`sd_create_dir` restore the card before a boot, and
  // `sd_take_changes` reports what the guest touched so only that is written
  // back.

  sd_write_file(path, bytes) {
    return withPath(path, (pptr, plen) =>
      withBytes(bytes, (dptr, dlen) =>
        api().switch_sd_write_file(handle(), pptr, plen, dptr, dlen)));
  },
  sd_create_dir(path) {
    return withPath(path, (ptr, len) => api().switch_sd_create_dir(handle(), ptr, len));
  },
  sd_remove(path) {
    return withPath(path, (ptr, len) => api().switch_sd_remove(handle(), ptr, len));
  },
  // The whole file, or null when the path is not one. Read in slices so a
  // large save does not need a single allocation twice its size.
  sd_read_file(path) {
    return withPath(path, (pptr, plen) => {
      const size = Number(api().switch_sd_file_size(handle(), pptr, plen));
      if (size < 0) return null;
      const out = new Uint8Array(size);
      const cap = Math.min(Math.max(size, 1), READ_CHUNK);
      return withBuffer(cap, (buf) => {
        let off = 0;
        while (off < size) {
          const n = Number(
            api().switch_sd_read_file(handle(), pptr, plen, BigInt(off), buf, cap));
          if (n <= 0) break;
          out.set(fromWasm(buf, n), off);
          off += n;
        }
        return out;
      });
    });
  },
  sd_pending_changes() {
    return handle() < 0 ? 0 : api().switch_sd_pending_changes(handle());
  },
  sd_take_changes() {
    if (handle() < 0) return [];
    const pending = api().switch_sd_pending_changes(handle());
    if (!pending) return [];
    return readJson<FsChange[]>(
      changesCap(pending),
      (buf, cap) => api().switch_sd_take_changes_json(handle(), buf, cap),
      [],
    );
  },

  // save data
  //
  // The same calls as the SD card above with a save id in front, because a
  // console keeps saves on its NAND rather than its card and one title's save
  // is not something another title can see.

  save_ids() {
    if (handle() < 0) return [];
    return readJson<string[]>(
      4096,
      (buf, cap) => api().switch_save_ids_json(handle(), buf, cap),
      [],
    );
  },
  save_pending_changes(id) {
    return handle() < 0 ? 0 : api().switch_save_pending_changes(handle(), saveId(id));
  },
  save_take_changes(id) {
    if (handle() < 0) return [];
    const save = saveId(id);
    const pending = api().switch_save_pending_changes(handle(), save);
    if (!pending) return [];
    return readJson<FsChange[]>(
      changesCap(pending),
      (buf, cap) => api().switch_save_take_changes_json(handle(), save, buf, cap),
      [],
    );
  },
  save_create(id) {
    return api().switch_save_create(handle(), saveId(id));
  },
  save_write_file(id, path, bytes) {
    return withPath(path, (pptr, plen) =>
      withBytes(bytes, (dptr, dlen) =>
        api().switch_save_write_file(handle(), saveId(id), pptr, plen, dptr, dlen)));
  },
  save_create_dir(id, path) {
    return withPath(path, (ptr, len) =>
      api().switch_save_create_dir(handle(), saveId(id), ptr, len));
  },
  // The whole file, or null when the path is not one. Sliced, so a large save
  // does not need one allocation twice its size.
  save_read_file(id, path) {
    const save = saveId(id);
    return withPath(path, (pptr, plen) => {
      const size = Number(api().switch_save_file_size(handle(), save, pptr, plen));
      if (size < 0) return null;
      const out = new Uint8Array(size);
      const cap = Math.min(Math.max(size, 1), READ_CHUNK);
      return withBuffer(cap, (buf) => {
        let off = 0;
        while (off < size) {
          const n = Number(
            api().switch_save_read_file(handle(), save, pptr, plen, BigInt(off), buf, cap));
          if (n <= 0) break;
          out.set(fromWasm(buf, n), off);
          off += n;
        }
        return out;
      });
    });
  },
};
