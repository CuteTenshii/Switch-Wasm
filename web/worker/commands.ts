// Command handlers: each returns a plain value or `{ error }` for `index.ts` to reply with.

import type {
  CommandHandlers, CrashReport, FsChange, GpuReport, IpcGaps, JitStats, KeyboardRequest,
  UserRecord,
} from '../shared/protocol';
import { fmtSize } from '../shared/format';
import { noteRegistered, resetActivity } from './activity';
import { addHostFile, openHostFile, resetHostFiles } from './hostfiles';
import { workerLog } from './log';
import { releaseLatchIfSeen, resetInput, setGamepad, setTouch } from './latch';
import {
  api,
  drain,
  fromWasm,
  handle,
  lastError,
  readJson,
  readString,
  readWholeJson,
  state,
  withBuffer,
  withBytes,
  withPath,
} from './wasm';

// wasm32-unknown-unknown has no clock; push the host time into the emulated RTC.
function pushTime(): void {
  if (handle() < 0) return;
  api().switch_set_time(handle(), BigInt(Math.floor(Date.now() / 1000)));
}

// The Battery Status API is Window-only, so the main thread sends it. Cached for new sessions.
let lastBattery = { percent: 100, charging: true };
function pushBattery(): void {
  if (handle() < 0) return;
  api().switch_set_battery(handle(), lastBattery.percent, lastBattery.charging ? 1 : 0);
}

// Cached so a reset session keeps the dock state.
let docked = false;
function pushOperationMode(): void {
  if (handle() < 0) return;
  api().switch_set_operation_mode(handle(), docked ? 1 : 0);
}

// A uid (32 hex digits in memory order) as two little-endian halves.
export function uidHalves(hex: string): [bigint, bigint] {
  const half = (from: number) => {
    let value = 0n;
    for (let i = 7; i >= 0; i--) {
      value = (value << 8n) | BigInt(parseInt(hex.slice(from + i * 2, from + i * 2 + 2), 16));
    }
    return value;
  };
  return [half(0), half(16)];
}

// `<hex save id>[@<user uid>]`; no uid means no owning user.
function saveKey(id: string): [bigint, bigint, bigint] {
  const [save, user] = id.split('@');
  const [lo, hi] = user ? uidHalves(user) : [0n, 0n];
  return [BigInt('0x' + save), lo, hi];
}

// The wasm side drains whether or not the JSON fits, so size for 0x301-byte paths per entry.
const changesCap = (pending: number) => 2 + pending * (0x301 * 2 + 64);

const READ_CHUNK = 1 << 20;

const NACP_SIZE = 0x4000;

function describe(file: Blob): string {
  const name = file instanceof File ? `"${file.name}" ` : '';
  return name + '(' + fmtSize(file.size) + ')';
}

// `result` is negative on failure.
function logLoad(what: string, result: number | bigint): void {
  if (Number(result) < 0) workerLog(`[io] ${what}: refused (${lastError()})`, 'warn');
  else workerLog(`[io] ${what}`);
}

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
    resetActivity();
    resetHostFiles();
    return 0;
  },

  set_trace(on) {
    api().switch_set_trace(handle(), on ? 1 : 0);
    return 0;
  },
  set_trace_channels(on) {
    api().switch_set_trace_channels(on ? 1 : 0);
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
    const result = withBytes(bytes, (ptr, len) => api().switch_load_font(handle(), ptr, len));
    logLoad(`loaded the shared font (${fmtSize(bytes.length)})`, result);
    return result;
  },
  load_nro(bytes) {
    const result =
      withBytes(bytes, (ptr, len) => Number(api().switch_load_nro(handle(), ptr, len)));
    logLoad(`loaded an NRO (${fmtSize(bytes.length)})`, result);
    return result;
  },
  load_elf(bytes) {
    const result =
      withBytes(bytes, (ptr, len) => Number(api().switch_load_elf(handle(), ptr, len)));
    logLoad(`loaded an ELF (${fmtSize(bytes.length)})`, result);
    return result;
  },

  // The File is kept and read by range.
  open_nsp(file) {
    const result = api().switch_open_nsp(handle(), openHostFile(file));
    logLoad(`opened ${describe(file)} as the container`, result);
    return result;
  },
  open_nca(file) {
    const result = api().switch_open_nca(handle(), openHostFile(file));
    logLoad(`opened ${describe(file)} as a standalone NCA`, result);
    return result;
  },
  // Register a firmware NCA (picked File or NAND Blob) as a system data archive.
  add_archive(file) {
    const index = addHostFile(file);
    const result = api().switch_add_archive(handle(), index, BigInt(file.size));
    if (result < 0) {
      workerLog(`[io] system archive ${describe(file)}: refused (${lastError()})`, 'warn');
    } else noteRegistered('system archives', file.size);
    return result;
  },
  // Returns the base title id the update patches, or '' if it is not an update.
  add_update(file) {
    const index = addHostFile(file);
    const id = api().switch_add_update(handle(), index, BigInt(file.size));
    const title = id ? id.toString(16).padStart(16, '0') : '';
    if (title) workerLog(`[io] opened ${describe(file)} as an update for ${title}`);
    else workerLog(`[io] ${describe(file)} is not an update`);
    return title;
  },
  update_version() {
    return readString(256, (buf, cap) => api().switch_update_version(handle(), buf, cap));
  },
  // Returns how many pieces the add-on content container holds.
  add_dlc(file) {
    const index = addHostFile(file);
    const pieces = api().switch_add_dlc(handle(), index, BigInt(file.size));
    // Zero, not a negative, is a refusal.
    if (pieces === 0) {
      workerLog(`[io] add-on content ${describe(file)}: refused (${lastError()})`, 'warn');
    } else {
      workerLog(`[io] opened ${describe(file)} as add-on content, ${pieces} pieces`);
    }
    return pieces;
  },
  dlc_json() {
    return readString(8192, (buf, cap) => api().switch_dlc_json(handle(), buf, cap));
  },
  clear_dlc() {
    api().switch_clear_dlc(handle());
    return 0;
  },
  clear_update() {
    api().switch_clear_update(handle());
    return 0;
  },
  // Header-only probe. Kind 0 is a program, 1 a data archive, 2 anything else; null if unreadable.
  nand_identify(file) {
    const index = addHostFile(file);
    noteRegistered('NAND files to identify', file.size);
    return withBuffer(4, (kindPtr) => {
      const id = api().switch_nand_identify(handle(), index, BigInt(file.size), kindPtr);
      const kind = new DataView(api().memory.buffer).getUint32(kindPtr, true);
      return id ? { id: id.toString(16).padStart(16, '0'), kind } : null;
    });
  },
  nand_launch(bytes) {
    const result =
      withBytes(bytes, (ptr, len) => Number(api().switch_nand_launch(handle(), ptr, len)));
    logLoad(`launched a program from the NAND (${fmtSize(bytes.length)})`, result);
    return result;
  },
  load_nca_from_nsp(index) {
    const result = Number(api().switch_load_nca_from_nsp(handle(), index));
    logLoad(`booted the program in container file ${index}`, result);
    return result;
  },
  load_nca() {
    const result = Number(api().switch_load_nca(handle()));
    logLoad('booted the program in the standalone NCA', result);
    return result;
  },
  program_nca_index() {
    return api().switch_program_nca_index(handle());
  },
  load_keys(prod, title) {
    const prodBytes = prod || new Uint8Array(0);
    const titleBytes = title || new Uint8Array(0);
    const result = withBytes(prodBytes, (pptr, plen) =>
      withBytes(titleBytes, (tptr, tlen) =>
        api().switch_load_keys(handle(), plen ? pptr : 0, plen, tlen ? tptr : 0, tlen)));
    logLoad(
      `loaded keys: prod ${fmtSize(prodBytes.length)}, title ${fmtSize(titleBytes.length)}`,
      result,
    );
    return result;
  },
  nsp_files_json() {
    return readString(8192, (buf, cap) => api().switch_nsp_files_json(handle(), buf, cap));
  },
  read_file(index, offset, len) {
    return withBuffer(len, (buf) => {
      const got = Number(api().switch_read_file(handle(), index, BigInt(offset), buf, len));
      if (got < 0) return { error: lastError() };
      return fromWasm(buf, got);
    });
  },

  load_control_from_nsp() {
    return api().switch_load_control_from_nsp(handle());
  },
  load_control_from_nca() {
    return api().switch_load_control_from_nca(handle());
  },
  control_json() {
    // Worst case: the export truncates silently.
    return readString(16384, (buf, cap) => api().switch_control_json(handle(), buf, cap));
  },
  // `size` is control_json's icon_size.
  control_icon(size) {
    if (!size) return new Uint8Array(0);
    return withBuffer(size, (buf) => {
      const n = Number(api().switch_control_icon(handle(), buf, size));
      return n > 0 ? fromWasm(buf, n) : new Uint8Array(0);
    });
  },
  control_nacp() {
    return withBuffer(NACP_SIZE, (buf) => {
      const n = Number(api().switch_control_nacp(handle(), buf, NACP_SIZE));
      return n > 0 ? fromWasm(buf, n) : new Uint8Array(0);
    });
  },
  add_application_control(titleId, nacp, icon) {
    return withBytes(nacp, (nptr, nlen) =>
      withBytes(icon, (iptr, ilen) =>
        api().switch_add_application_control(
          handle(), BigInt('0x' + titleId), nptr, nlen, iptr, ilen)));
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
  guest_fatal() {
    return readString(2048, (buf, cap) => api().switch_guest_fatal(handle(), buf, cap));
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
  // Large enough to keep the trace.
  crash_report() {
    return readJson<CrashReport>(
      1024 * 1024,
      (buf, cap) => api().switch_crash_report_json(handle(), buf, cap),
      { version: 'unknown', panicked: false, traceMask: 0 },
    );
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
  ram() {
    return {
      guest: handle() < 0 ? 0 : Number(api().switch_guest_ram(handle())),
      wasm: api().memory.buffer.byteLength,
    };
  },
  jit_stats() {
    if (handle() < 0) {
      return { enabled: false, blocks: 0, translated: 0, executed: 0, linked: 0, invalidated: 0 };
    }
    return readWholeJson<JitStats>(
      256,
      (buf, cap) => api().switch_jit_stats_json(handle(), buf, cap),
      { enabled: false, blocks: 0, translated: 0, executed: 0, linked: 0, invalidated: 0 },
    );
  },
  gpu_report() {
    if (handle() < 0) return {};
    return readWholeJson<GpuReport>(
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
  // Interleaved 16-bit PCM as raw bytes.
  audio_pull(maxSamples) {
    return withBuffer(maxSamples * 2, (buf) => {
      const n = api().switch_audio_pull(handle(), buf, maxSamples);
      return n > 0 ? fromWasm(buf, n * 2) : null;
    });
  },

  // SD card, mirrored to IndexedDB by the main thread.

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
  // The whole file, read in slices, or null when the path is not one.
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

  // Save data: the SD card calls keyed by save.

  save_ids() {
    if (handle() < 0) return [];
    return readJson<string[]>(
      4096,
      (buf, cap) => api().switch_save_ids_json(handle(), buf, cap),
      [],
    );
  },
  save_pending_changes(id) {
    return handle() < 0 ? 0 : api().switch_save_pending_changes(handle(), ...saveKey(id));
  },
  save_take_changes(id) {
    if (handle() < 0) return [];
    const save = saveKey(id);
    const pending = api().switch_save_pending_changes(handle(), ...save);
    if (!pending) return [];
    return readJson<FsChange[]>(
      changesCap(pending),
      (buf, cap) => api().switch_save_take_changes_json(handle(), ...save, buf, cap),
      [],
    );
  },
  save_create(id) {
    return api().switch_save_create(handle(), ...saveKey(id));
  },
  save_write_file(id, path, bytes) {
    return withPath(path, (pptr, plen) =>
      withBytes(bytes, (dptr, dlen) =>
        api().switch_save_write_file(handle(), ...saveKey(id), pptr, plen, dptr, dlen)));
  },
  save_create_dir(id, path) {
    return withPath(path, (ptr, len) =>
      api().switch_save_create_dir(handle(), ...saveKey(id), ptr, len));
  },
  save_remove(id, path) {
    return withPath(path, (ptr, len) =>
      api().switch_save_remove(handle(), ...saveKey(id), ptr, len));
  },
  // The whole file, read in slices, or null when the path is not one.
  save_read_file(id, path) {
    const save = saveKey(id);
    return withPath(path, (pptr, plen) => {
      const size = Number(api().switch_save_file_size(handle(), ...save, pptr, plen));
      if (size < 0) return null;
      const out = new Uint8Array(size);
      const cap = Math.min(Math.max(size, 1), READ_CHUNK);
      return withBuffer(cap, (buf) => {
        let off = 0;
        while (off < size) {
          const n = Number(
            api().switch_save_read_file(handle(), ...save, pptr, plen, BigInt(off), buf, cap));
          if (n <= 0) break;
          out.set(fromWasm(buf, n), off);
          off += n;
        }
        return out;
      });
    });
  },

  // Users: staged one at a time, committed whole before a title starts.

  users_set(users, current) {
    if (handle() < 0) return 1;
    for (const user of users) {
      const [lo, hi] = uidHalves(user.uid);
      withPath(user.nickname, (nptr, nlen) =>
        withBytes(user.picture ?? new Uint8Array(0), (pptr, plen) =>
          api().switch_user_stage(
            handle(), lo, hi, nptr, nlen, BigInt(user.editedAt), pptr,
            user.picture ? plen : 0)));
    }
    return api().switch_users_commit(handle(), ...uidHalves(current));
  },
  users_take_edits() {
    return handle() >= 0 && api().switch_take_profile_edits(handle()) !== 0;
  },
  users_read() {
    if (handle() < 0) return [];
    const users = readJson<(Omit<UserRecord, 'picture'> & { pictureLen: number })[]>(
      8192,
      (buf, cap) => api().switch_users_json(handle(), buf, cap),
      [],
    );
    return users.map(({ pictureLen, ...user }) => ({
      ...user,
      picture: pictureLen
        ? withBuffer(pictureLen, (buf) =>
            fromWasm(buf, api().switch_user_picture(handle(), ...uidHalves(user.uid), buf, pictureLen)))
        : null,
    }));
  },

  keyboard_request() {
    if (handle() < 0) return null;
    return readJson<KeyboardRequest | null>(
      8192, (buf, cap) => api().switch_keyboard_json(handle(), buf, cap), null);
  },
  keyboard_answer(text) {
    if (handle() < 0) return 0;
    return withPath(text ?? '', (ptr, len) =>
      api().switch_keyboard_answer(handle(), ptr, len, text === null ? 0 : 1));
  },
};
