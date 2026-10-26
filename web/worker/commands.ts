// Command handlers: each returns a plain value or `{ error }` for `index.ts` to reply with.

import type {
  CommandHandlers, ErrorDialog, FsChange, GpuReport, JitStats, KeyboardRequest,
  UserRecord,
} from '../shared/protocol';
import { fmtSize } from '../shared/format';
import { noteRegistered } from './activity';
import { serveCores } from './coreworkers';
import { addHostFile, openHostFile } from './hostfiles';
import { flushLog, workerLog } from './log';
import { switch_clear_screen } from '@core/switch_wasm.js';
import { pauseLoop, startLoop, stopLoop } from './loop';
import { flushStorage, restoreStorage } from './persist';
import { releaseLatchIfSeen, resetInput, setGamepad, setTouch } from './latch';
import {
  core,
  fromWasm,
  lastError,
  readJson,
  readString,
  readWholeJson,
  state,
  withBuffer,
  withBytes,
  withPath,
} from './wasm';

// The Battery Status API is Window-only, so the main thread sends it. Cached for new sessions.
let lastBattery = { percent: 100, charging: true };
function pushBattery(): void {
  if (state.handle < 0) return;
  core.switch_set_battery(state.handle, lastBattery.percent, lastBattery.charging ? 1 : 0);
}

// Cached so a reset session keeps the dock state.
let docked = false;
function pushOperationMode(): void {
  if (state.handle < 0) return;
  core.switch_set_operation_mode(state.handle, docked ? 1 : 0);
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
    state.handle = core.switch_new();
    serveCores(state.handle);
    resetInput(); // the new session's frame counter restarts at 0
    pushOperationMode();
    pushBattery();
    return state.handle;
  },
  end_title() {
    stopLoop();
    core.switch_end_title(state.handle);
    resetInput(); // the frame counter restarts at 0
    return 0;
  },

  set_trace(on) {
    core.switch_set_trace(state.handle, on ? 1 : 0);
    return 0;
  },
  set_trace_channels(on) {
    core.switch_set_trace_channels(on ? 1 : 0);
  },
  set_jit(on) {
    core.switch_set_jit(state.handle, on ? 1 : 0);
    return 0;
  },
  vibration() {
    return core.switch_vibration(state.handle);
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
    const result = withBytes(bytes, (ptr, len) => core.switch_load_font(state.handle, ptr, len));
    logLoad(`loaded the shared font (${fmtSize(bytes.length)})`, result);
    return result;
  },
  load_nro(bytes, sdPath) {
    const result = withBytes(bytes, (ptr, len) => withPath(sdPath ?? '', (pathPtr, pathLen) =>
      Number(core.switch_load_nro(state.handle, ptr, len, pathPtr, pathLen))));
    logLoad(`loaded an NRO (${fmtSize(bytes.length)})${sdPath ? ' from ' + sdPath : ''}`, result);
    return result;
  },
  load_elf(bytes) {
    const result =
      withBytes(bytes, (ptr, len) => Number(core.switch_load_elf(state.handle, ptr, len)));
    logLoad(`loaded an ELF (${fmtSize(bytes.length)})`, result);
    return result;
  },

  // The File is kept and read by range.
  open_nsp(file) {
    const result = core.switch_open_nsp(state.handle, openHostFile(file));
    logLoad(`opened ${describe(file)} as the container`, result);
    return result;
  },
  open_nca(file) {
    const result = core.switch_open_nca(state.handle, openHostFile(file));
    logLoad(`opened ${describe(file)} as a standalone NCA`, result);
    return result;
  },
  // Register a firmware NCA (picked File or NAND Blob) as a system data archive.
  add_archive(file) {
    const index = addHostFile(file);
    const result = core.switch_add_archive(state.handle, index, BigInt(file.size));
    if (result < 0) {
      workerLog(`[io] system archive ${describe(file)}: refused (${lastError()})`, 'warn');
    } else noteRegistered('system archives', file.size);
    return result;
  },
  // Register a library applet's Program NCA from the NAND.
  add_applet(file) {
    const result = core.switch_add_applet(state.handle, addHostFile(file), BigInt(file.size));
    if (result < 0) workerLog(`[io] applet ${describe(file)}: refused (${lastError()})`, 'warn');
    return result;
  },
  // Returns the base title id the update patches, or '' if it is not an update.
  add_update(file) {
    const index = addHostFile(file);
    const id = core.switch_add_update(state.handle, index, BigInt(file.size));
    const title = id ? id.toString(16).padStart(16, '0') : '';
    if (title) workerLog(`[io] opened ${describe(file)} as an update for ${title}`);
    else workerLog(`[io] ${describe(file)} is not an update`);
    return title;
  },
  update_version() {
    return readString(256, (buf, cap) => core.switch_update_version(state.handle, buf, cap));
  },
  // Returns how many pieces the add-on content container holds.
  add_dlc(file) {
    const index = addHostFile(file);
    const pieces = core.switch_add_dlc(state.handle, index, BigInt(file.size));
    // Zero, not a negative, is a refusal.
    if (pieces === 0) {
      workerLog(`[io] add-on content ${describe(file)}: refused (${lastError()})`, 'warn');
    } else {
      workerLog(`[io] opened ${describe(file)} as add-on content, ${pieces} pieces`);
    }
    return pieces;
  },
  dlc_json() {
    return readString(8192, (buf, cap) => core.switch_dlc_json(state.handle, buf, cap));
  },
  clear_dlc() {
    core.switch_clear_dlc(state.handle);
    return 0;
  },
  clear_update() {
    core.switch_clear_update(state.handle);
    return 0;
  },
  // Header-only probe. Kind 0 is a program, 1 a data archive, 2 anything else; null if unreadable.
  nand_identify(file) {
    const index = addHostFile(file);
    noteRegistered('NAND files to identify', file.size);
    return withBuffer(4, (kindPtr) => {
      const id = core.switch_nand_identify(state.handle, index, BigInt(file.size), kindPtr);
      const kind = new DataView(core.memory.buffer).getUint32(kindPtr, true);
      return id ? { id: id.toString(16).padStart(16, '0'), kind } : null;
    });
  },
  nand_launch(bytes) {
    const result =
      withBytes(bytes, (ptr, len) => Number(core.switch_nand_launch(state.handle, ptr, len)));
    logLoad(`launched a program from the NAND (${fmtSize(bytes.length)})`, result);
    return result;
  },
  load_nca_from_nsp(index) {
    const result = Number(core.switch_load_nca_from_nsp(state.handle, index));
    logLoad(`booted the program in container file ${index}`, result);
    return result;
  },
  load_nca() {
    const result = Number(core.switch_load_nca(state.handle));
    logLoad('booted the program in the standalone NCA', result);
    return result;
  },
  program_nca_index() {
    return core.switch_program_nca_index(state.handle);
  },
  load_keys(prod, title) {
    const prodBytes = prod || new Uint8Array(0);
    const titleBytes = title || new Uint8Array(0);
    const result = withBytes(prodBytes, (pptr, plen) =>
      withBytes(titleBytes, (tptr, tlen) =>
        core.switch_load_keys(state.handle, plen ? pptr : 0, plen, tlen ? tptr : 0, tlen)));
    logLoad(
      `loaded keys: prod ${fmtSize(prodBytes.length)}, title ${fmtSize(titleBytes.length)}`,
      result,
    );
    return result;
  },
  nsp_files_json() {
    return readString(8192, (buf, cap) => core.switch_nsp_files_json(state.handle, buf, cap));
  },
  read_file(index, offset, len) {
    return withBuffer(len, (buf) => {
      const got = Number(core.switch_read_file(state.handle, index, BigInt(offset), buf, len));
      if (got < 0) return { error: lastError() };
      return fromWasm(buf, got);
    });
  },

  load_control_from_nsp() {
    return core.switch_load_control_from_nsp(state.handle);
  },
  load_control_from_nca() {
    return core.switch_load_control_from_nca(state.handle);
  },
  control_json() {
    // Worst case: the export truncates silently.
    return readString(16384, (buf, cap) => core.switch_control_json(state.handle, buf, cap));
  },
  // `size` is control_json's icon_size.
  control_icon(size) {
    if (!size) return new Uint8Array(0);
    return withBuffer(size, (buf) => {
      const n = Number(core.switch_control_icon(state.handle, buf, size));
      return n > 0 ? fromWasm(buf, n) : new Uint8Array(0);
    });
  },
  control_nacp() {
    return withBuffer(NACP_SIZE, (buf) => {
      const n = Number(core.switch_control_nacp(state.handle, buf, NACP_SIZE));
      return n > 0 ? fromWasm(buf, n) : new Uint8Array(0);
    });
  },
  add_application_control(titleId, nacp, icon) {
    return withBytes(nacp, (nptr, nlen) =>
      withBytes(icon, (iptr, ilen) =>
        core.switch_add_application_control(
          state.handle, BigInt('0x' + titleId), nptr, nlen, iptr, ilen)));
  },
  parse_nca(header) {
    return withBytes(header, (ptr, len) =>
      readString(4096, (buf, cap) => core.switch_parse_nca(state.handle, ptr, len, buf, cap)));
  },

  run(budget) {
    const steps = Number(core.switch_run(state.handle, BigInt(budget)));
    releaseLatchIfSeen();
    return steps;
  },
  start_loop(tracing) {
    startLoop(tracing);
    return 0;
  },
  pause_loop() {
    pauseLoop();
    return 0;
  },
  restore_storage() {
    return restoreStorage();
  },
  async flush_storage() {
    await flushStorage();
    return 0;
  },
  clear_screen() {
    switch_clear_screen();
    return 0;
  },
  halted() {
    return core.switch_halted(state.handle);
  },
  guest_fatal() {
    return readString(2048, (buf, cap) => core.switch_guest_fatal(state.handle, buf, cap));
  },
  flush_log() {
    return flushLog();
  },
  log_regs() {
    core.switch_log_regs(state.handle);
    return 0;
  },
  log_threads() {
    core.switch_log_threads(state.handle);
    return 0;
  },
  get_steps() {
    return Number(core.switch_get_steps(state.handle));
  },
  get_reg(i) {
    return '0x' + core.switch_get_reg(state.handle, i).toString(16).padStart(16, '0');
  },
  ram() {
    return {
      guest: state.handle < 0 ? 0 : Number(core.switch_guest_ram(state.handle)),
      wasm: core.memory.buffer.byteLength,
    };
  },
  jit_stats() {
    if (state.handle < 0) {
      return { enabled: false, blocks: 0, translated: 0, executed: 0, linked: 0, invalidated: 0 };
    }
    return readWholeJson<JitStats>(
      256,
      (buf, cap) => core.switch_jit_stats_json(state.handle, buf, cap),
      { enabled: false, blocks: 0, translated: 0, executed: 0, linked: 0, invalidated: 0 },
    );
  },
  gpu_report() {
    if (state.handle < 0) return {};
    return readWholeJson<GpuReport>(
      2048,
      (buf, cap) => core.switch_gpu_report_json(state.handle, buf, cap),
      {},
    );
  },
  last_error() {
    return lastError();
  },

  fb_width() {
    return core.switch_fb_width(state.handle);
  },
  fb_height() {
    return core.switch_fb_height(state.handle);
  },
  frame_count() {
    return core.switch_frame_count(state.handle);
  },
  fb_snapshot(len) {
    return withBuffer(len, (buf) => {
      const n = core.switch_fb_snapshot(state.handle, buf, len);
      return n > 0 ? fromWasm(buf, n) : null;
    });
  },

  // SD card, mirrored to IndexedDB by the main thread.

  sd_write_file(path, bytes) {
    return withPath(path, (pptr, plen) =>
      withBytes(bytes, (dptr, dlen) =>
        core.switch_sd_write_file(state.handle, pptr, plen, dptr, dlen)));
  },
  sd_create_dir(path) {
    return withPath(path, (ptr, len) => core.switch_sd_create_dir(state.handle, ptr, len));
  },
  sd_remove(path) {
    return withPath(path, (ptr, len) => core.switch_sd_remove(state.handle, ptr, len));
  },
  // The whole file, read in slices, or null when the path is not one.
  sd_read_file(path) {
    return withPath(path, (pptr, plen) => {
      const size = Number(core.switch_sd_file_size(state.handle, pptr, plen));
      if (size < 0) return null;
      const out = new Uint8Array(size);
      const cap = Math.min(Math.max(size, 1), READ_CHUNK);
      return withBuffer(cap, (buf) => {
        let off = 0;
        while (off < size) {
          const n = Number(
            core.switch_sd_read_file(state.handle, pptr, plen, BigInt(off), buf, cap));
          if (n <= 0) break;
          out.set(fromWasm(buf, n), off);
          off += n;
        }
        return out;
      });
    });
  },
  sd_take_changes() {
    if (state.handle < 0) return [];
    const pending = core.switch_sd_pending_changes(state.handle);
    if (!pending) return [];
    return readJson<FsChange[]>(
      changesCap(pending),
      (buf, cap) => core.switch_sd_take_changes_json(state.handle, buf, cap),
      [],
    );
  },

  // Save data: the SD card calls keyed by save.

  save_ids() {
    if (state.handle < 0) return [];
    return readJson<string[]>(
      4096,
      (buf, cap) => core.switch_save_ids_json(state.handle, buf, cap),
      [],
    );
  },
  saves_pending() {
    if (state.handle < 0) return false;
    const ids = CMD.save_ids() as string[];
    return ids.some((id) => core.switch_save_pending_changes(state.handle, ...saveKey(id)) !== 0);
  },
  save_take_changes(id) {
    if (state.handle < 0) return [];
    const save = saveKey(id);
    const pending = core.switch_save_pending_changes(state.handle, ...save);
    if (!pending) return [];
    return readJson<FsChange[]>(
      changesCap(pending),
      (buf, cap) => core.switch_save_take_changes_json(state.handle, ...save, buf, cap),
      [],
    );
  },
  save_write_file(id, path, bytes) {
    return withPath(path, (pptr, plen) =>
      withBytes(bytes, (dptr, dlen) =>
        core.switch_save_write_file(state.handle, ...saveKey(id), pptr, plen, dptr, dlen)));
  },
  save_create_dir(id, path) {
    return withPath(path, (ptr, len) =>
      core.switch_save_create_dir(state.handle, ...saveKey(id), ptr, len));
  },
  save_remove(id, path) {
    return withPath(path, (ptr, len) =>
      core.switch_save_remove(state.handle, ...saveKey(id), ptr, len));
  },
  // The whole file, read in slices, or null when the path is not one.
  save_read_file(id, path) {
    const save = saveKey(id);
    return withPath(path, (pptr, plen) => {
      const size = Number(core.switch_save_file_size(state.handle, ...save, pptr, plen));
      if (size < 0) return null;
      const out = new Uint8Array(size);
      const cap = Math.min(Math.max(size, 1), READ_CHUNK);
      return withBuffer(cap, (buf) => {
        let off = 0;
        while (off < size) {
          const n = Number(
            core.switch_save_read_file(state.handle, ...save, pptr, plen, BigInt(off), buf, cap));
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
    if (state.handle < 0) return 1;
    for (const user of users) {
      const [lo, hi] = uidHalves(user.uid);
      withPath(user.nickname, (nptr, nlen) =>
        withBytes(user.picture ?? new Uint8Array(0), (pptr, plen) =>
          core.switch_user_stage(
            state.handle, lo, hi, nptr, nlen, BigInt(user.editedAt), pptr,
            user.picture ? plen : 0)));
    }
    return core.switch_users_commit(state.handle, ...uidHalves(current));
  },
  users_take_edits() {
    return state.handle >= 0 && core.switch_take_profile_edits(state.handle) !== 0;
  },
  users_read() {
    if (state.handle < 0) return [];
    const users = readJson<(Omit<UserRecord, 'picture'> & { pictureLen: number })[]>(
      8192,
      (buf, cap) => core.switch_users_json(state.handle, buf, cap),
      [],
    );
    return users.map(({ pictureLen, ...user }) => ({
      ...user,
      picture: pictureLen
        ? withBuffer(pictureLen, (buf) =>
            fromWasm(buf, core.switch_user_picture(state.handle, ...uidHalves(user.uid), buf, pictureLen)))
        : null,
    }));
  },

  error_dialog() {
    if (state.handle < 0) return null;
    // Taken from the queue; 16 KiB holds any message.
    return readJson<ErrorDialog | null>(
      16384, (buf, cap) => core.switch_error_dialog_json(state.handle, buf, cap), null);
  },
  keyboard_request() {
    if (state.handle < 0) return null;
    return readJson<KeyboardRequest | null>(
      8192, (buf, cap) => core.switch_keyboard_json(state.handle, buf, cap), null);
  },
  keyboard_answer(text) {
    if (state.handle < 0) return 0;
    return withPath(text ?? '', (ptr, len) =>
      core.switch_keyboard_answer(state.handle, ptr, len, text === null ? 0 : 1));
  },
};
