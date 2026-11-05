// Command handlers: each returns a plain value or `{ error }` for `index.ts` to reply with.

import type {
  Bytes, CommandHandlers, GpuReport, JitStats, MemoryReport,
} from '../shared/protocol';
import { idbGet, SD_STORE, sdIdb, type StoredEntry } from '../shared/db';
import { fmtSize } from '../shared/format';
import { noteRegistered } from './activity';
import { addHostFile, openHostFile } from './hostfiles';
import { flushLog, workerLog } from './log';
import {
  switch_clear_screen, switch_flush_storage, switch_remember_control, switch_restore_nand,
  switch_restore_storage, switch_store_profile_edits,
} from '@core/switch_wasm.js';
import { pauseLoop, startLoop, stopLoop } from './loop';
import { nandContent, saveKey, STORAGE, uidHalves } from './storage';
import {
  core,
  fromWasm,
  lastError,
  readJson,
  readString,
  readWholeJson,
  withBuffer,
  withBytes,
  withPath,
} from './wasm';

const READ_CHUNK = 1 << 20;

function describe(file: Blob): string {
  const name = file instanceof File ? `"${file.name}" ` : '';
  return name + '(' + fmtSize(file.size) + ')';
}

// `result` is negative on failure.
function logLoad(what: string, result: number | bigint): void {
  if (Number(result) < 0) workerLog(`[io] ${what}: refused (${lastError()})`, 'warn');
  else workerLog(`[io] ${what}`);
}

function loadNro(bytes: Bytes, sdPath?: string): number {
  const result = withBytes(bytes, (ptr, len) => withPath(sdPath ?? '', (pathPtr, pathLen) =>
    Number(core.switch_load_nro(ptr, len, pathPtr, pathLen))));
  if (result < 0) logLoad(`loaded an NRO (${fmtSize(bytes.length)})${sdPath ? ' from ' + sdPath : ''}`, result);
  return result;
}

export const CMD: CommandHandlers = {
  end_title() {
    stopLoop();
    core.switch_end_title();
    return 0;
  },

  set_trace(on) {
    core.switch_set_trace(on ? 1 : 0);
    return 0;
  },
  set_trace_channels(on) {
    core.switch_set_trace_channels(on ? 1 : 0);
  },
  set_jit(on) {
    core.switch_set_jit(on ? 1 : 0);
    return 0;
  },
  vibration() {
    return core.switch_vibration();
  },
  set_input(mask, slx, sly, srx, sry) {
    core.switch_set_input(BigInt(mask), slx, sly, srx, sry);
    return 0;
  },
  set_touch(points) {
    const bytes = new Uint8Array(points.buffer, points.byteOffset, points.byteLength) as Bytes;
    withBytes(bytes, (ptr) => core.switch_set_touch(ptr, points.length / 3));
    return 0;
  },
  // The Battery Status API is Window-only, so the main thread sends it.
  set_battery(percent, charging) {
    core.switch_set_battery(percent, charging ? 1 : 0);
    return 0;
  },
  set_operation_mode(next) {
    core.switch_set_operation_mode(next ? 1 : 0);
    return 0;
  },

  async load_nro(file) {
    return loadNro(new Uint8Array(await file.arrayBuffer()));
  },
  async load_sd_nro(path) {
    const entry = await idbGet<StoredEntry>(await sdIdb(), SD_STORE, path);
    if (entry?.kind !== 'file') throw new Error(path + ' is not on the SD card.');
    return loadNro(entry.data ?? new Uint8Array(0), path);
  },
  async load_elf(file) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const result =
      withBytes(bytes, (ptr, len) => Number(core.switch_load_elf(ptr, len)));
    logLoad(`loaded an ELF (${fmtSize(bytes.length)})`, result);
    return result;
  },

  // The File is kept and read by range.
  open_nsp(file) {
    const result = core.switch_open_nsp(openHostFile(file));
    logLoad(`opened ${describe(file)} as the container`, result);
    return result;
  },
  open_nca(file) {
    const result = core.switch_open_nca(openHostFile(file));
    logLoad(`opened ${describe(file)} as a standalone NCA`, result);
    return result;
  },
  // Register the stored firmware with the session; how many system data archives it took.
  restore_nand: switch_restore_nand,
  // Returns the base title id the update patches, or '' if it is not an update.
  add_update(file) {
    const index = addHostFile(file);
    const id = core.switch_add_update(index, BigInt(file.size));
    const title = id ? id.toString(16).padStart(16, '0') : '';
    if (title) workerLog(`[io] opened ${describe(file)} as an update for ${title}`);
    else workerLog(`[io] ${describe(file)} is not an update`);
    return title;
  },
  update_version() {
    return readString(256, (buf, cap) => core.switch_update_version(buf, cap));
  },
  // Returns how many pieces the add-on content container holds.
  add_dlc(file) {
    const index = addHostFile(file);
    const pieces = core.switch_add_dlc(index, BigInt(file.size));
    // Zero, not a negative, is a refusal.
    if (pieces === 0) {
      workerLog(`[io] add-on content ${describe(file)}: refused (${lastError()})`, 'warn');
    } else {
      workerLog(`[io] opened ${describe(file)} as add-on content, ${pieces} pieces`);
    }
    return pieces;
  },
  dlc_json() {
    return readString(8192, (buf, cap) => core.switch_dlc_json(buf, cap));
  },
  clear_dlc() {
    core.switch_clear_dlc();
    return 0;
  },
  clear_update() {
    core.switch_clear_update();
    return 0;
  },
  // Header-only probe. Kind 0 is a program, 1 a data archive, 2 anything else; null if unreadable.
  nand_identify(file) {
    const index = addHostFile(file);
    noteRegistered('NAND files to identify', file.size);
    return withBuffer(4, (kindPtr) => {
      const id = core.switch_nand_identify(index, BigInt(file.size), kindPtr);
      const kind = new DataView(core.memory.buffer).getUint32(kindPtr, true);
      return id ? { id: id.toString(16).padStart(16, '0'), kind } : null;
    });
  },
  async nand_launch(name) {
    const content = await nandContent(name);
    if (!content) throw new Error(name + ' is indexed but its content is missing.');
    const bytes = new Uint8Array(await content.arrayBuffer());
    const result =
      withBytes(bytes, (ptr, len) => Number(core.switch_nand_launch(ptr, len)));
    logLoad(`launched a program from the NAND (${fmtSize(bytes.length)})`, result);
    return result;
  },
  load_nca_from_nsp(index) {
    const result = Number(core.switch_load_nca_from_nsp(index));
    logLoad(`booted the program in container file ${index}`, result);
    return result;
  },
  load_nca() {
    const result = Number(core.switch_load_nca());
    logLoad('booted the program in the standalone NCA', result);
    return result;
  },
  program_nca_index() {
    return core.switch_program_nca_index();
  },
  load_keys(prod, title) {
    const prodBytes = prod || new Uint8Array(0);
    const titleBytes = title || new Uint8Array(0);
    const result = withBytes(prodBytes, (pptr, plen) =>
      withBytes(titleBytes, (tptr, tlen) =>
        core.switch_load_keys(plen ? pptr : 0, plen, tlen ? tptr : 0, tlen)));
    logLoad(
      `loaded keys: prod ${fmtSize(prodBytes.length)}, title ${fmtSize(titleBytes.length)}`,
      result,
    );
    return result;
  },
  nsp_files_json() {
    return readString(8192, (buf, cap) => core.switch_nsp_files_json(buf, cap));
  },
  read_file(index, offset, len) {
    return withBuffer(len, (buf) => {
      const got = Number(core.switch_read_file(index, BigInt(offset), buf, len));
      if (got < 0) return { error: lastError() };
      return fromWasm(buf, got);
    });
  },

  load_control_from_nsp() {
    return core.switch_load_control_from_nsp();
  },
  load_control_from_nca() {
    return core.switch_load_control_from_nca();
  },
  control_json() {
    // Worst case: the export truncates silently.
    return readString(16384, (buf, cap) => core.switch_control_json(buf, cap));
  },
  // `size` is control_json's icon_size.
  control_icon(size) {
    if (!size) return new Uint8Array(0);
    return withBuffer(size, (buf) => {
      const n = Number(core.switch_control_icon(buf, size));
      return n > 0 ? fromWasm(buf, n) : new Uint8Array(0);
    });
  },
  // Store the control data just read, for the next session's ns.
  remember_control: switch_remember_control,
  parse_nca(header) {
    return withBytes(header, (ptr, len) =>
      readString(4096, (buf, cap) => core.switch_parse_nca(ptr, len, buf, cap)));
  },

  run(budget) {
    return Number(core.switch_run(BigInt(budget)));
  },
  start_loop(tracing) {
    startLoop(tracing);
    return 0;
  },
  pause_loop() {
    pauseLoop();
    return 0;
  },
  restore_storage: switch_restore_storage,
  flush_storage: switch_flush_storage,
  clear_screen() {
    switch_clear_screen();
    return 0;
  },
  halted() {
    return core.switch_halted();
  },
  guest_fatal() {
    return readString(2048, (buf, cap) => core.switch_guest_fatal(buf, cap));
  },
  flush_log() {
    return flushLog();
  },
  log_regs() {
    core.switch_log_regs();
    return 0;
  },
  log_threads() {
    core.switch_log_threads();
    return 0;
  },
  get_steps() {
    return Number(core.switch_get_steps());
  },
  get_reg(i) {
    return '0x' + core.switch_get_reg(i).toString(16).padStart(16, '0');
  },
  ram() {
    return {
      guest: Number(core.switch_guest_ram()),
      wasm: core.memory.buffer.byteLength,
    };
  },
  memory() {
    const wasmHeap = core.memory.buffer.byteLength;
    const none: Omit<MemoryReport, 'wasmHeap'> = {
      backed: 0, cap: 0, regions: [], elsewhere: 0, allProcesses: 0,
    };
    const report = readJson<Omit<MemoryReport, 'wasmHeap'>>(
      4096,
      (buf, cap) => core.switch_memory_json(buf, cap),
      none,
    );
    return { ...report, wasmHeap };
  },
  jit_stats() {
    return readWholeJson<JitStats>(
      256,
      (buf, cap) => core.switch_jit_stats_json(buf, cap),
      { enabled: false, blocks: 0, translated: 0, executed: 0, linked: 0, invalidated: 0 },
    );
  },
  gpu_report() {
    return readWholeJson<GpuReport>(
      2048,
      (buf, cap) => core.switch_gpu_report_json(buf, cap),
      {},
    );
  },
  last_error: lastError,

  fb_width() {
    return core.switch_fb_width();
  },
  fb_height() {
    return core.switch_fb_height();
  },
  frame_count() {
    return core.switch_frame_count();
  },
  fb_snapshot(len) {
    return withBuffer(len, (buf) => {
      const n = core.switch_fb_snapshot(buf, len);
      return n > 0 ? fromWasm(buf, n) : null;
    });
  },

  // SD card, mirrored to IndexedDB (`storage.ts`).

  // The whole file, read in slices, or null when the path is not one.
  sd_read_file(path) {
    return withPath(path, (pptr, plen) => {
      const size = Number(core.switch_sd_file_size(pptr, plen));
      if (size < 0) return null;
      const out = new Uint8Array(size);
      const cap = Math.min(Math.max(size, 1), READ_CHUNK);
      return withBuffer(cap, (buf) => {
        let off = 0;
        while (off < size) {
          const n = Number(
            core.switch_sd_read_file(pptr, plen, BigInt(off), buf, cap));
          if (n <= 0) break;
          out.set(fromWasm(buf, n), off);
          off += n;
        }
        return out;
      });
    });
  },

  // Save data: the SD card calls keyed by save.

  // The whole file, read in slices, or null when the path is not one.
  save_read_file(id, path) {
    const save = saveKey(id);
    return withPath(path, (pptr, plen) => {
      const size = Number(core.switch_save_file_size(...save, pptr, plen));
      if (size < 0) return null;
      const out = new Uint8Array(size);
      const cap = Math.min(Math.max(size, 1), READ_CHUNK);
      return withBuffer(cap, (buf) => {
        let off = 0;
        while (off < size) {
          const n = Number(
            core.switch_save_read_file(...save, pptr, plen, BigInt(off), buf, cap));
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
    for (const user of users) {
      const [lo, hi] = uidHalves(user.uid);
      withPath(user.nickname, (nptr, nlen) =>
        withBytes(user.picture ?? new Uint8Array(0), (pptr, plen) =>
          core.switch_user_stage(
            lo, hi, nptr, nlen, BigInt(user.editedAt), pptr,
            user.picture ? plen : 0)));
    }
    return core.switch_users_commit(...uidHalves(current));
  },
  store_profile_edits: switch_store_profile_edits,

  keyboard_answer(text) {
    return withPath(text ?? '', (ptr, len) =>
      core.switch_keyboard_answer(ptr, len, text === null ? 0 : 1));
  },

  ...STORAGE,
};
