// Command handlers: each returns a plain value or `{ error }` for `index.ts` to reply with.

import type {
  CommandHandlers, GpuReport, JitStats, MemoryReport, NandIdentity, UpdateInfo,
} from '../shared/protocol';
import { fmtSize } from '../shared/format';
import { noteRegistered } from './activity';
import { addHostFile, openHostFile } from './hostfiles';
import { flushLog, workerLog } from './log';
import {
  switch_add_update, switch_clear_screen, switch_control_icon, switch_control_json, switch_dlc_json,
  switch_flush_storage, switch_gpu_report_json, switch_guest_fatal, switch_jit_stats_json,
  switch_keyboard_answer, switch_last_error, switch_load_keys, switch_load_program,
  switch_load_sd_nro, switch_memory_json, switch_nand_identify, switch_nand_launch,
  switch_nsp_files_json, switch_parse_nca, switch_read_file, switch_remember_control,
  switch_restore_nand, switch_restore_storage, switch_set_touch, switch_store_profile_edits,
  switch_user_stage, switch_users_commit,
} from '@core/switch_wasm.js';
import { pauseLoop, startLoop, stopLoop } from './loop';
import { nandContent, STORAGE } from './storage';
import { core, parseOr } from './wasm';

const NOTHING = new Uint8Array(0);

function describe(file: Blob): string {
  const name = file instanceof File ? `"${file.name}" ` : '';
  return name + '(' + fmtSize(file.size) + ')';
}

// `result` is negative on failure.
function logLoad(what: string, result: number | bigint): void {
  if (Number(result) < 0) workerLog(`[io] ${what}: refused (${switch_last_error()})`, 'warn');
  else workerLog(`[io] ${what}`);
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
    switch_set_touch(points);
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

  // The core reads the file itself, so the page never holds a copy.
  async load_program(file) {
    const result = await switch_load_program(file);
    if (result < 0) logLoad(`loaded ${describe(file)} as a program`, result);
    return result;
  },
  load_sd_nro(path) {
    const result = switch_load_sd_nro(path);
    if (result < 0) logLoad(`loaded the NRO at ${path}`, result);
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
  // The base title the update patches and its version, or null if it is not an update.
  add_update(file) {
    const index = addHostFile(file);
    const found = switch_add_update(index, BigInt(file.size));
    if (!found) {
      workerLog(`[io] ${describe(file)} is not an update`);
      return null;
    }
    const update = JSON.parse(found) as UpdateInfo;
    workerLog(`[io] opened ${describe(file)} as an update for ${update.title}`);
    return update;
  },
  // Returns how many pieces the add-on content container holds.
  add_dlc(file) {
    const index = addHostFile(file);
    const pieces = core.switch_add_dlc(index, BigInt(file.size));
    // Zero, not a negative, is a refusal.
    if (pieces === 0) {
      workerLog(`[io] add-on content ${describe(file)}: refused (${switch_last_error()})`, 'warn');
    } else {
      workerLog(`[io] opened ${describe(file)} as add-on content, ${pieces} pieces`);
    }
    return pieces;
  },
  dlc_json: switch_dlc_json,
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
    const found = switch_nand_identify(index, BigInt(file.size));
    return found ? JSON.parse(found) as NandIdentity : null;
  },
  // The content stays in the browser's copy and the core reads it by range.
  async nand_launch(name) {
    const content = await nandContent(name);
    if (!content) throw new Error(name + ' is indexed but its content is missing.');
    const result = switch_nand_launch(content);
    logLoad(`launched ${name} from the NAND`, result);
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
    const prodBytes = prod ?? NOTHING;
    const titleBytes = title ?? NOTHING;
    const result = switch_load_keys(prodBytes, titleBytes);
    logLoad(
      `loaded keys: prod ${fmtSize(prodBytes.length)}, title ${fmtSize(titleBytes.length)}`,
      result,
    );
    return result;
  },
  nsp_files_json: switch_nsp_files_json,
  read_file(index, offset, len) {
    return switch_read_file(index, BigInt(offset), len);
  },

  load_control() {
    return core.switch_load_control();
  },
  control_json: switch_control_json,
  control_icon: switch_control_icon,
  // Store the control data just read, for the next session's ns.
  remember_control: switch_remember_control,
  parse_nca: switch_parse_nca,

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
  guest_fatal: switch_guest_fatal,
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
    return { ...parseOr(switch_memory_json(), none), wasmHeap };
  },
  jit_stats() {
    return parseOr<JitStats>(
      switch_jit_stats_json(),
      { enabled: false, blocks: 0, translated: 0, executed: 0, linked: 0, invalidated: 0 },
    );
  },
  gpu_report() {
    return parseOr<GpuReport>(switch_gpu_report_json(), {});
  },
  last_error: switch_last_error,

  fb_width() {
    return core.switch_fb_width();
  },
  fb_height() {
    return core.switch_fb_height();
  },
  frame_count() {
    return core.switch_frame_count();
  },

  // Users: staged one at a time, committed whole before a title starts.

  users_set(users, current) {
    for (const user of users) {
      switch_user_stage(user.uid, user.nickname, user.editedAt, user.picture);
    }
    return switch_users_commit(current);
  },
  store_profile_edits: switch_store_profile_edits,

  keyboard_answer: switch_keyboard_answer,

  ...STORAGE,
};
