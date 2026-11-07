// Command handlers: each returns a plain value or `{ error }` for `index.ts` to reply with.

import type {
  CommandHandlers, DlcEntry, GpuReport, JitStats, MemoryReport, NandIdentity, NspFile, UpdateInfo,
} from '../shared/protocol';
import { fmtSize } from '../shared/format';
import { noteRegistered } from './activity';
import { addHostFile, openHostFile } from './hostfiles';
import { flushLog, workerLog } from './log';
import {
  switch_add_dlc, switch_add_update, switch_clear_dlc, switch_clear_screen, switch_clear_update,
  switch_control_icon, switch_control_json, switch_end_title, switch_flush_storage, switch_get_reg,
  switch_get_steps, switch_gpu_report_json, switch_guest_fatal, switch_guest_ram, switch_halted,
  switch_jit_stats_json, switch_keyboard_answer, switch_last_error, switch_load_control,
  switch_load_keys, switch_load_nca, switch_load_nca_from_nsp, switch_load_program,
  switch_load_sd, switch_memory_json, switch_nand_identify, switch_nand_launch, switch_net_connect,
  switch_net_connections_json, switch_net_disconnect, switch_next_load,
  switch_open_nca,
  switch_open_nsp, switch_parse_nca, switch_program_nca_index, switch_read_file, switch_reg_dump,
  switch_remember_control, switch_restore_nand, switch_restore_storage, switch_run,
  switch_set_battery, switch_set_jit, switch_set_operation_mode, switch_set_touch,
  switch_set_trace, switch_set_trace_channels, switch_store_profile_edits, switch_thread_dump,
  switch_user_stage, switch_users_commit,
} from '@core/switch_wasm.js';
import { pauseLoop, startLoop, stopLoop } from './loop';
import { nandContent, STORAGE } from './storage';
import { core, parseOr } from './wasm';

function describe(file: Blob): string {
  const name = file instanceof File ? `"${file.name}" ` : '';
  return name + '(' + fmtSize(file.size) + ')';
}

export const CMD = {
  endTitle() {
    stopLoop();
    switch_end_title();
  },

  setTrace: switch_set_trace,
  setTraceChannels: switch_set_trace_channels,
  setJit: switch_set_jit,
  setTouch: switch_set_touch,
  // The Battery Status API is Window-only, so the main thread sends it.
  setBattery: switch_set_battery,
  setOperationMode: switch_set_operation_mode,
  connectNetwork: switch_net_connect,
  disconnectNetwork: switch_net_disconnect,
  netConnections: switch_net_connections_json,

  // The core reads the file itself, so the page never holds a copy.
  loadProgram: switch_load_program,
  loadSd: switch_load_sd,
  nextLoad: switch_next_load,

  // The File is kept and read by range.
  openNsp(file) {
    return JSON.parse(switch_open_nsp(openHostFile(file))) as NspFile[];
  },
  openNca(file) {
    const result = switch_open_nca(openHostFile(file));
    workerLog(`[io] opened ${describe(file)} as a standalone NCA`);
    return result;
  },
  // Register the stored firmware with the session; how many system data archives it took.
  restoreNand: switch_restore_nand,
  // The base title the update patches and its version, or null if it is not an update.
  addUpdate(file) {
    const index = addHostFile(file);
    const found = switch_add_update(index, BigInt(file.size));
    if (!found) return null;
    const update = JSON.parse(found) as UpdateInfo;
    workerLog(`[io] opened ${describe(file)} as an update for ${update.title}`);
    return update;
  },
  // The pieces the add-on content container holds.
  addDlc(file) {
    const index = addHostFile(file);
    const pieces = JSON.parse(switch_add_dlc(index, BigInt(file.size))) as DlcEntry[];
    if (pieces.length > 0) {
      workerLog(`[io] opened ${describe(file)} as add-on content, ${pieces.length} pieces`);
    }
    return pieces;
  },
  clearDlc: switch_clear_dlc,
  clearUpdate: switch_clear_update,
  // Header-only probe. Kind 0 is a program, 1 a data archive, 2 anything else; null if unreadable.
  nandIdentify(file) {
    const index = addHostFile(file);
    noteRegistered('NAND files to identify', file.size);
    const found = switch_nand_identify(index, BigInt(file.size));
    return found ? JSON.parse(found) as NandIdentity : null;
  },
  // The content stays in the browser's copy and the core reads it by range.
  async nandLaunch(name) {
    const content = await nandContent(name);
    if (!content) throw new Error(name + ' is indexed but its content is missing.');
    const entry = switch_nand_launch(content);
    workerLog(`[io] launched ${name} from the NAND`);
    return entry;
  },
  loadNcaFromNsp: switch_load_nca_from_nsp,
  loadNca: switch_load_nca,
  programNcaIndex: switch_program_nca_index,
  loadKeys: switch_load_keys,
  readFile(index, offset, len) {
    return switch_read_file(index, BigInt(offset), len);
  },

  loadControl: switch_load_control,
  controlJson: switch_control_json,
  controlIcon: switch_control_icon,
  // Store the control data just read, for the next session's ns.
  rememberControl: switch_remember_control,
  parseNca: switch_parse_nca,

  run(budget) {
    return switch_run(budget);
  },
  startLoop,
  pauseLoop,
  restoreStorage: switch_restore_storage,
  flushStorage: switch_flush_storage,
  clearScreen: switch_clear_screen,
  halted: switch_halted,
  guestFatal: switch_guest_fatal,
  flushLog: flushLog,
  regDump: switch_reg_dump,
  threadDump: switch_thread_dump,
  getSteps: switch_get_steps,
  getReg(i) {
    return '0x' + switch_get_reg(i).toString(16).padStart(16, '0');
  },
  ram() {
    return {
      guest: Number(switch_guest_ram()),
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
  jitStats() {
    return parseOr<JitStats>(
      switch_jit_stats_json(),
      { enabled: false, blocks: 0, translated: 0, executed: 0, linked: 0, invalidated: 0 },
    );
  },
  gpuReport() {
    return parseOr<GpuReport>(switch_gpu_report_json(), {});
  },
  lastError: switch_last_error,

  // Users: staged one at a time, committed whole before a title starts.

  usersSet(users, current) {
    for (const user of users) {
      switch_user_stage(user.uid, user.nickname, user.editedAt, user.picture);
    }
    return switch_users_commit(current);
  },
  storeProfileEdits: switch_store_profile_edits,

  keyboardAnswer: switch_keyboard_answer,

  ...STORAGE,
} satisfies CommandHandlers;
