// The wasm module, its session, and the buffer plumbing commands go through.
// C ABI: pointers and lengths into linear memory, u64 as BigInt.

import type { Bytes } from '../shared/protocol';

export interface WasmExports {
  memory: WebAssembly.Memory;

  switch_alloc(len: number): number;
  switch_free(ptr: number, len: number): void;

  switch_init(): void;
  switch_new(): number;
  switch_add_core(handle: number): number;
  switch_end_title(handle: number): void;
  switch_last_error(handle: number, buf: number, maxlen: number): number;

  switch_open_nsp(handle: number, size: bigint): number;
  switch_open_nca(handle: number, size: bigint): number;
  switch_add_archive(handle: number, file: number, size: bigint): number;
  switch_add_applet(handle: number, file: number, size: bigint): number;
  switch_add_update(handle: number, file: number, size: bigint): bigint;
  switch_update_version(handle: number, buf: number, maxlen: number): number;
  switch_add_dlc(handle: number, file: number, size: bigint): number;
  switch_dlc_json(handle: number, buf: number, maxlen: number): number;
  switch_clear_dlc(handle: number): void;
  switch_clear_update(handle: number): void;
  switch_nand_identify(handle: number, file: number, size: bigint, kindOut: number): bigint;
  switch_nand_launch(handle: number, ptr: number, len: number): bigint;
  switch_nsp_files_json(handle: number, buf: number, maxlen: number): number;
  switch_read_file(
    handle: number, index: number, fileOffset: bigint, buf: number, maxlen: number): bigint;
  switch_parse_nca(
    handle: number, ptr: number, len: number, buf: number, maxlen: number): number;

  switch_load_control_from_nsp(handle: number): number;
  switch_load_control_from_nca(handle: number): number;
  switch_control_json(handle: number, buf: number, maxlen: number): number;
  switch_control_icon(handle: number, buf: number, maxlen: number): bigint;
  switch_control_nacp(handle: number, buf: number, maxlen: number): bigint;
  switch_add_application_control(
    handle: number, applicationId: bigint, nacpPtr: number, nacpLen: number,
    iconPtr: number, iconLen: number): number;

  switch_load_keys(
    handle: number, prodPtr: number, prodLen: number,
    titlePtr: number, titleLen: number): number;
  switch_load_font(handle: number, ptr: number, len: number): number;
  switch_load_nro(
    handle: number, ptr: number, len: number, pathPtr: number, pathLen: number): bigint;
  switch_load_elf(handle: number, ptr: number, len: number): bigint;
  switch_load_nca(handle: number): bigint;
  switch_load_nca_from_nsp(handle: number, index: number): bigint;
  switch_program_nca_index(handle: number): number;

  switch_set_trace(handle: number, enabled: number): void;
  switch_set_jit(handle: number, enabled: number): void;
  switch_set_trace_channels(on: number): void;
  switch_jit_stats_json(handle: number, buf: number, maxlen: number): number;
  switch_gpu_report_json(handle: number, buf: number, maxlen: number): number;
  switch_gpu_lost(handle: number): number;
  switch_log_activity(handle: number, seconds: number): void;
  switch_log_next(handle: number): number;
  switch_log_line(buf: number, maxlen: number): number;
  switch_set_input(
    handle: number, buttons: bigint,
    stickLx: number, stickLy: number, stickRx: number, stickRy: number): void;
  switch_set_touch(handle: number, ptr: number, count: number): void;
  switch_set_battery(handle: number, percent: number, charging: number): void;
  switch_set_operation_mode(handle: number, docked: number): void;
  switch_vibration(handle: number): number;

  switch_run(handle: number, maxSteps: bigint): bigint;
  switch_halted(handle: number): number;
  switch_guest_fatal(handle: number, buf: number, maxlen: number): number;
  switch_log_regs(handle: number): void;
  switch_log_threads(handle: number): void;
  switch_get_reg(handle: number, idx: number): bigint;
  switch_get_steps(handle: number): bigint;
  switch_guest_ram(handle: number): bigint;
  switch_memory_json(handle: number, buf: number, maxlen: number): number;

  switch_fb_width(handle: number): number;
  switch_fb_height(handle: number): number;
  switch_frame_count(handle: number): number;
  switch_fb_snapshot(handle: number, buf: number, maxlen: number): number;

  switch_audio_format(handle: number): number;
  switch_audio_feed(handle: number): void;
  switch_audio_ring(): number;
  switch_audio_ring_samples(): number;

  switch_sd_write_file(
    handle: number, pathPtr: number, pathLen: number,
    dataPtr: number, dataLen: number): number;
  switch_sd_create_dir(handle: number, pathPtr: number, pathLen: number): number;
  switch_sd_remove(handle: number, pathPtr: number, pathLen: number): number;
  switch_sd_file_size(handle: number, pathPtr: number, pathLen: number): bigint;
  switch_sd_read_file(
    handle: number, pathPtr: number, pathLen: number,
    offset: bigint, buf: number, maxlen: number): bigint;
  switch_sd_pending_changes(handle: number): number;
  switch_sd_take_changes_json(handle: number, buf: number, maxlen: number): number;

  switch_save_ids_json(handle: number, buf: number, maxlen: number): number;

  switch_user_stage(
    handle: number, uidLo: bigint, uidHi: bigint, namePtr: number, nameLen: number,
    editedAt: bigint, picturePtr: number, pictureLen: number): void;
  switch_users_commit(handle: number, currentLo: bigint, currentHi: bigint): number;
  switch_take_profile_edits(handle: number): number;
  switch_keyboard_json(handle: number, buf: number, maxlen: number): number;
  switch_error_dialog_json(handle: number, buf: number, maxlen: number): number;
  switch_keyboard_answer(
    handle: number, textPtr: number, textLen: number, submitted: number): number;
  switch_users_json(handle: number, buf: number, maxlen: number): number;
  switch_user_picture(
    handle: number, uidLo: bigint, uidHi: bigint, buf: number, maxlen: number): number;
  // A save is keyed by its id and its user's uid in two halves.
  switch_save_pending_changes(
    handle: number, saveId: bigint, userLo: bigint, userHi: bigint): number;
  switch_save_take_changes_json(
    handle: number, saveId: bigint, userLo: bigint, userHi: bigint,
    buf: number, maxlen: number): number;
  switch_save_write_file(
    handle: number, saveId: bigint, userLo: bigint, userHi: bigint,
    pathPtr: number, pathLen: number, dataPtr: number, dataLen: number): number;
  switch_save_create_dir(
    handle: number, saveId: bigint, userLo: bigint, userHi: bigint,
    pathPtr: number, pathLen: number): number;
  switch_save_remove(
    handle: number, saveId: bigint, userLo: bigint, userHi: bigint,
    pathPtr: number, pathLen: number): number;
  switch_save_file_size(
    handle: number, saveId: bigint, userLo: bigint, userHi: bigint,
    pathPtr: number, pathLen: number): bigint;
  switch_save_read_file(
    handle: number, saveId: bigint, userLo: bigint, userHi: bigint,
    pathPtr: number, pathLen: number, offset: bigint, buf: number, maxlen: number): bigint;
}

export const state = { handle: -1 };

// The core's exports, set once at load. A live binding: importers see the assignment.
export let core = null as unknown as WasmExports;

export function setCore(exports: WasmExports): void {
  core = exports;
}

// IndexedDB commands can arrive while the module is still loading.
export function coreLoaded(): boolean {
  return (core as WasmExports | null) !== null;
}

// A refused allocation returns 0. `>>> 0` because the i32 reads signed past 2 GiB.
export function alloc(len: number): number {
  const ptr = core.switch_alloc(len) >>> 0;
  if (!ptr) throw new Error('cannot allocate ' + len + ' bytes in the emulator');
  return ptr;
}

export function free(ptr: number, len: number): void {
  core.switch_free(ptr, len);
}

export function toWasm(jsbuf: Bytes, ptr: number): void {
  const view = new Uint8Array(core.memory.buffer, ptr, jsbuf.length);
  view.set(jsbuf);
}

export function fromWasm(ptr: number, len: number): Bytes {
  return new Uint8Array(core.memory.buffer, ptr, len).slice();
}

// Stage `bytes` in wasm for `body`, always freeing it. Empty still allocates one byte.
export function withBytes<T>(bytes: Bytes, body: (ptr: number, len: number) => T): T {
  const len = bytes.length;
  const ptr = alloc(len || 1);
  try {
    if (len) toWasm(bytes, ptr);
    return body(ptr, len);
  } finally {
    free(ptr, len || 1);
  }
}

// A scratch out-buffer of `cap` bytes, always freed.
export function withBuffer<T>(cap: number, body: (ptr: number) => T): T {
  const ptr = alloc(cap);
  try {
    return body(ptr);
  } finally {
    free(ptr, cap);
  }
}

const decoder = new TextDecoder();
const encoder = new TextEncoder();

export function decode(bytes: Bytes): string {
  return decoder.decode(bytes);
}

// A guest path as UTF-8 with no terminator.
export function withPath<T>(path: string, body: (ptr: number, len: number) => T): T {
  return withBytes(encoder.encode(path), body);
}

export function readString(cap: number, fill: (ptr: number, cap: number) => number): string {
  return withBuffer(cap, (ptr) => decode(fromWasm(ptr, fill(ptr, cap))));
}

const WHOLE_ANSWER_MAX = 1 << 20;

// Retry with a doubled buffer while the answer fills it. Only for idempotent reads.
export function readWholeString(cap: number, fill: (ptr: number, cap: number) => number): string {
  for (let size = cap; ; size *= 2) {
    const [written, text] = withBuffer(size, (ptr): [number, string] => {
      const n = fill(ptr, size);
      return [n, decode(fromWasm(ptr, n))];
    });
    if (written < size || size >= WHOLE_ANSWER_MAX) return text;
  }
}

// An empty answer is an empty list, so fall back instead of throwing.
function parseOr<T>(text: string, fallback: T): T {
  try {
    return JSON.parse(text) as T;
  } catch {
    return fallback;
  }
}

export function readWholeJson<T>(
  cap: number,
  fill: (ptr: number, cap: number) => number,
  fallback: T,
): T {
  return parseOr(readWholeString(cap, fill), fallback);
}

export function readJson<T>(
  cap: number,
  fill: (ptr: number, cap: number) => number,
  fallback: T,
): T {
  return parseOr(readString(cap, fill), fallback);
}

export function lastError(): string {
  const s = readString(2048, (buf, cap) => core.switch_last_error(state.handle, buf, cap));
  // The buffer is NUL-padded past the message.
  const end = s.indexOf('\0');
  return end < 0 ? s : s.slice(0, end);
}
