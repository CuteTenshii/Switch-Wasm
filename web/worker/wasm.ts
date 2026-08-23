/* The wasm module and the session it holds, plus the buffer plumbing every
   command goes through.

   The ABI is C: pointers and lengths into the module's linear memory, u64
   arguments and returns as BigInt. Nothing here knows what a command means -
   `commands.ts` does. */

import type { Bytes } from '../shared/protocol';

export interface WasmExports {
  memory: WebAssembly.Memory;

  switch_alloc(len: number): number;
  switch_free(ptr: number, len: number): void;

  switch_new(): number;
  switch_free_session(handle: number): void;
  switch_last_error(handle: number, buf: number, maxlen: number): number;

  switch_open_nsp(handle: number, size: bigint): number;
  switch_open_nca(handle: number, size: bigint): number;
  switch_add_archive(handle: number, file: number, size: bigint): number;
  switch_nand_add_archive(handle: number, ptr: number, len: number): bigint;
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

  switch_load_keys(
    handle: number, prodPtr: number, prodLen: number,
    titlePtr: number, titleLen: number): number;
  switch_load_font(handle: number, ptr: number, len: number): number;
  switch_load_nro(handle: number, ptr: number, len: number): bigint;
  switch_load_elf(handle: number, ptr: number, len: number): bigint;
  switch_load_nca(handle: number): bigint;
  switch_load_nca_from_nsp(handle: number, index: number): bigint;

  switch_set_trace(handle: number, enabled: number): void;
  switch_set_jit(handle: number, enabled: number): void;
  switch_jit_stats_json(handle: number, buf: number, maxlen: number): number;
  switch_set_input(
    handle: number, buttons: bigint,
    stickLx: number, stickLy: number, stickRx: number, stickRy: number): void;
  switch_set_touch(handle: number, ptr: number, count: number): void;
  switch_set_time(handle: number, unixSeconds: bigint): void;
  switch_set_battery(handle: number, percent: number, charging: number): void;
  switch_vibration(handle: number): number;

  switch_run(handle: number, maxSteps: bigint): bigint;
  switch_halted(handle: number): number;
  switch_drain_output(handle: number, buf: number, maxlen: number): number;
  switch_drain_trace(handle: number, buf: number, maxlen: number): number;
  switch_dump_regs(handle: number, buf: number, maxlen: number): number;
  switch_get_pc(handle: number): number;
  switch_get_reg(handle: number, idx: number): bigint;
  switch_get_cycles(handle: number): bigint;
  switch_guest_ram(handle: number): bigint;

  switch_fb_width(handle: number): number;
  switch_fb_height(handle: number): number;
  switch_frame_count(handle: number): number;
  switch_fb_snapshot(handle: number, buf: number, maxlen: number): number;

  switch_audio_format(handle: number): number;
  switch_audio_pull(handle: number, buf: number, maxSamples: number): number;

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
  switch_save_create(handle: number, saveId: bigint): number;
  switch_save_pending_changes(handle: number, saveId: bigint): number;
  switch_save_take_changes_json(
    handle: number, saveId: bigint, buf: number, maxlen: number): number;
  switch_save_write_file(
    handle: number, saveId: bigint, pathPtr: number, pathLen: number,
    dataPtr: number, dataLen: number): number;
  switch_save_create_dir(
    handle: number, saveId: bigint, pathPtr: number, pathLen: number): number;
  switch_save_file_size(
    handle: number, saveId: bigint, pathPtr: number, pathLen: number): bigint;
  switch_save_read_file(
    handle: number, saveId: bigint, pathPtr: number, pathLen: number,
    offset: bigint, buf: number, maxlen: number): bigint;
}

/** The instance and the session it is running. One object rather than two
 *  exported `let`s, so a module that imports this sees the current session
 *  rather than a copy of whatever it was at import time. */
export const state: { exports: WasmExports | null; handle: number } = {
  exports: null,
  handle: -1,
};

/** The wasm exports, or a clear error if the module never instantiated -
 *  which otherwise surfaces as a `null is not an object` from whichever
 *  command the page happened to send first. */
export function api(): WasmExports {
  if (!state.exports) throw new Error('the emulator core did not load');
  return state.exports;
}

export function handle(): number {
  return state.handle;
}

// Every buffer that crosses into wasm goes through here. A refused request
// comes back as null (switch_alloc will not trap on one), and writing at
// address 0 would corrupt the module's own data rather than fail, so this is
// where an impossible size has to stop.
export function alloc(len: number): number {
  const ptr = api().switch_alloc(len);
  if (!ptr) throw new Error('cannot allocate ' + len + ' bytes in the emulator');
  return ptr;
}

export function free(ptr: number, len: number): void {
  api().switch_free(ptr, len);
}

export function toWasm(jsbuf: Bytes, ptr: number): void {
  const view = new Uint8Array(api().memory.buffer, ptr, jsbuf.length);
  view.set(jsbuf);
}

export function fromWasm(ptr: number, len: number): Bytes {
  return new Uint8Array(api().memory.buffer, ptr, len).slice();
}

/** Hand `bytes` to `body` as a wasm buffer and give the buffer back however
 *  `body` ends: the staging copy is the emulator's own heap, and leaking one
 *  per load is memory the browser never gets back. A zero-length buffer is
 *  still allocated, because `alloc` refuses address 0. */
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

/** The same for an out-parameter: a scratch buffer of `cap` bytes. */
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

/** A guest path as the fs protocol carries it: UTF-8, no terminator. */
export function withPath<T>(path: string, body: (ptr: number, len: number) => T): T {
  return withBytes(encoder.encode(path), body);
}

/** Read a string out of a buffer the module fills. */
export function readString(cap: number, fill: (ptr: number, cap: number) => number): string {
  return withBuffer(cap, (ptr) => decode(fromWasm(ptr, fill(ptr, cap))));
}

/** The same, parsed - falling back rather than throwing when the module wrote
 *  nothing, which is what an empty list looks like. */
export function readJson<T>(
  cap: number,
  fill: (ptr: number, cap: number) => number,
  fallback: T,
): T {
  const text = readString(cap, fill);
  try {
    return JSON.parse(text) as T;
  } catch {
    return fallback;
  }
}

export function lastError(): string {
  const s = readString(512, (buf, cap) => api().switch_last_error(state.handle, buf, cap));
  return s.replace(/\u0000.*$/, '');
}

// Drain a ring buffer (output/trace) to completion, concatenating the chunks.
export function drain(
  fn: (handle: number, buf: number, cap: number) => number,
  cap: number,
): Bytes {
  const chunks: Bytes[] = [];
  for (;;) {
    const n = withBuffer(cap, (buf) => {
      const got = fn(state.handle, buf, cap);
      if (got > 0) chunks.push(fromWasm(buf, got));
      return got;
    });
    if (n < cap) break;
  }
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}
