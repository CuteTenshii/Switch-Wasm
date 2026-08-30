/* The contract between the page and the worker that hosts the emulator.

   Both sides compile against this: the page's `call('run', slice)` is checked
   against the handler the worker actually implements, and a handler that
   returns the wrong shape is a build error rather than a `undefined is not a
   function` somewhere in the run loop. */

/** Byte payloads always own a plain ArrayBuffer: they are either read from a
 *  File or sliced out of wasm memory, and saying so is what lets one be
 *  handed straight to ImageData, Blob or a transfer list. */
export type Bytes = Uint8Array<ArrayBuffer>;

/** One path the guest touched, as `switch_sd_take_changes_json` reports it. */
export interface FsChange {
  kind: 'file' | 'dir' | 'deleted';
  path: string;
}

/** Guest RAM is the emulated console's own memory use; `wasm` is what the
 *  worker's linear memory costs the browser. */
export interface RamUsage {
  guest: number;
  wasm: number;
}

/** What the block translator has been doing. `executed` counts blocks
 *  entered, so `executed / translated` is how much each translation paid for
 *  itself; `invalidated` counts blocks dropped because the guest wrote over
 *  the code they came from. */
/** What the installed GPU backend has been doing. `backend` is absent while
 *  the software rasterizer has the frame — it never declines a draw, so it has
 *  nothing to report. The timings are milliseconds over the whole run and are
 *  present whenever a device is. */
export interface GpuReport {
  backend?: 'device';
  /** Frames presented. Flush is a per-frame cost — one readback of the scanout
   *  surface — so this is what its timing has to be divided by, not draws. */
  frames?: number;
  drawn?: number;
  fallbacks?: number;
  pipelines?: number;
  modules?: number;
  /** Surfaces the backend is holding on the device. Every flush writes back
   *  all of them, so this growing is the flush time growing. */
  held?: number;
  evicted?: number;
  pending?: number;
  /** Bytes lifted out of guest memory by `Uploads::of`, by category. Textures
   *  dominate: 96.5% of them, deswizzled from block-linear on every draw. */
  read?: { textures: number; vertex: number; constants: number; index: number };
  /** Texture reads served from already-deswizzled bytes, against those that
   *  had to be read out of guest memory again. */
  textureHits?: number;
  textureMisses?: number;
  /** Once this latches, every frame after it is the rasterizer's however well
   *  the device is working. */
  softwareFrame?: boolean;
  gaveUp?: boolean;
  /** Every distinct reason a draw fell back, in the order first seen. */
  reasons?: string[];
  /** What the device itself rejected, which is not the same as a fallback:
   *  the backend only learns of a rejection when it next asks, so a frame can
   *  count as 100% device and still be wrong. `deviceErrorCount` includes the
   *  repeats; `deviceErrors` holds each distinct message once. */
  deviceErrorCount?: number;
  deviceErrors?: string[];
  /** Milliseconds over the whole run, by phase. Nested because `modules` is
   *  both a count above and a phase here. */
  times?: {
    translate: number;
    upload: number;
    modules: number;
    pipeline: number;
    encode: number;
    flush: number;
  };
}

export interface JitStats {
  enabled: boolean;
  blocks: number;
  translated: number;
  executed: number;
  invalidated: number;
}

/** What a firmware NCA is, without reading it: kind 0 is a program, 1 a data
 *  archive, 2 anything else. */
export interface NandIdentity {
  id: string;
  kind: number;
}

/** One piece of add-on content the session holds, as `switch_dlc_json`
 *  reports it: its own content id, the base title it belongs to, and the index
 *  the title knows it by. */
export interface DlcEntry {
  id: string;
  title_id: string;
  index: number;
}

/** A file inside the open PFS0 container. */
export interface NspFile {
  name: string;
  size: number;
}

export interface NcaSection {
  fs_type: string;
  offset: number;
  size: number;
}

/** `switch_parse_nca`'s JSON. `error` is set instead of the rest when the
 *  header could not be read - an encrypted CDN header with no keys loaded. */
export interface NcaInfo {
  error?: string;
  title_id: string;
  content_type: string;
  sdk_version: string;
  crypto_type: number;
  encrypted: boolean;
  file_size: number;
  sections: NcaSection[];
}

export interface AgeRating {
  organisation: string;
  age: number;
}

/** The NACP, as `switch_control_json` renders it. Every field past the name is
 *  optional in practice: most titles set only a handful. */
export interface ControlInfo {
  name: string;
  publisher?: string;
  version?: string;
  demo?: boolean;
  title_id: string;
  icon_size: number;
  icon_mime: string;
  language?: string;
  languages?: string[];
  ratings?: AgeRating[];
  startup_user_account?: string;
  screenshot?: string;
  video_capture?: string;
  user_save_size?: number;
  user_save_journal_size?: number;
  device_save_size?: number;
  device_save_journal_size?: number;
  bcat_storage_size?: number;
  add_on_content_base_id?: string;
  save_data_owner_id?: string;
  error_code_category?: string;
  isbn?: string;
}

/** Every command the worker answers, with the types the *page* sees: a
 *  handler that fails by returning `{ error }` shows up here as the value it
 *  returns on success, because the message loop turns that into a rejection. */
export interface Commands {
  'new'(): number;
  free_session(): number;

  set_trace(on: number): number;
  set_jit(on: number): number;
  set_input(mask: number, slx: number, sly: number, srx: number, sry: number): number;
  set_touch(points: Uint32Array): number;
  set_battery(percent: number, charging: number): number;
  set_operation_mode(docked: number): number;
  vibration(): number;

  load_font(bytes: Bytes): number;
  load_nro(bytes: Bytes): number;
  load_elf(bytes: Bytes): number;

  open_nsp(file: File): number;
  open_nca(file: File): number;
  add_archive(file: File): number;
  add_update(file: File): string;
  update_version(): string;
  add_dlc(file: File): number;
  dlc_json(): string;
  clear_dlc(): number;
  clear_update(): number;
  nand_identify(file: File): NandIdentity | null;
  nand_launch(bytes: Bytes): number;
  nand_add_archive(bytes: Bytes): string;
  load_nca_from_nsp(index: number): number;
  load_nca(): number;
  program_nca_index(): number;
  load_keys(prod: Bytes | null, title: Bytes | null): number;
  nsp_files_json(): string;
  read_file(index: number, offset: number, len: number): Bytes;

  load_control_from_nsp(): number;
  load_control_from_nca(): number;
  control_json(): string;
  control_icon(size: number): Bytes;
  parse_nca(header: Bytes): string;

  run(budget: number): number;
  halted(): number;
  drain_output(): Bytes;
  drain_trace(): Bytes;
  dump_regs(): string;
  get_pc(): number;
  get_cycles(): number;
  get_steps(): number;
  get_reg(i: number): string;
  ram(): RamUsage;
  jit_stats(): JitStats;
  gpu_report(): GpuReport;
  last_error(): string;

  fb_width(): number;
  fb_height(): number;
  frame_count(): number;
  fb_snapshot(len: number): Bytes | null;

  audio_format(): number;
  audio_pull(maxSamples: number): Bytes | null;

  sd_write_file(path: string, bytes: Bytes): number;
  sd_create_dir(path: string): number;
  sd_remove(path: string): number;
  sd_read_file(path: string): Bytes | null;
  sd_pending_changes(): number;
  sd_take_changes(): FsChange[];

  save_ids(): string[];
  save_pending_changes(id: string): number;
  save_take_changes(id: string): FsChange[];
  save_create(id: string): number;
  save_write_file(id: string, path: string, bytes: Bytes): number;
  save_create_dir(id: string, path: string): number;
  save_read_file(id: string, path: string): Bytes | null;
}

export type CommandName = keyof Commands;

/** What the worker implements: the same signatures, plus the option of
 *  reporting failure as `{ error }` instead of a value. */
export type CommandHandlers = {
  [K in CommandName]: (
    ...args: Parameters<Commands[K]>
  ) => ReturnType<Commands[K]> | { error: string };
};

export interface CallRequest {
  id: number;
  cmd: CommandName;
  args: unknown[];
}

export type WorkerMessage =
  | { type: 'ready'; error?: string }
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: string };
