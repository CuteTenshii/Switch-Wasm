// The page/worker command contract; both TypeScript builds check against it.

// Byte payloads own a plain ArrayBuffer, so they can go to ImageData, Blob or a transfer.
export type Bytes = Uint8Array<ArrayBuffer>;

// One path the guest touched, from `switch_sd_take_changes_json`.
export interface FsChange {
  kind: 'file' | 'dir' | 'deleted';
  path: string;
}

// `wasm` is the worker's linear memory size.
export interface RamUsage {
  guest: number;
  wasm: number;
}

// A part of the title's address space; `key` names it for the page's labels.
export type MemoryRegionKey = 'code' | 'stacks' | 'runtime' | 'loaded' | 'heap' | 'alias';

// The title's backed guest memory by region, and the WebAssembly heap it lives in.
export interface MemoryReport {
  backed: number;
  // Past it, guest writes fault.
  cap: number;
  regions: { key: MemoryRegionKey; start: number; end: number; bytes: number }[];
  // Backed outside every region: display buffers, the homebrew environment.
  elsewhere: number;
  // Every process's, library applets included.
  allProcesses: number;
  wasmHeap: number;
}

// GPU backend counters; `backend` is absent while the software rasterizer has the frame.
// Timings are milliseconds over the whole run.
export interface GpuReport {
  backend?: 'device';
  // Frames presented; flush is a per-frame cost.
  frames?: number;
  drawn?: number;
  fallbacks?: number;
  pipelines?: number;
  modules?: number;
  // Surfaces held on the device; every flush writes them all back.
  held?: number;
  evicted?: number;
  pending?: number;
  // Bytes lifted from guest memory by `Uploads::of`, by category.
  read?: { textures: number; vertex: number; constants: number; index: number };
  // Deswizzled texture bytes kept in the WebAssembly heap, and texture bytes on the device.
  textureCacheBytes?: number;
  deviceTextureBytes?: number;
  // Texture reads served from cached deswizzled bytes vs re-read from guest memory.
  textureHits?: number;
  textureMisses?: number;
  // The rasterizer holds frames after a fallback until enough in a row could run on the device.
  softwareFrame?: boolean;
  // Times the software-frame latch let go.
  unlatched?: number;
  gaveUp?: boolean;
  lostBecause?: string | null;
  // Distinct fallback reasons, in first-seen order.
  reasons?: string[];
  // Device-side errors, learned late, so a frame can be 100% device and still wrong.
  // `deviceErrorCount` includes repeats.
  deviceErrorCount?: number;
  deviceErrors?: string[];
  // Milliseconds per phase.
  times?: {
    translate: number;
    upload: number;
    modules: number;
    bind: number;
    pipeline: number;
    encode: number;
    flush: number;
    // `flush` phases: `flushAsk` encodes copies, `flushWait` waits for maps (~0 in a
    // browser), `flushLand` writes into guest memory.
    flushAsk?: number;
    flushWait?: number;
    flushLand?: number;
  };
}

export interface JitStats {
  enabled: boolean;
  blocks: number;
  translated: number;
  executed: number;
  linked: number;
  invalidated: number;
  // Blocks compiled to wasm; zero when the build can't emit code.
  emitted?: number;
  // Block entries that ran compiled code.
  enteredEmitted?: number;
  // Of those, entries a compiled block made by jumping into the next.
  chained?: number;
}

// Firmware NCA kind: 0 program, 1 data archive, 2 other.
export interface NandIdentity {
  id: string;
  kind: number;
}

// Add-on content from `switch_dlc_json`.
export interface DlcEntry {
  id: string;
  title_id: string;
  index: number;
}

export interface NspFile {
  name: string;
  size: number;
}

export interface NcaSection {
  fs_type: string;
  offset: number;
  size: number;
}

// `switch_parse_nca`'s JSON; `error` replaces the rest if the header can't be read.
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

// The NACP from `switch_control_json`; fields past the name are often absent.
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

export interface UserRecord {
  // The `AccountUid`: 32 hex digits in memory byte order.
  uid: string;
  nickname: string;
  // Last edit time in POSIX seconds; 0 for never.
  editedAt: number;
  // A baseline JPEG, or null for the core's default picture.
  picture: Bytes | null;
}

// What the guest's software keyboard shows and accepts; lengths are UTF-16 units.
export interface KeyboardRequest {
  header: string;
  sub: string;
  guide: string;
  // The submit button's label; empty for the default.
  ok: string;
  maxLength: number;
  minLength: number;
  password: boolean;
}

// What the title asked the error applet to show; `code` is as the console prints it.
export interface ErrorDialog {
  code: string | null;
  message: string;
}

export interface Commands {
  // Quoted, since unquoted `new()` is a construct signature.
  // eslint-disable-next-line @stylistic/quote-props
  'new'(): number;
  // End the running title and its applets; the console stays.
  end_title(): number;

  set_trace(on: number): number;
  set_jit(on: number): number;
  set_trace_channels(on: boolean): void;
  set_input(mask: number, slx: number, sly: number, srx: number, sry: number): number;
  set_touch(points: Uint32Array): number;
  set_battery(percent: number, charging: number): number;
  set_operation_mode(docked: number): number;
  vibration(): number;

  load_font(bytes: Bytes): number;
  // `sdPath`: where on the SD card it came from, told to it as argv[0].
  load_nro(bytes: Bytes, sdPath?: string): number;
  load_elf(bytes: Bytes): number;

  open_nsp(file: File): number;
  open_nca(file: File): number;
  add_archive(file: Blob): number;
  add_applet(file: Blob): number;
  add_update(file: File): string;
  update_version(): string;
  add_dlc(file: File): number;
  dlc_json(): string;
  clear_dlc(): number;
  clear_update(): number;
  nand_identify(file: File): NandIdentity | null;
  nand_launch(bytes: Bytes): number;
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
  control_nacp(): Bytes;
  // Record a title's NACP and icon for ns's GetApplicationControlData.
  add_application_control(titleId: string, nacp: Bytes, icon: Bytes): number;
  parse_nca(header: Bytes): string;

  run(budget: number): number;
  // The worker runs slices itself until paused, halted or faulted, reporting each.
  start_loop(tracing: boolean): number;
  pause_loop(): number;
  // The SD card and saves: loaded from IndexedDB into the core, and written back.
  restore_storage(): number;
  flush_storage(): number;
  clear_screen(): number;
  halted(): number;
  guest_fatal(): string;
  // Passes the core's queued lines on now; how many there were.
  flush_log(): number;
  log_regs(): number;
  log_threads(): number;
  get_steps(): number;
  get_reg(i: number): string;
  ram(): RamUsage;
  memory(): MemoryReport;
  jit_stats(): JitStats;
  gpu_report(): GpuReport;
  last_error(): string;

  fb_width(): number;
  fb_height(): number;
  frame_count(): number;
  fb_snapshot(len: number): Bytes | null;

  sd_write_file(path: string, bytes: Bytes): number;
  sd_create_dir(path: string): number;
  sd_remove(path: string): number;
  sd_read_file(path: string): Bytes | null;
  sd_take_changes(): FsChange[];

  save_ids(): string[];
  saves_pending(): boolean;
  save_take_changes(id: string): FsChange[];
  save_write_file(id: string, path: string, bytes: Bytes): number;
  save_create_dir(id: string, path: string): number;
  save_remove(id: string, path: string): number;
  save_read_file(id: string, path: string): Bytes | null;

  // Install `users` with `current` playing; 0 or `switch_users_commit`'s error code.
  users_set(users: UserRecord[], current: string): number;
  // Whether the guest edited a profile since the last call.
  users_take_edits(): boolean;
  // The users as the session holds them, including guest edits.
  users_read(): UserRecord[];

  // The next error the title asked to show, or null.
  error_dialog(): ErrorDialog | null;
  // The keyboard waiting for text, or null.
  keyboard_request(): KeyboardRequest | null;
  // Send the typed text, or null to cancel; 1 if a keyboard was waiting.
  keyboard_answer(text: string | null): number;
}

export type CommandName = keyof Commands;

// The worker's implementation: same signatures, or `{ error }` on failure.
export type CommandHandlers = {
  [K in CommandName]: (
    ...args: Parameters<Commands[K]>
  ) => ReturnType<Commands[K]> | Promise<ReturnType<Commands[K]>> | { error: string };
};

export interface CallRequest {
  id: number;
  cmd: CommandName;
  args: unknown[];
}

// A log line's severity; the console shows those at or above its chosen level.
export type LogLevel = 'error' | 'warn' | 'info' | 'log' | 'debug';

// What the worker gathers every few slices, so the page polls for nothing.
export interface Housekeeping {
  steps: number;
  ram: { guest: number; wasm: number };
  // Changed since the last report; the page reads the profiles itself.
  profilesEdited: boolean;
  keyboard: KeyboardRequest | null;
  errors: ErrorDialog[];
}

// One slice of the worker's run loop.
export interface SliceReport {
  // Retired this slice; fewer than `budget` means the guest halted, negative a fault.
  steps: number;
  budget: number;
  ms: number;
  // The loop has ended: paused, halted or faulted.
  stopped: boolean;
  frame: number;
  // The frame's size, and how long painting it took if this slice painted.
  width: number;
  height: number;
  paintMs: number | null;
  // `(channels << 24) | rate` of the audio in the ring; 0 before a device opens.
  audioFormat: number;
  housekeeping: Housekeeping | null;
}

// `log` is unprompted worker output, routed through the page's log; `slice` and
// `loop_failed` come from the worker's run loop.
export type WorkerMessage =
  | { type: 'ready'; error?: string }
  | { type: 'log'; text: string; level: LogLevel }
  | { type: 'slice'; report: SliceReport }
  | { type: 'loop_failed'; error: string }
  // Once at startup: the ring audio is written to, in the worker's shared memory.
  | { type: 'audio_ring'; buffer: SharedArrayBuffer; at: number; samples: number }
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: string };
