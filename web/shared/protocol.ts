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

// A refused service command; `cmd` is null without a command id.
export interface IpcGap {
  iface: string;
  cmd: number | null;
}

// `unimplemented` was refused; `stubbed` answered with nothing behind it.
export interface IpcGaps {
  unimplemented: IpcGap[];
  stubbed: IpcGap[];
}

// Everything for a bug report about one run. `panicked` means an emulator bug;
// `session` is null if requested after the session ended.
export interface CrashReport {
  version: string;
  panicked: boolean;
  traceMask: number;
  lastError?: string;
  guestFatal?: string | null;
  title?: { id: string; name?: string; version?: string };
  cpu?: {
    pc: number;
    mode: string;
    steps: number;
    cycles: number;
    halted: boolean;
    thread: number;
    guestRam: number;
    docked: boolean;
  };
  jit?: JitStats & { interpreted?: number };
  gpu?: GpuReport;
  backtrace?: number[];
  registers?: string;
  threads?: string;
  unimplemented?: IpcGap[];
  stubbed?: IpcGap[];
  trace?: string;
  session?: null;
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

export interface Commands {
  // Quoted, since unquoted `new()` is a construct signature.
  // eslint-disable-next-line @stylistic/quote-props
  'new'(): number;
  free_session(): number;

  set_trace(on: number): number;
  set_jit(on: number): number;
  set_trace_channels(on: boolean): void;
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
  add_archive(file: Blob): number;
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
  parse_nca(header: Bytes): string;

  run(budget: number): number;
  halted(): number;
  guest_fatal(): string;
  drain_output(): Bytes;
  drain_trace(): Bytes;
  dump_regs(): string;
  thread_dump(): string;
  backtrace(depth: number): number[];
  wake_blocked(): number;
  start_created_threads(): number;
  ipc_gaps(): IpcGaps;
  crash_report(): CrashReport;
  version(): string;
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
  save_remove(id: string, path: string): number;
  save_read_file(id: string, path: string): Bytes | null;

  // Install `users` with `current` playing; 0 or `switch_users_commit`'s error code.
  users_set(users: UserRecord[], current: string): number;
  // Whether the guest edited a profile since the last call.
  users_take_edits(): boolean;
  // The users as the session holds them, including guest edits.
  users_read(): UserRecord[];

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
  ) => ReturnType<Commands[K]> | { error: string };
};

export interface CallRequest {
  id: number;
  cmd: CommandName;
  args: unknown[];
}

// Log level; DevTools mirrors `err`/`warn` as named, `ok` as info, `dim` as debug, none as log.
export type LogClass = 'err' | 'warn' | 'ok' | 'dim';

// `log` is unprompted worker output, routed through the page's log.
export type WorkerMessage =
  | { type: 'ready'; error?: string }
  | { type: 'log'; text: string; cls?: LogClass }
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: string };
