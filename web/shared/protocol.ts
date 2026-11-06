// The page/worker command contract; both TypeScript builds check against it.

import type { NandEntry, StoredEntry } from './db';

// Byte payloads own a plain ArrayBuffer, so they can go to ImageData, Blob or a transfer.
export type Bytes = Uint8Array<ArrayBuffer>;

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

// An update container the core accepted: the base title it patches and its display version.
export interface UpdateInfo {
  title: string;
  version: string;
}

// Add-on content from `switch_dlc_json`.
export interface DlcEntry {
  id: string;
  title_id: string;
  index: number;
}

// An installed NAND title with the size of its content.
export interface NandTitle {
  id: string;
  name: string;
  kind: number;
  size: number;
}

// Where a stored path lives: the SD card, or one save's files.
export type StorageVolume = { kind: 'sd' } | { kind: 'save'; id: string };

export interface StorageInfo {
  kind: 'dir' | 'file';
  size: number;
}

// `ids` are the stored saves (`<hex save id>[@<user uid>]`); `names` map a uid to its nickname.
export interface SaveListing {
  ids: string[];
  names: [string, string][];
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
  // End the running title and its applets; the console stays.
  endTitle(): void;

  setTrace(on: boolean): void;
  setJit(on: boolean): void;
  setTraceChannels(on: boolean): void;
  setInput(mask: number, slx: number, sly: number, srx: number, sry: number): void;
  setTouch(points: Uint32Array): void;
  setBattery(percent: number, charging: boolean): void;
  setOperationMode(docked: boolean): void;
  vibration(): number;

  // An ELF or an NRO, told apart by its magic; the worker reads the file itself.
  loadProgram(file: File): number;
  // The NRO at `path` on the SD card, which is told to it as argv[0].
  loadSdNro(path: string): number;

  // The container's file table; throws the reason if it cannot be read.
  openNsp(file: File): NspFile[];
  openNca(file: File): number;
  restoreNand(): number;
  addUpdate(file: File): UpdateInfo | null;
  // The pieces the container holds; none if it is not add-on content.
  addDlc(file: File): DlcEntry[];
  clearDlc(): void;
  clearUpdate(): void;
  nandIdentify(file: File): NandIdentity | null;
  // The stored content named `name`; it is read here, not by the page.
  nandLaunch(name: string): number;
  loadNcaFromNsp(index: number): number;
  loadNca(): number;
  programNcaIndex(): number;
  loadKeys(prod: Bytes | null, title: Bytes | null): number;
  readFile(index: number, offset: number, len: number): Bytes;

  // The title's NACP and icon, from the container's Control NCA or a standalone one.
  // Throws the reason when there is none.
  loadControl(): void;
  controlJson(): string;
  controlIcon(): Bytes;
  // Store the control data just read, so a later session's ns can describe the title.
  rememberControl(): void;
  parseNca(header: Bytes): string;

  run(budget: number): number;
  // The worker runs slices itself until paused, halted or faulted, reporting each.
  startLoop(tracing: boolean): void;
  pauseLoop(): void;
  // The SD card and saves: the worker loads them from IndexedDB and writes changes back itself.
  restoreStorage(): number;
  flushStorage(): number;
  clearScreen(): void;
  halted(): boolean;
  guestFatal(): string;
  // Passes the core's queued lines on now; how many there were.
  flushLog(): number;
  regDump(): string;
  threadDump(): string;
  getSteps(): number;
  getReg(i: number): string;
  ram(): RamUsage;
  memory(): MemoryReport;
  jitStats(): JitStats;
  gpuReport(): GpuReport;
  lastError(): string;

  // The stored SD card and saves. Changes go to IndexedDB first, then to the running session.
  storageScan(volume: StorageVolume): Map<string, StorageInfo>;
  storageRead(volume: StorageVolume, path: string): StoredEntry | undefined;
  // A null entry deletes the path.
  storageApply(volume: StorageVolume, changes: [string, StoredEntry | null][]): void;
  // Store `file` at `path`; its size.
  storageUpload(volume: StorageVolume, path: string, file: File): number;
  // Delete `path` and everything under it; `/` empties the volume.
  storageRemove(volume: StorageVolume, path: string): void;
  storageRename(volume: StorageVolume, from: string, to: string): void;
  // A store-only zip of `root` and its entry count.
  storageExport(volume: StorageVolume, root: string): { blob: Blob; entries: number };
  savesList(): SaveListing;
  // Delete the saves a profile owns; how many entries went.
  savesDeleteUser(uid: string): number;

  // Profile records by uid, plus the playing one under its own key.
  usersRead(): [string, unknown][];
  // A null value deletes the key.
  usersStore(entries: [string, unknown][]): void;

  // The installed titles' index, and the same with content sizes.
  nandTitles(): [string, NandEntry][];
  nandList(): NandTitle[];
  nandContent(name: string): Blob | undefined;
  // Store `file` under its name and index it as title `id`.
  nandInstall(file: File, id: string, kind: number): void;
  nandRemove(id: string): void;
  nandErase(): void;

  // Install `users` with `current` playing; 0 or `switch_users_commit`'s error code.
  usersSet(users: UserRecord[], current: string): number;
  // Stores any profile edits the guest made; whether there were any.
  storeProfileEdits(): boolean;

  // Send the typed text, or null to cancel; 1 if a keyboard was waiting.
  keyboardAnswer(text: string | null): number;
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
export type LogLevel = 'error' | 'warn' | 'info' | 'log' | 'debug' | 'trace';

// What the worker gathers every few slices, so the page polls for nothing.
export interface Housekeeping {
  steps: number;
  ram: { guest: number; wasm: number };
}

// One slice of the worker's run loop.
export interface SliceReport {
  // The guest ended or failed; a slice also stops early for a pause.
  ended: boolean;
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
  housekeeping: Housekeeping | null;
}

// `log` is unprompted worker output, routed through the page's log; `slice` and
// `loop_failed` come from the worker's run loop.
export type WorkerMessage =
  | { type: 'ready'; error?: string }
  | { type: 'log'; text: string; level: LogLevel }
  | { type: 'slice'; report: SliceReport }
  | { type: 'loop_failed'; error: string }
  // From `switch_restore_nand`: system data archives registered so far, of all stored.
  | { type: 'nand_progress'; done: number; total: number }
  // Once at startup: the ring audio is written to, in the worker's shared memory.
  | { type: 'audio_ring'; buffer: SharedArrayBuffer; at: number; samples: number }
  // `(channels << 24) | rate` of the audio in the ring, sent when it changes; 0 before a device opens.
  | { type: 'audio_format'; format: number }
  // From `crates/switch-wasm/src/keyboard.rs`: the guest raised an error applet, or waits for text.
  | ({ type: 'error_dialog' } & ErrorDialog)
  | ({ type: 'keyboard' } & KeyboardRequest)
  // The guest's profile edits are stored; the page reloads them.
  | { type: 'profiles_edited' }
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: string };
