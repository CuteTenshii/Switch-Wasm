// Module declarations for what the bundler adds; vite/client is not used since
// half of the frontend compiles against the WebWorker library.

declare module '*?worker&url' {
  const url: string;
  export default url;
}

declare module '*?url' {
  const url: string;
  export default url;
}

// The File System Access API's folder picker, permissions and listing, which Chromium has and
// TypeScript's DOM library leaves out.
interface FileSystemHandlePermissionDescriptor {
  mode?: 'read' | 'readwrite';
}
interface FileSystemHandle {
  queryPermission(descriptor?: FileSystemHandlePermissionDescriptor): Promise<PermissionState>;
  requestPermission(descriptor?: FileSystemHandlePermissionDescriptor): Promise<PermissionState>;
}
interface FileSystemDirectoryHandle {
  values(): AsyncIterableIterator<FileSystemFileHandle | FileSystemDirectoryHandle>;
}
interface Window {
  showDirectoryPicker?(options?: {
    id?: string;
    mode?: 'read' | 'readwrite';
  }): Promise<FileSystemDirectoryHandle>;
}

// The wasm-bindgen core, declared because it does not exist until `make wasm`.
declare module '@core/switch_wasm.js' {
  export default function init(
    options?: { module_or_path: string | WebAssembly.Module },
  ): Promise<Record<string, unknown>>;
  // Another thread's instance, on the memory the first one made.
  export function initSync(
    options: { module: WebAssembly.Module; memory: WebAssembly.Memory },
  ): Record<string, unknown>;
  // `crates/wasm/src/gpu.rs`: what it renders on, or why it cannot.
  export function switch_gpu_open(
    deviceMsaa: boolean, interleave: boolean): Promise<string>;
  // `crates/wasm/src/screen.rs`; the first throws when the canvas has no 2d context.
  export function switch_attach_screen(canvas: OffscreenCanvas): void;
  export function switch_present(): boolean;
  export function switch_clear_screen(): void;
  // The scalar exports. Each `u64` is a bigint, each `u32` and `i32` a number, each `bool` a boolean.
  export function switch_init(): void;
  export function switch_add_core(): number;
  // `hostfiles.rs`: the worker (`worker/reader.ts`) that fills the host file cache; throws without `SharedArrayBuffer`.
  export function switch_set_reader(worker: Worker): void;
  // Blocks the calling core worker until the session stops the core.
  export function switch_serve_core(slot: number): void;
  export function switch_end_title(): void;
  export function switch_open_nca(file: Blob, name: string): number;
  export function switch_set_trace(enabled: boolean): void;
  export function switch_set_jit(enabled: boolean): void;
  export function switch_set_trace_channels(on: boolean): void;
  export function switch_gpu_lost(): boolean;
  export function switch_log_activity(seconds: number): void;
  export function switch_set_battery(percent: number, charging: boolean): void;
  export function switch_set_operation_mode(docked: boolean): void;
  export function switch_motion_attach(): Promise<string>;
  export function switch_motion_detach(): Promise<void>;
  export function switch_force_software(on: boolean): void;
  export function switch_set_device_timing(mode: string): void;
  // Throws when `url` is not a WebSocket address.
  export function switch_net_connect(url: string): void;
  export function switch_net_disconnect(): void;
  export function switch_net_connections_json(): string;
  export function switch_net_status_json(): string;
  // Throws when no relay that has settings is connected.
  export function switch_net_set_settings(
    port: number, allowPrivate: boolean, allowInbound: boolean, origins: string[],
  ): void;
  export function switch_run(maxSteps: number): number;
  export function switch_pace(elapsed: number): number;
  export function switch_halted(): boolean;
  export function switch_reg_dump(): string;
  export function switch_thread_dump(): string;
  export function switch_get_reg(idx: number): bigint;
  export function switch_get_steps(): number;
  export function switch_guest_ram(): number;
  // `[width, height]`.
  export function switch_fb_size(): Uint32Array;
  export function switch_frame_count(): number;
  // `crates/wasm/src/idb/persist.rs`: the SD card and saves, to and from IndexedDB.
  export function switch_restore_storage(): Promise<number>;
  export function switch_flush_storage(): Promise<number>;
  // `idb/nand.rs`: registers the stored firmware; the system data archives it took.
  export function switch_restore_nand(): Promise<number>;
  // `idb/content.rs`: NAND content in OPFS, or IndexedDB without it. Content read from stays on
  // disk until the next start, even once removed.
  export function switch_nand_content(name: string): Promise<Blob | undefined>;
  export function switch_nand_content_size(name: string): Promise<number | undefined>;
  export function switch_nand_install(
    content: Blob, name: string, id: string, kind: number): Promise<void>;
  export function switch_nand_remove(id: string): Promise<void>;
  export function switch_nand_erase(): Promise<void>;
  // `idb/users.rs`: stores the guest's profile edits; whether there were any.
  export function switch_store_profile_edits(): Promise<boolean>;
  // The exports that carry bytes or text, in `crates/wasm/src`. The ones that return a
  // `Result` throw its error.
  export function switch_last_error(): string;
  // The panic that trapped the last call, which is then forgotten.
  export function switch_panic_message(): string;
  // `container.rs`
  // An `OpenedContainer` as JSON; throws the reason when the file is no container.
  export function switch_open_container(file: Blob, name: string): string;
  // An `NcaInfo` as JSON.
  export function switch_inspect_nca(index?: number | null): string;
  export function switch_control_json(): string;
  export function switch_control_icon(): Uint8Array<ArrayBuffer>;
  // A `PeekedTitle` as JSON; throws the reason when the file holds no title.
  export function switch_peek_title(file: Blob): string;
  // The icon of the file last peeked, or empty; taken, so asked once.
  export function switch_peek_icon(): Uint8Array<ArrayBuffer>;
  export function switch_load_keys(prod: Uint8Array | null, title: Uint8Array | null): void;
  export function switch_load_control(): Promise<void>;
  // `boot.rs` and `nand.rs`: the entry address; they throw the reason when a program will not boot.
  export function switch_load_program(file: Blob): Promise<number>;
  export function switch_load_sd(path: string): number;
  export function switch_next_load(): string;
  export function switch_launch(index?: number | null): number;
  export function switch_nand_launch(content: Blob): number;
  // `{ id, kind }` as JSON, or empty when the file is no NCA.
  export function switch_nand_identify(content: Blob): string;
  // `log.rs`, `run.rs`, `stats.rs`
  // Every queued line, each as its level digit (1 error to 5 debug) then its text.
  export function switch_log_drain(): string[];
  export function switch_guest_fatal(): string;
  export function switch_jit_stats_json(): string;
  export function switch_gpu_report_json(): string;
  export function switch_memory_json(): string;
  export function switch_audio_json(): string;
  export function switch_threads_json(): string;
  // `input.rs`, `keyboard.rs`, `users.rs`
  export function switch_set_touch(points: Uint32Array): void;
  export function switch_keyboard_answer(text: string | null): number;
  export function switch_user_stage(
    uid: string, nickname: string, editedAt: number, picture: Uint8Array | null): void;
  export function switch_users_commit(current: string): number;
  // `storage.rs`: the session's files, which the next flush stores. `storage` is `sd`, or a
  // save's id, `<16 hex digits>` or `<id>@<32 hex digits of the owner's uid>`.
  // Whether this tab stores anything: another tab holding the storage makes it read-only.
  export function switch_storage_writable(): Promise<boolean>;
  export function switch_fs_write(storage: string, path: string, data: Uint8Array): void;
  export function switch_fs_create_dir(storage: string, path: string): void;
  export function switch_fs_remove(storage: string, path: string): void;
  // `[path, kind, size]`, kind 0 for a directory and 1 for a file.
  export function switch_fs_entries(storage: string): [string, number, number][];
  // A file's bytes; undefined for a directory or a missing path.
  export function switch_fs_read(storage: string, path: string): Uint8Array<ArrayBuffer> | undefined;
  // The saves holding anything.
  export function switch_saves(): string[];
  // `opfs/volumes.rs`: copied into OPFS by range where it holds the card.
  export function switch_fs_upload(storage: string, path: string, file: Blob): Promise<void>;
}

// The commit the site was built from; see `vite.config.ts`.
declare const __BUILD__: string;

interface ImportMetaEnv {
  readonly MODE: string;
  readonly BASE_URL: string;
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly SSR: boolean;
  readonly [key: string]: string | boolean | undefined;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
