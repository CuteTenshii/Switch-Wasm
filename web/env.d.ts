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

// The wasm-bindgen core, declared because it does not exist until `make wasm`.
declare module '@core/switch_wasm.js' {
  export default function init(
    options?: { module_or_path: string | WebAssembly.Module },
  ): Promise<Record<string, unknown>>;
  // Another thread's instance, on the memory the first one made.
  export function initSync(
    options: { module: WebAssembly.Module; memory: WebAssembly.Memory },
  ): Record<string, unknown>;
  // `crates/switch-wasm/src/gpu.rs`: what it renders on, or why it cannot.
  export function switch_gpu_open(
    deviceMsaa: boolean, interleave: boolean): Promise<string>;
  // `crates/switch-wasm/src/screen.rs`; the first throws when the canvas has no 2d context.
  export function switch_attach_screen(canvas: OffscreenCanvas): void;
  export function switch_present(): boolean;
  export function switch_clear_screen(): void;
  // `crates/switch-wasm/src/idb/persist.rs`: the SD card and saves, to and from IndexedDB.
  export function switch_restore_storage(): Promise<number>;
  export function switch_flush_storage(): Promise<number>;
  // `idb/nand.rs`: registers the stored firmware; the system data archives it took.
  export function switch_restore_nand(): Promise<number>;
  // `idb/controls.rs`: stores the control data the session has just read.
  export function switch_remember_control(): Promise<void>;
  // `idb/users.rs`: stores the guest's profile edits; whether there were any.
  export function switch_store_profile_edits(): Promise<boolean>;
  // The exports that carry bytes or text, in `crates/switch-wasm/src`. A failure's reason is
  // also in `switch_last_error`; the ones that return a `Result` throw it.
  export function switch_last_error(): string;
  // `container.rs`
  export function switch_update_version(): string;
  export function switch_dlc_json(): string;
  export function switch_nsp_files_json(): string;
  export function switch_read_file(
    index: number, offset: bigint, len: number): Uint8Array<ArrayBuffer>;
  export function switch_parse_nca(header: Uint8Array): string;
  export function switch_control_json(): string;
  export function switch_control_icon(): Uint8Array<ArrayBuffer>;
  export function switch_load_keys(prod: Uint8Array, title: Uint8Array): number;
  // `boot.rs` and `nand.rs`: the entry address, or -1.
  export function switch_load_program(file: Blob): Promise<number>;
  export function switch_load_sd_nro(path: string): number;
  export function switch_nand_launch(content: Blob): number;
  // `{ id, kind }` as JSON, or empty when the file is no NCA.
  export function switch_nand_identify(file: number, size: bigint): string;
  // `log.rs`, `run.rs`, `stats.rs`
  export function switch_log_line(): string;
  export function switch_guest_fatal(): string;
  export function switch_jit_stats_json(): string;
  export function switch_gpu_report_json(): string;
  export function switch_memory_json(): string;
  // `input.rs`, `keyboard.rs`, `users.rs`
  export function switch_set_touch(points: Uint32Array): void;
  export function switch_keyboard_answer(text: string | null): number;
  export function switch_user_stage(
    uid: string, nickname: string, editedAt: number, picture: Uint8Array | null): void;
  export function switch_users_commit(current: string): number;
  // `storage.rs`: the running session's SD card and saves; a save is its id and uid halves.
  export function switch_sd_write_file(path: string, data: Uint8Array): void;
  export function switch_sd_create_dir(path: string): void;
  export function switch_sd_remove(path: string): void;
  export function switch_save_write_file(
    saveId: bigint, userLo: bigint, userHi: bigint, path: string, data: Uint8Array): void;
  export function switch_save_create_dir(
    saveId: bigint, userLo: bigint, userHi: bigint, path: string): void;
  export function switch_save_remove(
    saveId: bigint, userLo: bigint, userHi: bigint, path: string): void;
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
