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
  // The scalar exports. Each `u64` is a bigint, each `u32` and `i32` a number, each `bool` a boolean.
  export function switch_init(): void;
  export function switch_add_core(): number;
  // Blocks the calling core worker until the session stops the core.
  export function switch_serve_core(slot: number): void;
  export function switch_end_title(): void;
  export function switch_open_nca(size: bigint): number;
  export function switch_clear_dlc(): void;
  export function switch_clear_update(): void;
  export function switch_set_trace(enabled: boolean): void;
  export function switch_set_jit(enabled: boolean): void;
  export function switch_set_trace_channels(on: boolean): void;
  export function switch_gpu_lost(): boolean;
  export function switch_log_activity(seconds: number): void;
  export function switch_set_input(
    buttons: number, stickLx: number, stickLy: number, stickRx: number, stickRy: number): void;
  export function switch_set_battery(percent: number, charging: boolean): void;
  export function switch_set_operation_mode(docked: boolean): void;
  export function switch_vibration(): number;
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
  // `crates/switch-wasm/src/idb/persist.rs`: the SD card and saves, to and from IndexedDB.
  export function switch_restore_storage(): Promise<number>;
  export function switch_flush_storage(): Promise<number>;
  // `idb/nand.rs`: registers the stored firmware; the system data archives it took.
  export function switch_restore_nand(): Promise<number>;
  // `idb/controls.rs`: stores the control data the session has just read.
  export function switch_remember_control(): Promise<void>;
  // `idb/users.rs`: stores the guest's profile edits; whether there were any.
  export function switch_store_profile_edits(): Promise<boolean>;
  // The exports that carry bytes or text, in `crates/switch-wasm/src`. The ones that return a
  // `Result` throw its error.
  export function switch_last_error(): string;
  // The panic that trapped the last call, which is then forgotten.
  export function switch_panic_message(): string;
  // `container.rs`
  // `{ title, version }` as JSON, or empty when the file is no update.
  export function switch_add_update(file: number, size: bigint): string;
  // The pieces as JSON: `[{ id, title_id, index }]`, empty when the file is no add-on content.
  export function switch_add_dlc(file: number, size: bigint): string;
  // The file table as JSON; throws the reason when the container cannot be read.
  export function switch_open_nsp(size: bigint): string;
  export function switch_read_file(
    index: number, offset: bigint, len: number): Uint8Array<ArrayBuffer>;
  export function switch_parse_nca(header: Uint8Array): string;
  export function switch_control_json(): string;
  export function switch_control_icon(): Uint8Array<ArrayBuffer>;
  export function switch_load_keys(prod: Uint8Array | null, title: Uint8Array | null): void;
  export function switch_load_control(): void;
  // `boot.rs` and `nand.rs`: the entry address; they throw the reason when a program will not boot.
  export function switch_load_program(file: Blob): Promise<number>;
  export function switch_load_sd_nro(path: string): number;
  export function switch_load_nca(): number;
  export function switch_load_nca_from_nsp(index: number): number;
  export function switch_program_nca_index(): number;
  export function switch_nand_launch(content: Blob): number;
  // `{ id, kind }` as JSON, or empty when the file is no NCA.
  export function switch_nand_identify(file: number, size: bigint): string;
  // `log.rs`, `run.rs`, `stats.rs`
  // Every queued line, each as its level digit (1 error to 5 debug) then its text.
  export function switch_log_drain(): string[];
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
  // `storage.rs`: the running session's files. `storage` is `sd`, or a save's id,
  // `<16 hex digits>` or `<id>@<32 hex digits of the owner's uid>`.
  export function switch_fs_write(storage: string, path: string, data: Uint8Array): void;
  export function switch_fs_create_dir(storage: string, path: string): void;
  export function switch_fs_remove(storage: string, path: string): void;
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
