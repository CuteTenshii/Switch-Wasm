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
    handle: number, deviceMsaa: boolean, interleave: boolean): Promise<string>;
  // `crates/switch-wasm/src/screen.rs`; the first throws when the canvas has no 2d context.
  export function switch_attach_screen(canvas: OffscreenCanvas): void;
  export function switch_present(handle: number): boolean;
  export function switch_clear_screen(): void;
  // `crates/switch-wasm/src/idb/persist.rs`: the SD card and saves, to and from IndexedDB.
  export function switch_restore_storage(handle: number): Promise<number>;
  export function switch_flush_storage(handle: number): Promise<number>;
  // `idb/nand.rs`: registers the stored firmware; the system data archives it took.
  export function switch_restore_nand(handle: number): Promise<number>;
  // `idb/controls.rs`: stores the control data the session has just read.
  export function switch_remember_control(handle: number): Promise<void>;
  // `idb/users.rs`: stores the guest's profile edits; whether there were any.
  export function switch_store_profile_edits(handle: number): Promise<boolean>;
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
