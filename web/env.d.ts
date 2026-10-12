// Module declarations for what the bundler adds; vite/client is not used since
// half of the frontend compiles against the WebWorker library.

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
