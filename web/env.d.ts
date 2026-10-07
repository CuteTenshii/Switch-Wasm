// Module declarations for what the bundler adds; vite/client is not used since
// half of the frontend compiles against the WebWorker library.

declare module '*?url' {
  const url: string;
  export default url;
}

// The wasm-bindgen core, declared because it does not exist until `make wasm`.
declare module '@core/switch_wasm.js' {
  export default function init(
    options?: { module_or_path: string },
  ): Promise<Record<string, unknown>>;
}

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
