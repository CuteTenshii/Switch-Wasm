import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';

// Frontend build: `web/` is the root, `web/public` is copied verbatim.

// The dev server serves `make wasm`'s build; the site build ships `make wasm-release`'s.
const coreDir = (command: 'serve' | 'build') => fileURLToPath(
  new URL(`./target/wasm32-unknown-unknown/${command === 'serve' ? 'debug' : 'release'}`,
    import.meta.url));

// Required for `SharedArrayBuffer`; production sets the same via `web/public/_headers`.
const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig(({ command }) => ({
  root: 'web',
  // Relative: the site is published below a host path.
  base: './',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    target: 'es2022',
    // Keep every asset a file so it gets a hashed URL.
    assetsInlineLimit: 0,
  },
  // Must match `{ type: 'module' }` in `main/rpc.ts`, in dev and production alike.
  worker: { format: 'es' },
  resolve: {
    alias: {
      '@core': coreDir(command),
      // The `host_read` import named by wasm-bindgen's generated glue.
      '@host/files': fileURLToPath(
        new URL('./web/worker/hostfiles.ts', import.meta.url)),
    },
  },
  server: {
    port: 8000,
    // The core is a cargo artifact outside the project root.
    fs: { allow: ['..'] },
    headers: crossOriginIsolation,
  },
  preview: { headers: crossOriginIsolation },
}));
