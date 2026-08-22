import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';

/* The frontend build.

   `web/` is the root, so `web/index.html` is the entry and everything the page
   pulls in - the stylesheet, the icon, the worker, the font, the emulator core
   - is discovered from there and emitted into `dist/assets` under a
   content-hashed name. That is the point of building it this way: the .wasm and
   the bundles are the files a browser most eagerly caches, and a hash in the
   name is what makes a new build a new URL.

   The exception is `web/public`, which is copied verbatim: it holds the social
   card, whose URL is baked into other people's caches by the meta tags, and the
   font's licence, which has to stay readable next to the font it covers. */

const coreDir = fileURLToPath(
  new URL('./target/wasm32-unknown-unknown/release', import.meta.url));

export default defineConfig({
  root: 'web',
  // Relative, because the site is published under a path
  // (tenshii.moe/Switch-Wasm/) rather than at a host's root. The default of '/'
  // emits /assets/... , which is a 404 anywhere but the root - and one that
  // only shows up once deployed.
  base: './',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    // The emulator needs a browser with WebAssembly and workers; nothing older
    // than this can run it anyway, so there is no reason to down-level.
    target: 'es2022',
    // Every asset stays a file. An inlined data: URI would defeat the hashing
    // above, and `switch_wasm.wasm` is fetched by URL rather than imported as
    // bytes, so it has to be one.
    assetsInlineLimit: 0,
  },
  // A module worker, which the `new Worker` call in `main/rpc.ts` must match
  // with `{ type: 'module' }`. The pair is not optional and not conditional:
  //   - 'iife' is not an escape. The dev server serves the worker entry
  //     unbundled whatever this says, so a classic worker gets a file full of
  //     `import` and dies with "Cannot use import statement outside a module".
  //   - branching on `import.meta.env.DEV` to drop the `type` in production is
  //     worse: the built chunk is still an ES module, and it only loads as a
  //     classic script for as long as bundling happens to leave no `import` in
  //     it. One more chunk and production breaks while dev stays green.
  worker: { format: 'es' },
  resolve: {
    // cargo's output, imported as an asset. The path lives here rather than in
    // the worker so that the profile and target triple are named once, beside
    // the Makefile variables that build it.
    alias: { '@core': coreDir },
  },
  server: {
    port: 8000,
    // The core is outside the project root (it is a cargo artifact), so the
    // dev server has to be allowed to serve it.
    fs: { allow: ['..'] },
  },
});
