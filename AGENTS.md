# Switch WASM

Switch WASM is a Nintendo Switch emulator for browsers. The shipped product is
the TypeScript frontend plus the Rust core compiled to `wasm32-unknown-unknown`.

## Target

- Browser WebAssembly is the only product target. Do not turn this into a native
  emulator, desktop application, or native-first library.
- Web APIs are part of the architecture: WebGPU, Web Workers, IndexedDB,
  `SharedArrayBuffer`, browser input, and browser audio.
- Native Rust examples and tests are diagnostic harnesses. They may exercise the
  core faster or provide reference output, but native behavior alone does not
  prove that a change works in the product.
- Keep browser constraints in view: wasm32 has a 4 GiB address space, workers
  cannot block on browser promises, WebAssembly memory growth detaches cached
  views, and large game containers must be read by range instead of staged in
  memory.

## Repository map

- `crates/switch-core`: dependency-free emulator core, including the AArch64
  interpreter and JIT, loaders, services, memory, audio, and software GPU.
- `crates/switch-gpu`: optional `wgpu` renderer behind the core renderer
  interface. The software renderer is its correctness reference.
- `crates/switch-wasm`: browser bindings and session handles. Buffers cross the
  boundary through WebAssembly linear memory. Keep the existing hand-written
  JSON boundary; do not add serde merely for bindings.
- `web/main`: DOM, persistence, controls, display, audio, and worker RPC.
- `web/worker`: module worker, WebAssembly loading, host file ranges, commands,
  and the run loop.
- `web/shared/protocol.ts`: command contract checked by both worker and main
  TypeScript builds.
- `docs`: subsystem details and reproduction notes. Read the relevant document
  before changing GPU, audio, services, decoding, or debugging workflows.
- `PROGRESS.md`: implementation history, failures, and lessons that should not
  be rediscovered.

`web` is source and `dist` is generated. Files in `web/public` are copied as-is;
other assets must be imported so Vite can hash and rewrite their URLs.

## Commands

```sh
bun install
make test
make wasm
make assets
bun run typecheck
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

- `node tools/jit_wasm_check.mjs` checks that the browser build runs the blocks
  it emits, against the interpreter, after `make wasm`. Host tests cannot: the
  offsets emitted code uses are wasm32's, and only the browser build has
  anything that can compile a module.
- `make all` runs the Rust tests and builds the browser site.
- `make wasm` builds `switch-wasm` with the WebGPU feature and runs
  `wasm-bindgen`. The installed CLI version must match `Cargo.lock`.
- `make assets` builds the WebAssembly input and then the Vite site in `dist`.
- `PROFILE=quick` shortens local emulator builds. Use `release` for shipped
  artifacts and quoted performance results.
- `bun run dev` serves the source on port 8000 after the core has been built.
- `bun run preview` serves `dist`.
- Vite does not type-check. Run `bun run typecheck` explicitly.

Use the narrowest relevant check while iterating, then run checks proportional
to the changed surface. Browser-facing changes need a WebAssembly build or an
actual browser run, not just host tests.

## Browser invariants

- Keep `base: './'` in `vite.config.ts`; deployment lives below a host path.
- Keep the worker a module on both sides: the `Worker` constructor and Vite's
  worker output must agree.
- Keep cross-origin isolation headers in development, preview, and deployment.
  `SharedArrayBuffer` depends on them.
- Host file reads are synchronous because guest execution requests ranges from
  inside a run slice. They belong in the worker through `FileReaderSync`.
- Container offsets remain `u64` end to end. A cast to `usize` can truncate
  offsets above 4 GiB on wasm32.
- Never load an entire NSP, XCI, NCA, or RomFS into WebAssembly memory. Preserve
  the `ByteSource` range-reading stack and per-file caching.
- Do not retain typed-array views across operations that can grow WebAssembly
  memory. Free staging buffers after use.
- Reset must invalidate the main-thread session before the worker frees the
  handle, and an aborted run slice must not call into a freed session.
- Browser GPU work must not synchronously wait for promises. Unsupported GPU
  behavior falls back to the software renderer, and renderer changes must be
  checked against byte-identical reference output where the harness supports it.
- Anything users need to see must use the emulator diagnostic channel. Native
  stderr and environment variables are unavailable on
  `wasm32-unknown-unknown`.

## Change discipline

- Preserve the zero-dependency policy of `switch-core` unless the user
  explicitly approves a change to it.
- Keep service implementations in their existing domain modules and update
  `docs/services.md` when service coverage changes.
- For AArch64 work, verify encodings against the existing differential tools or
  an assembler rather than hand-deriving expected values. The interpreter is
  the JIT correctness reference.
- Do not optimize from host timing alone. Browser performance claims require the
  produced WebAssembly artifact and `tools/wasm_bench.mjs`; native examples are
  for diagnosis and work counts.
- Keep generated output, game images, keys, firmware, and other copyrighted or
  secret material out of the repository.
- Do not edit unrelated user changes. `dist` and Cargo target output are build
  products, not source patches.
