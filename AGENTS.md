# AGENTS.md

Browser-oriented Nintendo Switch emulation core: an ARM64 (A64) integer interpreter plus PFS0/NSP, NCA, NRO and ELF parsers, compiled to WASM for the frontend.

## Commands

- `make all` — test + wasm + assets (the full pipeline).
- `make test` — `cargo test -p switch-core`. This is the only crate with tests (unit tests in each module + `crates/switch-core/tests/cpu_test.rs`).
- `make wasm` — `cargo build --target wasm32-unknown-unknown --release -p switch-wasm`.
- `make assets` — wasm, then **copies the built `.wasm` into `web/assets/`**.
- `make serve` — `python3 tools/serve.py` (after assets). Frontend is plain static JS; no bundler.
- Single test: `cargo test -p switch-core <test_name>`.
- `rustup target add wasm32-unknown-unknown` is required for the wasm build.

## Build pipeline traps

- The browser fetches `web/assets/switch_wasm.wasm` and bundled `.nro` files as committed copies. Building `switch-wasm` alone does **not** refresh them — run `make assets`.
- `make test` only covers `switch-core`; there are no tests for `switch-wasm`.

## Architecture

- `crates/switch-core` — the emulation core, zero external deps. Modules: `cpu` (A64 interpreter), `mem` (sparse 4 GiB page table), `disasm`, `nsp`/`nca`/`nro`/`elf` (parsers), `error`.
- `crates/switch-wasm` — `cdylib` for `wasm32-unknown-unknown`, exports a raw `extern "C"` ABI (no wasm-bindgen). Buffers cross the boundary via wasm linear memory (`switch_alloc`/`switch_free`); a handle is an index into a global session table. **No external deps by design** — JSON results are emitted by the hand-rolled `json_escape`/`write_into` helpers in `crates/switch-wasm/src/lib.rs`. Don't add serde/etc.
- `tools/serve.py` — local static server with no-cache headers.
- `web/main.js` — glue over the wasm ABI; also the source of truth for how the ABI is used.
- Emulated CPU addresses are `u32` (32-bit PC). Memory is a sparse page table (`PAGE_COUNT` computed in `u64` to survive wasm32 `usize` truncation); reads/writes to unmapped addresses fault with `Error::Cpu`.
- The demo framebuffer is memory-mapped at `FB_BASE = 0x3F00_0000` (640x360 RGBA) and input at `INPUT_ADDR = 0x3F10_0000`. Syscall ABI: `SyscallMode::Horizon` stubs for real homebrew. Commercial encrypted content is out of scope.

## Gotchas

- `cargo clippy` on the whole workspace **fails** in `switch-wasm` (deliberate raw-pointer `extern "C"` signatures trigger `not_unsafe_ptr_arg_deref`). `clippy`/`fmt` are not gating; `cargo fmt --check` is also not clean.
- CPU test encodings in `tests/cpu_test.rs` are hand-assembled and cross-checked against QEMU's `a64.decode`.
