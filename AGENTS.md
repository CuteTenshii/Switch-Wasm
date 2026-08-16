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
- `web/worker.js` — hosts the wasm module (`WebAssembly.instantiateStreaming` of `assets/switch_wasm.wasm`) and is the glue over the ABI. `web/main.js` is promise-based RPC over `postMessage`. Step budgets: non-trace `Number.MAX_SAFE_INTEGER`, trace mode 5000.
- Emulated CPU addresses are `u32` (32-bit PC). Memory is a sparse page table (`PAGE_COUNT` computed in `u64` to survive wasm32 `usize` truncation); reads/writes to unmapped addresses fault with `Error::Cpu`.
- The demo framebuffer is memory-mapped at `FB_BASE = 0x3F00_0000` (640x360 RGBA) and input at `INPUT_ADDR = 0x3F10_0000`. Syscall ABI: `SyscallMode::Horizon` stubs for real homebrew. Commercial encrypted content is out of scope.

## Homebrew memory layout (cpu.rs constants)

- Stack: `STACK_TOP = 0x1010_0000`, size 1 MiB (so `STACK_BASE = 0x1000_0000`), SP seeded at `STACK_TOP`.
- TLS base: `tpidr = 0x0FF0_0000` (deliberately clear of the heap). libnx ThreadVars at TLS+0x1E0 (magic `0x21545624` "!TV$"), `_REENT` zeroed at `0x1FF1_0000`.
- Heap via `svcSetHeapSize` returns `0x3000_0000` (address out-param; deliberately NOT `0x2000_0000` — with a 512 MiB heap there, `malloc`'s arena lands at `STACK_BASE + 0x1a80` and the app's 8 MiB memblock memset overwrites the stack). `svcQueryMemory` reports per-page state (allocated pages = RWX, untouched soft pages = unmapped) so libnx virtmem reservations find free address space. `svcGetInfo` uses the hbmenu libnx InfoType numbering (2/3 Alias, 4/5 Heap, 6/7 Total/Used memory, 12/13 Aslr, 14/15 Stack). Env block at `ENV_BLOCK_ADDR = 0x0010_0000`.
- `boot_homebrew` (cpu.rs): runs crt0 through the `bl main` at entry+0xc0, seeds env/ThreadVars, runs `DT_INIT_ARRAY` (`init_array_entries` in nro.rs parses MOD0/dynamic), then resumes at the `bl main`. **Static constructors only run through this path.**
- `stub_nvdrv` (cpu.rs): svc ConnectToNamedPort("nvdrv") — cmd 0 → fd=1, cmd 1/2 → 0, cmd 3 (Initialize) → empty success reply.
- `svcWaitSynchronization` reports the wait satisfied with X1 = 1 ("one handle
  signaled") — the libnx wrapper stores X1 to the caller's out pointer, and
  deko3d uses it; leaving it garbage made the fence wait read a bogus waiter
  index. A Timeout (0xEA01) during deko3d device init is treated as fatal
  (`svcBreak` 0x1159).
- hbmenu state: runs its whole init + `graphicsInit` + enters the menu loop,
  but spins in a deko3d GPU fence wait (`nvFenceWait` → `nvHostOpMultiWait` →
  `svcWaitSynchronization` → retry, ~669 steps/iteration). `internalPoll()`
  reads a GPU semaphore that never signals because nvdrv/GPU isn't emulated,
  so no frame is ever rendered. Previously aborted earlier (svcBreak 0x1159
  reading a NULL slot base at `0x08254080`) and before that (memset over the
  stack).

## Gotchas

- `cargo clippy` on the whole workspace **fails** in `switch-wasm` (deliberate raw-pointer `extern "C"` signatures trigger `not_unsafe_ptr_arg_deref`). `clippy`/`fmt` are not gating; `cargo fmt --check` is also not clean.
- CPU test encodings in `tests/cpu_test.rs` are hand-assembled and cross-checked against QEMU's `a64.decode`.
- **SIMD&FP LDR/STR addressing**: the register-offset form has `bits[25:24]=00` (mode 0b00) — it must NOT be detected via bit 21, because bit 21 is the top bit of `imm12` in the immediate (unsigned-offset, mode 0b01) form. `ldr b29, [x0, #0xc80]` was being misread as a register load using a garbage Rm (this broke a hbmenu constructor). Cross-check any new load/store decode against `llvm-mc -triple=aarch64 -disassemble`.
