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

- `crates/switch-core` — the emulation core, zero external deps. Modules: `cpu` (A64 interpreter, split into `mod`/`alu`/`bits`/`fp`/`ipc`/`loadstore`/`simd`/`svc`/`system`), `gpu` (GM20B model), `display` (binder buffer queue), `vfs` (emulated SD card), `mem` (sparse 4 GiB page table), `disasm`, `nsp`/`nca`/`nro`/`elf` (parsers), `error`.
- `crates/switch-wasm` — `cdylib` for `wasm32-unknown-unknown`, exports a raw `extern "C"` ABI (no wasm-bindgen). Buffers cross the boundary via wasm linear memory (`switch_alloc`/`switch_free`); a handle is an index into a global session table. **No external deps by design** — JSON results are emitted by the hand-rolled `json_escape`/`write_into` helpers in `crates/switch-wasm/src/lib.rs`. Don't add serde/etc.
- `tools/serve.py` — local static server with no-cache headers and the right `application/wasm` MIME type (`instantiateStreaming` refuses anything else). Takes an optional port.
- `tools/make_og.py` — renders `web/assets/og.png`, the social preview card, around a real captured frame (`web/assets/screenshot.png`).
- `web/index.html` is the only file at the web root; everything else lives in `web/assets/`, which therefore exists in a fresh checkout. (`*.wasm`/`*.nro` are gitignored, so the directory used to be absent and the Pages build failed on `cp`.)
- `web/assets/worker.js` hosts the wasm module (`WebAssembly.instantiateStreaming` of `switch_wasm.wasm`) and is the glue over the ABI; `web/assets/main.js` is promise-based RPC over `postMessage` plus the app shell. Step budgets: non-trace 5,000,000 per slice, trace mode 5000.
- **Path trap**: `new Worker(...)` resolves against the *document*, so main.js says `assets/worker.js`; a `fetch` inside the worker resolves against the *worker script*, so worker.js says `switch_wasm.wasm` with no prefix.
- The UI is an app shell, not a page: the canvas owns the viewport, the tools drawer is closed by default (backtick toggles it, Space is run/pause), and the canvas resizes to whatever resolution the guest presents.
- Emulated CPU addresses are `u32` (32-bit PC). Memory is a sparse page table (`PAGE_COUNT` computed in `u64` to survive wasm32 `usize` truncation); reads/writes to unmapped addresses fault with `Error::Cpu`.
- The demo framebuffer is memory-mapped at `FB_BASE = 0x3F00_0000` (640x360 RGBA) and input at `INPUT_ADDR = 0x3F10_0000`. Syscall ABI: `SyscallMode::Horizon` stubs for real homebrew. Commercial encrypted content is out of scope.

## Homebrew memory layout (cpu.rs constants)

- Stack: `STACK_TOP = 0x1010_0000`, size 1 MiB (so `STACK_BASE = 0x1000_0000`), SP seeded at `STACK_TOP`.
- TLS base: `tpidr = 0x0FF0_0000` (deliberately clear of the heap). libnx ThreadVars at TLS+0x1E0 (magic `0x21545624` "!TV$"), `_REENT` zeroed at `0x1FF1_0000`.
- Heap via `svcSetHeapSize` returns `0x3000_0000` (address out-param; deliberately NOT `0x2000_0000` — with a 512 MiB heap there, `malloc`'s arena lands at `STACK_BASE + 0x1a80` and the app's 8 MiB memblock memset overwrites the stack). `svcQueryMemory` reports per-page state (allocated pages = RWX, untouched soft pages = unmapped) so libnx virtmem reservations find free address space. `svcGetInfo` uses the hbmenu libnx InfoType numbering (2/3 Alias, 4/5 Heap, 6/7 Total/Used memory, 12/13 Aslr, 14/15 Stack). Env block at `ENV_BLOCK_ADDR = 0x0010_0000`.
- `boot_homebrew` (cpu.rs): runs crt0 through the `bl main` at entry+0xc0, seeds env/ThreadVars, runs `DT_INIT_ARRAY` (`init_array_entries` in nro.rs parses MOD0/dynamic), then resumes at the `bl main`. **Static constructors only run through this path.**
- `nvdrv_request` (cpu/ipc.rs): the real `INvDrvServices` interface — Open/Ioctl/Ioctl2/Ioctl3/Close/Initialize/QueryEvent, dispatched into `gpu::nvdrv`.
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

## GPU (`crates/switch-core/src/gpu`)

A model of the Tegra X1's GM20B, not a stub. Register numbers come from
deko3d's generated Maxwell headers and the ioctl ABI from libnx's
`nvidia/ioctl`, so the command streams real homebrew emits are decoded as-is.

- `nvdrv` — the driver the guest opens: `/dev/nvmap`, `/dev/nvhost-ctrl`,
  `/dev/nvhost-ctrl-gpu`, `/dev/nvhost-as-gpu`, `/dev/nvhost-gpu`. ioctl
  numbers are the Linux-style `dir|size|type|nr` words libnx builds.
- `nvmap` — memory objects. On Tegra the *guest* allocates the buffer and
  hands nvmap its CPU address, so GPU memory is ordinary guest memory.
- `vmm` — the graphics MMU: whole-buffer GPU VA ranges, small-page region at
  `0x04000000` and big-page region from `0x1_00000000`.
- `syncpt` — host1x syncpoints and `/dev/nvhost-ctrl` event slots. A
  submission runs to completion inside its ioctl, so fences are already
  expired when the guest waits on them.
- `channel` + `engine/*` — the command processor: GPFIFO entries → pushbuffer
  → method headers (`Increasing`/`NonIncreasing`/`Inline`/`IncreaseOnce`) →
  the class bound to that subchannel. Classes: 3D (0xB197), compute (0xB1C0),
  inline-to-memory (0xA140), 2D (0x902D), copy (0xB0B5), gpfifo (0xB06F).
- `macro_engine` — the MME. Methods ≥ 0xE00 are macro slots, and deko3d
  compiles its draws into macros, so nothing draws without it.
- `surface` — block-linear (GOB) swizzling and the colour formats. A naive
  memory dump of a Switch framebuffer looks shredded because of this.

Scan-out: the app hands a finished image to `display::BufferQueue`
(`QUEUE_BUFFER`), which resolves the `NvGraphicBuffer` to an nvmap id and
`Gpu::present` de-swizzles it into `Gpu::framebuffer` (RGBA8888). The wasm
`switch_fb_*`/`switch_frame_count` exports feed that to the canvas, which
resizes itself to whatever resolution the guest picked.

**Not yet implemented**: the shader core. `ClearBuffers`, the copy engine, the
2D blitter and inline uploads all execute for real; `VertexBegin`/draw calls
are decoded and recorded but not rasterized, and compute dispatches record
their QMD address without running warps.

## Emulated SD card (`vfs.rs`)

`fsp-srv` is backed by a real path-addressed tree, so `GetEntryType`,
`OpenDirectory`, `fsDirRead`, `OpenFile` and `Read` all agree with each other
and a missing path returns `FsError_PathNotFound`. Returning a fixed listing
for every path made menus recurse forever. The running NRO is published at
`nro::HOMEBREW_NRO_PATH` and advertised as `argv[0]` through the homebrew ABI
environment block, which is how libnx's `romfsMountSelf` finds the RomFS
appended to it.

## A64 traps found the hard way

- **Register 31 in ADD/SUB**: SP in the immediate and extended-register forms,
  XZR in the shifted-register form. `neg x1, x0` is `sub x1, xzr, x0`; reading
  SP there silently corrupted every `aligned_alloc`.
- **SIMD&FP load/store mode 0b00** is not just the unscaled STUR/LDUR form:
  bits[11:10] select unscaled / post-index / pre-index. Missing the write-back
  left `str q0, [x2], #16` looping forever.
- `Cpu::backtrace` walks the guest's X29 frame chain (devkitA64 keeps frame
  pointers), which is the fastest way to find which libnx function issued an
  IPC request.
