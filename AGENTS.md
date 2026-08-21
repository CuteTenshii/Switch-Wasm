# AGENTS.md

Browser-oriented Nintendo Switch emulation core: an ARM64 (A64) integer interpreter plus PFS0/NSP, NCA, NRO and ELF parsers, compiled to WASM for the frontend.

## Commands

- `make all` — test + wasm + assets (the full pipeline).
- `make test` — `cargo test -p switch-core`. This is the only crate with tests (unit tests in each module + `crates/switch-core/tests/cpu_test.rs`).
- `make wasm` — `cargo build --target wasm32-unknown-unknown --release -p switch-wasm`.
- `make assets` — wasm, then **copies the built `.wasm` into `web/assets/`**.
- `make serve` — `python3 tools/serve.py` (after assets). Frontend is plain static JS; no bundler.
- Single test: `cargo test -p switch-core <test_name>`.
- `python3 tools/difftest.py` — differential-test the decode against **real ARM semantics**: it assembles the instruction list in the script, runs it under `qemu-aarch64`, runs the identical bytes through `cargo run --example difftest`, and prints the first register that differs. `--scalar` does the same for the integer instructions (dumping x0..x25 instead of the vector registers). Needs `clang` + `lld` + `qemu-aarch64`. **Add an instruction there before hand-deriving expected values.** This caught TRN1/TRN2 taking the wrong lanes, and the scalar sweep found seven more in one run (EXTR's operand order, ADCS/SBC's op bit, SDIV's sign extension, SMADDL's operand width, CLS off by one).
- `rustup target add wasm32-unknown-unknown` is required for the wasm build.

## Build pipeline traps

- The browser fetches `web/assets/switch_wasm.wasm` and bundled `.nro` files as committed copies. Building `switch-wasm` alone does **not** refresh them — run `make assets`.
- `make test` only covers `switch-core`; there are no tests for `switch-wasm`.

## Architecture

- `crates/switch-core` — the emulation core, zero external deps. Modules: `cpu` (A64 interpreter, split into `mod`/`alu`/`bits`/`fp`/`ipc`/`loadstore`/`simd`/`svc`/`system`), `gpu` (GM20B model), `display` (binder buffer queue), `vfs` (emulated SD card), `mem` (sparse 4 GiB page table), `disasm`, `nsp`/`nca`/`nro`/`elf`/`romfs` (parsers), `control` (a title's NACP name/publisher/icon, out of the Control NCA's RomFS), `error`.
- `crates/switch-wasm` — `cdylib` for `wasm32-unknown-unknown`, exports a raw `extern "C"` ABI (no wasm-bindgen). Buffers cross the boundary via wasm linear memory (`switch_alloc`/`switch_free`); a handle is an index into a global session table. It declares exactly **one import**, `env.host_read` — how it reads the open container (see the container section below). **No external deps by design** — JSON results are emitted by the hand-rolled `json_escape`/`write_into` helpers in `crates/switch-wasm/src/lib.rs`. Don't add serde/etc.
- `tools/serve.py` — local static server with no-cache headers and the right `application/wasm` MIME type (`instantiateStreaming` refuses anything else). Takes an optional port.
- `tools/make_og.py` — renders `web/assets/og.png`, the social preview card, around a real captured frame (`web/assets/screenshot.png`).
- `web/index.html` is the only file at the web root; everything else lives in `web/assets/`, which therefore exists in a fresh checkout. (`*.wasm`/`*.nro` are gitignored, so the directory used to be absent and the Pages build failed on `cp`.)
- `web/assets/worker.js` hosts the wasm module (`WebAssembly.instantiateStreaming` of `switch_wasm.wasm`, with `{ env: { host_read } }` as its imports) and is the glue over the ABI; `web/assets/main.js` is promise-based RPC over `postMessage` plus the app shell. Step budgets: non-trace 5,000,000 per slice, trace mode 5000.
- **Path trap**: `new Worker(...)` resolves against the *document*, so main.js says `assets/worker.js`; a `fetch` inside the worker resolves against the *worker script*, so worker.js says `switch_wasm.wasm` with no prefix.
- The UI is an app shell, not a page: the canvas owns the viewport, the tools drawer is closed by default (backtick toggles it, Space is run/pause), and the canvas resizes to whatever resolution the guest presents. The boot splash is dismissed when a program **loads**, not when the first frame arrives — homebrew can run (or fault) for a long time before presenting, and waiting for a frame made the stage look dead until it crashed. `Res` reads `—` and the canvas stays blank until `frame_count > 0`; `fb_snapshot`'s pre-first-frame fallback (the memory-mapped demo framebuffer) is deliberately *not* painted, since real homebrew never writes that region.
- Emulated CPU addresses are `u32` (32-bit PC). Memory is a sparse page table (`PAGE_COUNT` computed in `u64` to survive wasm32 `usize` truncation); reads/writes to unmapped addresses fault with `Error::Cpu`.
- The demo framebuffer is memory-mapped at `FB_BASE = 0x3F00_0000` (640x360 RGBA) and input at `INPUT_ADDR = 0x3F10_0000`. Syscall ABI: `SyscallMode::Horizon` stubs for real homebrew. Commercial encrypted content is out of scope.

## Containers are never staged in memory

**A retail `.nsp` does not fit and never will.** A modern title's container
runs to several gigabytes; wasm32 linear memory tops out at 4 GiB, and Rust's
allocator on that target refuses any *single* allocation above `isize::MAX`
(2 GiB) — `Layout::from_size_align` returns `Err`, and the `.unwrap()` behind
it lowers to `unreachable`, so an oversized request used to take the whole
module down as `RuntimeError: unreachable executed` with nothing to say what
had asked for what. `switch_alloc` now returns null there instead, and
`worker.js`'s `alloc` throws on it.

Nothing reads a container as a buffer. `switch_core::source::ByteSource` is a
`u64`-addressed random-access range, and the pieces compose:

    HostSource (the open file)
      └ Window          — the NCA at file-table entry N
          └ SectionSource — decrypting view of one NCA section (AES-CTR is
             │              seekable: the keystream block a byte gets is
             │              decided by its own position, so a range costs
             │              exactly that range)
             └ Window     — the RomFS image, past the IVFC hash levels

- The browser keeps the `File` and serves ranges through the `host_read`
  import, which **must answer synchronously** — the emulator asks for RomFS
  ranges from inside `switch_run`, where there is nowhere to await. That is
  `FileReaderSync`, which exists only in a worker, and is the second reason
  the emulator lives in one. `worker.js` caches 1 MiB chunks (32 of them, LRU)
  because a RomFS mount walks its tables in small reads; a read larger than a
  chunk bypasses the cache rather than evicting it.
- `switch_open_nsp(handle, size)` / `switch_open_nca(handle, size)` open what
  the host has ready; only the PFS0 header is read. `switch_load_nca_from_nsp`
  boots straight out of it.
- The ExeFS **is** read in full (it is a title's executables, tens of MB at
  the outside) and hash-verified. The RomFS is not: `Cpu::set_romfs_source`
  takes the decrypting view, and `IStorage::Read` (`cpu/ipc.rs`) copies
  through a 64 KiB staging buffer, so a guest can ask for any range of a
  gigabytes-large RomFS.
- Offsets stay `u64` end to end. `f.offset as usize` on wasm32 silently
  truncated every entry past the 4 GiB mark *and* made the bounds check that
  should have caught it pass — `Pfs0::image_size` is `u64` for the same
  reason.

## Homebrew memory layout (cpu.rs constants)

- Stack: `STACK_TOP = 0x1010_0000`, size 1 MiB (so `STACK_BASE = 0x1000_0000`), SP seeded at `STACK_TOP`.
- TLS base: `tpidr = 0x0FF0_0000` (deliberately clear of the heap). libnx ThreadVars at TLS+0x1E0 (magic `0x21545624` "!TV$"), `_REENT` zeroed at `0x1FF1_0000`.
- Heap via `svcSetHeapSize` returns `0x3000_0000` (address out-param; deliberately NOT `0x2000_0000` — with a 512 MiB heap there, `malloc`'s arena lands at `STACK_BASE + 0x1a80` and the app's 8 MiB memblock memset overwrites the stack). `svcQueryMemory` reports per-page state (allocated pages = RWX, untouched soft pages = unmapped) so libnx virtmem reservations find free address space. `svcGetInfo` uses the hbmenu libnx InfoType numbering (0/1 Core/Priority mask, 2/3 Alias, 4/5 Heap, 6/7 Total/Used memory, 12/13 Aslr, 14/15 Stack, 16/17 System resource, 21/22 Total/Used non-system memory) — see the memory-region section below. Env block at `ENV_BLOCK_ADDR = 0x0010_0000`.
- `boot_homebrew` (cpu.rs): runs crt0 through the `bl` at entry+0xc0, seeds env/ThreadVars, runs `DT_INIT_ARRAY` (`init_array_entries` in nro.rs parses MOD0/dynamic), then resumes at that `bl`. **Static constructors only run through this path.** That call is libnx's `__libnx_init(ctx, main_thread, saved_lr)`: the constructor pass zeroes the registers, so all three arguments are re-seeded before resuming — including `saved_lr` = `SELF_RETURN_TRAMPOLINE`, which `envSetup` keeps as the exit function pointer. With it left at 0, `__nx_exit` branched to NULL and a clean exit looked like a crash.
- `nvdrv_request` (cpu/ipc.rs): the real `INvDrvServices` interface — Open/Ioctl/Ioctl2/Ioctl3/Close/Initialize/QueryEvent, dispatched into `gpu::nvdrv`.
- `svcWaitSynchronization` reports the wait satisfied with X1 = **0**: X1 is
  the *index* of the handle that signaled, and with every object pretended
  signaled the first one is it. The libnx wrapper stores X1 to the caller's
  out pointer and callers index their own waiter array by it, so garbage
  there sends them to the wrong object. It used to answer 1 unconditionally,
  which is out of range for a single-handle wait — `nnSdk`'s system worker
  (`nn::os::detail::MultiWaitImpl::WaitAny`) then read a
  `MultiWaitHolderType` past the end of its list and `blr`'d its null handler.
  A Timeout (0xEA01) during deko3d device init is treated as fatal
  (`svcBreak` 0x1159).
- hbmenu state: **its menu renders correctly** — title, theme background, entry
  tile and the icon (JPEG-decoded, pixel-exact against a reference decode) all
  composite through the real path: CPU-drawn linear buffer → deko3d copy-engine
  blit → swapchain → binder present.
- **hbmenu does not need the shader core.** `nx_graphics.c` draws with the CPU
  into a linear memblock and its command list is just
  `dkCmdBufCopyBufferToImage` + `dkCmdBufSignalFence`; its assets are raw RGBA
  `.bin` bitmaps. Only the copy engine and syncpoints are involved.

## Guest memory regions (`cpu/mod.rs`, `svcGetInfo`)

**Every region `svcGetInfo` reports has to be an address the emulator can
actually represent.** Guest memory is indexed with a `u32`, so the whole
address space is the low 4 GiB and `Cpu::bootstrap` soft-maps 0..0x8000_0000
(unwritten pages read as zeros and allocate on first write — that *is* demand
paging, and it is why a region can be reported far larger than the
`MAX_MAPPED_BYTES` RAM cap without costing anything).

`svcGetInfo` used to report Horizon's real region bases literally — alias at
0x10_0000_0000, heap at 0x2_0000_0000. `nnSdk` took the alias base at its word
and asked `svcMapPhysicalMemory` to back 0x10_0000_0000, which truncates to 0.
The regions now live in representable space: alias at
`GUEST_ALIAS_REGION_ADDR` (0x4000_0000, 1 GiB, up to the end of the soft map),
heap at `GUEST_HEAP_REGION_ADDR` (0x3000_0000, ending where `FB_BASE` begins).

- **`svcMapPhysicalMemory` (0x2c) is how a retail title grows its heap**, not
  `svcSetHeapSize` (0x01) — an application built for the 39-bit address space
  picks the address itself out of the alias region, which is why tracing one
  shows syscall 0x01 never being issued at all. 0x2c leaves the pages to the
  soft map and only validates the range; 0x2d (`UnmapPhysicalMemory`) does the
  real work of handing pages back.
- **InfoType 21/22 (Total/UsedNonSystemMemorySize) size the application
  heap.** `nn::init`'s startup hands their difference straight to
  `nn::mem::StandardAllocator::Initialize`, which asserts on any page-aligned
  span under 16 KiB — so the `_ => 0` default aborted the boot.
- **InfoType 16 (SystemResourceSizeTotal) is deliberately 0**, and is the one
  figure not taken from the title's own NPDM (which asks for 16 MiB). A
  non-zero answer makes `nn::os::detail::VammManager::InitializeIfEnabled`
  switch the heap onto a virtual-address-memory manager that reserves address
  space out of the alias region and backs it page by page — kernel machinery
  that does not exist here — and `nn::os::AllocateAddressRegion` fails with os
  result 3-12.

**There is no stderr in the browser.** `wasm32-unknown-unknown` has no WASI,
so an `eprintln!` goes nowhere and `std::env::var` always fails — every
`TRACE_*`-gated trace is host-CLI-only. A diagnostic that has to reach a
browser user goes through `Cpu::diagnostic`, which writes stderr *and* the
trace buffer the page drains (`switch_drain_trace`), and is recorded whether or
not per-instruction tracing is on. `main.js` drains it every run slice, so
`[ipc] unimplemented` and `[ipc] no implementation` show up in the page's
console panel as they happen.

`TRACE_WAIT=1` prints every event as it is handed out and every
`svcWaitSynchronization` with the state of what it is waiting on — the fastest
way to see whether a guest is holding a real event handle or 0.

`TRACE_SVC=1` prints every syscall except the three hot ones
(`SendSyncRequest`, `WaitSynchronization`, `SleepThread`), plus each
`svcGetInfo` answer. It is the counterpart of `TRACE_IPC` and the fastest way
to see which InfoType a stuck `nnSdk` is unhappy with.

## Retail process entry (`Cpu::boot_retail_program`)

Horizon's process entry ABI is two registers, and `rtld`'s first two
instructions read both literally (`cmp x0, #0` / `mov w19, w1`):

- **X0** — the launch argument. `0` for a normal process launch; non-zero only
  for the homebrew loader's config block, which sends `rtld` down a different
  path entirely.
- **X1** — the **main thread's handle** ([`MAIN_THREAD_HANDLE`]). `nnSdk`
  stores it in the main `nn::os::ThreadType` at **+0x1B0**, and
  `nn::os::detail::InternalCriticalSectionImplByHorizon::IsLockedByCurrentThread`
  compares every `SdkMutex` lock word (masked with `0xBFFFFFFF`, dropping the
  has-waiters bit) against it. Leaving X1 at 0 makes an *unlocked* mutex (lock
  word 0) compare equal to "owned by the current thread", so the very first
  `nn::os::SdkMutexType::Lock` fires its recursive-lock assertion and the
  title aborts inside `nn::oe::Initialize` before reaching any service.

`svcGetInfo`'s CoreMask (0) and PriorityMask (1) come from the NPDM's
`ThreadInfo` kernel capability; every retail application carries cores 0..2
(mask `0b111`) and priorities 28..59, which is what "A Short Hike"'s own
`main.npdm` says. Reporting 0 there hands
`nn::os::GetThreadAvailableCoreMask` an empty mask, whose inlined
highest-set-bit scan inside `nn::os::RegisterSystemWorkerHandler` asserts.

## Guest threads (`cpu/mod.rs`, `cpu/svc.rs`)

Cooperative: a thread runs until it makes a blocking syscall and only then does
another get the CPU. Real Horizon preempts, but every libnx synchronization
primitive re-checks its predicate in a loop, so co-operative switching completes
the same handshakes.

- `svcCreateThread` builds a `ThreadContext` with its own TLS block (and the
  `ThreadVars` libnx reads through TPIDRRO_EL0); `svcStartThread` marks it
  runnable; returning from a thread entry hits `THREAD_EXIT_TRAMPOLINE`, which
  is `svcExitThread`.
- **Mutexes and condvars are real**: Horizon keeps the lock word in guest memory
  (owner handle, plus bit30 for "has listeners"), and libnx re-reads it after
  every arbitration — so `svcArbitrateUnlock` has to actually hand ownership to
  a waiter and `svcWaitProcessWideKeyAtomic` has to release the mutex. Returning
  success from those stubs left hbmenu's worker spinning on a lock its main
  thread held.
- The SVC path retires the instruction (`self.pc = next_pc`) *before* dispatching,
  so a syscall that switches threads can install the incoming PC.
- If every thread is blocked, `reschedule` wakes them all rather than hanging:
  a spurious wake degrades to the old spin.

## Gotchas

- `cargo clippy` on the whole workspace **fails** in `switch-wasm` (deliberate raw-pointer `extern "C"` signatures trigger `not_unsafe_ptr_arg_deref`). `clippy`/`fmt` are not gating; `cargo fmt --check` is also not clean.
- CPU test encodings in `tests/cpu_test.rs` are hand-assembled and cross-checked against QEMU's `a64.decode`.
- **SIMD&FP LDR/STR addressing**: the register-offset form has `bits[25:24]=00` (mode 0b00) — it must NOT be detected via bit 21, because bit 21 is the top bit of `imm12` in the immediate (unsigned-offset, mode 0b01) form. `ldr b29, [x0, #0xc80]` was being misread as a register load using a garbage Rm (this broke a hbmenu constructor). Cross-check any new load/store decode against `llvm-mc -triple=aarch64 -disassemble`.
- **Decode groups whose opcode bits are split**: several AArch64 groups put part of an opcode either side of a fixed field, so matching a contiguous slice silently matches nothing. The scalar-FP 1-source group is `opcode(6) 10000`, i.e. bits[15:10] are `opcode<0>:10000` — testing `bits[15:10] == 0b100000` made the whole group (FMOV/FABS/FSQRT/FRINTx) dead code, and the 3-source group (FMADD family) has its own top byte (`00011111`), so it has to be matched before the `00011110` space. When adding a group, check the guard actually reaches it with a real encoding.
- **BLR reads its target before linking.** `blr x30` is a legal
  return-and-relink (hbmenu's NEON IDCT ends that way); writing x30 first makes
  it branch to itself+4, which looked like an infinite loop inside the JPEG
  decoder.
- **The permute trio place their results differently**: TRN takes the even (or
  odd) elements of *both* operands and interleaves them, ZIP interleaves one
  half of each, UZP packs every other element of Vn into the low half and Vm's
  into the high half. Conflating them scrambles a matrix transpose — `trn1`
  picking Vm's odd elements is what stalled hbmenu's icon decode.
- **AdvSIMD structure loads/stores** (`LD1`–`LD4`/`ST1`–`ST4`): writeback is bit 23 (the post-index forms) and `Rm == 31` means "increment by the transfer size"; a different `Rm` is a register increment. Keying writeback off `Rm` alone left `ld1 {v1.16b, v2.16b}, [x2], #32` without its base update, so newlib's `strrchr` returned a pointer 32 bytes below the string and `PHYSFS_init` failed on a garbage `argv[0]` directory. The single-lane forms spread their lane index across `Q:S:size`, and `scale == 0b11` is the load-and-replicate group (`LD1R`), not a doubleword lane insert.
- **BSL/BIT/BIF** differ only in which register is the mask: BSL selects with Vd, BIT and BIF with Vm. Getting that wrong broke newlib's vectorised `strchr` (it folds the "matched" and "end of string" predicates with `bif`), so every `device:` prefix lookup fell through to the default device and `romfs:/…` was looked up on the SD card.
- **A write to a W register zeroes bits 63:32.** Every 32-bit result has to be truncated: SBFM's sign extension was filling the top half, so `asr w0, w0, #31` produced `0xFFFF_FFFF_FFFF_FFFF` and any later 64-bit use of that register saw a huge value. Related: a 32-bit operand must be sign-extended from *bit 31* before an arithmetic shift or a signed divide — masking it to 32 bits and treating it as a positive `i64` made `asr w, w, w` and `sdiv w, w, w` unsigned, which is how libjpeg-turbo's `HUFF_EXTEND` lost the sign of every JPEG DC difference (hbmenu's icon decoded with grey luma and magenta chroma).
- **A guard that includes a fixed bit kills the whole group.** Three FP classes were dead code for this reason: the 1-source group (bits[15:10] matched as a unit though the opcode's low bit is bit15), FCSEL/FCCMP (guarded on bit21 == 0 when they have it set), and the int↔float conversions (`rmode`:`opcode` read as bits[21:16], which folds in the fixed bit21 — that made `ucvtf d0, x1` execute as FCVTMU and write **x0**). After adding a group, prove the guard reaches it with a real encoding from `llvm-mc`.
- The AdvSIMD **scalar** forms are separate encodings from the vector ones: shift-by-immediate has bit28 set, two-register-misc is `01 U 11110 …`. Both share the vector implementation with a one-lane count.
- **EXT** (`0 Q 101110 00 0 Rm 0 imm4 0 Rn Rd`) shares bits[28:24] with the permute group, which is why permute also has to require bit29 == 0.
- **TBL/TBX** (`0 Q 001110 00 0 Rm 0 len op 00 Rn Rd`) share bits[29:21] with the copy group (DUP/INS/UMOV/SMOV), so the copy decode has to let them past: every copy encoding sets bit10, table lookup has bit15 == 0 and bits[11:10] == 00. The table is `len+1` consecutive vector registers and **wraps past v31**; an index beyond it reads zero for TBL and leaves the destination byte alone for TBX. This is how a compiler spells a byte shuffle no fixed permute matches — NXpotify's SHA-512 hits `tbl v31.16b, {v29.16b}, v28.16b` (`0x4e1c03bf`) and faulted on it.
- **Vector FP** lives in two groups the integer three-same decode must not swallow: three-same opcodes from `0b11000` up (where bits[23:22] are `a:sz`, not an element size) and two-register misc (`bits[21:17] == 10000`, `bits[11:10] == 10`), whose FP forms are identified by `(U, size<1>, opcode)` together — opcode `11101` is SCVTF when `size<1> == 0` but FRECPE when it is 1.
- **CTR_EL0** reports the Cortex-A57 value `0x8444C004`; cache-flush loops stride by `4 << DminLine`, so reporting 0 walked buffers 4 bytes at a time.

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

Subchannel 6 (`SUBCHANNEL_GPFIFO`) is pre-bound to the channel's own
`MAXWELL_CHANNEL_GPFIFO_A` class, because nvhost binds it at channel creation and
userspace never issues a `SetObject` for it — deko3d writes its syncpoint
increments and cache-flush ops straight there.

Scan-out: the app hands a finished image to `display::BufferQueue`
(`QUEUE_BUFFER`), which resolves the `NvGraphicBuffer` to an nvmap id and
`Gpu::present` de-swizzles it into `Gpu::framebuffer` (RGBA8888). The wasm
`switch_fb_*`/`switch_frame_count` exports feed that to the canvas, which
resizes itself to whatever resolution the guest picked.

Draws are rasterized for real (`gpu/raster.rs`) with the SASS interpreter in
`gpu/shader/interp.rs` shading each vertex and each covered fragment; compute
dispatches still record their QMD address without running warps.

**A fragment shader runs once per covered pixel, so anything per-invocation is
per-pixel.** A full-screen pass is 921 600 of them, and NXpotify's frame is
five such passes. That makes the usual instinct — allocate a small `Vec`, use
a `HashMap` for a sparse thing, rescan the program to answer a question —
cost whole seconds a frame. Three that did: `texs` rescanned the program to
find where its result lands (a hundred heap allocations per pixel; it is a
property of the decoded program and is cached on `Program` now), `a[]` was a
`HashMap` (it is a 256-word flat array now, with a written-mask so "never
written" stays distinguishable from "wrote zero"), and the rasterizer built a
fresh `Invocation` per pixel (one per draw now, reset per pixel). Together
those were 2.6 s/frame → 0.67 s. Measure before and after with
`examples/screenshot` at two frame indices; the frames must stay
byte-identical.

## IPC dialects (`cpu/ipc.rs`)

Two guest IPC stacks reach these stubs and they disagree about how strict the
replies are:

- **libnx** (hbmenu, NX-Shell) ignores the reply's `type` field and its raw-data
  size, and converts `fsp-srv` to a **domain**, so sub-interfaces come back as
  out-object ids on one session handle.
- **libtransistor** (sdl-hello) validates both. A reply's `type` must be 0 or 4
  (`0x40` failed every reply with its error `0x7E0DD`), the move-handle count
  must match what the caller declared, and the raw-data size must be exactly
  what the command documents — `nvdrv` `Initialize` really does return a `u32
  error`, and omitting it failed nv init. It also never converts to a domain, so
  a sub-interface has to come back as a **session handle in a move handle**;
  `Cpu::reply_with_interface` picks the right shape per request and files the
  object's state under `Cpu::object_key(handle, object_id)`.

There are **two encodings of every message kind**: the plain one (`Request` =
4, `Control` = 5) and the "with context" one (`RequestWithContext` = 6,
`ControlWithContext` = 7), which prefixes the raw data with a 16-byte tracing
context. `libnx` sends the plain form; **`nnSdk` sends the context form for
everything**. Test control-ness with `Cpu::ipc_is_control_request`, never `type
== 5` — `appletOE`'s opening message from a retail title is
`QueryPointerBufferSize` as type 7, and reading it as an ordinary command
killed the applet chain before it opened.

The generic reply for a service with no dedicated stub
(`Cpu::reply_with_fabricated_object`) answers with a fresh object id **and a
real sub-session**, allocated once per `(session, command)` and reused. The
handle is not optional: `nnSdk` reads an out-object off a plain session as a
move handle, and a reply carrying none is not an error to it — the handle
parses as 0, the client silently skips constructing the proxy, and the command
still returns **success**. The caller's first virtual call then goes through a
null pointer, which is how boot2 reached `pc=0` one instruction after `gpio`'s
`OpenSession2` was answered "successfully", with nothing in the log between the
lie and the crash.

That reply used to guess the *applet* state commands (`ReceiveMessage` → 15,
`GetOperationMode` → 1, …) for any service whose name started with "applet";
those numbers leaked — `pl:u`'s `GetLoadState` is also command 1, and answering
it with 15 left NX-Shell polling the shared-font service 190k times.

`ssl` (`ssl_request`) is the system TLS stack — Switch does not let a title
bring its own, so a title asks the OS to build connections
(`CreateContext` → `ISslContext::CreateConnection` → an `ISslConnection`
wrapping a `bsd:u` socket). The local half is implemented, because contexts and
their options are ordinary objects that exist whether or not anything is
reachable; **`CreateConnection` deliberately reports itself as unimplemented**
rather than handing back a connection that can never connect, since there is no
socket layer under it. An offline retail title only ever calls
`SetInterfaceVersion`, which `nnSdk` issues at startup because `ssl` is in the
NPDM service list.

`hid` (`hid_request`) is the **negotiation** around input, not the input
itself. The data — buttons, sticks, touch — lives in a 256 KiB shared memory
region the guest reads directly with no IPC per frame, which
`Cpu::set_gamepad_state` already fills. `IHidServer::CreateAppletResource` →
`IAppletResource::GetSharedMemoryHandle` is what hands that region over, and
the handle it returns is now what `svcMapSharedMemory` recognises (the old
match on a 0x40000 size stays as a fallback, and is the reason `libnx` got
working input out of a fabricated reply while `nnSdk` called a method on a null
`IAppletResource`).

Two things there are easy to get wrong:

- **`QueryPointerBufferSize` must be non-zero for `hid`.**
  `nn::hid::SetSupportedNpadIdType` marshals its npad id array as a
  send-static ("pointer") buffer, and `nnSdk`'s client checks the negotiated
  size *before* sending, failing outright when the server claims it can take
  nothing. Note this also changes how `libnx`'s AutoSelect marshals — read
  input buffers with `Cpu::ipc_input_buffer`, which tries both forms.
- **The `Set*`/`Get*` pairs have to agree.** `SetSupportedNpadStyleSet` /
  `GetSupportedNpadStyleSet` and the joy-hold-type pair are read back; a caller
  that reads back something it did not set decides the controller it wanted is
  not there.

Vibration comes back out the same way: `SendVibrationValue` carries a
`HidVibrationValue` (amplitude and frequency for a low band and a high band),
the two amplitudes are kept in `Cpu::vibration`, and the page maps them onto
the Gamepad API's `dual-rumble` `strongMagnitude`/`weakMagnitude` — Switch
rumble is two independently driven actuators, which is the same shape.

**Events are copy handles, and they start unsignalled.** `Cpu::alloc_event`
names one and records it in `Cpu::events`; it has to reach the guest through
`Cpu::write_ipc_reply`'s **copy** list, not the move list — a move handle
transfers ownership (a sub-session from `reply_with_interface`), a copy handle
duplicates one the server keeps, they occupy different fields of the handle
descriptor, and an event sent in the move slot is read back as **0**. That is
why `nnSdk` spent whole boots waiting on handle 0.

`svcWaitSynchronization` then answers from real state:

- A handle that is *not* in `Cpu::events` still counts as ready. Thread handles
  and every service handle this emulator does not model keep behaving as they
  always have — do not "fix" this without checking homebrew.
- A **poll** (timeout 0, which is what `nn::os::TryWaitSystemEvent` and libnx's
  `waitSingle` issue) on events that have not fired reports Timeout. This is
  the fix that stopped `nn::oe::GpuErrorHandler` being told the GPU had
  faulted.
- A **blocking** wait with nothing signalled still reports the first handle
  ready, which is a lie and is deliberate. `nn::os::detail::MultiWaitImpl::
  WaitAny` answers a timeout by returning a **null holder** that
  `nn::os::RegisterSystemWorkerHandler` calls without checking, so telling that
  thread the truth jumps to 0; and blocking it for real is worse, because
  nothing here fires those events and the last runnable thread would have
  nowhere to go. Fixing this properly needs events that actually fire, not a
  different answer here.
- `Cpu::vsync_event` is signalled from the guest's own presented frames, which
  is the only periodic tick this emulator has — there is no clock behind the
  display.

**`CloneCurrentObject` (control command 2, and 4 for the Ex form) must hand
back a new session handle as a move handle.** It is answered centrally in
`svc.rs`, before per-service dispatch, because it is session management and
identical for every service: the clone reaches the same interface and inherits
the same domain objects. Every service's control path used to answer it with a
bare success and no handle at all, and `nnSdk` clones `fsp-srv` before it
mounts anything — so `nn::fs::MountRom("rom", …)` failed without ever issuing a
filesystem command, and the title only noticed much later, when
`nn::fs::OpenDirectory("rom:/Data")` found no such mount.

**Mount names live in the guest, not here.** `nn::fs`'s `MountTable` is
client-side inside `sdk`: a successful mount registers the name and the
emulator never sees it. So there is nothing to implement for `rom:` beyond
making the calls underneath it work — `OpenDataStorageByCurrentProcess`
(command 200) handing back an `IStorage` over the process's RomFS, and that
storage reading correctly.

**`IStorage::Read` is `(s64 offset, u64 size)` — not `IFile::Read`.** A file
read leads with a `u32 option` and pads to 8, putting its offset at +8 and its
size at +0x10; a storage read has neither. Reading the file layout on a storage
made every RomFS read return "0 bytes at offset 0x50", so the guest mounted its
RomFS, parsed an empty header, and found none of its own files.

`lm` (`lm_request`) is where a title's **own** diagnostic output goes —
`nnSdk`'s `NN_LOG` and everything on top of it, rather than
`svcOutputDebugString`. `ILogger::Log` carries one LogPacket in a map-alias
send buffer (`Cpu::ipc_send_buffer`): a 0x18-byte header (pid, thread id,
flags, severity, verbosity, payload size) then TLV chunks, of which key 2 is
the message text and key 6 the module name. A long message is split across
packets with `flags` bit 0 marking the head and bit 1 the tail, so the text
accumulates and only the tail ends the line. Output lands in `Cpu::out`,
alongside the guest's `svcOutputDebugString` writes. Retail builds often
compile their logging out, so an empty log is not evidence this is broken —
`lm_writes_the_guests_own_log_to_the_console` is.

`pctl` (parental controls, `pctl_request`) reports the console as
**unrestricted**, which is the true state here — no PIN, no play timer, no
linked guardian, and the one local account `acc` reports has no Nintendo
Account behind it. Watch the direction of the answers: a `Confirm*`/
`Check*Permission` command replies with a bare `Result` where success *is*
"permitted" (a restriction is an error the caller checks for by value), an
`IsRestriction*`/`IsRestrictedBy*` query is `false`, and an
`IsFreeCommunicationAvailable`/`IsStereoVisionPermitted` query is `true`. The
last family reads the opposite way from the middle one, so a blanket `false`
would report free communication as unavailable.

`acc` (user accounts, `acc_request`) models a console with **exactly one
user**, always signed in, whose uid is `ACCOUNT_UID` — nonzero, because zero is
`AccountUid`'s "no user" sentinel and a title handed it back concludes nobody
is signed in. `acc:u0` is the application-facing service and `acc:u1`/`acc:su`
the system-facing ones; they share commands 0..=51 and **diverge from 100 up,
where the same command id means different things** (100 is
`InitializeApplicationInfo` on `acc:u0` but `GetUserRegistrationNotifier` on
`acc:u1`), so those arms dispatch on which service the session was opened under
rather than on the command alone. The nickname is the one piece of real state:
`IProfileEditor::Store` writes it into `Cpu::account_nickname` and
`IProfile::GetBase` reads it back out. `LoadImage` hands over a real JPEG that
`solid_jpeg` encodes — a caller feeds what it gets straight to a decoder, so
zero bytes is nothing to decode.

`IProfile::Get` is why `Cpu::ipc_recv_static_buffers` exists: its
`AccountUserData` comes back through a **receive-static ("pointer") buffer**,
the one descriptor kind that sits *after* the raw data — at the unaligned data
offset plus `num_data_words`, which counts the padding that aligns the CMIF
header. It is also why `acc` answers `QueryPointerBufferSize` with a real size:
a client told the server has no room marshals no descriptor at all and then
reads the icon id and background colour back out of its own stack. Read one
with `Cpu::ipc_output_buffer` (map-alias first, pointer buffer second), the
mirror of `ipc_input_buffer`.

`apm` (performance management, `apm_request`) is the clock profiles, and there
is nothing here to clock — the CPU is an interpreter and the GPU a software
rasterizer. What it has to do is **agree**: `GetPerformanceMode` reports the
same Normal that `am`'s `ICommonStateGetter::GetPerformanceMode` does, and
`GetPerformanceConfiguration` gives back, per mode, whatever
`SetPerformanceConfiguration` was last handed (the defaults are nonzero, since
0 is `ApmPerformanceConfiguration_Invalid`).

`bsd` (sockets, `bsd_request`) models a console whose link is up — which is
what `nifm` already claims — and on which **nothing ever answers**. A browser
tab cannot open a TCP socket and nothing here proxies one, so the split is:
every genuinely local operation really succeeds (socket, bind, listen,
get/setsockopt, fcntl, shutdown, close, dup) and every operation needing a peer
fails immediately with a definite errno — `connect` is `ECONNREFUSED`, the data
path is `ENOTCONN` for a stream socket and `ENETUNREACH` for a datagram one,
`accept` on a listening socket is `EAGAIN`, and `select`/`poll` report nothing
ready. An empty answer is not the same as an *instant* answer, though: a `poll`
that was handed a timeout is a wait, and `bsd_request` asks for a reschedule
(`Cpu::pending_yield`, consumed by `svcSendSyncRequest` once the reply and X0
are written — switching threads swaps the register file) before returning zero.
Threads here only hand over at blocking syscalls, so without that a guest
looping on `if (poll(&pfd, 1, 200) <= 0) continue;` around an idle socket owns
the CPU forever and starves every other thread, its own main one included.
NXpotify's Zeroconf listener is exactly that loop, and it never drew a frame.
A `poll` with a zero timeout is an explicit non-blocking probe and still
returns at once, as it does on hardware. **The errnos are FreeBSD's** (`EAGAIN` is 35, not 11) because that is what the
real service returns. `fcntl`'s flags word is stored and returned *verbatim*
rather than decoded — `O_NONBLOCK` is a different bit in FreeBSD, newlib and
Linux, and what has to hold is that a guest reads back what it set.

`ts` (temperatures, `ts_request`) reports the two sensors real hardware has —
the SoC (`TsLocation_Internal`) and the PCB (`TsLocation_External`) — at a
fixed idle reading, which is the true state of a console with no silicon to
heat. Three things to keep straight. The constraint is internal consistency:
`GetTemperatureMilliC` is `GetTemperature` times a thousand, and both sit
inside what `GetTemperatureRange` reports, since a caller scaling a gauge by
that range would otherwise draw the needle off the end. **`ISession` is a
different interface from the server it came from**: its `GetTemperature` is
command 4, the same id the server uses for `OpenSession`, and it returns a
`float` where the server's commands return integers — one shared dispatch
answered a session's temperature request with another session object, which
NX-Fetch printed as "8 C". And the **device code's high byte** picks the
sensor (`0x41……` SoC, `0x43……` PCB), not its low byte: NX-Fetch asks for
`0x41000002` and calls the result "CPU".

`set:sys`'s `GetFirmwareVersion`/`GetFirmwareVersion2` (commands 3 and 4) are
**not cosmetic**. libnx's `__appInit` seeds `hosversionGet()` from them, and
everything version-gated branches on that — which `acc` commands exist, which
`ts` interface carries the measurement, which audio-renderer revision gets
negotiated. Answering them with the generic empty success left each caller
reading its own uninitialized buffer as the system version: NX-Fetch reported
"Horizon OS 115.119.105", which is the ASCII of `switch-wasm user` — the `acc`
uid this emulator had left in that same buffer earlier. The struct goes back
through a pointer buffer, so it needs `ipc_output_buffer` and a non-zero
`QueryPointerBufferSize`, like `acc`'s.

`csrng`, `spl:`, `pdm:qry`, `pm:*` and `pcv`/`clkrst` are the rest of what a
system-info or save-management homebrew opens.

- **csrng** fills the caller's buffer from `Cpu::next_random_u64` (splitmix64,
  seeded from the emulated clock). It is **not** a CSPRNG — there is no OS
  entropy under `wasm32-unknown-unknown` and no security processor here — but
  the generic reply left the buffer untouched, which is a "random" value that
  is whatever was on the caller's stack, non-random *and* undetectably so.
- **spl:** answers `GetConfig`: an original (Icosa) retail console, not in
  debug mode. Note what a real guest actually asks it for first — NX-Fetch
  wants **Atmosphère's extensions at 65000+** (CFW API version, emummc type),
  and zero there reads as "no custom firmware", which is true.
- **pdm:qry** reports a console **nothing has ever been played on**: empty
  event lists, an empty available range, zeroed statistics. That is a state a
  factory-fresh console has, which is what makes it truthful rather than a
  placeholder.
- **pm:\*** are four different interfaces on four service names, so
  `pm_request` dispatches on the name. What they can answer honestly is
  identity, and `pm`'s process id must equal `svcGetProcessId`'s — the same
  question through a different door. `pm:info`'s program id defaults to the
  Album applet's, which is what hbmenu-launched homebrew runs as on hardware;
  a loader that knows a real title id sets it with `Cpu::set_program_id`.
- **pcv/clkrst** are the same clock manager either side of 8.0.0. Their
  numbering differs *by an offset*: a `clkrst` device code is
  `0x40000000 + module + 1` over the `PcvModule` value `pcv` takes directly,
  so reading the code's low bits as the module is off by one — NX-Fetch asks
  for `0x40000001`/`0x40000002`/`0x40000039` and labels them CPU, GPU and
  Memory. Rates are the **handheld** ones, and a rate a guest sets reads back.

`sfdnsres` (the DNS resolver, `sfdnsres_request`) is the other half of the
socket stack, opened alongside `bsd:u` by `socketInitialize`. Nothing resolves:
`EAI_NONAME` for the `getaddrinfo` family, `HOST_NOT_FOUND` for the
`gethostbyname` one — the *definitive* failure, not the try-again one, for
`bsd`'s reason. A numeric address string fails too, which real hardware would
resolve without any DNS: serializing an `addrinfo` into Horizon's packed form
is guesswork nothing here can check against a real console, and the connect
that follows is refused anyway, so the lookup fails where the guest can act on
it. The failure goes in the **first** word of `SfdnsresRequestResults`, which
is what makes it robust to the field order; the error *strings* are answered
properly, so a guest that prints why a lookup failed gets a sentence.

**One console, one answer.** These services describe the same machine from
different angles, and a guest reads several of them: `am`'s operation mode,
`apm`'s performance mode and `clkrst`'s rates have to describe one console, or
NX-Fetch prints "Docked" beside a 720p framebuffer clocked for handheld. That
mismatch was real — `GetOperationMode` answered 1 (Console) under a comment
saying Handheld, and `AppletOperationMode_Handheld` is 0.

**A stub that answers by fabrication says so.** The generic reply for a service
with no dedicated handler is load-bearing for homebrew that only checks the
Result, so it still succeeds — but `Cpu::warn_no_implementation` records
`[ipc] no implementation: <service> cmd=<n>` once per pair, and that list *is*
the inventory of what a given guest is asking for and not getting.

**An unimplemented `am` command must not answer with a bare success.**
Everything `am` hands back is a live kernel object or a piece of applet state
the caller acts on, so a fabricated success is a wrong answer the guest
believes rather than a neutral placeholder: the old catch-all answered
`IApplicationFunctions::GetGpuErrorDetectedSystemEvent` (command 130) with
success and *no copy handle*, and `nnSdk`'s system worker spent the rest of the
boot waiting on handle 0. `Cpu::unimplemented_command` reports `cmif`'s
`UnknownCommandId` (module 10, description 221 — `0x1ba0a`) and warns once per
`(interface, command)` on stderr, which is how you find the next command to
implement. A bare success is still the *right* answer for a genuine
setter/notifier whose whole reply is a Result — but those are listed by command
id, not caught by a `_` arm.

Both callers reach the same `Cpu::applet_request`, by two different routes:
`libnx` converts `appletOE` to a domain and addresses each sub-interface by
object id, while `nnSdk` never converts and gets a **session handle per
interface**, so the `am:*` names are also listed in `svc.rs`'s dispatch (the
same split as `fsp-srv-fs` and `time:system-clock`).

A **Close** request (message type 2) carries no command id. Dispatching one on
whatever command id is left in the TLS buffer runs a real command — closing an
`fsp-srv` session was landing on `CreateFile` and adding an empty file to the SD
card — so `svc.rs` answers type 2 before dispatch and calls
`Cpu::forget_handle`.

## Where a CMIF header lands (`cpu/ipc.rs`)

`Cpu::ipc_cmif_header_offset` finds the "SFCI" header by walking the request's
descriptors (`ipc_reply_start`), checking the domain offset too, and only then
scanning the message buffer. A request with buffer descriptors pushes the header
well past the start — nvdrv's `KICKOFF_PB` puts it at 0x40 — and a fixed scan of
the first 0x40 bytes reported "no command id", so **the GPU submit was answered
as an unknown command with a generic success**: no pushbuffer ever ran, the frame
fence never signalled, and hbmenu spun in `dkFenceWait` forever.

## IPC payload offsets (`cpu/ipc.rs`)

`Cpu::ipc_request_data` locates a CMIF request's raw payload by finding the
"SFCI" magic rather than adding a fixed offset to the data area: a domain
request carries a `CmifDomainInHeader` in front of the `CmifInHeader`, so its
payload sits 0x20 rather than 0x10 bytes in. libnx converts the `fsp-srv`
session to a domain, so assuming 0x10 made `fsFileRead` read its offset and
size out of the header — every read asked for 0 bytes at offset 0, and
`romfsMountSelf` failed with `LibnxError_IoError`.

## Emulated SD card (`vfs.rs`)

`fsp-srv` is backed by a real path-addressed tree, so `GetEntryType`,
`OpenDirectory`, `fsDirRead`, `OpenFile` and `Read` all agree with each other
and a missing path returns `FsError_PathNotFound`. Returning a fixed listing
for every path made menus recurse forever. The running NRO is published at
`nro::HOMEBREW_NRO_PATH` and advertised as `argv[0]` through the homebrew ABI
environment block, which is how libnx's `romfsMountSelf` finds the RomFS
appended to it.

## Controller input (`cpu/mod.rs`, `web/assets/main.js`)

`Cpu::set_gamepad_state` publishes the host pad two ways: the memory-mapped
`INPUT_ADDR` register, and libnx's `HidSharedMemory` layout that `padUpdate`
reads. The offsets are not guessable — they were taken from libnx's
`services/hid.h` by compiling the struct on the host: `npad` at **0x9A00**, one
`HidNpadSharedMemoryEntry` every **0x5000**, `full_key_lifo` at **+0x28** and
`handheld_lifo` at **+0x378** inside `HidNpadInternalState`, `device_type` at
**+0x4188**; each LIFO is a 0x20-byte header (unused/buffer_count/tail/count)
then 0x30-byte entries of `{sampling_number, HidNpadCommonState}`. A single
entry at index 0 with `tail = 0`, `count = 1` and `HidNpadAttribute_IsConnected`
is all `hidGetNpadStates*` needs.

The pad is published in **two slots**: player 1 as a Pro Controller and slot 8
as the handheld controller, because homebrew polls whichever it was built to
expect and `padUpdate` merges them. Slot 8 is `HidNpadIdType_Handheld`.

The button bits are Horizon's order (A=1<<0 … StickR=1<<5, L=1<<6, ZL=1<<8,
Plus=1<<10, d-pad from 1<<12), not the old `KEY_*` order, and the
`HidNpadButton_StickL*`/`StickR*` pseudo-buttons are derived from the analog
values — `HidNpadButton_AnyUp` and friends are what menus navigate with. Stick Y
is positive **up**, the opposite of the browser Gamepad API.

**The run slice is the input sampling period.** The worker is single threaded,
so a `set_input` posted mid-slice sits in its queue until `switch_run` returns —
at ~23M steps/s the old 5,000,000-step slice was a 240ms block before the guest
could see anything. Slice size costs the interpreter nothing (`Cpu::run` is a
bare loop; throughput is flat from 100k to 5M steps), only the frontend's
postMessage round trips, which is why `RUN_SLICE` is 1,000,000 and the
debug-panel refreshes run on their own `HOUSEKEEPING_EVERY` cadence.

The worker latches a press until the **frame counter has advanced twice**, not
for a fixed number of slices. The guest polls hid once per iteration of its own
loop, and one of those spans many slices (hbmenu: ~25M instructions, five
slices), so holding a tap for one slice dropped most taps outright — and the
poll sits *inside* the loop, so only a complete present-to-present interval is
guaranteed to contain one. Only bits the guest may not have seen are latched: a
key the host still reports as down is published from `heldButtons` alone, so
releasing it lands on the next slice instead of a slice later (that extra slice
of stickiness made one d-pad tap step two menu entries). `MAX_LATCH_SLICES` is
the escape hatch for a program that has stopped presenting.

## Touchscreen (`Cpu::set_touch_state`)

`HidSharedMemory.touch_screen` is at **0x400**, straight after the debug pad's
0x400, and holds a `HidTouchScreenLifo`: the same 0x20-byte header as the npad
LIFOs, then storage entries of `{u64 sampling_number, HidTouchScreenState}`.
That state is `{u64 sampling_number, s32 count, u32 reserved, HidTouchState
touches[16]}`, and a `HidTouchState` is **0x28** bytes — `delta_time`,
`attributes`, `finger_id` at **+0x0C**, `x` at **+0x10**, `y` at **+0x14**, the
two diameters, `rotation_angle`. One entry at index 0 with `tail = 0`,
`count = 1`, exactly as for the pad.

- **hid reports touches in the console's own 1280x720 digitizer space**
  (`TOUCH_SCREEN_WIDTH`/`HEIGHT`), *not* in whatever resolution the guest is
  presenting at, so `main.js` scales the canvas onto that. `#screen` is
  `object-fit: contain`, so a tap must be mapped through the **contained rect**
  — going by the element's bounding box offsets every tap by the letterbox bars.
- Touch is handheld-only on real hardware, and `am`'s `GetOperationMode` already
  answers `AppletOperationMode_Handheld`, so there is no docked case to
  suppress. `ActivateTouchScreen` (`hid:server` cmd 11) was already a no-op
  setter.
- **A lift is a published state with `count = 0`, not silence.** Nothing is
  buffered, so the frontend keeps sending while a finger is down — which is also
  what makes a touch that began before the guest mapped hid's shared memory show
  up as soon as it has. Vacated slots are zeroed so a reader that scans the
  array instead of trusting `count` finds no ghost contact.
- Taps ride the same latch as button presses, for the same reason, and
  `finger_id` is a slot claimed for the life of the pointer so a title can
  follow a drag.

## The shared system font (`cpu/ipc.rs`, `web/assets/font.ttf`)

Homebrew does not ship fonts: it asks `pl:u` for the console's shared fonts and
renders them with its own FreeType (`plGetSharedFont` → `FT_New_Memory_Face`).
`Cpu::set_shared_font` takes a TrueType/OpenType file, `stub_pl` reports it as
every shared font type at offset 0, and mapping pl's shared memory (recognised
by its size, `PL_SHMEM_SIZE`) fills the region with it. Reporting an empty font
set — which is what this did before — means **no text renders at all**.

`tools/make_font.py` builds the shipped subset. It strips the TrueType hinting
programs for two reasons: hinted glyphs come out collapsed horizontally under
the interpreter (see PROGRESS.md — a real bug, not a font problem), and running
the hinting bytecode costs about **8x** more emulated instructions per frame
(hbmenu's first frame: 46M steps without, 350M with). It also points Nintendo's
private-use button codepoints (0xE0E0…, 0xE0A0…) at the matching letters so
on-screen button hints read as "A Launch" instead of arbitrary glyphs.

## Thread stacks live in the stack region (`cpu/svc.rs`)

`threadCreate` allocates a stack on the heap, asks `virtmemFindStack` for a free
range in the region `svcGetInfo` reports, and `svcMapMemory`s the stack there —
from then on it uses only that mirror. Two things therefore have to be true:

- The reported stack region (`GUEST_STACK_REGION_ADDR`) must have room. It used
  to be the 1 MiB the *main* stack already occupied, so every lookup failed,
  `virtmemFindStack` returned NULL, and each thread's mirror was address 0.
- `svcMapMemory` must really back the destination. While it was a no-op success,
  the next lookup saw the range as still free and handed out the same address,
  so **two threads shared one stack** and silently overwrote each other's
  frames; the crash only surfaced when input woke the parked thread and it
  returned through a clobbered link register.

Page storage is not shareable, so the alias is a copy (`Memory::copy_range`),
copied back by `svcUnmapMemory`, which then frees the pages. The guest only ever
touches one side of such an alias, so it cannot tell the difference.

## Performance: measure the wasm build, not just the host

Three tools, and they do not agree — measure the one you care about:

- `cargo run --release -p switch-core --example bench` — per-instruction-class
  throughput on the host. `b .` is the floor (one instruction, first check in the
  decoder), so the gap between it and a class is that class's decode+execute cost.
- `cargo run --release -p switch-core --example hotspots -- <nro>` — every
  instruction of one steady-state frame, bucketed by guest address and by encoding
  byte. This is how you find out that 72% of an hbmenu frame is hbmenu's own
  software gradient fill and only ~10% is the emulator's GPU work.
- `node tools/wasm_bench.mjs <nro>` — the build the browser runs, reporting the
  steady-frame cost in fps. `node --cpu-prof` on it produces a profile whose
  samples name the wasm functions; it is the only profiler available for that
  build.

What the numbers taught us:

- **Dispatch order matters on the host, inlining matters in wasm.** Routing by
  the A64 top-level group (bits 28:25) before running a group's decoder was worth
  ~25% natively and *nothing* in wasm. Splitting `Memory`'s accessors into an
  `#[inline(always)]` in-page fast path plus `#[cold]` page-straddling and
  unmapped fallbacks was worth ~15% in wasm — V8 had been emitting `read_u32` as a
  real call on the path of every instruction fetch.
- The interpreter's floor is ~9ns per instruction natively and ~20ns in wasm, so a
  frame of ~30M guest instructions cannot go below about a second in the browser
  no matter how the decoder is arranged. Getting past that needs a decoded-block
  cache (see PROGRESS.md), not more guard reordering.
- Anything on the per-instruction path is worth checking for accidental cost: the
  GPU's `read_pixel`/`write_pixel` used to translate a GPU address **per byte**,
  which is four `BTreeMap` searches per pixel and millions per blit.

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
