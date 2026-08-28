# AGENTS.md

Browser Switch emulator: an A64 interpreter with a block-translating JIT, a
software GPU with an optional WebGPU backend, and the container stack
(PFS0/NSP, XCI, NCA, NSO/NRO/ELF, RomFS) needed to boot retail titles and
system applets. Compiled to WASM; frontend is TypeScript on Vite.

PROGRESS.md is the long-form log of what was tried and why. `docs/` holds the
reference material that is looked up rather than read. This file is the
standing state.

## Commands

- `make all` — `test` then `assets`.
- `make test` — `cargo test` over all three crates. 908 tests.
- `make wasm` — release wasm build `--features gpu`, then `wasm-bindgen
  --target web`. Needs `rustup target add wasm32-unknown-unknown` and a
  `wasm-bindgen-cli` matching the `Cargo.lock` version.
- `make assets` — `make wasm` + `vite build` → `dist/`. The only frontend
  target, because the core is an *input* to the frontend build.
- `bun run dev` (:8000) / `bun run preview` — both need `make wasm` once.
- `bun run typecheck` — the only thing that type-checks; Vite never does.
- `python3 tools/difftest.py [--scalar]` — differential-test the decode
  against real ARM under `qemu-aarch64`. **Add an instruction there before
  hand-deriving expected values.** Needs `clang` + `lld` + `qemu-aarch64`.
  **Two separate harnesses**: bare, it runs the SIMD table only; `--scalar`
  runs the integer one. A change to the integer ALU that reports "matches
  qemu" from the bare run has not been tested at all.
- `cargo run --release -p switch-core --example jit_difftest -- <nro>` — both
  engines side by side, with every state difference between them.

## Crates

- `switch-core` — the emulator, **zero dependencies**. `cpu` (`mod`/`alu`/
  `bits`/`fp`/`jit`/`loadstore`/`simd`/`svc`/`system`, plus `ipc` and one
  module per service domain — see below), `gpu`, `display`, `mem`, `vfs`,
  `source`, `crypto`/`keys`/`ticket`, `nsp`/`xci`/`nca`/`romfs`/`npdm`/`nso`/`nro`/
  `elf`/`lz4`, `control`, `disasm`, `error`.
- `switch-gpu` — a `wgpu` backend behind `gpu::renderer::Renderer`. Separate
  because `wgpu` brings hundreds of crates and the core has none.
- `switch-wasm` — browser bindings, `cdylib`. Buffers cross via linear memory
  (`switch_alloc`/`switch_free`); a handle indexes a global session table.
  JSON is hand-rolled (`json_escape`/`write_into`) — **don't add serde**.

**Services are one module per domain.** `ipc.rs` is the marshalling layer —
descriptor walks, domain/control messages, handle bookkeeping,
`write_ipc_reply` — plus `sm:` and the services whose whole implementation is
an answer or two (`csrng`, `spl`, `pm`, `btm`, `nfc`). Everything else lives
beside it: `acc`, `am`, `audout`, `audren`, `erpt`, `fs`, `hid`, `ldr`, `log`,
`mii`, `net`, `ns`, `nv`, `online`, `pl`, `power`, `settings`, `time`, `vi`.
`svc.rs` dispatches to them by session name, and each owns its own constants,
its state structs and its tests. Shared request builders for those tests are
`ipc::testing`.

- **Decode the header once, through `Cpu::ipc_header`.** Seven walks used to
  re-derive the same counts with their own shifts, which is how
  `ipc_static_buffers` came to skip a special header's pid but not the copy and
  move handles behind it — reading a path out of the handle words.
- **A map-alias descriptor's address is its low word.** Guest memory is
  `u32`-indexed, so the packed word's address bits all land above bit 32 and are
  truncated straight back off. `Cpu::ipc_map_descriptor` is the one decode.
- **A caller marshals a buffer one of four ways** (map-alias send/receive,
  send-static, receive-static) and a service that reads only the form it expects
  reads nothing at all. Reach for `ipc_input_buffer`/`ipc_output_buffer` unless
  you know which form you are being sent.

**The shipped module is a wasm-bindgen module**: `make wasm` always builds
`--features gpu`, so the worker `import`s generated glue rather than calling
`WebAssembly.instantiateStreaming`. That also moves the host read into
wasm-bindgen's world (`raw_module = "@host/files"`, aliased in
`vite.config.ts`), so the module declares no `env` import at all. Without the
feature it declares exactly one, `env.host_read`. No WebGPU device → the
software rasterizer takes the frame.

## Frontend

`web/` is source, `dist/` is generated output, and nothing committed sits
beside a build artifact. `web/public/` is the one verbatim-copied directory
(the social card, the font licence). Everything else is content-hashed, which
is what makes a rebuilt core a new URL — so assets are named through the
bundler (`import fontUrl from '../font.ttf?url'`), never as literal paths.
`@core` aliases cargo's release dir. No source maps.

- `web/worker/` — `wasm.ts` (exports + buffer plumbing), `hostfiles.ts` (host
  read, 1 MiB × 32 LRU chunk cache), `latch.ts`, `commands.ts`, `index.ts`.
- `web/main/` — `rpc.ts` plus one module per part of the page; `index.ts` is
  the composition root and owns Reset.
- `web/shared/protocol.ts` — the `Commands` interface both sides are checked
  against, so a drifted signature is a build error.
- `runloop.ts`: `RUN_SLICE` 1,000,000, `TRACE_SLICE` 5000,
  `HOUSEKEEPING_EVERY` 8.

Traps:

- **`base: './'` in `vite.config.ts` is load-bearing.** The site is published
  under a path; Vite's default `'/'` 404s anywhere but a host's root, and only
  once deployed. Test by serving `dist/` from a subdirectory.
- **The worker is a module worker and both halves must say so** — `rpc.ts`'s
  `{ type: 'module' }` and `worker: { format: 'es' }`. Naming it by its source
  (`new URL('../worker/index.ts', import.meta.url)`) is what lets the bundler
  rewrite the path. `importScripts` is unavailable.
- **A `switch_*` call on a freed handle traps the module** — a table miss is a
  Rust panic, i.e. `unreachable`. Reset frees without waiting for the slice in
  flight, so three things guard it: `worker/index.ts` refuses all but
  `new`/`set_battery` with no session, `main/index.ts` sets `setSession(-1)`
  before posting the free, and `runloop.run` bails once `abortRun` is called.
- The `.wasm` is an input to the frontend build; building `switch-wasm` alone
  does not update the site. `.forgejo/workflows/pages.yml` deploys `dist/`.

## Containers are never staged in memory

A retail `.nsp` runs to gigabytes; wasm32 memory caps at 4 GiB and Rust refuses
any single allocation over 2 GiB. `switch_alloc` returns null there (it used to
trap) and `wasm.ts`'s `alloc` throws.

`source::ByteSource` is a `u64`-addressed random-access range and the pieces
compose: `HostSource` → `Window` (the NCA) → `SectionSource` (AES-CTR is
seekable, so a range costs exactly that range) → `Window` (the RomFS past the
IVFC levels).

- The host read is `(file, offset, ptr, len)` and **must answer
  synchronously** — RomFS ranges are asked for inside `switch_run`, where
  there is nowhere to await. That means `FileReaderSync`, which exists only in
  a worker: the second reason the emulator lives in one. **File 0 is the open
  container**; the rest are system data archives (`switch_add_archive`).
- The ExeFS is read in full and hash-verified. The RomFS is not:
  `Cpu::set_romfs_source` takes the decrypting view and `IStorage::Read` copies
  through a 64 KiB staging buffer.
- **Offsets stay `u64` end to end.** `as usize` on wasm32 truncates past 4 GiB
  *and* makes the bounds check that should catch it pass.

## Guest address space (`cpu/mod.rs`)

Guest memory is `u32`-indexed, so the whole space is the low 4 GiB.
`Cpu::bootstrap` soft-maps 0..`GUEST_SPACE_END`: unwritten pages read as zeros
and allocate on first write, which is why a region can be reported far larger
than the `MAX_MAPPED_BYTES` (3.125 GiB) cap for free.

```text
0x0010_0000  ENV_BLOCK_ADDR (nro.rs)  homebrew ABI environment block
0x0800_0000  ASLR region, 496 MiB     svcGetInfo 12/13
0x1800_0000  GUEST_STACK_REGION_ADDR  thread-stack mirrors, 128 MiB (14/15)
0x2000_0000  SELF_RETURN_TRAMPOLINE   +0x100 THREAD_EXIT_TRAMPOLINE
0x2010_0000  main thread TLS          children from THREAD_TLS_BASE, page each
0x2800_0000  STACK_BASE               main stack 1 MiB, SP at STACK_TOP
0x2900_0000  RO_MODULE_REGION_ADDR    ldr:ro maps run-time NROs, 112 MiB
0x3000_0000  heap / alias             per MemoryLayout, below
0xFA00_0000  SHARED_BUFFER_ADDR       system shared buffer, ~59 MiB reserved
0xFE00_0000  FB_BASE (lib.rs)         demo framebuffer, 640x360 RGBA
0xFE10_0000  INPUT_ADDR               memory-mapped input block
0xFF00_0000  GUEST_SPACE_END          above this a read faults
```

**Every region `svcGetInfo` reports must be representable here.** Horizon's
real bases (alias 0x10_0000_0000) truncate to 0 when `nnSdk` asks
`svcMapPhysicalMemory` to back them.

```text
MemoryLayout::PLAIN                 MemoryLayout::VIRTUAL_ADDRESS
0x3000_0000  heap,  3.125 GiB       0x3000_0000  heap,  128 MiB
0xF800_0000  alias, 32 MiB          0x3800_0000  alias, 3.03 GiB
total memory 3.125 GiB              total memory 896 MiB
system resource 0                   system resource 16 MiB
```

- **InfoType 16 picks the layout.** `VammManager::IsVirtualAddressMemoryEnabled`
  is this answering non-zero and nothing else, and the figure is the title's own
  NPDM `system_resource_size`. Both kinds of title are real, so each is
  answered what its manifest says (`npdm.rs`).
- **The syscall follows the layout.** VAMM titles use `svcMapPhysicalMemory`
  (0x2c) and never issue 0x01; plain titles issue exactly one `svcSetHeapSize`
  (0x01) for the whole reported total and never touch 0x2c. So each layout
  spends its address space on the region its own titles grow into. 0x2c only
  validates the range; 0x2d does the real unmapping work.
- **`nn::init` asks for the whole reported total**, so a region smaller than
  that figure is one the guest overruns. 0x01 refuses a heap larger than its
  region, and
  `the_guest_regions_are_disjoint_and_big_enough_for_what_they_promise` holds
  the layout together.
- Under VAMM the alias region must satisfy `size >= VAMM_ARENA_SIZE
  (0x3FE0_0000) + VAMM_TOTAL_MEMORY_SIZE + the title's own`. Falling short
  fails quietly: allocators here return null, nothing checks it, and the crash
  lands tens of millions of instructions away.
- **The reported total is what a title believes about the console.** Titles
  size pools from it against numbers baked into their own code, and those
  numbers are not negotiable: Persona 5 Royal's three pools are 2.98 GiB of
  `.data` constants, so anything under a 3 GiB heap aborts it in
  `RsdxDevice11CoreCommonUtil.cpp`. The alias region keeps only enough to be a
  region, because nothing on this layout ever maps into it.
- **InfoType 21/22** size the application heap — their difference goes straight
  to `nn::mem::StandardAllocator::Initialize`, which asserts under 16 KiB.
- **InfoType 11 (RandomEntropy) must not be zero** — real `sdk` startup
  `svcBreak`s on an all-zero pool.
- **InfoType 0/1** come from the NPDM `ThreadInfo` capability: cores 0..2
  (`0b111`), priorities 28..59. Zero makes
  `nn::os::RegisterSystemWorkerHandler` assert on an empty mask.

## Booting

**Homebrew** (`Cpu::boot_homebrew`): run crt0 to the `bl` at entry+0xc0, seed
the env block and `ThreadVars` (TLS+0x1E0, magic `0x21545624`, `_REENT` at
`0x1FF1_0000`), run `DT_INIT_ARRAY`, resume at that `bl`. **Static constructors
only run through this path.** The constructor pass zeroes registers, so all
three `__libnx_init` arguments are re-seeded before resuming — including
`saved_lr` = `SELF_RETURN_TRAMPOLINE`, without which `__nx_exit` branches to
NULL and a clean exit looks like a crash.

**Retail** (`Cpu::boot_retail_program`): `rtld` reads both entry registers
literally. **X0** is the launch argument (0 for a normal launch). **X1** is
`MAIN_THREAD_HANDLE` (1) — `nnSdk` stores it at ThreadType+0x1B0 and compares
every `SdkMutex` lock word against it, so X1 = 0 makes an *unlocked* mutex read
as "owned by me" and the first `Lock` fires its recursive-lock assertion.

## How long a run has to be

One emulated instruction ≈ one cycle of the 1.02 GHz CPU `svcGetSystemTick` is
scaled from, so **a billion steps is about a second of console time**. A retail
title spends seconds of console time before its first frame, an IL2CPP game
longer. Before concluding a title does not render, check the run was long
enough, and prefer `SHOT=<file.ppm>` over reading `frames presented: 0` off a
budget that was never going to get there. Per-title step counts live in
PROGRESS.md; keep them there, not here.

**The clock and the step count are not the same number.** `Cpu::cycles` is the
clock, and `reschedule` idles it forward to the earliest sleeper whenever every
thread is blocked — the console's own idle, covering cycles nobody executed.
`Cpu::steps` counts retired instructions and the idle never touches it; both
engines go through `Cpu::retire` so they cannot drift. The page's *Steps*
readout is `switch_get_steps`: a figure that leaps while the guest is stopped
is useless as a loading screen's sign of life.

## Block translation (`cpu/jit/`)

`ir` is what a block is made of, `cache` is which blocks exist and when a
guest store takes one away, `decode` builds them and `exec` runs them.

First visit to an address translates forward into `Op`s — operands extracted,
immediates decoded, and every field the interpreter re-reads per execution
(load width and direction, register-offset extension, add-vs-subtract, which
system register, which floating-point form) resolved to the one thing the
instruction does. Register 31 is resolved too: the file has 34 slots so its
three meanings (`XZR` read, `XZR` write, `SP`) are an index, not a branch.
Translation runs *through* `b.cond`/`cbz`/`tbz`, which become `Exit`s checked
on the way past, so only an always-taken branch ends a block; a `cmp` feeding
the branch after it is then folded into that `Exit` (`fuse_compares`). It
generates no code — every op calls the same helper the interpreter would, so
the two engines are the same computation and anything untranslated falls back.
`SWITCH_NO_JIT=1` for host tools, the debug panel's *Translation* section in
the browser.

**That shared helper is literal, not aspirational.** An instruction's body is
written once, keyed on register-file *slots*, and lives with its semantics
rather than with either engine: the ALU, bitfield, multiply and conditional
forms in `alu.rs`, `Acc`/`Ext`/`PairKind`/`Wb` with `access`/`indexed`/`pair`
in `loadstore.rs`, and `SysReg`/`SysOp` in `system.rs`. The interpreter
resolves slots from the encoding per execution, the translator resolves them
once into an `Op`, and both then call the same function. They used to be two
transcriptions of each other — which is what `jit_difftest` was watching for,
and what a `movk w0` that forgot to zero bits 63:32 hid in *both* engines at
once until `tools/difftest.py --scalar` was pointed at it.

**A block must always retire at least one instruction.** `run_jit` advances by
what `exec_block` reports, so a block returning `Ok(0)` spins forever. A fused
pair at index 0 under a one-instruction budget did exactly that; the budget now
runs the compare's half alone and stops on the branch, a valid entry point.
`jit_test` catches this by hanging, not by failing.

Emitting wasm per block is blocked by the memory model, not the browser: a
generated module can only address its own linear memory, and guest memory is a
page table with soft regions, read-only ranges and watchpoints. Flattening the
address space behind a base-plus-bounds check has to come first.

## Guest threads (`cpu/mod.rs`, `cpu/svc.rs`)

**Preemptive, on a `TIME_SLICE` of 20,000 instructions**, plus the blocking
syscalls. Cooperative switching completes the same handshakes but does not
*share* the CPU — one applet's audio thread measured 99.9% of all instructions
executed. Between instructions is safe (all state is in `ThreadContext`), and
`yield_thread` is a no-op when nothing else can run.

- **Mutexes and condvars are real.** Horizon keeps the lock word in guest
  memory and libnx re-reads it after every arbitration, so
  `svcArbitrateUnlock` must actually hand ownership over and
  `svcWaitProcessWideKeyAtomic` must release the mutex. `nn::os` skips the
  syscall entirely when a condvar's word is zero, so the *kernel* has to write
  `CONDVAR_HAS_WAITERS`.
- **The exclusive monitor is real, and preemption is why.** `LDXR`/`LDXP` set
  `Cpu::exclusive`, `STXR`/`STXP` require and consume it, a context switch
  clears it.
- **Thread stacks live in the stack region.** `GUEST_STACK_REGION_ADDR` must
  have room or `virtmemFindStack` hands back address 0, and `svcMapMemory` must
  really back the destination or two threads share one stack. Pages are not
  shareable, so the alias is a copy (`Memory::copy_range`), copied back by
  `svcUnmapMemory`. **Nothing of the emulator's own may be inside it**: a guest
  picks the address itself and asks only `svcQueryMemory`, so the region stops
  at `SELF_RETURN_TRAMPOLINE` and the trampolines and TLS blocks live above it.
- **A blocking wait parks, it does not re-ask.** `ThreadState::WaitEvent` holds
  the thread with its PC on the `svc`; `signal_event` wakes every parked waiter
  on an event's *transition* (re-firing a signalled event wakes nobody, or
  `audio_tick` would wake them on every wait in the process), and the deadline
  is the display tick. Re-asking on each slice is what it replaced, and it cost
  the Home Menu 131 of the 170M steps it took to reach frame 10.
- The SVC path retires the instruction *before* dispatching, so a syscall that
  switches threads can install the incoming PC.
- `svcQueryMemory` finds bounds through `Memory::state_run`, which skips 2 MiB
  blocks with no backed page — walking page by page is O(address space).
- **A wait on no handles is not answered.** `MultiWaitImpl::WaitAny` maps any
  answer onto a holder from its own list, and an empty list has none — either
  answer jumps the thread to 0. So it rewinds onto the `svc` and yields unless
  nothing else can run.
- **A satisfied wait reports X1 = 0** — X1 is the *index* of the handle that
  signalled, and callers index their own waiter arrays by it.

## Diagnostics

`wasm32-unknown-unknown` has no WASI: `eprintln!` goes nowhere and
`std::env::var` always fails, so every `TRACE_*` is host-CLI-only. Anything
that must reach a browser user goes through `Cpu::diagnostic`, which writes
stderr *and* the trace buffer the page drains every slice.

`TRACE_SVC` (every syscall bar the three hot ones, plus each `svcGetInfo`
answer), `TRACE_IPC`, `TRACE_WAIT` (events as handed out, and what each wait is
waiting on), `TRACE_FONT`, `TRACE_AUDIO`, `TRACE_ERPT` (every context record a
guest journals, by category), `TRACE_MAP`, `TRACE_REGS`, and the
GPU set — `TRACE_GPU`, `TRACE_NV`, `TRACE_DRAW`, `TRACE_PIPELINE`,
`TRACE_SHADER`, `TRACE_CFG`, `TRACE_WGSL`, `TRACE_UPLOAD`. `Cpu::backtrace`
walks the guest X29 frame chain.

The `wgpu` backend's own flags are not traces but switches: `GPU_ONLY=<i>` or
`<a>..<b>` renders only those draws of each frame on the device,
`GPU_DEVICE_MSAA=1` lets the device do the multisampling, `GPU_INTERLEAVE=1`
keeps fallback draws inside a device frame where readbacks land late,
`GPU_DEFER_READBACKS=1` makes them land late on purpose, `GPU_TIMES=1` says
where a draw's time went, and `GPU_DUMP_WGSL=<dir>` writes each draw's two
modules out.

## IPC (`cpu/ipc.rs`, `cpu/svc.rs`)

**Two dialects.** libnx ignores the reply's `type` and raw-data size and
converts `fsp-srv` to a **domain** (sub-interfaces are object ids on one
handle). libtransistor validates both and never converts, so a sub-interface
must come back as a **session handle in a move handle**;
`Cpu::reply_with_interface` picks the shape per request.

**Two encodings of every message kind**: plain (`Request` 4, `Control` 5) and
with-context (6, 7), which prefixes a 16-byte tracing context. libnx sends
plain, **`nnSdk` sends the context form for everything** — test with
`Cpu::ipc_is_control_request`, never `type == 5`.

A **Close** (type 2) carries no command id; `svc.rs` answers it before dispatch
and calls `Cpu::forget_handle`, or the leftover id runs a real command.

### The bug class: success with an unfilled out parameter

A reply is written *over* the request in the same TLS buffer and declares four
words of padding past the SFCO header, so a bare success passes every length
check and hands back stale *request* bytes. `Cpu::write_ipc_reply` clears the
section it declares, so unimplemented out parameters read as 0 — still wrong,
but the same wrong every time. When a command has an out parameter, implement
it; when its width is unclear, reply with a zeroed block **wider** than needed.
A reply may be longer than expected, never shorter.

Same trap in the handle slots: `nnSdk` reads an out-object off a plain session
as a move handle, and a reply with none parses as 0, silently skips the proxy,
returns success, and calls through null. `Cpu::reply_with_fabricated_object`
carries **a sub-session in the move slot and an event in the copy slot**,
allocated once per `(session, command)` — nothing here can tell which the
caller wanted.

**An unimplemented `am` command must not answer with a bare success** —
everything `am` returns is a live object or applet state.
`Cpu::unimplemented_command` reports `cmif`'s `UnknownCommandId` (`0x1ba0a`).
A bare success is still right for a genuine setter/notifier, but those are
listed by id, never caught by `_`. `Cpu::warn_no_implementation` logs `[ipc] no
implementation` once per pair, and that list *is* the inventory of what a guest
asks for and does not get.

### Events

`Cpu::alloc_event` records one in `Cpu::events`, and it must reach the guest
through the **copy** list — a move handle transfers ownership, a copy handle
duplicates one the server keeps, and an event in the move slot reads back as
**0**. **An event handed out twice must be the same event** (`Cpu::kept_event`,
keyed per `(purpose, object)`), or the caller waits on something the service
never signals.

`svcWaitSynchronization`: a handle not in `Cpu::events` counts as ready (do not
"fix" without checking homebrew); a **poll** (timeout 0) on unfired events
reports Timeout; a **blocking** wait with nothing signalled reports the first
handle ready, a deliberate lie — see `WaitAny`. `vsync_event` fires on a period
as well as on presents.

### Where headers and payloads land

`Cpu::ipc_cmif_header_offset` finds "SFCI" by walking the request's descriptors
first — buffer descriptors push it well past the start (nvdrv's `KICKOFF_PB`
puts it at 0x40). `Cpu::ipc_request_data` does the same for the payload: a
domain request carries a `CmifDomainInHeader` in front, so its payload sits
0x20 rather than 0x10 in.

Buffers: `ipc_input_buffer`/`ipc_output_buffer` try map-alias then pointer,
`ipc_send_buffer` is the map-alias send side, `ipc_recv_static_buffers` handles
the kind that sits *after* the raw data (the unaligned data offset plus
`num_data_words`).

- **`QueryPointerBufferSize` must be non-zero** wherever pointer buffers are
  used (`hid`, `acc`, `set:sys`) — `nnSdk` checks the negotiated size *before*
  sending and gives up when the server claims no room.
- **`CloneCurrentObject` (control 2, 4 for Ex) returns a new session handle as
  a move handle.** Answered centrally in `svc.rs`; the clone reaches the same
  interface and inherits the same domain objects. `nnSdk` clones `fsp-srv`
  before mounting anything.
- **Mount names live in the guest** — `nn::fs`'s `MountTable` is client-side,
  so `rom:` needs nothing beyond `OpenDataStorageByCurrentProcess` (200) and a
  storage that reads correctly.
- **`IStorage::Read` is `(s64 offset, u64 size)` — not `IFile::Read`**, which
  leads with a `u32 option` and puts its offset at +8.

### One console, one answer

`am`'s operation mode, `apm`'s performance mode, `vi`'s display size, the
shared buffer's geometry, `clkrst`'s GPU rate and whether touch reports
contacts all derive from one `OperationMode` (the page's dock switch):
1280x720 handheld / 1920x1080 docked, Normal / Boost, 384 / 768 MHz. Handheld
is **0**. The display size must agree everywhere or a caller sizes its
framebuffer from one source and scans out through another; `vi`'s display
queries are dispatched once in `Cpu::vi_common_command` ahead of the
sub-interface match.

`SHARED_BUFFER_ADDR` is the one surface the whole system draws into — an applet
asks `vi` for a slot rather than owning a layer. Seven slots, only the first
two handed out; address space is reserved for the docked geometry whatever mode
we are in, which costs nothing because the pages are soft-mapped.

Both IPC routes reach `Cpu::applet_request`: libnx by domain object id, `nnSdk`
by a session handle per interface — so the `am:*` names are also in `svc.rs`'s
dispatch, the same split as `fsp-srv-fs` and `time:system-clock`.

### Services

The per-service inventory — what each of `nvdrv`, `hid`, `lm`, `erpt`, `ssl`,
`bsd`, `acc`, `ts`, `set:sys`, `pm:*`, `pcv`/`clkrst` and the rest must answer
— is **`docs/services.md`**, including what the Home Menu opens that homebrew
never does.

## Audio (`audout`, `audren`)

Two paths, not variants of each other. `audout` is a *device* — the guest hands
it finished PCM. `audren` is a *mixer* — the guest hands it sources and gets
mixed PCM back. Homebrew mostly takes the first, nearly every retail title the
second. Both end in `Cpu::queue_audio`; whichever produced samples last sets
`Cpu::audio_format`.

**Both play in time, on the `cycles` clock the display and thread deadlines
use.** Releasing a buffer on arrival hands the guest an infinitely fast sound
card, and a title's audio clock is what its video is scheduled against.

### `audout`

A buffer is released once the CPU has run for as long as its samples take at
the device's rate (`Cpu::audio_play_cycles`), queued behind whatever is still
playing (`AudioOut::free_at`). Samples still copy to the host on arrival; it is
the *tag* that waits.

- **The buffer event fires on the clock, not the append** (`Cpu::audio_tick`),
  and a blocking wait on it rewinds onto the `svc` — safe because the wake-up
  is a known cycle count away.
- **Do not answer that wait with a bare success.** `nn::audio`'s mixer takes
  the event as proof a buffer is waiting and reads its queue head unchecked.

### `audren` (`cpu/audren.rs`)

One update carries the whole renderer state as a flat buffer whose header
declares each section's size; `Cpu::audren_parse_update` walks it by **those
declared sizes, not by strides computed here** — that is what makes one parser
serve REV1 through REV15, whose effect and mix entries are different widths.
Layout is libnx's `audren.h`, which is the authority.

Signal path: wave buffers → decode (PCM8/16/24/32/float, and Nintendo 4-bit
ADPCM, which is what retail voices actually are) → linear resample by
`rate × pitch` → per-voice biquads → per-channel gains into the destination
mix's buffers → submixes into their destinations, highest mix id first → the
device sink's channel map → interleaved i16.

- **A frame every 5 ms, counted off `cycles`** (`FRAME_CYCLES`), never "one per
  update". `QuerySystemEvent` must hand back a **real event as a copy handle**:
  a bare handle is "not an event", which `WaitSynchronization` reads as always
  ready, so `audrenWaitFrame` returns instantly and the renderer has no clock.
- **`num_wavebufs_consumed` is load-bearing.** The guest advances its own ring
  head by the delta and refills only what this accounts for; a reply of zero is
  a title that queues four buffers, waits, and stops.
- **A renderer opens *started*.** `StartAudioRenderer` exists and libnx never
  calls it.
- **Voice state is re-sent whole every update**; only position, ADPCM history
  and filter state survive, and `is_new` clears those. The playing slot is
  re-seeded from `wavebuf_head` each update.
- **`end_sample_offset` is a claim; `size` is the allocation, and it wins.**
  The buffer is still consumed — one that never comes back stalls its voice.
- Not modelled: effects (parsed for sizing), splitters (stepped over), the
  circular-buffer sink (reported once, never written). Each is a truthful zero.

### `hwopus` (`cpu/hwopus.rs`, `src/opus/`)

The one service whose implementation is a codec. The work buffer the caller
allocates as transfer memory is sized and never read, but `GetWorkBufferSize*`
must still answer — the caller allocates before it opens anything.

- **The packets are not bare Opus.** Each carries an eight-byte
  `{ size, final_range }` header, **big-endian**, and the reply's
  bytes-consumed counts it. Reading it little-endian, or reporting only the
  payload, desynchronises the caller after one packet.
- **The decoder is conformant, and that is checked**: `--example
  opus_testvectors` runs the RFC 8251 vectors and requires the range coder's
  state to match the encoder's on *every* packet. Samples themselves only have
  to pass `opus_compare` — Opus is specified in floating point above SILK.
- **SILK is integer arithmetic and has to be**: the filter a frame ends with is
  what the next predicts from, so a float implementation drifts. CELT's is
  float and matches the reference bit for bit on every unconcealed frame.
- **Concealment is where the two diverge** — an LPC extrapolation amplifies one
  ulp into an audible difference. Ours tracks the reference to within
  `opus_compare`'s tolerance and no closer.

## GPU (`switch-core/src/gpu`, `switch-gpu`)

A model of the Tegra X1's GM20B. Registers from deko3d's generated Maxwell
headers, ioctls from libnx's `nvidia/ioctl`.

- `nvdrv` — `/dev/nvmap`, `nvhost-ctrl`, `nvhost-ctrl-gpu`, `nvhost-as-gpu`,
  `nvhost-gpu`.
- `nvmap` — on Tegra the *guest* allocates and hands nvmap a CPU address, so
  GPU memory is ordinary guest memory.
- `vmm` — small-page region at `0x04000000`, big-page from `0x1_00000000`.
- `syncpt` — a submission completes inside its ioctl, so fences are already
  expired when the guest waits.
- `channel` + `engine/*` — GPFIFO → pushbuffer → method headers → the class on
  that subchannel. 3D `0xB197`, compute `0xB1C0`, inline `0xA140`, 2D `0x902D`,
  copy `0xB0B5`, gpfifo `0xB06F`. **Subchannel 6 is pre-bound** to the
  channel's own gpfifo class; userspace never issues `SetObject`.
- `macro_engine` — methods ≥ 0xE00 are macro slots, and deko3d compiles its
  draws into macros, so nothing draws without it.
- `surface` — block-linear (GOB) swizzling. `texture` + `bcn/` cover BC1–BC7
  and ASTC, the packed 32-bit formats (`A2B10G10R10`, `B10G11R11`) a title
  samples its own HDR targets back through, and `ZF32_X24S8` — a depth buffer
  read as a texture, which is its own `TexelKind` because its texel is not a
  colour in any layout `ColorFormat` names.
- `shader/` — `isa` (SASS decode), `cfg`, `interp` (software shading), `wgsl`
  (translation for the backend).
- `raster` — the software rasterizer, and **the reference every other path must
  agree with**. `qmd` + `compute` are its counterpart for a dispatch.

**`texs` is the short texture sample and `tex` is the general one.** The
operands `texs` has no room for — an explicit level, an `.AOFFI` texel offset,
a shadow reference — come out of one register in that order, each present only
if its own modifier bit is set, so getting the order wrong reads a coordinate
as an offset. The offset is scaled by `TextureSource::texel_step` and added to
the normalized coordinate, which differs from hardware only outside the image
and only for the wrap modes that do not clamp.

**`exit` carries a condition-code test beside its predicate**, and both have
to hold. `EXIT.F` and `EXIT.FCSM_TR` never fire, so they are decoded as `nop`
and the walk carries on past them — Persona 5 Royal's vertex shaders open with
one and write `gl_Position` *after* it, and treating any `exit` as the end of
the program left every vertex at the default clip position, every triangle
collapsed to the viewport centre, and 39,000 draws a run drawing a black frame.
`bra`'s and `kil`'s field is the same one.

**A compute dispatch runs on the CPU, one thread at a time.** Almost none of a
launch is in the class's register file — the grid, block, constant buffers and
shared-memory size come out of a 256-byte QMD in memory (`clb1c0qmd.h`'s
`MW(hi:lo)` fields). Sequential threads are exact except for `bar` (a thread
suspends; the CTA resumes once all have arrived) and `shfl` (answered once its
warp of 32 catches up), and they make an atomic atomic by construction — a
kernel whose answer depends on a race gets a valid one here and a different one
on hardware. Named barriers are not told apart and `bar.arrive` synchronises
like `bar.sync`; both over-synchronise, which no well-formed kernel can detect.
`compute::MAX_DISPATCH_THREADS` refuses a grid past a million threads.

**A fragment shader that shuffles runs in 2x2 quads.** `shfl` reads a register
belonging to *another* invocation — which is what `dFdx`/`fwidth` are built out
of — so a quad's four pixels reach the instruction together. `Halt::Shuffle`
suspends a lane the way a barrier suspends a thread, and
`interp::resolve_shuffles` answers every lane of the warp at once. Lanes are
`(x, y)`, `(x+1, y)`, `(x, y+1)`, `(x+1, y+1)`; `sr0` (`SR_LANEID`) is which.
The quad walk is gated on the program containing a `shfl` or `fswzadd`, since
the pixels a quad shades that the triangle misses are work no other draw does.
Neither op translates to WGSL, so those draws fall back.

Scan-out: `display::BufferQueue` (`QUEUE_BUFFER`) resolves the
`NvGraphicBuffer` to an nvmap id and `Gpu::present` de-swizzles into
`Gpu::framebuffer`.

**The queue transform is part of the frame.** `QueueBufferInput` carries
`NATIVE_WINDOW_TRANSFORM_*` at offset 0x20: how the image is *stored* against
how it is to be *shown*. A title that renders y-down says so there rather than
by mirroring its viewport (Minecraft queues `FLIP_V`), so `Gpu::present` emits
the row the transform names and refuses `ROT_90`. The applet path
(`PresentSharedFrameBuffer`) passes `0` — its transform's offset is unverified.

**The `wgpu` backend never blocks in a browser.** Reading a render target back
means awaiting a promise, and a blocking wait there is not slow but
*deadlocked*. So a surface stays on the device across every draw targeting it
and returns to guest memory only at `Renderer::flush`, which the engine calls
before `present`. Opening the device is likewise deferred: `worker/index.ts`
calls `switch_gpu_open` *between* run slices, since the channel does not exist
until the title has run. **Any draw the backend cannot express falls back to
`Software`** — a backend that guessed would produce a frame nobody could check.

`Renderer::flush` polls with `PollType::Wait`, which is a real wait natively
and *no-ops* on WebGPU, where callbacks come from the event loop. So a browser
gets `Flush::Pending` and the present waits for a later slice
(`Cpu::complete_pending_present`).

**Where a readback lands late, a frame is all one renderer's.** The first flush
answering `Pending` sets `Gpu::deferred_readbacks`, and from then on a frame in
which anything fell back makes every frame after it the rasterizer's whole
(`Gpu::software_frame`). Guest memory is then the only copy of every surface
and no readback is ever owed. **It latches on purpose** — alternating is the
one behaviour this must not have. `GPU_INTERLEAVE=1` (off, named in
`web/worker/index.ts`) trades correctness for speed; `GPU_DEFER_READBACKS=1`
reproduces the browser's late map natively, the only way to measure any of it.

**What buys the acceleration back is `shader::wgsl`.** Every fallback that
latches it is an opcode with no WGSL form, not anything WebGPU withholds. A
title the translator covers completely never reaches it.

**A copy out of a held surface flushes first.** The 2D blitter and the copy
engine read guest memory, so `channel.rs` hands the surfaces back before
`Engine2D::LAUNCHES_BLIT` and `copy::LAUNCH_DMA` — the same guard compute had.

**Depth, clears and multisampling all run on the device.** A depth surface is
held like a colour one, converted to `depth16unorm` or `depth32float` — the two
formats a copy can read — with the stencil byte read back out of guest memory
and put where it was, since neither renderer tests stencil. Nothing copies
*into* `depth32float`, so a surface gets there by being drawn: a fullscreen
triangle writing `@builtin(frag_depth)`. Clears are a pass's load operation
where they cover the whole surface and a scissored fullscreen draw where they
do not; a whole clear skips the upload, since nothing reads a surface it is
about to overwrite.

**Multisampling has two routes, and the default is the exact one.** Maxwell
stores samples *spatially* — a pixel owns a `samples_x` by `samples_y` tile of
texels — so guest memory holds the expanded image and the default route renders
exactly that, one fragment per texel. Coverage is tested at texel centres,
which is where Maxwell's samples are, so it reproduces the rasterizer texel for
texel; the sample mask and alpha-to-coverage become the fragment shader's job
(`wgsl::Coverage`). Two multisampled draws still fall back, both deliberately:
`MultisampleSampleLocations` away from texel centres
(`SampleGrid::samples_at_texel_centres`), and per-pixel coverage with a
*partial* sample mask, which has nothing to act on. `GPU_DEVICE_MSAA=1` lets
the device multisample instead — off, because WebGPU fixes its sample positions
at a rotated grid that is not Maxwell's, so every edge comes out correct and
*different*; core WebGPU guarantees four samples, so `2x1`, `4x2` and `4x4`
take the expanded route regardless.

**Two renderers disagree by a 255th where a channel lands on a half.**
`ColorFormat::encode` rounds `127.5` up and a device's unorm conversion rounds
it down. Not a bug in either — a test wanting byte-identity picks values off
the eight-bit half-way points.

**The reference is how you check the backend.** Run `screenshot_title` and
`screenshot_gpu` over the same frame and `cmp` the PPMs; a byte-identical pair
is the only evidence the backend renders what the rasterizer does.
`GPU_ONLY=<i>` runs only the i-th draw on the device, so a difference is
exactly one draw's — that is what caught a doubled y flip.

`switch-gpu`'s own tests are the other half, and the faster half:
`gpu::testing::Harness` builds a drawable `Engine3D` (a 16x8 target, two real
shaders, three vertices) that both renderers are driven over, so a route can be
checked without booting a title — every multisample mode, the sample mask,
alpha-to-coverage, per-pixel coverage, depth-tested and depth-only draws, fans,
instanced arrays, BGRA and unbound attributes, and clears. It exists because a
title exercises only what that title happens to do.

**hbmenu is not a shader-core test.** `nx_graphics.c` draws with the CPU into a
linear memblock and its command list is `dkCmdBufCopyBufferToImage` + a fence,
so only the copy engine and syncpoints are involved.

Per-title results live in PROGRESS.md.

## Input (`cpu/mod.rs`, `web/main/input.ts`)

`Cpu::set_gamepad_state` publishes the pad two ways: the `INPUT_ADDR` register
and libnx's `HidSharedMemory`. The offsets came from compiling libnx's
`services/hid.h` on the host and are not guessable: `npad` at **0x9A00**, one
entry every **0x5000**, `full_key_lifo` **+0x28**, `handheld_lifo` **+0x378**,
`device_type` **+0x4188**; each LIFO is a 0x20-byte header then 0x30-byte
entries. One entry at index 0 with `tail = 0`, `count = 1` and
`IsConnected` is all `hidGetNpadStates*` needs.

Published in **two slots** — player 1 as a Pro Controller and slot 8 as
handheld — because software polls whichever it expects and `padUpdate` merges
them. Buttons are Horizon's order (A=1<<0 … StickR=1<<5, L=1<<6, ZL=1<<8,
Plus=1<<10, d-pad from 1<<12), the `StickL*`/`StickR*` pseudo-buttons are
derived from the analog values, and **stick Y is positive up** — the opposite
of the Gamepad API.

**The run slice is the input sampling period.** The worker is single threaded,
so a `set_input` posted mid-slice waits for `switch_run` to return. Slice size
costs the interpreter nothing, only postMessage round trips.

The worker latches a press until the **frame counter has advanced twice**, not
for a fixed number of slices: the guest polls hid once per iteration of a loop
that spans many slices, and the poll sits *inside* that loop, so only a
complete present-to-present interval is guaranteed to contain one. Only bits
the guest may not have seen are latched — a still-held key publishes from
`heldButtons` alone, so a release lands on the next slice.
`MAX_LATCH_SLICES` (64) covers a program that has stopped presenting.

**Touch.** `HidSharedMemory.touch_screen` is at **0x400**; a `HidTouchState` is
**0x28** bytes with `finger_id` at +0x0C, `x` at +0x10, `y` at +0x14.

- hid reports touches in the console's own **1280x720 digitizer space**, not
  the guest's presented resolution, and `#screen` is `object-fit: contain` — so
  a tap must be mapped through the **contained rect**, or the letterbox bars
  offset every one.
- **Touch is handheld-only**; docked reports zero contacts whatever the page
  sends.
- **A lift is a published state with `count = 0`, not silence.** Vacated slots
  are zeroed so a reader that scans the array finds no ghost contact.
- Taps ride the same latch, and `finger_id` is held for the life of the pointer
  so a title can follow a drag.

## Shared system font (`cpu/ipc.rs`, `web/font.ttf`)

Software ships no fonts: it asks `pl:u` by type, gets an offset and size, and
reads out of pl's shared memory (mapping the region, recognised by
`PL_SHMEM_SIZE`, is what fills it).

`Cpu::build_shared_fonts`: **the real fonts come from firmware** — a BFTTF in
each system data archive's RomFS, decoded by `decode_bfttf`. With no firmware,
`Cpu::set_shared_font`'s host font stands in for **every** type. Reporting an
empty set means no text renders at all, and registering one face as only the
standard type is nearly as bad — the Home Menu asks for the whole set and looks
each character up in each face in turn.

`GetSharedFontInOrderOfPriority` (5, and 6 system-side) must fill its three
output buffers *and* its count, or a caller reads "loaded, zero fonts" and
retries forever.

`tools/make_font.py` builds the shipped subset. It strips TrueType hinting
because hinted glyphs collapse horizontally under the interpreter (a real bug —
see PROGRESS.md) and the bytecode costs about **8x** more instructions per
frame. It also points Nintendo's private-use button codepoints (0xE0E0…,
0xE0A0…) at matching letters.

## Storage

- **SD card** (`vfs.rs`) — a real path-addressed tree, so `GetEntryType`,
  `OpenDirectory`, `fsDirRead`, `OpenFile` and `Read` agree with each other and
  a missing path is `FsError_PathNotFound`. A fixed listing made menus recurse
  forever. The running NRO is published at `nro::HOMEBREW_NRO_PATH` and
  advertised as `argv[0]`, which is how `romfsMountSelf` works.
- **Saves** — the same filesystem one level in, created empty on open.
- **NAND** — a browser will not hand a page a file it was not asked for, so an
  archive registered by `switch_add_archive` (a `File` reference) dies on
  reload. Bytes can be kept: `switch_nand_add_archive` stores them,
  `switch_nand_identify` reads just an NCA header out of a `File` the host
  still holds, and `switch_nand_launch` boots a Program NCA from stored bytes —
  the Home Menu and Mii editor ship as bare NCAs with no NSP to open them from.

All three persist in IndexedDB from the page side (`main/db.ts`, `sdcard.ts`,
`saves.ts`, `nand.ts`); the core keeps them in memory and reports what changed.

## Performance

**One tool holds a clock, and it is not a host example.** A frame costs the
work it asks for times what that work costs on the machine running it, and only
the first half is a fact about this emulator. The second is a fact about a
compiler: the host examples are x86-64 out of rustc's backend, over 64-bit
pointers and an unbounded address space, and what ships is wasm32 recompiled by
a browser with its own register allocator and a bounds check on every guest
load. hbmenu's boot-to-frame-4 is ~79 M instructions/s here and ~59 M/s under
V8 on the same machine (min of three, the same 130 M instructions), and that
1.3x is not a constant — removing a libcall only wasm pays for was ~1.15x
natively and ~1.44x in the browser. So the host tools count, and `wasm_bench`
times.

- `node tools/wasm_bench.mjs <nro>` — the artefact `make wasm` produced, timed
  from inside one run. **The only tool here whose milliseconds mean anything**,
  and with `node --cpu-prof` the only profiler for that build. It loads the
  shipped module through its own wasm-bindgen glue rather than building one, so
  what it times is what the site serves.
- `--example frame_work -- <nro>` — one steady frame as work: instructions,
  block entries, fallbacks, methods, draws, clears, copies, pixels. Every count
  is the same under V8, and its instructions/frame should match `wasm_bench`'s.
- `--example jit_coverage -- <nro>` — the instructions a real frame runs that
  the translator has no op for, hottest first and disassembled, then rolled up
  by encoding group to read against `hotspots`. This is where the next block of
  CPU speed is.
- `--example present_work` — scan-out as bytes lifted, deswizzle lookups and
  pixels converted, per layout and crop. It runs whatever the renderer drew, so
  a title issuing no draws still pays it in full.
- `--example hotspots -- <nro>` — one steady frame bucketed by guest address.
  It counts **guest** instructions only and cannot see the emulator's own cost:
  what falls outside the image buckets is more guest code, not GPU work.
- `--example jit_difftest -- <nro>` — both engines over the same program, with
  every state difference. A correctness tool: the interpreter is the reference,
  and a translated run that disagrees is wrong however fast it was.

**A change that moves no count and looks faster on the host made this host
faster.** Fix what the work counters name, then confirm it under `wasm_bench`.
`perf` on the host still finds *where*, never *how much*.

**The per-pixel and per-instruction paths are where whole seconds hide.** A
full-screen pass is 921,600 fragment invocations, so a small `Vec`, a `HashMap`
for a sparse thing, or a rescan to answer a question costs seconds a frame; the
same goes for anything translating an address per byte. Frames must stay
byte-identical across any such change.

An identity map for guest memory does not fit — addresses reach
`GUEST_SPACE_END` (0xF5000000) and wasm32 caps at 4 GiB — but codegen does not
need one: only ≤3.125 GiB is ever backed (`MAX_MAPPED_BYTES`), so slabs in one
arena with a flat `u32` offset table would let generated code translate inline.

PROGRESS.md carries the measurements behind all of this.

## A64 decode traps

`docs/a64-decode.md` — the encodings that are easy to get wrong (register 31 in
ADD/SUB, W-register zero-extension, the scalar-FP guards, SIMD structure
loads, the permute trio, TBL/TBX) and the guards that get them wrong.
Cross-check any new decode against `llvm-mc -triple=aarch64 -disassemble` and
`tools/difftest.py`.

## Gotchas

- `cargo clippy` **fails** in `switch-wasm` — 28 `not_unsafe_ptr_arg_deref` on
  the deliberate raw-pointer `extern "C"` signatures. Not gating.
- **The crate is `cargo fmt`-formatted** at rustfmt's defaults (no
  `rustfmt.toml`); `--check` is clean. Run it rather than hand-aligning.
- `json_escape` **walks characters, not bytes** — a `\uXXXX` escape names a code
  point, so escaping a multi-byte character per byte spells a different string.
  Above-ASCII goes out as itself; the page decodes with a UTF-8 `TextDecoder`.
- CPU test encodings in `tests/cpu_test.rs` are hand-assembled and cross-checked
  against QEMU's `a64.decode`.
