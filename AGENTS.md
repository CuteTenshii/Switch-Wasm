# AGENTS.md

Browser Switch emulator: an A64 interpreter with a block-translating JIT, a
software GPU with an optional WebGPU backend, and the container stack
(PFS0/NSP, NCA, NSO/NRO/ELF, RomFS) needed to boot retail titles and system
applets. Compiled to WASM; frontend is TypeScript on Vite.

PROGRESS.md is the long-form log of what was tried and why. This file is the
standing state.

## Commands

- `make all` — `test` then `assets`.
- `make test` — `cargo test` over all three crates. 752 tests.
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
- `cargo run --release -p switch-core --example jit_bench -- <nro>` — both
  engines side by side, with every state difference between them.

## Crates

- `switch-core` — the emulator, **zero dependencies**. `cpu` (`mod`/`alu`/
  `bits`/`fp`/`jit`/`loadstore`/`simd`/`svc`/`system`, plus `ipc` and one
  module per service domain — see below), `gpu`, `display`, `mem`, `vfs`,
  `source`, `crypto`/`keys`/`ticket`, `nsp`/`nca`/`romfs`/`npdm`/`nso`/`nro`/
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
than the `MAX_MAPPED_BYTES` (512 MiB) cap for free.

```text
0x0010_0000  ENV_BLOCK_ADDR (nro.rs)  homebrew ABI environment block
0x0800_0000  ASLR region, 496 MiB     svcGetInfo 12/13
0x1800_0000  GUEST_STACK_REGION_ADDR  thread-stack mirrors, 128 MiB (14/15)
0x1F00_0000  SELF_RETURN_TRAMPOLINE   +0x100 THREAD_EXIT_TRAMPOLINE
0x1FE0_0000  main thread TLS          children from THREAD_TLS_BASE, page each
0x2800_0000  STACK_BASE               main stack 1 MiB, SP at STACK_TOP
0x2900_0000  RO_MODULE_REGION_ADDR    ldr:ro maps run-time NROs, 112 MiB
0x3000_0000  heap / alias             per MemoryLayout, below
0xF000_0000  SHARED_BUFFER_ADDR       system shared buffer, ~59 MiB reserved
0xF400_0000  FB_BASE (lib.rs)         demo framebuffer, 640x360 RGBA
0xF410_0000  INPUT_ADDR               memory-mapped input block
0xF500_0000  GUEST_SPACE_END          above this a read faults
```

**Every region `svcGetInfo` reports must be representable here.** Horizon's
real bases (alias 0x10_0000_0000) truncate to 0 when `nnSdk` asks
`svcMapPhysicalMemory` to back them.

```text
MemoryLayout::PLAIN                 MemoryLayout::VIRTUAL_ADDRESS
0x3000_0000  heap,  2.5 GiB         0x3000_0000  heap,  128 MiB
0xD000_0000  alias, 512 MiB         0x3800_0000  alias, 2.875 GiB
total memory 2.5 GiB                total memory 896 MiB
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
  size pools from it against numbers baked into their own code, so 2.5 GiB is a
  floor rather than a fidelity target.
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
scaled from, so **a billion steps is about a second of console time**.

**The clock and the step count are not the same number.** `Cpu::cycles` is the
clock, and `reschedule` idles it forward to the earliest sleeper whenever every
thread is blocked — the console's own idle, covering cycles nobody executed.
`Cpu::steps` counts retired instructions and the idle never touches it; both
engines go through `Cpu::retire` so they cannot drift. The page's *Steps*
readout is `switch_get_steps`, because a figure that leaps while the guest is
stopped is useless as the loading screen's sign of life: a parked Home Menu
jumped 24M → 313M having run nothing.

| steps | console time | reaches |
| --- | --- | --- |
| 2,000,000 (`boot_nsp` default) | 2 ms | barely past `rtld` |
| 400,000,000 | 0.4 s | service init, a layer, the first frame loop |
| 1,500,000,000 | 1.5 s | "A Short Hike" running clean, still no frame |

A retail title spends **seconds** of console time before its first frame, an
IL2CPP game longer. Before concluding a title does not render, check the run
was long enough for it to, and prefer `SHOT=<file.ppm>` over reading `frames
presented: 0` off a budget that was never going to get there. PROGRESS.md's
status table is where per-title results live; keep it, not this file, current.

## Block translation (`cpu/jit.rs`)

First visit to an address translates forward into `Op`s — operands extracted,
immediates decoded — until something changes the PC; that plus its terminator
is a cached `Block`. Worth **1.9–2.1x**. It removes decode, not dispatch, and
generates no code: every op calls the same helper the interpreter would, so the
two engines are the same computation and anything untranslated falls back.
`SWITCH_NO_JIT=1` for host tools, the debug panel's *Translation* section in
the browser.

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
  `svcUnmapMemory`.
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

## IPC (`cpu/ipc.rs`, `cpu/svc.rs`)

**Two dialects.** libnx ignores the reply's `type` and raw-data size and
converts `fsp-srv` to a **domain** (sub-interfaces are object ids on one
handle). libtransistor validates both and never converts, so a sub-interface
must come back as a **session handle in a move handle**;
`Cpu::reply_with_interface` picks the shape per request.

**Two encodings of every message kind**: plain (`Request` 4, `Control` 5) and
with-context (6, 7), which prefixes a 16-byte tracing context. libnx sends
plain, **`nnSdk` sends the context form for everything** — so test with
`Cpu::ipc_is_control_request`, never `type == 5`.

A **Close** (type 2) carries no command id; `svc.rs` answers it before dispatch
and calls `Cpu::forget_handle`, or the leftover id in the TLS buffer runs a
real command.

### The bug class: success with an unfilled out parameter

Not visible from the reply. A reply is written *over* the request in the same
TLS buffer and declares four words of padding past the SFCO header, so a bare
success passes every length check and hands back stale *request* bytes.
`Cpu::write_ipc_reply` clears the section it is about to declare, so
unimplemented out parameters read as 0 — still wrong, but the same wrong every
time. When a command *does* have an out parameter, implement it; when its width
is unclear, reply with a zeroed block **wider** than needed. A reply may be
longer than expected, never shorter.

The same trap in the handle slots. `nnSdk` reads an out-object off a plain
session as a move handle, and a reply with none is *not an error* to it: the
handle parses as 0, the proxy is silently skipped, the command returns success,
and the first virtual call goes through null. So
`Cpu::reply_with_fabricated_object` carries **a sub-session in the move slot and
an event in the copy slot**, allocated once per `(session, command)` — nothing
here can tell which the caller wanted.

**An unimplemented `am` command must not answer with a bare success** —
everything `am` returns is a live object or applet state.
`Cpu::unimplemented_command` reports `cmif`'s `UnknownCommandId` (`0x1ba0a`)
and warns once per `(interface, command)`. A bare success is still right for a
genuine setter/notifier, but those are listed by id, never caught by `_`.

`Cpu::warn_no_implementation` logs `[ipc] no implementation` once per pair, and
that list *is* the inventory of what a guest is asking for and not getting.

### Events

`Cpu::alloc_event` records one in `Cpu::events`, and it must reach the guest
through the **copy** list — a move handle transfers ownership, a copy handle
duplicates one the server keeps, they occupy different descriptor fields, and
an event in the move slot reads back as **0**.

**An event handed out twice must be the same event** (`Cpu::kept_event`, keyed
per `(purpose, object)`), or the caller waits on an object the service would
not signal.

`svcWaitSynchronization`: a handle not in `Cpu::events` counts as ready (do not
"fix" this without checking homebrew); a **poll** (timeout 0) on unfired events
reports Timeout; a **blocking** wait with nothing signalled reports the first
handle ready, which is a deliberate lie — see `WaitAny` above. `vsync_event`
fires on a period as well as on presents.

### Where headers and payloads land

`Cpu::ipc_cmif_header_offset` finds "SFCI" by walking the request's descriptors
first, because buffer descriptors push it well past the start (nvdrv's
`KICKOFF_PB` puts it at 0x40) — a fixed 0x40 scan reported "no command id" and
answered the whole GPU submit as an unknown command.

`Cpu::ipc_request_data` does the same for the payload: a domain request carries
a `CmifDomainInHeader` in front, so its payload sits 0x20 rather than 0x10 in.

Buffers: `ipc_input_buffer` / `ipc_output_buffer` try map-alias then pointer,
`ipc_send_buffer` is the map-alias send side, and `ipc_recv_static_buffers`
handles the one kind that sits *after* the raw data (at the unaligned data
offset plus `num_data_words`).

**`QueryPointerBufferSize` must be non-zero** wherever pointer buffers are used
(`hid`, `acc`, `set:sys`) — `nnSdk` checks the negotiated size *before* sending
and gives up when the server claims no room.

**`CloneCurrentObject` (control 2, 4 for Ex) must return a new session handle
as a move handle.** Answered centrally in `svc.rs`; the clone reaches the same
interface and inherits the same domain objects. `nnSdk` clones `fsp-srv` before
mounting anything.

**Mount names live in the guest** — `nn::fs`'s `MountTable` is client-side, so
`rom:` needs nothing beyond `OpenDataStorageByCurrentProcess` (200) and a
storage that reads correctly.

**`IStorage::Read` is `(s64 offset, u64 size)` — not `IFile::Read`**, which
leads with a `u32 option` and puts its offset at +8.

### One console, one answer

`am`'s operation mode, `apm`'s performance mode, `vi`'s display size, the
shared buffer's geometry, `clkrst`'s GPU rate and whether touch reports
contacts all derive from one `OperationMode` (set by the page's dock switch):
1280x720 handheld / 1920x1080 docked, Normal / Boost, 384 / 768 MHz. Handheld
is **0**. The display size must agree everywhere or a caller sizes its
framebuffer from one source and scans out through another; `vi`'s display
queries are dispatched once in `Cpu::vi_common_command` ahead of the
sub-interface match.

`SHARED_BUFFER_ADDR` is the one surface the whole system draws into — an applet
asks `vi` for a slot rather than owning a layer. Seven slots, only the first
two handed out (as on a console); address space is reserved for the docked
geometry whatever mode we are in, which costs nothing because the pages are
soft-mapped.

Both IPC routes reach `Cpu::applet_request`: libnx by domain object id, `nnSdk`
by a session handle per interface — so the `am:*` names are also in `svc.rs`'s
dispatch, the same split as `fsp-srv-fs` and `time:system-clock`.

### Services

- **`nvdrv`** — the real `INvDrvServices`, dispatched into `gpu::nvdrv`.
- **`hid`** is the *negotiation*, not the input: the data lives in a 256 KiB
  shared memory region the guest reads with no IPC per frame.
  `CreateAppletResource` → `GetSharedMemoryHandle` hands it over. The
  `Set*`/`Get*` pairs are read back and must agree. Vibration comes back
  through `SendVibrationValue` → `Cpu::vibration` → the Gamepad API's
  `dual-rumble`.
- **`pl:u`/`pl:s`** — the shared fonts; see below.
- **`lm`** carries a title's own `NN_LOG` output: a 0x18-byte LogPacket header
  then TLV chunks (key 2 message, key 6 module) in a map-alias buffer, split
  across packets with `flags` bit 0 head / bit 1 tail. Retail builds often
  compile logging out, so an empty log is not evidence of a bug.
- **`fatal:u`** carries the `Result` that stopped a process — the first account
  a guest ever gives of why.
- **`erpt`** is the second, and far more detailed: a journal of *context*, one
  record per category (`ErrorInfo`, `GpuCrashInfo`, `ThermalInfo`), resubmitted
  rather than appended to, which `CreateReport` writes out whole the moment
  something notices a problem. A report being filed is a `diagnostic`, named by
  the categories it is about. Nothing persists and nothing uploads, so the
  journal lives exactly as long as the session — and `IManager`'s report-created
  event is the one event in these services that genuinely fires.
- **`ssl`** — contexts and options are real; **`CreateConnection` deliberately
  reports unimplemented** rather than handing back a connection that can never
  connect. Offline titles only call `SetInterfaceVersion`.
- **`bsd`** models a link that is up and a network where nothing answers: local
  operations succeed, anything needing a peer fails at once with a definite
  errno (`ECONNREFUSED`, `ENOTCONN`/`ENETUNREACH`, `EAGAIN`). **Errnos are
  FreeBSD's** (`EAGAIN` is 35) and `fcntl`'s flags are stored verbatim, since
  what has to hold is that a guest reads back what it set. A `poll` *with* a
  timeout is a wait, so it asks for a reschedule (`Cpu::pending_yield`) before
  returning zero — otherwise a poll loop owns the CPU forever.
- **`sfdnsres`** — `EAI_NONAME` / `HOST_NOT_FOUND`, the *definitive* failure
  rather than try-again, in the **first** word of `SfdnsresRequestResults`.
  Error strings are answered properly.
- **`pctl`** reports the console unrestricted. Watch the direction:
  `Confirm*`/`Check*Permission` reply with a bare `Result` where success *is*
  permitted, `IsRestriction*` is `false`, `IsFreeCommunicationAvailable`/
  `IsStereoVisionPermitted` are `true`.
- **`acc`** models exactly one user, always signed in, uid `ACCOUNT_UID`
  (nonzero — 0 is the "no user" sentinel). `acc:u0` and `acc:u1`/`acc:su` share
  0..=51 but **diverge from 100 up**, so those arms dispatch on the service
  name. The nickname is real state. `LoadImage` returns a real JPEG.
- **`apm`** must *agree* with `am`; `GetPerformanceConfiguration` returns what
  `Set*` was last handed (defaults nonzero — 0 is `Invalid`).
- **`ts`** reports the SoC and PCB sensors at an idle reading. `MilliC` is
  `GetTemperature` × 1000 and both sit inside `GetTemperatureRange`.
  **`ISession` is a different interface from its server** — its
  `GetTemperature` is command 4, the same id as the server's `OpenSession`. The
  device code's **high byte** picks the sensor (`0x41…` SoC, `0x43…` PCB).
- **`set:sys`**'s `GetFirmwareVersion`/`2` are **not cosmetic**: libnx seeds
  `hosversionGet()` from them and everything version-gated branches on that.
- **`csrng`** fills from `Cpu::next_random_u64` (splitmix64). Not a CSPRNG —
  but the generic reply left the buffer untouched, which is non-random *and*
  undetectably so.
- **`spl:`** — an Icosa retail console, not in debug mode. Atmosphère's
  extensions at 65000+ answer zero, i.e. "no CFW", which is true.
- **`pdm:qry`** — a console nothing has been played on. Factory-fresh is a
  truthful state; a placeholder is not.
- **`pm:*`** are four interfaces on four names. `pm`'s process id must equal
  `svcGetProcessId`'s. `pm:info`'s program id defaults to the Album applet's.
- **`pcv`/`clkrst`** are the same manager either side of 8.0.0, and their
  numbering differs **by an offset**: a `clkrst` device code is `0x40000000 +
  module + 1`. A rate a guest sets reads back.
- **`fsp-srv`**'s `DisableAutoSaveDataCreation` (1003) is accepted and
  deliberately **not** honoured — saves are created on open and there is no
  installer to have made them first.

### What the Home Menu opens that homebrew never does

`lbl`, `audctl`, `nfc:sys`, `btm:sys`, `ldn:m`, `lp2p:m`, `ovln:*`, `olsc:s`,
`friend:*`, `news:*`, `bcat:*`, `notif:*`.

- **Most are a creator plus the objects it creates** (`olsc:s` is five deep),
  and a fabricated object id is not callable — so the fallback ended each chain
  at its *first* command. Each sub-interface gets a name of its own
  (`Cpu::ipc_interface`), listed in `svc.rs`'s dispatch.
- **The answer is an empty console, not a broken one.** No friends, news, BCAT,
  cloud saves, local network, NFC or paired gamepad — every one a state a real
  console reaches, so callers already have a path for it. A *failure* puts them
  on the path built for hardware that broke. None of these events ever signal.
- **The settings among them are stored, not answered** (`backlight`,
  `audio_control`, `notif_alarms`, …) — one caller writes, another reads back.

## Audio (`audout`, `audren`)

Two paths, and they are not variants of each other. `audout` is a *device* —
the guest hands it finished PCM. `audren` is a *mixer* — the guest hands it
sources and gets mixed PCM back. Homebrew mostly takes the first; nearly every
retail title takes the second, through `nn::audio` or libnx's `audrv`. Both end
in `Cpu::queue_audio`, and whichever produced samples last sets
`Cpu::audio_format`.

### `audout`

**The device plays in time.** A buffer is released once the CPU has run for as
long as its samples take at the device's rate (`Cpu::audio_play_cycles`),
queued behind whatever is still playing (`AudioOut::free_at`), on the same
`cycles` clock the display and thread deadlines use. Samples still copy to the
host on arrival; it is the *tag* that waits. Releasing on arrival hands the
guest an infinitely fast sound card, and a title's audio clock is what its
video is scheduled against — one pushed 205× real time and its software video
player dropped every frame of the boot video.

- **The buffer event fires on the clock, not the append** (`Cpu::audio_tick`),
  and a blocking wait on it rewinds onto the `svc` — safe because the wake-up
  is a known cycle count away.
- **Do not answer that wait with a bare success.** `nn::audio`'s mixer takes
  the event as proof a buffer is waiting and reads its queue head unchecked.

### `audren` (`cpu/audren.rs`)

The renderer, and the same clock discipline. One update carries the whole
renderer state as a flat buffer whose header declares each section's size;
`Cpu::audren_parse_update` walks it by **those declared sizes, not by strides
computed here** — that is what makes one parser serve REV1 through REV15, whose
effect and mix entries are different widths. Layout is libnx's `audren.h`, which
is the authority; read it before changing an offset.

Signal path: wave buffers → decode (PCM8/16/24/32/float, and Nintendo 4-bit
ADPCM, which is what retail voices actually are) → linear resample by
`rate × pitch` → per-voice biquads → per-channel gains into the destination
mix's buffers → submixes into their destinations, highest mix id first → the
device sink's channel map → interleaved i16.

- **A frame every 5 ms, counted off `cycles`** (`FRAME_CYCLES`), never "one per
  update". `Cpu::audren_tick` fires the frame event, `Cpu::audio_tick` folds its
  deadline in with `audout`'s, and a wait on it parks like any buffer wait.
  `QuerySystemEvent` must hand back a **real event as a copy handle**: a bare
  handle is "not an event", which `WaitSynchronization` reads as always ready —
  `audrenWaitFrame` then returns instantly and the renderer has no clock at all.
- **`num_wavebufs_consumed` is load-bearing.** The guest advances its own ring
  head by the delta and refills only what this has accounted for, so a reply
  that reports zero — which the all-zero stub did — is a title that queues four
  buffers, waits, and stops.
- **A renderer opens *started*.** `StartAudioRenderer` exists and libnx never
  calls it.
- **Voice state is re-sent whole every update**; only position, ADPCM history
  and filter state survive, and `is_new` clears those. The playing slot is
  re-seeded from `wavebuf_head` each update, which is how both sides stay in
  step without exchanging a position.
- **`end_sample_offset` is a claim; `size` is the allocation, and it wins.**
  Same lesson as `audout`'s unplayable descriptor. The buffer is still consumed
  — a buffer that never comes back stalls the voice that queued it.
- Not modelled: effects (parsed for sizing, never processed), splitters (stepped
  over), and the circular-buffer sink (reported once, never written). Each is a
  truthful zero in the reply rather than a guess.

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
  channel's own gpfifo class — nvhost binds it at creation and userspace never
  issues `SetObject`.
- `macro_engine` — methods ≥ 0xE00 are macro slots, and deko3d compiles its
  draws into macros, so nothing draws without it.
- `surface` — block-linear (GOB) swizzling; a naive framebuffer dump looks
  shredded because of this. `texture` + `bcn/` cover BC1–BC7 and ASTC.
- `shader/` — `isa` (SASS decode), `cfg`, `interp` (software shading), `wgsl`
  (translation for the backend).
- `raster` — the software rasterizer, and **the reference every other path must
  agree with**. `qmd` + `compute` are its counterpart for a dispatch: the
  256-byte launch descriptor, then one scalar invocation per thread of the grid.

**A compute dispatch runs on the CPU, one thread at a time.** Almost none of a
launch is in the class's register file — the channel writes a QMD address and
the grid, the block, the constant buffers and the shared-memory size come out
of a 256-byte structure in memory (`clb1c0qmd.h`'s `MW(hi:lo)` fields). Threads
being sequential is exact for everything except `bar`, where a thread suspends
and the CTA resumes once every thread has arrived; it also makes an atomic
atomic by construction, so a kernel whose answer depends on a race gets a valid
one here and a different one on hardware. Named barriers are not told apart and
`bar.arrive` synchronises like `bar.sync` — both over-synchronise, which no
well-formed kernel can detect. `compute::MAX_DISPATCH_THREADS` refuses a grid
past a million threads: hardware runs one in microseconds, and the worker
thread the whole GPU stack shares does not.

Scan-out: `display::BufferQueue` (`QUEUE_BUFFER`) resolves the
`NvGraphicBuffer` to an nvmap id and `Gpu::present` de-swizzles into
`Gpu::framebuffer`.

**The `wgpu` backend never blocks.** Reading a render target back means
awaiting a promise, and a blocking wait in a browser is not slow but
*deadlocked*. So a surface stays on the device across every draw targeting it
and returns to guest memory only at `Renderer::flush`, which the engine calls
before `present` — eighty-eight round trips a frame become one. Opening the
device is likewise deferred: `worker/index.ts` calls `switch_gpu_open`
*between* run slices, since the channel does not exist until the title has run.
**Any draw the backend cannot express falls back to `Software`** — a backend
that guessed would produce a frame nobody could check.

**The reference is how you check the backend.** Run `switch-core`'s
`screenshot_nca` and `switch-gpu`'s `screenshot_gpu` over the same frame and
`cmp` the PPMs — a byte-identical pair is the only evidence the backend renders
what the rasterizer does. `GPU_ONLY=<i>` runs only the i-th draw of each frame
on the device and leaves the rest to the reference, so a difference is exactly
one draw's. That is what caught a doubled y flip: two flat greys trading places
is geometry, not colour. The Home Menu's 88 draws all run on the device with no
fallback, 99.88% pixel-identical, 811 → 513 ms a frame.

**hbmenu is not a shader-core test.** Its menu renders correctly, but
`nx_graphics.c` draws with the CPU into a linear memblock and its command list
is just `dkCmdBufCopyBufferToImage` + a fence. Only the copy engine and
syncpoints are involved.

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

Four tools, and they disagree — measure the one you care about:

- `--example bench` — per-instruction-class throughput on the host. `b .` is
  the floor, so the gap to a class is that class's decode+execute cost.
- `--example hotspots -- <nro>` — one steady frame bucketed by guest address
  and encoding byte. This is how you learn 72% of an hbmenu frame is hbmenu's
  own gradient fill and ~10% is the emulator's GPU work.
- `--example jit_bench -- <nro>` — both engines, with every state difference.
- `node tools/wasm_bench.mjs <nro>` — the build the browser runs, in fps.
  `node --cpu-prof` on it names wasm functions; it is the only profiler for
  that build.

What the numbers taught us:

- **Dispatch order matters on the host, inlining matters in wasm.** Top-level
  group routing was worth ~25% natively and *nothing* in wasm; splitting
  `Memory`'s accessors into an `#[inline(always)]` fast path plus `#[cold]`
  fallbacks was worth ~15% in wasm.
- The interpreter's floor is ~9ns/instruction natively and ~20ns in wasm.
  Block translation is what got past it; more guard reordering would not have.
- **A fragment shader runs once per covered pixel**, and a full-screen pass is
  921,600 of them — so a small `Vec`, a `HashMap` for a sparse thing, or a
  rescan to answer a question costs whole seconds a frame. Fixing three of
  those was 2.6 s/frame → 0.67 s. Measure with `examples/screenshot` at two
  frame indices; the frames must stay byte-identical.
- Anything on the per-instruction path deserves the same scrutiny: the GPU's
  `read_pixel`/`write_pixel` used to translate a GPU address **per byte**.

## A64 decode traps

Cross-check any new decode against `llvm-mc -triple=aarch64 -disassemble` and
`tools/difftest.py`.

- **Register 31 in ADD/SUB**: SP in the immediate and extended-register forms,
  XZR in the shifted-register form. `neg x1, x0` is `sub x1, xzr, x0`.
- **A write to a W register zeroes bits 63:32**, and a 32-bit operand must be
  sign-extended from *bit 31* before an arithmetic shift or signed divide —
  masking to 32 bits makes `asr w, w, w` and `sdiv w, w, w` unsigned.
- **BLR reads its target before linking.** `blr x30` is a legal
  return-and-relink; writing x30 first branches to itself+4.
- **A guard that includes a fixed bit kills the whole group.** The scalar-FP
  1-source group is `opcode(6) 10000` (bits[15:10] are `opcode<0>:10000`);
  FCSEL/FCCMP have bit21 *set*; the int↔float conversions read
  `rmode`:`opcode` as bits[21:16], which folds in fixed bit21 and made `ucvtf
  d0, x1` execute as FCVTMU. The 3-source group's top byte is `00011111`, so it
  must match before the `00011110` space. Prove a new guard reaches a real
  encoding.
- **SIMD&FP LDR/STR**: the register-offset form is `bits[25:24]=00` — do *not*
  detect it via bit 21, which is the top bit of `imm12` in the unsigned-offset
  form. Mode 0b00 is not only STUR/LDUR either: bits[11:10] select unscaled /
  post-index / pre-index.
- **AdvSIMD structure loads/stores**: writeback is **bit 23**, and `Rm == 31`
  means "increment by the transfer size" while any other `Rm` is a register
  increment. Single-lane forms spread the index across `Q:S:size`, and
  `scale == 0b11` is `LD1R`, not a doubleword lane insert.
- **The permute trio differ**: TRN interleaves the even (or odd) elements of
  *both* operands, ZIP interleaves one half of each, UZP packs every other
  element of Vn low and Vm's high.
- **BSL/BIT/BIF** differ only in the mask register: BSL selects with Vd, BIT
  and BIF with Vm.
- **EXT** shares bits[28:24] with the permute group, so permute must also
  require bit29 == 0.
- **TBL/TBX** share bits[29:21] with the copy group, so copy must let them past
  (every copy encoding sets bit10; table lookup has bit15 == 0 and
  bits[11:10] == 00). The table is `len+1` registers and **wraps past v31**; an
  out-of-range index reads zero for TBL and leaves the byte alone for TBX.
- **Vector FP** lives in two groups the integer three-same decode must not
  swallow: three-same opcodes from `0b11000` up (bits[23:22] are `a:sz`) and
  two-register misc, whose FP forms need `(U, size<1>, opcode)` together —
  opcode `11101` is SCVTF when `size<1> == 0` and FRECPE when it is 1.
- The AdvSIMD **scalar** forms are separate encodings: shift-by-immediate has
  bit28 set, two-register-misc is `01 U 11110 …`.
- **CTR_EL0** reports `0x8444C004`; cache-flush loops stride by `4 << DminLine`.

## Gotchas

- `cargo clippy` **fails** in `switch-wasm` — 28 `not_unsafe_ptr_arg_deref` on
  the deliberate raw-pointer `extern "C"` signatures. Not gating.
- **The crate is hand-formatted: do not run `cargo fmt`.** `--check` is not
  clean either.
- `json_escape` **walks characters, not bytes** — a `\uXXXX` escape names a code
  point, so escaping a multi-byte character per byte spells a different string.
  Above-ASCII goes out as itself; the page decodes with a UTF-8 `TextDecoder`.
- CPU test encodings in `tests/cpu_test.rs` are hand-assembled and cross-checked
  against QEMU's `a64.decode`.
