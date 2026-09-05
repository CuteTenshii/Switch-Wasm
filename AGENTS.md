# AGENTS.md

Browser Switch emulator: an A64 interpreter with a block-translating JIT, a
software GPU with an optional WebGPU backend, and the container stack
(PFS0/NSP, XCI, NCA, NSO/NRO/ELF, RomFS) needed to boot retail titles and system
applets. Compiled to WASM; frontend is TypeScript on Vite.

This file is the standing state — the rules a change has to keep. PROGRESS.md
logs what broke and what it taught. `docs/` is looked up, not read:
`a64-decode.md` (encodings that are easy to get wrong), `repro.md` (the examples
and their environment switches), `services.md` (the per-service inventory),
`audio.md` and `gpu-backend.md` (the two subsystems with the most detail).

## Commands

- `make all` — `test` then `assets`. `make test` — `cargo test`, 1094 tests.
- `make wasm` — release wasm `--features gpu` + `wasm-bindgen --target web`.
  Needs the `wasm32-unknown-unknown` target and a `wasm-bindgen-cli` matching
  `Cargo.lock`.
- `make assets` — `make wasm` + `vite build` → `dist/`. The only frontend target:
  the core is an *input* to the frontend build, so building `switch-wasm` alone
  does not update the site.
- **`PROFILE=quick` on either builds the same thing in half the time.**
  `release` is `lto = "fat"` and `codegen-units = 1`, which puts a 100k-line
  crate's codegen on one core: 21 s per host example and 44 s per wasm rebuild.
  `quick` is thin LTO over 16 units — 8.6 s and 20 s, a 4.5 MB module instead
  of 4.1 MB, and the same guest speed (min-of-3 on JD2017 to 400 M steps: 3.96 s
  against 4.02 s). Host examples take it as `cargo build --profile quick
  --example <name>`, landing in `target/quick/examples/`. **Ship and quote
  timings from `release`.**
- `bun run dev` (:8000) / `bun run preview` — both need `make wasm` once.
- `bun run typecheck` — the only thing that type-checks; Vite never does.
- `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings`
  — what `.forgejo/workflows/check.yml` runs on every push and pull request.
  `git config core.hooksPath .githooks` adds a pre-commit rustfmt check over
  the staged files; `--no-verify` skips it.
- `python3 tools/difftest.py [--scalar]` — diffs the decode against real ARM
  under `qemu-aarch64`. **Add an instruction here before hand-deriving an
  expected value.** **Two separate harnesses**: bare runs SIMD only, `--scalar`
  the integer table plus the loads and stores through the scratch `x29` points
  at. What belongs in the lists is what titles execute — walk a run's executed
  pcs and name the unknown encodings with `llvm-mc --disassemble
  --triple=aarch64`.
- `--example jit_difftest -- <nro>` — both engines side by side, every state
  difference. The interpreter is the reference.

## Layout

- `switch-core` — the emulator, **zero dependencies**. `cpu` (`mod`/`alu`/`bits`/
  `fp`/`jit`/`loadstore`/`simd`/`svc`/`system`, `ipc`, one module per service
  domain), `gpu`, `display`, `mem`, `vfs`, `source`, `crypto`/`keys`/`ticket`,
  `nsp`/`xci`/`nca`/`romfs`/`npdm`/`nso`/`nro`/`elf`/`lz4`, `control`, `disasm`.
- `switch-gpu` — a `wgpu` backend behind `gpu::renderer::Renderer`, separate
  because `wgpu` brings hundreds of crates and the core has none.
- `switch-wasm` — browser bindings, `cdylib`. Buffers cross via linear memory
  (`switch_alloc`/`switch_free`); a handle indexes a global session table. JSON
  is hand-rolled — **don't add serde**.
- **Services are one module per domain**: `acc`, `am`, `audout`, `audren`,
  `erpt`, `fs`, `hid`, `ldr`, `log`, `mii`, `net`, `ns`, `nv`, `online`, `pl`,
  `power`, `settings`, `time`, `vi`. `ipc.rs` is the marshalling layer plus `sm:`
  and the one-answer services (`csrng`, `spl`, `pm`, `btm`, `nfc`); `svc.rs`
  dispatches by session name. Test request builders: `ipc::testing`.
- **The shipped module is a wasm-bindgen module** (`make wasm` always builds
  `--features gpu`), so the worker imports generated glue and the host read goes
  through `raw_module = "@host/files"` rather than an `env` import — without the
  feature there is exactly one, `env.host_read`, and no `instantiateStreaming`.

## Frontend

`web/` is source, `dist/` is generated, and nothing committed sits beside a build
artifact. `web/public/` is the one verbatim-copied directory; everything else is
content-hashed, so assets are named through the bundler (`import fontUrl from
'../font.ttf?url'`), never as literal paths. `@core` aliases cargo's release dir.
`.forgejo/workflows/pages.yml` deploys `dist/`.

- `web/worker/` — `wasm.ts` (exports + buffer plumbing), `hostfiles.ts` (host
  read, 1 MiB × 32 LRU chunk cache), `latch.ts`, `commands.ts`, `index.ts`.
- `web/main/` — `rpc.ts` plus one module per part of the page; `index.ts` is the
  composition root and owns Reset.
- `log.ts` keeps the log in an array, not in the DOM: the view is capped at
  2000 entries (an instruction trace put tens of thousands of elements on the
  page), Copy/Save read the array, and the tail is mirrored to IndexedDB every
  5 s. A `localStorage` mark cleared on `pagehide` — plus a `BroadcastChannel`
  ping, because the mark is per origin and a second tab is not a dead one —
  is what decides whether the mirrored log is offered back.
- `web/shared/protocol.ts` — the `Commands` interface both sides are checked
  against, so a drifted signature is a build error.
- `runloop.ts`: `RUN_SLICE` 1,000,000, `TRACE_SLICE` 5000, `HOUSEKEEPING_EVERY` 8.
- **`base: './'` in `vite.config.ts` is load-bearing** — the site is published
  under a path, and Vite's default 404s anywhere but a host's root, only once
  deployed. Test by serving `dist/` from a subdirectory.
- **The worker is a module worker and both halves must say so** — `rpc.ts`'s
  `{ type: 'module' }` and `worker: { format: 'es' }`, with the worker named by
  its source URL so the bundler can rewrite the path. No `importScripts`.
- **A `switch_*` call on a freed handle traps the module**, and Reset frees
  without waiting for the slice in flight: `worker/index.ts` refuses all but
  `new`/`set_battery` with no session, `main/index.ts` sets `setSession(-1)`
  before posting the free, `runloop.run` bails once `abortRun` is called.
- **Wasm buffers detach on growth**, so a cached view goes stale; staging buffers
  must be freed or repeated loads overflow linear memory.

## Containers are never staged in memory

A retail `.nsp` runs to gigabytes; wasm32 caps at 4 GiB and Rust refuses any
single allocation over 2 GiB. `source::ByteSource` is a `u64`-addressed
random-access range and the pieces compose: `HostSource` → `Window` (the NCA) →
`SectionSource` (AES-CTR is seekable, so a range costs exactly that range) →
`Window` (the RomFS past the IVFC levels).

- The host read is `(file, offset, ptr, len)` and **must answer synchronously** —
  RomFS ranges are asked for inside `switch_run`, so it must be `FileReaderSync`,
  which exists only in a worker. **File 0 is the open container**; the rest are
  archives (`switch_add_archive`), each with its own chunk cache so a patched
  read does not evict across containers.
- The ExeFS is hash-verified; the RomFS is not, so a wrong byte is believed and
  surfaces hundreds of millions of instructions later. `--example romfs_selftest`
  checks the invariant that needs no reference image: **the bytes of a range must
  not depend on how the range was asked for**. `INJECT=1` plants a boundary bug
  and expects to be told — run it once, or "consistent" means nothing.
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
0xFE00_0000  FB_BASE (lib.rs)         direct framebuffer, 640x360 RGBA
0xFE10_0000  INPUT_ADDR               memory-mapped pad register
0xFF00_0000  GUEST_SPACE_END          above this a read faults

MemoryLayout::PLAIN                 MemoryLayout::VIRTUAL_ADDRESS
0x3000_0000  heap,  3.125 GiB       0x3000_0000  heap,  128 MiB
0xF800_0000  alias, 32 MiB          0x3800_0000  alias, 3.03 GiB
total memory 3.125 GiB              total memory 896 MiB
system resource 0                   system resource 16 MiB
```

- **`FB_BASE` and `INPUT_ADDR` are not console facilities.** A title reaches
  the screen through `vi`/nvnflinger and the pad through `hid`'s shared
  memory; these two are holes in guest memory for a program that has neither.
  `switch_fb_snapshot` reads the GPU's framebuffer and only falls back here.
- **Every region `svcGetInfo` reports must be representable here** — Horizon's
  real bases (alias 0x10_0000_0000) truncate to 0.
- **InfoType 16 picks the layout**, from the title's own NPDM
  `system_resource_size` (`npdm.rs`); both kinds of title are real. VAMM titles
  use `svcMapPhysicalMemory` (0x2c validates, 0x2d unmaps) and never issue 0x01;
  plain titles issue one `svcSetHeapSize` for the whole reported total.
- **`nn::init` asks for the whole reported total**, and titles size pools from it
  against constants in their own code, so a region smaller than the figure is one
  the guest overruns.
  `the_guest_regions_are_disjoint_and_big_enough_for_what_they_promise` holds it
  together. Under VAMM the alias region needs `size >= VAMM_ARENA_SIZE
  (0x3FE0_0000) + VAMM_TOTAL_MEMORY_SIZE + the title's own`; falling short fails
  quietly, because allocators return null and nothing checks.
- **InfoType 21/22** size the application heap; their difference goes to
  `nn::mem::StandardAllocator::Initialize`, which asserts under 16 KiB.
- **InfoType 11 (RandomEntropy) must not be zero** — `sdk` startup `svcBreak`s.
- **InfoType 0/1** come from the NPDM `ThreadInfo` capability: cores 0..2
  (`0b111`), priorities 28..59. Zero asserts in
  `nn::os::RegisterSystemWorkerHandler`.

## Booting

**Homebrew** (`Cpu::boot_homebrew`): run crt0 to the `bl` at entry+0xc0, seed the
env block and `ThreadVars` (TLS+0x1E0, magic `0x21545624`, `_REENT` at
`0x1FF1_0000`), run `DT_INIT_ARRAY`, resume at that `bl`. **Static constructors
only run through this path**, and the pass zeroes registers, so all three
`__libnx_init` arguments are re-seeded before resuming — including `saved_lr` =
`SELF_RETURN_TRAMPOLINE`, without which `__nx_exit` branches to NULL.

**Retail** (`Cpu::boot_retail_program`): `rtld` reads both entry registers
literally. **X0** is the launch argument (0 normally); **X1** is
`MAIN_THREAD_HANDLE` (1), which `nnSdk` stores at ThreadType+0x1B0 and compares
every `SdkMutex` lock word against — X1 = 0 makes an unlocked mutex read as
self-owned and the first `Lock` asserts.

## How long a run has to be

One emulated instruction ≈ one cycle of the 1.02 GHz CPU `svcGetSystemTick` is
scaled from, so **a billion steps is about a second of console time** and a
retail title spends seconds of it before its first frame. Check the run was long
enough before concluding a title does not render, and prefer `SHOT=<f.ppm>` over
reading `frames presented: 0` off a budget that was never going to get there.

**The clock and the step count are not the same number.** `Cpu::cycles` is the
clock and `reschedule` idles it forward when every thread is blocked;
`Cpu::steps` counts retired instructions and the idle never touches it. Both
engines go through `Cpu::retire`, so a *Steps* readout (`switch_get_steps`) that
climbs while the guest is stopped is useless as a sign of life.

## Block translation (`cpu/jit/`)

`ir` is what a block is made of, `cache` which blocks exist and when a guest
store takes one away, `decode` builds them, `exec` runs them. First visit
translates forward into `Op`s, resolving every field the interpreter would
re-read per execution; register 31 is resolved too, so its three meanings (`XZR`
read, `XZR` write, `SP`) are an index into a 34-slot file, not a branch.
Translation runs *through* `b.cond`/`cbz`/`tbz` as `Exit`s checked on the way
past, so only an always-taken branch ends a block, and a `cmp` feeding the next
branch folds into that `Exit` (`fuse_compares`). Staleness is the memory's job: a
dirty bit per page, and a block never spans a page. `SWITCH_NO_JIT=1` for host
tools, the debug panel's *Translation* section in the browser.

- **The shared helper is literal, not aspirational.** An instruction's body is
  written once, keyed on register-file *slots*, and lives with its semantics
  (`alu.rs`; `Acc`/`Ext`/`PairKind`/`Wb` with `access`/`indexed`/`pair` in
  `loadstore.rs`; `SysReg`/`SysOp` in `system.rs`). Two transcriptions of each
  other is what `jit_difftest` watches for.
- **A block must always retire at least one instruction** — `run_jit` advances by
  what `exec_block` reports, so `Ok(0)` spins forever. `jit_test` catches this by
  hanging, not by failing.
- Emitting wasm per block is blocked by the memory model, not the browser: a
  generated module can only address its own linear memory, and guest memory is a
  page table with soft regions, read-only ranges and watchpoints.

## Guest threads (`cpu/mod.rs`, `cpu/svc.rs`)

**Preemptive, on a `TIME_SLICE` of 20,000 instructions**, plus the blocking
syscalls. Between instructions is safe (all state is in `ThreadContext`), and
`yield_thread` is a no-op when nothing else can run.

- **Mutexes and condvars are real.** Horizon keeps the lock word in guest memory
  and libnx re-reads it after every arbitration, so `svcArbitrateUnlock` must
  hand ownership over and `svcWaitProcessWideKeyAtomic` must release the mutex.
  `nn::os` skips the syscall when a condvar's word is zero, so the *kernel* has
  to write `CONDVAR_HAS_WAITERS`.
- **The exclusive monitor is real, and preemption is why.** `LDXR`/`LDXP` set
  `Cpu::exclusive`, `STXR`/`STXP` require and consume it, a switch clears it.
- **Thread stacks live in the stack region**, which must have room or
  `virtmemFindStack` returns 0; `svcMapMemory` must really back the destination
  or two threads share one stack. Pages are not shareable, so the alias is a copy
  (`Memory::copy_range`) copied back by `svcUnmapMemory`. **Nothing of the
  emulator's own may be inside it** — the region stops at
  `SELF_RETURN_TRAMPOLINE`, trampolines and TLS live above.
- **A blocking wait parks, it does not re-ask.** `ThreadState::WaitEvent` holds
  the thread with its PC on the `svc`; `signal_event` wakes parked waiters on an
  event's *transition* only, and the deadline is the display tick.
- **A wait on no handles is not answered** — `MultiWaitImpl::WaitAny` maps any
  answer onto a holder from its own list, so it rewinds onto the `svc` and yields
  unless nothing else can run. **A satisfied wait reports X1 = 0**: X1 is the
  *index* of the signalling handle.
- **A `poll` with a non-zero timeout must yield** — threads only hand over at
  blocking syscalls, so a polling guest loop starves everything else.
- The SVC path retires the instruction *before* dispatching, so a syscall that
  switches threads can install the incoming PC.
- `svcQueryMemory` finds bounds through `Memory::state_run`, which skips 2 MiB
  blocks with no backed page — walking page by page is O(address space).

## Diagnostics

`wasm32-unknown-unknown` has no WASI: `eprintln!` goes nowhere and
`std::env::var` always fails. **`trace.rs` is what the `TRACE_*` switches are**
— a process-global mask the environment *seeds* and `switch_set_trace_mask`
sets, so the page's Diagnostic channels section offers the same nineteen a
shell does. Add a channel to `trace::ALL` and it appears there with no
frontend change. `trace!(Trace::X, ..)` gates; `traceln!` writes to stderr
(natively) and to the sink `Cpu::absorb_traces` folds into the trace buffer —
which is how the rasterizer, the shader translator and the texture decoder,
none of which have a `Cpu`, reach a browser at all.

Anything that must reach a user goes through `Cpu::diagnostic(Level, ..)`.
**The level is part of the line**: a leading `0x01`–`0x04` byte the page maps
to a colour, and an unmarked line inherits the one before it, so a fault's
register dump and instruction trail stay with the fault. The trace buffer is a
**ring** — past 512 KiB the *oldest* goes, and `[trace] the buffer filled` is
written where the loss happened. It used to stop appending instead, which threw
away every fault that came after a full buffer.

`Cpu::backtrace` walks the guest X29 frame chain; `dump_exefs` writes a sorted
`symbols.txt` that turns an address into `sdk!nn::diag::detail::Abort+0x18`.
`Cpu::thread_dump`, `wake_all_blocked` and `start_created_threads` are the
hang levers, and they are exported — a hang is the failure a browser user hits
most.

`switch_crash_report_json` is the bundle an issue needs: build (`build.rs`
bakes the commit), title, cpu, jit, gpu, registers, threads, backtrace, the
`unimplemented`/`stubbed` lists and the trace. It works on a **dead handle**,
because that is when it is wanted.

The backend's own flags stay environment switches, not traces: `GPU_ONLY=<i>`
or `<a>..<b>`, `GPU_DEVICE_MSAA`, `GPU_INTERLEAVE`, `GPU_DEFER_READBACKS`,
`GPU_TIMES`, `GPU_DUMP_WGSL=<dir>`, plus `SWITCH_NO_JIT` and
`NO_VSYNC_THROTTLE`.

## IPC (`cpu/ipc.rs`, `cpu/svc.rs`)

**Two dialects.** libnx ignores the reply's `type` and raw-data size and converts
`fsp-srv` to a **domain** (sub-interfaces are object ids on one handle);
libtransistor validates both and never converts, so a sub-interface must come
back as a **session handle in a move handle**. `Cpu::reply_with_interface` picks
the shape per request.

**Two encodings of every message kind**: plain (`Request` 4, `Control` 5) and
with-context (6, 7), which prefixes a 16-byte tracing context. libnx sends plain,
**`nnSdk` sends the context form for everything** — test with
`Cpu::ipc_is_control_request`, never `type == 5`. A **Close** (type 2) carries no
command id; `svc.rs` answers it before dispatch, or the leftover id runs a real
command.

- **Decode the header once, through `Cpu::ipc_header`.**
- **A map-alias descriptor's address is its low word** — the packed word's
  address bits land above bit 32 and truncate off. `Cpu::ipc_map_descriptor` is
  the one decode.
- **A caller marshals a buffer one of four ways** (map-alias send/receive,
  send-static, receive-static) and a service that reads only the form it expects
  reads nothing. Reach for `ipc_input_buffer`/`ipc_output_buffer`;
  `ipc_pick_buffer` is the rule (`ipc_send_buffer` is the map-alias send side,
  `ipc_recv_static_buffers` the kind that sits *after* the raw data), since
  `cmifRequestInAutoBuffer` fills in both a
  static and a map-alias descriptor and nulls the one it did not choose.
- **Headers and payloads move.** `Cpu::ipc_cmif_header_offset` finds "SFCI" by
  walking the descriptors first (nvdrv's `KICKOFF_PB` puts it at 0x40), and
  `Cpu::ipc_request_data` allows for a domain request's `CmifDomainInHeader`,
  which puts the payload 0x20 in rather than 0x10.
- **`QueryPointerBufferSize` must be non-zero** — `nnSdk` measures a pointer
  argument against it *before* sending. Answered centrally in `svc.rs` as 0x8000.
- **`CloneCurrentObject` (control 2, 4 for Ex) returns a new session handle as a
  move handle**, answered centrally; the clone reaches the same interface and
  inherits the same domain objects. `nnSdk` clones `fsp-srv` before mounting.
- **Mount names live in the guest** — `nn::fs`'s `MountTable` is client-side, so
  `rom:` needs nothing beyond `OpenDataStorageByCurrentProcess` (200) and a
  storage that reads correctly.
- **`IStorage::Read` is `(s64 offset, u64 size)` — not `IFile::Read`**, which
  leads with a `u32 option` and puts its offset at +8.

**The bug class: success with an unfilled out parameter.** A reply is written
*over* the request in the same TLS buffer and declares four words of padding past
the SFCO header, so a bare success passes every length check and hands back stale
*request* bytes. `Cpu::write_ipc_reply` clears the section it declares, so
unimplemented out parameters read as 0 — still wrong, but the same wrong every
time. Implement out parameters; where the width is unclear reply with a zeroed
block **wider** than needed, since a reply may be longer than expected, never
shorter. Same trap in the handle slots: an out-object with no move handle parses
as 0, skips the proxy, returns success and calls through null, so
`Cpu::reply_with_fabricated_object` carries **a sub-session in the move slot and
an event in the copy slot**.

**An unimplemented `am` command must not answer with a bare success** —
everything `am` returns is a live object or applet state, so
`Cpu::unimplemented_command` reports `cmif`'s `UnknownCommandId` (`0x1ba0a`). A
bare success is right for a genuine setter/notifier, but those are listed by id,
never caught by `_`. The two inventories: `Cpu::warn_no_implementation` (`[ipc]
no implementation`) is what a guest asks for and does not get;
`Cpu::warn_stub` (`[ipc] stub:`) is what it is *answered* with nothing behind —
an invented value, a latch nobody records, an event nothing will signal. Not for
a *modelled* answer (one user account, no DLC, no parental controls are true
statements about this console), or the real ones get buried.

**Events reach the guest through the copy list.** A move handle transfers
ownership and an event in the move slot reads back as **0**, and **an event
handed out twice must be the same event** (`Cpu::kept_event`, keyed per
`(purpose, object)`) or the caller waits on something nothing signals. In
`svcWaitSynchronization` a handle not in `Cpu::events` counts as ready (do not
"fix" without checking homebrew), a **poll** on unfired events reports Timeout,
and a **blocking** wait with nothing signalled reports the first handle ready — a
deliberate lie, see `WaitAny`. `vsync_event` fires on a period as well as on
presents.

**One console, one answer.** `am`'s operation mode, `apm`'s performance mode,
`vi`'s display size, the shared buffer's geometry, `clkrst`'s GPU rate and
whether touch reports contacts all derive from one `OperationMode` (the page's
dock switch): 1280x720 handheld / 1920x1080 docked, Normal / Boost, 384 / 768
MHz. Handheld is **0**, and the size must agree everywhere or a caller sizes its
framebuffer from one source and scans out through another; `vi`'s display
queries are dispatched once in `Cpu::vi_common_command` ahead of the
sub-interface match. Host examples default
to handheld while the browser is usually docked — check that before debugging two
frames that disagree. `SHARED_BUFFER_ADDR` is the one surface the whole system
draws into: an applet asks `vi` for a slot rather than owning a layer. Both IPC
routes reach `Cpu::applet_request` — libnx by domain object id, `nnSdk` by a
session handle per interface — so the `am:*` names are also in `svc.rs`'s
dispatch, the same split as `fsp-srv-fs` and `time:system-clock`.

**Answer as the console you actually are**: one user account (uid nonzero — zero
means "nobody is signed in"), an idle temperature, a link up with nothing behind
it, a factory-fresh console. Not a stub, and not a failure, because a failure
puts callers on the path built for hardware that broke. Where a stub would have
to invent something unverifiable it fails instead (`ssl`'s `CreateConnection`,
`sfdnsres`'s definite `EAI_NONAME`, `acc`'s `LoadIdTokenCache`). `Set`/`Get`
pairs are read back. Inventory: `docs/services.md`.

## Audio (`audout`, `audren`, `hwopus`)

`audout` is a *device* — the guest hands it finished PCM; `audren` is a *mixer* —
the guest hands it sources and gets mixed PCM back. Homebrew mostly takes the
first, retail the second, and both end in `Cpu::queue_audio`. **Both play in
time, on the `cycles` clock the display and thread deadlines use**: releasing a
buffer on arrival hands the guest an infinitely fast sound card, and a title's
audio clock is what its video is scheduled against.

- **`audren` runs a frame every 5 ms counted off `cycles`** (`FRAME_CYCLES`),
  never one per update, and `QuerySystemEvent` must hand back a real event as a
  **copy** handle or `audrenWaitFrame` returns instantly and the renderer has no
  clock.
- **`Cpu::audren_parse_update` walks an update by the sizes its header declares,
  never by strides computed here** — that is what makes one parser serve REV1
  through REV15. Layout is libnx's `audren.h`.
- **Do not answer an `audout` buffer wait with a bare success** — `nn::audio`'s
  mixer takes the event as proof a buffer is waiting and reads its queue head
  unchecked.
- **`hwopus` packets are not bare Opus**: each carries an eight-byte
  `{ size, final_range }` header, **big-endian**, and the reply's bytes-consumed
  counts it.

`docs/audio.md` has the rest — the signal path, the update fields that are
load-bearing, what is deliberately not modelled, and how the Opus decoder's
conformance is checked.

## GPU (`switch-core/src/gpu`, `switch-gpu`)

A model of the Tegra X1's GM20B. Registers from deko3d's generated Maxwell
headers, ioctls from libnx's `nvidia/ioctl`.

- `nvdrv` — `/dev/nvmap`, `nvhost-ctrl`, `nvhost-ctrl-gpu`, `nvhost-as-gpu`,
  `nvhost-gpu`. On Tegra the *guest* allocates and hands nvmap a CPU address, so
  GPU memory is ordinary guest memory.
- `vmm` — small-page region at `0x04000000`, big-page from `0x1_00000000`.
- `syncpt` — a submission completes inside its ioctl, so fences are already
  expired when the guest waits.
- `channel` + `engine/*` — GPFIFO → pushbuffer → method headers → the class on
  that subchannel. 3D `0xB197`, compute `0xB1C0`, inline `0xA140`, 2D `0x902D`,
  copy `0xB0B5`, gpfifo `0xB06F`. **Subchannel 6 is pre-bound** to the channel's
  own gpfifo class; userspace never issues `SetObject`.
- `macro_engine` — methods ≥ 0xE00 are macro slots, and deko3d compiles its draws
  into macros, so nothing draws without it.
- `surface` — block-linear (GOB) swizzling. `texture` + `bcn/` cover BC1–BC7 and
  ASTC, the packed 32-bit formats (`A2B10G10R10`, `B10G11R11`) a title samples
  its HDR targets back through, and `ZF32_X24S8`.
- `shader/` — `isa` (SASS decode), `cfg`, `interp` (software shading), `wgsl`.
- `raster` — the software rasterizer, and **the reference every other path must
  agree with**; `qmd` + `compute` are its counterpart for a dispatch.
- Scan-out: `display::BufferQueue` (`QUEUE_BUFFER`) resolves the
  `NvGraphicBuffer` to an nvmap id and `Gpu::present` de-swizzles into
  `Gpu::framebuffer`.

Easy to get backwards:

- **`texs` is the short texture sample, `tex` the general one.** The operands
  `texs` has no room for — explicit level, `.AOFFI` texel offset, shadow
  reference — come out of one register in that order, each present only if its
  own modifier bit is set. The offset is scaled by `TextureSource::texel_step`.
- **`exit` carries a condition-code test beside its predicate**, and both have to
  hold. `EXIT.F` and `EXIT.FCSM_TR` never fire, so they decode as `nop` and the
  walk carries on past them; `bra`'s and `kil`'s field is the same one.
- **Which winding is front is decided in NDC, not on screen.** `SetWindowOrigin`
  bit 4 is the only thing that reverses it — deko3d drives it from
  `windingFlip()` and the viewport's y sign from `viewportFlipY()`, two separate
  flags. `viewport_mirrors` is that correction, applied once per draw.
- **A surface's size comes from `MsaaMode`, never `AntiAliasEnable`**, which is
  GL's `GL_MULTISAMPLE`: coverage evaluated once at the pixel centre
  (`SampleGrid::per_pixel_coverage`), moving no texels.
- **The queue transform is part of the frame.** `QueueBufferInput` carries
  `NATIVE_WINDOW_TRANSFORM_*` at offset 0x20; a title that renders y-down says so
  there rather than by mirroring its viewport, so `Gpu::present` emits the row
  the transform names and refuses `ROT_90`. The applet path
  (`PresentSharedFrameBuffer`) passes `0` — its offset is unverified.

**A compute dispatch runs on the CPU, one thread at a time.** Almost none of a
launch is in the register file — grid, block, constant buffers and shared-memory
size come out of a 256-byte QMD in memory (`clb1c0qmd.h`'s `MW(hi:lo)` fields).
Sequential threads are exact except for `bar` (a thread suspends; the CTA resumes
once all have arrived) and `shfl` (answered once its warp of 32 catches up), and
they make an atomic atomic by construction. Named barriers are not told apart and
`bar.arrive` synchronises like `bar.sync`; both over-synchronise, which no
well-formed kernel can detect. `compute::MAX_DISPATCH_THREADS` refuses a grid
past a million threads.

**A fragment shader that shuffles runs in 2x2 quads.** `shfl` reads a register
belonging to *another* invocation — what `dFdx`/`fwidth` are built out of — so
the four pixels reach the instruction together: `Halt::Shuffle` suspends a lane
the way a barrier suspends a thread and `interp::resolve_shuffles` answers the
whole warp at once. Lanes are `(x, y)`, `(x+1, y)`, `(x, y+1)`, `(x+1, y+1)`;
`sr0` (`SR_LANEID`) is which. The walk is gated on the program containing a
`shfl` or `fswzadd`, since helper lanes are work no other draw does. Neither op
translates to WGSL, so those draws fall back.

### The `wgpu` backend

**It never blocks in a browser** — reading a render target back means awaiting a
promise, and a blocking wait there is not slow but *deadlocked*. A surface stays
on the device across every draw targeting it and returns to guest memory only at
`Renderer::flush`, which answers `Flush::Pending` in a browser so the present
waits for a later slice. **Any draw the backend cannot express falls back to
`Software`**, and **the first late readback latches the whole frame to the
rasterizer** (`Gpu::software_frame`) — on purpose, because alternating is the one
behaviour this must not have. What buys the acceleration back is `shader::wgsl`:
every fallback that latches it is an opcode with no WGSL form, not anything
WebGPU withholds.

**Checking it** is running `screenshot_title` and `screenshot_gpu` over the same
frame and `cmp`ing the PPMs — a byte-identical pair is the only evidence it
renders what the rasterizer does. `GPU_ONLY=<i>` narrows a difference to one
draw, and `gpu::testing::Harness` drives both renderers over a drawable
`Engine3D` without booting a title.

`docs/gpu-backend.md` has the rest — flush and readback ordering, depth, clears,
the two multisampling routes and why the default is the exact one, and the
one-in-255 rounding difference between the two renderers.

## Input (`cpu/mod.rs`, `web/main/input.ts`)

`Cpu::set_gamepad_state` publishes the pad two ways: the `INPUT_ADDR` register
and libnx's `HidSharedMemory`. The offsets came from compiling libnx's
`services/hid.h` and are not guessable: `npad` at **0x9A00**, one entry every
**0x5000**, `full_key_lifo` **+0x28**, `handheld_lifo` **+0x378**, `device_type`
**+0x4188**; each LIFO is a 0x20-byte header then 0x30-byte entries.
`touch_screen` is at **0x400**, a `HidTouchState` is **0x28** bytes with
`finger_id` at +0x0C, `x` at +0x10, `y` at +0x14.

Published in **two slots** — player 1 as a Pro Controller and slot 8 as handheld —
because software polls whichever it expects and `padUpdate` merges them. Buttons
are Horizon's order (A=1<<0 … StickR=1<<5, L=1<<6, ZL=1<<8, Plus=1<<10, d-pad
from 1<<12), `StickL*`/`StickR*` are derived from the analog values, and **stick
Y is positive up** — the opposite of the Gamepad API.

- **A state is read under a seqlock, and bit 0 is the lock.** An entry's sampling
  number is the state's own **doubled**, so the low bit means "being written" and
  `nn::hid`'s reader spins until it clears. `libnx` never looks, so no homebrew
  can show a mistake here.
- **`hid` samples on a clock, not on input.** `Cpu::hid_tick` writes a fresh
  entry into every LIFO every 5 ms, republishing the host's last pad and contacts
  rather than inventing any; without it a title waiting for a newer sample waits
  forever — and a hand on the browser's keyboard hides that, which is why the CLI
  and the browser once disagreed.
- **The run slice is the input sampling period** — the worker is single threaded,
  so a `set_input` posted mid-slice waits for `switch_run` to return.
- The worker latches a press until the **frame counter has advanced twice**, not
  for a fixed number of slices, because the guest polls hid once per iteration of
  a loop that spans many. Only bits the guest may not have seen are latched — a
  still-held key publishes from `heldButtons` alone. `MAX_LATCH_SLICES` (64)
  covers a program that has stopped presenting.
- Touches are reported in the console's own **1280x720 digitizer space**, not the
  presented resolution, and `#screen` is `object-fit: contain` — so a tap maps
  through the **contained rect** or the letterbox bars offset every one.
- **Touch is handheld-only**; docked reports zero contacts whatever the page
  sends. **A lift is a published state with `count = 0`, not silence**, and
  vacated slots are zeroed so no ghost contact is found. `finger_id` is held for
  the life of the pointer so a title can follow a drag.

## Shared system font (`cpu/ipc.rs`, `web/font.ttf`)

Software ships no fonts: it asks `pl:u` by type, gets an offset and size, and
reads out of pl's shared memory (mapping the region, recognised by
`PL_SHMEM_SIZE`, is what fills it).

- **The real fonts come from firmware** — a BFTTF in each system data archive's
  RomFS, decoded by `decode_bfttf` (`Cpu::build_shared_fonts`). With no firmware
  `Cpu::set_shared_font`'s host font stands in for **every** type: an empty set
  means no text renders at all, and registering one face as only the standard
  type is nearly as bad, since the Home Menu looks each character up in each face
  in turn.
- `GetSharedFontInOrderOfPriority` (5, and 6 system-side) must fill its three
  output buffers *and* its count, or a caller reads "loaded, zero fonts" and
  retries forever.
- `tools/make_font.py` builds the shipped subset. It strips TrueType hinting
  because hinted glyphs collapse horizontally under the interpreter (an open bug,
  see PROGRESS.md) and the bytecode costs about **8x** more instructions per
  frame; it also points Nintendo's private-use button codepoints (0xE0E0…,
  0xE0A0…) at matching letters.

## Storage

- **SD card** (`vfs.rs`) — a real path-addressed tree, so `GetEntryType`,
  `OpenDirectory`, `fsDirRead`, `OpenFile` and `Read` agree with each other and a
  missing path is `FsError_PathNotFound` (a fixed listing made menus recurse
  forever). The running NRO is published at `nro::HOMEBREW_NRO_PATH` and
  advertised as `argv[0]`, which is how `romfsMountSelf` works.
- **Saves** — the same filesystem one level in, created empty on open.
- **NAND** — a browser will not hand a page a file it was not asked for, so a
  firmware dump is kept rather than re-picked: the page stores each NCA as a
  `Blob` in IndexedDB and hands it back to `switch_add_archive` as a host file,
  so re-registering a whole dump costs its headers rather than the gigabytes
  behind them. `switch_nand_identify` reads just an NCA header out of a `File`
  the host still holds, and `switch_nand_launch` boots a Program NCA from stored
  bytes — the Home Menu and Mii editor ship as bare NCAs with no NSP to open them
  from. Registering happens off the start-up path.
- All three persist in IndexedDB from the page side (`main/db.ts`, `sdcard.ts`,
  `saves.ts`, `nand.ts`); the core keeps them in memory and reports what changed.
- **An update or DLC is paired, not opened.** Dropping either on the container
  panel registers it against the open title, in either order, since neither file
  is read until Launch. A `File` reference dies on reload, so the page remembers
  which update a title was last launched with and asks for that file again rather
  than quietly running the base version.

## Performance

**One tool holds a clock, and it is not a host example.** The host examples are
x86-64 out of rustc's backend over an unbounded address space; what ships is
wasm32 recompiled by a browser with its own register allocator and a bounds check
on every guest load. hbmenu's boot-to-frame-4 is ~79 M instructions/s natively
and ~59 M/s under V8, and that 1.3x is not a constant — removing a libcall only
wasm pays for was ~1.15x natively and ~1.44x in the browser.

- `node tools/wasm_bench.mjs <nro>` — the artefact `make wasm` produced, timed
  from inside one run through its own wasm-bindgen glue. **The only tool here
  whose milliseconds mean anything**, and with `node --cpu-prof` the only
  profiler for that build.
- `--example frame_work` — one steady frame as work: instructions, block entries,
  fallbacks, methods, draws, clears, copies, pixels. Every count is the same
  under V8.
- `--example jit_coverage` — what a real frame runs that the translator has no op
  for, hottest first, rolled up by encoding group.
- `--example present_work` — scan-out as bytes lifted, deswizzle lookups and
  pixels converted, per layout and crop.
- `--example hotspots` — one steady frame bucketed by guest address; **guest**
  instructions only, so it cannot see the emulator's own cost.

**A change that moves no count and looks faster on the host made this host
faster.** Fix what the work counters name, then confirm under `wasm_bench`.
`perf` on the host still finds *where*, never *how much*.

**The per-pixel and per-instruction paths are where whole seconds hide.** A
full-screen pass is 921,600 fragment invocations, so a small `Vec`, a `HashMap`
for a sparse thing, or a rescan to answer a question costs seconds a frame; the
same goes for anything translating an address per byte. Frames must stay
byte-identical across any such change.

Two costs that are not emulation and keep coming back: `std::env::var` is a
linear scan, so every switch reads once into a `OnceLock` via `env_flag!`; and
`HashMap`'s default hasher is built for adversarial keys while every key here is
a handle or guest address this emulator minted, so `IdMap` is the same map with a
multiply for a hash. A multiply moves entropy **upward** only and `HashMap` picks
buckets from the **low** bits, so `finish` shifts and xors.

An identity map for guest memory does not fit — wasm32 caps at 4 GiB — but
codegen does not need one: only ≤3.125 GiB is ever backed, so slabs in one arena
with a flat `u32` offset table would let generated code translate inline.

## Gotchas

- `cargo clippy` **fails** in `switch-wasm` — 28 `not_unsafe_ptr_arg_deref` on
  the deliberate raw-pointer `extern "C"` signatures. CI gates on `-D
  warnings`, so a deliberate one needs an `allow` naming why.
- **The crate is `cargo fmt`-formatted** at rustfmt's defaults; `rustfmt.toml`
  only pins the edition and style edition so a bare `rustfmt` — an editor's
  format-on-save — agrees with `cargo fmt`. Run it rather than hand-aligning.
- `json_escape` **walks characters, not bytes** — a `\uXXXX` escape names a code
  point. Above-ASCII goes out as itself; the page decodes as UTF-8.
- CPU test encodings in `tests/cpu_test.rs` are hand-assembled and cross-checked
  against QEMU's `a64.decode`; check any new decode against `llvm-mc
  -triple=aarch64 -disassemble` too.
- **PFS0 offsets are counted from the end of the string table**, and the base is
  chosen from the extents — a repack padded to an alignment boundary has no entry
  at offset 0 to detect it from.
