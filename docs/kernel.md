# Kernel model

Guest memory layout, threads, scheduling, synchronization, boot and diagnostics.

## crates/switch-core/src/cpu/mod.rs (guest memory layout)

- Guest addresses are `u32`, so every region shares 4 GiB. `GUEST_SPACE_END` (0xFF00_0000): below is soft-mapped (reads see zeros, writes allocate), above faults on purpose; hbmenu probes the top of the range to size the address space.
- Host stack (`STACK_TOP`, 1 MiB) sits past the ASLR region and before the heap. It used to sit at 0x1000_0000 inside the ASLR region; Mesa/Nouveau's growing buffer-object pool (JKSV) does not re-check allocations with `svcQueryMemory` and overwrote the stack.
- Emulator furniture (return trampolines, TLS blocks) sits just past the ASLR region, not in the stack region: `nn::os::CreateThread` maps stacks anywhere `svcQueryMemory` reports free, and Just Dance 2023 mapped stacks one page short of the main thread's TLS.
- Stack region is 240 MiB because `AddressSpaceAllocatorBase::AllocateSpace` places stacks at random (512 tries, first free range with guard pages). The limit is the longest gap, not free space: at 128 MiB, Just Dance 2023's 39th thread (8 MiB stack, 93 MiB free, longest run 8244 KiB) failed and `MapAliasStack` aborted. A console's region is 2 GiB.
- Heap vs alias region: `nnSdk` uses only one route per process, chosen by the same manifest figure that picks the layout (heap via `svcSetHeapSize` for plain titles and libnx homebrew; alias reservations for virtual-address-memory titles). Each layout spends its address space on the route its titles use. Splitting evenly cost Tomodachi Life 1.25 GiB of heap and its allocator ran dry (null `ThreadType` passed to `CreateThread`). The plain layout's alias region is 32 MiB because Persona 5 Royal needs a heap of at least 3 GiB (pools of 1.73 GiB, 704 MiB and 650 MiB hardcoded in `.data`).
- Horizon's real alias region starts at 0x10_0000_0000, which cannot be represented; reporting it made `svcMapPhysicalMemory` truncate addresses to 0.
- `GUEST_TOTAL_MEMORY_SIZE` (3.125 GiB) is what `nn::init` requests as heap; titles size pools from hardcoded constants (Just Dance 2019's 699 MiB graphics pool), so reporting 480 MiB made allocations fail and titles used the returned null.
- Vamm layout (nonzero `SystemResourceSizeTotal`, e.g. Just Dance 2023 declares 16 MiB, Just Dance 2019 declares 0): `VammManagerImplByHorizon` claims `VAMM_ARENA_SIZE` (0x3FE0_0000, an SDK constant from `movz w9, #0x3fe0, lsl #16`) at the alias base, then `nn::init` reserves `VAMM_TOTAL_MEMORY_SIZE` (896 MiB) in the alias region. Invariant: `VAMM_ALIAS_REGION_SIZE >= VAMM_ARENA_SIZE + VAMM_TOTAL_MEMORY_SIZE + the title's own reservations`. With only 274 MiB left for the last term, Just Dance 2023's block allocator failed, its dlmalloc treated the null as a segment at address 0, and it crashed 30M instructions later at `pc=0`. The heap region is only 128 MiB here (Just Dance 2023 never calls `svcSetHeapSize`). Applying the Vamm layout to plain titles fails quietly: Just Dance 2019 aborts 378.8M steps in when told 896 MiB.
- Library applets (e.g. `LibAppletWeb`) claim a Vamm arena and also call `svcSetHeapSize` for 328 MiB, so they get their own layout with a 512 MiB heap.
- `ldr:ro` modules go between `STACK_TOP` and the heap base (112 MiB), outside every region the guest's allocators might claim.
- System shared buffer pool is laid out at the handheld (720p) layer size regardless of dock state: qlaunch draws at 1280x720 whatever `vi` reports, a display-sized pool gave a docked frame that was the 720p frame plus black, and a pool that moved with the dock relocated slots under an applet still drawing (black screen 13 frames after docking). Address space reserved is the docked size as headroom. Only slots 0 and 1 are handed out, as on a console.
- Operation mode drives resolution, performance mode, GPU clock and touchscreen presence; they must agree (NX-Fetch printed "Docked" beside a handheld resolution when they did not).

## crates/switch-core/src/cpu/mod.rs (threads and events)

- Threads switch only at blocking syscalls, so non-blocking critical sections are atomic.
- `Sleeping` parks a thread with its PC on the `svc` instead of spinning: re-entering the handler for each wait on an audio buffer (tens of millions of cycles) dropped Just Dance from 20M to 1.7M emulated instructions per second.
- `WaitEvent` parks until `signal_event` or the display tick rather than re-polling each slice: two threads waiting on `am:gpu-error` (never signalled on a console) took 70% of Just Dance 2023's retired instructions.
- Events start unsignalled; reporting signalled made `nn::os::TryWaitSystemEvent` tell `nn::oe::GpuErrorHandler` the GPU had faulted.
- `CONDVAR_HAS_WAITERS`: `nn::os::SignalConditionVariable` makes no syscall when the condvar word is zero, so the kernel must write it while a thread waits.
- Shared fonts: `.bfttf` is xored with a key derived from the known plaintext magic; the header's size field stays byte-reversed as on a console. The host fallback font is wrapped with `encode_bfttf` so it takes the same path; padding to whole words is safe since trailing zeros are past every table.

## crates/switch-core/src/cpu/mod.rs (CPU state and scheduling)

- Register file: 256 slots indexed by the `u8` slot byte. Register 31 is resolved at decode time to `ZR_SLOT` (31, never written, reads zero), `ZR_DISCARD` (32, write sink) or `SP_SLOT` (33). Testing for register 31 on every operand cost 12% of a translated frame; a 34-entry array indexed by `u8` needed a bounds check that cost 14% of a retail frame. The cost is 1,776 bytes per saved register file. The discard slot is not guest state (the JIT may skip writes the interpreter makes there), so difftests skip it.
- `CONDITION_MASKS`: a `match` on the condition compiled to a 14-way jump table that mispredicts; a shift-and-mask lookup is branchless (`B.cond` was 12% of an hbmenu frame).
- `cycles` vs `steps`: `cycles` (1.02 GHz, `svcGetSystemTick`) idles forward to the earliest sleeper when nothing can run, so it is not an instruction count; `steps` counts retired instructions only (the "Steps" readout jumped from 24M to 313M on an idle Home Menu when they were one counter).
- TPIDR_EL0 and TPIDRRO_EL0 must be separate: `nnSdk` init writes TPIDR_EL0, and aliasing them clobbered the kernel TLS pointer IPC depends on.
- Vsync fires on a timer (60 Hz) as well as on present: titles wait for vsync before rendering the frame that would fire it. hid samples every LIFO at 200 Hz even without input: Tomodachi Life waits for a sample newer than the last one.
- `TIME_SLICE` (20,000 instructions) preemption: without it an applet's audio thread took 99.9% of instructions and three system applets never reached a frame. Shorter slices (Horizon's 1 ms tick would be about 1000 instructions) cost too much since a switch copies the whole register file including vector registers.
- The exclusive monitor is cleared on context switch so an interrupted read-modify-write fails and retries.
- Service events are kept and returned again on repeat requests: a fresh handle per call leaves the waiter holding an event nothing signals. Several events start signalled because the guest waits on them (`GetApplicationRecordUpdateSystemEvent`, lock accessor, binder).
- AM queues each state change once and then reports no message; answering every poll with a message made `appletMainLoop` reprocess focus changes. Applications get `FocusStateChanged`, applets (Home Menu included) get `ChangeIntoForeground`.
- `pending_yield`/`pending_sleep`: a blocking service call cannot reschedule inside its handler (switching swaps the register file, and the result must land in the caller's X0), so `svcSendSyncRequest` applies it after the reply.
- Settings, options and modes that have getters (auto-sleep, access log mode, npad style, ssl options, apm configuration) are stored so they round-trip; `nnSdk` reads the access log mode at startup.

## crates/switch-core/src/cpu/mod.rs (synchronization and pacing)

- Mutexes: Horizon keeps the lock word in guest memory (owner handle plus `MUTEX_HAS_LISTENERS`); libnx re-reads it after arbitration, so ownership must actually move (returning success from stubs left hbmenu spinning).
- A condvar waiter is always woken holding its mutex (or queued for it), including on timeout: otherwise its next unlock releases a mutex owned by nobody, and `nn::os::UnlockMutex` aborts (the Mii editor's boot ended this way).
- Timed-wait expiry is checked against `cycles`, at most once per `TIME_SLICE`, not on the preemption tick: `slice_used` resets on every switch, so Album's frequently yielding threads kept its main thread's 10 ms sleep from ever expiring.
- `svcWaitForAddress` decides before blocking so X0 is written before a thread switch; the wrong order gave a new thread a zeroed X0 instead of its `ThreadType`.
- Present pacing: the presenting thread is parked until the next 60 Hz refresh. Without it Just Dance 2019 presented every 0.19 ms of emulated time (88 frames per displayed one), leaving loading threads a fifth of the CPU; pacing tripled guest progress.
- Deferred present: a device backend's readback completes only after the host event loop runs, so `vi` keeps the buffer and presents the same surface a slice later. Presenting whatever guest memory held instead came out black for double-buffered titles.
- When nothing can run, the clock idles to the earliest deadline of any kind (sleep, timed wait, display tick), not just the display tick: loading threads sleep in single milliseconds, and idling to the 16.7 ms tick throttled loading.
- The fault trail records runs `(start, count)` instead of per-instruction pairs; a translated block averages seven instructions, so this cuts the per-step cost sevenfold.
- Main TLS base history: 0x0FF0_0000 was overwritten by Mesa/Nouveau's address-range scan (`svcQueryMemory`-based, not via the heap), corrupting `ThreadVars` and failing `__syscall_getreent`'s `BadReent` check.

## crates/switch-core/src/cpu/mod.rs (boot, events, hid)

- Retail boot enters `rtld`, not `main`: `rtld` processes every module's relocations; entering `main` directly leaves its GOT full of placeholders.
- Entry ABI: X1 must be the main thread handle. `nnSdk` stores it in the main `ThreadType` (+0x1B0) and `SdkMutex` compares lock words against it; X1 = 0 made unlocked mutexes look owned and `nn::oe::Initialize` aborted.
- Launch parameters: without a `PreselectedUser`, `nn::account::OpenPreselectedUser` asserts (Just Dance 2019). Library applets get `LibAppletCommonArguments` then their own launch structs; the keyboard and controller applets pop two (stopping after one aborted with `2128-0003`).
- `signal_event` wakes every parked waiter (threads do not record their handles; a wrong set would be a lost wakeup). Only transitions wake: `audio_tick` re-signals due devices on every wait, which otherwise turned parking back into spinning.
- Thread handles are signalled only when the thread exits (`nn::os::WaitThread`). Treating them as always ready let Just Dance 2019 tear down a reused HTTP `ThreadType` while the new thread ran in it.
- Docking queues `OperationModeChanged` and `PerformanceModeChanged` because titles read `GetOperationMode` once; no messages for a no-op change.
- hid: the pad is published as player 1 and handheld, in the requested styles, plus SystemExt (the Home Menu reads only SystemExt and never calls `SetSupportedNpadStyleSet`). Power info must be written (zero reads as an empty battery), for the pad and each half. LIFO sampling numbers are doubled because bit 0 is the seqlock flag.
- Touch: `HidTouchState.attributes` bit 0 `start_touch`, bit 1 `end_touch`. UIs act on these transitions, so a lifted id is published once more with `end_touch`; publishing zero for both made every Home Menu tap do nothing.

## crates/switch-core/src/cpu/mod.rs (execution and diagnostics)

- ADD/SUB: register 31 is SP in the immediate and extended forms but XZR in the shifted-register form; getting it wrong turns `neg x1, x0` into a read of SP and corrupts every `aligned_alloc`.
- `execute` dispatches on bits 28:25 first; trying every group in turn cost about 40 ns per instruction (three quarters of interpreter time on integer code). The full chain is `#[cold]` so the hot dispatcher stays small.
- `BLR x30`: read the target before writing the link register (hbmenu's NEON JPEG IDCT ends with `blr x30`).
- `RET` to 0 is redirected to the exit trampoline: libnx's atexit runner returns with x30 = 0.
- SVC is retired before dispatch because a thread switch installs the incoming thread's PC.
- The trace buffer drops its oldest quarter when full; dropping the newest lost the fault report, which is written last.
- Backtraces only follow frames at or above SP and only report addresses that follow a call: zlib's `inflate_fast` (Just Dance 2019) keeps data in x29/x30 and produced a "return address" of `0x74736964` ("dist").
- On wasm there is no stderr; user-facing diagnostics go through the trace buffer with a level.

## crates/switch-core/src/cpu/mod.rs (hid styles, homebrew boot, misc)

- npad styles: a style the title did not list in `SetSupportedNpadStyleSet` is no pad at all to `nn::hid`; publishing a fixed Pro Controller made titles that accept only Joy-Con pairs abort with `2202-0710`. Presentations are ordered by how much of a dual-stick pad survives. `supported_npad_style_set` is the console's capability and must be the same everywhere (the controller applet gets it in its launch struct and re-asks `hid:sys`).
- `NpadCondition` at 0x3E200 is read directly by `nn::hid::GetNpadJoyHoldType`, which aborts with `2202-0710` unless `is_valid` is set (where the 21.2.0 Home Menu stopped).
- `STARVE_DECISIONS`: Horizon's strict priority rule is safe on three cores but here all threads share one host thread, so a high-priority thread spinning on a lower one would wait forever. The bound gives roughly an 8:1 split.
- Homebrew boot: when the crt0 skips `__libnx_init`, C++ globals stay empty (NX-Shell's SD path resolved to ""), so the emulator runs `.init_array` itself. `__libnx_init`'s x2 is the loader return address used by `__nx_exit`; leaving it 0 made clean exits jump to NULL.
- `csrng` is splitmix64 seeded from the emulated clock; not cryptographic (no entropy on wasm32-unknown-unknown).
- `add_with_carry` avoids `u128` (a libcall on wasm on the hottest path) and avoids branching on `sf` (mispredicts); ADD/SUB/CMP are 15% of a frame.

## `crates/switch-core/src/trace.rs`

- Traces used to be env-var gated and stderr-only, both unavailable on `wasm32-unknown-unknown`. Channels are now a runtime mask (seeded from the environment natively) and output goes to stderr plus a sink the host drains. Channel names are the env var names, so there is one spelling.
- Severity travels in the text stream as a leading control byte (no printable prefix is unambiguous against disassembly or register dumps). Unmarked lines inherit the previous level, so a fault's register dump stays with the fault.
- The global `PENDING` sink exists so code with no `Cpu` in reach (rasterizer, shader translator, texture decoder) can trace without threading a sink through hot free functions. Its length is mirrored in an atomic so the empty case costs a relaxed load. It is capped and drops oldest first (native runs never drain it).
- `trace!` evaluates its format arguments only when the channel is on, which lets it sit in per-syscall and per-draw paths. `traceln!` exists for sites already gated by their own condition.

## `crates/switch-core/tests/trace_test.rs`

- Separate test binary because the sink is process-global and taken (not copied); tests in it are serialized by a lock.
- The trace buffer is a ring that drops from the front: a fault writes its report after everything leading up to it, so dropping from the back lost the crash report.

## `crates/switch-core/tests/cpu_kernel_test.rs`

- Guest memory is addressed with `u32`; every region reported by `svcGetInfo` must be representable (Horizon's real 0x10_0000_0000 alias base would truncate to 0).
- GetSystemTick: 19.2 MHz tick vs 1.02 GHz CPU, about 1/53 tick per instruction.
- InfoType 16 (SystemResourceSizeTotal) non-zero is what puts `nnSdk` on VAMM; titles without an NPDM system resource must read 0. Just Dance 2023 declares 16 MiB, Just Dance 2019 declares 0, and each breaks with the other's layout.
- QueryMemory's MemoryInfo is 40 bytes. Untouched soft pages report unmapped so libnx virtmem finds free space. `.text` must report R-X and following pages RW- as separate regions, or `rtld` mistakes `.rodata` for a module.
- Scheduler: blocking syscalls must write their result to X0 before switching threads; waits on an empty handle set park forever; unsatisfiable waits park instead of re-polling; threads are preempted without syscalls; timed waits are swept independently of context switches; `AppendAudioOutBuffer` and timed `poll` yield the CPU.
- Mutexes: lock word is owner handle plus bit30 when contended; `ArbitrateUnlock` hands ownership over; `WaitProcessWideKeyAtomic` re-acquires on timeout too.
- Layout: regions are disjoint in one 4 GiB space; shared buffer is reserved for docked geometry; plain layout grows the heap region to total memory, VAMM layout fits the SDK arena plus heap reservation in the alias region with at least 512 MiB headroom (274 MiB was not enough for Just Dance 2023). Advertised total must not exceed `MAX_MAPPED_BYTES`.

## `crates/switch-core/src/mem.rs`

- `MAX_MAPPED_BYTES` (3.125 GiB) must never be below `TotalMemorySize` from `svcGetInfo`: titles size pools from it. With a 512 MiB cap and 2.5 GiB advertised, a title reserved 1.5 GiB and died memsetting it. Backing is lazy, so the cap only decides when a runaway fails. Hardware gives an application about 3.2 GiB of 4.
- The page table is a fixed-size boxed array (not `Vec`) so indexing with a masked page number has no bounds check; loads/stores are 31% of a retail frame's instructions.
- `readonly_span`/`module_span` envelopes: `is_readonly` runs on every store; with four modules it cost four comparisons per store (clears write ~11M texels/frame). Protected ranges are low in the image, so almost all stores answer in two comparisons.
- `watched_pages`: one shared bitmap for JIT and GPU caches, because a second test in the hottest write path would cost every store. Both drains see all reports; spurious ones are wasted work only. Allocated on first mark (128 KiB) so runs with neither cache miss on the null check. Fixed-size behind one pointer so emitted wasm stores can perform the exact test. `copy_watch`/`fill_watch` are only reached from `report_written`, so they cost nothing on the hot path. Drains are separate lists so one consumer cannot swallow the other's notification. `gpu_watching` exists because without a GPU backend nobody drained `gpu_dirty` and it grew forever. `fill_watch` is separate from `copy_watch` because a staged blit and a depth clear each write memory the other watches.
- Module memory states: Horizon maps a module as `Code` (`.text`+`.rodata`) and `CodeData` (`.data`+`.bss`). `nn::ro::detail::GetExceptionInfo` (used by every `nn::diag` log and abort) walks `Code` up then requires `CodeData` directly above. Reporting one state killed Asphalt 9 on its first `puts` (377M steps). `rtld` finds modules by scanning for R-X regions, so write-protected `.text` must be reported as its own run. Unmark operations never split partially overlapping ranges.
- `state_run` uses a 2 MiB block summary: the old page-at-a-time walk was O(address space) and titles that query as they allocate spent more time in `svcQueryMemory` than their own code. Queries above the limit describe the single page (hbmenu probes there to size the address space).
- `map_zero` zeroes already-backed pages too (`.bss` shares a page with `.data`, recycled TLS slots, `MapSharedMemory` promises cleared memory), and only exactly the requested range.
- `peek`/`poke` are the JIT fast path and call nothing: under V8 values live across a call are spilled even on the path that skips it.
- `fill_le` writes unit-width stores rather than memcpy from a stamped pattern: in wasm a variable-length copy becomes an out-of-line `memory.copy` call (Just Dance 2019's colour clear: 920k calls/frame, 1.5%). `write_bytes` exists because byte-wise service writes (fsp-srv RomFS reads) made loading Just Dance 2019's JSON as slow as decrypting it. `write_from` serves render target writeback (3.7M texels at 720p 2x MSAA).
- `copy_range` emulates `svcMapMemory` aliasing by copying (and copying back on unmap); libnx only uses one side at a time (e.g. thread stacks mirrored into the stack region).

## `crates/switch-core/src/kernel/svc.rs`

- MapMemory must really back the destination: `virtmemFindStack` picks the next thread's stack mirror by searching for unmapped ranges (a no-op gave every thread the same stack).
- MapPhysicalMemory relies on the soft-mapped low 2 GiB for demand paging; `nn::init` asks for far more than the RAM cap. UnmapPhysicalMemory must really free pages.
- QueryMemory: only module `.text` is executable. Retail `rtld` finds modules by walking for executable `CodeStatic` regions and reading a MOD0 offset; blanket RWX made it relocate itself twice. Non-module mapped memory still reports `CodeStatic`.
- A zeroed hid touch LIFO header is a zero-capacity ring; the Home Menu walked hundreds of millions of entries. Publish an empty sample.
- SleepThread must really sleep (only 0/-1/-2 yield): treating every sleep as a yield turned `nn::os` poll loops into busy-waits (Just Dance 2019 spent 12.7% of instructions in one 15 ms poll).
- WaitSynchronization: results must be written before yielding (register files swap). Polls time out; blocking waits park only while another thread is runnable, because a timeout makes `MultiWaitImpl::WaitAny` return a null holder that `RegisterSystemWorkerHandler` calls. Waits on zero handles rewind onto the `svc` and yield ("A Short Hike"). Vsync and audio-buffer waits park until the known deadline (answering immediately starved Home Menu; answering "satisfied" made Just Dance read a null audio buffer). Reporting index 0 as signalled ran the wrong handler (qlaunch `PopFromGeneralChannel`). Re-asking every slice wasted the whole machine on waits that never complete (`am:gpu-error`).
- GetSystemTick: one instruction is one cycle of the 1.02 GHz CPU, so ~53 instructions per 19.2 MHz tick (the old `cycles * 1000` ran guest time 53,000x fast).
- IPC: Close requests (type 2) and domain Close carry no command id and must be handled before dispatch. CloneCurrentObject must return a real move handle (`nnSdk` clones `fsp-srv` before mounting). QueryPointerBufferSize must not return 0 (`sf` 11-141 abort) nor a fabricated id (callers then use pointer buffers this IPC layer doesn't read).
- GetThreadId must be per-thread (Zelda: Echoes of Wisdom deadlocked when all ids were 1). GetProcessId must return X0 = 0.
- GetInfo: core/priority masks must be non-zero (from NPDM); RandomEntropy must be non-zero; SystemResourceSize non-zero enables `nnSdk`'s VAMM (Just Dance 2023 needs it); TotalNonSystemMemorySize (21) sizes the heap; alias/heap regions must fit in u32 guest addresses.
