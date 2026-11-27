# Kernel model

Guest memory layout, threads, scheduling, synchronization, SVCs, IPC framing, boot and diagnostics.

Section headings are paths under `crates/core/src/`; a heading that starts with `crates/`, `tools/` or `web/` is from the repository root.

## `kernel/layout.rs`

- Guest addresses are `u64` over a 39-bit space (`mem::ADDRESS_SPACE_SIZE`), as Horizon gives a 64-bit application. The emulator's own furniture (modules, stacks, TLS, trampolines, the shared buffer) stays below 4 GiB, so the guest pc and code addresses are `u32`; an indirect branch above 4 GiB faults (`Cpu::branch_target`) rather than truncating. A 64-bit process's space is soft-mapped (reads see zeros, writes allocate) from `GUEST_SPACE_START` to its end; a 32-bit process's stops at `GUEST_SPACE_END` (0xFF00_0000) and faults above, and hbmenu probes the top of the range to size the address space.
- Host stack (`STACK_TOP`, 1 MiB) and emulator furniture (return trampolines, TLS blocks) sit just past the ASLR region, outside the stack region and before the heap. The stack and main TLS used to sit in the ASLR region, where Mesa/Nouveau's growing buffer-object pool (JKSV) and its `svcQueryMemory` scan overwrote them; `nn::os::CreateThread` maps stacks anywhere `svcQueryMemory` reports free, and Just Dance 2023 mapped stacks one page short of the main thread's TLS.
- The stack region is 240 MiB because `AddressSpaceAllocatorBase::AllocateSpace` places stacks at random (512 tries, first free range with guard pages), so the limit is the longest gap, not free space: at 128 MiB, Just Dance 2023's 39th thread (8 MiB stack) found no gap and `MapAliasStack` aborted. A console's region is 2 GiB.
- A 64-bit process gets a console's regions above 4 GiB (`MemoryLayout::WIDE`): a 6 GiB heap region at 64 GiB and a 64 GiB alias region at 128 GiB, so both routes `nnSdk` can take fit at once and no title is patched or given a split. Titles reserve address space far beyond the RAM they commit: Paper Mario: The Origami King's `nninitStartup` reserves 62 GiB (`StandardAllocator::Initialize(nullptr, 0xf80000000)`), Pokémon Legends: Z-A's 12 GiB, and `VammManagerImplByHorizon::Initialize` keeps a 1 GiB bookkeeping arena at the alias base (Minecraft touched 8 KiB of it). With a 4 GiB space these had to be rewritten at load per title, and Z-A, whose reservation no rewrite matched, panicked in `StandardAllocator::Initialize` (162-0001). Tears of the Kingdom declares a system resource (24 MiB) yet also calls `svcSetHeapSize` with `(TotalNonSystemMemorySize & ~0x1fffff) - 0x43800000`, keeping 1080 MiB for the alias route; here both fit, as on a console.
- A 64-bit process whose NPDM declares the 36-bit space gets `MemoryLayout::NARROW` instead: Horizon's 6 GiB heap and 6 GiB alias regions, at 4 GiB and 16 GiB, with the space ending at 64 GiB. Its `nnSdk` packs only 36 address bits into IPC buffer descriptors, so under `WIDE` Breath of the Wild 1.0.0's binder parcels, on a heap at exactly 2^36, reached `vi` as addresses near zero; the empty reply made `android::Parcel::readInplace` return null and `nn::vi::CreateLayer` fault on it.
- `MemoryLayout::system_resource` is the NPDM's `system_resource_size`; nonzero puts `nnSdk` on its Vamm manager (Just Dance 2023 declares 16 MiB, Just Dance 2019 0).
- A 32-bit process keeps everything inside 4 GiB, so `nnSdk` gets one route per process (`MemoryLayout::LOW` for the heap route, `LOW_VAMM` for the alias route). The heap route's alias region is 32 MiB because Persona 5 Royal needs a heap of at least 3 GiB (pools hardcoded in `.data`).
- `GUEST_TOTAL_MEMORY_SIZE` (3.125 GiB) is what `nn::init` requests as heap, or reserves in the alias region. Titles size pools from hardcoded constants (Just Dance 2019's 699 MiB graphics pool); with 480 MiB reported, allocations failed and titles used the returned null. Library applets report `APPLET_TOTAL_MEMORY_SIZE` (512 MiB).
- `ldr:ro` modules go between `STACK_TOP` and `LOW_HEAP_REGION_ADDR` (112 MiB), outside every region the guest's allocators might claim.
- The system shared buffer pool is laid out at the handheld (720p) layer size whatever the dock state: qlaunch draws at 1280x720 whatever `vi` reports, a display-sized pool gave a docked frame that was the 720p frame plus black, and a pool that moved with the dock relocated slots under an applet still drawing. Address space is reserved at the docked size as headroom, since docking can happen after creation. Only slots 0 and 1 are handed out, as on a console.
- `OperationMode` drives resolution, performance mode, GPU clock and touchscreen presence, so they cannot disagree (NX-Fetch printed "1280x720 @ 60Hz [Docked]" when they did).

## `kernel/memory_use.rs`

- Regions are counted with `Memory::mapped_bytes_in`, which reads the per-2 MiB block counts wherever a whole block fits, so a report costs about two thousand loads, not a walk of a million pages. Every layout's regions are disjoint (tested); what falls outside them is reported as `elsewhere`, not dropped.

## `kernel/process.rs` and `kernel/program.rs`

- A `Process` holds what one guest process owns: its `AddressSpace`, threads, modules, memory layout, program id and the service objects it opened. `Machine.processes` keeps them by pid; `Cpu.pid` names the one the core runs, `enter_process` switches the core to another, and `in_process` runs kernel bookkeeping in another's address space and comes back.
- Handles, events, saves and devices stay console-wide on `Machine`.
- `Machine.root` is the running title's pid; its exit halts the console. Any other process ends once the `svc` that ended it has replied (`process_ending`, acted on in `syscall`).
- `end_title` ends the title and every process it started, leaving an empty process as the new root for the boot paths to load the next title into. It also resets what only meant something to the ended title: the display queue, the shared buffer, npad styles, the frame count, queued error dialogs and audio, and the once-per-request warnings. Storage, users, settings, keys and archives stay.
- `end_process` finishes every thread, reports the ending through the diagnostic channel, finishes the caller's library applet accessor, reschedules away and forgets the process's translations.
- `create_process` starts from an empty `AddressSpace` and runs `map_runtime` (soft-mapped space, host stack, trampolines); nothing runs until `start_main_thread` adds a main thread at the entry with the loader's X0/X1.
- `spawn_installed` starts a program registered with `install_program` beside the running one: it adopts the program's NPDM settings, loads its modules and starts the main thread. Only 64-bit programs start this way; A32 runs only as the booted title. `boot_program` boots a `Program` into the running process instead.
- Each process has its own handle table (`Process.handles`, with a count per handle), filled wherever a handle reaches the guest: IPC replies, `ConnectToNamedPort`, `CreateThread`, `CreateTransferMemory`. Values come from one console-wide counter, so maps keyed by a handle stay valid and never collide. `CloseHandle` closes a session once no process holds it. When a process ends, the sessions only it held close, and events go once their owner is gone and no live process holds them; console events (`alloc_console_event`: vsync, binder, npad style, ns and fs notifications) stay. Events outlive the guest's `CloseHandle` because services hand the same event out again.
- Shared memory objects are `SharedPages` blocks several address spaces map. The system shared buffer is one, covering the usable slots and mapped at `SHARED_BUFFER_ADDR` the first time a process uses the `vi` shared-buffer commands, so every process draws on the same pages. `CreateTransferMemory` turns the creator's range into a block (its contents kept) and keeps it by handle until no process holds the handle; the creator keeps access, which a console would revoke while the memory is mapped elsewhere. A `CreateTransferMemoryStorage` storage reads the block's contents. hid's and `time`'s shared memory are console-wide blocks too, recognised in `svcMapSharedMemory` by the handle their service gave (hid's also by `HID_SHMEM_SIZE`); `pl`'s is recognised by `PL_SHMEM_SIZE` and mapped zeroed with the shared font written in.
- Each core keeps translated code per process: `Cpu.jit` is the running process's cache and `parked_jits` holds the others by pid, since an address means different code in each process.
- The scheduler picks across every process's threads; switching to a thread of another process goes through `enter_process`, which swaps the address space and JIT cache. When nothing can run, timed waits expire and waiters wake in every process.

## `cpu/regs.rs`

- Register file: 256 slots indexed by the `u8` slot byte. Register 31 is resolved at decode time to `ZR_SLOT` (31, never written, reads zero), `ZR_DISCARD` (32, write sink) or `SP_SLOT` (33). Testing for register 31 on every operand cost 12% of a translated frame; a 34-entry array indexed by `u8` needed a bounds check that cost 14% of a retail frame. The price is 1,776 extra bytes per saved register file. The discard slot is not guest state (the JIT may skip writes the interpreter makes there), so difftests skip it.
- `CONDITION_MASKS`: a `match` on the condition compiled to a 14-way jump table that mispredicts; a shift-and-mask lookup is branchless (`B.cond` was 12% of an hbmenu frame).
- `add_carry_overflow` avoids `u128` (a libcall on wasm, on the hottest path) and a branch on `sf` for the carry chain; ADD/SUB/CMP are 15% of a frame.

## `cpu/mod.rs`

- `cycles` (1.02 GHz, `svcGetSystemTick`) idles forward to the earliest deadline when nothing can run, so it is not an instruction count; `steps` counts retired instructions only (as one counter, the "Steps" readout jumped from 24M to 313M on an idle Home Menu).
- TPIDR_EL0 (`tpidr_rw`) and TPIDRRO_EL0 (`tpidr`) must be separate: `nnSdk` init writes TPIDR_EL0, and aliasing them clobbered the kernel TLS pointer IPC depends on.
- `pending_yield`/`pending_sleep`: a blocking service call cannot reschedule inside its handler (switching swaps the register file, and the result must land in the caller's X0), so `svcSendSyncRequest` applies it after the reply.
- The fault trail records runs `(start, count)` instead of per-instruction pairs, cutting the per-step cost by the average run length.

## `kernel/sched.rs`

- `TIME_SLICE` (20,000 instructions) preemption: without it an applet's audio thread took 99.9% of instructions and three system applets never reached a frame. Shorter slices (Horizon's 1 ms tick would be about 1,000 instructions) cost too much, since a switch copies the whole register file, vector registers included. The JIT preempts only between blocks.
- The exclusive monitor is cleared on context switch so an interrupted read-modify-write fails and retries.
- `STARVE_DECISIONS`: Horizon's strict priority rule is safe on three cores, but here core 0 runs every thread not lent to another core on one host thread (and a native run may have no other cores), so a high-priority thread spinning on a lower one would wait forever. The bound gives roughly an 8:1 split.
- `Sleeping` parks a thread until a deadline instead of spinning. `svcSleepThread` and a service's `pending_sleep` resume after the `svc`; a wait on an audio buffer and a retried service call rewind onto it, so it is reissued on wake. Re-entering the handler for each wait on an audio buffer dropped Just Dance from 20M to 1.7M emulated instructions per second.
- `WaitEvent` parks until `signal_event` or its deadline rather than re-polling each slice: two threads waiting on `am:gpu-error` (never signalled on a console) took 70% of Just Dance 2023's retired instructions.
- Vsync fires on a timer (60 Hz) as well as on present, since titles wait for vsync before rendering the frame that would fire it. hid samples every LIFO at 200 Hz even without input: Tomodachi Life waits for a sample newer than the last one.
- `pace_present` parks the presenting thread until the next 60 Hz refresh. Without it Just Dance 2019 presented many frames per displayed one and starved its loading threads; pacing tripled guest progress.
- Deferred present: a device backend's readback completes only after the host event loop runs, so `vi` queues the buffer with the readback generation it waits on and presents the same surface a slice later. Up to two frames' readbacks are in flight (`Renderer::submit_frame`, `frames_landed`); a present that finds both slots full is dropped, never an older queued one. Presenting whatever guest memory held instead came out black for double-buffered titles.
- When nothing can run, the clock idles to the earliest deadline of any kind (sleep, timed wait, display tick), not just the display tick: loading threads sleep in single milliseconds, and idling to the 16.7 ms tick throttled loading.

## `kernel/sync.rs`

- Mutexes: Horizon keeps the lock word in guest memory (owner handle plus `MUTEX_HAS_LISTENERS`, bit 30, when contended); libnx re-reads it after arbitration, so ownership must actually move (stubs returning success left hbmenu spinning).
- `CONDVAR_HAS_WAITERS`: `nn::os::SignalConditionVariable` makes no syscall when the condvar word is zero, so the kernel must write it while a thread waits.
- A condvar waiter is always woken holding its mutex (or queued for it), timeout included: otherwise its next unlock releases a mutex owned by nobody and `nn::os::UnlockMutex` aborts (the Mii editor's boot ended this way).
- Timed-wait expiry is checked against `cycles`, at most once per `TIME_SLICE`, not on the preemption tick: `slice_used` resets on every switch, so Album's frequently yielding threads kept its main thread's 10 ms sleep from ever expiring.
- `svcWaitForAddress` decides (`arbitrate_address`) before blocking so X0 is written before a thread switch; the wrong order gave a new thread a zeroed X0 instead of its `ThreadType`.

## `kernel/events.rs`

- Events start unsignalled; reporting signalled made `nn::os::TryWaitSystemEvent` tell `nn::oe::GpuErrorHandler` the GPU had faulted.
- `signal_event` wakes every parked waiter (threads do not record their handles; a wrong set would be a lost wakeup). Only transitions wake: `audio_tick` re-signals due devices on every wait, which otherwise turned parking back into spinning.
- Thread handles are signalled only when the thread exits (`nn::os::WaitThread`). Treating them as always ready let Just Dance 2019 tear down a reused HTTP `ThreadType` while the new thread ran in it.
- Docking queues `OperationModeChanged` and `PerformanceModeChanged` because titles read `GetOperationMode` once; a no-op change queues nothing.

## `kernel/boot.rs`

- Retail boot enters `rtld`, not `main`: `rtld` processes every module's relocations, and entering `main` directly leaves its GOT full of placeholders.
- Entry ABI: X1 must be the main thread handle. `nnSdk` stores it in the main `ThreadType` (+0x1B0) and `SdkMutex` compares lock words against it; X1 = 0 made unlocked mutexes look owned and `nn::oe::Initialize` aborted.
- A library applet booted directly gets `LibAppletCommonArguments` then its own launch structs (an applet started by `am` gets what its caller pushed); the keyboard and controller applets pop two, swkbd's second a 0x1000 work buffer (stopping after one aborted with `2128-0002`).
- Homebrew boot enters the crt0 at its entry, as HBL does; `__libnx_init` (the crt0's `bl` at entry+0xc0) sets up the heap and `ThreadVars`, then runs `.init_array`. The emulator once ran the constructors itself before that call, so they ran twice, the first time before the heap: newlib's `sbrk` latched the end of BSS and grew through the soft-mapped stack region, and Taisei's `memset` zeroed a thread's stack. x30 at entry is the trampoline, which the crt0 hands `__libnx_init` as the loader return for `__nx_exit`.

## `kernel/svc.rs`

- MapMemory must really back the destination: `virtmemFindStack` picks the next thread's stack mirror by searching for unmapped ranges (a no-op gave every thread the same stack).
- MapPhysicalMemory relies on the soft-mapped address space for demand paging; `nn::init` asks for far more than the RAM cap. UnmapPhysicalMemory must really free pages.
- QueryMemory past the end of the address space answers one `Inaccessible` region from the end to 2^64, as Horizon does: retail `rtld` walks regions until the address wraps, and describing one page at a time there kept Z-A's `rtld` walking for billions of instructions.
- QueryMemory: `MemoryInfo` is 40 bytes. Only module `.text` is executable, reported R-X as its own run with the following pages RW-: retail `rtld` finds modules by walking for executable `Code` (libnx `CodeStatic`) regions and reading a MOD0 offset, so blanket RWX made it relocate itself twice. Untouched soft pages report unmapped so libnx virtmem finds free space; the heap `svcSetHeapSize` granted reports `Normal` as one run, touched or not; other mapped memory reports `Code`.
- SleepThread must really sleep (only 0/-1/-2 yield): treating every sleep as a yield turned `nn::os` poll loops into busy-waits (Just Dance 2019 spent 12.7% of instructions in one 15 ms poll).
- WaitSynchronization: results must be written before yielding (register files swap). Polls time out; blocking waits never do: they rewind onto the `svc` and park until a signal or the display tick, because a timeout makes `MultiWaitImpl::WaitAny` return a null holder that `RegisterSystemWorkerHandler` calls. Waits on zero handles park the same way until `svcCancelSynchronization`, which answers the thread's next wait once with cancelled (spinning on them stalled Just Dance 2022's media threads). Vsync and audio-buffer waits park until the known deadline (answering immediately starved Home Menu; answering "satisfied" made Just Dance read a null audio buffer). Reporting index 0 as signalled ran the wrong handler (qlaunch `PopFromGeneralChannel`).
- GetSystemTick: one instruction is one cycle of the 1.02 GHz CPU, so about 53 instructions per 19.2 MHz tick.
- Close requests (type 2) and domain closes carry no command id, so they are handled before dispatch; scanning for SFCI would find the previous request's id (the Home Menu's `IStorage` close ran as a Read). `ssl` context counts and Opus decoder state are released on close.
- `CloneCurrentObject` must return a new session move handle; `nnSdk` clones `fsp-srv` before `MountRom`.
- GetThreadId must be per-thread (Zelda: Echoes of Wisdom deadlocked when all ids were 1).
- GetInfo: core/priority masks must be nonzero (the core mask from NPDM, the priority mask fixed at 28..=59); RandomEntropy must be nonzero; SystemResourceSizeTotal (16) nonzero enables `nnSdk`'s Vamm manager and must read 0 for titles whose NPDM declares none; TotalNonSystemMemorySize (21) sizes the heap. Advertised total memory must not exceed `MAX_MAPPED_BYTES`.
- `svcResetSignal` must fail when nothing was signalled; succeeding unconditionally made drain loops spin.
- `svcSynchronizePreemptionState` does nothing: the kernel only unpins a pinned thread, and threads are never pinned here. Just Dance 2023 reaches it from `AK::EventManager`.
- `svcBreak` with `0x80000000` set is a notification, such as `PreLoadDll` before a module loads, and returns success. Ending the process on it killed Shop as it loaded its first module.

## `kernel/ipc.rs`

- `POINTER_BUFFER_SIZE` is 0x8000 (what a real `fsp-srv` reports), and `QueryPointerBufferSize` answers it for every session before dispatch. `nnSdk` measures pointer arguments against it before sending; with 0 it refused explicit `HipcPointer` arguments (`sf` 11-141 `PointerBufferTooSmall`, a Tomodachi Life abort). Both descriptor forms land in the same address space, so the number only picks which descriptor callers fill.
- AutoSelect buffers: `cmifRequestInAutoBuffer` fills a static and a map-alias descriptor and nulls the unused one, so services must use `ipc_input_buffer`/`ipc_output_buffer`. Reading only the map-alias side handed `nvdrv` nothing once ioctl args went through pointers.
- Header counts are decoded in one place (`HipcHeader`, `ipc_header`): separate walks each re-derived them, and one skipped a special header's pid but not its copy/move handles (four bytes short per handle).
- TIPC (12.0.0+ `sm:`): command id in the type field as `16 + cmd`, arguments directly in the data area, no SFCI, no 16-byte alignment, no domains. Without it, `cabinet`'s first request had no command id and the applet aborted.
- The SFCI header position depends on descriptors: nvdrv `KICKOFF_PB` lands at 0x40; a fixed scan missed it, GPU submits got a generic success, and hbmenu's fence never signalled. Domain requests push the payload to 0x20; assuming 0x10 made `fsFileRead` read 0 bytes at offset 0 and broke `romfsMountSelf`. Pre-CMIF libnx sessions (NX-Shell's `fsDirRead`) send `{type=2, object_id, cmd_id}` without SFCI.
- Descriptor addresses are decoded whole: a map-alias descriptor's bits 32..36 and 36.. sit in its third word, a send-static's in its packed word, a receive-static's in the high half of its second. Buffers live wherever the guest allocated them, in the heap and alias regions above 4 GiB.
- Receive-static descriptors sit after the raw data (`IProfile::Get`'s `AccountUserData`).
- Copy and move handles live in different descriptor fields, in that order; a copy handle in the move slot reads as 0 (`nnSdk` waited on handle 0 after `GetGpuErrorDetectedSystemEvent`; `vi`'s vsync event once did the same).
- The reply type field must be 0: libtransistor rejects anything but 0 or 4 (error 0x7E0DD; a libtransistor title failed to open fsp-srv).
- Replies are written over the request, so the declared section (header, raw data and four padding words) is zeroed first. An empty success on a command with an out parameter still hands back stale request bytes past that: `ListDisplayModes` read its count from the previous request and cost the Home Menu a billion instructions, and `set:sys` blocks wider than 16 bytes (`TvSettings` 0x20, `NotificationSettings` 0x18) leaked a NaN gamma. Write the full out width, or refuse.
- Plain sessions (libtransistor never converts to domains) need a real moved session handle for sub-interfaces; domain objects broke `fsp_srv_open_sd_card_filesystem`.
- Control messages share command ids with service commands, so `ipc_is_control_request` checks them first: `nnSdk` sends them as type 7 (with context) and requests as 6, and testing `== 5` ran `appletOE`'s and the Home Menu's `QueryPointerBufferSize` as command 3 (`set` 3 told `nnSdk` its pointer buffer was 18 bytes, aborting Just Dance 2017; on `vi` it ran a binder transaction). Domain-ness is the domain header's type byte, not the hipc type.
- `kept_event`: a second ask must return the same event or the caller waits on an unsignalled copy. Most are never signalled, describing things that never happen here.
- A command an implemented service does not know is refused with cmif 10-221 (`unimplemented_command`); a blanket success made `nn::oe::SetupGpuErrorHandler` wait on handle 0. A service with no implementation at all gets `reply_with_fabricated_object`: a real sub-session or domain object, the raw id, and an unsignalled copy event, reused per `(session, command)`, because the intended out type is unknown. A missing move handle made boot2 call through a null `SharedPointer` after `gpio`'s `OpenSession2`, and a missing event stalled the Home Menu's message thread.
- `warn_stub` marks answers with nothing behind them (invented values, unrecorded latches, never-signalled events), once per `(interface, command)` via the diagnostic channel; true emulated facts (the host's users, no DLC, no network) are not marked. Refusals, missing services and stubs are also counted for `Cpu::take_service_gaps`.
- `btm:sys`'s `GetCore` must be real because every other command goes through it; the radio flag is `set:sys`'s Bluetooth flag. The `nfc:sys` enabled flag is `set:sys`'s NFC flag; device commands are refused since no device handle was ever handed out. `ngc` must write the output text (callers otherwise read uninitialised buffers) and return a nonzero content version. `npns` `Receive`/`ReceiveRaw` are refused (the empty-queue error is undocumented).

## `mem.rs`

- `Memory` is one process's address space. A core holds it as an `AddressSpace`: an `Arc<Memory>` shared with the process, plus the raw `Memory` address first so emitted code reaches it in one load (`MEMORY_AT`).
- `MAX_MAPPED_BYTES` (3.125 GiB) must never be below `TotalMemorySize` from `svcGetInfo`, since titles size pools from it: with a 512 MiB cap and 2.5 GiB advertised, a title reserved 1.5 GiB and died memsetting it. Backing is lazy, so the cap only decides when a runaway fails. Hardware gives an application about 3.2 GiB of 4.
- The page table has two levels: `blocks`, a fixed-size boxed array (not `Vec`, so a masked index has no bounds check) of 2^18 entries for 2 MiB blocks, each a `Block` of 512 page pointers and its bitmaps. A flat table for 39 bits would be 512 MiB on wasm32. Untouched entries point at one shared empty block, never null, so a walk (and emitted code) tests only the page entry; loads and stores are 31% of a retail frame's instructions. Address bits above the 39-bit space are dropped, the same in `Memory` and in emitted code.
- `span_blocks` counts blocks with a backed page per 1 GiB, so `state_run`, `mapped_bytes_in`, `unmap` and loader dirty-range walks skip untouched space: `UnmapPhysicalMemory` over tens of GiB, and `svcQueryMemory` over the empty heap and alias regions, would otherwise visit millions of pages.
- `readonly_span`/`module_span` envelopes: `is_readonly` runs on every store, and with four modules it cost four comparisons per store (clears write about 11M texels a frame). Protected ranges are low in the image, so almost all stores answer in two comparisons.
- `Block::watched`: one bitmap for JIT and GPU caches, because a second test in the hottest write path would cost every store. Beside it `code` and `gpu` say whose page it is, so each drain hears only of its own (GPU pages filling the code lists overflowed them, and an overflow marks every translation stale). The bitmaps live in the blocks, so a page's bit is one load past its block, which emitted stores already hold, and nothing scales with the 39-bit space. `copy`/`fill` are only reached from `report_written`, so they cost nothing on the hot path, and are separate because a staged blit and a depth clear each write memory the other watches. Drains are separate lists of page indices so one consumer cannot swallow the other's notification. `gpu_watching` exists because without a GPU backend nobody drained `gpu_dirty` and it grew forever.
- Module memory states: Horizon maps a module as `Code` (`.text`+`.rodata`) and `CodeData` (`.data`+`.bss`). `nn::ro::detail::GetExceptionInfo` (used by every `nn::diag` log and abort) walks `Code` up, then requires `CodeData` directly above; reporting one state killed Asphalt 9 on its first `puts`. Unmark operations never split partially overlapping ranges.
- `state_run` skips untouched 2 MiB blocks (`Block::mapped`) and 1 GiB spans: a page-at-a-time walk was O(address space), and titles that query as they allocate spent more time in `svcQueryMemory` than in their own code. Queries above the limit describe the single page; `svcQueryMemory` answers past the space's end itself.
- `map_shared` points a range's page-table entries at a `SharedPages` block, so several address spaces share those pages; the mapping keeps the block alive, and unmapping or dropping the address space never frees its pages.
- `map_zero` zeroes already-backed pages too (`.bss` shares a page with `.data`, recycled TLS slots, `MapSharedMemory` promises cleared memory), and only exactly the requested range.
- `peek`/`poke` are the JIT fast path and call nothing: under V8, values live across a call are spilled even on the path that skips it.
- `fill_le` writes unit-width stores rather than memcpy from a stamped pattern: in wasm a variable-length copy becomes an out-of-line `memory.copy` call (Just Dance 2019's colour clear: 920k calls a frame, 1.5%). `write_bytes` exists because byte-wise service writes (fsp-srv RomFS reads) made loading Just Dance 2019's JSON as slow as decrypting it. `write_from` serves render target writeback.
- `copy_range` emulates `svcMapMemory` aliasing by copying (and copying back on unmap); libnx only uses one side at a time (e.g. thread stacks mirrored into the stack region).

## `cpu/exec.rs` and `cpu/debug.rs`

- `execute` dispatches on bits 28:25 first; trying every group in turn cost about 40 ns per instruction (three quarters of interpreter time on integer code). The full chain (`execute_chain`) is `#[cold]` so the hot dispatcher stays small.
- `RET` to 0 is redirected to the exit trampoline: libnx's atexit runner returns with x30 = 0.
- SVC is retired before dispatch because a thread switch installs the incoming thread's PC.
- The trace buffer drops its oldest quarter when full: a fault writes its report after everything leading up to it, so dropping the newest lost the report.
- Backtraces only follow frames at or above SP and only report addresses that follow a call: zlib's `inflate_fast` (Just Dance 2019) keeps data in x29/x30 and produced a "return address" of `0x74736964` ("dist").

## `kernel/thread_report.rs`

- `THREAD_TYPE_SIZE` was measured on Just Dance 2017, which allocates its `ThreadType`s back to back (consecutive entry arguments are 0x1C0 apart); searching further would pick up the neighbour's name.
- A thread's name is found as the first word in its `ThreadType` pointing back into the struct at readable text, since the offset varies by SDK version. Unnamed threads (`Thread_0x...`) print their entry function, since all share the SDK trampoline.
- Stack walks take the backtrace rule above: only frames whose return address follows a call.

## `trace.rs`

- Channels are a runtime mask (set by the host: from `--trace` in `common::parse` by the examples, all on or off with the log level by the page) and output goes to stderr plus a sink the host drains, since environment variables and stderr do not exist on `wasm32-unknown-unknown`. Channel names (`TRACE_IPC`) are the one spelling; `--trace` takes them without the prefix.
- Severity travels in the text stream as a leading control byte (no printable prefix is unambiguous against disassembly or register dumps). Unmarked lines inherit the previous level, so a fault's register dump stays with the fault.
- The global `PENDING` sink lets code with no `Cpu` in reach (rasterizer, shader translator, texture decoder) trace without threading a sink through hot free functions. Its length is mirrored in an atomic so the empty case costs a relaxed load. It is capped and drops oldest first (native runs never drain it).
- `trace!` evaluates its format arguments only when the channel is on, so it can sit in per-syscall and per-draw paths. `traceln!` is for sites already gated by their own condition.
- `tests/trace_test.rs` is a separate binary because the sink is process-global and taken (not copied); its tests are serialized by a lock.
