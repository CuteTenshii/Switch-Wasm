# Multi-core

How guest cores run on several workers over one shared memory. The console gives a title three cores; the system worker runs one and two core workers run the others.

## Build

- One wasm module, instantiated by every worker over a shared `WebAssembly.Memory`, so guest RAM, the page table and the emulator's heap are the same bytes everywhere. That needs std built with atomics, hence the pinned nightly in the `Makefile` and the `+atomics`/`--shared-memory` flags in `.cargo/config.toml`.
- `web/worker/core.ts` instantiates the module with `initSync` on the system worker's module and memory, then serves one core of each session (`switch_add_core` on the system worker, `switch_serve_core` here, which blocks until the session stops its cores). `web/worker/coreworkers.ts` starts them and gives each new session its cores.
- Function tables are per instance, so a region compiled by one worker cannot be called from another: each core keeps its own JIT caches, one per process (`Cpu.jit` for the running process, `parked_jits` for the rest), and installs its own regions; `switch-wasm`'s free-slot list is thread-local.
- The global allocator in `crates/switch-wasm/src/heap.rs` is dlmalloc behind a spin lock, since every worker allocates.

## State

- `Cpu` is one core: registers, the exclusive monitor, the running process (`pid`) and its address space (`mem`), the current thread, the JIT caches and slice accounting.
- `Machine` is everything else: kernel objects, handles, services and devices, and the processes by pid. `Cpu` holds it as a `Box<Machine>` and reaches it through `Deref`.
- `Process` is one guest process: its threads, modules, layout and the service objects it opened, plus its `AddressSpace`.
- Each process has its own `Memory`, shared with every core that runs it through an `AddressSpace` (an `Arc<Memory>`).
- `Memory` is safe to use from several cores at once:
  - page table entries are atomic pointers; a first touch allocates with a compare-and-swap, and pages are freed only while every other core has handed its thread back (`quiesce_cores`);
  - what stores update (mapped-page counts, the cached-pages bitmap, the copy and fill watches) is atomic;
  - the dirty-page lists are behind a mutex, reached only by the first store to a watched page; each core has its own code list (`dirty_code_pages(core)`), since each drops its own translations, and one that overflows reports every translation stale;
  - rare metadata (module and read-only ranges, the soft region, watchpoints) changes only while every other core is paused, so stores read it with plain loads.

## Lending

Core 0 (the system worker) is the only kernel: it runs the scheduler, every SVC, the services and the GPU, single-threaded as before. Cores 1 and 2 borrow threads (`kernel/cores.rs`).

- `service_cores`, from the timed-wait sweep every `TIME_SLICE` cycles and from `reschedule`, takes back what the cores returned and lends each idle core the most urgent ready thread whose affinity names it and that core 0 is not running. A `Lent` carries the thread's registers and its process's `AddressSpace`.
- A borrowing core (`Cpu::borrower`, its own JIT caches, a `Machine` it never uses) runs the thread with `run_lent` until an `svc` (left unexecuted, the pc on it), a fault (left on the faulting instruction), its quantum (1M instructions) or a recall, and hands back the registers.
- A more urgent ready thread for a core preempts the one it holds: core 0 sets the core's recall flag, and it hands back at its next slice boundary. Without this a pinned high-priority thread waited out a million-instruction quantum, and Just Dance 2019's frames got slower with cores.
- A thread handed back on an `svc` or a fault is `kernel_pending`: core 0 runs it before anything else, whatever its affinity, and does not lend it again until it has issued its `svc`.
- With extra cores, core 0 does not run threads whose affinity leaves out core 0 and names a core that is present; titles pin their workers this way (Minecraft's streaming pool is `0x6`), and they would otherwise take core 0 from the main thread.
- The clock is core 0's. Instructions on the other cores count as steps but do not advance it, as on the console where every core runs its own 17M cycles a frame; the borrowing cores read the counter from the value core 0 publishes at each sweep (`Cpu.clock`), so a thread's time never runs backwards when it comes home. Counting every core's instructions on the clock made a frame cost three cores' worth of instructions and slowed Minecraft's frames down.
- In a browser the clock also follows real time (`Cpu::pace_to_real_time`, called by `web/worker/loop.ts` before each slice): behind, it jumps forward, so a title too slow for the panel misses frames instead of slowing down; ahead, by idling, the slice ends and the worker waits, at most 50 ms in all, since a long timeout may be all that is left to idle to and core 0 is the only one serving the other cores. While they hold threads, `reschedule` waits for them rather than idling past real time. A slice credits at most 100 ms, so one long stall does not run out every timeout in the title. Native tools and `wasm_bench` never call it and keep the instruction-counted clock, which replays exactly.
- `reschedule`: switch, else take back and lend, else idle the clock to the earliest deadline, else wait for the cores while any holds a thread, and only then wake everything. A running thread that can go on keeps going at each step: idling past it, or waking every waiter while threads were out on other cores, made condition-variable waiters return without their mutex (Asphalt 9's `nn::os` assertion).
- Before memory changes under guest code (the memory SVCs, `ldr:ro` loads and unloads), and before `SetThreadActivity` and `GetThreadContext3`, core 0 recalls every lent thread: Unity's garbage collector pauses every thread and reads its registers to scan stacks, and a lent thread went on running with stale registers.
- The kernel's updates of guest mutex, condition-variable and arbiter words are compare-and-swap loops (`update_word`), since guest code on the other cores changes them with exclusives at the same time.
- `ArbitrateLock` blocks only on exactly `owner | listeners`, as Horizon does. An `svc` from another core arrives late: the owner may have let go and taken the lock back by the fast path meanwhile, and a waiter blocked on a word without the listeners bit is never woken (A Short Hike stalled on such a mutex).
- Measured natively with `NO_DRAW=1`, 3B core-0 instructions, against one core: Asphalt 9 about 3.5x the frames per second, Just Dance 2019 1.8x, A Short Hike 1.5x, Mario Kart 8 Deluxe 1.25x, Minecraft's boot even. In `wasm_bench --cores=2`, whose software rasterizer on core 0 bounds the frame, Just Dance 2019 is even to slightly ahead.

## Ordering

- Exclusive stores are a compare-and-swap against what the exclusive load read (`Memory::compare_exchange`), so another core's store in between makes them fail as on the console; a store that writes back the same value is not noticed. 16-byte pairs have no wasm atomic and are serialised against each other only.
- `DMB`/`DSB` (and A32's CP15 barriers) are host fences, `LDAR`/`STLR` and A32's `LDA`/`STL` are fenced on both sides, in the interpreter and emitted as `atomic.fence`: the host may reorder plain wasm accesses as an ARM host does.

## Next

- The synchronisation SVCs (mutexes, condition variables, the arbiter, events) on the borrowing core under a kernel lock, rather than a round trip through core 0 each.
