# Multi-core

How guest cores run on several workers over one shared memory. The console gives a title three cores; the system worker runs one and two core workers run the others.

## Build

- One wasm module, instantiated by every worker over a shared `WebAssembly.Memory`, so guest RAM, the page table and the emulator's heap are the same bytes everywhere. That needs std built with atomics, hence the pinned nightly in the `Makefile` and the `+atomics`/`--shared-memory` flags in `.cargo/config.toml`.
- `web/worker/core.ts` instantiates the module with `initSync` on the system worker's module and memory. Core workers run no guest code yet.
- Function tables are per instance, so a region compiled by one worker cannot be called from another: each core keeps its own JIT caches, one per process (`Cpu.jit` for the running process, `parked_jits` for the rest), and installs its own regions; `switch-wasm`'s free-slot list is thread-local.
- The global allocator in `crates/switch-wasm/src/heap.rs` has no lock, so it is only sound while one worker at a time allocates.

## State

- `Cpu` is one core: registers, the exclusive monitor, the running process (`pid`) and its address space (`mem`), the current thread, the JIT caches and slice accounting.
- `Machine` is everything else: kernel objects, handles, services and devices, and the processes by pid. `Cpu` holds it as a `Box<Machine>` and reaches it through `Deref`.
- `Process` is one guest process: its threads, modules, layout and the service objects it opened, plus its `AddressSpace`.
- Each process has its own `Memory`, shared with every core that runs it through an `AddressSpace` (an `Arc<Memory>`).
- `Memory` is safe to use from several cores at once:
  - page table entries are atomic pointers; a first touch allocates with a compare-and-swap, and pages are freed only while every other core is paused;
  - what stores update (mapped-page counts, the cached-pages bitmap, the copy and fill watches) is atomic;
  - the dirty-page lists are behind a mutex, reached only by the first store to a watched page;
  - rare metadata (module and read-only ranges, the soft region, watchpoints) changes only while every other core is paused, so stores read it with plain loads.

## Stages

1. Shared-memory build, core workers that start. Done.
2. Split per-core state from the machine (`Machine` behind `Deref`) and make `Memory` shareable: done. Remaining: one `Cpu` per core over one `Machine`.
3. Kernel lock, pausing the other cores, cross-core exclusive monitor, multi-core scheduler honouring affinity.
4. Cores 1 and 2 run guest code on their workers; services and the GPU stay on the system worker.
