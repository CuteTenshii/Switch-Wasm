# Multi-core

How guest cores run on several workers over one shared memory. The console gives a title three cores; the system worker runs one and two core workers run the others.

## Build

- One wasm module, instantiated by every worker over a shared `WebAssembly.Memory`, so guest RAM, the page table and the emulator's heap are the same bytes everywhere. That needs std built with atomics, hence the pinned nightly in the `Makefile`.
- Function tables are per instance, so an emitted block installed by one worker cannot be called from another: each core keeps its own JIT cache and installs its own blocks.
- `web/worker/core.ts` instantiates the module with `initSync` on the system worker's module and memory.

## State

- `Cpu` is one core: registers, the exclusive monitor, the current thread, the JIT and slice accounting.
- `Machine` is everything else: kernel objects, threads, services and devices. A core reaches it through `Deref`, and only while it holds the kernel lock.
- `Memory` is shared by every core and safe to use from several at once:
  - page table entries are atomic pointers; a first touch allocates with a compare-and-swap, and pages are freed only while every other core is paused;
  - what stores update (mapped-page counts, the cached-pages bitmap, the copy and fill watches) is atomic;
  - the dirty-page lists are behind a mutex, reached only by the first store to a watched page;
  - rare metadata (module and read-only ranges, the soft region, watchpoints) changes only while every other core is paused, so stores read it with plain loads.

## Stages

1. Shared-memory build, core workers that start. Done.
2. Split per-core state from the machine (`Machine` behind `Deref`, done), make `Memory` shareable, then one `Cpu` per core over one `Machine`.
3. Kernel lock, pausing the other cores, cross-core exclusive monitor, multi-core scheduler honouring affinity.
4. Cores 1 and 2 run guest code on their workers; services and the GPU stay on the system worker.
