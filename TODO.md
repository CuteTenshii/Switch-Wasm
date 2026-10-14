# TODO

## Multi-process kernel

Library applets must run as processes beside their caller, in one kernel with one set of console services.
Each stage builds, passes `make test` and clippy, keeps every title in `docs/progress.md` booting to the same frames, and is committed on its own.

1. Per-process handle tables, and shared memory for the shared framebuffer and transfer memory.
2. nvmap handles are console-wide and name a CPU address in the allocating process; a handle another process maps or presents is read from that process's memory at the same address.
3. am focus: the caller is told it lost and regained focus, and `GetCurrentFocusState` answers per process.

## Browser bindings

- Stretch audio when emulation runs below full speed, so it plays slowed down rather than in bursts with gaps.

## Mobile

- Mobile support: layout, touch controls and performance on phones and tablets.

## Open gaps

- `npns:s` cmd 3 (Receive) is refused; qlaunch polls it constantly. Needs the empty-queue result from switchbrew.
- `ssl:s` cmd 5 is stubbed and the `ssl:s` name is not routed like `ssl`.
- The `crates/switch-wasm/src/heap.rs` allocator can be entered concurrently by core workers at startup (unsynchronized dlmalloc under `+atomics`; `__wbindgen_start` allocates their stack and TLS).
- Guest output since the last drain is lost when the run loop dies on a trap or panic, or the session is recycled mid-run (`web/main/runloop.ts` catch path, `recycleSession`).
- The log file is hard to read: per-call `[ipc]`/`[wait]` trace lines, `[thread]` reports repeated every few seconds even when nothing changed, and periodic `[audio]`/`[io]`/`[fs]` stats.
- Minecraft frame 20 is clean on the WebGPU device but the software rasterizer differs from it, mostly in the right half.
