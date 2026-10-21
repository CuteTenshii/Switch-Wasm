# TODO

## Multi-process kernel

Library applets must run as processes beside their caller, in one kernel with one set of console services.
Each stage builds, passes `make test` and clippy, keeps every title in `docs/progress.md` booting to the same frames, and is committed on its own.

1. Shared memory objects for the shared framebuffer and transfer memory.
2. nvmap handles are console-wide and name a CPU address in the allocating process; a handle another process maps or presents is read from that process's memory at the same address.
3. am focus: the caller is told it lost and regained focus, and `GetCurrentFocusState` answers per process.

## Browser bindings

- Stretch audio when emulation runs below full speed, so it plays slowed down rather than in bursts with gaps.

## Performance

- Blocks with an instruction the emitter lacks stay in the op walk and break their region: on Asphalt 9 the hot one left is `MRS` of `CNTVCT_EL0` (the clock, which needs the host). `LDXP`/`STXP` are also interpreted; the 64-bit pair store compares 16 bytes, which wasm has no atomic for.

## Mobile

- Performance on phones and tablets.

## Open gaps

- The worker's host file table keeps every update and DLC added during a visit, each with up to 16 MiB of chunk cache; it should drop a file the session no longer reads.
- `npns:s` cmd 3 (Receive) is refused; qlaunch polls it constantly. Needs the empty-queue result from switchbrew.
- `ssl:s` cmd 5 is stubbed and the `ssl:s` name is not routed like `ssl`.
- Guest output since the last drain is lost when the run loop dies on a trap or panic, or the session is recycled mid-run (`web/main/runloop.ts` catch path, `recycleSession`).
- The log file is hard to read: per-call `[ipc]`/`[wait]` trace lines, `[thread]` reports repeated every few seconds even when nothing changed, and periodic `[audio]`/`[io]`/`[fs]` stats.
- Minecraft frame 20 is clean on the WebGPU device but the software rasterizer differs from it, mostly in the right half.
