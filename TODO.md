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

- Held surfaces are written back and dropped at every flush, so a depth target drawn into without a clear is converted and uploaded again each frame (Just Dance 2019: about 20 uploads a second). Keep a clean held surface until the guest writes to its pages.
- About half the block entries of Asphalt 9 run the op walk, not compiled code, and neither `HOT` nor a larger block cache moved the rate over its first 3B instructions.

## Mobile

- Performance on phones and tablets.

## Open gaps

- The worker's host file table keeps every update and DLC added during a visit, each with up to 16 MiB of chunk cache; it should drop a file the session no longer reads.
- `npns:s` cmd 3 (Receive) is refused; qlaunch polls it constantly. Needs the empty-queue result from switchbrew.
- `ssl:s` cmd 5 is stubbed and the `ssl:s` name is not routed like `ssl`.
- Guest output since the last drain is lost when the run loop dies on a trap or panic, or the session is recycled mid-run (`web/main/runloop.ts` catch path, `recycleSession`).
- The log file is hard to read: per-call `[ipc]`/`[wait]` trace lines, `[thread]` reports repeated every few seconds even when nothing changed, and periodic `[audio]`/`[io]`/`[fs]` stats.
- Minecraft frame 20 is clean on the WebGPU device but the software rasterizer differs from it, mostly in the right half.
