# TODO

## Multi-process kernel

Library applets must run as processes beside their caller, in one kernel with one set of console services.
Each stage builds, passes `make test` and clippy, keeps every title in `docs/progress.md` booting to the same frames, and is committed on its own.

1. `Process` struct: move the per-process fields of `Machine` (handles, threads, layout, modules, AM applet state, session-keyed service objects) into it, still with one process.
2. Memory and JIT per process: `Cpu.mem` points at the running process's `Memory`, the JIT loads it at block entry, threads carry a pid, sync waiters are keyed by `(pid, addr)`.
3. Per-process handle tables, and shared memory objects with aliased pages (hid, `pl` font, shared framebuffer, transfer memory).
4. GPU and display: per-process nvdrv fds, nvmap handles and address spaces; a vi layer table; layers composited by z-order into one frame, in software and wgpu.
5. Process creation in the core: installed applets registered as sources, NCA boot moved from `switch-wasm` into the core, real process ids in svc and pm.
6. am across processes: Start creates the applet process with the pushed storages, out and interactive data reach the other side, exit finishes the caller's accessor, focus messages, the real caller identity.
7. Docs: `kernel.md`, `services.md`, `multicore.md`, `gpu.md`.

## Open gaps

- `npns:s` cmd 3 (Receive) is refused; qlaunch polls it constantly. Needs the empty-queue result from switchbrew.
- `ssl:s` cmd 5 is stubbed and the `ssl:s` name is not routed like `ssl`.
- The log file is hard to read: per-call `[ipc]`/`[wait]` trace lines, `[thread]` reports repeated every few seconds even when nothing changed, and periodic `[audio]`/`[io]`/`[fs]` stats.
- The title icon gets two object URLs for one Blob (`web/main/container.ts` and `web/main/title.ts`), so the browser fetches it twice.
