# TODO

## Multi-process kernel

Library applets must run as processes beside their caller, in one kernel with one set of console services.
Each stage builds, passes `make test` and clippy, keeps every title in `docs/progress.md` booting to the same frames, and is committed on its own.

1. Per-process handle tables, and shared memory objects with aliased pages (hid, `pl` font, shared framebuffer, transfer memory). Until then input reaches whichever process is running when it arrives.
2. GPU and display: per-process nvdrv fds, nvmap handles and address spaces; a vi layer table; layers composited by z-order into one frame, in software and wgpu. Until then the last process to present owns the screen.
3. am focus: the caller is told it lost and regained focus, and `GetCurrentFocusState` answers per process.
4. Docs: `kernel.md`, `services.md`, `multicore.md`, `gpu.md`.

## Browser bindings

- Call browser APIs from Rust through `js_sys` and `web_sys` instead of hand-written JS glue, to avoid needless JS/wasm round trips.

## Open gaps

- `npns:s` cmd 3 (Receive) is refused; qlaunch polls it constantly. Needs the empty-queue result from switchbrew.
- `ssl:s` cmd 5 is stubbed and the `ssl:s` name is not routed like `ssl`.
- The log file is hard to read: per-call `[ipc]`/`[wait]` trace lines, `[thread]` reports repeated every few seconds even when nothing changed, and periodic `[audio]`/`[io]`/`[fs]` stats.
- The title icon gets two object URLs for one Blob (`web/main/container.ts` and `web/main/title.ts`), so the browser fetches it twice.
