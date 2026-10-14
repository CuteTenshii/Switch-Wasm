# Web frontend and worker

The page, the worker, and the wasm bindings between them.

## `crates/switch-wasm/src/lib.rs`

- Exports are `extern "C"` taking pointers JS owns for the call; marking them `unsafe fn` would not help a JS caller, hence `#![allow(clippy::not_unsafe_ptr_arg_deref)]`.
- The session table uses `SyncCell` (an `UnsafeCell`), not `Mutex`: a reentrant `lock()` deadlocks and a panic while held leaves it locked. It is sound only because one worker calls exports; the core workers on the shared memory only instantiate. Host tests serialize on a `HOST` mutex because `cargo test` is multi-threaded, and a panic crossing `extern "C"` aborts the harness.
- The panic hook writes to a fixed 2048-byte buffer (512 was too small for payload, location and assertion values), truncated on a char boundary so the page's `TextDecoder` gets valid UTF-8. `switch_init` installs it first thing (so do `switch_new` and `switch_alloc`), so panics before the first session are captured instead of trapping as bare `unreachable`. Panic and trace state is cleared on each new session, otherwise the first error after Reset described the dead session.
- With the `gpu` feature the host read is a wasm-bindgen import (`@host/files`, resolved by Vite) because wasm-bindgen owns the whole import object.
- `switch_alloc` returns null rather than trapping for sizes above `isize::MAX` (2 GiB on wasm32).
- `nul_reserved`: the terminating NUL takes a byte of the buffer, not of the message.
- `json_escape` escapes per character; per-byte `\u00XX` escapes turned "JUST DANCE® 2017" into "JUST DANCEÂ® 2017".
- Updates: the update's Program NCA carries the base title's program id (the `...800` id is only on its Meta NCA); its ExeFS replaces all modules and its RomFS is a patch over the base's. An update for a different title is refused at boot, which is where pairing is checked since the page may add the pieces in any order. Updates and DLC carry their own tickets.
- DLC: one Data NCA with a RomFS, title id = base + 0x1000 + 11-bit index, mounted by id like system data; `aoc:u` only lists indices. Mounted after the program id and NACP override are known; content for another title is reported and skipped.
- NRO control data is cached for display only, not via `cache_control`: homebrew runs in another title's process, so its NACP never governs saves.
- Boot reports ExeFS hash coverage and the module list next to what loaded (a missing `sdk`/`subsdk0` looks like a missing service otherwise). The NPDM system resource size picks the address space layout and must precede the boot (`nn::init` reads it immediately); the 32/64-bit flag changes the entry ABI.
- `switch_gpu_report_json` and `switch_activity_json` exist because `eprintln!` and env vars are unavailable on wasm32. Activity lists are fitted to `maxlen` in whole entries so the JSON always parses.
- `switch_present` (and `switch_fb_snapshot`) force opaque alpha: titles leave arbitrary alpha (Just Dance 2019 uses 0) and Chromium/macOS `putImageData` multiplies by it even with `alpha: false`.
- The page hands its canvas to the worker once (`transferControlToOffscreen`); `screen.rs` paints each new frame on it through `web_sys` after the slice that presented it, into one reused `ImageData`. The page never reads pixels, and maps touches with the size the slice report carries.
- SD and save host writes are not reported as changes (restores would otherwise be written straight back). Drains happen even if the JSON does not fit.
- The device opens as the run loop starts, between slices since it is a promise. The backend installs on the session's single `Gpu`, not a channel (Asphalt 9 opens four).

## `crates/switch-wasm/src/gpu.rs`

- `requestAdapter`/`requestDevice` are promises the emulator cannot wait on, so opening is `async` via `wasm-bindgen-futures`. The `gpu` feature exists because wgpu's web backend requires `wasm-bindgen` glue.
- Use `switch_gpu::device_descriptor`, not `DeviceDescriptor::default()`: compressed texture formats are behind optional features, and creating a BC texture without them panics (bare `unreachable` on wasm).
- `device_msaa`: device multisampling at WebGPU's only count, 4, shades once per pixel (different AA than the rasterizer, see `route` in `switch-gpu`'s `multisample.rs`). `interleave`: keep sending single fallback draws to the rasterizer within a device frame.

## `web/worker/index.ts`

- `gpuSession`: the backend belongs to the `Gpu` inside the session it was opened on, so freeing the session drops it. Every boot recycles the session, so without tracking this every session after the first silently ran on the rasterizer.
- `GPU_BACKEND_READY`: `Renderer::flush` answers `Pending` instead of blocking, and `Cpu::complete_pending_present` presents from a later slice, after the event loop let the map complete. The constant is the switch if the browser disagrees with the CLI.
- `GPU_DEVICE_MSAA` off: Maxwell samples sit at texel centres, WebGPU's on the spec's rotated grid, so device MSAA anti-aliases differently from the software reference. On would save fragment work. WebGPU guarantees only 4 samples, so 2x1, 4x2 and 4x4 always render expanded.
- `GPU_INTERLEAVE` off: fallback draws inside a device frame race the readback and lose pixels. A frame that is slow without interleaving is slow because a shader `gpu::shader::wgsl` cannot translate; the fix is the translator, not the switch.
- Adapter naming: wgpu names from `GPUAdapterInfo.description`, which Chrome leaves empty on macOS (it fills `vendor`/`architecture`), so the worker asks once with a second `requestAdapter`. Firefox (`dom/webgpu/Adapter.h`) blanks all four fields against fingerprinting, so "an unnamed adapter" is final there.
- `GPU_REOPENS = 3`: lost devices are replaced (the rasterizer is far slower), bounded so a device that dies on sight is not re-requested every slice. A new device shares no caches, so the first frames after pay for pipelines again.
- Once a browser answers with no adapter or device, `gpu = 'never'`: asking each slice cost a `wgpu::Instance`, a `requestAdapter`, and two Chrome "Failed to create WebGPU Context Provider" warnings each.

## `web/worker/core.ts`

- The system worker compiles the module once and starts `CORE_WORKERS` core workers on it and its shared memory, so guest RAM and the emulator's heap are the same bytes on every worker. They only instantiate for now; exports are still called from the system worker alone.

## `web/worker/wasm.ts`

- `switch_alloc` returns an i32 that reads negative past 2 GiB; `>>> 0` is required. A refused allocation returns 0, and writing at 0 would corrupt module data.
- `readWholeString` retries with a doubled buffer (capped at 1 MiB) and is only for idempotent reads; the activity report is consumed on read.

## `web/worker/commands.ts`

- Battery and dock state are cached in the worker so a reset session inherits them; time is sampled in the worker directly.
- Change-list buffers are sized from the pending count (0x301-byte paths) because the wasm side drains whether or not the JSON fits.
- `switch_control_json` truncates silently, so its buffer is sized for the worst-case NACP strings.

## `web/worker/hostfiles.ts`

- Containers are never handed to wasm whole (multi-GB, beyond wasm32). The File/Blob stays in the browser and wasm pulls ranges through the synchronous `hostRead` import; it must be synchronous because RomFS reads happen inside `switch_run`, and `FileReaderSync` only exists in workers.
- NAND content comes back from IndexedDB as Blobs, so firmware is registered for the cost of its headers rather than copied through the wasm heap.
- The chunk cache is per file, not shared: a title with an update reads two containers at once and a shared LRU thrashes at every crossing.
- Reads larger than a chunk (ExeFS pulled in one go) bypass the cache so they don't evict the working set.
- The host file table only grows within a session (wasm sources address archives by index); `resetHostFiles` clears it when the session is freed so resets don't accumulate handles and caches.
- A failed read (file moved or replaced) reports a short read; the wasm side turns it into an error with the offset.

## `web/worker/latch.ts`

- Input messages pile up while the worker is inside `switch_run`, so a tap can press and release entirely between guest polls. The unit a press must survive is a guest frame, not a run slice: the guest polls hid and presents once per loop iteration, so a press is held until the frame counter advances twice (only a full present-to-present interval is guaranteed to contain a poll). A slice cap releases the latch for programs that never present.
- Only bits the guest may not have seen are latched; keys the host still reports down publish live, so release takes effect on the next slice (an extra slice of stickiness made one d-pad tap step two menu entries).
- Stick flicks past `HID_STICK_THRESHOLD` latch like buttons, since menus navigate with the derived stick pseudo-buttons. Touch uses the same latch.

## `web/main/session.ts`

- A session is a whole console (RAM, threads, handles) and nothing is cleared per title, so every boot recycles the session first. Loading into the running session left the previous title's pages mapped (hundreds of MiB for retail), and after a few swaps guest RAM hit the cap and failed inside unrelated code.
- `recycleSession` invalidates the main-thread session (`setSession(-1)`) before posting the free so timers stop immediately, then restages font, SD card, saves, controls, users, keys, then archives and library applets (both need keys). A never-booted session is not recycled unless forced (Reset). `reopen` is passed by the caller to avoid a `container.ts` import cycle.

## `web/main/index.ts`, `boot.ts`, `dock.ts`, `wakelock.ts`

- After a homebrew load the screen is shown but the loading screen stays over it until `display.noteFrame` sees a real frame (homebrew can run a long time before presenting).
- Init awaits only the NAND index; archives register in the background so a firmware dump does not delay the first boot. A core that fails to start keeps the loading screen with the error rather than an idle splash.
- Operation mode affects `vi` resolution, `am`/`apm` performance mode, `clkrst` GPU clock and touchscreen presence. Titles read it once, so changes queue the two AM messages a real dock sends.
- Screen Wake Lock needs a secure context (missing in Firefox and Safari before 16.4). The browser drops the lock when the document is hidden and does not return it, so it is re-taken on visibility; a lock acquired after the run ended is released immediately.
- Icons are Lucide (`icons.ts`), imported per icon so the bundle carries only those used.

## `web/worker/loop.ts` and `web/main/runloop.ts`

- The worker runs the loop: slices back to back, each queued as a `MessageChannel` task so input and commands still get in between them (a `setTimeout` chain is clamped to 4 ms). Slices are sized by wall time to about one display frame.
- After each slice the worker posts one report: steps, timing, frame count and size, audio format, and every 8 slices the housekeeping (readouts, guest output, diagnostics, whether the SD card, saves or profiles changed, keyboard and error dialogs). The page reacts and polls nothing; it reads file contents only when told they changed.
- Run starts the loop and Pause asks it to stop after the slice in flight. Freeing the session stops it, and the page ignores reports that arrive after a reset.
- The release profile aborts on panic, which on wasm is an empty `unreachable` trap; the panic hook stores the message and `switch_last_error` returns it without a live session. Linear memory survives the trap, so panic context is collected afterwards, each piece separately.
- The status bar shows instructions retired, not the guest clock, which jumps forward when all threads are blocked.

## `web/main/loading.ts`

- The loading screen only owns its markup. `awaitFirstFrame` is the one open-ended phase, ended by `display.noteFrame` on the first painted frame or by the run loop on a fault, halt or pause. Failures keep the screen up with the reason rather than revealing a black stage.

## `web/main/input.ts`

- hid reports touch in the console's 1280x720 digitizer space regardless of the guest's presentation size. Taps are mapped through the `object-fit: contain` rect, not the element box. A touch slot (finger id) stays fixed for a contact's life, claimed from the lowest free slot.
- Stick Y axes are negated (Horizon's points up); the core derives stick pseudo-buttons from these values, so the sign convention matters.

## `web/main/container.ts`

- Updates are paired, not opened: an update's Program NCA has the patched modules in full but its RomFS is ranges indexed against the base game's. A held update waits until its title is open and is applied at Launch. Pairing lives in the page because the session refuses to launch a container whose update is for another title, and a held update for another game must not make the open one unlaunchable. Add-on content is always handed to the session (a title mounts what is numbered against it).
- The page remembers (localStorage `switch-wasm:updates` / `switch-wasm:dlc`) which update or DLC a title last ran with, since files do not survive a reload, so it can say so instead of launching unpatched silently.
- Reset re-hands the open container to the new session; the page still shows it, so "no container is open" would be wrong.
- NSP files are named by hash, so the boot finds the Program NCA by content type. XCI partitions are flattened by the wasm side, so the page treats them like NSPs.
- Without prod.keys the Control NCA cannot be identified (content type is in the encrypted header); a container without one (update or DLC) is a dim note, not an error.

## `web/main/filetype.ts`

- `accept` on an input is defeated by the picker's "All files" option and ignored by drops, so files are identified by header, not name. CDN NCAs are the exception: their header is encrypted until keys are loaded.
- Keys files are validated as `name = hex` lines so a binary picked by mistake is not stored as keys.

## `web/main/nand.ts`

- The NAND lives in IndexedDB in two stores: `content` (NCA bytes by file name, stored as `Blob` so the browser owns the bytes and the worker reads ranges from a handle) and `titles` (index by title id, so the panel lists installs without reading content).
- Data archives and library applets (`add_applet`) are registered per session; a rebuilt session needs them again (`restoreArchives` is awaited on rebuild because a title mounting an archive still being registered will not find it). A generation ticket lets a newer restore cancel an older one.
- A launched program is booted from full bytes (the loader maps the whole image); archives and applets are range-read, and the core runs an applet as a process beside its caller.

## `web/worker/persist.ts`

- The worker owns the SD card's and saves' persistence: it restores them from IndexedDB into the core when a session starts and stores guest-changed paths as its housekeeping finds them, so no file crosses to the page. Restores go through host entry points (not recorded as changes) so restoring does not queue a full write-back. Drained changes cannot be returned to the core, so IndexedDB failures (quota) stay in a per-path backlog until the next flush. Flushes run one after another, so one that resolves has stored its changes.
- The page asks for persistent storage (`navigator.storage.persist`, main thread only) so the card is not evicted.

## `web/main/files.ts`

- IndexedDB is the source of truth: the dialog lists and edits the stored entries, so it works with no session. Pending guest changes are flushed before a listing; edits are then mirrored into a live session through the host entry points.
- A folder can exist only as the prefix of deeper paths, so listings derive folders from paths, and delete and rename act on every path under the target. `Vfs::remove` is not recursive, so the session is told deepest first.
- Exports are store-only zips (`zip.ts`): saves and SD folders are small, and it needs no dependency. No zip64, so it refuses past 4 GiB or 65535 entries.
- `shell.ts` binds `.panel .tab` only; the dialog's tabs are `.files-tab` so the panel's handler does not take them.
- Opening a file (`fileview.ts`): valid UTF-8 without NUL bytes is edited in a textarea (read-only past 4 MiB), PNG/JPEG/GIF/WebP/BMP is shown by magic number, anything else as a hex dump of its first 4 KiB. Leaving or closing with unsaved edits asks first, Escape included.

## `web/main/users.ts`

- A title asks which user is playing once at startup, so profile changes apply from the next title. Idle sessions get changes immediately; running titles keep their user.
- The last profile cannot be deleted (no title starts without a user), nor the one a running title uses (its saves would be orphaned).
- Profiles are stored in the NAND database next to their saves.

## `web/main/log.ts`

- The page view keeps 2000 entries (layout cost with tracing on); the backlog keeps 50,000 for copy and save.
- The log tail is mirrored to IndexedDB on a timer (not per line) so it survives the browser killing the tab; capped so it can't trigger origin eviction (which would take the SD card and NAND).
- Unclean-exit detection: a `pagehide` handler writes a mark synchronously to `localStorage` (no time to await IndexedDB). The mark is per origin, so other tabs are pinged over `BroadcastChannel` before treating it as a dead session; pings carry a tab id because a page receives its own channel messages.
- Clipboard falls back to a selection copy because `navigator.clipboard` needs a secure context (e.g. testing on a phone over plain http).

## `web/main/battery.ts`

- Only Chromium exposes the Battery Status API; elsewhere the emulated battery stays at the wasm default (full, charging). Updates are event-driven and cached worker-side, so a new session (including after reset) picks them up.

## `vite.config.ts`

- `web/public` holds the social card (its URL is baked into other sites' caches by meta tags), the home screen icons and `manifest.webmanifest` (fixed URLs), the font licence (must stay beside the font) and `_headers` (cross-origin isolation in deployment). Everything else is content-hashed; `assetsInlineLimit: 0` keeps every asset a file with a hashed URL.
- `base: './'` because the site lives at tenshii.moe/Switch-Wasm/; '/' 404s only once deployed.
- The worker must be a module on both sides: 'iife' fails in dev (the dev server serves the entry unbundled, so `import` breaks a classic worker), and dropping `type` in production only works until a chunk gains an `import`.
- `@host/files` is a bare specifier so wasm-bindgen's glue in cargo's target dir can import `web/worker` code without a cross-directory relative path.
