# Web frontend and worker

The page, the worker, and the wasm bindings between them.

## `crates/switch-wasm/src/lib.rs`

- Exports are `extern "C"` taking pointers JS owns for the call; marking them `unsafe fn` would not help a JS caller, hence `#![allow(clippy::not_unsafe_ptr_arg_deref)]`.
- Session table uses `SyncCell` (UnsafeCell) not `Mutex`: the wasm `no_threads` backend aborts on reentrant `lock()` and a panic while held leaves it locked. Host tests serialize on a `HOST` mutex because `cargo test` is multi-threaded, and a panic crossing `extern "C"` aborts the harness.
- Panic hook writes to a fixed 2048-byte buffer (512 was too small for payload + location + assertion values), truncated on a char boundary so the page's `TextDecoder` gets valid UTF-8. `PANICKED` is separate because `switch_last_error` takes the message but the crash report still needs to know. `switch_init` installs the hook (also called from `switch_new`/`switch_alloc`) so panics before the first session are captured. Module-level panic/trace state is cleared on each new session, otherwise a report after Reset described the dead session.
- With the `gpu` feature the host read is a wasm-bindgen import (`@host/files`, resolved by Vite) because wasm-bindgen owns the whole import object. A retail container exceeds the wasm32 address space, so it stays in the browser and is served synchronously by range (`FileReaderSync`).
- `switch_alloc` returns null rather than trapping for sizes above `isize::MAX` (2 GiB on wasm32); the old `unwrap` trapped with `unreachable`.
- `nul_reserved`: the terminating NUL comes from the buffer, not the message (the old code cut "no container is open" to "...ope").
- `json_escape` escapes per character; per-byte `\u00XX` escapes turned "JUST DANCE® 2017" into "JUST DANCEÂ® 2017".
- Updates: the update's Program NCA program id is the base title's (the `...800` id is only on its Meta NCA); its ExeFS replaces all modules and its RomFS is a patch over the base's. An update for a different title is refused at boot. Updates and DLC carry their own tickets. Pairing is checked at boot since the page may add them in any order.
- DLC: one Data NCA with a RomFS, title id = base + 0x1000 + 11-bit index, mounted by id like system data; `aoc:u` only lists indices. Mounted after the program id and NACP override are known; content for another title is reported and skipped.
- NRO control data is cached for display only, not via `cache_control`: homebrew runs in another title's process, so its NACP never governs saves.
- Boot reports ExeFS hash coverage and the module list next to what loaded (missing `sdk`/`subsdk0` looks like a missing service otherwise). NPDM system resource size picks the address space layout and must precede the boot (`nn::init` reads it immediately); the 32/64-bit flag changes the entry ABI.
- `switch_gpu_report_json` and `switch_activity_json` exist because `eprintln!` and env vars are unavailable on wasm32. Activity lists are fitted to `maxlen` (whole entries) so the JSON always parses.
- The crash report uses `session_opt` so it works after the session died; the trace goes last so truncation loses it first.
- `switch_fb_snapshot` forces opaque alpha: titles leave arbitrary alpha (Just Dance 2019 uses 0) and Chromium/macOS `putImageData` multiplies by it even on `alpha: false`.
- SD and save host writes are not reported as changes (restores would otherwise be written straight back). Drains happen even if the JSON does not fit.
- `gpu_channel_open`: avoid building a device before the title opens a channel (wgpu web frees nothing on drop; Home Menu opens its channel 11.6M steps in). The backend installs on the session's single `Gpu`, not a channel (Asphalt 9 opens four).

## `web/worker/latch.ts`

- Input messages pile up while the worker is inside `switch_run`, so a tap can press and release entirely between guest polls. The unit a press must survive is a guest frame, not a run slice: the guest polls hid once per loop iteration and presents once per iteration, so a press is held until the frame counter advances twice (only a full present-to-present interval is guaranteed to contain a poll). A slice cap releases the latch for programs that never present.
- Only bits the guest may not have seen are latched; keys the host still reports down publish live, so release takes effect on the next slice (an extra slice of stickiness made one d-pad tap step two menu entries).
- Stick flicks past `HID_STICK_THRESHOLD` latch like buttons, since menus navigate with the derived stick pseudo-buttons. Touch uses the same latch.

## `web/main/input.ts`

- hid reports touch in the console's 1280x720 digitizer space regardless of the guest's presentation size, and the console always reports handheld mode, so touch is always live. Taps are mapped through the `object-fit: contain` rect, not the element box. A touch slot (finger id) stays fixed for a contact's life, claimed from the lowest free slot.
- Stick Y axes are negated (Horizon's points up); the core derives stick pseudo-buttons from these values, so the sign convention matters.

## `web/main/container.ts`

- Updates are paired, not opened: an update's Program NCA has the patched modules in full but its RomFS is ranges indexed against the base game's. A held update waits until its title is open and is applied at Launch. Pairing lives in the page because the session refuses to launch a container whose update is for another title, and a held update for another game must not make the open one unlaunchable. Add-on content is always handed to the session (a title mounts what is numbered against it).
- The page remembers (localStorage `switch-wasm:updates` / `switch-wasm:dlc`) which update/DLC a title last ran with, since files do not survive a reload, so it can say so instead of launching unpatched silently.
- Reset re-hands the open container to the new session; the page still shows it, so "no container is open" would be wrong.
- The container `File` is passed to the worker and read by range; retail containers exceed wasm32's address space.
- NSP files are named by hash, so the stage boot finds the Program NCA by content type. XCI partitions are flattened by the wasm side, so the page treats them like NSPs.
- Without prod.keys the Control NCA cannot be identified (content type is in the encrypted header); a container without one (update/DLC) is a dim note, not an error.

## `web/worker/hostfiles.ts`

- Containers are never handed to wasm whole (multi-GB, beyond wasm32). The File/Blob stays in the browser and wasm pulls ranges through the synchronous `host_read` import; it must be synchronous because RomFS reads happen inside `switch_run`, and `FileReaderSync` only exists in workers.
- NAND content comes back from IndexedDB as Blobs, so firmware is registered for the cost of its headers rather than copied through the wasm heap.
- Chunk cache is per file, not shared: a title with an update reads two containers at once and a shared LRU thrashes at every crossing.
- Reads larger than a chunk (ExeFS pulled in one go) bypass the cache so they don't evict the working set.
- The host file table only grows within a session (wasm sources address archives by index); `resetHostFiles` clears it when the session is freed so resets don't accumulate handles and caches.
- A failed read (file moved/replaced) reports a short read; the wasm side turns it into an error with the offset.

## `web/main/users.ts`

- A title asks which user is playing once at startup, so profile changes apply from the next title. Idle sessions get changes immediately; running titles keep their user.
- The last profile cannot be deleted (no title starts without a user), nor the one a running title uses (its saves would be orphaned).
- Profiles are stored in the NAND database next to their saves.

## `web/main/log.ts`

- The page view keeps 2000 entries (layout cost with tracing on); the backlog keeps 50,000 for copy/save.
- The log tail is mirrored to IndexedDB on a timer (not per line) so it survives the browser killing the tab; capped so it can't trigger origin eviction (which would take the SD card and NAND).
- Unclean-exit detection: a `pagehide` handler writes a mark synchronously to `localStorage` (no time to await IndexedDB). The mark is per origin, so other tabs are pinged over `BroadcastChannel` before treating it as a dead session; pings carry a tab id because a page receives its own channel messages.
- Clipboard falls back to a selection copy because `navigator.clipboard` needs a secure context (e.g. testing on a phone over plain http).

## `web/main/runloop.ts`

- Slices are sized by wall time to about one display frame: input reaches the worker only between slices, and each slice costs a worker round trip.
- Debug panel housekeeping (several postMessage round trips) runs every 8 slices.
- Reset aborts without waiting for the in-flight slice; the loop must not call into a freed session afterwards.
- The release profile aborts on panic, which on wasm is an empty `unreachable` trap; the panic hook stores the message and `switch_last_error` returns it without a live session. Linear memory survives the trap, so panic context is collected afterwards, each piece separately.
- The status bar shows instructions retired, not the guest clock, which jumps forward when all threads are blocked.

## `web/main/battery.ts`

- Only Chromium exposes the Battery Status API; elsewhere the emulated battery stays at the wasm default (full, charging). Updates are event-driven and cached worker-side, so a new session (including after reset) picks them up.

## `web/main/filetype.ts`

- `accept` on an input is defeated by the picker's "All files" option and ignored by drops, so files are identified by header, not name. CDN NCAs are the exception: their header is encrypted until keys are loaded.
- Keys files are validated as `name = hex` lines so a binary picked by mistake is not stored as keys.

## `web/main/loading.ts`

- The loading screen only owns its markup. `awaitFirstFrame` is the one open-ended phase, ended by `display.renderFb` on the first painted frame or by the run loop on a fault, halt or pause. Failures keep the screen up with the reason rather than revealing a black stage.

## `crates/switch-wasm/src/gpu.rs`

- `requestAdapter`/`requestDevice` are promises the emulator cannot wait on, so opening is `async` via `wasm-bindgen-futures`. The `gpu` feature exists because wgpu's web backend requires `wasm-bindgen` glue.
- Check `gpu_channel_open` before requesting a device: a device built too early and dropped frees nothing on wgpu's web backend.
- Use `switch_gpu::device_descriptor`, not `DeviceDescriptor::default()`: compressed texture formats are behind optional features, and creating a BC texture without them panics (bare `unreachable` on wasm).
- `GPUAdapterInfo.description` is empty on Chrome/macOS and always on Firefox; the worker supplies names.
- `device_msaa`: device multisampling at WebGPU's only count, 4, shades once per pixel (different AA than the rasterizer, see `Gpu::route`). `interleave`: keep sending single fallback draws to the rasterizer within a device frame (see `Gpu::interleave`).

## web/main (boot.ts, index.ts, dock.ts, wakelock.ts)

- Boot always recycles the session first: loading into the running session left the previous title's guest RAM mapped under the new one.
- After a homebrew load the screen is shown but the loading screen stays over it until `display.renderFb` has a real frame (homebrew can run a long time before presenting).
- Init awaits only the NAND index; archives register in the background so a firmware dump does not delay the first boot. A core that fails to start keeps the loading screen with the error rather than an idle splash.
- Operation mode affects `vi` resolution, `am`/`apm` performance mode, `clkrst` GPU clock and touchscreen presence. Titles read it once, so changes queue the two AM messages a real dock sends.
- Screen Wake Lock needs a secure context (missing in Firefox and Safari before 16.4). The browser drops the lock when the document is hidden and does not return it, so it is re-taken on visibility; a lock acquired after the run ended is released immediately.

## `web/worker/wasm.ts`

- `switch_alloc` returns an i32 that reads negative past 2 GiB; `>>> 0` is required. A refused allocation returns 0, and writing at 0 would corrupt module data.
- `readWholeString` retries with a doubled buffer (capped at 1 MiB) and is only for idempotent reads; the activity report is consumed on read.

## `web/worker/commands.ts`

- Battery and dock state are cached in the worker so a reset session inherits them; time is sampled in the worker directly.
- Change-list buffers are sized from the pending count (0x301-byte paths) because the wasm side drains whether or not the JSON fits.
- `switch_control_json` truncates silently, so its buffer is sized for the worst-case NACP strings.

## `web/main/sdcard.ts`

- The emulated SD card lives in session memory; IndexedDB persists it as a path -> bytes map, flushing only guest-changed paths. Restores go through host entry points (not recorded as changes) so restoring does not queue a full write-back. Drained changes cannot be returned to the core, so IndexedDB failures (quota) stay in a per-path backlog until the next flush. Persistent storage is requested so the card is not evicted.

## `web/main/session.ts`

- A session is a whole console (RAM, threads, handles) and nothing is cleared per title, so booting a second title must recycle the session. Before booting did this, titles loaded on top of each other: old pages stayed mapped (hundreds of MiB for retail), and after a few swaps guest RAM hit the cap and failed inside unrelated code.
- `recycleSession` invalidates the main-thread session (`setSession(-1)`) before posting the free so timers stop immediately, then restages font, SD card, saves, users, keys and archives (archives depend on keys). A never-booted session is not recycled unless forced (Reset). `reopen` is passed by the caller to avoid a `container.ts` import cycle.

## `web/worker/index.ts`

- `gpuSession`: the backend belongs to the `Gpu` inside the session it was opened on, so freeing the session drops it. Before tracking this, every session after the first (every boot recycles the session, not only Reset) silently ran on the rasterizer at about 1/30 speed.
- `GPU_BACKEND_READY`: `Renderer::flush` answers pending instead of blocking and `Cpu::complete_pending_present` presents from a later slice (a later message, after the event loop let the map complete). Verified natively: 20 deferred presents over 300 frames, all 300 presented. The constant is the switch if the browser disagrees with the CLI.
- `GPU_DEVICE_MSAA` off: Maxwell samples sit at texel centres, WebGPU's on the spec's rotated grid, so device MSAA anti-aliases differently from the software reference. On would save fragment work (4x at 4x MSAA). WebGPU guarantees only 4 samples, so 2x1, 4x2, 4x4 always render expanded.
- `GPU_INTERLEAVE` off: measured on the Home Menu at frame 60 with deferred readback, interleaving loses 795 of 921,600 pixels at 0.10 s frames; not interleaving is byte-identical at 1.03 s frames because the Home Menu has a shader `gpu::shader::wgsl` cannot translate. The fix is the translator, not the switch.
- Adapter naming: wgpu names from `GPUAdapterInfo.description`, which Chrome leaves empty on macOS (it fills `vendor`/`architecture`); asked once with a second `requestAdapter`. Firefox (`dom/webgpu/Adapter.h`) hardcodes all four fields empty against fingerprinting, so "an unnamed adapter" is final there.
- `GPU_REOPENS = 3`: lost devices are replaced (rasterizer fallback measured 30x slower), bounded so a device that dies on sight is not re-requested every slice. A new device shares no caches, so the first frames after pay for pipelines again.
- Once a browser answers with no adapter/device, `gpu = 'never'`: asking each slice cost a `wgpu::Instance`, a `requestAdapter`, and two Chrome "Failed to create WebGPU Context Provider" warnings each (thousands per sitting).
- The `RENDERING_ON` prefix is matched rather than the exact string because older cores left a trailing space where the name would be.
- `switch_init` installs the panic hook first thing; when `switch_new` installed it, failures before the first session trapped as bare `unreachable`.

## `vite.config.ts`

- `web/public` holds the social card (URL baked into other caches by meta tags) and the font licence (must stay beside the font). Everything else is content-hashed; no assets are inlined because `switch_wasm.wasm` is fetched by URL.
- `base: './'` because the site lives at tenshii.moe/Switch-Wasm/; '/' 404s only once deployed.
- The worker must be a module on both sides: 'iife' fails in dev (the dev server serves the entry unbundled, so `import` breaks a classic worker), and dropping `type` in production only works until a chunk gains an `import`.
- `@host/files` is a bare specifier so wasm-bindgen's glue in cargo's target dir can import `web/worker` code without a cross-directory relative path.

## `web/main/nand.ts`

- The NAND lives in IndexedDB in two stores: `content` (NCA bytes by file name, stored as `Blob` so the browser owns the bytes and the worker reads ranges from a handle) and `titles` (index by title id, so the panel lists installs without reading content).
- Data archives are registered per session; reset/rebuild requires re-registration (`restoreArchives` is awaited on rebuild because a title mounting an archive still being registered will not find it). A generation ticket lets a newer restore cancel an older one.
- Installed programs are booted from full bytes (the loader maps the whole image); archives are range-read.
