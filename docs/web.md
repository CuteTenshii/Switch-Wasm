# Web frontend and worker

The page, the worker, and the wasm bindings between them.

Section headings are paths under `crates/wasm/src/`; a heading that starts with `crates/`, `tools/` or `web/` is from the repository root.

## `lib.rs`

- Exports are `wasm-bindgen` exports. Bytes and text take `&[u8]`/`&str` and return `String`/`Vec<u8>`, so no caller sizes a buffer and nothing truncates silently; flags are `bool`; a `Result` error is thrown to the caller. The page-facing declarations are in `web/env.d.ts`.
- The session table uses `SyncCell` (an `UnsafeCell`), not `Mutex`: a reentrant `lock()` deadlocks and a panic while held leaves it locked. It is sound only because one worker calls exports; the core workers on the shared memory call only `switch_serve_core` (`cores.rs`). Host tests serialize on a `HOST` mutex because `cargo test` is multi-threaded and a panic crossing `extern "C"` aborts the harness.
- The panic hook writes to a fixed 2048-byte buffer (512 was too small for payload, location and assertion values), truncated on a char boundary so the page's `TextDecoder` gets valid UTF-8. `switch_init` installs it first, so panics are captured instead of trapping as bare `unreachable`. Building the machine clears the panic message, so an earlier session's panic is not reported for the new one; log lines are kept, so what the dead session said still reaches the page.
- Everything the page parses is written with `switch_core::json`, which keeps keys in the order written and escapes per character: per-byte `\u00XX` escapes once turned "JUST DANCE® 2017" into "JUST DANCEÂ® 2017". The backend's report and `frames` go into one object (`Gpu::write_renderer_report`), so nothing splices JSON text.
- `switch_open_container` reads a dropped file's table and NCA headers once and holds it as an update or add-on content, or makes it the open container. The open container's parsed NCAs stay in the session (`ncas`, parsed again when keys load) for the control data, the launch and the Manual.
- Updates: the session holds the last update added and applies it only when its title launches, so the pieces may come in any order. Updates and DLC carry their own tickets. The update NCA's layout is in [formats.md](formats.md#ncars).
- DLC: one Data NCA with a RomFS, title id = base + 0x1000 + 11-bit index, mounted by id like system data; `aoc:u` only lists indices. It is mounted once the program id and NACP override are known; content for another title is reported and skipped.
- NRO control data is cached for display only, not in `container_nacp`: homebrew runs in another title's process, so its NACP never governs saves.
- Boot reports ExeFS hash coverage and the module list next to what loaded (otherwise a missing `sdk`/`subsdk0` looks like a missing service). The NPDM must be applied before the boot, since `nn::init` reads the system resource size immediately ([formats.md](formats.md#npdmrs)); its 32/64-bit flag changes the entry ABI.
- `switch_gpu_report_json` exists because wasm32 has no `eprintln!` or env vars. The activity report is not data for the page: `activity.rs` writes its lines into the log.
- `switch_present` forces opaque alpha: titles leave arbitrary alpha (Just Dance 2019 uses 0) and Chromium/macOS `putImageData` multiplies by it even with `alpha: false`.
- The page hands its canvas to the worker once (`transferControlToOffscreen`). A canvas takes one kind of context for good, so `screen.rs` decides at the first presented frame, once the device has finished opening: when the WebGPU device is installed, the canvas becomes its screen (see [gpu-backend.md](gpu-backend.md)); otherwise `screen.rs` paints each new frame on a 2D context through `web_sys` after the slice that presented it, into one reused `ImageData`. Once the device owns the canvas, a session without a device cannot paint it. The page never reads pixels and maps touches with the size the slice report carries.
- The device opens between slices as the run loop starts. The backend installs on the session's single `Gpu`, not per channel (Asphalt 9 opens four).

## `log.rs`

- The page gets finished lines with a level number each, so it parses no text: guest `[lm/LEVEL]` tags and the trace's marker bytes are decoded here.
- The queue is global, not per session, so lines outlive the session that wrote them and code with no session still logs.
- Boot empties the core's output and trace buffers, so lines are collected first; clearing them lost the ExeFS verification line every retail boot logs.
- The worker flushes after every command (before replying, so the caller finds the command's lines on the page) and every slice (before the slice report, so a halt's last lines precede it).

## `gpu.rs`

- `requestAdapter`/`requestDevice` are promises the emulator cannot wait on, so opening is `async` via `wasm-bindgen-futures`.
- Use `switch_gpu::device_descriptor`, not `DeviceDescriptor::default()`: compressed texture formats are behind optional features, and creating a BC texture without them panics (bare `unreachable` on wasm).
- `device_msaa`: device multisampling at WebGPU's only count, 4, shading once per pixel (see `route` in `switch-gpu`'s `multisample.rs`). `interleave`: keep sending single fallback draws to the rasterizer within a device frame. [`web/worker/gpu.ts`](#webworkergputs) says why both are off.

## `hostfiles.rs`

- Containers are never handed to wasm whole (multi-GB, beyond wasm32). The File/Blob stays in the browser and the wasm side keeps it as a `web_sys::Blob`. Reads must be synchronous because RomFS reads happen inside `switch_run`, so a read waits on the reader worker or, when the cache will not hold it, calls `FileReaderSync` itself (only workers have it). The exports that take a file (`switch_open_container`, `switch_open_nca`, ...) take the `Blob` itself.
- NAND content comes back from storage as Blobs ([`web/main/nand.ts`](#webmainnandts)), so firmware is registered for the cost of its headers rather than copied through the wasm heap. The wasm side reads the index and content itself (`idb/nand.rs`, `idb/content.rs`) and registers each Blob with `hostfiles::add`, which returns its index; one host file serves every registration made from an NCA.
- Firmware installs from a folder or a .zip (`web/main/nand.ts`, `web/main/unzip.ts`). The zip is read by range: its directory, then one member at a time. A stored member is a slice of the archive; a deflated one is inflated through `DecompressionStream` with its CRC checked. The console's `x.nca/00`, `x.nca/01`… pieces are joined into one Blob without copying. A folder install takes only the NCAs directly in the folder picked, and the pieces of a split one, as Eden's folder install does not recurse; the picker hands over the whole tree, so files below are left out (`gatherNcas`, `web/main/firmware.ts`). A .zip's NCAs are taken at any depth, as Eden takes an extracted .zip's.
- Chunks are 1 MiB, each in its own `SharedArrayBuffer` created on first use (up to 2048, 2 GiB), so the cache lives outside linear memory, which the 4 GiB cap shares with guest memory. A browser that refuses a buffer shrinks the cache; reads it cannot hold go straight to the file. `switch_core::chunk_cache::ChunkIndex` decides what is held (one LRU for all files, with read-ahead: consecutive misses ask for 1, 2, 4, 8, then 16 chunks); `hostfiles.rs` moves the bytes.
- The reader worker (`web/worker/reader.ts`, started by `switch_set_reader`) does the `FileReaderSync` reads, one chunk at a time. It is sent every `Blob` and every cache buffer as they appear, and answers per slot in a shared `Int32Array`: the generation of the request it filled the slot for, or its negation on failure. The emulator blocks (`Atomics.wait`, 30 s) only for the chunk it needs; the read-ahead behind it fills while the guest runs. Generations stop a stale request, for a slot since evicted and reused, from being taken as the new chunk's data; requests are answered in order.
- Reads larger than a chunk (ExeFS pulled in one go) bypass the cache so they don't evict the working set.
- The host file table only grows (wasm sources address archives by index). System archives and applets are added once per page; each update or DLC added appends a file. A dropped file is added before it is known to be a game; one that becomes the open container moves to slot 0 and its added slot is forgotten.
- A failed read (file moved or replaced) reports a short read, which becomes an error with the offset; the host's reason is in the next `[io]` line, which `switch_log_activity` writes with the rest of the report.

## `input.rs`

- The latch lives in Rust: the page writes buttons and sticks into a `Mailbox` in shared memory (`exchange` takes it each slice), `switch_set_touch` applies touches, and `switch_run` lets go of what the guest has had a frame to see. Input piles up while the worker is inside `switch_run`, so a tap can press and release entirely between guest polls.
- A press must survive a guest frame, not a run slice: the guest polls hid and presents once per loop iteration, so a press is held until the frame counter advances twice (only a full present-to-present interval is guaranteed to contain a poll). A slice cap releases the latch for programs that never present.
- Only bits the guest may not have seen are latched; keys the host still reports down publish live, so release takes effect on the next slice (an extra slice of stickiness made one d-pad tap step two menu entries).
- Stick flicks past `HID_STICK_THRESHOLD` latch like buttons, since menus navigate with the derived stick pseudo-buttons. Touch uses the same latch.

## `idb/persist.rs`

- The wasm side owns the SD card's and saves' persistence and talks to OPFS and IndexedDB itself, so no file crosses JavaScript. `switch_restore_storage` loads them into the core when a session starts; `switch_flush_storage` stores changed paths, called by `switch_run` through `persist::tick` (at most every 100 ms) and by the page's `flushStorage` and after every Files dialog edit.
- They live in OPFS (`opfs/volumes.rs`) as `sd/<path>` and `saves/<save key>/<path>`, a backslash in a name escaped as `%5C` and `%` as `%25` (OPFS refuses the backslash; names are case-sensitive). IndexedDB (this module, with `idb/mod.rs`) holds them only where OPFS cannot be written, or when reading it failed this session.
- OPFS answers in promises except for an open sync handle's `read`, `write`, `truncate`, `flush` and `getSize`, and guest file requests cannot wait. So the session's `Vfs` holds the tree: a restore mounts every file as an `opfs/file.rs` `OpfsFile`, closed, read from a `getFile()` snapshot through `FileReaderSync`, without a handle or a copy in wasm memory. Writes to a closed file wait in memory over the snapshot until the next flush opens its handle and applies them; an open file is read and written through the handle, straight from wasm memory (Chromium takes views of shared memory; others go through a copy). Open handles stay open for the session: `getFile()` still reads an open file, from any worker.
- A file the guest or the page creates is held in memory; the flush creates it in OPFS, copies it in without yielding, and gives the session the open `OpfsFile` instead. Deletions close the file (the `Vfs` drops it) and remove the entry; a directory goes with what is under it.
- Older builds kept the card and saves in IndexedDB: a restore copies them into OPFS and only then empties both stores, so a move cut short is redone whole at the next start.
- With IndexedDB, restores go through host entry points (not recorded as changes), so restoring does not queue a full write-back. IndexedDB creates the databases with the same versions and stores as `web/shared/db.ts`; keep both in step.
- Drained changes cannot be returned to the core, so failures (quota) stay in a per-path backlog until the next flush. Flushes run one after another, so one that resolves has stored its changes.
- One tab stores. `storage.rs` asks for a Web Lock (through `Reflect`: `navigator.locks` is an unstable web-sys API) at `switch_init` and holds it for the worker's life; a reload's worker waits up to 2 s for the old one to let go. A tab without it is read-only: it reads OPFS, keeps the guest's changes in its session, refuses the page's edits, leaves the NAND alone (its sweep could remove files the first tab reads) and posts `storage_read_only`, which shows `#read-only` (`web/main/readonly.ts`). Nothing is stored before the claim settles.
- Free and total space come from the wasm side (`idb/usage.rs`): `navigator.storage.estimate()` gives the origin's free bytes, the OPFS `nand` directory plus any `content` left in IndexedDB the SYSTEM partition's, and the session's card and saves (which hold every file, whichever store keeps them) the SD card's and USER's. A scan starts on the first slice and again after a guest space query (at most every 2 s); queries answer from the last scan. Databases are opened only if `indexedDB.databases()` lists them, so the scan never creates one with the wrong version.
- The page asks for persistent storage (`navigator.storage.persist`, main thread only) so the card is not evicted.

## `web/worker/wasm.ts`

- `WasmExports` is only the module's `memory`; every function is imported from `@core/switch_wasm.js`. `parseOr` falls back to an empty report when a JSON export answers nothing.

## `web/worker/commands.ts`

- `loadProgram` hands the core the `File`, which reads it in 8 MiB chunks straight into linear memory and tells an ELF from an NRO by its magic; JavaScript holds no copy. An SD NRO is read from the session's own SD card (`loadSd`), and a NAND program is registered as a host file and read by range (`nandLaunch`), like archives and applets, so neither image is copied into the heap.

## `web/worker/core.ts`

- The system worker compiles the module once and `coreworkers.ts` starts `CORE_WORKERS` core workers on it and its shared memory, so guest RAM and the emulator's heap are the same bytes on every worker. Each runs a guest core the session lends it (`switch_add_core`, `switch_serve_core`), blocking until the session stops it; every other export is called from the system worker alone.

## `web/worker/gpu.ts`

- `GPU_BACKEND_READY`: `Renderer::flush` answers `Pending` instead of blocking, and `Cpu::complete_pending_present` presents the queued frames from a later slice, oldest first, after the event loop let the map complete. The constant is the switch if the browser disagrees with the CLI.
- `GPU_DEVICE_MSAA` off: Maxwell samples sit at texel centres, WebGPU's on the spec's rotated grid, so device MSAA anti-aliases differently from the software reference. On would save fragment work. WebGPU guarantees only 4 samples, so 2x1, 4x2 and 4x4 always render expanded.
- `GPU_INTERLEAVE` off: fallback draws inside a device frame race the readback and lose pixels. A frame that is slow without interleaving is slow because of a shader `gpu::shader::wgsl` cannot translate; fix the translator, not the switch.
- Adapter naming: wgpu names from `GPUAdapterInfo.description`, which Chrome leaves empty on macOS (it fills `vendor`/`architecture`), so the Debug panel (`web/main/debug.ts`) asks the page's own `requestAdapter` and joins all three. Firefox (`dom/webgpu/Adapter.h`) blanks all four fields against fingerprinting, so "an unnamed adapter" is final there.
- `GPU_REOPENS = 3`: lost devices are replaced (the rasterizer is far slower), bounded so a device that dies on sight is not re-requested every slice. A new device shares no caches, so the first frames after it pay for pipelines again.
- Once a browser answers with no adapter or device, `gpu = 'never'`: asking each slice cost a `wgpu::Instance`, a `requestAdapter`, and two Chrome "Failed to create WebGPU Context Provider" warnings.

## `web/worker/loop.ts` and `web/main/runloop.ts`

- The worker runs the loop: slices back to back, each queued as a `MessageChannel` task so input and commands still get in between them (a `setTimeout` chain is clamped to 4 ms). Slices are sized by wall time to about one display frame.
- After each slice the worker posts one report: steps, timing, frame count and size, paint time, and every 8 slices the housekeeping (readouts). The wasm side posts the rest itself: log lines go through the worker, and `error_dialog`, `keyboard` and `profiles_edited` messages arrive when the guest raises one (`keyboard.rs`) or an edit is stored. The page polls nothing; it reads file contents only when told they changed.
- Run starts the loop and Pause asks it to stop after the slice in flight. Stop (`endTitle`) stops it at once, and the page ignores reports that arrive after Stop.
- The release profile aborts on panic, an empty `unreachable` trap on wasm; `switch_panic_message` returns the stored message without a live session. Linear memory survives the trap, so panic context is collected afterwards, each piece separately.
- The CPU tab's rate counts instructions retired, not the guest clock, which jumps forward when all threads are blocked. The status bar leaves the total out; it is on `<body data-instructions>`, where `browser_boot.mjs` reads it.

## `web/worker/storage.ts`

- The page never opens IndexedDB for the SD card, saves, profiles or NAND: it sends `storage*`, `saves*`, `users*` and `nand*` commands and this module does the reads, writes and zip, handing NAND content to `idb/content.rs`. Big payloads (an upload, a NAND image, an NRO) go as a `File` or a stored name and are read here, not copied through the page.
- The SD card and saves are read and changed in the session (`storage.rs` exports), which is live from startup, and stored by a flush after each edit, so the dialog and the guest share one view and one set of open files. An upload is copied into OPFS by range (`switch_fs_upload`), not read into memory. A command that touches the core's storage runs between slices like any command, so a large export or install pauses emulation, not the page.
- Only the log's mirror (`web/main/log.ts`) still uses IndexedDB from the page.

## `web/main/session.ts`

- The session is the console, created and staged once per page (SD card, saves, controls, users, keys, archives, library applets). A title is a process in it: every launch and Stop calls `endTitle`, which has the core end the title and its applets (`Cpu::end_title`) and leaves an empty process for the next one, so nothing is restaged.
- Before titles were processes, every boot rebuilt the session: loading into the running one left the previous title's pages mapped (hundreds of MiB for retail) until guest RAM hit the cap.
- `endTitle` then restages the users: a running title keeps the player it started with, and profile edits made meanwhile reach the next one.
- The NACP figures (save sizes, cache storage, add-on base id) live on the process, so `endTitle` drops the ones `loadControl` set when the container opened. `loadControl` keeps the container's NACP in the session (`container_nacp`), and `switch_launch` gives it to the new process before booting.

## `web/main/index.ts`, `web/main/boot.ts`, `web/main/dock.ts`, `web/main/wakelock.ts`

- Init awaits only the NAND index, so a firmware dump does not delay the first boot. Archives register in the background: `switch_restore_nand` registers archives, library applets and the SYSTEM listing in one call and posts a `nand_progress` message after each archive, which drives the firmware panel's count.
- Operation mode affects `vi` resolution, `am`/`apm` performance mode, `clkrst` GPU clock and touchscreen presence. Titles read it once, so changes queue the two AM messages a real dock sends.
- Screen Wake Lock needs a secure context (missing in Firefox and Safari before 16.4). The browser drops the lock when the document is hidden and does not return it, so it is re-taken on visibility; a lock acquired after the run ended is released immediately.
- Icons are Lucide (`icons.ts`), imported per icon so the bundle carries only those used.

## `web/main/shell.ts`

- At 900px and below the top bar doesn't fit, so `placeMoreItems` moves profile, dock, the Discord link, fullscreen and Step into the More popover, and back beside comment markers when the window widens. Moving the nodes keeps their listeners and the desktop order exactly.

## `web/main/memory.ts`

- The history comes from housekeeping, so it is there when the group is opened: one sample a second, five minutes kept, reset by `endTitle`. The region table and the heap are asked for once a second, only while the group is on screen.
- `switch_guest_ram` counts the title's process (`Cpu::title_backed`), not the running one; otherwise an applet on screen would dip the line and the status bar.
- The heap section reports before a title boots. The guest RAM section waits for `titleBooted`, not `hasSession`, which is true as soon as the core has loaded and built the console's one session.
- The chart's `viewBox` stretches to the panel: strokes use `vector-effect: non-scaling-stroke`, and the marker and the scale are HTML, since SVG circles and text would stretch with it. The scale sits on the caption row, where a line near the top cannot run through it.

## `web/main/loading.ts`

- The loading screen only owns its markup. `awaitFirstFrame` is the one open-ended phase, ended by `display.noteFrame` on the first painted frame or by the run loop on a fault, halt or pause. After a homebrew load the screen is shown under it until then, since homebrew can run a long time before presenting.
- Failures keep the loading screen up with the reason rather than revealing a black stage; a core that fails to start shows its error rather than an idle splash.

## `web/main/input.ts`

- hid reports touch in the console's 1280x720 digitizer space regardless of the guest's presentation size. Taps are mapped through the `object-fit: contain` rect, not the element box. A touch slot (finger id) stays fixed for a contact's life, claimed from the lowest free slot.
- Stick Y axes are negated (Horizon's points up); the core derives stick pseudo-buttons from these values, so the sign matters.

## `web/main/touchpad.ts`

- The on-screen controller shows on a coarse pointer with no gamepad connected, while a program is loaded, running or paused. Portrait puts it below a 16:9 game; landscape lays it over the game's sides, inside the safe area.
- Each finger is hit-tested again as it moves, so a thumb slides across the D-pad or face buttons; a stick keeps the finger that started on it. `input.ts` ORs the held buttons into its mask and takes the sticks unless a gamepad is connected.
- Landscape phones (height 500px and below) hide the top and status bars while a program is loaded; the chevron at the top shows them again.
- Chromium's mobile emulation sends no pointer events for mouse input, so `tests/e2e/touchpad.spec.ts` drives fingers through CDP touch events.

## `web/main/container.ts`

- Updates are paired, not opened: an update's RomFS is ranges indexed against the base game's ([formats.md](formats.md#ncars)). A held update is applied when its title launches; the session skips one for another game. Each update or add-on content file is handed to the session once, when it is dropped (a title mounts the add-on content numbered against it).
- The page remembers (localStorage `switch-wasm:updates` / `switch-wasm:dlc`) which update or DLC a title last ran with, since files do not survive a reload, so it can say so instead of silently launching unpatched.
- Stop re-hands the open container to the new session; the page still shows it, so "no container is open" would be wrong.
- XCI partitions are flattened by the wasm side, so the page treats them like NSPs.
- Without prod.keys the Control NCA cannot be identified (content type is in the encrypted header); a container without one (update or DLC) is a dim note, not an error.

## `web/main/filetype.ts`

- `accept` on an input is defeated by the picker's "All files" option and ignored by drops, so files are identified by header, not name. CDN NCAs are the exception: their header is encrypted until keys are loaded.
- iOS greys out every file whose extension it has no type for, which is all of ours, so on iPhone and iPad (iPadOS reports a Mac with a touchscreen) `index.ts` removes the extension `accept` lists at startup.
- Keys files are validated as `name = hex` lines so a binary picked by mistake is not stored as keys.

## `web/main/nand.ts`

- NCA content lives in OPFS, one file per NCA in the `nand` directory, read through `getFile()` so the reader worker takes it like any Blob. Where OPFS cannot be written (no `createSyncAccessHandle`, or Firefox's private windows) it stays in IndexedDB's `content` store as Blobs. The index is IndexedDB's `titles` store (by title id, so the panel lists installs without reading content). A third store, `controls`, holds each opened title's NACP and icon: `idb/controls.rs` stores them when the page loads a title's details and hands them to the session at start.
- `idb/content.rs` owns the content; `opfs/mod.rs` is the OPFS layer under it. An install is written to `<name>.part` with a sync access handle, moved over `<name>`, and only then indexed, so the index never names a partial file. Writes and removals run one at a time.
- A File from OPFS stops reading once its file is removed or rewritten, and the session keeps its host files for the worker's life. So content handed out (registered, launched, downloaded) is not deleted when its title is removed or the NAND erased; the sweep at the next start removes every file the index does not name, and every `.part`.
- Content older builds kept in IndexedDB moves to OPFS in the background after the sweep, one file per queued job, each IndexedDB entry deleted once its copy is whole. Reads look in OPFS first and IndexedDB second, so nothing waits for the move. The emptied `content` store stays: dropping it takes a version bump of a database that also holds saves, and the fallback still writes it.
- Every stored NCA is also listed read-only at `/Contents/registered/<name>` on the SYSTEM BIS partition (`add_nand_content`), read by range from its host file.
- Data archives and library applets (`add_applet`) are registered by `restoreNand` (`idb/nand.rs`); the page queues registrations so each runs after the last. The core runs an applet as a process beside its caller.
- Registering, installing firmware, erasing the NAND and removing a title are NAND operations (`nandOperation`, `web/main/nandbusy.ts`). While one runs no title boots: every launch path checks first (`refuseBootWhileNandBusy`, before a running title is ended for it) and the Open buttons and launch rows are disabled. Applets register last, after every archive, so a title booted during the background registration found offlineWeb "not on the NAND" and Minecraft's NgWord2 missing.
- Each firmware NCA is identified from its encrypted header before it is stored, so the install buttons stay disabled until prod.keys is loaded, as Eden checks for keys before its picker opens. Without the check an install read every NCA and listed them all as failed.
- The Keys section heads the NAND tab: keys belong to the console as its firmware does, and installing firmware needs them first. Messages that ask for prod.keys from elsewhere (the Container section's encrypted NCA) name the NAND tab.

## `web/main/files.ts`

- The dialog lists and edits the session's card and saves through [`web/worker/storage.ts`](#webworkerstoragets), so a new card shows the `switch` folder the core makes for homebrew menus.
- A folder can exist only as the prefix of deeper paths, so listings derive folders from paths, and delete and rename act on every path under the target. Removing a directory from the `Vfs` removes what is under it too, each path recorded as a change.
- Exports are store-only zips (`web/worker/zip.ts`): saves and SD folders are small, and it needs no dependency. No zip64, so it refuses past 4 GiB or 65535 entries.
- `shell.ts` binds `.panel .tab` only; the dialog's tabs are `.files-tab` so the panel's handler does not take them.
- Opening a file (`fileview.ts`): valid UTF-8 without NUL bytes is edited in a textarea (read-only past 4 MiB), PNG/JPEG/GIF/WebP/BMP is shown by magic number, anything else as a hex dump of its first 4 KiB. Leaving or closing with unsaved edits asks first, Escape included.

## `web/main/users.ts`

- A title asks which user is playing once, at startup. Idle sessions get profile changes immediately; a running title keeps its user ([`endTitle`](#webmainsessionts) passes edits to the next one).
- The last profile cannot be deleted (no title starts without a user), nor the one a running title uses (its saves would be orphaned).
- Profiles are stored in the NAND database next to their saves.
- Profile edits the guest makes through `acc` are stored by the wasm side (`idb/users.rs`) over the stored profile of the same uid; the same tick posts `profiles_edited` and the page reloads its list from IndexedDB.

## `web/main/log.ts`

- The page view keeps 2000 entries (layout cost with tracing on); the backlog keeps every entry until Clear, for copy and save.
- The log is mirrored to IndexedDB on a timer (not per line) so it survives the browser killing the tab. The mirror is whole, so a very long session's log counts against the origin's quota, which the SD card and NAND share.
- Unclean-exit detection: a `pagehide` handler writes a mark synchronously to `localStorage` (no time to await IndexedDB). The mark is per origin, so other tabs are pinged over `BroadcastChannel` before it is treated as a dead session; pings carry a tab id because a page receives its own channel messages.
- Clipboard falls back to a selection copy because `navigator.clipboard` needs a secure context (e.g. testing on a phone over plain http).

## `web/main/battery.ts`

- Only Chromium exposes the Battery Status API; elsewhere the emulated battery stays at the wasm default (full, charging). Updates are event-driven.

## `vite.config.ts`

- `web/public` holds the social card (other sites cache its URL from meta tags), the home screen icons and `manifest.webmanifest` (fixed URLs), the font licence (the font itself is embedded in the wasm) and `_headers` (cross-origin isolation in deployment). Everything else is content-hashed; `assetsInlineLimit: 0` keeps every asset a file with a hashed URL.
- `base: './'` because the site lives at tenshii.moe/Switch-Wasm/; '/' 404s only once deployed.
- The worker must be a module on both sides: 'iife' fails in dev (the dev server serves the entry unbundled, so `import` breaks a classic worker), and dropping `type` in production only works until a chunk gains an `import`.
