# Tooling

Build profiles, hooks, benchmarks and diagnostic examples.

## `Cargo.toml`

- `[profile.test]` uses opt-level 0: optimizing the large test targets made CI spend far longer compiling than running them. `switch-core` tests keep opt-level 1 because they execute guest code.
- `[profile.dev]` uses `debug = "line-tables-only"`: enough for `perf -s srcline`.
- `identity_op` and `unusual_byte_groupings` are allowed so hardware encodings (instruction words, Maxwell methods, system-register literals grouped as `op0_op1_CRn_CRm_op2`) can be written field by field, zero fields included, and read against the reference.

## `Makefile`

- The wasm build always includes the WebGPU backend: `wgpu` reaches WebGPU through `wasm-bindgen`, so the artifact is a wasm-bindgen module with glue, and one shape of core is cheaper than two loaders. Without WebGPU the backend fails to open a device and the software rasterizer takes the frame.
- Both wasm builds use `NIGHTLY` (plain `nightly`; override it per build to pin a date), and `.cargo/config.toml` turns on `build-std` for it (stable cargo ignores that table): the core's memory is shared between workers, which needs `+atomics`, and stable's prebuilt std has none.
- `make assets` exists because the core is an input to the Vite build, and only make knows how to build it.
- `switch-gpu` tests take turns on one shared device (`tests::device` holds a lock): concurrent use of a software Vulkan device fails at random, and the lock lets one `cargo test --workspace` run every crate.

## `lefthook.yml`

- The rustfmt hook runs `cargo fmt --all --check` rather than rustfmt on staged files: bare rustfmt formats `include!`d files (generated ASTC tables), cargo fmt does not. Clippy covers the whole workspace and all targets, since switch-core changes break switch-wasm lints. Hooks read the working tree because cargo can't build from the index.

## `crates/switch-gpu/examples/screenshot.rs`

- Draws on the GPU backend through native wgpu and falls back per draw; with no adapter, or with `--software`, every draw is the rasterizer's. The rasterizer is the reference and is far slower once draws cover pixels, so use `--software` only to check the device against it. Run it as `cargo run --release -p switch-gpu --example screenshot`; building `-p switch-core` leaves a stale binary.
- It prints the Rendering panel's report (`rendering: {...}`) at the end. The device switches: `--only <i>` (or `<a>..<b>`) puts only those draws on the device; `--web-limits` hides native-only features and `--defer-readbacks` never waits for a readback, so a native run takes the browser's routes and latches as it does, which narrows down a fallback a browser run reported but does not list the browser's (only `browser_boot.mjs` does); `--interleave`, `--sync-every-present`, `--device-msaa` and `--times` are for comparing routes, and `--software-compute` runs every compute launch in software, to compare against the device's. `--software` refuses them.
- `--docked` docks before boot; `--dock-at <frame>` docks mid-run like a real dock, telling the running title through AM messages. It includes `switch-core`'s `examples/common` by path, so the two crates' tools share one copy.
- Flags beyond the shared ones (see [repro.md](repro.md#arguments); `--help` lists them all): `--steps <n>` caps the run (unlimited for a retail title, 200 million for an `.nro`); `--sd <dir>` copies a directory onto the SD card and launches an `.nro` inside it from there, as `boot`'s does; `--profile <interval>` samples pcs by thread, page and return address, as `boot` does, and `--stacks` adds a call stack per sample (interval 4096 if `--profile` is unset); `--cover <lo>:<hi>` records executed instructions in a range; `--watch-mem <addr>` reports the first step a 4 KiB window becomes non-zero; `--scan-mem <addr>:<size>[,...]` lists non-zero spans at the end; `--dump-verts <addr>[,...]` reads three 60-byte rows as floats; `--dump-surface <addr>:<w>x<h>:<format>[:<block height>][,...]` writes a block-linear colour surface to `<out>.<addr>.ppm` (format as in the draw trace, block height in GOBs, default 16; float channels clamped, brightest reported); `--poke-u32 <addr>:<value>` writes every 4096 steps, or once at `--poke-at <step>`; `--gate-sniff` finds the applet frame-skip gate (`ldr w8, [xN, #0x3e8]`) and holds it at zero; `--find-magic SARC` scans memory for a magic.
- `--profile`, `--stacks`, `--cover`, `--poke-u32`, `--gate-sniff`, `--trap-write`, `--trap-read` and `--watch-pc` force `Pace::Instructions`; otherwise the run uses the block translator like the frontend.

## `tools/browser_boot.mjs`

- Samples the run state (`body[data-state]`) and the page's console every 5 s so stalls are observed, then saves the Debug panel's Rendering and Display sections and the full console, periodic activity and thread reports included. `--jit-stats` adds the Translation section's block stats to every sample; only the browser build can measure them.
- WebGPU is force-enabled. A window opens so the browser gets the machine's GPU; without one Chromium falls back to SwiftShader, where readbacks never land and no device frame is presented. Each run uses a fresh profile so stored keys, SD card, saves and NAND do not leak between runs.
- A fresh profile has an empty NAND, so library applets report themselves cancelled; `--firmware=<dir>` installs a firmware directory's NCAs first.
- `--update=<nsp>` opens an update before the container and waits for the page to hold it, so the title launches patched, as dropping the update first does in the page.
- The run fails at once, with the console's last lines and exit code 1, when the title is not running within 2 minutes or the page logs an error. Before this check, a boot that never started sampled an idle page for the whole run.
- The console comes from the page's Save download, because the view keeps only the newest 2000 lines. Trace from the start floods the log and can leave the page too busy for the Rendering section, so `--trace-from=<text>` switches to trace once a line contains the text.
- `--no-webgpu` gives the software rasterizer every frame, the reference a device frame is compared against. No Chromium switch removes WebGPU (`--disable-features=WebGPU` leaves it on), so the tool deletes `gpu` from the page's and the worker's navigator. `--screenshot=<file>` saves the screen canvas at the end.
- `--nand=<name>` launches a title from the NAND tab instead of a container, and `--docked` starts docked. `--dump-wgsl=<dir>` writes every shader the worker hands the browser. `--browser-arg=<switch>` passes a Chromium switch. `--stream=<regex>` prints matching console lines as they appear, to read a run while it goes. `--port=<n>` serves on that port instead of a free one, so a relay can be started with `--allow-origin http://127.0.0.1:<n>`. `--relay=<ws url>` turns the Network tab on through that relay before the title starts and reports the tab's socket line and connection table at the end, so a guest's network use is readable without opening the panel. `--trace=<file>@<seconds>:<length>` records a Chromium trace from that long after the title starts, which `tools/trace_summary.mjs` totals by event per thread: the only view into the GPU process, which refuses perf.
- `--script=<file|steps>` drives a title step by step (`wait`, `press`, `pressuntil`, `text`, `log`, `shot`; see the tool's header) to reach a late screen without a human. `--press` can only tap keys, and its taps cancel a keyboard dialog, which Just Dance 2022's nickname prompt then asks for again. `tools/scripts/just-dance-2022.txt` reaches the menu in 160 s. A failed step stops the script and sets the exit code.
- `--seconds=N` (default 60) is how long the title runs; a fault or halt ends it sooner. `--out=<file>` writes the report instead of printing the console, and `--log-level=<level>` sets the console's level before the boot. `--press=<key>@<seconds>[:<every>],...` taps a key as `web/main/input.ts` maps it, closing an open dialog first. `--dock-at=N` toggles the dock N s in. `--software-at=N` hands drawing to the software rasterizer N s in, `--software-after=B` once B billion instructions have run and the screen is lit. `--dump-texture=<dir>@<W>x<H>[x<layers>]` reads back the device textures of that size at the end; not with `--dump-wgsl` or `--no-webgpu`. `--trace-cpu` adds V8's sampling profiler to `--trace`, for `tools/trace_cpu.mjs`. `--screenshot-every=N` also saves the screen every N s beside `--screenshot`'s file.

## `tools/common.mjs`

- What the tools share: the repo root, `fail`, flag parsing and launching Chromium. Playwright loads only when a tool launches it.

## `tools/shader_compile.mjs`

- Times the driver over shaders from `--dump-wgsl`, slowest first, with Chromium's GPU watchdog off. A literal is changed per run, since the driver's disk cache otherwise answers in about 100 ms.
- `--replace=pattern::text` times a change to the WGSL next to the original. It found a branch in `ftz` costing 50 s of a 54 s compile.
- Opens a window, because only a headed browser gets the machine's GPU. Times are `createRenderPipelineAsync`, not what a synchronous pipeline costs the worker.

## `tools/profile_ingame.mjs`

- Runs `browser_boot.mjs`, waits until `--settle` samples past `--after` sit inside `--when-fps=low-high`, then records `--record` seconds with perf and ends the run. Other flags go to `browser_boot.mjs`.
- Writes `report.txt`, `perf.data`, `top.txt` and `boot.txt` to `--dir`. It resolves wasm function names itself from V8's `/tmp/perf-<pid>.map`, because perf does not apply it to V8's anonymous pages.
- Chromium's sandbox is off for the run, or perf cannot attach. The GPU process refuses perf, so it only appears through `top`'s per-thread CPU.
- On macOS it uses `sample` on the page process (1 ms interval, `sample.txt`) instead of perf and `top`. The report ranks the busy frames at the top of stack, idle kernel waits set aside, and gives the page's and the GPU process's CPU from `ps`, without per-thread CPU. Wasm frames are named by function index, as V8 names them.

## `tools/log_summary.mjs`

- Prints a timeline, faults, refused commands, the pad, device timers, the last frames and the busiest guest threads from a `browser_boot.mjs --out` report or a saved page log.

## `crates/switch-core/examples/difftest.rs`

- Driven by `tools/difftest.py`: assembles instructions, runs them under qemu-aarch64 and here, and reports the first differing register (it found the TRN1/TRN2 lane mix-up behind hbmenu's JPEG decode). Usage `difftest <code.bin> --inputs <inputs.bin> --out <out.bin> --inputs-addr <hex>`. The dump buffer is sized from the program: a fixed 128 once silently stopped comparing past 128.

## `crates/switch-core/examples/hotspots.rs`

- The group table mirrors `jit::decode`'s arms, so it ranks work inside the translator (loads and stores walk the page table; SIMD/FP re-derive operands each execution). `jit_coverage` measures what the emitter has no writer for (`cpu::emits`); this measures where the instructions go, written or not. NRO rankings do not transfer to retail. Addresses are keyed by page, not a fixed window, since retail modules load anywhere.

## `crates/switch-core/examples/frame_work.rs`

- Work counts (instructions, blocks, interpreter fallbacks, draws, pixels) are identical under native and V8. Use them to decide what to fix and to prove a fix landed, and a `tools/browser_boot.mjs` or `tools/profile_ingame.mjs` run for what it was worth. A change that moves no count but looks faster on the host only made the host faster.
- Host timings are not a scaled version of browser timings: removing a libcall only wasm pays for was about 1.15x natively and 1.44x in the browser.
- Takes the `Play` flags ([repro.md](repro.md#arguments)); with `--no-draw` a run times the CPU as it is with a GPU doing the drawing.
- Startup frames (loader, allocator, first upload) are skipped so they don't skew the steady-state mean.

## `tools/make_font.py`

- Needs `fonttools`: `pip install -r requirements.txt`.
- `web/font.ttf` is a stand-in shared font, built by this script. The guest renders the shared font with its own FreeType, so it must be a real TTF/OTF; `pl:u` serves the firmware's fonts instead when their system data archives are registered.
- The charset is subset to keep it small (full CJK is tens of MB).
- Hinting is stripped (`--no-hinting`): hinted glyphs rendered collapsed horizontally under the emulator, and hinting bytecode cost about 8x more emulated instructions per frame.
- Nintendo's private-use button glyphs (the dark and light theme sets nx-hbmenu uses) are mapped to matching letters and signs; otherwise hints render as wrong glyphs.
