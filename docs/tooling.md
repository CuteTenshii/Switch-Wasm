# Tooling

Build profiles, hooks, benchmarks and diagnostic examples.

## `tools/wasm_bench.mjs`

- Host timings are not a scaled version of browser timings: removing a libcall only wasm pays for was about 1.15x natively and 1.44x in the browser. Use host examples (`frame_work`) for work counts, and this tool for milliseconds.
- The bench mirrors `web/worker/hostfiles.ts`'s per-file LRU (same 1 MiB chunks, depth, and large-read bypass) so the wasm side sees the shipped access pattern, even though node's `readSync` is cheaper than `FileReaderSync`.
- Frames are timed by sampling the frame counter between slices of a single run, because boot time varies by seconds between runs. One extra frame is timed and dropped because it absorbs the warmup's tail.
- It loads `target/wasm32-unknown-unknown/release` unless `--build=<dir>` names another wasm-bindgen output: comparing two builds means copying each out and naming it, or both runs measure the same one.
- `--cores=N` serves N more guest cores on `worker_threads` over the shared memory, as `web/worker/coreworkers.ts` does in the page.
- Native examples take `--cores <n>` for the same, and `--no-draw` to swap in a renderer that draws nothing, so a run times the CPU as it is with a GPU doing the drawing.

## `Cargo.toml`

- `[profile.test]` uses opt-level 0 because optimizing the large test targets made CI spend far longer compiling than running them; `switch-core` tests keep opt-level 1 because they execute guest code.
- `[profile.dev]` uses `debug = "line-tables-only"`: enough for `perf -s srcline`.
- `identity_op` and `unusual_byte_groupings` are allowed so hardware encodings (instruction words, Maxwell methods, system-register literals grouped as `op0_op1_CRn_CRm_op2`) can be written field by field, zero fields included, and read against the reference.

## `Makefile`

- The wasm build always includes the WebGPU backend: `wgpu` reaches WebGPU through `wasm-bindgen`, so the artifact is a wasm-bindgen module with glue, and shipping one shape of core is cheaper than maintaining two loaders. Without WebGPU the backend fails to open a device and the software rasterizer takes the frame.
- Both wasm builds use the pinned `NIGHTLY`, and `.cargo/config.toml` turns on `build-std` for it (stable cargo ignores that table): the core's memory is shared between workers, which needs `+atomics`, and stable's prebuilt std has none.
- `make assets` exists because the core is an input to the Vite build, and only make knows how to build it.
- `switch-gpu` tests run with `--test-threads=1`: parallel device creation can crash the software Vulkan driver on headless CI.

## `lefthook.yml`

- The rustfmt hook runs `cargo fmt --all --check` rather than rustfmt on staged files, because `include!`d files (generated ASTC tables) are formatted by bare rustfmt but not by cargo fmt. Clippy covers the whole workspace and all targets since switch-core changes break switch-wasm lints. Hooks read the working tree since cargo can't build from the index.

## `crates/switch-core/examples/screenshot_title.rs`

- Stops at the Nth presented frame rather than a step budget (a title needs billions of steps before its first frame).
- Flags beyond the shared ones (see [repro.md](repro.md); `--help` lists them all): `--firmware <dir>` registers system data archives; `--steps <n>` caps the run; `--profile <interval>` samples pcs by thread, page and return address, as `boot_nsp` does, and `--stacks` adds a call stack per sample (interval 4096 if `--profile` is unset); `--cover <lo>:<hi>` records executed instructions in a range; `--watch-mem <addr>` reports the first step a 4 KiB window becomes non-zero; `--scan-mem <addr>:<size>[,...]` lists non-zero spans at the end; `--dump-verts <addr>[,...]` reads three 60-byte rows as floats; `--dump-surface <addr>:<w>x<h>:<format>[:<block height>][,...]` writes a block-linear colour surface to `<out>.<addr>.ppm` (format as in the draw trace, block height in GOBs, default 16; float channels clamped, brightest reported); `--poke-u32 <addr>:<value>` writes every 4096 steps, or once at `--poke-at <step>`; `--gate-sniff` finds the applet frame-skip gate (`ldr w8, [xN, #0x3e8]`) and holds it at zero; `--find-magic SARC` scans memory for a magic.
- `--profile`, `--stacks`, `--cover`, `--poke-u32`, `--gate-sniff`, `--trap-write`, `--trap-read` and `--watch-pc` force `Pace::Instructions`; otherwise the run uses the block translator like the frontend.

## `tools/browser_boot.mjs`

- Samples the run state (`body[data-state]`) and the page's console every 5 s so stalls are observed, then saves the Debug panel's Rendering section and the full console, periodic activity and thread reports included. `--jit-stats` adds the Translation section's block stats to every sample, which only the browser build can measure.
- WebGPU is force-enabled (headless Chromium leaves it off); headless adapters are usually SwiftShader, valid for draw-path coverage but not speed. Each run uses a fresh profile so stored keys, SD card, saves and NAND do not leak between runs.
- A fresh profile has an empty NAND, so library applets report themselves cancelled; `--firmware=<dir>` installs a firmware directory's NCAs first.
- The run fails at once, with the console's last lines and exit code 1, when the title is not running within 2 minutes or the page logs an error; before this check a boot that never started sampled an idle page for the whole run.
- The console comes from the page's Save download: the view keeps only the newest 2000 lines. Trace from the start floods the log and can leave the page too busy for the Rendering section, so `--trace-from=<text>` switches to trace once a line contains the text.
- `--no-webgpu` gives the software rasterizer every frame, the reference a device frame is compared against. No Chromium switch removes WebGPU (`--disable-features=WebGPU` leaves it on), so the tool deletes `gpu` from the page's and the worker's navigator. `--screenshot=<file>` saves the screen canvas at the end.

- `--nand=<name>` launches a title from the NAND tab instead of a container, and `--docked` starts docked. `--dump-wgsl=<dir>` writes every shader the worker hands the browser. `--browser-arg=<switch>` passes a Chromium switch. `--stream=<regex>` prints matching console lines as they appear, for reading a run while it goes. `--trace=<file>@<seconds>:<length>` records a Chromium trace from that long after the title starts, which `tools/trace_summary.mjs` totals by event per thread: the only view into the GPU process, which refuses perf.

## `tools/common.mjs`

- What the tools share: the repo root, `fail`, flag parsing and launching Chromium. Playwright loads only when a tool launches it.

## `tools/shader_compile.mjs`

- Times the driver over shaders from `--dump-wgsl`, slowest first, with Chromium's GPU watchdog off. A literal is changed per run, since the driver's disk cache otherwise answers in about 100 ms.
- `--replace=pattern::text` times a change to the WGSL next to the original. This is how a branch in `ftz` was found to cost 50 s of a 54 s compile.
- Opens a window, because only a headed browser gets the machine's GPU. Times are `createRenderPipelineAsync`, not what a synchronous pipeline costs the worker.

## `tools/profile_ingame.mjs`

- Runs `browser_boot.mjs`, waits until `--settle` samples past `--after` sit inside `--when-fps=low-high`, then records `--record` seconds with perf and ends the run. Other flags go to `browser_boot.mjs`.
- Writes `report.txt`, `perf.data`, `top.txt` and `boot.txt` to `--dir`. It resolves wasm function names itself from V8's `/tmp/perf-<pid>.map`, because perf does not apply it to V8's anonymous pages.
- Chromium's sandbox is off for the run, or perf cannot attach. The GPU process refuses perf, so it only appears through `top`'s per-thread CPU.

## `tools/log_summary.mjs`

- Prints a timeline, faults, refused commands, the pad, device timers, the last frames and the busiest guest threads from a `browser_boot.mjs --out` report or a saved page log.

## `crates/switch-core/examples/difftest.rs`

- Driven by `tools/difftest.py`: assembles instructions, runs them under qemu-aarch64 and here, and reports the first differing register (found the TRN1/TRN2 lane mix-up behind hbmenu's JPEG decode). Usage `difftest <code.bin> <inputs.bin> <out.bin> <inputs-address-hex>`. The dump buffer is sized from the program: a fixed 128 once silently stopped comparing past 128.

## `crates/switch-core/examples/hotspots.rs`

- The group table mirrors `jit::decode`'s arms, so it ranks work inside the translator (loads and stores walk the page table; SIMD/FP re-derive operands each execution). `jit_coverage` measures what has no op; this measures what has an op and still costs. NRO rankings do not transfer to retail. Addresses are keyed by page, not a fixed window, since retail modules load anywhere.

## `crates/switch-core/examples/frame_work.rs`

- Work counts (instructions, blocks, interpreter fallbacks, draws, pixels) are identical under native and V8; use them to decide what to fix and prove a fix landed, and `tools/wasm_bench.mjs` for what it was worth. A change that moves no count but looks faster on the host only made the host faster.
- Startup frames (loader, allocator, first upload) are skipped so they don't skew the steady-state mean.

## `tools/make_font.py`

- `web/font.ttf` is a stand-in shared font, built by this script. The guest renders the shared font with its own FreeType, so it must be a real TTF/OTF; `pl:u` serves the firmware's fonts instead when their system data archives are registered.
- The charset is subset to keep it small (full CJK is tens of MB).
- Hinting is stripped (`--no-hinting`): hinted glyphs rendered collapsed horizontally under the emulator, and hinting bytecode cost about 8x more emulated instructions per frame.
- Nintendo's private-use button glyphs (the dark and light theme sets nx-hbmenu uses) are mapped to matching letters and signs; otherwise hints render as wrong glyphs.
