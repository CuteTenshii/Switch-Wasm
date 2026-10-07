# Tooling

Build profiles, hooks, benchmarks and diagnostic examples.

## `tools/wasm_bench.mjs`

- Host timings are not a scaled version of browser timings: removing a libcall only wasm pays for was about 1.15x natively and 1.44x in the browser. Use host examples (`frame_work`) for work counts, and this tool for milliseconds.
- The bench mirrors `web/worker/hostfiles.ts`'s 1 MiB LRU (same chunk size, depth, and large-read bypass) so the wasm side sees the shipped access pattern, even though node's `readSync` is cheaper than `FileReaderSync`.
- Frames are timed by sampling the frame counter between slices of a single run, because boot time (first frame around step 900M on retail) varies by seconds between runs. One extra frame is timed and dropped because it absorbs the warmup's tail.
- Running without the shared font understates frame cost (no text is drawn).

## `Cargo.toml`

- `[profile.test]` uses opt-level 0 because optimizing the large test targets made CI spend far longer compiling than running them; `switch-core` tests keep opt-level 1 because they execute guest code.
- `[profile.dev]` uses `debug = "line-tables-only"`: enough for `perf -s srcline`.
- `identity_op` and `unusual_byte_groupings` are allowed so hardware encodings (instruction words, Maxwell methods, system-register literals grouped as `op0_op1_CRn_CRm_op2`) can be written field by field, zero fields included, and read against the reference.

## `Makefile`

- The wasm build always includes the WebGPU backend: `wgpu` reaches WebGPU through `wasm-bindgen`, so the artifact is a wasm-bindgen module with glue. Shipping one shape of core is cheaper than maintaining two loaders. Without WebGPU the backend fails to open a device and the software rasterizer takes the frame.
- `wasm` is the incremental dev profile; `wasm-release` is fat LTO in one codegen unit (slow to build), and the only build performance is quoted from.
- `make assets` exists because the core is an input to the Vite build, and only make knows how to build it.
- `switch-gpu` tests run with `--test-threads=1`: parallel device creation can crash the software Vulkan driver on headless CI.

## `crates/switch-core/examples/screenshot_title.rs`

- Stops at the Nth presented frame rather than a step budget (a title needs billions of steps before its first frame).
- Knobs: `SWITCH_FIRMWARE=<dir>` system data archives (applets need fonts/icons/settings); `STEPS=<n>` cap; `PROFILE=<interval>` samples hot pcs/pages/threads, `STACKS=1` adds a return-address histogram (implies interval 4096); `COVER=<lo>:<hi>` records executed instructions in a range; `WATCH_MEM=<addr>` reports first step a 4 KiB window becomes non-zero; `SCAN_MEM=<addr>:<size>[,...]` lists non-zero spans at the end; `DUMP_VERTS=<addr>[,...]` reads three 60-byte rows as floats; `DUMP_SURFACE=<addr>:<w>x<h>:<format>[:<block height>][,...]` writes a block-linear colour surface to `<out>.<addr>.ppm` (format as in the draw trace, block height in GOBs, default 16; float channels clamped, max reported); `POKE_U32=<addr>:<value>` writes every tick or once at `POKE_AT=<step>`; `START_THREADS=<step>` makes never-started threads runnable; `WAKE_ALL=<period>` wakes all blocked threads periodically; `GATE_SNIFF=1` finds the applet frame-skip gate (`ldr w8, [xN, #0x3e8]`) and holds it at zero (on 18.0.1 qlaunch this yields 0 draws instead of 8); `FIND_MAGIC=SARC` scans memory for a magic.
- Any hook forces `Pace::Instructions` (about half speed); without hooks the run uses the block translator like the frontend.

## `tools/browser_boot.mjs`

- Samples run state and console row count every 5 s so stalls are observed, then presses "GPU stats" and "Thread dump" and saves the full console. WebGPU is force-enabled (headless Chromium leaves it off); headless adapters are usually SwiftShader, valid for draw-path coverage but not speed. `--jit-stats` deltas between samples show what the translator and emitter did, which only the browser build can measure. Each run uses a fresh profile so stored keys, SD card, saves and NAND do not leak between runs.

## `crates/switch-core/examples/difftest.rs`

- Driven by `tools/difftest.py`: assembles instructions, runs them under qemu-aarch64 and here, reports the first differing register (found the TRN1/TRN2 lane mix-up behind hbmenu's JPEG decode). Usage `difftest <code.bin> <inputs.bin> <out.bin> <inputs-address-hex>`. The dump buffer is sized from the program: a fixed 128 once silently stopped comparing past 128.

## `crates/switch-core/examples/hotspots.rs`

- The group table mirrors `jit::decode`'s arms, so it ranks work inside the translator (loads/stores walk the page table; SIMD/FP re-derive operands each execution). `jit_coverage` measures what has no op (under 1.5% on homebrew and retail); this measures what has an op and still costs. NRO rankings do not transfer to retail. hbmenu spends most of a frame in its own software gradient fill. Addresses are keyed by page, not a fixed 16 MiB window, since retail modules load anywhere.

## `crates/switch-core/examples/frame_work.rs`

- Work counts (instructions, blocks, interpreter fallbacks, draws, pixels) are identical under native and V8; use them to decide what to fix and prove a fix landed, and `tools/wasm_bench.mjs` for what it was worth. A change that moves no count but looks faster on the host only made the host faster.
- Startup frames (loader, allocator, first upload) are skipped so they don't skew the steady-state mean. NRO workloads don't rank like retail titles.

## `lefthook.yml`

- rustfmt hook runs `cargo fmt --all --check` rather than rustfmt on staged files, because `include!`d files (generated ASTC tables) are formatted by bare rustfmt but not cargo fmt. Clippy covers the whole workspace and all targets since switch-core changes break switch-wasm lints (~9 s warm). Hooks read the working tree since cargo can't build from the index.

## `tools/make_font.py`

- The guest renders the shared font with its own FreeType, so it must be a real TTF/OTF. The charset is subset to keep it small (full CJK is tens of MB).
- Hinting is stripped: hinted glyphs rendered collapsed horizontally under the emulator, and hinting bytecode cost about 8x more emulated instructions per frame.
- Nintendo's private-use button glyphs (0xE0Ax dark theme, 0xE0Ex light theme, used by nx-hbmenu) are mapped to similar letters/signs; otherwise hints render as wrong glyphs.
