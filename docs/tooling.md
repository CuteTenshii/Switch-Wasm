# Tooling

Build profiles, hooks, benchmarks and diagnostic examples.

## `tools/wasm_bench.mjs`

- Host timings are not a scaled version of browser timings: removing a libcall only wasm pays for was about 1.15x natively and 1.44x in the browser. Use host examples (`frame_work`) for work counts, and this tool for milliseconds.
- The bench mirrors `web/worker/hostfiles.ts`'s per-file LRU (same 1 MiB chunks, depth, and large-read bypass) so the wasm side sees the shipped access pattern, even though node's `readSync` is cheaper than `FileReaderSync`.
- Frames are timed by sampling the frame counter between slices of a single run, because boot time varies by seconds between runs. One extra frame is timed and dropped because it absorbs the warmup's tail.
- Running without `web/font.ttf` understates frame cost (no text is drawn).

## `Cargo.toml`

- `[profile.test]` uses opt-level 0 because optimizing the large test targets made CI spend far longer compiling than running them; `switch-core` tests keep opt-level 1 because they execute guest code.
- `[profile.dev]` uses `debug = "line-tables-only"`: enough for `perf -s srcline`.
- `identity_op` and `unusual_byte_groupings` are allowed so hardware encodings (instruction words, Maxwell methods, system-register literals grouped as `op0_op1_CRn_CRm_op2`) can be written field by field, zero fields included, and read against the reference.

## `Makefile`

- The wasm build always includes the WebGPU backend: `wgpu` reaches WebGPU through `wasm-bindgen`, so the artifact is a wasm-bindgen module with glue, and shipping one shape of core is cheaper than maintaining two loaders. Without WebGPU the backend fails to open a device and the software rasterizer takes the frame.
- Both wasm builds use the pinned `NIGHTLY` with `-Z build-std`: the core's memory is shared between workers, which needs `+atomics`, and stable's prebuilt std has none.
- `make assets` exists because the core is an input to the Vite build, and only make knows how to build it.
- `switch-gpu` tests run with `--test-threads=1`: parallel device creation can crash the software Vulkan driver on headless CI.

## `lefthook.yml`

- The rustfmt hook runs `cargo fmt --all --check` rather than rustfmt on staged files, because `include!`d files (generated ASTC tables) are formatted by bare rustfmt but not by cargo fmt. Clippy covers the whole workspace and all targets since switch-core changes break switch-wasm lints. Hooks read the working tree since cargo can't build from the index.

## `crates/switch-core/examples/screenshot_title.rs`

- Stops at the Nth presented frame rather than a step budget (a title needs billions of steps before its first frame).
- Knobs beyond the shared debugging ones (see [repro.md](repro.md)): `SWITCH_FIRMWARE=<dir>` registers system data archives; `STEPS=<n>` caps the run; `PROFILE=<interval>` samples hot pcs, pages and threads, and `STACKS=1` adds a return-address histogram (interval 4096 if `PROFILE` is unset); `COVER=<lo>:<hi>` records executed instructions in a range; `WATCH_MEM=<addr>` reports the first step a 4 KiB window becomes non-zero; `SCAN_MEM=<addr>:<size>[,...]` lists non-zero spans at the end; `DUMP_VERTS=<addr>[,...]` reads three 60-byte rows as floats; `DUMP_SURFACE=<addr>:<w>x<h>:<format>[:<block height>][,...]` writes a block-linear colour surface to `<out>.<addr>.ppm` (format as in the draw trace, block height in GOBs, default 16; float channels clamped, brightest reported); `POKE_U32=<addr>:<value>` writes every 4096 steps, or once at `POKE_AT=<step>`; `GATE_SNIFF=1` finds the applet frame-skip gate (`ldr w8, [xN, #0x3e8]`) and holds it at zero; `FIND_MAGIC=SARC` scans memory for a magic.
- `PROFILE`, `STACKS`, `COVER`, `POKE_U32`, `GATE_SNIFF`, `TRAP_WRITE`, `TRAP_READ` and `WATCH_PC` force `Pace::Instructions`; otherwise the run uses the block translator like the frontend.

## `tools/browser_boot.mjs`

- Samples the run state (`body[data-state]`) and the page's console every 5 s so stalls are observed, then saves the Debug panel's Rendering section and the full console, periodic activity and thread reports included. `--jit-stats` adds the Translation section's block stats to every sample, which only the browser build can measure.
- WebGPU is force-enabled (headless Chromium leaves it off); headless adapters are usually SwiftShader, valid for draw-path coverage but not speed. Each run uses a fresh profile so stored keys, SD card, saves and NAND do not leak between runs.

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
