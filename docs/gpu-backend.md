# The `wgpu` backend

The rules AGENTS.md summarises, in full. `switch-gpu` is the `wgpu` backend
behind `gpu::renderer::Renderer`; the software rasterizer is the reference it
must agree with.

**It never blocks in a browser.** Reading a render target back means awaiting a
promise, and a blocking wait there is not slow but *deadlocked*. So a surface
stays on the device across every draw targeting it and returns to guest memory
only at `Renderer::flush`, which the engine calls before `present`; opening the
device is likewise deferred: `worker/index.ts` calls `switch_gpu_open`
*between* run slices, since the channel does not exist until the title has run.
**Any draw the backend cannot express falls back to `Software`.**
`Renderer::flush` polls with `PollType::Wait`, a real wait natively and a
*no-op* on WebGPU, so a browser gets `Flush::Pending` and the present waits for
a later slice (`Cpu::complete_pending_present`).

**Where a readback lands late, a frame is all one renderer's.** The first flush
answering `Pending` sets `Gpu::deferred_readbacks`, and from then on a frame in
which anything fell back makes the frames after it the rasterizer's whole
(`Gpu::software_frame`). **It latches on purpose**, because alternating costs
a frame with the rasterizer's draws under a readback each time, **but it lets
go**: each draw of a rasterizer's frame goes through `Gpu::check` (pipeline
state, surfaces, route, cached translations; no uploads), and after enough
frames in a row in which every draw passed, the device has the frames again.
One clean frame releases it the first time; each time it has to close again
after a release, the wait doubles, since the check cannot see a fallback that
an upload or the device itself causes. Tomodachi Life is why: one draw in its
first 740 frames samples a per-pixel bindless handle, and a latch that never
let go gave the rest of the session to the rasterizer. What buys the
acceleration back is still `shader::wgsl`: almost every fallback that latches
it is an opcode with no WGSL form, not anything WebGPU withholds.

**A copy out of a held surface flushes first.** The 2D blitter and the copy
engine read guest memory, so `channel.rs` hands the surfaces back before
`Engine2D::LAUNCHES_BLIT` and `copy::LAUNCH_DMA`, the same guard compute had.

**Depth, clears and multisampling all run on the device.** A depth surface is
held like a colour one, converted to `depth16unorm` or `depth32float` (the two
formats a copy can read) with the stencil byte read out of guest memory and
put back, since neither renderer tests stencil. Nothing copies *into*
`depth32float`, so a surface gets there by being drawn: a fullscreen triangle
writing `@builtin(frag_depth)`. Clears are a pass's load operation where they
cover the whole surface and a scissored fullscreen draw where they do not.

**Multisampling has two routes, and the default is the exact one.** Maxwell
stores samples *spatially*: a pixel owns a `samples_x` by `samples_y` tile of
texels, so guest memory holds the expanded image and the default route renders
exactly that, one fragment per texel, testing coverage at texel centres where
Maxwell's samples are; the sample mask and alpha-to-coverage become the
fragment shader's job (`wgsl::Coverage`). Two draws still fall back
deliberately: `MultisampleSampleLocations` away from texel centres
(`SampleGrid::samples_at_texel_centres`), and per-pixel coverage with a
*partial* sample mask. `GPU_DEVICE_MSAA=1` lets the device multisample instead;
off, because WebGPU's sample positions are a rotated grid that is not
Maxwell's, and core WebGPU only guarantees four samples, so `2x1`, `4x2` and
`4x4` take the expanded route regardless.

**Two renderers disagree by a 255th where a channel lands on a half**:
`ColorFormat::encode` rounds `127.5` up and a device's unorm conversion rounds
it down, so a test wanting byte-identity picks values off the eight-bit
half-way points.

**Checking the backend** is running `screenshot_title` and `screenshot_gpu`
over the same frame and `cmp`ing the PPMs; a byte-identical pair is the only
evidence it renders what the rasterizer does, and `GPU_ONLY=<i>` narrows a
difference to one draw. `gpu::testing::Harness` is the faster half: a drawable
`Engine3D` (a 16x8 target, two real shaders, three vertices) both renderers are
driven over, so a route can be checked without booting a title. **hbmenu is not
a shader-core test**: its command list is `dkCmdBufCopyBufferToImage` plus a
fence.

## Notes by file

### `crates/switch-gpu/src/convert.rs`

- Formats behind optional WebGPU features must be refused up front: `createTexture` throws on a missing feature and wgpu's web backend unwraps it, which on wasm is an `unreachable` trap that kills the core (Just Dance 2019's first BC1 texture). Refusal routes the draw to the software rasterizer.
- Normalized 16-bit formats (`TEXTURE_FORMAT_16BIT_NORM`) are native-only; no browser offers them. Sampled textures in those formats are widened to the 32-bit float sibling, which is exact (`f32` has 24 significand bits) and needs `float32-filterable`. Render targets in those formats stay refused: float targets neither clamp nor blend like normalized ones, and readback copies texels straight into a 16-bit guest surface.
- Attachment support is checked through the format's allowed usages, not required features: e.g. `rg11b10ufloat` samples without a feature but rendering needs `RG11B10UFLOAT_RENDERABLE`.

### `crates/switch-gpu/src/stats.rs`

- All device rejections are kept (distinct messages capped at 16, plus a total count), not just the first: the only production reader runs before pipeline creation, so later errors went unread.
- Upload accounting is in bytes rather than time (same on host and V8); it showed textures are ~96.5% of upload bytes, which is why only textures are cached. Cached textures are not counted.
- `flush` splits into ask/wait/land. In a browser `flush_wait` is ~0 because the wait moved to the slice boundary (`Flush::Pending`).
- Fallback reasons are JSON-escaped because the page parses the stats with `JSON.parse`.

### `crates/switch-gpu/src/lib.rs`

- `wgpu` lives in its own crate to keep `switch-core` dependency-free; the `Renderer` trait is the only shared surface.
- Never block in a draw: reading a texture back means awaiting a promise, which deadlocks in a browser. Surfaces stay on the device across draws and go back to guest memory only at `Renderer::flush` (before present). This also turned 88 round trips per frame into one.
- Device features requested: compressed texture families masked to the adapter; `TEXTURE_FORMAT_16BIT_NORM` (native only, else widen to `r32float`, which needs `FLOAT32_FILTERABLE`); `SUBGROUP` for warp shuffles (native only; browsers use the derivative-based `QUAD_SWAP`); `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` for 2/8 sample counts (without it the device allows only 1 and 4, and the adapter's answer lies, silently producing empty draws); `RG11B10UFLOAT_RENDERABLE` (Tomodachi Life's HDR target); `FLOAT32_BLENDABLE`. Storage buffer limit raised to the adapter's because shaders can read more than the 8 guaranteed banks.
- The `wgpu::Instance` and adapter must be held for the device's lifetime: a browser loses the device ("A valid external Instance reference no longer exists") once they are collected.
- Per-draw buffers/textures, readback staging buffers and flushed surfaces are `destroy()`ed rather than dropped: a browser only frees dropped resources on GC, which led to `VK_ERROR_OUT_OF_DEVICE_MEMORY` (Just Dance 2019, 55k draws) and OOM in Persona 5 Royal. `destroy()` is safe once the submission is made.
- Caches: shader modules keyed by WGSL hash (compiling was ~59 ms vs 2 ms translation); pipelines keyed by `PipelineKey` (Home Menu: 480 draws, 7 pipelines); samplers (Tomodachi Life spent 62% of time in `createSampler`); bind group layouts; translated shaders by address and stage (qlaunch translated 278k times for 24 programs), evicted by page writes and validated against `brx` jump tables read from constant buffers; deswizzled textures (textures are 96.5% of upload bytes), evicted on watched page writes, whole-cache eviction at 256 MiB. Texture source pages are walked only as far as the mapping (Asphalt 9 textures claim 16 MiB in a 6 MiB mapping).
- Device errors are captured by the uncaptured-error handler and read later (error scopes would require waiting). Device loss is otherwise silent (submissions accepted, readbacks never map); on loss, `give_up` hands every later frame to the rasterizer and the reason goes into the report rather than an error, since flush also runs inside GPU submissions (an error faulted Persona 5 Royal).
- Readback timing: natively flush waits (`PollType::Wait`, with timeout); in a browser maps complete from the event loop, so flush returns `Flush::Pending` and `Cpu::complete_pending_present` presents from a later slice. Presenting guest memory meanwhile came out black for double-buffered titles. Once a readback is observed to land late (`deferred_readbacks`), frames are not interleaved: a mid-frame fallback would read stale memory and be overwritten by the readback. `GPU_DEFER_READBACKS=1` reproduces this natively; `GPU_INTERLEAVE=1` trades ~0.09% wrong pixels (Home Menu frame 60) for much faster frames. A proper browser fix would yield the pushbuffer at the fallback so the next slice resumes with the readback landed.
- Software-frame latch: a frame the device cannot fully render goes wholly to the rasterizer. The latch releases after `clean_frames_needed` consecutive frames whose draws all pass `check`, doubling after each relatch (Tomodachi Life has one untranslatable frame in 740). Most fallbacks are `shader::wgsl` coverage gaps (e.g. `ldg b128`).
- Flush early-out when nothing is held/evicted/pending: flush runs before every fallback draw, and a no-draw browser trace charged 1,755 ms to empty polls.
- Multisampling: Maxwell stores samples as a texel tile per pixel with samples at texel centres. The default "expanded" route renders the expanded surface per texel, reproducing the rasterizer exactly; the device MSAA route (`GPU_DEVICE_MSAA`) shades per pixel but WebGPU's fixed sample positions differ, so it is off. Sample positions moved off texel centres fall back. `AntiAliasEnable = 0` uses a per-pixel companion.
- Held surfaces sampled as textures must be copied from the device, not read stale from guest memory (Tomodachi Life's HDR reduction chain and cube faces). `ZF32` held depth is copied via a buffer into `r32float` (Nintendo Switch Sports); padded surfaces sampled at their drawn size use the corner path when row layouts match (`same_rows`). Draws may sample their own target, so held layers are copied in a separate submission, never bound directly.
- Depth cannot be copied into a `depth32float` from a buffer, so depth uploads and shadow maps go through an `r32float` staging image drawn with `frag_depth` (`LOAD_DEPTH_WGSL`), one layer at a time.
- Attachments must match in size: the pass covers the colour/depth intersection; a larger depth surface is cropped, a larger colour target draws into a scratch texture copied back (WebGPU copies part of a colour texture but only whole depth textures).
- WebGPU details: no border address mode (wgpu web panics; the rasterizer also treats border as edge); strip pipelines must name the index format and non-strip pipelines must not, or the whole command buffer is rejected; Metal drops vertices whose stride overruns the buffer; no BGRA vertex format (swapped in the shader); no per-texture swizzle (passed into the layout); y is negated because WebGPU mirrors y itself; stencil is neither tested nor held on the device, so stencil clears go to guest memory.
- `report_json` nests timings to avoid duplicate "modules" keys (`JSON.parse` keeps only the last), and escapes fallback reasons.
- `shader::wgsl` dispatch keeps an unreachable trailing `return false;` because naga requires it (Tint warns); a test detects when naga stops requiring it.

### `crates/switch-gpu/src/readback.rs`

- Readback is copy to staging, map, read: three steps that cannot happen in one call on the web. The map callback runs from the event loop, so state is an atomic polled by the next slice. `Gpu::write_back` currently does both halves with a wait between.
- Depth row bytes are the device format's: `Z24S8` is 4 bytes in memory and 4 bytes of `f32` on the device; `ZF32_X24S8` is 8 and 4.
- Companion surfaces are gathered on creation and scattered back before read so guest memory only sees the expanded form; the grid is stored because the register file has moved on by flush time.

### `crates/switch-gpu/src/builtin.rs`

- Depth uploads go through a fullscreen draw writing frag depth from an `r32float` texture because a copy cannot write `depth32float`. `depth16unorm` could be copied but uses the same draw path so the two formats cannot diverge.
- The resample grid is a storage buffer, not a uniform: its tables are indexed by a runtime value and a uniform array pads every element to 16 bytes.

### `crates/switch-gpu/examples/screenshot_gpu.rs`

- Run `screenshot_title` and `screenshot_gpu` on the same frame and `cmp` the PPMs: byte-identical output is the evidence a GPU backend matches the software reference. The example includes `switch-core`'s `examples/common` by path to avoid drifting copies. Docking mid-run (`DOCK_AT`) mirrors a real dock: the running title is told through AM messages. The GPU backend reaches late faults much faster, so it suits `TRAP_WRITE`/`WATCH_PC` debugging.
