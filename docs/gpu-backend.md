# The `wgpu` backend

`switch-gpu` is the `wgpu` backend behind `gpu::renderer::Renderer`; the software rasterizer is the reference it must agree with.

**It never blocks in a browser.** Reading a render target back means awaiting a promise, and a blocking wait there is not slow but *deadlocked*. So a surface stays on the device across every draw targeting it and returns to guest memory only at `Renderer::flush`, which the engine calls before `present`. Opening the device is likewise asynchronous: the worker's loop calls `switch_gpu_open` as it starts and the first slices draw on the rasterizer until it resolves. **Any draw the backend cannot express falls back to `Software`.** `Renderer::flush` polls with `PollType::Wait`, a real wait natively and a *no-op* on WebGPU, so a browser gets `Flush::Pending` and the present waits for a later slice (`Cpu::complete_pending_present`), since presenting guest memory before the readback landed came out black for double-buffered titles.

**Where a readback lands late, a frame is all one renderer's.** The first flush answering `Pending` sets `Gpu::deferred_readbacks`; only then does a frame in which anything fell back make the frames after it wholly the rasterizer's (`Gpu::software_frame`), since a mid-frame fallback would read stale memory that the late readback then overwrites. **It latches on purpose**, because alternating costs a frame with the rasterizer's draws under a readback each time, **but it lets go**: each draw of a rasterizer's frame goes through `Gpu::check` (pipeline state, surfaces, route, shader translation; no uploads), and after `clean_frames_needed` frames in a row in which every draw passed, the device has the frames again. One clean frame releases it the first time; each relatch after a release doubles the wait, since the check cannot see a fallback that an upload or the device itself causes. Tomodachi Life is why: one draw in its first 740 frames samples a per-pixel bindless handle, and a latch that never let go gave the rest of the session to the rasterizer. What buys the acceleration back is still `shader::wgsl`: almost every fallback that latches it is an opcode with no WGSL form, not anything WebGPU withholds. **A draw the rasterizer only vertex-shades for its stores runs there without counting as a fallback** (`raster::only_shades_vertices`: points or lines whose vertex shader stores to global memory and samples nothing), since it reads no render target and nothing landing late can overwrite it. Nintendo Switch Sports resets records through pointers with one every frame, and it relatched the frames on its own, the wait doubling to thousands of frames.

**Each GPU address space has its own surfaces and caches.** Processes map different memory at the same GPU addresses, so held surfaces, readbacks and the texture and shader caches live in a `Space` per address space; every renderer entry enters the `ExecCtx`'s (`Gpu::enter_space`) and parks the rest, a flush writes back only its own space's surfaces, and a closed address space's device textures are destroyed (`Renderer::address_space_closed`). The texture cache's byte limit is shared, and reaching it clears every space.

**Dropping the backend destroys what it made.** wgpu's web backend frees nothing on drop and leaves it to the garbage collector, so `Gpu`'s `Drop` destroys every space's textures, the scratch resources and the device, which happens when a session is freed or a lost device is replaced.

**A copy out of a held surface flushes first.** The 2D blitter, the copy engine and compute read guest memory, so `channel.rs` hands the surfaces back before `Engine2D::LAUNCHES_BLIT`, `copy::LAUNCH_DMA` and `compute::SEND_SIGNALING_PCAS_B`.

**Depth, clears and multisampling all run on the device.** A depth surface is held like a colour one, as `depth16unorm` or `depth32float` (see `upload.rs` in [gpu.md](gpu.md)); stencil stays in guest memory and is put back around the depth bytes, since neither renderer tests it. Clears are a pass's load operation where they cover the whole surface and a scissored fullscreen draw where they do not.

**Multisampling has two routes, and the default is the exact one.** Maxwell stores samples *spatially*: a pixel owns a `samples_x` by `samples_y` tile of texels, so guest memory holds the expanded image and the default route renders exactly that, one fragment per texel, testing coverage at texel centres where Maxwell's samples are; the sample mask and alpha-to-coverage become the fragment shader's job (`wgsl::Coverage`). `AntiAliasEnable = 0` renders through a per-pixel companion whose result fills every texel of the tile. Samples programmed away from their texel centres (`SampleGrid::samples_at_texel_centres`) draw once per sample: each pass moves the viewport by the sample's offset inside its texel (`SampleGrid::texel_shift`) and its pipeline's `pass_sample` constant keeps only that sample's texels, so coverage and depth are the rasterizer's; varyings interpolate at the moved texel centre. Every sample at the pixel centre takes the per-pixel route instead. One case still falls back deliberately: per-pixel coverage with a *partial* sample mask or alpha-to-coverage. `GPU_DEVICE_MSAA=1` lets the device multisample instead; it is off because WebGPU's sample positions are a rotated grid that is not Maxwell's, and core WebGPU only guarantees four samples, so `2x1`, `4x2` and `4x4` take the expanded route regardless.

**Two renderers disagree by a 255th where a channel lands on a half**: `ColorFormat::encode` rounds `127.5` up and a device's unorm conversion rounds it down, so a test wanting byte-identity picks values off the eight-bit half-way points.

**Checking the backend** is running `screenshot_title` and `screenshot_gpu` over the same frame and `cmp`ing the PPMs; a byte-identical pair is the only evidence it renders what the rasterizer does, and `GPU_ONLY=<i>` (or `<a>..<b>`) narrows a difference to one draw. `gpu::testing::Harness` is the faster half: a drawable `Engine3D` both renderers are driven over, so a route can be checked without booting a title. **hbmenu is not a shader-core test**: its command list is `dkCmdBufCopyBufferToImage` plus a fence.

## Notes by file

### `crates/switch-gpu/src/convert.rs`

- Formats behind optional WebGPU features must be refused up front: `createTexture` throws on a missing feature and wgpu's web backend unwraps it, which on wasm is an `unreachable` trap that kills the core (Just Dance 2019's first BC1 texture). Refusal routes the draw to the software rasterizer.
- Normalized 16-bit formats (`TEXTURE_FORMAT_16BIT_NORM`) are native-only; no browser offers them. Sampled textures in those formats are widened to the 32-bit float sibling, which is exact (`f32` has 24 significand bits) and needs `float32-filterable`. Render targets in those formats stay refused: float targets neither clamp nor blend like normalized ones, and readback copies texels straight into a 16-bit guest surface.
- Attachment support is checked through the format's allowed usages, not required features: e.g. `rg11b10ufloat` samples without a feature but rendering needs `RG11B10UFLOAT_RENDERABLE`.

### `crates/switch-gpu/src/stats.rs`

- All device rejections are kept (distinct messages capped at 16, plus a total count), not just the first: the only production reader runs before pipeline creation, so later errors went unread.
- Upload accounting is in bytes rather than time (same on host and V8); it showed textures dominate upload bytes, which is why textures are cached. Cached textures are not counted.
- `flush` splits into ask/wait/land. In a browser `flush_wait` is ~0 because the wait moved to the slice boundary (`Flush::Pending`).
- Fallback reasons and device errors are JSON-escaped because the page parses the report with `JSON.parse`.

### `crates/switch-gpu/src/lib.rs` and `device.rs`

- `wgpu` lives in its own crate to keep `switch-core` dependency-free; the `Renderer` trait is the only shared surface.
- Device features requested (masked to the adapter's): the compressed texture families; `TEXTURE_FORMAT_16BIT_NORM` and `FLOAT32_FILTERABLE` (see `convert.rs`); `SUBGROUP` for warp shuffles (native only; browsers use the derivative-based `QUAD_SWAP`); `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` for 2/8 sample counts (without it the device allows only 1 and 4, and the adapter's answer lies, silently producing empty draws); `RG11B10UFLOAT_RENDERABLE` (Tomodachi Life's HDR target); `FLOAT32_BLENDABLE`. The storage buffer limit is raised to the adapter's because shaders can read more than the 8 guaranteed banks. `GPU_WEB_LIMITS=1` hides the native-only features so a native run takes the browser's routes.
- The `wgpu::Instance` and adapter must be held for the device's lifetime: a browser loses the device ("A valid external Instance reference no longer exists") once they are collected.
- Per-draw buffers/textures, readback staging buffers and flushed surfaces are `destroy()`ed rather than dropped: a browser only frees dropped resources on GC, which led to `VK_ERROR_OUT_OF_DEVICE_MEMORY` (Just Dance 2019) and OOM in Persona 5 Royal. `destroy()` is safe once the submission is made.
- Caches: shader modules keyed by WGSL hash (compiling costs far more than translating); pipelines keyed by `PipelineKey`; samplers (`createSampler` is slow in browsers); bind group layouts; translated shaders by address and stage, evicted by page writes and validated against `brx` jump tables read from constant buffers; deswizzled textures, evicted on watched page writes, with whole-cache eviction at 256 MiB (`TEXTURE_CACHE_BYTES`). Texture source pages are walked only as far as the mapping (Asphalt 9 textures claim 16 MiB in a 6 MiB mapping).
- Device errors are captured by the uncaptured-error handler and read later (error scopes would require waiting). Device loss is otherwise silent (submissions accepted, readbacks never map); on loss, `give_up` hands every later frame to the rasterizer and the reason goes into the report rather than an error, since flush also runs inside GPU submissions (an error faulted Persona 5 Royal). `lost()` then answers true and the worker opens a new device.
- `GPU_DEFER_READBACKS=1` reproduces the browser's late readbacks natively. `GPU_INTERLEAVE=1` interleaves fallback draws into device frames anyway, trading a few wrong pixels for much faster frames. A proper browser fix would yield the pushbuffer at the fallback so the next slice resumes with the readback landed.
- Flush returns early when nothing is held, evicted or pending: flush runs before every fallback draw, and empty polls showed up as seconds in a no-draw browser trace.
- Held surfaces sampled as textures must be copied from the device, not read stale from guest memory (Tomodachi Life's HDR reduction chain and cube faces). `ZF32` held depth is copied via a buffer into `r32float` (Nintendo Switch Sports); padded surfaces sampled at their drawn size use the corner path when row layouts match (`same_rows`). Draws may sample their own target, so held layers are copied into a fresh texture, never bound directly.
- Attachments must match in size: the pass covers the colour/depth intersection; a larger depth surface is cropped, a larger colour target draws into a scratch texture copied back (WebGPU copies part of a colour texture but only whole depth textures).
- WebGPU details: no border address mode (wgpu web panics; the rasterizer also treats border as edge); strip pipelines must name the index format and non-strip pipelines must not, or the whole command buffer is rejected; Metal drops vertices whose stride overruns the buffer; no BGRA vertex format (swapped in the shader); no per-texture swizzle (passed into the layout); y is negated because WebGPU mirrors y itself; stencil is neither tested nor held on the device, so stencil clears go to guest memory.
- `report_json` nests timings to avoid duplicate "modules" keys (`JSON.parse` keeps only the last).

### `crates/switch-gpu/src/readback.rs`

- Readback is copy to staging, map, read: three steps that cannot happen in one call on the web. The map callback runs from the event loop, so state is an atomic polled by the next slice. Each ask for a frame's surfaces is a generation, at most two in flight, landed in order; a present waits on its generation.
- Depth row bytes are the device format's: `Z24S8` is 4 bytes in memory and 4 bytes of `f32` on the device; `ZF32_X24S8` is 8 and 4.
- A companion surface is resolved back into the held surface before the readback, so guest memory only sees the expanded form; its grid is stored because the register file has moved on by flush time.

### `crates/switch-gpu/src/builtin.rs`

- Depth uploads and shadow maps go through a fullscreen draw writing `frag_depth` from an `r32float` texture (`LOAD_DEPTH_WGSL`, one layer at a time) because a copy cannot write `depth32float`. `depth16unorm` could be copied but uses the same draw path so the two formats cannot diverge.
- The resample grid is a storage buffer, not a uniform: its tables are indexed by a runtime value and a uniform array pads every element to 16 bytes.

### `crates/switch-gpu/src/tests.rs`

- `shader::wgsl` dispatch keeps an unreachable trailing `return false;` because naga requires it (Tint warns); a test fails once naga stops requiring it.

### `crates/switch-gpu/examples/screenshot_gpu.rs`

- Includes `switch-core`'s `examples/common` by path to avoid drifting copies. `DOCKED=1` docks before boot; `DOCK_AT=<frame>` docks mid-run like a real dock, telling the running title through AM messages. The GPU backend reaches late faults much faster, so it suits `TRAP_WRITE`/`WATCH_PC` debugging.
