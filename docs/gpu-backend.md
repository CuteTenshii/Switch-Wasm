# The `wgpu` backend

The rules AGENTS.md summarises, in full. `switch-gpu` is the `wgpu` backend
behind `gpu::renderer::Renderer`; the software rasterizer is the reference it
must agree with.

**It never blocks in a browser.** Reading a render target back means awaiting a
promise, and a blocking wait there is not slow but *deadlocked*. So a surface
stays on the device across every draw targeting it and returns to guest memory
only at `Renderer::flush`, which the engine calls before `present`; opening the
device is likewise deferred — `worker/index.ts` calls `switch_gpu_open`
*between* run slices, since the channel does not exist until the title has run.
**Any draw the backend cannot express falls back to `Software`.**
`Renderer::flush` polls with `PollType::Wait`, a real wait natively and a
*no-op* on WebGPU, so a browser gets `Flush::Pending` and the present waits for
a later slice (`Cpu::complete_pending_present`).

**Where a readback lands late, a frame is all one renderer's.** The first flush
answering `Pending` sets `Gpu::deferred_readbacks`, and from then on a frame in
which anything fell back makes every frame after it the rasterizer's whole
(`Gpu::software_frame`). **It latches on purpose** — alternating is the one
behaviour this must not have. What buys the acceleration back is
`shader::wgsl`: every fallback that latches it is an opcode with no WGSL form,
not anything WebGPU withholds.

**A copy out of a held surface flushes first.** The 2D blitter and the copy
engine read guest memory, so `channel.rs` hands the surfaces back before
`Engine2D::LAUNCHES_BLIT` and `copy::LAUNCH_DMA` — the same guard compute had.

**Depth, clears and multisampling all run on the device.** A depth surface is
held like a colour one, converted to `depth16unorm` or `depth32float` — the two
formats a copy can read — with the stencil byte read out of guest memory and
put back, since neither renderer tests stencil. Nothing copies *into*
`depth32float`, so a surface gets there by being drawn: a fullscreen triangle
writing `@builtin(frag_depth)`. Clears are a pass's load operation where they
cover the whole surface and a scissored fullscreen draw where they do not.

**Multisampling has two routes, and the default is the exact one.** Maxwell
stores samples *spatially* — a pixel owns a `samples_x` by `samples_y` tile of
texels — so guest memory holds the expanded image and the default route renders
exactly that, one fragment per texel, testing coverage at texel centres where
Maxwell's samples are; the sample mask and alpha-to-coverage become the
fragment shader's job (`wgsl::Coverage`). Two draws still fall back
deliberately: `MultisampleSampleLocations` away from texel centres
(`SampleGrid::samples_at_texel_centres`), and per-pixel coverage with a
*partial* sample mask. `GPU_DEVICE_MSAA=1` lets the device multisample instead
— off, because WebGPU's sample positions are a rotated grid that is not
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
a shader-core test** — its command list is `dkCmdBufCopyBufferToImage` plus a
fence.

