# GPU core

The Maxwell engines, shader translation, the software rasterizer and texture handling.

## `crates/switch-core/src/gpu/engine/twod.rs`

- deko3d routes `dkCmdBufBlitImage` to the 2D engine whenever the copy engine cannot express it (scaling, format conversion, filtering).
- Fast paths: bilinear with taps exactly on texel centres degenerates to point sampling (Just Dance 2019 resolves 2x2 MSAA with `du_dx=2, dv_dy=2, src0=(0.5,0.5)`); point sampling between byte-exact formats is a move, no decode/encode. Surfaces lying in one mapping are translated once instead of per texel (per-texel lookup was 40% of the copy). Column swizzle halves are precomputed once per blit, row halves once per row.
- Tiled walk: row-order walking of a block-linear source jumps a GOB column every 8 texels; tiles read a few contiguous blocks. Overlapping surfaces keep row order since only then the order is observable.
- `blit_staged`: byte-exact copies between disjoint surfaces are done in host memory (gather source texels, read target whole, scatter, write back including padding). Through guest memory the 2560x1440 Just Dance 2019 resolve was a chain of cache misses (13 ms/frame). Gathered texels are kept with the source pages watched, so an unchanged source is skipped entirely. Only used where it is exactly the per-texel copy: offsets in bounds, no watchpoint over either surface, no protected byte in the target.
- `scatter` relies on `blit_staged` having proven all indices in range, which removes bounds checks from the hot loop.

## `crates/switch-core/src/gpu/macro_engine.rs`

- A taken branch overrides an exit in its delay slot, so an exit combined with a branch fires only when the branch is not taken (deko3d's `FillRegisters` relies on this).
- Macro host reads and writes happen in program order, so a read sees the side effects of earlier emitted writes (e.g. `WriteHardwareReg` firmware-call completion polling).

## `crates/switch-core/src/gpu/mod.rs`

- The renderer backend lives on `Gpu`, not on a `Channel`: every `open("/dev/nvhost-gpu")` creates a channel (Asphalt 9 opens four) and `nvClose` destroys one, so installing on a channel lost the device. It is lent to the executing channel for each submission and taken back even on a fault. Flushes run once (one backend) through the last submitting channel's address space.
- Scan-out honors the queued `transform` (Minecraft queues every frame `FLIP_V`) and `crop` (A Short Hike renders 1280x720 into a 1920x1080 buffer). `ROT_90` is refused rather than shown wrong. Empty or off-surface crops mean the whole surface; crops are clamped since they are guest data.
- Scan-out performance: read the whole surface with one page-table walk (`read_into`) into a buffer reused across frames, compute row/column swizzle halves once, and clamp the slice so per-pixel bounds checks are elided. Per-pixel `read_le` was the full cost of a frame in titles that draw nothing.

## `crates/switch-core/src/gpu/surface.rs`

- A block-linear address is a sum of a row term and a column term (`block_linear_row` + `block_linear_column`), so walks hoist the row (which carries the only divisions) out of the inner loop. Within a GOB only `x % 16` is linear, so walks proceed in 16-byte runs (`Layout::run_at`).
- 3D block-linear surfaces interleave slices inside a block; slice `z` is not `z` layer-strides along. 2D arrays store layers back to back with `layer_stride`.
- MSAA is spatial: a 4x 1280x720 target is 2560x1440 texels; render/depth target registers describe the expanded surface while scissor/viewport/clear stay in pixels. A sample's slot is the texel its programmed location falls in (reproduces deko3d's `locationsMS4`/`MS8`); collisions fall back to raster order. Backends can only reproduce grids whose samples sit at texel centres (`samples_at_texel_centres`), compared by position so a programmed table naming the centres still qualifies. `AntiAliasEnable = 0` moves all sample positions to the pixel centre but keeps the slots.
- sRGB encode/decode lives in `ColorFormat` so render targets, the blitter and the sampler agree (previously surfaces came back darkened, and scan-out brightened sRGB frames). Colours are always linear at the API.
- The blend unit clamps only for fixed-point targets; `B10G11R11_FLOAT` must not clamp (Persona 5 Royal's bloom).
- `f32_to_f16` rounds to nearest even and produces subnormals; truncation biased fp16 shader math (`hadd2`/`hmul2`/`hfma2`) toward zero. Subnormal decode was reached via BC6H.
- `host_shuffle`/`is_byte_exact` let scan-out and copies move bytes for 8-bit non-sRGB formats instead of a decode/encode round trip; asked once per surface. `UNORM8` table replaces `v / 255.0` divisions in the blitter.
- `encode_registers`/`decode_registers` (`sust.p`/`suld.p`) keep integer channels as integers; going through `f32` would round values above 2^24.
- `A8B8G8R8_SNORM` was misread as UNORM; SNORM's most negative value clamps to -1.0.

## `crates/switch-core/examples/bcn_difftest.rs`

- Fixture comes from a throwaway C harness around `bcdec.h` (public domain): per codec, 4000 random 16-byte blocks, each written as the input then the decoded texels, in order BC1, BC2, BC3, BC7 (RGBA), BC4 (R), BC5 (RG).
- BC7 must match exactly. BC1-3 may differ by 1/255: this decoder expands 5:6:5 endpoints by bit replication (as BPTC does), `bcdec` by rounding; both are within S3TC tolerance.
- ASTC fixture is grouped by footprint (width, height, count as LE i32, then records of 16 input bytes and RGBA8 per texel). The reference converts from float as `clamp(v * 65536 + 0.5, 0, 65535) >> 8`, so the test applies the same.

## `crates/switch-core/src/gpu/shader/cfg.rs`

- Maxwell uses a reconvergence stack (`ssy`/`sync`, `pbk`/`brk`, `pcnt`/`cont`) rather than structured control flow. If every pop statically pairs with one push, each pair maps to an `if` or loop and WGSL translation is mechanical; if an instruction is reachable with two different stacks, it needs per-lane masking instead. `Cfg::pairing` answers this question; it does not build structured output yet.

## `crates/switch-core/src/gpu/bcn/bc7.rs`

- Every BC7 mode uses exactly 128 bits; summing the mode table is a test that catches a mistyped field width in any row.

## `crates/switch-core/src/gpu/shader/mod.rs`

- Programs are 32-byte blocks: one `sched` control word then three instructions. Binaries carry no length, so decoding walks the CFG from the entry and stops each path at its terminator (stopping at the first `exit` misses code reached by branches). Branch targets landing on a `sched` word mean the block's first instruction.
- Mesa-compiled binaries carry a 0x50-byte header whose first bytes are not reliably zero, so decode is tried without it and retried past it if slot 1 does not decode. deko3d/uam binaries have none. This lives in one place because the rasterizer, wgpu backend and compute all need the same answer.
- The SPH's `omap.target` is the only source of fragment output registers: registers are assigned in target order, targets with no writes are skipped, but disabled components inside a written target still take a register.
- `brx` targets come from a constant-bank jump table; the decoder walks the selector's use-def chain back to the `imnmx` clamp (which the scheduler may hoist far away, 36 slots in Home Menu shaders) and gives up on any unexpected or predicated write rather than guessing a table length. Jump targets are base + entry, not rounded. The resolved targets are kept for later CFG analysis.
- `DecodeReads` records the GPU-virtual pages and constant words a decode read so backends can cache translations; only constant words are re-read for validation, program pages are watched.
- The rasterizer only interpolates varyings the fragment shader reads (cost is per covered pixel).

## `crates/switch-core/examples/shader_coverage.rs`

- WGSL translation is per program and stops at the first blocker, so the tool reports the first blocker per program; fix and re-run.
- Reports twice: no optional features (browser) and with quad ops. Fragment shaders use `wgsl::QUAD_SWAP` without quad ops, so a gap between passes means a vertex program shuffles.
- Programs must be built into a full module: texture dimensionality is rejected at binding layout, not at `translate`.
- Same address can decode differently under different constant bank bindings; first decode is kept.
- qlaunch is the preferred subject (frame in ~35M instructions, almost pure rasterizer).

## `crates/switch-core/src/gpu/channel.rs`

- A channel's pushbuffers are one continuous stream: a method group may run past the end of one pushbuffer and finish in the next (a 64-word inline upload split across two GPFIFO entries used to fault).
- nvhost pre-binds `MAXWELL_CHANNEL_GPFIFO_A` to subchannel 6; deko3d writes syncpoint increments there without `SetObject` (hbmenu's fence never signalled otherwise).
- Methods below 0x40 are host methods on every subchannel.
- 2D blits and compute dispatches flush the wgpu renderer first, because it keeps render targets on the device until flushed (Just Dance 2019 resolves its MSAA target with a 2D blit every frame).
- Semaphore acquires are treated as satisfied because a submission runs to completion before the ioctl returns.

## `crates/switch-core/src/gpu/compute.rs`

- The software dispatch is scalar: CTA threads run one after another. Exact except for barriers and warp instructions (`shfl`, `vote`), where threads suspend until all (CTA or warp) arrive. Atomics need no locking and races cannot be observed (a racy kernel gets a valid but different answer than hardware).
- Programs without barriers or warp instructions reuse one invocation serially (cheaper); otherwise every thread keeps a live invocation.
- `MAX_DISPATCH_THREADS` (1<<20) is a liveness guard: the GPU stack runs on one browser worker thread.

## `crates/switch-core/src/gpu/nvdrv.rs`

- `nvIoctl3`'s inline output matters: ioctls with a `{ buf_size, buf_addr }` pair return payloads through it. libnx reads from `data`, `nnSdk` from the inline output; leaving it empty gave a retail title zeroed GPU characteristics and a null device. `GetCharacteristics` and `GetTpcMasks` write both.
- Failed ioctls are always traced (`eprintln!`); missing handlers are reported via the diagnostic channel by `Cpu::nvdrv_request`.
- Submissions retire inside their ioctl, so syncpoint waits only see completed work, `nvQueryEvent` events are always signalled, `EventWaitAsync` signals on arrival, and a wait that is not satisfied reports the timeout rather than hanging.
- `GetConfig` (`nv!` debug overrides) is refused with `ConfigVarNotFound`: `NvOsGetConfigString` treats any success as "set", and an empty success enabled `NVWSI_FILL` (per-pixel buffer fills, 45% of a Just Dance 2017 frame). Production consoles have no such settings.
- ZBC tables are kept only so `ZbcQueryTable` reads back what `ZbcSetTable` stored (otherwise drivers re-register until full). Same value takes another reference, no eviction.
- `VsmsMapping` (0x13, `0xc0084713`): not in libnx; identified from `nnSdk`'s `nvrm_gpu`; matches upstream `nvgpu_gpu_vsms_mapping_args`, entries are u16 `{gpc, tpc}` per TPC. Counts must agree with `GetCharacteristics` (TPC mask and VSM map derive from the same constants).
- `MapBufferEx` with `MODIFY` remap: names an existing mapping with handle 0; deko3d uses it to give block-linear kinds to sub-ranges. `Remap` (0x14) is a batch at explicit GPU VAs in big pages, used to back sparse reservations.
- `SetTimeslice` and similar channel configuration must succeed: nnSdk's nvn checks it.
- Nvmap `Alloc` on an already-mapped handle must remap the GPU address space (otherwise buffers read as zeroes).

## `crates/switch-core/src/gpu/shader/isa.rs`

- Perspective correction: the fragment shader reads interpolated `1/w` at `a[0x7c]` via `ipa pass`, computes `w = mufu rcp`, and multiplies other varyings by it in perspective `ipa`.
- Guard predicate is bits [16,19) plus negate at 19; register 7 is `PT`.
- Maxwell has no 32-bit integer multiply; multiplies are `xmad` chains (16x16 + accumulate).
- `brx`: base plus table entry is the target, so the base must not be aligned past `sched` words.

## `crates/switch-core/src/gpu/activity.rs`

- Activity is keyed by numbers, not strings: thousands of draws per frame, so labels are formatted only when a surface is first seen. Tallies cap at 256 entries (overflow summed into one) and refusal reasons at 32 (reasons may embed addresses).
- Refusals are tracked by reason, separately from per-surface `failed` counts, since the reason is what says what to implement next.

## `crates/switch-core/src/gpu/engine/compute.rs`

- A compute launch is the QMD address (>> 8) written to `SendPcasA` then `SendSignalingPcasB`; everything about the grid is in the QMD. The program region and texture pools sit at the same methods as the 3D class, so `gpu::texture` serves both.
- The channel flushes the 3D backend before `SEND_SIGNALING_PCAS_B`, since a dispatch reads/writes guest memory a GPU-resident render target may hold.
- Compute class inline upload (0x60..0x6D) is needed: Zelda: Echoes of Wisdom uploads every QMD this way.
- Refused dispatches are counted, not propagated, matching refused draws.

## `crates/switch-core/src/gpu/vmm.rs`

- There is no VRAM: guests allocate CPU memory, nvmap wraps it, and `/dev/nvhost-as-gpu` maps whole nvmap ranges at contiguous GPU VAs, so a sorted list of ranges translates exactly with no per-page cost.
- `MAP_BUFFER_EX` with `NVGPU_AS_MAP_BUFFER_FLAGS_MODIFY` (bit 8) re-maps a sub-range of an existing mapping with a new kind (how drivers give block-linear images their swizzle within one buffer). `offset` names the existing mapping; handle is 0. Treating it as a normal map failed with `BadParameter`. The covering mapping is split (the map is keyed by start VA), and backing addresses do not move.
- `translate` assumes non-overlapping ranges; `unmap_range` (for `REMAP`) trims partial overlaps for that reason.
- Translation cache: 8 ways, round-robin. A single entry thrashed (constant buffers, textures and render target per pixel) and left the BTreeMap search at 5% of a Home Menu frame. Fields are split into separate `Cell` arrays because a `Cell<Option<(u64,u32,u64)>>` per way copied 32 bytes on every rejected way. Any mapping change flushes the cache.

## `crates/switch-core/src/gpu/upload.rs`

- The software rasterizer reads attributes lazily through the MMU; a GPU backend must decide up front what to upload. Vertex array limits often point at the end of a heap, so uploads are bounded by the draw (`first`/`count`, or min/max index for indexed draws; reading indices is free since they are uploaded anyway). `MAX_UPLOAD` (64 MiB) makes absurd stride*count report instead of allocating.
- Array `limit` is the address of the last valid byte (inclusive). Fetches past it read zero, as hardware and `raster::fetch_attribute` do; Tomodachi Life reads 16 bytes past an array end, and refusing that latched every later frame onto the rasterizer.
- Vertex spans charge the last element only its attribute length (WebGPU sizing), not a whole stride.
- Constant banks: resolving only the banks shaders read (`Banks::Read`) matters: the Home Menu binds eight 64 KiB banks per draw but reads two (190 KiB vs 60 KiB per draw).
- `TextureKey` comes entirely from the TIC, so descriptor rewrites produce new keys; only texel writes need watched pages. Swizzle and sampler are applied at sample time and are not part of the key. Textures were 96.5% of uploaded bytes; `bytes` is an `Arc<[u8]>` so cache hits cost a refcount (average draw reads 1.76 MiB). WebGPU has no per-texture component swizzle, so backends apply it in the sampling hook.
- BC textures upload compressed (WebGPU supports BC). ASTC is decoded to `Rgba8Unorm` since desktop browsers lack `texture-compression-astc` and the Home Menu's textures are ASTC 4x4. Partial-block compressed images (Home Menu 1x1 BC4/BC5 default textures) are decoded because WebGPU requires whole-block extents and rounding up would change sampled coordinates. Decoded images keep their sRGB encoding; the format says so and the device applies the transfer function. Depth textures cannot be sampled on device (WebGPU fills `depth32float` only by texture copy), so those draws go to the rasterizer.
- `decode_blocks` reuses strip/block buffers: zeroing a fresh 12x12 block per 4x4 block was 24% of an ASTC title's upload time.
- `deswizzle` units are texels for plain formats and whole blocks for compressed; reading compressed surfaces in texels makes the row stride a block too wide (diagonal ribbons). Surfaces in one mapping are translated once (`ExecCtx::span`) rather than per texel (3.7M translations at 720p), walking contiguous runs via `run_at` (16 bytes within a GOB, whole rows for pitch).
- `Target::write` patches over existing bytes because block-linear padding is not the surface's to zero. `write_strided` takes padded readback rows directly to avoid a 3.7 MB/frame repack at 720p.
- Depth: WebGPU copies can read but not write `depth32float`, read and write `depth16unorm`, and touch `depth24plus` in neither direction. So Z16 stays 16-bit and everything else becomes f32 (24-bit depth is lossless in an f32 mantissa; a lossy round trip would break every `Equal` depth test next frame). Depth writeback preserves packed stencil bytes, matching `raster`'s `Fragments::write`. Just Dance 2017 renders all passes depth-only (no colour target).
- `texs` immediates index (in dwords) the constant bank `TexCbIndex` names (15 under nouveau, 0 under deko3d), which holds the bindless handle into the TIC/TSC pools. The two shader stages index different constant buffers with the same slot.
- `read_range` reads a word at a time where possible: per-byte reads paid translation and page lookup 8x.

## `crates/switch-core/src/gpu/shader/wgsl.rs`

- Control flow: Maxwell has a reconvergence stack (`ssy`/`sync`), backward `bra` loops and `brx` multi-way jumps. Even though `cfg` can prove push/pop pairing for every Home Menu shader, the translation emits a `switch` over `pc` inside a `loop` with an explicit stack (the `interp::Invocation` machine in WGSL): it cannot mistranslate control flow. It is slower on GPU; recovering structure where `cfg` proves it safe is a future change local to this module, with the state machine as fallback.
- Registers are untyped `u32` with `bitcast<f32>`; typing them would be wrong since shaders mix integer and float use of one register. Registers, helpers, bindings are recorded while emitting (no second pass that could disagree). `Layout::of` derives from the `Translation` for the same reason.
- Registers are `var<private>` because a fragment shader's colour is r0..r3 after the invocation; there is no output attribute.
- Varyings are `@interpolate(linear)`: Maxwell's `ipa` receives value/w linearly interpolated and multiplies by `rcp(a[0x7c])` itself. Perspective interpolation would divide twice (looks like a texcoord bug). Vertex stage multiplies by 1/w; fragment `position.w` is 1/w, matching `a[0x7c]`.
- `flip_y` must be set when the guest viewport does *not* mirror y (offscreen), because WebGPU's NDC->framebuffer already mirrors. Using `Viewport::flip_y` directly flips twice; the Home Menu still came out 94.87% correct because UIs are mostly symmetric.
- Depth: Maxwell clips z like GL (-w..w), WebGPU like Vulkan (0..w); detect via viewport z scale 0.5.
- Shadow samples: WGSL compares only via `textureSampleCompare` on `texture_depth_*`, and WebGPU copies into `depth32float` only from another same-format texture (spec 26.1.2.2), so guest shadow maps cannot be uploaded; the fix would be a render pass writing `frag_depth`. Until then the rasterizer takes these draws.
- Subgroups: browsers require `enable subgroups;`, naga rejects it, so `Caps::subgroup_enable` is separate. wgpu's web backend cannot request or report `subgroups`, so `QUAD_SWAP` emulates quad swaps from `dpdxFine`/`dpdyFine`: splitting a register into two 16-bit halves makes the difference and re-addition exact in f32, with clamp/round guarding the spec's approximate derivatives. `derivative_uniformity` diagnostic is turned off because derivatives sit inside the dispatch loop. `fswzadd` only needs the lane index (from `position`), not device support.
- Non-finite float literals (`bitcast<f32>(2139095040u)`) are constant expressions WGSL rejects (Chrome refused a Tomodachi Life module; naga accepted); they are bound to a `let`.
- Half precision: `pack2x16float` rounds to nearest-even like `f32_to_f16`, so backends agree on finite results; overflow differs (WGSL indeterminate, interpreter gives infinity). Half `.ftz` threshold is the half subnormal range, much higher than f32's.
- `ldg`: a descriptor in a constant bank plus a register index, or a whole descriptor loaded by `ldc.64`/two `mov`s (Nintendo Switch Sports). A Short Hike uses the indexed form 12 times. Other addresses are unsupported rather than guessed.
- Bindless `tex.b` handles are traced to a constant word via the nearest unguarded write in-block or the program's sole write; guessing would draw a plausible wrong texture instead of falling back.
- Texture results are stored immediately, unlike the interpreter's scoreboard-like deferral to first use; they differ only when the destination is overwritten before any read (interpreter still lands the sample late, hardware does not).
- Descriptor swizzles matter: two thirds of Home Menu draws sample single-channel images as `[R, R, R, One]`. WebGPU has no per-texture swizzle, so it is applied in the sampling hook.
- Each stage gets its own bind group because stages index different constant buffers/textures with the same numbers. Bank b binds at b, texture i at 32 + 2i with sampler beside, `ldg` buffers from 96.
- Centroid: emitted even though single-sample passes treat it like center, because with `GPU_DEVICE_MSAA` passes are really multisampled. Both stages must agree; the backend copies the fragment stage's set to the vertex stage.
- Depth-only passes (Just Dance 2017) return nothing from the fragment entry (an unbacked `@location(0)` fails pipeline creation) but still run for `kil` and alpha-to-coverage.
- Expanded multisample (per-texel) rendering must apply sample mask and alpha-to-coverage in the shader. Alpha-to-coverage keeps a prefix of `floor(alpha*count + 0.5)` samples (Rust `round` semantics, not WGSL's ties-to-even).
- Integer vertex attributes are declared `vec4<i32>/<u32>` (WebGPU matches base types) and bitcast into `a[]`; BGRA attributes swap components 0 and 2 in the entry point (no BGRA vertex format).

## `crates/switch-core/src/gpu/renderer.rs`

- The `Renderer` trait exists so a GPU backend can sit beside the software rasterizer, which stays as the bit-exact reference to check it against (about 85% of a Home Menu frame is rasterization, so only a GPU backend makes it fast).
- A backend that keeps surfaces GPU-side owns when to write render targets back to guest memory; `present` deswizzles straight out of guest memory.
- Clears are on the trait too: on a GPU they are a render pass load op, and draws on a GPU surface with clears in guest memory would disagree.
- `lost()`: a browser can drop the device (driver reset, GPU process restart, memory pressure). Falling back to the rasterizer for the rest of the session was measured 30x slower, while a lost device can just be requested again; it is cheap enough to poll once a slice.
- `Flush::Pending` must not be ignored: landing a readback one flush late produced black frames, since a double-buffered title presents the surface whose readback was just requested.

## `crates/switch-core/src/gpu/raster.rs`

- The CCW rewind before the top-left rule matters because the tie-break only splits an on-edge point between two triangles that walk the shared edge in opposite directions. SDL emits quads as one CCW and one CW triangle; JKSV's 128x128 save tiles have a 45-degree diagonal through pixel centres, and every tile showed a one-pixel gap.
- Barycentrics are screen-space linear; the shader's `ipa`/`mufu rcp` sequence does perspective correction, so they feed in as hardware's `attr/w` and `1/w`.
- `GENERIC_VARYINGS` covers all of Maxwell's `a[0x80]..a[0x280)`: it used to be four, but Home Menu panel shaders interpolate slots 4-6; the zero read became an `rcp` denominator, infinity, then `0 * inf` NaN encoded as black. `Program::interpolated_slots` keeps the cost down.
- Depth compare and blend factor/equation registers take both GL and D3D numberings (`Never_D3D = 1..=Always_D3D = 8` beside `Never_GL = 0x200..`). Mesa (JKSV) writes GL enums, deko3d/nvn D3D ones. Decoding one only made Just Dance 2019's `LessEqual` (4) fall into Always, and made the GPU backend fall back on every draw; the Home Menu's D3D `SrcAlpha`/`OneMinusSrcAlpha` fell to `One`/`One`, washing separators white.
- Fixed-point targets clamp the blend source (GL rule, and the ROP's), which is what kills NaN: the Album applet normalises by total alpha, a transparent texel gives `rcp(0)` then `0 * inf`, and every icon had a black box. NaN floors at 0, not SNORM's -1.
- Culling judges winding in window space after the viewport transform. Judging in NDC is the same through a y-mirroring viewport (every nnSdk main pass) but opposite otherwise; Echoes of Wisdom's offscreen post-processing (front=CCW, FlipY, back culling) lost every full-screen quad, leaving black frames.
- Near-plane clipping is required: `w <= 0` sends projected vertices to infinity or the wrong side, smearing triangles over the framebuffer.
- Depth-only passes: Just Dance 2017 binds its Z24S8 surface as colour target 0 and works in the depth buffer; requiring a colour target cost all 1870 draws. Extent comes from whichever target is bound.
- Attribute fetch history: "fixed" attributes read the default instead of erroring (JKSV leaves attribute 2 fixed; erroring dropped its full-screen background quad). Integer attributes carry bits (Just Dance 2019's signed-byte attribute dropped 6,480 of 6,844 draws). Minecraft's `4x16` halves (all 110 draws dropped). Echoes of Wisdom's `1x8`/`2x8` integer attributes (5,000+ draws a frame). 8-bit shapes read only their own bytes since a one-byte attribute may end a mapping. Past the array limit reads zeros, matching hardware and `upload`'s device padding.
- The vertex cache is keyed by guest index with `IdHasher`; re-running vertex shading per reference is the loop's most expensive possible cost. Constant buffers are read once per stage per draw.
- Quad walk (warp shuffles): Checkpoint's antialiased text differences coverage against the neighbour; per-pixel shading gave zero differences and solid blocks. Helper lanes outside the box are shaded but never depth-tested or written (they are outside the scissor too).
- `TRACE_WGSL` checks the flag before reading the env value (per-draw env scans were removed from hot paths). In a browser there is no directory, and taking the write branch produced a failed `std::fs::write` diagnostic per shader per draw.
- Alpha-to-coverage uses a fixed sample prefix rather than hardware's dither; average coverage is the same and a resolve averages the dither away.
- Others: "A Short Hike" masks alpha on a third of draws; Tomodachi Life composites through a y-mirroring viewport; Echoes of Wisdom's visibility boxes store from the vertex stage; the Home Menu draws UI as instances of a unit quad found by `gl_InstanceID`.

## `crates/switch-core/src/gpu/engine/copy.rs`

- `SetSrcWidth`/`SetDstWidth` count elements, not bytes; with the remap on an element is a pixel. Treating them as bytes worked for deko3d (remap off) but shredded JKSV's Mesa icon uploads into strips.
- Remap-off copies go a contiguous run at a time (pitch rows contiguous, block-linear in 16-byte pieces): per-byte swizzle and translation was ~1/5 of an hbmenu frame.

## `crates/switch-core/src/gpu/pipeline.rs`

- `Pipeline` resolves register state into WebGPU vocabulary so backends never re-decode raw codes. Anything WebGPU cannot express (fans handled by index rewriting excepted) is `Unsupported`, never approximated; the caller falls back to the software rasterizer. Unknown codes are errors here even where the rasterizer defaults (e.g. blend factor `One`).
- Blend factors come in GL numbering (Mesa) or D3D numbering (deko3d/nvn); depth compare in GL (0x200..) or D3D (1..=8). Both must be accepted (Just Dance 2019 fell back on every draw when 1..=8 was rejected).
- Viewport y mirror: WebGPU has no negative viewport height, so backends negate `position.y` where the guest does not mirror; facing is judged in window space and carries over unchanged.
- GL vs Vulkan clip-z convention is inferred from the depth transform shape (scale 0.5, translate 0.5 = GL); WebGPU clips 0..w.
- `Rg11b10Ufloat` needs `rg11b10ufloat-renderable`; `Rgba16Float` can't substitute (guest surface width). 16-bit norm formats are native-only. Integer colour formats are unnamed until the shader translator emits integer outputs.
- Colour format names must agree with `gpu::surface` on width, sRGB and number kind (SINT was once misnamed `Rgba8Unorm`).

## `crates/switch-core/src/gpu/shader/interp.rs`

- `ShaderResult` boxes `Error` (56 bytes) so a `Result` fits in a register pair; the unboxed form was ~8% of a Home Menu frame. Unbox only at `resume`'s boundary.
- Per-draw caches (`ConstCache`, TIC/TSC descriptors, decoded blocks) are owned by the caller so the per-fragment structs can be rebuilt while the pixel loop borrows `ctx` mutably. Constant buffers can't change mid-draw.
- `gpr` has 256 slots so `RZ` indexes without a bounds check; RZ is rewritten to 0 unconditionally (cheaper than a branch). Program ops and predicates are sliced to one length to share one bounds check (~7% of a frame otherwise).
- `texs` results are deferred: compiled code reads destination registers' old values between the fetch and first consumer, so each result lands just before its first reader, or at the next branch/exit. `reads()` must list every operand register of texture ops or queued results land late.
- `a[]` is a flat 256-word array with a written-mask (unwritten `clip.w` must default to 1.0); replaced a HashMap that dominated per-pixel cost.
- `centroid` needs no handling because shading is per covered sample at its centre.
- Half ops compute in f32 and round once at the merge; `.ftz` uses the half threshold only for half lanes.
- Warp shuffles to lanes absent from the warp (a quad is 4 of 32) read the caller's own value; votes count only lanes that reached them.
- `SR_Y_DIRECTION` must be +/-1.0 (zero deletes screen-space directions). Packed `SR_TID`/`SR_NTID` are refused as the layout is unconfirmed.
- `texs` handle immediate is a dword index (0x20 -> byte 0x80). `tex.aoffi` offsets are texels scaled by level size (Persona 5 Royal blur).

## `crates/switch-core/src/gpu/shader/compiled.rs`

- Lowering exists because fragment shaders run per covered pixel (921,600 times for a full-screen 720p quad): binary-searched branch targets, linear `texs` lookups and constant-cache reads are paid once instead.
- Ops are stored apart from predicates because an `Op` is 32 bytes vs 40 for `Instruction`: two per cache line instead of 1.6.
- Constant folding is sound because a constant buffer cannot change during a draw (one method). Unbound banks stay unfolded so the error still comes from the reading instruction. Undecoded branch targets error only when taken, since unreachable branches exist in shaders that run fine.

## `crates/switch-core/src/gpu/texture.rs`

- `texs` immediates are dword indices into the `TexCbIndex` bank (nouveau 15, deko3d 0). Reading them as bytes lands in nouveau's fixed header (`0, 1, 2, ...` on GM107), which looks like a handle table, so every draw resolved to the same handle and text drew one glyph repeatedly.
- Compressed surfaces are swizzled/addressed in blocks; using texel units makes the stride a block too large and shreds the image into diagonal ribbons.
- `BlockCache`: decoding a whole block per fetch was 7% of the Home Menu frame in ASTC alone. Four ways cover a bilinear footprint across a block corner; round-robin fits scanline order. Blocks decode in place because ASTC blocks are 2.3 KiB.
- Float/packed format support came from titles: "A Short Hike" composites from R16G16B16A16 FLOAT (refusing it left the frame transparent), Persona 5 Royal uses A2B10G10R10 and B10G11R11, Nintendo Switch Sports samples ZF32 as R32, Echoes of Wisdom reads S8D24 stencil as uint in a compute shader.
- Layer stride for mipmapped arrays/cubes: Tomodachi Life's 64x64 cubes with seven levels are 0x6000 apart; 0x4000 sampled every face from the first face's mips.
- Cubemaps: `DEPTH_MINUS_ONE` holds 1 but there are six faces. Software cube sampling clamps at face edges where the GPU filters across seams; nothing seen samples there.
- Shadow sampling is PCF (compare per tap, then filter), returning `[c, c, c, 1]` without swizzle. Integer textures are always nearest-sampled.
- `TRACE_TEX` is separate from `TRACE_GPU` because a frame has a million method traces but only dozens of textures.
