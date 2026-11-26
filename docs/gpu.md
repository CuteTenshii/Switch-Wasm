# GPU core

The Maxwell engines, shader translation, the software rasterizer and texture handling. The `wgpu` backend is in [gpu-backend.md](gpu-backend.md).

Section headings are paths under `crates/core/src/gpu/`; a heading that starts with `crates/`, `tools/` or `web/` is from the repository root.

## `nvdrv.rs`

- `nvIoctl3`'s inline output matters: ioctls with a `{ buf_size, buf_addr }` pair return payloads through it. libnx reads `data`, `nnSdk` the inline output; leaving it empty gave a retail title zeroed GPU characteristics and a null device. `GetCharacteristics` and `GetTpcMasks` write both.
- Failed ioctls are always traced (`traceln!`, which also reaches the diagnostic channel); `Cpu::nvdrv_request` reports missing handlers.
- Submissions retire inside their ioctl, so syncpoint waits see only completed work, `nvQueryEvent` events are always signalled, `EventWaitAsync` signals on arrival, and an unsatisfied wait reports the timeout instead of hanging.
- A method that needs guest memory the backend has not finished putting back (a 2D blit, copy or compute launch run on the CPU after a sync answering `Pending`, as in a browser) parks its submission (`Channel::resume`, `Gpu::submit`): the ioctl writes no reply and is asked again 1 ms later (`retry_after`), and the same call runs on from that method, so nothing before it runs twice. Until it finishes, any other submission is asked again too, so none overtakes it, and so is a change to the mappings of its address space (`FreeSpace`, `UnmapBuffer`, `MapBufferEx`, `Remap`): Mario Kart 8 Deluxe's movie decoder unmapped a frame while the copy out of it was parked, which the hardware would have run long before. The submission still retires inside its ioctl. A host paced to real time ends the slice when that wait idles past it, which is when a browser's readbacks land; the instruction-counted clock of `screenshot --defer-readbacks` idles through about a hundred retries a wait instead.
- `GetConfig` (`nv!` debug overrides) is refused with `ConfigVarNotFound`: `NvOsGetConfigString` treats any success as "set", and an empty success enabled `NVWSI_FILL` (per-pixel buffer fills, a large share of a Just Dance 2017 frame). Production consoles have no such settings.
- ZBC tables exist only so `ZbcQueryTable` reads back what `ZbcSetTable` stored (otherwise drivers re-register until full). A repeated value takes another reference; nothing is evicted.
- `VsmsMapping` (0x13, `0xc0084713`) is not in libnx; it was identified from `nnSdk`'s `nvrm_gpu` and matches upstream `nvgpu_gpu_vsms_mapping_args`: u16 `{gpc, tpc}` entries, one per TPC. Counts must agree with `GetCharacteristics` (TPC mask and VSM map derive from the same constants).
- `Remap` (0x14) is a batch at explicit GPU VAs in big pages, used to back sparse reservations. `MapBufferEx`'s `MODIFY` remap is under [vmm.rs](#vmmrs).
- `SetTimeslice` is accepted with no effect: it means nothing when each submission runs to completion, and nnSdk's nvn checks it, so refusing is wrong.
- `TRACE_NV` logs what an nvmap `Alloc`'s handle was bound to before: a second alloc on an already-mapped handle would leave the GPU address space on the old memory, which shows up only as buffers of zeroes.

## `channel.rs`

- A channel's pushbuffers are one continuous stream: a method group may run past the end of one pushbuffer and finish in the next (a 64-word inline upload split across two GPFIFO entries used to fault).
- nvhost pre-binds `MAXWELL_CHANNEL_GPFIFO_A` to subchannel 6; deko3d writes syncpoint increments there without `SetObject` (otherwise hbmenu's fence never signalled).
- Methods below 0x40 are host methods on every subchannel.
- The 2D blit launch (`Engine2D::LAUNCHES_BLIT`), copy `LAUNCH_DMA` and compute `SEND_SIGNALING_PCAS_B` flush the renderer first, since the wgpu backend keeps render targets on the device until flushed (Just Dance 2019 resolves its MSAA target with a 2D blit every frame). The exceptions (`Renderer::blit`, `Renderer::guest_memory_is_current`) are in [gpu-backend.md](gpu-backend.md).
- Semaphore acquires count as satisfied because a submission runs to completion before the ioctl returns.

## `vmm.rs`

- There is no VRAM: guests allocate CPU memory, nvmap wraps it, and `/dev/nvhost-as-gpu` maps whole nvmap ranges at contiguous GPU VAs, so a sorted list of ranges translates exactly at no per-page cost.
- `MAP_BUFFER_EX` with `NVGPU_AS_MAP_BUFFER_FLAGS_MODIFY` (bit 8) re-maps a sub-range of an existing mapping with a new kind; drivers (deko3d among them) give block-linear images their swizzle within one buffer this way. `offset` names the existing mapping and the handle is 0; treating it as a normal map failed with `BadParameter`. The covering mapping is split (the map is keyed by start VA); backing addresses do not move. A remap may span pieces of earlier splits: Unity re-kinds a reused range of its render-target pool across old cuts, and refusing that left A Short Hike's scene with no target bound.
- `translate` assumes non-overlapping ranges, so `unmap_range` (for `REMAP`) trims partial overlaps.
- Translation cache: 8 ways, round-robin, since a single entry thrashed between constant buffers, textures and the render target per pixel. Fields sit in separate `Cell` arrays because a `Cell<Option<(u64,u32,u64)>>` per way copied 32 bytes on every rejected way. Any mapping change flushes the cache.

## `macro_engine.rs`

- A taken branch overrides an exit in its delay slot, so an exit paired with a branch fires only when the branch is not taken (deko3d's `FillRegisters` relies on this).
- Macro host reads and writes happen in program order, so a read sees the side effects of earlier emitted writes (e.g. `WriteHardwareReg` firmware-call completion polling).

## `mod.rs`

- Every presenting process has a layer (`Gpu::present` takes it by pid), and the screen is the layers blended in pid order, so an applet sits above its caller. A layer alone on screen is the framebuffer itself, uncopied; `remove_layer` uncovers what was below when a process ends.
- The renderer backend lives on `Gpu`, not on a `Channel`: every `open("/dev/nvhost-gpu")` creates a channel (Asphalt 9 opens four) and `nvClose` destroys one, so installing it on a channel lost the device. It is lent to the executing channel for each submission and taken back even on a fault. A flush goes through the presenting process's last submitting channel, with that process's memory.
- Scan-out honors the queued `transform` (Minecraft queues `FLIP_V` every frame) and `crop` (A Short Hike renders 1280x720 into a 1920x1080 buffer). `ROT_90` is refused rather than shown wrong. An empty or off-surface crop means the whole surface; crops are guest data, so they are clamped.
- Scan-out reads the whole surface with one page-table walk (`read_into`) into a buffer reused across frames, computes row/column swizzle halves once, and clamps the slice so per-pixel bounds checks are elided. Per-pixel `read_le` was the full cost of a frame in titles that draw nothing.

## `engine/threed/`

- The Vulkan driver (Minecraft 1.26) draws with `DRAW_INDEX_BUFFER{32,16,8}_BEGIN_END_INSTANCE_FIRST` (0x5F9..0x5FB): first index, count and topology in one word, the index size from `SET_INDEX_BUFFER_E` as Eden reads it. Treated as inert, every one of the title's draws was dropped and the screen stayed black. The `_SUBSEQUENT` variants are not handled yet.
- `SET_PRIMITIVE_TOPOLOGY` (0x65C) replaces a one-word draw's topology while `SET_PRIMITIVE_TOPOLOGY_CONTROL` (0x652) is 1; the driver leaves the per-draw field at points. `VERTEX_BEGIN_GL` draws keep their own.
- The MME shadow RAM (`SET_MME_SHADOW_RAM_CONTROL`, 0x049): writes are recorded while tracking (0, 1), go only to the registers in passthrough (2), and take the recorded value in replay (3). The driver sets its present blit's state in passthrough and replays the same block to restore the title's; without replay its `SET_VIEWPORT_SCALE_OFFSET` 0 stuck and the title's next composite collapsed to a pixel.
- `SET_VIEWPORT_SCALE_OFFSET` (0x64B) off means the vertex stage writes window coordinates (the driver's blit computes pixels from the vertex id). The software rasterizer passes them through; the device path converts them back to clip space ([pipeline.rs](#pipeliners)). The register starts on, so titles that never write it keep the transform.
- `RenderTargetControl` (0x487) counts a draw's colour outputs and maps output `i` to a slot (`Engine3D::color_outputs`, `render_target_slot`); output 0 is written even at a count of zero. Each output blends with `blend_target(i)` and masks with its slot's `color_mask`, as output 0 always has. An output the fragment program's map leaves out keeps what its target holds: Breath of the Wild writes its deferred G-buffer through six outputs, binds one surface to two of them and writes only one. With output 0 alone drawn, only an `RG8` G-buffer plane was written and the lit scene came out as sky over black.
- `SET_RT_LAYER` (0x573) picks the layer of a layered colour target that draws render into, unless bit 16 hands the choice to a shader (not supported). `Engine3D::draw_target` moves the target to that layer's address, where a layer clear also lands and where the device holds surfaces, so a sampled array or cube finds each layer. Minecraft draws its title panorama's six cube faces this way; with the register ignored every face landed on layer 0, leaving a flat blue screen.

## `engine/twod.rs`

- deko3d routes `dkCmdBufBlitImage` to the 2D engine whenever the copy engine cannot express it (scaling, format conversion, filtering).
- Fast paths: bilinear with taps exactly on texel centres degenerates to point sampling (Just Dance 2019 resolves 2x2 MSAA with `du_dx=2, dv_dy=2, src0=(0.5,0.5)`); point sampling between byte-exact formats is a move with no decode/encode. A surface in one mapping is translated once, not per texel ([upload.rs](#uploadrs)). Column swizzle halves are precomputed once per blit, row halves once per row. A filtered blit between disjoint surfaces runs in host memory (`blit_filtered`), decoding each source row once and only the columns a tap lands on.
- Tiled walk: walking a block-linear source in row order jumps a GOB column every 8 texels; tiles read a few contiguous blocks. Overlapping surfaces keep row order, the only case where the order is observable.
- `blit_staged` does byte-exact copies between disjoint surfaces in host memory (gather source texels, read the target whole, scatter, write back including padding); through guest memory the 2560x1440 Just Dance 2019 resolve was a chain of cache misses. Gathered texels are kept with the source range marked, so an unchanged source is not gathered again. It runs only where it is exactly the per-texel copy: offsets in bounds, no watchpoint over either surface, no protected byte in the target.
- `scatter` relies on `blit_staged` having proven every index in range, which removes bounds checks from the hot loop.

## `engine/copy.rs`

- `SetSrcWidth`/`SetDstWidth` count elements, not bytes; with the remap on, an element is a pixel. Bytes worked for deko3d (remap off) but shredded JKSV's Mesa icon uploads into strips.
- `LaunchDma`'s `DATA_TRANSFER_TYPE` 0 (`NONE`) copies nothing and only releases its semaphore (`SEMAPHORE_TYPE` 1 writes the payload, 2 the payload as 64 bits and a timestamp at +8, as compute releases do). A remap whose components are all constants reads no source: Minecraft fills buffers that way with the source address left at 0.
- Remap-off copies go a contiguous run at a time (pitch rows contiguous, block-linear in 16-byte pieces); per-byte swizzle and translation was a fifth of an hbmenu frame.

## `engine/compute.rs`

- A compute launch is the QMD address (>> 8) written to `SEND_PCAS_A`, then `SEND_SIGNALING_PCAS_B`; the QMD holds everything about the grid. The program region and texture pools sit at the 3D class's methods, so `gpu::texture` serves both.
- Compute class inline upload (0x60..0x6D) is needed: Zelda: Echoes of Wisdom uploads every QMD this way.
- Refused dispatches are counted, not propagated, as refused draws are.
- `write` only records a launch; the channel offers it to the renderer (`Renderer::dispatch`) and runs it in software, after a flush, when the renderer declines. Both routes write the QMD's release semaphores (`compute::release`).

## `compute.rs`

- The software dispatch is scalar: CTA threads run one after another. This is exact except at barriers and warp instructions (`shfl`, `vote`), where threads suspend until the whole CTA or warp arrives. Atomics need no locking and races cannot be observed (a racy kernel gets a valid answer that differs from hardware's).
- Programs without barriers or warp instructions reuse one invocation serially (cheaper); otherwise every thread keeps a live invocation.
- `MAX_DISPATCH_THREADS` (1<<20) is a liveness guard: the GPU stack runs on one browser worker thread.
- The WGSL translator's compute stage (`Stage::Compute`) is one invocation per thread and one workgroup per CTA, its shape (`ComputeShape`) baked into the module. `s2r` reads go through `sreg`, which a compute module answers with the lane (`local_invocation_index % 32`), thread and CTA ids and the QMD's shared and local sizes; the graphics stages still read zero, as before.

## `surface.rs`

- A block-linear address is a row term plus a column term (`block_linear_row` + `block_linear_column`), so walks hoist the row, which carries the only divisions, out of the inner loop. A render target's tile mode carries a block depth (bits 8 to 11) when it is one slice of a volume: `Layout::BlockLinear::block_depth_gobs` then spaces its block columns and rows that many GOBs further apart, which with the slice's own address is the volume's addressing (`block_linear_volume_offset`). A sampled volume keeps 1 there and its depth in the TIC's. Within a GOB only `x % 16` is linear, so walks proceed in contiguous runs (`Layout::run_at`: 16 bytes within a GOB, whole rows for pitch).
- 3D block-linear surfaces interleave slices inside a block, so slice `z` is not `z` layer-strides along. 2D arrays store layers back to back, `layer_stride` apart.
- MSAA is spatial: a 4x 1280x720 target is 2560x1440 texels. Render/depth target registers describe the expanded surface; scissor, viewport and clear stay in pixels. A sample's slot is the texel its programmed location falls in (reproducing deko3d's `locationsMS4`/`MS8`); collisions fall back to raster order. Rendering the expanded surface once per texel reproduces only grids whose samples sit at texel centres (`samples_at_texel_centres`, compared by position so a programmed table naming the centres still qualifies). `AntiAliasEnable = 0` moves every sample position to the pixel centre but keeps the slots.
- sRGB encode/decode lives in `ColorFormat` so render targets, the blitter and the sampler agree (split, it darkened surfaces and brightened sRGB frames at scan-out). Colours are always linear at the API.
- The blend unit clamps only for fixed-point targets; `B10G11R11_FLOAT` must not clamp (Persona 5 Royal's bloom).
- `f32_to_f16` rounds to nearest even and produces subnormals; truncating biased fp16 shader math (`hadd2`/`hmul2`/`hfma2`) toward zero. BC6H reached subnormal decode.
- `host_shuffle`/`is_byte_exact`, asked once per surface, let scan-out and copies move bytes for 8-bit non-sRGB formats instead of a decode/encode round trip. The `UNORM8` table replaces the blitter's `v / 255.0` divisions.
- `encode_registers`/`decode_registers` (`sust.p`/`suld.p`) keep integer channels as integers; going through `f32` would round values above 2^24.
- `A8B8G8R8_SNORM` was misread as UNORM; SNORM's most negative value clamps to -1.0.

## `texture.rs`

- Vertex shaders sample through the same `MemoryTextures` as fragments, so the rasterizer can run a draw the device refused.
- `texs` immediates are dword indices into the `TexCbIndex` bank (nouveau 15, deko3d 0), which holds the bindless handle into the TIC/TSC pools; the two stages index different constant buffers with the same slot. Read as bytes, they land in nouveau's fixed header (`0, 1, 2, ...` on GM107), which looks like a handle table: every draw resolved to one handle and text drew one glyph repeatedly.
- Compressed surfaces are swizzled and addressed in blocks; texel units make the stride a block too large and shred the image into diagonal ribbons.
- `BlockCache`: decoding a whole block per fetch was a large share of an ASTC frame. Four ways cover a bilinear footprint across a block corner; round-robin fits scanline order. Blocks decode in place because an ASTC block is 2.3 KiB.
- Float and packed format support came from titles: A Short Hike composites from R16G16B16A16 FLOAT (refusing it left the frame transparent), Persona 5 Royal uses A2B10G10R10 and B10G11R11, Nintendo Switch Sports samples ZF32 as R32, Echoes of Wisdom reads S8D24 stencil as uint in a compute shader.
- Layer stride for mipmapped arrays and cubes includes the mip chain: Tomodachi Life's 64x64 cubes with seven levels are 0x6000 apart; 0x4000 sampled every face from the first face's mips.
- Cubemaps: `DEPTH_MINUS_ONE` holds 1 but there are six faces. Software cube sampling clamps at face edges where the GPU filters across seams; nothing seen samples there. Gathers do cross the seam (`cube_neighbour`).
- Shadow sampling is PCF (compare per tap, then filter) and returns `[c, c, c, 1]` without swizzle. Integer textures are always nearest-sampled.
- A TIC entry with address 0 is a null image, as in Eden's `IsValidEntry`: samples, gathers and surface loads read zero, size queries answer 0, stores are dropped. Echoes of Wisdom's cubemap filter samples four unbound slots (handle 0, an all-zero entry), which read as `COMPONENTS_SIZES 0`/`R_DATA_TYPE 0` and refused 7 of 8 dispatches. On the device a null image binds one transparent texel (`upload::null_texture`).
- `TRACE_TEX` is separate from `TRACE_GPU` because a frame has a million method traces but only dozens of textures.

## `upload.rs`

- The software rasterizer reads attributes lazily through the MMU; a GPU backend must decide up front what to upload. Vertex array limits often point at the end of a heap, so uploads are bounded by the draw (`first`/`count`, or min/max index for indexed draws; reading indices is free since they are uploaded anyway). `MAX_UPLOAD` (64 MiB) makes an absurd stride*count report instead of allocating.
- An array's `limit` is the address of its last valid byte (inclusive). Fetches past it read zero, on hardware, in `raster::fetch_attribute` and through the device upload's padding; Tomodachi Life reads 16 bytes past an array end, and refusing that latched every later frame onto the rasterizer.
- A vertex span charges its last element only the attribute's length (WebGPU sizing), not a whole stride.
- Constant banks: resolving only the banks shaders read (`Banks::Read`) matters, since the Home Menu binds eight 64 KiB banks per draw and reads two.
- `TextureKey` comes entirely from the TIC, so descriptor rewrites produce new keys and only texel writes need watched pages. Swizzle and sampler apply at sample time and are not in the key. Textures dominate uploaded bytes, so `bytes` is an `Arc<[u8]>` and a cache hit costs a refcount.
- BC textures upload compressed (WebGPU supports BC). ASTC is decoded to `Rgba8Unorm` since desktop browsers lack `texture-compression-astc`. Partial-block compressed images (the Home Menu's 1x1 BC4/BC5 default textures) are decoded, because WebGPU requires whole-block extents and rounding up would change sampled coordinates. Decoded images keep their sRGB encoding; the format says so and the device applies the transfer function.
- `ZF32_X24S8` uploads as `R32Float`, the low half of each 8-byte texel (`Copy::FloatDepth`). Its `TextureKey` keeps the guest's 8-byte rows, which held-surface matching and page watching address by, while the upload's rows are 4 bytes a texel. Persona 5 Royal samples one; refusing it latched every frame onto the rasterizer.
- `S8D24` read as uint uploads its words unchanged (`R32Uint`, `Widen::PackedDepth`), and `PACKED_DEPTH_WGSL` (`fs_words`) converts them on the device into the `Rg32Float` that `DepthTexel::decode` gives the rasterizer: stencil bits in red, unorm depth in green. A Short Hike samples its depth buffer this way; refusing it latched every frame onto the rasterizer. A device that flushes denormals reads that red as zero. 32-bit float textures also need `float32-filterable`; without it the binding falls back instead of failing validation.
- `decode_blocks` reuses its strip and block buffers: zeroing a fresh 12x12 block per 4x4 block was a quarter of an ASTC title's upload time.
- Surfaces in one mapping are translated once (`ExecCtx::span`) instead of per texel, walking contiguous runs with `run_at`.
- `Target::write` patches over existing bytes because block-linear padding is not the surface's to zero. `write_strided` takes padded readback rows directly, avoiding a per-frame repack.
- Depth: WebGPU copies can read but not write `depth32float`, read and write `depth16unorm`, and touch `depth24plus` in neither direction. So Z16 stays 16-bit and everything else becomes f32 (24-bit depth is lossless in an f32 mantissa; a lossy round trip would break every `Equal` depth test next frame). Depth writeback preserves packed stencil bytes, matching `raster`'s `Fragments::write`. Just Dance 2017 renders all passes depth-only (no colour target).
- `read_range` reads a word at a time where it can: per-byte reads paid translation and page lookup 8x.

## `bcn/bc7.rs`

- Every BC7 mode uses exactly 128 bits; a test sums the mode table, catching a mistyped field width in any row.

## `shader/mod.rs`

- Programs are 32-byte blocks: one `sched` control word, then three instructions. Binaries carry no length, so decoding walks the CFG from the entry and stops each path at its terminator (stopping at the first `exit` misses code reached by branches). A branch target on a `sched` word means the block's first instruction.
- Mesa-compiled binaries carry a 0x50-byte header whose first bytes are not reliably zero, so the header is assumed only when slot 1 does not decode; deko3d/uam binaries have none. This lives in one place because the rasterizer, the wgpu backend and compute all need the same answer.
- The SPH's `omap.target` is the only source of fragment output registers: registers are assigned in target order, targets with no writes are skipped, and disabled components inside a written target still take a register. The WGSL translator maps outputs the same way (`Translation::header`); it used to read output 0 from `r0..r3` whatever the map said.
- `brx` targets come from a constant-bank jump table. The decoder walks the selector's use-def chain back to the `imnmx` clamp (which the scheduler may hoist far away) and gives up on any unexpected or predicated write rather than guess a table length. Jump targets are base + entry, unrounded, so the base must not be aligned past `sched` words. The resolved targets are kept for later CFG analysis.
- `DecodeReads` records the GPU-virtual pages and constant words a decode read, so backends can cache translations; only constant words are re-read to validate, program pages are watched.
- The rasterizer interpolates only the varyings the fragment shader reads (`interpolated_slots`), since the cost is per covered pixel.

## `shader/cfg.rs`

- Maxwell uses a reconvergence stack (`ssy`/`sync`, `pbk`/`brk`, `pcnt`/`cont`), not structured control flow. If every pop statically pairs with one push, each pair maps to an `if` or loop and WGSL translation is mechanical; an instruction reachable with two different stacks needs per-lane masking instead. `Cfg::pairing` answers this question; it does not build structured output.

## `shader/isa/`

- Perspective correction: the fragment shader reads interpolated `1/w` at `a[0x7c]` via `ipa pass`, computes `w = mufu rcp`, and multiplies other varyings by it in perspective `ipa`. Barycentrics are therefore screen-space linear in both renderers.
- The guard predicate is bits [16,19) plus negate at 19; register 7 is `PT`.
- Maxwell has no 32-bit integer multiply; multiplies are `xmad` chains (16x16 + accumulate).
- `lea.hi` adds the low word of the 64-bit `hi:a` shifted right by `32 - shift`, and `neg` negates all 64 bits. WGSL has no `u64`, so the translation borrows across the two words by hand; `lea_computes_the_same_words_on_both_renderers` checks it against the interpreter. Echoes of Wisdom's first was `lea.hi r4, r4, r3, rz, 0x1f`. The `.x`, `.cc` and predicate-writing forms are refused, as Eden refuses them.
- `f2i` takes neg from bit 45 and abs from bit 49, as Mesa's `emitF2I` emits and the other conversions do; Eden's `F2I` names them the other way round, so its field names are no proof. It reads an f16 source from the half bit 41 names, which only Eden documents. `pset` writes 1.0 or all ones (bit 44) from three predicates. `fcmp` copies raw bits, picking between two values by how a third compares with 0.0. Just Dance 2023's shaders use all three.

## `shader/interp/`

- `ShaderResult` boxes `Error` (56 bytes) so a `Result` fits in a register pair; unboxed, it cost a measurable share of a Home Menu frame. Unbox only at `resume`'s boundary.
- Per-draw caches (`ConstCache`, TIC/TSC descriptors, decoded blocks) are owned by the caller so the per-fragment structs can be rebuilt while the pixel loop borrows `ctx` mutably. Constant buffers can't change mid-draw.
- `gpr` has 256 slots so `RZ` indexes without a bounds check; RZ is rewritten to 0 unconditionally (cheaper than a branch). Program ops and predicates are sliced to one length to share one bounds check.
- `texs` results are deferred: compiled code reads destination registers' old values between the fetch and its first consumer, so each result lands just before its first reader, or at the next branch/exit. `reads()` must list every operand register of texture ops or queued results land late.
- `a[]` is a flat 256-word array with a written-mask (unwritten `clip.w` must default to 1.0); it replaced a HashMap that dominated per-pixel cost.
- `centroid` needs no handling because shading is per covered sample at its centre.
- Half ops compute in f32 and round once at the merge; `.ftz` uses the half threshold only for half lanes.
- Warp shuffles to lanes absent from the warp (a quad is 4 of 32) read the caller's own value; votes count only lanes that reached them. In a fragment shader the device runs `vote` over the quad too (ballot bit n is lane n, `all` is four of four), as the rasterizer does, so a ballot has four bits where hardware's has up to 32; a compute program's runs over a 32-lane subgroup ([gpu-backend.md](gpu-backend.md)). The rasterizer shades a fragment shader with a `vote` in quads, as it does one with `shfl`.
- `SR_Y_DIRECTION` must be +/-1.0 (zero deletes screen-space directions). Packed `SR_TID`/`SR_NTID` are refused because their layout is unconfirmed.
- `tmml` (0xdf58, `tmml.b` 0xdf60) is decoded only with mask 0x2, the level a sample would read, which is 0 because every image is sampled at level 0. The hardware returns the raw LOD first and that level second, as nouveau's `handleTXLQ` reads them (Eden returns them in SPIR-V's order instead); a mask asking for the raw LOD stays undecoded and falls back (Breath of the Wild).
- `tex.aoffi` offsets are texels scaled by level size (Persona 5 Royal's blur); a 3D image's z nibble (bits 8 to 11) moves it by slices (Breath of the Wild).

## `shader/compiled.rs`

- Lowering exists because fragment shaders run once per covered pixel: binary-searched branch targets, linear `texs` lookups and constant-cache reads are paid once instead.
- Ops are stored apart from predicates because an `Op` is 32 bytes against 40 for `Instruction`: two per cache line instead of 1.6.
- Constant folding is sound because a constant buffer cannot change during a draw (one method). Unbound banks stay unfolded so the error still comes from the reading instruction. Undecoded branch targets error only when taken, since unreachable branches exist in shaders that run fine.

## `shader/wgsl/`

- Control flow: Maxwell has a reconvergence stack (`ssy`/`sync`), backward `bra` loops and `brx` multi-way jumps. Although `cfg` proves push/pop pairing for every Home Menu shader, the translation emits a `switch` over `pc` inside a `loop` with an explicit stack (the `interp::Invocation` machine in WGSL), which cannot mistranslate control flow. It is slower on the GPU; recovering structure where `cfg` proves it safe is a future change local to this module, with the state machine as fallback.
- Registers are untyped `u32` with `bitcast<f32>`: shaders mix integer and float use of one register, so typing them would be wrong. Registers, helpers and bindings are recorded while emitting, with no second pass that could disagree; `Layout::of` derives from the `Translation` for the same reason.
- Registers are `var<private>` because a fragment shader's colour is r0..r3 after the invocation; there is no output attribute.
- Varyings are `@interpolate(linear)`: `ipa` receives value/w and multiplies by `rcp(a[0x7c])` itself ([shader/isa/](#shaderisa)), so perspective interpolation would divide twice (it looks like a texcoord bug). The vertex stage multiplies by 1/w; fragment `position.w` is 1/w, matching `a[0x7c]`.
- `Layout::flip_y` must be set when the guest viewport does *not* mirror y (offscreen): WebGPU has no negative viewport height, and its NDC-to-framebuffer transform already mirrors. Using `Viewport::flip_y` directly flips twice and still looks almost right, because UIs are mostly symmetric.
- Global memory addresses for `ldg`, `stg` and atomics are either a descriptor in a constant bank plus a register index (A Short Hike) or a whole descriptor loaded by `ldc.64`/two `mov`s (Nintendo Switch Sports). Other addresses are unsupported rather than guessed.
- A `ldg` through a constant-bank descriptor binds the buffer the descriptor names, at the address register minus the descriptor's low word, read at the `ldg`. Taking the register the `iadd.cout` added from was wrong: shaders rewrite it before loading, even in place (`iadd.cout r12, r12, c0[0x110]`). Echoes of Wisdom's controller applet builds its glyph quads from records loaded that way, and on the device every line of its text was missing.
- `iscadd.cc` (bit 47) sets the carry `iadd.x` reads, like `iadd.cc`. The interpreter ignored it, so `iscadd.cc r0, r1, c0[0x510], 2` then `iadd.x r1, rz, c0[0x514]` took a stale carry on the rasterizer and was not recognised as descriptor plus index on the device. Echoes of Wisdom's visibility passes build their `ldg` and `red` addresses this way.
- A null descriptor (address 0) binds an empty buffer that reads zero instead of refusing the draw: Echoes of Wisdom's vertex shader takes one from a bank slot it never reaches. Any other unmapped address still refuses.
- `stg`, `red` and `atom` through a descriptor make its binding read-write (`array<atomic<u32>>`, fragment stage only: WebGPU bars writable storage from the vertex stage). Unsigned `add`, `min`, `max`, `and`, `or`, `xor` and `exch` are WGSL atomics; `cas`, signed `min`/`max` and `f32` `add` run a bounded compare-exchange loop (`layout::ATOMS`); other signed operations take the unsigned row. 64-bit and shared atomics, `inc`, `dec`, `safeadd` and the other `f32` operations are refused. Indices outside the buffer are dropped, so an address below the descriptor is lost where hardware would write the pool before it (one of 2299 writes in Echoes of Wisdom's frame 67).
- Bindless `tex.b` handles are traced to a constant word via the nearest unguarded write in-block or the program's sole write; a guess would draw a plausible wrong texture instead of falling back. NVN also ORs a handle from two words, image and sampler halves (`mov` then `lop.or` with a constant; Just Dance 2023), traced as `TextureSlot::BindlessPair`. Its sampling key is the first word's, so a program pairing one first word with two second words is refused (`SharedHandleKey`). The trace follows control flow: a handle written more than once still traces when one unguarded write reaches the sample on every path (Just Dance 2023 sets it up once, samples it in later blocks and reuses the register afterwards); paths that disagree are refused. A guarded write counts only for a sample under the same guard, when nothing between them rewrites that predicate (`interp::writes_pred`): the sample then runs only where the write did. Just Dance 2023's song select loads a handle and samples it all under `@!p0`.
- Texture results are stored immediately, unlike the interpreter's scoreboard-like deferral to first use; the two differ only when the destination is overwritten before any read (the interpreter still lands the sample late, hardware does not).
- Shadow samples bind as `texture_depth_*` with a comparison sampler; the backend draws guest shadow maps into `depth32float` ([builtin.rs](gpu-backend.md#builtinrs)). A slot sampled both with and without comparison cannot be one binding and is refused (`Unsupported::DepthCompare`).
- A folded binding (`Fold`, set by the backend like the swizzle) is declared `texture_2d_array` and sampled through `folded_coords`: the texel row picks the tile, `v` is made tile-relative, and a row offset is applied before the tile is chosen. `txq` still answers the unfolded size.
- Descriptor swizzles matter: two thirds of Home Menu draws sample single-channel images as `[R, R, R, One]`. WebGPU has no per-texture swizzle, so the sampling hook applies it.
- Each stage gets its own bind group because stages index different constant buffers and textures with the same numbers. A stage's constant banks are one storage buffer at binding 0 (`CONSTANTS_BINDING`: a table of first word and word count per bank, then the banks), texture i at 32 + 2i with its sampler beside it, and `ldg` buffers from 96 (`GLOBAL_BINDING`). The backend's bind group cache is under [lib.rs and device.rs](gpu-backend.md#librs-and-devicers).
- Subgroups: browsers require `enable subgroups;` and naga rejects it, so `Caps::subgroup_enable` is separate. wgpu's web backend cannot request or report `subgroups`, so `QUAD_SWAP` emulates quad swaps from `dpdxFine`/`dpdyFine`: splitting a register into two 16-bit halves makes the difference and re-addition exact in f32, with clamp/round guarding the spec's approximate derivatives. The `derivative_uniformity` diagnostic is off because derivatives sit inside the dispatch loop. `fswzadd` needs only the lane index (from `position`), not device support.
- Non-finite float literals (`bitcast<f32>(2139095040u)`) are constant expressions WGSL rejects (Chrome refused a Tomodachi Life module that naga accepted), so they are bound to a `let`.
- Half precision: `pack2x16float` rounds to nearest even like `f32_to_f16`, so the backends agree on finite results; overflow differs (WGSL indeterminate, interpreter infinity). The half `.ftz` threshold is the half subnormal range, far above f32's.
- Centroid is emitted even though single-sample passes treat it like center, because with `--device-msaa` passes really are multisampled. Both stages must agree; the backend copies the fragment stage's set to the vertex stage.
- Depth-only passes (Just Dance 2017) return nothing from the fragment entry (an unbacked `@location(0)` fails pipeline creation) but still run for `kil` and alpha-to-coverage.
- Expanded multisample (per-texel) rendering must apply the sample mask and alpha-to-coverage in the shader (`wgsl::Coverage`). Alpha-to-coverage keeps a prefix of `floor(alpha*count + 0.5)` samples (Rust `round` semantics, not WGSL's ties-to-even).
- Integer vertex attributes are declared `vec4<i32>/<u32>` (WebGPU matches base types) and bitcast into `a[]`; BGRA attributes swap components 0 and 2 in the entry point, since WebGPU has no BGRA vertex format.
- A draw whose fragment stage is disabled writes no colour (Persona 5 Royal's title screen): both renderers run `no_fragment_program`, a lone `exit`, as a depth-only pass, and skip the draw when it leaves depth alone (`writes_no_surface`). Refusing it latched every frame onto the rasterizer, which refused it too.

## `renderer.rs`

- The `Renderer` trait lets a GPU backend sit beside the software rasterizer, which stays the bit-exact reference to check it against (most of a Home Menu frame is rasterization, so only a GPU backend makes it fast).
- A backend that keeps surfaces GPU-side owns when render targets go back to guest memory; `present` deswizzles straight out of guest memory, unless the backend owns the screen and shows the surface itself (`Renderer::scan_out`, `Gpu::present_on_device`).
- Clears are on the trait too: on a GPU they are a render pass load op, and draws on a GPU surface would disagree with clears in guest memory.
- `lost()`: a browser can drop the device (driver reset, GPU process restart, memory pressure). Falling back to the rasterizer for the rest of the session is far slower, while a lost device can just be requested again (the worker does so a bounded number of times); `lost()` is cheap enough to poll once a slice.
- `Flush::Pending` must not be ignored: landing a readback one flush late produced black frames, since a double-buffered title presents the surface whose readback was just requested. An engine about to read guest memory parks its submission on it instead (above).

## `raster/`

- The CCW rewind before the top-left rule matters because the tie-break splits an on-edge point only between two triangles that walk the shared edge in opposite directions. SDL emits quads as one CCW and one CW triangle; JKSV's 128x128 save tiles have a 45-degree diagonal through pixel centres, and every tile showed a one-pixel gap.
- `NUM_VARYINGS` covers Maxwell's generic `a[0x80]..a[0x280)` and the fixed-function texture coordinates `a[0x300]..a[0x3a0)` (`shader::varying_slot`, slots 40 to 49, counting on from the generic ones so slot `n` is at `0x80 + n * 0x10`). It was four, but Home Menu panel shaders interpolate slots 4-6: the zero read became an `rcp` denominator, infinity, then a `0 * inf` NaN encoded as black. The Vulkan driver's blits pass their coordinates in the texture slots; without them every texel read was (0, 0), and Minecraft's panorama faces were one colour each. The WGSL translator names the same slots. `interpolated_slots` keeps the cost down.
- Depth compare and blend factor/equation registers take both GL and D3D numberings (`1..=8` beside `0x200..=0x207`): Mesa (JKSV) writes GL enums, deko3d/nvn D3D ones. Decoding only one made Just Dance 2019's `LessEqual` (4) fall into Always and the GPU backend fall back on every draw, and turned the Home Menu's D3D `SrcAlpha`/`OneMinusSrcAlpha` into `One`/`One`, washing separators white. `pipeline` accepts both for the same reason.
- Fixed-point targets clamp the blend source (the GL rule, and the ROP's), which is what kills NaN: the Album applet normalises by total alpha, a transparent texel gives `rcp(0)` then `0 * inf`, and every icon had a black box. NaN floors at 0, not SNORM's -1.
- Culling judges winding in window space after the viewport transform; window-origin `FlipY` swaps the front winding only through a viewport that keeps y. Judging in NDC agrees through a y-mirroring viewport (every nnSdk main pass) but is opposite otherwise: Echoes of Wisdom's offscreen post-processing (front=CCW, FlipY, back culling, y kept) lost every full-screen quad, leaving black frames. A Short Hike's Unity blits set FlipY *with* a y-mirroring viewport, and swapping there culled all 883 of its quads; Tomodachi Life's (no FlipY, y mirrored) pin the unswapped case. No title on hand has FlipY clear through a viewport that keeps y, so that case is unconfirmed.
- Near-plane clipping is required: `w <= 0` sends projected vertices to infinity or the wrong side, smearing triangles across the framebuffer.
- Depth-only passes: Just Dance 2017 binds its Z24S8 surface as colour target 0 and works in the depth buffer; requiring a colour target dropped every draw. Extent comes from whichever target is bound.
- Attribute fetch: "fixed" attributes read the default instead of erroring (JKSV leaves attribute 2 fixed; erroring dropped its full-screen background quad). Integer attributes carry bits, and every shape titles use must decode (Just Dance 2019's signed bytes, Minecraft's `4x16` halves and Echoes of Wisdom's `1x8`/`2x8` integers each dropped most of a frame). 8-bit shapes read only their own bytes, since a one-byte attribute may end a mapping. Reads past the array limit are zero ([upload.rs](#uploadrs)). A zero-stride array is one element every vertex reads (Just Dance 2023); `upload` skipping it left the device an attribute with no bytes, so every frame fell back.
- Vertex shading is cached per guest index (`IdMap`); re-running it per reference is the loop's most expensive possible cost. Constant buffers are read once per stage per draw.
- Quad walk (warp shuffles): Checkpoint's antialiased text differences coverage against the neighbour; per-pixel shading gave zero differences and solid blocks. Helper lanes outside the box are shaded but never depth-tested or written (they are outside the scissor too).
- Point and line draws are not rasterized, but a vertex shader that stores to global memory still runs for each vertex and its stores land: Nintendo Switch Sports resets records in memory with a points draw whose vertex shader only loads a pointer and stores through it. The refusal is still reported.
- `TRACE_WGSL` translates the unfolded program, as the device does (folding turns the constant operands a `ldg` descriptor is traced through into immediates, which made every such shader look untranslatable), and runs before the primitive check so point and line draws are traced too. It traces a summary per shader; the modules go to a `dump` sink only when a harness installed one that wants `Artefact::RasterWgsl`. The core writes no files, so a browser gets only the summary.
- Alpha-to-coverage uses a fixed sample prefix rather than hardware's dither; average coverage is the same and a resolve averages the dither away.

## `pipeline.rs`

- `Pipeline` resolves register state into WebGPU vocabulary so backends never re-decode raw codes. Anything WebGPU cannot express (except fans, handled by index rewriting) is `Unsupported`, never approximated, and the caller falls back to the software rasterizer. Unknown codes are errors here even where the rasterizer defaults (e.g. blend factor `One`).
- WebGPU's blend constant is per channel, so a blend reading only the constant's alpha (`ConstantAlpha`, `OneMinusConstantAlpha`) reads `Constant` with that alpha in every channel (`Pipeline::blend_constant`); beside a per-channel constant factor it is drawn so only when every channel of the constant equals its alpha (Breath of the Wild).
- Facing is judged in window space like `raster::culls`, so it carries over unchanged through the y negation (`Layout::flip_y`, [shader/wgsl/](#shaderwgsl)).
- With `SET_VIEWPORT_SCALE_OFFSET` off, `Viewport` covers colour target 0 (the surface clip is left at 16384, past WebGPU's limit) and carries `window_coordinates`, and the vertex entry maps x and y to `2 * p / size - w` before its y flip: WebGPU clips before the viewport, so passing pixels through would clip them away.
- Clip z: the GL (-w..w) or Vulkan (0..w) convention is inferred from the viewport's depth transform (`Viewport::depth_minus_one_to_one`: `translate - scale >= 0`, e.g. scale 0.5 and translate 0.5 is GL); WebGPU clips 0..w.
- `Rgba16Float` can't stand in for `Rg11b10Ufloat` (guest surface width). The feature `Rg11b10Ufloat` rendering needs and the native-only 16-bit norm formats are under [convert.rs](gpu-backend.md#convertrs). Integer colour formats are unnamed except `RGBA32_UINT` (`Rgba32Uint`), whose raw words the shader writes and reads, and `RGBA16_UINT` (`Rgba16Uint`, Breath of the Wild), whose words it saturates to 16 bits as the hardware does.
- Colour format names must agree with `gpu::surface` on width, sRGB and number kind (SINT was once misnamed `Rgba8Unorm`).

## `activity.rs`

- Activity is keyed by numbers, not strings: with thousands of draws a frame, labels are formatted only when a surface is first seen. Tallies cap at 256 entries (overflow summed into one) and refusal reasons at 32 (reasons may embed addresses).
- Refusals are tracked by reason, apart from per-surface `failed` counts, since the reason says what to implement next.

## `crates/core/examples/shader_coverage.rs`

- WGSL translation is per program and stops at the first blocker, so the tool reports the first blocker per program; fix and re-run.
- It reports twice: with no optional features (the browser) and with quad ops. Fragment shaders use `wgsl::QUAD_SWAP` without quad ops, so a gap between the passes means a vertex program shuffles.
- Programs are built into a full module, because texture dimensionality is rejected at binding layout, not at `translate`.
- One address can decode differently under different constant bank bindings; the first decode is kept.
- qlaunch is the preferred subject (a frame is almost pure rasterizer).

## `crates/core/examples/bcn_difftest.rs`

- The fixture comes from a throwaway C harness around `bcdec.h` (public domain): per codec, 4000 random 16-byte blocks, each written as the input then the decoded texels, in the order BC1, BC2, BC3, BC7 (RGBA8), BC4 (R8), BC5 (RG8), then BC6H UF16 and SF16 (RGB as `f32`).
- BC6H must match exactly; every 8-bit codec, BC7 included, passes within 1/255. BC1-3 differ legitimately: this decoder expands 5:6:5 endpoints by bit replication (as BPTC does), `bcdec` by rounding; both are within S3TC tolerance.
- The ASTC fixture is grouped by footprint (width, height, count as LE i32, then records of 16 input bytes and RGBA8 per texel). The reference converts from float as `clamp(v * 65536 + 0.5, 0, 65535) >> 8`, so the test does the same.
