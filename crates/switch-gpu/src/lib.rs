//! A `wgpu` backend for switch-core's 3D engine, behind
//! [`switch_core::gpu::renderer::Renderer`]. The software rasterizer is the reference;
//! any draw this cannot express runs there instead.
//!
//! Surfaces stay on the device across draws and return to guest memory only at
//! [`Renderer::flush`], so no draw ever waits on the device.

/// Whether a device with `features` can blend into a `format` target.
fn can_blend(format: wgpu::TextureFormat, features: wgpu::Features) -> bool {
    let float32 = matches!(
        format,
        wgpu::TextureFormat::R32Float
            | wgpu::TextureFormat::Rg32Float
            | wgpu::TextureFormat::Rgba32Float
    );
    !float32 || features.contains(wgpu::Features::FLOAT32_BLENDABLE)
}

pub fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    let wanted = wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::TEXTURE_COMPRESSION_ASTC
        | wgpu::Features::TEXTURE_COMPRESSION_ETC2
        // `R16Unorm` and friends, widened to a 32-bit float where missing; see `convert::sampled_texture_format`.
        | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        // Needed to filter a widened `r32float`.
        | wgpu::Features::FLOAT32_FILTERABLE
        // WGSL quad operations, for warp shuffles.
        | wgpu::Features::SUBGROUP
        // Lets the device use the adapter's real sample counts instead of only one and four.
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        // Makes `rg11b10ufloat` renderable.
        | wgpu::Features::RG11B10UFLOAT_RENDERABLE
        // Blending into 32-bit float targets.
        | wgpu::Features::FLOAT32_BLENDABLE;
    // Raise the storage buffer limit to the adapter's: a shader may read more than the guaranteed eight banks.
    let required_limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: wgpu::Limits::default()
            .max_storage_buffers_per_shader_stage
            .max(adapter.limits().max_storage_buffers_per_shader_stage),
        ..wgpu::Limits::default()
    };
    wgpu::DeviceDescriptor {
        required_features: adapter.features() & wanted,
        required_limits,
        ..Default::default()
    }
}

mod builtin;
mod convert;
mod readback;
mod stats;

pub use wgpu;

use switch_core::gpu::engine::threed::{Engine3D, ShaderStage};
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::pipeline::{self as state, Pipeline};
use switch_core::gpu::renderer::{Flush, Renderer, Software};
#[cfg(test)]
use switch_core::gpu::shader::compiled::Compiled;
#[cfg(test)]
use switch_core::gpu::shader::wgsl::{self, Stage};
use switch_core::gpu::shader::wgsl::{Layout, Translation};
use switch_core::gpu::surface::{Layout as SurfaceLayout, GOB_WIDTH};
use switch_core::gpu::upload::{Target, TextureKey, Uploads};
use switch_core::Result;

#[cfg(test)]
use builtin::grid_bytes;
use builtin::ResampleKey;
use readback::{Held, Pending, Scratch};
use stats::{json_string, DeviceErrors, Times, UploadBytes};

/// The memory a `ldg` reads, resolved from its descriptor.
struct GlobalUpload {
    stage: ShaderStage,
    slot: u32,
    bytes: Vec<u8>,
}

/// Upper bound on one `ldg` buffer, since the program does not say how far it reads.
const MAX_GLOBAL: u64 = 8 << 20;

/// Keep the leftmost `window` texels of each `stride`-texel row.
fn crop_rows(rows: Vec<u8>, stride: usize, window: usize, unit: usize) -> Vec<u8> {
    if window >= stride {
        return rows;
    }
    let mut out = Vec::with_capacity(rows.len() / stride * window);
    for row in rows.chunks_exact(stride * unit) {
        out.extend_from_slice(&row[..window * unit]);
    }
    out
}

/// How wide the red channel of a shadow map's texel is, and how it is read.
#[derive(Clone, Copy)]
enum Red {
    Unorm8,
    Unorm16,
    Float16,
    Float32,
}

/// `copyTextureToBuffer` wants each row of the destination aligned.
const COPY_ALIGNMENT: u32 = 256;

/// Where a module's textures start binding; see `switch_core::gpu::shader::wgsl`.
const TEXTURE_BINDING: u32 = 32;

/// What an unsupplied vertex attribute reads: the fixed `vec4` default, or zero for an unconfigured slot.
const ATTRIBUTE_DEFAULTS: [u8; 32] = {
    let mut bytes = [0u8; 32];
    let one = 1.0f32.to_le_bytes();
    bytes[28] = one[0];
    bytes[29] = one[1];
    bytes[30] = one[2];
    bytes[31] = one[3];
    bytes
};
const ABSENT_ATTRIBUTE: u64 = 0;
const DEFAULT_ATTRIBUTE: u64 = 16;

/// The shape of a surface's companion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Shape {
    /// A device multisample texture of `n` samples.
    Multisampled(u32),
    /// One sample per pixel at the centre (`AntiAliasEnable` off); every texel of the tile takes the result.
    PerPixel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ClearKey {
    color: Option<wgpu::TextureFormat>,
    depth: Option<wgpu::TextureFormat>,
    write_mask: [bool; 4],
}

/// A binding and what fills it, held until the bind group is built.
enum Resource {
    Buffer(u32, wgpu::Buffer),
    Texture(u32, wgpu::Texture, wgpu::TextureViewDimension),
    Sampler(u32, wgpu::Sampler),
}

/// Write a draw's two WGSL modules under `dir`, named by hash.
fn dump_wgsl(dir: &str, vs: &str, fs: &str) {
    let _ = std::fs::create_dir_all(dir);
    for (what, src) in [("vs", vs), ("fs", fs)] {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(src, &mut h);
        let key = std::hash::Hasher::finish(&h);
        let _ = std::fs::write(format!("{dir}/{key:016x}.{what}.wgsl"), src);
    }
}

/// `GPU_ONLY`'s value: one draw index, or the half-open range `a..b`.
fn draw_range(spec: &str) -> Option<std::ops::Range<u32>> {
    match spec.trim().split_once("..") {
        Some((from, to)) => Some(from.trim().parse().ok()?..to.trim().parse().ok()?),
        None => {
            let one: u32 = spec.trim().parse().ok()?;
            Some(one..one + 1)
        }
    }
}

/// Everything a render pipeline bakes in; viewport, scissor and blend constant are set per pass.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    /// The two module cache keys (hashes of the WGSL).
    vs: u64,
    fs: u64,
    /// `None` for a depth-only pass.
    target: Option<wgpu::TextureFormat>,
    /// Device depth format, whether depth is written, and the compare function.
    depth: Option<(wgpu::TextureFormat, bool, state::Compare)>,
    samples: u32,
    sample_mask: u64,
    alpha_to_coverage: bool,
    blend: Option<state::Blend>,
    write_mask: [bool; 4],
    topology: state::Topology,
    /// The index format a strip's primitive restart uses.
    strip_index_format: Option<wgpu::IndexFormat>,
    front_face: state::FrontFace,
    cull: state::Cull,
    /// Each bound vertex buffer, in the order they are bound.
    buffers: Vec<VertexBufferKey>,
}

/// Format, offset into the element, and shader location.
type AttributeKey = (wgpu::VertexFormat, u64, u32);

/// Stride, whether it steps per instance, and its attributes.
type VertexBufferKey = (u32, bool, Vec<AttributeKey>);

struct Bound {
    buffer: wgpu::Buffer,
    attributes: Vec<wgpu::VertexAttribute>,
    /// Zero for an instanced array and for the constant attribute: one element read by every vertex.
    stride: u64,
    step: wgpu::VertexStepMode,
}

/// A ceiling for pathological titles; real ones use a few dozen.
const SHADER_CACHE_ENTRIES: usize = 1024;

/// A translated shader and what it was translated from. See [`Gpu::shader_cache`].
#[derive(Debug)]
struct CachedShader {
    translation: Translation,
    reads: switch_core::gpu::shader::DecodeReads,
}

/// A device, and the rasterizer to fall back to.
#[derive(Debug)]
pub struct Gpu {
    /// Never read, but must be held: a browser loses the device once the instance is dropped.
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Surfaces this is holding, by guest address.
    held: std::collections::HashMap<u64, Held>,
    /// Per-draw resources, destroyed after submission: a browser only frees dropped resources on GC.
    scratch: Vec<Scratch>,
    /// Surfaces the guest rebound, still owing a write-back.
    evicted: Vec<Held>,
    /// Readbacks asked for and not yet collected.
    pending: Vec<Pending>,
    /// Compiled shader modules, keyed by a hash of the WGSL source.
    modules: std::collections::HashMap<u64, wgpu::ShaderModule>,
    /// Render pipelines, by everything they were built from.
    pipelines: std::collections::HashMap<PipelineKey, wgpu::RenderPipeline>,
    /// Samplers by their settings; `createSampler` is slow in browsers.
    samplers: std::collections::HashMap<SamplerKey, wgpu::Sampler>,
    /// Bind group layouts, by the entries they describe.
    group_layouts:
        std::collections::HashMap<Vec<wgpu::BindGroupLayoutEntry>, wgpu::BindGroupLayout>,
    /// Depth upload pipelines by depth format. See [`LOAD_DEPTH_WGSL`].
    depth_loaders: std::collections::HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    /// Rectangle clear pipelines. See [`CLEAR_RECT_WGSL`].
    clear_pipelines: std::collections::HashMap<ClearKey, wgpu::RenderPipeline>,
    /// Multisample resampling pipelines. See [`resample_wgsl`].
    resample_pipelines: std::collections::HashMap<ResampleKey, wgpu::RenderPipeline>,
    /// `GPU_WEB_LIMITS=1`: hide native-only features so a native run takes the browser's routes.
    web_limits: bool,
    /// `GPU_DEVICE_MSAA=1`: let the device multisample where it can. Off: see [`Gpu::route`].
    device_msaa: bool,
    /// Device rejections, written by the uncaptured-error callback.
    failed: std::sync::Arc<std::sync::Mutex<DeviceErrors>>,
    /// Set with the browser's reason when the device is lost, which is otherwise silent.
    lost: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// Whether a readback has ever completed after the flush that asked for it (always in a browser).
    /// Then a mid-frame fallback would read stale memory, so frames are not interleaved.
    deferred_readbacks: bool,
    /// `GPU_DEFER_READBACKS=1`: never wait for readbacks, to reproduce a browser natively.
    defer_readbacks: bool,
    /// `GPU_INTERLEAVE=1`: interleave fallback draws into device frames even with late readbacks.
    /// Slightly wrong frames for speed; off by default.
    interleave: bool,
    /// Whether the rasterizer has the whole frame. Latches; released after
    /// [`Gpu::clean_frames_needed`] frames whose draws all pass [`Gpu::check`], doubling on each relatch.
    software_frame: bool,
    /// Whether anything fell back during the frame in progress.
    fell_back_this_frame: bool,
    /// Whether a draw of the rasterizer's frame was checked, and whether one would have fallen back.
    checked_this_frame: bool,
    would_fall_back_this_frame: bool,
    /// Rasterizer's frames in a row in which every draw passed the check.
    clean_frames: u32,
    /// Clean frames needed to release the latch; doubles on each relatch.
    clean_frames_needed: u32,
    /// How many times the latch has let go.
    unlatched: u32,
    /// Whether [`Gpu::give_up`] has already handed the frame back.
    gave_up: bool,
    /// The loss reason, reported by the next flush.
    report: Option<String>,
    /// What this cannot express runs here instead.
    software: Software,
    /// Draws rendered here, and draws that fell back.
    pub drawn: u64,
    pub fallbacks: u64,
    pub last_fallback: Option<String>,
    /// Every distinct reason a draw fell back, in the order first seen.
    pub reasons: Vec<String>,
    /// Draws by the route they took. See [`Render`].
    pub direct: u64,
    pub expanded: u64,
    pub multisampled: u64,
    pub per_pixel: u64,
    /// Which draw of the current frame this is; only `GPU_ONLY` reads it.
    in_frame: u32,
    /// Phase timings; always on under wasm, `GPU_TIMES=1` natively.
    times: Option<Times>,
    /// What every draw's `Uploads::of` read, by category.
    uploaded: UploadBytes,
    /// Deswizzled texture bytes, evicted when the guest writes their pages.
    texture_cache: std::collections::HashMap<TextureKey, std::sync::Arc<[u8]>>,
    /// Translated shaders by address and stage.
    shader_cache: std::collections::HashMap<(u64, ShaderStage), CachedShader>,
    /// Which cached translations a guest page holds program words for.
    shader_pages: std::collections::HashMap<u32, Vec<(u64, ShaderStage)>>,
    /// Pages to watch once a mutable `ExecCtx` is available.
    shader_to_watch: Vec<(u32, (u64, ShaderStage))>,
    /// Which cached textures a guest page holds bytes for.
    page_owners: std::collections::HashMap<u32, Vec<TextureKey>>,
    /// Device copies of cached textures; only keys `texture_cache` also holds.
    gpu_textures:
        std::collections::HashMap<TextureKey, Vec<(wgpu::TextureViewDimension, wgpu::Texture)>>,
    cached_bytes: u64,
    pub texture_hits: u64,
    pub texture_misses: u64,
    /// Draws served from [`Gpu::shader_cache`], and draws that missed.
    pub shader_hits: u64,
    pub shader_misses: u64,
    gpu_texture_bytes: u64,
    /// Textures `prepare` read and has not watched yet.
    to_remember: Vec<(TextureKey, std::sync::Arc<[u8]>, u64)>,
    /// `GPU_ONLY=<i>` or `<a>..<b>`: render only those draws of each frame here, for bisecting.
    only: Option<std::ops::Range<u32>>,
}

/// Bounds a run over endless fresh textures; a normal working set is far smaller.
const TEXTURE_CACHE_BYTES: u64 = 256 << 20;
/// Guest pages are 4 KiB, the granularity `Memory` watches writes at.
const PAGE_BITS: u32 = 12;

impl Drop for Gpu {
    fn drop(&mut self) {
        eprintln!(
            "[gpu] {} draws rendered, {} fell back, from {} pipelines and {} modules",
            self.drawn,
            self.fallbacks,
            self.pipelines.len(),
            self.modules.len()
        );
        if self.expanded + self.multisampled + self.per_pixel > 0 {
            eprintln!(
                "[gpu] {} single-sample, {} multisampled by the device, {} expanded, \
                 {} per-pixel coverage",
                self.direct, self.multisampled, self.expanded, self.per_pixel
            );
        }
        let mib = |v: u64| v as f64 / (1024.0 * 1024.0);
        let u = self.uploaded;
        eprintln!(
            "[gpu] read {:.1} MiB of textures, {:.1} MiB of vertices, {:.1} MiB of constants, \
             {:.1} MiB of indices; {} texture reads served from cache, {} not",
            mib(u.textures),
            mib(u.vertex),
            mib(u.constants),
            mib(u.index),
            self.texture_hits,
            self.texture_misses,
        );
        eprintln!(
            "[gpu] {} shader translations served from cache, {} not",
            self.shader_hits, self.shader_misses,
        );
        if let Some(t) = self.times {
            let ms = |v: u128| v as f64 / 1000.0;
            eprintln!(
                "[gpu] translate {:.0}ms  upload {:.0}ms  modules {:.0}ms  \
                 pipeline {:.0}ms  encode {:.0}ms  flush {:.0}ms \
                 (ask {:.0}ms, wait {:.0}ms, land {:.0}ms)",
                ms(t.translate),
                ms(t.upload),
                ms(t.modules),
                ms(t.pipeline),
                ms(t.encode),
                ms(t.flush),
                ms(t.flush_ask),
                ms(t.flush_wait),
                ms(t.flush_land),
            );
        }
    }
}

/// Add `at.elapsed()` to `slot`, if timing is on.
macro_rules! timed {
    ($self:ident, $field:ident, $body:expr) => {{
        if $self.times.is_none() {
            $body
        } else {
            let at = web_time::Instant::now();
            let out = $body;
            if let Some(t) = $self.times.as_mut() {
                t.$field += at.elapsed().as_micros();
            }
            out
        }
    }};
}

mod bind;
mod clear;
mod device;
mod held;
mod multisample;
mod prepare;
mod render;
mod resources;

#[cfg(test)]
mod parity_tests;
#[cfg(test)]
mod tests;

/// How a draw reaches its surfaces; only multisampled surfaces have a choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Render {
    /// Straight into the held surface, whose texels are its pixels.
    Direct,
    /// Into the held surface at texel resolution, with the shader handling mask and alpha-to-coverage.
    Expanded,
    /// Into a companion of this shape, gathered on the way in and scattered on the way out.
    Companion(Shape),
}

/// What a sampler is made of, and so what one is cached by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SamplerKey {
    compare: Option<wgpu::CompareFunction>,
    wrap_u: wgpu::AddressMode,
    wrap_v: wgpu::AddressMode,
    mag: wgpu::FilterMode,
    min: wgpu::FilterMode,
}

/// Whether a texture and a surface at one address share row layout, making the texture its corner.
fn same_rows(surface: &Target, texture: &TextureKey) -> bool {
    let stride = |layout: SurfaceLayout, row_bytes: u32| match layout {
        SurfaceLayout::BlockLinear { .. } => row_bytes.div_ceil(GOB_WIDTH),
        SurfaceLayout::Pitch { pitch } => pitch,
    };
    surface.layout == texture.layout
        && stride(surface.layout, surface.row_bytes) == stride(texture.layout, texture.row_bytes)
}

/// A held surface standing in for one layer of a sampled texture: see [`Gpu::held_layers`].
enum HeldLayer {
    /// Copied texture to texture.
    Colour(wgpu::Texture),
    /// A `depth32float` copied into an `r32float` through a buffer.
    Depth(wgpu::Texture),
    /// A `depth32float` copied whole into its layer of a shadow map.
    Shadow(wgpu::Texture),
    /// A larger `depth32float` whose corner is drawn into a shadow map layer.
    ShadowCorner(wgpu::Texture),
}

/// What [`Gpu::check`] settled; fields mean what [`Prepared`]'s do.
struct Checked {
    state: Pipeline,
    render: Render,
    color: Option<Target>,
    color_scratch: Option<(u32, u32)>,
    depth: Option<Target>,
    vs: Translation,
    fs: Translation,
}

/// One draw, resolved into everything a device needs.
struct Prepared {
    state: Pipeline,
    render: Render,
    /// `None` for a depth-only pass.
    color: Option<Target>,
    /// The scratch texture extent when the colour target is larger than the depth surface.
    color_scratch: Option<(u32, u32)>,
    /// The depth surface, or `None` for a draw that neither tests nor writes depth.
    depth: Option<Target>,
    vs: Translation,
    fs: Translation,
    vs_layout: Layout,
    fs_layout: Layout,
    uploads: Uploads,
    /// The memory each stage's `ldg`s read, by descriptor.
    globals: Vec<GlobalUpload>,
    /// Vertices, indices, or the length of [`Prepared::assembled`].
    count: u32,
    /// Triangle list indices and base vertex for topologies WebGPU lacks, from `raster::assemble`.
    assembled: Option<(Vec<u32>, i32)>,
    /// `gl_InstanceID`, reproduced as the first instance of a one-instance draw.
    instance: u32,
}

impl Renderer for Gpu {
    fn draw(&mut self, engine: &Engine3D, ctx: &mut ExecCtx) -> Result<()> {
        if self.give_up() {
            return self.software.draw(engine, ctx);
        }
        // Any failure runs the draw on the rasterizer.
        let index = self.in_frame;
        self.in_frame += 1;
        if self
            .only
            .as_ref()
            .is_some_and(|only| !only.contains(&index))
        {
            self.flush(ctx)?;
            return self.software.draw(engine, ctx);
        }
        // See [`Gpu::software_frame`].
        if self.software_frame {
            self.check_for_release(engine, ctx);
            self.flush(ctx)?;
            return self.software.draw(engine, ctx);
        }
        self.evict_written(ctx);
        let mut route = None;
        let attempt = match self.prepare(engine, &*ctx) {
            Ok(prepared) => {
                route = Some(prepared.render);
                self.render(&prepared, ctx)
            }
            Err(why) => Err(why),
        };
        // Watch `prepare`'s reads now that `ctx` is mutable.
        self.remember_textures(ctx);
        // See [`Gpu::scratch`].
        self.release_scratch();
        match attempt {
            Ok(()) => {
                self.drawn += 1;
                match route {
                    Some(Render::Direct) => self.direct += 1,
                    Some(Render::Expanded) => self.expanded += 1,
                    Some(Render::Companion(Shape::Multisampled(_))) => self.multisampled += 1,
                    Some(Render::Companion(Shape::PerPixel)) => self.per_pixel += 1,
                    None => {}
                }
                return Ok(());
            }
            Err(why) => self.fall_back(why),
        }
        // The rasterizer needs guest memory to be current.
        self.flush(ctx)?;
        self.software.draw(engine, ctx)
    }

    fn clear_color(
        &mut self,
        engine: &Engine3D,
        ctx: &mut ExecCtx,
        target: u32,
        layer: u32,
        channels: [bool; 4],
    ) -> Result<()> {
        // A frame starts at its clear.
        self.in_frame = 0;
        if self.deferred_readbacks
            && self.fell_back_this_frame
            && !self.software_frame
            && !self.interleave
        {
            self.software_frame = true;
            if self.unlatched > 0 {
                self.clean_frames_needed = self.clean_frames_needed.saturating_mul(2);
            }
            eprintln!(
                "[gpu] a draw fell back where a readback lands later than the call that \
                 asked for it; the rasterizer has the frames from here until {} in a row \
                 could have been the device's. What it fell back on: {:?}",
                self.clean_frames_needed, self.reasons
            );
        }
        self.fell_back_this_frame = false;
        self.release_if_clean();
        if self.give_up() || self.software_frame {
            self.flush(ctx)?;
            return self
                .software
                .clear_color(engine, ctx, target, layer, channels);
        }
        let attempt = self.clear_color_here(engine, ctx, target, layer, channels);
        self.release_scratch();
        match attempt {
            Ok(()) => Ok(()),
            Err(why) => {
                self.fall_back(why);
                // Write held surfaces back before the rasterizer clears.
                self.flush(ctx)?;
                self.software
                    .clear_color(engine, ctx, target, layer, channels)
            }
        }
    }

    fn clear_depth_stencil(
        &mut self,
        engine: &Engine3D,
        ctx: &mut ExecCtx,
        depth: bool,
        stencil: bool,
    ) -> Result<()> {
        if self.give_up() || self.software_frame {
            self.flush(ctx)?;
            return self
                .software
                .clear_depth_stencil(engine, ctx, depth, stencil);
        }
        // The device holds no stencil, so a stencil clear goes straight to guest memory.
        if stencil {
            self.software
                .clear_depth_stencil(engine, ctx, false, true)?;
        }
        if !depth {
            return Ok(());
        }
        let attempt = self.clear_depth_here(engine, ctx);
        self.release_scratch();
        match attempt {
            Ok(()) => Ok(()),
            Err(why) => {
                self.fall_back(why);
                self.flush(ctx)?;
                self.software.clear_depth_stencil(engine, ctx, true, false)
            }
        }
    }

    fn flush(&mut self, ctx: &mut ExecCtx) -> Result<Flush> {
        let at = self.times.map(|_| web_time::Instant::now());
        let result = self.flush_inner(ctx);
        if let (Some(at), Some(t)) = (at, self.times.as_mut()) {
            t.flush += at.elapsed().as_micros();
        }
        result
    }

    fn lost(&self) -> bool {
        // Only a lost device is fixed by replacing it.
        self.gave_up
    }

    fn report_json(&self) -> String {
        // The `Drop` summary, for the browser.
        let ms = |v: u128| v as f64 / 1000.0;
        // Nested so the two "modules" keys do not collide.
        let times = match self.times {
            Some(t) => format!(
                ",\"times\":{{\"translate\":{:.1},\"upload\":{:.1},\"modules\":{:.1},\
                 \"pipeline\":{:.1},\"encode\":{:.1},\"flush\":{:.1},\
                 \"flushAsk\":{:.1},\"flushWait\":{:.1},\"flushLand\":{:.1}}}",
                ms(t.translate),
                ms(t.upload),
                ms(t.modules),
                ms(t.pipeline),
                ms(t.encode),
                ms(t.flush),
                ms(t.flush_ask),
                ms(t.flush_wait),
                ms(t.flush_land),
            ),
            None => String::new(),
        };
        let reasons: Vec<String> = self.reasons.iter().map(|why| json_string(why)).collect();
        // Device rejections, which fallback counts do not show.
        let (error_count, errors) = self.device_errors();
        let errors: Vec<String> = errors.iter().map(|e| json_string(e)).collect();
        // `held` grows with flush cost.
        format!(
            "{{\"backend\":\"device\",\"drawn\":{},\"fallbacks\":{},\"pipelines\":{},\
             \"modules\":{},\"held\":{},\"evicted\":{},\"pending\":{},\
             \"read\":{{\"textures\":{},\"vertex\":{},\"constants\":{},\"index\":{}}},\
             \"textureHits\":{},\"textureMisses\":{},\
             \"shaderHits\":{},\"shaderMisses\":{},\
             \"softwareFrame\":{},\"unlatched\":{},\"gaveUp\":{},\"lostBecause\":{},\"reasons\":[{}],\
             \"deviceErrorCount\":{},\"deviceErrors\":[{}]{}}}",
            self.drawn,
            self.fallbacks,
            self.pipelines.len(),
            self.modules.len(),
            self.held.len(),
            self.evicted.len(),
            self.pending.len(),
            self.uploaded.textures,
            self.uploaded.vertex,
            self.uploaded.constants,
            self.uploaded.index,
            self.texture_hits,
            self.texture_misses,
            self.shader_hits,
            self.shader_misses,
            self.software_frame,
            self.unlatched,
            self.gave_up,
            self.report
                .as_deref()
                .map_or("null".to_string(), json_string),
            reasons.join(","),
            error_count,
            errors.join(","),
            times,
        )
    }
}
