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
use switch_core::gpu::pipeline::{self as state, AttributeBase, Pipeline};
use switch_core::gpu::renderer::{Flush, Renderer, Software};
use switch_core::gpu::shader::compiled::Compiled;
use switch_core::gpu::shader::wgsl::{self, Coverage, Layout, Stage, Translation};
use switch_core::gpu::surface::{Layout as SurfaceLayout, SampleGrid, GOB_WIDTH};
use switch_core::gpu::texture::TextureSlot;
use switch_core::gpu::upload::{Banks, DepthKind, Target, Targets, TextureKey, Uploads};
use switch_core::{Error, Result};

use builtin::{grid_bytes, resample_wgsl, ResampleKey, CLEAR_RECT_WGSL, LOAD_DEPTH_WGSL};
use convert::{
    blend, compare, depth_texture_format, device_attachment_format, index_format,
    sampled_texture_format, topology, vertex_format, widen, write_mask, Widen,
};
use readback::{Companion, Held, Pending, Scratch, MAP_FAILED, MAP_READY, MAP_WAITING};
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

impl Gpu {
    /// Take a device opened elsewhere (asynchronously, in a browser).
    /// The instance and adapter must be kept: see [`Gpu::_instance`].
    pub fn with_device(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Gpu {
        let failed: std::sync::Arc<std::sync::Mutex<DeviceErrors>> =
            std::sync::Arc::new(std::sync::Mutex::new(DeviceErrors::default()));
        let sink = failed.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            if let Ok(mut slot) = sink.lock() {
                slot.record(e.to_string());
            }
        }));
        let lost: std::sync::Arc<std::sync::Mutex<Option<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let sink = lost.clone();
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut slot) = sink.lock() {
                slot.get_or_insert(format!("{reason:?}: {message}"));
            }
        });
        Gpu {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            held: std::collections::HashMap::new(),
            scratch: Vec::new(),
            evicted: Vec::new(),
            pending: Vec::new(),
            modules: std::collections::HashMap::new(),
            pipelines: std::collections::HashMap::new(),
            samplers: std::collections::HashMap::new(),
            group_layouts: std::collections::HashMap::new(),
            depth_loaders: std::collections::HashMap::new(),
            clear_pipelines: std::collections::HashMap::new(),
            resample_pipelines: std::collections::HashMap::new(),
            failed,
            lost,
            deferred_readbacks: false,
            defer_readbacks: switch_core::env_flag!("GPU_DEFER_READBACKS"),
            interleave: switch_core::env_flag!("GPU_INTERLEAVE"),
            software_frame: false,
            fell_back_this_frame: false,
            checked_this_frame: false,
            would_fall_back_this_frame: false,
            clean_frames: 0,
            clean_frames_needed: 1,
            unlatched: 0,
            gave_up: false,
            web_limits: switch_core::env_flag!("GPU_WEB_LIMITS"),
            device_msaa: switch_core::env_flag!("GPU_DEVICE_MSAA"),
            report: None,
            software: Software,
            drawn: 0,
            fallbacks: 0,
            last_fallback: None,
            reasons: Vec::new(),
            direct: 0,
            expanded: 0,
            multisampled: 0,
            per_pixel: 0,
            in_frame: 0,
            times: (cfg!(target_arch = "wasm32") || switch_core::env_flag!("GPU_TIMES"))
                .then(Times::default),
            uploaded: UploadBytes::default(),
            texture_cache: std::collections::HashMap::new(),
            shader_cache: std::collections::HashMap::new(),
            shader_pages: std::collections::HashMap::new(),
            shader_to_watch: Vec::new(),
            page_owners: std::collections::HashMap::new(),
            gpu_textures: std::collections::HashMap::new(),
            cached_bytes: 0,
            texture_hits: 0,
            shader_hits: 0,
            shader_misses: 0,
            texture_misses: 0,
            gpu_texture_bytes: 0,
            to_remember: Vec::new(),
            only: std::env::var("GPU_ONLY")
                .ok()
                .as_deref()
                .and_then(draw_range),
        }
    }

    /// Open a device by blocking on it, which only a native thread may do.
    pub fn open() -> std::result::Result<Gpu, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| format!("no adapter: {e}"))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&device_descriptor(&adapter)))
                .map_err(|e| format!("no device: {e}"))?;
        Ok(Gpu::with_device(instance, adapter, device, queue))
    }

    pub fn describe(&self) -> String {
        format!("{:?}", self.device.limits().max_texture_dimension_2d)
    }

    /// Release the device; dropping it does not on the web.
    pub fn destroy(&self) {
        self.device.destroy();
    }

    fn fall_back(&mut self, why: String) {
        self.fallbacks += 1;
        self.fell_back_this_frame = true;
        if !self.reasons.contains(&why) {
            eprintln!("[gpu] falling back: {why}");
            self.reasons.push(why.clone());
        }
        self.last_fallback = Some(why);
    }

    /// Hand the frame to the rasterizer for good once the device is lost; answers whether it has been.
    fn give_up(&mut self) -> bool {
        if self.gave_up {
            return true;
        }
        let Some(why) = self.lost.lock().ok().and_then(|slot| slot.clone()) else {
            return false;
        };
        self.gave_up = true;
        let said = format!("the device was lost ({why}); the rasterizer has the frame from here");
        eprintln!("[gpu] {said}");
        self.report = Some(said);
        self.held.clear();
        self.evicted.clear();
        self.pending.clear();
        self.scratch.clear();
        true
    }

    /// Check a draw of the rasterizer's frame to see whether the device could have drawn it.
    fn check_for_release(&mut self, engine: &Engine3D, ctx: &mut ExecCtx) {
        self.evict_written(ctx);
        let verdict = self.check(engine, &*ctx);
        self.remember_textures(ctx);
        self.checked_this_frame = true;
        if verdict.is_err() {
            self.would_fall_back_this_frame = true;
        }
    }

    /// At the clear that ends a rasterizer's frame, count it towards releasing the latch.
    fn release_if_clean(&mut self) {
        let (checked, would_fall_back) = (self.checked_this_frame, self.would_fall_back_this_frame);
        self.checked_this_frame = false;
        self.would_fall_back_this_frame = false;
        if !self.software_frame || !checked {
            return;
        }
        if would_fall_back {
            self.clean_frames = 0;
            return;
        }
        self.clean_frames += 1;
        if self.clean_frames < self.clean_frames_needed {
            return;
        }
        self.software_frame = false;
        self.clean_frames = 0;
        self.unlatched += 1;
        eprintln!(
            "[gpu] every draw of the last {} frame(s) could have run on the device; \
             it has the frames again",
            self.clean_frames_needed
        );
    }

    /// The `GPU_INTERLEAVE` flag. See [`Gpu::interleave`].
    pub fn set_interleave(&mut self, interleave: bool) {
        self.interleave = interleave;
    }

    /// The `GPU_WEB_LIMITS` flag. See [`Gpu::web_limits`].
    pub fn set_web_limits(&mut self, web_limits: bool) {
        self.web_limits = web_limits;
    }

    /// What this device may be asked for; see [`Gpu::web_limits`].
    fn features(&self) -> wgpu::Features {
        let features = self.device.features();
        if self.web_limits {
            features - wgpu::Features::SUBGROUP - wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        } else {
            features
        }
    }

    /// The `GPU_DEVICE_MSAA` flag. See [`Gpu::route`].
    pub fn set_device_msaa(&mut self, device_msaa: bool) {
        self.device_msaa = device_msaa;
    }

    /// The device texture for a surface, uploaded on first use.
    fn hold(&mut self, target: &Target, ctx: &ExecCtx) -> Result<()> {
        match self.held.get(&target.addr) {
            // The same surface as last time.
            Some(held) if held.target == *target => return Ok(()),
            // A different surface at the same address: the guest rebound it; write the old one back later.
            Some(_) => {
                if let Some(held) = self.held.remove(&target.addr) {
                    self.evicted.push(held);
                }
            }
            None => {}
        }
        let texture = self.upload_target(target, ctx)?;
        self.held.insert(
            target.addr,
            Held {
                texture,
                target: *target,
                dirty: false,
                companion: None,
            },
        );
        Ok(())
    }

    /// Write one held surface back into guest memory and stop holding it.
    fn flush_one(&mut self, addr: u64) {
        let Some(held) = self.held.remove(&addr) else {
            return;
        };
        self.ask_for(held);
    }

    /// Ask for a surface back without waiting; [`Gpu::flush`] collects it.
    fn ask_for(&mut self, held: Held) {
        // A companion's contents must be in the surface before it is copied.
        if let Some(companion) = &held.companion {
            let depth = held.target.depth_kind().is_some();
            if let Err(why) = self.resolve_into(&held.texture, companion, depth) {
                self.fall_back(format!("putting a companion surface back: {why}"));
            }
        }
        // A surface nothing drew into is already what guest memory says.
        if held.dirty {
            let pending = self.start_read_back(&held.target, &held.texture);
            self.pending.push(pending);
        }
        // Destroyed rather than dropped, so a browser frees it without waiting for GC.
        held.texture.destroy();
        if let Some(companion) = &held.companion {
            companion.texture.destroy();
        }
    }

    fn upload_target(&mut self, target: &Target, ctx: &ExecCtx) -> Result<wgpu::Texture> {
        if let Some(kind) = target.depth_kind() {
            return self.upload_depth_target(target, kind, ctx);
        }
        let format = device_attachment_format(self.features(), target.format)?;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render target"),
            size: wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let rows = target.read(ctx)?;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rows,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(target.row_bytes),
                rows_per_image: Some(target.rows),
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(texture)
    }

    /// Upload a depth surface via an `r32float` staging image and a pass. See [`LOAD_DEPTH_WGSL`].
    fn upload_depth_target(
        &mut self,
        target: &Target,
        kind: DepthKind,
        ctx: &ExecCtx,
    ) -> Result<wgpu::Texture> {
        let format = depth_texture_format(kind);
        let size = wgpu::Extent3d {
            width: target.width,
            height: target.height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // A cropped target takes the left of each row.
        let values = target.read_depth(ctx)?;
        let surface_texels = (target.row_bytes / target.unit.max(1)) as usize;
        let values = crop_rows(
            values,
            surface_texels,
            target.width as usize,
            kind.unit() as usize,
        );
        // The staging image is always `r32float`.
        let staging = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth upload"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let floats: Vec<u8> = match kind {
            DepthKind::Float32 => values,
            DepthKind::Unorm16 => values
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|v| {
                    let stored = u16::from_le_bytes(*v);
                    (f32::from(stored) / 65535.0).to_le_bytes()
                })
                .collect(),
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &staging,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &floats,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(target.width * 4),
                rows_per_image: Some(target.height),
            },
            size,
        );
        self.load_depth(&texture, &staging, format)?;
        staging.destroy();
        Ok(texture)
    }

    /// Draw `staging` into `texture`'s depth.
    fn load_depth(
        &mut self,
        texture: &wgpu::Texture,
        staging: &wgpu::Texture,
        format: wgpu::TextureFormat,
    ) -> Result<()> {
        for layer in 0..texture.depth_or_array_layers() {
            self.load_depth_layer(texture, layer, staging, layer, format)?;
        }
        Ok(())
    }

    /// One layer at a time: a depth attachment is a single layer.
    fn load_depth_layer(
        &mut self,
        texture: &wgpu::Texture,
        layer: u32,
        staging: &wgpu::Texture,
        staging_layer: u32,
        format: wgpu::TextureFormat,
    ) -> Result<()> {
        let one_layer = |base_array_layer: u32| wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer,
            array_layer_count: Some(1),
            ..Default::default()
        };
        let pipeline = self.depth_loader(format);
        let layout = pipeline.get_bind_group_layout(0);
        let view = staging.create_view(&one_layer(staging_layer));
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth upload"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        let target = texture.create_view(&one_layer(layer));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("depth upload"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth upload"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// The depth loader for this format, built once.
    fn depth_loader(&mut self, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        if let Some(pipeline) = self.depth_loaders.get(&format) {
            return pipeline.clone();
        }
        let (_, module) = self.module("load depth", LOAD_DEPTH_WGSL);
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("load depth"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.depth_loaders.insert(format, pipeline.clone());
        pipeline
    }

    /// Start copying a surface off the device, with rows padded to 256 bytes.
    fn start_read_back(&self, target: &Target, texture: &wgpu::Texture) -> Pending {
        // Device row width, which for depth is not the guest's texel width.
        let row_bytes = match target.depth_kind() {
            Some(kind) => target.width * kind.unit(),
            None => target.row_bytes,
        };
        let aspect = match target.depth_kind() {
            Some(_) => wgpu::TextureAspect::DepthOnly,
            None => wgpu::TextureAspect::All,
        };
        let padded = row_bytes.div_ceil(COPY_ALIGNMENT) * COPY_ALIGNMENT;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(padded) * u64::from(target.rows),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(target.rows),
                },
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        // Asked for, not waited on.
        let state = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(MAP_WAITING));
        let sink = state.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let done = if r.is_ok() { MAP_READY } else { MAP_FAILED };
            sink.store(done, std::sync::atomic::Ordering::Release);
        });
        Pending {
            staging,
            target: *target,
            row_bytes,
            padded,
            state,
        }
    }

    /// Copy a finished readback into guest memory, dropping the row padding.
    fn land(&self, pending: &Pending, ctx: &mut ExecCtx) -> Result<()> {
        let slice = pending.staging.slice(..);
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| Error::Gpu(format!("mapping the readback: {e}")))?;
        let target = &pending.target;
        // Colour writes straight from the padded mapping; depth repacks.
        let outcome = match target.depth_kind() {
            None => target.write_strided(ctx, &mapped, pending.padded),
            Some(kind) => {
                let mut rows = Vec::with_capacity((pending.row_bytes * target.rows) as usize);
                for y in 0..target.rows {
                    let at = (y * pending.padded) as usize;
                    rows.extend_from_slice(&mapped[at..at + pending.row_bytes as usize]);
                }
                Self::land_depth(target, kind, &rows, ctx)
            }
        };
        drop(mapped);
        pending.staging.unmap();
        // Destroyed rather than dropped; see [`Gpu::scratch`].
        pending.staging.destroy();
        outcome
    }

    /// Put a depth readback back, repacked.
    fn land_depth(target: &Target, kind: DepthKind, rows: &[u8], ctx: &mut ExecCtx) -> Result<()> {
        // A cropped depth target holds the left of each row; the rest is preserved.
        let surface_texels = (target.row_bytes / target.unit.max(1)) as usize;
        let window = target.width as usize;
        if window >= surface_texels {
            return target.write_depth(ctx, rows);
        }
        let unit = kind.unit() as usize;
        let mut full = target.read_depth(ctx)?;
        for y in 0..target.rows as usize {
            let from = y * window * unit;
            let to = y * surface_texels * unit;
            let len = window * unit;
            if to + len <= full.len() && from + len <= rows.len() {
                full[to..to + len].copy_from_slice(&rows[from..from + len]);
            }
        }
        target.write_depth(ctx, &full)
    }
}

impl Gpu {
    /// Build (or reuse) the pipeline and run the pass.
    fn render(&mut self, p: &Prepared, ctx: &mut ExecCtx) -> std::result::Result<(), String> {
        let target_format = match p.color {
            Some(color) => Some(
                device_attachment_format(self.features(), color.format)
                    .map_err(|e| format!("{e:?}"))?,
            ),
            None => None,
        };
        // A blend the device cannot do must fall back here; a rejected pipeline silently draws nothing.
        if let Some(format) = target_format {
            let blends = p.state.target.is_some_and(|t| t.blend.is_some());
            if blends && !can_blend(format, self.features()) {
                return Err(format!(
                    "blending into a {format:?} target, which this device cannot blend"
                ));
            }
        }
        let depth_format = p
            .depth
            .and_then(|d| d.depth_kind())
            .map(depth_texture_format);
        // Sample mask and alpha-to-coverage are the device's only on the companion route.
        let multisample = match p.render {
            Render::Companion(Shape::Multisampled(count)) => wgpu::MultisampleState {
                count,
                mask: u64::from(p.state.sample_mask),
                alpha_to_coverage_enabled: p.state.alpha_to_coverage,
            },
            _ => wgpu::MultisampleState::default(),
        };
        if let Some(e) = self.device_error() {
            return Err(format!("the device rejected an earlier draw: {e}"));
        }
        let ((vs_key, vs_module), (fs_key, fs_module)) = timed!(self, modules, {
            let vs_source = wgsl::module(&p.vs, Stage::Vertex, &p.vs_layout);
            let fs_source = wgsl::module(&p.fs, Stage::Fragment, &p.fs_layout);
            match (vs_source, fs_source) {
                (Ok(vs), Ok(fs)) => {
                    if let Ok(dir) = std::env::var("GPU_DUMP_WGSL") {
                        dump_wgsl(&dir, &vs, &fs);
                    }
                    Ok((self.module("vertex", &vs), self.module("fragment", &fs)))
                }
                (Err(e), _) | (_, Err(e)) => Err(format!("module: {e}")),
            }
        })?;

        // Vertex buffers the shader reads; the stride travels with each buffer.
        let mut bound: Vec<Bound> = Vec::new();
        for buffer in &p.state.vertex_buffers {
            let attributes: Vec<wgpu::VertexAttribute> = buffer
                .attributes
                .iter()
                .filter(|a| p.vs_layout.attributes.contains(&(a.location as usize)))
                .map(|a| wgpu::VertexAttribute {
                    format: vertex_format(a.format),
                    offset: u64::from(a.offset),
                    shader_location: a.location,
                })
                .collect();
            if attributes.is_empty() {
                continue;
            }
            let upload = p
                .uploads
                .vertex
                .iter()
                .find(|v| v.array == buffer.index)
                .ok_or("a bound vertex array with no bytes")?;
            // An instanced array uploads only this instance's element, so its stride is zero.
            let stride = match buffer.step {
                state::StepMode::Instance => 0,
                state::StepMode::Vertex => u64::from(buffer.stride),
            };
            // Metal drops a vertex whose stride runs past the end of the buffer.
            let mut bytes = std::borrow::Cow::Borrowed(&upload.bytes[..]);
            let whole = bytes.len().next_multiple_of(stride.max(1) as usize);
            if whole != bytes.len() {
                bytes.to_mut().resize(whole, 0);
            }
            bound.push(Bound {
                buffer: self.buffer("vertex", &bytes, wgpu::BufferUsages::VERTEX),
                attributes,
                stride,
                step: match buffer.step {
                    state::StepMode::Instance => wgpu::VertexStepMode::Instance,
                    state::StepMode::Vertex => wgpu::VertexStepMode::Vertex,
                },
            });
        }
        // Unbound locations read a constant buffer of the two default vectors.
        let fed: Vec<usize> = bound
            .iter()
            .flat_map(|b| b.attributes.iter().map(|a| a.shader_location as usize))
            .collect();
        let unfed: Vec<wgpu::VertexAttribute> = p
            .vs_layout
            .attributes
            .iter()
            .filter(|l| !fed.contains(l))
            .map(|&location| wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: if p.state.fixed_attributes.contains(&(location as u32)) {
                    DEFAULT_ATTRIBUTE
                } else {
                    ABSENT_ATTRIBUTE
                },
                shader_location: location as u32,
            })
            .collect();
        if !unfed.is_empty() {
            bound.push(Bound {
                buffer: self.buffer("defaults", &ATTRIBUTE_DEFAULTS, wgpu::BufferUsages::VERTEX),
                attributes: unfed,
                stride: 0,
                step: wgpu::VertexStepMode::Vertex,
            });
        }
        let layouts: Vec<Option<wgpu::VertexBufferLayout>> = bound
            .iter()
            .map(|b| {
                Some(wgpu::VertexBufferLayout {
                    array_stride: b.stride,
                    step_mode: b.step,
                    attributes: &b.attributes,
                })
            })
            .collect();

        let (vs_group_layout, vs_group) =
            timed!(self, pipeline, self.bind_group(p, ShaderStage::VertexB, 0))?;
        let (fs_group_layout, fs_group) =
            timed!(self, pipeline, self.bind_group(p, ShaderStage::Fragment, 1))?;
        // An assembled topology is always `u32` indices.
        let draw_index_format = match &p.assembled {
            Some(_) => Some(wgpu::IndexFormat::Uint32),
            None => p.uploads.index.as_ref().map(|i| index_format(i.format)),
        };
        // WebGPU requires the strip index format on strip pipelines and forbids it otherwise.
        let strip_index_format = match p.state.topology {
            state::Topology::LineStrip | state::Topology::TriangleStrip => draw_index_format,
            _ => None,
        };
        let key = PipelineKey {
            vs: vs_key,
            fs: fs_key,
            target: target_format,
            depth: depth_format
                .zip(p.state.depth)
                .map(|(format, d)| (format, d.write_enabled, d.compare)),
            samples: multisample.count,
            sample_mask: multisample.mask,
            alpha_to_coverage: multisample.alpha_to_coverage_enabled,
            blend: p.state.target.and_then(|t| t.blend),
            write_mask: p.state.target.map_or([true; 4], |t| t.write_mask),
            topology: p.state.topology,
            strip_index_format,
            front_face: p.state.front_face,
            cull: p.state.cull,
            buffers: bound
                .iter()
                .map(|b| {
                    let attributes = b
                        .attributes
                        .iter()
                        .map(|a| (a.format, a.offset, a.shader_location))
                        .collect();
                    (
                        b.stride as u32,
                        b.step == wgpu::VertexStepMode::Instance,
                        attributes,
                    )
                })
                .collect(),
        };
        if let Some(pipeline) = self.pipelines.get(&key) {
            return self.encode(p, &pipeline.clone(), &vs_group, &fs_group, &bound, ctx);
        }
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("draw"),
                bind_group_layouts: &[Some(&vs_group_layout), Some(&fs_group_layout)],
                immediate_size: 0,
            });
        // Empty for a depth-only pass.
        let colour_targets: Vec<Option<wgpu::ColorTargetState>> = target_format
            .map(|format| {
                Some(wgpu::ColorTargetState {
                    format,
                    blend: p.state.target.and_then(|t| t.blend).map(blend),
                    write_mask: p
                        .state
                        .target
                        .map_or(wgpu::ColorWrites::ALL, |t| write_mask(t.write_mask)),
                })
            })
            .into_iter()
            .collect();
        let descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("draw"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology: topology(p.state.topology),
                strip_index_format,
                front_face: match p.state.front_face {
                    state::FrontFace::Ccw => wgpu::FrontFace::Ccw,
                    state::FrontFace::Cw => wgpu::FrontFace::Cw,
                },
                cull_mode: match p.state.cull {
                    state::Cull::None => None,
                    state::Cull::Front => Some(wgpu::Face::Front),
                    state::Cull::Back => Some(wgpu::Face::Back),
                },
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: key.depth.map(|(format, write_enabled, test)| {
                wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(write_enabled),
                    depth_compare: Some(compare(test)),
                    // Neither renderer tests stencil.
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }
            }),
            multisample,
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &colour_targets,
            }),
            multiview_mask: None,
            cache: None,
        };
        let pipeline = timed!(
            self,
            pipeline,
            self.device.create_render_pipeline(&descriptor)
        );
        self.pipelines.insert(key, pipeline.clone());
        self.encode(p, &pipeline, &vs_group, &fs_group, &bound, ctx)
    }

    /// Record and submit one draw against a built pipeline.
    fn encode(
        &mut self,
        p: &Prepared,
        pipeline: &wgpu::RenderPipeline,
        vs_group: &wgpu::BindGroup,
        fs_group: &wgpu::BindGroup,
        bound: &[Bound],
        ctx: &mut ExecCtx,
    ) -> std::result::Result<(), String> {
        // Held across the frame.
        let colour_view = match p.color {
            Some(color) => {
                self.hold(&color, ctx).map_err(|e| format!("{e:?}"))?;
                Some(self.attachment(p, color.addr, true)?)
            }
            None => None,
        };
        // See [`Prepared::color_scratch`].
        let scratch = match (p.color, p.color_scratch) {
            (Some(color), Some((width, height))) => {
                let held = &self
                    .held
                    .get(&color.addr)
                    .ok_or("the surface was not held")?
                    .texture;
                let size = wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                };
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("colour scratch"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: held.format(),
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("colour scratch in"),
                        });
                encoder.copy_texture_to_texture(
                    held.as_image_copy(),
                    texture.as_image_copy(),
                    size,
                );
                self.queue.submit([encoder.finish()]);
                Some((held.clone(), texture, size))
            }
            _ => None,
        };
        let colour_view = match &scratch {
            Some((_, texture, _)) => {
                Some(texture.create_view(&wgpu::TextureViewDescriptor::default()))
            }
            None => colour_view,
        };
        let depth_view = match p.depth {
            Some(depth) => {
                self.hold(&depth, ctx).map_err(|e| format!("{e:?}"))?;
                // A depth test without writes leaves the surface clean.
                let writes = p.state.depth.is_some_and(|d| d.write_enabled);
                Some(self.attachment(p, depth.addr, writes)?)
            }
            None => None,
        };
        let index = match &p.assembled {
            // An assembled topology is always drawn indexed.
            Some((indices, base)) => {
                let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
                Some((
                    self.buffer("assembled", &bytes, wgpu::BufferUsages::INDEX),
                    wgpu::IndexFormat::Uint32,
                    *base,
                ))
            }
            None => p.uploads.index.as_ref().map(|index| {
                (
                    self.buffer("index", &index.bytes, wgpu::BufferUsages::INDEX),
                    index_format(index.format),
                    -(index.lowest as i32),
                )
            }),
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("draw"),
            });
        {
            let colour: Vec<Option<wgpu::RenderPassColorAttachment>> = colour_view
                .as_ref()
                .map(|view| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Loaded, never cleared: a clear is its own method.
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })
                })
                .into_iter()
                .collect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw"),
                color_attachments: &colour,
                depth_stencil_attachment: depth_view.as_ref().map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, vs_group, &[]);
            pass.set_bind_group(1, fs_group, &[]);
            for (slot, b) in bound.iter().enumerate() {
                pass.set_vertex_buffer(slot as u32, b.buffer.slice(..));
            }
            // In pixels, except on the expanded route where the attachment is in texels.
            let (sx, sy) = match p.render {
                Render::Expanded => (p.state.grid.samples_x, p.state.grid.samples_y),
                _ => (1, 1),
            };
            let viewport = &p.state.viewport;
            pass.set_viewport(
                viewport.x * sx as f32,
                viewport.y * sy as f32,
                viewport.width * sx as f32,
                viewport.height * sy as f32,
                viewport.min_depth.clamp(0.0, 1.0),
                viewport.max_depth.clamp(0.0, 1.0),
            );
            let scissor = p.state.scissor;
            pass.set_scissor_rect(
                scissor.x0 * sx,
                scissor.y0 * sy,
                scissor.x1.saturating_sub(scissor.x0) * sx,
                scissor.y1.saturating_sub(scissor.y0) * sy,
            );
            if let Some(constant) = p.state.target.and_then(|t| t.blend) {
                let _ = constant;
                let [r, g, b, a] = p.state.blend_constant;
                pass.set_blend_constant(wgpu::Color {
                    r: f64::from(r),
                    g: f64::from(g),
                    b: f64::from(b),
                    a: f64::from(a),
                });
            }
            // One instance, numbered so `instance_index` matches `gl_InstanceID`.
            let instances = p.instance..p.instance + 1;
            match &index {
                Some((buffer, format, base)) => {
                    pass.set_index_buffer(buffer.slice(..), *format);
                    // The vertex buffer starts at the draw's lowest index.
                    pass.draw_indexed(0..p.count, *base, instances);
                }
                None => pass.draw(0..p.count, instances),
            }
        }
        if let Some((held, texture, size)) = &scratch {
            encoder.copy_texture_to_texture(texture.as_image_copy(), held.as_image_copy(), *size);
            self.scratch.push(Scratch::Texture(texture.clone()));
        }
        timed!(self, encode, self.queue.submit([encoder.finish()]));
        Ok(())
    }

    /// The view a draw renders one of its surfaces through, with any companion in place.
    fn attachment(
        &mut self,
        p: &Prepared,
        addr: u64,
        writes: bool,
    ) -> std::result::Result<wgpu::TextureView, String> {
        match p.render {
            Render::Companion(shape) => self.companion(addr, shape, p.state.grid)?,
            // Put back whatever a companion holds first.
            Render::Direct | Render::Expanded => self.resolve_companion(addr)?,
        }
        let held = self.held.get_mut(&addr).ok_or("the surface was not held")?;
        held.dirty |= writes;
        let texture = match &held.companion {
            Some(companion) => &companion.texture,
            None => &held.texture,
        };
        Ok(texture.create_view(&wgpu::TextureViewDescriptor::default()))
    }

    /// One stage's bindings: constant banks, and textures with their samplers.
    fn bind_group(
        &mut self,
        p: &Prepared,
        stage: ShaderStage,
        group: u32,
    ) -> std::result::Result<(wgpu::BindGroupLayout, wgpu::BindGroup), String> {
        // The layout's dimensionality, as the module declared it.
        let declared = if stage == ShaderStage::VertexB {
            &p.vs_layout
        } else {
            &p.fs_layout
        };
        let mut entries = Vec::new();
        let mut resources: Vec<Resource> = Vec::new();

        for upload in p.uploads.constants.iter().filter(|c| c.stage == stage) {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: upload.bank,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            resources.push(Resource::Buffer(
                upload.bank,
                self.buffer("constants", &upload.bytes, wgpu::BufferUsages::STORAGE),
            ));
        }

        for (index, upload) in p
            .uploads
            .textures
            .iter()
            .filter(|t| t.stage == stage)
            .enumerate()
        {
            use switch_core::gpu::shader::isa::TexDim;
            let declared_texture = declared
                .textures
                .iter()
                .find(|b| b.slot == upload.slot)
                .ok_or("a texture the module never declared")?;
            let compare = declared_texture.compare;
            let dim = declared_texture.dim;
            let view_dimension = match dim {
                TexDim::T2dArray => wgpu::TextureViewDimension::D2Array,
                TexDim::T3d => wgpu::TextureViewDimension::D3,
                // Six faces of a 2D texture, viewed as one cube.
                TexDim::TCube => wgpu::TextureViewDimension::Cube,
                // Six faces to a cube, as many cubes as the array holds.
                TexDim::TCubeArray => wgpu::TextureViewDimension::CubeArray,
                _ => wgpu::TextureViewDimension::D2,
            };
            let binding = TEXTURE_BINDING + 2 * index as u32;
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: if compare {
                        wgpu::TextureSampleType::Depth
                    } else {
                        wgpu::TextureSampleType::Float { filterable: true }
                    },
                    view_dimension,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding + 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(if compare {
                    wgpu::SamplerBindingType::Comparison
                } else {
                    wgpu::SamplerBindingType::Filtering
                }),
                count: None,
            });
            // A size-only query binds as 2D, and the device refuses a 2D view of a layered texture.
            if view_dimension == wgpu::TextureViewDimension::D2 && upload.layers > 1 {
                return Err(format!(
                    "binds a {}-layer texture where the program declared a 2D one",
                    upload.layers
                ));
            }
            let held = self.held_layers(upload, compare, view_dimension)?;
            let texture = if !held.is_empty() {
                self.texture_over_held(upload, view_dimension, &held, compare)?
            } else if compare {
                self.shadow_texture(upload)?
            } else {
                self.texture(upload, view_dimension)?
            };
            resources.push(Resource::Texture(binding, texture, view_dimension));
            resources.push(Resource::Sampler(
                binding + 1,
                self.sampler(upload, compare),
            ));
        }

        for global in p.globals.iter().filter(|g| g.stage == stage) {
            let binding = wgsl::GLOBAL_BINDING + global.slot;
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            resources.push(Resource::Buffer(
                binding,
                self.buffer("global", &global.bytes, wgpu::BufferUsages::STORAGE),
            ));
        }

        let layout = match self.group_layouts.get(&entries) {
            Some(layout) => layout.clone(),
            None => {
                let layout =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("stage"),
                            entries: &entries,
                        });
                self.group_layouts.insert(entries, layout.clone());
                layout
            }
        };
        // The views have to outlive the descriptor that borrows them.
        let views: Vec<Option<wgpu::TextureView>> = resources
            .iter()
            .map(|r| match r {
                Resource::Texture(_, texture, dimension) => {
                    Some(texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(*dimension),
                        ..Default::default()
                    }))
                }
                _ => None,
            })
            .collect();
        let bindings: Vec<wgpu::BindGroupEntry> = resources
            .iter()
            .zip(&views)
            .map(|(resource, view)| match resource {
                Resource::Buffer(binding, buffer) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                },
                Resource::Texture(binding, _, _) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::TextureView(view.as_ref().expect("a view")),
                },
                Resource::Sampler(binding, sampler) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            })
            .collect();
        let group_name = format!("group {group}");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&group_name),
            layout: &layout,
            entries: &bindings,
        });
        Ok((layout, bind_group))
    }

    /// Layers of a sampled texture that are surfaces still held on the device, whose guest
    /// memory is stale. One that cannot be copied is refused.
    fn held_layers(
        &self,
        upload: &switch_core::gpu::upload::TextureUpload,
        compare: bool,
        view: wgpu::TextureViewDimension,
    ) -> std::result::Result<Vec<(u32, HeldLayer)>, String> {
        let mut layers = Vec::new();
        for layer in 0..upload.layers.max(1) {
            let addr = upload.key.addr + u64::from(layer) * u64::from(upload.key.layer_stride);
            let Some(held) = self.held.get(&addr) else {
                continue;
            };
            // A held `ZF32` is copied through a buffer into an `R32` view; a depth texture cannot copy to colour.
            if let Some(depth) = held.target.depth {
                let float =
                    depth.bytes == 4 && depth.depth_bits == 0 && depth.stencil_shift.is_none();
                let (format, _) = sampled_texture_format(self.features(), upload.format)
                    .map_err(|e| format!("{e:?}"))?;
                // A shadow map is copied depth to depth.
                if compare {
                    let (width, height) = (held.texture.width(), held.texture.height());
                    let depth32 = held.texture.format() == wgpu::TextureFormat::Depth32Float;
                    if depth32 && (width, height) == (upload.width, upload.height) {
                        layers.push((layer, HeldLayer::Shadow(held.texture.clone())));
                        continue;
                    }
                    // A padded depth surface sampled at its drawn size goes through the shadow map pass.
                    if depth32
                        && upload.width <= width
                        && upload.height <= height
                        && same_rows(&held.target, &upload.key)
                    {
                        layers.push((layer, HeldLayer::ShadowCorner(held.texture.clone())));
                        continue;
                    }
                    return Err(format!(
                        "samples a depth surface held on the device as a shadow map, \
                         held {:?} {width}x{height} as {:?} and sampled {}x{} with {} \
                         layer(s) as {:?}",
                        held.texture.format(),
                        held.target.layout,
                        upload.width,
                        upload.height,
                        upload.layers,
                        upload.key.layout
                    ));
                }
                let refused = if !float {
                    Some(format!(
                        "packed {depth:?}, which is not the float it would be read as"
                    ))
                } else if format != wgpu::TextureFormat::R32Float
                    || held.texture.format() != wgpu::TextureFormat::Depth32Float
                {
                    Some(format!(
                        "held as {:?} and sampled as {format:?}",
                        held.texture.format()
                    ))
                } else if (held.texture.width(), held.texture.height())
                    != (upload.width, upload.height)
                {
                    Some(format!(
                        "held {}x{} and sampled {}x{}",
                        held.texture.width(),
                        held.texture.height(),
                        upload.width,
                        upload.height
                    ))
                } else {
                    None
                };
                if let Some(why) = refused {
                    return Err(format!("samples a depth surface held on the device, {why}"));
                }
                layers.push((layer, HeldLayer::Depth(held.texture.clone())));
                continue;
            }
            if held.companion.is_some() {
                return Err("samples a multisampled surface held on the device".into());
            }
            // Deep-block volume slices interleave, so none is a surface of its own.
            if view == wgpu::TextureViewDimension::D3 && upload.key.block_depth_gobs > 1 {
                return Err(
                    "samples a surface held on the device as a slice of an interleaved volume"
                        .into(),
                );
            }
            // A texture may be the top-left corner of a larger surface with the same rows.
            let size = held.texture.size();
            if !same_rows(&held.target, &upload.key)
                || upload.width > size.width
                || upload.height > size.height
            {
                return Err(format!(
                    "samples a {}x{} image out of a {}x{} surface held on the device \
                     that lays its rows out differently (layer {layer} of {} at {:#x})",
                    upload.width,
                    upload.height,
                    size.width,
                    size.height,
                    upload.layers,
                    upload.key.addr
                ));
            }
            let (format, widening) = sampled_texture_format(self.features(), upload.format)
                .map_err(|e| format!("{e:?}"))?;
            // A copy may change nothing but whether the format is sRGB.
            if widening != Widen::None
                || format.remove_srgb_suffix() != held.texture.format().remove_srgb_suffix()
            {
                return Err(format!(
                    "samples a {:?} surface held on the device as {format:?}",
                    held.texture.format()
                ));
            }
            layers.push((layer, HeldLayer::Colour(held.texture.clone())));
        }
        Ok(layers)
    }

    /// Upload a texture, then copy the held layers over it; never cached.
    fn texture_over_held(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        view: wgpu::TextureViewDimension,
        held: &[(u32, HeldLayer)],
        compare: bool,
    ) -> std::result::Result<wgpu::Texture, String> {
        let texture = if compare {
            self.shadow_texture(upload)?
        } else {
            let (texture, _) = self.upload_texture(upload, view)?;
            self.scratch.push(Scratch::Texture(texture.clone()));
            texture
        };
        for (layer, surface) in held {
            if let HeldLayer::ShadowCorner(surface) = surface {
                self.draw_shadow_corner(&texture, *layer, surface)?;
            }
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("held layers"),
            });
        let extent = wgpu::Extent3d {
            width: upload.width,
            height: upload.height,
            depth_or_array_layers: 1,
        };
        for (layer, surface) in held {
            let into = wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: *layer,
                },
                aspect: wgpu::TextureAspect::All,
            };
            match surface {
                HeldLayer::Colour(surface) | HeldLayer::Shadow(surface) => {
                    encoder.copy_texture_to_texture(surface.as_image_copy(), into, extent);
                }
                // Drawn above, before these copies were encoded.
                HeldLayer::ShadowCorner(_) => {}
                HeldLayer::Depth(surface) => {
                    // Four bytes a texel, rows padded for a texture-buffer copy.
                    let row = (upload.width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                    let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("held depth"),
                        size: u64::from(row) * u64::from(upload.height),
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                    let layout = wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(upload.height),
                    };
                    encoder.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo {
                            texture: surface,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::DepthOnly,
                        },
                        wgpu::TexelCopyBufferInfo {
                            buffer: &staging,
                            layout,
                        },
                        extent,
                    );
                    encoder.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer: &staging,
                            layout,
                        },
                        into,
                        extent,
                    );
                    self.scratch.push(Scratch::Buffer(staging));
                }
            }
        }
        self.queue.submit([encoder.finish()]);
        Ok(texture)
    }

    /// Draw the top-left corner of a held `depth32float` into a smaller shadow map layer, via `r32float`.
    fn draw_shadow_corner(
        &mut self,
        shadow: &wgpu::Texture,
        layer: u32,
        surface: &wgpu::Texture,
    ) -> std::result::Result<(), String> {
        let size = surface.size();
        let row = (size.width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("held shadow"),
            size: u64::from(row) * u64::from(size.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("held shadow"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row),
            rows_per_image: Some(size.height),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("held shadow"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: surface,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout,
            },
            size,
        );
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout,
            },
            staging.as_image_copy(),
            size,
        );
        self.queue.submit([encoder.finish()]);
        let drawn = self
            .load_depth_layer(
                shadow,
                layer,
                &staging,
                0,
                wgpu::TextureFormat::Depth32Float,
            )
            .map_err(|e| format!("{e:?}"));
        staging.destroy();
        self.scratch.push(Scratch::Buffer(buffer));
        drawn
    }

    fn texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        view: wgpu::TextureViewDimension,
    ) -> std::result::Result<wgpu::Texture, String> {
        if let Some(made) = self.gpu_textures.get(&upload.key) {
            if let Some((_, texture)) = made.iter().find(|(v, _)| *v == view) {
                return Ok(texture.clone());
            }
        }
        let (texture, len) = self.upload_texture(upload, view)?;
        // Kept only alongside its cached bytes, which a guest write evicts.
        if self.texture_cache.contains_key(&upload.key) {
            self.gpu_texture_bytes += len as u64;
            self.gpu_textures
                .entry(upload.key)
                .or_default()
                .push((view, texture.clone()));
        } else {
            self.scratch.push(Scratch::Texture(texture.clone()));
        }
        Ok(texture)
    }

    /// A texture made from an upload's bytes, and how many bytes went into it.
    fn upload_texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        view: wgpu::TextureViewDimension,
    ) -> std::result::Result<(wgpu::Texture, usize), String> {
        let (format, widening) =
            sampled_texture_format(self.features(), upload.format).map_err(|e| format!("{e:?}"))?;
        // Formats the device lacks are widened first. See [`convert::Widen`].
        let widened = (widening != Widen::None).then(|| widen(&upload.bytes, widening));
        let (bytes, row_bytes) = match &widened {
            Some(bytes) => (bytes.as_slice(), upload.row_bytes * 2),
            None => (&*upload.bytes, upload.row_bytes),
        };
        let size = wgpu::Extent3d {
            width: upload.width.max(1),
            height: upload.height.max(1),
            depth_or_array_layers: upload.layers.max(1),
        };
        // A 3D image must be created as one so sampling filters between slices.
        let dimension = match view {
            wgpu::TextureViewDimension::D3 => wgpu::TextureDimension::D3,
            _ => wgpu::TextureDimension::D2,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(upload.rows),
            },
            size,
        );
        Ok((texture, bytes.len()))
    }

    /// The buffers a stage's `ldg`s read, one per descriptor traced back to a constant bank.
    fn global_uploads(
        &self,
        layout: &Layout,
        stage: ShaderStage,
        uploads: &Uploads,
        ctx: &ExecCtx,
    ) -> std::result::Result<Vec<GlobalUpload>, String> {
        let mut out = Vec::new();
        for (slot, &(bank, offset)) in layout.globals.iter().enumerate() {
            let held = uploads
                .constants
                .iter()
                .find(|c| c.stage == stage && c.bank == u32::from(bank))
                .ok_or("a `ldg` descriptor in a bank the draw never bound")?;
            let word = |at: usize| -> Option<u32> {
                held.bytes
                    .get(at..at + 4)
                    .map(|b| u32::from_le_bytes(b.try_into().expect("four bytes")))
            };
            let at = usize::from(offset);
            let (Some(lo), Some(hi)) = (word(at), word(at + 4)) else {
                return Err(format!(
                    "a `ldg` descriptor at c{bank}[{offset:#x}], past the bank's end"
                ));
            };
            let address = (u64::from(hi) << 32) | u64::from(lo);
            if switch_core::trace::enabled(switch_core::trace::Trace::GpuTex) {
                if let Some(h) = self
                    .held
                    .values()
                    .find(|h| (h.target.addr..h.target.addr + h.target.len()).contains(&address))
                {
                    switch_core::traceln!(
                        "[gpu-tex] ldg buffer of {stage:?} at {address:#x} is inside the held \
                         surface at {:#x}",
                        h.target.addr
                    );
                }
            }
            let mapping = ctx.vmm.mapping_at(address).ok_or_else(|| {
                format!("a `ldg` descriptor naming {address:#x}, which is unmapped")
            })?;
            let len = (mapping.gpu_va + mapping.size - address).min(MAX_GLOBAL) as usize;
            let mut bytes = vec![0u8; len];
            ctx.vmm
                .read_into(ctx.mem, address, &mut bytes)
                .map_err(|e| format!("{e:?}"))?;
            out.push(GlobalUpload {
                stage,
                slot: slot as u32,
                bytes,
            });
        }
        Ok(out)
    }

    /// A sampled depth image, drawn through [`Gpu::load_depth`] since it cannot be copied in.
    fn shadow_texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
    ) -> std::result::Result<wgpu::Texture, String> {
        let size = wgpu::Extent3d {
            width: upload.width.max(1),
            height: upload.height.max(1),
            depth_or_array_layers: upload.layers.max(1),
        };
        let format = wgpu::TextureFormat::Depth32Float;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow map"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // Copied into as well as drawn into.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // The staging image `load_depth` reads: `r32float`.
        let depths = self.shadow_depths(upload)?;
        let staging = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow upload"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &staging,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &depths,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width * 4),
                rows_per_image: Some(size.height),
            },
            size,
        );
        self.load_depth(&texture, &staging, format)
            .map_err(|e| format!("{e:?}"))?;
        staging.destroy();
        self.scratch.push(Scratch::Texture(texture.clone()));
        Ok(texture)
    }

    /// One `f32` per texel: the decoded red channel, which a comparison reads.
    fn shadow_depths(
        &self,
        upload: &switch_core::gpu::upload::TextureUpload,
    ) -> std::result::Result<Vec<u8>, String> {
        use switch_core::gpu::pipeline::Format;
        // (bytes per texel, where red starts in one, how wide red is).
        let (unit, at, red) = match upload.format {
            Format::R32Float => return Ok(upload.bytes.to_vec()),
            Format::R16Unorm => (2, 0, Red::Unorm16),
            Format::R16Float => (2, 0, Red::Float16),
            Format::R8Unorm => (1, 0, Red::Unorm8),
            Format::Rg8Unorm => (2, 0, Red::Unorm8),
            Format::Rgba8Unorm => (4, 0, Red::Unorm8),
            Format::Bgra8Unorm => (4, 2, Red::Unorm8),
            Format::Rgba16Float => (8, 0, Red::Float16),
            Format::Rgba32Float => (8 * 2, 0, Red::Float32),
            other => return Err(format!("a shadow map stored as {other:?}")),
        };
        let mut out = Vec::with_capacity(upload.bytes.len() / unit * 4);
        for texel in upload.bytes.chunks_exact(unit) {
            let b = &texel[at..];
            let value = match red {
                Red::Unorm8 => f32::from(b[0]) / 255.0,
                Red::Unorm16 => f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0,
                Red::Float16 => {
                    switch_core::gpu::surface::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
                }
                Red::Float32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            };
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(out)
    }

    /// `compare` is the binding's question, not the descriptor's.
    fn sampler(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        compare: bool,
    ) -> wgpu::Sampler {
        use switch_core::gpu::texture::Wrap;
        let wrap = |w: Wrap| match w {
            Wrap::Repeat => wgpu::AddressMode::Repeat,
            Wrap::Mirror => wgpu::AddressMode::MirrorRepeat,
            Wrap::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            // WebGPU has no border mode, and the rasterizer also samples border as edge.
            Wrap::ClampToBorder => wgpu::AddressMode::ClampToEdge,
        };
        let filter = |linear: bool| {
            if linear {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            }
        };
        use switch_core::gpu::texture::Compare;
        let compare = compare
            .then(|| upload.sampler.compare.unwrap_or(Compare::Always))
            .map(|c| match c {
                Compare::Never => wgpu::CompareFunction::Never,
                Compare::Less => wgpu::CompareFunction::Less,
                Compare::Equal => wgpu::CompareFunction::Equal,
                Compare::LessEqual => wgpu::CompareFunction::LessEqual,
                Compare::Greater => wgpu::CompareFunction::Greater,
                Compare::NotEqual => wgpu::CompareFunction::NotEqual,
                Compare::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
                Compare::Always => wgpu::CompareFunction::Always,
            });
        let key = SamplerKey {
            compare,
            wrap_u: wrap(upload.sampler.wrap_u),
            wrap_v: wrap(upload.sampler.wrap_v),
            mag: filter(upload.sampler.mag_linear),
            min: filter(upload.sampler.min_linear),
        };
        let device = &self.device;
        self.samplers
            .entry(key)
            .or_insert_with(|| {
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("sampler"),
                    compare: key.compare,
                    address_mode_u: key.wrap_u,
                    address_mode_v: key.wrap_v,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: key.mag,
                    min_filter: key.min,
                    ..Default::default()
                })
            })
            .clone()
    }

    /// The module for this WGSL and its cache key; rejections arrive via the error handler.
    fn module(&mut self, what: &str, source: &str) -> (u64, wgpu::ShaderModule) {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        let key = hasher.finish();
        if let Some(module) = self.modules.get(&key) {
            return (key, module.clone());
        }
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(what),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        self.modules.insert(key, module.clone());
        (key, module)
    }

    /// Whatever the device rejected since this was last asked.
    fn device_error(&self) -> Option<String> {
        self.failed.lock().ok().and_then(|mut e| e.fresh.take())
    }

    /// Every distinct rejection and the total count, without draining.
    fn device_errors(&self) -> (u64, Vec<String>) {
        match self.failed.lock() {
            Ok(e) => (e.count, e.distinct.clone()),
            Err(_) => (0, Vec::new()),
        }
    }

    /// A buffer for the draw in progress; see [`Gpu::scratch`].
    fn buffer(&mut self, what: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        // Padded to four bytes.
        let mut padded = bytes.to_vec();
        while !padded.len().is_multiple_of(4) || padded.is_empty() {
            padded.push(0);
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(what),
            size: padded.len() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, &padded);
        self.scratch.push(Scratch::Buffer(buffer.clone()));
        buffer
    }

    /// Decide how a draw reaches its surfaces. The expanded route is the default because
    /// it reproduces Maxwell's texel-centre sample positions exactly.
    fn route(
        &self,
        state: &Pipeline,
        color: Option<Target>,
        depth: Option<Target>,
    ) -> std::result::Result<Render, String> {
        if state.grid.is_single() {
            return Ok(Render::Direct);
        }
        if state.per_pixel_coverage {
            // Every texel of a pixel's tile takes the same value, so a partial mask cannot apply.
            let all = (1u64 << state.samples) - 1;
            if u64::from(state.sample_mask) & all != all || state.alpha_to_coverage {
                return Err("a draw with coverage per pixel and a mask that is per sample".into());
            }
            return Ok(Render::Companion(Shape::PerPixel));
        }
        let formats = [
            match color {
                Some(color) => Some(
                    device_attachment_format(self.features(), color.format)
                        .map_err(|e| format!("{e:?}"))?,
                ),
                None => None,
            },
            depth.and_then(|d| d.depth_kind()).map(depth_texture_format),
        ];
        // All attachments of a pass share one sample count.
        let offered = self.device_msaa
            && formats
                .into_iter()
                .flatten()
                .all(|format| self.samples_supported(format, state.samples));
        if offered {
            return Ok(Render::Companion(Shape::Multisampled(state.samples)));
        }
        // Samples moved off texel centres cannot be expressed by either route.
        if !state.grid.samples_at_texel_centres() {
            return Err("a draw with programmed sample locations".into());
        }
        Ok(Render::Expanded)
    }

    /// Whether this adapter will render `samples` samples into `format`.
    fn samples_supported(&self, format: wgpu::TextureFormat, samples: u32) -> bool {
        // The adapter's answer applies only with `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`.
        let features = self.device.features();
        let flags = if features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            self._adapter.get_texture_format_features(format).flags
        } else {
            format.guaranteed_format_features(features).flags
        };
        flags.sample_count_supported(samples)
    }

    /// Give the surface at `addr` a companion of `shape`, resolving any previous one first.
    fn companion(
        &mut self,
        addr: u64,
        shape: Shape,
        grid: SampleGrid,
    ) -> std::result::Result<(), String> {
        match self.held.get(&addr).and_then(|h| h.companion.as_ref()) {
            Some(have) if have.shape == shape && have.grid == grid => return Ok(()),
            Some(_) => self.resolve_companion(addr)?,
            None => {}
        }
        let held = self.held.get(&addr).ok_or("the surface was not held")?;
        let (width, height) = grid.pixels(held.target.width, held.target.height);
        let samples = match shape {
            Shape::Multisampled(n) => n,
            Shape::PerPixel => 1,
        };
        let depth = held.target.depth_kind();
        let format = held.texture.format();
        let companion = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("companion"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            // A multisampled texture accepts no copy usage.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // Fill the companion from the surface.
        let source = self
            .held
            .get(&addr)
            .ok_or("the surface was not held")?
            .texture
            .clone();
        self.resample(
            &companion,
            &source,
            ResampleKey {
                entry: if samples > 1 {
                    "fs_gather"
                } else {
                    "fs_gather_flat"
                },
                dst: format,
                samples,
                ms_source: false,
                depth: depth.is_some(),
            },
            grid,
        )?;
        let held = self.held.get_mut(&addr).ok_or("the surface was not held")?;
        held.companion = Some(Companion {
            shape,
            texture: companion,
            grid,
        });
        Ok(())
    }

    /// Drop the companion without resolving it, for a caller about to overwrite the surface.
    fn discard_companion(&mut self, addr: u64) {
        if let Some(held) = self.held.get_mut(&addr) {
            if let Some(companion) = held.companion.take() {
                companion.texture.destroy();
            }
        }
    }

    /// Resolve the companion back into the surface and drop it.
    fn resolve_companion(&mut self, addr: u64) -> std::result::Result<(), String> {
        let Some(held) = self.held.get_mut(&addr) else {
            return Ok(());
        };
        let Some(companion) = held.companion.take() else {
            return Ok(());
        };
        let surface = held.texture.clone();
        let depth = held.target.depth_kind().is_some();
        self.resolve_into(&surface, &companion, depth)
    }

    /// Scatter a companion back into the expanded surface it stands in for.
    fn resolve_into(
        &mut self,
        surface: &wgpu::Texture,
        companion: &Companion,
        depth: bool,
    ) -> std::result::Result<(), String> {
        self.resample(
            surface,
            &companion.texture,
            ResampleKey {
                entry: "fs_scatter",
                dst: surface.format(),
                samples: 1,
                ms_source: matches!(companion.shape, Shape::Multisampled(_)),
                depth,
            },
            companion.grid,
        )?;
        companion.texture.destroy();
        Ok(())
    }

    /// Run one resampling pass from `src` into `dst`.
    fn resample(
        &mut self,
        dst: &wgpu::Texture,
        src: &wgpu::Texture,
        key: ResampleKey,
        grid: SampleGrid,
    ) -> std::result::Result<(), String> {
        let pipeline = self.resample_pipeline(key)?;
        let layout = pipeline.get_bind_group_layout(0);
        let buffer = self.buffer("grid", &grid_bytes(grid), wgpu::BufferUsages::STORAGE);
        let source = src.create_view(&wgpu::TextureViewDescriptor::default());
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resample"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source),
                },
            ],
        });
        let view = dst.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resample"),
            });
        {
            let colour: Vec<Option<wgpu::RenderPassColorAttachment>> = (!key.depth)
                .then_some({
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Every texel of the destination is written.
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })
                })
                .into_iter()
                .collect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("resample"),
                color_attachments: &colour,
                depth_stencil_attachment: key.depth.then_some({
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// The pipeline for one resampling direction, built once per shape.
    fn resample_pipeline(
        &mut self,
        key: ResampleKey,
    ) -> std::result::Result<wgpu::RenderPipeline, String> {
        if let Some(pipeline) = self.resample_pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let (sampled, load) = match (key.depth, key.ms_source, key.entry) {
            (false, false, "fs_scatter") => ("texture_2d<f32>", "textureLoad(src, pixel, 0)"),
            (false, false, _) => ("texture_2d<f32>", "textureLoad(src, texel, 0)"),
            (false, true, _) => (
                "texture_multisampled_2d<f32>",
                "textureLoad(src, pixel, sample)",
            ),
            (true, false, "fs_scatter") => ("texture_depth_2d", "textureLoad(src, pixel, 0)"),
            (true, false, _) => ("texture_depth_2d", "textureLoad(src, texel, 0)"),
            (true, true, _) => (
                "texture_depth_multisampled_2d",
                "textureLoad(src, pixel, sample)",
            ),
        };
        let (_, module) = {
            let source = resample_wgsl(sampled, load, key.depth);
            self.module("resample", &source)
        };
        let targets: Vec<Option<wgpu::ColorTargetState>> = (!key.depth)
            .then_some({
                Some(wgpu::ColorTargetState {
                    format: key.dst,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .into_iter()
            .collect();
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("resample"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: key.depth.then(|| wgpu::DepthStencilState {
                    format: key.dst,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: key.samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(key.entry),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            });
        self.resample_pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// Clear part or all of the surfaces a `ClearBuffers` names; whole clears skip the upload.
    fn clear_on_device(
        &mut self,
        color: Option<(Target, [f32; 4], [bool; 4])>,
        depth: Option<(Target, f32)>,
        rect: state::ScissorRect,
        ctx: &ExecCtx,
    ) -> std::result::Result<(), String> {
        let extent = color.map(|(t, _, _)| t).or(depth.map(|(t, _)| t));
        let Some(extent) = extent else { return Ok(()) };
        let whole = rect.x0 == 0
            && rect.y0 == 0
            && rect.x1 >= extent.width
            && rect.y1 >= extent.height
            && color.is_none_or(|(_, _, channels)| channels.iter().all(|&c| c));
        if rect.x1 <= rect.x0 || rect.y1 <= rect.y0 {
            return Ok(());
        }

        let mut views = Vec::new();
        for (target, blank) in [color.map(|(t, _, _)| t), depth.map(|(t, _)| t)]
            .into_iter()
            .flatten()
            .map(|t| (t, whole))
        {
            // A surface about to be written whole need not be uploaded.
            if blank {
                self.hold_blank(&target).map_err(|e| format!("{e:?}"))?;
                // The companion is about to be overwritten, so drop it.
                self.discard_companion(target.addr);
            } else {
                self.hold(&target, ctx).map_err(|e| format!("{e:?}"))?;
                // A partial clear keeps what the companion holds outside the rectangle.
                self.resolve_companion(target.addr)?;
            }
            let held = self
                .held
                .get_mut(&target.addr)
                .ok_or("the surface was not held")?;
            held.dirty = true;
            views.push(
                held.texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
            );
        }
        let mut view = views.into_iter();
        let colour_view = color.map(|_| view.next().expect("a colour view"));
        let depth_view = depth.map(|_| view.next().expect("a depth view"));

        let key = ClearKey {
            color: match color {
                Some((target, _, _)) => Some(
                    device_attachment_format(self.features(), target.format)
                        .map_err(|e| format!("{e:?}"))?,
                ),
                None => None,
            },
            depth: depth
                .and_then(|(target, _)| target.depth_kind())
                .map(depth_texture_format),
            write_mask: color.map_or([true; 4], |(_, _, channels)| channels),
        };
        // Only a partial clear needs its value in a uniform.
        let uniform = (!whole).then(|| {
            let [r, g, b, a] = color.map_or([0.0; 4], |(_, colour, _)| colour);
            let mut bytes = Vec::new();
            for value in [r, g, b, a, depth.map_or(0.0, |(_, d)| d)] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            // A uniform binding is a multiple of sixteen bytes.
            bytes.resize(32, 0);
            self.buffer("clear", &bytes, wgpu::BufferUsages::UNIFORM)
        });
        let pipeline = match &uniform {
            Some(_) => Some(self.clear_pipeline(key)?),
            None => None,
        };
        let group = pipeline
            .as_ref()
            .zip(uniform.as_ref())
            .map(|(pipeline, buffer)| {
                let layout = pipeline.get_bind_group_layout(0);
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("clear"),
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                })
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("clear"),
            });
        {
            let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = colour_view
                .as_ref()
                .map(|view| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: match (whole, color) {
                                (true, Some((_, [r, g, b, a], _))) => {
                                    wgpu::LoadOp::Clear(wgpu::Color {
                                        r: f64::from(r),
                                        g: f64::from(g),
                                        b: f64::from(b),
                                        a: f64::from(a),
                                    })
                                }
                                _ => wgpu::LoadOp::Load,
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })
                })
                .into_iter()
                .collect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: &attachments,
                depth_stencil_attachment: depth_view.as_ref().map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: match (whole, depth) {
                                (true, Some((_, value))) => wgpu::LoadOp::Clear(value),
                                _ => wgpu::LoadOp::Load,
                            },
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let (Some(pipeline), Some(group)) = (&pipeline, &group) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.set_scissor_rect(rect.x0, rect.y0, rect.x1 - rect.x0, rect.y1 - rect.y0);
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Clear colour target `target`, on the device.
    fn clear_color_here(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
        target: u32,
        layer: u32,
        channels: [bool; 4],
    ) -> std::result::Result<(), String> {
        if layer != 0 {
            // Surfaces are held by address only, so layers are unsupported.
            return Err(format!("a clear of layer {layer}"));
        }
        let slot = engine.render_target_slot(target);
        let Some(surface) = Target::color(engine, slot).map_err(|e| format!("{e:?}"))? else {
            // Nothing bound is nothing to clear, which is what the rasterizer answers too.
            return Ok(());
        };
        let rect = self.clear_texels(engine, &surface)?;
        self.clear_on_device(
            Some((surface, engine.clear_color_value(), channels)),
            None,
            rect,
            ctx,
        )
    }

    /// Clear the depth surface, on the device.
    fn clear_depth_here(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
    ) -> std::result::Result<(), String> {
        let Some(surface) = Target::depth_surface(engine).map_err(|e| format!("{e:?}"))? else {
            return Ok(());
        };
        let rect = self.clear_texels(engine, &surface)?;
        self.clear_on_device(None, Some((surface, engine.clear_depth_value())), rect, ctx)
    }

    /// The clear rectangle in texels: pixels scaled by the sample tile.
    fn clear_texels(
        &self,
        engine: &Engine3D,
        surface: &Target,
    ) -> std::result::Result<state::ScissorRect, String> {
        let grid = engine.sample_grid().map_err(|e| format!("{e:?}"))?;
        let (width, height) = grid.pixels(surface.width, surface.height);
        let rect = engine.clear_rectangle(width, height);
        Ok(state::ScissorRect {
            x0: rect.x0 * grid.samples_x,
            y0: rect.y0 * grid.samples_y,
            x1: rect.x1 * grid.samples_x,
            y1: rect.y1 * grid.samples_y,
        })
    }

    /// Hold a surface without reading it, for a clear that writes every texel.
    fn hold_blank(&mut self, target: &Target) -> Result<()> {
        match self.held.get(&target.addr) {
            // Already held: clear it in place.
            Some(held) if held.target == *target => return Ok(()),
            Some(_) => {
                if let Some(held) = self.held.remove(&target.addr) {
                    self.evicted.push(held);
                }
            }
            None => {}
        }
        let texture = self.blank_target(target)?;
        self.held.insert(
            target.addr,
            Held {
                texture,
                target: *target,
                dirty: false,
                companion: None,
            },
        );
        Ok(())
    }

    /// A device texture for a surface, with nothing in it.
    fn blank_target(&self, target: &Target) -> Result<wgpu::Texture> {
        let (format, usage) = match target.depth_kind() {
            Some(kind) => (
                depth_texture_format(kind),
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            None => (
                device_attachment_format(self.features(), target.format)?,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
        };
        Ok(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cleared target"),
            size: wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        }))
    }

    /// The rectangle clear pipeline for these formats, built once. See [`CLEAR_RECT_WGSL`].
    fn clear_pipeline(
        &mut self,
        key: ClearKey,
    ) -> std::result::Result<wgpu::RenderPipeline, String> {
        if let Some(pipeline) = self.clear_pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let (_, module) = self.module("clear", CLEAR_RECT_WGSL);
        let targets: Vec<Option<wgpu::ColorTargetState>> = key
            .color
            .map(|format| {
                Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: write_mask(key.write_mask),
                })
            })
            .into_iter()
            .collect();
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("clear"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: key.depth.map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(if key.color.is_some() {
                        "fs_color"
                    } else {
                        "fs_depth"
                    }),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            });
        self.clear_pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// Drop every cached texture the guest has written over.
    fn evict_written(&mut self, ctx: &mut ExecCtx) {
        if !ctx.mem.has_dirty_gpu() {
            return;
        }
        let dirty = ctx.mem.dirty_gpu_pages();
        self.evict_shaders(&dirty);
        for page in dirty {
            let Some(keys) = self.page_owners.remove(&page) else {
                continue;
            };
            for key in keys {
                if let Some(bytes) = self.texture_cache.remove(&key) {
                    self.cached_bytes -= bytes.len() as u64;
                    self.drop_gpu_texture(&key, bytes.len() as u64);
                }
            }
        }
    }

    /// Destroy a cached texture's device copy.
    fn drop_gpu_texture(&mut self, key: &TextureKey, bytes: u64) {
        if let Some(made) = self.gpu_textures.remove(key) {
            for (_, texture) in made {
                texture.destroy();
                self.gpu_texture_bytes = self.gpu_texture_bytes.saturating_sub(bytes);
            }
        }
    }

    /// Keep what this draw read and watch its pages; textures with no watchable pages are not kept.
    fn remember_textures(&mut self, ctx: &mut ExecCtx) {
        for (page, key) in std::mem::take(&mut self.shader_to_watch) {
            ctx.mem.mark_gpu_page(page << PAGE_BITS);
            self.shader_pages.entry(page).or_default().push(key);
        }
        for (key, bytes, source_len) in std::mem::take(&mut self.to_remember) {
            if self.texture_cache.contains_key(&key) {
                continue;
            }
            self.texture_misses += 1;
            // `source_len` is an upper bound that can exceed the mapping, so stop where the mapping does.
            let end = key.addr.saturating_add(source_len);
            let mut pages: Vec<u32> = Vec::new();
            let mut at = key.addr;
            while at < end {
                let Some((cpu, run)) = ctx.vmm.translate(at) else {
                    break;
                };
                let take = run.min(end - at);
                if take == 0 {
                    break;
                }
                let first = u64::from(cpu) >> PAGE_BITS;
                let last = (u64::from(cpu) + take - 1) >> PAGE_BITS;
                pages.extend((first..=last).map(|p| p as u32));
                at += take;
            }
            if pages.is_empty() {
                continue;
            }
            // Whole-cache eviction: reaching the limit means the textures changed wholesale.
            let len = bytes.len() as u64;
            if self.cached_bytes + len > TEXTURE_CACHE_BYTES {
                for (_, made) in self.gpu_textures.drain() {
                    for (_, texture) in made {
                        texture.destroy();
                    }
                }
                self.gpu_texture_bytes = 0;
                self.texture_cache.clear();
                self.page_owners.clear();
                self.cached_bytes = 0;
            }
            for &page in &pages {
                ctx.mem.mark_gpu_page(page << PAGE_BITS);
                self.page_owners.entry(page).or_default().push(key);
            }
            self.texture_cache.insert(key, bytes);
            self.cached_bytes += len;
        }
    }

    fn release_scratch(&mut self) {
        for made in self.scratch.drain(..) {
            match made {
                Scratch::Buffer(buffer) => buffer.destroy(),
                Scratch::Texture(texture) => texture.destroy(),
            }
        }
    }
}

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

impl Gpu {
    /// Decide whether a draw can run here, without uploading anything.
    fn check(&mut self, engine: &Engine3D, ctx: &ExecCtx) -> std::result::Result<Checked, String> {
        let mut state = Pipeline::of(engine).map_err(|e| e.to_string())?;
        let targets = Targets::of(engine).map_err(|e| format!("{e:?}"))?;
        let color = targets.color;
        // Attach depth only for draws that test or write it.
        let uses_depth = state
            .depth
            .is_some_and(|d| d.write_enabled || d.compare != state::Compare::Always);
        let depth = targets.depth.filter(|_| uses_depth);
        // Attachments must match in size, so the pass covers the intersection. A larger depth surface is
        // cropped; a larger colour target draws into a scratch texture copied back afterwards.
        let mut depth = depth;
        let mut color_scratch = None;
        if let (Some(full_color), Some(full_depth)) = (color, depth) {
            let (cw, ch) = (full_color.width, full_color.height);
            let (dw, dh) = (full_depth.width, full_depth.height);
            if dw >= cw && dh >= ch {
                depth = Some(Target {
                    width: cw,
                    height: ch,
                    rows: ch,
                    ..full_depth
                });
            } else if dw <= cw && dh <= ch {
                color_scratch = Some((dw, dh));
                // The scissor is confined to the depth surface only for draws that reach it.
                let (pixels_x, pixels_y) = state.grid.pixels(dw, dh);
                state.scissor.x1 = state.scissor.x1.min(pixels_x);
                state.scissor.y1 = state.scissor.y1.min(pixels_y);
                state.scissor.x0 = state.scissor.x0.min(state.scissor.x1);
                state.scissor.y0 = state.scissor.y0.min(state.scissor.y1);
            } else {
                return Err(format!(
                    "a {cw}x{ch} colour target beside a {dw}x{dh} depth one, each larger one way"
                ));
            }
        }
        if color.is_none() && depth.is_none() {
            return Err("a draw into neither a colour nor a depth surface".into());
        }
        let render = self.route(&state, color, depth)?;
        if color_scratch.is_some() && matches!(render, Render::Companion(_)) {
            return Err(
                "a colour target larger than its depth surface, drawn through a multisample companion"
                    .into(),
            );
        }

        // Unfolded, so a module depends only on the shader binary.
        let vs = timed!(
            self,
            translate,
            self.translate(engine, ctx, ShaderStage::VertexB)
        );
        let fs = timed!(
            self,
            translate,
            self.translate(engine, ctx, ShaderStage::Fragment)
        );
        let (vs, fs) = (vs?, fs?);
        Ok(Checked {
            state,
            render,
            color,
            color_scratch,
            depth,
            vs,
            fs,
        })
    }

    fn prepare(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
    ) -> std::result::Result<Prepared, String> {
        let Checked {
            state,
            render,
            color,
            color_scratch,
            depth,
            vs,
            fs,
        } = self.check(engine, ctx)?;

        let mut vs_layout = Layout::of(&vs, Stage::Vertex);
        let mut fs_layout = Layout::of(&fs, Stage::Fragment);
        // A depth-only pass has no colour output.
        fs_layout.targets = u32::from(color.is_some());
        // Both stages must name the same varyings; missing ones read as zero.
        let mut varyings = vs_layout.varyings.clone();
        varyings.extend(fs_layout.varyings.iter().copied());
        varyings.sort_unstable();
        varyings.dedup();
        vs_layout.varyings = varyings.clone();
        fs_layout.varyings = varyings;
        // Only the fragment program says which varyings are centroid.
        vs_layout.centroid_varyings = fs_layout.centroid_varyings.clone();
        // Negated because WebGPU mirrors y itself. See `Layout::flip_y`.
        vs_layout.flip_y = !state.viewport.flip_y;
        vs_layout.depth_minus_one_to_one = state.viewport.depth_minus_one_to_one();
        // On the expanded route the shader applies the sample mask and alpha-to-coverage.
        if render == Render::Expanded {
            fs_layout.coverage = Some(Coverage {
                samples_x: state.grid.samples_x,
                samples_y: state.grid.samples_y,
                sample_of_slot: state.grid.sample_of_slot()[..state.samples as usize].to_vec(),
                sample_mask: state.sample_mask,
                alpha_to_coverage: state.alpha_to_coverage,
            });
        }
        // Integer attributes come from the draw's registers.
        vs_layout.integer_attributes = state
            .vertex_buffers
            .iter()
            .flat_map(|buffer| &buffer.attributes)
            .filter(|a| a.format.base() != AttributeBase::Float)
            .map(|a| (a.location as usize, a.format.base()))
            .collect();
        // WebGPU has no BGRA vertex format, so the entry point swaps.
        vs_layout.bgra_attributes = state
            .vertex_buffers
            .iter()
            .flat_map(|buffer| &buffer.attributes)
            .filter(|a| a.is_bgra)
            .map(|a| a.location as usize)
            .collect();
        // And which arrive as one 10-10-10-2 word to unpack.
        vs_layout.packed_attributes = state
            .vertex_buffers
            .iter()
            .flat_map(|buffer| &buffer.attributes)
            .filter_map(|a| match a.format {
                state::VertexFormat::Packed1010102(packing) => Some((a.location as usize, packing)),
                _ => None,
            })
            .collect();
        // One bind group per stage; see `Layout::group`.
        vs_layout.group = 0;
        fs_layout.group = 1;

        let mut slots: Vec<(ShaderStage, TextureSlot)> = Vec::new();
        slots.extend(
            vs.textures
                .iter()
                .map(|&(slot, _, _)| (ShaderStage::VertexB, slot)),
        );
        slots.extend(
            fs.textures
                .iter()
                .map(|&(slot, _, _)| (ShaderStage::Fragment, slot)),
        );
        let mut banks: Vec<(ShaderStage, u32)> = Vec::new();
        banks.extend(
            vs.const_banks
                .iter()
                .map(|&b| (ShaderStage::VertexB, u32::from(b))),
        );
        banks.extend(
            fs.const_banks
                .iter()
                .map(|&b| (ShaderStage::Fragment, u32::from(b))),
        );
        // Taken out of `self` because the closure holds it while `timed!` borrows `self`.
        let cache = std::mem::take(&mut self.texture_cache);
        let mut hits = 0u64;
        let uploads = timed!(self, upload, {
            Uploads::of_cached(
                engine,
                &state,
                ctx,
                Banks::Read(&banks),
                &slots,
                &mut |key| {
                    let hit = cache.get(key).cloned();
                    hits += u64::from(hit.is_some());
                    hit
                },
            )
        });
        self.texture_cache = cache;
        self.texture_hits += hits;
        let uploads = uploads.map_err(|e| format!("{e:?}"))?;
        self.uploaded.add_but_textures(&uploads);
        for upload in &uploads.textures {
            if !self.texture_cache.contains_key(&upload.key) {
                self.uploaded.add_texture(upload.bytes.len());
                self.to_remember
                    .push((upload.key, upload.bytes.clone(), upload.source_len));
            }
        }

        // The texture swizzle is in the descriptor, and WebGPU has no per-texture swizzle.
        for (layout, stage) in [
            (&mut vs_layout, ShaderStage::VertexB),
            (&mut fs_layout, ShaderStage::Fragment),
        ] {
            for binding in &mut layout.textures {
                if let Some(upload) = uploads
                    .textures
                    .iter()
                    .find(|t| t.stage == stage && t.slot == binding.slot)
                {
                    binding.swizzle = upload.swizzle;
                }
            }
        }

        // Fans, quad strips and polygons become triangle lists.
        let assembled = match state.expand {
            Some(primitive) => {
                let triangles =
                    switch_core::gpu::raster::assemble(primitive, engine.last_draw.count);
                let mut indices = Vec::with_capacity(triangles.len() * 3);
                match &uploads.index {
                    // Indexed: triples index the index list; base vertex is the lowest index.
                    Some(index) => {
                        let list = index.indices();
                        for triangle in triangles {
                            for at in triangle {
                                indices.push(*list.get(at as usize).ok_or_else(|| {
                                    format!("assembling {primitive:?}: index {at} is past the list")
                                })?);
                            }
                        }
                        Some((indices, -(index.lowest as i32)))
                    }
                    // Sequential: triples are vertex ordinals, already relative to the upload.
                    None => {
                        for triangle in triangles {
                            indices.extend_from_slice(&triangle);
                        }
                        Some((indices, 0))
                    }
                }
            }
            None => None,
        };

        let mut globals = self.global_uploads(&vs_layout, ShaderStage::VertexB, &uploads, ctx)?;
        globals.extend(self.global_uploads(&fs_layout, ShaderStage::Fragment, &uploads, ctx)?);

        if switch_core::trace::enabled(switch_core::trace::Trace::GpuTex) {
            // Trace what the draw renders into.
            switch_core::traceln!(
                "[gpu-draw] colour={} depth={} state={:?} viewport={:?} scissor={:?} \
                 topology={:?} call={:?} vertex_buffers={} cull={:?} front={:?} buffers={:?}",
                color.map_or("none".to_string(), |c| format!(
                    "{:#x} {:?} {}x{}",
                    c.addr, c.format, c.width, c.height
                )),
                depth.map_or("none".to_string(), |d| format!("{:#x}", d.addr)),
                state.target,
                state.viewport,
                state.scissor,
                state.topology,
                engine.last_draw,
                state.vertex_buffers.len(),
                state.cull,
                state.front_face,
                state.vertex_buffers
            );
            // A buffer over a held surface would read stale memory too.
            let held_at = |addr: u64| {
                self.held
                    .values()
                    .find(|h| (h.target.addr..h.target.addr + h.target.len()).contains(&addr))
                    .map(|h| h.target.addr)
            };
            for c in &uploads.constants {
                if let Some((addr, _)) = engine.bound_constbuf(c.stage, c.bank) {
                    if let Some(surface) = held_at(addr) {
                        switch_core::traceln!(
                            "[gpu-tex] constant bank {} of {:?} at {addr:#x} is inside the held \
                             surface at {surface:#x}",
                            c.bank,
                            c.stage
                        );
                    }
                }
            }
            for t in &uploads.textures {
                switch_core::traceln!(
                    "[gpu-tex] {:?} {:?} {}x{} swizzle={:?} sampler={:?} addr={:#x}{}",
                    t.slot,
                    t.format,
                    t.width,
                    t.height,
                    t.swizzle,
                    t.sampler,
                    t.key.addr,
                    // Sampling a held surface would read stale guest memory.
                    match self.held.values().find(|h| {
                        (h.target.addr..h.target.addr + h.target.len()).contains(&t.key.addr)
                    }) {
                        Some(h) if h.target.addr == t.key.addr => {
                            " (held on the device)".to_string()
                        }
                        Some(h) => format!(" (inside the held surface at {:#x})", h.target.addr),
                        None if self.evicted.iter().any(|h| h.target.addr == t.key.addr) => {
                            " (evicted, not yet written back)".to_string()
                        }
                        None if self.pending.iter().any(|p| p.target.addr == t.key.addr) => {
                            " (being read back)".to_string()
                        }
                        None => String::new(),
                    }
                );
            }
        }
        Ok(Prepared {
            state,
            render,
            color,
            color_scratch,
            depth,
            vs,
            fs,
            vs_layout,
            fs_layout,
            uploads,
            globals,
            count: match &assembled {
                Some((indices, _)) => indices.len() as u32,
                None => engine.last_draw.count,
            },
            assembled,
            instance: engine.instance_id(),
        })
    }

    fn translate(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
        stage: ShaderStage,
    ) -> std::result::Result<Translation, String> {
        let binding = engine
            .program(stage)
            .ok_or_else(|| format!("no {stage:?} program"))?;
        let key = (binding.addr, stage);
        // Validate hits: a `brx` jump table read from a constant buffer may have changed.
        if let Some(cached) = self.shader_cache.get(&key) {
            if cached.reads.constants_unchanged(ctx) {
                self.shader_hits += 1;
                return Ok(cached.translation.clone());
            }
        }
        self.shader_misses += 1;
        let (program, reads) = switch_core::gpu::shader::decode_program_from_memory_recording(
            ctx,
            binding.addr,
            &|bank: u8| engine.bound_constbuf(stage, u32::from(bank)),
        )
        .map_err(|e| format!("{e:?}"))?;
        let caps = wgsl::Caps {
            subgroups: self.features().contains(wgpu::Features::SUBGROUP),
            // A browser wants the directive; naga rejects it.
            subgroup_enable: cfg!(target_arch = "wasm32"),
        };
        let translation =
            wgsl::translate_for(&Compiled::new(&program), caps).map_err(|e| e.to_string())?;
        // Watch the CPU page behind each virtual page the decode read.
        for &page in &reads.pages {
            if let Some((cpu, _)) = ctx.vmm.translate(page << PAGE_BITS) {
                self.shader_to_watch
                    .push(((u64::from(cpu) >> PAGE_BITS) as u32, key));
            }
        }
        // Whole-cache eviction, as for textures.
        if self.shader_cache.len() >= SHADER_CACHE_ENTRIES {
            self.shader_cache.clear();
            self.shader_pages.clear();
        }
        self.shader_cache.insert(
            key,
            CachedShader {
                translation: translation.clone(),
                reads,
            },
        );
        Ok(translation)
    }

    /// Drop every translation a written page held program words for.
    fn evict_shaders(&mut self, pages: &[u32]) {
        for page in pages {
            let Some(keys) = self.shader_pages.remove(page) else {
                continue;
            };
            for key in keys {
                self.shader_cache.remove(&key);
            }
        }
    }
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

impl Gpu {
    fn flush_inner(&mut self, ctx: &mut ExecCtx) -> Result<Flush> {
        // After a loss, report rather than error: a flush also runs inside GPU submissions.
        if self.give_up() {
            return Ok(Flush::Done);
        }
        // Nothing held, owed or in flight: nothing to do (and no poll).
        if self.held.is_empty() && self.evicted.is_empty() && self.pending.is_empty() {
            return Ok(Flush::Done);
        }
        // Only once per frame.
        if self.pending.is_empty() {
            timed!(self, flush_ask, {
                for held in std::mem::take(&mut self.evicted) {
                    self.ask_for(held);
                }
                let addresses: Vec<u64> = self.held.keys().copied().collect();
                for addr in addresses {
                    self.flush_one(addr);
                }
            });
        }
        // `Wait` blocks natively and does nothing on the web, where the present waits for a later slice.
        // `GPU_DEFER_READBACKS=1` skips it to reproduce the browser natively.
        let _ = timed!(self, flush_wait, {
            if self.defer_readbacks {
                self.device.poll(wgpu::PollType::Poll)
            } else {
                self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(5)),
                })
            }
        });
        use std::sync::atomic::Ordering;
        if self
            .pending
            .iter()
            .any(|p| p.state.load(Ordering::Acquire) == MAP_WAITING)
        {
            // A readback outlived its flush. See [`Gpu::deferred_readbacks`].
            self.deferred_readbacks = true;
            return Ok(Flush::Pending);
        }
        timed!(self, flush_land, {
            for pending in std::mem::take(&mut self.pending) {
                if pending.state.load(Ordering::Acquire) == MAP_FAILED {
                    // Include the device's reason if it left one.
                    return Err(Error::Gpu(match self.device_error() {
                        Some(e) => format!("the readback was not mapped: {e}"),
                        None => "the readback was not mapped".into(),
                    }));
                }
                self.land(&pending, ctx)?;
            }
        });
        Ok(Flush::Done)
    }
}

#[cfg(test)]
mod tests {
    /// The device, or `None` with a notice; with `REQUIRE_GPU` set, a panic.
    fn device() -> Option<super::Gpu> {
        match super::Gpu::open() {
            Ok(gpu) => Some(gpu),
            Err(why) if std::env::var_os("REQUIRE_GPU").is_some_and(|v| !v.is_empty()) => {
                panic!("REQUIRE_GPU is set and there is no device: {why}")
            }
            Err(why) => {
                eprintln!("[gpu] skipped: {why}");
                None
            }
        }
    }

    /// naga rejects `shader::wgsl`'s dispatch function without its unreachable trailing
    /// `return false;` (Tint warns about it); this fails once naga stops requiring it.
    #[test]
    fn naga_still_needs_a_return_after_a_loop_that_cannot_fall_through() {
        let Some(gpu) = device() else { return };
        let module = |name: &str, trailing: &str| {
            let src = [
                "fn f() -> bool {",
                "  var pc: u32 = 0u;",
                "  loop {",
                "    switch (pc) {",
                "      case 0u: { return false; }",
                "      default: { return false; }",
                "    }",
                "  }",
                trailing,
                "}",
                "@fragment fn fs_main() -> @location(0) vec4<f32> {",
                "  if (f()) { discard; }",
                "  return vec4<f32>(0.0);",
                "}",
            ]
            .join("\n");
            let _m = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                });
            let _ = gpu.device.poll(wgpu::PollType::Poll);
            gpu.failed.lock().ok().and_then(|mut e| e.fresh.take())
        };
        assert!(
            module("with", "  return false;").is_none(),
            "naga rejected the form `shader::wgsl` actually emits"
        );
        assert!(
            module("without", "").is_some(),
            "naga now accepts a function whose loop cannot fall through: drop the \
             trailing `return false;` from `shader::wgsl`, and Tint stops warning"
        );
    }

    /// The browser's derivative-based quad swap (`Caps::NONE`) passes validation.
    #[test]
    fn the_quad_swap_a_browser_gets_is_wgsl_naga_accepts() {
        use super::{wgpu, wgsl, Compiled, Layout, Stage};
        use switch_core::gpu::shader::isa::{Instruction, Operand, Pred, ShflMode};
        use switch_core::gpu::shader::{Op, Program};

        let Some(gpu) = device() else { return };
        let mut program = Program::default();
        for (index, op) in [
            Op::Shfl {
                dst: 1,
                pred: 0,
                src: 2,
                index: Operand::Imm(1),
                mask: Operand::Imm(0x1c),
                mode: ShflMode::Bfly,
            },
            Op::Exit,
        ]
        .into_iter()
        .enumerate()
        {
            program.insns.push(Instruction {
                pred: Pred::ALWAYS,
                op,
            });
            program.offsets.push(index as u32 * 8);
        }
        let translated = wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE)
            .expect("a shuffle translates without the device's quad operations");
        let layout = Layout::of(&translated, Stage::Fragment);
        let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
        let _ = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quad swap"),
                source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
            });
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
        assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
    }

    /// Tomodachi Life's cube-array `tex` forms and its `vmnmx` translate to valid WGSL.
    #[test]
    fn tomodachi_lifes_cube_array_and_video_min_are_wgsl_naga_accepts() {
        use super::{wgpu, wgsl, Compiled, Layout, Stage};
        use switch_core::gpu::shader::isa::{self, Instruction, Pred};
        use switch_core::gpu::shader::{Op, Program};

        let Some(gpu) = device() else { return };
        let mut program = Program::default();
        let words = [0xc03a0087fff70400, 0xc1ba0087f0970400, 0x3a2c03e060c70907];
        let ops = words.map(|word| isa::decode(word).op);
        for (index, op) in ops.into_iter().chain([Op::Exit]).enumerate() {
            assert!(!matches!(op, Op::Unimplemented { .. }), "{op:?}");
            program.insns.push(Instruction {
                pred: Pred::ALWAYS,
                op,
            });
            program.offsets.push(index as u32 * 8);
        }
        let translated = wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE)
            .expect("a cube-array tex and a vmnmx translate");
        let layout = Layout::of(&translated, Stage::Fragment);
        let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
        assert!(source.contains("texture_cube_array<f32>"), "{source}");
        let _ = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cube array and vmnmx"),
                source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
            });
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
        assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
    }

    #[test]
    fn a_float32_target_blends_only_where_the_device_offers_it() {
        use super::{can_blend, wgpu};
        let none = wgpu::Features::empty();
        let offered = wgpu::Features::FLOAT32_BLENDABLE;
        for format in [
            wgpu::TextureFormat::R32Float,
            wgpu::TextureFormat::Rg32Float,
            wgpu::TextureFormat::Rgba32Float,
        ] {
            assert!(!can_blend(format, none), "{format:?}");
            assert!(can_blend(format, offered), "{format:?}");
        }
        assert!(can_blend(wgpu::TextureFormat::Rgba16Float, none));
        assert!(can_blend(wgpu::TextureFormat::Rgba8Unorm, none));
    }

    /// A 10-10-10-2 colour attribute renders the same on both renderers.
    #[test]
    fn a_10_10_10_2_colour_reaches_the_same_pixels_on_both_renderers() {
        const UNORM: u32 = 2;
        const SNORM: u32 = 1;
        for (ty, word) in [
            // Magenta, opaque: red and blue at their largest, alpha 3 of 3.
            (UNORM, 0x3ff | 0x3ff << 20 | 0b11 << 30),
            // Through snorm: 511 is 1, -512 clamps to -1 then 0, and a two-bit 1 is 1.
            (SNORM, 0x1ff | 0x200 << 10 | 0x1ff << 20 | 0b01 << 30),
        ] {
            let set_up = move |h: &mut Harness| {
                h.triangle([0.0; 4]);
                // Attribute 1, the colour: offset 16, size 0x30.
                h.engine
                    .regs
                    .set(0x458 + 1, (16 << 7) | (0x30 << 21) | (ty << 27));
                let vertices = h.base + 0x400;
                for vertex in 0..3u64 {
                    h.vmm
                        .write_u32(&mut h.mem, vertices + vertex * 32 + 16, word)
                        .unwrap();
                }
            };
            agrees(set_up);
            let mut h = Harness::new();
            set_up(&mut h);
            h.draw_with(&mut Software).expect("the draw");
            assert_eq!(h.texel(1, 1), 0xffff_00ff, "type {ty}: opaque magenta");
        }
    }

    /// A colour through local memory (`st.64 l[RZ + 0x10]` and back) renders the same on both.
    #[test]
    fn a_colour_through_local_memory_reaches_the_same_pixels_on_both_renderers() {
        use switch_core::gpu::shader::isa::{self, MemSize};
        use switch_core::gpu::shader::Op;
        use switch_core::gpu::testing::block;
        const ALWAYS: u64 = 7 << 16;
        const RZ: u64 = 0xff << 8;
        let at = 0x10u64 << 20;
        let stl = ((0xef50u64 | 5) << 48) | at | ALWAYS | RZ;
        let ldl = ((0xef40u64 | 5) << 48) | at | ALWAYS | RZ;
        assert_eq!(
            isa::decode(stl).op,
            Op::Stl {
                addr: 0xff,
                offset: 0x10,
                src: 0,
                size: MemSize::B64
            }
        );
        assert_eq!(
            isa::decode(ldl).op,
            Op::Ldl {
                dst: 0,
                addr: 0xff,
                offset: 0x10,
                size: MemSize::B64
            }
        );
        let split = |w: u64| (w as u32, (w >> 32) as u32);
        let shader = move || {
            let mut bytes = block(
                (0xe1a0070f, 0x00240401),
                (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c]
                (0x00470003, 0x50800000), // mufu rcp $r3 $r0
                (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3
            );
            bytes.extend(block(
                (0xb0400341, 0x055c8400),
                (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3
                (0x8037ff02, 0xe043ff88), // ipa $r2 a[0x88] $r3
                (0xc037ff03, 0xe043ff88), // ipa $r3 a[0x8c] $r3
            ));
            bytes.extend(block(
                (0xffe1ffef, 0x001f8000),
                split(stl),
                split(ldl),
                (0x0007000f, 0xe3000000), // exit
            ));
            bytes
        };
        let set_up = |h: &mut Harness| h.triangle([1.0, 0.0, 1.0, 1.0]);
        agrees_shading(
            move || Harness::with_fragment_shader(shader()),
            |_| {},
            set_up,
        );
        let mut h = Harness::with_fragment_shader(shader());
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        assert_eq!(h.texel(1, 1), 0xffff_00ff, "red survived the round trip");
    }

    /// A lost device falls back without failing the flush.
    #[test]
    fn a_lost_device_is_reported_rather_than_failing_the_flush() {
        use switch_core::gpu::renderer::Renderer;
        let Some(mut gpu) = device() else { return };
        let mut h = Harness::new();
        h.triangle([1.0, 0.0, 1.0, 1.0]);
        h.draw_with(&mut gpu).expect("the draw");
        *gpu.lost.lock().unwrap() = Some("Out of memory".into());
        // `flush_with` panics on an error.
        h.flush_with(&mut gpu);
        h.flush_with(&mut gpu);
        let json = gpu.report_json();
        assert!(json.contains("\"gaveUp\":true"), "{json}");
        assert!(json.contains("Out of memory"), "{json}");
    }

    /// A rejection the backend never asks about still reaches the report.
    #[test]
    fn a_rejection_nothing_asked_about_is_still_counted_and_reported() {
        let Some(gpu) = device() else { return };
        assert_eq!(gpu.device_errors(), (0, Vec::new()), "nothing rejected yet");

        // `bool` is not a valid fragment output.
        let _m = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rejected"),
                source: wgpu::ShaderSource::Wgsl(
                    "@fragment fn fs_main() -> @location(0) bool { return true; }".into(),
                ),
            });
        let _ = gpu.device.poll(wgpu::PollType::Poll);

        let (count, distinct) = gpu.device_errors();
        assert!(
            count >= 1,
            "the device rejected the module and said nothing"
        );
        assert_eq!(distinct.len(), 1, "one rejection, one distinct message");
        // Not drained.
        assert_eq!(gpu.device_errors(), (count, distinct.clone()));

        use switch_core::gpu::renderer::Renderer;
        let json = gpu.report_json();
        assert!(
            json.contains(&format!("\"deviceErrorCount\":{count}")),
            "the count is missing from the report: {json}"
        );
        assert!(
            json.contains("\"deviceErrors\":[\""),
            "the message is missing from the report: {json}"
        );
    }

    /// Every internal pipeline this backend builds compiles.
    use switch_core::gpu::renderer::Software;
    use switch_core::gpu::testing::{self, Harness};

    /// A solid white triangle over a multisample mode, by both renderers; only coverage can differ.
    fn compare(mode: u32, samples_x: u32, samples_y: u32, set_up: impl Fn(&mut Harness)) {
        let Some(mut gpu) = device() else { return };
        let colour = [1.0f32, 1.0, 1.0, 1.0];

        let build = |gpu: Option<&mut super::Gpu>| {
            let mut h = Harness::new();
            h.multisample(mode, samples_x, samples_y);
            set_up(&mut h);
            h.triangle(colour);
            match gpu {
                Some(gpu) => {
                    h.draw_with(gpu).expect("the draw");
                    h.flush_with(gpu);
                }
                None => h.draw_with(&mut Software).expect("the draw"),
            }
            h.target()
        };
        let want = build(None);
        let before = gpu.fallbacks;
        let got = build(Some(&mut gpu));
        assert_eq!(
            gpu.fallbacks, before,
            "the draw did not run on the device: {:?}",
            gpu.last_fallback
        );
        // Surface any device rejection before comparing.
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert_eq!(gpu.device_error(), None, "the device rejected the pass");
        assert_eq!(
            got, want,
            "mode {mode} ({samples_x}x{samples_y}) came out differently on the device"
        );
    }

    /// One draw, rendered by both renderers, colour and depth.
    fn agrees(set_up: impl Fn(&mut Harness)) {
        agrees_shading(Harness::new, |_| {}, set_up);
    }

    /// [`agrees`] with a custom fragment shader, and `tune` applied to the device.
    fn agrees_shading(
        new: impl Fn() -> Harness,
        tune: impl Fn(&mut super::Gpu),
        set_up: impl Fn(&mut Harness),
    ) {
        let Some(mut gpu) = device() else { return };
        tune(&mut gpu);
        let build = |gpu: Option<&mut super::Gpu>| {
            let mut h = new();
            set_up(&mut h);
            match gpu {
                Some(gpu) => {
                    h.draw_with(gpu).expect("the draw");
                    h.flush_with(gpu);
                }
                None => h.draw_with(&mut Software).expect("the draw"),
            }
            (h.target(), h.depth())
        };
        let want = build(None);
        let before = gpu.fallbacks;
        let got = build(Some(&mut gpu));
        assert_eq!(
            gpu.fallbacks, before,
            "the draw did not run on the device: {:?}",
            gpu.last_fallback
        );
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert_eq!(gpu.device_error(), None, "the device rejected the pass");
        assert_eq!(got.0, want.0, "the colour surface differs");
        assert_eq!(got.1, want.1, "the depth surface differs");
    }

    /// A shader reading its quad neighbour's register reads the same on both renderers,
    /// with native quad operations and with the browser's `QUAD_SWAP`.
    #[test]
    fn a_shuffling_fragment_shader_reads_the_same_neighbour_the_rasterizer_reads() {
        for web_limits in [false, true] {
            agrees_shading(
                || Harness::with_fragment_shader(testing::derivative_fragment_shader()),
                move |gpu| gpu.set_web_limits(web_limits),
                |h| {
                    h.depth_target(0x0207);
                    // Red ramps across the target, so neighbours differ by a sixteenth.
                    h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
                    h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
                    h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
                },
            );
        }
    }

    /// A `tld4` gathers the same four texels in the same order on both renderers.
    #[test]
    fn a_gather_reads_the_texels_the_rasterizer_reads() {
        for component in [0, 1] {
            let set_up = |h: &mut Harness| {
                h.bindless_texture();
                // `TexCbIndex`: the bank the bindless fixture writes its handle into.
                h.engine.regs.set(0x982, testing::BINDLESS_HANDLE_BANK);
                h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
                h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
                h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
            };
            let new =
                move || Harness::with_fragment_shader(testing::gather_fragment_shader(component));

            // The reference must be the image's channel, or two empty renders would agree.
            let mut h = new();
            set_up(&mut h);
            h.draw_with(&mut Software).expect("the draw");
            let channel = |texel: u32| (texel >> (8 * component)) & 0xff;
            let size = testing::BINDLESS_TEXTURE_SIZE;
            let values: Vec<u32> = (0..size * size)
                .map(|i| channel(testing::bindless_texel(i % size, i / size)))
                .collect();
            let drawn: Vec<u32> = h.target().into_iter().filter(|&c| c != 0).collect();
            assert!(
                !drawn.is_empty(),
                "component {component}: nothing was drawn"
            );
            for pixel in &drawn {
                for byte in pixel.to_le_bytes() {
                    assert!(
                        values.contains(&u32::from(byte)),
                        "component {component}: {pixel:#010x} holds {byte:#x}, no texel's value"
                    );
                }
            }

            agrees_shading(new, |_| {}, set_up);
        }
    }

    /// Nintendo Switch Sports' `txq` and `tld4` translate to a module the device accepts.
    #[test]
    fn nintendo_switch_sports_txq_and_tld4_are_wgsl_naga_accepts() {
        use super::{wgpu, wgsl, Compiled, Layout, Stage};
        use switch_core::gpu::shader::isa::{self, Instruction, Pred};
        use switch_core::gpu::shader::{Op, Program};

        let Some(gpu) = device() else { return };
        let mut program = Program::default();
        let words = [0xdf48008180470800, 0xc83a0086aff70208];
        let ops = words.map(|word| isa::decode(word).op);
        for (index, op) in ops.into_iter().chain([Op::Exit]).enumerate() {
            assert!(!matches!(op, Op::Unimplemented { .. }), "{op:?}");
            program.insns.push(Instruction {
                pred: Pred::ALWAYS,
                op,
            });
            program.offsets.push(index as u32 * 8);
        }
        let translated = wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE)
            .expect("a txq of a texture the program gathers from translates");
        let layout = Layout::of(&translated, Stage::Fragment);
        let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
        assert!(source.contains("textureGather(0, tex0"), "{source}");
        assert!(source.contains("textureNumLevels(tex0)"), "{source}");
        let _ = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("txq and tld4"),
                source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
            });
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
        assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
    }

    /// Nintendo Switch Sports' `i2i.cc` and `csetp.neu` translate to a module the device accepts.
    #[test]
    fn nintendo_switch_sports_csetp_is_wgsl_naga_accepts() {
        use super::{wgpu, wgsl, Compiled, Layout, Stage};
        use switch_core::gpu::shader::isa::{self, Instruction, Pred};
        use switch_core::gpu::shader::{Op, Program};

        let Some(gpu) = device() else { return };
        let mut program = Program::default();
        let words = [0x5ce0800000170aff, 0x50a0038000070d07];
        let ops = words.map(|word| isa::decode(word).op);
        for (index, op) in ops.into_iter().chain([Op::Exit]).enumerate() {
            assert!(!matches!(op, Op::Unimplemented { .. }), "{op:?}");
            program.insns.push(Instruction {
                pred: Pred::ALWAYS,
                op,
            });
            program.offsets.push(index as u32 * 8);
        }
        let translated = wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE)
            .expect("an i2i.cc and a csetp translate");
        let layout = Layout::of(&translated, Stage::Fragment);
        let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
        assert!(source.contains("var ccZ: bool"), "{source}");
        let _ = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("csetp"),
                source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
            });
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
        assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
    }

    /// A `tex.aoffi` with an immediate offset samples the same texel on both renderers.
    #[test]
    fn a_constant_texel_offset_samples_what_the_rasterizer_samples() {
        let set_up = |h: &mut Harness| {
            h.bindless_texture();
            h.engine.regs.set(0x982, testing::BINDLESS_HANDLE_BANK);
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
            h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
            h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        };
        let new = || Harness::with_fragment_shader(testing::offset_fragment_shader());

        let mut h = new();
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        let size = testing::BINDLESS_TEXTURE_SIZE;
        let texels: Vec<u32> = (0..size * size)
            .map(|i| testing::bindless_texel(i % size, i / size))
            .collect();
        let drawn: std::collections::BTreeSet<u32> =
            h.target().into_iter().filter(|&c| c != 0).collect();
        assert!(drawn.len() >= 8, "only {drawn:x?} was drawn");
        assert!(drawn.iter().all(|c| texels.contains(c)), "{drawn:x?}");

        agrees_shading(new, |_| {}, set_up);
    }

    /// A held `ZF32` surface sampled as a float texture reads the device's depth, not stale memory.
    #[test]
    fn a_held_float_depth_surface_samples_as_the_depth_it_holds() {
        let Some(mut gpu) = device() else { return };
        let build = |gpu: Option<&mut super::Gpu>| {
            let mut h = Harness::with_fragment_shader(testing::bindless_fragment_shader());
            h.bindless_texture();
            h.depth_target(0x0207);
            // ZF32, neither tested nor written, so the draw samples it without attaching it.
            h.engine.regs.set(0x3FA, 0x0A);
            h.engine.regs.set(testing::DEPTH_TEST_ENABLE, 0);
            h.engine.regs.set(testing::DEPTH_WRITE_ENABLE, 0);
            h.engine.regs.set(0x364, 0.25f32.to_bits());
            // The image descriptor, pointed at the depth surface.
            let (tic, depth) = (h.base + 0x1400 + 32, h.base + 0x1000);
            let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
            let mut ctx = h.ctx();
            for (word, value) in [
                (0, 0x2f | (7 << 7) | identity),
                (1, depth as u32),
                (2, (depth >> 32) as u32 | (3 << 21)),
                (3, 0),
                (4, (testing::TARGET_WIDTH - 1) | (1 << 23)),
                (5, testing::TARGET_HEIGHT - 1),
            ] {
                ctx.write_u32(tic + word * 4, value).unwrap();
            }
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
            h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
            h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
            match gpu {
                Some(gpu) => {
                    h.clear_depth_with(gpu).expect("the clear");
                    h.draw_with(gpu).expect("the draw");
                    h.flush_with(gpu);
                }
                None => {
                    h.clear_depth_with(&mut Software).expect("the clear");
                    h.draw_with(&mut Software).expect("the draw");
                }
            }
            h.target()
        };
        let want = build(None);
        assert!(
            want.contains(&0xff00_0040),
            "the reference did not draw the cleared depth: {want:x?}"
        );
        let (fallbacks, drawn) = (gpu.fallbacks, gpu.drawn);
        let got = build(Some(&mut gpu));
        assert_eq!(
            gpu.fallbacks, fallbacks,
            "the draw fell back: {:?}",
            gpu.last_fallback
        );
        assert_eq!(gpu.drawn, drawn + 1, "the draw did not run on the device");
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert_eq!(gpu.device_error(), None, "the device rejected the pass");
        assert_eq!(got, want, "the colour surface differs");
    }

    /// A held depth surface sampled as a shadow map, whole and as a padded surface's corner.
    #[test]
    fn a_held_depth_surface_is_the_shadow_map_the_rasterizer_compares_against() {
        let Some(mut gpu) = device() else { return };
        for width in [testing::TARGET_WIDTH, testing::TARGET_WIDTH / 2] {
            let build = |gpu: Option<&mut super::Gpu>, clear: f32| {
                let mut h = Harness::with_fragment_shader(testing::shadow_fragment_shader());
                h.bindless_texture();
                h.engine.regs.set(0x982, testing::BINDLESS_HANDLE_BANK);
                h.depth_target(0x0207);
                h.engine.regs.set(0x3FA, 0x0A); // ZF32
                h.engine.regs.set(testing::DEPTH_TEST_ENABLE, 0);
                h.engine.regs.set(testing::DEPTH_WRITE_ENABLE, 0);
                h.engine.regs.set(0x364, clear.to_bits());
                let tic = h.base + 0x1400 + 32;
                let tsc = h.base + 0x1480 + 32;
                let depth = h.base + 0x1000;
                let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
                let mut ctx = h.ctx();
                for (word, value) in [
                    (0, 0x2f | (7 << 7) | identity),
                    (1, depth as u32),
                    (2, (depth >> 32) as u32 | (3 << 21)),
                    (3, 0),
                    (4, (width - 1) | (1 << 23)),
                    (5, testing::TARGET_HEIGHT - 1),
                ] {
                    ctx.write_u32(tic + word * 4, value).unwrap();
                }
                // The fixture's sampler, comparing with Less.
                ctx.write_u32(tsc, 2 | (2 << 3) | (2 << 6) | (1 << 9) | (1 << 10))
                    .unwrap();
                h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
                h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
                h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
                match gpu {
                    Some(gpu) => {
                        h.clear_depth_with(gpu).expect("the clear");
                        h.draw_with(gpu).expect("the draw");
                        h.flush_with(gpu);
                    }
                    None => {
                        h.clear_depth_with(&mut Software).expect("the clear");
                        h.draw_with(&mut Software).expect("the draw");
                    }
                }
                h.target()
            };
            // A stale read would compare against zero, a different picture.
            let want = build(None, 0.75);
            assert_ne!(want, build(None, 0.0), "{width} wide: 0.75 reads as 0");
            let (fallbacks, drawn) = (gpu.fallbacks, gpu.drawn);
            let got = build(Some(&mut gpu), 0.75);
            assert_eq!(
                gpu.fallbacks, fallbacks,
                "{width} wide: the draw fell back: {:?}",
                gpu.last_fallback
            );
            assert_eq!(gpu.drawn, drawn + 1, "{width} wide: not on the device");
            let _ = gpu.device.poll(wgpu::PollType::Poll);
            assert_eq!(gpu.device_error(), None, "{width} wide: rejected");
            assert_eq!(got, want, "{width} wide: the colour surface differs");
        }
    }

    /// A texture that is the corner of a held surface is copied from the device.
    #[test]
    fn a_held_surface_stands_in_for_a_smaller_texture_laid_out_the_same_way() {
        let Some(mut gpu) = device() else { return };
        let build = |gpu: Option<&mut super::Gpu>| {
            let mut h = Harness::with_fragment_shader(testing::bindless_fragment_shader());
            h.bindless_texture();
            // The fixture's image descriptor, pointed at the colour target.
            let tic = h.base + 0x1400 + 32;
            let target = h.base;
            let mut ctx = h.ctx();
            ctx.write_u32(tic + 4, target as u32).unwrap();
            ctx.write_u32(tic + 8, (target >> 32) as u32 | (2 << 21))
                .unwrap();
            ctx.write_u32(tic + 12, testing::TARGET_WIDTH * 4 / 32)
                .unwrap();
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
            h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
            h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
            // No half anywhere: see the clear test.
            for (i, value) in [0.0f32, 0.2, 0.6, 1.0].into_iter().enumerate() {
                h.engine.regs.set(0x360 + i as u32, value.to_bits());
            }
            match gpu {
                Some(gpu) => {
                    h.clear_with(gpu, [true; 4]).expect("the clear");
                    h.draw_with(gpu).expect("the draw");
                    h.flush_with(gpu);
                }
                None => {
                    h.clear_with(&mut Software, [true; 4]).expect("the clear");
                    h.draw_with(&mut Software).expect("the draw");
                }
            }
            h.target()
        };
        let want = build(None);
        let (fallbacks, drawn) = (gpu.fallbacks, gpu.drawn);
        let got = build(Some(&mut gpu));
        assert_eq!(
            gpu.fallbacks, fallbacks,
            "the draw fell back: {:?}",
            gpu.last_fallback
        );
        assert_eq!(gpu.drawn, drawn + 1, "the draw did not run on the device");
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert_eq!(gpu.device_error(), None, "the device rejected the pass");
        assert_eq!(got, want, "the colour surface differs");
    }

    /// Culling matches on both renderers whether or not the viewport mirrors y.
    #[test]
    fn culling_keeps_the_faces_the_rasterizer_keeps() {
        const VIEWPORT_TRANSFORM: u32 = 0x280;
        const CULL: u32 = 0x646;
        const FRONT_FACE: u32 = 0x647;
        const CULL_FACE: u32 = 0x648;
        const CW: u32 = 0x900;
        const CCW: u32 = 0x901;
        const BACK: u32 = 0x405;
        let set_up = |mirrored: bool, front: u32| {
            move |h: &mut Harness| {
                h.triangle([0.0, 1.0, 0.0, 1.0]);
                let (w, height) = (
                    testing::TARGET_WIDTH as f32 / 2.0,
                    testing::TARGET_HEIGHT as f32 / 2.0,
                );
                let scale_y = if mirrored { -height } else { height };
                for (i, value) in [w, scale_y, 0.5, w, height, 0.5].into_iter().enumerate() {
                    h.engine
                        .regs
                        .set(VIEWPORT_TRANSFORM + i as u32, value.to_bits());
                }
                h.engine.regs.set(CULL, 1);
                h.engine.regs.set(FRONT_FACE, front);
                h.engine.regs.set(CULL_FACE, BACK);
            }
        };
        for mirrored in [true, false] {
            // One winding keeps the triangle and the other culls it.
            let drawn = [CW, CCW].map(|front| {
                let mut h = Harness::new();
                set_up(mirrored, front)(&mut h);
                h.draw_with(&mut Software).expect("the draw");
                h.target().iter().any(|&c| c != 0)
            });
            assert_ne!(drawn[0], drawn[1], "mirrored={mirrored}: {drawn:?}");
            for front in [CW, CCW] {
                agrees(set_up(mirrored, front));
            }
        }
    }

    /// A bindless `tex.b` resolves its handle from the constant word the shader loaded.
    #[test]
    fn a_bindless_texture_is_the_one_the_rasterizer_samples() {
        // Pixel centres fall a quarter or three quarters into a texel.
        let set_up = |h: &mut Harness| {
            h.bindless_texture();
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
            h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
            h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        };
        let new = || Harness::with_fragment_shader(testing::bindless_fragment_shader());

        // The reference must be the image first.
        let mut h = new();
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        let size = testing::BINDLESS_TEXTURE_SIZE;
        let texels: Vec<u32> = (0..size * size)
            .map(|i| testing::bindless_texel(i % size, i / size))
            .collect();
        let drawn: std::collections::BTreeSet<u32> =
            h.target().into_iter().filter(|&c| c != 0).collect();
        assert!(drawn.len() >= 8, "only {drawn:x?} was drawn");
        assert!(
            drawn.iter().all(|c| texels.contains(c)),
            "{drawn:x?} is not all texels"
        );

        agrees_shading(new, |_| {}, set_up);
    }

    #[test]
    fn a_depth_tested_draw_writes_the_same_depth_the_rasterizer_writes() {
        // The depth functions titles use. Colours are ones and zeros since `mufu rcp` and
        // WGSL division differ by a rounding step.
        for (func, passes) in [
            (0x0201, false),
            (0x0203, false),
            (0x0204, true),
            (0x0207, true),
        ] {
            let set_up = move |h: &mut Harness| {
                h.depth_target(func);
                h.triangle([1.0, 0.0, 1.0, 1.0]);
            };
            agrees(set_up);
            // The surface starts at 0 and the triangle sits at 0.5 (Z24 0x800000).
            let mut h = Harness::new();
            set_up(&mut h);
            h.draw_with(&mut Software).expect("the draw");
            let (colour, depth) = if passes {
                (0xffff_00ff, 0x8000_0000)
            } else {
                (0, 0)
            };
            assert_eq!(h.texel(1, 1), colour, "func {func:#x}: colour");
            assert_eq!(h.depth()[17], depth, "func {func:#x}: depth");
        }
    }

    /// A depth surface smaller than the colour target confines the draw on both renderers.
    #[test]
    fn a_depth_surface_smaller_than_the_colour_target_confines_the_draw() {
        // `Always` with writes on still reaches the depth surface.
        let set_up = |h: &mut Harness| {
            h.depth_target_sized(0x0207, 8, 4);
            h.triangle([1.0, 0.0, 1.0, 1.0]);
        };
        agrees(set_up);

        // (9, 1) is inside the triangle and outside the depth surface, (1, 1) inside both.
        let mut h = Harness::new();
        let (inside, outside) = (h.texel(1, 1), h.texel(9, 1));
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        assert_ne!(h.texel(1, 1), inside, "drawn where both surfaces exist");
        assert_eq!(h.texel(9, 1), outside, "untouched past the depth surface");
    }

    #[test]
    fn a_depth_only_pass_still_writes_depth() {
        // No colour target at all; the fragment shader still runs.
        let set_up = |h: &mut Harness| {
            h.depth_target(0x0207);
            // Unbind colour target 0: an address of zero is no surface.
            h.engine.regs.set(0x200, 0);
            h.engine.regs.set(0x201, 0);
            h.triangle([1.0, 1.0, 1.0, 1.0]);
        };
        agrees(set_up);
        let mut h = Harness::new();
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        assert_eq!(h.depth()[17], 0x8000_0000, "depth 0.5 at (1, 1)");
    }

    #[test]
    fn a_triangle_fan_is_assembled_the_way_the_rasterizer_assembles_one() {
        // Fans go through `raster::assemble` on both renderers.
        for primitive in [6, 9] {
            // A quad as a four-vertex fan: a list or strip would leave (0, 7) bare.
            let set_up = move |h: &mut Harness| {
                h.engine.last_draw.primitive = primitive;
                h.engine.last_draw.count = 4;
                let limit = h.vertices() as u32 + 4 * 32 - 1;
                h.engine.regs.set(0x7C1, limit);
                h.depth_target(0x0207);
                let colour = [1.0, 0.0, 1.0, 1.0];
                h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], colour);
                h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], colour);
                h.write_vertex(2, [1.0, -1.0, 0.0, 1.0], colour);
                h.write_vertex(3, [-1.0, -1.0, 0.0, 1.0], colour);
            };
            agrees(set_up);
            let mut h = Harness::new();
            set_up(&mut h);
            h.draw_with(&mut Software).expect("the draw");
            assert_eq!(h.texel(0, 7), 0xffff_00ff, "primitive {primitive}");
        }
    }

    #[test]
    fn an_instanced_array_reads_this_instance_and_not_the_first() {
        // An instanced array's single uploaded element needs a zero stride.
        for instance in [0, 1, 2] {
            agrees(move |h| {
                h.depth_target(0x0207);
                h.triangle([0.0, 0.0, 0.0, 1.0]);
                h.instanced_colour(
                    instance,
                    &[
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ],
                );
            });
        }
    }

    #[test]
    fn a_bgra_attribute_is_swapped_the_way_the_rasterizer_swaps_one() {
        // Red and blue tell a BGRA swap apart.
        let set_up = |h: &mut Harness| {
            h.depth_target(0x0207);
            h.triangle([1.0, 0.0, 0.0, 1.0]);
            let raw = h.engine.regs.get(0x459);
            h.engine.regs.set(0x459, raw | 1 << 31);
        };
        agrees(set_up);
        let mut h = Harness::new();
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        assert_eq!(h.texel(1, 1), 0xffff_0000, "red arrives as blue");
    }

    /// With late readbacks, the frame after a fallback and those following go to the rasterizer.
    #[test]
    fn a_frame_the_device_cannot_finish_is_a_frame_it_does_not_start() {
        let Some(mut gpu) = device() else { return };
        // What a browser teaches it on its first present.
        gpu.deferred_readbacks = true;
        let colour = [1.0f32, 0.0, 1.0, 1.0];

        let mut h = Harness::new();
        h.triangle(colour);
        h.clear_with(&mut gpu, [true; 4]).expect("the clear");
        assert!(!gpu.software_frame, "nothing has fallen back yet");
        // A line loop must fall back.
        h.engine.last_draw.primitive = 2;
        // The rasterizer refuses it too; the fallback is what is under test.
        let _ = h.draw_with(&mut gpu);
        assert!(
            gpu.fell_back_this_frame,
            "a line loop should not have been expressible"
        );

        // The next frame's clear is where that becomes a decision.
        h.engine.last_draw.primitive = 4;
        h.clear_with(&mut gpu, [true; 4]).expect("the clear");
        assert!(
            gpu.software_frame,
            "the frame after a fallback is the rasterizer's"
        );
        let drawn = gpu.drawn;
        h.draw_with(&mut gpu).expect("the draw");
        assert_eq!(
            gpu.drawn, drawn,
            "a draw ran on the device in a rasterizer's frame"
        );

        // Read before anything clears it again.
        let got = h.target();
        let mut want = Harness::new();
        want.triangle(colour);
        want.clear_with(&mut Software, [true; 4])
            .expect("the clear");
        want.draw_with(&mut Software).expect("the draw");
        assert_eq!(got, want.target());

        // One clean frame releases the latch the first time.
        h.clear_with(&mut gpu, [true; 4]).expect("the clear");
        assert!(
            !gpu.software_frame,
            "a clean frame did not release the latch"
        );
        assert_eq!(gpu.unlatched, 1);
    }

    /// The latch holds through unrunnable frames, releases after a clean one, and doubles each relatch.
    #[test]
    fn the_latch_lets_go_after_clean_frames_and_waits_longer_each_time() {
        let Some(mut gpu) = device() else { return };
        gpu.deferred_readbacks = true;
        let mut h = Harness::new();
        h.triangle([1.0, 0.0, 1.0, 1.0]);
        // A line loop has no pipeline, so the device and the check refuse it.
        let line_loop = |h: &mut Harness, gpu: &mut super::Gpu| {
            h.engine.last_draw.primitive = 2;
            let _ = h.draw_with(gpu);
            h.engine.last_draw.primitive = 4;
        };
        let frame = |h: &mut Harness, gpu: &mut super::Gpu| {
            h.clear_with(gpu, [true; 4]).expect("the clear");
        };

        frame(&mut h, &mut gpu);
        line_loop(&mut h, &mut gpu);
        frame(&mut h, &mut gpu);
        assert!(gpu.software_frame, "a fallback did not latch");

        line_loop(&mut h, &mut gpu);
        frame(&mut h, &mut gpu);
        assert!(
            gpu.software_frame,
            "released after a frame the device could not have drawn"
        );

        // Nothing drawn is nothing learned.
        frame(&mut h, &mut gpu);
        assert!(gpu.software_frame, "released after a frame with no draws");

        h.draw_with(&mut gpu).expect("the draw");
        frame(&mut h, &mut gpu);
        assert!(!gpu.software_frame, "a clean frame did not release it");

        // Falling back again closes it, and the next release waits for two.
        line_loop(&mut h, &mut gpu);
        frame(&mut h, &mut gpu);
        assert!(gpu.software_frame, "a second fallback did not latch");
        assert_eq!(gpu.clean_frames_needed, 2);
        h.draw_with(&mut gpu).expect("the draw");
        frame(&mut h, &mut gpu);
        assert!(gpu.software_frame, "released after one of two clean frames");
        h.draw_with(&mut gpu).expect("the draw");
        frame(&mut h, &mut gpu);
        assert!(!gpu.software_frame, "two clean frames did not release it");
        assert_eq!(gpu.unlatched, 2);
    }

    #[test]
    fn a_clear_writes_what_the_rasterizer_would_have_written() {
        let Some(mut gpu) = device() else { return };
        for channels in [[true; 4], [true, false, true, false], [false; 4]] {
            let build = |gpu: Option<&mut super::Gpu>| {
                let mut h = Harness::new();
                // A channel in each of the four so a masked clear has one to leave alone.
                // No 0.5: the two renderers round 127.5 differently.
                h.engine.regs.set(0x360, 0.0f32.to_bits());
                h.engine.regs.set(0x361, 0.2f32.to_bits());
                h.engine.regs.set(0x362, 0.6f32.to_bits());
                h.engine.regs.set(0x363, 1.0f32.to_bits());
                match gpu {
                    Some(gpu) => {
                        h.clear_with(gpu, channels).expect("the clear");
                        h.flush_with(gpu);
                    }
                    None => h.clear_with(&mut Software, channels).expect("the clear"),
                }
                h.target()
            };
            let want = build(None);
            let before = gpu.fallbacks;
            let got = build(Some(&mut gpu));
            assert_eq!(
                gpu.fallbacks, before,
                "the clear fell back: {:?}",
                gpu.last_fallback
            );
            let _ = gpu.device.poll(wgpu::PollType::Poll);
            assert_eq!(gpu.device_error(), None, "the device rejected the clear");
            assert_eq!(got, want, "a clear of channels {channels:?}");
        }
    }

    #[test]
    fn an_attribute_the_draw_binds_nothing_to_reads_what_the_rasterizer_reads() {
        // Fixed attributes get a constant buffer rather than falling back.
        agrees(|h| {
            h.depth_target(0x0207);
            h.triangle([1.0, 1.0, 1.0, 1.0]);
            // VertexAttribState[1], the colour: fixed, so no buffer feeds it.
            let raw = h.engine.regs.get(0x459);
            h.engine.regs.set(0x459, raw | 1 << 6);
        });
    }

    /// Every multisample mode; `4x4` takes the expanded route and `2x2` the device's.
    #[test]
    fn a_multisampled_draw_reaches_the_same_texels_the_rasterizer_reaches() {
        for (mode, x, y) in [(1, 2, 1), (2, 2, 2), (3, 4, 2), (6, 4, 4)] {
            compare(mode, x, y, |_| {});
        }
    }

    /// The device multisampling route: fully covered pixels match, edges may differ.
    #[test]
    fn the_device_route_agrees_wherever_an_edge_is_not() {
        let Some(mut gpu) = device() else { return };
        gpu.set_device_msaa(true);
        let colour = [1.0f32, 1.0, 1.0, 1.0];
        let mut ran = 0;
        for (mode, x, y) in [(1, 2, 1), (2, 2, 2), (3, 4, 2), (6, 4, 4)] {
            let build = |gpu: Option<&mut super::Gpu>| {
                let mut h = Harness::new();
                h.multisample(mode, x, y);
                h.triangle(colour);
                match gpu {
                    Some(gpu) => {
                        h.draw_with(gpu).expect("the draw");
                        h.flush_with(gpu);
                    }
                    None => h.draw_with(&mut Software).expect("the draw"),
                }
                h.target()
            };
            let want = build(None);
            let before = gpu.multisampled;
            let got = build(Some(&mut gpu));
            if gpu.multisampled == before {
                // Not offered, so it went the expanded way, already covered.
                continue;
            }
            ran += 1;
            let _ = gpu.device.poll(wgpu::PollType::Poll);
            assert_eq!(gpu.device_error(), None, "the device rejected the pass");
            let width = switch_core::gpu::testing::TARGET_WIDTH;
            for py in 0..switch_core::gpu::testing::TARGET_HEIGHT / y {
                for px in 0..width / x {
                    let tile: Vec<u32> = (0..y)
                        .flat_map(|dy| (0..x).map(move |dx| (dx, dy)))
                        .map(|(dx, dy)| want[((py * y + dy) * width + px * x + dx) as usize])
                        .collect();
                    if tile.iter().any(|&t| t != tile[0]) {
                        continue;
                    }
                    for (dx, dy) in (0..y).flat_map(|dy| (0..x).map(move |dx| (dx, dy))) {
                        let at = ((py * y + dy) * width + px * x + dx) as usize;
                        assert_eq!(
                            got[at], tile[0],
                            "mode {mode}: pixel ({px}, {py}) is not on an edge and differs"
                        );
                    }
                }
            }
        }
        assert!(
            ran > 0,
            "this device offered none of the sample counts under test"
        );
    }

    #[test]
    fn a_sample_mask_keeps_the_same_samples_on_the_device() {
        for (mode, x, y) in [(2, 2, 2), (6, 4, 4)] {
            for mask in [0b0001, 0b1010, 0b0110] {
                compare(mode, x, y, move |h| {
                    h.engine.regs.set(testing::MULTISAMPLE_SAMPLE_MASK, mask);
                });
            }
        }
    }

    #[test]
    fn alpha_to_coverage_keeps_the_same_samples_on_the_device() {
        // Alpha-to-coverage on the expanded route at 4x4, against the reference.
        for alpha in [0.0f32, 0.25, 0.5, 1.0] {
            compare(6, 4, 4, move |h| {
                h.engine.regs.set(testing::MULTISAMPLE_CONTROL, 1);
                let colour = [1.0, 1.0, 1.0, alpha];
                h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], colour);
                h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], colour);
                h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], colour);
            });
        }
    }

    /// `AntiAliasEnable` off over a multisampled surface: whole-pixel coverage.
    #[test]
    fn coverage_per_pixel_covers_whole_pixels_on_the_device_too() {
        for (mode, x, y) in [(1, 2, 1), (2, 2, 2), (6, 4, 4)] {
            compare(mode, x, y, |h| {
                h.engine.regs.set(testing::MULTISAMPLE_ENABLE, 0);
            });
        }
    }

    #[test]
    fn every_pass_the_backend_builds_for_itself_compiles() {
        let Some(mut gpu) = device() else { return };
        // A multisampled colour format and the two readback depth formats.
        let colour = wgpu::TextureFormat::Bgra8Unorm;
        let depths = [
            wgpu::TextureFormat::Depth16Unorm,
            wgpu::TextureFormat::Depth32Float,
        ];

        for depth in depths {
            let _ = gpu.depth_loader(depth);
            gpu.clear_pipeline(super::ClearKey {
                color: None,
                depth: Some(depth),
                write_mask: [true; 4],
            })
            .expect("a depth clear pipeline");
        }
        for write_mask in [[true; 4], [true, false, true, false]] {
            gpu.clear_pipeline(super::ClearKey {
                color: Some(colour),
                depth: None,
                write_mask,
            })
            .expect("a colour clear pipeline");
        }

        // Four samples is the count core WebGPU guarantees.
        let samples = 4;
        assert!(
            gpu.samples_supported(colour, samples),
            "an adapter that will not multisample {colour:?} four ways"
        );
        for (dst, is_depth) in [(colour, false), (depths[0], true), (depths[1], true)] {
            if is_depth && !gpu.samples_supported(dst, samples) {
                continue;
            }
            for key in [
                // Into a device multisample companion, and into a per-pixel one.
                super::ResampleKey {
                    entry: "fs_gather",
                    dst,
                    samples,
                    ms_source: false,
                    depth: is_depth,
                },
                super::ResampleKey {
                    entry: "fs_gather_flat",
                    dst,
                    samples: 1,
                    ms_source: false,
                    depth: is_depth,
                },
                // And back out of each.
                super::ResampleKey {
                    entry: "fs_scatter",
                    dst,
                    samples: 1,
                    ms_source: true,
                    depth: is_depth,
                },
                super::ResampleKey {
                    entry: "fs_scatter",
                    dst,
                    samples: 1,
                    ms_source: false,
                    depth: is_depth,
                },
            ] {
                gpu.resample_pipeline(key)
                    .unwrap_or_else(|e| panic!("{key:?}: {e}"));
            }
        }

        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert_eq!(
            gpu.device_error(),
            None,
            "the device rejected one of its own passes"
        );
    }

    /// The grid tables the resampling passes read.
    #[test]
    fn the_grid_a_resampling_pass_reads_is_the_one_the_rasterizer_uses() {
        use switch_core::gpu::surface::SampleGrid;
        let grid = SampleGrid::new(2, &[0; 16]).expect("a 2x2 grid");
        assert_eq!(grid.count(), 4);
        let bytes = super::grid_bytes(grid);
        let word = |i: usize| {
            u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ])
        };
        assert_eq!(bytes.len(), 8 + 3 * 16 * 4);
        assert_eq!((word(0), word(1)), (2, 2), "the tile a pixel owns");
        for sample in 0..grid.count() {
            let (x, y) = grid.slot(sample);
            assert_eq!(word(2 + sample as usize), x);
            assert_eq!(word(18 + sample as usize), y);
            // The inverse maps the slot back to this sample.
            assert_eq!(word(34 + (y * 2 + x) as usize), sample);
        }
    }

    /// Fallback reasons are escaped so `JSON.parse` accepts the report.
    #[test]
    fn a_reason_with_json_punctuation_in_it_stays_one_string() {
        assert_eq!(super::json_string("plain"), "\"plain\"");
        assert_eq!(
            super::json_string(r#"no WGSL form for Ldg { at: 3 } "x" \ y"#),
            r#""no WGSL form for Ldg { at: 3 } \"x\" \\ y""#
        );
        assert_eq!(
            super::json_string("two\nlines\ttabbed"),
            r#""two\nlines\ttabbed""#
        );
        // A control character has no literal form at all.
        assert_eq!(super::json_string("\u{1}"), r#""\u0001""#);
    }
}
