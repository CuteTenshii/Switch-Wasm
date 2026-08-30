//! A GPU backend for switch-core's 3D engine.
//!
//! [`switch_core::gpu::renderer::Renderer`] is the seam this plugs into, and
//! [`switch_core::gpu::renderer::Software`] is what it has to agree with.
//! That is the whole shape of the thing: the software rasterizer is the
//! reference, this is the fast path, and any draw this cannot express runs on
//! the reference instead. A backend that guessed at a draw it did not
//! understand would produce a frame nobody could check.
//!
//! # Why this is a crate of its own
//!
//! `switch-core` has no dependencies at all — its Wasm bindings are
//! hand-rolled and its display path is `putImageData`. `wgpu` brings a few
//! hundred crates. Keeping it out here means the core stays what it is, and
//! the trait is the only thing the two share.
//!
//! # A draw never blocks
//!
//! A render target lives in guest memory, so the obvious arrangement is to
//! read the surface in, render, and write it back per draw. That is what this
//! did first, and it was byte-identical to the rasterizer, and it cannot work
//! in a browser: reading a texture back means waiting on a promise, and a
//! blocking wait there is not slow but deadlocked, because the event loop
//! that would resolve it cannot run.
//!
//! So a surface stays on the device once it is there, across every draw that
//! targets it, and goes back to guest memory only at
//! [`Renderer::flush`] — which the engine calls before `present`, the one
//! reader that always matters. Draws encode and return. The waiting happens
//! at the frame boundary, which in a browser is exactly where a worker is
//! free to await.
//!
//! That is not only a portability change. It is the same change that turns
//! eighty-eight round trips a frame into one.

/// What to ask a device for: the compressed texture families this adapter
/// actually has, and nothing else.
///
/// A Switch title's textures are block-compressed, and WebGPU hands those out
/// only on request — `DeviceDescriptor::default()` asks for none of them, so
/// the first BC1 texture threw. Masking against the adapter keeps the request
/// itself from failing on hardware that lacks a family, and
/// [`device_texture_format`] turns whatever is still missing into a fallback
/// rather than a crash.
pub fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    let wanted = wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::TEXTURE_COMPRESSION_ASTC
        | wgpu::Features::TEXTURE_COMPRESSION_ETC2
        // `R16Unorm` and its wider siblings, which a title samples as an
        // ordinary texture and WebGPU does not offer. Asked for the same way
        // the compressed formats are: taken where an adapter has it, and the
        // draw falls to the rasterizer where it does not.
        | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        // WGSL's quad operations, which are what a warp shuffle is: a `shfl`
        // reads another lane of the 2x2 quad, and `quadSwapX`/`Y`/`Diagonal`
        // are exactly that. Without them the draw is the rasterizer's.
        | wgpu::Features::SUBGROUP
        // Without this a device offers the sample counts the WebGPU spec
        // guarantees — one and four — whatever the adapter underneath it can
        // do. Maxwell's multisample modes are two, four, eight and sixteen,
        // so asking for it is the difference between `2x1` and `4x2` being
        // multisampled by the device and being rendered the long way round.
        // It does not exist on the web, where four is all there is; see
        // `Gpu::samples_supported`, which asks whichever source is telling
        // the truth on this device.
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    // A constant bank is bound as a storage buffer, and WebGPU guarantees
    // only eight of those per stage. Maxwell has eighteen banks and a shader
    // is free to read nine, which `create_pipeline_layout` then rejects
    // outright — so the draw is lost to a limit rather than to anything it
    // asked the device to do. Raised to whatever the adapter really has,
    // which is the mechanism the specification provides for exactly this;
    // asking for more than that would fail device creation instead.
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

/// The `wgpu` this was built against, so a caller that has to name a device
/// type does not have to guess at a matching version.
pub use wgpu;

use switch_core::gpu::engine::threed::{Engine3D, ShaderStage};
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::pipeline::{self as state, AttributeBase, Format, Pipeline};
use switch_core::gpu::renderer::{Flush, Renderer, Software};
use switch_core::gpu::shader::compiled::Compiled;
use switch_core::gpu::shader::wgsl::{self, Coverage, Layout, Stage, Translation};
use switch_core::gpu::surface::{SampleGrid, MAX_SAMPLES};
use switch_core::gpu::upload::{Banks, DepthKind, Target, Targets, TextureKey, Uploads};
use switch_core::{Error, Result};

/// The memory a `ldg` reads, resolved from the descriptor its address was
/// built out of.
struct GlobalUpload {
    stage: ShaderStage,
    slot: u32,
    bytes: Vec<u8>,
}

/// How much of a mapping one `ldg` buffer may take. A shader indexes from a
/// descriptor and nothing in the program says how far it reaches, so the
/// upload runs to the end of the mapping the descriptor points into — and
/// stops here, well inside `maxStorageBufferBindingSize`, rather than moving
/// a mapping's worth of memory for a draw that reads a few words of it.
const MAX_GLOBAL: u64 = 8 << 20;

/// Keep the leftmost `window` texels of each row of a linear image whose rows
/// are `stride` texels wide.
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

/// Where a module's textures start binding; see
/// `switch_core::gpu::shader::wgsl`.
const TEXTURE_BINDING: u32 = 32;

/// What a vertex attribute the draw supplies no buffer for reads.
///
/// Two vectors, because the rasterizer has two answers. A *fixed* attribute
/// is the `vec4` default every graphics API hands an unsupplied input, which
/// is `raster::ATTRIB_DEFAULT`; a slot the register file never configured is
/// not fetched at all and stays at the zero its attribute space starts with.
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

/// The pass that puts a guest depth surface onto the device.
///
/// `depth32float` is a format a copy may read out of and never write into,
/// so the only way in is to draw it: a fullscreen triangle whose fragment
/// reads the texel that was uploaded to an ordinary `r32float` texture and
/// reports it as its own depth. `depth16unorm` needs none of this — it is
/// the one depth format a copy may write — but it goes the same way, because
/// two paths that must agree about a surface's contents are one more place
/// for them to disagree than there needs to be.
const LOAD_DEPTH_WGSL: &str = "\
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
  // One triangle covering the whole of clip space, out of three vertices and
  // no vertex buffer at all.
  let x = f32(i32(vertex) / 2) * 4.0 - 1.0;
  let y = f32(i32(vertex) & 1) * 4.0 - 1.0;
  return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @builtin(frag_depth) f32 {
  return textureLoad(src, vec2<u32>(position.xy), 0).x;
}
";

/// The pass that clears part of a surface.
///
/// A clear that covers the whole of one is a render pass's load operation and
/// needs none of this. A clear that covers a rectangle of it, or only some of
/// its channels, is not something a load operation can say — so it is a
/// fullscreen triangle under a scissor, with the write mask baked into the
/// pipeline and the value in a uniform.
const CLEAR_RECT_WGSL: &str = "\
struct Clear {
  color: vec4<f32>,
  depth: f32,
}
@group(0) @binding(0) var<uniform> clear: Clear;

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
  let x = f32(i32(vertex) / 2) * 4.0 - 1.0;
  let y = f32(i32(vertex) & 1) * 4.0 - 1.0;
  return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_color() -> @location(0) vec4<f32> {
  return clear.color;
}

@fragment
fn fs_depth() -> @builtin(frag_depth) f32 {
  return clear.depth;
}
";

/// The WGSL that moves a multisampled surface between the two shapes it has.
///
/// A Maxwell multisample surface stores its samples *spatially*: a pixel owns
/// a `samples_x` by `samples_y` tile of texels, and that expanded image is
/// what guest memory holds and what a readback has to produce. A device's own
/// multisampling stores them opaquely, in a texture a copy may not touch at
/// all. So the two shapes both exist, and this is the pass between them —
/// `gather` on the way in, `scatter` on the way out.
///
/// `sampled` is how the source is declared and `load` is how one texel comes
/// out of it, which is the only thing that differs between the four
/// directions. The grid is a storage buffer rather than a uniform because its
/// tables are indexed by a value only known at run time and a uniform array
/// pads every element to sixteen bytes.
fn resample_wgsl(sampled: &str, load: &str, depth: bool) -> String {
    let output = if depth {
        "@builtin(frag_depth) f32"
    } else {
        "@location(0) vec4<f32>"
    };
    format!(
        "\
struct Grid {{
  size: vec2<u32>,
  // Which texel of a pixel's tile holds each sample, as x and y.
  slot_x: array<u32, 16>,
  slot_y: array<u32, 16>,
  // The inverse: which sample each texel of the tile holds.
  sample_of_slot: array<u32, 16>,
}}
@group(0) @binding(0) var<storage, read> grid: Grid;
@group(0) @binding(1) var src: {sampled};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {{
  let x = f32(i32(vertex) / 2) * 4.0 - 1.0;
  let y = f32(i32(vertex) & 1) * 4.0 - 1.0;
  return vec4<f32>(x, y, 0.0, 1.0);
}}

// Into the companion: this fragment is one pixel, and `sample` says which of
// its samples. The texel it comes from is that sample's slot in the tile.
@fragment
fn fs_gather(
  @builtin(position) position: vec4<f32>,
  @builtin(sample_index) sample: u32,
) -> {output} {{
  let pixel = vec2<u32>(position.xy);
  let slot = vec2<u32>(grid.slot_x[sample], grid.slot_y[sample]);
  let texel = pixel * grid.size + slot;
  return {load};
}}

// The same fragment for a companion that has one sample per pixel: every
// texel of the tile holds the same value, so sample 0's slot is the one to
// read and `sample_index` means nothing.
@fragment
fn fs_gather_flat(@builtin(position) position: vec4<f32>) -> {output} {{
  let pixel = vec2<u32>(position.xy);
  let slot = vec2<u32>(grid.slot_x[0], grid.slot_y[0]);
  let texel = pixel * grid.size + slot;
  let sample = 0u;
  // Named whether or not this direction's load reads it, so that one text
  // serves all four: a phony assignment is what WGSL has for saying so.
  _ = sample;
  return {load};
}}

// Out of the companion: this fragment is one texel of the expanded surface,
// and which sample it holds is its place in its pixel's tile.
@fragment
fn fs_scatter(@builtin(position) position: vec4<f32>) -> {output} {{
  let texel = vec2<u32>(position.xy);
  let pixel = texel / grid.size;
  let slot = texel % grid.size;
  let sample = grid.sample_of_slot[slot.y * grid.size.x + slot.x];
  _ = sample;
  return {load};
}}
"
    )
}

/// What a resampling pipeline is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResampleKey {
    /// Which fragment entry point of [`resample_wgsl`] runs.
    entry: &'static str,
    /// The destination's format, which is also the source's — a companion is
    /// the same format as the surface it stands in for.
    dst: wgpu::TextureFormat,
    /// The destination's sample count.
    samples: u32,
    /// Whether the source is a device multisample texture.
    ms_source: bool,
    depth: bool,
}

/// The grid, as the storage buffer [`resample_wgsl`] reads.
fn grid_bytes(grid: SampleGrid) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + 3 * 16 * 4);
    out.extend_from_slice(&grid.samples_x.to_le_bytes());
    out.extend_from_slice(&grid.samples_y.to_le_bytes());
    for sample in 0..MAX_SAMPLES as u32 {
        let (x, _) = grid.slot(sample.min(grid.count() - 1));
        out.extend_from_slice(&x.to_le_bytes());
    }
    for sample in 0..MAX_SAMPLES as u32 {
        let (_, y) = grid.slot(sample.min(grid.count() - 1));
        out.extend_from_slice(&y.to_le_bytes());
    }
    for sample in grid.sample_of_slot() {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// Which shape a surface's companion has, and so which pass moves values
/// between it and the expanded surface guest memory holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Shape {
    /// A device multisample texture of `n` samples, one pixel per texel of
    /// its own. What a draw with per-sample coverage renders into when the
    /// adapter offers that sample count.
    Multisampled(u32),
    /// One sample per pixel, at the pixel centre — `AntiAliasEnable` off over
    /// a surface that still has a tile of texels per pixel. A device's
    /// multisampling cannot be told to test coverage there, so the draw is
    /// rendered at pixel resolution and every texel of the tile takes the
    /// answer.
    PerPixel,
}

/// What a clear pipeline is built from.
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

fn topology(topology: state::Topology) -> wgpu::PrimitiveTopology {
    match topology {
        state::Topology::PointList => wgpu::PrimitiveTopology::PointList,
        state::Topology::LineList => wgpu::PrimitiveTopology::LineList,
        state::Topology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        state::Topology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        state::Topology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
    }
}

/// The guest's per-channel colour write enables, as WebGPU spells them.
fn write_mask(mask: [bool; 4]) -> wgpu::ColorWrites {
    let channels = [
        wgpu::ColorWrites::RED,
        wgpu::ColorWrites::GREEN,
        wgpu::ColorWrites::BLUE,
        wgpu::ColorWrites::ALPHA,
    ];
    let mut writes = wgpu::ColorWrites::empty();
    for (enabled, channel) in mask.into_iter().zip(channels) {
        if enabled {
            writes |= channel;
        }
    }
    writes
}

/// Write a draw's two WGSL modules under `dir`, named by their hash, so the
/// exact source a title makes can be compiled somewhere other than here.
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

fn vertex_format(format: state::VertexFormat) -> wgpu::VertexFormat {
    match format {
        state::VertexFormat::Float32 => wgpu::VertexFormat::Float32,
        state::VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
        state::VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
        state::VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
        state::VertexFormat::Float16x2 => wgpu::VertexFormat::Float16x2,
        state::VertexFormat::Float16x4 => wgpu::VertexFormat::Float16x4,
        state::VertexFormat::Unorm16x2 => wgpu::VertexFormat::Unorm16x2,
        state::VertexFormat::Unorm16x4 => wgpu::VertexFormat::Unorm16x4,
        state::VertexFormat::Snorm16x2 => wgpu::VertexFormat::Snorm16x2,
        state::VertexFormat::Snorm16x4 => wgpu::VertexFormat::Snorm16x4,
        state::VertexFormat::Sint16x2 => wgpu::VertexFormat::Sint16x2,
        state::VertexFormat::Sint16x4 => wgpu::VertexFormat::Sint16x4,
        state::VertexFormat::Uint16x2 => wgpu::VertexFormat::Uint16x2,
        state::VertexFormat::Uint16x4 => wgpu::VertexFormat::Uint16x4,
        state::VertexFormat::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
        state::VertexFormat::Snorm8x4 => wgpu::VertexFormat::Snorm8x4,
        state::VertexFormat::Sint8x4 => wgpu::VertexFormat::Sint8x4,
        state::VertexFormat::Uint8x4 => wgpu::VertexFormat::Uint8x4,
    }
}

fn blend_factor(factor: state::BlendFactor) -> wgpu::BlendFactor {
    match factor {
        state::BlendFactor::Zero => wgpu::BlendFactor::Zero,
        state::BlendFactor::One => wgpu::BlendFactor::One,
        state::BlendFactor::Src => wgpu::BlendFactor::Src,
        state::BlendFactor::OneMinusSrc => wgpu::BlendFactor::OneMinusSrc,
        state::BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
        state::BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        state::BlendFactor::Dst => wgpu::BlendFactor::Dst,
        state::BlendFactor::OneMinusDst => wgpu::BlendFactor::OneMinusDst,
        state::BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
        state::BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        state::BlendFactor::SrcAlphaSaturated => wgpu::BlendFactor::SrcAlphaSaturated,
        state::BlendFactor::Constant => wgpu::BlendFactor::Constant,
        state::BlendFactor::OneMinusConstant => wgpu::BlendFactor::OneMinusConstant,
    }
}

fn blend_operation(operation: state::BlendOperation) -> wgpu::BlendOperation {
    match operation {
        state::BlendOperation::Add => wgpu::BlendOperation::Add,
        state::BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
        state::BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        state::BlendOperation::Min => wgpu::BlendOperation::Min,
        state::BlendOperation::Max => wgpu::BlendOperation::Max,
    }
}

fn compare(compare: state::Compare) -> wgpu::CompareFunction {
    match compare {
        state::Compare::Never => wgpu::CompareFunction::Never,
        state::Compare::Less => wgpu::CompareFunction::Less,
        state::Compare::Equal => wgpu::CompareFunction::Equal,
        state::Compare::LessEqual => wgpu::CompareFunction::LessEqual,
        state::Compare::Greater => wgpu::CompareFunction::Greater,
        state::Compare::NotEqual => wgpu::CompareFunction::NotEqual,
        state::Compare::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        state::Compare::Always => wgpu::CompareFunction::Always,
    }
}

/// The depth format a device holds a guest surface in. See
/// [`switch_core::gpu::upload::DepthKind`] for why there are only two.
fn depth_texture_format(kind: DepthKind) -> wgpu::TextureFormat {
    match kind {
        DepthKind::Unorm16 => wgpu::TextureFormat::Depth16Unorm,
        DepthKind::Float32 => wgpu::TextureFormat::Depth32Float,
    }
}

fn blend(blend: state::Blend) -> wgpu::BlendState {
    let component = |c: state::BlendComponent| wgpu::BlendComponent {
        src_factor: blend_factor(c.src_factor),
        dst_factor: blend_factor(c.dst_factor),
        operation: blend_operation(c.operation),
    };
    wgpu::BlendState {
        color: component(blend.color),
        alpha: component(blend.alpha),
    }
}

/// A readback that has been asked for and not yet copied out.
///
/// Kept as a type because asking and collecting are the two halves a browser
/// has to put an `await` between — see [`Gpu::write_back`], which today does
/// both with a wait in the middle.
#[derive(Debug)]
struct Pending {
    staging: wgpu::Buffer,
    target: Target,
    /// Bytes in one row of what the device holds. The surface's own for a
    /// colour target; for a depth one it is the device format's, which is
    /// not the guest's — a `Z24S8` texel is four bytes in memory and four
    /// bytes of `f32` on the device, and a `ZF32_X24S8` texel is eight and
    /// four.
    row_bytes: u32,
    /// That stride rounded up to the 256 bytes `copyTextureToBuffer` wants.
    padded: u32,
    /// What the map callback reported: [`MAP_WAITING`] until it runs, then
    /// [`MAP_READY`] or [`MAP_FAILED`]. Read rather than waited on, because
    /// on the web the callback runs from the event loop and nothing inside a
    /// slice can make that happen.
    state: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

const MAP_WAITING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;

/// Everything a render pipeline is built from, as a value that can be
/// compared and hashed.
///
/// Not the whole of [`Pipeline`]: the viewport, the scissor and the blend
/// constant are set on the pass rather than baked in, and they are the parts
/// that change from draw to draw. What is left is what a title reuses.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    /// The two module cache keys, which are hashes of the WGSL — so two
    /// draws share a pipeline exactly when they would share both modules.
    vs: u64,
    fs: u64,
    /// `None` for a depth-only pass, which has no colour attachment and so
    /// no colour target state either.
    target: Option<wgpu::TextureFormat>,
    /// The device depth format, whether depth is written, and what the test
    /// compares — all three of which a pipeline bakes in.
    depth: Option<(wgpu::TextureFormat, bool, state::Compare)>,
    /// The multisample state, which is only ever more than one sample on the
    /// route that lets the device do the multisampling.
    samples: u32,
    sample_mask: u64,
    alpha_to_coverage: bool,
    blend: Option<state::Blend>,
    write_mask: [bool; 4],
    topology: state::Topology,
    front_face: state::FrontFace,
    cull: state::Cull,
    /// `(stride, instanced, [(format, offset, location)])` for each bound
    /// vertex buffer, in the order they are bound.
    buffers: Vec<(u32, bool, Vec<(wgpu::VertexFormat, u64, u32)>)>,
}

/// A vertex buffer a draw binds, and how it is stepped through.
struct Bound {
    buffer: wgpu::Buffer,
    attributes: Vec<wgpu::VertexAttribute>,
    /// Zero for an instanced array and for the constant that feeds attribute
    /// slots the draw binds nothing to — both are one element read by every
    /// vertex.
    stride: u64,
    step: wgpu::VertexStepMode,
}

/// One resource a single draw made, held only until that draw is submitted.
#[derive(Debug)]
enum Scratch {
    Buffer(wgpu::Buffer),
    Texture(wgpu::Texture),
}

/// A render target held on the device, and where in guest memory it came
/// from.
#[derive(Debug)]
struct Held {
    texture: wgpu::Texture,
    target: Target,
    /// Whether anything has been drawn into it since it was uploaded. A
    /// surface nothing touched need not be written back.
    dirty: bool,
    /// What draws render into, when the expanded surface's own texels are
    /// not what a draw's coverage is measured in — see [`Shape`]. Gathered
    /// from the surface when it is made and scattered back into it before
    /// the surface is read, so that guest memory only ever sees the expanded
    /// form.
    companion: Option<Companion>,
}

/// A surface's stand-in, and the grid it stands in for.
///
/// The grid is kept rather than read again at the end: what puts a companion
/// back is a flush, and by then the register file has moved on to whatever
/// the next frame is doing.
#[derive(Debug)]
struct Companion {
    shape: Shape,
    texture: wgpu::Texture,
    grid: SampleGrid,
}

/// How many distinct device errors are worth keeping. A rejected draw repeats
/// its rejection every frame, so the list stops growing almost immediately and
/// the count carries the rest.
const MAX_DEVICE_ERRORS: usize = 16;

/// Everything the device has rejected since it was opened.
///
/// Keeping only the first — which is what this was — reported nothing at all
/// once the pipelines were built: the only production reader runs before a
/// pipeline is created, and a title that builds its four in the first frames
/// never creates another. Every rejection after that sat unread.
#[derive(Debug, Default)]
struct DeviceErrors {
    /// The oldest rejection nothing has asked about yet, taken by
    /// [`Gpu::device_error`]. Oldest rather than newest because the first
    /// rejection is the one with a cause; the rest are usually its echo.
    fresh: Option<String>,
    /// Each distinct message once, in the order first seen — the same shape
    /// as `reasons`, and for the same reason.
    distinct: Vec<String>,
    /// Every rejection, including the repeats and anything past
    /// [`MAX_DEVICE_ERRORS`].
    count: u64,
}

impl DeviceErrors {
    fn record(&mut self, message: String) {
        self.count += 1;
        if !self.distinct.contains(&message) {
            eprintln!("[gpu] the device rejected something: {message}");
            if self.distinct.len() < MAX_DEVICE_ERRORS {
                self.distinct.push(message.clone());
            }
        }
        self.fresh.get_or_insert(message);
    }
}

/// A device, and the rasterizer to fall back to.
#[derive(Debug)]
pub struct Gpu {
    /// The instance and adapter the device came from, held for as long as it
    /// is — never read again, and not droppable either.
    ///
    /// A browser loses a device when the last *external* reference to the
    /// instance behind it goes, and reports it as "A valid external Instance
    /// reference no longer exists". Dropping these at the end of the call that
    /// opened the device is what released it: the device's own handle does not
    /// count as one, so the loss arrived whenever the collector next ran, and
    /// from then on every readback failed to map and every frame was dropped.
    /// Natively they are held for nothing — wgpu-core keeps the instance alive
    /// behind the device — which is why this cost a browser to find.
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Surfaces this is holding, by the guest address they came from.
    ///
    /// A frame's draws all target the same surface, so this is what makes a
    /// frame one upload and one readback instead of eighty-eight of each.
    held: std::collections::HashMap<u64, Held>,
    /// The buffers and textures the draw in progress made, destroyed once it
    /// has been submitted.
    ///
    /// A draw's vertices, indices, constants and textures are built fresh
    /// every time, and dropping them is not enough to give the memory back:
    /// `wgpu` frees a dropped resource when the device is next polled, and a
    /// browser never polls — WebGPU reclaims on garbage collection, which
    /// does not run per draw and cannot be made to. Just Dance 2019 issues
    /// 55,465 draws in a six-billion-instruction run, each with a texture and
    /// three or four buffers, and an 8 GB card answered
    /// `VK_ERROR_OUT_OF_DEVICE_MEMORY` a long way short of the end of them.
    /// So each is destroyed outright, which WebGPU defines as safe once the
    /// submission that reads it has been made: the memory comes back when
    /// that work finishes rather than when the call does.
    scratch: Vec<Scratch>,
    /// Surfaces the guest rebound out from under, still owing a write-back.
    /// Kept rather than written back where it happened, so that no draw ever
    /// waits on a device.
    evicted: Vec<Held>,
    /// Readbacks asked for and not yet collected. A flush that finds one of
    /// these unfinished answers `Flush::Pending` rather than waiting: the
    /// wait it would have to do is the one a browser cannot perform.
    pending: Vec<Pending>,
    /// Compiled shader modules, by a hash of the WGSL that produced them.
    ///
    /// Compiling a module is the whole cost of a draw: WGSL through naga to
    /// SPIR-V is about 59 ms, against 2 ms to translate the shader and half
    /// a millisecond to read every buffer the draw touches. The Home Menu
    /// has five distinct shader pairs and drew eighty-eight times a frame,
    /// so this is not an optimisation so much as not doing the same work
    /// eighty-eight times.
    ///
    /// Keyed by the source rather than by the shader's address, because the
    /// source is what the module *is*: two draws whose shaders live at the
    /// same address but were assembled with a different texture swizzle are
    /// two different modules, and a guest is free to overwrite a shader in
    /// place.
    modules: std::collections::HashMap<u64, wgpu::ShaderModule>,
    /// Render pipelines, by everything they were built from.
    ///
    /// Building one is the device validating both modules against the fixed
    /// function state around them, and it happened once per draw. A title
    /// draws the same few pipelines over and over: the Home Menu's frame 60
    /// renders 480 draws through 7 of them, and Just Dance 2019's first 3,481
    /// through 4. So this is the argument [`Gpu::modules`] makes, one level
    /// up — and like that one it is unbounded, because a program that walks
    /// endlessly over fresh shaders is not a thing a title does.
    pipelines: std::collections::HashMap<PipelineKey, wgpu::RenderPipeline>,
    /// Bind group layouts, by the entries they describe.
    ///
    /// A layout is a description, not a resource, and two draws through the
    /// same pair of shaders describe the same one — so this hands the same
    /// object to the pipeline that was built from it and to every later
    /// draw's bind group. WebGPU matches the two structurally rather than by
    /// identity, so a cache is not what makes a cached pipeline usable; it is
    /// what makes it obvious that it is.
    group_layouts:
        std::collections::HashMap<Vec<wgpu::BindGroupLayoutEntry>, wgpu::BindGroupLayout>,
    /// The pipeline that puts a guest depth surface onto the device, by the
    /// depth format it writes. See [`LOAD_DEPTH_WGSL`].
    depth_loaders: std::collections::HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    /// The pipeline that clears a rectangle of a surface, by the formats and
    /// the write mask it was built for. See [`CLEAR_RECT_WGSL`].
    clear_pipelines: std::collections::HashMap<ClearKey, wgpu::RenderPipeline>,
    /// The pipelines that move a multisampled surface between its expanded
    /// form and a device companion. See [`resample_wgsl`].
    resample_pipelines: std::collections::HashMap<ResampleKey, wgpu::RenderPipeline>,
    /// `GPU_DEVICE_MSAA=1`: let the device do the multisampling where it
    /// offers the sample count, instead of rendering the expanded surface a
    /// texel at a time.
    ///
    /// Off, because the two do not produce the same frame — see
    /// [`Gpu::route`]. It is a speed-for-fidelity trade and the frame it
    /// gives up is the one the rasterizer can be compared against.
    device_msaa: bool,
    /// What the device has rejected. Written by the uncaptured-error
    /// callback, read on the next draw, because asking sooner means waiting.
    failed: std::sync::Arc<std::sync::Mutex<DeviceErrors>>,
    /// Set when the device is lost, with the browser's reason.
    ///
    /// The one failure nothing else here can see: a lost device raises no
    /// error and rejects nothing. It accepts every submission and performs
    /// none of them, and the only symptom is a readback that never maps —
    /// which read as "the readback was not mapped", every frame, for as long
    /// as the title ran.
    lost: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// Whether a flush has ever answered [`Flush::Pending`] — that is,
    /// whether a readback on this host completes later than the call that
    /// asked for it.
    ///
    /// Natively it does not: the flush waits, so guest memory is the truth by
    /// the time the call returns and a draw may hand itself to the rasterizer
    /// in the middle of a frame. In a browser it does, because a map completes
    /// from the event loop and nothing inside a run slice can make that
    /// happen — and there a mid-frame fallback reads what was in memory
    /// *before* the device drew, then has the readback land on top of what it
    /// wrote.
    ///
    /// Observed rather than compiled in, because it is a fact about how the
    /// host answers and not about which target this was built for.
    deferred_readbacks: bool,
    /// `GPU_DEFER_READBACKS=1`: do not wait for a readback even where waiting
    /// is possible, so that a native run reproduces what a browser does. See
    /// [`Gpu::deferred_readbacks`], which this is the way to provoke.
    defer_readbacks: bool,
    /// `GPU_INTERLEAVE=1`: keep handing single draws to the rasterizer in the
    /// middle of a device frame even where readbacks land late, instead of
    /// giving the whole frame to the rasterizer.
    ///
    /// It is wrong, and it is not very wrong, and how wrong is measurable:
    /// with `GPU_DEFER_READBACKS=1` to make a native run behave like a
    /// browser, the Home Menu's frame 60 comes out with **795 of its 921,600
    /// pixels** written by a draw the readback then overwrote — 0.09% of the
    /// frame, in the places the fallback draws touched. Against that, the
    /// latch costs the Home Menu every device draw after the first frame,
    /// which is 0.10 s a frame becoming 1.03 s.
    ///
    /// So it is a real trade and the numbers are per title: a title the
    /// translator covers has nothing to trade, and one where the device draws
    /// half the frame would lose more than 795 pixels. Off, because a frame
    /// nobody produced is the one thing this backend is built not to make.
    interleave: bool,
    /// Whether the rasterizer has the frame, and every frame after it.
    ///
    /// The answer to a readback that cannot land inside a slice is not to
    /// interleave more carefully — it is not to interleave. A frame the
    /// device cannot render *all* of is one it renders none of: guest memory
    /// is then the only copy of every surface, and no readback is ever owed.
    ///
    /// It latches, because nothing can tell it to unlatch. A draw falls back
    /// on the shader it runs, a title runs the same shaders every frame, and
    /// a frame rendered entirely on the rasterizer never discovers whether
    /// the next one would have fallen back — so alternating is the one
    /// behaviour this must not have.
    ///
    /// **What buys the acceleration back is `shader::wgsl`.** Every fallback
    /// this latches on today is an opcode with no WGSL form, not anything
    /// WebGPU withholds: the Home Menu's are one `ldg b128`, and A Short
    /// Hike's are a handful of `Unimplemented`. A title the translator covers
    /// completely never reaches this.
    software_frame: bool,
    /// Whether anything fell back during the frame in progress.
    fell_back_this_frame: bool,
    /// Whether [`Gpu::give_up`] has already handed the frame back.
    gave_up: bool,
    /// The loss, kept for the first flush after it.
    ///
    /// A browser discards what `eprintln!` writes, and a flush's error is the
    /// only way a reason reaches the frontend's diagnostics from here.
    report: Option<String>,
    /// What this cannot express runs here instead.
    software: Software,
    /// Draws this rendered, and draws that fell back with why the last one
    /// did — the two numbers that say how much of a frame is really running
    /// here.
    pub drawn: u64,
    pub fallbacks: u64,
    pub last_fallback: Option<String>,
    /// Every distinct reason a draw fell back, in the order first seen.
    pub reasons: Vec<String>,
    /// Draws by the route they took — see [`Render`]. Which of the two
    /// multisampling routes a title actually gets is a question about the
    /// adapter as much as about the title, so it is counted rather than
    /// assumed.
    pub direct: u64,
    pub expanded: u64,
    pub multisampled: u64,
    pub per_pixel: u64,
    /// Which draw of the current frame this is, counting from the clear that
    /// starts it. Only `GPU_ONLY` reads it.
    in_frame: u32,
    /// Where a draw's time goes, printed when this is dropped and readable
    /// at any point through [`Renderer::report_json`].
    ///
    /// Always on under wasm and `GPU_TIMES=1` natively. `env_flag!` reads
    /// `std::env::var`, which is empty in a browser, so an env-gated clock
    /// is one the target can never switch on — and the browser has no other
    /// way to find out where a frame went. It costs one `performance.now()`
    /// per phase per draw, which is microseconds across a whole run.
    times: Option<Times>,
    /// What every draw's `Uploads::of` read, by category.
    uploaded: UploadBytes,
    /// Texture bytes already deswizzled, by what decides them.
    ///
    /// A title samples a handful of images over and over — 96.5% of
    /// everything `Uploads::of` lifts is texture bytes, 1.76 MiB a draw, and
    /// nearly all of it the same images read again. An entry lives until the
    /// guest writes to the memory behind it, which `Memory`'s watched pages
    /// report.
    texture_cache: std::collections::HashMap<TextureKey, std::sync::Arc<[u8]>>,
    /// Which cached textures a guest page holds bytes for, so a write to it
    /// evicts them without walking the cache. A key left here after its entry
    /// has gone evicts nothing, which is why nothing prunes these.
    page_owners: std::collections::HashMap<u32, Vec<TextureKey>>,
    /// The device's copy of a cached texture, by the same key and paired with
    /// the view it was created for — a 3D image and an array of the same
    /// bytes are different textures.
    ///
    /// Only ever holds a key `texture_cache` also holds. That is what ties it
    /// to a watched page: a texture kept here whose bytes were not cached
    /// would never be told the guest had overwritten it.
    gpu_textures:
        std::collections::HashMap<TextureKey, Vec<(wgpu::TextureViewDimension, wgpu::Texture)>>,
    cached_bytes: u64,
    pub texture_hits: u64,
    pub texture_misses: u64,
    gpu_texture_bytes: u64,
    /// Textures `prepare` read and has not watched yet. `prepare` is handed a
    /// shared `ExecCtx` and watching a page needs a mutable one, so the two
    /// halves happen either side of it.
    to_remember: Vec<(TextureKey, std::sync::Arc<[u8]>, u64)>,
    /// `GPU_ONLY=<i>` renders only the i-th draw of each frame here and
    /// leaves the rest to the rasterizer; `GPU_ONLY=<a>..<b>` renders the
    /// half-open range of them.
    ///
    /// Which is how you find the draw that renders differently. The
    /// difference between a frame and the reference is then exactly that
    /// range's — and it takes a range rather than an index because that is
    /// what a bisection needs: a frame is a hundred draws, and halving turns
    /// that into seven runs instead of a hundred.
    only: Option<std::ops::Range<u32>>,
}

/// How much deswizzled texture to keep. A frame's working set is a handful of
/// images at 1.76 MiB each, so this is generous on purpose: the point is to
/// bound a run that walks endlessly over fresh textures, not to ration a
/// normal one.
const TEXTURE_CACHE_BYTES: u64 = 256 << 20;
/// Guest pages are 4 KiB, the granularity `Memory` watches writes at.
const PAGE_BITS: u32 = 12;

/// What `Uploads::of` lifted out of guest memory over a whole run.
#[derive(Debug, Default, Clone, Copy)]
struct UploadBytes {
    vertex: u64,
    index: u64,
    constants: u64,
    textures: u64,
}

impl UploadBytes {
    /// What one draw lifted out of guest memory, by what it was.
    ///
    /// Bytes rather than milliseconds: a byte read out of guest memory is the
    /// same byte under V8, and counting them is what showed textures to be
    /// 96.5% of this, and so the only one of the four worth caching.
    ///
    /// Textures are counted by [`UploadBytes::add_texture`] instead, and only
    /// when one was really read. A draw served from the cache never touched
    /// guest memory for it, and counting it here would report reads that did
    /// not happen — which is exactly what this said before the cache existed
    /// to make the two differ.
    fn add_but_textures(&mut self, uploads: &Uploads) {
        self.vertex += uploads
            .vertex
            .iter()
            .map(|v| v.bytes.len() as u64)
            .sum::<u64>();
        self.index += uploads.index.as_ref().map_or(0, |i| i.bytes.len() as u64);
        self.constants += uploads
            .constants
            .iter()
            .map(|c| c.bytes.len() as u64)
            .sum::<u64>();
    }

    fn add_texture(&mut self, bytes: usize) {
        self.textures += bytes as u64;
    }
}

/// Where a draw's time goes, in microseconds, over a whole run.
#[derive(Debug, Default, Clone, Copy)]
struct Times {
    /// Decoding both shaders out of guest memory and translating them.
    translate: u128,
    /// Reading the vertices, indices, constants and textures.
    upload: u128,
    /// Generating the WGSL and handing it to the device.
    modules: u128,
    /// Building the pipeline and its bind groups.
    pipeline: u128,
    /// Encoding and submitting the pass.
    encode: u128,
    /// Handing surfaces back to guest memory.
    flush: u128,
}

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
        if let Some(t) = self.times {
            let ms = |v: u128| v as f64 / 1000.0;
            eprintln!(
                "[gpu] translate {:.0}ms  upload {:.0}ms  modules {:.0}ms  \
                 pipeline {:.0}ms  encode {:.0}ms  flush {:.0}ms",
                ms(t.translate),
                ms(t.upload),
                ms(t.modules),
                ms(t.pipeline),
                ms(t.encode),
                ms(t.flush),
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
    /// Take a device somebody else opened.
    ///
    /// The browser's entry point. Opening a device there is asynchronous —
    /// `requestAdapter` and `requestDevice` are promises — and nothing in
    /// this crate may wait on a promise, so the waiting happens outside and
    /// the result is handed in. [`Gpu::open`] is the native convenience that
    /// does it by blocking, which is fine on a thread that owns itself.
    ///
    /// The instance and the adapter come too, and are not optional: see
    /// [`Gpu::_instance`] for what a browser does when they are dropped.
    pub fn with_device(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Gpu {
        // Where a rejection lands, since nothing in a draw ever stops to ask.
        let failed: std::sync::Arc<std::sync::Mutex<DeviceErrors>> =
            std::sync::Arc::new(std::sync::Mutex::new(DeviceErrors::default()));
        let sink = failed.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            if let Ok(mut slot) = sink.lock() {
                slot.record(e.to_string());
            }
        }));
        // Asked for by name, because a lost device is silent everywhere else.
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
            gave_up: false,
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
            page_owners: std::collections::HashMap::new(),
            gpu_textures: std::collections::HashMap::new(),
            cached_bytes: 0,
            texture_hits: 0,
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

    /// What adapter this opened.
    pub fn describe(&self) -> String {
        format!("{:?}", self.device.limits().max_texture_dimension_2d)
    }

    /// Give the device back, for one that will not be used.
    ///
    /// Dropping it does not: wgpu's web backend frees nothing on drop, so an
    /// abandoned device lives in the GPU process until the collector runs.
    pub fn destroy(&self) {
        self.device.destroy();
    }

    fn fall_back(&mut self, why: String) {
        self.fallbacks += 1;
        self.fell_back_this_frame = true;
        // Each distinct reason once: a draw that falls back does it every
        // frame, and the interesting thing is the list rather than the count.
        if !self.reasons.contains(&why) {
            eprintln!("[gpu] falling back: {why}");
            self.reasons.push(why.clone());
        }
        self.last_fallback = Some(why);
    }

    /// Hand the frame back to the rasterizer for good, once the device is
    /// lost. Answers whether it has been.
    ///
    /// A lost device cannot copy a surface back, so everything held on it is
    /// gone and what guest memory holds is the last thing the rasterizer
    /// wrote. That is what the display gets from here on — which is the
    /// point: the alternative, and what this replaces, was a readback that
    /// failed and a frame that was dropped, once per frame, forever.
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

    /// Keep interleaving single fallback draws into a device frame on a host
    /// whose readbacks land late — the `GPU_INTERLEAVE` flag, for the build
    /// with no environment to read it from. See [`Gpu::interleave`] for what
    /// it trades and what it costs.
    pub fn set_interleave(&mut self, interleave: bool) {
        self.interleave = interleave;
    }

    /// Let the device do the multisampling where it offers the sample count
    /// — the `GPU_DEVICE_MSAA` flag, for the build that has no environment to
    /// read it from. See [`Gpu::route`] for what it trades.
    pub fn set_device_msaa(&mut self, device_msaa: bool) {
        self.device_msaa = device_msaa;
    }

    /// The device texture for a surface, uploading it if this is the first
    /// draw to reach it.
    ///
    /// A draw that blends reads what is already there. The first time that
    /// is guest memory; afterwards it is whatever the previous draw left on
    /// the device, which is the same thing and already in the right place.
    fn hold(&mut self, target: &Target, ctx: &ExecCtx) -> Result<()> {
        match self.held.get(&target.addr) {
            // The same surface as last time. Anything about it having
            // changed — a different format or extent at the same address —
            // means the guest rebound it, and the old contents are not this
            // one's.
            Some(held) if held.target == *target => return Ok(()),
            // A different surface at the same address: the guest rebound
            // it. The old one still has to go back, but not here — a draw
            // that read a texture back is a draw that blocks.
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

    /// Ask for a surface back, without waiting for it.
    ///
    /// The waiting used to happen here, with `Device::poll`, which is a real
    /// wait natively and a no-op on the web — WebGPU has no polling, and a map
    /// completes when the event loop runs. So in a browser the collection read
    /// a buffer that was not mapped yet.
    ///
    /// Nothing waits now. [`Gpu::flush`] collects what has arrived and says
    /// `Flush::Pending` for what has not, and the *present* is what waits —
    /// `Cpu::complete_pending_present` puts the frame up from a later slice,
    /// once the host has had its turn. Which is not the same as landing a
    /// readback one flush late and presenting guest memory meanwhile: that
    /// came out black, because a double-buffered title queues the surface
    /// whose readback was just asked for.
    fn ask_for(&mut self, held: Held) {
        // Whatever a companion holds is part of the surface, and a readback
        // copies the surface — so it has to be in it first. This is where a
        // multisampled frame's samples land in the expanded layout guest
        // memory keeps them in.
        if let Some(companion) = &held.companion {
            let depth = held.target.depth_kind().is_some();
            if let Err(why) = self.resolve_into(&held.texture, companion, depth) {
                self.fall_back(format!("putting a companion surface back: {why}"));
            }
        }
        // A surface nothing drew into is already what guest memory says.
        if !held.dirty {
            return;
        }
        let pending = self.start_read_back(&held.target, &held.texture);
        self.pending.push(pending);
    }

    /// Bring a guest surface onto the device.
    fn upload_target(&mut self, target: &Target, ctx: &ExecCtx) -> Result<wgpu::Texture> {
        if let Some(kind) = target.depth_kind() {
            return self.upload_depth_target(target, kind, ctx);
        }
        let format = device_texture_format(&self.device, target.format)?;
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

    /// Bring a guest depth surface onto the device, converted.
    ///
    /// Nothing copies into a depth texture in the format this needs, so what
    /// is copied is an `r32float` image of the same texels and what puts it
    /// where it belongs is a pass — see [`LOAD_DEPTH_WGSL`].
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
        // `read_depth` walks the surface, whose rows are `row_bytes` wide
        // whatever the pass covers — so a cropped target takes the left of
        // each one. `Target::rows` is already the pass's height.
        let values = target.read_depth(ctx)?;
        let surface_texels = (target.row_bytes / target.unit.max(1)) as usize;
        let values = crop_rows(
            values,
            surface_texels,
            target.width as usize,
            kind.unit() as usize,
        );
        // The staging image is always `r32float`, whatever the depth format
        // is: one WGSL text, one pipeline per depth format, and a `u16` that
        // has to be widened on the way in either way.
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
                .chunks_exact(2)
                .flat_map(|v| {
                    let stored = u16::from_le_bytes([v[0], v[1]]);
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

    /// Draw `staging` into `texture`'s depth, which is the only way a value
    /// gets there.
    fn load_depth(
        &mut self,
        texture: &wgpu::Texture,
        staging: &wgpu::Texture,
        format: wgpu::TextureFormat,
    ) -> Result<()> {
        let pipeline = self.depth_loader(format);
        let layout = pipeline.get_bind_group_layout(0);
        let view = staging.create_view(&wgpu::TextureViewDescriptor::default());
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth upload"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        let target = texture.create_view(&wgpu::TextureViewDescriptor::default());
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

    /// The pipeline that draws a depth surface into a depth texture of this
    /// format, built once per format.
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

    /// Take a surface off the device.
    ///
    /// `copyTextureToBuffer` wants rows aligned to 256 bytes, which a
    /// surface's own rows need not be, so the padding is added on the way
    /// out and dropped on the way in.
    ///
    /// This is the one place that waits, and the reason [`Renderer::flush`]
    /// exists to be the only caller: on a browser the wait is a deadlock
    /// anywhere the event loop is not free to run.
    fn start_read_back(&self, target: &Target, texture: &wgpu::Texture) -> Pending {
        // What a device row holds is the device format's, which for a depth
        // surface is not the guest's texel width.
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

        // Asked for, not waited on. The map completes when the device is next
        // polled — which on a browser is when the event loop next runs, and
        // there is no way to make that happen from inside a blocking call.
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

    /// Copy a finished readback into guest memory, dropping the row padding
    /// `copyTextureToBuffer` insisted on.
    fn land(&self, pending: &Pending, ctx: &mut ExecCtx) -> Result<()> {
        let slice = pending.staging.slice(..);
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| Error::Gpu(format!("mapping the readback: {e}")))?;
        let target = &pending.target;
        // A colour surface goes back straight out of the mapping, padding and
        // all: `write_strided` skips whatever each row's alignment left over,
        // which is a whole-surface copy — 3.7 MB a frame at 720p — that only
        // ever existed to hand the walk a packed buffer.
        //
        // Depth still repacks. Its write-back reads the surface and patches a
        // window into it, so it wants the rows contiguous, and it is the
        // rarer path.
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
        // Destroyed rather than merely dropped, for the reason
        // [`Gpu::scratch`] gives: a readback is a whole surface, several a
        // frame, and in a browser a dropped buffer waits on a collector.
        pending.staging.destroy();
        outcome
    }

    /// Put a depth readback back, repacked.
    fn land_depth(target: &Target, kind: DepthKind, rows: &[u8], ctx: &mut ExecCtx) -> Result<()> {
        // A cropped depth target holds the left of each row; the rest is what
        // the surface already had, and reading it back is how it survives
        // being written whole.
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
    /// Build the pipeline and run the pass.
    ///
    /// Everything is created per draw: the modules, the pipeline, the
    /// buffers, the bind groups. That is the slow arrangement and the one
    /// worth having first, because a cache is only ever as right as the
    /// thing it caches.
    fn render(&mut self, p: &Prepared, ctx: &mut ExecCtx) -> std::result::Result<(), String> {
        let target_format = match p.color {
            Some(color) => Some(
                device_texture_format(&self.device, color.format).map_err(|e| format!("{e:?}"))?,
            ),
            None => None,
        };
        let depth_format = p
            .depth
            .and_then(|d| d.depth_kind())
            .map(depth_texture_format);
        // The sample mask and alpha-to-coverage belong to the device only on
        // the route where the device is doing the multisampling. On the
        // expanded route the fragment shader has them, and saying them twice
        // would apply them twice.
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

        // Vertex buffers, and the attributes the vertex shader actually
        // reads: a draw binds sixteen slots and a shader reads one.
        // The stride travels *with* the buffer rather than beside it: a
        // bound array whose attributes the shader never reads is skipped, so
        // the two lists are not the same length and pairing them by position
        // gave one array another's stride.
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
            bound.push(Bound {
                buffer: self.buffer("vertex", &upload.bytes, wgpu::BufferUsages::VERTEX),
                attributes,
                // An instanced array advances once per instance, and only
                // this instance's element was uploaded — so the stride is
                // nothing and every instance reads the one element there is.
                stride: match buffer.step {
                    state::StepMode::Instance => 0,
                    state::StepMode::Vertex => u64::from(buffer.stride),
                },
                step: match buffer.step {
                    state::StepMode::Instance => wgpu::VertexStepMode::Instance,
                    state::StepMode::Vertex => wgpu::VertexStepMode::Vertex,
                },
            });
        }
        // Every location the shader declares has to be fed, or the pipeline
        // will not build. A slot the draw binds no buffer to is not a gap:
        // `fetch_attribute` answers `(0, 0, 0, 1)` for a fixed attribute and
        // an unconfigured slot is left at zero, so both are a constant — one
        // buffer of two vectors, read with a stride of nothing.
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
        // A pipeline is keyed by what it is built from, so the cache can be
        // consulted before any of it is created. The bind group layouts are
        // not part of the key: they follow from the two modules, and WebGPU
        // matches a bind group to a pipeline structurally rather than by
        // identity, so the ones built for this draw fit a cached pipeline.
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
        // Empty for a depth-only pass: a colour state with no attachment
        // behind it is a pipeline that will not build.
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
                strip_index_format: None,
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
                    // Neither renderer tests stencil: `raster` reads the byte
                    // back only to put it where it was. When one of them
                    // learns to, they both must.
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

    /// Record and submit one draw against a pipeline that is already built.
    ///
    /// Split out from [`Gpu::render`] because it is the half that runs on
    /// every draw, cached pipeline or not.
    fn encode(
        &mut self,
        p: &Prepared,
        pipeline: &wgpu::RenderPipeline,
        vs_group: &wgpu::BindGroup,
        fs_group: &wgpu::BindGroup,
        bound: &[Bound],
        ctx: &mut ExecCtx,
    ) -> std::result::Result<(), String> {
        // Held across the frame: the first draw brings the surface onto the
        // device and every later one finds it already there.
        let colour_view = match p.color {
            Some(color) => {
                self.hold(&color, ctx).map_err(|e| format!("{e:?}"))?;
                Some(self.attachment(p, color.addr, true)?)
            }
            None => None,
        };
        let depth_view = match p.depth {
            Some(depth) => {
                self.hold(&depth, ctx).map_err(|e| format!("{e:?}"))?;
                // A draw that only tests depth leaves the surface as it
                // found it, and a surface nothing wrote need not go back.
                let writes = p.state.depth.is_some_and(|d| d.write_enabled);
                Some(self.attachment(p, depth.addr, writes)?)
            }
            None => None,
        };
        let index = match &p.assembled {
            // An assembled topology is always drawn indexed, whether or not
            // the draw it came from was: the triangles are a list of
            // ordinals, and that is what an index buffer is.
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
                    match index.format {
                        switch_core::gpu::upload::IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                        switch_core::gpu::upload::IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
                    },
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
                            // Loaded, never cleared: a clear is its own
                            // method, and this pass is one draw in the middle
                            // of a frame.
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
            // The viewport and the scissor are in pixels, which is what the
            // attachment is measured in on every route but the expanded one —
            // there the attachment is the surface itself, and its extent is
            // texels.
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
            // One instance, numbered so that `@builtin(instance_index)` is
            // the `gl_InstanceID` the rasterizer would have used.
            let instances = p.instance..p.instance + 1;
            match &index {
                Some((buffer, format, base)) => {
                    pass.set_index_buffer(buffer.slice(..), *format);
                    // The vertex buffer starts at the draw's lowest index, so
                    // every index in it is that much too high.
                    pass.draw_indexed(0..p.count, *base, instances);
                }
                None => pass.draw(0..p.count, instances),
            }
        }
        timed!(self, encode, self.queue.submit([encoder.finish()]));
        Ok(())
    }

    /// The view a draw renders one of its surfaces through, with whatever
    /// companion the route asks for already in place.
    fn attachment(
        &mut self,
        p: &Prepared,
        addr: u64,
        writes: bool,
    ) -> std::result::Result<wgpu::TextureView, String> {
        match p.render {
            Render::Companion(shape) => self.companion(addr, shape, p.state.grid)?,
            // A surface drawn into directly has to have whatever a previous
            // draw left on a companion put back first — a frame is allowed to
            // change its mind about how it renders a surface, and the two
            // shapes are not the same pixels.
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

    /// One stage's bindings: its constant banks, and its textures with
    /// their samplers.
    fn bind_group(
        &mut self,
        p: &Prepared,
        stage: ShaderStage,
        group: u32,
    ) -> std::result::Result<(wgpu::BindGroupLayout, wgpu::BindGroup), String> {
        // The dimensionality is the layout's, because it is what the module
        // declared the binding as.
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
                .find(|b| b.immediate == upload.immediate)
                .ok_or("a texture the module never declared")?;
            let compare = declared_texture.compare;
            let dim = declared_texture.dim;
            let view_dimension = match dim {
                TexDim::T2dArray => wgpu::TextureViewDimension::D2Array,
                TexDim::T3d => wgpu::TextureViewDimension::D3,
                // Six faces of a 2D texture, viewed as one cube.
                TexDim::TCube => wgpu::TextureViewDimension::Cube,
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
            let texture = if compare {
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
        let format =
            device_texture_format(&self.device, upload.format).map_err(|e| format!("{e:?}"))?;
        let size = wgpu::Extent3d {
            width: upload.width.max(1),
            height: upload.height.max(1),
            depth_or_array_layers: upload.layers.max(1),
        };
        // A 3D image's slices are the same bytes an array's are, but the
        // texture has to be created as one: a `texture_3d` binding filters
        // between them, and a `D2Array` does not.
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
            &upload.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(upload.row_bytes),
                rows_per_image: Some(upload.rows),
            },
            size,
        );
        // Kept only alongside its bytes, which is what a guest write evicts;
        // a texture whose source could not be watched goes back to being one
        // draw's scratch, as everything was before this.
        if self.texture_cache.contains_key(&upload.key) {
            self.gpu_texture_bytes += upload.bytes.len() as u64;
            self.gpu_textures
                .entry(upload.key)
                .or_default()
                .push((view, texture.clone()));
        } else {
            self.scratch.push(Scratch::Texture(texture.clone()));
        }
        Ok(texture)
    }

    /// The buffers a stage's `ldg`s read, one per descriptor the translation
    /// tracked back to a constant bank.
    ///
    /// The address is not something the shader computed: it is a descriptor
    /// the driver wrote into a bank, so it can be read out of the bank this
    /// draw already uploads and the memory it names bound beside it. Eden
    /// does the same in `global_memory_to_storage_buffer`.
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

    /// A sampled depth image, which cannot be uploaded and so is *drawn*.
    ///
    /// WebGPU allows a copy into a `depth32float` only from another texture
    /// of the same format (§26.1.2.2), so the guest's shadow map goes into an
    /// `r32float` staging image and [`Gpu::load_depth`] — the same pass that
    /// puts a depth *target* on the device — writes it through `frag_depth`.
    /// That is the workaround the specification itself names.
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // Whatever the guest stored it as, the staging image is the one
        // `load_depth` reads: `r32float`, one value a texel.
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

    /// One `f32` a texel out of a sampled depth image: its red channel,
    /// which is the one a comparison reads.
    ///
    /// A shadow map is not always stored in a depth format — a title is free
    /// to render its depth into an ordinary colour surface and compare that,
    /// and A Short Hike does — so this reads whichever format the descriptor
    /// named. Red rather than the first byte, because `sample_compare_with`
    /// compares `texel[0]` of the *decoded* texel and the two have to agree.
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

    /// The sampler a texture is read through. `compare` is the *binding's*
    /// question, not the descriptor's: a `texs.dc` asks for a comparison
    /// whatever the TSC left in `depth_compare_enable`, and the rasterizer
    /// answers such a sample with `Always` — see `sample_compare_with`.
    fn sampler(
        &self,
        upload: &switch_core::gpu::upload::TextureUpload,
        compare: bool,
    ) -> wgpu::Sampler {
        use switch_core::gpu::texture::Wrap;
        let wrap = |w: Wrap| match w {
            Wrap::Repeat => wgpu::AddressMode::Repeat,
            Wrap::Mirror => wgpu::AddressMode::MirrorRepeat,
            Wrap::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            // WebGPU has no border mode at all -- `clamp-to-edge`, `repeat`
            // and `mirror-repeat` are the whole list -- and asking wgpu's web
            // backend for one is not an error it returns but a panic it
            // raises, which on wasm stops the core.
            //
            // Edge is also what this has to agree with: the rasterizer takes
            // `ClampToBorder` down the same arm as `ClampToEdge`
            // (`texture::Wrap`), so neither renderer samples a border colour
            // and the two match. When one of them learns to, they both must.
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
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler"),
            compare,
            address_mode_u: wrap(upload.sampler.wrap_u),
            address_mode_v: wrap(upload.sampler.wrap_v),
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: filter(upload.sampler.mag_linear),
            min_filter: filter(upload.sampler.min_linear),
            ..Default::default()
        })
    }

    /// A shader module, without asking the device whether it liked it.
    ///
    /// Asking means an error scope, and popping one means waiting. What the
    /// device rejects arrives through its uncaptured-error handler instead
    /// and is read at the start of the next draw — a frame late, which is
    /// what "do not block" costs and is cheap: the WGSL has already been
    /// through `naga` before it gets here.
    /// The module for this WGSL, with the cache key that named it — which is
    /// a hash of the source, and so identifies the module to a pipeline key
    /// as well.
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

    /// Every distinct rejection the device has raised, and how many it has
    /// raised in total — for the report, which is the only channel a browser
    /// has. Taking nothing: a rejection stays reportable for the whole run.
    fn device_errors(&self) -> (u64, Vec<String>) {
        match self.failed.lock() {
            Ok(e) => (e.count, e.distinct.clone()),
            Err(_) => (0, Vec::new()),
        }
    }

    /// A buffer holding `bytes`, for the draw in progress — see
    /// [`Gpu::scratch`].
    fn buffer(&mut self, what: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        // Padded to four bytes, which every buffer binding wants and a
        // twelve-byte index buffer is not.
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

    /// Decide how a draw reaches its surfaces.
    ///
    /// The expanded route is the default, and the reason is *where the
    /// samples are*. Maxwell puts them at the centres of the texels they are
    /// stored in, which is exactly where rendering the expanded surface one
    /// texel at a time tests coverage — so that route reproduces the
    /// rasterizer's frame texel for texel. WebGPU's sample positions are
    /// fixed by the spec at a rotated grid that is not Maxwell's and cannot
    /// be programmed, so the device's own multisampling anti-aliases every
    /// edge *differently*: correct, and not the reference.
    ///
    /// What the device's route buys is shading once per pixel instead of once
    /// per sample, which at `4x4` is sixteen times the fragment work. That is
    /// worth having and it is not worth having silently, so it is
    /// [`Gpu::device_msaa`] and it is off.
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
            // Every texel of a pixel's tile takes the same value, so a mask
            // that keeps some of them and not others has nothing to act on.
            let all = (1u64 << state.samples) - 1;
            if u64::from(state.sample_mask) & all != all || state.alpha_to_coverage {
                return Err("a draw with coverage per pixel and a mask that is per sample".into());
            }
            return Ok(Render::Companion(Shape::PerPixel));
        }
        let formats = [
            match color {
                Some(color) => Some(
                    device_texture_format(&self.device, color.format)
                        .map_err(|e| format!("{e:?}"))?,
                ),
                None => None,
            },
            depth.and_then(|d| d.depth_kind()).map(depth_texture_format),
        ];
        // Every attachment of a pass has to have the same sample count, so
        // one format the device will not multisample decides it for both.
        let offered = self.device_msaa
            && formats
                .into_iter()
                .flatten()
                .all(|format| self.samples_supported(format, state.samples));
        if offered {
            return Ok(Render::Companion(Shape::Multisampled(state.samples)));
        }
        // The expanded route tests coverage at texel centres, because that is
        // where a fragment is. A guest that has moved its samples somewhere
        // else inside their texels is asking for coverage neither route can
        // express — the device's positions are fixed by the spec and not
        // Maxwell's either — so this is a draw to hand back rather than one
        // to draw a fraction of a texel wrong.
        if !state.grid.samples_at_texel_centres() {
            return Err("a draw with programmed sample locations".into());
        }
        Ok(Render::Expanded)
    }

    /// Whether this adapter will render `samples` samples into `format`.
    ///
    /// Core WebGPU guarantees four and nothing else, and a browser offers
    /// exactly that; a native adapter usually adds two and eight. Which is
    /// why there are two ways to render a multisampled surface here — asking
    /// is how a draw picks one.
    fn samples_supported(&self, format: wgpu::TextureFormat, samples: u32) -> bool {
        // The adapter's answer is only the device's answer when the device
        // was given `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`. Without it a
        // device permits the counts the spec guarantees and no more — and
        // asking the adapter anyway is how a `2x1` draw built a two-sample
        // pipeline that the device rejected, silently, leaving the surface
        // exactly as empty as if nothing had been drawn.
        let features = self.device.features();
        let flags = if features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            self._adapter.get_texture_format_features(format).flags
        } else {
            format.guaranteed_format_features(features).flags
        };
        flags.sample_count_supported(samples)
    }

    /// Give the surface at `addr` a companion of `shape`, replacing whatever
    /// it has.
    ///
    /// A frame that changes its mind — the same surface drawn with per-sample
    /// coverage and then with per-pixel — is why replacing is a case rather
    /// than an error: what is already on the companion goes back into the
    /// expanded surface first, and the new one is gathered out of it.
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
            // No copy usage at all: a multisampled texture accepts none, and
            // everything that reads this one reads it through a shader.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // What is already in the surface has to be in the companion, or a
        // draw that blends against it — or tests depth against it — reads a
        // texture nothing has written.
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

    /// Drop the companion of the surface at `addr` without putting it back,
    /// for a caller that is about to overwrite every texel of the surface.
    fn discard_companion(&mut self, addr: u64) {
        if let Some(held) = self.held.get_mut(&addr) {
            if let Some(companion) = held.companion.take() {
                companion.texture.destroy();
            }
        }
    }

    /// Put what the companion of the surface at `addr` holds back into it,
    /// and stop holding it.
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
                .then(|| {
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Every texel of the destination is written, so
                            // there is nothing to preserve under it.
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
                depth_stencil_attachment: key.depth.then(|| {
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

    /// The pipeline for one resampling direction, built once per shape it
    /// moves between.
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
            .then(|| {
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

    /// Clear part or all of the surfaces a `ClearBuffers` names, on the
    /// device.
    ///
    /// A clear that covers a whole surface is the cheapest thing here and the
    /// most common: it is a load operation, and a surface about to be
    /// overwritten whole need not be uploaded at all — which is a megabyte a
    /// frame that used to cross the bus twice for nothing.
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
            // Nothing reads a surface that is about to be written whole, so
            // nothing has to be uploaded to write it.
            if blank {
                self.hold_blank(&target).map_err(|e| format!("{e:?}"))?;
                // Whatever a companion holds is about to be overwritten, so
                // it is dropped rather than scattered back into the surface
                // first. The next draw gathers a fresh one out of what the
                // clear leaves.
                self.discard_companion(target.addr);
            } else {
                self.hold(&target, ctx).map_err(|e| format!("{e:?}"))?;
                // A clear of part of the surface is not: what the companion
                // holds outside the rectangle survives it.
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
                    device_texture_format(&self.device, target.format)
                        .map_err(|e| format!("{e:?}"))?,
                ),
                None => None,
            },
            depth: depth
                .and_then(|(target, _)| target.depth_kind())
                .map(depth_texture_format),
            write_mask: color.map_or([true; 4], |(_, _, channels)| channels),
        };
        // A partial clear draws, so it needs its value where a shader can
        // read it. A whole one does not, and pays nothing for this.
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
            // A layered surface is `layer_stride` bytes further on, and
            // nothing here holds a surface by anything but its address.
            return Err(format!("a clear of layer {layer}"));
        }
        let slot = engine.render_target_slot(target);
        let Some(surface) = Target::color(engine, slot).map_err(|e| format!("{e:?}"))? else {
            // Nothing bound is nothing to clear, which is what the
            // rasterizer answers too.
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

    /// The rectangle a clear covers, in the surface's own texels.
    ///
    /// `clear_rectangle` answers in pixels, because that is what the scissor
    /// and the viewport it is cut against are in. On a multisampled surface a
    /// pixel is a tile of texels, and every one of them is cleared — so the
    /// rectangle scales, which is the same reading `Engine3D::clear_color`
    /// makes of it.
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

    /// Hold a surface without reading it, for a clear that is about to write
    /// every texel of it.
    fn hold_blank(&mut self, target: &Target) -> Result<()> {
        match self.held.get(&target.addr) {
            // Already here and already this surface: the clear writes it
            // where it is.
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
                device_texture_format(&self.device, target.format)?,
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

    /// The pipeline that clears a rectangle of these formats, built once per
    /// combination. See [`CLEAR_RECT_WGSL`].
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

    /// Give back everything the draw that has just finished made — whether it
    /// was submitted or handed to the rasterizer, nothing will read any of it
    /// again.
    /// Drop every cached texture the guest has written over.
    ///
    /// The pages come from the same bitmap the JIT drains, so this sees pages
    /// nothing here cached from as well; those find no owner and cost a miss
    /// in a hash map.
    fn evict_written(&mut self, ctx: &mut ExecCtx) {
        if !ctx.mem.has_dirty_gpu() {
            return;
        }
        for page in ctx.mem.dirty_gpu_pages() {
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

    /// Give a cached texture's device copy back, destroyed rather than merely
    /// dropped: in a browser a dropped texture waits on a collector, and this
    /// is a whole image.
    fn drop_gpu_texture(&mut self, key: &TextureKey, bytes: u64) {
        if let Some(made) = self.gpu_textures.remove(key) {
            for (_, texture) in made {
                texture.destroy();
                self.gpu_texture_bytes = self.gpu_texture_bytes.saturating_sub(bytes);
            }
        }
    }

    /// Keep what this draw read, and watch the memory it came from.
    ///
    /// The pages a texture's source covers are what a later write to it is
    /// noticed through, so a texture with none of them is not kept: its bytes
    /// could change with nothing to say so.
    fn remember_textures(&mut self, ctx: &mut ExecCtx) {
        for (key, bytes, source_len) in std::mem::take(&mut self.to_remember) {
            if self.texture_cache.contains_key(&key) {
                continue;
            }
            self.texture_misses += 1;
            // `source_len` is an upper bound on where a read could reach —
            // `dense.max(strided)` in `upload.rs` — and not the size of the
            // allocation, which a title routinely maps less of: Asphalt 9's
            // textures claim 16 MiB against a 6 MiB mapping.
            //
            // So the walk stops where the mapping does and keeps what it
            // found. Bytes that are not mapped cannot be written through this
            // address space, and the decode did not read them either — it
            // would have faulted rather than produced the image being cached,
            // and a faulted upload is a draw on the rasterizer, which
            // `fallbacks` would have counted. Requiring the whole bound
            // instead cost this title every texture it had: 240 of 242
            // decodes in a 40-frame run reproduced an image the cache had
            // already decoded and would not take.
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
            // Whole-cache rather than least-recently-used: the working set is
            // a handful of images, so a run that reaches this is one whose
            // textures have changed wholesale, and picking victims out of a
            // set that is all about to be replaced buys nothing.
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

/// The wgpu name for a format `switch_core::gpu::pipeline` resolved, checked
/// against what this device was actually given.
///
/// WebGPU gates the compressed families behind optional features, and creating
/// a texture in one the device does not have is not an error you can catch --
/// `createTexture` throws, and wgpu's web backend unwraps it. That is a panic
/// in the middle of a draw, which on wasm is a bare `unreachable` that takes
/// the whole core down: Just Dance 2019 reached its first BC1 texture and the
/// run loop stopped.
///
/// Refusing it here instead makes it what every other thing this backend
/// cannot express already is -- one draw on the software rasterizer.
fn device_texture_format(device: &wgpu::Device, format: Format) -> Result<wgpu::TextureFormat> {
    let wanted = texture_format(format)?;
    let needs = wanted.required_features();
    if !device.features().contains(needs) {
        return Err(Error::Gpu(format!(
            "the device was not given {needs:?}, which {wanted:?} needs"
        )));
    }
    Ok(wanted)
}

/// The wgpu name for a format `switch_core::gpu::pipeline` resolved.
fn texture_format(format: Format) -> Result<wgpu::TextureFormat> {
    use wgpu::TextureFormat as T;
    Ok(match format {
        Format::R8Unorm => T::R8Unorm,
        Format::Rg8Unorm => T::Rg8Unorm,
        Format::Rgba8Unorm => T::Rgba8Unorm,
        Format::Rgba8UnormSrgb => T::Rgba8UnormSrgb,
        Format::Bgra8Unorm => T::Bgra8Unorm,
        Format::Bgra8UnormSrgb => T::Bgra8UnormSrgb,
        Format::Rgb10a2Unorm => T::Rgb10a2Unorm,
        Format::R32Float => T::R32Float,
        Format::R16Float => T::R16Float,
        Format::R16Unorm => T::R16Unorm,
        Format::Rgba16Float => T::Rgba16Float,
        Format::Rgba32Float => T::Rgba32Float,
        Format::Depth16Unorm => T::Depth16Unorm,
        Format::Depth24Plus => T::Depth24Plus,
        Format::Depth24PlusStencil8 => T::Depth24PlusStencil8,
        Format::Depth32Float => T::Depth32Float,
        Format::Depth32FloatStencil8 => T::Depth32FloatStencil8,
        Format::Bc1RgbaUnorm => T::Bc1RgbaUnorm,
        Format::Bc1RgbaUnormSrgb => T::Bc1RgbaUnormSrgb,
        Format::Bc2RgbaUnorm => T::Bc2RgbaUnorm,
        Format::Bc2RgbaUnormSrgb => T::Bc2RgbaUnormSrgb,
        Format::Bc3RgbaUnorm => T::Bc3RgbaUnorm,
        Format::Bc3RgbaUnormSrgb => T::Bc3RgbaUnormSrgb,
        Format::Bc4RUnorm => T::Bc4RUnorm,
        Format::Bc4RSnorm => T::Bc4RSnorm,
        Format::Bc5RgUnorm => T::Bc5RgUnorm,
        Format::Bc5RgSnorm => T::Bc5RgSnorm,
        Format::Bc6hRgbUfloat => T::Bc6hRgbUfloat,
        Format::Bc6hRgbFloat => T::Bc6hRgbFloat,
        Format::Bc7RgbaUnorm => T::Bc7RgbaUnorm,
        Format::Bc7RgbaUnormSrgb => T::Bc7RgbaUnormSrgb,
    })
}

/// How a draw reaches the surfaces it renders into.
///
/// Only a multisampled surface has more than one answer. Its samples live in
/// guest memory as a tile of texels per pixel, and there are two ways to put
/// them there: let the device multisample and scatter the result, or render
/// the expanded image directly, one fragment per texel. Neither is a
/// compromise for the other — the first shades once per pixel, which is what
/// multisampling *is* and what the rasterizer does; the second works for
/// every mode on every adapter, which the first does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Render {
    /// Straight into the held surface, whose texels are its pixels.
    Direct,
    /// Into the held surface at texel resolution, one fragment per sample.
    /// The sample mask and alpha-to-coverage are the fragment shader's job
    /// here, because there is no multisample state to carry them.
    Expanded,
    /// Into a companion of this shape, gathered on the way in and scattered
    /// on the way out.
    Companion(Shape),
}

/// One draw, resolved into everything a device needs.
struct Prepared {
    state: Pipeline,
    /// How this draw reaches its surfaces.
    render: Render,
    /// `None` for a depth-only pass.
    color: Option<Target>,
    /// The depth surface the draw reads or writes, or `None` for a draw that
    /// does neither — which is not the same as a draw with no depth surface
    /// bound. A test of `Always` with writes off depends on nothing, so
    /// attaching the surface would upload it for no reason.
    depth: Option<Target>,
    vs: Translation,
    fs: Translation,
    vs_layout: Layout,
    fs_layout: Layout,
    uploads: Uploads,
    /// The memory each stage's `ldg`s read, by descriptor.
    globals: Vec<GlobalUpload>,
    /// Vertices for a sequential draw, indices for an indexed one — or, for
    /// a topology that had to be assembled, the length of
    /// [`Prepared::assembled`].
    count: u32,
    /// The index list that makes this draw a triangle list, for a topology
    /// WebGPU has no name for, paired with the base vertex to draw it with.
    ///
    /// Built with `raster::assemble`, which is the same call the rasterizer
    /// assembles a fan or a quad with — so the two cannot come to disagree
    /// about which triangles a quad is made of.
    assembled: Option<(Vec<u32>, i32)>,
    /// `gl_InstanceID`, which WebGPU reproduces as the first instance of a
    /// one-instance draw.
    instance: u32,
}

impl Gpu {
    /// Read a draw out of the engine, or say what stops it running here.
    ///
    /// Everything this reaches for is `switch-core`'s: the translation, the
    /// pipeline state and the uploads all already exist and are already
    /// tested against the rasterizer. What is left is arranging them.
    fn prepare(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
    ) -> std::result::Result<Prepared, String> {
        let state = Pipeline::of(engine).map_err(|e| e.to_string())?;
        let targets = Targets::of(engine).map_err(|e| format!("{e:?}"))?;
        let color = targets.color;
        // A draw that neither tests nor writes depth depends on the surface
        // not at all, and attaching it would upload a megabyte to be read
        // once and put straight back. Every other draw gets it.
        let uses_depth = state
            .depth
            .is_some_and(|d| d.write_enabled || d.compare != state::Compare::Always);
        let depth = targets.depth.filter(|_| uses_depth);
        // WebGPU wants every attachment of a pass to be the same size, and
        // the rasterizer does not — it addresses each surface with its own
        // extent and simply misses where they disagree. A depth surface
        // larger than the colour one is the case that arises, and the pass
        // only ever touches the part of it the colour target covers, so it
        // is attached cropped to that: the rest is read and written by
        // nobody, which is what missing it means.
        let mut depth = depth;
        if let (Some(color), Some(full)) = (color, depth) {
            if (color.width, color.height) != (full.width, full.height) {
                if full.width < color.width || full.height < color.height {
                    return Err(format!(
                        "a {}x{} colour target beside a smaller {}x{} depth one",
                        color.width, color.height, full.width, full.height
                    ));
                }
                // The stride stays the surface's, because that is what its
                // block-linear addressing is in terms of; only how much of
                // it the pass covers changes.
                depth = Some(Target {
                    width: color.width,
                    height: color.height,
                    rows: color.height,
                    ..full
                });
            }
        }
        if color.is_none() && depth.is_none() {
            return Err("a draw into neither a colour nor a depth surface".into());
        }
        let render = self.route(&state, color, depth)?;

        // Unfolded, so a module depends only on the shader binary and not on
        // what happened to be in a constant buffer when it was translated.
        let (vs, fs) = timed!(self, translate, {
            (
                self.translate(engine, ctx, ShaderStage::VertexB),
                self.translate(engine, ctx, ShaderStage::Fragment),
            )
        });
        let (vs, fs) = (vs?, fs?);

        let mut vs_layout = Layout::of(&vs, Stage::Vertex);
        let mut fs_layout = Layout::of(&fs, Stage::Fragment);
        // A depth-only pass has nowhere to put a colour, and a fragment
        // shader that names `@location(0)` with no attachment behind it is a
        // pipeline that will not build.
        fs_layout.targets = u32::from(color.is_some());
        // The two stages have to name the same varyings: WebGPU will not
        // link a fragment input nothing produces. The union is what agrees
        // with the rasterizer, where a varying the vertex shader never wrote
        // reads as zero rather than as absent.
        let mut varyings = vs_layout.varyings.clone();
        varyings.extend(fs_layout.varyings.iter().copied());
        varyings.sort_unstable();
        varyings.dedup();
        vs_layout.varyings = varyings.clone();
        fs_layout.varyings = varyings;
        // And on the same qualifier for each of them. Only the fragment
        // program's `ipa` says which are sampled at the centroid, so the
        // vertex stage is told rather than asked.
        vs_layout.centroid_varyings = fs_layout.centroid_varyings.clone();
        // Neither correction is anything the program says. Negated, because
        // WebGPU mirrors y on its own: the two agree exactly when the
        // guest's transform mirrors too, and the shader has to do it when
        // the guest's does not. See `Layout::flip_y`.
        vs_layout.flip_y = !state.viewport.flip_y;
        vs_layout.depth_minus_one_to_one = state.viewport.depth_minus_one_to_one();
        // Rendering the expanded surface directly makes every texel its own
        // fragment, so there is no multisample state to carry the sample mask
        // or alpha-to-coverage and the shader does both. `raster` applies the
        // mask before shading and the coverage after, and so does this.
        if render == Render::Expanded {
            fs_layout.coverage = Some(Coverage {
                samples_x: state.grid.samples_x,
                samples_y: state.grid.samples_y,
                sample_of_slot: state.grid.sample_of_slot()[..state.samples as usize].to_vec(),
                sample_mask: state.sample_mask,
                alpha_to_coverage: state.alpha_to_coverage,
            });
        }
        // Nor is which attributes are integers: the format is in the draw's
        // registers, and WebGPU will not feed one to a `vec4<f32>` input.
        vs_layout.integer_attributes = state
            .vertex_buffers
            .iter()
            .flat_map(|buffer| &buffer.attributes)
            .filter(|a| a.format.base() != AttributeBase::Float)
            .map(|a| (a.location as usize, a.format.base()))
            .collect();
        // Nor is which of them are BGRA. WebGPU has no BGRA vertex format,
        // so the swap happens in the entry point — which is where
        // `raster::fetch_attribute` does it too.
        vs_layout.bgra_attributes = state
            .vertex_buffers
            .iter()
            .flat_map(|buffer| &buffer.attributes)
            .filter(|a| a.is_bgra)
            .map(|a| a.location as usize)
            .collect();
        // One bind group per stage; see `Layout::group`.
        vs_layout.group = 0;
        fs_layout.group = 1;

        let mut immediates: Vec<(ShaderStage, u16)> = Vec::new();
        immediates.extend(
            vs.textures
                .iter()
                .map(|&(imm, _, _)| (ShaderStage::VertexB, imm)),
        );
        immediates.extend(
            fs.textures
                .iter()
                .map(|&(imm, _, _)| (ShaderStage::Fragment, imm)),
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
        // Out of `self` and back, because the closure below holds it while
        // `timed!` wants `self` for the clock.
        let cache = std::mem::take(&mut self.texture_cache);
        let mut hits = 0u64;
        let uploads = timed!(self, upload, {
            Uploads::of_cached(
                engine,
                &state,
                ctx,
                Banks::Read(&banks),
                &immediates,
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

        // The swizzle is in the descriptor, which is guest memory the draw
        // points at, so the translation cannot know it and the layout has to
        // be told. WebGPU has no per-texture component swizzle.
        for (layout, stage) in [
            (&mut vs_layout, ShaderStage::VertexB),
            (&mut fs_layout, ShaderStage::Fragment),
        ] {
            for binding in &mut layout.textures {
                if let Some(upload) = uploads
                    .textures
                    .iter()
                    .find(|t| t.stage == stage && t.immediate == binding.immediate)
                {
                    binding.swizzle = upload.swizzle;
                }
            }
        }

        // A fan, a quad strip and a polygon are a triangle list once their
        // indices have been rewritten, which is what `Pipeline::expand` names
        // and what the rasterizer does to them too.
        let assembled = match state.expand {
            Some(primitive) => {
                let triangles =
                    switch_core::gpu::raster::assemble(primitive, engine.last_draw.count);
                let mut indices = Vec::with_capacity(triangles.len() * 3);
                match &uploads.index {
                    // An indexed draw's triples are positions in its index
                    // list, so each one names the index to draw with. The
                    // base vertex is the one the unexpanded draw would have
                    // used: the vertex buffer starts at the lowest index.
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
                    // A sequential draw's triples are vertex ordinals, and
                    // the upload starts at the draw's first vertex — so the
                    // ordinals are already what to draw with.
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

        if switch_core::env_flag!("TRACE_GPU_TEX") {
            for t in &uploads.textures {
                eprintln!(
                    "[gpu-tex] imm={} {:?} {}x{} swizzle={:?} sampler={:?}",
                    t.immediate, t.format, t.width, t.height, t.swizzle, t.sampler
                );
            }
        }
        Ok(Prepared {
            state,
            render,
            color,
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
        &self,
        engine: &Engine3D,
        ctx: &ExecCtx,
        stage: ShaderStage,
    ) -> std::result::Result<Translation, String> {
        let binding = engine
            .program(stage)
            .ok_or_else(|| format!("no {stage:?} program"))?;
        let program =
            switch_core::gpu::shader::decode_program_from_memory(ctx, binding.addr, &|bank: u8| {
                engine.bound_constbuf(stage, u32::from(bank))
            })
            .map_err(|e| format!("{e:?}"))?;
        let caps = wgsl::Caps {
            subgroups: self.device.features().contains(wgpu::Features::SUBGROUP),
            // On the web the browser compiles the text and wants the
            // directive; natively naga does, and rejects it.
            subgroup_enable: cfg!(target_arch = "wasm32"),
        };
        wgsl::translate_for(&Compiled::new(&program), caps).map_err(|e| e.to_string())
    }
}

impl Renderer for Gpu {
    fn draw(&mut self, engine: &Engine3D, ctx: &mut ExecCtx) -> Result<()> {
        if self.give_up() {
            return self.software.draw(engine, ctx);
        }
        // Anything at all going wrong here runs the draw on the rasterizer
        // instead. That is not timidity: the rasterizer is the reference, so
        // a frame is always either right or a frame the reference produced,
        // and `cmp` against it measures how much of the work this actually
        // did rather than how much it got away with.
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
        // A frame the device is not rendering all of, it is not rendering any
        // of — see [`Gpu::software_frame`]. Nothing is held, so the flush
        // here has nothing to hand back after the first draw of it.
        if self.software_frame {
            self.flush(ctx)?;
            return self.software.draw(engine, ctx);
        }
        // Anything the guest wrote since the last draw is no longer what was
        // read from it, whoever wrote it and whatever they meant by it.
        self.evict_written(ctx);
        let mut route = None;
        let attempt = match self.prepare(engine, &*ctx) {
            Ok(prepared) => {
                route = Some(prepared.render);
                self.render(&prepared, ctx)
            }
            Err(why) => Err(why),
        };
        // What `prepare` read, now that there is a mutable `ExecCtx` to watch
        // its pages through. After the draw rather than before: a draw that
        // failed read the same bytes, and they are as reusable either way.
        self.remember_textures(ctx);
        // Whether it was submitted or abandoned, nothing reads this draw's
        // buffers and textures again — see [`Gpu::scratch`].
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
        // The rasterizer reads and writes guest memory, so it has to be the
        // truth before a draw runs there.
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
        // A frame starts at its clear, which is where the last one's answer
        // becomes this one's decision.
        self.in_frame = 0;
        if self.deferred_readbacks
            && self.fell_back_this_frame
            && !self.software_frame
            && !self.interleave
        {
            self.software_frame = true;
            eprintln!(
                "[gpu] a draw fell back where a readback lands later than the call that \
                 asked for it; the rasterizer has every frame from here. The fallbacks \
                 are opcodes `shader::wgsl` does not translate: {:?}",
                self.reasons
            );
        }
        self.fell_back_this_frame = false;
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
                // The rasterizer writes guest memory, so anything held has to
                // go back before it does — or the clear would be overwritten
                // by a surface handed back after it.
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
        // The device holds no stencil at all — `depth32float` and
        // `depth16unorm` are the two formats a readback can reach, and
        // neither carries one. So a stencil clear goes straight to guest
        // memory, which is where the stencil byte lives and stays: nothing
        // here writes it, and `Target::write_depth` reads it back and puts it
        // where it was. No flush is owed, because the two never touch the
        // same bits.
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
        // `gave_up` and not `software_frame`: the latch is a decision about
        // this device, which is working, and replacing it would throw away a
        // warm cache to reach the same conclusion. A lost device is the one
        // condition a fresh one actually fixes.
        self.gave_up
    }

    fn report_json(&self) -> String {
        // The same numbers `Drop` prints, except that a browser never sees
        // those: the module outlives the page's interest in it, and stderr
        // goes nowhere. `software_frame` is the one that matters most — once
        // it latches, every frame after it is the rasterizer's however well
        // the device is working.
        let ms = |v: u128| v as f64 / 1000.0;
        // Nested rather than flattened: the phase that builds modules and the
        // count of modules built are both called "modules", and one JSON
        // object cannot hold that name twice — `JSON.parse` keeps whichever
        // came last and drops the other without saying so.
        let times = match self.times {
            Some(t) => format!(
                ",\"times\":{{\"translate\":{:.1},\"upload\":{:.1},\"modules\":{:.1},\
                 \"pipeline\":{:.1},\"encode\":{:.1},\"flush\":{:.1}}}",
                ms(t.translate),
                ms(t.upload),
                ms(t.modules),
                ms(t.pipeline),
                ms(t.encode),
                ms(t.flush),
            ),
            None => String::new(),
        };
        let reasons: Vec<String> = self.reasons.iter().map(|why| json_string(why)).collect();
        // A rejection is not a fallback: the backend does not learn about one
        // until it next asks, so a frame can be counted as wholly the
        // device's and still be wrong. These two are the only evidence of
        // that, and a browser sees no stderr.
        let (error_count, errors) = self.device_errors();
        let errors: Vec<String> = errors.iter().map(|e| json_string(e)).collect();
        // `held` is what a flush costs: `flush_inner` writes back every
        // surface in it, every time, so this growing is the flush time
        // growing. Nothing caps it — a title that renders to fresh addresses
        // accumulates them — and the count is the only way to see that from
        // outside.
        format!(
            "{{\"backend\":\"device\",\"drawn\":{},\"fallbacks\":{},\"pipelines\":{},\
             \"modules\":{},\"held\":{},\"evicted\":{},\"pending\":{},\
             \"read\":{{\"textures\":{},\"vertex\":{},\"constants\":{},\"index\":{}}},\
             \"textureHits\":{},\"textureMisses\":{},\
             \"softwareFrame\":{},\"gaveUp\":{},\"reasons\":[{}],\
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
            self.software_frame,
            self.gave_up,
            reasons.join(","),
            error_count,
            errors.join(","),
            times,
        )
    }
}

/// One JSON string literal. A fallback reason is an error message and is free
/// to contain a quote or a backslash; the page parses this with `JSON.parse`,
/// which is entitled to reject the whole object over one of them.
fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl Gpu {
    fn flush_inner(&mut self, ctx: &mut ExecCtx) -> Result<Flush> {
        if self.give_up() {
            // The first flush after the loss carries the reason out; every one
            // after it says the frame is ready, because guest memory is now
            // the whole truth and there is nothing left to wait for.
            return match self.report.take() {
                Some(why) => Err(Error::Gpu(why)),
                None => Ok(Flush::Done),
            };
        }
        // Nothing on the device, nothing owed, nothing in flight: guest
        // memory cannot disagree with a backend holding no surfaces, so there
        // is nothing to wait for and nothing to land.
        //
        // Worth checking because a flush is not once a frame. It also runs
        // before every fallback draw, and `flush_one` empties `held` as it
        // writes back — so every flush after a frame's first one reaches this
        // with all three empty, and used to poll the device to find that out.
        // On the web that poll cannot even do anything (callbacks come from
        // the event loop) and still costs a crossing: a browser trace with
        // *no draws at all* charged 1,755 ms to flush.
        if self.held.is_empty() && self.evicted.is_empty() && self.pending.is_empty() {
            return Ok(Flush::Done);
        }
        // Only once per frame: asking again while the first ask is in flight
        // would copy the same surface twice and read the second copy.
        if self.pending.is_empty() {
            for held in std::mem::take(&mut self.evicted) {
                self.ask_for(held);
            }
            let addresses: Vec<u64> = self.held.keys().copied().collect();
            for addr in addresses {
                self.flush_one(addr);
            }
        }
        // Let the device run its callbacks.
        //
        // `Wait` rather than `Poll`, which is not a change of mind about
        // blocking: on WebGPU it has no effect at all — callbacks are invoked
        // from the event loop and nothing here can make that happen — so the
        // browser still gets `Flush::Pending` and the present still waits for
        // a later slice. Natively it blocks until the copies are done, which
        // is what the one caller that can afford to block actually wants.
        //
        // It matters because a flush is *also* what runs before a draw hands
        // itself to the rasterizer, and the rasterizer reads guest memory. A
        // flush that answered "not yet" there left the draw reading whatever
        // was in memory before the device drew, and the readback then landed
        // on top of what it wrote. The timeout is so that a submission that
        // never completes is a dropped frame rather than a hung emulator.
        //
        // **This leaves the browser half-fixed, and the other half is not a
        // browser limit.** There, the map still completes only when the event
        // loop runs, so a draw that falls back mid-slice still reads stale
        // memory. The browser-native fix is to *yield*: the backend says it
        // needs the event loop, the channel suspends the pushbuffer at that
        // method and `switch_run` returns, and the slice after it resumes with
        // the readback landed. That is a change to how a channel is driven
        // rather than anything WebGPU withholds — and the fallback set it
        // matters for is itself a shortfall in `shader::wgsl`, not a fact
        // about the platform.
        // `GPU_DEFER_READBACKS=1` declines the wait, so a native run behaves
        // the way a browser does — the map completes on some later call
        // rather than this one. It is how the browser-only half of this is
        // measured at all.
        let _ = if self.defer_readbacks {
            self.device.poll(wgpu::PollType::Poll)
        } else {
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
        };
        use std::sync::atomic::Ordering;
        if self
            .pending
            .iter()
            .any(|p| p.state.load(Ordering::Acquire) == MAP_WAITING)
        {
            // The one place that learns a readback does not land inside the
            // call that asked for it. From here on a frame is all one
            // renderer's — see [`Gpu::deferred_readbacks`].
            self.deferred_readbacks = true;
            return Ok(Flush::Pending);
        }
        for pending in std::mem::take(&mut self.pending) {
            if pending.state.load(Ordering::Acquire) == MAP_FAILED {
                // With the reason, if the device left one: on its own this
                // message names the symptom and nothing else, and the cause is
                // in the browser rather than in the frame.
                return Err(Error::Gpu(match self.device_error() {
                    Some(e) => format!("the readback was not mapped: {e}"),
                    None => "the readback was not mapped".into(),
                }));
            }
            self.land(&pending, ctx)?;
        }
        Ok(Flush::Done)
    }
}

#[cfg(test)]
mod tests {
    /// Why `shader::wgsl` ends its dispatch function with a `return false;`
    /// nothing can reach — and the check that will say when it can go.
    ///
    /// The dispatch loop has no `break`, so control cannot fall out of it.
    /// Chrome's Tint knows that and warns `code is unreachable` about the
    /// statement after it, twice per shader module. naga, which validates the
    /// same WGSL for every native backend, rejects the function without it.
    /// The two disagree, native has to compile, so the statement stays and the
    /// warning is what it costs.
    ///
    /// The second assertion is the interesting one: when naga learns what Tint
    /// already knows, this fails, and the trailing statement — and the console
    /// full of warnings — can go.
    #[test]
    fn naga_still_needs_a_return_after_a_loop_that_cannot_fall_through() {
        let Ok(gpu) = super::Gpu::open() else { return };
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

    /// A rejection the backend never asks about still reaches the report.
    ///
    /// The failure this guards is what made a magenta frame in the browser
    /// read as `0 fell back`: the only production reader of `device_error`
    /// runs before a pipeline is built, and a title that builds its pipelines
    /// in the first frames never builds another. Everything the device
    /// rejected from then on was captured and never looked at, and the report
    /// had nowhere to put it.
    #[test]
    fn a_rejection_nothing_asked_about_is_still_counted_and_reported() {
        let Ok(gpu) = super::Gpu::open() else { return };
        assert_eq!(gpu.device_errors(), (0, Vec::new()), "nothing rejected yet");

        // Rejected for a reason that cannot become valid: `bool` is not a
        // fragment return type WebGPU accepts at a location.
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
        assert!(count >= 1, "the device rejected the module and said nothing");
        assert_eq!(distinct.len(), 1, "one rejection, one distinct message");
        // Not taken: `device_error` is what drains, and the report has to keep
        // saying so for the rest of the run.
        assert_eq!(gpu.device_errors(), (count, distinct.clone()));

        // `report_json` is the renderer's, and the page reads it through that
        // trait rather than through this type.
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

    /// Every internal pipeline this backend builds for itself, built.
    ///
    /// These are the passes a draw never mentions and a title cannot be asked
    /// to exercise: the one that puts a depth surface on the device, the two
    /// that clear part of one, and the four that move a multisampled surface
    /// between its expanded form and a device companion. All of them are WGSL
    /// this crate generates, and until this ran the only thing that compiled
    /// them was a frame — where a rejected module is a silent fallback rather
    /// than a failure, because nothing here asks the device whether it liked
    /// what it was given.
    use switch_core::gpu::renderer::Software;
    use switch_core::gpu::testing::{self, Harness};

    /// A solid white triangle over one of Maxwell's multisample modes,
    /// rendered by the rasterizer and by the device, as two pictures of the
    /// same expanded surface.
    ///
    /// Solid rather than interpolated on purpose: a colour that is the same
    /// at all three vertices interpolates to itself, so the only thing left
    /// to disagree about is *coverage* — which sample of which pixel the
    /// triangle reached, and which texel of guest memory that sample is. That
    /// is the whole of what multisampling is, and the whole of what the two
    /// routes through it have to get right.
    fn compare(mode: u32, samples_x: u32, samples_y: u32, set_up: impl Fn(&mut Harness)) {
        let Ok(mut gpu) = super::Gpu::open() else {
            return;
        };
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
        // Nothing in a draw asks the device whether it liked what it was
        // given — that would mean waiting — so a rejection is silent until
        // the next draw reads it. A test is the one place that can afford to
        // ask, and a rejected pass looks exactly like a surface nothing drew
        // into.
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert_eq!(gpu.device_error(), None, "the device rejected the pass");
        assert_eq!(
            got, want,
            "mode {mode} ({samples_x}x{samples_y}) came out differently on the device"
        );
    }

    /// One draw, set up by `set_up`, rendered by the rasterizer and by the
    /// device — colour and depth both.
    fn agrees(set_up: impl Fn(&mut Harness)) {
        let Ok(mut gpu) = super::Gpu::open() else {
            return;
        };
        let build = |gpu: Option<&mut super::Gpu>| {
            let mut h = Harness::new();
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

    /// A depth-tested draw, which is what this backend could not do at all
    /// before it held a depth buffer.
    ///
    /// The depth surface is checked as well as the colour one, because it is
    /// guest memory the rasterizer owns: a frame that comes out right with a
    /// depth buffer left untouched is a frame the *next* draw gets wrong.
    #[test]
    fn a_depth_tested_draw_writes_the_same_depth_the_rasterizer_writes() {
        // Less, less-equal, greater and always, in the numbering deko3d
        // writes — the four a title actually uses.
        //
        // The colour is ones and zeros because these tests are about depth:
        // `mufu rcp` is a hardware approximation and WGSL's `1.0 / x` is not
        // the same approximation, so an interpolated channel can land a
        // 255th either side of a rounding boundary. A half is exactly such a
        // boundary; a one and a zero are not near one.
        for func in [0x0201, 0x0203, 0x0204, 0x0207] {
            agrees(move |h| {
                h.depth_target(func);
                h.triangle([1.0, 0.0, 1.0, 1.0]);
            });
        }
    }

    #[test]
    fn a_depth_only_pass_still_writes_depth() {
        // No colour target at all, which Just Dance 2017 renders every pass
        // as. The fragment shader has nowhere to put its colour and still has
        // to run.
        agrees(|h| {
            h.depth_target(0x0207);
            // Unbind colour target 0: an address of zero is no surface.
            h.engine.regs.set(0x200, 0);
            h.engine.regs.set(0x201, 0);
            h.triangle([1.0, 1.0, 1.0, 1.0]);
        });
    }

    #[test]
    fn a_triangle_fan_is_assembled_the_way_the_rasterizer_assembles_one() {
        // WebGPU has no fan, and the index rewriting that turns one into a
        // triangle list is `raster::assemble` — the same call the rasterizer
        // makes, so there is nothing for the two to disagree about.
        for primitive in [6, 9] {
            agrees(move |h| {
                h.engine.last_draw.primitive = primitive;
                h.depth_target(0x0207);
                h.triangle([1.0, 0.0, 1.0, 1.0]);
            });
        }
    }

    #[test]
    fn an_instanced_array_reads_this_instance_and_not_the_first() {
        // WebGPU fetches an instanced array at the absolute instance index,
        // and the upload holds one element — the one this instance reaches.
        // A stride of nothing is what makes those the same thing; without it
        // the draw read past the end of a sixteen-byte buffer.
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
        // WebGPU has no BGRA vertex format, so the swap happens in the entry
        // point. Red and blue are the two it exchanges, so a colour with one
        // and not the other is what tells the two apart.
        agrees(|h| {
            h.depth_target(0x0207);
            h.triangle([1.0, 0.0, 0.0, 1.0]);
            let raw = h.engine.regs.get(0x459);
            h.engine.regs.set(0x459, raw | 1 << 31);
        });
    }

    /// What a host whose readbacks land late does with a frame the device
    /// cannot render all of.
    ///
    /// Natively a flush waits, so this never arises and a frame interleaves
    /// freely. In a browser the map completes from the event loop and nothing
    /// inside a run slice can make that happen — so a mid-frame fallback read
    /// guest memory the device had not written back yet, and the readback
    /// then landed on top of what the rasterizer wrote. The frame after such
    /// a fallback is the rasterizer's whole, and so is every frame after it.
    #[test]
    fn a_frame_the_device_cannot_finish_is_a_frame_it_does_not_start() {
        let Ok(mut gpu) = super::Gpu::open() else {
            return;
        };
        // What a browser teaches it on its first present.
        gpu.deferred_readbacks = true;
        let colour = [1.0f32, 0.0, 1.0, 1.0];

        let mut h = Harness::new();
        h.triangle(colour);
        h.clear_with(&mut gpu, [true; 4]).expect("the clear");
        assert!(!gpu.software_frame, "nothing has fallen back yet");
        // A line loop is a topology neither renderer draws and no pipeline
        // can describe, so this is a draw that must fall back.
        h.engine.last_draw.primitive = 2;
        // The rasterizer will not draw one either, and says so — which is
        // what a fallback landing somewhere that also refuses looks like. The
        // fallback is the part under test.
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

        // What guest memory holds is the frame the rasterizer draws — read
        // before anything clears it again.
        let got = h.target();
        let mut want = Harness::new();
        want.triangle(colour);
        want.clear_with(&mut Software, [true; 4])
            .expect("the clear");
        want.draw_with(&mut Software).expect("the draw");
        assert_eq!(got, want.target());

        // It latches: a frame that renders nothing on the device cannot
        // discover that the next one would have been fine, and alternating is
        // the one behaviour this must not have.
        h.clear_with(&mut gpu, [true; 4]).expect("the clear");
        assert!(gpu.software_frame, "the decision came undone");
    }

    #[test]
    fn a_clear_writes_what_the_rasterizer_would_have_written() {
        // Clears used to go to the rasterizer, which meant handing every
        // surface back first — a whole frame's readback, at every clear.
        let Ok(mut gpu) = super::Gpu::open() else {
            return;
        };
        for channels in [[true; 4], [true, false, true, false], [false; 4]] {
            let build = |gpu: Option<&mut super::Gpu>| {
                let mut h = Harness::new();
                // A colour with a channel in each of the four, so a masked
                // clear has one to leave alone.
                //
                // Not a half anywhere: `0.5` is `127.5` in eight bits, and
                // the two renderers break that tie in opposite directions —
                // `ColorFormat::encode` rounds it up and a device's unorm
                // conversion rounds it down. It is a 255th, it is real, and
                // it is not what this test is about.
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
        // `fetch_attribute` answers `(0, 0, 0, 1)` for a fixed attribute, and
        // the pipeline needs *something* bound to every location the shader
        // declares — so the backend feeds it a constant rather than handing
        // the draw back.
        agrees(|h| {
            h.depth_target(0x0207);
            h.triangle([1.0, 1.0, 1.0, 1.0]);
            // VertexAttribState[1], the colour: fixed, so no buffer feeds it.
            let raw = h.engine.regs.get(0x459);
            h.engine.regs.set(0x459, raw | 1 << 6);
        });
    }

    /// Every multisample mode, over whichever of the two routes this adapter
    /// puts it down.
    ///
    /// Both are exercised on any real adapter: four samples is the count core
    /// WebGPU guarantees, and sixteen is one nothing offers — so `4x4` takes
    /// the expanded route here whatever the machine, and `2x2` takes the
    /// device's own.
    #[test]
    fn a_multisampled_draw_reaches_the_same_texels_the_rasterizer_reaches() {
        for (mode, x, y) in [(1, 2, 1), (2, 2, 2), (3, 4, 2), (6, 4, 4)] {
            compare(mode, x, y, |_| {});
        }
    }

    /// The other route: the device doing the multisampling.
    ///
    /// It cannot be checked against the reference texel for texel, and that
    /// is the point of it being off by default — WebGPU's sample positions
    /// are a rotated grid the spec fixes, Maxwell's are the texel centres,
    /// and an edge falls differently under the two. What *must* still hold is
    /// the thing multisampling promises: a pixel the triangle covers
    /// completely is completely covered, one it misses entirely is untouched,
    /// and only the pixels an edge crosses are free to differ.
    #[test]
    fn the_device_route_agrees_wherever_an_edge_is_not() {
        let Ok(mut gpu) = super::Gpu::open() else {
            return;
        };
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
                // This device does not offer that many samples, so the draw
                // went the expanded way and the strict test already covers it.
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
        // A device turns alpha into a coverage mask its own way, and the
        // rasterizer keeps a prefix of `round(alpha * count)` samples. They
        // agree on nothing in between, so this is the expanded route's
        // arithmetic being checked against the reference — and at 4x4, which
        // no adapter multisamples, that is the only route there is.
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

    /// `AntiAliasEnable` off over a surface that still has a tile of texels
    /// per pixel: coverage is whole pixels, and every texel of a covered
    /// pixel gets the answer.
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
        let Ok(mut gpu) = super::Gpu::open() else {
            return;
        };
        // A surface format that is certainly multisampled, and the two depth
        // formats a readback can reach.
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

        // Four samples is the one count core WebGPU guarantees, so it is the
        // one a test may assume an adapter has.
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
                // Into a device multisample companion, and into a
                // one-sample-per-pixel one.
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

    /// The grid tables the resampling passes read, which are the only thing
    /// standing between a sample and the texel it belongs in.
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
            // And the inverse really is one: the slot this sample sits in
            // names this sample back.
            assert_eq!(word(34 + (y * 2 + x) as usize), sample);
        }
    }

    #[test]
    fn there_is_a_device_to_render_on() {
        match super::Gpu::open() {
            Ok(gpu) => println!("[gpu] opened, max texture {}", gpu.describe()),
            Err(why) => println!("[gpu] {why}"),
        }
    }

    /// A fallback reason is an error message and can hold anything. The page
    /// parses the whole report with `JSON.parse`, which rejects the entire
    /// object over one unescaped quote — so the counters would vanish because
    /// of a string beside them.
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
