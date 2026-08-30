//! The shaders the backend writes for itself.
//!
//! None of these come from the guest. They are the passes a device needs that
//! a Maxwell draw does not describe: putting a depth surface the guest wrote
//! by hand back onto the device, clearing part of a surface, and moving a
//! surface between the sample grid it is stored on and the one a pass renders
//! at.

use switch_core::gpu::surface::{SampleGrid, MAX_SAMPLES};

/// The pass that puts a guest depth surface onto the device.
///
/// `depth32float` is a format a copy may read out of and never write into,
/// so the only way in is to draw it: a fullscreen triangle whose fragment
/// reads the texel that was uploaded to an ordinary `r32float` texture and
/// reports it as its own depth. `depth16unorm` needs none of this — it is
/// the one depth format a copy may write — but it goes the same way, because
/// two paths that must agree about a surface's contents are one more place
/// for them to disagree than there needs to be.
pub(crate) const LOAD_DEPTH_WGSL: &str = "\
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
pub(crate) const CLEAR_RECT_WGSL: &str = "\
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
pub(crate) fn resample_wgsl(sampled: &str, load: &str, depth: bool) -> String {
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
pub(crate) struct ResampleKey {
    /// Which fragment entry point of [`resample_wgsl`] runs.
    pub(crate) entry: &'static str,
    /// The destination's format, which is also the source's — a companion is
    /// the same format as the surface it stands in for.
    pub(crate) dst: wgpu::TextureFormat,
    /// The destination's sample count.
    pub(crate) samples: u32,
    /// Whether the source is a device multisample texture.
    pub(crate) ms_source: bool,
    pub(crate) depth: bool,
}

/// The grid, as the storage buffer [`resample_wgsl`] reads.
pub(crate) fn grid_bytes(grid: SampleGrid) -> Vec<u8> {
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
