//! Vertex fetch, primitive assembly, rasterization, and the fragment-shader
//! integration that turns coverage into real pixels.
//!
//! [`draw`] is the top-level entry point `Engine3D::draw_arrays`/
//! `draw_elements` call. Everything it calls is independently testable
//! against synthetic inputs, which is how each earlier stage validated its
//! own piece before this stage wired them together.
//!
//! Coverage is per *sample* and shading is per *pixel*, which is what makes
//! this multisampling rather than rendering the whole frame at the sample
//! grid's resolution — see [`crate::gpu::surface::SampleGrid`] for how a
//! sample becomes a texel. The sample mask and alpha-to-coverage narrow that
//! coverage; `MultisampleCoverageToColor` and the target-independent
//! rasterization `SetMultisampleRasterEnable` turns on are not implemented,
//! and content that switches either on will draw as though it had not.

use crate::gpu::engine::threed::{
    BlendTarget, CullState, DepthState, DepthTarget, Engine3D, RenderTarget, ScissorRect,
    ShaderStage, VertexArray, VertexAttrib, ViewportTransform,
};
use crate::gpu::exec::ExecCtx;
use crate::gpu::shader::interp::{
    resolve_shuffles, ConstantSource, Env, Halt, Invocation, MemoryConstants, MemoryGlobal,
    MemoryTextures, NoTextures,
};
use crate::gpu::shader::compiled::Compiled;
use crate::gpu::shader::{decode_program_from_memory, Op, Program};
use crate::gpu::surface::{f16_to_f32, ColorFormat, SampleGrid, MAX_SAMPLES};
use crate::{Error, Result};

/// The `DkPrimitive` topologies this rasterizer assembles (deko3d.h,
/// devkitPro/libnx, MIT). Point and line topologies are recognised so a draw
/// that uses them is reported as such rather than as an unknown number;
/// nothing here rasterizes them yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
    Quads,
    QuadStrip,
    Polygon,
}

impl Primitive {
    pub fn from_raw(raw: u32) -> Result<Primitive> {
        match raw {
            0 => Ok(Primitive::Points),
            1 => Ok(Primitive::Lines),
            2 => Ok(Primitive::LineLoop),
            3 => Ok(Primitive::LineStrip),
            4 => Ok(Primitive::Triangles),
            5 => Ok(Primitive::TriangleStrip),
            6 => Ok(Primitive::TriangleFan),
            7 => Ok(Primitive::Quads),
            8 => Ok(Primitive::QuadStrip),
            9 => Ok(Primitive::Polygon),
            other => Err(Error::Gpu(format!("raster: unknown DkPrimitive {other}"))),
        }
    }
}

/// Break a `count`-vertex draw into triangles, as vertex-ordinal triples.
///
/// `TriangleStrip` alternates the first two indices of odd triangles so
/// every triangle in the strip winds the same way, which matters once
/// back-face culling is on. `Quads`/`QuadStrip`/`Polygon` fan out into
/// triangles the way the fixed-function pipeline does. Point and line
/// topologies produce nothing: they need their own rasterization, and
/// silently turning them into triangles would draw the wrong thing.
pub fn assemble(primitive: Primitive, count: u32) -> Vec<[u32; 3]> {
    match primitive {
        Primitive::Points | Primitive::Lines | Primitive::LineLoop | Primitive::LineStrip => {
            Vec::new()
        }
        Primitive::Triangles => (0..count / 3).map(|t| [t * 3, t * 3 + 1, t * 3 + 2]).collect(),
        Primitive::TriangleStrip => {
            if count < 3 {
                return Vec::new();
            }
            (0..count - 2)
                .map(|i| if i % 2 == 0 { [i, i + 1, i + 2] } else { [i + 1, i, i + 2] })
                .collect()
        }
        Primitive::TriangleFan | Primitive::Polygon => {
            if count < 3 {
                return Vec::new();
            }
            (0..count - 2).map(|i| [0, i + 1, i + 2]).collect()
        }
        Primitive::Quads => (0..count / 4)
            .flat_map(|q| {
                let b = q * 4;
                [[b, b + 1, b + 2], [b, b + 2, b + 3]]
            })
            .collect(),
        Primitive::QuadStrip => {
            if count < 4 {
                return Vec::new();
            }
            (0..(count - 2) / 2)
                .flat_map(|q| {
                    let b = q * 2;
                    [[b, b + 1, b + 2], [b + 2, b + 1, b + 3]]
                })
                .collect()
        }
    }
}

/// A vertex position in screen space (pixels, `y` growing downward).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenVertex {
    pub x: f32,
    pub y: f32,
}

/// Inclusive-exclusive pixel bounds: `[x0, x1) x [y0, y1)` — a resolved
/// viewport/scissor intersection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

/// Rasterize one triangle to the pixels it covers, using the standard
/// top-left fill rule: a pixel's center is covered if it's strictly inside,
/// or lies exactly on a "top" or "left" edge. That tie-break is what keeps
/// two triangles sharing an edge from double-covering or gapping the pixels
/// along it. No fragment shading or depth test yet — this is coverage only.
pub fn rasterize_triangle(
    v0: ScreenVertex,
    v1: ScreenVertex,
    v2: ScreenVertex,
    bounds: Bounds,
) -> Vec<(u32, u32)> {
    rasterize_triangle_weighted(v0, v1, v2, bounds)
        .into_iter()
        .map(|(x, y, ..)| (x, y))
        .collect()
}

/// Like [`rasterize_triangle`], but also returns each covered pixel's
/// barycentric weights (`w0, w1, w2`, summing to 1, one per vertex). These
/// are screen-space-linear, not perspective-corrected — the shader's own
/// `ipa`/`mufu rcp` sequence does that correction (see `isa`'s module
/// docs), so the caller feeds these straight in as the linearly-interpolated
/// `attr/w` and `1/w` values a real Maxwell rasterizer would hand it.
pub fn rasterize_triangle_weighted(
    v0: ScreenVertex,
    v1: ScreenVertex,
    v2: ScreenVertex,
    bounds: Bounds,
) -> Vec<(u32, u32, f32, f32, f32)> {
    let Some(tri) = TriangleSetup::new(v0, v1, v2) else {
        return Vec::new();
    };
    let (min_x, max_x, min_y, max_y) = tri.bbox(bounds);
    let mut out = Vec::new();
    for y in min_y..max_y {
        for x in min_x..max_x {
            if let Some([w0, w1, w2]) = tri.coverage(x as f32 + 0.5, y as f32 + 0.5) {
                out.push((x, y, w0, w1, w2));
            }
        }
    }
    out
}

/// The samples alpha-to-coverage leaves for a fragment of this alpha.
///
/// Hardware dithers the mask so that neighbouring pixels of equal alpha keep
/// *different* samples. Keeping a fixed prefix instead gives every pixel the
/// same fraction of its samples, which is the same average coverage — the
/// difference only shows in the spatial noise of the dither, and a resolve
/// averages that away.
fn alpha_coverage(alpha: f32, count: u32) -> u32 {
    let kept = (alpha.clamp(0.0, 1.0) * count as f32).round() as u32;
    if kept >= count {
        u32::MAX
    } else {
        (1u32 << kept) - 1
    }
}

fn edge(a: ScreenVertex, b: ScreenVertex, px: f32, py: f32) -> f32 {
    (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x)
}

fn is_top_left(a: ScreenVertex, b: ScreenVertex) -> bool {
    (a.y == b.y && a.x > b.x) || (a.y > b.y)
}

/// A triangle prepared for coverage queries: its edge functions, the top-left
/// tie-break each edge falls under, and the winding fix-up, all resolved once.
///
/// Multisampling is why this is a type rather than a loop: a pixel is asked
/// about once per sample, at a different point each time, and re-deriving the
/// winding and the fill rule for every one of them would be both slower and a
/// chance for the samples of one pixel to disagree about the triangle.
#[derive(Debug, Clone, Copy)]
pub struct TriangleSetup {
    v0: ScreenVertex,
    v1: ScreenVertex,
    v2: ScreenVertex,
    /// The caller's winding was clockwise, so `w1` and `w2` swap back on the
    /// way out of [`TriangleSetup::weights_from`].
    clockwise: bool,
    area: f32,
    /// Whether each of the edges `v0-v1`, `v1-v2`, `v2-v0` is a top or left
    /// one, in that order.
    top_left: [bool; 3],
}

impl TriangleSetup {
    /// `None` for a degenerate triangle: zero area, nothing covered.
    pub fn new(v0: ScreenVertex, v1: ScreenVertex, v2: ScreenVertex) -> Option<TriangleSetup> {
        let signed_area = edge(v0, v1, v2.x, v2.y);
        if signed_area == 0.0 {
            return None;
        }
        // Wind the triangle counter-clockwise before applying the fill rule,
        // swapping the weights back on the way out.
        //
        // The top-left tie-break only assigns an on-edge point to exactly one
        // of the two triangles sharing that edge when they *walk the edge in
        // opposite directions* — which is true of consistently-wound geometry
        // and false when a quad is emitted as one counter-clockwise and one
        // clockwise triangle, as SDL's does. There both triangles walk the
        // shared diagonal the same way, so they agree on `is_top_left` and the
        // points exactly on it belong to both or, when the answer is `false`,
        // to neither. JKSV's save tiles are 128x128 quads whose diagonal runs
        // at exactly 45 degrees, so pixel centres land on it — and every tile
        // came out with a one-pixel gap straight through it.
        let clockwise = signed_area < 0.0;
        let (v1, v2) = if clockwise { (v2, v1) } else { (v1, v2) };
        Some(TriangleSetup {
            v0,
            v1,
            v2,
            clockwise,
            area: signed_area.abs(),
            top_left: [is_top_left(v0, v1), is_top_left(v1, v2), is_top_left(v2, v0)],
        })
    }

    /// The half-open pixel range the triangle can reach, clipped to `bounds`.
    pub fn bbox(&self, bounds: Bounds) -> (u32, u32, u32, u32) {
        let (v0, v1, v2) = (self.v0, self.v1, self.v2);
        (
            v0.x.min(v1.x).min(v2.x).floor().max(bounds.x0 as f32) as u32,
            (v0.x.max(v1.x).max(v2.x).ceil() as u32).min(bounds.x1),
            v0.y.min(v1.y).min(v2.y).floor().max(bounds.y0 as f32) as u32,
            (v0.y.max(v1.y).max(v2.y).ceil() as u32).min(bounds.y1),
        )
    }

    /// The three edge functions at `(px, py)`, in `v0-v1`, `v1-v2`, `v2-v0`
    /// order — the same order as `top_left`.
    fn edges(&self, px: f32, py: f32) -> [f32; 3] {
        [
            edge(self.v0, self.v1, px, py),
            edge(self.v1, self.v2, px, py),
            edge(self.v2, self.v0, px, py),
        ]
    }

    /// `w0` is opposite `v0` (edge `v1-v2`), and so on. The counter-clockwise
    /// rewind in `new` moved the caller's `v1` and `v2`, so their weights swap
    /// back here.
    fn weights_from(&self, e: [f32; 3]) -> [f32; 3] {
        let (w0, w1, w2) = (e[1] / self.area, e[2] / self.area, e[0] / self.area);
        if self.clockwise {
            [w0, w2, w1]
        } else {
            [w0, w1, w2]
        }
    }

    /// Barycentric weights at `(px, py)` whether or not it is covered.
    ///
    /// This is where a fragment shader runs. The default interpolation
    /// qualifier evaluates at the pixel centre even for a partially covered
    /// pixel whose centre falls outside the triangle, where the weights come
    /// out extrapolated — that is the behaviour, not a rounding slip.
    pub fn weights(&self, px: f32, py: f32) -> [f32; 3] {
        self.weights_from(self.edges(px, py))
    }

    /// Weights at `(px, py)`, or `None` where the fill rule leaves it
    /// uncovered: a point is covered if it is strictly inside, or lies exactly
    /// on a "top" or "left" edge.
    pub fn coverage(&self, px: f32, py: f32) -> Option<[f32; 3]> {
        let e = self.edges(px, py);
        let inside = |value: f32, top_left: bool| value > 0.0 || (top_left && value == 0.0);
        (0..3)
            .all(|i| inside(e[i], self.top_left[i]))
            .then(|| self.weights_from(e))
    }
}

/// `DkVtxAttribSize`'s component count and per-component bit width
/// (deko3d.h). Only the shapes this fetcher decodes are listed; anything
/// else is a clear "unsupported" error rather than a guess.
fn attrib_shape(size: u32) -> Option<(u32, u32)> {
    match size {
        0x01 => Some((4, 32)), // 4x32
        0x02 => Some((3, 32)), // 3x32
        0x04 => Some((2, 32)), // 2x32
        0x12 => Some((1, 32)), // 1x32
        0x03 => Some((4, 16)), // 4x16
        0x05 => Some((3, 16)), // 3x16
        0x0f => Some((2, 16)), // 2x16
        0x1b => Some((1, 16)), // 1x16
        0x0a => Some((4, 8)),  // 4x8
        _ => None,
    }
}

/// `DkVtxAttribType` (deko3d.h): `Float = 7`, `Unorm = 2`.
/// `DkVtxAttribType`, as Eden's `VertexAttribute::Type` names them.
const ATTRIB_TYPE_SNORM: u32 = 1;
const ATTRIB_TYPE_UNORM: u32 = 2;
const ATTRIB_TYPE_SINT: u32 = 3;
const ATTRIB_TYPE_UINT: u32 = 4;
const ATTRIB_TYPE_FLOAT: u32 = 7;

/// What a "fixed" attribute reads as: the `vec4` default every graphics API
/// hands a vertex input the draw supplies no data for.
const ATTRIB_DEFAULT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// Fetch one vertex's worth of a single attribute out of GPU memory,
/// returning it padded to 4 components (`0,0,0,1` for the ones the format
/// doesn't carry — the usual `vec4` default). `is_bgra` swaps the first and
/// third component after decoding, matching a packed-colour attribute
/// declared BGRA instead of RGBA. A "fixed" attribute has no buffer behind
/// it and reads [`ATTRIB_DEFAULT`] outright.
pub fn fetch_attribute(
    attrib: VertexAttrib,
    array: VertexArray,
    vertex_index: u32,
    ctx: &ExecCtx,
) -> Result<[f32; 4]> {
    // A "fixed" attribute is not backed by a vertex buffer at all -- it is
    // the shader reading an input this draw binds nothing to, which is a
    // well-defined thing to do rather than a gap in this rasterizer. Erroring
    // dropped the whole draw, and with it every attribute that *was* bound.
    if attrib.is_fixed {
        return Ok(ATTRIB_DEFAULT);
    }
    // A disabled buffer is different: the attribute claims to be fetched from
    // an array the draw never turned on, so some piece of state has been read
    // wrong. Say so instead of inventing a value.
    if !array.enabled {
        return Err(Error::Gpu(format!(
            "raster: attribute reads from disabled vertex buffer {}",
            attrib.buffer_id
        )));
    }
    let (components, bits) = attrib_shape(attrib.size).ok_or_else(|| {
        Error::Gpu(format!(
            "raster: unsupported vertex attribute size {:#x}",
            attrib.size
        ))
    })?;

    let addr = array.start
        + vertex_index as u64 * array.stride as u64
        + attrib.offset as u64;

    let mut out = [0.0f32, 0.0, 0.0, 1.0];
    match (attrib.ty, bits) {
        (ATTRIB_TYPE_FLOAT, 32) => {
            for c in 0..components {
                let bits = ctx.read_u32(addr + c as u64 * 4)?;
                out[c as usize] = f32::from_bits(bits);
            }
        }
        // The 16-bit shapes, read as one packed value the way the 8-bit ones
        // are: `read_pixel` translates the address once for the whole
        // attribute rather than once per component.
        (ATTRIB_TYPE_FLOAT, 16) => {
            let packed = ctx.read_pixel(addr, components * 2)?;
            for c in 0..components {
                out[c as usize] = f16_to_f32((packed >> (c * 16)) as u16);
            }
        }
        (ATTRIB_TYPE_UNORM, 16) => {
            let packed = ctx.read_pixel(addr, components * 2)?;
            for c in 0..components {
                out[c as usize] = f32::from((packed >> (c * 16)) as u16) / 65535.0;
            }
        }
        (ATTRIB_TYPE_SNORM, 16) => {
            let packed = ctx.read_pixel(addr, components * 2)?;
            for c in 0..components {
                let value = (packed >> (c * 16)) as u16 as i16;
                // -32768 and -32767 both mean -1, as at eight bits.
                out[c as usize] = (f32::from(value) / 32767.0).max(-1.0);
            }
        }
        (ATTRIB_TYPE_SINT, 16) => {
            let packed = ctx.read_pixel(addr, components * 2)?;
            for c in 0..components {
                let value = (packed >> (c * 16)) as u16 as i16;
                out[c as usize] = f32::from_bits(i32::from(value) as u32);
            }
        }
        (ATTRIB_TYPE_UINT, 16) => {
            let packed = ctx.read_pixel(addr, components * 2)?;
            for c in 0..components {
                out[c as usize] = f32::from_bits(u32::from((packed >> (c * 16)) as u16));
            }
        }
        (ATTRIB_TYPE_UNORM, 8) => {
            let packed = ctx.read_u32(addr)?;
            for c in 0..components {
                let byte = (packed >> (c * 8)) & 0xff;
                out[c as usize] = byte as f32 / 255.0;
            }
        }
        (ATTRIB_TYPE_SNORM, 8) => {
            let packed = ctx.read_u32(addr)?;
            for c in 0..components {
                let byte = ((packed >> (c * 8)) & 0xff) as u8 as i8;
                // -128 and -127 both mean -1: the negative side has one more
                // step than the positive, and every API maps both onto it.
                out[c as usize] = (byte as f32 / 127.0).max(-1.0);
            }
        }
        // An integer attribute is not a number the shader averages, it is a
        // value it indexes or masks with — so the slot carries its *bits*,
        // the same way `shade_vertex` hands over `vertex_id` and
        // `instance_id`. Converting it to a float instead would read back as
        // whatever that float's bit pattern happened to be.
        (ATTRIB_TYPE_SINT, 8) => {
            let packed = ctx.read_u32(addr)?;
            for c in 0..components {
                let byte = ((packed >> (c * 8)) & 0xff) as u8 as i8;
                out[c as usize] = f32::from_bits(byte as i32 as u32);
            }
        }
        (ATTRIB_TYPE_UINT, 8) => {
            let packed = ctx.read_u32(addr)?;
            for c in 0..components {
                out[c as usize] = f32::from_bits((packed >> (c * 8)) & 0xff);
            }
        }
        (ty, bits) => {
            return Err(Error::Gpu(format!(
                "raster: unsupported vertex attribute type {} at {} bits",
                ty, bits
            )));
        }
    }
    if attrib.is_bgra {
        out.swap(0, 2);
    }
    Ok(out)
}

/// The `a[]` offset `gl_Position`/clip position lands at, and the fixed
/// interpolated-`1/w` slot — both established in Stage 0's recon (see
/// `isa`'s module docs) and already load-bearing in `shader::interp`'s
/// tests.
const CLIP_POS_OFFSET: u16 = 0x70;
const INV_W_OFFSET: u16 = 0x7c;
/// Generic varying `i`'s `a[]` slot: `VARYING_BASE + i * VARYING_STRIDE`.
/// This is the same numeric convention on both sides of the interpolator —
/// a vertex shader's output slot `i` is the fragment shader's input slot
/// `i` — because they're literally the same fixed-function wires.
const VARYING_BASE: u16 = 0x80;
const VARYING_STRIDE: u16 = 0x10;

/// `gl_InstanceID`'s and `gl_VertexID`'s slots in a vertex shader's `a[]`
/// input space. They are integers, not floats: the shader shifts and adds
/// them, so the register has to hold the value's *bits*, not its numeric
/// value as an `f32`.
const INSTANCE_ID_OFFSET: u16 = 0x2f8;
const VERTEX_ID_OFFSET: u16 = 0x2fc;
/// How many generic varying slots get fetched/interpolated: the whole of
/// Maxwell's generic attribute space, `a[0x80]..a[0x280)`.
///
/// This used to be four, on the reasoning that colour, texcoord and a spare
/// is all a 2D UI needs. It is not: the Home Menu's panel shaders interpolate
/// slots 4, 5 and 6, and a slot past the end reads as zero rather than
/// failing. Those zeros became the denominator of a normalising `rcp`, which
/// returned infinity, and a `0 * inf` later turned every one of the panel's
/// pixels into a NaN that encoded as black.
///
/// Interpolating all 32 for every pixel would cost far more than the four
/// did, so [`Program::interpolated_slots`] narrows it back down to the ones
/// the fragment shader actually reads.
const NUM_VARYINGS: usize = 32;
/// Real Maxwell guarantees at least this many vertex attributes; scanning a
/// fixed range means vertex fetch doesn't need the shader to declare how
/// many it reads.
const MAX_VERTEX_ATTRIBS: u32 = 16;
/// One vertex after the vertex shader ran: clip-space position plus every
/// generic varying, ready for the perspective divide and interpolation.
#[derive(Clone, Copy)]
struct ShadedVertex {
    clip: [f32; 4],
    varyings: [[f32; 4]; NUM_VARYINGS],
}

fn shade_vertex(
    program: &Compiled,
    attribs: &[VertexAttrib],
    arrays: &[VertexArray],
    // The two ordinals that pick this invocation's data out of the arrays.
    (vertex_index, instance_id): (u32, u32),
    ctx: &ExecCtx,
    consts: &dyn ConstantSource,
    y_negate: bool,
) -> Result<ShadedVertex> {
    let mut inv = Invocation::new();
    inv.attr_in.set(VERTEX_ID_OFFSET, f32::from_bits(vertex_index));
    inv.attr_in.set(INSTANCE_ID_OFFSET, f32::from_bits(instance_id));
    for (i, attrib) in attribs.iter().enumerate() {
        // size 0 isn't a valid DkVtxAttribSize — it's what an unconfigured
        // `VertexAttribState` slot reads back as, so it means "not used"
        // rather than "unsupported format".
        if attrib.size == 0 {
            continue;
        }
        let array = arrays[attrib.buffer_id as usize];
        if !array.enabled {
            continue;
        }
        // A non-zero divisor makes the array instanced: every `divisor`
        // instances advance it by one element, and the vertex ordinal does
        // not move it at all.
        let element = instance_id.checked_div(array.divisor).unwrap_or(vertex_index);
        let v = fetch_attribute(*attrib, array, element, ctx)?;
        let base = VARYING_BASE + i as u16 * VARYING_STRIDE;
        for (c, &component) in v.iter().enumerate() {
            inv.attr_in.set(base + c as u16 * 4, component);
        }
    }
    let global = MemoryGlobal { ctx };
    let mut env = Env::new(consts, &NoTextures);
    env.memory = Some(&global);
    env.special.y_negate = y_negate;
    inv.execute(program, &env)?;

    let mut clip = [0.0, 0.0, 0.0, 1.0];
    for (c, slot) in clip.iter_mut().enumerate() {
        if let Some(v) = inv.attr_out.written(CLIP_POS_OFFSET + c as u16 * 4) {
            *slot = v;
        }
    }
    let mut varyings = [[0.0f32; 4]; NUM_VARYINGS];
    for (i, varying) in varyings.iter_mut().enumerate() {
        let base = VARYING_BASE + i as u16 * VARYING_STRIDE;
        for (c, slot) in varying.iter_mut().enumerate() {
            if let Some(v) = inv.attr_out.written(base + c as u16 * 4) {
                *slot = v;
            }
        }
    }
    Ok(ShadedVertex { clip, varyings })
}

/// Clip position to window space: perspective-divide, then apply the
/// viewport transform the guest programmed — `window = ndc * scale +
/// translate` per axis. Whether that flips y is the transform's business
/// (see [`Engine3D::viewport_transform`]), not a convention baked in here.
///
/// Also returns `1/w` and the window-space depth. Both are affine in screen
/// space, so they interpolate with plain (not perspective-corrected)
/// barycentrics.
fn to_screen(clip: [f32; 4], vt: ViewportTransform) -> (ScreenVertex, f32, f32) {
    let inv_w = 1.0 / clip[3];
    let screen = ScreenVertex {
        x: clip[0] * inv_w * vt.scale[0] + vt.translate[0],
        y: clip[1] * inv_w * vt.scale[1] + vt.translate[1],
    };
    (screen, inv_w, clip[2] * inv_w * vt.scale[2] + vt.translate[2])
}

/// `DEPTH_TEST_FUNC` carries **either** numbering, and hardware takes both.
///
/// Homebrew going through Mesa's GL driver writes the literal OpenGL enum,
/// `GL_NEVER`(0x0200)`..=GL_ALWAYS`(0x0207) — confirmed by dumping a live
/// JKSV capture's registers, which is why only those were decoded here. A
/// title that came through a D3D-shaped path writes the one-based numbering
/// instead, and Maxwell's register documents both: Eden's `ComparisonOp`
/// (`maxwell_3d.h`) lists `Never_D3D = 1 ..= Always_D3D = 8` beside
/// `Never_GL = 0x200 ..= Always_GL = 0x207`.
///
/// Decoding only one of them was not a draw that failed loudly. The
/// unrecognised half fell into `Always`, so Just Dance 2019's `LessEqual`
/// (`4`) became "the depth test passes" and every fragment it should have
/// hidden was drawn. The GPU backend, which refuses what it cannot express,
/// turned the same value into a fallback for *every* draw in the title.
fn depth_test_passes(func: u32, new: f32, old: f32) -> bool {
    match func {
        1 | 0x0200 => false,
        2 | 0x0201 => new < old,
        3 | 0x0202 => new == old,
        4 | 0x0203 => new <= old,
        5 | 0x0204 => new > old,
        6 | 0x0205 => new != old,
        7 | 0x0206 => new >= old,
        _ => true, // Always (8 or 0x0207), and any unrecognised code.
    }
}

/// `BLEND_FUNC_*`'s real hardware type is `G80_BLEND_FACTOR`
/// (`nv_3ddefs.xml`): literal OpenGL blend-factor enum values (`0x4000`+ for
/// the plain factors, `0xc000`+ for the constant-colour ones), not deko3d's
/// simplified `DkBlendFactor` numbering — see [`depth_test_passes`]'s doc
/// comment for how that was confirmed. `SrcColor`/`DstColor` are genuinely
/// per-channel; the rest just broadcast a scalar.
fn blend_factor(code: u32, src: [f32; 4], dst: [f32; 4], constant: [f32; 4]) -> [f32; 4] {
    match code {
        // The D3D numbering. Both numberings name the same set of factors and
        // the hardware takes either; which one a register holds is down to
        // whose driver wrote it. Mesa (JKSV) writes the GL enum straight
        // through, deko3d and nvn write this one — the Home Menu blends every
        // one of its draws `SrcAlpha`/`OneMinusSrcAlpha`, which fell through
        // to `One`/`One` here and turned its whole UI into `src + dst`.
        0x01 => [0.0; 4],                    // Zero
        0x02 => [1.0; 4],                    // One
        0x03 => src,                         // SrcColor
        0x04 => src.map(|c| 1.0 - c),        // OneMinusSrcColor
        0x05 => [src[3]; 4],                 // SrcAlpha
        0x06 => [1.0 - src[3]; 4],           // OneMinusSrcAlpha
        0x07 => [dst[3]; 4],                 // DstAlpha
        0x08 => [1.0 - dst[3]; 4],           // OneMinusDstAlpha
        0x09 => dst,                         // DstColor
        0x0a => dst.map(|c| 1.0 - c),        // OneMinusDstColor
        0x61 => constant,                    // ConstantColor
        0x62 => constant.map(|c| 1.0 - c),   // OneMinusConstantColor
        0x63 => [constant[3]; 4],            // ConstantAlpha
        0x64 => [1.0 - constant[3]; 4],      // OneMinusConstantAlpha

        // The GL numbering.
        0x4000 => [0.0; 4],                  // Zero
        0x4300 => src,                       // SrcColor
        0x4301 => src.map(|c| 1.0 - c),      // OneMinusSrcColor
        0x4302 => [src[3]; 4],               // SrcAlpha
        0x4303 => [1.0 - src[3]; 4],         // OneMinusSrcAlpha
        0x4304 => [dst[3]; 4],               // DstAlpha
        0x4305 => [1.0 - dst[3]; 4],         // OneMinusDstAlpha
        0x4306 => dst,                       // DstColor
        0x4307 => dst.map(|c| 1.0 - c),      // OneMinusDstColor
        0xc001 => constant,                  // ConstantColor
        0xc002 => constant.map(|c| 1.0 - c), // OneMinusConstantColor
        0xc003 => [constant[3]; 4],          // ConstantAlpha
        0xc004 => [1.0 - constant[3]; 4],    // OneMinusConstantAlpha

        // SrcAlphaSaturate, in both numberings. Alpha's factor is 1, not the
        // saturated value the colour channels get.
        0x0b | 0x4308 => {
            let f = src[3].min(1.0 - dst[3]);
            [f, f, f, 1.0]
        }

        _ => [1.0; 4], // One (0x4001), and anything unrecognised.
    }
}

/// `BLEND_EQUATION_*`'s real hardware type is `gl_blend_equation`
/// (`nv_3ddefs.xml`): literal `GL_FUNC_ADD`(0x8006)`..=GL_FUNC_REVERSE_
/// SUBTRACT`(0x800b), not deko3d's simplified 1-5 `DkBlendOp` numbering.
/// `BLEND_EQUATION_*` in the D3D numbering the same register also takes —
/// see [`blend_factor`].
fn blend_equation(op: u32, src: f32, dst: f32) -> f32 {
    match op {
        0x2 | 0x800a => src - dst,    // FuncSubtract
        0x3 | 0x800b => dst - src,    // FuncReverseSubtract
        0x4 | 0x8007 => src.min(dst), // Min
        0x5 | 0x8008 => src.max(dst), // Max
        _ => src + dst,               // FuncAdd (1, 0x8006), and anything unrecognised.
    }
}

/// The shader's output colour as the blend unit sees it.
///
/// Blending into a **fixed-point** render target clamps the incoming colour
/// into the range that target can store first — GL says so for a fixed-point
/// colour buffer, and it is what the ROP does. A float target takes the
/// colour as it is.
///
/// The range is the smaller half of it. What this really settles is **NaN**,
/// which clamps to zero here and is otherwise indestructible: every blend
/// factor is a multiply, and `NaN * 0` is `NaN`, so a NaN source survives even
/// a source alpha of zero and lands in the framebuffer as an opaque black
/// pixel. That is not a hypothetical — the Album applet's image shaders
/// normalise a weighted sum of samples by dividing by the total alpha, and a
/// fully transparent texel makes that `rcp(0)`, an infinity, and then
/// `0 * inf`. Every icon it drew came out inside a black box.
fn source_color(color: [f32; 4], format: ColorFormat) -> [f32; 4] {
    if format.is_float() {
        return color;
    }
    // `f32::max` returns the operand that is not NaN, so this floors a NaN at
    // zero where `clamp` would carry it straight through.
    color.map(|c| c.max(0.0).min(1.0))
}

fn blend(target: BlendTarget, constant: [f32; 4], src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let src_rgb = blend_factor(target.func_rgb_src, src, dst, constant);
    let dst_rgb = blend_factor(target.func_rgb_dst, src, dst, constant);
    let src_a = blend_factor(target.func_alpha_src, src, dst, constant)[3];
    let dst_a = blend_factor(target.func_alpha_dst, src, dst, constant)[3];
    let mut out = [0.0f32; 4];
    for i in 0..3 {
        out[i] = blend_equation(target.equation_rgb, src[i] * src_rgb[i], dst[i] * dst_rgb[i]);
    }
    out[3] = blend_equation(target.equation_alpha, src[3] * src_a, dst[3] * dst_a);
    out
}

/// Put one fragment's interpolated inputs in place, ready to run.
///
/// `inv` is threaded in rather than created here so that a draw allocates one
/// invocation instead of one per covered pixel — a full-screen quad covers
/// 921 600 of them.
fn seed_fragment(
    inv: &mut Invocation,
    program: &Compiled,
    verts: &[ShadedVertex; 3],
    inv_w: [f32; 3],
    weights: [f32; 3],
) {
    inv.reset();
    let interp_inv_w = weights[0] * inv_w[0] + weights[1] * inv_w[1] + weights[2] * inv_w[2];
    inv.attr_in.set(INV_W_OFFSET, interp_inv_w);
    for &slot in program.interpolated_slots() {
        let base = VARYING_BASE + slot as u16 * VARYING_STRIDE;
        for c in 0..4 {
            let over_w = weights[0] * verts[0].varyings[slot][c] * inv_w[0]
                + weights[1] * verts[1].varyings[slot][c] * inv_w[1]
                + weights[2] * verts[2].varyings[slot][c] * inv_w[2];
            inv.attr_in.set(base + c as u16 * 4, over_w);
        }
    }
}

/// The colour an invocation that has run to `exit` leaves behind, or `None`
/// if `kil` discarded the fragment.
///
/// Which register holds which component is the program header's business: a
/// program that leaves a component to the driver still spends a register on
/// it, and one that writes nothing to a target spends none at all. A program
/// with no header — `uam`/deko3d builds, and this module's own fixtures —
/// keeps the plain `r0..r3`.
fn fragment_color(inv: &Invocation, program: &Compiled) -> Option<[f32; 4]> {
    if inv.discarded {
        return None;
    }
    let Some(header) = program.header().filter(|h| h.writes_any_color()) else {
        return Some([inv.reg_f32(0), inv.reg_f32(1), inv.reg_f32(2), inv.reg_f32(3)]);
    };
    Some(std::array::from_fn(|component| {
        match header.fragment_output_reg(0, component as u32) {
            Some(reg) => inv.reg_f32(reg),
            // Nothing wrote it: colour reads zero and alpha opaque, which is
            // what a blend against it expects rather than a stale register.
            None => (component == 3) as u32 as f32,
        }
    }))
}

/// Shade one covered pixel.
fn shade_fragment(
    inv: &mut Invocation,
    program: &Compiled,
    verts: &[ShadedVertex; 3],
    inv_w: [f32; 3],
    weights: [f32; 3],
    env: &Env,
) -> Result<Option<[f32; 4]>> {
    seed_fragment(inv, program, verts, inv_w, weights);
    inv.execute(program, env)?;
    Ok(fragment_color(inv, program))
}

/// The four pixels hardware shades together, in lane order: `(x, y)`,
/// `(x + 1, y)`, `(x, y + 1)`, `(x + 1, y + 1)`.
pub const QUAD: usize = 4;

/// Where lane `lane` of a quad based at `(x, y)` sits.
fn quad_pixel(x: u32, y: u32, lane: usize) -> (u32, u32) {
    (x + lane as u32 % 2, y + lane as u32 / 2)
}

/// Shade a 2x2 quad of pixels in lock-step.
///
/// A warp shuffle reads a register belonging to another invocation, so the
/// four pixels have to reach it together: each lane runs to its next shuffle,
/// and only once every lane that is still going has arrived does the exchange
/// happen and all of them go on. Lanes whose pixel the triangle misses are
/// shaded anyway and their colour thrown away — they exist so that the
/// covered lanes have a neighbour to difference against, which is the whole
/// reason a derivative can be computed at all.
///
/// Lanes that diverge — one at a shuffle, another somewhere else entirely —
/// are answered from wherever the other lane happens to be. Hardware answers
/// them from an inactive lane, which is to say with a value the shader is not
/// entitled to rely on either.
fn shade_quad(
    lanes: &mut [Invocation; QUAD],
    program: &Compiled,
    verts: &[ShadedVertex; 3],
    inv_w: [f32; 3],
    weights: [[f32; 3]; QUAD],
    env: &mut Env,
) -> Result<[Option<[f32; 4]>; QUAD]> {
    for (lane, invocation) in lanes.iter_mut().enumerate() {
        seed_fragment(invocation, program, verts, inv_w, weights[lane]);
        invocation.begin();
    }
    let mut running = [true; QUAD];
    loop {
        let mut shuffled = false;
        for (lane, invocation) in lanes.iter_mut().enumerate() {
            if !running[lane] {
                continue;
            }
            env.special.lane = lane as u32;
            match invocation.resume(program, env)? {
                Halt::Exited => running[lane] = false,
                Halt::Shuffle => shuffled = true,
                Halt::Barrier => {
                    return Err(Error::Gpu(
                        "raster: bar in a fragment shader, where there is no CTA to \
                         synchronise with"
                            .into(),
                    ))
                }
            }
        }
        if !shuffled {
            break;
        }
        resolve_shuffles(lanes);
    }
    Ok(std::array::from_fn(|lane| fragment_color(&lanes[lane], program)))
}

/// The per-pixel half of a draw: which samples of a pixel a triangle covers
/// and passes the depth test at, and what a shaded colour does to the
/// targets.
///
/// Held apart from the loop over pixels because there are two such loops — a
/// shader containing a warp shuffle is walked in 2x2 quads instead of pixel
/// by pixel — and they do exactly this to every pixel they reach.
struct Fragments {
    grid: SampleGrid,
    sample_mask: u32,
    depth: Option<DepthTarget>,
    depth_state: DepthState,
    rt: Option<RenderTarget>,
    blend_target: BlendTarget,
    blend_constant: [f32; 4],
    color_mask: [bool; 4],
    writes_all_channels: bool,
    writes_any_channel: bool,
    alpha_to_coverage: bool,
}

impl Fragments {
    /// Which samples of pixel `(x, y)` this triangle covers and the depth
    /// test lets through, and the interpolated depth at each of them.
    ///
    /// Coverage and depth are per sample; shading is not. That split is what
    /// multisampling buys over rendering the whole frame at the sample grid's
    /// resolution: the edges get every sample's worth of coverage, but the
    /// fragment shader still runs once for the pixel.
    /// `sample_z` is filled for the samples the returned mask names, and left
    /// alone for the rest — so the caller may reuse one buffer across pixels
    /// without clearing it. [`Fragments::write`] reads exactly the samples the
    /// mask names, and alpha-to-coverage only narrows that mask.
    fn coverage(
        &self,
        tri: &TriangleSetup,
        window_z: [f32; 3],
        (x, y): (u32, u32),
        sample_z: &mut [f32; MAX_SAMPLES],
        ctx: &mut ExecCtx,
    ) -> Result<u32> {
        let mut covered = 0u32;
        for sample in 0..self.grid.count() {
            if self.sample_mask >> sample & 1 == 0 {
                continue;
            }
            let [offset_x, offset_y] = self.grid.position(sample);
            let Some(w) = tri.coverage(x as f32 + offset_x, y as f32 + offset_y) else {
                continue;
            };
            let z = w[0] * window_z[0] + w[1] * window_z[1] + w[2] * window_z[2];
            if let (true, Some(dt)) = (self.depth_state.test_enabled, self.depth) {
                let (tx, ty) = self.grid.texel(x, y, sample);
                let bytes = dt.format.bytes;
                let dva = dt.addr + dt.layout.offset(tx * bytes, ty, dt.width * bytes) as u64;
                let old = dt.format.decode_depth(ctx.read_pixel(dva, bytes)?);
                if !depth_test_passes(self.depth_state.func, z, old) {
                    continue;
                }
            }
            covered |= 1 << sample;
            sample_z[sample as usize] = z;
        }
        Ok(covered)
    }

    /// Put a shaded pixel into the targets, for every sample of it still
    /// covered.
    fn write(
        &self,
        (x, y): (u32, u32),
        covered: u32,
        sample_z: &[f32; MAX_SAMPLES],
        color: [f32; 4],
        ctx: &mut ExecCtx,
        tally: &mut DrawTally,
    ) -> Result<()> {
        tally.shaded(color);

        // Alpha-to-coverage narrows the mask *after* shading, since it is the
        // shaded alpha it turns into coverage.
        let mut covered = covered;
        if self.alpha_to_coverage {
            covered &= alpha_coverage(color[3], self.grid.count());
            if covered == 0 {
                tally.alpha_killed += 1;
                return Ok(());
            }
        }

        for sample in 0..self.grid.count() {
            if covered & (1 << sample) == 0 {
                continue;
            }
            let (tx, ty) = self.grid.texel(x, y, sample);
            if self.depth_state.test_enabled && self.depth_state.write_enabled {
                if let Some(dt) = self.depth {
                    let bytes = dt.format.bytes;
                    let dva = dt.addr + dt.layout.offset(tx * bytes, ty, dt.width * bytes) as u64;
                    let z = sample_z[sample as usize];
                    // A packed depth-stencil pixel holds a stencil byte this
                    // draw is not writing. Read it back and merge rather than
                    // flattening it to zero — the extra read is only for the
                    // formats that actually share the pixel.
                    let value = if dt.format.packs_stencil() {
                        dt.format.with_depth(ctx.read_pixel(dva, bytes)?, z)
                    } else {
                        dt.format.encode_depth(z)
                    };
                    ctx.write_pixel(dva, bytes, value)?;
                }
            }

            // A depth-only pass shaded the fragment for its `kil` and its
            // alpha coverage, and has nowhere to put the colour that came out
            // of it — either because no colour target is bound or because the
            // write mask closed every channel.
            if let Some(rt) = self.rt.filter(|_| self.writes_any_channel) {
                let bpp = rt.format.bytes_per_pixel;
                let va = rt.addr + rt.texel_offset(tx, ty) as u64;
                // Blending and a masked channel both need what the target
                // already holds, so read it once for both.
                let dst = if self.blend_target.enabled || !self.writes_all_channels {
                    Some(rt.format.decode(ctx.read_pixel(va, bpp)?)?)
                } else {
                    None
                };
                let mut out = color;
                if let Some(dst) = dst {
                    if self.blend_target.enabled {
                        let src = source_color(color, rt.format);
                        out = blend(self.blend_target, self.blend_constant, src, dst);
                    }
                    for (channel, keep) in self.color_mask.iter().enumerate() {
                        if !keep {
                            out[channel] = dst[channel];
                        }
                    }
                }
                tally.wrote(out);
                ctx.write_pixel(va, bpp, rt.format.encode(out)?)?;
            }
        }
        Ok(())
    }
}

/// Build the environment a fragment shader runs under and hand it to `f`.
///
/// Every source in it borrows the execution context immutably, and the pixel
/// loop needs that context mutably to write what comes out — so the
/// environment cannot outlive one shading step, which is what the callback is
/// for.
fn with_fragment_env<T>(
    engine: &Engine3D,
    ctx: &ExecCtx,
    consts: &std::cell::RefCell<crate::gpu::shader::interp::ConstCache>,
    descriptors: &std::cell::RefCell<crate::IdMap<u32, crate::gpu::texture::Descriptors>>,
    blocks: &std::cell::RefCell<crate::gpu::texture::BlockCache>,
    f: impl FnOnce(&mut Env) -> Result<T>,
) -> Result<T> {
    let fs_consts = MemoryConstants {
        ctx,
        bindings: &|bank: u8| engine.bound_constbuf(ShaderStage::Fragment, bank as u32),
        cache: consts,
    };
    let fs_textures = MemoryTextures {
        ctx,
        tex_header_pool: engine.tex_header_pool(),
        tex_sampler_pool: engine.tex_sampler_pool(),
        descriptors,
        blocks,
    };
    let fs_global = MemoryGlobal { ctx };
    let mut env = Env::with_tex_cb_index(&fs_consts, &fs_textures, engine.tex_cb_index());
    env.memory = Some(&fs_global);
    env.special.y_negate = engine.window_origin().lower_left;
    f(&mut env)
}

/// Whether `cull` throws this triangle away.
///
/// Face determination happens in *window* space, after the viewport
/// transform — so which winding is front depends on the sign of that
/// transform's y scale, exactly as it does on hardware, and is not decided
/// here. A zero-area triangle covers no pixels either way; reporting it
/// culled saves the rasterizer the walk.
fn culls(cull: CullState, v: [ScreenVertex; 3]) -> bool {
    if !cull.enabled {
        return false;
    }
    let area = (v[1].x - v[0].x) * (v[2].y - v[0].y) - (v[1].y - v[0].y) * (v[2].x - v[0].x);
    if area == 0.0 {
        return true;
    }
    let front = (area > 0.0) == cull.front_ccw;
    if front {
        cull.cull_front
    } else {
        cull.cull_back
    }
}

/// Read the `i`th index of an indexed draw out of the bound index buffer.
fn read_index(ctx: &ExecCtx, base: u64, format: u32, i: u32) -> Result<u32> {
    Ok(match format {
        0 => u32::from(ctx.vmm_read_u8(base + u64::from(i))?),
        1 => {
            let at = base + u64::from(i) * 2;
            u32::from(ctx.vmm_read_u8(at)?) | (u32::from(ctx.vmm_read_u8(at + 1)?) << 8)
        }
        2 => ctx.read_u32(base + u64::from(i) * 4)?,
        other => {
            return Err(Error::Gpu(format!("raster: unknown index format {other}")));
        }
    })
}

/// A vertex after clipping: clip-space position plus its varyings, which
/// interpolate linearly in clip space (that's what makes clipping able to
/// produce new vertices at all).
#[derive(Debug, Clone, Copy)]
struct ClipVertex {
    clip: [f32; 4],
    varyings: [[f32; 4]; NUM_VARYINGS],
}

impl ClipVertex {
    fn lerp(a: &ClipVertex, b: &ClipVertex, t: f32) -> ClipVertex {
        let mut out = ClipVertex { clip: [0.0; 4], varyings: [[0.0; 4]; NUM_VARYINGS] };
        for c in 0..4 {
            out.clip[c] = a.clip[c] + (b.clip[c] - a.clip[c]) * t;
        }
        for slot in 0..NUM_VARYINGS {
            for c in 0..4 {
                out.varyings[slot][c] =
                    a.varyings[slot][c] + (b.varyings[slot][c] - a.varyings[slot][c]) * t;
            }
        }
        out
    }
}

/// Clip a triangle against the near plane (`w > epsilon`).
///
/// This is not an optimisation. A vertex at or behind the eye has `w <= 0`,
/// and the perspective divide by it sends the projected position to infinity
/// or flips it to the wrong side of the screen — one off-screen vertex
/// smears a triangle across the whole framebuffer. Clipping first replaces
/// the offending vertices with real ones on the plane, so the rasterizer
/// only ever sees geometry that projects.
fn clip_near(tri: [ClipVertex; 3]) -> Vec<[ClipVertex; 3]> {
    /// Far enough from zero that the reciprocal stays finite.
    const NEAR_W: f32 = 1e-6;

    let inside: Vec<bool> = tri.iter().map(|v| v.clip[3] > NEAR_W).collect();
    let count = inside.iter().filter(|&&i| i).count();
    if count == 3 {
        return vec![tri];
    }
    if count == 0 {
        return Vec::new();
    }
    // Walk the edges, emitting kept vertices and the crossings between them.
    let mut poly: Vec<ClipVertex> = Vec::with_capacity(4);
    for i in 0..3 {
        let j = (i + 1) % 3;
        let (a, b) = (tri[i], tri[j]);
        if inside[i] {
            poly.push(a);
        }
        if inside[i] != inside[j] {
            let t = (NEAR_W - a.clip[3]) / (b.clip[3] - a.clip[3]);
            poly.push(ClipVertex::lerp(&a, &b, t));
        }
    }
    // A triangle clipped by one plane is a triangle or a quad; fan it.
    (1..poly.len().saturating_sub(1))
        .map(|i| [poly[0], poly[i], poly[i + 1]])
        .collect()
}

/// Run `engine.last_draw` for real: fetch vertices, shade them, clip,
/// rasterize, shade covered pixels, and write real colour into the bound
/// render target.
pub fn draw(engine: &Engine3D, ctx: &mut ExecCtx) -> Result<()> {
    let call = engine.last_draw;
    // Logical target 0, through whatever slot RenderTargetControl maps it onto
    // — the same resolution a clear does.
    let rt = engine.render_target(engine.render_target_slot(0))?;

    let vs_binding = engine
        .program(ShaderStage::VertexB)
        .ok_or_else(|| Error::Gpu("raster: draw with no bound vertex program".into()))?;
    let fs_binding = engine
        .program(ShaderStage::Fragment)
        .ok_or_else(|| Error::Gpu("raster: draw with no bound fragment program".into()))?;

    let vs_program = decode_program_from_memory(&*ctx, vs_binding.addr, &|bank: u8| {
        engine.bound_constbuf(ShaderStage::VertexB, bank as u32)
    })?;
    let fs_program = decode_program_from_memory(&*ctx, fs_binding.addr, &|bank: u8| {
        engine.bound_constbuf(ShaderStage::Fragment, bank as u32)
    })?;

    let attribs: Vec<VertexAttrib> = (0..MAX_VERTEX_ATTRIBS).map(|i| engine.vertex_attrib(i)).collect();
    let arrays: Vec<VertexArray> = (0..MAX_VERTEX_ATTRIBS).map(|i| engine.vertex_array(i)).collect();
    let viewport = engine.viewport_transform();
    let grid = engine.sample_grid()?;
    let sample_mask = engine.sample_mask();
    let alpha_to_coverage = engine.alpha_to_coverage();
    let depth = engine.depth_target()?;
    // The scissor, the viewport and this draw's bounds are all in pixels;
    // a target's width and height are texels, which differ on a multisampled
    // one.
    //
    // A depth-only pass binds no colour target, so the extent comes from
    // whichever target the draw does have. Just Dance 2017 runs every pass
    // that way — it binds its Z24S8 surface as colour target 0, which is a
    // depth surface and so no colour target at all, and does its work in the
    // depth buffer. Requiring a colour target here cost it every one of its
    // 1870 draws.
    let (target_width, target_height) = match (rt, depth) {
        (Some(rt), _) => (rt.width, rt.height),
        (None, Some(dt)) => (dt.width, dt.height),
        (None, None) => {
            return Err(Error::Gpu("raster: draw with neither a colour nor a depth target".into()))
        }
    };
    let (rt_width, rt_height) = grid.pixels(target_width, target_height);
    let clip = engine.apply_scissor(ScissorRect { x0: 0, y0: 0, x1: rt_width, y1: rt_height });
    let bounds = Bounds { x0: clip.x0, y0: clip.y0, x1: clip.x1, y1: clip.y1 };
    let depth_state = engine.depth_state();
    let blend_target = engine.blend_target(0);
    let blend_constant = engine.blend_constant();
    // Which channels this draw is allowed into. A mask with nothing set is a
    // draw that writes depth and no colour at all, which is a real pass rather
    // than a mistake — so it is resolved once here, not per pixel.
    let color_mask = engine.color_mask(engine.render_target_slot(0));
    let writes_all_channels = color_mask == [true; 4];
    let writes_any_channel = color_mask.iter().any(|&channel| channel);
    let cull = engine.cull_state();

    let index_base = if call.indexed { engine.index_array_start() } else { 0 };
    let instance_id = engine.instance_id();
    let primitive = Primitive::from_raw(call.primitive)?;
    // A point or line topology assembles into nothing, and "nothing" is
    // indistinguishable on screen from a draw that worked and covered no
    // pixels. Say so, so it lands in `draws_skipped` and the trace, rather
    // than a whole line-drawn UI reporting a clean frame.
    if matches!(
        primitive,
        Primitive::Points | Primitive::Lines | Primitive::LineLoop | Primitive::LineStrip
    ) {
        return Err(Error::Gpu(format!("raster: {primitive:?} is not rasterized")));
    }
    let triangles = assemble(primitive, call.count);
    let mut tally = DrawTally::new(&fs_program);
    // One shaded vertex per *index*, cached: an indexed mesh reuses vertices
    // heavily, and re-running the vertex shader for each reference is the
    // single most expensive thing this loop can do. Keyed by an index the
    // guest minted, so it wants `crate::IdHasher` rather than a hash built to
    // resist a key being chosen to collide.
    let mut cache: crate::IdMap<u32, ShadedVertex> = crate::IdMap::default();
    // One fragment invocation for the whole draw, reset per pixel.
    let mut fragment = Invocation::new();
    let fragments = Fragments {
        grid,
        sample_mask,
        depth,
        depth_state,
        rt,
        blend_target,
        blend_constant,
        color_mask,
        writes_all_channels,
        writes_any_channel,
        alpha_to_coverage,
    };
    // Parsed TIC/TSC pairs, and the compressed blocks they decode to, shared
    // by every fragment of this draw.
    let descriptors = std::cell::RefCell::new(crate::IdMap::default());
    let blocks = std::cell::RefCell::new(crate::gpu::texture::BlockCache::default());
    // A constant buffer cannot change while a draw runs, so each stage reads
    // every constant it uses from memory once rather than once per invocation.
    let vs_const_cache = std::cell::RefCell::new(crate::gpu::shader::interp::ConstCache::default());
    let fs_const_cache = std::cell::RefCell::new(crate::gpu::shader::interp::ConstCache::default());
    // Lower both programs for this draw: branch targets resolved to indices,
    // and every constant the bound banks can supply folded into an immediate.
    // Both are things the interpreter would otherwise redo per invocation, and
    // a fragment shader runs once per covered pixel.
    let vs_program = {
        let consts = MemoryConstants {
            ctx: &*ctx,
            bindings: &|bank: u8| engine.bound_constbuf(ShaderStage::VertexB, bank as u32),
            cache: &vs_const_cache,
        };
        Compiled::with_constants(&vs_program, &consts)
    };
    let fs_program = {
        let consts = MemoryConstants {
            ctx: &*ctx,
            bindings: &|bank: u8| engine.bound_constbuf(ShaderStage::Fragment, bank as u32),
            cache: &fs_const_cache,
        };
        Compiled::with_constants(&fs_program, &consts)
    };
    // A shader that reads a register belonging to another lane has to be run
    // the way hardware runs it: four pixels of a quad in lock-step, helper
    // lanes and all. Nothing else is, because those helpers are work no other
    // draw has to do and this is a draw's innermost loop.
    let mut quad: Option<Box<[Invocation; QUAD]>> = fs_program
        .ops()
        .iter()
        .any(|op| matches!(op, Op::Shfl { .. } | Op::Fswzadd { .. }))
        .then(|| Box::new(std::array::from_fn(|_| Invocation::new())));

    // `TRACE_PIPELINE=1`: the fixed-function state this draw runs under, as
    // a GPU backend would have to describe it — or what stopped it being
    // describable, which is the more useful half.
    if crate::env_flag!("TRACE_PIPELINE") {
        // What the viewport was resolved *from* as well as what it resolved
        // to: `Viewport::flip_y` is the sign of a scale the window origin may
        // already have flipped, and the two are not the same claim.
        eprintln!(
            "[pipe] {:?} {:?} clip_height={}",
            engine.viewport_transform(),
            engine.window_origin(),
            engine.surface_clip_height()
        );
        match crate::gpu::pipeline::Pipeline::of(engine) {
            Ok(pipeline) => eprintln!("[pipe] {pipeline:?}"),
            Err(e) => eprintln!("[pipe] undescribable: {e}"),
        }
    }
    // `TRACE_UPLOAD=1`: how many bytes of guest memory this draw would have
    // to be handed to a device, which is the number that decides whether
    // uploading per draw is affordable at all.
    if crate::env_flag!("TRACE_UPLOAD") {
        match crate::gpu::pipeline::Pipeline::of(engine)
            .map_err(|e| Error::Gpu(format!("pipeline: {e}")))
            .and_then(|p| {
                // The `texs` immediates are the shaders' business, not the
                // register file's, and the two stages index *different*
                // constant buffers with the same immediate.
                let mut immediates: Vec<(ShaderStage, u16)> = Vec::new();
                for (stage, program) in [
                    (ShaderStage::VertexB, &vs_program),
                    (ShaderStage::Fragment, &fs_program),
                ] {
                    if let Ok(translated) = crate::gpu::shader::wgsl::translate(program) {
                        immediates
                            .extend(translated.textures.iter().map(|&(imm, _)| (stage, imm)));
                    }
                }
                crate::gpu::upload::Uploads::of(
                    engine,
                    &p,
                    &*ctx,
                    crate::gpu::upload::Banks::Bound,
                    &immediates,
                )
            })
        {
            Ok(uploads) => eprintln!(
                "[up] {} bytes: {} vertex ({}), {} index, {} constant ({} banks), \
                 {} texture ({})",
                uploads.len(),
                uploads.vertex.iter().map(|v| v.bytes.len()).sum::<usize>(),
                uploads.vertex.len(),
                uploads.index.as_ref().map_or(0, |i| i.bytes.len()),
                uploads.constants.iter().map(|c| c.bytes.len()).sum::<usize>(),
                uploads.constants.len(),
                uploads.textures.iter().map(|t| t.bytes.len()).sum::<usize>(),
                uploads.textures.len(),
            ),

            Err(e) => eprintln!("[up] cannot resolve: {e:?}"),
        }
        // The other direction: what a backend holding its surfaces on the
        // device has to hand back before the guest looks at them.
        match crate::gpu::upload::Targets::of(engine) {
            Ok(targets) => eprintln!(
                "[rt] {} bytes back: colour {:?}, depth {:?}",
                targets.len(),
                targets.color.map(|t| (t.format, t.width, t.height, t.len())),
                targets.depth.map(|t| (t.format, t.width, t.height, t.len())),
            ),
            Err(e) => eprintln!("[rt] cannot resolve: {e:?}"),
        }
    }
    // `TRACE_CFG=1`: what each shader's control flow looks like to a
    // translator. Structured control flow is what any shading language wants
    // and what Maxwell's reconvergence stack does not have, so this is the
    // measurement that says how hard translating a given shader would be.
    if crate::env_flag!("TRACE_CFG") {
        for (stage, addr, program) in
            [("vs", vs_binding.addr, &vs_program), ("fs", fs_binding.addr, &fs_program)]
        {
            eprintln!(
                "[cfg] {stage}@{addr:#x} {}",
                crate::gpu::shader::cfg::Cfg::new(program).describe()
            );
        }
    }
    // `TRACE_WGSL=1`: whether each shader can be translated, and what it
    // needs bound. `TRACE_WGSL=dir` writes each complete module to
    // `dir/<stage>_<addr>.wgsl` instead, which is how the emitted text gets
    // in front of a real shader compiler — nothing in this crate can parse
    // WGSL, so `naga --validate` on those files is the only thing that says
    // whether a translation is one.
    // Asked as a flag first: this one wants the *value*, and reading it per
    // draw would put the environment scan back in the path everything else
    // here was just taken out of.
    if crate::env_flag!("TRACE_WGSL") {
        let where_to = std::env::var("TRACE_WGSL").unwrap_or_default();
        use crate::gpu::shader::wgsl::{self, Layout, Stage};
        for (name, stage, addr, program) in [
            ("vs", Stage::Vertex, vs_binding.addr, &vs_program),
            ("fs", Stage::Fragment, fs_binding.addr, &fs_program),
        ] {
            let translated = match wgsl::translate(program) {
                Ok(translated) => translated,
                Err(e) => {
                    eprintln!("[wgsl] {name}@{addr:#x} untranslated: {e}");
                    continue;
                }
            };
            let mut layout = Layout::of(&translated, stage);
            // Neither correction is anything the program says; both come out
            // of the draw's viewport transform.
            if let Ok(pipeline) = crate::gpu::pipeline::Pipeline::of(engine) {
                layout.flip_y = pipeline.viewport.flip_y;
                layout.depth_minus_one_to_one = pipeline.viewport.depth_minus_one_to_one();
            }
            match wgsl::module(&translated, stage, &layout) {
                Ok(_) if where_to == "1" => eprintln!(
                    "[wgsl] {name}@{addr:#x} {} regs, {} attribs, {} varyings, \
                     {} banks, {} textures",
                    translated.registers.len(),
                    layout.attributes.len(),
                    layout.varyings.len(),
                    layout.const_banks.len(),
                    layout.textures.len()
                ),
                Ok(module) => {
                    let path = format!("{where_to}/{name}_{addr:x}.wgsl");
                    match std::fs::write(&path, module) {
                        Ok(()) => eprintln!("[wgsl] {name}@{addr:#x} -> {path}"),
                        Err(e) => eprintln!("[wgsl] {name}@{addr:#x} cannot write {path}: {e}"),
                    }
                }
                Err(e) => eprintln!("[wgsl] {name}@{addr:#x} no module: {e}"),
            }
        }
    }

    for tri in triangles {
        let mut shaded: Vec<ShadedVertex> = Vec::with_capacity(3);
        for &ordinal in &tri {
            let index = if call.indexed {
                read_index(&*ctx, index_base, call.index_format, call.first + ordinal)?
            } else {
                call.first + ordinal
            };
            if let Some(v) = cache.get(&index) {
                shaded.push(*v);
                continue;
            }
            let vs_consts = MemoryConstants {
                ctx: &*ctx,
                bindings: &|bank: u8| engine.bound_constbuf(ShaderStage::VertexB, bank as u32),
                cache: &vs_const_cache,
            };
            let v = shade_vertex(
                &vs_program,
                &attribs,
                &arrays,
                (index, instance_id),
                &*ctx,
                &vs_consts,
                engine.window_origin().lower_left,
            )?;
            cache.insert(index, v);
            shaded.push(v);
        }

        let unclipped = [
            ClipVertex { clip: shaded[0].clip, varyings: shaded[0].varyings },
            ClipVertex { clip: shaded[1].clip, varyings: shaded[1].varyings },
            ClipVertex { clip: shaded[2].clip, varyings: shaded[2].varyings },
        ];
        for piece in clip_near(unclipped) {
            let shaded: [ShadedVertex; 3] = [
                ShadedVertex { clip: piece[0].clip, varyings: piece[0].varyings },
                ShadedVertex { clip: piece[1].clip, varyings: piece[1].varyings },
                ShadedVertex { clip: piece[2].clip, varyings: piece[2].varyings },
            ];
            let projected: Vec<(ScreenVertex, f32, f32)> =
                shaded.iter().map(|v| to_screen(v.clip, viewport)).collect();
            let screen = [projected[0].0, projected[1].0, projected[2].0];
            let inv_w = [projected[0].1, projected[1].1, projected[2].1];
            let window_z = [projected[0].2, projected[1].2, projected[2].2];

            tally.triangles += 1;
            tally.geometry(screen);
            if culls(cull, screen) {
                tally.culled += 1;
                continue;
            }

            let Some(tri) = TriangleSetup::new(screen[0], screen[1], screen[2]) else {
                tally.degenerate += 1;
                continue;
            };
            let (min_x, max_x, min_y, max_y) = tri.bbox(bounds);
            let Some(quad) = quad.as_mut() else {
                let mut sample_z = [0.0f32; MAX_SAMPLES];
                for y in min_y..max_y {
                    for x in min_x..max_x {
                        let covered =
                            fragments.coverage(&tri, window_z, (x, y), &mut sample_z, ctx)?;
                        if covered == 0 {
                            tally.uncovered += 1;
                            continue;
                        }
                        tally.covered += 1;

                        let weights = tri.weights(x as f32 + 0.5, y as f32 + 0.5);
                        let color = with_fragment_env(
                            engine,
                            &*ctx,
                            &fs_const_cache,
                            &descriptors,
                            &blocks,
                            |env| {
                                shade_fragment(
                                    &mut fragment,
                                    &fs_program,
                                    &shaded,
                                    inv_w,
                                    weights,
                                    env,
                                )
                            },
                        )?;
                        // `kil` discards the fragment: no colour, and no depth
                        // write either, which is why the depth store waits
                        // until after shading rather than happening with the
                        // test.
                        let Some(color) = color else {
                            tally.killed += 1;
                            continue;
                        };
                        fragments.write((x, y), covered, &sample_z, color, ctx, &mut tally)?;
                    }
                }
                continue;
            };

            // The quad walk: the same work, in the 2x2 groups a warp shuffle
            // needs. It starts at the even pixel at or below the bounding
            // box's corner, since which pixels share a quad is a property of
            // the grid rather than of the triangle. Pixels outside the box are
            // never sampled — they are outside the scissor as well, and a
            // depth test there would read a target this draw has no business
            // touching — but they are still shaded, as the neighbours the
            // covered lanes difference against.
            let mut sample_z = [[0.0f32; MAX_SAMPLES]; QUAD];
            for y in ((min_y & !1)..max_y).step_by(2) {
                for x in ((min_x & !1)..max_x).step_by(2) {
                    let mut covered = [0u32; QUAD];
                    let mut weights = [[0.0f32; 3]; QUAD];
                    let mut any_covered = false;
                    for lane in 0..QUAD {
                        let (px, py) = quad_pixel(x, y, lane);
                        weights[lane] = tri.weights(px as f32 + 0.5, py as f32 + 0.5);
                        if px < min_x || px >= max_x || py < min_y || py >= max_y {
                            continue;
                        }
                        let mask =
                            fragments.coverage(&tri, window_z, (px, py), &mut sample_z[lane], ctx)?;
                        covered[lane] = mask;
                        if mask == 0 {
                            tally.uncovered += 1;
                        } else {
                            tally.covered += 1;
                            any_covered = true;
                        }
                    }
                    if !any_covered {
                        continue;
                    }

                    let colors = with_fragment_env(
                        engine,
                        &*ctx,
                        &fs_const_cache,
                        &descriptors,
                        &blocks,
                        |env| shade_quad(quad, &fs_program, &shaded, inv_w, weights, env),
                    )?;
                    for lane in 0..QUAD {
                        if covered[lane] == 0 {
                            continue;
                        }
                        let Some(color) = colors[lane] else {
                            tally.killed += 1;
                            continue;
                        };
                        fragments.write(
                            quad_pixel(x, y, lane),
                            covered[lane],
                            &sample_z[lane],
                            color,
                            ctx,
                            &mut tally,
                        )?;
                    }
                }
            }
        }
    }
    tally.report(&call, primitive, blend_target, bounds);
    Ok(())
}

/// Per-draw fragment accounting for `TRACE_DRAW=1`.
///
/// A draw that runs to completion and leaves the framebuffer looking exactly
/// as it did before is indistinguishable, from the outside, from a draw that
/// never ran. This says which stage the fragments died at.
struct DrawTally {
    enabled: bool,
    fs_len: usize,
    triangles: u64,
    culled: u64,
    degenerate: u64,
    uncovered: u64,
    covered: u64,
    killed: u64,
    alpha_killed: u64,
    written: u64,
    first_shaded: Option<[f32; 4]>,
    first_written: Option<[f32; 4]>,
    first_screen: Option<[ScreenVertex; 3]>,
}

impl DrawTally {
    fn new(fs_program: &Program) -> DrawTally {
        DrawTally {
            enabled: crate::env_flag!("TRACE_DRAW"),
            fs_len: fs_program.insns.len(),
            triangles: 0,
            culled: 0,
            degenerate: 0,
            uncovered: 0,
            covered: 0,
            killed: 0,
            alpha_killed: 0,
            written: 0,
            first_shaded: None,
            first_written: None,
            first_screen: None,
        }
    }

    fn geometry(&mut self, screen: [ScreenVertex; 3]) {
        if self.enabled && self.first_screen.is_none() {
            self.first_screen = Some(screen);
        }
    }

    fn shaded(&mut self, color: [f32; 4]) {
        if self.enabled && self.first_shaded.is_none() {
            self.first_shaded = Some(color);
        }
    }

    fn wrote(&mut self, color: [f32; 4]) {
        self.written += 1;
        if self.enabled && self.first_written.is_none() {
            self.first_written = Some(color);
        }
    }

    fn report(
        &self,
        call: &crate::gpu::engine::threed::DrawCall,
        primitive: Primitive,
        blend: BlendTarget,
        bounds: Bounds,
    ) {
        if !self.enabled {
            return;
        }
        let blend = if blend.enabled {
            format!(
                "{:#x},{:#x},{:#x}/{:#x},{:#x},{:#x}",
                blend.equation_rgb,
                blend.func_rgb_src,
                blend.func_rgb_dst,
                blend.equation_alpha,
                blend.func_alpha_src,
                blend.func_alpha_dst
            )
        } else {
            "off".to_string()
        };
        let bounds = (bounds.x0, bounds.y0, bounds.x1, bounds.y1);
        eprintln!(
            "[draw] {primitive:?} count={} indexed={} fs_ops={} bounds={bounds:?} \
             blend={blend} tris={} culled={} degen={} covered={} uncovered={} kil={} \
             a2c={} wrote={} shaded={:?} out={:?} screen={:?}",
            call.count,
            call.indexed,
            self.fs_len,
            self.triangles,
            self.culled,
            self.degenerate,
            self.covered,
            self.uncovered,
            self.killed,
            self.alpha_killed,
            self.written,
            self.first_shaded,
            self.first_written,
            self.first_screen.map(|s| s.map(|v| (v.x, v.y))),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::engine::threed::DrawCall;
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
    use crate::mem::Memory;

    #[test]
    fn triangles_assemble_into_disjoint_triples() {
        assert_eq!(
            assemble(Primitive::Triangles, 6),
            vec![[0, 1, 2], [3, 4, 5]]
        );
    }

    #[test]
    fn triangle_strip_alternates_winding() {
        assert_eq!(
            assemble(Primitive::TriangleStrip, 5),
            vec![[0, 1, 2], [2, 1, 3], [2, 3, 4]]
        );
    }

    #[test]
    fn a_right_triangle_covers_exactly_its_staircase_of_pixels() {
        // (0,0)-(4,0)-(0,4): a clean, hand-verifiable case — the covered set
        // is the classic staircase, with the top-left rule resolving the
        // shared hypotenuse/edges so nothing doubles up or gaps.
        let covered = rasterize_triangle(
            ScreenVertex { x: 0.0, y: 0.0 },
            ScreenVertex { x: 4.0, y: 0.0 },
            ScreenVertex { x: 0.0, y: 4.0 },
            Bounds { x0: 0, y0: 0, x1: 8, y1: 8 },
        );
        let mut covered = covered;
        covered.sort();
        assert_eq!(
            covered,
            vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
        );
    }

    #[test]
    fn a_quad_split_into_two_oppositely_wound_triangles_is_watertight() {
        // The six vertices SDL emits for an 8x8 quad: v0,v1,v2 counter-
        // clockwise and v1,v2,v3 clockwise, so both triangles walk the shared
        // diagonal in the *same* direction. The diagonal runs at exactly 45
        // degrees, which puts pixel centres right on it — the case that left
        // a one-pixel gap through every one of JKSV's save tiles.
        let (a, b) = (ScreenVertex { x: 0.0, y: 0.0 }, ScreenVertex { x: 8.0, y: 0.0 });
        let (c, d) = (ScreenVertex { x: 0.0, y: 8.0 }, ScreenVertex { x: 8.0, y: 8.0 });
        let bounds = Bounds { x0: 0, y0: 0, x1: 8, y1: 8 };

        let mut covered = rasterize_triangle(a, b, c, bounds);
        covered.extend(rasterize_triangle(b, c, d, bounds));
        covered.sort();

        let expected: Vec<(u32, u32)> =
            (0..8).flat_map(|x| (0..8).map(move |y| (x, y))).collect();
        let mut sorted = expected.clone();
        sorted.sort();
        assert_eq!(covered, sorted, "every pixel of the quad exactly once");
    }

    #[test]
    fn rasterization_is_clipped_to_bounds() {
        let covered = rasterize_triangle(
            ScreenVertex { x: 0.0, y: 0.0 },
            ScreenVertex { x: 10.0, y: 0.0 },
            ScreenVertex { x: 0.0, y: 10.0 },
            Bounds { x0: 2, y0: 2, x1: 4, y1: 4 },
        );
        let mut covered = covered;
        covered.sort();
        assert_eq!(covered, vec![(2, 2), (2, 3), (3, 2), (3, 3)]);
    }

    #[test]
    fn a_degenerate_triangle_covers_nothing() {
        let covered = rasterize_triangle(
            ScreenVertex { x: 1.0, y: 1.0 },
            ScreenVertex { x: 2.0, y: 2.0 },
            ScreenVertex { x: 3.0, y: 3.0 },
            Bounds { x0: 0, y0: 0, x1: 8, y1: 8 },
        );
        assert!(covered.is_empty());
    }

    fn harness() -> (Memory, AddressSpace, u64) {
        let mut mem = Memory::new();
        mem.map_zero(0x6000_0000, 0x1000).unwrap();
        let mut vmm = AddressSpace::new();
        let gpu_va = vmm
            .map(0x6000_0000, 0x1000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        (mem, vmm, gpu_va)
    }

    #[test]
    fn fetch_attribute_reads_a_float32_vec4_by_stride() {
        let (mut mem, vmm, base) = harness();
        // Vertex 1's position at base + stride*1 + offset(0).
        let stride = 16u32;
        let addr = base + stride as u64;
        for (i, v) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
            vmm.write_u32(&mut mem, addr + i as u64 * 4, v.to_bits())
                .unwrap();
        }
        let mut stats = Default::default();
        let mut host1x = Host1x::new();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };

        let attrib = VertexAttrib { buffer_id: 0, is_fixed: false, offset: 0, size: 0x01, ty: ATTRIB_TYPE_FLOAT, is_bgra: false };
        let array = VertexArray { enabled: true, stride, start: base, limit: base + 0x1000, divisor: 0 };

        let v = fetch_attribute(attrib, array, 1, &ctx).unwrap();
        assert_eq!(v, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn fetch_attribute_unpacks_unorm8_and_honours_is_bgra() {
        let (mut mem, vmm, base) = harness();
        // Packed BGRA8: B=0x40, G=0x80, R=0xff, A=0x00 (little-endian word).
        let packed = 0x00u32 << 24 | 0xffu32 << 16 | 0x80u32 << 8 | 0x40u32;
        vmm.write_u32(&mut mem, base, packed).unwrap();
        let mut stats = Default::default();
        let mut host1x = Host1x::new();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };

        let attrib = VertexAttrib { buffer_id: 0, is_fixed: false, offset: 0, size: 0x0a, ty: ATTRIB_TYPE_UNORM, is_bgra: true };
        let array = VertexArray { enabled: true, stride: 4, start: base, limit: base + 0x1000, divisor: 0 };

        let v = fetch_attribute(attrib, array, 0, &ctx).unwrap();
        // Decoded as BGRA then swapped to RGBA: R=0xff, G=0x80, B=0x40, A=0x00.
        assert_eq!(v, [1.0, 0x80 as f32 / 255.0, 0x40 as f32 / 255.0, 0.0]);
    }

    /// An integer attribute carries its bits, not its magnitude — the slot is
    /// read back as an integer by the shader, the same way `shade_vertex`
    /// hands over `vertex_id`. Just Dance 2019 binds a signed-byte attribute
    /// (`size 0xa type 3`), and every draw that read one was dropped: 6,480 of
    /// 6,844 in a 400-frame run.
    #[test]
    fn fetch_attribute_unpacks_the_eight_bit_integer_and_normalised_types() {
        let (mut mem, vmm, base) = harness();
        // 0x7F, 0x80, 0x01, 0xFF as four bytes: signed 127, -128, 1, -1.
        let packed = 0xFFu32 << 24 | 0x01u32 << 16 | 0x80u32 << 8 | 0x7Fu32;
        vmm.write_u32(&mut mem, base, packed).unwrap();
        let mut stats = Default::default();
        let mut host1x = Host1x::new();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let array = VertexArray { enabled: true, stride: 4, start: base, limit: base + 0x1000, divisor: 0 };
        let fetch = |ty| {
            let attrib = VertexAttrib { buffer_id: 0, is_fixed: false, offset: 0, size: 0x0a, ty, is_bgra: false };
            fetch_attribute(attrib, array, 0, &ctx).unwrap()
        };

        let sint = fetch(ATTRIB_TYPE_SINT);
        let as_int = |v: f32| v.to_bits() as i32;
        assert_eq!(
            [as_int(sint[0]), as_int(sint[1]), as_int(sint[2]), as_int(sint[3])],
            [127, -128, 1, -1],
            "sint8 sign-extends, and keeps its bits rather than its value"
        );

        let uint = fetch(ATTRIB_TYPE_UINT);
        assert_eq!(
            [uint[0].to_bits(), uint[1].to_bits(), uint[2].to_bits(), uint[3].to_bits()],
            [0x7F, 0x80, 0x01, 0xFF],
            "uint8 is zero-extended"
        );

        let snorm = fetch(ATTRIB_TYPE_SNORM);
        assert_eq!(snorm[0], 1.0);
        assert_eq!(snorm[1], -1.0, "-128 clamps onto -1 rather than past it");
        assert_eq!(snorm[3], -1.0 / 127.0);
    }

    /// Minecraft's every draw reads `4x16` halves (`size 0x3 type 7`), and
    /// with no shape for them all 110 of a frame's 110 draws were dropped —
    /// by the backend, which had no vertex format to build a pipeline from,
    /// and then by the rasterizer it fell back to.
    #[test]
    fn fetch_attribute_unpacks_the_sixteen_bit_types() {
        let (mut mem, vmm, base) = harness();
        // 1.0, -2.0, 0.5, 65504 (the largest finite half) as four halves.
        let halves: [u16; 4] = [0x3C00, 0xC000, 0x3800, 0x7BFF];
        let packed = halves
            .iter()
            .enumerate()
            .fold(0u64, |acc, (i, &h)| acc | u64::from(h) << (i * 16));
        vmm.write_u64(&mut mem, base, packed).unwrap();
        // The signed pattern, eight bytes on: 0x8000 is -32768, the one value
        // both ends of the range map onto -1, and 0x7FFF is +1 exactly.
        vmm.write_u64(&mut mem, base + 8, 0x0001_7FFF_8000_8001).unwrap();
        let mut stats = Default::default();
        let mut host1x = Host1x::new();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let array = VertexArray { enabled: true, stride: 16, start: base, limit: base + 0x1000, divisor: 0 };
        let fetch = |size, ty, offset| {
            let attrib = VertexAttrib { buffer_id: 0, is_fixed: false, offset, size, ty, is_bgra: false };
            fetch_attribute(attrib, array, 0, &ctx).unwrap()
        };

        assert_eq!(fetch(0x03, ATTRIB_TYPE_FLOAT, 0), [1.0, -2.0, 0.5, 65504.0]);
        // A shape that carries fewer than four components pads the rest
        // `(0, 0, 0, 1)`, which is what WebGPU hands a `vec4<f32>` too.
        assert_eq!(fetch(0x0f, ATTRIB_TYPE_FLOAT, 0), [1.0, -2.0, 0.0, 1.0]);
        assert_eq!(fetch(0x05, ATTRIB_TYPE_FLOAT, 0), [1.0, -2.0, 0.5, 1.0]);
        assert_eq!(fetch(0x1b, ATTRIB_TYPE_FLOAT, 0), [1.0, 0.0, 0.0, 1.0]);

        let sint = fetch(0x03, ATTRIB_TYPE_SINT, 0);
        let as_int = |v: f32| v.to_bits() as i32;
        assert_eq!(
            [as_int(sint[0]), as_int(sint[1]), as_int(sint[2]), as_int(sint[3])],
            [0x3C00, -0x4000, 0x3800, 0x7BFF],
            "sint16 sign-extends, and keeps its bits rather than its value"
        );

        let uint = fetch(0x03, ATTRIB_TYPE_UINT, 0);
        assert_eq!(
            [uint[0].to_bits(), uint[1].to_bits(), uint[2].to_bits(), uint[3].to_bits()],
            [0x3C00, 0xC000, 0x3800, 0x7BFF],
            "uint16 is zero-extended"
        );

        let unorm = fetch(0x03, ATTRIB_TYPE_UNORM, 0);
        assert_eq!(unorm[0], 0x3C00 as f32 / 65535.0);

        let snorm = fetch(0x03, ATTRIB_TYPE_SNORM, 8);
        assert_eq!(snorm[0], -1.0, "-32767 is -1");
        assert_eq!(snorm[1], -1.0, "-32768 clamps onto -1 rather than past it");
        assert_eq!(snorm[2], 1.0);
        assert_eq!(snorm[3], 1.0 / 32767.0);
    }

    #[test]
    fn fetch_attribute_from_a_disabled_buffer_is_an_error() {
        let (mut mem, vmm, base) = harness();
        let mut stats = Default::default();
        let mut host1x = Host1x::new();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let attrib = VertexAttrib { buffer_id: 0, is_fixed: false, offset: 0, size: 0x01, ty: ATTRIB_TYPE_FLOAT, is_bgra: false };
        let array = VertexArray { enabled: false, stride: 16, start: base, limit: base, divisor: 0 };
        assert!(fetch_attribute(attrib, array, 0, &ctx).is_err());
    }

    #[test]
    fn a_fixed_attribute_reads_the_vec4_default_instead_of_failing_the_draw() {
        // JKSV leaves attribute 2 marked fixed on draws whose shader never
        // reads it. Erroring dropped those draws entirely -- including its
        // full-screen background quad, which then left the previous frame's
        // chrome showing through.
        let (mut mem, vmm, base) = harness();
        let mut stats = Default::default();
        let mut host1x = Host1x::new();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let attrib = VertexAttrib { buffer_id: 0, is_fixed: true, offset: 0, size: 0x12, ty: ATTRIB_TYPE_FLOAT, is_bgra: false };
        // Deliberately a disabled array: a fixed attribute is not fetched
        // from one at all, so the buffer's state must not matter.
        let array = VertexArray { enabled: false, stride: 0, start: base, limit: base, divisor: 0 };
        assert_eq!(fetch_attribute(attrib, array, 0, &ctx).unwrap(), [0.0, 0.0, 0.0, 1.0]);
    }

    // -- Full-pipeline integration: vertex fetch -> vertex shading ->
    // rasterization -> fragment shading -> real pixel write. Register
    // numbers below are the same raw values `threed.rs`'s own
    // `setup_pitch_target` test helper uses, since its constants are
    // private to that module.

    fn word(low: u32, high: u32) -> [u8; 8] {
        (((high as u64) << 32) | low as u64).to_le_bytes()
    }
    fn block(sched: (u32, u32), a: (u32, u32), b: (u32, u32), c: (u32, u32)) -> Vec<u8> {
        let mut out = Vec::with_capacity(32);
        out.extend_from_slice(&word(sched.0, sched.1));
        out.extend_from_slice(&word(a.0, a.1));
        out.extend_from_slice(&word(b.0, b.1));
        out.extend_from_slice(&word(c.0, c.1));
        out
    }

    /// `gl_Position = aPosition; vColor = aColor;` — composed directly from
    /// the same real, oracle-verified `ld`/`st` b128 attribute-space words
    /// `mvp.vert`'s fixture uses (see `isa`'s module docs), so no new
    /// bit-level guessing is needed for this passthrough.
    fn passthrough_vertex_shader() -> Vec<u8> {
        // Sched words are placeholders reused from mvp.vert's real capture —
        // never all-zero, since `decode_program_from_memory` treats an
        // all-zero first word as "this binary has a Mesa header" (see
        // `MESA_SHADER_HEADER_BYTES`'s doc comment).
        let mut bytes = block(
            (0xfc20070f, 0x081f8441),
            (0x0807ff00, 0xefd9ff80), // ld b128 $r0 a[0x80] 0x0  (aPosition)
            (0x0707ff00, 0xeff1ff80), // st b128 a[0x70] $r0 0x0  (gl_Position)
            (0x0907ff00, 0xefd9ff80), // ld b128 $r0 a[0x90] 0x0  (aColor)
        );
        bytes.extend(block(
            (0xfc2207e1, 0x001f8c40),
            (0x0807ff00, 0xeff1ff80), // st b128 a[0x80] $r0 0x0  (vColor)
            (0x0007000f, 0xe3000000), // exit
            (0, 0),
        ));
        bytes
    }

    /// `oColor = vColor;` — the same real capture `isa`'s module docs and
    /// `shader::interp`'s tests use.
    fn solid_fragment_shader() -> Vec<u8> {
        let mut bytes = block(
            (0xe1a0070f, 0x00240401),
            (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
            (0x00470003, 0x50800000), // mufu rcp $r3 $r0
            (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
        );
        bytes.extend(block(
            (0xb0400341, 0x055c8400),
            (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
            (0x8037ff02, 0xe043ff88), // ipa $r2 a[0x88] $r3 0x0 0x1
            (0xc037ff03, 0xe043ff88), // ipa $r3 a[0x8c] $r3 0x0 0x1
        ));
        bytes.extend(block(
            (0xffe1ffef, 0x001f8000),
            (0x0007000f, 0xe3000000), // exit
            (0, 0),
            (0, 0),
        ));
        bytes
    }

    /// `oColor.r = vColor.r - vColor.r of the pixel beside me`, which is a
    /// horizontal derivative: the `ipa` chain of [`solid_fragment_shader`] up
    /// to the first component, then a `shfl.bfly` reading the lane whose
    /// number differs in the low bit and a subtract.
    ///
    /// The result lands in `r0`, the neighbour's own value stays in `r1`, and
    /// `r3` is still the `w` the interpolation needed — so the colour written
    /// is `(dFdx, neighbour, 0, 1)`.
    fn derivative_fragment_shader() -> Vec<u8> {
        let mut bytes = block(
            (0xe1a0070f, 0x00240401),
            (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
            (0x00470003, 0x50800000), // mufu rcp $r3 $r0
            (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
        );
        bytes.extend(block(
            (0xb0400341, 0x055c8400),
            (0xf0170001, 0xef100070), // shfl.bfly $p0 $r1 $r0 0x1 0x1c
            (0x00170000, 0x5c590000), // fadd $r0 -$r0 $r1
            (0x0007000f, 0xe3000000), // exit
        ));
        bytes
    }

    /// Lay out a render target, both programs, and a 3-vertex buffer
    /// (position vec4 @ offset 0, colour vec4 @ offset 16, stride 32) in one
    /// mapped region, and program `engine`'s registers to match. Returns
    /// `(mem, vmm, engine)`; the caller still needs to write vertex data and
    /// call [`draw`].
    fn pipeline_harness() -> (Memory, AddressSpace, Engine3D) {
        pipeline_harness_with(solid_fragment_shader())
    }

    fn pipeline_harness_with(fragment_shader: Vec<u8>) -> (Memory, AddressSpace, Engine3D) {
        let mut mem = Memory::new();
        mem.map_zero(0x7000_0000, 0x2000).unwrap();
        let mut vmm = AddressSpace::new();
        let base = vmm.map(0x7000_0000, 0x2000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();

        let rt_addr = base;
        let vs_addr = base + 0x200;
        let fs_addr = base + 0x300;
        let vbuf_addr = base + 0x400;

        {
            let mut host1x = Host1x::new();
            let mut stats = Default::default();
            let mut ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
            for (words, addr) in
                [(passthrough_vertex_shader(), vs_addr), (fragment_shader, fs_addr)]
            {
                for (i, chunk) in words.chunks_exact(4).enumerate() {
                    let word = u32::from_le_bytes(chunk.try_into().unwrap());
                    ctx.write_u32(addr + i as u64 * 4, word).unwrap();
                }
            }
        }

        let mut engine = Engine3D::new();
        // Render target: 16x8 pitch-linear RGBA8, matching setup_pitch_target.
        engine.regs.set(0x200, (rt_addr >> 32) as u32);
        engine.regs.set(0x201, rt_addr as u32);
        engine.regs.set(0x202, 16 * 4);
        engine.regs.set(0x203, 8);
        engine.regs.set(0x204, 0xD5); // RGBA8Unorm
        engine.regs.set(0x205, 1 << 12); // IsLinear
        engine.regs.set(0x206, 1);
        // Viewport 0: x=0,y=0,w=16,h=8.
        engine.regs.set(0x300, 16 << 16);
        engine.regs.set(0x301, 8 << 16);
        // SetProgramRegion.
        engine.regs.set(0x582, (base >> 32) as u32);
        engine.regs.set(0x583, base as u32);
        // SetProgram[VertexB] (StageId 1): enabled, offset 0x200.
        engine.regs.set(0x800 + 0x10, 1 | (1 << 4));
        engine.regs.set(0x800 + 0x11, 0x200);
        engine.regs.set(0x800 + 0x13, 8);
        // SetProgram[Fragment] (StageId 5): enabled, offset 0x300.
        engine.regs.set(0x800 + 5 * 0x10, 1 | (5 << 4));
        engine.regs.set(0x800 + 5 * 0x10 + 1, 0x300);
        engine.regs.set(0x800 + 5 * 0x10 + 3, 8);
        // VertexAttribState[0] = aPosition: buffer 0, offset 0, 4x32 float.
        engine.regs.set(0x458, 0x01 << 21 | 7 << 27);
        // VertexAttribState[1] = aColor: buffer 0, offset 16, 4x32 float.
        engine.regs.set(0x458 + 1, (16 << 7) | (0x01 << 21) | (7 << 27));
        // VertexArray[0]: stride 32, enabled.
        engine.regs.set(0x700, 32 | (1 << 12));
        engine.regs.set(0x701, (vbuf_addr >> 32) as u32);
        engine.regs.set(0x702, vbuf_addr as u32);
        engine.regs.set(0x7C0, (vbuf_addr >> 32) as u32);
        engine.regs.set(0x7C1, vbuf_addr as u32 + 3 * 32);

        engine.last_draw = DrawCall { primitive: 4, first: 0, count: 3, indexed: false, index_format: 0 };
        (mem, vmm, engine)
    }

    fn write_vertex(mem: &mut Memory, vmm: &AddressSpace, base: u64, index: u32, pos: [f32; 4], color: [f32; 4]) {
        let addr = base + index as u64 * 32;
        for (i, v) in pos.iter().enumerate() {
            vmm.write_u32(mem, addr + i as u64 * 4, v.to_bits()).unwrap();
        }
        for (i, v) in color.iter().enumerate() {
            vmm.write_u32(mem, addr + 16 + i as u64 * 4, v.to_bits()).unwrap();
        }
    }

    #[test]
    fn a_solid_colour_triangle_matches_a_clear_color_equivalent_fill() {
        let (mut mem, vmm, engine) = pipeline_harness();
        let vbuf_addr = engine.vertex_array(0).start;
        let color = [0.2f32, 0.4, 0.6, 1.0];
        // Screen (0,0)-(16,0)-(0,8): covers the whole 16x8 target's upper
        // triangle half. clip.w = 1 everywhere (no projection), so NDC and
        // clip are the same.
        write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], color);

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: true };
        draw(&engine, &mut ctx).unwrap();

        let rt = engine.render_target(0).unwrap().unwrap();
        let expected = rt.format.encode(color).unwrap();
        // (2,2) and (0,0) are inside the covered half; (12,6) is in the
        // untouched half and must still read as the mapping's initial zero.
        assert_eq!(ctx.read_u32(rt.addr).unwrap() as u128, expected);
        assert_eq!(ctx.read_u32(rt.addr + rt.layout.offset(2 * 4, 2, 16 * 4) as u64).unwrap() as u128, expected);
        assert_eq!(ctx.read_u32(rt.addr + rt.layout.offset(12 * 4, 6, 16 * 4) as u64).unwrap(), 0);
    }

    /// A shader that reads its neighbour's register gets the neighbour, not
    /// its own value: the four pixels of a quad run in lock-step so that the
    /// shuffle has something to read.
    ///
    /// This is what Checkpoint's antialiased text is drawn with — a coverage
    /// differenced against the pixel beside it. Run one pixel at a time,
    /// every lane reads whatever it holds itself and the difference is zero,
    /// which is why the text came out as solid blocks.
    #[test]
    fn a_shuffling_fragment_shader_reads_the_pixel_beside_it() {
        let (mut mem, vmm, engine) = pipeline_harness_with(derivative_fragment_shader());
        let vbuf_addr = engine.vertex_array(0).start;
        // Red ramps from 0 at the left edge of the 16-pixel target to 1 at
        // the right, so it is exactly `(x + 0.5) / 16` at each pixel centre
        // and the difference between neighbours is 1/16 everywhere.
        write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
        write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        draw(&engine, &mut ctx).unwrap();

        let rt = engine.render_target(0).unwrap().unwrap();
        let pixel = |ctx: &mut ExecCtx, x: u32, y: u32| {
            let raw = ctx.read_pixel(rt.addr + rt.texel_offset(x, y) as u64, 4).unwrap();
            rt.format.decode(raw).unwrap()
        };
        let close = |got: f32, want: f32| (got - want).abs() < 1.0 / 255.0;

        // Lane 0 of the quad reads lane 1, which is half a pixel further
        // along the ramp in each direction.
        let left = pixel(&mut ctx, 0, 0);
        assert!(close(left[0], 1.0 / 16.0), "dFdx at (0,0): {left:?}");
        assert!(close(left[1], 1.5 / 16.0), "the neighbour's own value: {left:?}");
        // Lane 1 differences the other way, and a negative colour clamps at
        // zero on the way into an unorm target.
        let right = pixel(&mut ctx, 1, 0);
        assert!(close(right[0], 0.0), "dFdx at (1,0): {right:?}");
        assert!(close(right[1], 0.5 / 16.0), "the neighbour's own value: {right:?}");
    }

    /// The colour write mask keeps the channels it turns off, and a mask with
    /// nothing set writes no colour at all.
    ///
    /// "A Short Hike" turns alpha off for a third of its draws and turns
    /// every channel off for one of them. Writing all four regardless is how
    /// a title's own opacity gets overwritten with whatever its shader left
    /// in alpha.
    #[test]
    fn a_masked_channel_keeps_what_the_target_already_held() {
        let full = [0.2f32, 0.4, 0.6, 0.25];
        // Whatever the target held before the draw, in the same format.
        let before = [1.0f32, 1.0, 1.0, 1.0];

        // `0x1111` is every channel, `0x0111` drops alpha, `0` drops all four.
        for (mask, expected) in [
            (0x1111u32, Some([full[0], full[1], full[2], full[3]])),
            (0x0111, Some([full[0], full[1], full[2], before[3]])),
            (0x1000, Some([before[0], before[1], before[2], full[3]])),
            (0, None),
        ] {
            let (mut mem, vmm, mut engine) = pipeline_harness();
            engine.regs.set(0x680, mask);
            let vbuf_addr = engine.vertex_array(0).start;
            for (i, pos) in [[-1.0f32, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [-1.0, -1.0, 0.0, 1.0]]
                .into_iter()
                .enumerate()
            {
                write_vertex(&mut mem, &vmm, vbuf_addr, i as u32, pos, full);
            }

            let mut host1x = Host1x::new();
            let mut stats = Default::default();
            let mut ctx =
                ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
            let rt = engine.render_target(0).unwrap().unwrap();
            let held = rt.format.encode(before).unwrap();
            ctx.write_pixel(rt.addr, rt.format.bytes_per_pixel, held).unwrap();

            draw(&engine, &mut ctx).unwrap();

            let want = match expected {
                Some(colour) => rt.format.encode(colour).unwrap(),
                // Nothing written: the target still holds what it did.
                None => held,
            };
            assert_eq!(
                ctx.read_u32(rt.addr).unwrap() as u128,
                want,
                "mask {mask:#06x}"
            );
        }
    }

    /// A guest that never writes the mask registers must still draw. Zero is
    /// the register file's initial value and would mean "no channels".
    #[test]
    fn an_unwritten_write_mask_lets_every_channel_through() {
        assert_eq!(Engine3D::new().color_mask(0), [true; 4]);
        assert_eq!(Engine3D::new().color_mask(7), [true; 4]);
    }

    #[test]
    fn a_multisampled_edge_covers_some_samples_of_a_pixel_and_not_others() {
        // The same 16x8 surface, read as 8x4 pixels of 2x2 samples. The
        // triangle's hypotenuse runs from (8,0) to (0,4) in pixels, so it
        // crosses pixel (7, 0) — the one pixel a single-sample rasterizer has
        // to call either wholly covered or wholly empty.
        let (mut mem, vmm, mut engine) = pipeline_harness();
        engine.regs.set(0x300, 8 << 16); // viewport width, in pixels
        engine.regs.set(0x301, 4 << 16); // viewport height, in pixels
        engine.regs.set(0x54D, 1); // MultisampleEnable
        engine.regs.set(0x574, 2); // MultisampleMode = 2x2
        let vbuf_addr = engine.vertex_array(0).start;
        let color = [1.0f32, 1.0, 1.0, 1.0];
        write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], color);

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx =
            ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        draw(&engine, &mut ctx).unwrap();

        let rt = engine.render_target(0).unwrap().unwrap();
        let texel = |ctx: &ExecCtx, x: u32, y: u32| {
            ctx.read_u32(rt.addr + rt.texel_offset(x, y) as u64).unwrap()
        };
        // Pixel (0, 0) is wholly inside: all four of its texels are written.
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert_ne!(texel(&ctx, x, y), 0, "texel ({x}, {y}) of a covered pixel");
        }
        // Pixel (7, 0) straddles the edge. Only the sample at the top left of
        // it — texel (14, 0) — falls inside the triangle.
        assert_ne!(texel(&ctx, 14, 0), 0, "the covered sample of pixel (7, 0)");
        for (x, y) in [(15, 0), (14, 1), (15, 1)] {
            assert_eq!(texel(&ctx, x, y), 0, "texel ({x}, {y}) is outside the edge");
        }
        // Pixel (7, 3) is wholly outside.
        for (x, y) in [(14, 6), (15, 6), (14, 7), (15, 7)] {
            assert_eq!(texel(&ctx, x, y), 0, "texel ({x}, {y}) of an uncovered pixel");
        }
    }

    /// The 16x8 target read as 8x4 pixels of 2x2 samples, with a triangle
    /// that wholly covers pixel (0, 0) so the only thing under test is which
    /// of that pixel's four samples get written.
    fn multisampled_harness() -> (Memory, AddressSpace, Engine3D) {
        let (mut mem, vmm, mut engine) = pipeline_harness();
        engine.regs.set(0x300, 8 << 16); // viewport width, in pixels
        engine.regs.set(0x301, 4 << 16); // viewport height, in pixels
        engine.regs.set(0x54D, 1); // MultisampleEnable
        engine.regs.set(0x574, 2); // MultisampleMode = 2x2
        let vbuf_addr = engine.vertex_array(0).start;
        let color = [1.0f32, 1.0, 1.0, 1.0];
        write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], color);
        (mem, vmm, engine)
    }

    #[test]
    fn a_sample_mask_keeps_only_the_samples_it_names() {
        let (mut mem, vmm, mut engine) = multisampled_harness();
        engine.regs.set(0x3EF, 0b0001); // MultisampleSampleMask: sample 0 only

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx =
            ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        draw(&engine, &mut ctx).unwrap();

        let rt = engine.render_target(0).unwrap().unwrap();
        let texel =
            |ctx: &ExecCtx, x: u32, y: u32| ctx.read_u32(rt.addr + rt.texel_offset(x, y) as u64).unwrap();
        assert_ne!(texel(&ctx, 0, 0), 0, "sample 0 is in the mask");
        for (x, y) in [(1, 0), (0, 1), (1, 1)] {
            assert_eq!(texel(&ctx, x, y), 0, "texel ({x}, {y}) is masked off");
        }
    }

    #[test]
    fn alpha_to_coverage_turns_half_alpha_into_half_the_samples() {
        let (mut mem, vmm, mut engine) = multisampled_harness();
        engine.regs.set(0x54F, 1); // MultisampleControl: AlphaToCoverage
        let vbuf_addr = engine.vertex_array(0).start;
        let color = [1.0f32, 1.0, 1.0, 0.5];
        write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], color);

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx =
            ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        draw(&engine, &mut ctx).unwrap();

        // Half of four samples is two, so the pixel is half covered even
        // though the triangle covers all of it.
        let rt = engine.render_target(0).unwrap().unwrap();
        let texel =
            |ctx: &ExecCtx, x: u32, y: u32| ctx.read_u32(rt.addr + rt.texel_offset(x, y) as u64).unwrap();
        assert_ne!(texel(&ctx, 0, 0), 0);
        assert_ne!(texel(&ctx, 1, 0), 0);
        assert_eq!(texel(&ctx, 0, 1), 0);
        assert_eq!(texel(&ctx, 1, 1), 0);
    }

    #[test]
    fn alpha_to_coverage_spans_none_to_all_of_the_samples() {
        assert_eq!(alpha_coverage(0.0, 4), 0);
        assert_eq!(alpha_coverage(0.25, 4), 0b0001);
        assert_eq!(alpha_coverage(0.5, 4), 0b0011);
        assert_eq!(alpha_coverage(1.0, 4), u32::MAX);
        // Out-of-range alpha clamps rather than shifting past the sample count.
        assert_eq!(alpha_coverage(2.0, 4), u32::MAX);
        assert_eq!(alpha_coverage(-1.0, 4), 0);
        assert_eq!(alpha_coverage(1.0, 16), u32::MAX);
    }

    /// RenderTargetControl lets a guest address its targets in an order other
    /// than the one it bound them in. A clear honoured that mapping and a draw
    /// did not, so a frame could be cleared on one surface and drawn on
    /// another; this pins the two to the same answer.
    #[test]
    fn a_draw_follows_the_render_target_control_mapping() {
        let (mut mem, vmm, mut engine) = pipeline_harness();
        let slot0 = engine.render_target(0).unwrap().unwrap().addr;
        let slot1 = slot0 + 0x800;
        // Bind a second target in physical slot 1, same 16x8 pitch-linear form.
        engine.regs.set(0x210, (slot1 >> 32) as u32);
        engine.regs.set(0x211, slot1 as u32);
        engine.regs.set(0x212, 16 * 4);
        engine.regs.set(0x213, 8);
        engine.regs.set(0x214, 0xD5);
        engine.regs.set(0x215, 1 << 12);
        engine.regs.set(0x216, 1);
        // One target in use, and logical 0 maps onto physical slot 1.
        engine.regs.set(0x487, 1 | (1 << 4));
        assert_eq!(engine.render_target_slot(0), 1);

        let vbuf_addr = engine.vertex_array(0).start;
        let color = [1.0f32, 1.0, 1.0, 1.0];
        write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], color);

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx =
            ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        draw(&engine, &mut ctx).unwrap();

        assert_ne!(ctx.read_u32(slot1).unwrap(), 0, "the draw belongs in slot 1");
        assert_eq!(ctx.read_u32(slot0).unwrap(), 0, "slot 0 is not the mapped target");
    }

    #[test]
    fn an_indexed_draw_reads_its_vertices_through_the_index_buffer() {
        // The same covered half as the direct draw above, but the vertices
        // are stored in the reverse order and an index buffer puts them
        // back. Unity and every other real engine draws this way.
        let (mut mem, vmm, mut engine) = pipeline_harness();
        let vbuf_addr = engine.vertex_array(0).start;
        let color = [0.2f32, 0.4, 0.6, 1.0];
        write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, -1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, 1.0, 0.0, 1.0], color);

        // A u16 index buffer of [2, 1, 0], right after the vertex data.
        let ibuf_addr = vbuf_addr + 3 * 32;
        {
            let mut host1x = Host1x::new();
            let mut stats = Default::default();
            let mut ctx =
                ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
            ctx.write_u32(ibuf_addr, 2 | (1 << 16)).unwrap();
            ctx.write_u32(ibuf_addr + 4, 0).unwrap();
        }
        engine.regs.set(0x5F2, (ibuf_addr >> 32) as u32);
        engine.regs.set(0x5F3, ibuf_addr as u32);
        engine.last_draw =
            DrawCall { primitive: 4, first: 0, count: 3, indexed: true, index_format: 1 };

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx =
            ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: true };
        draw(&engine, &mut ctx).unwrap();

        let rt = engine.render_target(0).unwrap().unwrap();
        let expected = rt.format.encode(color).unwrap();
        assert_eq!(ctx.read_u32(rt.addr).unwrap() as u128, expected);
        assert_eq!(
            ctx.read_u32(rt.addr + rt.layout.offset(2 * 4, 2, 16 * 4) as u64).unwrap() as u128,
            expected
        );
        assert_eq!(ctx.read_u32(rt.addr + rt.layout.offset(12 * 4, 6, 16 * 4) as u64).unwrap(), 0);
    }

    #[test]
    fn back_face_culling_drops_the_wrongly_wound_triangle() {
        let (mut mem, vmm, mut engine) = pipeline_harness();
        let vbuf_addr = engine.vertex_array(0).start;
        let color = [0.2f32, 0.4, 0.6, 1.0];
        // Clockwise in NDC, i.e. the back face when front is CCW.
        write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 1, [-1.0, -1.0, 0.0, 1.0], color);
        write_vertex(&mut mem, &vmm, vbuf_addr, 2, [1.0, 1.0, 0.0, 1.0], color);

        // OGL_SET_CULL enable, front = CCW, cull = BACK.
        engine.regs.set(0x646, 1);
        engine.regs.set(0x647, 0x901);
        engine.regs.set(0x648, 0x405);

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx =
            ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: true };
        draw(&engine, &mut ctx).unwrap();
        let rt = engine.render_target(0).unwrap().unwrap();
        assert_eq!(ctx.read_u32(rt.addr).unwrap(), 0, "a back face must not be drawn");

        // Culling the front face instead draws it.
        engine.regs.set(0x648, 0x404);
        draw(&engine, &mut ctx).unwrap();
        assert_ne!(ctx.read_u32(rt.addr).unwrap(), 0);
    }

    #[test]
    fn a_triangle_crossing_the_near_plane_is_clipped_not_projected() {
        // One vertex behind the eye. Dividing by its w would throw the
        // projected position to the far side of the screen and smear the
        // triangle across the whole target; clipping replaces it with real
        // vertices on the plane first.
        let far = ClipVertex { clip: [-1.0, 1.0, 0.0, 1.0], varyings: [[0.0; 4]; NUM_VARYINGS] };
        let also_far = ClipVertex { clip: [1.0, 1.0, 0.0, 1.0], varyings: [[0.0; 4]; NUM_VARYINGS] };
        let behind = ClipVertex { clip: [0.0, -1.0, 0.0, -1.0], varyings: [[0.0; 4]; NUM_VARYINGS] };

        let pieces = clip_near([far, also_far, behind]);
        assert_eq!(pieces.len(), 2, "a triangle cut by one plane fans into two");
        for piece in &pieces {
            for v in piece {
                assert!(v.clip[3] > 0.0, "no vertex may survive at or behind the eye");
            }
        }

        // Wholly in front: untouched. Wholly behind: gone.
        assert_eq!(clip_near([far, also_far, far]).len(), 1);
        assert!(clip_near([behind, behind, behind]).is_empty());
    }

    #[test]
    fn the_other_triangle_topologies_assemble() {
        assert_eq!(assemble(Primitive::TriangleFan, 5), vec![[0, 1, 2], [0, 2, 3], [0, 3, 4]]);
        assert_eq!(assemble(Primitive::Quads, 4), vec![[0, 1, 2], [0, 2, 3]]);
        assert_eq!(assemble(Primitive::QuadStrip, 4), vec![[0, 1, 2], [2, 1, 3]]);
        // Point and line topologies need their own rasterization; turning
        // them into triangles would draw something that isn't there.
        assert!(assemble(Primitive::Lines, 6).is_empty());
        assert!(assemble(Primitive::Points, 6).is_empty());
    }

    #[test]
    fn three_vertex_colours_interpolate_correctly_at_a_known_interior_point() {
        let (mut mem, vmm, engine) = pipeline_harness();
        let vbuf_addr = engine.vertex_array(0).start;
        let red = [1.0f32, 0.0, 0.0, 1.0];
        let green = [0.0f32, 1.0, 0.0, 1.0];
        let blue = [0.0f32, 0.0, 1.0, 1.0];
        write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], red);
        write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], green);
        write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], blue);

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: true };
        draw(&engine, &mut ctx).unwrap();

        let rt = engine.render_target(0).unwrap().unwrap();
        // Pixel (4,2), centre (4.5,2.5): barycentric weights against
        // (0,0)-(16,0)-(0,8) are (0.40625, 0.28125, 0.3125).
        let expected_color = [
            0.40625 * red[0] + 0.28125 * green[0] + 0.3125 * blue[0],
            0.40625 * red[1] + 0.28125 * green[1] + 0.3125 * blue[1],
            0.40625 * red[2] + 0.28125 * green[2] + 0.3125 * blue[2],
            1.0,
        ];
        let expected = rt.format.encode(expected_color).unwrap();
        let va = rt.addr + rt.layout.offset(4 * 4, 2, 16 * 4) as u64;
        assert_eq!(ctx.read_u32(va).unwrap() as u128, expected);
    }

    #[test]
    fn depth_test_passes_matches_gl_comparison_op() {
        assert!(!depth_test_passes(0x0200, 0.1, 0.5)); // Never
        assert!(depth_test_passes(0x0201, 0.1, 0.5)); // Less
        assert!(!depth_test_passes(0x0201, 0.5, 0.5));
        assert!(depth_test_passes(0x0202, 0.5, 0.5)); // Equal
        assert!(depth_test_passes(0x0203, 0.5, 0.5)); // Lequal
        assert!(depth_test_passes(0x0204, 0.6, 0.5)); // Greater
        assert!(depth_test_passes(0x0205, 0.6, 0.5)); // NotEqual
        assert!(depth_test_passes(0x0206, 0.5, 0.5)); // Gequal
        assert!(depth_test_passes(0x0207, 0.9, 0.1)); // Always
    }

    #[test]
    fn blend_factor_reads_src_and_dst_color_and_alpha() {
        let src = [1.0, 0.5, 0.25, 0.75];
        let dst = [0.0, 1.0, 0.0, 0.2];
        let constant = [0.1, 0.2, 0.3, 0.4];
        assert_eq!(blend_factor(0x4000, src, dst, constant), [0.0; 4]); // Zero
        assert_eq!(blend_factor(0x4001, src, dst, constant), [1.0; 4]); // One
        assert_eq!(blend_factor(0x4300, src, dst, constant), src); // SrcColor
        assert_eq!(blend_factor(0x4302, src, dst, constant), [0.75; 4]); // SrcAlpha
        assert_eq!(blend_factor(0x4303, src, dst, constant), [0.25; 4]); // OneMinusSrcAlpha
        assert_eq!(blend_factor(0x4306, src, dst, constant), dst); // DstColor
        assert_eq!(blend_factor(0xc001, src, dst, constant), constant); // ConstantColor
    }

    #[test]
    fn a_fixed_point_target_clamps_the_blend_source_and_a_float_one_does_not() {
        // A NaN is what this is really for. Every blend factor is a multiply
        // and `NaN * 0` is `NaN`, so a NaN source colour survives a source
        // alpha of zero and reaches the framebuffer as opaque black — which is
        // what put a black box around every icon the Album applet drew. A
        // fixed-point target clamps it to zero before the blend unit ever sees
        // it; a float target stores what it is given.
        let unorm = ColorFormat::from_raw(0xD5).unwrap(); // RGBA8Unorm
        let float = ColorFormat::from_raw(0xCA).unwrap(); // RGBA16Float
        let color = [f32::NAN, 2.0, -1.0, 0.5];
        assert_eq!(source_color(color, unorm), [0.0, 1.0, 0.0, 0.5]);
        let through = source_color(color, float);
        assert!(through[0].is_nan());
        assert_eq!(&through[1..], &[2.0, -1.0, 0.5]);

        // And the whole of it: a transparent NaN over an opaque background
        // leaves the background alone rather than blacking it out.
        let target = BlendTarget {
            enabled: true,
            equation_rgb: 0x8006,   // FuncAdd
            func_rgb_src: 0x4302,   // SrcAlpha
            func_rgb_dst: 0x4303,   // OneMinusSrcAlpha
            equation_alpha: 0x8006,
            func_alpha_src: 0x4302,
            func_alpha_dst: 0x4303,
        };
        let dst = [0.9, 0.9, 0.9, 1.0];
        let src = source_color([f32::NAN, f32::NAN, f32::NAN, 0.0], unorm);
        assert_eq!(blend(target, [0.0; 4], src, dst), dst);
    }

    #[test]
    fn blend_composites_the_default_alpha_blend_state() {
        // dkBlendStateDefaults: colorBlendOp=Add, src=SrcAlpha, dst=InvSrcAlpha;
        // alphaBlendOp=Add, src=One, dst=Zero -- so out.a is just src.a.
        // Values are the real hardware's GL enum codes (see
        // `blend_factor`/`blend_equation`'s doc comments), not deko3d's API
        // numbering.
        let target = BlendTarget {
            enabled: true,
            equation_rgb: 0x8006,   // FuncAdd
            func_rgb_src: 0x4302,   // SrcAlpha
            func_rgb_dst: 0x4303,   // OneMinusSrcAlpha
            equation_alpha: 0x8006, // FuncAdd
            func_alpha_src: 0x4001, // One
            func_alpha_dst: 0x4000, // Zero
        };
        let src = [1.0, 0.0, 0.0, 0.5]; // 50% opaque red
        let dst = [0.0, 0.0, 1.0, 1.0]; // opaque blue
        let out = blend(target, [0.0; 4], src, dst);
        assert_eq!(out, [0.5, 0.0, 0.5, 0.5]);
    }

    #[test]
    fn a_vertex_shader_reads_its_vertex_and_instance_ids() {
        // The Home Menu draws each UI element as one instance of a unit quad
        // and finds that element by `gl_InstanceID`, so a vertex shader that
        // reads zero for it draws every element on top of the first.
        // Both are integers: what has to reach the register is the value's
        // bit pattern, not its numeric value converted to a float.
        use crate::gpu::shader::isa::{Instruction, MemSize, Op, Pred, RZ};

        let mut program = Program::default();
        for (at, op) in [
            (8u32, Op::Ld { dst: 0, offset: INSTANCE_ID_OFFSET, idx: RZ, size: MemSize::B32 }),
            (16, Op::Ld { dst: 1, offset: VERTEX_ID_OFFSET, idx: RZ, size: MemSize::B32 }),
            (24, Op::St { offset: CLIP_POS_OFFSET, idx: RZ, src: 0, size: MemSize::B32 }),
            (32, Op::St { offset: CLIP_POS_OFFSET + 4, idx: RZ, src: 1, size: MemSize::B32 }),
            (40, Op::Exit),
        ] {
            program.offsets.push(at);
            program.insns.push(Instruction { pred: Pred::ALWAYS, op });
        }

        let (mut mem, vmm, _) = harness();
        let mut stats = Default::default();
        let mut host1x = Host1x::new();
        let ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        let consts: std::collections::HashMap<(u8, u16), f32> = Default::default();

        let program = Compiled::new(&program);
        let v = shade_vertex(&program, &[], &[], (7, 42), &ctx, &consts, false).unwrap();
        assert_eq!(v.clip[0].to_bits(), 42, "gl_InstanceID");
        assert_eq!(v.clip[1].to_bits(), 7, "gl_VertexID");
    }

    #[test]
    fn an_instanced_vertex_array_advances_with_the_instance_not_the_vertex() {
        let (mut mem, vmm, base) = harness();
        // Four consecutive floats, one per element of a divisor-2 array.
        for (i, v) in [10.0f32, 20.0, 30.0, 40.0].iter().enumerate() {
            vmm.write_u32(&mut mem, base + i as u64 * 4, v.to_bits()).unwrap();
        }
        let mut stats = Default::default();
        let mut host1x = Host1x::new();
        let ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        let attrib = VertexAttrib {
            buffer_id: 0,
            is_fixed: false,
            offset: 0,
            size: 0x12, // 1x32
            ty: ATTRIB_TYPE_FLOAT,
            is_bgra: false,
        };
        let array =
            VertexArray { enabled: true, stride: 4, start: base, limit: base + 0x1000, divisor: 2 };

        // Instances 0 and 1 share element 0; instances 2 and 3 share element 1.
        for (instance, expected) in [(0u32, 10.0f32), (1, 10.0), (2, 20.0), (3, 20.0)] {
            let element = instance / array.divisor;
            let v = fetch_attribute(attrib, array, element, &ctx).unwrap();
            assert_eq!(v[0], expected, "instance {instance}");
        }
    }

    #[test]
    fn the_same_blend_state_composites_the_same_in_either_numbering() {
        // The Home Menu writes its blend state in the D3D numbering, which
        // this understood as "unrecognised" and therefore `One`/`One` — every
        // one of its draws came out as `src + dst`, which put 2.0 in the
        // alpha channel and washed its dark separator lines to white.
        let gl = BlendTarget {
            enabled: true,
            equation_rgb: 0x8006,
            func_rgb_src: 0x4302,
            func_rgb_dst: 0x4303,
            equation_alpha: 0x8006,
            func_alpha_src: 0x4001,
            func_alpha_dst: 0x4000,
        };
        let d3d = BlendTarget {
            enabled: true,
            equation_rgb: 1,   // Add
            func_rgb_src: 5,   // SrcAlpha
            func_rgb_dst: 6,   // OneMinusSrcAlpha
            equation_alpha: 1, // Add
            func_alpha_src: 2, // One
            func_alpha_dst: 1, // Zero
        };
        let src = [1.0, 0.0, 0.0, 0.5];
        let dst = [0.0, 0.0, 1.0, 1.0];
        assert_eq!(blend(d3d, [0.0; 4], src, dst), [0.5, 0.0, 0.5, 0.5]);
        assert_eq!(blend(d3d, [0.0; 4], src, dst), blend(gl, [0.0; 4], src, dst));
    }

    #[test]
    fn blend_factor_reads_the_d3d_numbering_too() {
        let src = [1.0, 0.5, 0.25, 0.75];
        let dst = [0.0, 1.0, 0.0, 0.2];
        let constant = [0.1, 0.2, 0.3, 0.4];
        for (d3d, gl) in [
            (0x01u32, 0x4000u32), // Zero
            (0x02, 0x4001),       // One
            (0x03, 0x4300),       // SrcColor
            (0x04, 0x4301),       // OneMinusSrcColor
            (0x05, 0x4302),       // SrcAlpha
            (0x06, 0x4303),       // OneMinusSrcAlpha
            (0x07, 0x4304),       // DstAlpha
            (0x08, 0x4305),       // OneMinusDstAlpha
            (0x09, 0x4306),       // DstColor
            (0x0a, 0x4307),       // OneMinusDstColor
            (0x0b, 0x4308),       // SrcAlphaSaturate
            (0x61, 0xc001),       // ConstantColor
            (0x62, 0xc002),       // OneMinusConstantColor
            (0x63, 0xc003),       // ConstantAlpha
            (0x64, 0xc004),       // OneMinusConstantAlpha
        ] {
            assert_eq!(
                blend_factor(d3d, src, dst, constant),
                blend_factor(gl, src, dst, constant),
                "factor {d3d:#x} and {gl:#x} name the same thing"
            );
        }
        // SrcAlphaSaturate is min(srcA, 1 - dstA) on colour and 1 on alpha.
        assert_eq!(blend_factor(0x0b, src, dst, constant), [0.75, 0.75, 0.75, 1.0]);
    }

    #[test]
    fn blend_equation_reads_the_d3d_numbering_too() {
        for (d3d, gl) in [(1u32, 0x8006u32), (2, 0x800a), (3, 0x800b), (4, 0x8007), (5, 0x8008)] {
            assert_eq!(
                blend_equation(d3d, 0.75, 0.25),
                blend_equation(gl, 0.75, 0.25),
                "equation {d3d:#x} and {gl:#x} name the same thing"
            );
        }
    }

    #[test]
    fn a_depth_only_pass_draws_without_a_colour_target() {
        // Just Dance 2017 runs every pass this way: its Z24S8 surface bound
        // as colour target 0, which is a depth surface and so not a colour
        // target at all, and the work done in the depth buffer. The draw has
        // to happen and its colour has to go nowhere.
        let (mut mem, vmm, mut engine) = pipeline_harness();
        let vbuf_addr = engine.vertex_array(0).start;
        let rt_addr = engine.regs.iova(0x200);
        let depth_addr = vbuf_addr + 0x200; // past the vertex buffer, still mapped.

        engine.regs.set(0x204, 0x14); // colour target 0 in Z24S8, as the title binds it
        engine.regs.set(0x3F8, (depth_addr >> 32) as u32);
        engine.regs.set(0x3F9, depth_addr as u32);
        engine.regs.set(0x3FA, 0x0A); // Z32Float
        engine.regs.set(0x3FB, 0); // block_height_gobs = 1
        engine.regs.set(0x48A, 16);
        engine.regs.set(0x48B, 8);
        engine.regs.set(0x4B3, 1); // DepthTestEnable
        engine.regs.set(0x4BA, 1); // DepthWriteEnable
        engine.regs.set(0x4C3, 0x0201); // DepthTestFunc = GL_LESS

        {
            let mut host1x = Host1x::new();
            let mut stats = Default::default();
            let mut ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
            engine.regs.set(0x364, 1.0f32.to_bits()); // CLEAR_DEPTH = far
            engine.write(0x674, 0b1, true, &mut ctx).unwrap();
        }

        let color = [0.2f32, 0.8, 0.2, 1.0];
        for (i, pos) in [[-1.0f32, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [-1.0, -1.0, 0.0, 1.0]]
            .into_iter()
            .enumerate()
        {
            write_vertex(&mut mem, &vmm, vbuf_addr, i as u32, pos, color);
        }

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        assert!(engine.render_target(0).unwrap().is_none(), "no colour target is bound");
        draw(&engine, &mut ctx).unwrap();

        // NDC z = 0 is window z = 0.5 through this viewport, and it passes
        // GL_LESS against the 1.0 the clear left.
        assert_eq!(f32::from_bits(ctx.read_u32(depth_addr).unwrap()), 0.5);
        // And nothing was written where a colour target would have been.
        assert_eq!(ctx.read_u32(rt_addr).unwrap(), 0);
    }

    #[test]
    fn depth_test_keeps_the_nearer_of_two_overlapping_triangles() {
        let (mut mem, vmm, mut engine) = pipeline_harness();
        let vbuf_addr = engine.vertex_array(0).start;
        let depth_addr = vbuf_addr + 0x200; // past the vertex buffer, still inside the mapped region.

        engine.regs.set(0x3F8, (depth_addr >> 32) as u32);
        engine.regs.set(0x3F9, depth_addr as u32);
        engine.regs.set(0x3FA, 0x0A); // Z32Float
        engine.regs.set(0x3FB, 0); // block_height_gobs = 1
        engine.regs.set(0x48A, 16);
        engine.regs.set(0x48B, 8);
        engine.regs.set(0x4B3, 1); // DepthTestEnable
        engine.regs.set(0x4BA, 1); // DepthWriteEnable
        engine.regs.set(0x4C3, 0x0201); // DepthTestFunc = GL_LESS

        {
            // Clear depth to 1.0 (far) first, as real content always does —
            // an unwritten (zeroed) depth buffer would otherwise read as the
            // nearest possible value and reject every draw.
            let mut host1x = Host1x::new();
            let mut stats = Default::default();
            let mut ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
            engine.regs.set(0x364, 1.0f32.to_bits()); // CLEAR_DEPTH
            engine.write(0x674, 0b1, true, &mut ctx).unwrap(); // CLEAR_BUFFERS: clear_depth
        }

        let near = [0.2f32, 0.8, 0.2, 1.0];
        let far = [0.8f32, 0.2, 0.2, 1.0];

        // Far triangle first, at NDC z = 0.5 (covers the whole target).
        write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.5, 1.0], far);
        write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.5, 1.0], far);
        write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.5, 1.0], far);
        {
            let mut host1x = Host1x::new();
            let mut stats = Default::default();
            let mut ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
            draw(&engine, &mut ctx).unwrap();
        }
        // Near triangle second, at NDC z = -0.5 (closer — smaller depth).
        write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, -0.5, 1.0], near);
        write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, -0.5, 1.0], near);
        write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, -0.5, 1.0], near);
        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        draw(&engine, &mut ctx).unwrap();

        let rt = engine.render_target(0).unwrap().unwrap();
        let expected = rt.format.encode(near).unwrap();
        assert_eq!(ctx.read_u32(rt.addr).unwrap() as u128, expected);

        // Drawing the far triangle a third time must not overwrite the
        // nearer surface already there.
        write_vertex(ctx.mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.5, 1.0], far);
        write_vertex(ctx.mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.5, 1.0], far);
        write_vertex(ctx.mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.5, 1.0], far);
        draw(&engine, &mut ctx).unwrap();
        assert_eq!(ctx.read_u32(rt.addr).unwrap() as u128, expected);
    }
}
