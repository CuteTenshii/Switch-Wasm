//! Vertex fetch, primitive assembly, rasterization, and the fragment-shader
//! integration that turns coverage into real pixels.
//!
//! [`draw`] is the top-level entry point `Engine3D::draw_arrays`/
//! `draw_elements` call. Everything it calls is independently testable
//! against synthetic inputs, which is how each earlier stage validated its
//! own piece before this stage wired them together.

use crate::gpu::engine::threed::{
    decode_depth, encode_depth, BlendTarget, CullState, Engine3D, ScissorRect, ShaderStage,
    VertexArray, VertexAttrib, ViewportTransform,
};
use crate::gpu::exec::ExecCtx;
use crate::gpu::shader::interp::{
    ConstantSource, Env, Invocation, MemoryConstants, MemoryTextures, NoTextures, TextureSource,
};
use crate::gpu::shader::{self, Op, Program};
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
    fn edge(a: ScreenVertex, b: ScreenVertex, px: f32, py: f32) -> f32 {
        (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x)
    }
    fn is_top_left(a: ScreenVertex, b: ScreenVertex) -> bool {
        (a.y == b.y && a.x > b.x) || (a.y > b.y)
    }

    let signed_area = edge(v0, v1, v2.x, v2.y);
    if signed_area == 0.0 {
        return Vec::new(); // degenerate triangle: zero area, nothing covered.
    }
    // Wind the triangle counter-clockwise before applying the rule, swapping
    // the weights back on the way out.
    //
    // The top-left tie-break only assigns an on-edge pixel to exactly one of
    // the two triangles sharing that edge when they *walk the edge in
    // opposite directions* — which is true of consistently-wound geometry
    // and false when a quad is emitted as one counter-clockwise and one
    // clockwise triangle, as SDL's does. There both triangles walk the
    // shared diagonal the same way, so they agree on `is_top_left` and the
    // pixels exactly on it belong to both or, when the answer is `false`, to
    // neither. JKSV's save tiles are 128x128 quads whose diagonal runs at
    // exactly 45 degrees, so pixel centres land on it — and every tile came
    // out with a one-pixel gap straight through it.
    let clockwise = signed_area < 0.0;
    let (v1, v2) = if clockwise { (v2, v1) } else { (v1, v2) };
    let area = signed_area.abs();

    let inside = |e: f32, top_left: bool| e > 0.0 || (top_left && e == 0.0);
    let tl01 = is_top_left(v0, v1);
    let tl12 = is_top_left(v1, v2);
    let tl20 = is_top_left(v2, v0);

    let min_x = v0.x.min(v1.x).min(v2.x).floor().max(bounds.x0 as f32) as u32;
    let max_x = (v0.x.max(v1.x).max(v2.x).ceil() as u32).min(bounds.x1);
    let min_y = v0.y.min(v1.y).min(v2.y).floor().max(bounds.y0 as f32) as u32;
    let max_y = (v0.y.max(v1.y).max(v2.y).ceil() as u32).min(bounds.y1);

    let mut out = Vec::new();
    for y in min_y..max_y {
        for x in min_x..max_x {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let e01 = edge(v0, v1, px, py);
            let e12 = edge(v1, v2, px, py);
            let e20 = edge(v2, v0, px, py);
            if inside(e01, tl01) && inside(e12, tl12) && inside(e20, tl20) {
                // w0 is opposite v0 (edge v1-v2), etc. The swap above moved
                // the caller's v1 and v2, so their weights swap back here.
                let (w0, w1, w2) = (e12 / area, e20 / area, e01 / area);
                out.push(if clockwise { (x, y, w0, w2, w1) } else { (x, y, w0, w1, w2) });
            }
        }
    }
    out
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
        0x0a => Some((4, 8)),  // 4x8
        _ => None,
    }
}

/// `DkVtxAttribType` (deko3d.h): `Float = 7`, `Unorm = 2`.
const ATTRIB_TYPE_UNORM: u32 = 2;
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
        (ATTRIB_TYPE_UNORM, 8) => {
            let packed = ctx.read_u32(addr)?;
            for c in 0..components {
                let byte = (packed >> (c * 8)) & 0xff;
                out[c as usize] = byte as f32 / 255.0;
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
/// How many generic varying slots get fetched/interpolated. Real Maxwell
/// supports far more; this is enough for 2D UI (colour, texcoord, a spare)
/// without needing to know how many a given shader pair actually uses —
/// unused slots just carry unread zeros.
const NUM_VARYINGS: usize = 4;
/// Real Maxwell guarantees at least this many vertex attributes; scanning a
/// fixed range means vertex fetch doesn't need the shader to declare how
/// many it reads.
const MAX_VERTEX_ATTRIBS: u32 = 16;
/// Words of shader binary to scan looking for `exit` before giving up —
/// generous for 2D UI, not unbounded (see `shader::MAX_INSTRUCTIONS`'s doc
/// comment for the same reasoning). Reading stops as soon as `exit` is
/// found, so a short real program never touches memory past its own end.
const MAX_PROGRAM_WORDS: u64 = 1024;

/// One vertex after the vertex shader ran: clip-space position plus every
/// generic varying, ready for the perspective divide and interpolation.
#[derive(Clone, Copy)]
struct ShadedVertex {
    clip: [f32; 4],
    varyings: [[f32; 4]; NUM_VARYINGS],
}

/// Decode a shader program straight out of GPU memory, stripping `sched`
/// words and stopping at the first `exit` (mirrors
/// `shader::decode_program`, which needs the whole binary as a slice
/// up front — reading incrementally here means a short real program never
/// touches memory past its own end).
/// Nouveau/Mesa's shader upload convention prepends a fixed-size header
/// (driver bookkeeping, not part of the Maxwell ISA) before the real `sched`/
/// instruction stream — confirmed empirically against a live JKSV capture
/// (its vertex and fragment programs both have a recognisable `sched` word,
/// followed by a real `ld`, starting exactly 0x50 bytes in;
/// `/tmp/dump_vs.bin`/`dump_fs.bin` via a temporary dump added and removed
/// for this investigation). `uam`/deko3d-compiled binaries (hbmenu, this
/// module's own test fixtures) have no such header. The header's own first
/// bytes aren't reliably zero (they carry real driver metadata), so this
/// can't be detected by peeking the first word — instead, decode
/// speculatively assuming no header, and if the very first real instruction
/// (slot 1, right after the first `sched` word) doesn't decode, that's the
/// header showing through: retry assuming one.
const MESA_SHADER_HEADER_BYTES: u64 = 0x50;

fn decode_program_from_memory(ctx: &ExecCtx, addr: u64) -> Result<Program> {
    let first_real_word = ctx.read_u64(addr + 8)?;
    let addr = if matches!(shader::isa::decode(first_real_word).op, Op::Unimplemented { .. }) {
        addr + MESA_SHADER_HEADER_BYTES
    } else {
        addr
    };
    let limit = MAX_PROGRAM_WORDS * 8;
    shader::decode_program_with(&mut |offset: u32| {
        if u64::from(offset) >= limit {
            return Err(Error::Gpu(format!(
                "raster: program read at {offset:#x} is past the {limit:#x}-byte cap"
            )));
        }
        ctx.read_u64(addr + u64::from(offset))
    })
}

fn shade_vertex(
    program: &Program,
    attribs: &[VertexAttrib],
    arrays: &[VertexArray],
    vertex_index: u32,
    ctx: &ExecCtx,
    consts: &dyn ConstantSource,
) -> Result<ShadedVertex> {
    let mut inv = Invocation::new();
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
        let v = fetch_attribute(*attrib, array, vertex_index, ctx)?;
        let base = VARYING_BASE + i as u16 * VARYING_STRIDE;
        for (c, &component) in v.iter().enumerate() {
            inv.attr_in.set(base + c as u16 * 4, component);
        }
    }
    inv.execute(program, &Env::new(consts, &NoTextures))?;

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

/// `DEPTH_TEST_FUNC`'s real hardware type is `gl_comparison_op`
/// (`nv_3ddefs.xml`): literal OpenGL `GL_NEVER`(0x0200)`..=GL_ALWAYS`(0x0207)
/// enum values, not deko3d's simplified 1-8 `DkCompareOp` numbering — real
/// content goes through Mesa's GL driver (JKSV), which writes the GL enum
/// straight through. Confirmed by dumping a live JKSV capture's actual
/// register contents.
fn depth_test_passes(func: u32, new: f32, old: f32) -> bool {
    match func {
        0x0200 => false,
        0x0201 => new < old,
        0x0202 => new == old,
        0x0203 => new <= old,
        0x0204 => new > old,
        0x0205 => new != old,
        0x0206 => new >= old,
        _ => true, // GL_ALWAYS (0x0207), and any unrecognised code.
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
        0x4000 => [0.0; 4],                  // Zero
        0x4300 => src,                       // SrcColor
        0x4301 => src.map(|c| 1.0 - c),       // OneMinusSrcColor
        0x4302 => [src[3]; 4],               // SrcAlpha
        0x4303 => [1.0 - src[3]; 4],         // OneMinusSrcAlpha
        0x4304 => [dst[3]; 4],               // DstAlpha
        0x4305 => [1.0 - dst[3]; 4],         // OneMinusDstAlpha
        0x4306 => dst,                       // DstColor
        0x4307 => dst.map(|c| 1.0 - c),       // OneMinusDstColor
        0xc001 => constant,                  // ConstantColor
        0xc002 => constant.map(|c| 1.0 - c),  // OneMinusConstantColor
        0xc003 => [constant[3]; 4],          // ConstantAlpha
        0xc004 => [1.0 - constant[3]; 4],    // OneMinusConstantAlpha
        _ => [1.0; 4],                        // One (0x4001), and anything unrecognised.
    }
}

/// `BLEND_EQUATION_*`'s real hardware type is `gl_blend_equation`
/// (`nv_3ddefs.xml`): literal `GL_FUNC_ADD`(0x8006)`..=GL_FUNC_REVERSE_
/// SUBTRACT`(0x800b), not deko3d's simplified 1-5 `DkBlendOp` numbering.
fn blend_equation(op: u32, src: f32, dst: f32) -> f32 {
    match op {
        0x800a => src - dst,   // FuncSubtract
        0x800b => dst - src,   // FuncReverseSubtract
        0x8007 => src.min(dst), // Min
        0x8008 => src.max(dst), // Max
        _ => src + dst,         // FuncAdd (0x8006), and anything unrecognised.
    }
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

/// Shade one covered pixel. `inv` is threaded in rather than created here so
/// that a draw allocates one invocation instead of one per covered pixel — a
/// full-screen quad covers 921 600 of them.
fn shade_fragment(
    inv: &mut Invocation,
    program: &Program,
    verts: &[ShadedVertex; 3],
    inv_w: [f32; 3],
    weights: [f32; 3],
    consts: &dyn ConstantSource,
    textures: &dyn TextureSource,
) -> Result<Option<[f32; 4]>> {
    inv.reset();
    let interp_inv_w = weights[0] * inv_w[0] + weights[1] * inv_w[1] + weights[2] * inv_w[2];
    inv.attr_in.set(INV_W_OFFSET, interp_inv_w);
    for slot in 0..NUM_VARYINGS {
        let base = VARYING_BASE + slot as u16 * VARYING_STRIDE;
        for c in 0..4 {
            let over_w = weights[0] * verts[0].varyings[slot][c] * inv_w[0]
                + weights[1] * verts[1].varyings[slot][c] * inv_w[1]
                + weights[2] * verts[2].varyings[slot][c] * inv_w[2];
            inv.attr_in.set(base + c as u16 * 4, over_w);
        }
    }
    inv.execute(program, &Env::new(consts, textures))?;
    if inv.discarded {
        return Ok(None);
    }
    Ok(Some([inv.reg_f32(0), inv.reg_f32(1), inv.reg_f32(2), inv.reg_f32(3)]))
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
    let rt = engine
        .render_target(0)?
        .ok_or_else(|| Error::Gpu("raster: draw with no bound render target".into()))?;

    let vs_binding = engine
        .program(ShaderStage::VertexB)
        .ok_or_else(|| Error::Gpu("raster: draw with no bound vertex program".into()))?;
    let fs_binding = engine
        .program(ShaderStage::Fragment)
        .ok_or_else(|| Error::Gpu("raster: draw with no bound fragment program".into()))?;

    let vs_program = decode_program_from_memory(&*ctx, vs_binding.addr)?;
    let fs_program = decode_program_from_memory(&*ctx, fs_binding.addr)?;

    let attribs: Vec<VertexAttrib> = (0..MAX_VERTEX_ATTRIBS).map(|i| engine.vertex_attrib(i)).collect();
    let arrays: Vec<VertexArray> = (0..MAX_VERTEX_ATTRIBS).map(|i| engine.vertex_array(i)).collect();
    let viewport = engine.viewport_transform();
    let clip = engine.apply_scissor(ScissorRect { x0: 0, y0: 0, x1: rt.width, y1: rt.height });
    let bounds = Bounds { x0: clip.x0, y0: clip.y0, x1: clip.x1, y1: clip.y1 };
    let depth = engine.depth_target()?;
    let depth_state = engine.depth_state();
    let blend_target = engine.blend_target(0);
    let blend_constant = engine.blend_constant();
    let cull = engine.cull_state();

    let index_base = if call.indexed { engine.index_array_start() } else { 0 };
    let triangles = assemble(Primitive::from_raw(call.primitive)?, call.count);
    // One shaded vertex per *index*, cached: an indexed mesh reuses vertices
    // heavily, and re-running the vertex shader for each reference is the
    // single most expensive thing this loop can do.
    let mut cache: std::collections::HashMap<u32, ShadedVertex> = std::collections::HashMap::new();
    // One fragment invocation for the whole draw, reset per pixel.
    let mut fragment = Invocation::new();

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
            };
            let v = shade_vertex(&vs_program, &attribs, &arrays, index, &*ctx, &vs_consts)?;
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

            if culls(cull, screen) {
                continue;
            }

            for (x, y, w0, w1, w2) in
                rasterize_triangle_weighted(screen[0], screen[1], screen[2], bounds)
            {
                let z01 = w0 * window_z[0] + w1 * window_z[1] + w2 * window_z[2];
                if let (true, Some(dt)) = (depth_state.test_enabled, depth) {
                    let dva = dt.addr + dt.layout.offset(x * dt.bytes, y, dt.width * dt.bytes) as u64;
                    let old_raw = ctx.read_pixel(dva, dt.bytes)?;
                    let old = decode_depth(old_raw, dt.depth_bits);
                    if !depth_test_passes(depth_state.func, z01, old) {
                        continue;
                    }
                }

                let color = {
                    let fs_consts = MemoryConstants {
                        ctx: &*ctx,
                        bindings: &|bank: u8| engine.bound_constbuf(ShaderStage::Fragment, bank as u32),
                    };
                    let fs_textures = MemoryTextures {
                        ctx: &*ctx,
                        tex_header_pool: engine.tex_header_pool(),
                        tex_sampler_pool: engine.tex_sampler_pool(),
                    };
                    shade_fragment(
                        &mut fragment,
                        &fs_program,
                        &shaded,
                        [inv_w[0], inv_w[1], inv_w[2]],
                        [w0, w1, w2],
                        &fs_consts,
                        &fs_textures,
                    )?
                };
                // `kil` discards the fragment: no colour, and no depth
                // write either, which is why the depth store waits until
                // after shading rather than happening with the test.
                let Some(color) = color else { continue };

                if depth_state.test_enabled && depth_state.write_enabled {
                    if let Some(dt) = depth {
                        let dva =
                            dt.addr + dt.layout.offset(x * dt.bytes, y, dt.width * dt.bytes) as u64;
                        ctx.write_pixel(dva, dt.bytes, encode_depth(z01, dt.depth_bits))?;
                    }
                }

                let bpp = rt.format.bytes_per_pixel;
                let va = rt.addr + rt.layout.offset(x * bpp, y, rt.width * bpp) as u64;
                let out = if blend_target.enabled {
                    let dst = rt.format.decode(ctx.read_pixel(va, bpp)?)?;
                    blend(blend_target, blend_constant, color, dst)
                } else {
                    color
                };
                ctx.write_pixel(va, bpp, rt.format.encode(out)?)?;
            }
        }
    }
    Ok(())
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

    /// Lay out a render target, both programs, and a 3-vertex buffer
    /// (position vec4 @ offset 0, colour vec4 @ offset 16, stride 32) in one
    /// mapped region, and program `engine`'s registers to match. Returns
    /// `(mem, vmm, engine)`; the caller still needs to write vertex data and
    /// call [`draw`].
    fn pipeline_harness() -> (Memory, AddressSpace, Engine3D) {
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
            for (words, addr) in [
                (passthrough_vertex_shader(), vs_addr),
                (solid_fragment_shader(), fs_addr),
            ] {
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
