//! The fixed-function state a draw runs under, in one typed value.
//!
//! Everything here is already in [`Engine3D`]'s registers, and the software
//! rasterizer reads it straight out of them a field at a time. A GPU backend
//! cannot: a pipeline is built once and then drawn with, so the state has to
//! be a *value* — something to hash, compare against the last draw's, and
//! look a cached pipeline up by. That is what this is.
//!
//! # It answers in the target's vocabulary, not the hardware's
//!
//! A blend factor arrives as `0x4302` or as `0x05` depending on whose driver
//! wrote the register — Mesa writes the GL enum, deko3d and nvn write the
//! D3D one — and neither number means anything to a shading API. So this
//! resolves them, and everything else, into the vocabulary WebGPU uses. A
//! backend that had to re-decode `0x4302` would be a second place for the
//! two to disagree about what a draw meant.
//!
//! # What it refuses
//!
//! Maxwell can describe draws WebGPU has no way to express: a triangle fan,
//! a blend factor built from the constant colour's alpha alone, a vertex
//! attribute stepped once per two instances. Every one of those is an
//! [`Unsupported`] rather than an approximation, because the point of a
//! second backend is to agree with the first — and the caller's answer to
//! being told is to run that draw on the software rasterizer, which is a
//! normal thing to do and not a failure.
//!
//! It also refuses things the software rasterizer *does* accept by falling
//! back on a default. `blend_factor` answers `One` for a code it does not
//! know, which is a reasonable thing for a rasterizer that must produce a
//! pixel and the wrong thing for a description that can say "I don't know".

use crate::gpu::engine::threed::{
    BlendTarget, DepthLayout, DepthState, Engine3D, ScissorRect, VertexArray, ViewportTransform,
};
use crate::gpu::raster::Primitive;
use crate::gpu::surface::ColorFormat;
use std::fmt;

/// A piece of state that has no WebGPU spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {
    /// A primitive topology WebGPU does not have. Fans, quads, loops and
    /// polygons all need the index buffer expanded first, which is work for
    /// a backend rather than a description.
    Topology(Primitive),
    /// A blend factor with no equivalent, or a code neither numbering
    /// recognises.
    BlendFactor { code: u32 },
    BlendEquation { code: u32 },
    /// A depth comparison the software rasterizer does not implement either
    /// — see [`Depth::compare`].
    DepthCompare { code: u32 },
    /// A colour or depth surface format with no equivalent.
    Format { raw: u32 },
    /// A vertex attribute's component count and type, as
    /// `DkVtxAttribSize`/`DkVtxAttribType`.
    VertexFormat { size: u32, ty: u32 },
    /// An attribute declared BGRA, which swaps two components after
    /// fetching. WebGPU has no BGRA vertex format; a backend could swizzle
    /// it in the entry point instead.
    BgraAttribute { location: u32 },
    /// A vertex array stepped once every `divisor` instances. WebGPU steps
    /// per instance or per vertex and has nothing in between.
    InstanceDivisor { divisor: u32 },
    /// The engine could not resolve a piece of state at all.
    State(String),
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsupported::Topology(primitive) => write!(f, "no topology for {primitive:?}"),
            Unsupported::BlendFactor { code } => write!(f, "no blend factor for {code:#x}"),
            Unsupported::BlendEquation { code } => write!(f, "no blend equation for {code:#x}"),
            Unsupported::DepthCompare { code } => write!(f, "no depth comparison for {code:#x}"),
            Unsupported::Format { raw } => write!(f, "no surface format for {raw:#x}"),
            Unsupported::VertexFormat { size, ty } => {
                write!(f, "no vertex format for size {size:#x} type {ty}")
            }
            Unsupported::BgraAttribute { location } => {
                write!(f, "attribute {location} is BGRA, which has no vertex format")
            }
            Unsupported::InstanceDivisor { divisor } => {
                write!(f, "a vertex array stepped once every {divisor} instances")
            }
            Unsupported::State(why) => write!(f, "{why}"),
        }
    }
}

/// How vertices assemble into primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Topology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

/// Which winding is the front face, **in window space** — after the viewport
/// transform, so a transform that mirrors y has already reversed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrontFace {
    Ccw,
    Cw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cull {
    None,
    Front,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compare {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// A surface format, named as WebGPU names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Rgb10a2Unorm,
    R32Float,
    Rgba16Float,
    Rgba32Float,
    Depth16Unorm,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
    Depth32FloatStencil8,
    // Compressed, and so sampled-only: nothing renders into these. They are
    // in the same enum because WebGPU has one format enum for both, and
    // because a texture upload and a render target ask the same question of
    // the same raw codes.
    Bc1RgbaUnorm,
    Bc1RgbaUnormSrgb,
    Bc2RgbaUnorm,
    Bc2RgbaUnormSrgb,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSrgb,
    Bc4RUnorm,
    Bc4RSnorm,
    Bc5RgUnorm,
    Bc5RgSnorm,
    Bc6hRgbUfloat,
    Bc6hRgbFloat,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSrgb,
}

/// What a vertex format's components arrive as. WebGPU makes this part of
/// the match between a format and the shader input it feeds, so it decides
/// whether an attribute is declared `vec4<f32>`, `vec4<i32>` or `vec4<u32>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttributeBase {
    Float,
    Sint,
    Uint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Unorm8x4,
    Snorm8x4,
    Sint8x4,
    Uint8x4,
}

impl VertexFormat {
    /// The normalized formats are floats by the time a shader sees them;
    /// only the integer ones carry their bits through, which is what
    /// `raster::fetch_attribute` leaves in the slot for one as well.
    pub fn base(self) -> AttributeBase {
        match self {
            VertexFormat::Sint8x4 => AttributeBase::Sint,
            VertexFormat::Uint8x4 => AttributeBase::Uint,
            _ => AttributeBase::Float,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepMode {
    Vertex,
    Instance,
}

/// One side of a blend: `src * src_factor <op> dst * dst_factor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Blend {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorTarget {
    pub format: Format,
    /// `None` when blending is off, which is WebGPU's own spelling for it.
    pub blend: Option<Blend>,
    /// Which of R, G, B, A the draw may write; see `Engine3D::color_mask`.
    /// Part of the pipeline rather than of the pass because that is where
    /// WebGPU keeps it, and because it is what the guest changes it with.
    pub write_mask: [bool; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Depth {
    pub format: Format,
    pub write_enabled: bool,
    /// `Always` when the test is off, which is what a disabled depth test
    /// does and what WebGPU wants written down.
    pub compare: Compare,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    /// Byte offset within the buffer's element.
    pub offset: u32,
    /// Which `@location` the shader reads it as. Maxwell's attribute slot
    /// number, which is also the generic `a[]` slot it lands in.
    pub location: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexBuffer {
    /// Which of Maxwell's sixteen vertex arrays this is.
    pub index: u32,
    pub stride: u32,
    pub step: StepMode,
    pub attributes: Vec<VertexAttribute>,
}

/// The viewport, as a rectangle rather than as the scale-and-translate pair
/// the hardware holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
    /// The transform's z scale and translate, unresolved.
    ///
    /// Kept because `min_depth`/`max_depth` are read under one of two
    /// conventions and these are not — see
    /// [`Viewport::depth_minus_one_to_one`].
    pub depth_scale: f32,
    pub depth_translate: f32,
    /// Whether the guest's transform mirrors y — a negative `scale_y`, which
    /// is how a driver reconciles GL's bottom-left window origin with a
    /// render target whose row 0 is at the top.
    ///
    /// WebGPU has no negative viewport height, so a backend reproduces this
    /// by negating `position.y` in the vertex entry point. It also reverses
    /// which winding is front, which is why [`Pipeline::front_face`] is
    /// already resolved in window space.
    pub flip_y: bool,
}

/// Everything a draw's pipeline is built from.
#[derive(Debug, Clone, PartialEq)]
pub struct Pipeline {
    pub topology: Topology,
    pub front_face: FrontFace,
    pub cull: Cull,
    /// Colour target 0, or `None` for a depth-only pass — which is a real
    /// thing a title does, not a gap: Just Dance 2017 renders every pass
    /// that way.
    pub target: Option<ColorTarget>,
    pub depth: Option<Depth>,
    pub vertex_buffers: Vec<VertexBuffer>,
    /// Attribute slots the draw binds no buffer to. They read `(0, 0, 0, 1)`
    /// — a shader reading an input this draw supplies nothing for is
    /// well-defined — and a backend feeds them from a constant of its own.
    pub fixed_attributes: Vec<u32>,
    pub samples: u32,
    pub sample_mask: u32,
    pub alpha_to_coverage: bool,
    pub viewport: Viewport,
    pub scissor: ScissorRect,
    pub blend_constant: [f32; 4],
}

/// How many vertex arrays and attribute slots the engine has.
const MAX_VERTEX_ATTRIBS: u32 = 16;

impl Pipeline {
    /// Read the state [`Engine3D::last_draw`] would run under.
    pub fn of(engine: &Engine3D) -> Result<Pipeline, Unsupported> {
        let state = |what: &str, e: crate::Error| Unsupported::State(format!("{what}: {e:?}"));

        let primitive = Primitive::from_raw(engine.last_draw.primitive)
            .map_err(|e| state("primitive", e))?;
        let cull = engine.cull_state();
        let viewport = viewport(engine.viewport_transform());

        let slot = engine.render_target_slot(0);
        let rt = engine.render_target(slot).map_err(|e| state("colour target", e))?;
        let dt = engine.depth_target().map_err(|e| state("depth target", e))?;
        let target = match rt {
            Some(rt) => Some(ColorTarget {
                format: color_format(rt.format)?,
                blend: blend(engine.blend_target(0))?,
                write_mask: engine.color_mask(slot),
            }),
            None => None,
        };
        let depth = match dt {
            Some(dt) => Some(self::depth(dt.format, engine.depth_state())?),
            None => None,
        };

        // The extent the scissor is resolved against is the target's, in
        // pixels — which differ from texels on a multisampled surface.
        let grid = engine.sample_grid().map_err(|e| state("sample grid", e))?;
        let extent = match (rt, dt) {
            (Some(rt), _) => (rt.width, rt.height),
            (None, Some(dt)) => (dt.width, dt.height),
            (None, None) => {
                return Err(Unsupported::State(
                    "a draw with neither a colour nor a depth target".into(),
                ))
            }
        };
        let (width, height) = grid.pixels(extent.0, extent.1);
        let scissor = engine.apply_scissor(ScissorRect { x0: 0, y0: 0, x1: width, y1: height });

        let (vertex_buffers, fixed_attributes) = vertex_buffers(engine)?;

        Ok(Pipeline {
            topology: topology(primitive)?,
            // In window space, after the transform: mirroring y reverses
            // which way a triangle winds, so a front face resolved from the
            // register alone would be back-to-front on a flipped viewport.
            front_face: match (cull.front_ccw, viewport.flip_y) {
                (true, false) | (false, true) => FrontFace::Ccw,
                _ => FrontFace::Cw,
            },
            cull: match (cull.enabled, cull.cull_front, cull.cull_back) {
                (false, _, _) | (_, false, false) => Cull::None,
                (_, true, true) => {
                    // Culling both faces draws nothing, which WebGPU cannot
                    // say. It is also not something a draw means to do.
                    return Err(Unsupported::State("a draw that culls both faces".into()));
                }
                (_, true, false) => Cull::Front,
                (_, false, true) => Cull::Back,
            },
            target,
            depth,
            vertex_buffers,
            fixed_attributes,
            samples: grid.samples_x * grid.samples_y,
            sample_mask: engine.sample_mask(),
            alpha_to_coverage: engine.alpha_to_coverage(),
            viewport,
            scissor,
            blend_constant: engine.blend_constant(),
        })
    }
}

/// `window = ndc * scale + translate` as a rectangle. NDC spans `-1..=1`, so
/// each axis covers `translate - |scale| ..= translate + |scale|` and a
/// negative scale means the axis is mirrored rather than that the rectangle
/// is.
fn viewport(transform: ViewportTransform) -> Viewport {
    let [sx, sy, sz] = transform.scale;
    let [tx, ty, tz] = transform.translate;
    Viewport {
        x: tx - sx.abs(),
        y: ty - sy.abs(),
        width: 2.0 * sx.abs(),
        height: 2.0 * sy.abs(),
        // Depth is not mirrored by anything this has seen, so the near plane
        // is the low end of the range the transform maps onto.
        min_depth: (tz - sz).min(tz + sz),
        max_depth: (tz - sz).max(tz + sz),
        depth_scale: sz,
        depth_translate: tz,
        flip_y: sy < 0.0,
    }
}

impl Viewport {
    /// Whether the guest's clip space runs z from `-w` to `w` rather than
    /// from `0` to `w`, which decides whether a vertex entry point has to
    /// remap `position.z` — WebGPU clips z the way Vulkan does, and a shader
    /// whose z is left alone has the near half of its frustum clipped away.
    ///
    /// This is an inference, not a reading. A driver using GL's range writes
    /// scale 0.5 and translate 0.5, because that is what maps `-1..1` onto a
    /// `0..1` window depth; one using Vulkan's writes scale 1.0 and
    /// translate 0.0. Both give the same window range, so the two numbers
    /// tell them apart only by which of those shapes they have — and a
    /// Vulkan-convention guest that also narrowed its depth range would look
    /// like neither. Every transform this has seen is the first shape.
    pub fn depth_minus_one_to_one(&self) -> bool {
        self.depth_translate - self.depth_scale >= 0.0
    }
}

fn topology(primitive: Primitive) -> Result<Topology, Unsupported> {
    match primitive {
        Primitive::Points => Ok(Topology::PointList),
        Primitive::Lines => Ok(Topology::LineList),
        Primitive::LineStrip => Ok(Topology::LineStrip),
        Primitive::Triangles => Ok(Topology::TriangleList),
        Primitive::TriangleStrip => Ok(Topology::TriangleStrip),
        // A fan, a quad, a loop and a polygon all become triangles by
        // rewriting the index buffer, which `raster::assemble` does on the
        // CPU and a pipeline cannot describe.
        other => Err(Unsupported::Topology(other)),
    }
}

/// Both numberings, as [`crate::gpu::raster`]'s `blend_factor` takes them:
/// Mesa writes the GL enum straight through and deko3d and nvn write the D3D
/// one. A code in neither is an error here where the rasterizer answers
/// `One` — a rasterizer has to produce a pixel, and this does not.
fn blend_factor(code: u32) -> Result<BlendFactor, Unsupported> {
    Ok(match code {
        0x01 | 0x4000 => BlendFactor::Zero,
        0x02 | 0x4001 => BlendFactor::One,
        0x03 | 0x4300 => BlendFactor::Src,
        0x04 | 0x4301 => BlendFactor::OneMinusSrc,
        0x05 | 0x4302 => BlendFactor::SrcAlpha,
        0x06 | 0x4303 => BlendFactor::OneMinusSrcAlpha,
        0x07 | 0x4304 => BlendFactor::DstAlpha,
        0x08 | 0x4305 => BlendFactor::OneMinusDstAlpha,
        0x09 | 0x4306 => BlendFactor::Dst,
        0x0a | 0x4307 => BlendFactor::OneMinusDst,
        0x0b | 0x4308 => BlendFactor::SrcAlphaSaturated,
        0x61 | 0xc001 => BlendFactor::Constant,
        0x62 | 0xc002 => BlendFactor::OneMinusConstant,
        // ConstantAlpha and its complement broadcast the blend constant's
        // alpha to all four channels. WebGPU's `Constant` is per-channel and
        // there is no alpha-only form of it.
        code => return Err(Unsupported::BlendFactor { code }),
    })
}

fn blend_equation(code: u32) -> Result<BlendOperation, Unsupported> {
    Ok(match code {
        0x1 | 0x8006 => BlendOperation::Add,
        0x2 | 0x800a => BlendOperation::Subtract,
        0x3 | 0x800b => BlendOperation::ReverseSubtract,
        0x4 | 0x8007 => BlendOperation::Min,
        0x5 | 0x8008 => BlendOperation::Max,
        code => return Err(Unsupported::BlendEquation { code }),
    })
}

fn blend(target: BlendTarget) -> Result<Option<Blend>, Unsupported> {
    if !target.enabled {
        return Ok(None);
    }
    Ok(Some(Blend {
        color: BlendComponent {
            src_factor: blend_factor(target.func_rgb_src)?,
            dst_factor: blend_factor(target.func_rgb_dst)?,
            operation: blend_equation(target.equation_rgb)?,
        },
        alpha: BlendComponent {
            src_factor: blend_factor(target.func_alpha_src)?,
            dst_factor: blend_factor(target.func_alpha_dst)?,
            operation: blend_equation(target.equation_alpha)?,
        },
    }))
}

/// The GL comparison enums, which are the ones [`crate::gpu::raster`]'s
/// `depth_test_passes` implements. `DepthState`'s doc describes a one-based
/// `DepthTestFunc` numbering as well; nothing decodes that, so a pipeline
/// claiming to would disagree with the rasterizer it is meant to match.
fn depth_compare(code: u32) -> Result<Compare, Unsupported> {
    Ok(match code {
        1 | 0x0200 => Compare::Never,
        2 | 0x0201 => Compare::Less,
        3 | 0x0202 => Compare::Equal,
        4 | 0x0203 => Compare::LessEqual,
        5 | 0x0204 => Compare::Greater,
        6 | 0x0205 => Compare::NotEqual,
        7 | 0x0206 => Compare::GreaterEqual,
        8 | 0x0207 => Compare::Always,
        code => return Err(Unsupported::DepthCompare { code }),
    })
}

/// The WebGPU format a depth surface's layout names.
pub fn depth_format(layout: DepthLayout) -> Result<Format, Unsupported> {
    Ok(match (layout.bytes, layout.depth_bits, layout.stencil_shift.is_some()) {
        (2, 16, false) => Format::Depth16Unorm,
        (4, 24, false) => Format::Depth24Plus,
        (4, 24, true) => Format::Depth24PlusStencil8,
        (4, 0, false) => Format::Depth32Float,
        (8, 0, true) => Format::Depth32FloatStencil8,
        _ => return Err(Unsupported::Format { raw: layout.bytes }),
    })
}

fn depth(layout: DepthLayout, state: DepthState) -> Result<Depth, Unsupported> {
    Ok(Depth {
        format: depth_format(layout)?,
        write_enabled: state.write_enabled,
        // A disabled test passes everything, which is what `Always` says.
        compare: if state.test_enabled { depth_compare(state.func)? } else { Compare::Always },
    })
}

/// The WebGPU format a colour target's raw code names.
///
/// Follows [`crate::gpu::surface`]'s reading of those codes rather than a
/// second one — the tests below check the two still agree about how wide a
/// pixel is and whether it is sRGB, which is what would drift.
pub(crate) fn color_format(format: ColorFormat) -> Result<Format, Unsupported> {
    Ok(match format.raw {
        0xD5 | 0xD7 | 0xD8 | 0xD9 | 0xF9 => Format::Rgba8Unorm,
        0xD6 | 0xFA => Format::Rgba8UnormSrgb,
        0xCF | 0xE6 | 0xFD | 0xFE => Format::Bgra8Unorm,
        0xD0 | 0xE7 => Format::Bgra8UnormSrgb,
        0xD1 => Format::Rgb10a2Unorm,
        0xE5 => Format::R32Float,
        0xCA | 0xCE => Format::Rgba16Float,
        0xC0 | 0xC3 => Format::Rgba32Float,
        0xEA => Format::Rg8Unorm,
        0xF3 => Format::R8Unorm,
        raw => return Err(Unsupported::Format { raw }),
    })
}

/// `DkVtxAttribType` (deko3d.h), as `crate::gpu::raster` also names them.
const ATTRIB_TYPE_SNORM: u32 = 1;
const ATTRIB_TYPE_UNORM: u32 = 2;
const ATTRIB_TYPE_SINT: u32 = 3;
const ATTRIB_TYPE_UINT: u32 = 4;
const ATTRIB_TYPE_FLOAT: u32 = 7;

/// The formats [`crate::gpu::raster`]'s `fetch_attribute` decodes, and no
/// others: a pipeline that claimed one the rasterizer cannot fetch would
/// draw something the reference could not be compared against.
fn vertex_format(size: u32, ty: u32) -> Result<VertexFormat, Unsupported> {
    Ok(match (size, ty) {
        (0x01, ATTRIB_TYPE_FLOAT) => VertexFormat::Float32x4,
        (0x02, ATTRIB_TYPE_FLOAT) => VertexFormat::Float32x3,
        (0x04, ATTRIB_TYPE_FLOAT) => VertexFormat::Float32x2,
        (0x12, ATTRIB_TYPE_FLOAT) => VertexFormat::Float32,
        // Size `0x0a` is `4x8`, the only 8-bit shape both `fetch_attribute`
        // decodes and WebGPU spells: it has no one- or three-component 8-bit
        // format, and a shorter one would be padded `(0, 0, 0, 1)` as floats
        // where the rasterizer pads an integer slot with those *bits*.
        (0x0a, ATTRIB_TYPE_UNORM) => VertexFormat::Unorm8x4,
        (0x0a, ATTRIB_TYPE_SNORM) => VertexFormat::Snorm8x4,
        (0x0a, ATTRIB_TYPE_SINT) => VertexFormat::Sint8x4,
        (0x0a, ATTRIB_TYPE_UINT) => VertexFormat::Uint8x4,
        (size, ty) => return Err(Unsupported::VertexFormat { size, ty }),
    })
}

/// Group the attributes by the array they read, which is the shape WebGPU
/// wants and the opposite of the register file's.
fn vertex_buffers(engine: &Engine3D) -> Result<(Vec<VertexBuffer>, Vec<u32>), Unsupported> {
    let mut buffers: Vec<VertexBuffer> = Vec::new();
    let mut fixed = Vec::new();
    for location in 0..MAX_VERTEX_ATTRIBS {
        let attrib = engine.vertex_attrib(location);
        // Size 0 is what an unconfigured slot reads back as, so it means
        // "not used" rather than "a format I do not know".
        if attrib.size == 0 {
            continue;
        }
        if attrib.is_fixed {
            fixed.push(location);
            continue;
        }
        let array = engine.vertex_array(attrib.buffer_id);
        if !array.enabled {
            // The attribute claims to read an array the draw never turned
            // on. `fetch_attribute` calls that an error rather than
            // inventing a value, and so does this.
            return Err(Unsupported::State(format!(
                "attribute {location} reads from disabled vertex buffer {}",
                attrib.buffer_id
            )));
        }
        if attrib.is_bgra {
            return Err(Unsupported::BgraAttribute { location });
        }
        let attribute = VertexAttribute {
            format: vertex_format(attrib.size, attrib.ty)?,
            offset: attrib.offset,
            location,
        };
        match buffers.iter_mut().find(|b| b.index == attrib.buffer_id) {
            Some(buffer) => buffer.attributes.push(attribute),
            None => buffers.push(VertexBuffer {
                index: attrib.buffer_id,
                stride: array.stride,
                step: step_mode(array)?,
                attributes: vec![attribute],
            }),
        }
    }
    Ok((buffers, fixed))
}

/// A divisor of zero steps per vertex and one steps per instance, which are
/// WebGPU's two. Anything else — every `n` instances — it cannot say.
fn step_mode(array: VertexArray) -> Result<StepMode, Unsupported> {
    match array.divisor {
        0 => Ok(StepMode::Vertex),
        1 => Ok(StepMode::Instance),
        divisor => Err(Unsupported::InstanceDivisor { divisor }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two numberings, paired: the D3D code a deko3d or nvn driver
    /// writes, and the GL code Mesa writes, for the same factor.
    const FACTOR_PAIRS: &[(u32, u32, BlendFactor)] = &[
        (0x01, 0x4000, BlendFactor::Zero),
        (0x02, 0x4001, BlendFactor::One),
        (0x03, 0x4300, BlendFactor::Src),
        (0x04, 0x4301, BlendFactor::OneMinusSrc),
        (0x05, 0x4302, BlendFactor::SrcAlpha),
        (0x06, 0x4303, BlendFactor::OneMinusSrcAlpha),
        (0x07, 0x4304, BlendFactor::DstAlpha),
        (0x08, 0x4305, BlendFactor::OneMinusDstAlpha),
        (0x09, 0x4306, BlendFactor::Dst),
        (0x0a, 0x4307, BlendFactor::OneMinusDst),
        (0x0b, 0x4308, BlendFactor::SrcAlphaSaturated),
        (0x61, 0xc001, BlendFactor::Constant),
        (0x62, 0xc002, BlendFactor::OneMinusConstant),
    ];

    #[test]
    fn both_numberings_name_the_same_blend_factor() {
        // Which one a register holds is down to whose driver wrote it, and a
        // backend that only knew one would blend a third of this emulator's
        // draws wrongly — the Home Menu's whole UI blends
        // SrcAlpha/OneMinusSrcAlpha, in the numbering that is not GL's.
        for &(d3d, gl, expected) in FACTOR_PAIRS {
            assert_eq!(blend_factor(d3d), Ok(expected), "D3D {d3d:#x}");
            assert_eq!(blend_factor(gl), Ok(expected), "GL {gl:#x}");
        }
    }

    #[test]
    fn a_factor_neither_numbering_knows_is_reported_rather_than_defaulted() {
        // `raster::blend_factor` answers `One` here, which is what a
        // rasterizer that must produce a pixel has to do. A description can
        // say it does not know, and saying so is what sends the draw to the
        // rasterizer instead of drawing it differently.
        assert_eq!(blend_factor(0x1234), Err(Unsupported::BlendFactor { code: 0x1234 }));
    }

    #[test]
    fn a_constant_alpha_factor_has_no_webgpu_spelling() {
        // ConstantAlpha broadcasts the blend constant's alpha to all four
        // channels; WebGPU's `Constant` is per-channel and has no alpha-only
        // form.
        for code in [0x63, 0x64, 0xc003, 0xc004] {
            assert_eq!(blend_factor(code), Err(Unsupported::BlendFactor { code }));
        }
    }

    #[test]
    fn both_numberings_name_the_same_blend_equation() {
        for (simple, gl, expected) in [
            (0x1, 0x8006, BlendOperation::Add),
            (0x2, 0x800a, BlendOperation::Subtract),
            (0x3, 0x800b, BlendOperation::ReverseSubtract),
            (0x4, 0x8007, BlendOperation::Min),
            (0x5, 0x8008, BlendOperation::Max),
        ] {
            assert_eq!(blend_equation(simple), Ok(expected));
            assert_eq!(blend_equation(gl), Ok(expected));
        }
        assert_eq!(blend_equation(0x77), Err(Unsupported::BlendEquation { code: 0x77 }));
    }

    #[test]
    fn blending_that_is_off_is_no_blend_state_at_all() {
        let mut target = BlendTarget {
            enabled: false,
            equation_rgb: 0,
            func_rgb_src: 0,
            func_rgb_dst: 0,
            equation_alpha: 0,
            func_alpha_src: 0,
            func_alpha_dst: 0,
        };
        // The codes are nonsense, and unreachable while it is disabled.
        assert_eq!(blend(target), Ok(None));
        target.enabled = true;
        assert!(blend(target).is_err(), "an enabled blend reads its codes");
    }

    #[test]
    fn topologies_webgpu_lacks_are_reported() {
        // A fan, a quad, a loop and a polygon all become triangles by
        // rewriting the index buffer, which is work for a backend.
        for primitive in [
            Primitive::TriangleFan,
            Primitive::Quads,
            Primitive::QuadStrip,
            Primitive::Polygon,
            Primitive::LineLoop,
        ] {
            assert_eq!(topology(primitive), Err(Unsupported::Topology(primitive)));
        }
        assert_eq!(topology(Primitive::Triangles), Ok(Topology::TriangleList));
        assert_eq!(topology(Primitive::TriangleStrip), Ok(Topology::TriangleStrip));
    }

    #[test]
    fn a_depth_test_that_is_off_compares_always() {
        // What a disabled test does, said in the only vocabulary a pipeline
        // has for it. The `func` here is nonsense and never read.
        let layout =
            DepthLayout { bytes: 4, depth_bits: 24, depth_shift: 8, stencil_shift: Some(0) };
        let off = DepthState { test_enabled: false, write_enabled: true, func: 0xdead };
        assert_eq!(
            depth(layout, off),
            Ok(Depth {
                format: Format::Depth24PlusStencil8,
                write_enabled: true,
                compare: Compare::Always
            })
        );
    }

    #[test]
    fn both_numberings_of_a_depth_comparison_decode_the_same() {
        // Maxwell's register takes either, and titles use both: Mesa's GL
        // driver writes 0x200..=0x207, a D3D-shaped path writes 1..=8. Eden's
        // `ComparisonOp` lists the two side by side. This used to reject
        // 1..=8 on the grounds that the rasterizer only decoded the GL half —
        // which was true, and the reason Just Dance 2019 fell back to software
        // on every draw and then had its depth test ignored there.
        let layout = DepthLayout { bytes: 4, depth_bits: 24, depth_shift: 8, stencil_shift: None };
        let of = |func| {
            let on = DepthState { test_enabled: true, write_enabled: false, func };
            depth(layout, on).unwrap().compare
        };
        for (d3d, gl) in (1..=8u32).zip(0x0200..=0x0207u32) {
            assert_eq!(of(d3d), of(gl), "D3D {d3d} against GL {gl:#x}");
        }
        assert_eq!(of(4), Compare::LessEqual, "the one Just Dance 2019 sends");
        assert_eq!(of(0x0203), Compare::LessEqual);

        // A code in neither numbering is still reported rather than guessed.
        let on = DepthState { test_enabled: true, write_enabled: false, func: 0x40 };
        assert_eq!(depth(layout, on), Err(Unsupported::DepthCompare { code: 0x40 }));
    }

    /// How wide a pixel of each colour format is, and whether it is sRGB.
    fn shape(format: Format) -> (u32, bool) {
        match format {
            Format::R8Unorm => (1, false),
            Format::Rg8Unorm => (2, false),
            Format::Rgba8Unorm | Format::Bgra8Unorm | Format::Rgb10a2Unorm | Format::R32Float => {
                (4, false)
            }
            Format::Rgba8UnormSrgb | Format::Bgra8UnormSrgb => (4, true),
            Format::Rgba16Float => (8, false),
            Format::Rgba32Float => (16, false),
            other => panic!("{other:?} is not a colour target format"),
        }
    }

    #[test]
    fn a_colour_format_is_what_the_surface_module_makes_of_the_same_code() {
        // These names are a second reading of the raw codes `gpu::surface`
        // already interprets, and two readings drift. This is the coupling:
        // every code named here has to have the width and the transfer
        // function `surface` gives it, or one of the two is wrong.
        for raw in 0u32..=0xff {
            let Ok(format) = ColorFormat::from_raw(raw) else { continue };
            let Ok(named) = color_format(format) else { continue };
            let (bytes, srgb) = shape(named);
            assert_eq!(bytes, format.bytes_per_pixel, "{raw:#x} is {named:?}");
            assert_eq!(srgb, format.is_srgb(), "{raw:#x} is {named:?}");
        }
    }

    #[test]
    fn a_mirrored_viewport_is_a_rectangle_and_a_flag() {
        // A driver writes a negative scale_y to reconcile GL's bottom-left
        // window origin with a target whose row 0 is at the top. The
        // rectangle is the same either way; which way up it is is not.
        let flipped = viewport(ViewportTransform {
            scale: [640.0, -360.0, 0.5],
            translate: [640.0, 360.0, 0.5],
        });
        assert_eq!((flipped.x, flipped.y), (0.0, 0.0));
        assert_eq!((flipped.width, flipped.height), (1280.0, 720.0));
        assert!(flipped.flip_y);
        let upright = viewport(ViewportTransform {
            scale: [128.0, 128.0, 0.5],
            translate: [128.0, 128.0, 0.5],
        });
        assert_eq!((upright.width, upright.height), (256.0, 256.0));
        assert!(!upright.flip_y);
    }

    #[test]
    fn a_gl_depth_range_is_told_apart_from_a_vulkan_one() {
        // Both map onto a 0..1 window depth, and differ only in what the
        // shader's z is expected to span — which decides whether a vertex
        // entry point has to remap it. Every transform this has seen is the
        // first shape.
        let gl = viewport(ViewportTransform { scale: [1.0, 1.0, 0.5], translate: [0.0, 0.0, 0.5] });
        assert_eq!((gl.min_depth, gl.max_depth), (0.0, 1.0));
        assert!(gl.depth_minus_one_to_one());
        let vulkan =
            viewport(ViewportTransform { scale: [1.0, 1.0, 1.0], translate: [0.0, 0.0, 0.0] });
        assert!(!vulkan.depth_minus_one_to_one());
    }

    #[test]
    fn an_instance_step_webgpu_cannot_take_is_reported() {
        let array =
            |divisor| VertexArray { enabled: true, stride: 16, start: 0, limit: 0, divisor };
        assert_eq!(step_mode(array(0)), Ok(StepMode::Vertex));
        assert_eq!(step_mode(array(1)), Ok(StepMode::Instance));
        // Every two instances: WebGPU steps per instance or per vertex and
        // has nothing in between.
        assert_eq!(step_mode(array(2)), Err(Unsupported::InstanceDivisor { divisor: 2 }));
    }

    #[test]
    fn vertex_formats_are_the_ones_the_rasterizer_can_fetch() {
        assert_eq!(vertex_format(0x01, ATTRIB_TYPE_FLOAT), Ok(VertexFormat::Float32x4));
        assert_eq!(vertex_format(0x0a, ATTRIB_TYPE_UNORM), Ok(VertexFormat::Unorm8x4));
        assert_eq!(vertex_format(0x0a, ATTRIB_TYPE_SNORM), Ok(VertexFormat::Snorm8x4));
        assert_eq!(vertex_format(0x0a, ATTRIB_TYPE_SINT), Ok(VertexFormat::Sint8x4));
        assert_eq!(vertex_format(0x0a, ATTRIB_TYPE_UINT), Ok(VertexFormat::Uint8x4));
        // An integer attribute reaches the shader as its bits, so it is the
        // one kind that cannot be declared `vec4<f32>`.
        assert_eq!(VertexFormat::Sint8x4.base(), AttributeBase::Sint);
        assert_eq!(VertexFormat::Uint8x4.base(), AttributeBase::Uint);
        assert_eq!(VertexFormat::Snorm8x4.base(), AttributeBase::Float);
        assert_eq!(VertexFormat::Unorm8x4.base(), AttributeBase::Float);
        // A shape `fetch_attribute` cannot decode. Claiming it would draw
        // something the reference could not be compared against.
        assert_eq!(
            vertex_format(0x03, ATTRIB_TYPE_FLOAT),
            Err(Unsupported::VertexFormat { size: 0x03, ty: ATTRIB_TYPE_FLOAT })
        );
    }

    #[test]
    fn a_draw_with_no_targets_at_all_says_so_rather_than_panicking() {
        // A register file nothing has written. The rasterizer raises the
        // same thing rather than picking an extent out of the air.
        let engine = Engine3D::new();
        assert_eq!(
            Pipeline::of(&engine),
            Err(Unsupported::State("a draw with neither a colour nor a depth target".into()))
        );
    }
}
