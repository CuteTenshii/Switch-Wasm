//! The fixed-function state a draw runs under as one hashable value, resolved
//! into WebGPU's vocabulary so a backend can cache pipelines by it. State WebGPU
//! can't express, or codes not recognised, are [`Unsupported`] and the draw falls
//! back to the software rasterizer.

use crate::gpu::engine::threed::{
    BlendTarget, DepthLayout, DepthState, Engine3D, RenderTarget, VertexArray, ViewportTransform,
};
// Re-exported for backends reading `Pipeline::scissor`.
pub use crate::gpu::engine::threed::ScissorRect;
use crate::gpu::raster::Primitive;
use crate::gpu::surface::{ColorFormat, SampleGrid};
use std::fmt;
use switch_shader::attribute::{AttributeBase, Packed1010102};

/// A piece of state that has no WebGPU spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {
    /// A topology neither renderer assembles into triangles (lines, points);
    /// fans, quads and polygons go through [`Pipeline::expand`].
    Topology(Primitive),
    /// A blend factor with no equivalent, or an unknown code.
    BlendFactor {
        code: u32,
    },
    BlendEquation {
        code: u32,
    },
    /// A depth comparison the software rasterizer doesn't implement either.
    DepthCompare {
        code: u32,
    },
    /// A colour or depth surface format with no equivalent.
    Format {
        raw: u32,
    },
    /// A vertex attribute's `DkVtxAttribSize`/`DkVtxAttribType`.
    VertexFormat {
        size: u32,
        ty: u32,
    },
    /// A vertex array stepped every `divisor` instances.
    InstanceDivisor {
        divisor: u32,
    },
    /// The engine could not resolve a piece of state.
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
            Unsupported::InstanceDivisor { divisor } => {
                write!(f, "a vertex array stepped once every {divisor} instances")
            }
            Unsupported::State(why) => write!(f, "{why}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Topology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

/// Front-face winding in the backend's NDC: the guest's, judged before any y flip.
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

impl Format {
    /// Whether texels are integers, read and written as raw words.
    pub fn is_uint(self) -> bool {
        self.uint_max().is_some()
    }

    /// The largest value an integer channel holds; written words saturate to it.
    pub fn uint_max(self) -> Option<u32> {
        match self {
            Format::R8Uint => Some(0xFF),
            Format::Rgba16Uint => Some(0xFFFF),
            Format::R32Uint | Format::Rgba32Uint => Some(u32::MAX),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Rgb10a2Unorm,
    // SNORM: sampled but not renderable; the backend rejects them as targets.
    R8Snorm,
    Rg8Snorm,
    Rgba8Snorm,
    R16Snorm,
    Rg16Snorm,
    Rgba16Snorm,
    R16Unorm,
    Rg16Unorm,
    Rgba16Unorm,
    R16Float,
    Rg16Float,
    Rgba16Float,
    R32Float,
    /// A texture's raw 32-bit words, for a conversion pass to turn into something sampleable.
    PackedDepth,
    Rg32Float,
    Rgba32Float,
    /// `R8_UINT`: as `Rgba32Uint`, one channel saturated to 8 bits.
    R8Uint,
    /// `R32_UINT`: as `Rgba32Uint`, one channel.
    R32Uint,
    /// `RGBA16_UINT`: as `Rgba32Uint`, each word saturated to 16 bits.
    Rgba16Uint,
    /// `RGBA32_UINT`: raw words, written and read as integers and never blended.
    Rgba32Uint,
    /// `B10G11R11_FLOAT`; needs `rg11b10ufloat-renderable`, and `Rgba16Float`
    /// can't stand in since the guest surface is four bytes a pixel.
    Rg11b10Ufloat,
    Depth16Unorm,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
    Depth32FloatStencil8,
    // Compressed, sampled-only.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Float16,
    Unorm16,
    Snorm16,
    Sint16,
    Uint16,
    Float16x2,
    Float16x4,
    Unorm16x2,
    Unorm16x4,
    Snorm16x2,
    Snorm16x4,
    Sint16x2,
    Sint16x4,
    Uint16x2,
    Uint16x4,
    Unorm8,
    Snorm8,
    Sint8,
    Uint8,
    Unorm8x2,
    Snorm8x2,
    Sint8x2,
    Uint8x2,
    Unorm8x4,
    Snorm8x4,
    Sint8x4,
    Uint8x4,
    Packed1010102(Packed1010102),
}

impl VertexFormat {
    pub fn size(self) -> u32 {
        match self {
            VertexFormat::Unorm8
            | VertexFormat::Snorm8
            | VertexFormat::Sint8
            | VertexFormat::Uint8 => 1,
            VertexFormat::Float16
            | VertexFormat::Unorm16
            | VertexFormat::Snorm16
            | VertexFormat::Sint16
            | VertexFormat::Uint16
            | VertexFormat::Unorm8x2
            | VertexFormat::Snorm8x2
            | VertexFormat::Sint8x2
            | VertexFormat::Uint8x2 => 2,
            VertexFormat::Float32
            | VertexFormat::Sint32
            | VertexFormat::Uint32
            | VertexFormat::Float16x2
            | VertexFormat::Unorm16x2
            | VertexFormat::Snorm16x2
            | VertexFormat::Sint16x2
            | VertexFormat::Uint16x2
            | VertexFormat::Unorm8x4
            | VertexFormat::Snorm8x4
            | VertexFormat::Sint8x4
            | VertexFormat::Uint8x4
            | VertexFormat::Packed1010102(_) => 4,
            VertexFormat::Float32x2
            | VertexFormat::Sint32x2
            | VertexFormat::Uint32x2
            | VertexFormat::Float16x4
            | VertexFormat::Unorm16x4
            | VertexFormat::Snorm16x4
            | VertexFormat::Sint16x4
            | VertexFormat::Uint16x4 => 8,
            VertexFormat::Float32x3 | VertexFormat::Sint32x3 | VertexFormat::Uint32x3 => 12,
            VertexFormat::Float32x4 | VertexFormat::Sint32x4 | VertexFormat::Uint32x4 => 16,
        }
    }

    /// Normalized formats are floats to the shader; integer ones keep their bits.
    pub fn base(self) -> AttributeBase {
        match self {
            VertexFormat::Sint32
            | VertexFormat::Sint32x2
            | VertexFormat::Sint32x3
            | VertexFormat::Sint32x4
            | VertexFormat::Sint16
            | VertexFormat::Sint16x2
            | VertexFormat::Sint16x4
            | VertexFormat::Sint8
            | VertexFormat::Sint8x2
            | VertexFormat::Sint8x4
            | VertexFormat::Packed1010102(Packed1010102::Sint) => AttributeBase::Sint,
            VertexFormat::Uint32
            | VertexFormat::Uint32x2
            | VertexFormat::Uint32x3
            | VertexFormat::Uint32x4
            | VertexFormat::Uint16
            | VertexFormat::Uint16x2
            | VertexFormat::Uint16x4
            | VertexFormat::Uint8
            | VertexFormat::Uint8x2
            | VertexFormat::Uint8x4
            | VertexFormat::Packed1010102(Packed1010102::Uint) => AttributeBase::Uint,
            _ => AttributeBase::Float,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepMode {
    Vertex,
    Instance,
}

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

impl BlendComponent {
    /// Whether the result stays within `[0, 1]` for a source and destination already there,
    /// `constant` being the blend constant's channels this component reads.
    fn stays_in_unit_range(&self, constant: &[f32]) -> bool {
        use BlendFactor::*;
        let constant_in_range = constant.iter().all(|c| (0.0..=1.0).contains(c));
        let in_range = |factor: BlendFactor| match factor {
            Constant | OneMinusConstant => constant_in_range,
            _ => true,
        };
        if !in_range(self.src_factor) || !in_range(self.dst_factor) {
            return false;
        }
        // Factors summing to one weigh the two terms into a mix of them.
        let complements = matches!(
            (self.src_factor, self.dst_factor),
            (Src, OneMinusSrc)
                | (OneMinusSrc, Src)
                | (SrcAlpha, OneMinusSrcAlpha)
                | (OneMinusSrcAlpha, SrcAlpha)
                | (Dst, OneMinusDst)
                | (OneMinusDst, Dst)
                | (DstAlpha, OneMinusDstAlpha)
                | (OneMinusDstAlpha, DstAlpha)
                | (Constant, OneMinusConstant)
                | (OneMinusConstant, Constant)
        );
        match self.operation {
            BlendOperation::Min | BlendOperation::Max => true,
            BlendOperation::Add => {
                complements || self.src_factor == Zero || self.dst_factor == Zero
            }
            BlendOperation::Subtract => self.dst_factor == Zero,
            BlendOperation::ReverseSubtract => self.src_factor == Zero,
        }
    }
}

impl Blend {
    /// Whether every channel stays within `[0, 1]` for a source and destination already there:
    /// blending into a float surface standing in for a unorm one then needs no clamp.
    pub fn stays_in_unit_range(&self, constant: [f32; 4]) -> bool {
        self.color.stays_in_unit_range(&constant[..3])
            && self.alpha.stays_in_unit_range(&constant[3..])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorTarget {
    pub format: Format,
    /// `None` when blending is off.
    pub blend: Option<Blend>,
    /// See `Engine3D::color_mask`.
    pub write_mask: [bool; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Depth {
    pub format: Format,
    pub write_enabled: bool,
    /// `Always` when the test is off.
    pub compare: Compare,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u32,
    /// The shader `@location`: Maxwell's attribute slot.
    pub location: u32,
    /// Swap the first and third components; WebGPU has no BGRA vertex format.
    pub is_bgra: bool,
    /// An integer `format` converted to the float of its value; WebGPU has no scaled formats.
    pub scaled: bool,
}

impl VertexAttribute {
    pub fn base(&self) -> AttributeBase {
        match (self.format.base(), self.scaled) {
            (AttributeBase::Sint, true) => AttributeBase::Sscaled,
            (AttributeBase::Uint, true) => AttributeBase::Uscaled,
            (base, _) => base,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexBuffer {
    pub index: u32,
    pub stride: u32,
    pub step: StepMode,
    pub attributes: Vec<VertexAttribute>,
}

/// The viewport as a rectangle rather than scale-and-translate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
    /// The unresolved z scale; see [`Viewport::depth_minus_one_to_one`].
    pub depth_scale: f32,
    pub depth_translate: f32,
    /// Whether the guest's transform mirrors y (negative `scale_y`). A backend
    /// negates `position.y` where it doesn't, since WebGPU has no negative viewport height.
    pub flip_y: bool,
    /// The surface size a vertex stage writing window coordinates is measured in; the
    /// backend turns them into clip space. See [`Engine3D::window_coordinates`].
    pub window_coordinates: Option<(u32, u32)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pipeline {
    pub topology: Topology,
    /// The requested topology when it becomes a triangle list by index rewriting.
    pub expand: Option<Primitive>,
    pub front_face: FrontFace,
    pub cull: Cull,
    /// Colour target 0, or `None` for a depth-only pass.
    pub target: Option<ColorTarget>,
    /// Colour outputs 1 onward, as `RenderTargetControl` counts them; `None` where unbound.
    pub extra_targets: Vec<Option<ColorTarget>>,
    pub depth: Option<Depth>,
    pub vertex_buffers: Vec<VertexBuffer>,
    /// Attribute slots with no buffer bound; they read `(0, 0, 0, 1)`.
    pub fixed_attributes: Vec<u32>,
    /// Sample layout and coverage positions; [`Pipeline::samples`] is its count.
    pub grid: SampleGrid,
    /// Coverage tested per pixel over a multisampled surface (`AntiAliasEnable` off, or every
    /// sample programmed to the pixel centre), which device multisampling can't do.
    pub per_pixel_coverage: bool,
    pub samples: u32,
    pub sample_mask: u32,
    pub alpha_to_coverage: bool,
    pub viewport: Viewport,
    pub scissor: ScissorRect,
    pub blend_constant: [f32; 4],
}

const MAX_VERTEX_ATTRIBS: u32 = 16;

impl Pipeline {
    pub fn of(engine: &Engine3D) -> Result<Pipeline, Unsupported> {
        let state = |what: &str, e: crate::Error| Unsupported::State(format!("{what}: {e:?}"));

        let primitive =
            Primitive::from_raw(engine.last_draw.primitive).map_err(|e| state("primitive", e))?;
        let cull = engine.cull_state();
        let viewport = match engine.window_coordinates() {
            Some((width, height)) => Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
                depth_scale: 1.0,
                depth_translate: 0.0,
                flip_y: false,
                window_coordinates: Some((width, height)),
            },
            None => viewport(engine.viewport_transform()),
        };

        let slot = engine.render_target_slot(0);
        let rt = engine
            .render_target(slot)
            .map_err(|e| state("colour target", e))?;
        let dt = engine
            .depth_target()
            .map_err(|e| state("depth target", e))?;
        let given_constant = engine.blend_constant();
        // Every output blends against the one constant, which a factor may rewrite.
        let mut blend_constant = None;
        let mut color_target = |output: u32,
                                slot: u32,
                                rt: Option<RenderTarget>|
         -> Result<Option<ColorTarget>, Unsupported> {
            let Some(rt) = rt else {
                return Ok(None);
            };
            let format = color_format(rt.format)?;
            // An integer target is never blended.
            let blend = match format.is_uint() {
                true => None,
                false => {
                    let (state, constant) = blend(engine.blend_target(output), given_constant)?;
                    if state.is_some() {
                        if blend_constant.is_some_and(|earlier| earlier != constant) {
                            return Err(Unsupported::State(
                                "outputs blending against differently rewritten constants".into(),
                            ));
                        }
                        blend_constant = Some(constant);
                    }
                    state
                }
            };
            Ok(Some(ColorTarget {
                format,
                blend,
                write_mask: engine.color_mask(slot),
            }))
        };
        let target = color_target(0, slot, rt)?;
        let mut extra_targets = Vec::new();
        for output in 1..engine.color_outputs() {
            let slot = engine.render_target_slot(output);
            let rt = engine
                .render_target(slot)
                .map_err(|e| state("colour target", e))?;
            extra_targets.push(color_target(output, slot, rt)?);
        }
        let blend_constant = blend_constant.unwrap_or(given_constant);
        let depth = match dt {
            Some(dt) => Some(self::depth(dt.format, engine.depth_state())?),
            None => None,
        };

        // Scissor extent is in pixels, which differ from texels when multisampled.
        let grid = engine.sample_grid().map_err(|e| state("sample grid", e))?;
        let Some(extent) = crate::gpu::engine::threed::draw_extent(
            rt.map(|rt| (rt.width, rt.height)),
            dt.map(|dt| (dt.width, dt.height)),
            engine.depth_state(),
        ) else {
            return Err(Unsupported::State(
                "a draw with neither a colour nor a depth target".into(),
            ));
        };
        let (width, height) = grid.pixels(extent.0, extent.1);
        let scissor = engine.apply_scissor(ScissorRect {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        });

        let (vertex_buffers, fixed_attributes) = vertex_buffers(engine)?;

        let (topology, expand) = topology(primitive)?;
        Ok(Pipeline {
            topology,
            expand,
            // Facing is judged in window space like `raster::culls`, so it carries over.
            front_face: if cull.front_ccw {
                FrontFace::Ccw
            } else {
                FrontFace::Cw
            },
            cull: match (cull.enabled, cull.cull_front, cull.cull_back) {
                (false, _, _) | (_, false, false) => Cull::None,
                // Culling applies to triangles only.
                (_, true, true) if !primitive.is_rasterized() => Cull::None,
                (_, true, true) => {
                    // A draw that culls both faces draws nothing, which WebGPU can't say;
                    // see [`culls_every_triangle`].
                    return Err(Unsupported::State("a draw that culls both faces".into()));
                }
                (_, true, false) => Cull::Front,
                (_, false, true) => Cull::Back,
            },
            target,
            extra_targets,
            depth,
            vertex_buffers,
            fixed_attributes,
            grid,
            per_pixel_coverage: !grid.is_single()
                && (!engine.multisample_enabled() || grid.all_at_pixel_centre()),
            samples: grid.count(),
            sample_mask: engine.sample_mask(),
            alpha_to_coverage: engine.alpha_to_coverage(),
            viewport,
            scissor,
            blend_constant,
        })
    }
}

/// `window = ndc * scale + translate` as a rectangle; negative scale mirrors the axis.
fn viewport(transform: ViewportTransform) -> Viewport {
    let [sx, sy, sz] = transform.scale;
    let [tx, ty, tz] = transform.translate;
    Viewport {
        x: tx - sx.abs(),
        y: ty - sy.abs(),
        width: 2.0 * sx.abs(),
        height: 2.0 * sy.abs(),
        min_depth: (tz - sz).min(tz + sz),
        max_depth: (tz - sz).max(tz + sz),
        depth_scale: sz,
        depth_translate: tz,
        flip_y: sy < 0.0,
        window_coordinates: None,
    }
}

impl Viewport {
    /// Whether clip z runs `-w..w` (GL) rather than `0..w`, inferred from scale 0.5
    /// and translate 0.5. WebGPU clips like Vulkan, so GL-range shaders need z remapped.
    pub fn depth_minus_one_to_one(&self) -> bool {
        self.depth_translate - self.depth_scale >= 0.0
    }
}

/// Whether the draw culls both faces of a triangle primitive, so it changes nothing.
pub fn culls_every_triangle(engine: &Engine3D) -> bool {
    let cull = engine.cull_state();
    cull.enabled
        && cull.cull_front
        && cull.cull_back
        && Primitive::from_raw(engine.last_draw.primitive).is_ok_and(Primitive::is_rasterized)
}

/// Whether the draw's fragment stage is disabled and it leaves depth alone, so it changes
/// nothing: with no fragment program the hardware writes no colour, and stencil is not emulated.
pub fn writes_no_surface(engine: &Engine3D) -> bool {
    engine
        .program(crate::gpu::engine::threed::ShaderStage::Fragment)
        .is_none()
        && !(engine.depth_state().reaches_surface()
            && engine.depth_target().is_ok_and(|d| d.is_some()))
}

/// WebGPU topology, and the primitive whose indices `raster::assemble` must
/// rewrite first. Line loops, points and lines are drawn by neither renderer.
fn topology(primitive: Primitive) -> Result<(Topology, Option<Primitive>), Unsupported> {
    match primitive {
        Primitive::Points => Ok((Topology::PointList, None)),
        Primitive::Lines => Ok((Topology::LineList, None)),
        Primitive::LineStrip => Ok((Topology::LineStrip, None)),
        Primitive::Triangles => Ok((Topology::TriangleList, None)),
        Primitive::TriangleStrip => Ok((Topology::TriangleStrip, None)),
        Primitive::TriangleFan | Primitive::Quads | Primitive::QuadStrip | Primitive::Polygon => {
            Ok((Topology::TriangleList, Some(primitive)))
        }
        other => Err(Unsupported::Topology(other)),
    }
}

/// GL (Mesa) or D3D (deko3d, nvn) numbering. Unknown codes are errors here,
/// where the rasterizer answers `One`.
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
        // ConstantAlpha has no WebGPU form: `Constant` is per-channel.
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

/// The blend state and the constant to draw it with. A blend reading only the constant's alpha
/// reads `Constant` with the alpha in every channel.
fn blend(
    target: BlendTarget,
    constant: [f32; 4],
) -> Result<(Option<Blend>, [f32; 4]), Unsupported> {
    if !target.enabled {
        return Ok((None, constant));
    }
    let codes = [
        target.func_rgb_src,
        target.func_rgb_dst,
        target.func_alpha_src,
        target.func_alpha_dst,
    ];
    let alpha_only = codes
        .into_iter()
        .find(|code| matches!(code, 0x63 | 0x64 | 0xc003 | 0xc004));
    let per_channel = codes
        .into_iter()
        .any(|code| matches!(code, 0x61 | 0x62 | 0xc001 | 0xc002));
    if let Some(code) = alpha_only {
        if per_channel && constant[..3] != [constant[3]; 3] {
            return Err(Unsupported::BlendFactor { code });
        }
    }
    let factor = |code| {
        blend_factor(match code {
            0x63 | 0xc003 => 0x61,
            0x64 | 0xc004 => 0x62,
            code => code,
        })
    };
    // Min and max ignore the factors, and WebGPU refuses any but one beside them.
    let component = |src: u32, dst: u32, equation: u32| -> Result<BlendComponent, Unsupported> {
        let operation = blend_equation(equation)?;
        Ok(match operation {
            BlendOperation::Min | BlendOperation::Max => BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation,
            },
            _ => BlendComponent {
                src_factor: factor(src)?,
                dst_factor: factor(dst)?,
                operation,
            },
        })
    };
    let blend = Blend {
        color: component(
            target.func_rgb_src,
            target.func_rgb_dst,
            target.equation_rgb,
        )?,
        alpha: component(
            target.func_alpha_src,
            target.func_alpha_dst,
            target.equation_alpha,
        )?,
    };
    let constant = match alpha_only {
        Some(_) => [constant[3]; 4],
        None => constant,
    };
    Ok((Some(blend), constant))
}

/// The GL comparison enums `raster::depth_test_passes` implements.
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

pub fn depth_format(layout: DepthLayout) -> Result<Format, Unsupported> {
    Ok(
        match (
            layout.bytes,
            layout.depth_bits,
            layout.stencil_shift.is_some(),
        ) {
            (2, 16, false) => Format::Depth16Unorm,
            (4, 24, false) => Format::Depth24Plus,
            (4, 24, true) => Format::Depth24PlusStencil8,
            (4, 0, false) => Format::Depth32Float,
            (8, 0, true) => Format::Depth32FloatStencil8,
            _ => return Err(Unsupported::Format { raw: layout.bytes }),
        },
    )
}

fn depth(layout: DepthLayout, state: DepthState) -> Result<Depth, Unsupported> {
    Ok(Depth {
        format: depth_format(layout)?,
        write_enabled: state.write_enabled,
        compare: if state.test_enabled {
            depth_compare(state.func)?
        } else {
            Compare::Always
        },
    })
}

/// The WebGPU format for a colour code, following [`crate::gpu::surface`]'s
/// reading. Also used for textures. Integer formats and packings WebGPU lacks are unnamed.
pub(crate) fn color_format(format: ColorFormat) -> Result<Format, Unsupported> {
    Ok(match format.raw {
        // Eight bits a channel. `0xD8`/`0xD9` are the SINT/UINT codes.
        0xD5 | 0xF9 => Format::Rgba8Unorm,
        0xD6 | 0xFA => Format::Rgba8UnormSrgb,
        0xD7 => Format::Rgba8Snorm,
        0xCF | 0xE6 | 0xFD | 0xFE => Format::Bgra8Unorm,
        0xD0 | 0xE7 => Format::Bgra8UnormSrgb,
        0xEA => Format::Rg8Unorm,
        0xEB => Format::Rg8Snorm,
        0xF3 => Format::R8Unorm,
        0xF4 => Format::R8Snorm,
        0xF6 => Format::R8Uint,
        // Sixteen. Normalized needs `texture-format-16bit-norm` (native only).
        0xC6 => Format::Rgba16Unorm,
        0xC7 => Format::Rgba16Snorm,
        0xC9 => Format::Rgba16Uint,
        0xCA | 0xCE => Format::Rgba16Float,
        0xDA => Format::Rg16Unorm,
        0xDB => Format::Rg16Snorm,
        0xDE => Format::Rg16Float,
        0xEE => Format::R16Unorm,
        0xEF => Format::R16Snorm,
        0xF2 => Format::R16Float,
        // Thirty-two, and the packed formats WebGPU has.
        0xC0 | 0xC3 => Format::Rgba32Float,
        0xC2 => Format::Rgba32Uint,
        0xE4 => Format::R32Uint,
        0xCB => Format::Rg32Float,
        0xE5 => Format::R32Float,
        0xD1 => Format::Rgb10a2Unorm,
        0xE0 => Format::Rg11b10Ufloat,
        raw => return Err(Unsupported::Format { raw }),
    })
}

/// `DkVtxAttribType` (deko3d.h).
const ATTRIB_TYPE_SNORM: u32 = 1;
const ATTRIB_TYPE_UNORM: u32 = 2;
const ATTRIB_TYPE_SINT: u32 = 3;
const ATTRIB_TYPE_UINT: u32 = 4;
const ATTRIB_TYPE_USCALED: u32 = 5;
const ATTRIB_TYPE_SSCALED: u32 = 6;
const ATTRIB_TYPE_FLOAT: u32 = 7;

/// Only the formats `raster::fetch_attribute` decodes.
fn vertex_format(size: u32, ty: u32) -> Result<VertexFormat, Unsupported> {
    Ok(match (size, ty) {
        (0x01, ATTRIB_TYPE_FLOAT) => VertexFormat::Float32x4,
        (0x02, ATTRIB_TYPE_FLOAT) => VertexFormat::Float32x3,
        (0x04, ATTRIB_TYPE_FLOAT) => VertexFormat::Float32x2,
        (0x12, ATTRIB_TYPE_FLOAT) => VertexFormat::Float32,
        (0x01, ATTRIB_TYPE_SINT) => VertexFormat::Sint32x4,
        (0x02, ATTRIB_TYPE_SINT) => VertexFormat::Sint32x3,
        (0x04, ATTRIB_TYPE_SINT) => VertexFormat::Sint32x2,
        (0x12, ATTRIB_TYPE_SINT) => VertexFormat::Sint32,
        (0x01, ATTRIB_TYPE_UINT) => VertexFormat::Uint32x4,
        (0x02, ATTRIB_TYPE_UINT) => VertexFormat::Uint32x3,
        (0x04, ATTRIB_TYPE_UINT) => VertexFormat::Uint32x2,
        (0x12, ATTRIB_TYPE_UINT) => VertexFormat::Uint32,
        // 16- and 8-bit shapes WebGPU has; three-component ones stay on the rasterizer.
        (0x1b, ATTRIB_TYPE_FLOAT) => VertexFormat::Float16,
        (0x1b, ATTRIB_TYPE_UNORM) => VertexFormat::Unorm16,
        (0x1b, ATTRIB_TYPE_SNORM) => VertexFormat::Snorm16,
        (0x1b, ATTRIB_TYPE_SINT) => VertexFormat::Sint16,
        (0x1b, ATTRIB_TYPE_UINT) => VertexFormat::Uint16,
        (0x03, ATTRIB_TYPE_FLOAT) => VertexFormat::Float16x4,
        (0x03, ATTRIB_TYPE_UNORM) => VertexFormat::Unorm16x4,
        (0x03, ATTRIB_TYPE_SNORM) => VertexFormat::Snorm16x4,
        (0x03, ATTRIB_TYPE_SINT) => VertexFormat::Sint16x4,
        (0x03, ATTRIB_TYPE_UINT) => VertexFormat::Uint16x4,
        (0x0f, ATTRIB_TYPE_FLOAT) => VertexFormat::Float16x2,
        (0x0f, ATTRIB_TYPE_UNORM) => VertexFormat::Unorm16x2,
        (0x0f, ATTRIB_TYPE_SNORM) => VertexFormat::Snorm16x2,
        (0x0f, ATTRIB_TYPE_SINT) => VertexFormat::Sint16x2,
        (0x0f, ATTRIB_TYPE_UINT) => VertexFormat::Uint16x2,
        (0x1d, ATTRIB_TYPE_UNORM) => VertexFormat::Unorm8,
        (0x1d, ATTRIB_TYPE_SNORM) => VertexFormat::Snorm8,
        (0x1d, ATTRIB_TYPE_SINT) => VertexFormat::Sint8,
        (0x1d, ATTRIB_TYPE_UINT) => VertexFormat::Uint8,
        (0x18, ATTRIB_TYPE_UNORM) => VertexFormat::Unorm8x2,
        (0x18, ATTRIB_TYPE_SNORM) => VertexFormat::Snorm8x2,
        (0x18, ATTRIB_TYPE_SINT) => VertexFormat::Sint8x2,
        (0x18, ATTRIB_TYPE_UINT) => VertexFormat::Uint8x2,
        (0x0a, ATTRIB_TYPE_UNORM) => VertexFormat::Unorm8x4,
        (0x0a, ATTRIB_TYPE_SNORM) => VertexFormat::Snorm8x4,
        (0x0a, ATTRIB_TYPE_SINT) => VertexFormat::Sint8x4,
        (0x0a, ATTRIB_TYPE_UINT) => VertexFormat::Uint8x4,
        // Size `0x30` is 10-10-10-2; see [`Packed1010102`].
        (0x30, ATTRIB_TYPE_SNORM) => VertexFormat::Packed1010102(Packed1010102::Snorm),
        (0x30, ATTRIB_TYPE_UNORM) => VertexFormat::Packed1010102(Packed1010102::Unorm),
        (0x30, ATTRIB_TYPE_SINT) => VertexFormat::Packed1010102(Packed1010102::Sint),
        (0x30, ATTRIB_TYPE_UINT) => VertexFormat::Packed1010102(Packed1010102::Uint),
        (size, ty) => return Err(Unsupported::VertexFormat { size, ty }),
    })
}

/// Group attributes by vertex array, as WebGPU wants.
fn vertex_buffers(engine: &Engine3D) -> Result<(Vec<VertexBuffer>, Vec<u32>), Unsupported> {
    let mut buffers: Vec<VertexBuffer> = Vec::new();
    let mut fixed = Vec::new();
    for location in 0..MAX_VERTEX_ATTRIBS {
        let attrib = engine.vertex_attrib(location);
        // Size 0 is an unconfigured slot.
        if attrib.size == 0 {
            continue;
        }
        if attrib.is_fixed {
            fixed.push(location);
            continue;
        }
        let array = engine.vertex_array(attrib.buffer_id);
        if !array.enabled {
            // An attribute reading a disabled array is an error, as in `fetch_attribute`.
            return Err(Unsupported::State(format!(
                "attribute {location} reads from disabled vertex buffer {}",
                attrib.buffer_id
            )));
        }
        let scaled = matches!(attrib.ty, ATTRIB_TYPE_USCALED | ATTRIB_TYPE_SSCALED);
        let ty = match attrib.ty {
            ATTRIB_TYPE_USCALED => ATTRIB_TYPE_UINT,
            ATTRIB_TYPE_SSCALED => ATTRIB_TYPE_SINT,
            ty => ty,
        };
        let attribute = VertexAttribute {
            format: vertex_format(attrib.size, ty)?,
            offset: attrib.offset,
            location,
            is_bgra: attrib.is_bgra,
            scaled,
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

/// Divisor 0 steps per vertex, 1 per instance; others are unsupported.
fn step_mode(array: VertexArray) -> Result<StepMode, Unsupported> {
    match array.divisor {
        0 => Ok(StepMode::Vertex),
        1 => Ok(StepMode::Instance),
        divisor => Err(Unsupported::InstanceDivisor { divisor }),
    }
}

#[cfg(test)]
mod tests {
    use super::{Blend, BlendComponent, BlendFactor, BlendOperation};

    fn component(
        src_factor: BlendFactor,
        dst_factor: BlendFactor,
        operation: BlendOperation,
    ) -> BlendComponent {
        BlendComponent {
            src_factor,
            dst_factor,
            operation,
        }
    }

    fn both(c: BlendComponent) -> Blend {
        Blend { color: c, alpha: c }
    }

    /// Min and max ignore the factors, and WebGPU refuses a pipeline that names any but one
    /// beside them: Breath of the Wild leaves source alpha beside a min.
    #[test]
    fn a_min_or_max_blend_takes_one_for_both_factors() {
        let target = crate::gpu::engine::threed::BlendTarget {
            enabled: true,
            equation_rgb: 0x8007,
            func_rgb_src: 0x4302,
            func_rgb_dst: 0x4303,
            equation_alpha: 5,
            func_alpha_src: 5,
            func_alpha_dst: 1,
        };
        let (Some(blend), _) = super::blend(target, [0.0; 4]).unwrap() else {
            panic!("blending is on");
        };
        let one = component(BlendFactor::One, BlendFactor::One, BlendOperation::Min);
        assert_eq!(blend.color, one);
        assert_eq!(
            blend.alpha,
            BlendComponent {
                operation: BlendOperation::Max,
                ..one
            }
        );
    }

    /// Breath of the Wild's `R16G16B16A16_UNORM` probes blend a constant alpha against its
    /// complement, which a float stand-in can do without the clamp.
    #[test]
    fn a_constant_mix_stays_in_range_while_its_constant_does() {
        use BlendFactor::*;
        let mix = both(component(Constant, OneMinusConstant, BlendOperation::Add));
        assert!(mix.stays_in_unit_range([0.25; 4]));
        assert!(!mix.stays_in_unit_range([1.5; 4]));
        assert!(!mix.stays_in_unit_range([-0.5; 4]));
    }

    #[test]
    fn adding_both_terms_whole_leaves_the_range() {
        use BlendFactor::*;
        assert!(!both(component(One, One, BlendOperation::Add)).stays_in_unit_range([0.0; 4]));
        assert!(both(component(One, Zero, BlendOperation::Add)).stays_in_unit_range([0.0; 4]));
        assert!(
            both(component(SrcAlpha, OneMinusSrcAlpha, BlendOperation::Add))
                .stays_in_unit_range([0.0; 4])
        );
    }

    #[test]
    fn subtracting_stays_in_range_only_with_nothing_taken_away() {
        use BlendFactor::*;
        assert!(!both(component(One, One, BlendOperation::Subtract)).stays_in_unit_range([0.0; 4]));
        assert!(both(component(One, Zero, BlendOperation::Subtract)).stays_in_unit_range([0.0; 4]));
        assert!(both(component(Zero, One, BlendOperation::ReverseSubtract))
            .stays_in_unit_range([0.0; 4]));
        assert!(both(component(One, One, BlendOperation::Max)).stays_in_unit_range([0.0; 4]));
    }

    use super::*;

    #[test]
    fn culling_both_faces_changes_nothing_only_for_triangles() {
        let mut engine = Engine3D::new();
        engine.regs.set(0x646, 1); // OGL_SET_CULL: enabled
        engine.regs.set(0x648, 0x408); // OGL_SET_CULL_FACE: front and back
        engine.last_draw.primitive = 4; // triangles
        assert!(culls_every_triangle(&engine));
        engine.last_draw.primitive = 1; // lines
        assert!(!culls_every_triangle(&engine));
        engine.regs.set(0x648, 0x405); // back only
        engine.last_draw.primitive = 4;
        assert!(!culls_every_triangle(&engine));
    }

    #[test]
    fn both_numberings_name_the_same_blend_factor() {
        // Both numberings occur; the Home Menu uses the D3D one.
        for (d3d, gl) in [
            (0x01, 0x4000),
            (0x02, 0x4001),
            (0x03, 0x4300),
            (0x04, 0x4301),
            (0x05, 0x4302),
            (0x06, 0x4303),
            (0x07, 0x4304),
            (0x08, 0x4305),
            (0x09, 0x4306),
            (0x0a, 0x4307),
            (0x0b, 0x4308),
            (0x61, 0xc001),
            (0x62, 0xc002),
        ] {
            let factor = blend_factor(d3d);
            assert!(factor.is_ok(), "D3D {d3d:#x}: {factor:?}");
            assert_eq!(factor, blend_factor(gl), "D3D {d3d:#x} against GL {gl:#x}");
        }
    }

    #[test]
    fn a_factor_neither_numbering_knows_is_reported_rather_than_defaulted() {
        // `raster::blend_factor` answers `One`; this reports it instead.
        assert_eq!(
            blend_factor(0x1234),
            Err(Unsupported::BlendFactor { code: 0x1234 })
        );
    }

    #[test]
    fn a_constant_alpha_factor_has_no_webgpu_spelling() {
        for code in [0x63, 0x64, 0xc003, 0xc004] {
            assert_eq!(blend_factor(code), Err(Unsupported::BlendFactor { code }));
        }
    }

    #[test]
    fn both_numberings_name_the_same_blend_equation() {
        for (simple, gl) in [
            (0x1, 0x8006),
            (0x2, 0x800a),
            (0x3, 0x800b),
            (0x4, 0x8007),
            (0x5, 0x8008),
        ] {
            let operation = blend_equation(simple);
            assert!(operation.is_ok(), "{simple:#x}: {operation:?}");
            assert_eq!(operation, blend_equation(gl), "{simple:#x} against {gl:#x}");
        }
        assert_eq!(
            blend_equation(0x77),
            Err(Unsupported::BlendEquation { code: 0x77 })
        );
    }

    #[test]
    fn a_blend_reading_only_the_constant_alpha_draws_with_it_in_every_channel() {
        let target = |rgb_src, rgb_dst| BlendTarget {
            enabled: true,
            equation_rgb: 0x8006,
            func_rgb_src: rgb_src,
            func_rgb_dst: rgb_dst,
            equation_alpha: 0x8006,
            func_alpha_src: 0x4001,
            func_alpha_dst: 0x4000,
        };
        let constant = [0.1, 0.2, 0.3, 0.4];
        let (state, drawn_with) = blend(target(0xc003, 0xc004), constant).unwrap();
        let color = state.unwrap().color;
        assert_eq!(
            (color.src_factor, color.dst_factor),
            (BlendFactor::Constant, BlendFactor::OneMinusConstant)
        );
        assert_eq!(drawn_with, [0.4; 4]);

        assert_eq!(
            blend(target(0xc003, 0xc001), constant),
            Err(Unsupported::BlendFactor { code: 0xc003 }),
            "the colour constant needs its own channels"
        );
        let grey = [0.5; 4];
        assert_eq!(blend(target(0x63, 0x61), grey).unwrap().1, grey);
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
        assert_eq!(blend(target, [0.0; 4]), Ok((None, [0.0; 4])));
        target.enabled = true;
        assert!(
            blend(target, [0.0; 4]).is_err(),
            "an enabled blend reads its codes"
        );
    }

    #[test]
    fn topologies_webgpu_lacks_are_assembled_into_triangles_instead() {
        // These all become triangle lists; a line loop does not.
        for primitive in [
            Primitive::TriangleFan,
            Primitive::Quads,
            Primitive::QuadStrip,
            Primitive::Polygon,
        ] {
            assert_eq!(
                topology(primitive),
                Ok((Topology::TriangleList, Some(primitive)))
            );
            assert!(
                !crate::gpu::raster::assemble(primitive, 6).is_empty(),
                "{primitive:?} names an expansion the rasterizer does not perform"
            );
        }
        assert_eq!(
            topology(Primitive::LineLoop),
            Err(Unsupported::Topology(Primitive::LineLoop))
        );
        assert_eq!(
            topology(Primitive::Triangles),
            Ok((Topology::TriangleList, None))
        );
        assert_eq!(
            topology(Primitive::TriangleStrip),
            Ok((Topology::TriangleStrip, None))
        );
    }

    #[test]
    fn a_depth_test_that_is_off_compares_always() {
        // A disabled test is `Always`; `func` is never read.
        let layout = DepthLayout {
            bytes: 4,
            depth_bits: 24,
            depth_shift: 8,
            stencil_shift: Some(0),
        };
        let off = DepthState {
            test_enabled: false,
            write_enabled: true,
            func: 0xdead,
        };
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
        // Maxwell accepts both GL (0x200..=0x207) and D3D (1..=8) comparison codes.
        let layout = DepthLayout {
            bytes: 4,
            depth_bits: 24,
            depth_shift: 8,
            stencil_shift: None,
        };
        let of = |func| {
            let on = DepthState {
                test_enabled: true,
                write_enabled: false,
                func,
            };
            depth(layout, on).unwrap().compare
        };
        for (d3d, gl) in (1..=8u32).zip(0x0200..=0x0207u32) {
            assert_eq!(of(d3d), of(gl), "D3D {d3d} against GL {gl:#x}");
        }
        assert_eq!(of(4), Compare::LessEqual, "the one Just Dance 2019 sends");
        assert_eq!(of(0x0203), Compare::LessEqual);

        let on = DepthState {
            test_enabled: true,
            write_enabled: false,
            func: 0x40,
        };
        assert_eq!(
            depth(layout, on),
            Err(Unsupported::DepthCompare { code: 0x40 })
        );
    }

    fn shape(format: Format) -> (u32, bool) {
        match format {
            Format::R8Unorm | Format::R8Snorm | Format::R8Uint => (1, false),
            Format::Rg8Unorm | Format::Rg8Snorm => (2, false),
            Format::R16Unorm | Format::R16Snorm | Format::R16Float => (2, false),
            Format::Rgba8Unorm
            | Format::Rgba8Snorm
            | Format::Bgra8Unorm
            | Format::Rgb10a2Unorm
            | Format::Rg16Unorm
            | Format::Rg16Snorm
            | Format::Rg16Float
            | Format::R32Float
            | Format::R32Uint
            | Format::Rg11b10Ufloat => (4, false),
            Format::Rgba8UnormSrgb | Format::Bgra8UnormSrgb => (4, true),
            Format::Rgba16Unorm
            | Format::Rgba16Snorm
            | Format::Rgba16Float
            | Format::Rgba16Uint
            | Format::Rg32Float => (8, false),
            Format::Rgba32Float | Format::Rgba32Uint => (16, false),
            other => panic!("{other:?} is not a colour target format"),
        }
    }

    /// The blend clamp range per format; `None` for floats.
    fn clamp(format: Format) -> Option<(f32, f32)> {
        match format {
            Format::R16Float
            | Format::Rg16Float
            | Format::Rgba16Float
            | Format::R32Float
            | Format::Rg32Float
            | Format::Rgba32Float
            | Format::R8Uint
            | Format::R32Uint
            | Format::Rgba16Uint
            | Format::Rgba32Uint
            | Format::Rg11b10Ufloat => None,
            Format::R8Snorm
            | Format::Rg8Snorm
            | Format::Rgba8Snorm
            | Format::R16Snorm
            | Format::Rg16Snorm
            | Format::Rgba16Snorm => Some((-1.0, 1.0)),
            _ => Some((0.0, 1.0)),
        }
    }

    #[test]
    fn a_colour_format_is_what_the_surface_module_makes_of_the_same_code() {
        // These names must agree with `gpu::surface` on width and transfer function.
        for raw in 0u32..=0xff {
            let Ok(format) = ColorFormat::from_raw(raw) else {
                continue;
            };
            let Ok(named) = color_format(format) else {
                continue;
            };
            let (bytes, srgb) = shape(named);
            assert_eq!(bytes, format.bytes_per_pixel, "{raw:#x} is {named:?}");
            assert_eq!(srgb, format.is_srgb(), "{raw:#x} is {named:?}");
            // And on the channel number kind.
            assert_eq!(clamp(named), format.source_clamp(), "{raw:#x} is {named:?}");
        }
    }

    /// Codes `gpu::surface` stores that have no device name, deliberately.
    #[test]
    fn the_codes_with_no_webgpu_name_are_the_ones_that_cannot_have_one() {
        const UNNAMED: [u32; 22] = [
            0xC1, 0xC4, 0xC5, 0xC8, 0xCC, 0xCD, 0xD2, 0xD8, 0xD9, 0xDC, 0xDD, 0xE3, 0xEC, 0xED,
            0xF0, 0xF1, 0xF5, // `A2R10G10B10`, `R5G6B5`, `A1R5G5B5`, `X1R5G5B5`, `A8`.
            0xDF, 0xE8, 0xE9, 0xF7, 0xF8,
        ];
        let mut unnamed = Vec::new();
        for raw in 0u32..=0xFF {
            let Ok(format) = ColorFormat::from_raw(raw) else {
                continue;
            };
            if format.encode([0.0; 4]).is_err() {
                continue;
            }
            if color_format(format).is_err() {
                unnamed.push(raw);
            }
        }
        let mut expected = UNNAMED;
        expected.sort_unstable();
        assert_eq!(unnamed, expected);
    }

    #[test]
    fn a_mirrored_viewport_is_a_rectangle_and_a_flag() {
        // A negative scale_y flips the viewport, same rectangle.
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
        // Both map onto 0..1 window depth, differing in the expected clip z range.
        let gl = viewport(ViewportTransform {
            scale: [1.0, 1.0, 0.5],
            translate: [0.0, 0.0, 0.5],
        });
        assert_eq!((gl.min_depth, gl.max_depth), (0.0, 1.0));
        assert!(gl.depth_minus_one_to_one());
        let vulkan = viewport(ViewportTransform {
            scale: [1.0, 1.0, 1.0],
            translate: [0.0, 0.0, 0.0],
        });
        assert!(!vulkan.depth_minus_one_to_one());
    }

    #[test]
    fn a_scaled_attribute_keeps_its_integer_format_and_says_it_is_scaled() {
        let attribute = |format, scaled| VertexAttribute {
            format,
            offset: 0,
            location: 0,
            is_bgra: false,
            scaled,
        };
        assert_eq!(
            attribute(VertexFormat::Uint8x4, true).base(),
            AttributeBase::Uscaled
        );
        assert_eq!(
            attribute(VertexFormat::Sint16x2, true).base(),
            AttributeBase::Sscaled
        );
        assert_eq!(
            attribute(VertexFormat::Uint8x4, false).base(),
            AttributeBase::Uint
        );
    }

    #[test]
    fn an_instance_step_webgpu_cannot_take_is_reported() {
        let array = |divisor| VertexArray {
            enabled: true,
            stride: 16,
            start: 0,
            limit: 0,
            divisor,
        };
        assert_eq!(step_mode(array(0)), Ok(StepMode::Vertex));
        assert_eq!(step_mode(array(1)), Ok(StepMode::Instance));
        assert_eq!(
            step_mode(array(2)),
            Err(Unsupported::InstanceDivisor { divisor: 2 })
        );
    }

    /// Every format named here is one the rasterizer fetches, over as many bytes as it says.
    #[test]
    fn vertex_formats_are_the_ones_the_rasterizer_can_fetch() {
        use crate::gpu::engine::threed::VertexAttrib;
        use crate::gpu::exec::ExecCtx;
        use crate::gpu::raster::fetch_attribute;
        use crate::gpu::syncpt::Host1x;
        use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
        use crate::mem::Memory;

        let mem = Memory::new();
        mem.map(0x6000_0000, &[0xFF; 0x1000]).unwrap();
        let mut vmm = AddressSpace::new();
        let base = vmm
            .map(0x6000_0000, 0x1000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx = ExecCtx {
            mem: &mem,
            vmm: &vmm,
            as_id: 0,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        let mut named = 0;
        for size in 0..0x40 {
            for ty in [
                ATTRIB_TYPE_SNORM,
                ATTRIB_TYPE_UNORM,
                ATTRIB_TYPE_SINT,
                ATTRIB_TYPE_UINT,
                ATTRIB_TYPE_FLOAT,
            ] {
                let Ok(format) = vertex_format(size, ty) else {
                    continue;
                };
                named += 1;
                let attrib = VertexAttrib {
                    buffer_id: 0,
                    is_fixed: false,
                    offset: 0,
                    size,
                    ty,
                    is_bgra: false,
                };
                // Past the array's limit the fetch reads zero in place of the 0xFF bytes there.
                let first_channel = |limit| {
                    let array = VertexArray {
                        enabled: true,
                        stride: 0,
                        start: base,
                        limit,
                        divisor: 0,
                    };
                    fetch_attribute(attrib, array, 0, &ctx)
                        .unwrap_or_else(|e| panic!("{format:?} at {size:#x}/{ty}: {e:?}"))[0]
                        .to_bits()
                };
                let bytes = format.size();
                let last = base + u64::from(bytes) - 1;
                assert_ne!(
                    first_channel(last),
                    0,
                    "{format:?}: more than {bytes} bytes fetched"
                );
                assert_eq!(
                    first_channel(last - 1),
                    0,
                    "{format:?}: under {bytes} bytes fetched"
                );
                // The rasterizer carries an integer type's bits and turns every other into floats.
                let base_type = match ty {
                    ATTRIB_TYPE_SINT => AttributeBase::Sint,
                    ATTRIB_TYPE_UINT => AttributeBase::Uint,
                    _ => AttributeBase::Float,
                };
                assert_eq!(format.base(), base_type, "{format:?}");
            }
        }
        assert_ne!(named, 0);
    }

    #[test]
    fn a_draw_with_no_targets_at_all_says_so_rather_than_panicking() {
        // An unwritten register file is an error, as in the rasterizer.
        let engine = Engine3D::new();
        assert_eq!(
            Pipeline::of(&engine),
            Err(Unsupported::State(
                "a draw with neither a colour nor a depth target".into()
            ))
        );
    }
}
