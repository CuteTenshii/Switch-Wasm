//! MAXWELL_B (class 0xB197), the 3D engine.
//! Register numbers come from deko3d's `engine_3d.def`.

use crate::gpu::engine::Registers;
use crate::gpu::macro_engine::MacroEngine;
use crate::gpu::renderer::Renderer;
use crate::gpu::surface::{ColorFormat, Layout};
use crate::{Error, Result};

mod clear;
mod dispatch;
mod state;

#[cfg(test)]
mod tests;

// Registers with behaviour attached. Everything else is plain state.
const MME_INSTRUCTION_RAM_POINTER: u32 = 0x045;
const MME_INSTRUCTION_RAM_LOAD: u32 = 0x046;
const MME_START_ADDRESS_RAM_POINTER: u32 = 0x047;
const MME_START_ADDRESS_RAM_LOAD: u32 = 0x048;
const SYNCPT_ACTION: u32 = 0x0B2;
const RENDER_TARGET_BASE: u32 = 0x200;
const RENDER_TARGET_STRIDE: u32 = 0x10;
const VIEWPORT_TRANSFORM_BASE: u32 = 0x280;
const VIEWPORT_BASE: u32 = 0x300;
const SCISSOR_BASE: u32 = 0x380;
// NV9097_OGL_SET_CULL / _FRONT_FACE / _CULL_FACE (cl9097.h), as dword indices.
const OGL_SET_CULL: u32 = 0x646;
const OGL_SET_FRONT_FACE: u32 = 0x647;
const OGL_SET_CULL_FACE: u32 = 0x648;
// NV9097_SET_INDEX_BUFFER_A (method 0x17c8).
const INDEX_ARRAY_START: u32 = 0x5F2;
const DRAW_ARRAYS_COUNT: u32 = 0x35E;
const CLEAR_COLOR: u32 = 0x360;
const CLEAR_DEPTH: u32 = 0x364;
const CLEAR_STENCIL: u32 = 0x368;
const DEPTH_TARGET_ADDR: u32 = 0x3F8;
const DEPTH_TARGET_FORMAT: u32 = 0x3FA;
const DEPTH_TARGET_TILE_MODE: u32 = 0x3FB;
const SCREEN_SCISSOR_HORIZONTAL: u32 = 0x3FD;
const SCREEN_SCISSOR_VERTICAL: u32 = 0x3FE;
/// `SET_WINDOW_ORIGIN_MODE` (method 0x13AC).
const WINDOW_ORIGIN: u32 = 0x4EB;
const CLEAR_BUFFER_FLAGS: u32 = 0x43E;
const RENDER_TARGET_CONTROL: u32 = 0x487;
const DEPTH_TARGET_HORIZONTAL: u32 = 0x48A;
const DEPTH_TARGET_VERTICAL: u32 = 0x48B;
const VERTEX_END_GL: u32 = 0x585;
const VERTEX_BEGIN_GL: u32 = 0x586;
/// `VertexBeginGl.InstanceNext`: step the instance counter instead of resetting it.
const VERTEX_BEGIN_INSTANCE_NEXT: u32 = 1 << 26;
const DRAW_ELEMENTS_COUNT: u32 = 0x5F8;
const CLEAR_BUFFERS: u32 = 0x674;
const REPORT_SEMAPHORE_OFFSET: u32 = 0x6C0;
const REPORT_SEMAPHORE_PAYLOAD: u32 = 0x6C2;
const REPORT_SEMAPHORE: u32 = 0x6C3;
const CONSTBUF_SELECTOR_SIZE: u32 = 0x8E0;
const CONSTBUF_SELECTOR_ADDR: u32 = 0x8E1;
const LOAD_CONSTBUF_OFFSET: u32 = 0x8E3;
const LOAD_CONSTBUF_DATA: u32 = 0x8E4;
const LOAD_CONSTBUF_DATA_LAST: u32 = 0x8F3;

// Falcon firmware method-call interface; see `firmware_call`.
const FIRMWARE_CALL: u32 = 0x8C0;
const FIRMWARE_CALL_LAST: u32 = 0x8DF;
const MME_FIRMWARE_ARGS: u32 = 0xD00;

// Inline-to-memory methods, which the 3D class also implements.
const INLINE_FIRST: u32 = 0x060;
const INLINE_LAST: u32 = 0x06D;

// --- Shader program binding ---
// `SetProgram[stage].Offset` is relative to `SetProgramRegion`.
const SET_PROGRAM_REGION: u32 = 0x582;
const SET_PROGRAM: u32 = 0x800;
const SET_PROGRAM_STRIDE: u32 = 0x10;

// --- Vertex format ---
const VERTEX_ATTRIB_STATE: u32 = 0x458;
const VERTEX_ARRAY: u32 = 0x700;
const VERTEX_ARRAY_STRIDE: u32 = 0x4;
const VERTEX_ARRAY_LIMIT: u32 = 0x7C0;
/// `VertexStreamInstances[i]`: whether array `i` steps per instance.
const VERTEX_ARRAY_PER_INSTANCE: u32 = 0x620;

// --- Texture/sampler pools ---
const TEX_CB_INDEX: u32 = 0x982;
const SET_TEX_SAMPLER_POOL: u32 = 0x557;
const SET_TEX_HEADER_POOL: u32 = 0x55D;

// --- Depth/stencil test ---
const DEPTH_TEST_ENABLE: u32 = 0x4B3;
const INDEPENDENT_BLEND_ENABLE: u32 = 0x4B9;
const DEPTH_WRITE_ENABLE: u32 = 0x4BA;
const DEPTH_TEST_FUNC: u32 = 0x4C3;

// --- Blend ---
const BLEND_CONSTANT: u32 = 0x4C7;
// Shared blend state, used when `IndependentBlendEnable` is off.
const BLEND_EQUATION_RGB: u32 = 0x4D0;
const BLEND_FUNC_SRC_RGB: u32 = 0x4D1;
const BLEND_FUNC_DST_RGB: u32 = 0x4D2;
const BLEND_EQUATION_ALPHA: u32 = 0x4D3;
const BLEND_FUNC_SRC_ALPHA: u32 = 0x4D4;
const BLEND_FUNC_DST_ALPHA: u32 = 0x4D6;
const COLOR_BLEND_ENABLE: u32 = 0x4D8;
const INDEPENDENT_BLEND: u32 = 0x780;
const INDEPENDENT_BLEND_STRIDE: u32 = 0x8;

// --- Colour write mask ---
// `SetCtWrite[i]`: one nibble per channel per colour target.
const COLOR_MASK: u32 = 0x680;
const COLOR_MASK_COMMON: u32 = 0x3E4;
const COLOR_TARGETS: u32 = 8;
/// Every channel enabled, the reading of an unwritten mask.
const COLOR_MASK_ALL: u32 = 0x1111;

// --- Multisampling ---
const MULTISAMPLE_SAMPLE_MASK: u32 = 0x3EF;
// Sixteen one-byte sample locations across four registers.
const MULTISAMPLE_SAMPLE_LOCATIONS: u32 = 0x478;
const MULTISAMPLE_ENABLE: u32 = 0x54D;
const MULTISAMPLE_CONTROL: u32 = 0x54F;
/// A `MsaaMode`; see [`SampleGrid`].
const MULTISAMPLE_MODE: u32 = 0x574;

// --- Constant-buffer stage binding ---
// Writing `Bind[slot].Constbuf` snapshots the selected constbuf into that stage's bank.
const BIND: u32 = 0x900;
const BIND_STRIDE: u32 = 0x8;
/// How many stages have a `Bind` block of their own.
const BIND_SLOTS: usize = 5;
const BIND_LAST: u32 = BIND + BIND_SLOTS as u32 * BIND_STRIDE - 1;
const BIND_CONSTBUF_OFFSET: u32 = 0x4;
/// `Bind.ConstantBuffer.Index` is five bits wide.
const CONSTBUF_BANKS: usize = 32;

/// Pipeline stage, numbered as `SetProgram[i].Config.StageId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    VertexA,
    VertexB,
    TessCtrl,
    TessEval,
    Geometry,
    Fragment,
}

impl ShaderStage {
    const ALL: [ShaderStage; 6] = [
        ShaderStage::VertexA,
        ShaderStage::VertexB,
        ShaderStage::TessCtrl,
        ShaderStage::TessEval,
        ShaderStage::Geometry,
        ShaderStage::Fragment,
    ];

    fn index(self) -> u32 {
        Self::ALL.iter().position(|&s| s == self).unwrap() as u32
    }

    /// `Bind`'s array index for this stage; `VertexA` and `VertexB` share one.
    fn bind_slot(self) -> u32 {
        match self {
            ShaderStage::VertexA | ShaderStage::VertexB => 0,
            ShaderStage::TessCtrl => 1,
            ShaderStage::TessEval => 2,
            ShaderStage::Geometry => 3,
            ShaderStage::Fragment => 4,
        }
    }
}

/// A `SetProgram[stage]` resolved into an absolute address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramBinding {
    pub addr: u64,
    pub num_registers: u32,
}

/// A `VertexAttribState[i]` entry, with raw size/type enum values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttrib {
    pub buffer_id: u32,
    pub is_fixed: bool,
    pub offset: u32,
    pub size: u32,
    pub ty: u32,
    pub is_bgra: bool,
}

/// A `VertexArray[i]` + its `VertexArrayLimit[i]`: one vertex buffer binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexArray {
    pub enabled: bool,
    pub stride: u32,
    pub start: u64,
    pub limit: u64,
    pub divisor: u32,
}

/// One `IndependentBlend[i]` entry plus its `ColorBlendEnable[i]` bit, as raw enum codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendTarget {
    pub enabled: bool,
    pub equation_rgb: u32,
    pub func_rgb_src: u32,
    pub func_rgb_dst: u32,
    pub equation_alpha: u32,
    pub func_alpha_src: u32,
    pub func_alpha_dst: u32,
}

/// Depth-test state; `func` is the raw one-based `DepthTestFunc` code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthState {
    pub test_enabled: bool,
    pub write_enabled: bool,
    pub func: u32,
}

impl DepthState {
    /// Whether a draw under this state reads or writes its depth surface.
    pub fn reaches_surface(&self) -> bool {
        const ALWAYS: [u32; 2] = [8, 0x0207];
        self.test_enabled && (self.write_enabled || !ALWAYS.contains(&self.func))
    }
}

/// The texel extent a draw is confined to: the intersection of its colour and depth targets.
pub fn draw_extent(
    color: Option<(u32, u32)>,
    depth: Option<(u32, u32)>,
    state: DepthState,
) -> Option<(u32, u32)> {
    match (color, depth) {
        (Some((w, h)), Some((dw, dh))) if state.reaches_surface() => Some((w.min(dw), h.min(dh))),
        (Some(extent), _) | (None, Some(extent)) => Some(extent),
        (None, None) => None,
    }
}

/// A depth/stencil render target resolved from the register file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthTarget {
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    pub layout: Layout,
    /// Where this format keeps its depth and its stencil inside a pixel.
    pub format: DepthLayout,
}

/// One viewport's transform: `window = ndc * scale + translate` per axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportTransform {
    pub scale: [f32; 3],
    pub translate: [f32; 3],
}

/// Which corner window coordinates are measured from, and which winding is front.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WindowOrigin {
    /// Window y is measured from the bottom of the surface clip.
    pub lower_left: bool,
    /// The front face is the opposite winding to `OGL_SET_FRONT_FACE`'s.
    pub flip_y: bool,
}

/// A resolved pixel rectangle, `[x0, x1) x [y0, y1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScissorRect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

/// Which faces a draw throws away before rasterizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CullState {
    pub enabled: bool,
    /// Whether counter-clockwise winding (in NDC) is the front face.
    pub front_ccw: bool,
    pub cull_front: bool,
    pub cull_back: bool,
}

/// A draw the engine was asked to perform.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DrawCall {
    /// `VertexBeginGl` primitive type.
    pub primitive: u32,
    pub first: u32,
    pub count: u32,
    pub indexed: bool,
    /// Index buffer format (0 = u8, 1 = u16, 2 = u32) for indexed draws.
    pub index_format: u32,
}

/// A colour render target resolved from the register file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTarget {
    pub addr: u64,
    /// Width in pixels; for a pitch target, derived from the stride.
    pub width: u32,
    pub height: u32,
    pub format: ColorFormat,
    pub layout: Layout,
    pub layers: u32,
    pub layer_stride: u32,
}

impl RenderTarget {
    /// Byte offset of a texel (one sample on a multisampled target).
    pub fn texel_offset(&self, x: u32, y: u32) -> u32 {
        let bpp = self.format.bytes_per_pixel;
        let width_bytes = self.width * bpp;
        self.layout.offset(x * bpp, y, width_bytes)
    }

    /// [`Target::texel_offset`] plus how many texels from there are contiguous (at least one).
    pub fn texel_run(&self, x: u32, y: u32) -> (u32, u32) {
        let bpp = self.format.bytes_per_pixel;
        let width_bytes = self.width * bpp;
        let (offset, run) = self.layout.run_at(x * bpp, y, width_bytes);
        (offset, (run / bpp).max(1))
    }
}

/// The parameters and span of a depth clear, for skipping identical repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DepthFill {
    addr: u64,
    span: u32,
    value: u128,
    written: u128,
    bytes: u32,
    rect: (u32, u32, u32, u32),
    tile_mode: u32,
    width_bytes: u32,
}

#[derive(Debug)]
pub struct Engine3D {
    pub regs: Registers,
    /// Backend that turns draws and clears into pixels.
    renderer: Box<dyn Renderer>,
    /// `gl_InstanceID` for the next draw.
    instance_id: u32,
    traced_regs: Option<Vec<u32>>,
    pub macros: MacroEngine,
    /// Inline-to-memory unit; lives here so macro writes reach it too.
    pub inline: crate::gpu::engine::inline::EngineInline,
    /// The last draw the engine was asked to perform.
    pub last_draw: DrawCall,
    /// The last depth clear, to skip an identical repeat over untouched memory.
    depth_fill: std::cell::RefCell<Option<DepthFill>>,
    /// Write cursor for `LoadConstbufData`, in bytes.
    constbuf_cursor: u32,
    /// Constant banks bound per bind slot and bank.
    bound_constbufs: [[Option<(u64, u32)>; CONSTBUF_BANKS]; BIND_SLOTS],
    /// Draws and clears by target: see [`crate::gpu::activity`].
    pub activity: crate::gpu::activity::GpuActivity,
}

impl Default for Engine3D {
    fn default() -> Self {
        Engine3D::new()
    }
}

/// How a depth format packs depth and stencil into one pixel.
/// NVIDIA names fields most significant first: `Z24S8` keeps stencil in the low byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthLayout {
    /// Bytes per pixel.
    pub bytes: u32,
    /// How many bits of depth, or `0` for a 32-bit float.
    pub depth_bits: u32,
    /// Where the depth field starts within the pixel.
    pub depth_shift: u32,
    /// Where the stencil byte starts, for the formats that carry one.
    pub stencil_shift: Option<u32>,
}

impl DepthLayout {
    /// The bits of a pixel the depth field occupies.
    pub fn depth_mask(&self) -> u128 {
        u128::from(self.depth_mask64())
    }

    /// Depth fields sit in the low 64 bits; `u128` shifts are a libcall in wasm.
    fn depth_mask64(&self) -> u64 {
        let width: u64 = match self.depth_bits {
            0 => 0xFFFF_FFFF,
            bits => (1u64 << bits) - 1,
        };
        width << self.depth_shift
    }

    /// `depth` in `0.0..=1.0`, encoded in place; rounds in `f64` since `f32` loses 24-bit precision.
    pub fn encode_depth(&self, depth: f32) -> u128 {
        let stored = match self.depth_bits {
            0 => u64::from(depth.to_bits()),
            bits => {
                let max = ((1u64 << bits) - 1) as f64;
                (depth.clamp(0.0, 1.0) as f64 * max + 0.5) as u64
            }
        };
        u128::from(stored << self.depth_shift)
    }

    /// Inverse of [`DepthLayout::encode_depth`], from a whole stored pixel.
    pub fn decode_depth(&self, pixel: u128) -> f32 {
        let stored = (pixel as u64 & self.depth_mask64()) >> self.depth_shift;
        match self.depth_bits {
            0 => f32::from_bits(stored as u32),
            bits => {
                let max = ((1u64 << bits) - 1) as f64;
                (stored as f64 / max) as f32
            }
        }
    }

    /// `pixel` with its depth replaced and every other bit left alone.
    pub fn with_depth(&self, pixel: u128, depth: f32) -> u128 {
        let mask = self.depth_mask();
        (pixel & !mask) | (self.encode_depth(depth) & mask)
    }

    /// Whether depth shares its pixel with a stencil byte.
    pub fn packs_stencil(&self) -> bool {
        self.stencil_shift.is_some()
    }
}

/// The [`DepthLayout`] of a `SET_ZT_FORMAT` value.
pub(crate) fn depth_format_layout(raw: u32) -> Result<DepthLayout> {
    // `S8` has no depth field.
    let (bytes, depth_bits, depth_shift, stencil_shift) = match raw {
        0x0A => (4, 0, 0, None),      // ZF32
        0x13 => (2, 16, 0, None),     // Z16
        0x14 => (4, 24, 8, Some(0)),  // Z24S8
        0x15 => (4, 24, 0, None),     // X8Z24
        0x16 => (4, 24, 0, Some(24)), // S8Z24
        0x17 => (1, 0, 0, Some(0)),   // S8
        0x19 => (8, 0, 0, Some(32)),  // ZF32_X24S8
        other => {
            return Err(Error::Gpu(format!(
                "3d: unsupported depth format {:#x}",
                other
            )))
        }
    };
    Ok(DepthLayout {
        bytes,
        depth_bits,
        depth_shift,
        stencil_shift,
    })
}
