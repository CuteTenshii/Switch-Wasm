//! MAXWELL_B (class 0xB197) — the 3D engine.
//!
//! Register numbers come from the Maxwell class headers deko3d generates
//! (`source/maxwell/engine_3d.def`), so they match the command streams real
//! homebrew emits exactly.

use crate::gpu::engine::{field, Registers, REGISTER_COUNT};
use crate::gpu::exec::ExecCtx;
use crate::gpu::macro_engine::{MacroEngine, MacroHost, MacroWrite, MACRO_METHODS_START};
use crate::gpu::renderer::{Renderer, Software};
use crate::gpu::surface::{
    ColorFormat, Layout, SampleGrid, GOB_HEIGHT, GOB_SIZE, GOB_WIDTH, MAX_SAMPLES,
};
use crate::{Error, Result};

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
// NV9097_OGL_SET_CULL / _FRONT_FACE / _CULL_FACE at methods 0x1918/0x191c/
// 0x1920 (NVIDIA's cl9097.h), as dword indices.
const OGL_SET_CULL: u32 = 0x646;
const OGL_SET_FRONT_FACE: u32 = 0x647;
const OGL_SET_CULL_FACE: u32 = 0x648;
// NV9097_SET_INDEX_BUFFER_A at method 0x17c8: the address pair the format,
// first and count registers this file already names follow on from.
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
const CLEAR_BUFFER_FLAGS: u32 = 0x43E;
const RENDER_TARGET_CONTROL: u32 = 0x487;
const DEPTH_TARGET_HORIZONTAL: u32 = 0x48A;
const DEPTH_TARGET_VERTICAL: u32 = 0x48B;
const VERTEX_END_GL: u32 = 0x585;
const VERTEX_BEGIN_GL: u32 = 0x586;
/// `VertexBeginGl.InstanceNext`. An instanced draw is not one method with an
/// instance count: it is one `Begin`/`End` pair per instance, every pair after
/// the first carrying this bit, which tells the hardware to step its instance
/// counter rather than reset it. Without it every instance is instance zero —
/// and a UI that draws each of its elements as one instance of a unit quad
/// stacks all of them on top of each other.
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

// The Falcon firmware "method call" interface: writing one of these kicks off
// a firmware routine whose arguments live in the MmeFirmwareArgs registers
// (0xD00..). deko3d's `WriteHardwareReg` macro uses FirmwareCall[4] to poke
// PGRAPH registers, then polls MmeFirmwareArgs[0] until the firmware reports
// completion — see `firmware_call` below.
const FIRMWARE_CALL: u32 = 0x8C0;
const FIRMWARE_CALL_LAST: u32 = 0x8DF;
const MME_FIRMWARE_ARGS: u32 = 0xD00;

// The 3D class also implements the inline-to-memory methods; deko3d issues
// them on the 3D subchannel (see `gpu_transfer.cpp`).
const INLINE_FIRST: u32 = 0x060;
const INLINE_LAST: u32 = 0x06D;

// --- Shader program binding ---
// `SetProgramRegion` is a base iova; each `SetProgram[stage]` entry's `Offset`
// is relative to it. Register numbers and layout from deko3d's
// `engine_3d.def`, cross-checked against JKSV's own real shader uploads (see
// `gpu/shader/mod.rs`'s module doc for how that was derived).
const SET_PROGRAM_REGION: u32 = 0x582;
const SET_PROGRAM: u32 = 0x800;
const SET_PROGRAM_STRIDE: u32 = 0x10;

// --- Vertex format ---
const VERTEX_ATTRIB_STATE: u32 = 0x458;
const VERTEX_ARRAY: u32 = 0x700;
const VERTEX_ARRAY_STRIDE: u32 = 0x4;
const VERTEX_ARRAY_LIMIT: u32 = 0x7C0;

// --- Texture/sampler pools ---
const TEX_CB_INDEX: u32 = 0x982;
const SET_TEX_SAMPLER_POOL: u32 = 0x557;
const SET_TEX_HEADER_POOL: u32 = 0x55D;

// --- Depth/stencil test (as opposed to depth *clear*, above) ---
const DEPTH_TEST_ENABLE: u32 = 0x4B3;
const INDEPENDENT_BLEND_ENABLE: u32 = 0x4B9;
const DEPTH_WRITE_ENABLE: u32 = 0x4BA;
const DEPTH_TEST_FUNC: u32 = 0x4C3;

// --- Blend ---
const BLEND_CONSTANT: u32 = 0x4C7;
// The shared, non-independent blend state (`gf100_3d.xml`: BLEND_EQUATION_RGB
// 0x1340..BLEND_FUNC_DST_ALPHA 0x1358, word-indexed) — what real hardware
// actually reads when `IndependentBlendEnable` is off, which is the common
// case. `IndependentBlend[i]` below is a *different* register block that
// only applies when that bit is set; reading it unconditionally left every
// blend factor/equation at 0 (not a valid `DkBlendFactor`/`DkBlendOp`) for
// any content that never turns independent blending on.
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
// `SetCtWrite[i]` (method 0x1A00), one register per colour target with each
// channel's enable in a nibble of its own. `ColorMaskCommon` (0x0F90) makes
// every target read slot 0's mask rather than its own.
const COLOR_MASK: u32 = 0x680;
const COLOR_MASK_COMMON: u32 = 0x3E4;
/// How many colour targets Maxwell has, and so how many masks.
const COLOR_TARGETS: u32 = 8;
/// Every channel enabled — hardware's usable state, and what a target whose
/// mask the guest never writes has to read as. Zero would mean "write
/// nothing", which is a blank frame for every guest that leaves it alone.
const COLOR_MASK_ALL: u32 = 0x1111;

// --- Multisampling ---
// Which samples a draw may write, one bit each; deko3d's
// `dkCmdBufSetSampleMask` writes the same mask to all four registers.
const MULTISAMPLE_SAMPLE_MASK: u32 = 0x3EF;
// Sixteen one-byte sample locations across four registers, as
// `dkMultisampleStateSetLocations` writes them.
const MULTISAMPLE_SAMPLE_LOCATIONS: u32 = 0x478;
const MULTISAMPLE_ENABLE: u32 = 0x54D;
const MULTISAMPLE_CONTROL: u32 = 0x54F;
/// A `MsaaMode`; see [`SampleGrid`].
const MULTISAMPLE_MODE: u32 = 0x574;

// --- Constant-buffer stage binding ---
// `Bind[slot].Constbuf{Valid, Index}` is a *trigger*, not plain state: on a
// valid write, real hardware snapshots the currently-selected
// `ConstbufSelectorAddr`/`Size` into a per-(stage, bank) table that the
// shader ISA's `cN[offset]` operand reads through directly — it is a
// separate mechanism from `LoadConstbufData`'s upload cursor, which just
// writes bytes to whatever `ConstbufSelectorAddr` currently points at.
// `Bind` has one slot per stage *except* the vertex shader's `VertexA`/
// `VertexB` split, which share a slot (see `ShaderStage::bind_slot`).
const BIND: u32 = 0x900;
const BIND_STRIDE: u32 = 0x8;
/// How many stages have a `Bind` block of their own.
const BIND_SLOTS: usize = 5;
const BIND_LAST: u32 = BIND + BIND_SLOTS as u32 * BIND_STRIDE - 1;
const BIND_CONSTBUF_OFFSET: u32 = 0x4;
/// `Bind.ConstantBuffer.Index` is five bits wide, so a stage has this many
/// constant banks.
const CONSTBUF_BANKS: usize = 32;

/// Which pipeline stage a `SetProgram`/`Bind` entry configures. Numbering
/// matches `SetProgram[i].Config.StageId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    /// `Bind`'s array index for this stage. `VertexA` and `VertexB` share a
    /// slot — Maxwell's constant-buffer bindings don't distinguish the split
    /// vertex-shader halves.
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

/// A `SetProgram[stage]` resolved into an absolute address, ready for
/// `gpu/shader::decode_program`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramBinding {
    pub addr: u64,
    pub num_registers: u32,
}

/// A `VertexAttribState[i]` entry: where in a vertex buffer this attribute's
/// data lives and how to interpret it. `size`/`ty` are the raw
/// `DkVtxAttribSize`/`DkVtxAttribType`-shaped enum values — decoding those
/// into an actual component count/format is `gpu/raster.rs`'s job.
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

/// One `IndependentBlend[i]` entry plus its shared `ColorBlendEnable[i]` bit.
/// Function/equation values are the raw Maxwell enum codes (GL blend
/// func/equation order) — resolving those is `gpu/raster.rs`'s job.
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

/// The depth-test state (as opposed to `DEPTH_TARGET_*`'s depth-clear
/// state). `func` is the raw `DepthTestFunc` enum code (`1..=8`,
/// `Never..=Always` — one-based, unlike GL's zero-based `GL_NEVER..=GL_ALWAYS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthState {
    pub test_enabled: bool,
    pub write_enabled: bool,
    pub func: u32,
}

/// A depth/stencil render target resolved from the register file, mirroring
/// [`RenderTarget`] for the colour side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthTarget {
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    pub layout: Layout,
    /// Where this format keeps its depth and its stencil inside a pixel.
    pub format: DepthLayout,
}

/// One viewport's NDC-to-window transform: `window = ndc * scale + translate`
/// on each of x, y and z. See [`Engine3D::viewport_transform`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportTransform {
    pub scale: [f32; 3],
    pub translate: [f32; 3],
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

/// A draw the engine was asked to perform, kept so the rasterizer stage (and
/// tests) can see what was requested.
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
    /// Width in pixels (block-linear) — for a pitch target this is derived
    /// from the stride.
    pub width: u32,
    pub height: u32,
    pub format: ColorFormat,
    pub layout: Layout,
    pub layers: u32,
    pub layer_stride: u32,
}

impl RenderTarget {
    /// Byte offset of a *texel* within the target. On a multisampled target a
    /// texel is one sample, not one pixel — [`SampleGrid::texel`] converts.
    pub fn texel_offset(&self, x: u32, y: u32) -> u32 {
        let bpp = self.format.bytes_per_pixel;
        let width_bytes = self.width * bpp;
        self.layout.offset(x * bpp, y, width_bytes)
    }

    /// [`Target::texel_offset`], plus how many texels from there on are
    /// contiguous — what a walk over the whole surface wants, rather than a
    /// swizzle per texel. At least one, so a caller can always make progress.
    pub fn texel_run(&self, x: u32, y: u32) -> (u32, u32) {
        let bpp = self.format.bytes_per_pixel;
        let width_bytes = self.width * bpp;
        let (offset, run) = self.layout.run_at(x * bpp, y, width_bytes);
        (offset, (run / bpp).max(1))
    }
}

#[derive(Debug)]
pub struct Engine3D {
    pub regs: Registers,
    /// What turns this engine's draws and clears into pixels. Swappable so a
    /// GPU backend can be a second implementation rather than a rewrite —
    /// see [`crate::gpu::renderer`].
    renderer: Box<dyn Renderer>,
    /// `gl_InstanceID` for the draw about to be issued — see
    /// [`VERTEX_BEGIN_INSTANCE_NEXT`].
    instance_id: u32,
    traced_regs: Option<Vec<u32>>,
    pub macros: MacroEngine,
    /// The inline-to-memory unit, which the 3D class exposes as its own
    /// methods and which lives here rather than on the channel so that
    /// **every** route into `write` reaches it. A macro's method writes do not
    /// go through the channel at all, and this class's inline upload is
    /// exactly what a macro is used for.
    pub inline: crate::gpu::engine::inline::EngineInline,
    /// The last draw the engine was asked to perform.
    pub last_draw: DrawCall,
    /// Write cursor for `LoadConstbufData`, in bytes.
    constbuf_cursor: u32,
    /// `(Bind slot, hardware bank index) -> (addr, size)` snapshots taken on
    /// each valid `Bind[slot].Constbuf` write — see the constant `BIND`'s
    /// doc comment for why this needs to be its own table rather than a
    /// plain register read.
    /// Where each stage's constant banks are bound, indexed by bind slot and
    /// bank rather than hashed.
    ///
    /// This was a `HashMap<(u32, u32), _>`, which meant a SipHash of the key
    /// for *every constant a shader reads* — one per instruction with a `c[]`
    /// operand, once per covered pixel. Hashing was 10% of the Home Menu's
    /// whole frame time. Both indices are small and bounded by the register
    /// layout, so an array is both faster and a better description of what the
    /// hardware has.
    bound_constbufs: [[Option<(u64, u32)>; CONSTBUF_BANKS]; BIND_SLOTS],
}

impl Default for Engine3D {
    fn default() -> Self {
        Engine3D::new()
    }
}

impl Engine3D {
    pub fn new() -> Engine3D {
        let mut regs = Registers::new();
        // See [`COLOR_MASK_ALL`]: an unwritten mask has to mean "all
        // channels", not "none".
        for target in 0..COLOR_TARGETS {
            regs.set(COLOR_MASK + target, COLOR_MASK_ALL);
        }
        Engine3D {
            regs,
            renderer: Box::new(Software),
            instance_id: 0,
            traced_regs: None,
            macros: MacroEngine::new(),
            inline: crate::gpu::engine::inline::EngineInline::new(),
            last_draw: DrawCall::default(),
            constbuf_cursor: 0,
            bound_constbufs: [[None; CONSTBUF_BANKS]; BIND_SLOTS],
        }
    }

    /// Handle one method write. `last_call` is true for the final write of a
    /// pushbuffer method group, which is what makes a pending macro run.
    pub fn write(
        &mut self,
        method: u32,
        arg: u32,
        last_call: bool,
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        if method >= MACRO_METHODS_START {
            return self.write_macro(method, arg, last_call, ctx);
        }
        self.regs.set(method, arg);
        match method {
            MME_INSTRUCTION_RAM_POINTER => self.macros.instruction_ram_pointer = arg,
            MME_INSTRUCTION_RAM_LOAD => self.macros.push_instruction(arg),
            MME_START_ADDRESS_RAM_POINTER => self.macros.start_address_pointer = arg,
            MME_START_ADDRESS_RAM_LOAD => self.macros.push_start_address(arg),
            SYNCPT_ACTION => self.syncpt_action(arg, ctx)?,
            CLEAR_BUFFERS => self.clear_buffers(arg, ctx)?,
            REPORT_SEMAPHORE => self.report_semaphore(arg, ctx)?,
            FIRMWARE_CALL..=FIRMWARE_CALL_LAST => self.firmware_call(arg, ctx)?,
            LOAD_CONSTBUF_OFFSET => self.constbuf_cursor = field(arg, 0, 15),
            LOAD_CONSTBUF_DATA..=LOAD_CONSTBUF_DATA_LAST => self.load_constbuf(arg, ctx)?,
            BIND..=BIND_LAST => self.bind(method, arg),
            DRAW_ARRAYS_COUNT => self.draw_arrays(arg, ctx)?,
            DRAW_ELEMENTS_COUNT => self.draw_elements(arg, ctx)?,
            VERTEX_BEGIN_GL => {
                self.last_draw.primitive = field(arg, 0, 15);
                self.instance_id = if arg & VERTEX_BEGIN_INSTANCE_NEXT != 0 {
                    self.instance_id.wrapping_add(1)
                } else {
                    0
                };
            }
            VERTEX_END_GL => {}
            // The inline-to-memory methods the 3D class shares with
            // KEPLER_INLINE_TO_MEMORY_B. These used to do nothing here on the
            // reasoning that the channel drives them — which it does, but only
            // for methods that arrive in a pushbuffer. A **macro**'s writes go
            // straight to this engine, and uploading a small buffer from a
            // macro is precisely what the class is for: "A Short Hike" pushes
            // 576 `LoadInlineData` words that way, its vertex buffer among
            // them, and every one of them was dropped on the floor. Its
            // triangles then came out with all three corners at the same
            // point, and it presented black frames.
            INLINE_FIRST..=INLINE_LAST => self.inline.write(method, arg, ctx)?,
            _ => {
                ctx.stats.inert_methods += 1;
                if ctx.trace {
                    eprintln!("[gpu] inert method={method:#x} arg={arg:#010x}");
                }
            }
        }
        Ok(())
    }

    fn write_macro(
        &mut self,
        method: u32,
        arg: u32,
        last_call: bool,
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        let offset = method - MACRO_METHODS_START;
        let slot = offset >> 1;
        if offset & 1 == 0 {
            self.macros.start(slot, arg);
        } else {
            self.macros.push_argument(arg);
        }
        if last_call {
            // The macro's writes are applied to the class as they are emitted,
            // so a later `read` in the macro sees their effect (e.g. the
            // `WriteHardwareReg` firmware-call poll reads `MmeFirmwareArgs[0]`
            // back after the firmware call writes it). Take the engine out so
            // the mutable borrow of `macros` doesn't conflict with `self.write`.
            let mut macros = std::mem::take(&mut self.macros);
            struct Host<'e, 'c, 'x> {
                engine: &'e mut Engine3D,
                ctx: &'c mut ExecCtx<'x>,
            }
            impl<'e, 'c, 'x> MacroHost for Host<'e, 'c, 'x> {
                fn read_method(&self, method: u32) -> u32 {
                    self.engine.regs.get(method)
                }
                fn write_method(&mut self, write: MacroWrite) -> Result<()> {
                    if self.ctx.trace {
                        eprintln!("[gpu] mme method={:#05x} arg={:#010x}", write.method, write.arg);
                    }
                    self.engine.write(write.method, write.arg, true, self.ctx)
                }
            }
            let mut host = Host { engine: self, ctx };
            let result = macros.run(&mut host);
            self.macros = macros;
            ctx.stats.macros += 1;
            result?;
        }
        Ok(())
    }

    fn syncpt_action(&mut self, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        let id = field(arg, 0, 11);
        if field(arg, 20, 20) != 0 {
            ctx.host1x.increment(id)?;
        }
        Ok(())
    }

    /// `SetReportSemaphore`: the 3D class's fence release.
    fn report_semaphore(&mut self, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        const OPERATION_RELEASE: u32 = 0;
        const STRUCTURE_ONE_WORD: u32 = 1;
        let operation = field(arg, 0, 1);
        if operation != OPERATION_RELEASE {
            // Acquire/counter/trap: work is already retired by the time the
            // guest sees this, so there is nothing to wait for.
            return Ok(());
        }
        let addr = self.regs.iova(REPORT_SEMAPHORE_OFFSET);
        let payload = self.regs.get(REPORT_SEMAPHORE_PAYLOAD);
        if field(arg, 28, 28) == STRUCTURE_ONE_WORD {
            ctx.write_u32(addr, payload)?;
        } else {
            ctx.write_u64(addr, payload as u64)?;
            ctx.write_u64(addr + 8, ctx.stats.submissions)?;
        }
        Ok(())
    }

    /// `FirmwareCall[n]`: the Falcon firmware method-call interface. Writing
    /// one of these starts a firmware routine that reads its arguments from the
    /// `MmeFirmwareArgs` registers and writes a completion code back to
    /// `MmeFirmwareArgs[0]`. deko3d's `WriteHardwareReg` macro fires
    /// `FirmwareCall[4]` to write a PGRAPH register and then polls
    /// `MmeFirmwareArgs[0]` until it reads 1.
    ///
    /// There is no firmware and PGRAPH MMIO is not modelled, so the actual
    /// register poke is a no-op; just acknowledge completion so the macro's
    /// poll loop terminates.
    fn firmware_call(&mut self, _arg: u32, _ctx: &mut ExecCtx) -> Result<()> {
        self.regs.set(MME_FIRMWARE_ARGS, 1);
        Ok(())
    }

    /// `LoadConstbufData`: stream data into the selected constant buffer.
    fn load_constbuf(&mut self, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        let addr = self.regs.iova(CONSTBUF_SELECTOR_ADDR);
        let size = self.regs.field(CONSTBUF_SELECTOR_SIZE, 0, 16);
        if self.constbuf_cursor + 4 > size {
            return Err(Error::Gpu(format!(
                "3d: constant-buffer upload at {:#x} exceeds its {:#x}-byte size",
                self.constbuf_cursor, size
            )));
        }
        ctx.write_u32(addr + self.constbuf_cursor as u64, arg)?;
        self.constbuf_cursor += 4;
        Ok(())
    }

    /// `Bind[slot]`: on a valid `Constbuf` sub-write, snapshot the
    /// currently-selected constant buffer into `bound_constbufs` under
    /// `(slot, bank index)`; on an invalid one, forget that bank.
    fn bind(&mut self, method: u32, arg: u32) {
        let offset = method - BIND;
        let slot = offset / BIND_STRIDE;
        if offset % BIND_STRIDE != BIND_CONSTBUF_OFFSET {
            return;
        }
        let valid = field(arg, 0, 0) != 0;
        let index = field(arg, 4, 8);
        let Some(entry) = self
            .bound_constbufs
            .get_mut(slot as usize)
            .and_then(|banks| banks.get_mut(index as usize))
        else {
            return;
        };
        *entry = valid.then(|| {
            let addr = self.regs.iova(CONSTBUF_SELECTOR_ADDR);
            let size = self.regs.field(CONSTBUF_SELECTOR_SIZE, 0, 16);
            (addr, size)
        });
    }

    fn draw_arrays(&mut self, count: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.last_draw = DrawCall {
            primitive: self.regs.field(VERTEX_BEGIN_GL, 0, 15),
            first: self.regs.get(0x35D),
            count,
            indexed: false,
            index_format: 0,
        };
        ctx.stats.draws += 1;
        self.trace_reg_diff();
        self.rasterize_or_log(ctx);
        Ok(())
    }

    fn draw_elements(&mut self, count: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.last_draw = DrawCall {
            primitive: self.regs.field(VERTEX_BEGIN_GL, 0, 15),
            first: self.regs.get(0x5F7),
            count,
            indexed: true,
            index_format: self.regs.get(0x5F6),
        };
        ctx.stats.draws += 1;
        self.trace_reg_diff();
        self.rasterize_or_log(ctx);
        Ok(())
    }

    /// Run the real rasterizer for `last_draw`. This is deliberately
    /// non-fatal: the ISA/feature subset it supports is still growing (see
    /// `gpu/shader`'s staging), and content that hits something outside it
    /// — real deko3d/Mesa shaders are far richer than our fixtures — must
    /// keep running exactly as it did before this existed, just without
    /// real pixels for that draw. `TRACE_GPU` surfaces why.
    /// `TRACE_REGS=1`: which registers this draw changed since the previous
    /// one. Two draws that read the same state must draw the same thing, so
    /// when a frame's draws all land on top of each other, this says what the
    /// guest varied — and by omission, what the rasterizer is failing to read.
    fn trace_reg_diff(&mut self) {
        if !crate::env_flag!("TRACE_REGS") {
            return;
        }
        let now: Vec<u32> = (0..REGISTER_COUNT as u32).map(|m| self.regs.get(m)).collect();
        if let Some(prev) = &self.traced_regs {
            let diff: Vec<String> = now
                .iter()
                .enumerate()
                .filter(|(i, v)| prev[*i] != **v)
                .map(|(i, v)| format!("{i:#x}={v:#010x}"))
                .collect();
            eprintln!(
                "[regs] begin={:#010x} {}",
                self.regs.get(VERTEX_BEGIN_GL),
                diff.join(" ")
            );
        }
        self.traced_regs = Some(now);
    }

    fn rasterize_or_log(&mut self, ctx: &mut ExecCtx) {
        if ctx.trace && ctx.stats.draws == 1 {
            self.dump_vertex_input();
        }
        let result = self.with_renderer(ctx, |renderer, engine, ctx| renderer.draw(engine, ctx));
        if let Err(e) = result {
            ctx.stats.draws_skipped += 1;
            if ctx.trace {
                // With the vertex array the draw was going to read: a draw
                // that fails and one that reads an empty buffer look the same
                // on screen, and this is what tells them apart.
                let va = self.vertex_array(0);
                eprintln!(
                    "[gpu] raster: {e} [vtx0 start={:#x} stride={} count={}]",
                    va.start, va.stride, self.last_draw.count
                );
            }
        }
    }

    /// The whole vertex-input state of the first draw: which streams are
    /// bound and which attributes read them. A draw whose attributes all point
    /// at a stream that is off is fed constants, not memory, and an empty
    /// vertex buffer is then the expected state rather than a missing upload.
    fn dump_vertex_input(&self) {
        for i in 0..16 {
            let a = self.vertex_attrib(i);
            if a.size == 0 && !a.is_fixed {
                continue;
            }
            eprintln!(
                "[gpu] attrib{i} buf={} fixed={} off={:#x} size={:#x} ty={}",
                a.buffer_id, a.is_fixed, a.offset, a.size, a.ty
            );
        }
        for i in 0..8 {
            let v = self.vertex_array(i);
            eprintln!(
                "[gpu] stream{i} en={} stride={} start={:#x} limit={:#x}",
                v.enabled, v.stride, v.start, v.limit
            );
        }
    }

    /// Resolve `stage`'s bound program, if `SetProgram[stage].Config.Enable`
    /// is set.
    pub fn program(&self, stage: ShaderStage) -> Option<ProgramBinding> {
        let base = SET_PROGRAM + stage.index() * SET_PROGRAM_STRIDE;
        // `VertexB` is the exception: it is the stage a draw cannot do
        // without, so Maxwell keeps it active whether or not `Config.Enable`
        // is set, and drivers do not bother setting it. "A Short Hike" never
        // writes 0x810 at all -- not from the pushbuffer and not from a macro
        // -- while it writes the Config of every other stage on every draw:
        // 0x800 = 0 (VertexA off), 0x820/0x830/0x840 (tessellation and
        // geometry off), 0x850 = 0x51 (fragment on, type 5). It does write
        // VertexB's *offset* and *register count*, so the program is bound;
        // only the bit is missing. Requiring it rejected the vertex program
        // and every one of the title's 325 draws failed with "draw with no
        // bound vertex program".
        if stage != ShaderStage::VertexB && field(self.regs.get(base), 0, 0) == 0 {
            return None;
        }
        let offset = self.regs.get(base + 1);
        let num_registers = self.regs.get(base + 3);
        let addr = self.regs.iova(SET_PROGRAM_REGION).wrapping_add(u64::from(offset));
        Some(ProgramBinding { addr, num_registers })
    }

    /// Resolve `VertexAttribState[i]`.
    pub fn vertex_attrib(&self, i: u32) -> VertexAttrib {
        let raw = self.regs.get(VERTEX_ATTRIB_STATE + i);
        VertexAttrib {
            buffer_id: field(raw, 0, 4),
            is_fixed: field(raw, 6, 6) != 0,
            offset: field(raw, 7, 20),
            size: field(raw, 21, 26),
            ty: field(raw, 27, 29),
            is_bgra: field(raw, 31, 31) != 0,
        }
    }

    /// Resolve `VertexArray[i]` plus its `VertexArrayLimit[i]`.
    pub fn vertex_array(&self, i: u32) -> VertexArray {
        let base = VERTEX_ARRAY + i * VERTEX_ARRAY_STRIDE;
        let config = self.regs.get(base);
        VertexArray {
            enabled: field(config, 12, 12) != 0,
            stride: field(config, 0, 11),
            start: self.regs.iova(base + 1),
            limit: self.regs.iova(VERTEX_ARRAY_LIMIT + i * 2),
            divisor: self.regs.get(base + 3),
        }
    }

    /// Base of the texture-image-control descriptor pool `SetTexHeaderPool`
    /// points at.
    pub fn tex_header_pool(&self) -> u64 {
        self.regs.iova(SET_TEX_HEADER_POOL)
    }

    /// Base of the texture-sampler-control descriptor pool
    /// `SetTexSamplerPool` points at.
    pub fn tex_sampler_pool(&self) -> u64 {
        self.regs.iova(SET_TEX_SAMPLER_POOL)
    }

    /// The `(addr, size)` of the constant buffer bound to `stage`'s hardware
    /// bank `bank` — see `BIND`'s doc comment.
    pub fn bound_constbuf(&self, stage: ShaderStage, bank: u32) -> Option<(u64, u32)> {
        *self
            .bound_constbufs
            .get(stage.bind_slot() as usize)?
            .get(bank as usize)?
    }

    /// Resolve `IndependentBlend[index]` plus its `ColorBlendEnable[index]`
    /// bit.
    pub fn blend_target(&self, index: u32) -> BlendTarget {
        let enabled = self.regs.get(COLOR_BLEND_ENABLE + index) != 0;
        if self.independent_blend_enabled() {
            let base = INDEPENDENT_BLEND + index * INDEPENDENT_BLEND_STRIDE;
            BlendTarget {
                enabled,
                equation_rgb: self.regs.get(base + 1),
                func_rgb_src: self.regs.get(base + 2),
                func_rgb_dst: self.regs.get(base + 3),
                equation_alpha: self.regs.get(base + 4),
                func_alpha_src: self.regs.get(base + 5),
                func_alpha_dst: self.regs.get(base + 6),
            }
        } else {
            BlendTarget {
                enabled,
                equation_rgb: self.regs.get(BLEND_EQUATION_RGB),
                func_rgb_src: self.regs.get(BLEND_FUNC_SRC_RGB),
                func_rgb_dst: self.regs.get(BLEND_FUNC_DST_RGB),
                equation_alpha: self.regs.get(BLEND_EQUATION_ALPHA),
                func_alpha_src: self.regs.get(BLEND_FUNC_SRC_ALPHA),
                func_alpha_dst: self.regs.get(BLEND_FUNC_DST_ALPHA),
            }
        }
    }

    /// Whether blend targets are independent (`IndependentBlendEnable`) —
    /// when false, every render target uses `blend_target(0)`.
    pub fn independent_blend_enabled(&self) -> bool {
        self.regs.get(INDEPENDENT_BLEND_ENABLE) != 0
    }

    pub fn blend_constant(&self) -> [f32; 4] {
        [
            self.regs.float(BLEND_CONSTANT),
            self.regs.float(BLEND_CONSTANT + 1),
            self.regs.float(BLEND_CONSTANT + 2),
            self.regs.float(BLEND_CONSTANT + 3),
        ]
    }

    pub fn depth_state(&self) -> DepthState {
        DepthState {
            test_enabled: self.regs.get(DEPTH_TEST_ENABLE) != 0,
            write_enabled: self.regs.get(DEPTH_WRITE_ENABLE) != 0,
            func: self.regs.get(DEPTH_TEST_FUNC),
        }
    }

    /// Resolve the depth/stencil render target from the register file,
    /// mirroring [`Self::render_target`].
    pub fn depth_target(&self) -> Result<Option<DepthTarget>> {
        let addr = self.regs.iova(DEPTH_TARGET_ADDR);
        let raw_format = self.regs.get(DEPTH_TARGET_FORMAT);
        if addr == 0 || raw_format == 0 {
            return Ok(None);
        }
        let format = depth_format_layout(raw_format)?;
        let tile_mode = self.regs.get(DEPTH_TARGET_TILE_MODE);
        Ok(Some(DepthTarget {
            addr,
            width: self.regs.get(DEPTH_TARGET_HORIZONTAL),
            height: self.regs.get(DEPTH_TARGET_VERTICAL),
            layout: Layout::BlockLinear { block_height_gobs: 1 << field(tile_mode, 4, 7) },
            format,
        }))
    }

    /// Viewport 0's `(x, y, width, height)` in pixels — the NDC-to-screen
    /// transform's target rectangle. Real Maxwell has up to 16 viewports;
    /// only the first is wired up, matching `clear_rect`'s existing
    /// single-viewport assumption.
    pub fn viewport(&self) -> (f32, f32, f32, f32) {
        (
            self.regs.field(VIEWPORT_BASE, 0, 15) as f32,
            self.regs.field(VIEWPORT_BASE + 1, 0, 15) as f32,
            self.regs.field(VIEWPORT_BASE, 16, 31) as f32,
            self.regs.field(VIEWPORT_BASE + 1, 16, 31) as f32,
        )
    }

    /// Viewport 0's NDC-to-window transform, as the hardware actually holds
    /// it: three scales and three translates, applied per axis.
    ///
    /// **Whether y is flipped is this register's business, not a constant.**
    /// GL's window origin is bottom-left and a render target's row 0 is at
    /// the top, so Mesa hands the default framebuffer a *negative* `scale_y`
    /// — and a user FBO, whose contents are sampled back with the same
    /// convention they were written in, a positive one. JKSV's own capture
    /// shows both: `scale_y = -360` for the 1280x720 window, `+128` for the
    /// 256x256 target it renders a save tile into. Hard-coding the flip drew
    /// every offscreen target upside down, which is how a tile's text came
    /// out mirrored.
    pub fn viewport_transform(&self) -> ViewportTransform {
        let f = |i: u32| f32::from_bits(self.regs.get(VIEWPORT_TRANSFORM_BASE + i));
        let scale = [f(0), f(1), f(2)];
        if scale[0] != 0.0 || scale[1] != 0.0 {
            return ViewportTransform { scale, translate: [f(3), f(4), f(5)] };
        }
        // Nothing has written the transform. A zero scale on both axes maps
        // every vertex to one point, so it cannot be a viewport a guest
        // meant; it is a register file no draw has configured. Fall back to
        // the viewport *rectangle* with the window-system flip, which is
        // what the synthetic fixtures in the rasterizer's tests program.
        let (vx, vy, vw, vh) = self.viewport();
        ViewportTransform {
            scale: [vw / 2.0, -vh / 2.0, 0.5],
            translate: [vx + vw / 2.0, vy + vh / 2.0, 0.5],
        }
    }

    /// Which constant bank a `texs`'s immediate indexes for its texture
    /// handle (`TexCbIndex`).
    ///
    /// Both drivers this emulator sees program it, to different values:
    /// nouveau writes 15, the bank it reserves for driver constants, and
    /// deko3d writes 0. Hard-coding nouveau's answer made every deko3d draw
    /// that samples a texture read a bank deko3d never binds — Checkpoint
    /// lost eighteen draws a frame to "read from unbound constant bank 15".
    pub fn tex_cb_index(&self) -> u8 {
        self.regs.field(TEX_CB_INDEX, 0, 4) as u8
    }

    /// Where the bound index buffer starts.
    /// Replace the backend this engine draws and clears through.
    ///
    /// The one place a GPU backend gets installed — see
    /// [`crate::gpu::renderer`].
    pub fn set_renderer(&mut self, renderer: Box<dyn Renderer>) {
        self.renderer = renderer;
    }

    /// Tell the backend that something outside it is about to read a render
    /// target — see [`Renderer::flush`].
    pub fn flush_renderer(&mut self, ctx: &mut ExecCtx) -> Result<()> {
        self.with_renderer(ctx, |renderer, _, ctx| renderer.flush(ctx))
    }

    /// `gl_InstanceID` for [`Engine3D::last_draw`].
    pub fn instance_id(&self) -> u32 {
        self.instance_id
    }

    pub fn index_array_start(&self) -> u64 {
        self.regs.iova(INDEX_ARRAY_START)
    }

    /// Narrow `rect` — normally a render target's full extent — by whichever
    /// of the two clip rectangles the guest has actually configured.
    ///
    /// Both are skipped when unset rather than treated as empty: a register
    /// file that has never been written would otherwise clip every draw to
    /// nothing, and "no scissor programmed" means "do not clip".
    pub fn apply_scissor(&self, rect: ScissorRect) -> ScissorRect {
        let mut out = rect;
        let screen_w = self.regs.field(SCREEN_SCISSOR_HORIZONTAL, 16, 31);
        let screen_h = self.regs.field(SCREEN_SCISSOR_VERTICAL, 16, 31);
        if screen_w != 0 && screen_h != 0 {
            let x0 = self.regs.field(SCREEN_SCISSOR_HORIZONTAL, 0, 15);
            let y0 = self.regs.field(SCREEN_SCISSOR_VERTICAL, 0, 15);
            out.x0 = out.x0.max(x0);
            out.y0 = out.y0.max(y0);
            out.x1 = out.x1.min(x0 + screen_w);
            out.y1 = out.y1.min(y0 + screen_h);
        }
        if self.regs.get(SCISSOR_BASE) != 0 {
            out.x0 = out.x0.max(self.regs.field(SCISSOR_BASE + 1, 0, 15));
            out.x1 = out.x1.min(self.regs.field(SCISSOR_BASE + 1, 16, 31));
            out.y0 = out.y0.max(self.regs.field(SCISSOR_BASE + 2, 0, 15));
            out.y1 = out.y1.min(self.regs.field(SCISSOR_BASE + 2, 16, 31));
        }
        ScissorRect { x0: out.x0, y0: out.y0, x1: out.x1.max(out.x0), y1: out.y1.max(out.y0) }
    }

    /// Face culling, as `OGL_SET_CULL`/`_FRONT_FACE`/`_CULL_FACE` describe
    /// it. The face constants are the GL ones the hardware inherited.
    pub fn cull_state(&self) -> CullState {
        let face = self.regs.get(OGL_SET_CULL_FACE);
        CullState {
            enabled: field(self.regs.get(OGL_SET_CULL), 0, 0) != 0,
            front_ccw: self.regs.get(OGL_SET_FRONT_FACE) != 0x900,
            cull_front: face == 0x404 || face == 0x408,
            cull_back: face == 0x405 || face == 0x408,
        }
    }

    /// The sample grid the bound targets are laid out on.
    ///
    /// `MultisampleEnable` off means one sample per pixel whatever
    /// `MultisampleMode` still holds — the mode register survives a pass that
    /// turns multisampling off, so reading it alone would keep expanding
    /// coordinates long after the guest stopped asking for it.
    /// How the bound surfaces lay their samples out, and where coverage is
    /// tested inside a pixel.
    ///
    /// **`AntiAliasEnable` does not decide how many texels a pixel owns.**
    /// `MsaaMode` does, and the render- and depth-target registers describe
    /// the expanded surface whether or not the bit is set — Eden reads the
    /// mode alone to size a render target and never reads the enable at all.
    /// Taking the bit as "one sample per pixel" made "A Short Hike"'s
    /// 2560x720 2x1 surface a 2560x720 *pixel* one, so its 1280x720 clear
    /// rect and viewport covered the left half of it and the title's own
    /// resolve blit shrank that to a quarter of the frame.
    ///
    /// What the bit does decide is whether coverage is per sample, which is
    /// [`SampleGrid::per_pixel_coverage`].
    pub fn sample_grid(&self) -> Result<SampleGrid> {
        let mut locations = [0u8; MAX_SAMPLES];
        for (i, byte) in locations.iter_mut().enumerate() {
            let word = self.regs.get(MULTISAMPLE_SAMPLE_LOCATIONS + (i / 4) as u32);
            *byte = (word >> (8 * (i % 4))) as u8;
        }
        let grid = SampleGrid::new(self.regs.get(MULTISAMPLE_MODE), &locations)?;
        if self.regs.get(MULTISAMPLE_ENABLE) == 0 {
            return Ok(grid.per_pixel_coverage());
        }
        Ok(grid)
    }

    /// Which samples a draw is allowed to write, as a bit per sample.
    ///
    /// A register file nothing has written reads as zero, and taking that
    /// literally would discard every fragment of every draw — so an all-zero
    /// mask means "unprogrammed", the same reading `apply_scissor` gives an
    /// unset scissor. Hardware resets this register to all-ones, so no guest
    /// can tell the difference.
    pub fn sample_mask(&self) -> u32 {
        match self.regs.get(MULTISAMPLE_SAMPLE_MASK) {
            0 => u32::MAX,
            mask => mask,
        }
    }

    /// Whether a fragment's alpha turns into coverage (`MultisampleControl`).
    pub fn alpha_to_coverage(&self) -> bool {
        self.regs.bit(MULTISAMPLE_CONTROL, 0)
    }

    /// The physical slot `RenderTargetControl` maps logical colour target
    /// `index` onto.
    ///
    /// A guest may bind its targets in one order and address them in another,
    /// and the two places that resolve a target — a clear and a draw — have to
    /// agree about it. They did not: the clear mapped and the draw did not, so
    /// content that remapped target 0 cleared one surface and drew into
    /// whichever one happened to be bound in slot 0.
    pub fn render_target_slot(&self, index: u32) -> u32 {
        let count = self.regs.field(RENDER_TARGET_CONTROL, 0, 3);
        if index < count {
            self.regs.field(RENDER_TARGET_CONTROL, 4 + index * 3, 6 + index * 3)
        } else {
            index
        }
    }

    /// Which channels of colour target `index` a draw may write.
    ///
    /// Not a detail: "A Short Hike" turns alpha off for a third of its draws
    /// and turns every channel off for one, so a rasterizer that writes all
    /// four regardless overwrites exactly what the title meant to keep — and
    /// alpha is what the display reads a frame's opacity out of.
    pub fn color_mask(&self, index: u32) -> [bool; 4] {
        let slot = if self.regs.get(COLOR_MASK_COMMON) != 0 { 0 } else { index };
        let raw = self.regs.get(COLOR_MASK + slot.min(COLOR_TARGETS - 1));
        [
            field(raw, 0, 0) != 0,
            field(raw, 4, 4) != 0,
            field(raw, 8, 8) != 0,
            field(raw, 12, 12) != 0,
        ]
    }

    /// Resolve colour render target `index` from the register file.
    pub fn render_target(&self, index: u32) -> Result<Option<RenderTarget>> {
        let base = RENDER_TARGET_BASE + index * RENDER_TARGET_STRIDE;
        let addr = self.regs.iova(base);
        let raw_format = self.regs.get(base + 4);
        if addr == 0 || raw_format == 0 {
            return Ok(None);
        }
        // The colour and depth format enums are disjoint, so a depth format
        // here is not an unrecognised colour one: the guest has bound a depth
        // surface where a colour target goes. Just Dance 2017 does it for
        // every depth-only pass it runs — the same Z24S8 surface as both
        // colour target 0 and the Z target — and then clears it with the
        // colour channels enabled. There is no colour in a depth surface to
        // write, so report nothing bound rather than invent an encoding for
        // it; a clear then does nothing instead of failing, which matters
        // because a clear that fails takes the channel, and the title, with
        // it.
        if depth_format_layout(raw_format).is_ok() {
            return Ok(None);
        }
        let format = ColorFormat::from_raw(raw_format)?;
        let tile_mode = self.regs.get(base + 5);
        let horizontal = self.regs.get(base + 2);
        let vertical = self.regs.get(base + 3);
        let is_linear = tile_mode >> 12 & 1 != 0;
        let (layout, width) = if is_linear {
            (Layout::Pitch { pitch: horizontal }, horizontal / format.bytes_per_pixel.max(1))
        } else {
            let block_width_gobs = field(tile_mode, 0, 3);
            if block_width_gobs != 0 {
                return Err(Error::Gpu(format!(
                    "3d: render target {} uses a {}-GOB-wide block, which Maxwell does not have",
                    index,
                    1 << block_width_gobs
                )));
            }
            let block_height_gobs = 1 << field(tile_mode, 4, 7);
            (Layout::BlockLinear { block_height_gobs }, horizontal)
        };
        Ok(Some(RenderTarget {
            addr,
            width,
            height: vertical,
            format,
            layout,
            layers: self.regs.field(base + 6, 0, 15).max(1),
            layer_stride: self.regs.get(base + 7),
        }))
    }

    /// The rectangle a clear covers: the screen scissor, further clipped by
    /// the scissor and viewport when `ClearBufferFlags` asks for it.
    ///
    /// Everything here is in **pixels**, so `width`/`height` must be the
    /// target's pixel extent rather than the texel extent its registers hold.
    /// The two differ on a multisampled target, and passing the texel extent
    /// leaves the clear covering only the top-left `1/samples_x` by
    /// `1/samples_y` of it.
    fn clear_rect(&self, width: u32, height: u32) -> (u32, u32, u32, u32) {
        let mut x0 = self.regs.field(SCREEN_SCISSOR_HORIZONTAL, 0, 15);
        let mut y0 = self.regs.field(SCREEN_SCISSOR_VERTICAL, 0, 15);
        let mut x1 = x0 + self.regs.field(SCREEN_SCISSOR_HORIZONTAL, 16, 31);
        let mut y1 = y0 + self.regs.field(SCREEN_SCISSOR_VERTICAL, 16, 31);

        let flags = self.regs.get(CLEAR_BUFFER_FLAGS);
        if field(flags, 8, 8) != 0 && self.regs.get(SCISSOR_BASE) != 0 {
            x0 = x0.max(self.regs.field(SCISSOR_BASE + 1, 0, 15));
            x1 = x1.min(self.regs.field(SCISSOR_BASE + 1, 16, 31));
            y0 = y0.max(self.regs.field(SCISSOR_BASE + 2, 0, 15));
            y1 = y1.min(self.regs.field(SCISSOR_BASE + 2, 16, 31));
        }
        if field(flags, 12, 12) != 0 {
            let vx = self.regs.field(VIEWPORT_BASE, 0, 15);
            let vy = self.regs.field(VIEWPORT_BASE + 1, 0, 15);
            x0 = x0.max(vx);
            x1 = x1.min(vx + self.regs.field(VIEWPORT_BASE, 16, 31));
            y0 = y0.max(vy);
            y1 = y1.min(vy + self.regs.field(VIEWPORT_BASE + 1, 16, 31));
        }
        // An unprogrammed screen scissor means "the whole target".
        if x1 <= x0 {
            x0 = 0;
            x1 = width;
        }
        if y1 <= y0 {
            y0 = 0;
            y1 = height;
        }
        (x0, y0, x1.min(width), y1.min(height))
    }

    /// Run `f` against the bound renderer.
    ///
    /// The renderer is taken out of the engine for the call and put back
    /// after, because it needs `&mut` while the backend needs `&Engine3D` to
    /// read the draw's state from — the same swap `write_macro` does with the
    /// macro engine, and for the same reason.
    fn with_renderer<T>(
        &mut self,
        ctx: &mut ExecCtx,
        f: impl FnOnce(&mut dyn Renderer, &Engine3D, &mut ExecCtx) -> T,
    ) -> T {
        let mut renderer = std::mem::replace(&mut self.renderer, Box::new(Software));
        let out = f(renderer.as_mut(), self, ctx);
        self.renderer = renderer;
        out
    }

    fn clear_buffers(&mut self, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        ctx.stats.clears += 1;
        let clear_depth = field(arg, 0, 0) != 0;
        let clear_stencil = field(arg, 1, 1) != 0;
        let channels = [
            field(arg, 2, 2) != 0,
            field(arg, 3, 3) != 0,
            field(arg, 4, 4) != 0,
            field(arg, 5, 5) != 0,
        ];
        let target = field(arg, 6, 9);
        let layer = field(arg, 10, 20);

        if channels.iter().any(|&c| c) {
            if ctx.trace {
                let colour = [
                    self.regs.float(CLEAR_COLOR),
                    self.regs.float(CLEAR_COLOR + 1),
                    self.regs.float(CLEAR_COLOR + 2),
                    self.regs.float(CLEAR_COLOR + 3),
                ];
                match self.render_target(target) {
                    Ok(Some(rt)) => eprintln!(
                        "[gpu] clear target={target} addr={:#x} {}x{} texels colour={colour:?}",
                        rt.addr, rt.width, rt.height
                    ),
                    other => eprintln!("[gpu] clear target={target} -> {other:x?}"),
                }
            }
            self.with_renderer(ctx, |renderer, engine, ctx| {
                renderer.clear_color(engine, ctx, target, layer, channels)
            })?;
        }
        if clear_depth || clear_stencil {
            self.with_renderer(ctx, |renderer, engine, ctx| {
                renderer.clear_depth_stencil(engine, ctx, clear_depth, clear_stencil)
            })?;
        }
        Ok(())
    }

    pub(crate) fn clear_color(
        &self,
        target: u32,
        layer: u32,
        channels: [bool; 4],
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        let slot = self.render_target_slot(target);
        let rt = match self.render_target(slot)? {
            Some(rt) => rt,
            None => return Ok(()),
        };
        let color = [
            self.regs.float(CLEAR_COLOR),
            self.regs.float(CLEAR_COLOR + 1),
            self.regs.float(CLEAR_COLOR + 2),
            self.regs.float(CLEAR_COLOR + 3),
        ];
        let raw = rt.format.encode(color)?;
        let bpp = rt.format.bytes_per_pixel;
        let all_channels = channels.iter().all(|&c| c);
        let base = rt.addr + (layer as u64) * (rt.layer_stride as u64) * 4;
        let grid = self.sample_grid()?;
        let (width, height) = grid.pixels(rt.width, rt.height);
        let (x0, y0, x1, y1) = self.clear_rect(width, height);
        if ctx.trace {
            eprintln!(
                "[gpu] clear color rt{} {:#x} {width}x{height}px {}x{} samples fmt={:#x} \
                 rect=({x0},{y0})..({x1},{y1}) rgba={color:?}",
                slot, rt.addr, grid.samples_x, grid.samples_y, rt.format.raw
            );
        }
        // Every sample of a pixel is a texel of that pixel's own tile, and
        // `SampleGrid` hands out `samples_x * samples_y` distinct ones — so
        // clearing whole pixels *is* clearing the texel rectangle they cover.
        // Written that way it is runs of contiguous texels rather than a
        // swizzle and an address translation each, which for a 720p target at
        // 2x2 samples is 3.7 million of them per attachment per frame.
        if all_channels {
            let (tx0, ty0) = (x0 * grid.samples_x, y0 * grid.samples_y);
            let (tx1, ty1) = (x1 * grid.samples_x, y1 * grid.samples_y);
            // A GOB is 512 *contiguous* bytes -- `gob_offset` is a bijection
            // from its 64x8 bytes onto them -- and a clear writes one value to
            // every texel in it. So a GOB that lies entirely inside the
            // rectangle is one fill of 512 bytes rather than 128 swizzled
            // writes, and a 2560x1440 target is 28,800 of them instead of 3.7
            // million. Only the GOBs on the edges of a partial rectangle need
            // the per-texel walk.
            let gob_texels = GOB_WIDTH / bpp;
            let whole_gobs = matches!(rt.layout, Layout::BlockLinear { .. }) && gob_texels > 0;
            let mut ty = ty0;
            while ty < ty1 {
                let gob_row = ty - ty % GOB_HEIGHT;
                let row_whole = whole_gobs && ty == gob_row && ty + GOB_HEIGHT <= ty1;
                let mut tx = tx0;
                while tx < tx1 {
                    let gob_col = tx - tx % gob_texels;
                    if row_whole && tx == gob_col && tx + gob_texels <= tx1 {
                        let (offset, _) = rt.texel_run(tx, ty);
                        ctx.fill_pixels(base + offset as u64, bpp, raw, GOB_SIZE / bpp)?;
                        tx += gob_texels;
                        continue;
                    }
                    let (offset, run) = rt.texel_run(tx, ty);
                    let count = run.min(tx1 - tx);
                    ctx.fill_pixels(base + offset as u64, bpp, raw, count)?;
                    tx += count;
                }
                ty += if row_whole { GOB_HEIGHT } else { 1 };
            }
            return Ok(());
        }

        for y in y0..y1 {
            for x in x0..x1 {
                for sample in 0..grid.count() {
                    let (tx, ty) = grid.texel(x, y, sample);
                    let va = base + rt.texel_offset(tx, ty) as u64;
                    {
                        let old = rt.format.decode(ctx.read_pixel(va, bpp)?)?;
                        let mut merged = old;
                        for (i, &enabled) in channels.iter().enumerate() {
                            if enabled {
                                merged[i] = color[i];
                            }
                        }
                        ctx.write_pixel(va, bpp, rt.format.encode(merged)?)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn clear_depth_stencil(
        &self,
        clear_depth: bool,
        clear_stencil: bool,
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        let addr = self.regs.iova(DEPTH_TARGET_ADDR);
        let raw_format = self.regs.get(DEPTH_TARGET_FORMAT);
        if addr == 0 || raw_format == 0 {
            return Ok(());
        }
        let format = depth_format_layout(raw_format)?;
        let bytes = format.bytes;
        let texels_x = self.regs.get(DEPTH_TARGET_HORIZONTAL);
        let texels_y = self.regs.get(DEPTH_TARGET_VERTICAL);
        let tile_mode = self.regs.get(DEPTH_TARGET_TILE_MODE);
        let layout = Layout::BlockLinear { block_height_gobs: 1 << field(tile_mode, 4, 7) };
        let depth = self.regs.float(CLEAR_DEPTH).clamp(0.0, 1.0);
        let stencil = self.regs.get(CLEAR_STENCIL) & 0xFF;
        let grid = self.sample_grid()?;
        let (width, height) = grid.pixels(texels_x, texels_y);
        let (x0, y0, x1, y1) = self.clear_rect(width, height);
        let width_bytes = texels_x * bytes;
        if ctx.trace {
            eprintln!(
                "[gpu] clear depth {addr:#x} {width}x{height}px {}x{} samples fmt={raw_format:#x} \
                 rect=({x0},{y0})..({x1},{y1}) depth={clear_depth}/{depth} stencil={clear_stencil}/{stencil}",
                grid.samples_x, grid.samples_y
            );
        }

        // Which bits this clear owns, and what it puts in them. Both are the
        // same for every texel, so the whole clear is one masked value —
        // which is what lets it go run by run instead of texel by texel.
        let mut written = 0u128;
        let mut value = 0u128;
        if clear_depth {
            let mask = format.depth_mask();
            written |= mask;
            value |= format.encode_depth(depth) & mask;
        }
        if clear_stencil {
            if let Some(shift) = format.stencil_shift {
                written |= 0xFFu128 << shift;
                value |= u128::from(stencil) << shift;
            }
        }
        if written == 0 {
            return Ok(());
        }

        // The same GOB walk [`Engine3D::clear_color`] does, and for the same
        // reason: a GOB is 512 contiguous bytes holding 128 texels in some
        // permuted order, and this applies one mask and one value to all of
        // them — an operation that does not care what the order is. Just Dance
        // 2019 clears depth twice a frame and draws nothing, so this was the
        // whole cost of its frame.
        let (tx0, ty0) = (x0 * grid.samples_x, y0 * grid.samples_y);
        let (tx1, ty1) = (x1 * grid.samples_x, y1 * grid.samples_y);
        let gob_texels = GOB_WIDTH / bytes;
        let mut ty = ty0;
        while ty < ty1 {
            let gob_row = ty - ty % GOB_HEIGHT;
            let row_whole = gob_texels > 0 && ty == gob_row && ty + GOB_HEIGHT <= ty1;
            let mut tx = tx0;
            while tx < tx1 {
                let gob_col = if gob_texels > 0 { tx - tx % gob_texels } else { tx };
                if row_whole && tx == gob_col && tx + gob_texels <= tx1 {
                    let (offset, _) = layout.run_at(tx * bytes, ty, width_bytes);
                    let va = addr + offset as u64;
                    ctx.merge_pixels(va, bytes, value, written, GOB_SIZE / bytes)?;
                    tx += gob_texels;
                    continue;
                }
                let (offset, run) = layout.run_at(tx * bytes, ty, width_bytes);
                let count = (run / bytes).max(1).min(tx1 - tx);
                ctx.merge_pixels(addr + offset as u64, bytes, value, written, count)?;
                tx += count;
            }
            ty += if row_whole { GOB_HEIGHT } else { 1 };
        }
        Ok(())
    }
}

/// How a depth format packs depth and stencil into one pixel.
///
/// NVIDIA names the fields most significant first, so `Z24S8` keeps its
/// stencil in the *low* byte and `S8Z24` in the high one — the opposite of
/// how the names read. Mesa's nv50 format table is what settles it: it maps
/// `PIPE_FORMAT_Z24_UNORM_S8_UINT`, whose depth is in the low 24 bits, onto
/// NVIDIA's `S8Z24`, and `PIPE_FORMAT_S8_UINT_Z24_UNORM` onto `Z24S8`. The
/// table below had those two the wrong way round, which wrote Just Dance
/// 2017's depth into its stencil byte and its stencil into the top of its
/// depth.
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
        let width: u128 = match self.depth_bits {
            0 => 0xFFFF_FFFF,
            bits => (1u128 << bits) - 1,
        };
        width << self.depth_shift
    }

    /// `depth`, in `0.0..=1.0`, encoded where the pixel keeps it.
    ///
    /// The rounding is done in `f64`. In `f32` it cannot be: 24 bits of depth
    /// scale to 16777215, and `16777215.0 + 0.5` is not representable as an
    /// `f32` — it rounds up to 16777216, one bit past the field, so a depth
    /// buffer cleared to 1.0 came back cleared to 0 and every depth test
    /// against it failed.
    pub fn encode_depth(&self, depth: f32) -> u128 {
        let stored = match self.depth_bits {
            0 => depth.to_bits() as u128,
            bits => {
                let max = ((1u64 << bits) - 1) as f64;
                (depth.clamp(0.0, 1.0) as f64 * max + 0.5) as u128
            }
        };
        stored << self.depth_shift
    }

    /// Inverse of [`DepthLayout::encode_depth`], from a whole stored pixel.
    pub fn decode_depth(&self, pixel: u128) -> f32 {
        let stored = (pixel & self.depth_mask()) >> self.depth_shift;
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

    /// Whether depth shares its pixel with a stencil byte, and so whether
    /// writing depth has to read the pixel back rather than overwrite it.
    pub fn packs_stencil(&self) -> bool {
        self.stencil_shift.is_some()
    }
}

/// The [`DepthLayout`] of a `SET_ZT_FORMAT` value, named as
/// `NVB197_SET_ZT_FORMAT_V` names it.
pub(crate) fn depth_format_layout(raw: u32) -> Result<DepthLayout> {
    // `S8` has no depth field at all; its depth columns are never read,
    // because a guest that clears depth into a stencil-only surface is
    // asking for something that does not exist.
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
    Ok(DepthLayout { bytes, depth_bits, depth_shift, stencil_shift })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::exec::GpuStats;
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
    use crate::mem::Memory;

    struct Harness {
        mem: Memory,
        vmm: AddressSpace,
        host1x: Host1x,
        stats: GpuStats,
        base: u64,
    }

    impl Harness {
        fn new(size: u32) -> Harness {
            let mut mem = Memory::new();
            mem.map_zero(0x3000_0000, size as usize).unwrap();
            let mut vmm = AddressSpace::new();
            let base = vmm
                .map(0x3000_0000, size as u64, 1, 0, SMALL_PAGE_SIZE, 0, 0)
                .unwrap();
            Harness { mem, vmm, host1x: Host1x::new(), stats: GpuStats::default(), base }
        }

        fn ctx(&mut self) -> ExecCtx<'_> {
            ExecCtx {
                mem: &mut self.mem,
                vmm: &self.vmm,
                host1x: &mut self.host1x,
                stats: &mut self.stats,
                trace: false,
            }
        }
    }

    #[derive(Debug, Default, PartialEq, Eq)]
    struct Recorded {
        draws: u32,
        color_clears: Vec<(u32, u32, [bool; 4])>,
        depth_clears: Vec<(bool, bool)>,
    }

    /// A renderer that records what it was asked to do and writes nothing.
    ///
    /// Its whole job is to prove the seam is real: if the engine produced a
    /// single pixel without going through the bound renderer, the target in
    /// the test below would not still be zero. It shares its log rather than
    /// being read back out of the engine, which owns it once installed.
    #[derive(Debug, Default)]
    struct Recorder(std::rc::Rc<std::cell::RefCell<Recorded>>);

    impl Renderer for Recorder {
        fn draw(&mut self, _: &Engine3D, _: &mut ExecCtx) -> Result<()> {
            self.0.borrow_mut().draws += 1;
            Ok(())
        }

        fn clear_color(
            &mut self,
            _: &Engine3D,
            _: &mut ExecCtx,
            target: u32,
            layer: u32,
            channels: [bool; 4],
        ) -> Result<()> {
            self.0.borrow_mut().color_clears.push((target, layer, channels));
            Ok(())
        }

        fn clear_depth_stencil(
            &mut self,
            _: &Engine3D,
            _: &mut ExecCtx,
            depth: bool,
            stencil: bool,
        ) -> Result<()> {
            self.0.borrow_mut().depth_clears.push((depth, stencil));
            Ok(())
        }
    }

    /// Program a 16x8 pitch-linear RGBA8 render target.
    fn setup_pitch_target(engine: &mut Engine3D, base: u64, width: u32, height: u32) {
        engine.regs.set(0x200, (base >> 32) as u32);
        engine.regs.set(0x201, base as u32);
        engine.regs.set(0x202, width * 4); // pitch in bytes
        engine.regs.set(0x203, height);
        engine.regs.set(0x204, 0xD5); // RGBA8Unorm
        engine.regs.set(0x205, 1 << 12); // IsLinear
        engine.regs.set(0x206, 1);
        engine.regs.set(SCREEN_SCISSOR_HORIZONTAL, width << 16);
        engine.regs.set(SCREEN_SCISSOR_VERTICAL, height << 16);
    }

    /// Program a single-pixel block-linear depth target in `format`.
    fn setup_depth_target(engine: &mut Engine3D, base: u64, format: u32) {
        engine.regs.set(DEPTH_TARGET_ADDR, (base >> 32) as u32);
        engine.regs.set(DEPTH_TARGET_ADDR + 1, base as u32);
        engine.regs.set(DEPTH_TARGET_FORMAT, format);
        engine.regs.set(DEPTH_TARGET_TILE_MODE, 0);
        engine.regs.set(DEPTH_TARGET_HORIZONTAL, 1);
        engine.regs.set(DEPTH_TARGET_VERTICAL, 1);
        engine.regs.set(SCREEN_SCISSOR_HORIZONTAL, 1 << 16);
        engine.regs.set(SCREEN_SCISSOR_VERTICAL, 1 << 16);
    }

    #[test]
    fn every_pixel_the_engine_produces_goes_through_the_bound_renderer() {
        // The seam is only worth anything if nothing gets past it: a GPU
        // backend whose surfaces live on the GPU cannot have half a frame
        // written into guest memory behind its back.
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        let log = std::rc::Rc::new(std::cell::RefCell::new(Recorded::default()));
        engine.set_renderer(Box::new(Recorder(log.clone())));
        setup_pitch_target(&mut engine, h.base, 16, 8);
        {
            let mut ctx = h.ctx();
            // A colour clear of all four channels, a depth and stencil clear,
            // and a draw — every way this engine makes pixels.
            engine.write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx).unwrap();
            engine.write(CLEAR_BUFFERS, 0b11, true, &mut ctx).unwrap();
            engine.write(VERTEX_BEGIN_GL, 4, true, &mut ctx).unwrap();
            engine.write(DRAW_ARRAYS_COUNT, 3, true, &mut ctx).unwrap();
        }

        let log = log.borrow();
        assert_eq!(log.draws, 1, "the draw reached the renderer");
        assert_eq!(log.color_clears, vec![(0, 0, [true; 4])], "and the colour clear");
        assert_eq!(log.depth_clears, vec![(true, true)], "and the depth clear");

        // And the render target is untouched, because the recorder wrote
        // nothing. `clear_fills_a_pitch_render_target` runs the same clear
        // through the software renderer and finds it filled, which is what
        // makes this assertion mean something.
        let target = h.mem.dump(0x3000_0000, 16 * 8 * 4).unwrap();
        assert!(target.iter().all(|&b| b == 0), "no pixel was written past the renderer");
    }

    #[test]
    fn clear_fills_a_pitch_render_target() {
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        setup_pitch_target(&mut engine, h.base, 16, 8);
        engine.regs.set(CLEAR_COLOR, 1.0f32.to_bits());
        engine.regs.set(CLEAR_COLOR + 1, 0.0f32.to_bits());
        engine.regs.set(CLEAR_COLOR + 2, 0.0f32.to_bits());
        engine.regs.set(CLEAR_COLOR + 3, 1.0f32.to_bits());

        let mut ctx = h.ctx();
        // Clear all four colour channels of target 0.
        engine.write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx).unwrap();

        assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0xFF00_00FF);
        assert_eq!(h.mem.read_u32(0x3000_0000 + 15 * 4).unwrap(), 0xFF00_00FF);
        assert_eq!(h.mem.read_u32(0x3000_0000 + 7 * 64).unwrap(), 0xFF00_00FF);
        // One past the last row must be untouched.
        assert_eq!(h.mem.read_u32(0x3000_0000 + 8 * 64).unwrap(), 0);
        assert_eq!(h.stats.clears, 1);
    }

    #[test]
    fn clear_of_a_multisampled_target_fills_every_sample() {
        // Just Dance 2019's frame, shrunk: the render target registers hold a
        // 16x8 *texel* surface, the scissor holds the 8x4 *pixel* area, and
        // 4x multisampling is what reconciles them. Clamping the clear against
        // the texel extent covered a quarter of the target and left the rest
        // of the frame black.
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        setup_pitch_target(&mut engine, h.base, 16, 8);
        engine.regs.set(SCREEN_SCISSOR_HORIZONTAL, 8 << 16);
        engine.regs.set(SCREEN_SCISSOR_VERTICAL, 4 << 16);
        engine.regs.set(MULTISAMPLE_ENABLE, 1);
        engine.regs.set(MULTISAMPLE_MODE, 2); // 2x2
        for i in 0..4 {
            engine.regs.set(MULTISAMPLE_SAMPLE_LOCATIONS + i, 0xEAA2_6E26);
        }
        engine.regs.set(CLEAR_COLOR, 1.0f32.to_bits());
        engine.regs.set(CLEAR_COLOR + 1, 1.0f32.to_bits());
        engine.regs.set(CLEAR_COLOR + 2, 1.0f32.to_bits());
        engine.regs.set(CLEAR_COLOR + 3, 1.0f32.to_bits());

        let mut ctx = h.ctx();
        engine.write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx).unwrap();

        // Every one of the 16x8 texels, not just the 8x4 the scissor names.
        for y in 0..8u32 {
            for x in 0..16u32 {
                assert_eq!(
                    h.mem.read_u32(0x3000_0000 + y * 64 + x * 4).unwrap(),
                    0xFFFF_FFFF,
                    "texel ({x}, {y})"
                );
            }
        }
        // One past the last row is still outside the target.
        assert_eq!(h.mem.read_u32(0x3000_0000 + 8 * 64).unwrap(), 0);
    }

    /// `MsaaMode` alone says how many texels a pixel owns; `AntiAliasEnable`
    /// says only where coverage is tested.
    ///
    /// This used to read the other way — the enable bit gating the whole grid
    /// — on the reasoning that a guest might leave a stale mode behind. It
    /// cannot work: the surface registers count texels either way, so
    /// answering "one sample per pixel" turns a multisampled target into a
    /// pixel-space one that many times too wide. That is what made "A Short
    /// Hike" paint a quarter of its frame, the same way Just Dance 2019
    /// painted a quarter of its own before any of this was converted at all.
    #[test]
    fn the_msaa_mode_sizes_the_surface_and_the_enable_bit_only_moves_coverage() {
        let mut engine = Engine3D::new();
        engine.regs.set(MULTISAMPLE_MODE, 2); // 2x2
        let off = engine.sample_grid().unwrap();
        assert_eq!(off.count(), 4, "the surface is multisampled either way");
        assert_eq!((off.samples_x, off.samples_y), (2, 2));
        // With antialiasing off every sample tests the pixel's centre, so a
        // pixel is covered whole or not at all.
        assert!((0..off.count()).all(|s| off.position(s) == [0.5, 0.5]));
        // Its samples still land in texels of their own.
        assert_eq!(off.texel(3, 5, 0), (6, 10));
        assert_ne!(off.texel(3, 5, 3), off.texel(3, 5, 0));

        engine.regs.set(MULTISAMPLE_ENABLE, 1);
        let on = engine.sample_grid().unwrap();
        assert_eq!(on.count(), 4);
        assert!((0..on.count()).any(|s| on.position(s) != [0.5, 0.5]), "coverage is per sample");
    }

    /// A guest that never touches either register is not multisampled.
    #[test]
    fn an_unprogrammed_multisample_mode_is_one_sample_a_pixel() {
        assert!(Engine3D::new().sample_grid().unwrap().is_single());
    }

    #[test]
    fn clear_respects_the_scissor() {
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        setup_pitch_target(&mut engine, h.base, 16, 8);
        engine.regs.set(CLEAR_COLOR + 3, 1.0f32.to_bits());
        engine.regs.set(CLEAR_BUFFER_FLAGS, 1 << 8);
        engine.regs.set(SCISSOR_BASE, 1); // enable
        engine.regs.set(SCISSOR_BASE + 1, 4 | (8 << 16)); // x in [4, 8)
        engine.regs.set(SCISSOR_BASE + 2, 0 | (8 << 16));

        let mut ctx = h.ctx();
        engine.write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx).unwrap();

        assert_eq!(h.mem.read_u32(0x3000_0000 + 3 * 4).unwrap(), 0);
        assert_eq!(h.mem.read_u32(0x3000_0000 + 4 * 4).unwrap(), 0xFF00_0000);
        assert_eq!(h.mem.read_u32(0x3000_0000 + 7 * 4).unwrap(), 0xFF00_0000);
        assert_eq!(h.mem.read_u32(0x3000_0000 + 8 * 4).unwrap(), 0);
    }

    #[test]
    fn clear_with_a_channel_mask_preserves_the_others() {
        let mut h = Harness::new(0x1000);
        h.mem.write_u32(0x3000_0000, 0x1122_3344).unwrap();
        let mut engine = Engine3D::new();
        setup_pitch_target(&mut engine, h.base, 1, 1);
        engine.regs.set(CLEAR_COLOR, 1.0f32.to_bits()); // red = 1.0

        let mut ctx = h.ctx();
        // Only the red channel.
        engine.write(CLEAR_BUFFERS, 0b100, true, &mut ctx).unwrap();

        assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0x1122_33FF);
    }

    #[test]
    fn a_colour_target_holding_a_depth_format_is_not_a_colour_target() {
        // Just Dance 2017 binds one Z24S8 surface as both the Z target and
        // colour target 0, then clears it with the colour channels enabled.
        // The clear has to come back having done nothing: a clear that
        // returns an error takes the channel, and the title, with it.
        let mut h = Harness::new(0x1000);
        h.mem.write_u32(0x3000_0000, 0x1122_3344).unwrap();
        let mut engine = Engine3D::new();
        setup_pitch_target(&mut engine, h.base, 1, 1);
        engine.regs.set(0x204, 0x14); // Z24S8, where a colour format goes
        engine.regs.set(CLEAR_COLOR, 1.0f32.to_bits());
        {
            let mut ctx = h.ctx();
            engine.write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx).unwrap();
        }

        assert_eq!(engine.render_target(0).unwrap(), None);
        assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0x1122_3344);
    }

    #[test]
    fn a_colour_target_in_a_format_that_is_neither_is_still_an_error() {
        // Recognising a depth format is not licence to swallow a value that
        // is not one: that is still the guest, or this model, being wrong.
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        setup_pitch_target(&mut engine, h.base, 1, 1);
        engine.regs.set(0x204, 0x77);
        let mut ctx = h.ctx();
        assert!(engine.write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx).is_err());
    }

    #[test]
    fn z24s8_keeps_its_stencil_low_and_s8z24_keeps_it_high() {
        // The two names are mirror images and read backwards: NVIDIA names
        // the fields most significant first. Clearing depth to 1.0 and
        // stencil to 0xAB says which end each one landed on.
        for (format, expected) in [(0x14u32, 0xFFFF_FFAB_u32), (0x16, 0xABFF_FFFF)] {
            let mut h = Harness::new(0x1000);
            let mut engine = Engine3D::new();
            setup_depth_target(&mut engine, h.base, format);
            engine.regs.set(CLEAR_DEPTH, 1.0f32.to_bits());
            engine.regs.set(CLEAR_STENCIL, 0xAB);
            {
                let mut ctx = h.ctx();
                engine.write(CLEAR_BUFFERS, 0b11, true, &mut ctx).unwrap();
            }

            assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), expected, "format {format:#x}");
        }
    }

    #[test]
    fn writing_depth_leaves_a_packed_stencil_byte_alone() {
        let z24s8 = depth_format_layout(0x14).unwrap();
        assert!(z24s8.packs_stencil());
        assert_eq!(z24s8.with_depth(0x0000_00AB, 1.0), 0xFFFF_FFAB);
        assert!((z24s8.decode_depth(z24s8.with_depth(0x0000_00AB, 0.5)) - 0.5).abs() < 1e-6);

        // Z32Float owns its whole pixel, so a depth write need not read first.
        let zf32 = depth_format_layout(0x0A).unwrap();
        assert!(!zf32.packs_stencil());
        assert_eq!(zf32.decode_depth(zf32.encode_depth(0.25)), 0.25);
    }

    /// The depth clear walks runs of texels rather than swizzling each one,
    /// which is only allowed if it lands in the same place with the same
    /// bits. `Z24S8` clearing depth alone is the case that can go wrong twice:
    /// it owns 24 of each pixel's 32 bits and must leave the stencil byte it
    /// shares with, and a GOB's 128 texels are contiguous only because the
    /// value written to all of them is the same.
    #[test]
    fn a_depth_clear_writes_runs_where_a_per_texel_clear_would() {
        const WIDTH: u32 = 32;
        const HEIGHT: u32 = 16;
        const BYTES: u32 = 4;
        /// The stencil byte this texel starts with, distinct across the
        /// surface so that preserving it is visible rather than lucky.
        fn seeded(tx: u32, ty: u32) -> u32 {
            0xDEAD_0000 | (tx << 8) | ((tx * 7 + ty) & 0xFF)
        }

        // A rectangle covering whole GOBs (16x8 texels at four bytes), and one
        // cutting across them in both directions, so the run path and the
        // per-texel edges it falls back to are both exercised.
        for &(rx, rw, ry, rh) in &[(0u32, WIDTH, 0u32, HEIGHT), (4, 20, 3, 9)] {
            let mut h = Harness::new(0x10000);
            let layout = Layout::BlockLinear { block_height_gobs: 1 };
            let width_bytes = WIDTH * BYTES;
            for ty in 0..HEIGHT {
                for tx in 0..WIDTH {
                    let at = 0x3000_0000 + layout.offset(tx * BYTES, ty, width_bytes);
                    h.mem.write_u32(at, seeded(tx, ty)).unwrap();
                }
            }

            let mut engine = Engine3D::new();
            setup_depth_target(&mut engine, h.base, 0x14);
            engine.regs.set(DEPTH_TARGET_HORIZONTAL, WIDTH);
            engine.regs.set(DEPTH_TARGET_VERTICAL, HEIGHT);
            engine.regs.set(SCREEN_SCISSOR_HORIZONTAL, rx | (rw << 16));
            engine.regs.set(SCREEN_SCISSOR_VERTICAL, ry | (rh << 16));
            engine.regs.set(CLEAR_DEPTH, 0.5f32.to_bits());
            {
                let mut ctx = h.ctx();
                engine.write(CLEAR_BUFFERS, 0b01, true, &mut ctx).unwrap();
            }

            let format = depth_format_layout(0x14).unwrap();
            for ty in 0..HEIGHT {
                for tx in 0..WIDTH {
                    let at = 0x3000_0000 + layout.offset(tx * BYTES, ty, width_bytes);
                    let old = seeded(tx, ty);
                    let inside =
                        (rx..rx + rw).contains(&tx) && (ry..ry + rh).contains(&ty);
                    let want = if inside {
                        format.with_depth(u128::from(old), 0.5) as u32
                    } else {
                        old
                    };
                    assert_eq!(
                        h.mem.read_u32(at).unwrap(),
                        want,
                        "texel ({tx},{ty}) of rect ({rx},{ry}) {rw}x{rh}"
                    );
                    if inside {
                        assert_eq!(want & 0xFF, old & 0xFF, "the stencil byte survived");
                    }
                }
            }
        }
    }

    #[test]
    fn clear_of_a_block_linear_target_uses_the_swizzle() {
        let mut h = Harness::new(0x10000);
        let mut engine = Engine3D::new();
        engine.regs.set(0x200, (h.base >> 32) as u32);
        engine.regs.set(0x201, h.base as u32);
        engine.regs.set(0x202, 16); // 16 pixels wide
        engine.regs.set(0x203, 8);
        engine.regs.set(0x204, 0xD5);
        engine.regs.set(0x205, 0); // block-linear, one GOB per block
        engine.regs.set(SCREEN_SCISSOR_HORIZONTAL, 16 << 16);
        engine.regs.set(SCREEN_SCISSOR_VERTICAL, 8 << 16);
        engine.regs.set(CLEAR_COLOR + 3, 1.0f32.to_bits());

        let mut ctx = h.ctx();
        engine.write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx).unwrap();

        // A 16x8 RGBA8 surface is exactly one GOB; every byte of it is written.
        for i in 0..512u32 / 4 {
            assert_eq!(
                h.mem.read_u32(0x3000_0000 + i * 4).unwrap(),
                0xFF00_0000,
                "word {}",
                i
            );
        }
    }

    #[test]
    fn report_semaphore_release_writes_the_payload() {
        let mut h = Harness::new(0x1000);
        let base = h.base;
        let mut engine = Engine3D::new();
        engine.regs.set(REPORT_SEMAPHORE_OFFSET, (base >> 32) as u32);
        engine.regs.set(REPORT_SEMAPHORE_OFFSET + 1, base as u32);
        engine.regs.set(REPORT_SEMAPHORE_PAYLOAD, 0x1234_5678);

        let mut ctx = h.ctx();
        // Release, one-word structure.
        engine.write(REPORT_SEMAPHORE, 1 << 28, true, &mut ctx).unwrap();
        assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0x1234_5678);
    }

    #[test]
    fn syncpt_action_increments_the_counter() {
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        let mut ctx = h.ctx();
        engine.write(SYNCPT_ACTION, 9 | (1 << 20), true, &mut ctx).unwrap();
        assert_eq!(h.host1x.read(9).unwrap(), 1);
    }

    #[test]
    fn constbuf_upload_walks_the_cursor() {
        let mut h = Harness::new(0x1000);
        let base = h.base;
        let mut engine = Engine3D::new();
        engine.regs.set(CONSTBUF_SELECTOR_SIZE, 0x100);
        engine.regs.set(CONSTBUF_SELECTOR_ADDR, (base >> 32) as u32);
        engine.regs.set(CONSTBUF_SELECTOR_ADDR + 1, base as u32);

        let mut ctx = h.ctx();
        engine.write(LOAD_CONSTBUF_OFFSET, 0, true, &mut ctx).unwrap();
        engine.write(LOAD_CONSTBUF_DATA, 0xAAAA_AAAA, false, &mut ctx).unwrap();
        engine.write(LOAD_CONSTBUF_DATA, 0xBBBB_BBBB, true, &mut ctx).unwrap();

        assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0xAAAA_AAAA);
        assert_eq!(h.mem.read_u32(0x3000_0004).unwrap(), 0xBBBB_BBBB);
    }

    #[test]
    fn constbuf_upload_past_the_end_is_rejected() {
        let mut h = Harness::new(0x1000);
        let base = h.base;
        let mut engine = Engine3D::new();
        engine.regs.set(CONSTBUF_SELECTOR_SIZE, 4);
        engine.regs.set(CONSTBUF_SELECTOR_ADDR, (base >> 32) as u32);
        engine.regs.set(CONSTBUF_SELECTOR_ADDR + 1, base as u32);

        let mut ctx = h.ctx();
        engine.write(LOAD_CONSTBUF_DATA, 1, false, &mut ctx).unwrap();
        assert!(engine.write(LOAD_CONSTBUF_DATA, 2, true, &mut ctx).is_err());
    }

    #[test]
    fn draw_arrays_records_the_call() {
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        let mut ctx = h.ctx();
        engine.write(VERTEX_BEGIN_GL, 4, true, &mut ctx).unwrap(); // Triangles
        engine.write(0x35D, 6, true, &mut ctx).unwrap(); // first
        engine.write(DRAW_ARRAYS_COUNT, 3, true, &mut ctx).unwrap();
        assert_eq!(
            engine.last_draw,
            DrawCall { primitive: 4, first: 6, count: 3, indexed: false, index_format: 0 }
        );
        assert_eq!(h.stats.draws, 1);
    }

    #[test]
    fn instance_next_steps_the_instance_counter_and_a_plain_begin_resets_it() {
        // An instanced draw is one Begin/End pair per instance, every pair
        // after the first carrying `InstanceNext`. The Home Menu draws each
        // of its UI elements as one instance of a unit quad and reads
        // `gl_InstanceID` to find that element's position, so a counter stuck
        // at zero puts all 111 glyphs of a label on the same two pixels.
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        let mut ctx = h.ctx();

        engine.write(VERTEX_BEGIN_GL, 4, true, &mut ctx).unwrap();
        assert_eq!(engine.instance_id(), 0);
        for expected in 1..=3 {
            engine
                .write(VERTEX_BEGIN_GL, 4 | VERTEX_BEGIN_INSTANCE_NEXT, true, &mut ctx)
                .unwrap();
            assert_eq!(engine.instance_id(), expected);
        }

        // The next draw's first instance starts over.
        engine.write(VERTEX_BEGIN_GL, 4, true, &mut ctx).unwrap();
        assert_eq!(engine.instance_id(), 0);
    }

    #[test]
    fn instance_next_does_not_disturb_the_primitive() {
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        let mut ctx = h.ctx();
        engine
            .write(VERTEX_BEGIN_GL, 4 | VERTEX_BEGIN_INSTANCE_NEXT, true, &mut ctx)
            .unwrap();
        engine.write(0x5F7, 0, true, &mut ctx).unwrap();
        engine.write(DRAW_ELEMENTS_COUNT, 6, true, &mut ctx).unwrap();
        assert_eq!(engine.last_draw.primitive, 4, "Triangles, not the raw argument");
    }

    #[test]
    fn the_vertex_b_stage_is_bound_without_its_enable_bit() {
        // VertexB is the stage a draw cannot do without, so Maxwell keeps it
        // active whether or not `Config.Enable` is set. "A Short Hike" never
        // writes 0x810 at all -- not from the pushbuffer and not from a macro
        // -- while it writes every other stage's Config on every draw. It does
        // write VertexB's offset and register count, so the program is bound
        // and only the bit is missing; requiring it rejected the vertex
        // program and every one of the title's 325 draws failed with "draw
        // with no bound vertex program", leaving the frame black.
        let mut h = Harness::new(0x1000);
        let base_addr = h.base;
        let mut engine = Engine3D::new();
        let mut ctx = h.ctx();
        engine.write(SET_PROGRAM_REGION, (base_addr >> 32) as u32, true, &mut ctx).unwrap();
        engine.write(SET_PROGRAM_REGION + 1, base_addr as u32, true, &mut ctx).unwrap();

        // Offset and register count, and no Config write at all.
        let base = SET_PROGRAM + ShaderStage::VertexB.index() * SET_PROGRAM_STRIDE;
        engine.write(base + 1, 0x200, true, &mut ctx).unwrap();
        engine.write(base + 3, 0xd, true, &mut ctx).unwrap();
        assert_eq!(
            engine.program(ShaderStage::VertexB),
            Some(ProgramBinding { addr: base_addr + 0x200, num_registers: 0xd }),
        );

        // Every other stage still needs the bit: VertexA sits right next to
        // VertexB and the title disables it by writing Config = 0, which has
        // to keep meaning "off".
        let a = SET_PROGRAM + ShaderStage::VertexA.index() * SET_PROGRAM_STRIDE;
        engine.write(a, 0, true, &mut ctx).unwrap();
        engine.write(a + 1, 0x300, true, &mut ctx).unwrap();
        assert_eq!(engine.program(ShaderStage::VertexA), None);
        assert_eq!(engine.program(ShaderStage::Geometry), None);
    }

    #[test]
    fn program_binding_resolves_relative_to_its_region() {
        let mut h = Harness::new(0x1000);
        let base_addr = h.base;
        let mut engine = Engine3D::new();
        let mut ctx = h.ctx();
        engine.write(SET_PROGRAM_REGION, (base_addr >> 32) as u32, true, &mut ctx).unwrap();
        engine.write(SET_PROGRAM_REGION + 1, base_addr as u32, true, &mut ctx).unwrap();
        // Fragment (StageId 5) enabled, at +0x100, using 4 registers.
        let base = SET_PROGRAM + 5 * SET_PROGRAM_STRIDE;
        engine.write(base, 1 | (5 << 4), true, &mut ctx).unwrap();
        engine.write(base + 1, 0x100, true, &mut ctx).unwrap();
        engine.write(base + 3, 4, true, &mut ctx).unwrap();

        assert_eq!(
            engine.program(ShaderStage::Fragment),
            Some(ProgramBinding { addr: base_addr + 0x100, num_registers: 4 })
        );
        assert_eq!(engine.program(ShaderStage::Geometry), None);
    }

    #[test]
    fn vertex_attrib_state_decodes_its_bit_fields() {
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        let mut ctx = h.ctx();
        // BufferId=2, IsFixed=0, Offset=0x10, Size=0x1F, Type=3, IsBgra=1.
        let raw = 2 | (0x10 << 7) | (0x1F << 21) | (3 << 27) | (1 << 31);
        engine.write(VERTEX_ATTRIB_STATE + 3, raw, true, &mut ctx).unwrap();

        let attrib = engine.vertex_attrib(3);
        assert_eq!(attrib.buffer_id, 2);
        assert!(!attrib.is_fixed);
        assert_eq!(attrib.offset, 0x10);
        assert_eq!(attrib.size, 0x1F);
        assert_eq!(attrib.ty, 3);
        assert!(attrib.is_bgra);
    }

    #[test]
    fn vertex_array_resolves_start_and_limit() {
        let mut h = Harness::new(0x1000);
        let base_addr = h.base;
        let mut engine = Engine3D::new();
        let mut ctx = h.ctx();
        let base = VERTEX_ARRAY + 2 * VERTEX_ARRAY_STRIDE;
        engine.write(base, 0x20 | (1 << 12), true, &mut ctx).unwrap(); // stride 0x20, enabled
        engine.write(base + 1, (base_addr >> 32) as u32, true, &mut ctx).unwrap();
        engine.write(base + 2, base_addr as u32, true, &mut ctx).unwrap();
        engine.write(base + 3, 5, true, &mut ctx).unwrap(); // divisor
        engine.write(VERTEX_ARRAY_LIMIT + 2 * 2, (base_addr >> 32) as u32, true, &mut ctx).unwrap();
        engine
            .write(VERTEX_ARRAY_LIMIT + 2 * 2 + 1, base_addr as u32 + 0x1000, true, &mut ctx)
            .unwrap();

        let va = engine.vertex_array(2);
        assert!(va.enabled);
        assert_eq!(va.stride, 0x20);
        assert_eq!(va.start, base_addr);
        assert_eq!(va.limit, base_addr + 0x1000);
        assert_eq!(va.divisor, 5);
    }

    #[test]
    fn binding_a_constbuf_snapshots_the_current_selector() {
        let mut h = Harness::new(0x1000);
        let base_addr = h.base;
        let mut engine = Engine3D::new();
        let mut ctx = h.ctx();
        engine.write(CONSTBUF_SELECTOR_SIZE, 0x40, true, &mut ctx).unwrap();
        engine.write(CONSTBUF_SELECTOR_ADDR, (base_addr >> 32) as u32, true, &mut ctx).unwrap();
        engine.write(CONSTBUF_SELECTOR_ADDR + 1, base_addr as u32, true, &mut ctx).unwrap();

        // Fragment's bind slot (4), bank 2, valid.
        let base = BIND + 4 * BIND_STRIDE;
        engine.write(base + BIND_CONSTBUF_OFFSET, 1 | (2 << 4), true, &mut ctx).unwrap();

        assert_eq!(engine.bound_constbuf(ShaderStage::Fragment, 2), Some((base_addr, 0x40)));
        assert_eq!(engine.bound_constbuf(ShaderStage::Fragment, 3), None);
        // Vertex shares Fragment's data source but not its bank slot.
        assert_eq!(engine.bound_constbuf(ShaderStage::VertexB, 2), None);

        // A later selector change must not retroactively affect an already
        // bound bank — binding really does snapshot, not alias.
        engine.write(CONSTBUF_SELECTOR_ADDR + 1, base_addr as u32 + 0x40, true, &mut ctx).unwrap();
        assert_eq!(engine.bound_constbuf(ShaderStage::Fragment, 2), Some((base_addr, 0x40)));

        // Unbinding forgets it.
        engine.write(base + BIND_CONSTBUF_OFFSET, 0 | (2 << 4), true, &mut ctx).unwrap();
        assert_eq!(engine.bound_constbuf(ShaderStage::Fragment, 2), None);
    }

    #[test]
    fn blend_target_uses_the_shared_registers_when_independent_blend_is_off() {
        // `IndependentBlendEnable` defaults to off — this is the common case
        // (a single font/UI shader drawing everything with one blend
        // state), and it's a *different* register block from
        // `IndependentBlend[i]`, not just index 0 of the same array.
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        let mut ctx = h.ctx();
        engine.write(COLOR_BLEND_ENABLE, 1, true, &mut ctx).unwrap();
        engine.write(BLEND_EQUATION_RGB, 1, true, &mut ctx).unwrap(); // Add
        engine.write(BLEND_FUNC_SRC_RGB, 5, true, &mut ctx).unwrap(); // SrcAlpha
        engine.write(BLEND_FUNC_DST_RGB, 6, true, &mut ctx).unwrap(); // InvSrcAlpha
        engine.write(BLEND_EQUATION_ALPHA, 1, true, &mut ctx).unwrap();
        engine.write(BLEND_FUNC_SRC_ALPHA, 2, true, &mut ctx).unwrap(); // One
        engine.write(BLEND_FUNC_DST_ALPHA, 1, true, &mut ctx).unwrap(); // Zero

        assert!(!engine.independent_blend_enabled());
        let bt = engine.blend_target(0);
        assert!(bt.enabled);
        assert_eq!(
            (bt.equation_rgb, bt.func_rgb_src, bt.func_rgb_dst, bt.equation_alpha, bt.func_alpha_src, bt.func_alpha_dst),
            (1, 5, 6, 1, 2, 1)
        );
    }

    #[test]
    fn blend_and_depth_state_resolve_from_registers() {
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        let mut ctx = h.ctx();
        engine.write(COLOR_BLEND_ENABLE + 1, 1, true, &mut ctx).unwrap();
        let base = INDEPENDENT_BLEND + 1 * INDEPENDENT_BLEND_STRIDE;
        engine.write(base + 1, 1, true, &mut ctx).unwrap(); // EquationRgb = FUNC_ADD
        engine.write(base + 2, 1, true, &mut ctx).unwrap(); // FuncRgbSrc = ONE
        engine.write(base + 3, 0, true, &mut ctx).unwrap(); // FuncRgbDst = ZERO
        engine.write(INDEPENDENT_BLEND_ENABLE, 1, true, &mut ctx).unwrap();
        engine.write(DEPTH_TEST_ENABLE, 1, true, &mut ctx).unwrap();
        engine.write(DEPTH_WRITE_ENABLE, 1, true, &mut ctx).unwrap();
        engine.write(DEPTH_TEST_FUNC, 4, true, &mut ctx).unwrap(); // Lequal

        let bt = engine.blend_target(1);
        assert!(bt.enabled);
        assert_eq!((bt.equation_rgb, bt.func_rgb_src, bt.func_rgb_dst), (1, 1, 0));
        assert!(engine.independent_blend_enabled());
        assert_eq!(
            engine.depth_state(),
            DepthState { test_enabled: true, write_enabled: true, func: 4 }
        );
    }
}
