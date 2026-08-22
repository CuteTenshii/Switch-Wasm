//! MAXWELL_B (class 0xB197) — the 3D engine.
//!
//! Register numbers come from the Maxwell class headers deko3d generates
//! (`source/maxwell/engine_3d.def`), so they match the command streams real
//! homebrew emits exactly.

use crate::gpu::engine::{field, Registers};
use crate::gpu::exec::ExecCtx;
use crate::gpu::macro_engine::{MacroEngine, MacroHost, MacroWrite, MACRO_METHODS_START};
use crate::gpu::raster;
use crate::gpu::surface::{ColorFormat, Layout};
use crate::{Error, Result};
use std::collections::HashMap;

// Registers with behaviour attached. Everything else is plain state.
const MME_INSTRUCTION_RAM_POINTER: u32 = 0x045;
const MME_INSTRUCTION_RAM_LOAD: u32 = 0x046;
const MME_START_ADDRESS_RAM_POINTER: u32 = 0x047;
const MME_START_ADDRESS_RAM_LOAD: u32 = 0x048;
const SYNCPT_ACTION: u32 = 0x0B2;
const RENDER_TARGET_BASE: u32 = 0x200;
const RENDER_TARGET_STRIDE: u32 = 0x10;
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
const BIND_LAST: u32 = BIND + 5 * BIND_STRIDE - 1;
const BIND_CONSTBUF_OFFSET: u32 = 0x4;

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
    /// Bytes per pixel (matches [`depth_format_layout`]'s first field).
    pub bytes: u32,
    /// Depth bits (`0` means 32-bit float, matching `depth_format_layout`).
    pub depth_bits: u32,
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
    /// Byte offset of a pixel within the target.
    fn pixel_offset(&self, x: u32, y: u32) -> u32 {
        let bpp = self.format.bytes_per_pixel;
        let width_bytes = self.width * bpp;
        self.layout.offset(x * bpp, y, width_bytes)
    }
}

#[derive(Debug)]
pub struct Engine3D {
    pub regs: Registers,
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
    bound_constbufs: HashMap<(u32, u32), (u64, u32)>,
}

impl Default for Engine3D {
    fn default() -> Self {
        Engine3D::new()
    }
}

impl Engine3D {
    pub fn new() -> Engine3D {
        Engine3D {
            regs: Registers::new(),
            macros: MacroEngine::new(),
            inline: crate::gpu::engine::inline::EngineInline::new(),
            last_draw: DrawCall::default(),
            constbuf_cursor: 0,
            bound_constbufs: HashMap::new(),
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
            VERTEX_BEGIN_GL => self.last_draw.primitive = field(arg, 0, 15),
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
        if valid {
            let addr = self.regs.iova(CONSTBUF_SELECTOR_ADDR);
            let size = self.regs.field(CONSTBUF_SELECTOR_SIZE, 0, 16);
            self.bound_constbufs.insert((slot, index), (addr, size));
        } else {
            self.bound_constbufs.remove(&(slot, index));
        }
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
        self.rasterize_or_log(ctx);
        Ok(())
    }

    /// Run the real rasterizer for `last_draw`. This is deliberately
    /// non-fatal: the ISA/feature subset it supports is still growing (see
    /// `gpu/shader`'s staging), and content that hits something outside it
    /// — real deko3d/Mesa shaders are far richer than our fixtures — must
    /// keep running exactly as it did before this existed, just without
    /// real pixels for that draw. `TRACE_GPU` surfaces why.
    fn rasterize_or_log(&self, ctx: &mut ExecCtx) {
        if ctx.trace && ctx.stats.draws == 1 {
            self.dump_vertex_input();
        }
        if let Err(e) = raster::draw(self, ctx) {
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
        self.bound_constbufs.get(&(stage.bind_slot(), bank)).copied()
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
        let (bytes, depth_bits, _stencil_shift) = depth_format_layout(raw_format)?;
        let tile_mode = self.regs.get(DEPTH_TARGET_TILE_MODE);
        Ok(Some(DepthTarget {
            addr,
            width: self.regs.get(DEPTH_TARGET_HORIZONTAL),
            height: self.regs.get(DEPTH_TARGET_VERTICAL),
            layout: Layout::BlockLinear { block_height_gobs: 1 << field(tile_mode, 4, 7) },
            bytes,
            depth_bits,
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

    /// Where the bound index buffer starts.
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

    /// Resolve colour render target `index` from the register file.
    pub fn render_target(&self, index: u32) -> Result<Option<RenderTarget>> {
        let base = RENDER_TARGET_BASE + index * RENDER_TARGET_STRIDE;
        let addr = self.regs.iova(base);
        let raw_format = self.regs.get(base + 4);
        if addr == 0 || raw_format == 0 {
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
            self.clear_color(target, layer, channels, ctx)?;
        }
        if clear_depth || clear_stencil {
            self.clear_depth_stencil(clear_depth, clear_stencil, ctx)?;
        }
        Ok(())
    }

    fn clear_color(
        &self,
        target: u32,
        layer: u32,
        channels: [bool; 4],
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        // RenderTargetControl maps a logical target id onto a physical slot.
        let num_targets = self.regs.field(RENDER_TARGET_CONTROL, 0, 3);
        let slot = if target < num_targets {
            self.regs.field(RENDER_TARGET_CONTROL, 4 + target * 3, 6 + target * 3)
        } else {
            target
        };
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
        let (x0, y0, x1, y1) = self.clear_rect(rt.width, rt.height);
        if ctx.trace {
            eprintln!(
                "[gpu] clear color rt{} {:#x} {}x{} fmt={:#x} rect=({},{})..({},{}) rgba={:?}",
                slot, rt.addr, rt.width, rt.height, rt.format.raw, x0, y0, x1, y1, color
            );
        }
        for y in y0..y1 {
            for x in x0..x1 {
                let va = base + rt.pixel_offset(x, y) as u64;
                if all_channels {
                    ctx.write_pixel(va, bpp, raw)?;
                } else {
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
        Ok(())
    }

    fn clear_depth_stencil(
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
        let (bytes, depth_bits, stencil_shift) = depth_format_layout(raw_format)?;
        let width = self.regs.get(DEPTH_TARGET_HORIZONTAL);
        let height = self.regs.get(DEPTH_TARGET_VERTICAL);
        let tile_mode = self.regs.get(DEPTH_TARGET_TILE_MODE);
        let layout = Layout::BlockLinear { block_height_gobs: 1 << field(tile_mode, 4, 7) };
        let depth = self.regs.float(CLEAR_DEPTH).clamp(0.0, 1.0);
        let stencil = self.regs.get(CLEAR_STENCIL) & 0xFF;
        let (x0, y0, x1, y1) = self.clear_rect(width, height);
        let width_bytes = width * bytes;

        for y in y0..y1 {
            for x in x0..x1 {
                let va = addr + layout.offset(x * bytes, y, width_bytes) as u64;
                let mut value = ctx.read_pixel(va, bytes)?;
                if clear_depth {
                    let encoded = encode_depth(depth, depth_bits);
                    let mask = depth_mask(depth_bits);
                    value = (value & !mask) | (encoded & mask);
                }
                if clear_stencil {
                    if let Some(shift) = stencil_shift {
                        value = (value & !(0xFFu128 << shift)) | ((stencil as u128) << shift);
                    }
                }
                ctx.write_pixel(va, bytes, value)?;
            }
        }
        Ok(())
    }
}

/// `(bytes per pixel, depth bits, stencil bit offset)` for a depth format.
/// Depth bits of 32 with a float layout is signalled by `depth_bits == 0`.
pub(crate) fn depth_format_layout(raw: u32) -> Result<(u32, u32, Option<u32>)> {
    Ok(match raw {
        0x0A => (4, 0, None),        // Z32Float
        0x13 => (2, 16, None),       // Z16Unorm
        0x14 => (4, 24, Some(24)),   // S8Z24Unorm
        0x15 => (4, 24, None),       // Z24X8Unorm
        0x16 => (4, 24, Some(24)),   // Z24S8Unorm
        0x17 => (1, 0, Some(0)),     // S8Uint
        0x19 => (8, 0, Some(32)),    // Z32S8X24Float
        other => {
            return Err(Error::Gpu(format!(
                "3d: unsupported depth format {:#x}",
                other
            )))
        }
    })
}

pub(crate) fn depth_mask(depth_bits: u32) -> u128 {
    match depth_bits {
        0 => 0xFFFF_FFFF,
        bits => (1u128 << bits) - 1,
    }
}

pub(crate) fn encode_depth(depth: f32, depth_bits: u32) -> u128 {
    match depth_bits {
        0 => depth.to_bits() as u128,
        bits => {
            let max = ((1u64 << bits) - 1) as f32;
            (depth * max + 0.5) as u128
        }
    }
}

/// Inverse of [`encode_depth`]: a stored depth value back to `0.0..=1.0`.
pub(crate) fn decode_depth(raw: u128, depth_bits: u32) -> f32 {
    match depth_bits {
        0 => f32::from_bits(raw as u32),
        bits => {
            let max = ((1u64 << bits) - 1) as f32;
            (raw & depth_mask(bits)) as f32 / max
        }
    }
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
