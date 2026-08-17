//! MAXWELL_B (class 0xB197) — the 3D engine.
//!
//! Register numbers come from the Maxwell class headers deko3d generates
//! (`source/maxwell/engine_3d.def`), so they match the command streams real
//! homebrew emits exactly.

use crate::gpu::engine::{field, Registers};
use crate::gpu::exec::ExecCtx;
use crate::gpu::macro_engine::{MacroEngine, MACRO_METHODS_START};
use crate::gpu::surface::{ColorFormat, Layout};
use crate::{Error, Result};

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

// The 3D class also implements the inline-to-memory methods; deko3d issues
// them on the 3D subchannel (see `gpu_transfer.cpp`).
const INLINE_FIRST: u32 = 0x060;
const INLINE_LAST: u32 = 0x06D;

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
    /// The last draw the engine was asked to perform.
    pub last_draw: DrawCall,
    /// Write cursor for `LoadConstbufData`, in bytes.
    constbuf_cursor: u32,
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
            last_draw: DrawCall::default(),
            constbuf_cursor: 0,
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
            LOAD_CONSTBUF_OFFSET => self.constbuf_cursor = field(arg, 0, 15),
            LOAD_CONSTBUF_DATA..=LOAD_CONSTBUF_DATA_LAST => self.load_constbuf(arg, ctx)?,
            DRAW_ARRAYS_COUNT => self.draw_arrays(arg, ctx)?,
            DRAW_ELEMENTS_COUNT => self.draw_elements(arg, ctx)?,
            VERTEX_BEGIN_GL => self.last_draw.primitive = field(arg, 0, 15),
            VERTEX_END_GL => {}
            INLINE_FIRST..=INLINE_LAST => {
                // Shares the register file with the inline-to-memory class,
                // which the channel drives; nothing extra to do here.
            }
            _ => ctx.stats.inert_methods += 1,
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
            let writes = {
                let regs = &self.regs;
                self.macros.run(|m| regs.get(m))?
            };
            ctx.stats.macros += 1;
            for write in writes {
                // A macro can only emit ordinary method writes, so this
                // recursion is one level deep.
                self.write(write.method, write.arg, true, ctx)?;
            }
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

    fn draw_arrays(&mut self, count: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.last_draw = DrawCall {
            primitive: self.regs.field(VERTEX_BEGIN_GL, 0, 15),
            first: self.regs.get(0x35D),
            count,
            indexed: false,
            index_format: 0,
        };
        ctx.stats.draws += 1;
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
        Ok(())
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
fn depth_format_layout(raw: u32) -> Result<(u32, u32, Option<u32>)> {
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

fn depth_mask(depth_bits: u32) -> u128 {
    match depth_bits {
        0 => 0xFFFF_FFFF,
        bits => (1u128 << bits) - 1,
    }
}

fn encode_depth(depth: f32, depth_bits: u32) -> u128 {
    match depth_bits {
        0 => depth.to_bits() as u128,
        bits => {
            let max = ((1u64 << bits) - 1) as f32;
            (depth * max + 0.5) as u128
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
}
