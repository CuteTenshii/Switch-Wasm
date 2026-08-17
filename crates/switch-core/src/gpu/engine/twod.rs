//! FERMI_TWOD_A (class 0x902D) — the 2D blitter.
//!
//! `PixelsFromMemory` scales a source rectangle into a destination rectangle
//! using 32.32 fixed-point stepping, with point or bilinear sampling. deko3d
//! routes `dkCmdBufBlitImage` here whenever the copy engine cannot express the
//! operation (scaling, format conversion, filtering).

use crate::gpu::engine::Registers;
use crate::gpu::exec::ExecCtx;
use crate::gpu::surface::{ColorFormat, Layout};
use crate::{Error, Result};

const SET_DST_FORMAT: u32 = 0x080;
const SET_DST_MEMORY_LAYOUT: u32 = 0x081;
const SET_DST_BLOCK_SIZE: u32 = 0x082;
const SET_DST_PITCH: u32 = 0x085;
const SET_DST_WIDTH: u32 = 0x086;
const SET_DST_HEIGHT: u32 = 0x087;
const SET_DST_OFFSET: u32 = 0x088;
const SET_SRC_FORMAT: u32 = 0x08C;
const SET_SRC_MEMORY_LAYOUT: u32 = 0x08D;
const SET_SRC_BLOCK_SIZE: u32 = 0x08E;
const SET_SRC_PITCH: u32 = 0x091;
const SET_SRC_WIDTH: u32 = 0x092;
const SET_SRC_HEIGHT: u32 = 0x093;
const SET_SRC_OFFSET: u32 = 0x094;
const SET_OPERATION: u32 = 0x0AB;
const SAMPLE_MODE: u32 = 0x223;
const DST_X0: u32 = 0x22C;
const DST_Y0: u32 = 0x22D;
const DST_WIDTH: u32 = 0x22E;
const DST_HEIGHT: u32 = 0x22F;
const DU_DX_FRAC: u32 = 0x230;
const DU_DX_INT: u32 = 0x231;
const DV_DY_FRAC: u32 = 0x232;
const DV_DY_INT: u32 = 0x233;
const SRC_X0_FRAC: u32 = 0x234;
const SRC_X0_INT: u32 = 0x235;
const SRC_Y0_FRAC: u32 = 0x236;
/// Writing this register triggers the blit.
const SRC_Y0_INT: u32 = 0x237;

const MEMORY_LAYOUT_PITCH: u32 = 1;
const OPERATION_SRC_COPY: u32 = 3;
const FILTER_BILINEAR: u32 = 1;

#[derive(Debug, Default)]
pub struct Engine2D {
    pub regs: Registers,
}

impl Engine2D {
    pub fn new() -> Engine2D {
        Engine2D { regs: Registers::new() }
    }

    pub fn write(&mut self, method: u32, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.regs.set(method, arg);
        if method == SRC_Y0_INT {
            self.blit(ctx)?;
        }
        Ok(())
    }

    fn surface(&self, dst: bool) -> Result<Surface> {
        let (format, layout_reg, block_reg, pitch_reg, width_reg, height_reg, offset_reg) = if dst {
            (
                SET_DST_FORMAT,
                SET_DST_MEMORY_LAYOUT,
                SET_DST_BLOCK_SIZE,
                SET_DST_PITCH,
                SET_DST_WIDTH,
                SET_DST_HEIGHT,
                SET_DST_OFFSET,
            )
        } else {
            (
                SET_SRC_FORMAT,
                SET_SRC_MEMORY_LAYOUT,
                SET_SRC_BLOCK_SIZE,
                SET_SRC_PITCH,
                SET_SRC_WIDTH,
                SET_SRC_HEIGHT,
                SET_SRC_OFFSET,
            )
        };
        let format = ColorFormat::from_raw(self.regs.get(format))?;
        let pitch_linear = self.regs.get(layout_reg) == MEMORY_LAYOUT_PITCH;
        let layout = if pitch_linear {
            Layout::Pitch { pitch: self.regs.get(pitch_reg) }
        } else {
            Layout::BlockLinear {
                block_height_gobs: 1 << self.regs.field(block_reg, 4, 6),
            }
        };
        Ok(Surface {
            addr: self.regs.iova(offset_reg),
            width: self.regs.get(width_reg),
            height: self.regs.get(height_reg),
            format,
            layout,
        })
    }

    fn blit(&mut self, ctx: &mut ExecCtx) -> Result<()> {
        let operation = self.regs.get(SET_OPERATION);
        if operation != OPERATION_SRC_COPY {
            return Err(Error::Gpu(format!(
                "2d: blit operation {} is not implemented (only SrcCopy)",
                operation
            )));
        }
        let src = self.surface(false)?;
        let dst = self.surface(true)?;
        let dst_x0 = self.regs.get(DST_X0);
        let dst_y0 = self.regs.get(DST_Y0);
        let dst_w = self.regs.get(DST_WIDTH);
        let dst_h = self.regs.get(DST_HEIGHT);
        let du_dx = fixed(self.regs.get(DU_DX_INT), self.regs.get(DU_DX_FRAC));
        let dv_dy = fixed(self.regs.get(DV_DY_INT), self.regs.get(DV_DY_FRAC));
        let src_x0 = fixed(self.regs.get(SRC_X0_INT), self.regs.get(SRC_X0_FRAC));
        let src_y0 = fixed(self.regs.get(SRC_Y0_INT), self.regs.get(SRC_Y0_FRAC));
        let bilinear = self.regs.field(SAMPLE_MODE, 4, 4) == FILTER_BILINEAR;

        if ctx.trace {
            eprintln!(
                "[gpu] 2d blit src {:#x} {}x{} fmt={:#x} -> dst {:#x} ({},{}) {}x{} fmt={:#x}",
                src.addr, src.width, src.height, src.format.raw,
                dst.addr, dst_x0, dst_y0, dst_w, dst_h, dst.format.raw
            );
        }

        for y in 0..dst_h {
            let v = src_y0 + dv_dy * y as f64;
            for x in 0..dst_w {
                let u = src_x0 + du_dx * x as f64;
                let color = if bilinear {
                    src.sample_bilinear(u, v, ctx)?
                } else {
                    src.sample_point(u, v, ctx)?
                };
                let va = dst.addr + dst.offset(dst_x0 + x, dst_y0 + y) as u64;
                ctx.write_pixel(va, dst.format.bytes_per_pixel, dst.format.encode(color)?)?;
            }
        }
        ctx.stats.copies += 1;
        Ok(())
    }
}

struct Surface {
    addr: u64,
    width: u32,
    height: u32,
    format: ColorFormat,
    layout: Layout,
}

impl Surface {
    fn offset(&self, x: u32, y: u32) -> u32 {
        let bpp = self.format.bytes_per_pixel;
        let width_bytes = match self.layout {
            Layout::Pitch { pitch } => pitch,
            Layout::BlockLinear { .. } => self.width * bpp,
        };
        self.layout.offset(x * bpp, y, width_bytes)
    }

    fn texel(&self, x: u32, y: u32, ctx: &ExecCtx) -> Result<[f32; 4]> {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        let va = self.addr + self.offset(x, y) as u64;
        self.format.decode(ctx.read_pixel(va, self.format.bytes_per_pixel)?)
    }

    fn sample_point(&self, u: f64, v: f64, ctx: &ExecCtx) -> Result<[f32; 4]> {
        self.texel(u.max(0.0) as u32, v.max(0.0) as u32, ctx)
    }

    fn sample_bilinear(&self, u: f64, v: f64, ctx: &ExecCtx) -> Result<[f32; 4]> {
        let u = (u - 0.5).max(0.0);
        let v = (v - 0.5).max(0.0);
        let x0 = u as u32;
        let y0 = v as u32;
        let fx = (u - x0 as f64) as f32;
        let fy = (v - y0 as f64) as f32;
        let c00 = self.texel(x0, y0, ctx)?;
        let c10 = self.texel(x0 + 1, y0, ctx)?;
        let c01 = self.texel(x0, y0 + 1, ctx)?;
        let c11 = self.texel(x0 + 1, y0 + 1, ctx)?;
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            let top = c00[i] + (c10[i] - c00[i]) * fx;
            let bottom = c01[i] + (c11[i] - c01[i]) * fx;
            out[i] = top + (bottom - top) * fy;
        }
        Ok(out)
    }
}

/// Recombine the 32.32 fixed-point pairs the engine takes.
fn fixed(int_part: u32, frac: u32) -> f64 {
    int_part as i32 as f64 + frac as f64 / 4_294_967_296.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::exec::GpuStats;
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
    use crate::mem::Memory;

    fn set_iova(engine: &mut Engine2D, method: u32, va: u64) {
        engine.regs.set(method, (va >> 32) as u32);
        engine.regs.set(method + 1, va as u32);
    }

    #[test]
    fn one_to_one_blit_copies_pixels() {
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, 0x2000).unwrap();
        for i in 0..16u32 {
            mem.write_u32(0x3000_0000 + i * 4, 0xFF00_0000 | i).unwrap();
        }
        let mut vmm = AddressSpace::new();
        let base = vmm.map(0x3000_0000, 0x2000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
        let mut host1x = Host1x::new();
        let mut stats = GpuStats::default();

        let mut engine = Engine2D::new();
        engine.regs.set(SET_SRC_FORMAT, 0xD5);
        engine.regs.set(SET_SRC_MEMORY_LAYOUT, MEMORY_LAYOUT_PITCH);
        engine.regs.set(SET_SRC_PITCH, 16);
        engine.regs.set(SET_SRC_WIDTH, 4);
        engine.regs.set(SET_SRC_HEIGHT, 4);
        set_iova(&mut engine, SET_SRC_OFFSET, base);

        engine.regs.set(SET_DST_FORMAT, 0xD5);
        engine.regs.set(SET_DST_MEMORY_LAYOUT, MEMORY_LAYOUT_PITCH);
        engine.regs.set(SET_DST_PITCH, 16);
        engine.regs.set(SET_DST_WIDTH, 4);
        engine.regs.set(SET_DST_HEIGHT, 4);
        set_iova(&mut engine, SET_DST_OFFSET, base + 0x1000);

        engine.regs.set(SET_OPERATION, OPERATION_SRC_COPY);
        engine.regs.set(DST_WIDTH, 4);
        engine.regs.set(DST_HEIGHT, 4);
        engine.regs.set(DU_DX_INT, 1);
        engine.regs.set(DV_DY_INT, 1);

        let mut ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        engine.write(SRC_Y0_INT, 0, &mut ctx).unwrap();

        for i in 0..16u32 {
            assert_eq!(
                mem.read_u32(0x3000_1000 + i * 4).unwrap(),
                0xFF00_0000 | i,
                "pixel {}",
                i
            );
        }
        assert_eq!(stats.copies, 1);
    }

    #[test]
    fn fixed_point_conversion() {
        assert_eq!(fixed(1, 0), 1.0);
        assert_eq!(fixed(0, 0x8000_0000), 0.5);
        assert_eq!(fixed(2, 0x8000_0000), 2.5);
    }
}
