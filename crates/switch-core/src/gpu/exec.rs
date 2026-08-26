//! Execution context handed to the engines while a channel's pushbuffer runs.
//!
//! Bundles everything an engine can touch — guest memory through the channel's
//! GPU address space, the host1x syncpoints, and the frame statistics — so the
//! engines never need a reference back to the whole GPU.

use crate::gpu::syncpt::Host1x;
use crate::gpu::vmm::AddressSpace;
use crate::mem::Memory;
use crate::{Error, Result};

/// Counters describing what the GPU has done, for the frontend and for tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuStats {
    /// GPFIFO submissions processed.
    pub submissions: u64,
    /// Method writes dispatched to an engine.
    pub methods: u64,
    /// `ClearBuffers` operations executed.
    pub clears: u64,
    /// Draw calls seen (`VertexBegin`/`DrawArrays`/`DrawElements`).
    pub draws: u64,
    /// Copy-engine and 2D-engine transfers executed.
    pub copies: u64,
    /// Macros executed by the MME.
    pub macros: u64,
    /// Method writes that hit a register with no implemented behaviour. These
    /// are still stored in the register file, so state stays coherent.
    pub inert_methods: u64,
    /// Compute dispatches launched.
    pub dispatches: u64,
    /// Dispatches that did not run — an unparseable QMD, or a kernel using an
    /// instruction the interpreter does not decode. Counted for the same
    /// reason `draws_skipped` is: a kernel that never ran leaves memory
    /// holding whatever was there, and nothing on screen says so.
    pub dispatches_skipped: u64,
    /// Draws the rasterizer refused, almost always because the shader used an
    /// instruction the interpreter does not decode.
    ///
    /// Counted because the symptom otherwise carries no information: the draw
    /// is dropped, the render target keeps whatever was in it, and the frame
    /// is presented black with nothing to say why. `draws` minus this is what
    /// actually reached the framebuffer.
    pub draws_skipped: u64,
}

pub struct ExecCtx<'a> {
    pub mem: &'a mut Memory,
    pub vmm: &'a AddressSpace,
    pub host1x: &'a mut Host1x,
    pub stats: &'a mut GpuStats,
    /// Emit a per-method trace to stderr (`TRACE_GPU`).
    pub trace: bool,
}

impl ExecCtx<'_> {
    pub fn read_u32(&self, gpu_va: u64) -> Result<u32> {
        self.vmm.read_u32(self.mem, gpu_va)
    }

    pub fn read_u64(&self, gpu_va: u64) -> Result<u64> {
        self.vmm.read_u64(self.mem, gpu_va)
    }

    pub fn write_u32(&mut self, gpu_va: u64, value: u32) -> Result<()> {
        self.vmm.write_u32(self.mem, gpu_va, value)
    }

    pub fn write_u64(&mut self, gpu_va: u64, value: u64) -> Result<()> {
        self.vmm.write_u64(self.mem, gpu_va, value)
    }

    /// Read `len` bytes of a surface's raw pixel, little-endian. The GPU VA is
    /// translated once for the whole pixel rather than once per byte: a blit
    /// touches every pixel of a 1280x720 surface, and a per-byte translation
    /// meant millions of address-space lookups per frame.
    pub fn read_pixel(&self, gpu_va: u64, len: u32) -> Result<u128> {
        // One access per machine word rather than per byte, which is
        // `Memory::read_le` — the same walk `Gpu::present` needs, so it lives
        // there rather than here.
        self.mem.read_le(self.pixel_addr(gpu_va, len)?, len)
    }

    /// Write `len` bytes of a surface's raw pixel, little-endian.
    pub fn write_pixel(&mut self, gpu_va: u64, len: u32, value: u128) -> Result<()> {
        let cpu = self.pixel_addr(gpu_va, len)?;
        self.mem.write_le(cpu, len, value)
    }

    /// Write `count` consecutive pixels of `unit` bytes with the same value,
    /// translating the GPU address **once** for the whole run.
    ///
    /// The address translation is what a clear spends itself on. A 720p target
    /// at 2x2 samples is 3.7 million texels, a title that clears three
    /// attachments does that three times a frame, and Just Dance 2019 pays it
    /// on frames that carry no draw at all — so a per-texel translation was
    /// the most expensive thing in a frame with nothing in it.
    ///
    /// A run that would leave its own mapping is not a run: that falls back to
    /// one translation each rather than write past the end of it.
    pub fn fill_pixels(&mut self, gpu_va: u64, unit: u32, value: u128, count: u32) -> Result<()> {
        let bytes = u64::from(unit) * u64::from(count);
        let cpu = match self.vmm.translate(gpu_va) {
            Some((cpu, left)) if left >= bytes => cpu,
            _ => {
                for i in 0..count {
                    self.write_pixel(gpu_va + u64::from(i) * u64::from(unit), unit, value)?;
                }
                return Ok(());
            }
        };
        self.mem.fill_le(cpu, unit, value, count)
    }

    /// [`ExecCtx::fill_pixels`] for a write that owns only the `mask` bits of
    /// each pixel, leaving the rest as it found them.
    ///
    /// What a depth clear needs against a format that packs stencil beside
    /// depth: the value is uniform across the run even though the write is
    /// partial, so the run still costs one translation rather than two per
    /// texel. A mask covering the whole pixel is a fill, and takes that path.
    pub fn merge_pixels(
        &mut self,
        gpu_va: u64,
        unit: u32,
        value: u128,
        mask: u128,
        count: u32,
    ) -> Result<()> {
        let all = if unit >= 16 { u128::MAX } else { (1u128 << (unit * 8)) - 1 };
        if mask & all == all {
            return self.fill_pixels(gpu_va, unit, value, count);
        }
        let bytes = u64::from(unit) * u64::from(count);
        let cpu = match self.vmm.translate(gpu_va) {
            Some((cpu, left)) if left >= bytes => cpu,
            _ => {
                for i in 0..count {
                    let at = gpu_va + u64::from(i) * u64::from(unit);
                    let old = self.read_pixel(at, unit)?;
                    self.write_pixel(at, unit, (old & !mask) | (value & mask))?;
                }
                return Ok(());
            }
        };
        self.mem.merge_le(cpu, unit, value, mask, count)
    }

    /// Where a pixel's `len` bytes live in guest memory. A pixel never spans two
    /// mappings, so one translation covers all of it.
    fn pixel_addr(&self, gpu_va: u64, len: u32) -> Result<u32> {
        match self.vmm.translate(gpu_va) {
            Some((cpu, left)) if left >= u64::from(len) => Ok(cpu),
            _ => Err(Error::Gpu(format!(
                "gpu va {:#x}: {} bytes are not mapped",
                gpu_va, len
            ))),
        }
    }

    pub fn vmm_read_u8(&self, gpu_va: u64) -> Result<u8> {
        match self.vmm.translate(gpu_va) {
            Some((cpu, _)) => self.mem.read_u8(cpu),
            None => Err(Error::Gpu(format!("gpu va {:#x} is not mapped", gpu_va))),
        }
    }

    pub fn vmm_write_u8(&mut self, gpu_va: u64, value: u8) -> Result<()> {
        match self.vmm.translate(gpu_va) {
            Some((cpu, _)) => self.mem.write_u8(cpu, value),
            None => Err(Error::Gpu(format!("gpu va {:#x} is not mapped", gpu_va))),
        }
    }
}
