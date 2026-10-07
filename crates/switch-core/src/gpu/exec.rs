//! Execution context handed to the engines while a channel's pushbuffer runs.

use crate::gpu::syncpt::Host1x;
use crate::gpu::vmm::AddressSpace;
use crate::mem::Memory;
use crate::{Error, Result};

/// Counters describing what the GPU has done.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuStats {
    /// GPFIFO submissions processed.
    pub submissions: u64,
    /// Method writes dispatched to an engine.
    pub methods: u64,
    /// `ClearBuffers` operations executed.
    pub clears: u64,
    /// Depth clears skipped because the target already held the cleared bytes.
    pub clears_elided: u64,
    /// Draw calls seen (`VertexBegin`/`DrawArrays`/`DrawElements`).
    pub draws: u64,
    /// Copy-engine and 2D-engine transfers executed.
    pub copies: u64,
    /// Macros executed by the MME.
    pub macros: u64,
    /// Method writes to registers with no implemented behaviour.
    pub inert_methods: u64,
    /// Compute dispatches launched.
    pub dispatches: u64,
    /// Dispatches that did not run (bad QMD or undecodable kernel).
    pub dispatches_skipped: u64,
    /// Draws the rasterizer refused, usually due to an undecodable shader.
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

    /// Read `len` bytes of a surface's raw pixel, little-endian.
    pub fn read_pixel(&self, gpu_va: u64, len: u32) -> Result<u128> {
        self.mem.read_le(self.pixel_addr(gpu_va, len)?, len)
    }

    /// Write `len` bytes of a surface's raw pixel, little-endian.
    pub fn write_pixel(&mut self, gpu_va: u64, len: u32, value: u128) -> Result<()> {
        let cpu = self.pixel_addr(gpu_va, len)?;
        self.mem.write_le(cpu, len, value)
    }

    /// Write `count` consecutive `unit`-byte pixels with the same value.
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

    /// [`ExecCtx::fill_pixels`] that only writes the `mask` bits of each pixel.
    pub fn merge_pixels(
        &mut self,
        gpu_va: u64,
        unit: u32,
        value: u128,
        mask: u128,
        count: u32,
    ) -> Result<()> {
        let all = if unit >= 16 {
            u128::MAX
        } else {
            (1u128 << (unit * 8)) - 1
        };
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

    /// The CPU address of a `len`-byte span, if one mapping holds all of it.
    pub fn span(&self, gpu_va: u64, len: u64) -> Option<u32> {
        match self.vmm.translate(gpu_va) {
            Some((cpu, left)) if left >= len => Some(cpu),
            _ => None,
        }
    }

    pub fn read_span(&self, cpu: u32, out: &mut [u8]) -> Result<()> {
        self.mem.read_into(cpu, out)
    }

    pub fn write_span(&mut self, cpu: u32, bytes: &[u8]) -> Result<()> {
        self.mem.write_from(cpu, bytes)
    }

    fn pixel_addr(&self, gpu_va: u64, len: u32) -> Result<u32> {
        match self.vmm.translate(gpu_va) {
            Some((cpu, left)) if left >= u64::from(len) => Ok(cpu),
            _ => Err(Error::Gpu(format!(
                "gpu va {:#x}: {} bytes are not mapped",
                gpu_va, len
            ))),
        }
    }

    pub fn read_run(&self, gpu_va: u64, out: &mut [u8]) -> Result<()> {
        if let Some(cpu) = self.span(gpu_va, out.len() as u64) {
            return self.read_span(cpu, out);
        }
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = self.vmm_read_u8(gpu_va + i as u64)?;
        }
        Ok(())
    }

    pub fn write_run(&mut self, gpu_va: u64, bytes: &[u8]) -> Result<()> {
        if let Some(cpu) = self.span(gpu_va, bytes.len() as u64) {
            return self.write_span(cpu, bytes);
        }
        for (i, byte) in bytes.iter().enumerate() {
            self.vmm_write_u8(gpu_va + i as u64, *byte)?;
        }
        Ok(())
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
