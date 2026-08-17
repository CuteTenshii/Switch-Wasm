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
        let mut v = 0u128;
        for i in 0..len {
            v |= (self.vmm_read_u8(gpu_va + i as u64)? as u128) << (8 * i);
        }
        Ok(v)
    }

    /// Write `len` bytes of a surface's raw pixel, little-endian.
    pub fn write_pixel(&mut self, gpu_va: u64, len: u32, value: u128) -> Result<()> {
        for i in 0..len {
            self.vmm_write_u8(gpu_va + i as u64, (value >> (8 * i)) as u8)?;
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
