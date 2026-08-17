//! GM20B graphics MMU (GMMU): the GPU-side virtual address space.
//!
//! On the Switch the guest allocates its own CPU-visible memory and hands the
//! backing address to nvmap; `/dev/nvhost-as-gpu` then maps those nvmap
//! handles into a GPU address space. So a GPU virtual address resolves to a
//! CPU address in the same [`Memory`] the ARM core executes from — there is no
//! separate VRAM.
//!
//! Mappings are whole buffers rather than individual pages: an
//! `NVGPU_AS_IOCTL_MAP_BUFFER_EX` maps one contiguous nvmap range at one
//! contiguous GPU VA, so a sorted list of ranges translates exactly and costs
//! nothing per page.

use crate::mem::Memory;
use crate::{Error, Result};
use std::collections::BTreeMap;

/// GPU small page size (the GMMU's finest granularity).
pub const SMALL_PAGE_SIZE: u64 = 0x1000;
/// GPU big page size used by the Switch driver.
pub const BIG_PAGE_SIZE: u64 = 0x1_0000;

/// Base of the small-page VA region reported by `GET_VA_REGIONS`.
pub const SMALL_REGION_BASE: u64 = 0x0400_0000;
/// End of the small-page VA region / base of the big-page region.
pub const SMALL_REGION_END: u64 = 0x1_0000_0000;
/// End of the big-page VA region (the 40-bit GPU address space).
pub const BIG_REGION_END: u64 = 0x100_0000_0000;

/// `NVGPU_AS_ALLOC_SPACE_FLAGS_FIXED_OFFSET` / `..._MAP_BUFFER_FLAGS_FIXED_OFFSET`.
pub const FLAG_FIXED_OFFSET: u32 = 1 << 0;
/// `NVGPU_AS_MAP_BUFFER_FLAGS_MAPPABLE_COMPBITS` — irrelevant to us but
/// forwarded by the driver, so it must not be mistaken for a fixed offset.
pub const FLAG_MAPPABLE_COMPBITS: u32 = 1 << 1;
/// `NVGPU_AS_MAP_BUFFER_FLAGS_CACHEABLE`.
pub const FLAG_CACHEABLE: u32 = 1 << 2;

/// One mapped nvmap range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mapping {
    pub gpu_va: u64,
    pub size: u64,
    /// CPU address of the first byte, in the guest's address space.
    pub cpu_addr: u32,
    /// nvmap handle this range came from (0 for a raw/anonymous mapping).
    pub handle: u32,
    /// Memory kind (block-linear layout selector) the buffer was mapped with.
    pub kind: u8,
}

impl Mapping {
    fn contains(&self, gpu_va: u64) -> bool {
        gpu_va >= self.gpu_va && gpu_va - self.gpu_va < self.size
    }
}

/// A VA range reserved by `ALLOC_SPACE`. Mappings with a fixed offset land
/// inside one of these; the allocator never hands the same range out twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Reservation {
    base: u64,
    size: u64,
    page_size: u64,
}

/// One GPU address space (one `/dev/nvhost-as-gpu` fd).
#[derive(Debug)]
pub struct AddressSpace {
    /// Big page size negotiated by `INITIALIZE_EX` (0 until then).
    pub big_page_size: u64,
    mappings: BTreeMap<u64, Mapping>,
    reservations: BTreeMap<u64, Reservation>,
    /// Bump pointer for un-fixed small-page allocations.
    next_small: u64,
    /// Bump pointer for un-fixed big-page allocations.
    next_big: u64,
}

impl Default for AddressSpace {
    fn default() -> Self {
        AddressSpace::new()
    }
}

impl AddressSpace {
    pub fn new() -> AddressSpace {
        AddressSpace {
            big_page_size: BIG_PAGE_SIZE,
            mappings: BTreeMap::new(),
            reservations: BTreeMap::new(),
            next_small: SMALL_REGION_BASE,
            next_big: SMALL_REGION_END,
        }
    }

    /// Reserve `pages * page_size` bytes of address space. With
    /// [`FLAG_FIXED_OFFSET`] the caller picks the base; otherwise one is
    /// allocated from the region matching `page_size`.
    pub fn alloc_space(
        &mut self,
        pages: u32,
        page_size: u32,
        flags: u32,
        requested: u64,
    ) -> Result<u64> {
        let size = (pages as u64)
            .checked_mul(page_size as u64)
            .ok_or(Error::Overflow)?;
        if size == 0 {
            return Err(Error::Gpu("as: zero-sized address-space allocation".into()));
        }
        let base = if flags & FLAG_FIXED_OFFSET != 0 {
            requested
        } else {
            self.bump(size, page_size as u64)?
        };
        self.reservations.insert(
            base,
            Reservation { base, size, page_size: page_size as u64 },
        );
        Ok(base)
    }

    /// Release a reservation made by [`AddressSpace::alloc_space`]. Any
    /// mappings still inside it are torn down, mirroring the driver.
    pub fn free_space(&mut self, base: u64, pages: u32, page_size: u32) -> Result<()> {
        let size = (pages as u64)
            .checked_mul(page_size as u64)
            .ok_or(Error::Overflow)?;
        self.reservations.remove(&base);
        let doomed: Vec<u64> = self
            .mappings
            .range(base..base.saturating_add(size))
            .map(|(&k, _)| k)
            .collect();
        for key in doomed {
            self.mappings.remove(&key);
        }
        Ok(())
    }

    /// Map `size` bytes of the buffer at `cpu_addr` into the address space.
    /// With [`FLAG_FIXED_OFFSET`] the mapping lands at `requested`; otherwise
    /// a fresh VA is allocated. Returns the GPU VA.
    pub fn map(
        &mut self,
        cpu_addr: u32,
        size: u64,
        handle: u32,
        kind: u8,
        page_size: u64,
        flags: u32,
        requested: u64,
    ) -> Result<u64> {
        if size == 0 {
            return Err(Error::Gpu("as: zero-sized buffer mapping".into()));
        }
        let page_size = if page_size == 0 { SMALL_PAGE_SIZE } else { page_size as u64 };
        let gpu_va = if flags & FLAG_FIXED_OFFSET != 0 {
            requested
        } else {
            self.bump(size, page_size)?
        };
        self.mappings
            .insert(gpu_va, Mapping { gpu_va, size, cpu_addr, handle, kind });
        Ok(gpu_va)
    }

    /// Drop the mapping that starts at `gpu_va` (`UNMAP_BUFFER`).
    pub fn unmap(&mut self, gpu_va: u64) -> Result<()> {
        self.mappings.remove(&gpu_va);
        Ok(())
    }

    /// Allocate `size` bytes of VA from the region that matches `page_size`,
    /// aligned up to it.
    fn bump(&mut self, size: u64, page_size: u64) -> Result<u64> {
        let big = page_size >= BIG_PAGE_SIZE;
        let align = page_size.max(SMALL_PAGE_SIZE);
        let (cursor, limit) = if big {
            (&mut self.next_big, BIG_REGION_END)
        } else {
            (&mut self.next_small, SMALL_REGION_END)
        };
        let base = (*cursor + align - 1) & !(align - 1);
        let end = base.checked_add(size).ok_or(Error::Overflow)?;
        if end > limit {
            return Err(Error::Gpu(format!(
                "as: out of {} GPU address space ({:#x} bytes)",
                if big { "big-page" } else { "small-page" },
                size
            )));
        }
        *cursor = end;
        Ok(base)
    }

    /// Translate a GPU VA to `(cpu_addr, bytes_left_in_mapping)`.
    pub fn translate(&self, gpu_va: u64) -> Option<(u32, u64)> {
        let (_, m) = self.mappings.range(..=gpu_va).next_back()?;
        if !m.contains(gpu_va) {
            return None;
        }
        let off = gpu_va - m.gpu_va;
        Some((m.cpu_addr.wrapping_add(off as u32), m.size - off))
    }

    /// The mapping covering `gpu_va`, if any.
    pub fn mapping_at(&self, gpu_va: u64) -> Option<&Mapping> {
        let (_, m) = self.mappings.range(..=gpu_va).next_back()?;
        if m.contains(gpu_va) { Some(m) } else { None }
    }

    /// Every live mapping, in ascending GPU VA order.
    pub fn mappings(&self) -> impl Iterator<Item = &Mapping> {
        self.mappings.values()
    }

    fn cpu_addr(&self, gpu_va: u64, len: u64) -> Result<u32> {
        match self.translate(gpu_va) {
            Some((cpu, left)) if left >= len => Ok(cpu),
            Some((_, left)) => Err(Error::Gpu(format!(
                "gpu va {:#x}: access of {} bytes crosses the end of its mapping ({} left)",
                gpu_va, len, left
            ))),
            None => Err(Error::Gpu(format!("gpu va {:#x} is not mapped", gpu_va))),
        }
    }

    pub fn read_u32(&self, mem: &Memory, gpu_va: u64) -> Result<u32> {
        mem.read_u32(self.cpu_addr(gpu_va, 4)?)
    }

    pub fn read_u64(&self, mem: &Memory, gpu_va: u64) -> Result<u64> {
        mem.read_u64(self.cpu_addr(gpu_va, 8)?)
    }

    pub fn write_u32(&self, mem: &mut Memory, gpu_va: u64, value: u32) -> Result<()> {
        mem.write_u32(self.cpu_addr(gpu_va, 4)?, value)
    }

    pub fn write_u64(&self, mem: &mut Memory, gpu_va: u64, value: u64) -> Result<()> {
        mem.write_u64(self.cpu_addr(gpu_va, 8)?, value)
    }

    pub fn read_into(&self, mem: &Memory, gpu_va: u64, buf: &mut [u8]) -> Result<()> {
        mem.read_into(self.cpu_addr(gpu_va, buf.len() as u64)?, buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space_with_buffer(cpu_addr: u32, size: u64) -> (AddressSpace, u64) {
        let mut vmm = AddressSpace::new();
        let va = vmm.map(cpu_addr, size, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
        (vmm, va)
    }

    #[test]
    fn translate_inside_mapping() {
        let (vmm, va) = space_with_buffer(0x2000_0000, 0x4000);
        assert_eq!(vmm.translate(va), Some((0x2000_0000, 0x4000)));
        assert_eq!(vmm.translate(va + 0x100), Some((0x2000_0100, 0x3f00)));
        assert_eq!(vmm.translate(va + 0x4000), None);
    }

    #[test]
    fn unfixed_allocations_do_not_overlap() {
        let mut vmm = AddressSpace::new();
        let a = vmm.map(0x2000_0000, 0x2000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
        let b = vmm.map(0x2100_0000, 0x2000, 2, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
        assert!(b >= a + 0x2000);
    }

    #[test]
    fn fixed_offset_is_honoured() {
        let mut vmm = AddressSpace::new();
        let base = vmm.alloc_space(16, 0x1_0000, 0, 0).unwrap();
        let va = vmm
            .map(0x2000_0000, 0x1_0000, 3, 0, BIG_PAGE_SIZE, FLAG_FIXED_OFFSET, base)
            .unwrap();
        assert_eq!(va, base);
        assert_eq!(vmm.translate(base), Some((0x2000_0000, 0x1_0000)));
    }

    #[test]
    fn big_and_small_regions_are_separate() {
        let mut vmm = AddressSpace::new();
        let small = vmm.map(0x2000_0000, 0x1000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
        let big = vmm.map(0x2100_0000, 0x1_0000, 2, 0, BIG_PAGE_SIZE, 0, 0).unwrap();
        assert!(small < SMALL_REGION_END);
        assert!(big >= SMALL_REGION_END);
    }

    #[test]
    fn read_write_through_translation() {
        let mut mem = Memory::new();
        mem.map_zero(0x2000_0000, 0x1000).unwrap();
        let (vmm, va) = space_with_buffer(0x2000_0000, 0x1000);
        vmm.write_u32(&mut mem, va + 8, 0xDEAD_BEEF).unwrap();
        assert_eq!(vmm.read_u32(&mem, va + 8).unwrap(), 0xDEAD_BEEF);
        assert_eq!(mem.read_u32(0x2000_0008).unwrap(), 0xDEAD_BEEF);
    }

    #[test]
    fn access_past_the_mapping_end_faults() {
        let mut mem = Memory::new();
        mem.map_zero(0x2000_0000, 0x2000).unwrap();
        let (vmm, va) = space_with_buffer(0x2000_0000, 0x1000);
        assert!(vmm.read_u32(&mem, va + 0xffe).is_err());
        assert!(vmm.read_u32(&mem, va + 0x1000).is_err());
    }

    #[test]
    fn free_space_drops_mappings_inside_it() {
        let mut vmm = AddressSpace::new();
        let base = vmm.alloc_space(4, 0x1_0000, 0, 0).unwrap();
        vmm.map(0x2000_0000, 0x1_0000, 1, 0, BIG_PAGE_SIZE, FLAG_FIXED_OFFSET, base)
            .unwrap();
        vmm.free_space(base, 4, 0x1_0000).unwrap();
        assert!(vmm.translate(base).is_none());
    }
}
