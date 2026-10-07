//! GM20B GMMU: the GPU virtual address space. GPU addresses resolve into the
//! same guest [`Memory`] (no VRAM), as whole contiguous nvmap ranges.

use crate::mem::Memory;
use crate::{Error, Result};
use std::cell::Cell;
use std::collections::BTreeMap;

pub const SMALL_PAGE_SIZE: u64 = 0x1000;
pub const BIG_PAGE_SIZE: u64 = 0x1_0000;

/// Base of the small-page VA region reported by `GET_VA_REGIONS`.
pub const SMALL_REGION_BASE: u64 = 0x0400_0000;
pub const SMALL_REGION_END: u64 = 0x1_0000_0000;
/// End of the big-page region (40-bit GPU address space).
pub const BIG_REGION_END: u64 = 0x100_0000_0000;

/// `NVGPU_AS_ALLOC_SPACE_FLAGS_FIXED_OFFSET` / `..._MAP_BUFFER_FLAGS_FIXED_OFFSET`.
pub const FLAG_FIXED_OFFSET: u32 = 1 << 0;
/// `NVGPU_AS_MAP_BUFFER_FLAGS_MAPPABLE_COMPBITS`, ignored.
pub const FLAG_MAPPABLE_COMPBITS: u32 = 1 << 1;
/// `NVGPU_AS_MAP_BUFFER_FLAGS_CACHEABLE`.
pub const FLAG_CACHEABLE: u32 = 1 << 2;
/// `NVGPU_AS_MAP_BUFFER_FLAGS_MODIFY`: re-map a sub-range of an existing
/// mapping with a new kind; `offset` names it and the handle is unused.
pub const FLAG_REMAP_SUB_RANGE: u32 = 1 << 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mapping {
    pub gpu_va: u64,
    pub size: u64,
    pub cpu_addr: u32,
    /// 0 for a raw mapping.
    pub handle: u32,
    /// Memory kind (block-linear layout selector).
    pub kind: u8,
}

impl Mapping {
    fn contains(&self, gpu_va: u64) -> bool {
        gpu_va >= self.gpu_va && gpu_va - self.gpu_va < self.size
    }
}

/// A VA range reserved by `ALLOC_SPACE`; fixed-offset mappings land inside one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Reservation {
    base: u64,
    size: u64,
    page_size: u64,
}

const TRANSLATION_WAYS: usize = 8;

/// One `/dev/nvhost-as-gpu` fd.
#[derive(Debug)]
pub struct AddressSpace {
    /// Negotiated by `INITIALIZE_EX`; 0 until then.
    pub big_page_size: u64,
    mappings: BTreeMap<u64, Mapping>,
    reservations: BTreeMap<u64, Reservation>,
    next_small: u64,
    next_big: u64,
    /// Recently translated mappings, split by field so the per-pixel scan reads
    /// only two arrays. A `size` of zero is an empty way.
    recent_base: [Cell<u64>; TRANSLATION_WAYS],
    recent_size: [Cell<u64>; TRANSLATION_WAYS],
    recent_cpu: [Cell<u32>; TRANSLATION_WAYS],
    next_translation: Cell<usize>,
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
            recent_base: [const { Cell::new(0) }; TRANSLATION_WAYS],
            recent_size: [const { Cell::new(0) }; TRANSLATION_WAYS],
            recent_cpu: [const { Cell::new(0) }; TRANSLATION_WAYS],
            next_translation: Cell::new(0),
            next_small: SMALL_REGION_BASE,
            next_big: SMALL_REGION_END,
        }
    }

    /// Reserve address space at a fixed or allocated base.
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
            Reservation {
                base,
                size,
                page_size: page_size as u64,
            },
        );
        Ok(base)
    }

    /// Release a reservation, tearing down mappings inside it.
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
            self.forget_translations();
        }
        Ok(())
    }

    /// Map `size` bytes at `cpu_addr`, at `requested` or a fresh VA. Returns the GPU VA.
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
        let page_size = if page_size == 0 {
            SMALL_PAGE_SIZE
        } else {
            page_size
        };
        let gpu_va = if flags & FLAG_FIXED_OFFSET != 0 {
            requested
        } else {
            self.bump(size, page_size)?
        };
        self.mappings.insert(
            gpu_va,
            Mapping {
                gpu_va,
                size,
                cpu_addr,
                handle,
                kind,
            },
        );
        self.forget_translations();
        Ok(gpu_va)
    }

    /// Re-map a sub-range of an existing mapping with a different kind, splitting
    /// the covering mapping. Returns whether a mapping covered it.
    pub fn remap(&mut self, gpu_va: u64, size: u64, kind: Option<u8>) -> bool {
        if size == 0 {
            return false;
        }
        let Some((_, &covering)) = self.mappings.range(..=gpu_va).next_back() else {
            return false;
        };
        let covering_end = covering.gpu_va.saturating_add(covering.size);
        let end = gpu_va.saturating_add(size);
        if !covering.contains(gpu_va) || end > covering_end {
            return false;
        }
        // `NV_KIND_INVALID` keeps the existing kind.
        let kind = match kind {
            Some(kind) if kind != covering.kind => kind,
            _ => return true,
        };
        let piece = |gpu_va: u64, size: u64, kind: u8| Mapping {
            gpu_va,
            size,
            cpu_addr: covering
                .cpu_addr
                .wrapping_add((gpu_va - covering.gpu_va) as u32),
            handle: covering.handle,
            kind,
        };
        self.mappings.remove(&covering.gpu_va);
        if gpu_va > covering.gpu_va {
            self.mappings.insert(
                covering.gpu_va,
                piece(covering.gpu_va, gpu_va - covering.gpu_va, covering.kind),
            );
        }
        self.mappings.insert(gpu_va, piece(gpu_va, size, kind));
        if end < covering_end {
            self.mappings
                .insert(end, piece(end, covering_end - end, covering.kind));
        }
        self.forget_translations();
        true
    }

    pub fn unmap(&mut self, gpu_va: u64) -> Result<()> {
        self.mappings.remove(&gpu_va);
        self.forget_translations();
        Ok(())
    }

    /// Clear a range, trimming partly covered mappings so ranges stay disjoint.
    pub fn unmap_range(&mut self, gpu_va: u64, size: u64) {
        let end = gpu_va.saturating_add(size);
        let overlapping: Vec<Mapping> = self
            .mappings
            .range(..end)
            .map(|(_, m)| *m)
            .filter(|m| m.gpu_va.saturating_add(m.size) > gpu_va)
            .collect();
        for mapping in overlapping {
            let mapping_end = mapping.gpu_va.saturating_add(mapping.size);
            let piece = |at: u64, size: u64| Mapping {
                gpu_va: at,
                size,
                cpu_addr: mapping.cpu_addr.wrapping_add((at - mapping.gpu_va) as u32),
                handle: mapping.handle,
                kind: mapping.kind,
            };
            self.mappings.remove(&mapping.gpu_va);
            if mapping.gpu_va < gpu_va {
                self.mappings.insert(
                    mapping.gpu_va,
                    piece(mapping.gpu_va, gpu_va - mapping.gpu_va),
                );
            }
            if mapping_end > end {
                self.mappings.insert(end, piece(end, mapping_end - end));
            }
        }
        self.forget_translations();
    }

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

    #[inline]
    pub fn translate(&self, gpu_va: u64) -> Option<(u32, u64)> {
        for way in 0..TRANSLATION_WAYS {
            let size = self.recent_size[way].get();
            let off = gpu_va.wrapping_sub(self.recent_base[way].get());
            if off < size {
                return Some((
                    self.recent_cpu[way].get().wrapping_add(off as u32),
                    size - off,
                ));
            }
        }
        let (_, m) = self.mappings.range(..=gpu_va).next_back()?;
        if !m.contains(gpu_va) {
            return None;
        }
        let way = self.next_translation.get();
        self.recent_base[way].set(m.gpu_va);
        self.recent_size[way].set(m.size);
        self.recent_cpu[way].set(m.cpu_addr);
        self.next_translation.set((way + 1) % TRANSLATION_WAYS);
        let off = gpu_va - m.gpu_va;
        Some((m.cpu_addr.wrapping_add(off as u32), m.size - off))
    }

    /// a cached entry outliving its mapping would hand out a stale address.
    fn forget_translations(&self) {
        for way in 0..TRANSLATION_WAYS {
            self.recent_size[way].set(0);
        }
    }

    pub fn mapping_at(&self, gpu_va: u64) -> Option<&Mapping> {
        let (_, m) = self.mappings.range(..=gpu_va).next_back()?;
        if m.contains(gpu_va) {
            Some(m)
        } else {
            None
        }
    }

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

    pub fn read_u8(&self, mem: &Memory, gpu_va: u64) -> Result<u8> {
        mem.read_u8(self.cpu_addr(gpu_va, 1)?)
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
        let va = vmm
            .map(cpu_addr, size, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
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
        let a = vmm
            .map(0x2000_0000, 0x2000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        let b = vmm
            .map(0x2100_0000, 0x2000, 2, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        assert!(b >= a + 0x2000);
    }

    #[test]
    fn fixed_offset_is_honoured() {
        let mut vmm = AddressSpace::new();
        let base = vmm.alloc_space(16, 0x1_0000, 0, 0).unwrap();
        let va = vmm
            .map(
                0x2000_0000,
                0x1_0000,
                3,
                0,
                BIG_PAGE_SIZE,
                FLAG_FIXED_OFFSET,
                base,
            )
            .unwrap();
        assert_eq!(va, base);
        assert_eq!(vmm.translate(base), Some((0x2000_0000, 0x1_0000)));
    }

    #[test]
    fn big_and_small_regions_are_separate() {
        let mut vmm = AddressSpace::new();
        let small = vmm
            .map(0x2000_0000, 0x1000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        let big = vmm
            .map(0x2100_0000, 0x1_0000, 2, 0, BIG_PAGE_SIZE, 0, 0)
            .unwrap();
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
    fn unmap_range_trims_what_it_only_partly_covers() {
        let mut vmm = AddressSpace::new();
        let va = vmm
            .map(0x2000_0000, 0x4000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        vmm.unmap_range(va + 0x1000, 0x1000);
        assert_eq!(vmm.translate(va), Some((0x2000_0000, 0x1000)));
        assert_eq!(vmm.translate(va + 0x1000), None);
        assert_eq!(vmm.translate(va + 0x2000), Some((0x2000_2000, 0x2000)));
    }

    #[test]
    fn unmap_range_spans_several_mappings() {
        let mut vmm = AddressSpace::new();
        let a = vmm
            .map(
                0x2000_0000,
                0x1000,
                1,
                0,
                SMALL_PAGE_SIZE,
                FLAG_FIXED_OFFSET,
                0x10_0000,
            )
            .unwrap();
        vmm.map(
            0x2100_0000,
            0x1000,
            2,
            0,
            SMALL_PAGE_SIZE,
            FLAG_FIXED_OFFSET,
            0x10_1000,
        )
        .unwrap();
        vmm.map(
            0x2200_0000,
            0x1000,
            3,
            0,
            SMALL_PAGE_SIZE,
            FLAG_FIXED_OFFSET,
            0x10_2000,
        )
        .unwrap();
        vmm.unmap_range(a, 0x2000);
        assert_eq!(vmm.translate(a), None);
        assert_eq!(vmm.translate(a + 0x1000), None);
        assert_eq!(vmm.translate(a + 0x2000), Some((0x2200_0000, 0x1000)));
    }

    #[test]
    fn free_space_drops_mappings_inside_it() {
        let mut vmm = AddressSpace::new();
        let base = vmm.alloc_space(4, 0x1_0000, 0, 0).unwrap();
        vmm.map(
            0x2000_0000,
            0x1_0000,
            1,
            0,
            BIG_PAGE_SIZE,
            FLAG_FIXED_OFFSET,
            base,
        )
        .unwrap();
        vmm.free_space(base, 4, 0x1_0000).unwrap();
        assert!(vmm.translate(base).is_none());
    }
}
