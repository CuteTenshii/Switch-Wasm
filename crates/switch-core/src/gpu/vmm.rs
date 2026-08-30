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
use std::cell::Cell;
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
/// `NVGPU_AS_MAP_BUFFER_FLAGS_MODIFY`: `MAP_BUFFER_EX` is not mapping a new
/// nvmap handle at all, it is **re-mapping a sub-range of a mapping that
/// already exists** with a different memory kind. The `offset` field names the
/// existing mapping rather than requesting one, and the nvmap handle field is
/// unused — which is why treating this as an ordinary map rejected it with
/// `BadParameter` for handle 0.
pub const FLAG_REMAP_SUB_RANGE: u32 = 1 << 8;

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

/// How many mappings [`AddressSpace::translate`] remembers.
const TRANSLATION_WAYS: usize = 8;

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
    /// The mappings recent translations resolved to. Engines walk a surface
    /// address by address, so without this a blit pays a `BTreeMap` search
    /// for every pixel it touches.
    ///
    /// Several entries rather than one, because a shaded pixel does not stay
    /// in a single mapping: it reads two or three constant buffers, samples a
    /// texture and reads and writes the render target, each of which is its
    /// own. A one-entry cache is evicted by every one of those in turn and
    /// hits almost never — the `BTreeMap` search was still 5% of the Home
    /// Menu's frame with it in place.
    ///
    /// Split across three arrays because the scan is what runs per pixel and
    /// it only needs two of the three: a `Cell<Option<(u64, u32, u64)>>` per
    /// way made rejecting a way copy the whole thirty-two-byte option out of
    /// the cell. A `size` of zero is an empty way, so no discriminant is
    /// needed to say so — no mapping is zero bytes long.
    recent_base: [Cell<u64>; TRANSLATION_WAYS],
    recent_size: [Cell<u64>; TRANSLATION_WAYS],
    recent_cpu: [Cell<u32>; TRANSLATION_WAYS],
    /// Round-robin replacement cursor for the three arrays above.
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
            Reservation {
                base,
                size,
                page_size: page_size as u64,
            },
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
            self.forget_translations();
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

    /// Re-map `[gpu_va, gpu_va + size)` — a sub-range of a mapping that
    /// already exists — with a different memory kind
    /// ([`FLAG_REMAP_SUB_RANGE`]). Returns whether a mapping covered it.
    ///
    /// This is how a driver gives one buffer several layouts: the whole nvmap
    /// handle is mapped once, then the ranges holding block-linear images are
    /// re-mapped over the top with the kind that describes their swizzle. The
    /// backing memory does not move — the sub-range keeps resolving to exactly
    /// the CPU bytes it did before — so the only thing that changes is the
    /// kind recorded for those pages.
    ///
    /// The covering mapping is split rather than overwritten. Overwriting it
    /// would drop whatever lay past the sub-range (the map is keyed by start
    /// VA, so a sub-range starting at the same VA replaces the whole entry),
    /// and [`AddressSpace::translate`] relies on the ranges not overlapping.
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
        // `NV_KIND_INVALID` means "keep what the mapping already had", and a
        // kind that is already what it should be needs no split at all.
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

    /// Drop the mapping that starts at `gpu_va` (`UNMAP_BUFFER`).
    pub fn unmap(&mut self, gpu_va: u64) -> Result<()> {
        self.mappings.remove(&gpu_va);
        self.forget_translations();
        Ok(())
    }

    /// Clear `[gpu_va, gpu_va + size)`, keeping whatever lies outside it.
    ///
    /// `REMAP` addresses a range rather than a whole buffer, so what it
    /// replaces can be part of a larger mapping or several smaller ones. A
    /// partly-covered mapping is trimmed instead of dropped, because
    /// [`AddressSpace::translate`] takes the ranges to be disjoint: leaving an
    /// old mapping overlapping a new one resolves addresses through whichever
    /// starts lower, which is not the one the guest just asked for.
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

    /// Drop every cached translation. Called whenever a mapping changes, since
    /// a cached entry outliving its mapping would hand out a stale address.
    fn forget_translations(&self) {
        for way in 0..TRANSLATION_WAYS {
            self.recent_size[way].set(0);
        }
    }

    /// The mapping covering `gpu_va`, if any.
    pub fn mapping_at(&self, gpu_va: u64) -> Option<&Mapping> {
        let (_, m) = self.mappings.range(..=gpu_va).next_back()?;
        if m.contains(gpu_va) {
            Some(m)
        } else {
            None
        }
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
        // The hole is gone and both sides keep resolving to their own bytes.
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
