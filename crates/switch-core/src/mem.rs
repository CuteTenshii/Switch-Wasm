//! Sparse 4 GiB address space for the emulated Switch.
//!
//! Backed by fixed 4 KiB pages allocated on demand, so an idle system costs
//! almost nothing in the browser while still permitting the full 32-bit
//! address range the CPU can reach. Reads/writes to unmapped addresses fault
//! with [`Error::Cpu`], except inside an optional "soft" region (see
//! [`Memory::soft_map_zero`]) whose unmapped pages read as zero and allocate
//! a real page on first write — used to present homebrew with its expected
//! address space without reserving it up front.

use crate::{Error, Result};

pub const PAGE_SIZE: usize = 4096;
pub const PAGE_BITS: u32 = 12;
pub const ADDRESS_SPACE_SIZE: u64 = 0x1_0000_0000; // 4 GiB
/// Number of 4 KiB pages in the 4 GiB space. Computed in u64 first so it
/// survives the wasm32 (32-bit usize) truncation of the 4 GiB constant.
const PAGE_COUNT: usize = (ADDRESS_SPACE_SIZE >> PAGE_BITS) as usize;

#[derive(Debug)]
pub struct Memory {
    /// One slot per page. `None` means the page is not mapped.
    pages: Vec<Option<Box<[u8; PAGE_SIZE]>>>,
    /// Soft region as `(start, end)` (end-exclusive); `start > end` disables
    /// it. Unmapped pages in `[start, end)` read as zero from [`Memory::zero`]
    /// and allocate a private page on first write.
    soft: (u32, u32),
    /// Shared zero page served for reads inside the soft region.
    zero: Box<[u8; PAGE_SIZE]>,
    /// How many pages currently hold real storage. Counted as they are
    /// allocated so reporting guest RAM use never walks the million-entry
    /// page table.
    mapped_pages: usize,
}

impl Default for Memory {
    fn default() -> Self {
        Memory::new()
    }
}

impl Memory {
    pub fn new() -> Memory {
        Memory {
            pages: vec![None; PAGE_COUNT],
            soft: (1, 0),
            zero: Box::new([0u8; PAGE_SIZE]),
            mapped_pages: 0,
        }
    }

    /// Mark `[start, end)` as softly mapped: reads return zeros (served from
    /// a single shared page) and writes allocate a real page on first touch.
    /// This lets homebrew read its uninitialized address space without the
    /// host reserving the whole region up front.
    pub fn soft_map_zero(&mut self, start: u32, end: u32) {
        self.soft = (start, end);
    }

    #[inline]
    fn page_index(addr: u32) -> usize {
        (addr as usize) >> PAGE_BITS
    }

    #[inline]
    fn in_page_offset(addr: u32) -> usize {
        (addr as usize) & (PAGE_SIZE - 1)
    }

    fn page_mut(&mut self, idx: usize) -> Result<&mut Box<[u8; PAGE_SIZE]>> {
        if self.pages[idx].is_none() {
            self.pages[idx] = Some(Box::new([0u8; PAGE_SIZE]));
            self.mapped_pages += 1;
        }
        Ok(self.pages[idx].as_mut().unwrap())
    }

    /// Pages backed by real storage.
    pub fn mapped_pages(&self) -> usize {
        self.mapped_pages
    }

    /// Guest memory actually backed by host storage, in bytes. This is what the
    /// emulated console "uses": the image, stack, heap and every page the guest
    /// has touched inside a soft-mapped region.
    pub fn mapped_bytes(&self) -> u64 {
        self.mapped_pages as u64 * PAGE_SIZE as u64
    }

    fn page_ref(&self, idx: usize) -> Result<&[u8; PAGE_SIZE]> {
        if let Some(page) = self.pages[idx].as_deref() {
            return Ok(page);
        }
        let addr = idx << PAGE_BITS;
        if addr >= self.soft.0 as usize && addr < self.soft.1 as usize {
            return Ok(&self.zero);
        }
        Err(Error::Cpu(format!("read from unmapped address {:#010x}", addr)))
    }

    /// Whether a real page has been allocated at `addr` (as opposed to a
    /// soft-mapped page that has never been touched). Used by `svcQueryMemory`
    /// so address-space walks see genuinely free pages as unmapped.
    pub fn page_mapped(&self, addr: u32) -> bool {
        self.pages[Self::page_index(addr)].is_some()
    }

    /// Map `data` at `addr`, allocating pages as needed and zero-filling any
    /// gap between existing mappings. Wraps around page boundaries.
    pub fn map(&mut self, addr: u32, data: &[u8]) -> Result<()> {
        let mut pos = addr as usize;
        for chunk in data.chunks(PAGE_SIZE - (pos & (PAGE_SIZE - 1))) {
            let idx = pos >> PAGE_BITS;
            let off = pos & (PAGE_SIZE - 1);
            let page = self.page_mut(idx)?;
            let n = chunk.len();
            page[off..off + n].copy_from_slice(chunk);
            pos += n;
        }
        Ok(())
    }

    /// Map `size` zero-filled bytes at `addr`.
    pub fn map_zero(&mut self, addr: u32, size: usize) -> Result<()> {
        let mut pos = addr as usize;
        let end = pos.saturating_add(size);
        while pos < end {
            let idx = pos >> PAGE_BITS;
            self.page_mut(idx)?;
            pos = (pos & !(PAGE_SIZE - 1)) + PAGE_SIZE;
        }
        Ok(())
    }

    #[inline]
    pub fn read_u8(&self, addr: u32) -> Result<u8> {
        let page = self.page_ref(Self::page_index(addr))?;
        Ok(page[Self::in_page_offset(addr)])
    }

    #[inline]
    pub fn read_u16(&self, addr: u32) -> Result<u16> {
        Ok((self.read_u8(addr)? as u16)
            | ((self.read_u8(addr.wrapping_add(1))? as u16) << 8))
    }

    #[inline]
    pub fn read_u32(&self, addr: u32) -> Result<u32> {
        Ok((self.read_u8(addr)? as u32)
            | ((self.read_u8(addr.wrapping_add(1))? as u32) << 8)
            | ((self.read_u8(addr.wrapping_add(2))? as u32) << 16)
            | ((self.read_u8(addr.wrapping_add(3))? as u32) << 24))
    }

    #[inline]
    pub fn read_u64(&self, addr: u32) -> Result<u64> {
        Ok((self.read_u32(addr)? as u64) | ((self.read_u32(addr.wrapping_add(4))? as u64) << 32))
    }

    /// Fetch the next instruction (little-endian AArch64 word).
    #[inline]
    pub fn fetch(&self, pc: u32) -> Result<u32> {
        self.read_u32(pc)
    }

    #[inline]
    pub fn write_u8(&mut self, addr: u32, val: u8) -> Result<()> {
        let idx = Self::page_index(addr);
        let off = Self::in_page_offset(addr);
        let page = self.page_mut(idx)?;
        page[off] = val;
        Ok(())
    }

    #[inline]
    pub fn write_u16(&mut self, addr: u32, val: u16) -> Result<()> {
        self.write_u8(addr, val as u8)?;
        self.write_u8(addr.wrapping_add(1), (val >> 8) as u8)
    }

    #[inline]
    pub fn write_u32(&mut self, addr: u32, val: u32) -> Result<()> {
        self.write_u8(addr, val as u8)?;
        self.write_u8(addr.wrapping_add(1), (val >> 8) as u8)?;
        self.write_u8(addr.wrapping_add(2), (val >> 16) as u8)?;
        self.write_u8(addr.wrapping_add(3), (val >> 24) as u8)
    }

    #[inline]
    pub fn write_u64(&mut self, addr: u32, val: u64) -> Result<()> {
        self.write_u32(addr, val as u32)?;
        self.write_u32(addr.wrapping_add(4), (val >> 32) as u32)
    }

    /// Read `len` bytes into `buf` starting at `addr`.
    pub fn read_into(&self, addr: u32, buf: &mut [u8]) -> Result<()> {
        let mut pos = addr as usize;
        let end = pos.saturating_add(buf.len());
        let mut out = 0usize;
        while pos < end {
            let idx = pos >> PAGE_BITS;
            let page = self.page_ref(idx)?;
            let off = pos & (PAGE_SIZE - 1);
            let n = (PAGE_SIZE - off).min(end - pos);
            buf[out..out + n].copy_from_slice(&page[off..off + n]);
            pos += n;
            out += n;
        }
        Ok(())
    }

    /// Copy `size` bytes from `src` to `dst`, backing the destination with real
    /// pages. Horizon's `svcMapMemory` aliases two ranges; page storage here is
    /// not shareable, so the bytes are copied instead and copied back when the
    /// alias is torn down. The guest only uses one side of such an alias at a
    /// time — libnx maps a thread's stack into the stack region and from then on
    /// touches only the mirror — so a copy behaves like an alias to it.
    /// Source pages that were never touched contribute zeros, the same
    /// zero-filled memory the guest would have seen through the alias.
    pub fn copy_range(&mut self, dst: u32, src: u32, size: usize) -> Result<()> {
        let mut buf = vec![0u8; size];
        let mut pos = 0usize;
        while pos < size {
            let addr = src.wrapping_add(pos as u32);
            let off = Self::in_page_offset(addr);
            let n = (PAGE_SIZE - off).min(size - pos);
            if let Ok(page) = self.page_ref(Self::page_index(addr)) {
                buf[pos..pos + n].copy_from_slice(&page[off..off + n]);
            }
            pos += n;
        }
        self.map(dst, &buf)
    }

    /// Drop the real pages backing `size` bytes at `addr`, so address-space
    /// walks see the range as free again. Partial pages at either end are kept,
    /// since something else may still live in them.
    pub fn unmap(&mut self, addr: u32, size: usize) {
        // Whole pages only, and in page indices: the address space is 4 GiB, so
        // byte counts do not fit a 32-bit usize on wasm.
        let first = (addr as u64 + PAGE_SIZE as u64 - 1) >> PAGE_BITS;
        let last = (addr as u64 + size as u64) >> PAGE_BITS;
        for idx in first..last.min(PAGE_COUNT as u64) {
            if self.pages[idx as usize].take().is_some() {
                self.mapped_pages -= 1;
            }
        }
    }

    /// Dump a contiguous region for debugging.
    pub fn dump(&self, addr: u32, len: usize) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            out.push(self.read_u8(addr.wrapping_add(i as u32))?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_bytes_counts_pages_as_they_are_backed() {
        let mut m = Memory::new();
        assert_eq!(m.mapped_bytes(), 0);
        m.map_zero(0x1000, PAGE_SIZE).unwrap();
        assert_eq!(m.mapped_bytes(), PAGE_SIZE as u64);
        // Writing inside an already-backed page doesn't count again.
        m.write_u8(0x1FFF, 1).unwrap();
        assert_eq!(m.mapped_bytes(), PAGE_SIZE as u64);
        // A soft-mapped page only costs storage once the guest writes to it.
        m.soft_map_zero(0x2000, 0x4000);
        assert_eq!(m.read_u8(0x2000).unwrap(), 0);
        assert_eq!(m.mapped_bytes(), PAGE_SIZE as u64);
        m.write_u8(0x2000, 7).unwrap();
        assert_eq!(m.mapped_bytes(), 2 * PAGE_SIZE as u64);
        assert_eq!(m.mapped_pages(), 2);
    }

    #[test]
    fn unmmapped_read_faults() {
        let m = Memory::new();
        assert!(m.read_u32(0xDEAD_0000).is_err());
    }

    #[test]
    fn map_across_page_boundary() {
        let mut m = Memory::new();
        let data = (0..=255u8).collect::<Vec<_>>();
        let addr = 0x0000_1FF0; // starts 16 bytes before a page boundary
        m.map(addr, &data).unwrap();
        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(m.read_u8(addr + i as u32).unwrap(), expected);
        }
    }

    #[test]
    fn zero_fill_between_maps() {
        let mut m = Memory::new();
        m.map_zero(0x0000_0000, PAGE_SIZE).unwrap();
        m.map_zero(0x0000_3000, PAGE_SIZE).unwrap();
        assert_eq!(m.read_u8(0x0000_0000).unwrap(), 0);
        assert_eq!(m.read_u8(0x0000_3000).unwrap(), 0);
        // Unmapped pages still fault.
        assert!(m.read_u8(0x0000_1000).is_err());
    }

    #[test]
    fn u32_u64_roundtrip() {
        let mut m = Memory::new();
        m.map_zero(0x0000_0000, 16).unwrap();
        m.write_u32(0, 0xDEAD_BEEF).unwrap();
        assert_eq!(m.read_u32(0).unwrap(), 0xDEAD_BEEF);
        m.write_u64(8, 0x1234_5678_9ABC_DEF0).unwrap();
        assert_eq!(m.read_u64(8).unwrap(), 0x1234_5678_9ABC_DEF0);
    }

    #[test]
    fn dump_matches_mapped_bytes() {
        let mut m = Memory::new();
        let data = (0..=127u8).collect::<Vec<_>>();
        m.map(0x0001_0000, &data).unwrap();
        assert_eq!(m.dump(0x0001_0000, 128).unwrap(), data);
    }
}
