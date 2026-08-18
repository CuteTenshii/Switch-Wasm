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
/// Hard ceiling on real, host-backed guest RAM. Any homebrew's actual image +
/// heap + stack use sits far below this; it exists to bound a runaway guest
/// write (e.g. a stray pointer walking up from a null base, one soft-mapped
/// page at a time) to a fast, cheap failure instead of ballooning the host
/// process — a browser tab included — for seconds before anything faults.
pub const MAX_MAPPED_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB
const MAX_MAPPED_PAGES: usize = (MAX_MAPPED_BYTES / PAGE_SIZE as u64) as usize;

#[derive(Debug)]
pub struct Memory {
    /// One slot per page. `None` means the page is not mapped.
    pages: Vec<Option<Box<[u8; PAGE_SIZE]>>>,
    /// Soft region as `(start, end)` (end-exclusive); `start > end` disables
    /// it. Unmapped pages in `[start, end)` read as zero from [`Memory::zero`]
    /// and allocate a private page on first write.
    soft: (u32, u32),
    /// Read-only region as `(start, end)` (end-exclusive); `start > end`
    /// disables it. Sits over the loaded image's `.text` once the loader has
    /// finished patching it, so a guest write through a wild pointer faults
    /// instead of silently corrupting the running code — see
    /// [`Memory::mark_readonly`].
    readonly: (u32, u32),
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
            readonly: (1, 0),
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

    /// Mark `[start, end)` as read-only to the guest: a CPU store into it
    /// faults instead of writing through. Replaces any previous read-only
    /// region (there is only ever one loaded image). Loader-only writes
    /// (`map`/`map_zero`/`copy_range`) are unaffected — call this once a
    /// segment's relocations have been patched in, not before.
    pub fn mark_readonly(&mut self, start: u32, end: u32) {
        self.readonly = (start, end);
    }

    #[inline(always)]
    fn check_writable(&self, addr: u32) -> Result<()> {
        if addr >= self.readonly.0 && addr < self.readonly.1 {
            Err(Error::Cpu(format!("write to read-only address {:#010x}", addr)))
        } else {
            Ok(())
        }
    }

    #[inline]
    fn page_index(addr: u32) -> usize {
        (addr as usize) >> PAGE_BITS
    }

    #[inline]
    fn in_page_offset(addr: u32) -> usize {
        (addr as usize) & (PAGE_SIZE - 1)
    }

    #[inline(always)]
    fn page_mut(&mut self, idx: usize) -> Result<&mut Box<[u8; PAGE_SIZE]>> {
        if self.pages[idx].is_none() {
            self.allocate_page(idx)?;
        }
        Ok(self.pages[idx].as_mut().unwrap())
    }

    /// First touch of a page. Out of line so the common "already mapped" store
    /// path does not carry the allocation code with it.
    #[cold]
    #[inline(never)]
    fn allocate_page(&mut self, idx: usize) -> Result<()> {
        if self.mapped_pages >= MAX_MAPPED_PAGES {
            return Err(Error::Cpu(format!(
                "out of guest memory: exceeded the {} MiB cap",
                MAX_MAPPED_BYTES / (1024 * 1024)
            )));
        }
        self.pages[idx] = Some(Box::new([0u8; PAGE_SIZE]));
        self.mapped_pages += 1;
        Ok(())
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

    #[inline(always)]
    fn page_ref(&self, idx: usize) -> Result<&[u8; PAGE_SIZE]> {
        match self.pages.get(idx).and_then(|p| p.as_deref()) {
            Some(page) => Ok(page),
            None => self.page_ref_unmapped(idx),
        }
    }

    /// An access to a page with no storage behind it: zeros inside a soft-mapped
    /// region, a fault anywhere else. Kept out of line so the mapped case stays
    /// small enough to inline into its callers — it is on the path of every
    /// instruction fetch.
    #[cold]
    #[inline(never)]
    fn page_ref_unmapped(&self, idx: usize) -> Result<&[u8; PAGE_SIZE]> {
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

    #[inline(always)]
    pub fn read_u8(&self, addr: u32) -> Result<u8> {
        let page = self.page_ref(Self::page_index(addr))?;
        Ok(page[Self::in_page_offset(addr)])
    }

    /// The `N` bytes at `addr` when they all live in one page. Multi-byte
    /// accesses go through this so they cost a single page lookup instead of one
    /// per byte — the interpreter reads four bytes for every instruction it
    /// fetches, so this is the hottest path in the emulator.
    #[inline(always)]
    fn read_bytes_in_page<const N: usize>(&self, addr: u32) -> Option<[u8; N]> {
        let off = Self::in_page_offset(addr);
        if off + N > PAGE_SIZE {
            return None;
        }
        let page = self.page_ref(Self::page_index(addr)).ok()?;
        Some(page[off..off + N].try_into().unwrap())
    }

    /// Same, for writing. `None` means the access straddles a page boundary and
    /// the caller has to fall back to going byte by byte.
    #[inline(always)]
    fn write_bytes_in_page<const N: usize>(&mut self, addr: u32, val: [u8; N]) -> Option<()> {
        let off = Self::in_page_offset(addr);
        if off + N > PAGE_SIZE {
            return None;
        }
        self.check_writable(addr).ok()?;
        let page = self.page_mut(Self::page_index(addr)).ok()?;
        page[off..off + N].copy_from_slice(&val);
        Some(())
    }

    #[inline(always)]
    pub fn read_u16(&self, addr: u32) -> Result<u16> {
        match self.read_bytes_in_page::<2>(addr) {
            Some(bytes) => Ok(u16::from_le_bytes(bytes)),
            None => self.read_u16_straddling(addr),
        }
    }

    /// The rare read that crosses a page boundary.
    #[cold]
    #[inline(never)]
    fn read_u16_straddling(&self, addr: u32) -> Result<u16> {
        Ok((self.read_u8(addr)? as u16) | ((self.read_u8(addr.wrapping_add(1))? as u16) << 8))
    }

    #[inline(always)]
    pub fn read_u32(&self, addr: u32) -> Result<u32> {
        match self.read_bytes_in_page::<4>(addr) {
            Some(bytes) => Ok(u32::from_le_bytes(bytes)),
            None => self.read_u32_straddling(addr),
        }
    }

    #[cold]
    #[inline(never)]
    fn read_u32_straddling(&self, addr: u32) -> Result<u32> {
        Ok((self.read_u8(addr)? as u32)
            | ((self.read_u8(addr.wrapping_add(1))? as u32) << 8)
            | ((self.read_u8(addr.wrapping_add(2))? as u32) << 16)
            | ((self.read_u8(addr.wrapping_add(3))? as u32) << 24))
    }

    #[inline(always)]
    pub fn read_u64(&self, addr: u32) -> Result<u64> {
        match self.read_bytes_in_page::<8>(addr) {
            Some(bytes) => Ok(u64::from_le_bytes(bytes)),
            None => self.read_u64_straddling(addr),
        }
    }

    #[cold]
    #[inline(never)]
    fn read_u64_straddling(&self, addr: u32) -> Result<u64> {
        Ok((self.read_u32(addr)? as u64) | ((self.read_u32(addr.wrapping_add(4))? as u64) << 32))
    }

    /// Fetch the next instruction (little-endian AArch64 word).
    #[inline(always)]
    pub fn fetch(&self, pc: u32) -> Result<u32> {
        self.read_u32(pc)
    }

    #[inline(always)]
    pub fn write_u8(&mut self, addr: u32, val: u8) -> Result<()> {
        self.check_writable(addr)?;
        let idx = Self::page_index(addr);
        let off = Self::in_page_offset(addr);
        let page = self.page_mut(idx)?;
        page[off] = val;
        Ok(())
    }

    #[inline(always)]
    pub fn write_u16(&mut self, addr: u32, val: u16) -> Result<()> {
        if self.write_bytes_in_page(addr, val.to_le_bytes()).is_some() {
            return Ok(());
        }
        self.write_u16_straddling(addr, val)
    }

    /// The rare access that crosses a page boundary.
    #[cold]
    #[inline(never)]
    fn write_u16_straddling(&mut self, addr: u32, val: u16) -> Result<()> {
        self.write_u8(addr, val as u8)?;
        self.write_u8(addr.wrapping_add(1), (val >> 8) as u8)
    }

    #[inline(always)]
    pub fn write_u32(&mut self, addr: u32, val: u32) -> Result<()> {
        if self.write_bytes_in_page(addr, val.to_le_bytes()).is_some() {
            return Ok(());
        }
        self.write_u32_straddling(addr, val)
    }

    /// The rare access that crosses a page boundary.
    #[cold]
    #[inline(never)]
    fn write_u32_straddling(&mut self, addr: u32, val: u32) -> Result<()> {
        self.write_u8(addr, val as u8)?;
        self.write_u8(addr.wrapping_add(1), (val >> 8) as u8)?;
        self.write_u8(addr.wrapping_add(2), (val >> 16) as u8)?;
        self.write_u8(addr.wrapping_add(3), (val >> 24) as u8)
    }

    #[inline(always)]
    pub fn write_u64(&mut self, addr: u32, val: u64) -> Result<()> {
        if self.write_bytes_in_page(addr, val.to_le_bytes()).is_some() {
            return Ok(());
        }
        self.write_u64_straddling(addr, val)
    }

    /// The rare access that crosses a page boundary.
    #[cold]
    #[inline(never)]
    fn write_u64_straddling(&mut self, addr: u32, val: u64) -> Result<()> {
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
    fn runaway_soft_writes_fail_fast_at_the_ram_cap() {
        // A wild pointer walking up through a wide-open soft region (the
        // scenario a null buffer pointer plus a growing byte offset hits)
        // must not be allowed to fabricate pages forever: it should fail as
        // soon as it has touched the RAM cap's worth of pages, not after
        // exhausting the whole soft region.
        let mut m = Memory::new();
        m.soft_map_zero(0, 0x8000_0000);
        let mut addr = 0u32;
        let mut touched = 0u64;
        loop {
            match m.write_u8(addr, 1) {
                Ok(()) => {
                    touched += 1;
                    addr = addr.wrapping_add(PAGE_SIZE as u32);
                }
                Err(_) => break,
            }
        }
        assert_eq!(touched, MAX_MAPPED_PAGES as u64);
        assert_eq!(m.mapped_bytes(), MAX_MAPPED_BYTES);
    }

    #[test]
    fn readonly_region_rejects_guest_writes_but_not_reads() {
        let mut m = Memory::new();
        m.map_zero(0x1000, PAGE_SIZE).unwrap();
        m.write_u32(0x1000, 0x1111_1111).unwrap();
        m.mark_readonly(0x1000, 0x2000);
        // A wild write into the now-locked-down page faults instead of
        // silently corrupting it...
        assert!(m.write_u32(0x1000, 0x2222_2222).is_err());
        assert!(m.write_u8(0x1FFF, 1).is_err());
        // ...but reads, and writes just outside the range, are unaffected.
        assert_eq!(m.read_u32(0x1000).unwrap(), 0x1111_1111);
        m.map_zero(0x2000, PAGE_SIZE).unwrap();
        m.write_u32(0x2000, 3).unwrap();
        assert_eq!(m.read_u32(0x2000).unwrap(), 3);
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
