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
/// Pages per entry in the block summary [`Memory`] keeps beside the page
/// table. 512 pages is 2 MiB: small enough that the summary is a few
/// kilobytes, large enough that walking a multi-gigabyte untouched region
/// costs thousands of steps instead of millions. See [`Memory::state_run`].
const BLOCK_PAGES: usize = 512;
const BLOCK_COUNT: usize = PAGE_COUNT / BLOCK_PAGES;
/// The default ceiling on real, host-backed guest RAM — see
/// [`Memory::set_max_mapped_bytes`] to choose another.
///
/// It exists to bound a runaway guest write (e.g. a stray pointer walking up
/// from a null base, one soft-mapped page at a time) to a fast, cheap failure
/// instead of ballooning the host process — a browser tab included — for
/// seconds before anything faults.
///
/// **It may not be smaller than what `svcGetInfo` advertises as
/// `TotalMemorySize`**, because a title believes that figure and sizes its
/// pools from it. 512 MiB held here while `GUEST_TOTAL_MEMORY_SIZE` said
/// 2.5 GiB, and a title took the emulator at its word: it reserved a 1.5 GiB
/// pool and died part way through `memset`ting it, in a `stp q0, q0` loop with
/// no allocation in sight. A cap under the advertised total does not limit a
/// title, it makes the emulator lie to one.
///
/// Backing is lazy — a page costs nothing until the guest touches it — so this
/// only decides when a run fails, never what an idle title reserves. The
/// console being emulated has 4 GiB, of which an application gets about 3.2,
/// so this is still short of hardware rather than generous.
pub const MAX_MAPPED_BYTES: u64 = 0xA000_0000; // 2.5 GiB, `GUEST_TOTAL_MEMORY_SIZE`
const MAX_MAPPED_PAGES: usize = (MAX_MAPPED_BYTES / PAGE_SIZE as u64) as usize;

/// Horizon's `MemoryState`, as `svcQueryMemory` reports it in the first word
/// past a region's bounds. Only the states this emulator can tell regions
/// apart by are here.
///
/// A module is **two** states, not one: the kernel maps its static half
/// (`.text` + `.rodata`) and its mutable half (`.data` + `.bss`) separately,
/// and SDK code walks from one into the other — see [`Memory::mark_module`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MemoryState {
    Unmapped = 0,
    /// The process image's static half.
    Code = 3,
    /// The process image's mutable half.
    CodeData = 4,
    /// The same two for a module `ldr:ro` mapped after the process started.
    AliasCode = 8,
    AliasCodeData = 9,
}

/// One region as `svcQueryMemory` describes it: the bounds of a run of pages
/// that share a state, and the state itself. See [`Memory::state_run`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateRun {
    pub start: u32,
    /// End-exclusive, and clamped to the limit the run was asked for.
    pub end: u32,
    /// Whether the pages hold real storage, as opposed to being untouched
    /// soft-mapped ones an address-space walk should see as free.
    pub mapped: bool,
    /// Whether they are write-protected — in practice a module's `.text`.
    pub readonly: bool,
    /// What the guest is told the run is.
    pub state: MemoryState,
}

/// One loaded module's image, split the way the kernel maps it.
#[derive(Debug, Clone, Copy)]
struct ModuleImage {
    /// `.text` + `.rodata`, end-exclusive and page-aligned.
    static_range: (u32, u32),
    /// `.data` + `.bss`, likewise.
    mutable_range: (u32, u32),
    /// Whether this is an `ldr:ro` module rather than part of the process
    /// image, which decides between the `Alias*` states and the plain ones.
    alias: bool,
}

#[derive(Debug)]
pub struct Memory {
    /// One slot per page. `None` means the page is not mapped.
    pages: Vec<Option<Box<[u8; PAGE_SIZE]>>>,
    /// Soft region as `(start, end)` (end-exclusive); `start > end` disables
    /// it. Unmapped pages in `[start, end)` read as zero from [`Memory::zero`]
    /// and allocate a private page on first write.
    soft: (u32, u32),
    /// Read-only regions, each `(start, end)` (end-exclusive). Sits over each
    /// loaded module's `.text` once the loader has finished patching it, so
    /// a guest write through a wild pointer faults instead of silently
    /// corrupting running code, and so `svcQueryMemory` can report the
    /// real-world R-X permission code on it (retail `rtld` scans process
    /// memory for other modules by filtering on exactly that permission —
    /// see [`Memory::mark_readonly`]). A retail process loads several
    /// modules (`rtld`/`main`/`subsdk*`/`sdk`), so this is a list, not a
    /// single range.
    readonly: Vec<(u32, u32)>,
    /// The envelope of every range in `readonly`, as `(lowest start, highest
    /// end)`; `start >= end` when there are none.
    ///
    /// [`Memory::is_readonly`] runs on **every** guest store and walks that
    /// list linearly, so a process with four modules loaded paid four
    /// comparisons per store — against a clear that writes eleven million
    /// texels a frame, and a heap that writes far more. Every protected range
    /// is a module's `.text` down in the image, so almost every store a
    /// running title makes is outside the envelope and answers in two
    /// comparisons without touching the list at all.
    readonly_span: (u32, u32),
    /// Every module image currently mapped, in load order. Small — a retail
    /// process is four modules plus whatever `ldr:ro` has open.
    modules: Vec<ModuleImage>,
    /// The envelope of every range in `modules`, as `(lowest start, highest
    /// end)`; `start >= end` when there are none. Same trick as
    /// `readonly_span`: a page outside it is classified in two comparisons.
    module_span: (u32, u32),
    /// Shared zero page served for reads inside the soft region.
    zero: Box<[u8; PAGE_SIZE]>,
    /// How many pages currently hold real storage. Counted as they are
    /// allocated so reporting guest RAM use never walks the million-entry
    /// page table.
    mapped_pages: usize,
    /// The ceiling `mapped_pages` may reach, from [`MAX_MAPPED_BYTES`] unless
    /// a caller lowered it. A field rather than a constant so that a host with
    /// less to give — or a test that wants to reach the cap without allocating
    /// gigabytes to do it — can say so.
    max_mapped_pages: usize,
    /// How many pages of each 2 MiB block hold real storage. Maintained
    /// alongside `pages` so a scan can skip a block that is entirely
    /// untouched without looking at its pages; that is the case that grows
    /// with the address space, and the only one that ever got expensive.
    block_mapped: Vec<u16>,
    /// Watchpoint `[start, end)` and the address of the most recent guest
    /// write that landed in it. `start >= end` disables it. A host-side
    /// debugger arms the range and reads [`Memory::take_watch_hit`] after each
    /// step, which is the only way to attribute a buffer's contents to the
    /// code that produced them: polling the buffer cannot see a write that
    /// stores the value already there, and cannot name the writer at all.
    watch: (u32, u32),
    watch_hit: Option<u32>,
    /// The same, for reads. Separate because the read path takes `&self`, so
    /// the hit is recorded through a `Cell`.
    read_watch: (u32, u32),
    read_hit: std::cell::Cell<Option<u32>>,
    /// Pages the JIT has translated code out of, one bit each. A store that
    /// lands on one of them invalidates whatever was compiled from it, so the
    /// translator's view of guest code can never go stale behind its back.
    ///
    /// One bit per 4 KiB page is 128 KiB for the whole address space, but a
    /// program's stores cluster, so the handful of cache lines under them is
    /// all a run ever touches. It is allocated on the first
    /// [`Memory::mark_code_page`], so a run with the JIT off does not carry
    /// it and every store's test misses on the length check alone.
    code_pages: Vec<u64>,
    /// Code pages written since the JIT last drained this. A page is recorded
    /// once — marking it clears its bit — and stays out of the bitmap until
    /// something is translated from it again.
    code_dirty: Vec<u32>,
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
            block_mapped: vec![0u16; BLOCK_COUNT],
            soft: (1, 0),
            readonly: Vec::new(),
            readonly_span: (u32::MAX, 0),
            modules: Vec::new(),
            module_span: (u32::MAX, 0),
            zero: Box::new([0u8; PAGE_SIZE]),
            mapped_pages: 0,
            max_mapped_pages: MAX_MAPPED_PAGES,
            watch: (1, 0),
            watch_hit: None,
            read_watch: (1, 0),
            read_hit: std::cell::Cell::new(None),
            code_pages: Vec::new(),
            code_dirty: Vec::new(),
        }
    }

    /// Record that the page holding `addr` has had code translated out of it,
    /// so a later store there is reported by [`Memory::dirty_code_pages`].
    pub fn mark_code_page(&mut self, addr: u32) {
        if self.code_pages.is_empty() {
            self.code_pages = vec![0u64; PAGE_COUNT / 64];
        }
        let idx = Self::page_index(addr);
        self.code_pages[idx >> 6] |= 1u64 << (idx & 63);
    }

    /// Note a guest store, invalidating the page's translations if it holds
    /// any. Inlined into every write path, so it has to answer "no" in a
    /// couple of instructions: one bounds-checked load from the bitmap (which
    /// is empty, and so always misses, while nothing has been translated) and
    /// one bit test. Only the recording is out of line.
    #[inline(always)]
    fn note_code_write(&mut self, addr: u32) {
        let idx = Self::page_index(addr);
        let bit = 1u64 << (idx & 63);
        if let Some(&word) = self.code_pages.get(idx >> 6) {
            if word & bit != 0 {
                self.mark_code_dirty(idx, bit);
            }
        }
    }

    /// Record that a page's translations are stale. Rare: it happens once per
    /// page until something is translated out of it again.
    #[cold]
    #[inline(never)]
    fn mark_code_dirty(&mut self, idx: usize, bit: u64) {
        self.code_pages[idx >> 6] &= !bit;
        self.code_dirty.push(idx as u32);
    }

    /// Mark every code page overlapping `[addr, addr + size)` dirty. Used by
    /// the loader-side paths ([`Memory::map`], [`Memory::map_zero`],
    /// [`Memory::unmap`]), which move whole segments at a time and do not go
    /// through the per-store write paths.
    fn dirty_code_range(&mut self, addr: u32, size: usize) {
        if self.code_pages.is_empty() || size == 0 {
            return;
        }
        let first = (addr as u64) >> PAGE_BITS;
        let last = (addr as u64 + size as u64 - 1) >> PAGE_BITS;
        for idx in first..=last.min(PAGE_COUNT as u64 - 1) {
            let idx = idx as usize;
            let bit = 1u64 << (idx & 63);
            if self.code_pages[idx >> 6] & bit != 0 {
                self.code_pages[idx >> 6] &= !bit;
                self.code_dirty.push(idx as u32);
            }
        }
    }

    /// Whether any translated page has been written since it was last
    /// drained. Checked before every block the JIT enters, so it is a plain
    /// emptiness test rather than the drain itself.
    #[inline(always)]
    pub fn has_dirty_code(&self) -> bool {
        !self.code_dirty.is_empty()
    }

    /// Take the pages whose translations are stale, clearing the list. Returns
    /// an empty (unallocated) vector in the overwhelmingly common case that
    /// nothing has written to code.
    pub fn dirty_code_pages(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.code_dirty)
    }

    /// Mark `[start, end)` as softly mapped: reads return zeros (served from
    /// a single shared page) and writes allocate a real page on first touch.
    /// This lets homebrew read its uninitialized address space without the
    /// host reserving the whole region up front.
    pub fn soft_map_zero(&mut self, start: u32, end: u32) {
        self.soft = (start, end);
    }

    /// Mark `[start, end)` as read-only to the guest: a CPU store into it
    /// faults instead of writing through. Adds to the existing set of
    /// read-only ranges (a retail process has one per loaded module) rather
    /// than replacing it. Loader-only writes (`map`/`map_zero`/`copy_range`)
    /// are unaffected — call this once a segment's relocations have been
    /// patched in, not before.
    pub fn mark_readonly(&mut self, start: u32, end: u32) {
        self.readonly.push((start, end));
        self.refresh_readonly_span();
    }

    /// Recompute the envelope `is_readonly` tests before it walks the list.
    /// Called by everything that changes the list, which is the only way the
    /// envelope can be wrong.
    fn refresh_readonly_span(&mut self) {
        self.readonly_span = self
            .readonly
            .iter()
            .fold((u32::MAX, 0), |(lo, hi), &(start, end)| (lo.min(start), hi.max(end)));
    }

    /// Forget every loaded module — both its protection and the memory
    /// states it is reported under — so a fresh boot in a reused [`Memory`]
    /// doesn't inherit either from a previous title/homebrew. The two are one
    /// call because a module image that outlived its protection, or the other
    /// way round, is a state no boot can produce.
    pub fn clear_modules(&mut self) {
        self.readonly.clear();
        self.refresh_readonly_span();
        self.modules.clear();
        self.refresh_module_span();
    }

    /// Drop every read-only range that lies inside `[start, end)`.
    ///
    /// This is the other half of [`Memory::mark_readonly`], and it exists
    /// because a module can go away while the process keeps running:
    /// `ldr:ro`'s `UnloadModule` frees the address space a loaded NRO
    /// occupied, and leaving its `.text` protected would fault the next thing
    /// to be mapped over it. Ranges that merely overlap are left alone —
    /// nothing here splits a protected range, and a partial unmap of somebody
    /// else's module is not something to guess at.
    pub fn unmark_readonly(&mut self, start: u32, end: u32) {
        self.readonly.retain(|&(s, e)| !(s >= start && e <= end));
        self.refresh_readonly_span();
    }

    /// Whether `addr` falls in a range marked by [`Memory::mark_readonly`] —
    /// in practice, always a loaded module's `.text`. Used by `svcQueryMemory`
    /// to report the real R-X permission code on it instead of a blanket RWX.
    #[inline(always)]
    pub fn is_readonly(&self, addr: u32) -> bool {
        addr >= self.readonly_span.0
            && addr < self.readonly_span.1
            && self.readonly.iter().any(|&(s, e)| addr >= s && addr < e)
    }

    /// Record a loaded module's image so `svcQueryMemory` can report the two
    /// memory states the kernel maps it under. Both ranges are end-exclusive
    /// and rounded out to whole pages, because a state is a property of a
    /// page.
    ///
    /// A module is not one region to Horizon. Its static half (`.text` +
    /// `.rodata`) is `Code` and its mutable half (`.data` + `.bss`) is
    /// `CodeData`, and `nn::ro::detail::GetExceptionInfo` — which every
    /// `nn::diag` log line and every abort goes through to name the module an
    /// address belongs to — reads the boundary between them as the module's
    /// shape: it walks `Code` up from the queried address, then *requires*
    /// the region immediately above that run to be `CodeData` and walks that
    /// to find the image's end. Reporting one state for the whole image left
    /// nothing above the run to be `CodeData`, and it aborted — which is how
    /// Asphalt 9 died on its first `puts`, 377M steps in, with an assertion
    /// whose text the release SDK had compiled out.
    pub fn mark_module(&mut self, static_range: (u32, u32), mutable_range: (u32, u32), alias: bool) {
        const PAGE: u32 = PAGE_SIZE as u32;
        let page_out = |(start, end): (u32, u32)| {
            (start & !(PAGE - 1), end.wrapping_add(PAGE - 1) & !(PAGE - 1))
        };
        self.modules.push(ModuleImage {
            static_range: page_out(static_range),
            mutable_range: page_out(mutable_range),
            alias,
        });
        self.refresh_module_span();
    }

    /// Drop every module image that lies inside `[start, end)` — the other
    /// half of [`Memory::mark_module`], for `ldr:ro`'s `UnloadModule`. Same
    /// rule as [`Memory::unmark_readonly`]: an image that merely overlaps is
    /// left alone rather than split.
    pub fn unmark_module(&mut self, start: u32, end: u32) {
        self.modules
            .retain(|m| !(m.static_range.0 >= start && m.mutable_range.1 <= end));
        self.refresh_module_span();
    }

    /// Recompute the envelope [`Memory::module_state`] tests before it walks
    /// the list. Called by everything that changes the list.
    fn refresh_module_span(&mut self) {
        self.module_span = self.modules.iter().fold((u32::MAX, 0), |(lo, hi), m| {
            (lo.min(m.static_range.0), hi.max(m.mutable_range.1))
        });
    }

    /// Which half of which module image `addr` falls in, if any.
    fn module_state(&self, addr: u32) -> Option<MemoryState> {
        if addr < self.module_span.0 || addr >= self.module_span.1 {
            return None;
        }
        self.modules.iter().find_map(|m| {
            if addr >= m.static_range.0 && addr < m.static_range.1 {
                Some(if m.alias { MemoryState::AliasCode } else { MemoryState::Code })
            } else if addr >= m.mutable_range.0 && addr < m.mutable_range.1 {
                Some(if m.alias { MemoryState::AliasCodeData } else { MemoryState::CodeData })
            } else {
                None
            }
        })
    }

    /// Whether any module image overlaps `[start, end)`.
    fn module_intersects(&self, start: u32, end: u32) -> bool {
        start < self.module_span.1
            && self.module_span.0 < end
            && self.modules.iter().any(|m| start < m.mutable_range.1 && m.static_range.0 < end)
    }

    #[inline(always)]
    fn check_writable(&self, addr: u32) -> Result<()> {
        if self.is_readonly(addr) {
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
        if self.mapped_pages >= self.max_mapped_pages {
            return Err(Error::Cpu(format!(
                "out of guest memory: exceeded the {} MiB cap",
                self.max_mapped_bytes() / (1024 * 1024)
            )));
        }
        self.pages[idx] = Some(Box::new([0u8; PAGE_SIZE]));
        self.mapped_pages += 1;
        self.block_mapped[idx / BLOCK_PAGES] += 1;
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

    /// The ceiling [`Memory::mapped_bytes`] may reach.
    pub fn max_mapped_bytes(&self) -> u64 {
        self.max_mapped_pages as u64 * PAGE_SIZE as u64
    }

    /// Choose a different ceiling, rounded down to whole pages.
    ///
    /// Lower it where the host has less to give than [`MAX_MAPPED_BYTES`]
    /// assumes. Raising it above what `svcGetInfo` advertises buys nothing:
    /// a title sizes itself from the advertised figure, not from this.
    pub fn set_max_mapped_bytes(&mut self, bytes: u64) {
        self.max_mapped_pages = (bytes / PAGE_SIZE as u64) as usize;
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

    /// Whether any range marked by [`Memory::mark_readonly`] overlaps
    /// `[start, end)`.
    fn readonly_intersects(&self, start: u32, end: u32) -> bool {
        self.readonly.iter().any(|&(s, e)| start < e && s < end)
    }

    /// The run of pages around `addr` that share its state — backed or not,
    /// read-only or not — clamped to `[0, limit)`. This is the region
    /// `svcQueryMemory` reports, and the two facts it reports about it.
    ///
    /// Finding the run means finding where the state changes, which the
    /// syscall used to do one 4 KiB page at a time. That is O(address space),
    /// and it consulted the read-only list for every page: a guest whose heap
    /// region is measured in gigabytes made each query walk hundreds of
    /// thousands of untouched pages, and a title that queries as it allocates
    /// spent more time inside `svcQueryMemory` than in its own code. Blocks
    /// with no backed page in them are skipped whole, so the answer is the
    /// same and the cost follows the number of *regions* rather than the size
    /// of the address space.
    pub fn state_run(&self, addr: u32, limit: u32) -> StateRun {
        const PAGE: u32 = PAGE_SIZE as u32;
        const BLOCK: u32 = (BLOCK_PAGES * PAGE_SIZE) as u32;
        // A run is bounded by any of the three facts changing, not just by
        // backing: a module's `.rodata` and its `.data` are both mapped and
        // both writable here, and they are still two regions to the guest.
        let state = |a: u32| {
            let mapped = self.page_mapped(a);
            let reported = self.module_state(a).unwrap_or(if mapped {
                MemoryState::Code
            } else {
                MemoryState::Unmapped
            });
            (mapped, self.is_readonly(a), reported)
        };
        let page = addr & !(PAGE - 1);
        let (mapped, readonly, reported) = state(page);
        let limit = limit & !(PAGE - 1);
        // A query past the end of the address space this emulator presents
        // describes the page it named and nothing around it. Guests probe up
        // there deliberately (hbmenu reads the failure to size the address
        // space), so it is an answer rather than an error.
        if page >= limit {
            return StateRun {
                start: page,
                end: page.saturating_add(PAGE),
                mapped,
                readonly,
                state: reported,
            };
        }
        // Only a run of *untouched, unprotected, unclaimed* pages can be
        // skipped a block at a time: that is what the summary knows about.
        // Every other run is bounded by something the page table has to be
        // asked about.
        let skippable = !mapped && !readonly && reported == MemoryState::Unmapped;
        let empty = |block_start: u32| {
            self.block_mapped[(block_start >> PAGE_BITS) as usize / BLOCK_PAGES] == 0
                && !self.readonly_intersects(block_start, block_start + BLOCK)
                && !self.module_intersects(block_start, block_start + BLOCK)
        };

        let mut start = page;
        while start > 0 {
            if skippable && start % BLOCK == 0 && start >= BLOCK && empty(start - BLOCK) {
                start -= BLOCK;
                continue;
            }
            if state(start - PAGE) != (mapped, readonly, reported) {
                break;
            }
            start -= PAGE;
        }
        let mut end = page + PAGE;
        while end < limit {
            if skippable && end % BLOCK == 0 && limit - end >= BLOCK && empty(end) {
                end += BLOCK;
                continue;
            }
            if state(end) != (mapped, readonly, reported) {
                break;
            }
            end += PAGE;
        }
        StateRun { start, end, mapped, readonly, state: reported }
    }

    /// Map `data` at `addr`, allocating pages as needed and zero-filling any
    /// gap between existing mappings. Wraps around page boundaries.
    pub fn map(&mut self, addr: u32, data: &[u8]) -> Result<()> {
        self.dirty_code_range(addr, data.len());
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
    ///
    /// Zero-filled including where a page was already backed. This used to
    /// allocate the pages and stop, which is the same thing only while every
    /// page is fresh — and the callers that most need the zeros are the ones
    /// where it is not. `.bss` shares its first page with `.data`, a recycled
    /// thread's TLS slot is a page some earlier thread already wrote, and
    /// `MapSharedMemory` promises the guest a cleared buffer. Each of those
    /// handed back whatever the last user left.
    ///
    /// Exactly `[addr, addr + size)`, never the whole of the end pages: the
    /// byte before `.bss` is `.data`'s, and it has already been loaded.
    pub fn map_zero(&mut self, addr: u32, size: usize) -> Result<()> {
        self.dirty_code_range(addr, size);
        let mut pos = addr as usize;
        let end = pos.saturating_add(size);
        while pos < end {
            let idx = pos >> PAGE_BITS;
            let off = pos & (PAGE_SIZE - 1);
            let n = (PAGE_SIZE - off).min(end - pos);
            // A page allocated here is already zero; only one that survived
            // from an earlier use has to be cleared.
            let backed = self.pages[idx].is_some();
            let page = self.page_mut(idx)?;
            if backed {
                page[off..off + n].fill(0);
            }
            pos += n;
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
        if addr < self.read_watch.1 && addr.wrapping_add(N as u32) > self.read_watch.0 {
            self.read_hit.set(Some(addr));
        }
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
        if addr < self.watch.1 && addr.wrapping_add(N as u32) > self.watch.0 {
            self.watch_hit = Some(addr);
        }
        self.note_code_write(addr);
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

    /// Read `len` little-endian bytes as one value: one page lookup per
    /// machine word rather than one per byte.
    ///
    /// This is the shape every pixel walk in the GPU wants. `read_u8` costs a
    /// page lookup each, so assembling a 4-byte pixel a byte at a time is four
    /// of them, and the walks that do it run over every pixel of a surface —
    /// a blit, a scan-out, a blend. Both `ExecCtx::read_pixel` and
    /// `Gpu::present` used to carry their own copy of this; only one of them
    /// had been fixed.
    #[inline(always)]
    pub fn read_le(&self, addr: u32, len: u32) -> Result<u128> {
        Ok(match len {
            1 => u128::from(self.read_u8(addr)?),
            2 => u128::from(self.read_u16(addr)?),
            4 => u128::from(self.read_u32(addr)?),
            8 => u128::from(self.read_u64(addr)?),
            16 => {
                u128::from(self.read_u64(addr)?)
                    | (u128::from(self.read_u64(addr.wrapping_add(8))?) << 64)
            }
            _ => self.read_le_odd(addr, len)?,
        })
    }

    /// The byte-at-a-time fallback for a width no accessor covers — 3-byte
    /// formats, and nothing else in practice.
    #[cold]
    #[inline(never)]
    fn read_le_odd(&self, addr: u32, len: u32) -> Result<u128> {
        let mut value = 0u128;
        for i in 0..len {
            value |= u128::from(self.read_u8(addr.wrapping_add(i))?) << (8 * i);
        }
        Ok(value)
    }

    /// Write `len` little-endian bytes of `value`: the counterpart of
    /// [`Memory::read_le`], and worth more, since `write_u8` re-scans the
    /// read-only ranges on every call and a byte loop pays for that per byte.
    #[inline(always)]
    pub fn write_le(&mut self, addr: u32, len: u32, value: u128) -> Result<()> {
        match len {
            1 => self.write_u8(addr, value as u8),
            2 => self.write_u16(addr, value as u16),
            4 => self.write_u32(addr, value as u32),
            8 => self.write_u64(addr, value as u64),
            16 => {
                self.write_u64(addr, value as u64)?;
                self.write_u64(addr.wrapping_add(8), (value >> 64) as u64)
            }
            _ => self.write_le_odd(addr, len, value),
        }
    }

    #[cold]
    #[inline(never)]
    fn write_le_odd(&mut self, addr: u32, len: u32, value: u128) -> Result<()> {
        for i in 0..len {
            self.write_u8(addr.wrapping_add(i), (value >> (8 * i)) as u8)?;
        }
        Ok(())
    }

    /// Write `count` copies of a `unit`-byte little-endian value to
    /// consecutive addresses, looking the page up **once** for the whole run.
    ///
    /// A clear is what wants this: it writes one value across a whole surface,
    /// and going through [`Memory::write_le`] per unit paid a page lookup and
    /// a read-only scan for each of the eleven million texels a 720p 2x2 MSAA
    /// target costs per frame. The GPU hands runs of at most a GOB's linear
    /// stretch, which never crosses a page — anything that would falls back to
    /// writing them one at a time.
    ///
    /// The watchpoint and the JIT's code-page bookkeeping are kept exactly as
    /// the per-unit path would leave them: a run that lands in the watched
    /// range reports its first address, and a run that touches translated code
    /// invalidates it.
    pub fn fill_le(&mut self, addr: u32, unit: u32, value: u128, count: u32) -> Result<()> {
        let span = (unit as usize) * (count as usize);
        let off = Self::in_page_offset(addr);
        if count == 0 {
            return Ok(());
        }
        if off + span > PAGE_SIZE || !matches!(unit, 1 | 2 | 4 | 8 | 16) {
            for i in 0..count {
                self.write_le(addr.wrapping_add(i * unit), unit, value)?;
            }
            return Ok(());
        }
        let end = addr.wrapping_add(span as u32);
        // The whole run shares a page, so it shares its protection too.
        self.check_writable(addr)?;
        // Stamped once into a pattern and then memcpy'd, rather than `count`
        // separate little copies: a GOB-sized clear run is 128 four-byte
        // writes done the obvious way, and one 512-byte copy done this way.
        const PATTERN: usize = 512;
        let unit = unit as usize;
        let bytes = value.to_le_bytes();
        let mut pattern = [0u8; PATTERN];
        let repeats = (PATTERN / unit).max(1);
        for i in 0..repeats {
            pattern[i * unit..(i + 1) * unit].copy_from_slice(&bytes[..unit]);
        }
        let stride = repeats * unit;
        let page = self.page_mut(Self::page_index(addr))?;
        let mut done = 0;
        while done < span {
            let n = stride.min(span - done);
            page[off + done..off + done + n].copy_from_slice(&pattern[..n]);
            done += n;
        }
        if addr < self.watch.1 && end > self.watch.0 {
            self.watch_hit = Some(addr);
        }
        self.note_code_write(addr);
        Ok(())
    }

    /// [`Memory::fill_le`] for a write that only owns some of each unit's
    /// bits: every unit keeps whatever `mask` does not select.
    ///
    /// A depth clear against a packed format is this. `Z24S8` clearing depth
    /// alone may not touch the stencil byte beside it, so the run cannot be
    /// filled — but the mask and the value are the same for every unit, so it
    /// is still one page lookup and a linear walk rather than a translation
    /// and a swizzle per texel.
    pub fn merge_le(
        &mut self,
        addr: u32,
        unit: u32,
        value: u128,
        mask: u128,
        count: u32,
    ) -> Result<()> {
        let span = (unit as usize) * (count as usize);
        let off = Self::in_page_offset(addr);
        if count == 0 {
            return Ok(());
        }
        if off + span > PAGE_SIZE || !matches!(unit, 1 | 2 | 4 | 8 | 16) {
            for i in 0..count {
                let at = addr.wrapping_add(i * unit);
                let old = self.read_le(at, unit)?;
                self.write_le(at, unit, (old & !mask) | (value & mask))?;
            }
            return Ok(());
        }
        let end = addr.wrapping_add(span as u32);
        self.check_writable(addr)?;
        let keep = (!mask).to_le_bytes();
        let set = (value & mask).to_le_bytes();
        let unit = unit as usize;
        let page = self.page_mut(Self::page_index(addr))?;
        let run = &mut page[off..off + span];
        // A unit at a time in its own width, rather than a byte at a time: a
        // depth clear merges four bytes per texel and 921,600 texels per
        // attachment, and doing that bytewise is four masks, four loads and
        // four stores where the hardware has one of each.
        match unit {
            4 => {
                let keep = u32::from_le_bytes([keep[0], keep[1], keep[2], keep[3]]);
                let set = u32::from_le_bytes([set[0], set[1], set[2], set[3]]);
                for slot in run.chunks_exact_mut(4) {
                    let old = u32::from_le_bytes([slot[0], slot[1], slot[2], slot[3]]);
                    slot.copy_from_slice(&((old & keep) | set).to_le_bytes());
                }
            }
            2 => {
                let keep = u16::from_le_bytes([keep[0], keep[1]]);
                let set = u16::from_le_bytes([set[0], set[1]]);
                for slot in run.chunks_exact_mut(2) {
                    let old = u16::from_le_bytes([slot[0], slot[1]]);
                    slot.copy_from_slice(&((old & keep) | set).to_le_bytes());
                }
            }
            1 => {
                for byte in run.iter_mut() {
                    *byte = (*byte & keep[0]) | set[0];
                }
            }
            _ => {
                for slot in run.chunks_exact_mut(unit) {
                    for (i, byte) in slot.iter_mut().enumerate() {
                        *byte = (*byte & keep[i]) | set[i];
                    }
                }
            }
        }
        if addr < self.watch.1 && end > self.watch.0 {
            self.watch_hit = Some(addr);
        }
        self.note_code_write(addr);
        Ok(())
    }

    /// Fetch the next instruction (little-endian AArch64 word).
    #[inline(always)]
    pub fn fetch(&self, pc: u32) -> Result<u32> {
        self.read_u32(pc)
    }

    /// Arm the write watchpoint over `[start, start + size)`; a zero `size`
    /// disarms it.
    pub fn watch_writes(&mut self, start: u32, size: u32) {
        self.watch = if size == 0 { (1, 0) } else { (start, start.wrapping_add(size)) };
        self.watch_hit = None;
    }

    /// Arm the read watchpoint over `[start, start + size)`; a zero `size`
    /// disarms it. Finding every piece of code that *examines* a flag is how
    /// the one that would clear it gets named, when nothing ever does.
    pub fn watch_reads(&mut self, start: u32, size: u32) {
        self.read_watch = if size == 0 { (1, 0) } else { (start, start.wrapping_add(size)) };
        self.read_hit.set(None);
    }

    /// The address of the most recent read inside the watched range, clearing
    /// it so the next call reports only new reads.
    pub fn take_read_hit(&self) -> Option<u32> {
        self.read_hit.replace(None)
    }

    /// The address of the most recent write inside the watched range, clearing
    /// it so the next call reports only new writes.
    pub fn take_watch_hit(&mut self) -> Option<u32> {
        self.watch_hit.take()
    }

    #[inline(always)]
    pub fn write_u8(&mut self, addr: u32, val: u8) -> Result<()> {
        self.check_writable(addr)?;
        let idx = Self::page_index(addr);
        let off = Self::in_page_offset(addr);
        let page = self.page_mut(idx)?;
        page[off] = val;
        if addr >= self.watch.0 && addr < self.watch.1 {
            self.watch_hit = Some(addr);
        }
        self.note_code_write(addr);
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

    /// Copy `buf` into guest memory at `addr` — [`Memory::read_into`] run
    /// backwards, one page at a time rather than one word at a time.
    ///
    /// What a render target's write-back needs. A 720p surface at 2x2 samples
    /// is 3.7 million texels, and putting one back through `write_le` is 3.7
    /// million bounds checks and page lookups for what is a handful of
    /// `copy_from_slice` calls.
    pub fn write_from(&mut self, addr: u32, buf: &[u8]) -> Result<()> {
        self.check_writable(addr)?;
        let mut pos = addr as usize;
        let end = pos.saturating_add(buf.len());
        let mut at = 0usize;
        while pos < end {
            let idx = pos >> PAGE_BITS;
            let off = pos & (PAGE_SIZE - 1);
            let n = (PAGE_SIZE - off).min(end - pos);
            let page = self.page_mut(idx)?;
            page[off..off + n].copy_from_slice(&buf[at..at + n]);
            pos += n;
            at += n;
        }
        // A surface is not code, but nothing here knows that, and the
        // translator has to be told about any write it did not see.
        self.dirty_code_range(addr, buf.len());
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
        self.dirty_code_range(addr, size);
        // Whole pages only, and in page indices: the address space is 4 GiB, so
        // byte counts do not fit a 32-bit usize on wasm.
        let first = (addr as u64 + PAGE_SIZE as u64 - 1) >> PAGE_BITS;
        let last = (addr as u64 + size as u64) >> PAGE_BITS;
        for idx in first..last.min(PAGE_COUNT as u64) {
            if self.pages[idx as usize].take().is_some() {
                self.mapped_pages -= 1;
                self.block_mapped[idx as usize / BLOCK_PAGES] -= 1;
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

    /// `map_zero` over ground somebody has already used has to hand back
    /// zeros, and has to stop at the range it was given — the page holding the
    /// first byte of `.bss` holds the last bytes of `.data` too.
    #[test]
    fn map_zero_clears_a_page_that_was_already_backed() {
        let mut m = Memory::new();
        m.map_zero(0x1000, PAGE_SIZE * 2).unwrap();
        for i in 0..PAGE_SIZE as u32 * 2 {
            m.write_u8(0x1000 + i, 0xAB).unwrap();
        }

        // A range starting part way into a live page, and running into the
        // next one.
        m.map_zero(0x1000 + 0x40, PAGE_SIZE).unwrap();
        assert_eq!(m.read_u8(0x1000 + 0x3F).unwrap(), 0xAB, "before the range");
        assert_eq!(m.read_u8(0x1000 + 0x40).unwrap(), 0, "the first byte of it");
        assert_eq!(
            m.read_u8(0x1000 + 0x40 + PAGE_SIZE as u32 - 1).unwrap(),
            0,
            "the last byte of it"
        );
        assert_eq!(
            m.read_u8(0x1000 + 0x40 + PAGE_SIZE as u32).unwrap(),
            0xAB,
            "after the range"
        );
        // And it is still the same two pages: clearing is not unmapping.
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
        // Against its own small cap rather than the default: what is being
        // tested is that the walk stops at the ceiling, and reaching the real
        // one would mean allocating gigabytes inside a unit test.
        const CAP: u64 = 4 * 1024 * 1024;
        m.set_max_mapped_bytes(CAP);
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
        assert_eq!(touched, CAP / PAGE_SIZE as u64);
        assert_eq!(m.mapped_bytes(), CAP);
    }

    #[test]
    fn a_region_scan_reports_exact_bounds_over_an_empty_address_space() {
        // The bounds are what `svcQueryMemory` hands the guest, and they have
        // to be right whether the run is four pages or three gigabytes. What
        // the block summary changes is the cost: this walks a 3.75 GiB space
        // whose untouched blocks are skipped whole, and every assertion below
        // is a boundary the old page-at-a-time scan would have found by
        // looking at each of a million pages.
        const LIMIT: u32 = 0xF000_0000;
        let mut m = Memory::new();
        m.soft_map_zero(0, LIMIT);
        m.map_zero(0x1000_0000, PAGE_SIZE * 4).unwrap();

        let run = m.state_run(0x1000_2000, LIMIT);
        assert_eq!((run.start, run.end), (0x1000_0000, 0x1000_4000));
        assert!(run.mapped);

        let run = m.state_run(0x0800_0000, LIMIT);
        assert_eq!((run.start, run.end), (0, 0x1000_0000));
        assert!(!run.mapped);

        let run = m.state_run(0x8000_0000, LIMIT);
        assert_eq!((run.start, run.end), (0x1000_4000, LIMIT));
        assert!(!run.mapped);

        // A page above the limit describes itself and stops. Guests read the
        // top of the address space on purpose.
        let run = m.state_run(LIMIT + 0x5000, LIMIT);
        assert_eq!((run.start, run.end), (LIMIT + 0x5000, LIMIT + 0x6000));

        // Freeing the pages puts the run back together, so the summary has to
        // come back down with them.
        m.unmap(0x1000_0000, PAGE_SIZE * 4);
        let run = m.state_run(0x1000_2000, LIMIT);
        assert_eq!((run.start, run.end), (0, LIMIT));
    }

    #[test]
    fn a_read_only_range_is_a_region_boundary() {
        // `.text` is mapped like the pages around it and only differs in being
        // write-protected, which `svcQueryMemory` reports as R-X. A scan that
        // skipped it along with the rest of a mapped run would tell `rtld`
        // that a module's code and its data are one region, and `rtld` finds
        // modules by looking for executable ones.
        const LIMIT: u32 = 0xF000_0000;
        let mut m = Memory::new();
        m.map_zero(0x0800_0000, PAGE_SIZE * 8).unwrap();
        m.mark_readonly(0x0800_2000, 0x0800_4000);

        let run = m.state_run(0x0800_2000, LIMIT);
        assert_eq!((run.start, run.end), (0x0800_2000, 0x0800_4000));
        assert!(run.mapped && run.readonly);

        let run = m.state_run(0x0800_0000, LIMIT);
        assert_eq!((run.start, run.end), (0x0800_0000, 0x0800_2000));
        assert!(run.mapped && !run.readonly);
    }

    #[test]
    fn a_module_is_two_memory_states_and_the_boundary_between_them_is_a_region() {
        // `nn::ro::detail::GetExceptionInfo` walks a module by its states: it
        // runs `Code` up from an address it was given, then requires the very
        // next region to be `CodeData` and runs that to find the image's end.
        // Reporting one state for the whole image leaves nothing above the
        // run to be `CodeData`, and it aborts — which is exactly what stopped
        // Asphalt 9 on its first `puts`.
        const LIMIT: u32 = 0xF000_0000;
        let mut m = Memory::new();
        // .text 2 pages, .rodata 2, .data + .bss 4.
        m.map_zero(0x0800_0000, PAGE_SIZE * 8).unwrap();
        m.mark_readonly(0x0800_0000, 0x0800_2000);
        m.mark_module((0x0800_0000, 0x0800_4000), (0x0800_4000, 0x0800_8000), false);

        let run = m.state_run(0x0800_0000, LIMIT);
        assert_eq!((run.start, run.end), (0x0800_0000, 0x0800_2000), ".text");
        assert_eq!(run.state, MemoryState::Code);
        assert!(run.readonly, ".text is the only part that is R-X");

        let run = m.state_run(0x0800_2000, LIMIT);
        assert_eq!((run.start, run.end), (0x0800_2000, 0x0800_4000), ".rodata");
        assert_eq!(run.state, MemoryState::Code);

        // The static half ends where the mutable half begins, even though
        // both are mapped and both are writable here.
        let run = m.state_run(0x0800_4000, LIMIT);
        assert_eq!((run.start, run.end), (0x0800_4000, 0x0800_8000), ".data + .bss");
        assert_eq!(run.state, MemoryState::CodeData);

        // An `ldr:ro` module carries the alias states instead, and gives them
        // back when it is unloaded.
        m.map_zero(0x2900_0000, PAGE_SIZE * 4).unwrap();
        m.mark_module((0x2900_0000, 0x2900_2000), (0x2900_2000, 0x2900_4000), true);
        assert_eq!(m.state_run(0x2900_0000, LIMIT).state, MemoryState::AliasCode);
        assert_eq!(m.state_run(0x2900_2000, LIMIT).state, MemoryState::AliasCodeData);
        m.unmark_module(0x2900_0000, 0x2900_4000);
        assert_eq!(m.state_run(0x2900_0000, LIMIT).state, MemoryState::Code);
        // ...and the process image is untouched by that.
        assert_eq!(m.state_run(0x0800_4000, LIMIT).state, MemoryState::CodeData);
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
