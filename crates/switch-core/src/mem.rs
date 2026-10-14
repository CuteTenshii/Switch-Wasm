//! Sparse 4 GiB guest address space of lazily allocated 4 KiB pages.
//! Unmapped accesses fault, except in an optional soft region that reads as zero.

use crate::{Error, Result};
use std::cell::UnsafeCell;
use std::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed,
};
use std::sync::Mutex;

pub const PAGE_SIZE: usize = 4096;
pub const PAGE_BITS: u32 = 12;
pub const ADDRESS_SPACE_SIZE: u64 = 0x1_0000_0000; // 4 GiB
/// Computed in u64 so the 4 GiB constant survives a 32-bit usize.
const PAGE_COUNT: usize = (ADDRESS_SPACE_SIZE >> PAGE_BITS) as usize;
/// [`Memory::page_index`] masks with `PAGE_COUNT - 1`.
const _: () = assert!(PAGE_COUNT.is_power_of_two());
/// Pages per block-summary entry (2 MiB). See [`Memory::state_run`].
const BLOCK_PAGES: usize = 512;
const BLOCK_COUNT: usize = PAGE_COUNT / BLOCK_PAGES;
const WATCH_WORDS: usize = PAGE_COUNT / 64;
/// Default ceiling on host-backed guest RAM, bounding runaway writes. Must not
/// be below the `TotalMemorySize` that `svcGetInfo` advertises.
pub const MAX_MAPPED_BYTES: u64 = 0xC800_0000; // 3.125 GiB, `GUEST_TOTAL_MEMORY_SIZE`
const MAX_MAPPED_PAGES: usize = (MAX_MAPPED_BYTES / PAGE_SIZE as u64) as usize;

/// Horizon's `MemoryState` as `svcQueryMemory` reports it, for the states
/// distinguished here. See [`Memory::mark_module`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MemoryState {
    Unmapped = 0,
    Code = 3,
    CodeData = 4,
    /// The same two for an `ldr:ro` module.
    AliasCode = 8,
    AliasCodeData = 9,
}

/// A run of pages sharing a state, as `svcQueryMemory` describes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateRun {
    pub start: u32,
    /// End-exclusive, clamped to the requested limit.
    pub end: u32,
    /// Real storage, as opposed to untouched soft-mapped pages.
    pub mapped: bool,
    /// Write-protected, in practice a module's `.text`.
    pub readonly: bool,
    pub state: MemoryState,
}

#[derive(Debug, Clone, Copy)]
struct ModuleImage {
    /// `.text` + `.rodata`, end-exclusive and page-aligned.
    static_range: (u32, u32),
    /// `.data` + `.bss`, likewise.
    mutable_range: (u32, u32),
    /// An `ldr:ro` module, reported with the `Alias*` states.
    alias: bool,
}

/// Byte offsets of the fields emitted guest accesses read, taken here because
/// the fields are private. Not `#[repr(C)]`: valid for the build that emits the code.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Offsets {
    pub(crate) pages: u32,
    pub(crate) read_watch_lo: u32,
    pub(crate) read_watch_hi: u32,
    pub(crate) watch_lo: u32,
    pub(crate) watch_hi: u32,
    pub(crate) watched: u32,
}

impl Offsets {
    pub(crate) const OF_MEMORY: Offsets = Offsets {
        pages: std::mem::offset_of!(Memory, pages) as u32,
        read_watch_lo: Self::meta(std::mem::offset_of!(Meta, read_watch.0)),
        read_watch_hi: Self::meta(std::mem::offset_of!(Meta, read_watch.1)),
        watch_lo: Self::meta(std::mem::offset_of!(Meta, watch.0)),
        watch_hi: Self::meta(std::mem::offset_of!(Meta, watch.1)),
        watched: std::mem::offset_of!(Memory, watched_pages) as u32,
    };

    /// A [`Meta`] field's offset; `UnsafeCell` is transparent.
    const fn meta(field: usize) -> u32 {
        (std::mem::offset_of!(Memory, meta) + field) as u32
    }
}

/// A guest page. Cores share them, so bytes are reached through the cell.
type Page = UnsafeCell<[u8; PAGE_SIZE]>;

/// The cached-pages bitmap, one bit per page.
type WatchWords = [AtomicU64; WATCH_WORDS];

/// What changes only while no other core runs: loads, unloads and debugging.
#[derive(Debug)]
struct Meta {
    /// Soft region `(start, end)`, end-exclusive; `start > end` disables it.
    soft: (u32, u32),
    /// Read-only ranges, end-exclusive: each loaded module's `.text`, once patched.
    readonly: Vec<(u32, u32)>,
    /// Envelope of `readonly`; `start >= end` when empty. Lets most stores skip the list.
    readonly_span: (u32, u32),
    /// Every mapped module image, in load order.
    modules: Vec<ModuleImage>,
    /// Envelope of `modules`, like `readonly_span`.
    module_span: (u32, u32),
    /// Ceiling for the mapped pages, from [`MAX_MAPPED_BYTES`] unless lowered.
    max_mapped_pages: usize,
    /// Write watchpoint `[start, end)`; `start >= end` disables it.
    watch: (u32, u32),
    /// The same for reads.
    read_watch: (u32, u32),
    /// Shared memory mapped here, as `(first page, block)`; its pages are the block's.
    shared: Vec<(usize, std::sync::Arc<SharedPages>)>,
}

/// The last address a watchpoint saw, or none.
#[derive(Debug, Default)]
struct Hit(AtomicU64);

impl Hit {
    const NONE: u64 = u64::MAX;

    fn set(&self, addr: u32) {
        self.0.store(u64::from(addr), Relaxed);
    }

    fn take(&self) -> Option<u32> {
        match self.0.swap(Self::NONE, Relaxed) {
            Self::NONE => None,
            addr => Some(addr as u32),
        }
    }
}

/// Watched pages written since each drain last ran.
#[derive(Debug, Default)]
struct Dirty {
    code: Vec<u32>,
    gpu: Vec<u32>,
}

/// The guest address space, shared by every core. Page bytes and the counters
/// stores update are safe to touch from several at once; [`Meta`] and
/// unmapping are not, and happen only while the other cores are paused.
#[derive(Debug)]
pub struct Memory {
    /// One entry per page, null until written. A fixed-size array so indexing has no bounds check.
    pages: Box<[AtomicPtr<Page>; PAGE_COUNT]>,
    meta: UnsafeCell<Meta>,
    /// Shared zero page for soft-region reads.
    zero: Box<[u8; PAGE_SIZE]>,
    /// Pages holding real storage.
    mapped_pages: AtomicUsize,
    /// Backed pages per 2 MiB block, so scans skip untouched blocks.
    block_mapped: Box<[AtomicU32]>,
    /// The last write and read watchpoint hits.
    watch_hit: Hit,
    read_hit: Hit,
    /// One bit per page something has cached (JIT code, GPU textures); stores there
    /// are reported to both drains. Null until the first mark; fixed-size for emitted code.
    watched_pages: AtomicPtr<WatchWords>,
    /// Watched pages written since each drain, and whether either list is non-empty.
    dirty: Mutex<Dirty>,
    code_dirty: AtomicBool,
    gpu_dirty: AtomicBool,
    /// Whether the GPU backend has watched a page, so reports are not queued for nobody.
    gpu_watching: AtomicBool,
    /// Pages a host copy came from, and whether a store landed since [`Memory::take_copy_written`].
    copy_watch: AtomicPtr<WatchWords>,
    copy_written: AtomicBool,
    /// A second independent channel: bytes a host operation wrote (depth clears).
    fill_watch: AtomicPtr<WatchWords>,
    fill_written: AtomicBool,
}

// SAFETY: pages are reached through their cells and races between guest
// accesses are the guest's, as on the console; everything stores update is
// atomic or locked; `meta` and freeing pages are only touched while no other
// core runs.
unsafe impl Sync for Memory {}
unsafe impl Send for Memory {}

/// A process's address space as a core holds it: shared with the process, with the
/// `Memory` address first so emitted code reaches it in one load.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct AddressSpace {
    at: *const Memory,
    shared: std::sync::Arc<Memory>,
}

// SAFETY: `at` points into `shared`, which this keeps alive, and `Memory` is `Sync`.
unsafe impl Sync for AddressSpace {}
unsafe impl Send for AddressSpace {}

impl AddressSpace {
    pub fn new() -> Self {
        let shared = std::sync::Arc::new(Memory::new());
        Self {
            at: std::sync::Arc::as_ptr(&shared),
            shared,
        }
    }

    /// Where the core keeps the `Memory` address, from the start of an `AddressSpace`.
    pub(crate) const MEMORY_AT: u32 = std::mem::offset_of!(AddressSpace, at) as u32;
}

impl Default for AddressSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for AddressSpace {
    type Target = Memory;

    fn deref(&self) -> &Memory {
        &self.shared
    }
}

/// Pages several address spaces map at once: a shared memory object.
pub struct SharedPages {
    pages: Box<[Page]>,
}

// SAFETY: as for `Memory`'s pages, races on the bytes are the guest's.
unsafe impl Sync for SharedPages {}
unsafe impl Send for SharedPages {}

impl std::fmt::Debug for SharedPages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedPages({} pages)", self.pages.len())
    }
}

impl SharedPages {
    /// `size` bytes, rounded up to whole pages, zeroed.
    pub fn new(size: usize) -> std::sync::Arc<SharedPages> {
        let pages = std::iter::repeat_with(|| UnsafeCell::new([0u8; PAGE_SIZE]))
            .take(size.div_ceil(PAGE_SIZE))
            .collect();
        std::sync::Arc::new(SharedPages { pages })
    }

    /// Write `bytes` at `offset` into the block.
    pub fn write(&self, offset: u32, bytes: &[u8]) -> Result<()> {
        let end = offset as usize + bytes.len();
        if end > self.pages.len() * PAGE_SIZE {
            return Err(Error::Cpu(format!(
                "shared memory write at {offset:#x} runs past its {:#x} bytes",
                self.pages.len() * PAGE_SIZE
            )));
        }
        for (i, &byte) in bytes.iter().enumerate() {
            let at = offset as usize + i;
            // SAFETY: in bounds, checked above; races are the guest's, as for any page.
            unsafe { (*self.pages[at / PAGE_SIZE].get())[at % PAGE_SIZE] = byte };
        }
        Ok(())
    }

    pub fn write_u32(&self, offset: u32, value: u32) -> Result<()> {
        self.write(offset, &value.to_le_bytes())
    }

    pub fn read_u32(&self, offset: u32) -> Result<u32> {
        let at = offset as usize;
        if at + 4 > self.pages.len() * PAGE_SIZE {
            return Err(Error::Cpu(format!(
                "shared memory read at {offset:#x} is past its end"
            )));
        }
        let byte = |i: usize| {
            // SAFETY: in bounds, checked above.
            unsafe { (*self.pages[(at + i) / PAGE_SIZE].get())[(at + i) % PAGE_SIZE] }
        };
        Ok(u32::from_le_bytes([byte(0), byte(1), byte(2), byte(3)]))
    }

    pub fn write_u64(&self, offset: u32, value: u64) -> Result<()> {
        self.write(offset, &value.to_le_bytes())
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        let shared = self.shared_pages();
        for (idx, entry) in self.pages.iter().enumerate() {
            let page = entry.swap(std::ptr::null_mut(), Relaxed);
            if !page.is_null() && !shared.contains(&idx) {
                // SAFETY: allocated by `Box::into_raw` and unreachable from here on.
                drop(unsafe { Box::from_raw(page) });
            }
        }
        for words in [&self.watched_pages, &self.copy_watch, &self.fill_watch] {
            let words = words.swap(std::ptr::null_mut(), Relaxed);
            if !words.is_null() {
                // SAFETY: as above.
                drop(unsafe { Box::from_raw(words) });
            }
        }
    }
}

impl Default for Memory {
    fn default() -> Self {
        Memory::new()
    }
}

impl Memory {
    pub fn new() -> Memory {
        Memory {
            pages: std::iter::repeat_with(|| AtomicPtr::new(std::ptr::null_mut()))
                .take(PAGE_COUNT)
                .collect::<Box<[_]>>()
                .try_into()
                .expect("the page table is PAGE_COUNT long by construction"),
            meta: UnsafeCell::new(Meta {
                soft: (1, 0),
                readonly: Vec::new(),
                readonly_span: (u32::MAX, 0),
                modules: Vec::new(),
                module_span: (u32::MAX, 0),
                max_mapped_pages: MAX_MAPPED_PAGES,
                watch: (1, 0),
                read_watch: (1, 0),
                shared: Vec::new(),
            }),
            zero: Box::new([0u8; PAGE_SIZE]),
            mapped_pages: AtomicUsize::new(0),
            block_mapped: std::iter::repeat_with(|| AtomicU32::new(0))
                .take(BLOCK_COUNT)
                .collect(),
            watch_hit: Hit(AtomicU64::new(Hit::NONE)),
            read_hit: Hit(AtomicU64::new(Hit::NONE)),
            watched_pages: AtomicPtr::new(std::ptr::null_mut()),
            dirty: Mutex::default(),
            code_dirty: AtomicBool::new(false),
            gpu_dirty: AtomicBool::new(false),
            gpu_watching: AtomicBool::new(false),
            copy_watch: AtomicPtr::new(std::ptr::null_mut()),
            copy_written: AtomicBool::new(false),
            fill_watch: AtomicPtr::new(std::ptr::null_mut()),
            fill_written: AtomicBool::new(false),
        }
    }

    fn meta(&self) -> &Meta {
        // SAFETY: only written through `meta_mut`, while no other core runs.
        unsafe { &*self.meta.get() }
    }

    /// For the rare changes [`Meta`] holds. Callers run while the other cores are paused.
    #[allow(clippy::mut_from_ref)]
    fn meta_mut(&self) -> &mut Meta {
        // SAFETY: as for `meta`; nothing else holds a reference across this call.
        unsafe { &mut *self.meta.get() }
    }

    /// A lazily allocated bitmap: `slot`'s, allocated on first use.
    fn words(slot: &AtomicPtr<WatchWords>) -> &WatchWords {
        let mut words = slot.load(Relaxed);
        if words.is_null() {
            let fresh = Box::into_raw(
                std::iter::repeat_with(|| AtomicU64::new(0))
                    .take(WATCH_WORDS)
                    .collect::<Box<[_]>>()
                    .try_into()
                    .expect("the bitmap is WATCH_WORDS long by construction"),
            );
            words = match slot.compare_exchange(std::ptr::null_mut(), fresh, Relaxed, Relaxed) {
                Ok(_) => fresh,
                Err(won) => {
                    // SAFETY: never published.
                    drop(unsafe { Box::from_raw(fresh) });
                    won
                }
            };
        }
        // SAFETY: published bitmaps live as long as the memory.
        unsafe { &*words }
    }

    /// `slot`'s bitmap if one was allocated.
    fn words_if(slot: &AtomicPtr<WatchWords>) -> Option<&WatchWords> {
        // SAFETY: as for `words`.
        unsafe { slot.load(Relaxed).as_ref() }
    }

    /// Mark `addr`'s page as translated, so stores there are reported.
    pub fn mark_code_page(&self, addr: u32) {
        let idx = Self::page_index(addr);
        Self::words(&self.watched_pages)[idx >> 6].fetch_or(1u64 << (idx & 63), Relaxed);
    }

    /// Note a guest store; the miss path must stay a few instructions.
    #[inline(always)]
    fn note_code_write(&self, addr: u32) {
        let Some(words) = Self::words_if(&self.watched_pages) else {
            return;
        };
        let idx = Self::page_index(addr);
        if words[idx >> 6].load(Relaxed) & (1u64 << (idx & 63)) != 0 {
            self.mark_code_dirty(idx);
        }
    }

    /// Record a stale page; rare. Of cores storing to it at once, one reports it.
    #[cold]
    #[inline(never)]
    fn mark_code_dirty(&self, idx: usize) {
        let bit = 1u64 << (idx & 63);
        let Some(words) = Self::words_if(&self.watched_pages) else {
            return;
        };
        if words[idx >> 6].fetch_and(!bit, Relaxed) & bit != 0 {
            self.report_written(idx);
            let start = (idx << PAGE_BITS) as u32;
            if self.readonly_intersects(start, start.saturating_add(PAGE_SIZE as u32)) {
                words[idx >> 6].fetch_or(bit, Relaxed);
            }
        }
    }

    /// Report page `idx` as written; its `watched_pages` bit is already clear.
    fn report_written(&self, idx: usize) {
        {
            let mut dirty = self.dirty.lock().unwrap_or_else(|e| e.into_inner());
            dirty.code.push(idx as u32);
            self.code_dirty.store(true, Relaxed);
            if self.gpu_watching.load(Relaxed) {
                dirty.gpu.push(idx as u32);
                self.gpu_dirty.store(true, Relaxed);
            }
        }
        let bit = 1u64 << (idx & 63);
        for (watch, written) in [
            (&self.copy_watch, &self.copy_written),
            (&self.fill_watch, &self.fill_written),
        ] {
            if let Some(words) = Self::words_if(watch) {
                if words[idx >> 6].fetch_and(!bit, Relaxed) & bit != 0 {
                    written.store(true, Relaxed);
                }
            }
        }
    }

    /// Mark code pages in a range dirty, for whole-segment loader paths.
    fn dirty_code_range(&self, addr: u32, size: usize) {
        if Self::words_if(&self.watched_pages).is_none() || size == 0 {
            return;
        }
        let first = (addr as u64) >> PAGE_BITS;
        let last = (addr as u64 + size as u64 - 1) >> PAGE_BITS;
        for idx in first..=last.min(PAGE_COUNT as u64 - 1) {
            self.mark_code_dirty(idx as usize);
        }
    }

    /// Whether any translated page was written since the last drain.
    #[inline(always)]
    pub fn has_dirty_code(&self) -> bool {
        self.code_dirty.load(Relaxed)
    }

    /// Take the stale code pages, clearing the list.
    pub fn dirty_code_pages(&self) -> Vec<u32> {
        let mut dirty = self.dirty.lock().unwrap_or_else(|e| e.into_inner());
        self.code_dirty.store(false, Relaxed);
        std::mem::take(&mut dirty.code)
    }

    /// Watch `addr`'s page for the GPU backend.
    pub fn mark_gpu_page(&self, addr: u32) {
        self.gpu_watching.store(true, Relaxed);
        self.mark_code_page(addr);
    }

    /// Whether any GPU-watched page was written since the last drain.
    #[inline(always)]
    pub fn has_dirty_gpu(&self) -> bool {
        self.gpu_dirty.load(Relaxed)
    }

    pub fn dirty_gpu_pages(&self) -> Vec<u32> {
        let mut dirty = self.dirty.lock().unwrap_or_else(|e| e.into_inner());
        self.gpu_dirty.store(false, Relaxed);
        std::mem::take(&mut dirty.gpu)
    }

    /// Watch a range for a host copy; see [`Memory::take_copy_written`].
    pub fn mark_copy_range(&self, addr: u32, len: u32) {
        self.mark_watch_range(&self.copy_watch, addr, len);
    }

    /// Set `addr..addr + len`'s pages in `watch` and in the cached-pages bitmap.
    fn mark_watch_range(&self, watch: &AtomicPtr<WatchWords>, addr: u32, len: u32) {
        if len == 0 {
            return;
        }
        let (watched, words) = (Self::words(&self.watched_pages), Self::words(watch));
        let first = u64::from(addr) >> PAGE_BITS;
        let last = (u64::from(addr) + u64::from(len) - 1) >> PAGE_BITS;
        for idx in first..=last.min(PAGE_COUNT as u64 - 1) {
            let (word, bit) = ((idx >> 6) as usize, 1u64 << (idx & 63));
            watched[word].fetch_or(bit, Relaxed);
            words[word].fetch_or(bit, Relaxed);
        }
    }

    /// Whether a copy-watched page was written since last asked, clearing it.
    pub fn take_copy_written(&self) -> bool {
        self.copy_written.swap(false, Relaxed)
    }

    /// [`Memory::mark_copy_range`] for the second channel: bytes a host operation wrote.
    pub fn mark_fill_range(&self, addr: u32, len: u32) {
        self.mark_watch_range(&self.fill_watch, addr, len);
    }

    pub fn take_fill_written(&self) -> bool {
        self.fill_written.swap(false, Relaxed)
    }

    /// Soft-map `[start, end)`: reads return zeros, writes allocate a page.
    pub fn soft_map_zero(&self, start: u32, end: u32) {
        self.meta_mut().soft = (start, end);
    }

    /// Write-protect `[start, end)` against guest stores, adding to the existing
    /// ranges. Loader writes are unaffected; call after relocation.
    pub fn mark_readonly(&self, start: u32, end: u32) {
        self.meta_mut().readonly.push((start, end));
        self.refresh_readonly_span();
        // Emitted stores test only the cached-pages bitmap.
        if start < end {
            let watched = Self::words(&self.watched_pages);
            for idx in (start >> PAGE_BITS) as usize..=((end - 1) >> PAGE_BITS) as usize {
                watched[idx >> 6].fetch_or(1u64 << (idx & 63), Relaxed);
            }
        }
    }

    fn refresh_readonly_span(&self) {
        let meta = self.meta_mut();
        meta.readonly_span = meta
            .readonly
            .iter()
            .fold((u32::MAX, 0), |(lo, hi), &(start, end)| {
                (lo.min(start), hi.max(end))
            });
    }

    /// Forget every module's protection and memory states, for a fresh boot.
    pub fn clear_modules(&self) {
        self.meta_mut().readonly.clear();
        self.refresh_readonly_span();
        self.meta_mut().modules.clear();
        self.refresh_module_span();
    }

    /// Drop read-only ranges inside `[start, end)`, for `ldr:ro` unloads.
    /// Overlapping ranges are left alone.
    pub fn unmark_readonly(&self, start: u32, end: u32) {
        self.meta_mut()
            .readonly
            .retain(|&(s, e)| !(s >= start && e <= end));
        self.refresh_readonly_span();
    }

    /// Whether `addr` is in a read-only range, reported as R-X by `svcQueryMemory`.
    #[inline(always)]
    pub fn is_readonly(&self, addr: u32) -> bool {
        addr >= self.meta().readonly_span.0
            && addr < self.meta().readonly_span.1
            && self
                .meta()
                .readonly
                .iter()
                .any(|&(s, e)| addr >= s && addr < e)
    }

    /// Record a module image as `Code` (static half) and `CodeData` (mutable half)
    /// pages, as the kernel maps it. Ranges are end-exclusive and rounded to pages.
    pub fn mark_module(&self, static_range: (u32, u32), mutable_range: (u32, u32), alias: bool) {
        const PAGE: u32 = PAGE_SIZE as u32;
        let page_out = |(start, end): (u32, u32)| {
            (
                start & !(PAGE - 1),
                end.wrapping_add(PAGE - 1) & !(PAGE - 1),
            )
        };
        self.meta_mut().modules.push(ModuleImage {
            static_range: page_out(static_range),
            mutable_range: page_out(mutable_range),
            alias,
        });
        self.refresh_module_span();
    }

    /// Drop module images inside `[start, end)`; overlapping ones are left alone.
    pub fn unmark_module(&self, start: u32, end: u32) {
        self.meta_mut()
            .modules
            .retain(|m| !(m.static_range.0 >= start && m.mutable_range.1 <= end));
        self.refresh_module_span();
    }

    fn refresh_module_span(&self) {
        let meta = self.meta_mut();
        meta.module_span = meta.modules.iter().fold((u32::MAX, 0), |(lo, hi), m| {
            (lo.min(m.static_range.0), hi.max(m.mutable_range.1))
        });
    }

    fn module_state(&self, addr: u32) -> Option<MemoryState> {
        if addr < self.meta().module_span.0 || addr >= self.meta().module_span.1 {
            return None;
        }
        self.meta().modules.iter().find_map(|m| {
            if addr >= m.static_range.0 && addr < m.static_range.1 {
                Some(if m.alias {
                    MemoryState::AliasCode
                } else {
                    MemoryState::Code
                })
            } else if addr >= m.mutable_range.0 && addr < m.mutable_range.1 {
                Some(if m.alias {
                    MemoryState::AliasCodeData
                } else {
                    MemoryState::CodeData
                })
            } else {
                None
            }
        })
    }

    fn module_intersects(&self, start: u32, end: u32) -> bool {
        start < self.meta().module_span.1
            && self.meta().module_span.0 < end
            && self
                .meta()
                .modules
                .iter()
                .any(|m| start < m.mutable_range.1 && m.static_range.0 < end)
    }

    #[inline(always)]
    fn check_writable(&self, addr: u32) -> Result<()> {
        if self.is_readonly(addr) {
            Err(Error::Cpu(format!(
                "write to read-only address {:#010x}",
                addr
            )))
        } else {
            Ok(())
        }
    }

    /// The page a guest address falls in, masked so indexing skips the bounds check.
    #[inline]
    fn page_index(addr: u32) -> usize {
        ((addr as usize) >> PAGE_BITS) & (PAGE_COUNT - 1)
    }

    #[inline]
    fn in_page_offset(addr: u32) -> usize {
        (addr as usize) & (PAGE_SIZE - 1)
    }

    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    fn page_mut(&self, idx: usize) -> Result<&mut [u8; PAGE_SIZE]> {
        let mut page = self.pages[idx].load(Relaxed);
        if page.is_null() {
            page = self.allocate_page(idx)?;
        }
        // SAFETY: published pages live until unmapped, which only happens while
        // no other core runs; concurrent guest stores race as on the console.
        Ok(unsafe { &mut *(*page).get() })
    }

    /// First touch of a page, out of line. Cores touching it at once share one page.
    #[cold]
    #[inline(never)]
    fn allocate_page(&self, idx: usize) -> Result<*mut Page> {
        if self.mapped_pages.load(Relaxed) >= self.meta().max_mapped_pages {
            return Err(Error::Cpu(format!(
                "out of guest memory: exceeded the {} MiB cap",
                self.max_mapped_bytes() / (1024 * 1024)
            )));
        }
        let fresh = Box::into_raw(Box::new(UnsafeCell::new([0u8; PAGE_SIZE])));
        match self.pages[idx].compare_exchange(std::ptr::null_mut(), fresh, Relaxed, Relaxed) {
            Ok(_) => {
                self.mapped_pages.fetch_add(1, Relaxed);
                self.block_mapped[idx / BLOCK_PAGES].fetch_add(1, Relaxed);
                Ok(fresh)
            }
            Err(won) => {
                // SAFETY: never published.
                drop(unsafe { Box::from_raw(fresh) });
                Ok(won)
            }
        }
    }

    pub fn mapped_pages(&self) -> usize {
        self.mapped_pages.load(Relaxed)
    }

    /// Guest memory backed by host storage, in bytes.
    pub fn mapped_bytes(&self) -> u64 {
        self.mapped_pages() as u64 * PAGE_SIZE as u64
    }

    pub fn max_mapped_bytes(&self) -> u64 {
        self.meta().max_mapped_pages as u64 * PAGE_SIZE as u64
    }

    /// Choose a different ceiling, rounded down to whole pages.
    pub fn set_max_mapped_bytes(&self, bytes: u64) {
        self.meta_mut().max_mapped_pages = (bytes / PAGE_SIZE as u64) as usize;
    }

    #[inline(always)]
    fn page_ref(&self, idx: usize) -> Result<&[u8; PAGE_SIZE]> {
        match self.page(idx) {
            Some(page) => Ok(page),
            None => self.page_ref_unmapped(idx),
        }
    }

    /// Page `idx` if it is backed.
    #[inline(always)]
    fn page(&self, idx: usize) -> Option<&[u8; PAGE_SIZE]> {
        // SAFETY: as for `page_mut`.
        unsafe {
            self.pages
                .get(idx)?
                .load(Relaxed)
                .as_ref()
                .map(|page| &*page.get())
        }
    }

    /// Access to an unbacked page: zeros in the soft region, else a fault.
    #[cold]
    #[inline(never)]
    fn page_ref_unmapped(&self, idx: usize) -> Result<&[u8; PAGE_SIZE]> {
        let addr = idx << PAGE_BITS;
        let soft = self.meta().soft;
        if addr >= soft.0 as usize && addr < soft.1 as usize {
            return Ok(&self.zero);
        }
        Err(Error::Cpu(format!(
            "read from unmapped address {:#010x}",
            addr
        )))
    }

    /// Whether a real page backs `addr`, so `svcQueryMemory` walks see free pages.
    pub fn page_mapped(&self, addr: u32) -> bool {
        !self.pages[Self::page_index(addr)].load(Relaxed).is_null()
    }

    fn readonly_intersects(&self, start: u32, end: u32) -> bool {
        self.meta()
            .readonly
            .iter()
            .any(|&(s, e)| start < e && s < end)
    }

    /// Whether [`Memory::read_into`] matches per-access reads: no read watchpoint in range.
    pub fn plainly_readable(&self, addr: u32, len: u32) -> bool {
        let (start, end) = (u64::from(addr), u64::from(addr) + u64::from(len));
        !(start < u64::from(self.meta().read_watch.1) && u64::from(self.meta().read_watch.0) < end)
    }

    /// Whether [`Memory::write_from`] matches per-access writes: no protection or
    /// write watchpoint in range.
    pub fn plainly_writable(&self, addr: u32, len: u32) -> bool {
        let (start, end) = (u64::from(addr), u64::from(addr) + u64::from(len));
        let overlaps = |(s, e): (u32, u32)| start < u64::from(e) && u64::from(s) < end;
        !overlaps(self.meta().watch) && !self.meta().readonly.iter().any(|&range| overlaps(range))
    }

    /// The run of pages around `addr` sharing its state, clamped to `[0, limit)`,
    /// as `svcQueryMemory` reports. Untouched blocks are skipped whole.
    pub fn state_run(&self, addr: u32, limit: u32) -> StateRun {
        const PAGE: u32 = PAGE_SIZE as u32;
        const BLOCK: u32 = (BLOCK_PAGES * PAGE_SIZE) as u32;
        // A run ends when backing, protection, or reported state changes.
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
        // Past the limit, describe only the named page; hbmenu probes there.
        if page >= limit {
            return StateRun {
                start: page,
                end: page.saturating_add(PAGE),
                mapped,
                readonly,
                state: reported,
            };
        }
        // Only untouched, unprotected, unclaimed runs can skip whole blocks.
        let skippable = !mapped && !readonly && reported == MemoryState::Unmapped;
        let empty = |block_start: u32| {
            self.block_mapped[(block_start >> PAGE_BITS) as usize / BLOCK_PAGES].load(Relaxed) == 0
                && !self.readonly_intersects(block_start, block_start + BLOCK)
                && !self.module_intersects(block_start, block_start + BLOCK)
        };

        let mut start = page;
        while start > 0 {
            if skippable && start.is_multiple_of(BLOCK) && start >= BLOCK && empty(start - BLOCK) {
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
            if skippable && end.is_multiple_of(BLOCK) && limit - end >= BLOCK && empty(end) {
                end += BLOCK;
                continue;
            }
            if state(end) != (mapped, readonly, reported) {
                break;
            }
            end += PAGE;
        }
        StateRun {
            start,
            end,
            mapped,
            readonly,
            state: reported,
        }
    }

    /// Map `data` at `addr`, allocating pages and zero-filling gaps.
    pub fn map(&self, addr: u32, data: &[u8]) -> Result<()> {
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

    /// Zero exactly `[addr, addr + size)`, including already backed pages.
    pub fn map_zero(&self, addr: u32, size: usize) -> Result<()> {
        self.dirty_code_range(addr, size);
        let mut pos = addr as usize;
        let end = pos.saturating_add(size);
        while pos < end {
            let idx = pos >> PAGE_BITS;
            let off = pos & (PAGE_SIZE - 1);
            let n = (PAGE_SIZE - off).min(end - pos);
            // Only a reused page needs clearing.
            let backed = !self.pages[idx].load(Relaxed).is_null();
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

    /// The `N` bytes at `addr` if only the page table is needed, for JIT loads.
    /// `None` means the full read has more to do. Calls nothing.
    #[inline(always)]
    pub fn peek<const N: usize>(&self, addr: u32) -> Option<[u8; N]> {
        let off = Self::in_page_offset(addr);
        if off + N > PAGE_SIZE
            || (addr < self.meta().read_watch.1
                && addr.wrapping_add(N as u32) > self.meta().read_watch.0)
        {
            return None;
        }
        let page = self.page(Self::page_index(addr))?;
        Some(page[off..off + N].try_into().unwrap())
    }

    /// Write `val` if only the page table is needed, returning whether it did.
    #[inline(always)]
    pub fn poke<const N: usize>(&self, addr: u32, val: [u8; N]) -> bool {
        let off = Self::in_page_offset(addr);
        let idx = Self::page_index(addr);
        if off + N > PAGE_SIZE
            || self.is_readonly(addr)
            || (addr < self.meta().watch.1 && addr.wrapping_add(N as u32) > self.meta().watch.0)
            || self.watches_code(idx)
        {
            return false;
        }
        if self.pages[idx].load(Relaxed).is_null() {
            return false;
        }
        match self.page_mut(idx) {
            Ok(page) => {
                page[off..off + N].copy_from_slice(&val);
                true
            }
            Err(_) => false,
        }
    }

    /// [`Memory::poke`] for a pair in one page.
    #[inline(always)]
    pub fn poke_pair<const N: usize>(&self, addr: u32, first: [u8; N], second: [u8; N]) -> bool {
        let off = Self::in_page_offset(addr);
        let idx = Self::page_index(addr);
        if off + 2 * N > PAGE_SIZE
            || self.is_readonly(addr)
            || self.is_readonly(addr.wrapping_add(N as u32))
            || (addr < self.meta().watch.1 && addr.wrapping_add(2 * N as u32) > self.meta().watch.0)
            || self.watches_code(idx)
        {
            return false;
        }
        if self.pages[idx].load(Relaxed).is_null() {
            return false;
        }
        match self.page_mut(idx) {
            Ok(page) => {
                page[off..off + N].copy_from_slice(&first);
                page[off + N..off + 2 * N].copy_from_slice(&second);
                true
            }
            Err(_) => false,
        }
    }

    /// Whether page `idx` has had code translated since its last reported store.
    #[inline(always)]
    fn watches_code(&self, idx: usize) -> bool {
        Self::words_if(&self.watched_pages)
            .is_some_and(|words| words[idx >> 6].load(Relaxed) & (1u64 << (idx & 63)) != 0)
    }

    /// The `N` bytes at `addr` when they live in one page.
    #[inline(always)]
    fn read_bytes_in_page<const N: usize>(&self, addr: u32) -> Option<[u8; N]> {
        let off = Self::in_page_offset(addr);
        if off + N > PAGE_SIZE {
            return None;
        }
        let page = self.page_ref(Self::page_index(addr)).ok()?;
        if addr < self.meta().read_watch.1 && addr.wrapping_add(N as u32) > self.meta().read_watch.0
        {
            self.read_hit.set(addr);
        }
        Some(page[off..off + N].try_into().unwrap())
    }

    /// Same, for writing. `None` if the access straddles a page.
    #[inline(always)]
    fn write_bytes_in_page<const N: usize>(&self, addr: u32, val: [u8; N]) -> Option<()> {
        let off = Self::in_page_offset(addr);
        if off + N > PAGE_SIZE {
            return None;
        }
        self.check_writable(addr).ok()?;
        let page = self.page_mut(Self::page_index(addr)).ok()?;
        page[off..off + N].copy_from_slice(&val);
        if addr < self.meta().watch.1 && addr.wrapping_add(N as u32) > self.meta().watch.0 {
            self.watch_hit.set(addr);
        }
        self.note_code_write(addr);
        Some(())
    }

    /// Both halves of a pair in one page; the later half is the watch hit.
    #[inline(always)]
    fn read_pair_in_page<const N: usize>(&self, addr: u32) -> Option<([u8; N], [u8; N])> {
        let off = Self::in_page_offset(addr);
        if off + 2 * N > PAGE_SIZE {
            return None;
        }
        let page = self.page_ref(Self::page_index(addr)).ok()?;
        if addr < self.meta().read_watch.1
            && addr.wrapping_add(2 * N as u32) > self.meta().read_watch.0
        {
            self.read_hit
                .set(Self::later_half_hit(addr, N as u32, self.meta().read_watch));
        }
        Some((
            page[off..off + N].try_into().unwrap(),
            page[off + N..off + 2 * N].try_into().unwrap(),
        ))
    }

    /// [`Memory::read_pair_in_page`] for writing. `None` unless both halves can complete.
    #[inline(always)]
    fn write_pair_in_page<const N: usize>(
        &self,
        addr: u32,
        first: [u8; N],
        second: [u8; N],
    ) -> Option<()> {
        let off = Self::in_page_offset(addr);
        if off + 2 * N > PAGE_SIZE {
            return None;
        }
        self.check_writable(addr).ok()?;
        self.check_writable(addr.wrapping_add(N as u32)).ok()?;
        let page = self.page_mut(Self::page_index(addr)).ok()?;
        page[off..off + N].copy_from_slice(&first);
        page[off + N..off + 2 * N].copy_from_slice(&second);
        if addr < self.meta().watch.1 && addr.wrapping_add(2 * N as u32) > self.meta().watch.0 {
            self.watch_hit
                .set(Self::later_half_hit(addr, N as u32, self.meta().watch));
        }
        self.note_code_write(addr);
        Some(())
    }

    /// The half a watchpoint reports for a pair: the second when it overlaps.
    #[cold]
    #[inline(never)]
    fn later_half_hit(addr: u32, half: u32, range: (u32, u32)) -> u32 {
        let second = addr.wrapping_add(half);
        if second < range.1 && second.wrapping_add(half) > range.0 {
            second
        } else {
            addr
        }
    }

    /// The two `u64`s an `LDP` of X registers reads.
    #[inline(always)]
    pub fn read_u64_pair(&self, addr: u32) -> Result<(u64, u64)> {
        match self.read_pair_in_page::<8>(addr) {
            Some((a, b)) => Ok((u64::from_le_bytes(a), u64::from_le_bytes(b))),
            None => Ok((self.read_u64(addr)?, self.read_u64(addr.wrapping_add(8))?)),
        }
    }

    /// The two consecutive `u32`s an `LDP` of W registers reads.
    #[inline(always)]
    pub fn read_u32_pair(&self, addr: u32) -> Result<(u32, u32)> {
        match self.read_pair_in_page::<4>(addr) {
            Some((a, b)) => Ok((u32::from_le_bytes(a), u32::from_le_bytes(b))),
            None => Ok((self.read_u32(addr)?, self.read_u32(addr.wrapping_add(4))?)),
        }
    }

    /// Write two `u64`s, an `STP` of X registers, in order.
    #[inline(always)]
    pub fn write_u64_pair(&self, addr: u32, first: u64, second: u64) -> Result<()> {
        if self
            .write_pair_in_page(addr, first.to_le_bytes(), second.to_le_bytes())
            .is_some()
        {
            return Ok(());
        }
        self.write_u64(addr, first)?;
        self.write_u64(addr.wrapping_add(8), second)
    }

    /// Write two consecutive `u32`s, an `STP` of W registers.
    #[inline(always)]
    pub fn write_u32_pair(&self, addr: u32, first: u32, second: u32) -> Result<()> {
        if self
            .write_pair_in_page(addr, first.to_le_bytes(), second.to_le_bytes())
            .is_some()
        {
            return Ok(());
        }
        self.write_u32(addr, first)?;
        self.write_u32(addr.wrapping_add(4), second)
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

    /// Read `len` little-endian bytes as one value, a word per page lookup.
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

    /// Byte-at-a-time fallback, in practice 3-byte formats.
    #[cold]
    #[inline(never)]
    fn read_le_odd(&self, addr: u32, len: u32) -> Result<u128> {
        let mut value = 0u128;
        for i in 0..len {
            value |= u128::from(self.read_u8(addr.wrapping_add(i))?) << (8 * i);
        }
        Ok(value)
    }

    /// Write `len` little-endian bytes of `value`.
    #[inline(always)]
    pub fn write_le(&self, addr: u32, len: u32, value: u128) -> Result<()> {
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
    fn write_le_odd(&self, addr: u32, len: u32, value: u128) -> Result<()> {
        for i in 0..len {
            self.write_u8(addr.wrapping_add(i), (value >> (8 * i)) as u8)?;
        }
        Ok(())
    }

    /// Write `count` copies of a `unit`-byte value with one page lookup, keeping
    /// watchpoint and code-page bookkeeping as the per-unit path would.
    pub fn fill_le(&self, addr: u32, unit: u32, value: u128, count: u32) -> Result<()> {
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
        self.check_writable(addr)?;
        let unit = unit as usize;
        let bytes = value.to_le_bytes();
        let page = self.page_mut(Self::page_index(addr))?;
        let run = &mut page[off..off + span];
        // A unit at a time: a variable-length copy is a slow `memory.copy` in wasm.
        match unit {
            4 => {
                let v: [u8; 4] = bytes[..4].try_into().unwrap();
                for slot in run.as_chunks_mut::<4>().0 {
                    *slot = v;
                }
            }
            2 => {
                let v: [u8; 2] = bytes[..2].try_into().unwrap();
                for slot in run.as_chunks_mut::<2>().0 {
                    *slot = v;
                }
            }
            1 => run.fill(bytes[0]),
            8 => {
                let v: [u8; 8] = bytes[..8].try_into().unwrap();
                for slot in run.as_chunks_mut::<8>().0 {
                    *slot = v;
                }
            }
            _ => {
                for slot in run.as_chunks_mut::<16>().0 {
                    *slot = bytes;
                }
            }
        }
        if addr < self.meta().watch.1 && end > self.meta().watch.0 {
            self.watch_hit.set(addr);
        }
        self.note_code_write(addr);
        Ok(())
    }

    /// [`Memory::fill_le`] keeping the bits `mask` does not select, as a `Z24S8` depth clear.
    pub fn merge_le(
        &self,
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
        // A unit at a time in its own width, not bytewise.
        match unit {
            4 => {
                let keep = u32::from_le_bytes([keep[0], keep[1], keep[2], keep[3]]);
                let set = u32::from_le_bytes([set[0], set[1], set[2], set[3]]);
                for slot in run.as_chunks_mut::<4>().0 {
                    let old = u32::from_le_bytes([slot[0], slot[1], slot[2], slot[3]]);
                    slot.copy_from_slice(&((old & keep) | set).to_le_bytes());
                }
            }
            2 => {
                let keep = u16::from_le_bytes([keep[0], keep[1]]);
                let set = u16::from_le_bytes([set[0], set[1]]);
                for slot in run.as_chunks_mut::<2>().0 {
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
        if addr < self.meta().watch.1 && end > self.meta().watch.0 {
            self.watch_hit.set(addr);
        }
        self.note_code_write(addr);
        Ok(())
    }

    #[inline(always)]
    pub fn fetch(&self, pc: u32) -> Result<u32> {
        self.read_u32(pc)
    }

    /// Whether either watchpoint is armed; emitted code tests them only then.
    pub fn watchpoints_armed(&self) -> bool {
        self.meta().watch != (1, 0) || self.meta().read_watch != (1, 0)
    }

    /// Arm the write watchpoint over `[start, start + size)`; zero disarms it.
    pub fn watch_writes(&self, start: u32, size: u32) {
        self.meta_mut().watch = if size == 0 {
            (1, 0)
        } else {
            (start, start.wrapping_add(size))
        };
        self.watch_hit.take();
    }

    /// Arm the read watchpoint over `[start, start + size)`; zero disarms it.
    pub fn watch_reads(&self, start: u32, size: u32) {
        self.meta_mut().read_watch = if size == 0 {
            (1, 0)
        } else {
            (start, start.wrapping_add(size))
        };
        self.read_hit.take();
    }

    /// The last read hit, cleared.
    pub fn take_read_hit(&self) -> Option<u32> {
        self.read_hit.take()
    }

    /// The last write hit, cleared.
    pub fn take_watch_hit(&self) -> Option<u32> {
        self.watch_hit.take()
    }

    #[inline(always)]
    pub fn write_u8(&self, addr: u32, val: u8) -> Result<()> {
        self.check_writable(addr)?;
        let idx = Self::page_index(addr);
        let off = Self::in_page_offset(addr);
        let page = self.page_mut(idx)?;
        page[off] = val;
        if addr >= self.meta().watch.0 && addr < self.meta().watch.1 {
            self.watch_hit.set(addr);
        }
        self.note_code_write(addr);
        Ok(())
    }

    #[inline(always)]
    pub fn write_u16(&self, addr: u32, val: u16) -> Result<()> {
        if self.write_bytes_in_page(addr, val.to_le_bytes()).is_some() {
            return Ok(());
        }
        self.write_u16_straddling(addr, val)
    }

    /// The rare access that crosses a page boundary.
    #[cold]
    #[inline(never)]
    fn write_u16_straddling(&self, addr: u32, val: u16) -> Result<()> {
        self.write_u8(addr, val as u8)?;
        self.write_u8(addr.wrapping_add(1), (val >> 8) as u8)
    }

    #[inline(always)]
    pub fn write_u32(&self, addr: u32, val: u32) -> Result<()> {
        if self.write_bytes_in_page(addr, val.to_le_bytes()).is_some() {
            return Ok(());
        }
        self.write_u32_straddling(addr, val)
    }

    /// The rare access that crosses a page boundary.
    #[cold]
    #[inline(never)]
    fn write_u32_straddling(&self, addr: u32, val: u32) -> Result<()> {
        self.write_u8(addr, val as u8)?;
        self.write_u8(addr.wrapping_add(1), (val >> 8) as u8)?;
        self.write_u8(addr.wrapping_add(2), (val >> 16) as u8)?;
        self.write_u8(addr.wrapping_add(3), (val >> 24) as u8)
    }

    #[inline(always)]
    pub fn write_u64(&self, addr: u32, val: u64) -> Result<()> {
        if self.write_bytes_in_page(addr, val.to_le_bytes()).is_some() {
            return Ok(());
        }
        self.write_u64_straddling(addr, val)
    }

    /// The rare access that crosses a page boundary.
    #[cold]
    #[inline(never)]
    fn write_u64_straddling(&self, addr: u32, val: u64) -> Result<()> {
        self.write_u32(addr, val as u32)?;
        self.write_u32(addr.wrapping_add(4), (val >> 32) as u32)
    }

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

    /// Write `buf` with exactly the effect of a [`Memory::write_u8`] per byte, a page at a time.
    pub fn write_bytes(&self, addr: u32, buf: &[u8]) -> Result<()> {
        let mut done = 0usize;
        while done < buf.len() {
            let at = addr.wrapping_add(done as u32);
            let off = Self::in_page_offset(at);
            let n = (PAGE_SIZE - off).min(buf.len() - done);
            let end = u64::from(at) + n as u64;
            let protected = self
                .meta()
                .readonly
                .iter()
                .any(|&(s, e)| u64::from(at) < u64::from(e) && u64::from(s) < end);
            if protected {
                for (i, &byte) in buf[done..done + n].iter().enumerate() {
                    self.write_u8(at.wrapping_add(i as u32), byte)?;
                }
            } else {
                let page = self.page_mut(Self::page_index(at))?;
                page[off..off + n].copy_from_slice(&buf[done..done + n]);
                let (watch_start, watch_end) = (
                    u64::from(self.meta().watch.0),
                    u64::from(self.meta().watch.1),
                );
                if u64::from(at) < watch_end && end > watch_start {
                    self.watch_hit.set((end.min(watch_end) - 1) as u32);
                }
                self.note_code_write(at);
            }
            done += n;
        }
        Ok(())
    }

    /// Copy `buf` into guest memory a page at a time.
    pub fn write_from(&self, addr: u32, buf: &[u8]) -> Result<()> {
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
        // The translator must hear about writes it did not see.
        self.dirty_code_range(addr, buf.len());
        Ok(())
    }

    /// Copy `size` bytes from `src` to `dst` in place of `svcMapMemory` aliasing,
    /// which page storage here cannot share. Untouched source pages give zeros.
    pub fn copy_range(&self, dst: u32, src: u32, size: usize) -> Result<()> {
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

    /// Page indices whose storage belongs to a shared memory block.
    fn shared_pages(&self) -> Vec<usize> {
        self.meta()
            .shared
            .iter()
            .flat_map(|(first, block)| *first..first + block.pages.len())
            .collect()
    }

    /// Map shared memory `block` at page-aligned `addr`; stores through either side show
    /// in every address space that maps it. Runs while the other cores are paused.
    pub fn map_shared(&self, addr: u32, block: &std::sync::Arc<SharedPages>) -> Result<()> {
        let first = addr as usize >> PAGE_BITS;
        if addr as usize & (PAGE_SIZE - 1) != 0 || first + block.pages.len() > PAGE_COUNT {
            return Err(Error::Cpu(format!("shared memory cannot map at {addr:#x}")));
        }
        self.unmap(addr, block.pages.len() * PAGE_SIZE);
        for (i, page) in block.pages.iter().enumerate() {
            let idx = first + i;
            self.pages[idx].store(page as *const Page as *mut Page, Relaxed);
            self.mapped_pages.fetch_add(1, Relaxed);
            self.block_mapped[idx / BLOCK_PAGES].fetch_add(1, Relaxed);
        }
        self.meta_mut()
            .shared
            .push((first, std::sync::Arc::clone(block)));
        Ok(())
    }

    /// Drop whole backed pages in a range; partial end pages are kept.
    pub fn unmap(&self, addr: u32, size: usize) {
        self.dirty_code_range(addr, size);
        // In page indices: byte counts overflow a 32-bit usize.
        let first = (addr as u64 + PAGE_SIZE as u64 - 1) >> PAGE_BITS;
        let last = (addr as u64 + size as u64) >> PAGE_BITS;
        let shared = self.shared_pages();
        for idx in first..last.min(PAGE_COUNT as u64) {
            let idx = idx as usize;
            let page = self.pages[idx].swap(std::ptr::null_mut(), Relaxed);
            if page.is_null() {
                continue;
            }
            if !shared.contains(&idx) {
                // SAFETY: unmapping happens while no other core runs, so nothing holds the page.
                drop(unsafe { Box::from_raw(page) });
            }
            self.mapped_pages.fetch_sub(1, Relaxed);
            self.block_mapped[idx / BLOCK_PAGES].fetch_sub(1, Relaxed);
        }
        // A block stays alive while any of its pages is still mapped here.
        let pages = &self.pages;
        self.meta_mut().shared.retain(|(first, block)| {
            (*first..first + block.pages.len()).any(|idx| !pages[idx].load(Relaxed).is_null())
        });
    }

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
    fn a_shared_block_is_one_set_of_pages_and_outlives_a_mapping() {
        let block = SharedPages::new(2 * PAGE_SIZE);
        let a = Memory::new();
        let b = Memory::new();
        a.map_shared(0x1000_0000, &block).unwrap();
        b.map_shared(0x2000_0000, &block).unwrap();
        a.write_u32(0x1000_1004, 7).unwrap();
        assert_eq!(
            b.read_u32(0x2000_1004).unwrap(),
            7,
            "a store did not show in the other mapping"
        );
        block.write_u32(8, 9).unwrap();
        assert_eq!(
            a.read_u32(0x1000_0008).unwrap(),
            9,
            "the block's own write did not show"
        );

        a.unmap(0x1000_0000, 2 * PAGE_SIZE);
        assert!(
            a.read_u32(0x1000_1004).is_err(),
            "the unmapped range still reads"
        );
        drop(a);
        assert_eq!(
            b.read_u32(0x2000_1004).unwrap(),
            7,
            "unmapping freed the shared pages"
        );
    }

    #[test]
    fn mapped_bytes_counts_pages_as_they_are_backed() {
        let m = Memory::new();
        assert_eq!(m.mapped_bytes(), 0);
        m.map_zero(0x1000, PAGE_SIZE).unwrap();
        assert_eq!(m.mapped_bytes(), PAGE_SIZE as u64);
        m.write_u8(0x1FFF, 1).unwrap();
        assert_eq!(m.mapped_bytes(), PAGE_SIZE as u64);
        m.soft_map_zero(0x2000, 0x4000);
        assert_eq!(m.read_u8(0x2000).unwrap(), 0);
        assert_eq!(m.mapped_bytes(), PAGE_SIZE as u64);
        m.write_u8(0x2000, 7).unwrap();
        assert_eq!(m.mapped_bytes(), 2 * PAGE_SIZE as u64);
        assert_eq!(m.mapped_pages(), 2);
    }

    /// `map_zero` clears reused pages and stops at its range.
    #[test]
    fn map_zero_clears_a_page_that_was_already_backed() {
        let m = Memory::new();
        m.map_zero(0x1000, PAGE_SIZE * 2).unwrap();
        for i in 0..PAGE_SIZE as u32 * 2 {
            m.write_u8(0x1000 + i, 0xAB).unwrap();
        }

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
        // Clearing is not unmapping.
        assert_eq!(m.mapped_pages(), 2);
    }

    #[test]
    fn unmmapped_read_faults() {
        let m = Memory::new();
        assert!(m.read_u32(0xDEAD_0000).is_err());
    }

    #[test]
    fn runaway_soft_writes_fail_fast_at_the_ram_cap() {
        // A wild walk through a soft region stops at the RAM cap.
        let m = Memory::new();
        // A small cap, to avoid allocating gigabytes.
        const CAP: u64 = 4 * 1024 * 1024;
        m.set_max_mapped_bytes(CAP);
        m.soft_map_zero(0, 0x8000_0000);
        let mut addr = 0u32;
        let mut touched = 0u64;
        while m.write_u8(addr, 1).is_ok() {
            touched += 1;
            addr = addr.wrapping_add(PAGE_SIZE as u32);
        }
        assert_eq!(touched, CAP / PAGE_SIZE as u64);
        assert_eq!(m.mapped_bytes(), CAP);
    }

    #[test]
    fn a_region_scan_reports_exact_bounds_over_an_empty_address_space() {
        // Bounds must be right for runs of any size.
        const LIMIT: u32 = 0xF000_0000;
        let m = Memory::new();
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

        // A page above the limit describes itself.
        let run = m.state_run(LIMIT + 0x5000, LIMIT);
        assert_eq!((run.start, run.end), (LIMIT + 0x5000, LIMIT + 0x6000));

        m.unmap(0x1000_0000, PAGE_SIZE * 4);
        let run = m.state_run(0x1000_2000, LIMIT);
        assert_eq!((run.start, run.end), (0, LIMIT));
    }

    #[test]
    fn a_read_only_range_is_a_region_boundary() {
        // Write-protected `.text` is its own region, as `rtld` relies on.
        const LIMIT: u32 = 0xF000_0000;
        let m = Memory::new();
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
        // `Code` must be followed directly by `CodeData`.
        const LIMIT: u32 = 0xF000_0000;
        let m = Memory::new();
        // .text 2 pages, .rodata 2, .data + .bss 4.
        m.map_zero(0x0800_0000, PAGE_SIZE * 8).unwrap();
        m.mark_readonly(0x0800_0000, 0x0800_2000);
        m.mark_module(
            (0x0800_0000, 0x0800_4000),
            (0x0800_4000, 0x0800_8000),
            false,
        );

        let run = m.state_run(0x0800_0000, LIMIT);
        assert_eq!((run.start, run.end), (0x0800_0000, 0x0800_2000), ".text");
        assert_eq!(run.state, MemoryState::Code);
        assert!(run.readonly, ".text is the only part that is R-X");

        let run = m.state_run(0x0800_2000, LIMIT);
        assert_eq!((run.start, run.end), (0x0800_2000, 0x0800_4000), ".rodata");
        assert_eq!(run.state, MemoryState::Code);

        let run = m.state_run(0x0800_4000, LIMIT);
        assert_eq!(
            (run.start, run.end),
            (0x0800_4000, 0x0800_8000),
            ".data + .bss"
        );
        assert_eq!(run.state, MemoryState::CodeData);

        // `ldr:ro` modules use the alias states and drop them on unload.
        m.map_zero(0x2900_0000, PAGE_SIZE * 4).unwrap();
        m.mark_module((0x2900_0000, 0x2900_2000), (0x2900_2000, 0x2900_4000), true);
        assert_eq!(
            m.state_run(0x2900_0000, LIMIT).state,
            MemoryState::AliasCode
        );
        assert_eq!(
            m.state_run(0x2900_2000, LIMIT).state,
            MemoryState::AliasCodeData
        );
        m.unmark_module(0x2900_0000, 0x2900_4000);
        assert_eq!(m.state_run(0x2900_0000, LIMIT).state, MemoryState::Code);
        assert_eq!(m.state_run(0x0800_4000, LIMIT).state, MemoryState::CodeData);
    }

    #[test]
    fn readonly_region_rejects_guest_writes_but_not_reads() {
        let m = Memory::new();
        m.map_zero(0x1000, PAGE_SIZE).unwrap();
        m.write_u32(0x1000, 0x1111_1111).unwrap();
        m.mark_readonly(0x1000, 0x2000);
        assert!(m.write_u32(0x1000, 0x2222_2222).is_err());
        assert!(m.write_u8(0x1FFF, 1).is_err());
        assert_eq!(m.read_u32(0x1000).unwrap(), 0x1111_1111);
        m.map_zero(0x2000, PAGE_SIZE).unwrap();
        m.write_u32(0x2000, 3).unwrap();
        assert_eq!(m.read_u32(0x2000).unwrap(), 3);
    }

    /// Emitted stores see write protection only through the cached-pages bit.
    #[test]
    fn a_read_only_page_stays_watched_through_loader_writes() {
        let m = Memory::new();
        m.map_zero(0x1000, 3 * PAGE_SIZE).unwrap();
        m.mark_readonly(0x1800, 0x2800);
        assert!(!m.poke(0x1000, [0u8; 4]));
        assert!(!m.poke(0x2000, [0u8; 4]));
        assert!(m.poke(0x3000, [0u8; 4]));
        m.map(0x1000, &[1; 2 * PAGE_SIZE]).unwrap();
        assert!(m.watches_code(1) && m.watches_code(2));
    }

    #[test]
    fn map_across_page_boundary() {
        let m = Memory::new();
        let data = (0..=255u8).collect::<Vec<_>>();
        let addr = 0x0000_1FF0; // starts 16 bytes before a page boundary
        m.map(addr, &data).unwrap();
        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(m.read_u8(addr + i as u32).unwrap(), expected);
        }
    }

    #[test]
    fn zero_fill_between_maps() {
        let m = Memory::new();
        m.map_zero(0x0000_0000, PAGE_SIZE).unwrap();
        m.map_zero(0x0000_3000, PAGE_SIZE).unwrap();
        assert_eq!(m.read_u8(0x0000_0000).unwrap(), 0);
        assert_eq!(m.read_u8(0x0000_3000).unwrap(), 0);
        assert!(m.read_u8(0x0000_1000).is_err());
    }

    #[test]
    fn u32_u64_roundtrip() {
        let m = Memory::new();
        m.map_zero(0x0000_0000, 16).unwrap();
        m.write_u32(0, 0xDEAD_BEEF).unwrap();
        assert_eq!(m.read_u32(0).unwrap(), 0xDEAD_BEEF);
        m.write_u64(8, 0x1234_5678_9ABC_DEF0).unwrap();
        assert_eq!(m.read_u64(8).unwrap(), 0x1234_5678_9ABC_DEF0);
    }

    /// Single-lookup and split pair paths must agree.
    #[test]
    fn pair_accesses_match_two_single_accesses() {
        let m = Memory::new();
        m.map_zero(0x1000, 2 * PAGE_SIZE).unwrap();
        for addr in [0x1100, 0x1FF8, 0x1FFC] {
            m.write_u64_pair(addr, 0x1111_2222_3333_4444, 0x5555_6666_7777_8888)
                .unwrap();
            assert_eq!(m.read_u64(addr).unwrap(), 0x1111_2222_3333_4444);
            assert_eq!(m.read_u64(addr + 8).unwrap(), 0x5555_6666_7777_8888);
            assert_eq!(
                m.read_u64_pair(addr).unwrap(),
                (0x1111_2222_3333_4444, 0x5555_6666_7777_8888)
            );

            m.write_u32_pair(addr, 0x9999_AAAA, 0xBBBB_CCCC).unwrap();
            assert_eq!(m.read_u32(addr).unwrap(), 0x9999_AAAA);
            assert_eq!(m.read_u32(addr + 4).unwrap(), 0xBBBB_CCCC);
            assert_eq!(m.read_u32_pair(addr).unwrap(), (0x9999_AAAA, 0xBBBB_CCCC));
        }
        assert!(m.read_u64_pair(0x2FF8).is_err());
        assert!(m.read_u32_pair(0x2FFC).is_err());
    }

    /// A pair is two stores: the first lands when the second faults.
    #[test]
    fn a_pair_whose_second_half_is_read_only_still_writes_the_first() {
        let m = Memory::new();
        m.map_zero(0x1000, PAGE_SIZE).unwrap();
        m.mark_readonly(0x1108, 0x1110);
        assert!(m.write_u64_pair(0x1100, 7, 8).is_err());
        assert_eq!(m.read_u64(0x1100).unwrap(), 7);
        assert_eq!(m.read_u64(0x1108).unwrap(), 0);
    }

    /// A pair's watch hit is the half that would come second.
    #[test]
    fn a_watched_pair_reports_the_half_two_accesses_would() {
        let m = Memory::new();
        m.map_zero(0x1000, PAGE_SIZE).unwrap();
        for (start, size, expected) in [
            (0x1100, 8, 0x1100),
            (0x1108, 8, 0x1108),
            (0x1100, 16, 0x1108),
        ] {
            m.watch_writes(start, size);
            m.write_u64_pair(0x1100, 1, 2).unwrap();
            assert_eq!(m.take_watch_hit(), Some(expected));
            m.watch_reads(start, size);
            m.read_u64_pair(0x1100).unwrap();
            assert_eq!(m.take_read_hit(), Some(expected));
        }
        m.watch_writes(0x1110, 8);
        m.write_u64_pair(0x1100, 1, 2).unwrap();
        assert_eq!(m.take_watch_hit(), None);
    }

    /// `write_bytes` matches a `write_u8` loop in memory, fault, and watch hit.
    #[test]
    fn a_bulk_write_stops_and_reports_where_bytewise_writes_would() {
        let data: Vec<u8> = (1..=0x40u8).collect();
        let bytewise = |m: &Memory, at: u32| -> Result<()> {
            for (i, &b) in data.iter().enumerate() {
                m.write_u8(at + i as u32, b)?;
            }
            Ok(())
        };
        let setup = || {
            let m = Memory::new();
            m.map_zero(0x1000, 2 * PAGE_SIZE).unwrap();
            m.watch_writes(0x1FF0, 0x20);
            m
        };
        let (bulk, mut single) = (setup(), setup());
        bulk.write_bytes(0x1FE0, &data).unwrap();
        bytewise(&mut single, 0x1FE0).unwrap();
        assert_eq!(
            bulk.dump(0x1000, 2 * PAGE_SIZE),
            single.dump(0x1000, 2 * PAGE_SIZE)
        );
        assert_eq!(bulk.take_watch_hit(), single.take_watch_hit());
        assert_eq!(bulk.take_watch_hit(), None);

        let setup = || {
            let m = Memory::new();
            m.map_zero(0x1000, PAGE_SIZE).unwrap();
            m.mark_readonly(0x1110, 0x1200);
            m
        };
        let (bulk, mut single) = (setup(), setup());
        assert!(bulk.write_bytes(0x1100, &data).is_err());
        assert!(bytewise(&mut single, 0x1100).is_err());
        assert_eq!(bulk.dump(0x1000, PAGE_SIZE), single.dump(0x1000, PAGE_SIZE));
        assert_eq!(bulk.read_u8(0x110F).unwrap(), 0x10);
        assert_eq!(bulk.read_u8(0x1110).unwrap(), 0);
    }

    #[test]
    fn dump_matches_mapped_bytes() {
        let m = Memory::new();
        let data = (0..=127u8).collect::<Vec<_>>();
        m.map(0x0001_0000, &data).unwrap();
        assert_eq!(m.dump(0x0001_0000, 128).unwrap(), data);
    }

    /// The JIT and GPU drains are independent.
    #[test]
    fn a_write_reaches_both_drains_whichever_asks_first() {
        let mem = Memory::new();
        mem.map_zero(0x1000, 0x2000).unwrap();
        mem.mark_gpu_page(0x1000);
        mem.write_u32(0x1000, 1).unwrap();
        assert!(mem.has_dirty_gpu());
        assert!(mem.has_dirty_code());
        assert_eq!(mem.dirty_gpu_pages().len(), 1);
        assert!(!mem.has_dirty_gpu());
        assert_eq!(mem.dirty_code_pages().len(), 1);

        // A page is reported once until cached again.
        mem.write_u32(0x1000, 2).unwrap();
        assert!(!mem.has_dirty_gpu());
        mem.mark_gpu_page(0x1000);
        mem.write_u32(0x1000, 3).unwrap();
        assert_eq!(mem.dirty_gpu_pages().len(), 1);
    }

    // The mask in `Memory::page_index` is a no-op for every `u32` address.
    const _: () = assert!((u32::MAX as usize) >> PAGE_BITS == PAGE_COUNT - 1);
}
