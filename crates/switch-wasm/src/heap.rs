//! The module's heap: std's own allocator, asking for memory in larger steps.
//!
//! On `wasm32-unknown-unknown` std allocates with `dlmalloc`, which grows
//! linear memory in its granularity, 64 KiB. Guest RAM is backed a 4 KiB page
//! at a time as the guest first touches it, so a title loading assets grows
//! memory once per sixteen pages: about 500 `memory.grow`s for the 32 MiB Just
//! Dance 2019 touches in its first seconds of play.
//!
//! A grow costs V8 time in proportion to the number of instances importing the
//! memory, because every one of them caches its size, and every block the JIT
//! emits is an instance of its own (see `jit.rs`). With the eight thousand a
//! retail title builds up, that walk was half the main thread while assets
//! loaded: 0.4 s of `SetInstanceMemory` over a 60-frame run.
//!
//! This is the same allocator with a granularity of [`GRANULARITY`], so the
//! same run grows eight times instead. What it costs is up to one granularity
//! of memory grown but not yet handed out, which the browser does not back
//! with anything until it is written.

use dlmalloc::Dlmalloc;
use std::alloc::{GlobalAlloc, Layout};
use std::cell::UnsafeCell;

/// Bytes linear memory grows by at least, whenever it has to grow at all.
const GRANULARITY: usize = 4 << 20;

struct Heap(UnsafeCell<Dlmalloc>);

// SAFETY: this target has no threads. The module is built without the
// `atomics` feature, so nothing can reach the heap from a second thread, which
// is the same argument `dlmalloc`'s own global allocator makes for taking no
// lock here.
unsafe impl Sync for Heap {}

#[global_allocator]
static HEAP: Heap = Heap(UnsafeCell::new(
    const {
        let mut heap = Dlmalloc::new();
        assert!(heap.set_granularity(GRANULARITY));
        heap
    },
));

// SAFETY: every method forwards to the one `Dlmalloc` with the caller's own
// layout, which is the contract `Dlmalloc`'s methods state, and the `Sync`
// argument above is what makes the `&mut` each call takes unique.
unsafe impl GlobalAlloc for Heap {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        (*self.0.get()).malloc(layout.size(), layout.align())
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        (*self.0.get()).free(ptr, layout.size(), layout.align())
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        (*self.0.get()).calloc(layout.size(), layout.align())
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        (*self.0.get()).realloc(ptr, layout.size(), layout.align(), new_size)
    }
}
