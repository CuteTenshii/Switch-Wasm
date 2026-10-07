//! The module's heap: std's dlmalloc with a larger growth granularity, to
//! reduce `memory.grow` calls.

use dlmalloc::Dlmalloc;
use std::alloc::{GlobalAlloc, Layout};
use std::cell::UnsafeCell;

const GRANULARITY: usize = 4 << 20;

struct Heap(UnsafeCell<Dlmalloc>);

// SAFETY: the module is built without `atomics`, so there is only one thread.
unsafe impl Sync for Heap {}

#[global_allocator]
static HEAP: Heap = Heap(UnsafeCell::new(
    const {
        let mut heap = Dlmalloc::new();
        assert!(heap.set_granularity(GRANULARITY));
        heap
    },
));

// SAFETY: forwards to `Dlmalloc` with the caller's layout; single-threaded per `Sync` above.
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
