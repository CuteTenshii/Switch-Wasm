//! The module's heap: std's dlmalloc with a larger growth granularity, to
//! reduce `memory.grow` calls, behind a spin lock because every worker on the
//! shared memory allocates from it.

use dlmalloc::Dlmalloc;
use std::alloc::{GlobalAlloc, Layout};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};

const GRANULARITY: usize = 4 << 20;

struct Heap {
    locked: AtomicBool,
    dlmalloc: UnsafeCell<Dlmalloc>,
}

// SAFETY: `dlmalloc` is only reached while `locked` is held.
unsafe impl Sync for Heap {}

#[global_allocator]
static HEAP: Heap = Heap {
    locked: AtomicBool::new(false),
    dlmalloc: UnsafeCell::new(
        const {
            let mut heap = Dlmalloc::new();
            assert!(heap.set_granularity(GRANULARITY));
            heap
        },
    ),
};

impl Heap {
    /// Run `f` on the allocator with the lock held; spun, since the holder never blocks.
    #[inline]
    fn with<T>(&self, f: impl FnOnce(&mut Dlmalloc) -> T) -> T {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
        // SAFETY: the lock is held, so no other reference exists.
        let out = f(unsafe { &mut *self.dlmalloc.get() });
        self.locked.store(false, Ordering::Release);
        out
    }
}

// SAFETY: forwards to `Dlmalloc` with the caller's layout, under the lock.
unsafe impl GlobalAlloc for Heap {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.with(|heap| heap.malloc(layout.size(), layout.align()))
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.with(|heap| heap.free(ptr, layout.size(), layout.align()))
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.with(|heap| heap.calloc(layout.size(), layout.align()))
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.with(|heap| heap.realloc(ptr, layout.size(), layout.align(), new_size))
    }
}
