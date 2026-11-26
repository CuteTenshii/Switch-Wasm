//! Guest loads and stores: page checks, pairs and the exclusive monitor.

use super::*;

impl Emitter<'_> {
    /// Leave an access's guest address in `L_ADDR` and any writeback value in
    /// `L_W`, per [`crate::cpu::Cpu::indexed`].
    pub(super) fn address(&mut self, rn: u8, offset: i64, wb: Wb) {
        if matches!(wb, Wb::None) {
            self.read_reg_raw(rn);
            if offset != 0 {
                self.f.i64_const(offset);
                self.f.i64_add();
            }
        } else {
            self.read_reg_raw(rn);
            self.f.local_tee(L_A);
            self.f.i64_const(offset);
            self.f.i64_add();
            self.f.local_set(L_W);
            match wb {
                Wb::Post => self.f.local_get(L_A),
                _ => self.f.local_get(L_W),
            }
        }
        self.set_addr64();
    }

    /// Take the `i64` address on the stack as the access's, whole and in its low bits.
    pub(super) fn set_addr64(&mut self) {
        self.f.local_tee(L_ADDR64);
        self.f.i32_wrap_i64();
        self.f.local_set(L_ADDR);
    }

    /// The same for an `i32` address: an A32 one, or a literal below 4 GiB.
    pub(super) fn set_addr32(&mut self) {
        self.f.local_tee(L_ADDR);
        self.f.i64_extend_i32_u();
        self.f.local_set(L_ADDR64);
    }

    /// The same for the register-offset form, per
    /// [`crate::cpu::Cpu::reg_offset`].
    pub(super) fn address_reg(&mut self, rn: u8, rm: u8, ext: Ext, shift: u8) {
        self.read_reg_raw(rn);
        self.read_reg_raw(rm);
        match ext {
            Ext::Uxtw => self.mask_to(false),
            Ext::Sxtw => self.f.i64_extend32_s(),
            Ext::None => {}
        }
        if shift != 0 {
            self.f.i64_const(i64::from(shift));
            self.f.i64_shl();
        }
        self.f.i64_add();
        self.set_addr64();
    }

    /// Write `L_W` back to the base register, for the modes that have one.
    pub(super) fn write_back(&mut self, rn: u8, wb: Wb) {
        if !matches!(wb, Wb::None) {
            self.f.local_get(L_W);
            self.store_reg(rn);
        }
    }

    /// Leave the page `L_ADDR` is on in `L_PAGE`, or leave the block if
    /// reading `n` bytes there needs more than the page table
    /// ([`crate::mem::Memory::peek`]'s checks).
    pub(super) fn page_for_read(&mut self, n: u32, retired: usize) {
        self.page_number();
        let crosses = n > 1;
        if crosses {
            self.crosses_page(n);
        }
        if self.watches {
            self.covers(n, self.layout.read_watch_lo, self.layout.read_watch_hi);
            if crosses {
                self.f.i32_or();
            }
        }
        if crosses || self.watches {
            self.deopt_if(retired);
        }
        self.page_or_defer(retired);
    }

    /// The same for a store, with [`crate::mem::Memory::poke`]'s checks.
    pub(super) fn page_for_write(&mut self, n: u32, retired: usize) {
        self.page_number();
        let crosses = n > 1;
        if crosses {
            self.crosses_page(n);
        }
        if self.watches {
            self.covers(n, self.layout.watch_lo, self.layout.watch_hi);
            if crosses {
                self.f.i32_or();
            }
        }
        if crosses || self.watches {
            self.deopt_if(retired);
        }
        self.owes_a_report(retired);
        self.page_or_defer(retired);
    }

    /// Leave the table block the access is in in `L_BLOCK`, and the page's index within it
    /// in `L_IDX`. Address bits above the 39-bit space are dropped, as `Memory` drops them;
    /// a block nothing was backed in is a real, empty one, so this needs no test.
    pub(super) fn page_number(&mut self) {
        self.env(self.layout.blocks);
        self.f.local_get(L_ADDR64);
        self.f.i64_const(i64::from(PAGE_BITS + BLOCK_BITS));
        self.f.i64_shr_u();
        self.f.i32_wrap_i64();
        self.f.i32_const((BLOCK_COUNT - 1) as i32);
        self.f.i32_and();
        self.f.i32_const(PAGE_ENTRY_SHIFT);
        self.f.i32_shl();
        self.f.i32_add();
        self.f.i32_load(ALIGN_4, 0);
        self.f.local_set(L_BLOCK);
        self.f.local_get(L_ADDR);
        self.f.i32_const(PAGE_BITS as i32);
        self.f.i32_shr_u();
        self.f.i32_const((BLOCK_PAGES - 1) as i32);
        self.f.i32_and();
        self.f.local_set(L_IDX);
    }

    /// Leave that page's storage in `L_PAGE`, or leave the block if it is null.
    pub(super) fn page_or_defer(&mut self, retired: usize) {
        self.f.local_get(L_BLOCK);
        self.f.local_get(L_IDX);
        self.f.i32_const(PAGE_ENTRY_SHIFT);
        self.f.i32_shl();
        self.f.i32_add();
        self.f.i32_load(ALIGN_4, self.layout.block_pages);
        self.f.local_tee(L_PAGE);
        self.f.i32_eqz();
        self.deopt_if(retired);
    }

    /// Leave the block if anything has cached page `L_IDX`'s contents.
    pub(super) fn owes_a_report(&mut self, retired: usize) {
        self.f.local_get(L_BLOCK);
        self.f.local_get(L_IDX);
        self.f.i32_const(6);
        self.f.i32_shr_u();
        self.f.i32_const(3);
        self.f.i32_shl();
        self.f.i32_add();
        self.f.i64_load(ALIGN_8, self.layout.block_watched);
        // The shift is taken modulo 64, masking the bit index.
        self.f.i64_const(1);
        self.f.local_get(L_IDX);
        self.f.i64_extend_i32_u();
        self.f.i64_shl();
        self.f.i64_and();
        self.f.i64_const(0);
        self.f.i64_ne();
        self.deopt_if(retired);
    }

    /// Push whether the `n` bytes at `L_ADDR` run past the end of their page.
    pub(super) fn crosses_page(&mut self, n: u32) {
        self.f.local_get(L_ADDR);
        self.f.i32_const((PAGE_SIZE - 1) as i32);
        self.f.i32_and();
        self.f.i32_const((PAGE_SIZE as u32 - n) as i32);
        self.f.i32_gt_u();
    }

    /// Push whether the watchpoint at `lo`/`hi` overlaps the `n` bytes at
    /// `L_ADDR64`, wrapping included. Only regions emitted while one is armed test it.
    pub(super) fn covers(&mut self, n: u32, lo: u32, hi: u32) {
        self.f.local_get(L_ADDR64);
        self.memory_u64(hi);
        self.f.i64_lt_u();
        self.f.local_get(L_ADDR64);
        self.f.i64_const(i64::from(n));
        self.f.i64_add();
        self.memory_u64(lo);
        self.f.i64_gt_u();
        self.f.i32_and();
    }

    /// Push the `u64` at `offset` in the running process's `Memory`.
    fn memory_u64(&mut self, offset: u32) {
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, self.layout.memory);
        self.f.i64_load(ALIGN_8, offset);
    }

    /// Push the host address of the access: page plus offset.
    pub(super) fn in_page(&mut self) {
        self.f.local_get(L_PAGE);
        self.f.local_get(L_ADDR);
        self.f.i32_const((PAGE_SIZE - 1) as i32);
        self.f.i32_and();
        self.f.i32_add();
    }

    /// A load into `rt` from `L_ADDR`. Only the sign-extend-to-32 forms need
    /// anything after the wasm load.
    pub(super) fn load(&mut self, rt: u8, acc: Acc, retired: usize) {
        self.page_for_read(access_bytes(acc), retired);
        self.in_page();
        match acc {
            Acc::Load8 => self.f.i64_load8_u(0),
            Acc::Load16 => self.f.i64_load16_u(UNALIGNED, 0),
            Acc::Load32 => self.f.i64_load32_u(UNALIGNED, 0),
            Acc::Load64 => self.f.i64_load(UNALIGNED, 0),
            Acc::LoadS8 => self.f.i64_load8_s(0),
            Acc::LoadS16 => self.f.i64_load16_s(UNALIGNED, 0),
            Acc::LoadS32 => self.f.i64_load32_s(UNALIGNED, 0),
            Acc::LoadS8To32 => {
                self.f.i64_load8_s(0);
                self.mask_to(false);
            }
            Acc::LoadS16To32 => {
                self.f.i64_load16_s(UNALIGNED, 0);
                self.mask_to(false);
            }
            _ => unreachable!("only the loads reach here; `writes_rt` is what sorts them"),
        }
        self.store_reg(rt);
    }

    /// A word from `L_ADDR` into `L_TARGET`, leaving the block if its bit 0 is set:
    /// an interworking branch the interpreter runs.
    pub(super) fn load_branch_target(&mut self, retired: usize) {
        self.page_for_read(4, retired);
        self.in_page();
        self.f.i64_load32_u(UNALIGNED, 0);
        self.f.i32_wrap_i64();
        self.f.local_tee(L_TARGET);
        self.f.i32_const(1);
        self.f.i32_and();
        self.deopt_if(retired);
    }

    /// A store of `rt` to `L_ADDR`.
    pub(super) fn store(&mut self, rt: u8, acc: Acc, retired: usize) {
        self.page_for_write(access_bytes(acc), retired);
        self.in_page();
        self.read_reg_raw(rt);
        match acc {
            Acc::Store8 => self.f.i64_store8(0),
            Acc::Store16 => self.f.i64_store16(UNALIGNED, 0),
            Acc::Store32 => self.f.i64_store32(UNALIGNED, 0),
            Acc::Store64 => self.f.i64_store(UNALIGNED, 0),
            _ => unreachable!("only the stores reach here; `writes_rt` is what sorts them"),
        }
    }

    /// `LDXR`/`LDAXR`: the load, then the monitor armed at the address with what was read.
    pub(super) fn load_exclusive(&mut self, rt: u8, rn: u8, sz: u8, retired: usize) {
        self.address(rn, 0, Wb::None);
        self.load_exclusive_at(rt, sz, retired);
    }

    /// The same at `L_ADDR`.
    pub(super) fn load_exclusive_at(&mut self, rt: u8, sz: u8, retired: usize) {
        self.page_for_read(1 << sz, retired);
        self.in_page();
        match sz {
            0 => self.f.i64_load8_u(0),
            1 => self.f.i64_load16_u(UNALIGNED, 0),
            2 => self.f.i64_load32_u(UNALIGNED, 0),
            _ => self.f.i64_load(UNALIGNED, 0),
        }
        self.f.local_set(L_T);
        self.arm_monitor(None);
        self.f.local_get(L_T);
        self.store_reg(rt);
    }

    /// `STXR`/`STLXR`: swap in `rt` only if the monitor is armed at the address and memory
    /// still holds what it read, and say which in `rs` (0 stored, 1 not). A misaligned
    /// address is the interpreter's, since a wasm atomic would trap on it.
    pub(super) fn store_exclusive(&mut self, rs: u8, rt: u8, rn: u8, sz: u8, retired: usize) {
        self.address(rn, 0, Wb::None);
        self.store_exclusive_at(rs, rt, sz, retired);
    }

    /// The same at `L_ADDR`.
    pub(super) fn store_exclusive_at(&mut self, rs: u8, rt: u8, sz: u8, retired: usize) {
        self.page_for_write(1 << sz, retired);
        if sz > 0 {
            self.f.local_get(L_ADDR);
            self.f.i32_const((1 << sz) - 1);
            self.f.i32_and();
            self.deopt_if(retired);
        }
        let at = self.layout.exclusive;
        self.monitor_held();
        self.spend_monitor();
        self.f.local_get(L_C);
        self.f.if_void();
        let wide = sz == 3;
        self.f.local_get(STATE);
        self.f.i64_load(ALIGN_8, at + MONITOR_VALUE);
        self.f.local_set(L_T);
        self.in_page();
        self.f.local_get(L_T);
        if !wide {
            self.f.i32_wrap_i64();
        }
        self.read_reg_raw(rt);
        if !wide {
            self.f.i32_wrap_i64();
        }
        self.f.atomic_cmpxchg(sz);
        self.f.local_get(L_T);
        if wide {
            self.f.i64_ne();
        } else {
            self.f.i32_wrap_i64();
            self.f.i32_ne();
        }
        self.f.i64_extend_i32_u();
        self.f.local_set(L_R);
        self.f.end();
        self.f.local_get(L_R);
        self.store_reg(rs);
    }

    /// Arm the monitor at `L_ADDR` with `L_T` and `value1`, an `i64` local or zero.
    pub(super) fn arm_monitor(&mut self, value1: Option<u32>) {
        let at = self.layout.exclusive;
        self.f.local_get(STATE);
        self.f.local_get(L_ADDR64);
        self.f.i64_store(ALIGN_8, at + MONITOR_ADDR);
        self.f.local_get(STATE);
        self.f.i32_const(1);
        self.f.i32_store(ALIGN_4, at + MONITOR_ARMED);
        self.f.local_get(STATE);
        self.f.local_get(L_T);
        self.f.i64_store(ALIGN_8, at + MONITOR_VALUE);
        self.f.local_get(STATE);
        match value1 {
            Some(local) => self.f.local_get(local),
            None => self.f.i64_const(0),
        }
        self.f.i64_store(ALIGN_8, at + MONITOR_VALUE + 8);
    }

    /// `LDXP`/`LDAXP`: the pair loaded, then the monitor armed over both halves. The word
    /// form keeps its pair as the one doubleword it was read as.
    pub(super) fn load_exclusive_pair(
        &mut self,
        rt: u8,
        rt2: u8,
        rn: u8,
        wide: bool,
        retired: usize,
    ) {
        self.address(rn, 0, Wb::None);
        self.page_for_read(if wide { 16 } else { 8 }, retired);
        self.in_page();
        self.f.i64_load(UNALIGNED, 0);
        self.f.local_set(L_T);
        if wide {
            self.in_page();
            self.f.i64_load(UNALIGNED, 8);
            self.f.local_set(L_S);
        }
        self.arm_monitor(wide.then_some(L_S));
        self.f.local_get(L_T);
        if !wide {
            self.mask_to(false);
        }
        self.store_reg(rt);
        if wide {
            self.f.local_get(L_S);
        } else {
            self.f.local_get(L_T);
            self.f.i64_const(32);
            self.f.i64_shr_u();
        }
        self.store_reg(rt2);
    }

    /// Whether the monitor is armed at `L_ADDR`, in `L_C`.
    pub(super) fn monitor_held(&mut self) {
        let at = self.layout.exclusive;
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, at + MONITOR_ARMED);
        self.f.local_get(STATE);
        self.f.i64_load(ALIGN_8, at + MONITOR_ADDR);
        self.f.local_get(L_ADDR64);
        self.f.i64_eq();
        self.f.i32_and();
        self.f.local_set(L_C);
    }

    /// Disarm the monitor, whether or not the store happens, and start `L_R` at 1, the
    /// status of a store that did not.
    pub(super) fn spend_monitor(&mut self) {
        self.f.local_get(STATE);
        self.f.i32_const(0);
        self.f
            .i32_store(ALIGN_4, self.layout.exclusive + MONITOR_ARMED);
        self.f.i64_const(1);
        self.f.local_set(L_R);
    }

    /// Push the address of the byte [`crate::mem::Memory::compare_exchange_pair`] serialises on.
    pub(super) fn pair_lock(&mut self) {
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, self.layout.memory);
        self.f.i32_const(self.layout.pair_lock as i32);
        self.f.i32_add();
    }

    /// `STXP`/`STLXP`, as `STXR` is. The word form swaps its doubleword with one `cmpxchg`
    /// (a misaligned address is the interpreter's). Wasm has no 16-byte atomic, so the
    /// doubleword form compares and writes under the lock the interpreter's pair store takes.
    pub(super) fn store_exclusive_pair(
        &mut self,
        rs: u8,
        rt: u8,
        rt2: u8,
        rn: u8,
        wide: bool,
        retired: usize,
    ) {
        self.address(rn, 0, Wb::None);
        self.page_for_write(if wide { 16 } else { 8 }, retired);
        if !wide {
            self.f.local_get(L_ADDR);
            self.f.i32_const(7);
            self.f.i32_and();
            self.deopt_if(retired);
        }
        self.monitor_held();
        if wide && self.watches {
            // The interpreter's pair store reads both halves, which a read watchpoint reports.
            self.f.local_get(L_C);
            self.covers(16, self.layout.read_watch_lo, self.layout.read_watch_hi);
            self.f.i32_and();
            self.deopt_if(retired);
        }
        self.spend_monitor();
        self.f.local_get(L_C);
        self.f.if_void();
        let at = self.layout.exclusive;
        if wide {
            let done = self.f.block_void();
            let again = self.f.loop_void();
            self.pair_lock();
            self.f.i32_const(0);
            self.f.i32_const(1);
            self.f.atomic_cmpxchg(0);
            self.f.i32_eqz();
            self.f.if_void();
            self.f.br(done);
            self.f.end();
            self.f.br(again);
            self.f.end();
            self.f.end();
            for half in [0, 8] {
                self.in_page();
                self.f.i64_load(UNALIGNED, half);
                self.f.local_get(STATE);
                self.f.i64_load(ALIGN_8, at + MONITOR_VALUE + half);
                self.f.i64_eq();
            }
            self.f.i32_and();
            self.f.if_void();
            for (half, slot) in [(0, rt), (8, rt2)] {
                self.in_page();
                self.read_reg_raw(slot);
                self.f.i64_store(UNALIGNED, half);
            }
            self.f.i64_const(0);
            self.f.local_set(L_R);
            self.f.end();
            self.pair_lock();
            self.f.i32_const(1);
            self.f.i32_const(0);
            self.f.atomic_cmpxchg(0);
            self.f.drop_value();
        } else {
            self.f.local_get(STATE);
            self.f.i64_load(ALIGN_8, at + MONITOR_VALUE);
            self.f.local_set(L_T);
            self.in_page();
            self.f.local_get(L_T);
            self.read_reg_raw(rt);
            self.mask_to(false);
            self.read_reg_raw(rt2);
            self.f.i64_const(32);
            self.f.i64_shl();
            self.f.i64_or();
            self.f.atomic_cmpxchg(3);
            self.f.local_get(L_T);
            self.f.i64_ne();
            self.f.i64_extend_i32_u();
            self.f.local_set(L_R);
        }
        self.f.end();
        self.f.local_get(L_R);
        self.store_reg(rs);
    }

    /// A load of two `n`-byte registers from `L_ADDR`, checked as one access.
    pub(super) fn load_pair(&mut self, rt: u8, rt2: u8, n: u32, signed: bool, retired: usize) {
        self.page_for_read(2 * n, retired);
        for (slot, at) in [(rt, 0), (rt2, n)] {
            self.in_page();
            match (n, signed) {
                (8, _) => self.f.i64_load(UNALIGNED, at),
                (4, false) => self.f.i64_load32_u(UNALIGNED, at),
                _ => self.f.i64_load32_s(UNALIGNED, at),
            }
            self.store_reg(slot);
        }
    }

    /// A store of two `n`-byte registers to `L_ADDR`, checked as one access.
    pub(super) fn store_pair(&mut self, rt: u8, rt2: u8, n: u32, retired: usize) {
        self.page_for_write(2 * n, retired);
        for (slot, at) in [(rt, 0), (rt2, n)] {
            self.in_page();
            self.read_reg_raw(slot);
            match n {
                8 => self.f.i64_store(UNALIGNED, at),
                _ => self.f.i64_store32(UNALIGNED, at),
            }
        }
    }

    /// A single-register access. `PRFM` has no guard: it touches no memory.
    pub(super) fn access(&mut self, rt: u8, acc: Acc, retired: usize) {
        match acc {
            Acc::Prefetch => {}
            _ if acc.writes_rt() => self.load(rt, acc, retired),
            _ => self.store(rt, acc, retired),
        }
    }
}

/// How many bytes an access touches.
pub(super) fn access_bytes(acc: Acc) -> u32 {
    match acc {
        Acc::Load8 | Acc::LoadS8 | Acc::LoadS8To32 | Acc::Store8 => 1,
        Acc::Load16 | Acc::LoadS16 | Acc::LoadS16To32 | Acc::Store16 => 2,
        Acc::Load32 | Acc::LoadS32 | Acc::Store32 => 4,
        Acc::Load64 | Acc::Store64 => 8,
        // `PRFM` reads nothing, so nothing is ever asked of its address.
        Acc::Prefetch => 0,
    }
}
