//! The Horizon supervisor calls (`SVC`) libnx homebrew issues at runtime.

use crate::cpu::{
    ArbiterWait, Cpu, GUEST_ASLR_REGION_ADDR, GUEST_ASLR_REGION_SIZE, GUEST_STACK_REGION_ADDR,
    GUEST_STACK_REGION_SIZE, HID_SHMEM_SIZE, PL_SHMEM_SIZE,
};
use crate::kernel::cores::Handback;
use crate::mem::{MemoryState, Permission, StateRun};
use crate::{Error, Result};
use std::fmt::Write;

mod send_sync_request;

const RESULT_OK: u64 = 0;
const RESULT_INVALID_MEMORY_RANGE: u64 = 0x8000_DC01;
// Kernel (module 1) description 114: invalid handle.
const RESULT_INVALID_HANDLE: u64 = 1 | (114 << 9);
// Kernel descriptions 57 and 116: invalid core id and ideal core outside the mask.
const RESULT_INVALID_CORE_ID: u64 = 1 | (57 << 9);
const RESULT_INVALID_COMBINATION: u64 = 1 | (116 << 9);
const RESULT_INVALID_PRIORITY: u64 = 1 | (112 << 9);
const RESULT_INVALID_STATE: u64 = 1 | (125 << 9);
const RESULT_TIMED_OUT: u64 = 0xEA01;
// Ideal-core values that are not a core: process default, none, no update.
const IDEAL_CORE_USE_PROCESS_VALUE: i32 = -2;
const IDEAL_CORE_DONT_CARE: i32 = -1;
const IDEAL_CORE_NO_UPDATE: i32 = -3;

impl Cpu {
    /// Every SVC a guest issues. `svc #0` (unused by Horizon) is the host halt trap.
    pub(crate) fn syscall(&mut self, imm: u16) -> Result<()> {
        if self.core != 0 {
            // Only core 0 runs the kernel: it reissues the `svc`.
            self.pc = self.pc.wrapping_sub(4);
            self.handback = Some(Handback::Svc);
            return Ok(());
        }
        if imm == 0 {
            self.halted = true;
            return Ok(());
        }
        if !self.machine.cores.is_empty() {
            self.lend_again(imm);
        }
        self.horizon_syscall(imm)?;
        if let Some(ending) = self.process_ending.take() {
            self.end_process(self.pid, ending);
        }
        Ok(())
    }

    /// Horizon syscalls by libnx's numbering (`nx/source/kernel/svc.s`).
    /// X0 carries the Result; out-handles and values come back in X1.
    pub(crate) fn horizon_syscall(&mut self, imm: u16) -> Result<()> {
        // Trace every SVC except IPC (traced by `TRACE_IPC`) and the hot
        // WaitSynchronization, SleepThread and QueryMemory.
        if !matches!(imm, 0x21 | 0x18 | 0x0b | 0x06)
            && crate::trace::enabled(crate::trace::Trace::Svc)
        {
            crate::traceln!(
                "[svc] pc={:#x} #{:#04x} x0={:#x} x1={:#x} x2={:#x} x3={:#x}",
                self.pc,
                imm,
                self.read_zr(0),
                self.read_zr(1),
                self.read_zr(2),
                self.read_zr(3)
            );
        }
        match imm {
            0x01 => self.svc_set_heap_size(),
            0x02 | 0x03 | 0x14 => {
                // SetMemoryPermission / SetMemoryAttribute / UnmapSharedMemory
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x04 => self.svc_map_memory(),
            0x2C => self.svc_map_physical_memory(),
            0x2D => self.svc_unmap_physical_memory(),
            0x13 => self.svc_map_shared_memory(),
            0x05 => self.svc_unmap_memory(),
            0x06 => self.svc_query_memory(),
            0x07 => self.svc_exit_process(),
            0x0A => {
                // ExitThread: only the main thread ending stops the process.
                self.exit_thread();
                Ok(())
            }
            0x08 => self.svc_create_thread(),
            0x09 => self.svc_start_thread(),
            0x32 => self.svc_set_thread_activity(),
            0x33 => self.svc_get_thread_context3(),
            0x0B => self.svc_sleep_thread(),
            0x1A => self.svc_arbitrate_lock(),
            0x1B => self.svc_arbitrate_unlock(),
            0x1C => self.svc_wait_process_wide_key_atomic(),
            0x34 => self.svc_wait_for_address(),
            0x35 => self.svc_signal_to_address(),
            // SynchronizePreemptionState only unpins a pinned thread, and nothing here pins one.
            0x36 => Ok(()),
            0x1D => self.svc_signal_process_wide_key(),
            0x0C => self.svc_get_thread_priority(),
            0x0D => self.svc_set_thread_priority(),
            0x0E => self.svc_get_thread_core_mask(),
            0x0F => self.svc_set_thread_core_mask(),
            0x16 => {
                // CloseHandle(handle)
                self.close_handle(self.read_zr(0));
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x28 | 0x5F => {
                // ReturnFromException / FlushProcessDataCache
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x17 => self.svc_reset_signal(),
            0x19 => self.svc_cancel_synchronization(),
            0x10 => {
                // GetCurrentProcessorNumber: the running thread's core.
                let core = self.current_core();
                self.write_zr(0, u64::from(core));
                Ok(())
            }
            0x11 | 0x12 => {
                // SignalEvent / ClearEvent
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x15 => self.svc_create_transfer_memory(),
            0x18 => self.svc_wait_synchronization(),
            0x1E => {
                // GetSystemTick: 19.2 MHz, scaled from cycles of the 1.02 GHz CPU.
                self.svc_out64(0, 0, 1, self.system_tick());
                Ok(())
            }
            0x1F => self.svc_connect_to_named_port(),
            0x20..=0x23 => self.svc_send_sync_request(),
            0x24 => {
                // GetProcessId(out_process_id, process_handle): Result in X0, id in X1.
                self.write_zr(0, RESULT_OK);
                self.svc_out64(1, 1, 2, 1);
                Ok(())
            }
            0x25 => self.svc_get_thread_id(),
            0x26 => self.svc_break(),
            0x27 => self.svc_output_debug_string(),
            0x29 => self.svc_get_info(),
            0x6F => self.svc_get_system_info(),
            _ => Err(Error::Cpu(format!(
                "unimplemented Horizon syscall #{:#x}",
                imm
            ))),
        }
    }

    /// SetHeapSize(size): a heap larger than its region is refused.
    fn svc_set_heap_size(&mut self) -> Result<()> {
        const RESULT_OUT_OF_MEMORY: u64 = 1 | (104 << 9);
        let layout = self.memory_layout();
        let size = self.read_zr(1);
        if size > layout.heap_size {
            self.write_zr(0, RESULT_OUT_OF_MEMORY);
            return Ok(());
        }
        self.process_mut().heap_size = size;
        self.write_zr(0, RESULT_OK);
        self.write_zr(1, layout.heap_addr);
        Ok(())
    }

    /// MapMemory(dst, src, size): really back dst, since `virtmemFindStack` looks
    /// for unmapped ranges to place the next thread's stack mirror.
    fn svc_map_memory(&mut self) -> Result<()> {
        let dst = self.read_zr(0);
        let src = self.read_zr(1);
        let size = self.read_zr(2) as usize;
        if dst == 0 || size == 0 {
            self.write_zr(0, RESULT_INVALID_MEMORY_RANGE);
            return Ok(());
        }
        if crate::trace::enabled(crate::trace::Trace::Map) {
            crate::traceln!("[map] MapMemory dst={dst:#x} src={src:#x} size={size:#x}");
        }
        self.mem.copy_range(dst, src, size)?;
        self.write_zr(0, RESULT_OK);
        Ok(())
    }

    /// MapPhysicalMemory(address, size): how 39-bit titles grow their heap. Pages
    /// materialise on first write via the soft-mapped address space.
    fn svc_map_physical_memory(&mut self) -> Result<()> {
        let layout = self.memory_layout();
        let addr = self.read_zr(0);
        let size = self.read_zr(1);
        let fits = addr
            .checked_add(size)
            .is_some_and(|end| end <= layout.space_end);
        if size == 0 || (addr & 0xFFF) != 0 || (size & 0xFFF) != 0 || !fits {
            self.write_zr(0, RESULT_INVALID_MEMORY_RANGE);
            return Ok(());
        }
        self.write_zr(0, RESULT_OK);
        Ok(())
    }

    /// UnmapPhysicalMemory(address, size): pages are really freed for the RAM cap.
    fn svc_unmap_physical_memory(&mut self) -> Result<()> {
        let addr = self.read_zr(0);
        let size = self.read_zr(1);
        self.mem.unmap(addr, size);
        self.write_zr(0, RESULT_OK);
        Ok(())
    }

    /// MapSharedMemory(handle, addr, size, perm): hid's is the console's block, with a
    /// connected pad published at once; others are zeroed, pl's with the shared font.
    fn svc_map_shared_memory(&mut self) -> Result<()> {
        let addr = self.read_zr(1);
        let size = self.read_zr(2);
        // Prefer `hid`'s handle; the size match is a fallback.
        if Some(self.read_zr(0)) == self.process().hid_shmem_handle
            || size == u64::from(HID_SHMEM_SIZE)
        {
            let block = self.hid_shmem.clone();
            self.mem.map_shared(addr, &block)?;
            self.process_mut().hid_shmem_addr = addr;
            self.hid_mapped = true;
            self.set_gamepad_state(0, 0, 0, 0, 0);
            // An empty touch sample: a zeroed LIFO header is a ring with no capacity.
            self.set_touch_state(&[]);
        } else if Some(self.read_zr(0)) == self.process().time_shmem_handle {
            let block = self.time_shmem.clone();
            self.mem.map_shared(addr, &block)?;
        } else {
            self.mem.map_zero(addr, size)?;
            if size == u64::from(PL_SHMEM_SIZE) {
                self.process_mut().pl_shmem_addr = addr;
                self.write_shared_font(addr);
            }
        }
        self.write_zr(0, RESULT_OK);
        Ok(())
    }

    /// UnmapMemory(dst, src, size). An out-of-range unmap reports a 39-bit space,
    /// which hbmenu probes for.
    fn svc_unmap_memory(&mut self) -> Result<()> {
        let dst = self.read_zr(0);
        if (dst >> 48) == 0xFFFF {
            self.write_zr(0, 0x8000_D401);
            return Ok(());
        }
        let src = self.read_zr(1);
        let size = self.read_zr(2) as usize;
        if dst != 0 && src != 0 && size != 0 {
            self.mem.copy_range(src, dst, size)?;
            self.mem.unmap(dst, size as u64);
        }
        self.write_zr(0, RESULT_OK);
        Ok(())
    }

    /// QueryMemory(info, pageInfo, addr): the run of pages in the queried page's
    /// state. Untouched soft-mapped pages read as unmapped. Only module `.text` is
    /// executable, since `rtld` finds modules by walking for executable `CodeStatic`,
    /// and `.rodata` is read-only, where `nn::diag` reads a module's name.
    fn svc_query_memory(&mut self) -> Result<()> {
        let layout = self.memory_layout();
        let out = self.read_zr(0);
        let addr = self.read_zr(2);
        // See [`crate::mem::Memory::state_run`].
        let mut run = self.mem.state_run(addr, layout.space_end);
        // The granted heap is in use even where no page has been touched yet.
        let heap = layout.heap_addr..layout.heap_addr.saturating_add(self.process().heap_size);
        if heap.contains(&addr) {
            run = StateRun {
                start: heap.start,
                end: heap.end,
                mapped: true,
                perm: Permission::ReadWrite,
                state: MemoryState::Normal,
            };
        } else if addr < heap.start {
            run.end = run.end.min(heap.start);
        } else if !heap.is_empty() {
            run.start = run.start.max(heap.end);
        }
        // Past the end of the space is one inaccessible region up to 2^64, as Horizon
        // reports it: `rtld` walks regions until the address wraps.
        if addr >= layout.space_end {
            run = StateRun {
                start: layout.space_end,
                end: 0,
                mapped: false,
                perm: Permission::None,
                state: MemoryState::Inaccessible,
            };
        }
        let (base, size) = (run.start, run.end.wrapping_sub(run.start));
        let mut info = Vec::with_capacity(40);
        info.extend_from_slice(&base.to_le_bytes());
        info.extend_from_slice(&size.to_le_bytes());
        for v in [
            run.state as u32, // type
            0,
            run.perm as u32,
            0,
            0,
            0,
        ] {
            info.extend_from_slice(&v.to_le_bytes());
        }
        for (i, &b) in info.iter().enumerate() {
            self.mem.write_u8(out.wrapping_add(i as u64), b)?;
        }
        self.write_zr(0, RESULT_OK);
        self.write_zr(1, if run.mapped { 0x1000 } else { 0 }); // page info
        Ok(())
    }

    /// ExitProcess. Only the main thread returns to the exit stub; any other
    /// thread here is an emulator bug, so say so loudly.
    fn svc_exit_process(&mut self) -> Result<()> {
        let pc = self.get_pc();
        let at_stub = pc == crate::cpu::SELF_RETURN_TRAMPOLINE
            || pc.wrapping_sub(4) == crate::cpu::SELF_RETURN_TRAMPOLINE;
        if at_stub && self.current_thread_index() != 0 {
            let line = format!(
                "{} reached the emulator's process-exit stub, which only the main thread \
                 returns to; the process is ending because of it (x30={:#x}, sp={:#x})",
                self.thread_label(self.current_thread_handle()),
                self.read_zr(30),
                self.sp()
            );
            let line = format!("{line}\n{}", self.trail_text());
            self.log(crate::trace::Level::Error, &line);
        }
        self.exit_process(crate::cpu::Ending::Exited);
        Ok(())
    }

    /// CreateThread(entry = X1, arg = X2, stack_top = X3, priority = W4, core = W5)
    /// -> handle in X1. Starts suspended with its own TLS block.
    fn svc_create_thread(&mut self) -> Result<()> {
        let entry = self.branch_target(self.read_zr(1))?;
        let arg = self.read_zr(2);
        let stack_top = self.read_zr(3);
        // AArch32 passes the priority in r0 and the core in r4.
        let (priority, core) = if self.mode == crate::cpu::ExecMode::A32 {
            (self.read_zr(0), self.read_zr(4))
        } else {
            (self.read_zr(4), self.read_zr(5))
        };
        let priority = priority as u32;
        if priority > 63 {
            self.write_zr(0, RESULT_INVALID_PRIORITY);
            return Ok(());
        }
        let core = match core as u32 as i32 {
            IDEAL_CORE_USE_PROCESS_VALUE => self.process().main_thread_core,
            core @ 0..=3 if self.process().process_core_mask >> core & 1 != 0 => core as u8,
            _ => {
                self.write_zr(0, RESULT_INVALID_CORE_ID);
                return Ok(());
            }
        };
        let handle = self.create_thread(entry, arg, stack_top, priority as u8, core);
        if crate::trace::enabled(crate::trace::Trace::Wait) {
            crate::traceln!("[thread] create handle={handle:#x} entry={entry:#x} core={core}");
        }
        self.write_zr(0, RESULT_OK);
        self.write_zr(1, handle);
        Ok(())
    }

    /// StartThread
    fn svc_start_thread(&mut self) -> Result<()> {
        let handle = self.read_zr(0);
        let started = self.start_thread(handle);
        if crate::trace::enabled(crate::trace::Trace::Wait) {
            crate::traceln!("[thread] start handle={handle:#x} -> {started}");
        }
        self.write_zr(0, RESULT_OK);
        Ok(())
    }

    /// SetThreadActivity(handle, activity): 0 = Runnable, 1 = Paused. The caller
    /// can't suspend itself, and a no-op change reports busy.
    fn svc_set_thread_activity(&mut self) -> Result<()> {
        const RESULT_BUSY: u64 = 1 | (122 << 9);
        let handle = self.read_zr(0);
        let paused = self.read_zr(1) != 0;
        if self.current_thread_handle() == handle {
            self.write_zr(0, RESULT_BUSY);
            return Ok(());
        }
        let result = match self.set_thread_paused(handle, paused) {
            Some(true) => RESULT_OK,
            Some(false) => RESULT_INVALID_STATE,
            None => RESULT_INVALID_HANDLE,
        };
        self.write_zr(0, result);
        Ok(())
    }

    /// GetThreadContext3(out = X0, handle = X1): the suspended thread's registers.
    fn svc_get_thread_context3(&mut self) -> Result<()> {
        let out = self.read_zr(0);
        let handle = self.read_zr(1);
        let ok = self.write_thread_context(out, handle);
        self.write_zr(0, if ok { RESULT_OK } else { RESULT_INVALID_HANDLE });
        Ok(())
    }

    /// SleepThread(nanoseconds = X0). 0, -1 and -2 are yield modes; other values
    /// really sleep, which matters for `nn::os` poll loops.
    fn svc_sleep_thread(&mut self) -> Result<()> {
        let nanoseconds = self.svc_arg64(0, 0, 1) as i64;
        self.write_zr(0, RESULT_OK);
        match self.wait_deadline(nanoseconds) {
            Some(deadline) => self.sleep_until(deadline),
            None => self.yield_thread(),
        }
        Ok(())
    }

    /// ArbitrateLock(owner = W0, mutex = X1, self = W2)
    fn svc_arbitrate_lock(&mut self) -> Result<()> {
        let owner = self.read_zr(0) as u32;
        let mutex = self.read_zr(1);
        let requester = self.read_zr(2) as u32;
        if crate::trace::enabled(crate::trace::Trace::Wait) {
            crate::traceln!(
                "[wait] lock mutex={mutex:#x} owner={owner:#x} self={requester:#x} thread={:#x}",
                self.current_thread_handle()
            );
        }
        self.write_zr(0, RESULT_OK);
        self.arbitrate_lock(owner, mutex, requester);
        Ok(())
    }

    /// ArbitrateUnlock(mutex = X0)
    fn svc_arbitrate_unlock(&mut self) -> Result<()> {
        let mutex = self.read_zr(0);
        self.write_zr(0, RESULT_OK);
        self.arbitrate_unlock(mutex);
        Ok(())
    }

    /// WaitProcessWideKeyAtomic(mutex = X0, key = X1, self = W2, timeout = X3)
    fn svc_wait_process_wide_key_atomic(&mut self) -> Result<()> {
        let mutex = self.read_zr(0);
        let key = self.read_zr(1);
        let requester = self.read_zr(2) as u32;
        // A negative timeout waits forever; a positive one is in nanoseconds.
        let timeout = self.svc_arg64(3, 3, 4) as i64;
        if crate::trace::enabled(crate::trace::Trace::Wait) {
            crate::traceln!(
                "[wait] condvar key={key:#x} mutex={mutex:#x} timeout={timeout} thread={:#x} bt={:x?}",
                self.current_thread_handle(),
                self.backtrace(8)
            );
        }
        self.write_zr(0, RESULT_OK);
        self.wait_process_wide_key(mutex, key, requester, timeout);
        Ok(())
    }

    /// WaitForAddress(address = X0, arb_type = W1, value = W2, timeout = X3).
    /// If the arb_type predicate doesn't hold, report InvalidState instead of blocking.
    fn svc_wait_for_address(&mut self) -> Result<()> {
        let addr = self.read_zr(0);
        let arb_type = self.read_zr(1) as u32;
        let value = self.read_zr(2) as u32 as i32;
        let timeout = self.svc_arg64(3, 3, 4) as i64;
        if crate::trace::enabled(crate::trace::Trace::Wait) {
            crate::traceln!(
                "[wait] arbiter addr={addr:#x} type={arb_type} value={value} \
                 timeout={timeout} thread={:#x}",
                self.current_thread_handle()
            );
        }
        // Write the result before blocking: blocking switches register files.
        let outcome = self.arbitrate_address(addr, arb_type, value, timeout);
        self.write_zr(
            0,
            match outcome {
                ArbiterWait::Blocked => RESULT_OK,
                ArbiterWait::Mismatch => RESULT_INVALID_STATE,
                ArbiterWait::TimedOut => RESULT_TIMED_OUT,
            },
        );
        if outcome == ArbiterWait::Blocked {
            self.block_on_address(addr, timeout);
        }
        Ok(())
    }

    /// SignalToAddress(address = X0, signal_type = W1, value = W2, count = W3).
    fn svc_signal_to_address(&mut self) -> Result<()> {
        let addr = self.read_zr(0);
        let signal_type = self.read_zr(1) as u32;
        let value = self.read_zr(2) as u32 as i32;
        let count = self.read_zr(3) as u32 as i32;
        if crate::trace::enabled(crate::trace::Trace::Wait) {
            crate::traceln!(
                "[wait] arbiter signal addr={addr:#x} type={signal_type} value={value} \
                 count={count} thread={:#x}",
                self.current_thread_handle()
            );
        }
        let ok = self.signal_to_address(addr, signal_type, value, count);
        self.write_zr(0, if ok { RESULT_OK } else { RESULT_INVALID_STATE });
        Ok(())
    }

    /// SignalProcessWideKey(key = X0, count = W1)
    fn svc_signal_process_wide_key(&mut self) -> Result<()> {
        let key = self.read_zr(0);
        let count = self.read_zr(1) as u32 as i32;
        if crate::trace::enabled(crate::trace::Trace::Wait) {
            crate::traceln!(
                "[wait] signal key={key:#x} count={count} thread={:#x}",
                self.current_thread_handle()
            );
        }
        self.write_zr(0, RESULT_OK);
        self.signal_process_wide_key(key, count);
        Ok(())
    }

    /// GetThreadPriority(handle = X1) -> priority in W1.
    fn svc_get_thread_priority(&mut self) -> Result<()> {
        match self.thread_priority(self.read_zr(1)) {
            Some(priority) => {
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, u64::from(priority));
            }
            None => self.write_zr(0, RESULT_INVALID_HANDLE),
        }
        Ok(())
    }

    /// SetThreadPriority(handle = X0, priority = W1)
    fn svc_set_thread_priority(&mut self) -> Result<()> {
        let handle = self.read_zr(0);
        let priority = self.read_zr(1) as u32;
        let result = if priority > 63 {
            RESULT_INVALID_PRIORITY
        } else if self.set_thread_priority(handle, priority as u8) {
            RESULT_OK
        } else {
            RESULT_INVALID_HANDLE
        };
        self.write_zr(0, result);
        Ok(())
    }

    /// GetThreadCoreMask(handle = W2) -> ideal core W1, affinity X2 (A32 r2:r3).
    fn svc_get_thread_core_mask(&mut self) -> Result<()> {
        let handle = self.read_zr(2);
        match self.thread_core_mask(handle) {
            Some((ideal_core, affinity)) => {
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, u64::from(ideal_core as u32));
                self.svc_out64(2, 2, 3, affinity);
            }
            None => self.write_zr(0, RESULT_INVALID_HANDLE),
        }
        Ok(())
    }

    /// SetThreadCoreMask(handle = W0, ideal core = W1, affinity = X2, r2:r3 on A32),
    /// checked in the kernel's order.
    fn svc_set_thread_core_mask(&mut self) -> Result<()> {
        let handle = self.read_zr(0);
        let requested = self.read_zr(1) as u32 as i32;
        let affinity = self.svc_arg64(2, 2, 3);
        let Some((current_ideal, _)) = self.thread_core_mask(handle) else {
            self.write_zr(0, RESULT_INVALID_HANDLE);
            return Ok(());
        };
        let checked = match requested {
            IDEAL_CORE_USE_PROCESS_VALUE => Ok((
                i32::from(self.process().main_thread_core),
                1u64 << self.process().main_thread_core,
            )),
            _ if affinity & !self.process().process_core_mask != 0 => Err(RESULT_INVALID_CORE_ID),
            _ if affinity == 0 => Err(RESULT_INVALID_COMBINATION),
            core @ 0..=3 if affinity & (1 << core) == 0 => Err(RESULT_INVALID_COMBINATION),
            core @ 0..=3 => Ok((core, affinity)),
            IDEAL_CORE_DONT_CARE => Ok((IDEAL_CORE_DONT_CARE, affinity)),
            IDEAL_CORE_NO_UPDATE if current_ideal >= 0 && affinity & (1 << current_ideal) == 0 => {
                Err(RESULT_INVALID_COMBINATION)
            }
            IDEAL_CORE_NO_UPDATE => Ok((current_ideal, affinity)),
            _ => Err(RESULT_INVALID_CORE_ID),
        };
        let result = match checked {
            Ok((ideal, affinity)) if self.set_thread_core_mask(handle, ideal, affinity) => {
                RESULT_OK
            }
            Ok(_) => RESULT_INVALID_HANDLE,
            Err(result) => result,
        };
        self.write_zr(0, result);
        Ok(())
    }

    /// ResetSignal(handle): clear an event and report whether it was signalled.
    fn svc_reset_signal(&mut self) -> Result<()> {
        let handle = self.read_zr(0);
        let was_signalled = self.reset_signal(handle);
        if crate::trace::enabled(crate::trace::Trace::Wait) {
            crate::traceln!(
                "[wait] reset handle={handle:#x} {:?} -> {was_signalled}",
                self.event_name(handle)
            );
        }
        self.write_zr(
            0,
            if was_signalled {
                RESULT_OK
            } else {
                RESULT_INVALID_STATE
            },
        );
        Ok(())
    }

    /// CancelSynchronization(thread): wakes a `nn::os::MultiWait` waiting on no handle.
    fn svc_cancel_synchronization(&mut self) -> Result<()> {
        let cancelled = self.cancel_synchronization(self.read_zr(0));
        self.write_zr(
            0,
            if cancelled {
                RESULT_OK
            } else {
                RESULT_INVALID_HANDLE
            },
        );
        Ok(())
    }

    /// CreateTransferMemory(addr, size, perm): the creator's pages become a shared
    /// block, so a process the memory is passed to sees what the creator writes.
    fn svc_create_transfer_memory(&mut self) -> Result<()> {
        // Kernel description 106: the range is not the caller's memory.
        const RESULT_INVALID_CURRENT_MEMORY: u64 = 1 | (106 << 9);
        let addr = self.read_zr(1);
        let size = self.read_zr(2);
        // Untouched heap is the guest's too; the block backs every page of the range,
        // so it must fit under the memory cap once it replaces what is backed there.
        let owned = addr.checked_add(size).is_some_and(|end| {
            self.mem.owns_range(addr, end)
                && self.mem.mapped_bytes() - self.mem.mapped_bytes_in(addr, end) + size
                    <= self.mem.max_mapped_bytes()
        });
        if !owned {
            self.write_zr(0, RESULT_INVALID_CURRENT_MEMORY);
            return Ok(());
        }
        let size = size as usize;
        let handle = self.alloc_handle();
        self.grant_handle(handle);
        let page = crate::mem::PAGE_SIZE;
        if size != 0 && addr.is_multiple_of(page as u64) && size.is_multiple_of(page) {
            let block = crate::mem::SharedPages::new(size);
            let mut contents = [0u8; crate::mem::PAGE_SIZE];
            for offset in (0..size).step_by(page) {
                self.mem.read_into(addr + offset as u64, &mut contents)?;
                block.write(offset as u32, &contents)?;
            }
            self.mem.map_shared(addr, &block)?;
            self.transfer_memories.insert(handle, block);
        }
        self.write_zr(0, RESULT_OK);
        self.write_zr(1, handle);
        Ok(())
    }

    /// WaitSynchronization(out_index, handles, num_handles, timeout). X1 is the
    /// index of the signalled handle.
    fn svc_wait_synchronization(&mut self) -> Result<()> {
        const RESULT_CANCELLED: u64 = 1 | (118 << 9);
        let handles_ptr = self.read_zr(1);
        let count = (self.read_zr(2) as u32).min(0x40);
        let timeout = self.svc_arg64(3, 0, 3) as i64;
        let handles: Vec<u64> = (0..count)
            .map(|i| {
                u64::from(
                    self.mem
                        .read_u32(handles_ptr.wrapping_add(u64::from(i) * 4))
                        .unwrap_or(0),
                )
            })
            .collect();
        if crate::trace::enabled(crate::trace::Trace::Wait) {
            let named: Vec<String> = handles
                .iter()
                .map(|&h| match (self.event_name(h), self.event_signaled(h)) {
                    (Some(name), Some(true)) => format!("{h:#x} {name} signalled"),
                    (Some(name), _) => format!("{h:#x} {name}"),
                    _ => format!("{h:#x} (not an event)"),
                })
                .collect();
            crate::traceln!("[wait] pc={:#x} timeout={timeout} {named:?}", self.pc);
        }
        // Presents drive vsync.
        let refreshed =
            self.cycles.wrapping_sub(self.last_vsync_cycles) >= crate::cpu::VSYNC_PERIOD_CYCLES;
        if self.nv.gpu.frames != self.last_vsync_frame || refreshed {
            self.last_vsync_frame = self.nv.gpu.frames;
            self.last_vsync_cycles = self.cycles;
            for vsync in self.vsync_events.clone() {
                self.signal_event(vsync);
            }
        }
        // `hid` ticks for the same reason.
        self.hid_tick();
        // Audio buffer events fire here; see `Cpu::audio_tick`.
        let next_buffer = self.audio_tick(&handles);
        // First ready handle; unmodelled handles count as ready.
        let ready = handles
            .iter()
            .position(|&h| self.waitable_signaled(h) != Some(false));
        if let Some(index) = ready {
            self.consume_event(handles[index]);
            self.write_zr(0, RESULT_OK);
            self.write_zr(1, index as u64);
            self.yield_thread();
            return Ok(());
        }
        if self.take_sync_cancelled() {
            self.write_zr(0, RESULT_CANCELLED);
            return Ok(());
        }
        // Nothing fired. A poll times out; a blocking wait parks.
        if timeout == 0 {
            self.write_zr(0, RESULT_TIMED_OUT);
            return Ok(());
        }
        // A blocking wait on vsync is certain to fire, so park until the tick.
        if timeout != 0
            && self.vsync_throttle
            && self.vsync_events.iter().any(|v| handles.contains(v))
        {
            self.pc = self.pc.wrapping_sub(4);
            self.park_on_events(self.next_display_tick());
            return Ok(());
        }
        // Same for an audio buffer event: park until the buffer finishes.
        if let Some(done_at) = next_buffer {
            // Park until the buffer's own deadline, then reissue the `svc`.
            self.pc = self.pc.wrapping_sub(4);
            self.sleep_until(done_at);
            return Ok(());
        }
        // Rewind onto the `svc` and park until a signal or display tick wakes us.
        // Reporting a handle as signalled would run that object's handler.
        self.pc = self.pc.wrapping_sub(4);
        self.park_on_events(self.next_display_tick());
        Ok(())
    }

    /// ConnectToNamedPort
    fn svc_connect_to_named_port(&mut self) -> Result<()> {
        let name_ptr = self.read_zr(1);
        let name = if name_ptr != 0 {
            self.read_port_name(name_ptr)
        } else {
            String::new()
        };
        let handle = self.alloc_handle();
        self.record_handle(handle, &name);
        self.grant_handle(handle);
        self.write_zr(0, RESULT_OK);
        self.write_zr(1, handle);
        Ok(())
    }

    /// GetThreadId(out_thread_id, handle): each thread's own id, in X1.
    fn svc_get_thread_id(&mut self) -> Result<()> {
        let handle = self.read_zr(1);
        match self.thread_id(handle) {
            Some(id) => {
                self.write_zr(0, RESULT_OK);
                self.svc_out64(1, 1, 2, id);
            }
            None => self.write_zr(0, RESULT_INVALID_HANDLE),
        }
        Ok(())
    }

    /// Break(reason, arg, size): decode the reason, read the arg when it is an
    /// integer, and print a frame-pointer backtrace.
    fn svc_break(&mut self) -> Result<()> {
        let reason = self.read_zr(0);
        let arg = self.read_zr(1);
        let size = self.read_zr(2);
        let reason_name = match reason & 0xFF {
            0 => "Panic",
            1 => "Assert",
            2 => "User",
            3 => "PreLoadDll",
            4 => "PostLoadDll",
            5 => "PreUnloadDll",
            6 => "PostUnloadDll",
            7 => "CppException",
            _ => "Unknown",
        };
        // Only tells a debugger something, such as a module about to load: it returns.
        const NOTIFICATION_ONLY: u64 = 0x8000_0000;
        if reason & NOTIFICATION_ONLY != 0 {
            self.log(
                crate::trace::Level::Debug,
                &format!("[svcBreak] notification: {reason_name} arg={arg:#x}"),
            );
            self.write_zr(0, 0);
            return Ok(());
        }
        let mut msg =
            format!("[svcBreak] reason={reason_name} ({reason:#x}) arg={arg:#x} size={size:#x}");
        if let Some(value) = match size {
            1 => self.mem.read_u8(arg).ok().map(u64::from),
            2 => self.mem.read_u16(arg).ok().map(u64::from),
            4 => self.mem.read_u32(arg).ok().map(u64::from),
            8 => self.mem.read_u64(arg).ok(),
            _ => None,
        } {
            let _ = write!(msg, " value={value:#x}");
        }
        msg.push('\n');
        for addr in self.backtrace(16) {
            let _ = writeln!(msg, "  {addr:#010x}");
        }
        self.out.extend_from_slice(msg.as_bytes());
        let first = msg.lines().next().unwrap_or_default().to_owned();
        self.exit_process(crate::cpu::Ending::Faulted(first));
        Ok(())
    }

    /// OutputDebugString(ptr, size)
    fn svc_output_debug_string(&mut self) -> Result<()> {
        let ptr = self.read_zr(0);
        let len = (self.read_zr(1) as i64).clamp(0, 4096) as u64;
        if ptr != 0 && len > 0 {
            for i in 0..len {
                match self.mem.read_u8(ptr.wrapping_add(i)) {
                    Ok(b) => self.out.push(b),
                    Err(_) => break,
                }
            }
        }
        Ok(())
    }

    /// GetInfo(out, infoType, handle, infoSubValue): value in X1.
    fn svc_get_info(&mut self) -> Result<()> {
        // Memory pool and system resource figures come from the process layout (`MemoryLayout`).
        let layout = self.memory_layout();
        let total_memory_size = layout.total_memory;
        let system_resource_size = layout.system_resource;
        let info_type = self.read_zr(1);
        let value = match info_type {
            // Core/priority masks from the NPDM; 0 makes `nnSdk` assert.
            0 => self.process().process_core_mask, // CoreMask
            1 => 0x0FFF_FFFF_F000_0000,            // PriorityMask: 28..=59
            // Alias/heap regions from the layout, below 4 GiB (`u32` guest addresses).
            2 => layout.alias_addr, // AliasRegionAddress
            3 => layout.alias_size, // AliasRegionSize
            4 => layout.heap_addr,  // HeapRegionAddress
            5 => layout.heap_size,  // HeapRegionSize
            6 => total_memory_size, // TotalMemorySize
            7 => 0,                 // UsedMemorySize
            8 => 0,                 // DebuggerAttached
            9 => 0,                 // ResourceLimit
            // RandomEntropy: non-zero SplitMix64 per subvalue; `sdk` aborts on zero.
            11 => {
                let sub = self.svc_arg64(3, 0, 3).wrapping_add(1);
                let mut z = sub.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            }
            // System resource size. Non-zero enables `nnSdk`'s virtual address memory
            // manager; see [`GUEST_ALIAS_REGION_SIZE`] and [`VAMM_ARENA_SIZE`].
            16 => system_resource_size, // SystemResourceSizeTotal
            17 => 0,                    // SystemResourceSizeUsed
            // Total/UsedNonSystemMemorySize, which `nn::init` sizes the heap from.
            21 => total_memory_size - system_resource_size,
            22 => 0,
            12 => u64::from(GUEST_ASLR_REGION_ADDR),
            13 => u64::from(GUEST_ASLR_REGION_SIZE),
            // Stack mirror region: clear of `STACK_TOP`, room for several stacks.
            14 => u64::from(GUEST_STACK_REGION_ADDR),
            15 => u64::from(GUEST_STACK_REGION_SIZE),
            20 => 0, // UserExceptionContextAddress
            28 => 0, // AliasRegionExtraSize
            _ => 0,
        };
        if crate::trace::enabled(crate::trace::Trace::Svc) {
            crate::traceln!("[svc]   -> GetInfo({info_type}) = {value:#x}");
        }
        self.svc_out64(1, 1, 2, value);
        self.write_zr(0, RESULT_OK);
        Ok(())
    }

    /// GetSystemInfo(out, handle, infoType): value in X1.
    fn svc_get_system_info(&mut self) -> Result<()> {
        let info_type = self.read_zr(2);
        let value = match info_type {
            2 => 0x1000_0000, // TotalMemorySize
            3 => 0,           // UsedMemorySize
            _ => 0,
        };
        self.write_zr(1, value);
        self.write_zr(0, RESULT_OK);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::{Cpu, CURRENT_THREAD_PSEUDO_HANDLE};

    const RESULT_INVALID_CORE_ID: u64 = 1 | (57 << 9);
    const RESULT_INVALID_COMBINATION: u64 = 1 | (116 << 9);
    const DONT_CARE: u64 = -1i32 as u32 as u64;
    const USE_PROCESS_VALUE: u64 = -2i32 as u32 as u64;
    const NO_UPDATE: u64 = -3i32 as u32 as u64;

    fn create_thread(cpu: &mut Cpu, core: u64) -> (u64, u64) {
        cpu.write_zr(1, 0x0800_0000);
        cpu.write_zr(2, 0);
        cpu.write_zr(3, 0x1000_0000);
        cpu.write_zr(4, 44);
        cpu.write_zr(5, core);
        cpu.horizon_syscall(0x08).unwrap();
        (cpu.read_zr(0), cpu.read_zr(1))
    }

    fn processor_number(cpu: &mut Cpu) -> u64 {
        cpu.horizon_syscall(0x10).unwrap();
        cpu.read_zr(0)
    }

    fn core_mask(cpu: &mut Cpu, handle: u64) -> (u64, u64, u64) {
        cpu.write_zr(2, handle);
        cpu.horizon_syscall(0x0E).unwrap();
        (cpu.read_zr(0), cpu.read_zr(1), cpu.read_zr(2))
    }

    fn set_core_mask(cpu: &mut Cpu, handle: u64, core: u64, mask: u64) -> u64 {
        cpu.write_zr(0, handle);
        cpu.write_zr(1, core);
        cpu.write_zr(2, mask);
        cpu.horizon_syscall(0x0F).unwrap();
        cpu.read_zr(0)
    }

    /// `rtld` walks regions until the address wraps, so past the end is one region to 2^64.
    #[test]
    fn query_memory_past_the_address_space_is_one_inaccessible_region() {
        const OUT: u64 = 0x0800_0000;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(OUT, 0x1000).unwrap();
        let end = cpu.memory_layout().space_end;
        for addr in [end, end + 0x1234_5000, u64::MAX - 0xFFF] {
            cpu.write_zr(0, OUT);
            cpu.write_zr(2, addr);
            cpu.horizon_syscall(0x06).unwrap();
            assert_eq!(cpu.read_zr(0), 0);
            let (base, size) = (
                cpu.mem.read_u64(OUT).unwrap(),
                cpu.mem.read_u64(OUT + 8).unwrap(),
            );
            assert_eq!(base, end, "at {addr:#x}");
            assert_eq!(base.wrapping_add(size), 0, "the region runs to 2^64");
            assert_eq!(cpu.mem.read_u32(OUT + 16).unwrap(), 0x10, "Inaccessible");
            assert_eq!(cpu.mem.read_u32(OUT + 24).unwrap(), 0, "no permission");
        }
    }

    #[test]
    fn query_memory_reports_the_granted_heap_before_it_is_touched() {
        const OUT: u64 = 0x0800_0000;
        const SIZE: u64 = 0x0100_0000;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(OUT, 0x1000).unwrap();
        cpu.write_zr(1, SIZE);
        cpu.horizon_syscall(0x01).unwrap();
        let heap = cpu.read_zr(1);
        assert_eq!(heap, crate::cpu::HEAP_REGION_ADDR);
        cpu.write_zr(0, OUT);
        cpu.write_zr(2, heap + 0x0050_0000);
        cpu.horizon_syscall(0x06).unwrap();
        assert_eq!(cpu.mem.read_u64(OUT).unwrap(), heap);
        assert_eq!(cpu.mem.read_u64(OUT + 8).unwrap(), SIZE);
        assert_eq!(cpu.mem.read_u32(OUT + 16).unwrap(), 5, "Normal");
        cpu.write_zr(0, OUT);
        cpu.write_zr(2, heap - 0x1000);
        cpu.horizon_syscall(0x06).unwrap();
        let (base, len) = (
            cpu.mem.read_u64(OUT).unwrap(),
            cpu.mem.read_u64(OUT + 8).unwrap(),
        );
        assert_eq!(
            base + len,
            heap,
            "the run before the heap ends where it starts"
        );
    }

    #[test]
    fn synchronizing_preemption_state_leaves_an_unpinned_thread_alone() {
        let mut cpu = Cpu::new();
        cpu.write_zr(0, 0x1234);
        cpu.horizon_syscall(0x36).unwrap();
        assert_eq!(cpu.read_zr(0), 0x1234);
    }

    #[test]
    fn a_transfer_memory_is_the_creators_pages_and_goes_with_its_last_handle() {
        const ADDR: u32 = 0x0900_0000;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(u64::from(ADDR), 0x2000).unwrap();
        cpu.mem.write_u32(u64::from(ADDR + 4), 7).unwrap();
        cpu.write_zr(1, u64::from(ADDR));
        cpu.write_zr(2, 0x2000);
        cpu.horizon_syscall(0x15).unwrap();
        assert_eq!(cpu.read_zr(0), 0);
        let handle = cpu.read_zr(1);

        let memory = cpu.transfer_memories[&handle].clone();
        assert_eq!(memory.read_u32(4).unwrap(), 7, "the contents were not kept");
        cpu.mem.write_u32(u64::from(ADDR + 0x1000), 9).unwrap();
        assert_eq!(
            memory.read_u32(0x1000).unwrap(),
            9,
            "a store by the creator did not show in the object"
        );

        cpu.close_handle(handle);
        assert!(cpu.transfer_memories.is_empty());
    }

    #[test]
    fn a_transfer_memory_over_pages_the_guest_lacks_is_refused_before_allocating() {
        const ADDR: u64 = 0x0900_0000;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(ADDR, 0x2000).unwrap();
        let before = cpu.mem.mapped_bytes();
        for size in [0x3000, 0x100_0000_0000, u64::MAX & !0xFFF] {
            cpu.write_zr(1, ADDR);
            cpu.write_zr(2, size);
            cpu.horizon_syscall(0x15).unwrap();
            assert_eq!(cpu.read_zr(0), 1 | (106 << 9), "size {size:#x}");
        }
        assert!(cpu.transfer_memories.is_empty());
        assert_eq!(cpu.mem.mapped_bytes(), before);
    }

    /// Echoes of Wisdom passes the untouched end of its heap, which reads as zero.
    #[test]
    fn a_transfer_memory_over_untouched_heap_is_made() {
        const ADDR: u64 = 0x0900_0000;
        let mut cpu = Cpu::new();
        cpu.mem.soft_map_zero(ADDR, ADDR + 0x10_0000);
        cpu.mem.map_zero(ADDR, 0x1000).unwrap();
        cpu.mem.write_u32(ADDR + 4, 7).unwrap();
        cpu.write_zr(1, ADDR);
        cpu.write_zr(2, 0x4000);
        cpu.horizon_syscall(0x15).unwrap();
        assert_eq!(cpu.read_zr(0), 0);
        let memory = cpu.transfer_memories[&cpu.read_zr(1)].clone();
        assert_eq!(
            memory.read_u32(4).unwrap(),
            7,
            "the backed page was not kept"
        );
        assert_eq!(memory.read_u32(0x3000).unwrap(), 0);
        assert_eq!(cpu.mem.mapped_bytes_in(ADDR, ADDR + 0x4000), 0x4000);
    }

    #[test]
    fn a_transfer_memory_past_the_memory_cap_is_refused() {
        const ADDR: u64 = 0x0900_0000;
        let mut cpu = Cpu::new();
        cpu.mem.soft_map_zero(ADDR, ADDR + 0x10_0000);
        cpu.mem.map_zero(ADDR, 0x1000).unwrap();
        let before = cpu.mem.mapped_bytes();
        cpu.mem.set_max_mapped_bytes(before + 0x2000);
        cpu.write_zr(1, ADDR);
        cpu.write_zr(2, 0x4000);
        cpu.horizon_syscall(0x15).unwrap();
        assert_eq!(cpu.read_zr(0), 1 | (106 << 9));
        assert!(cpu.transfer_memories.is_empty());
        assert_eq!(cpu.mem.mapped_bytes(), before);
    }

    /// Each thread reports the core it was put on.
    #[test]
    fn each_thread_reports_the_core_it_was_created_on() {
        let mut cpu = Cpu::new();
        let (result, worker) = create_thread(&mut cpu, 2);
        assert_eq!(result, 0);
        assert!(cpu.start_thread(worker));
        assert_eq!(processor_number(&mut cpu), 0, "the main thread");
        while cpu.current_thread_handle() != worker {
            cpu.yield_thread();
        }
        assert_eq!(processor_number(&mut cpu), 2, "the worker");
    }

    #[test]
    fn the_process_default_core_is_the_main_threads() {
        let mut cpu = Cpu::new();
        cpu.set_main_thread_core(1);
        let (result, worker) = create_thread(&mut cpu, USE_PROCESS_VALUE);
        assert_eq!(result, 0);
        assert_eq!(core_mask(&mut cpu, worker), (0, 1, 0b10));
    }

    /// Core 3 is the system's; anything past it is no core.
    #[test]
    fn a_thread_cannot_be_created_on_a_core_the_process_lacks() {
        let mut cpu = Cpu::new();
        assert_eq!(create_thread(&mut cpu, 3).0, RESULT_INVALID_CORE_ID);
        assert_eq!(create_thread(&mut cpu, 7).0, RESULT_INVALID_CORE_ID);
    }

    /// A manifest granting core 3 alone may move a thread there.
    #[test]
    fn a_system_applet_gets_the_cores_its_manifest_grants() {
        let mut cpu = Cpu::new();
        cpu.set_process_core_mask(0b1000);
        let me = CURRENT_THREAD_PSEUDO_HANDLE;
        assert_eq!(set_core_mask(&mut cpu, me, DONT_CARE, 0b1000), 0);
        assert_eq!(create_thread(&mut cpu, 3).0, 0);
        assert_eq!(create_thread(&mut cpu, 0).0, RESULT_INVALID_CORE_ID);
        cpu.write_zr(1, 0);
        cpu.write_zr(2, 0xFFFF_8001);
        cpu.horizon_syscall(0x29).unwrap();
        assert_eq!(cpu.read_zr(1), 0b1000, "svcGetInfo CoreMask");
    }

    #[test]
    fn a_core_mask_is_checked_kept_and_read_back() {
        let mut cpu = Cpu::new();
        let me = CURRENT_THREAD_PSEUDO_HANDLE;
        assert_eq!(
            set_core_mask(&mut cpu, me, 2, 0b011),
            RESULT_INVALID_COMBINATION,
            "an ideal core outside its own mask"
        );
        assert_eq!(
            set_core_mask(&mut cpu, me, 1, 0b1000),
            RESULT_INVALID_CORE_ID,
            "a mask naming the system's core"
        );
        assert_eq!(set_core_mask(&mut cpu, me, 1, 0b011), 0);
        assert_eq!(core_mask(&mut cpu, me), (0, 1, 0b011));
        assert_eq!(processor_number(&mut cpu), 1);
        assert_eq!(set_core_mask(&mut cpu, me, NO_UPDATE, 0b110), 0);
        assert_eq!(core_mask(&mut cpu, me), (0, 1, 0b110));
        assert_eq!(
            set_core_mask(&mut cpu, me, NO_UPDATE, 0b100),
            RESULT_INVALID_COMBINATION,
            "a kept ideal core the new mask leaves out"
        );
        assert_eq!(set_core_mask(&mut cpu, 0xdead, 1, 0b011), 1 | (114 << 9));
    }

    #[test]
    fn a_cancelled_thread_answers_its_next_wait_once() {
        const RESULT_CANCELLED: u64 = 1 | (118 << 9);
        let mut cpu = Cpu::new();
        cpu.write_zr(0, CURRENT_THREAD_PSEUDO_HANDLE);
        cpu.horizon_syscall(0x19).unwrap();
        assert_eq!(cpu.read_zr(0), 0);
        cpu.write_zr(2, 0);
        cpu.write_zr(3, u64::MAX);
        cpu.horizon_syscall(0x18).unwrap();
        assert_eq!(cpu.read_zr(0), RESULT_CANCELLED);
        assert!(!cpu.take_sync_cancelled());
        cpu.write_zr(0, 0xdead);
        cpu.horizon_syscall(0x19).unwrap();
        assert_eq!(cpu.read_zr(0), 1 | (114 << 9));
    }

    fn thread_id(cpu: &mut Cpu, handle: u64) -> (u64, u64) {
        cpu.write_zr(1, handle);
        cpu.horizon_syscall(0x25).unwrap();
        (cpu.read_zr(0), cpu.read_zr(1))
    }

    /// Every thread has its own id.
    #[test]
    fn each_thread_has_an_id_of_its_own() {
        let mut cpu = Cpu::new();
        let (_, a) = create_thread(&mut cpu, 0);
        let (_, b) = create_thread(&mut cpu, 0);
        let main = thread_id(&mut cpu, CURRENT_THREAD_PSEUDO_HANDLE);
        let (a_id, b_id) = (thread_id(&mut cpu, a), thread_id(&mut cpu, b));
        assert_eq!(main.0, 0);
        assert_eq!((a_id.0, b_id.0), (0, 0));
        let ids = [main.1, a_id.1, b_id.1];
        assert!(ids.iter().all(|&id| id != 0), "{ids:?}");
        assert!(
            ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2],
            "{ids:?}"
        );
        cpu.start_thread(a);
        while cpu.current_thread_handle() != a {
            cpu.yield_thread();
        }
        assert_eq!(thread_id(&mut cpu, CURRENT_THREAD_PSEUDO_HANDLE).1, a_id.1);
        assert_eq!(thread_id(&mut cpu, 0xdead).0, 1 | (114 << 9));
    }

    /// No ideal core and a one-core mask moves the thread to that core.
    #[test]
    fn a_mask_without_an_ideal_core_moves_the_thread_only_when_it_must() {
        let mut cpu = Cpu::new();
        let me = CURRENT_THREAD_PSEUDO_HANDLE;
        assert_eq!(set_core_mask(&mut cpu, me, DONT_CARE, 0b100), 0);
        assert_eq!(core_mask(&mut cpu, me), (0, DONT_CARE, 0b100));
        assert_eq!(processor_number(&mut cpu), 2);
        assert_eq!(set_core_mask(&mut cpu, me, DONT_CARE, 0b110), 0);
        assert_eq!(processor_number(&mut cpu), 2, "still allowed, so not moved");
    }
}
