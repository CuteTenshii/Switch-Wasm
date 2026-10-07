//! The Horizon supervisor calls (`SVC`) libnx homebrew issues at runtime.

use crate::cpu::{
    ArbiterWait, Cpu, GUEST_ASLR_REGION_ADDR, GUEST_ASLR_REGION_SIZE, GUEST_SPACE_END,
    GUEST_STACK_REGION_ADDR, GUEST_STACK_REGION_SIZE, HID_SHMEM_SIZE, PL_SHMEM_SIZE,
};
use crate::{Error, Result};
use std::fmt::Write;

impl Cpu {
    /// Every SVC a guest issues. `svc #0` (unused by Horizon) is the host halt trap.
    pub(crate) fn syscall(&mut self, imm: u16) -> Result<()> {
        if imm == 0 {
            self.halted = true;
            return Ok(());
        }
        self.horizon_syscall(imm)
    }

    /// Horizon syscalls by libnx's numbering (`nx/source/kernel/svc.s`).
    /// X0 carries the Result; out-handles and values come back in X1.
    pub(crate) fn horizon_syscall(&mut self, imm: u16) -> Result<()> {
        const RESULT_OK: u64 = 0;
        const FAKE_HANDLE: u64 = 0x1000;
        const RESULT_INVALID_MEMORY_RANGE: u64 = 0x8000_DC01;
        // Kernel (module 1) description 114: invalid handle.
        const RESULT_INVALID_HANDLE: u64 = 1 | (114 << 9);
        // Kernel descriptions 57 and 116: invalid core id and ideal core outside the mask.
        const RESULT_INVALID_CORE_ID: u64 = 1 | (57 << 9);
        const RESULT_INVALID_COMBINATION: u64 = 1 | (116 << 9);
        // Ideal-core values that are not a core: process default, none, no update.
        const IDEAL_CORE_USE_PROCESS_VALUE: i32 = -2;
        const IDEAL_CORE_DONT_CARE: i32 = -1;
        const IDEAL_CORE_NO_UPDATE: i32 = -3;
        // Memory pool and system resource figures come from the process layout (`MemoryLayout`).
        let layout = self.memory_layout();
        let total_memory_size = u64::from(layout.total_memory);
        let system_resource_size = u64::from(layout.system_resource);
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
            0x01 => {
                // SetHeapSize(size): a heap larger than its region is refused.
                const RESULT_OUT_OF_MEMORY: u64 = 1 | (104 << 9);
                let size = self.read_zr(1);
                if size > u64::from(layout.heap_size) {
                    self.write_zr(0, RESULT_OUT_OF_MEMORY);
                    return Ok(());
                }
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, u64::from(layout.heap_addr));
                Ok(())
            }
            0x02 | 0x03 | 0x14 => {
                // SetMemoryPermission / SetMemoryAttribute / UnmapSharedMemory
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x04 => {
                // MapMemory(dst, src, size): really back dst, since `virtmemFindStack` looks
                // for unmapped ranges to place the next thread's stack mirror.
                let dst = self.read_zr(0) as u32;
                let src = self.read_zr(1) as u32;
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
            0x2C => {
                // MapPhysicalMemory(address, size): how 39-bit titles grow their heap. Pages
                // materialise on first write via the soft-mapped low 2 GiB.
                let addr = self.read_zr(0);
                let size = self.read_zr(1);
                let fits = addr
                    .checked_add(size)
                    .is_some_and(|end| end <= u64::from(u32::MAX) + 1);
                if size == 0 || (addr & 0xFFF) != 0 || (size & 0xFFF) != 0 || !fits {
                    self.write_zr(0, RESULT_INVALID_MEMORY_RANGE);
                    return Ok(());
                }
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x2D => {
                // UnmapPhysicalMemory(address, size): pages are really freed for the RAM cap.
                let addr = self.read_zr(0) as u32;
                let size = self.read_zr(1) as usize;
                self.mem.unmap(addr, size);
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x13 => {
                // MapSharedMemory(handle, addr, size, perm): zeroed backing. hid's gets a
                // connected pad published at once; pl's gets the shared font.
                let addr = self.read_zr(1) as u32;
                let size = self.read_zr(2) as u32;
                self.mem.map_zero(addr, size as usize)?;
                // Prefer `hid`'s handle; the size match is a fallback.
                if Some(self.read_zr(0)) == self.hid_shmem_handle || size == HID_SHMEM_SIZE {
                    self.hid_shmem_addr = addr;
                    self.set_gamepad_state(0, 0, 0, 0, 0);
                    // An empty touch sample: a zeroed LIFO header is a ring with no capacity.
                    self.set_touch_state(&[]);
                } else if size == PL_SHMEM_SIZE {
                    self.pl_shmem_addr = addr;
                    self.write_shared_font(addr);
                }
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x05 => {
                // UnmapMemory(dst, src, size). An out-of-range unmap reports a 39-bit space,
                // which hbmenu probes for.
                let dst = self.read_zr(0);
                if (dst >> 48) == 0xFFFF {
                    self.write_zr(0, 0x8000_D401);
                    return Ok(());
                }
                let src = self.read_zr(1) as u32;
                let size = self.read_zr(2) as usize;
                if dst != 0 && src != 0 && size != 0 {
                    self.mem.copy_range(src, dst as u32, size)?;
                    self.mem.unmap(dst as u32, size);
                }
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x06 => {
                // QueryMemory(info, pageInfo, addr): the run of pages in the queried page's
                // state. Untouched soft-mapped pages read as unmapped. Only module `.text` is
                // executable, since `rtld` finds modules by walking for executable `CodeStatic`.
                let out = self.read_zr(0) as u32;
                let addr = self.read_zr(2) as u32;
                // See [`crate::mem::Memory::state_run`].
                let run = self.mem.state_run(addr, GUEST_SPACE_END);
                let (base, end, mapped, text) = (run.start, run.end, run.mapped, run.readonly);
                let mut info = Vec::with_capacity(40);
                info.extend_from_slice(&(base as u64).to_le_bytes());
                info.extend_from_slice(&((end - base) as u64).to_le_bytes());
                for v in [
                    run.state as u32, // type
                    0,
                    if text {
                        0b101
                    } else if mapped {
                        0b011
                    } else {
                        0
                    }, // perm (R-X / RW- / none)
                    0,
                    0,
                    0,
                ] {
                    info.extend_from_slice(&v.to_le_bytes());
                }
                for (i, &b) in info.iter().enumerate() {
                    self.mem.write_u8(out.wrapping_add(i as u32), b)?;
                }
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, if mapped { 0x1000 } else { 0 }); // page info
                Ok(())
            }
            0x07 => {
                // ExitProcess. Only the main thread returns to the exit stub; any other
                // thread here is an emulator bug, so say so loudly.
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
                    self.diagnostic(crate::trace::Level::Error, &line);
                }
                self.halted = true;
                Ok(())
            }
            0x0A => {
                // ExitThread: only the main thread ending stops the process.
                self.exit_thread();
                Ok(())
            }
            0x08 => {
                // CreateThread(entry = X1, arg = X2, stack_top = X3, priority = W4, core = W5)
                // -> handle in X1. Starts suspended with its own TLS block.
                let entry = self.read_zr(1) as u32;
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
                    const RESULT_INVALID_PRIORITY: u64 = 1 | (112 << 9);
                    self.write_zr(0, RESULT_INVALID_PRIORITY);
                    return Ok(());
                }
                let core = match core as u32 as i32 {
                    IDEAL_CORE_USE_PROCESS_VALUE => self.main_thread_core,
                    core @ 0..=3 if self.process_core_mask >> core & 1 != 0 => core as u8,
                    _ => {
                        self.write_zr(0, RESULT_INVALID_CORE_ID);
                        return Ok(());
                    }
                };
                let handle = self.create_thread(entry, arg, stack_top, priority as u8, core);
                if crate::trace::enabled(crate::trace::Trace::Wait) {
                    crate::traceln!(
                        "[thread] create handle={handle:#x} entry={entry:#x} core={core}"
                    );
                }
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, handle);
                Ok(())
            }
            0x09 => {
                // StartThread
                let handle = self.read_zr(0);
                let started = self.start_thread(handle);
                if crate::trace::enabled(crate::trace::Trace::Wait) {
                    crate::traceln!("[thread] start handle={handle:#x} -> {started}");
                }
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x32 => {
                // SetThreadActivity(handle, activity): 0 = Runnable, 1 = Paused. The caller
                // can't suspend itself, and a no-op change reports busy.
                const RESULT_BUSY: u64 = 1 | (122 << 9);
                const RESULT_INVALID_STATE: u64 = 1 | (125 << 9);
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
            0x33 => {
                // GetThreadContext3(out = X0, handle = X1): the suspended thread's registers.
                let out = self.read_zr(0) as u32;
                let handle = self.read_zr(1);
                let ok = self.write_thread_context(out, handle);
                self.write_zr(0, if ok { RESULT_OK } else { RESULT_INVALID_HANDLE });
                Ok(())
            }
            0x0B => {
                // SleepThread(nanoseconds = X0). 0, -1 and -2 are yield modes; other values
                // really sleep, which matters for `nn::os` poll loops.
                let nanoseconds = self.svc_arg64(0, 0, 1) as i64;
                self.write_zr(0, RESULT_OK);
                match self.wait_deadline(nanoseconds) {
                    Some(deadline) => self.sleep_until(deadline),
                    None => self.yield_thread(),
                }
                Ok(())
            }
            0x1A => {
                // ArbitrateLock(owner = W0, mutex = X1, self = W2)
                let owner = self.read_zr(0) as u32;
                let mutex = self.read_zr(1) as u32;
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
            0x1B => {
                // ArbitrateUnlock(mutex = X0)
                let mutex = self.read_zr(0) as u32;
                self.write_zr(0, RESULT_OK);
                self.arbitrate_unlock(mutex);
                Ok(())
            }
            0x1C => {
                // WaitProcessWideKeyAtomic(mutex = X0, key = X1, self = W2, timeout = X3)
                let mutex = self.read_zr(0) as u32;
                let key = self.read_zr(1) as u32;
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
            0x34 => {
                // WaitForAddress(address = X0, arb_type = W1, value = W2, timeout = X3).
                // If the arb_type predicate doesn't hold, report InvalidState instead of blocking.
                const RESULT_INVALID_STATE: u64 = 1 | (125 << 9);
                const RESULT_TIMED_OUT: u64 = 0xEA01;
                let addr = self.read_zr(0) as u32;
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
            0x35 => {
                // SignalToAddress(address = X0, signal_type = W1, value = W2, count = W3).
                const RESULT_INVALID_STATE: u64 = 1 | (125 << 9);
                let addr = self.read_zr(0) as u32;
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
            0x1D => {
                // SignalProcessWideKey(key = X0, count = W1)
                let key = self.read_zr(0) as u32;
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
            0x0C => {
                // GetThreadPriority(handle = X1) -> priority in W1.
                match self.thread_priority(self.read_zr(1)) {
                    Some(priority) => {
                        self.write_zr(0, RESULT_OK);
                        self.write_zr(1, u64::from(priority));
                    }
                    None => self.write_zr(0, RESULT_INVALID_HANDLE),
                }
                Ok(())
            }
            0x0D => {
                // SetThreadPriority(handle = X0, priority = W1)
                const RESULT_INVALID_PRIORITY: u64 = 1 | (112 << 9);
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
            0x0E => {
                // GetThreadCoreMask(handle = W2) -> ideal core W1, affinity X2 (A32 r2:r3).
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
            0x0F => {
                // SetThreadCoreMask(handle = W0, ideal core = W1, affinity = X2, r2:r3 on A32),
                // checked in the kernel's order.
                let handle = self.read_zr(0);
                let requested = self.read_zr(1) as u32 as i32;
                let affinity = self.svc_arg64(2, 2, 3);
                let Some((current_ideal, _)) = self.thread_core_mask(handle) else {
                    self.write_zr(0, RESULT_INVALID_HANDLE);
                    return Ok(());
                };
                let checked = match requested {
                    IDEAL_CORE_USE_PROCESS_VALUE => Ok((
                        i32::from(self.main_thread_core),
                        1u64 << self.main_thread_core,
                    )),
                    _ if affinity & !self.process_core_mask != 0 => Err(RESULT_INVALID_CORE_ID),
                    _ if affinity == 0 => Err(RESULT_INVALID_COMBINATION),
                    core @ 0..=3 if affinity & (1 << core) == 0 => Err(RESULT_INVALID_COMBINATION),
                    core @ 0..=3 => Ok((core, affinity)),
                    IDEAL_CORE_DONT_CARE => Ok((IDEAL_CORE_DONT_CARE, affinity)),
                    IDEAL_CORE_NO_UPDATE
                        if current_ideal >= 0 && affinity & (1 << current_ideal) == 0 =>
                    {
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
            0x16 | 0x17 | 0x28 | 0x5F => {
                // CloseHandle / CancelSynchronization / ReturnFromException / FlushProcessDataCache
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x19 => {
                // ResetSignal(handle): clear an event and report whether it was signalled.
                const RESULT_INVALID_STATE: u64 = 1 | (125 << 9);
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
            0x15 => {
                // CreateTransferMemory
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, FAKE_HANDLE);
                Ok(())
            }
            0x18 => {
                // WaitSynchronization(out_index, handles, num_handles, timeout). X1 is the
                // index of the signalled handle.
                const RESULT_TIMED_OUT: u64 = 0xEA01;
                let handles_ptr = self.read_zr(1) as u32;
                let count = (self.read_zr(2) as u32).min(0x40);
                let timeout = self.svc_arg64(3, 0, 3) as i64;
                let handles: Vec<u64> = (0..count)
                    .map(|i| {
                        u64::from(
                            self.mem
                                .read_u32(handles_ptr.wrapping_add(i * 4))
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
                let refreshed = self.cycles.wrapping_sub(self.last_vsync_cycles)
                    >= crate::cpu::VSYNC_PERIOD_CYCLES;
                if self.nv.gpu.frames != self.last_vsync_frame || refreshed {
                    self.last_vsync_frame = self.nv.gpu.frames;
                    self.last_vsync_cycles = self.cycles;
                    if let Some(vsync) = self.vsync_event {
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
                // Nothing fired. A poll times out. A blocking wait parks only while another
                // thread can run (timing out would hand `MultiWaitImpl::WaitAny` a null holder).
                // Results are written before yielding, which switches register files.
                if timeout == 0 {
                    self.write_zr(0, RESULT_TIMED_OUT);
                    return Ok(());
                }
                // A wait on no handles can never be satisfied and either answer jumps to 0,
                // so rewind onto the `svc` and yield.
                if handles.is_empty() && self.has_other_runnable() {
                    self.pc = self.pc.wrapping_sub(4);
                    self.yield_thread();
                    return Ok(());
                }
                // A blocking wait on vsync is certain to fire, so park until the tick.
                if timeout != 0
                    && !crate::env_flag!("NO_VSYNC_THROTTLE")
                    && self.vsync_event.is_some_and(|v| handles.contains(&v))
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
            0x1E => {
                // GetSystemTick: 19.2 MHz, scaled from cycles of the 1.02 GHz CPU.
                self.svc_out64(0, 0, 1, self.system_tick());
                Ok(())
            }
            0x1F => {
                // ConnectToNamedPort
                let name_ptr = self.read_zr(1) as u32;
                let name = if name_ptr != 0 {
                    self.read_port_name(name_ptr)
                } else {
                    String::new()
                };
                let handle = self.alloc_handle();
                self.record_handle(handle, &name);
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, handle);
                Ok(())
            }
            0x20..=0x23 => {
                // SendSyncRequest variants: a named service's stub, else the generic reply.
                let tls = self.tpidr as u32;
                let handle = self.read_zr(0);
                let cmd_id = self.ipc_command_id(tls);
                let svc_name = self.service_name(handle).map(|s| s.to_string());
                if crate::trace::enabled(crate::trace::Trace::Ipc) {
                    let obj = self.ipc_domain_object_id(tls);
                    let iface = self.domain_interface(handle, obj).map(|s| s.to_string());
                    crate::traceln!(
                        "[ipc] pc={:#x} h={} svc={:?} obj={} iface={:?} domain={} type={} cmd={:?}",
                        self.pc,
                        handle,
                        svc_name,
                        obj,
                        iface,
                        self.ipc_is_domain_request(tls),
                        self.ipc_message_type(tls),
                        cmd_id
                    );
                    let words: Vec<String> = (0..8)
                        .map(|i| format!("{:08x}", self.mem.read_u32(tls + i * 4).unwrap_or(0)))
                        .collect();
                    crate::traceln!("[ipc]   svc={:#x} tls={:#x} {}", imm, tls, words.join(" "));
                }
                // A Close request (type 2) has no command id; it tears down the session.
                if self.ipc_message_type(tls) == 2 {
                    self.forget_handle(handle);
                    self.write_ipc_response(tls, 0, &[], &[], &[])?;
                    self.write_zr(0, RESULT_OK);
                    return Ok(());
                }
                // CloneCurrentObject (control 2) and its Ex form (4) reply with a new session
                // handle as a move handle, reaching the same interface and domain objects.
                if self.ipc_is_control_request(tls) && matches!(cmd_id, Some(2) | Some(4)) {
                    if let Some(name) = svc_name.clone() {
                        let clone = self.alloc_handle();
                        self.record_handle(clone, &name);
                        let objects: Vec<(u32, String)> = self
                            .domain_objects
                            .iter()
                            .filter(|((h, _), _)| *h == handle)
                            .map(|((_, obj), iface)| (*obj, iface.clone()))
                            .collect();
                        for (obj, iface) in objects {
                            self.record_domain_object(clone, obj, &iface);
                        }
                        self.write_ipc_response(tls, 0, &[clone], &[], &[])?;
                        self.write_zr(0, RESULT_OK);
                        return Ok(());
                    }
                }
                // QueryPointerBufferSize (control 3): `nnSdk` checks pointer args against it.
                if self.ipc_is_control_request(tls) && cmd_id == Some(3) {
                    let size = super::ipc::POINTER_BUFFER_SIZE.to_le_bytes();
                    self.write_ipc_response(tls, 0, &[], &size, &[])?;
                    self.write_zr(0, RESULT_OK);
                    return Ok(());
                }
                // A domain Close has no command id; handle it before any service sees it.
                if self.ipc_is_domain_close(tls) {
                    let object_id = self.ipc_domain_object_id(tls);
                    self.close_domain_object(tls, handle, object_id)?;
                    self.write_zr(0, RESULT_OK);
                    return Ok(());
                }
                if let Some(name) = svc_name {
                    match name.as_str() {
                        "sm:" | "sm" => self.sm_request(tls, cmd_id, handle)?,
                        "fsp-srv" | "fsp-srv:" => {
                            // Domain sub-objects on fsp-srv, routed by recorded interface.
                            let object_id = self.ipc_domain_object_id(tls);
                            match self.domain_interface(handle, object_id) {
                                Some("fsp-srv-fs") => self.fs_request(tls, cmd_id, handle)?,
                                Some("fsp-srv-fs-dir") => self.fs_dir_request(
                                    tls,
                                    cmd_id,
                                    Self::object_key(handle, object_id),
                                )?,
                                Some("fsp-srv-fs-file") => self.fs_file_request(
                                    tls,
                                    cmd_id,
                                    Self::object_key(handle, object_id),
                                )?,
                                Some("fsp-srv-storage") => {
                                    self.fs_storage_request(tls, handle, cmd_id)?
                                }
                                Some("fsp-srv-save-info-reader") => {
                                    self.fs_save_data_info_reader_request(tls, cmd_id)?
                                }
                                Some("fsp-srv-device-operator") => {
                                    self.fs_device_operator_request(tls, cmd_id)?
                                }
                                Some("fsp-srv-sd-detection")
                                | Some("fsp-srv-gamecard-detection") => {
                                    self.fs_detection_notifier_request(tls, handle, cmd_id)?
                                }
                                _ => self.fsp_srv_request(tls, cmd_id, handle)?,
                            }
                        }
                        // The same interfaces over their own session handle (non-domain callers).
                        "fsp-srv-fs" => self.fs_request(tls, cmd_id, handle)?,
                        "fsp-srv-fs-dir" => {
                            self.fs_dir_request(tls, cmd_id, Self::object_key(handle, 0))?
                        }
                        "fsp-srv-fs-file" => {
                            self.fs_file_request(tls, cmd_id, Self::object_key(handle, 0))?
                        }
                        "fsp-srv-storage" => self.fs_storage_request(tls, handle, cmd_id)?,
                        "fsp-srv-save-info-reader" => {
                            self.fs_save_data_info_reader_request(tls, cmd_id)?
                        }
                        "fsp-srv-device-operator" => {
                            self.fs_device_operator_request(tls, cmd_id)?
                        }
                        "fsp-srv-sd-detection" | "fsp-srv-gamecard-detection" => {
                            self.fs_detection_notifier_request(tls, handle, cmd_id)?
                        }
                        "vi:m" | "vi:m:" => self.vi_request(tls, handle, cmd_id)?,
                        // fatal:u, the guest giving up.
                        "fatal:u" | "fatal:p" => self.fatal_request(tls, cmd_id)?,
                        "set" => self.set_request(tls, handle, cmd_id)?,
                        "set:sys" => self.set_sys_request(tls, cmd_id)?,
                        "nvdrv" | "nvdrv:" | "nvdrv:a" | "nvdrv:a:" | "nvdrv:s" | "nvdrv:t" => {
                            self.nvdrv_request(tls, cmd_id, handle)?
                        }
                        "pl:u" | "pl:s" => self.pl_request(tls, cmd_id)?,
                        // caps:a, the screenshot album.
                        "caps:a" => self.caps_album_accessor_request(tls, handle, cmd_id)?,
                        // time:* sub-interfaces as domain out-objects.
                        "time:s" | "time:u" | "time:a" | "time:r" => {
                            let object_id = self.ipc_domain_object_id(tls);
                            match self.domain_interface(handle, object_id) {
                                Some("time:system-clock") => {
                                    self.time_system_clock_request(tls, cmd_id)?
                                }
                                Some("time:steady-clock") => {
                                    self.time_steady_clock_request(tls, cmd_id)?
                                }
                                Some("time:timezone") => self.time_timezone_request(tls, cmd_id)?,
                                _ => self.time_request(tls, cmd_id, handle)?,
                            }
                        }
                        // The same over their own session handle.
                        "time:system-clock" => self.time_system_clock_request(tls, cmd_id)?,
                        "time:steady-clock" => self.time_steady_clock_request(tls, cmd_id)?,
                        "time:timezone" => self.time_timezone_request(tls, cmd_id)?,
                        // psm (power state management).
                        "psm" => {
                            let object_id = self.ipc_domain_object_id(tls);
                            match self.domain_interface(handle, object_id) {
                                Some("psm-session") => self.psm_session_request(tls, cmd_id)?,
                                _ => self.psm_request(tls, cmd_id, handle)?,
                            }
                        }
                        "psm-session" => self.psm_session_request(tls, cmd_id)?,
                        // appletOE / appletAE and the `am` sub-interfaces over their own handles.
                        "appletOE"
                        | "appletAE"
                        | "am:proxy-service"
                        | "am:application-proxy"
                        | "am:common-state-getter"
                        | "am:self-controller"
                        | "am:window-controller"
                        | "am:audio-controller"
                        | "am:display-controller"
                        | "am:library-applet-creator"
                        | "am:application-functions"
                        | "am:library-applet-proxy"
                        | "am:system-applet-proxy"
                        | "am:library-applet-self-accessor"
                        | "am:applet-common-functions"
                        | "am:system-process-common-functions"
                        | "am:application-observer"
                        | "am:applet-alternative-functions"
                        | "am:process-winding-controller"
                        | "am:home-menu-functions"
                        | "am:global-state-controller"
                        | "am:application-creator"
                        | "am:lock-accessor"
                        | "am:storage"
                        | "am:storage-accessor"
                        | "am:debug-functions" => self.applet_request(tls, handle, cmd_id)?,
                        // nifm at all three privilege levels.
                        "nifm:u" | "nifm:s" | "nifm:a" => {
                            let object_id = self.ipc_domain_object_id(tls);
                            match self.domain_interface(handle, object_id) {
                                Some("nifm:general-service") => {
                                    self.nifm_general_service_request(tls, handle, cmd_id)?
                                }
                                Some("nifm:request") => {
                                    self.nifm_request_object_request(tls, cmd_id)?
                                }
                                _ => self.nifm_request(tls, cmd_id, handle)?,
                            }
                        }
                        // The same two over their own session handles.
                        "nifm:general-service" => {
                            self.nifm_general_service_request(tls, handle, cmd_id)?
                        }
                        "nifm:request" => self.nifm_request_object_request(tls, cmd_id)?,
                        // ssl and its contexts.
                        "ssl" | "ssl:service" | "ssl:context" => {
                            self.ssl_request(tls, handle, cmd_id)?
                        }
                        // hid, IAppletResource and IActiveVibrationDeviceList.
                        "hid"
                        | "hid:dbg"
                        | "hid:sys"
                        | "hid:server"
                        | "hid:applet-resource"
                        | "hid:vibration-devices" => self.hid_request(tls, handle, cmd_id)?,
                        // lm, the log manager.
                        "lm" | "lm:service" | "lm:logger" => {
                            self.lm_request(tls, handle, cmd_id)?
                        }
                        // acc, the user accounts.
                        "acc:u0" | "acc:u1" | "acc:su" | "acc:profile" | "acc:profile-editor"
                        | "acc:manager" | "acc:async-context" | "acc:notifier" => {
                            self.acc_request(tls, handle, cmd_id)?
                        }
                        // ns, installed content.
                        "ns:am"
                        | "ns:am2"
                        | "ns:ec"
                        | "ns:rid"
                        | "ns:rt"
                        | "ns:web"
                        | "ns:ro"
                        | "ns:su"
                        | "ns:vm"
                        | "ns:dev"
                        | "ns:app-manager"
                        | "ns:read-only-record"
                        | "ns:read-only-control"
                        | "ns:content-management"
                        | "ns:download-task"
                        | "ns:account-proxy"
                        | "ns:app-version"
                        | "ns:factory-reset"
                        | "ns:ecommerce"
                        | "ns:dynamic-rights"
                        | "ns:document" => self.ns_request(tls, handle, cmd_id)?,
                        // aoc, add-on content.
                        "aoc:u" => self.aoc_request(tls, handle, cmd_id)?,
                        // ldr:ro, used by `nn::ro`.
                        "ldr:ro" => self.ldr_ro_request(tls, handle, cmd_id)?,
                        // csrng and `spl:`.
                        "csrng" => self.csrng_request(tls, cmd_id)?,
                        "spl:" | "spl:mig" | "spl:fs" | "spl:ssl" | "spl:es" | "spl:manu" => {
                            self.spl_request(tls, cmd_id)?
                        }
                        // pdm, play history.
                        "pdm:qry" | "pdm:ntfy" | "pdm:info" => self.pdm_request(tls, cmd_id)?,
                        // prepo, play reports.
                        "prepo:u" | "prepo:a" | "prepo:a2" | "prepo:m" | "prepo:s" => {
                            self.prepo_request(tls, cmd_id)?
                        }
                        // pm, the process manager.
                        "pm:shell" | "pm:dmnt" | "pm:info" | "pm:bm" => {
                            self.pm_request(tls, handle, cmd_id)?
                        }
                        // pcv and clkrst, the clock manager.
                        "pcv" | "clkrst" | "clkrst:i" | "clkrst:session-0" | "clkrst:session-1"
                        | "clkrst:session-2" | "clkrst:session-3" => {
                            self.pcv_request(tls, handle, cmd_id)?
                        }
                        // mm:u, multimedia clocks.
                        "mm:u" => self.mm_request(tls, cmd_id)?,
                        // ts, temperature sensors.
                        "ts" | "ts:u" | "ts:s" | "ts:session-internal" | "ts:session-external" => {
                            self.ts_request(tls, handle, cmd_id)?
                        }
                        // sfdnsres, the DNS resolver.
                        "sfdnsres" => self.sfdnsres_request(tls, cmd_id)?,
                        // bsd, sockets.
                        "bsd:u" | "bsd:s" => self.bsd_request(tls, handle, cmd_id)?,
                        // apm, clock profiles.
                        "apm" | "apm:p" | "apm:am" | "apm:sys" | "apm:session" => {
                            self.apm_request(tls, handle, cmd_id)?
                        }
                        // pctl and IParentalControlService.
                        "pctl" | "pctl:s" | "pctl:a" | "pctl:r" | "pctl:factory"
                        | "pctl:service" => self.pctl_request(tls, handle, cmd_id)?,
                        // audout, PCM output.
                        "audout:u" | "audout:a" | "audout:d" => self.audout_request(tls, cmd_id)?,
                        "audout:iaudioout" => self.audio_out_request(tls, cmd_id, handle)?,
                        // psc, power-state notifications.
                        "psc:m" | "psc:service" | "psc:module" => {
                            self.psc_request(tls, handle, cmd_id)?
                        }
                        // gpio and its pad sessions.
                        "gpio" | "gpio:pad" => self.gpio_request(tls, handle, cmd_id)?,
                        // Mii database.
                        "mii:e" | "mii:u" | "mii:s" => self.mii_request(tls, handle, cmd_id)?,
                        "mii:database" | "mii:static" => self.mii_request(tls, handle, cmd_id)?,
                        "miiimg" => self.miiimg_request(tls, cmd_id)?,
                        // hwopus, the Opus decoder.
                        "hwopus" | "hwopus:decoder" => self.hwopus_request(tls, handle, cmd_id)?,
                        "audren:u" => self.audren_request(tls, handle, cmd_id)?,
                        "audren:iaudiorenderer" => {
                            self.audren_renderer_request(tls, cmd_id, handle)?
                        }
                        "audren:iaudiodevice" => self.audio_device_request(tls, cmd_id, handle)?,
                        // lbl, the backlight.
                        "lbl" => self.lbl_request(tls, handle, cmd_id)?,
                        // audctl, system audio settings.
                        "audctl" => self.audctl_request(tls, handle, cmd_id)?,
                        // nfc:sys; no reader attached.
                        "nfc:sys" | "nfc:system" => self.nfc_request(tls, handle, cmd_id)?,
                        // btm:sys, the Bluetooth manager.
                        "btm:sys" | "btm:core" => self.btm_request(tls, handle, cmd_id)?,
                        // ngc, the profanity filter.
                        "ngc:u" | "ngct:u" | "ngct:s" => self.ngc_request(tls, handle, cmd_id)?,
                        "npns:s" | "npns:u" => self.npns_request(tls, handle, cmd_id)?,
                        // ldn:m and lp2p:m, local wireless monitors.
                        "ldn:m" | "ldn:monitor" => self.ldn_monitor_request(tls, handle, cmd_id)?,
                        "lp2p:m" | "lp2p:monitor" => {
                            self.lp2p_monitor_request(tls, handle, cmd_id)?
                        }
                        // ovln, the overlay message queue.
                        "ovln:snd" | "ovln:rcv" | "ovln:sender" | "ovln:receiver" => {
                            self.ovln_request(tls, handle, cmd_id)?
                        }
                        // olsc, save-data cloud backup.
                        "olsc:s"
                        | "olsc:system-service"
                        | "olsc:transfer-task-list"
                        | "olsc:remote-storage"
                        | "olsc:daemon"
                        | "olsc:transfer-end-holder"
                        | "olsc:transfer-start-holder"
                        | "olsc:error-holder"
                        | "olsc:stopper" => self.olsc_request(tls, handle, cmd_id)?,
                        // friend and its IServiceCreator interfaces.
                        "friend:u"
                        | "friend:v"
                        | "friend:m"
                        | "friend:s"
                        | "friend:a"
                        | "friend:service"
                        | "friend:notification"
                        | "friend:daemon-suspend-session" => {
                            self.friend_request(tls, handle, cmd_id)?
                        }
                        // news and its article store.
                        "news:a"
                        | "news:c"
                        | "news:m"
                        | "news:p"
                        | "news:v"
                        | "news:service"
                        | "news:arrival-event"
                        | "news:overwrite-event"
                        | "news:data"
                        | "news:database" => self.news_request(tls, handle, cmd_id)?,
                        // bcat, background delivery.
                        "bcat:a" | "bcat:m" | "bcat:u" | "bcat:s" | "bcat:service"
                        | "bcat:storage" | "bcat:progress" | "bcat:notifier"
                        | "bcat:suspension" | "bcat:file" | "bcat:directory" => {
                            self.bcat_request(tls, handle, cmd_id)?
                        }
                        // notif, alarms and notifications.
                        "notif:a" | "notif:s" | "notif:event-accessor" => {
                            self.notif_request(tls, handle, cmd_id)?
                        }
                        // erpt, error reports.
                        "erpt:c" => self.erpt_context_request(tls, handle, cmd_id)?,
                        "erpt:r" | "erpt:report" | "erpt:manager" | "erpt:attachment" => {
                            self.erpt_session_request(tls, handle, cmd_id)?
                        }
                        name => {
                            // Known service without a stub: reply with a fabricated sub-session
                            // (`reply_with_fabricated_object`).
                            let name = name.to_string();
                            // Control commands first: a fabricated pointer buffer size would
                            // make callers use pointer buffers, which this IPC layer doesn't read.
                            if self.ipc_is_control_request(tls) {
                                match cmd_id {
                                    Some(0) => {
                                        let obj = self.alloc_domain_object();
                                        self.record_domain_object(handle, obj, &name);
                                        self.write_ipc_response(
                                            tls,
                                            0,
                                            &[],
                                            &obj.to_le_bytes(),
                                            &[],
                                        )?;
                                    }
                                    _ => {
                                        self.write_ipc_response(
                                            tls,
                                            0,
                                            &[],
                                            &0u16.to_le_bytes(),
                                            &[],
                                        )?;
                                    }
                                }
                                self.write_zr(0, RESULT_OK);
                                return Ok(());
                            }
                            self.warn_no_implementation(&name, cmd_id);
                            self.reply_with_fabricated_object(tls, handle, &name, cmd_id)?
                        }
                    }
                } else {
                    // Unrecognized session: try the vi stub, else the generic object-id reply.
                    if let Some(cmd) = cmd_id {
                        // vi: GetIApplicationDisplayService (2) and display/session (100+).
                        if cmd == 2 || cmd >= 100 {
                            return self.vi_request(tls, handle, cmd_id);
                        }
                    }
                    let start = self.ipc_reply_start(tls);
                    let is_domain = self
                        .mem
                        .read_u32(tls.wrapping_add(start + 0x10))
                        .unwrap_or(0)
                        == 0x4943_4653;
                    // A fresh object id for a caller that expects an out-object.
                    self.warn_no_implementation("<untracked session>", cmd_id);
                    let data = {
                        let obj = self.next_object_id;
                        self.next_object_id = obj.wrapping_add(1);
                        obj
                    };
                    if is_domain {
                        for i in 0..4u32 {
                            let _ = self.mem.write_u32(tls.wrapping_add(start + i * 4), 0);
                        }
                        let _ = self
                            .mem
                            .write_u32(tls.wrapping_add(start + 0x10), 0x4F43_4653);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x14), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x18), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x1C), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x20), data);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x24), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x28), data);
                    } else {
                        let _ = self.mem.write_u32(tls.wrapping_add(start), 0x4F43_4653);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x04), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x08), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x0C), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x10), data);
                    }
                }
                self.write_zr(0, RESULT_OK);
                // Yield after X0 is written: `yield_thread` swaps register files.
                if std::mem::take(&mut self.pending_yield) {
                    self.yield_thread();
                }
                // Same for a service that asked to sleep until a deadline (`vi`'s present).
                if let Some(until) = std::mem::take(&mut self.pending_sleep) {
                    self.sleep_until(until);
                }
                Ok(())
            }
            0x24 => {
                // GetProcessId(out_process_id, process_handle): Result in X0, id in X1.
                self.write_zr(0, RESULT_OK);
                self.svc_out64(1, 1, 2, 1);
                Ok(())
            }
            0x25 => {
                // GetThreadId(out_thread_id, handle): each thread's own id, in X1.
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
            0x26 => {
                // Break(reason, arg, size): decode the reason, read the arg when it is an
                // integer, and print a frame-pointer backtrace.
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
                let mut msg = format!(
                    "[svcBreak] reason={reason_name} ({reason:#x}) arg={arg:#x} size={size:#x}"
                );
                if let Some(value) = match size {
                    1 => self.mem.read_u8(arg as u32).ok().map(u64::from),
                    2 => self.mem.read_u16(arg as u32).ok().map(u64::from),
                    4 => self.mem.read_u32(arg as u32).ok().map(u64::from),
                    8 => self.mem.read_u64(arg as u32).ok(),
                    _ => None,
                } {
                    let _ = write!(msg, " value={value:#x}");
                }
                msg.push('\n');
                for addr in self.backtrace(16) {
                    let _ = writeln!(msg, "  {addr:#010x}");
                }
                self.out.extend_from_slice(msg.as_bytes());
                self.halted = true;
                Ok(())
            }
            0x27 => {
                // OutputDebugString(ptr, size)
                let ptr = self.read_zr(0) as u32;
                let len = (self.read_zr(1) as i64).clamp(0, 4096) as u32;
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
            0x29 => {
                // GetInfo(out, infoType, handle, infoSubValue): value in X1.
                let info_type = self.read_zr(1);
                let value = match info_type {
                    // Core/priority masks from the NPDM; 0 makes `nnSdk` assert.
                    0 => self.process_core_mask, // CoreMask
                    1 => 0x0FFF_FFFF_F000_0000,  // PriorityMask: 28..=59
                    // Alias/heap regions from the layout, below 4 GiB (`u32` guest addresses).
                    2 => u64::from(layout.alias_addr), // AliasRegionAddress
                    3 => u64::from(layout.alias_size), // AliasRegionSize
                    4 => u64::from(layout.heap_addr),  // HeapRegionAddress
                    5 => u64::from(layout.heap_size),  // HeapRegionSize
                    6 => total_memory_size,            // TotalMemorySize
                    7 => 0,                            // UsedMemorySize
                    8 => 0,                            // DebuggerAttached
                    9 => 0,                            // ResourceLimit
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
            0x6F => {
                // GetSystemInfo(out, handle, infoType): value in X1.
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
            _ => Err(Error::Cpu(format!(
                "unimplemented Horizon syscall #{:#x}",
                imm
            ))),
        }
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
