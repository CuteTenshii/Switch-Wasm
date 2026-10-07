//! Guest thread creation, priorities and affinity.

use crate::cpu::*;

/// A guest thread's state. Threads only switch at blocking syscalls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadState {
    /// Created but not yet started with `svcStartThread`.
    Created,
    Runnable,
    /// Returned from its entry point or called `svcExitThread`.
    Finished,
    /// Blocked in `svcArbitrateLock` on the mutex word at this address.
    WaitMutex(u32),
    /// Blocked in `svcWaitProcessWideKeyAtomic`; re-acquires `mutex` when woken.
    /// `deadline` is the expiry cycle for timed waits.
    WaitKey {
        key: u32,
        mutex: u32,
        deadline: Option<u64>,
    },
    /// Blocked in `svcWaitForAddress` until signalled or `deadline` passes.
    WaitAddress {
        addr: u32,
        deadline: Option<u64>,
    },
    /// Asleep until `deadline` with the PC on the `svc`, which is reissued on wake.
    Sleeping {
        deadline: u64,
    },
    /// Blocked in `svcWaitSynchronization` with the PC on the `svc`; reissued when
    /// [`Cpu::signal_event`] wakes it or at `deadline` (the display tick).
    WaitEvent {
        deadline: u64,
    },
}

#[derive(Debug, Clone)]
pub struct ThreadContext {
    pub handle: u64,
    /// Kernel thread id (`svcGetThreadId`), distinct from the handle.
    pub(crate) id: u64,
    pub state: ThreadState,
    /// Suspended by `svcSetThreadActivity`; independent of `state`.
    pub(crate) paused: bool,
    /// Saved register file, SP included; see [`REG_SLOTS`].
    pub(crate) regs: [u64; REG_FILE],
    pub(crate) pc: u32,
    pub(crate) nzcv: u32,
    pub(crate) mode: ExecMode,
    pub(crate) cpsr_q: bool,
    pub(crate) cpsr_ge: u8,
    pub(crate) fpscr_nzcv: u32,
    pub(crate) vregs: VRegs,
    pub(crate) fpcr: u32,
    pub(crate) fpsr: u32,
    pub(crate) tpidr: u64,
    pub(crate) tpidr_rw: u64,
    /// 0 (most urgent) to 63; see [`Cpu::pick_next`].
    pub(crate) priority: u8,
    /// Current core, ideal core (-1 for none) and affinity mask. Not scheduled on;
    /// `core` is what `GetCurrentProcessorNumber` answers.
    pub(crate) core: u8,
    pub(crate) ideal_core: i32,
    pub(crate) affinity: u64,
    /// Decisions passed over while runnable; see [`STARVE_DECISIONS`].
    pub(crate) passed_over: u32,
    /// Entry point and argument (an `nn::os` thread's `ThreadType`), for the thread report.
    pub(crate) entry: u32,
    pub(crate) arg: u64,
    /// Instructions retired and times scheduled since the last [`ThreadReport`].
    pub(crate) ran: u64,
    pub(crate) switches: u64,
    /// Clock when it last did real work; see [`ThreadReport::idle_ms`].
    pub(crate) busy_at: u64,
}

impl Cpu {
    // ---- guest threads ----

    /// Created on demand so a single-threaded program costs nothing.
    pub(crate) fn ensure_main_thread(&mut self) {
        if self.threads.is_empty() {
            self.threads.push(ThreadContext {
                handle: MAIN_THREAD_HANDLE,
                id: MAIN_THREAD_ID,
                state: ThreadState::Runnable,
                paused: false,
                regs: [0; REG_FILE],
                pc: 0,
                nzcv: 0,
                mode: self.mode,
                cpsr_q: false,
                cpsr_ge: 0,
                fpscr_nzcv: 0,
                vregs: VRegs::default(),
                fpcr: self.fpcr,
                fpsr: 0,
                tpidr: self.tpidr,
                tpidr_rw: self.tpidr_rw,
                priority: self.main_thread_priority,
                core: self.main_thread_core,
                ideal_core: i32::from(self.main_thread_core),
                affinity: 1 << self.main_thread_core,
                passed_over: 0,
                entry: 0,
                arg: 0,
                ran: 0,
                switches: 0,
                busy_at: self.cycles,
            });
            self.current_thread = 0;
        }
    }

    /// Create a thread as `svcCreateThread` does (TLS with libnx `ThreadVars`, stack,
    /// entry, argument in x0). Returns its handle.
    pub(crate) fn create_thread(
        &mut self,
        entry: u32,
        arg: u64,
        stack_top: u64,
        priority: u8,
        core: u8,
    ) -> u64 {
        self.ensure_main_thread();
        let handle = self.alloc_handle();
        let index = self.threads.len() as u32;
        let tls = THREAD_TLS_BASE + index * THREAD_TLS_STRIDE;
        let _ = self.mem.map_zero(tls, THREAD_TLS_STRIDE as usize);
        // ThreadVars at TLS+0x1E0: magic, handle, thread pointer, reent, tls_tp.
        const TV_MAGIC: u32 = 0x2154_5624; // "!TV$"
        let reent = tls + 0x400;
        let _ = self.mem.write_u32(tls + 0x1E0, TV_MAGIC);
        let _ = self.mem.write_u32(tls + 0x1E4, handle as u32);
        let _ = self.mem.write_u32(tls + 0x1E8, 0);
        let _ = self.mem.write_u32(tls + 0x1F0, reent);
        let _ = self.mem.write_u32(tls + 0x1F8, tls);

        let mut regs = [0u64; REG_FILE];
        regs[0] = arg;
        // Inherit the creator's execution state; LR and SP slots differ per mode.
        if self.mode == ExecMode::A32 {
            regs[14] = THREAD_EXIT_TRAMPOLINE as u64;
            regs[13] = stack_top;
        } else {
            regs[30] = THREAD_EXIT_TRAMPOLINE as u64;
            regs[SP_SLOT] = stack_top;
        }
        let id = self.next_thread_id;
        self.next_thread_id += 1;
        self.threads.push(ThreadContext {
            handle,
            id,
            state: ThreadState::Created,
            paused: false,
            regs,
            pc: entry,
            nzcv: 0,
            mode: self.mode,
            cpsr_q: false,
            cpsr_ge: 0,
            fpscr_nzcv: 0,
            vregs: VRegs::default(),
            fpcr: 0,
            fpsr: 0,
            tpidr: u64::from(tls),
            tpidr_rw: 0,
            priority: priority.min(LOWEST_PRIORITY),
            core,
            ideal_core: i32::from(core),
            affinity: 1 << core,
            passed_over: 0,
            entry,
            arg,
            ran: 0,
            switches: 0,
            busy_at: self.cycles,
        });
        let line = format!(
            "{} created by {}: arg {arg:#x}, stack top {stack_top:#x}, priority {priority}, \
             core {core}",
            self.thread_label(handle),
            self.thread_label(self.current_thread_handle())
        );
        self.log_thread(line);
        handle
    }

    /// Mark a created thread runnable (`svcStartThread`).
    pub(crate) fn start_thread(&mut self, handle: u64) -> bool {
        for thread in &mut self.threads {
            if thread.handle == handle && thread.state == ThreadState::Created {
                thread.state = ThreadState::Runnable;
                let line = format!("{} started", self.thread_label(handle));
                self.log_thread(line);
                return true;
            }
        }
        let line = format!(
            "{} asked to start, but it is not a thread waiting to be started",
            self.thread_label(handle)
        );
        self.log_thread(line);
        false
    }

    /// `svcSetThreadActivity`. `Err(())` when already in the requested state, as Horizon reports.
    pub(crate) fn set_thread_paused(&mut self, handle: u64, paused: bool) -> Option<bool> {
        let thread = self.threads.iter_mut().find(|t| t.handle == handle)?;
        if thread.paused == paused {
            return Some(false);
        }
        thread.paused = paused;
        let line = format!(
            "{} {} by {}",
            self.thread_label(handle),
            if paused { "paused" } else { "resumed" },
            self.thread_label(self.current_thread_handle())
        );
        self.log_thread(line);
        Some(true)
    }

    /// Fill the 0x320-byte `ThreadContext` for `svcGetThreadContext3` from the live or saved registers.
    pub(crate) fn write_thread_context(&mut self, out: u32, handle: u64) -> bool {
        self.ensure_main_thread();
        let Some(index) = self.threads.iter().position(|t| t.handle == handle) else {
            return false;
        };
        let (regs, pc, nzcv, vregs, fpcr, fpsr, tpidr) = if index == self.current_thread {
            (
                self.regs, self.pc, self.nzcv, self.vregs, self.fpcr, self.fpsr, self.tpidr,
            )
        } else {
            let t = &self.threads[index];
            (t.regs, t.pc, t.nzcv, t.vregs, t.fpcr, t.fpsr, t.tpidr)
        };
        let sp = regs[SP_SLOT];
        let put64 = |cpu: &mut Self, off: u32, v: u64| {
            let _ = cpu.mem.write_u64(out.wrapping_add(off), v);
        };
        for (i, &r) in regs.iter().take(29).enumerate() {
            put64(self, i as u32 * 8, r);
        }
        put64(self, 0xE8, regs[29]); // fp
        put64(self, 0xF0, regs[30]); // lr
        put64(self, 0xF8, sp);
        put64(self, 0x100, u64::from(pc));
        let _ = self.mem.write_u32(out.wrapping_add(0x108), nzcv);
        let _ = self.mem.write_u32(out.wrapping_add(0x10C), 0);
        for (i, &v) in vregs.iter().enumerate() {
            let at = 0x110 + i as u32 * 16;
            put64(self, at, v as u64);
            put64(self, at + 8, (v >> 64) as u64);
        }
        let _ = self.mem.write_u32(out.wrapping_add(0x310), fpcr);
        let _ = self.mem.write_u32(out.wrapping_add(0x314), fpsr);
        put64(self, 0x318, tpidr);
        true
    }

    pub(crate) fn has_other_runnable(&self) -> bool {
        self.threads
            .iter()
            .enumerate()
            .any(|(i, t)| i != self.current_thread && t.state == ThreadState::Runnable && !t.paused)
    }

    /// End the running thread and switch away; the process ends only when the main thread exits.
    pub(crate) fn exit_thread(&mut self) {
        self.ensure_main_thread();
        let line = format!(
            "{} exited at {}",
            self.thread_label(self.current_thread_handle()),
            self.locate(self.pc)
        );
        self.log_thread(line);
        if self.current_thread == 0 {
            self.halted = true;
            return;
        }
        self.threads[self.current_thread].state = ThreadState::Finished;
        // Joiners are parked on this thread's handle, now signalled.
        self.wake_event_waiters();
        if !self.switch_to_next_runnable() {
            // Nothing else can run: fall back to the main thread.
            self.threads[0].state = ThreadState::Runnable;
            self.switch_to_next_runnable();
        }
    }

    /// Give the CPU to [`Cpu::pick_next`]'s choice, unless the running thread should keep it.
    pub(crate) fn yield_thread(&mut self) {
        if self.threads.len() < 2 || !self.has_other_runnable() {
            return;
        }
        let current = self.current_thread;
        let still_runnable =
            self.threads[current].state == ThreadState::Runnable && !self.threads[current].paused;
        if still_runnable {
            if let Some(next) = self.pick_next() {
                let other = &self.threads[next];
                if other.passed_over < STARVE_DECISIONS
                    && self.threads[current].priority < other.priority
                {
                    self.pass_over_all_but(current);
                    return;
                }
            }
        }
        self.switch_to_next_runnable();
    }

    /// Next thread to run: a starved one first, then by priority, round-robin among equals.
    pub(crate) fn pick_next(&self) -> Option<usize> {
        let count = self.threads.len();
        let start = self.current_thread;
        let mut best: Option<usize> = None;
        for step in 1..count {
            let candidate = (start + step) % count;
            let thread = &self.threads[candidate];
            if thread.state != ThreadState::Runnable || thread.paused {
                continue;
            }
            if thread.passed_over >= STARVE_DECISIONS {
                return Some(candidate);
            }
            if best.is_none_or(|b| thread.priority < self.threads[b].priority) {
                best = Some(candidate);
            }
        }
        best
    }

    /// Count a passed-over decision against every runnable thread except `chosen`.
    pub(crate) fn pass_over_all_but(&mut self, chosen: usize) {
        for (index, thread) in self.threads.iter_mut().enumerate() {
            if index == chosen {
                thread.passed_over = 0;
            } else if thread.state == ThreadState::Runnable && !thread.paused {
                thread.passed_over = thread.passed_over.saturating_add(1);
            }
        }
    }

    /// `svcGetThreadPriority`; `None` for a non-thread handle.
    pub(crate) fn thread_priority(&self, handle: u64) -> Option<u8> {
        let handle = self.resolve_thread_handle(handle);
        match self.threads.iter().find(|t| t.handle == handle) {
            Some(thread) => Some(thread.priority),
            // No main thread slot yet; use the manifest priority.
            None if handle == MAIN_THREAD_HANDLE => Some(self.main_thread_priority),
            None => None,
        }
    }

    /// `svcSetThreadPriority`; `false` for a non-thread handle.
    pub(crate) fn set_thread_priority(&mut self, handle: u64, priority: u8) -> bool {
        let handle = self.resolve_thread_handle(handle);
        self.ensure_main_thread();
        match self.threads.iter_mut().find(|t| t.handle == handle) {
            Some(thread) => {
                thread.priority = priority;
                true
            }
            None => false,
        }
    }

    /// Resolve `CURRENT_THREAD` to the running thread's handle.
    fn resolve_thread_handle(&self, handle: u64) -> u64 {
        if handle == CURRENT_THREAD_PSEUDO_HANDLE {
            self.current_thread_handle()
        } else {
            handle
        }
    }

    /// The main thread's priority from `main.npdm`.
    pub fn set_main_thread_priority(&mut self, priority: u8) {
        self.main_thread_priority = priority.min(LOWEST_PRIORITY);
        if let Some(main) = self
            .threads
            .iter_mut()
            .find(|t| t.handle == MAIN_THREAD_HANDLE)
        {
            main.priority = self.main_thread_priority;
        }
    }

    /// The process core mask from `main.npdm`, for `svcGetInfo` and thread core checks.
    pub fn set_process_core_mask(&mut self, mask: u64) {
        self.process_core_mask = mask;
    }

    /// The main thread's core from `main.npdm`, also the process's default core.
    pub fn set_main_thread_core(&mut self, core: u8) {
        self.main_thread_core = core.min(LAST_CORE);
        if let Some(main) = self
            .threads
            .iter_mut()
            .find(|t| t.handle == MAIN_THREAD_HANDLE)
        {
            main.core = self.main_thread_core;
            main.ideal_core = i32::from(self.main_thread_core);
            main.affinity = 1 << self.main_thread_core;
        }
    }

    pub(crate) fn thread_id(&mut self, handle: u64) -> Option<u64> {
        let handle = self.resolve_thread_handle(handle);
        self.ensure_main_thread();
        self.threads
            .iter()
            .find(|t| t.handle == handle)
            .map(|t| t.id)
    }

    pub(crate) fn current_core(&self) -> u8 {
        self.threads
            .get(self.current_thread)
            .map_or(self.main_thread_core, |t| t.core)
    }

    /// A thread's ideal core (-1 for none) and affinity mask.
    pub(crate) fn thread_core_mask(&mut self, handle: u64) -> Option<(i32, u64)> {
        let handle = self.resolve_thread_handle(handle);
        self.ensure_main_thread();
        self.threads
            .iter()
            .find(|t| t.handle == handle)
            .map(|t| (t.ideal_core, t.affinity))
    }

    /// Set a thread's ideal core and affinity mask and migrate it as the kernel does.
    /// `false` for an unknown handle.
    pub(crate) fn set_thread_core_mask(
        &mut self,
        handle: u64,
        ideal_core: i32,
        affinity: u64,
    ) -> bool {
        let handle = self.resolve_thread_handle(handle);
        self.ensure_main_thread();
        match self.threads.iter_mut().find(|t| t.handle == handle) {
            Some(thread) => {
                thread.ideal_core = ideal_core;
                thread.affinity = affinity;
                if ideal_core >= 0 {
                    thread.core = ideal_core as u8;
                } else if affinity & (1 << thread.core) == 0 {
                    thread.core = affinity.trailing_zeros() as u8;
                }
                true
            }
            None => false,
        }
    }
}
