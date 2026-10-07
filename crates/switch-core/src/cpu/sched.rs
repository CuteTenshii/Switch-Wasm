//! Thread scheduling, display pacing and context switching.

use super::*;

/// Instructions per 60 Hz display refresh at 1.02 GHz; vsync fires even without a present.
pub const VSYNC_PERIOD_CYCLES: u64 = 1_020_000_000 / 60;

/// Instructions per 200 Hz `hid` sample; LIFOs advance even with no input.
pub const HID_SAMPLE_PERIOD_CYCLES: u64 = 1_020_000_000 / 200;

/// Instructions a thread runs before preemption.
pub(super) const TIME_SLICE: u64 = 20_000;

/// Default thread priority (also what most retail manifests declare); 0 is most urgent, 63 least.
pub const DEFAULT_THREAD_PRIORITY: u8 = 44;

pub(super) const MAIN_THREAD_ID: u64 = 1;

pub(super) const LOWEST_PRIORITY: u8 = 63;
/// The console has four cores, 0 to 3.
pub(super) const LAST_CORE: u8 = 3;

/// Decisions a runnable thread may be passed over before it runs regardless, since
/// all threads share one host core.
pub(super) const STARVE_DECISIONS: u32 = 8;

impl Cpu {
    /// Park the presenting thread until the next 60 Hz refresh, as a swapchain would.
    /// Takes effect after the reply is written; see [`Cpu::pending_sleep`].
    pub(super) fn pace_present(&mut self) {
        let tick = self.last_present_cycles.wrapping_add(VSYNC_PERIOD_CYCLES);
        if self.cycles < tick {
            self.pending_sleep = Some(tick);
        }
        // A title slower than the panel is not dragged backwards.
        self.last_present_cycles = self.cycles.max(tick);
    }

    /// Present a frame whose surface was still on the device, once it has come back.
    pub(super) fn complete_pending_present(&mut self) {
        let Some(buffer) = self.pending_present else {
            return;
        };
        match self.nv.gpu.flush_renderers(&mut self.mem) {
            Ok(crate::gpu::renderer::Flush::Pending) => {}
            Ok(crate::gpu::renderer::Flush::Done) => {
                self.pending_present = None;
                if let Err(e) = self.nv.gpu.present(&self.mem, &buffer) {
                    self.diagnostic(
                        Level::Error,
                        &format!("[vi] the deferred present failed: {e}"),
                    );
                }
            }
            Err(e) => {
                // Drop the frame rather than stall the display.
                self.pending_present = None;
                self.diagnostic(
                    Level::Error,
                    &format!("[vi] the deferred readback failed: {e}"),
                );
            }
        }
    }

    pub(super) fn sleep_until(&mut self, deadline: u64) {
        self.ensure_main_thread();
        self.threads[self.current_thread].state = ThreadState::Sleeping { deadline };
        self.reschedule();
    }

    /// The next display refresh, a deadline that always arrives.
    pub(super) fn next_display_tick(&self) -> u64 {
        self.last_vsync_cycles.wrapping_add(VSYNC_PERIOD_CYCLES)
    }

    /// Park on awaited events until one fires or `deadline` passes; the `svc` is reissued.
    pub(super) fn park_on_events(&mut self, deadline: u64) {
        self.ensure_main_thread();
        self.threads[self.current_thread].state = ThreadState::WaitEvent { deadline };
        self.reschedule();
    }

    /// Switch away after blocking. If nothing can run, idle or wake everything.
    pub(super) fn reschedule(&mut self) {
        if self.switch_to_next_runnable() {
            return;
        }
        // Idle the clock to the earliest deadline of any kind, as the console idles.
        if let Some(deadline) = self.earliest_deadline() {
            if deadline > self.cycles {
                self.cycles = deadline;
            }
            self.expire_timed_waits();
            if self.switch_to_next_runnable() {
                return;
            }
        }
        // No deadline either: wake everything rather than hang.
        for index in 0..self.threads.len() {
            match self.threads[index].state {
                ThreadState::WaitKey { mutex, .. } => self.wake_condvar_waiter(index, mutex),
                ThreadState::WaitMutex(_)
                | ThreadState::WaitAddress { .. }
                | ThreadState::WaitEvent { .. } => {
                    self.threads[index].state = ThreadState::Runnable;
                }
                _ => {}
            }
        }
        self.switch_to_next_runnable();
    }

    /// Account one retired instruction: a cycle and a step. Both engines call this.
    #[inline(always)]
    pub(super) fn retire(&mut self) {
        self.cycles += 1;
        self.steps += 1;
    }

    /// Record a run of `count` instructions from `start` in the fault trail.
    #[inline(always)]
    pub(super) fn record_run(&mut self, start: u32, count: u32) {
        // A single step that continues the last run extends it.
        if count == 1 && self.recent_len != 0 {
            let index = (self.recent_len - 1) & (RECENT_LEN - 1);
            let (last_start, last_count) = self.recent[index];
            if last_count < RECENT_LEN as u32 && last_start.wrapping_add(last_count * 4) == start {
                self.recent[index].1 = last_count + 1;
                return;
            }
        }
        self.push_run(start, count);
    }

    /// [`Cpu::record_run`] without the single-step merge, which only pays off
    /// for the interpreter's steps.
    #[inline(always)]
    pub(super) fn push_run(&mut self, start: u32, count: u32) {
        self.recent[self.recent_len % RECENT_LEN] = (start, count);
        self.recent_len = self.recent_len.wrapping_add(1);
    }

    /// Switch to the thread [`Cpu::pick_next`] chooses. Returns false if there
    /// is none (in which case the running thread keeps going).
    pub(super) fn switch_to_next_runnable(&mut self) -> bool {
        let start = self.current_thread;
        let Some(candidate) = self.pick_next() else {
            return false;
        };
        self.pass_over_all_but(candidate);
        self.account_slice(start);
        self.threads[candidate].switches += 1;
        self.save_context(start);
        self.load_context(candidate);
        true
    }

    fn save_context(&mut self, index: usize) {
        let thread = &mut self.threads[index];
        thread.regs = self.regs;
        thread.pc = self.pc;
        thread.nzcv = self.nzcv;
        thread.mode = self.mode;
        thread.cpsr_q = self.cpsr_q;
        thread.cpsr_ge = self.cpsr_ge;
        thread.fpscr_nzcv = self.fpscr_nzcv;
        thread.vregs = self.vregs;
        thread.fpcr = self.fpcr;
        thread.fpsr = self.fpsr;
        thread.tpidr = self.tpidr;
        thread.tpidr_rw = self.tpidr_rw;
    }

    fn load_context(&mut self, index: usize) {
        self.slice_used = 0;
        // A switch clears the local monitor.
        self.exclusive = None;
        let thread = self.threads[index].clone();
        self.regs = thread.regs;
        self.pc = thread.pc;
        self.nzcv = thread.nzcv;
        self.mode = thread.mode;
        self.cpsr_q = thread.cpsr_q;
        self.cpsr_ge = thread.cpsr_ge;
        self.fpscr_nzcv = thread.fpscr_nzcv;
        self.vregs = thread.vregs;
        self.fpcr = thread.fpcr;
        self.fpsr = thread.fpsr;
        self.tpidr = thread.tpidr;
        self.tpidr_rw = thread.tpidr_rw;
        self.current_thread = index;
    }

    pub fn current_thread_handle(&self) -> u64 {
        self.threads
            .get(self.current_thread)
            .map_or(MAIN_THREAD_HANDLE, |t| t.handle)
    }

    /// Threads created, including the main thread.
    pub fn thread_count(&self) -> usize {
        self.threads.len().max(1)
    }
}
