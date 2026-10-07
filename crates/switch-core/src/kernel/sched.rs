//! Thread scheduling, display pacing and context switching.

use crate::cpu::*;
use crate::trace::Level;

/// Instructions per 60 Hz display refresh at 1.02 GHz; vsync fires even without a present.
pub const VSYNC_PERIOD_CYCLES: u64 = 1_020_000_000 / 60;

/// Instructions per 200 Hz `hid` sample; LIFOs advance even with no input.
pub const HID_SAMPLE_PERIOD_CYCLES: u64 = 1_020_000_000 / 200;

/// Instructions a thread runs before preemption.
pub(crate) const TIME_SLICE: u64 = 20_000;

/// Default thread priority (also what most retail manifests declare); 0 is most urgent, 63 least.
pub const DEFAULT_THREAD_PRIORITY: u8 = 44;

pub(crate) const MAIN_THREAD_ID: u64 = 1;

pub(crate) const LOWEST_PRIORITY: u8 = 63;
/// The console has four cores, 0 to 3.
pub(crate) const LAST_CORE: u8 = 3;

/// Decisions a runnable thread may be passed over before it runs regardless, since
/// all threads share one host core.
pub(crate) const STARVE_DECISIONS: u32 = 8;

impl Cpu {
    /// Park the presenting thread until the next 60 Hz refresh, as a swapchain would.
    /// Takes effect after the reply is written; see [`Cpu::pending_sleep`].
    pub(crate) fn pace_present(&mut self) {
        let tick = self.last_present_cycles.wrapping_add(VSYNC_PERIOD_CYCLES);
        if self.cycles < tick {
            self.pending_sleep = Some(tick);
        }
        // A title slower than the panel is not dragged backwards.
        self.last_present_cycles = self.cycles.max(tick);
    }

    /// Present a frame whose surface was still on the device, once it has come back.
    pub(crate) fn complete_pending_present(&mut self) {
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

    pub(crate) fn sleep_until(&mut self, deadline: u64) {
        self.ensure_main_thread();
        self.threads[self.current_thread].state = ThreadState::Sleeping { deadline };
        self.reschedule();
    }

    /// The next display refresh, a deadline that always arrives.
    pub(crate) fn next_display_tick(&self) -> u64 {
        self.last_vsync_cycles.wrapping_add(VSYNC_PERIOD_CYCLES)
    }

    /// Park on awaited events until one fires or `deadline` passes; the `svc` is reissued.
    pub(crate) fn park_on_events(&mut self, deadline: u64) {
        self.ensure_main_thread();
        self.threads[self.current_thread].state = ThreadState::WaitEvent { deadline };
        self.reschedule();
    }

    /// Switch away after blocking. If nothing can run, idle or wake everything.
    pub(crate) fn reschedule(&mut self) {
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
    pub(crate) fn retire(&mut self) {
        self.cycles += 1;
        self.steps += 1;
    }

    /// Record a run of `count` instructions from `start` in the fault trail.
    #[inline(always)]
    pub(crate) fn record_run(&mut self, start: u32, count: u32) {
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
    pub(crate) fn push_run(&mut self, start: u32, count: u32) {
        self.recent[self.recent_len % RECENT_LEN] = (start, count);
        self.recent_len = self.recent_len.wrapping_add(1);
    }

    /// Switch to the thread [`Cpu::pick_next`] chooses. Returns false if there
    /// is none (in which case the running thread keeps going).
    pub(crate) fn switch_to_next_runnable(&mut self) -> bool {
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

#[cfg(test)]
mod tests {
    use crate::cpu::{
        Cpu, CURRENT_THREAD_PSEUDO_HANDLE, DEFAULT_THREAD_PRIORITY, MAIN_THREAD_HANDLE,
    };

    #[test]
    fn the_idle_moves_the_clock_and_leaves_the_step_count_alone() {
        // Idling advances the clock but not the instruction count.
        let mut cpu = Cpu::new();
        cpu.bootstrap();
        let (clock, steps) = (cpu.cycles, cpu.steps);

        cpu.sleep_until(clock + 1_000_000);

        assert_eq!(
            cpu.cycles,
            clock + 1_000_000,
            "the clock idled to the deadline"
        );
        assert_eq!(cpu.steps, steps, "the idle executed nothing");
    }

    /// Run `rounds` yielding decisions and count how often each thread got the CPU.
    fn shares(cpu: &mut Cpu, rounds: usize) -> Vec<usize> {
        let mut held = vec![0; cpu.threads.len()];
        for _ in 0..rounds {
            cpu.yield_thread();
            held[cpu.current_thread] += 1;
        }
        held
    }

    /// The most urgent thread gets most of the CPU; others still run within `STARVE_DECISIONS`.
    #[test]
    fn the_most_urgent_thread_runs_most_and_starves_nobody() {
        let mut cpu = Cpu::new();
        let urgent = cpu.create_thread(0x0800_0000, 0, 0x1000_0000, 30, 0);
        let idle = cpu.create_thread(0x0800_0000, 0, 0x1100_0000, 50, 0);
        assert!(cpu.start_thread(urgent) && cpu.start_thread(idle));

        let held = shares(&mut cpu, 900);
        assert!(
            held[1] > held[0] * 4,
            "the urgent thread dominates: {held:?}"
        );
        assert!(
            held[1] > held[2] * 4,
            "the urgent thread dominates: {held:?}"
        );
        assert!(held[0] > 0 && held[2] > 0, "nobody is starved: {held:?}");
    }

    /// Equal priorities take turns.
    #[test]
    fn threads_of_one_priority_take_turns() {
        let mut cpu = Cpu::new();
        let a = cpu.create_thread(0x0800_0000, 0, 0x1000_0000, DEFAULT_THREAD_PRIORITY, 0);
        let b = cpu.create_thread(0x0800_0000, 0, 0x1100_0000, DEFAULT_THREAD_PRIORITY, 0);
        assert!(cpu.start_thread(a) && cpu.start_thread(b));
        assert_eq!(shares(&mut cpu, 300), vec![100, 100, 100]);
    }

    /// `svcSetThreadPriority` through the pseudo handle affects the next decision.
    #[test]
    fn a_priority_set_through_the_pseudo_handle_is_the_one_scheduled_on() {
        let mut cpu = Cpu::new();
        let worker = cpu.create_thread(0x0800_0000, 0, 0x1000_0000, DEFAULT_THREAD_PRIORITY, 0);
        assert!(cpu.start_thread(worker));
        assert_eq!(cpu.thread_priority(CURRENT_THREAD_PSEUDO_HANDLE), Some(44));
        assert_eq!(cpu.thread_priority(worker), Some(44));
        assert_eq!(cpu.thread_priority(0xdead), None, "not a thread");

        assert!(cpu.set_thread_priority(CURRENT_THREAD_PSEUDO_HANDLE, 10));
        assert_eq!(cpu.thread_priority(MAIN_THREAD_HANDLE), Some(10));
        cpu.yield_thread();
        assert_eq!(
            cpu.current_thread, 0,
            "the more urgent main thread keeps running"
        );

        cpu.set_main_thread_priority(20);
        assert_eq!(cpu.thread_priority(MAIN_THREAD_HANDLE), Some(20));
    }
}
