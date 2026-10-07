//! Per-thread activity for host reports: a lifecycle journal plus a summary
//! of each thread's work, location and wait since the last reading.

use crate::cpu::{Cpu, ThreadState};

/// Lifecycle lines held between readings; the rest are only counted.
const LOG_CAP: usize = 256;

/// Size of `nn::os::ThreadType`, which bounds the name search.
const THREAD_TYPE_SIZE: u32 = 0x1C0;

/// Frames above a thread's pc its report walks.
const CALLERS: usize = 4;

/// The longest name the search accepts, `nn::os`'s own limit included.
const NAME_MAX: u32 = 64;

/// Fewer instructions than this between two reports counts as no work.
const IDLE_WORK: u64 = 20_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadReport {
    /// 0 is the main thread, the rest in creation order.
    pub index: usize,
    pub handle: u64,
    /// The `nn::os` name, formatted by [`Cpu::name_text`].
    pub name: Option<String>,
    /// Entry point, as `module+offset` when inside a loaded module.
    pub entry: String,
    /// Current pc and callers, as `module+offset`.
    pub at: String,
    pub state: String,
    /// 0 (most urgent) to 63.
    pub priority: u8,
    pub running: bool,
    /// Instructions retired since the last reading.
    pub ran: u64,
    /// Times scheduled since the last reading.
    pub switches: u64,
    /// Guest milliseconds since a report last saw real work; 0 if never started or ended.
    pub idle_ms: u64,
}

impl Cpu {
    /// Charge instructions retired since the last switch to thread `index`.
    pub(crate) fn account_slice(&mut self, index: usize) {
        let ran = self.steps.wrapping_sub(self.switched_in_at);
        if let Some(thread) = self.threads.get_mut(index) {
            thread.ran += ran;
        }
        self.switched_in_at = self.steps;
    }

    pub(crate) fn log_thread(&mut self, line: String) {
        if self.thread_log.len() < LOG_CAP {
            self.thread_log.push(line);
        } else {
            self.thread_log_dropped += 1;
        }
    }

    pub(crate) fn record_module_name(&mut self, start: u32, end: u32, name: &str) {
        self.module_names
            .retain(|&(s, e, _)| e <= start || s >= end);
        self.module_names.push((start, end, name.to_owned()));
    }

    pub(crate) fn forget_module_name(&mut self, start: u32) {
        self.module_names.retain(|&(s, _, _)| s != start);
    }

    /// An address as `module+offset`, or bare when no loaded module holds it.
    pub fn locate(&self, addr: u32) -> String {
        match self
            .module_names
            .iter()
            .find(|&&(start, end, _)| (start..end).contains(&addr))
        {
            Some((start, _, name)) => format!("{name}+{:#x}", addr - start),
            None => format!("{addr:#x}"),
        }
    }

    /// Lifecycle events since the last call, the overflow count, and every
    /// thread's state. Resets per-thread counters.
    pub fn take_thread_report(&mut self) -> (Vec<ThreadReport>, Vec<String>, u64) {
        let log = std::mem::take(&mut self.thread_log);
        let dropped = std::mem::take(&mut self.thread_log_dropped);
        // Without any created thread, everything ran on the main thread.
        if self.threads.is_empty() {
            let ran = self.steps.wrapping_sub(self.switched_in_at);
            self.switched_in_at = self.steps;
            let main = ThreadReport {
                index: 0,
                handle: crate::cpu::MAIN_THREAD_HANDLE,
                name: None,
                entry: "the process entry".to_owned(),
                at: self.where_is(0),
                state: "running".to_owned(),
                priority: self.main_thread_priority,
                running: true,
                ran,
                switches: 0,
                idle_ms: 0,
            };
            return (vec![main], log, dropped);
        }
        self.account_slice(self.current_thread);
        let mut reports = Vec::with_capacity(self.threads.len());
        for index in 0..self.threads.len() {
            let thread = &self.threads[index];
            let entry = if index == 0 {
                "the process entry".to_owned()
            } else {
                self.locate(thread.entry)
            };
            reports.push(ThreadReport {
                index,
                handle: thread.handle,
                name: self.thread_name(index).map(|name| self.name_text(&name)),
                entry,
                at: self.where_is(index),
                state: self.describe_thread(index),
                priority: thread.priority,
                running: index == self.current_thread,
                ran: 0,
                switches: 0,
                idle_ms: 0,
            });
        }
        let hz = u64::from(crate::services::power::CLOCK_RATES_HZ[0]).max(1);
        let now = self.cycles;
        for (report, thread) in reports.iter_mut().zip(self.threads.iter_mut()) {
            report.ran = std::mem::take(&mut thread.ran);
            report.switches = std::mem::take(&mut thread.switches);
            if report.ran >= IDLE_WORK {
                thread.busy_at = now;
            }
            report.idle_ms = match thread.state {
                ThreadState::Created | ThreadState::Finished => 0,
                _ => now.saturating_sub(thread.busy_at).saturating_mul(1000) / hz,
            };
        }
        (reports, log, dropped)
    }

    /// Thread `index`'s pc, then its frames' return addresses, innermost first.
    fn where_is(&self, index: usize) -> String {
        let (pc, regs, mode) = match self.threads.get(index) {
            Some(thread) if index != self.current_thread => (thread.pc, &thread.regs, thread.mode),
            _ => (self.pc, &self.regs, self.mode),
        };
        let callers: Vec<String> = self
            .walk_frames(regs, mode, CALLERS)
            .into_iter()
            .map(|addr| self.locate(addr))
            .collect();
        if callers.is_empty() {
            return format!("at {}", self.locate(pc));
        }
        format!(
            "at {}, called from {}",
            self.locate(pc),
            callers.join(" <- ")
        )
    }

    /// The name `nn::os` gave thread `index`: found via the first word in its
    /// `ThreadType` that points back into the struct at readable text.
    fn thread_name(&self, index: usize) -> Option<String> {
        let base = u32::try_from(self.threads.get(index)?.arg).ok()?;
        if base == 0 {
            return None;
        }
        let end = base.checked_add(THREAD_TYPE_SIZE)?;
        (0..THREAD_TYPE_SIZE).step_by(8).find_map(|offset| {
            let word = self.mem.read_u64(base + offset).ok()?;
            let target = u32::try_from(word).ok()?;
            if !(base..end).contains(&target) {
                return None;
            }
            self.read_name(target)
        })
    }

    /// A thread's printed name; unnamed `Thread_0x` threads print their function.
    fn name_text(&self, name: &str) -> String {
        match name
            .strip_prefix("Thread_0x")
            .and_then(|hex| u64::from_str_radix(hex, 16).ok())
            .and_then(|function| u32::try_from(function).ok())
        {
            Some(function) => format!("unnamed, runs {}", self.locate(function)),
            None => format!("\"{name}\""),
        }
    }

    /// Printable NUL-terminated text at `at` that could be a name.
    fn read_name(&self, at: u32) -> Option<String> {
        let mut name = String::new();
        for i in 0..NAME_MAX {
            match self.mem.read_u8(at.checked_add(i)?).ok()? {
                0 => break,
                byte @ 0x20..=0x7E => name.push(char::from(byte)),
                _ => return None,
            }
            if i + 1 == NAME_MAX {
                return None;
            }
        }
        (name.len() >= 2 && name.chars().any(|c| c.is_ascii_alphabetic())).then_some(name)
    }

    /// What a blocked thread is blocked on and, where known, who holds it.
    fn describe_thread(&self, index: usize) -> String {
        let thread = &self.threads[index];
        let running = index == self.current_thread;
        let hz = u64::from(crate::services::power::CLOCK_RATES_HZ[0]).max(1);
        let until = |deadline: u64| {
            let left = deadline.saturating_sub(self.cycles);
            format!("{:.1} ms", left as f64 * 1000.0 / hz as f64)
        };
        let mut text = match thread.state {
            ThreadState::Created => "created, never started".to_owned(),
            ThreadState::Runnable if running => "running".to_owned(),
            ThreadState::Runnable => "ready to run".to_owned(),
            ThreadState::Finished => "exited".to_owned(),
            ThreadState::WaitMutex(addr) => {
                // The lock word holds its owner's handle.
                let owner = self.mem.read_u32(addr).unwrap_or(0) & !crate::cpu::MUTEX_HAS_LISTENERS;
                format!(
                    "waiting for the mutex at {addr:#x}, held by {}",
                    self.thread_named_by(u64::from(owner))
                )
            }
            ThreadState::WaitKey {
                key,
                mutex,
                deadline,
            } => format!(
                "waiting on the condition variable at {key:#x} (mutex {mutex:#x}){}",
                deadline.map_or(String::new(), |d| format!(", times out in {}", until(d)))
            ),
            ThreadState::WaitAddress { addr, deadline } => format!(
                "waiting on the address {addr:#x} (holds {:#x}){}",
                self.mem.read_u32(addr).unwrap_or(0),
                deadline.map_or(String::new(), |d| format!(", times out in {}", until(d)))
            ),
            ThreadState::Sleeping { deadline } => format!("asleep, wakes in {}", until(deadline)),
            ThreadState::WaitEvent { .. } => {
                // Parked on `svcWaitSynchronization`: handles are in its saved registers.
                let list = thread.regs[1] as u32;
                let count = (thread.regs[2] as u32).min(0x40);
                let waits: Vec<String> = (0..count)
                    .map(|i| {
                        let at = list.wrapping_add(i * 4);
                        let handle = u64::from(self.mem.read_u32(at).unwrap_or(0));
                        match (self.event_name(handle), self.service_name(handle)) {
                            (Some(name), _) => format!("{name} ({handle:#x})"),
                            (None, Some(name)) => format!("session {name} ({handle:#x})"),
                            (None, None) => format!("handle {handle:#x}"),
                        }
                    })
                    .collect();
                // `nn::os` sleeps by waiting on no handles.
                if waits.is_empty() {
                    "asleep in a wait on no handles".to_owned()
                } else {
                    format!("waiting on {}", waits.join(", "))
                }
            }
        };
        if thread.paused {
            text.push_str(", paused");
        }
        text
    }

    fn thread_named_by(&self, handle: u64) -> String {
        match self.threads.iter().position(|t| t.handle == handle) {
            Some(index) => match self.thread_name(index) {
                Some(name) => format!("thread {index} ({})", self.name_text(&name)),
                None => format!("thread {index}"),
            },
            None => format!("handle {handle:#x}"),
        }
    }

    /// A thread as lifecycle lines name it: number, name, and entry point.
    pub(crate) fn thread_label(&self, handle: u64) -> String {
        match self.threads.iter().position(|t| t.handle == handle) {
            Some(0) => "thread 0 (main)".to_owned(),
            Some(index) => {
                let entry = self.locate(self.threads[index].entry);
                match self.thread_name(index) {
                    Some(name) => {
                        format!("thread {index} ({}, via {entry})", self.name_text(&name))
                    }
                    None => format!("thread {index} ({entry})"),
                }
            }
            None => format!("handle {handle:#x}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::THREAD_TYPE_SIZE;
    use crate::cpu::Cpu;

    // The decoy below sits at +0x400, which the search must not reach.
    const _: () = assert!(THREAD_TYPE_SIZE <= 0x400);

    #[test]
    fn an_address_is_named_by_the_module_it_is_in() {
        let mut cpu = Cpu::new();
        cpu.record_module_name(0x0800_4000, 0x0900_0000, "main");
        assert_eq!(cpu.locate(0x0800_5234), "main+0x1234");
        assert_eq!(cpu.locate(0x0700_0000), "0x7000000");
        cpu.forget_module_name(0x0800_4000);
        assert_eq!(cpu.locate(0x0800_5234), "0x8005234");
    }

    #[test]
    fn a_blocked_thread_counts_its_idle_time_until_it_works() {
        use crate::cpu::ThreadState;
        let mut cpu = Cpu::new();
        let handle = cpu.create_thread(0x0800_0100, 0, 0x1000_0000, 44, 0);
        assert!(cpu.start_thread(handle));
        cpu.take_thread_report();
        cpu.threads[1].state = ThreadState::WaitEvent { deadline: u64::MAX };
        let hz = u64::from(crate::services::power::CLOCK_RATES_HZ[0]);
        cpu.cycles += hz * 3;
        let (threads, ..) = cpu.take_thread_report();
        assert_eq!(threads[1].idle_ms, 3000, "three seconds without work");

        // A report that finds real work resets it.
        cpu.threads[1].ran = super::IDLE_WORK;
        cpu.cycles += hz;
        let (threads, ..) = cpu.take_thread_report();
        assert_eq!(threads[1].idle_ms, 0);
    }

    #[test]
    fn a_thread_is_reported_from_creation_to_exit() {
        let mut cpu = Cpu::new();
        cpu.record_module_name(0x0800_0000, 0x0900_0000, "main");
        let handle = cpu.create_thread(0x0800_0100, 7, 0x1000_0000, 44, 0);
        assert!(cpu.start_thread(handle));

        let (threads, log, dropped) = cpu.take_thread_report();
        assert_eq!(dropped, 0);
        assert_eq!(threads.len(), 2, "{threads:?}");
        assert_eq!(threads[1].entry, "main+0x100");
        assert_eq!(
            threads[1].at,
            format!(
                "at main+0x100, called from {:#x}",
                crate::cpu::THREAD_EXIT_TRAMPOLINE
            )
        );
        assert_eq!(threads[1].state, "ready to run");
        assert_eq!(threads[1].name, None, "an argument of 7 is no ThreadType");
        assert!(threads[0].running);
        assert!(
            log[0].starts_with("thread 1 (main+0x100) created"),
            "{log:?}"
        );
        assert!(
            log[1].starts_with("thread 1 (main+0x100) started"),
            "{log:?}"
        );
        assert!(cpu.take_thread_report().1.is_empty(), "taken, not read");
    }

    /// Frames are only reported while their return address follows a call.
    #[test]
    fn a_backtrace_reports_only_frames_that_return_after_a_call() {
        use crate::cpu::SP_SLOT;
        const BL: u32 = 0x9400_0000;
        const BLR_X8: u32 = 0xD63F_0100;
        const NOP: u32 = 0xD503_201F;
        const STACK: u32 = 0x1000_0000;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(0x0800_0000, 0x1000).unwrap();
        cpu.mem.write_u32(0x0800_0100, BL).unwrap();
        cpu.mem.write_u32(0x0800_0200, BLR_X8).unwrap();
        cpu.mem.write_u32(0x0800_0300, NOP).unwrap();
        cpu.mem.map_zero(STACK - 0x1000, 0x2000).unwrap();
        // Two good frames, then one whose return address follows a nop.
        cpu.mem
            .write_u64(STACK + 0x10, u64::from(STACK + 0x40))
            .unwrap();
        cpu.mem.write_u64(STACK + 0x18, 0x0800_0204).unwrap();
        cpu.mem
            .write_u64(STACK + 0x40, u64::from(STACK + 0x80))
            .unwrap();
        cpu.mem.write_u64(STACK + 0x48, 0x0800_0104).unwrap();
        cpu.mem
            .write_u64(STACK + 0x80, u64::from(STACK + 0xC0))
            .unwrap();
        cpu.mem.write_u64(STACK + 0x88, 0x0800_0304).unwrap();

        cpu.regs[SP_SLOT] = u64::from(STACK);
        cpu.regs[29] = u64::from(STACK + 0x10);
        cpu.regs[30] = 0xaa;
        assert_eq!(cpu.backtrace(8), [0x0800_0204, 0x0800_0104]);

        cpu.regs[30] = 0x0800_0104;
        assert_eq!(cpu.backtrace(8), [0x0800_0104, 0x0800_0204, 0x0800_0104]);

        // The same good record below the stack pointer is not a frame.
        cpu.regs[SP_SLOT] = u64::from(STACK + 0x20);
        cpu.regs[30] = 0xaa;
        assert_eq!(cpu.backtrace(8), [] as [u32; 0]);

        cpu.regs[SP_SLOT] = u64::from(STACK);
        cpu.regs[29] = 0x7473_6964;
        cpu.regs[30] = 0;
        assert_eq!(cpu.backtrace(8), [] as [u32; 0], "\"dist\" is no frame");
    }

    /// A thread handle stays unsignalled until the thread really ends.
    #[test]
    fn a_thread_handle_is_signalled_when_its_thread_exits_and_wakes_its_joiner() {
        use crate::cpu::ThreadState;
        let mut cpu = Cpu::new();
        let handle = cpu.create_thread(0x0800_0100, 0, 0x1000_0000, 44, 0);
        assert_eq!(cpu.waitable_signaled(handle), Some(false), "created");
        assert!(cpu.start_thread(handle));
        assert_eq!(cpu.waitable_signaled(handle), Some(false), "running");

        // The main thread parks on the handle; the worker then exits.
        cpu.threads[0].state = ThreadState::WaitEvent { deadline: u64::MAX };
        cpu.current_thread = 1;
        cpu.exit_thread();
        assert_eq!(cpu.threads[1].state, ThreadState::Finished);
        assert_eq!(cpu.waitable_signaled(handle), Some(true), "exited");
        assert_eq!(
            cpu.threads[0].state,
            ThreadState::Runnable,
            "the joiner is woken to recheck its handle"
        );
        assert_eq!(
            cpu.waitable_signaled(0xdead),
            None,
            "neither event nor thread"
        );
    }

    /// A name pointer that leaves the `ThreadType` is not taken.
    #[test]
    fn a_thread_is_named_from_the_pointer_its_thread_type_keeps_to_itself() {
        const TYPE: u32 = 0x3096_5000;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TYPE, 0x1000).unwrap();
        // A pointer out of the struct, at text, before the real one: skipped.
        cpu.mem
            .write_u64(TYPE + 0x08, u64::from(TYPE + 0x400))
            .unwrap();
        for (i, &b) in b"NotMine\0".iter().enumerate() {
            cpu.mem.write_u8(TYPE + 0x400 + i as u32, b).unwrap();
        }
        // The name buffer, and the pointer to it.
        for (i, &b) in b"LoadingThread\0".iter().enumerate() {
            cpu.mem.write_u8(TYPE + 0x188 + i as u32, b).unwrap();
        }
        cpu.mem
            .write_u64(TYPE + 0x1A8, u64::from(TYPE + 0x188))
            .unwrap();

        let handle = cpu.create_thread(0x0800_0100, u64::from(TYPE), 0x1000_0000, 44, 0);
        let (threads, _, _) = cpu.take_thread_report();
        assert_eq!(threads[1].name.as_deref(), Some("\"LoadingThread\""));
        assert!(cpu.thread_label(handle).contains("\"LoadingThread\""));

        // An SDK default name prints its function's address instead.
        cpu.record_module_name(0x0800_4000, 0x0900_0000, "main");
        for (i, &b) in b"Thread_0x00000000086563A8\0".iter().enumerate() {
            cpu.mem.write_u8(TYPE + 0x188 + i as u32, b).unwrap();
        }
        let (threads, _, _) = cpu.take_thread_report();
        assert_eq!(
            threads[1].name.as_deref(),
            Some("unnamed, runs main+0x6523a8")
        );

        // An inward pointer to non-text yields no name.
        cpu.mem.write_u8(TYPE + 0x188, 0x01).unwrap();
        let (threads, _, _) = cpu.take_thread_report();
        assert_eq!(threads[1].name, None);
    }
}
