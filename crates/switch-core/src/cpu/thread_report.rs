//! What each guest thread has been doing, for a host that reports it.
//!
//! Two halves. The lifecycle, every thread created, started, paused, resumed
//! and ended, is journalled as it happens: those are rare, and each one is
//! worth a line. Everything a thread does in between, blocking and waking
//! thousands of times a second, is summarised instead, as how many
//! instructions each thread ran since the last reading, where it is, and what
//! it is waiting on now, which is the question a stalled title raises: which
//! thread is spinning, and which one is asleep on something that will never
//! come.

use super::{Cpu, ThreadState};

/// How many lifecycle lines are held between two readings. Past it they are
/// counted instead.
const LOG_CAP: usize = 256;

/// The size of `nn::os::ThreadType`, which is how far into one the name
/// search looks. Measured rather than taken from a header: Just Dance 2017
/// allocates its threads' `ThreadType`s back to back, and consecutive entry
/// arguments are exactly this far apart. Looking further would read the next
/// thread's name buffer and report it as this one's.
const THREAD_TYPE_SIZE: u32 = 0x1C0;

/// How many frames above a thread's pc its report walks: enough to climb out
/// of the SDK's wrappers into the code that called them.
const CALLERS: usize = 4;

/// The longest name the search accepts, `nn::os`'s own limit included.
const NAME_MAX: u32 = 64;

/// One thread, as of the reading that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadReport {
    /// Its slot: 0 is the main thread, the rest in creation order.
    pub index: usize,
    /// Its handle, as the guest knows it.
    pub handle: u64,
    /// The name `nn::os` gave it, when one could be found, as the report
    /// prints it: in quotes, or for a thread never named, the function the
    /// SDK's default name records. See [`Cpu::name_text`].
    pub name: Option<String>,
    /// Where it started, as `module+offset` when the address is in a loaded
    /// module.
    pub entry: String,
    /// Where it is now and who called the function it is in, as
    /// `module+offset`.
    pub at: String,
    /// What it is doing, in words.
    pub state: String,
    /// Whether it holds the CPU right now.
    pub running: bool,
    /// Instructions it retired since the last reading.
    pub ran: u64,
    /// Times it was given the CPU since the last reading.
    pub switches: u64,
}

impl Cpu {
    /// Charge the instructions retired since the last switch to thread
    /// `index`, which is the one that ran them.
    pub(super) fn account_slice(&mut self, index: usize) {
        let ran = self.steps.wrapping_sub(self.switched_in_at);
        if let Some(thread) = self.threads.get_mut(index) {
            thread.ran += ran;
        }
        self.switched_in_at = self.steps;
    }

    /// Journal one lifecycle event, or count it when the journal is full.
    pub(super) fn log_thread(&mut self, line: String) {
        if self.thread_log.len() < LOG_CAP {
            self.thread_log.push(line);
        } else {
            self.thread_log_dropped += 1;
        }
    }

    /// Remember that `name` occupies `start..end`, so an address in it can be
    /// named as an offset into it.
    pub(super) fn record_module_name(&mut self, start: u32, end: u32, name: &str) {
        self.module_names
            .retain(|&(s, e, _)| e <= start || s >= end);
        self.module_names.push((start, end, name.to_owned()));
    }

    /// Forget the module at `start`, when it is unloaded.
    pub(super) fn forget_module_name(&mut self, start: u32) {
        self.module_names.retain(|&(s, _, _)| s != start);
    }

    /// An address as `module+offset`, or bare when no loaded module holds it.
    ///
    /// Offsets are what a disassembly of the module is read against, so this
    /// is the form a report can be checked with; a bare address only means
    /// something against this one run's layout.
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

    /// The lifecycle events since the last call, how many more there were
    /// than the journal had room for, and where every thread stands now.
    /// Each thread's instruction and switch counts restart from zero.
    pub fn take_thread_report(&mut self) -> (Vec<ThreadReport>, Vec<String>, u64) {
        let log = std::mem::take(&mut self.thread_log);
        let dropped = std::mem::take(&mut self.thread_log_dropped);
        // A process that never created a thread has no slot for its main one
        // yet; everything it ran is the main thread's.
        if self.threads.is_empty() {
            let ran = self.steps.wrapping_sub(self.switched_in_at);
            self.switched_in_at = self.steps;
            let main = ThreadReport {
                index: 0,
                handle: super::MAIN_THREAD_HANDLE,
                name: None,
                entry: "the process entry".to_owned(),
                at: self.where_is(0),
                state: "running".to_owned(),
                running: true,
                ran,
                switches: 0,
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
                running: index == self.current_thread,
                ran: 0,
                switches: 0,
            });
        }
        for (report, thread) in reports.iter_mut().zip(self.threads.iter_mut()) {
            report.ran = std::mem::take(&mut thread.ran);
            report.switches = std::mem::take(&mut thread.switches);
        }
        (reports, log, dropped)
    }

    /// Where thread `index` is: its pc, then the return addresses of the
    /// frames above it, innermost first.
    ///
    /// The callers are the part that says what the thread is *doing*. A
    /// blocked thread's pc is the `svc` in the SDK's wait wrapper, which
    /// every blocked thread shares, and the first caller or two are still
    /// the SDK's; the game's own code is further up.
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
        format!(
            "at {}, called from {}",
            self.locate(pc),
            callers.join(" <- ")
        )
    }

    /// The name `nn::os` gave thread `index`, when it can be found.
    ///
    /// A thread made by `nn::os::CreateThread` enters with its `ThreadType`
    /// as its argument, and a `ThreadType` carries its name as a pointer to a
    /// buffer inside itself. Where in it depends on the SDK version, so the
    /// pointer is looked for rather than read from a fixed offset: the first
    /// word in the struct that points back into the struct at readable text.
    /// Pointing *into the struct* is what makes this a name rather than any
    /// string the thread happens to reference, and a thread that was not
    /// made by `nn::os` has no such word and gets no name.
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

    /// A thread's name the way the report prints it.
    ///
    /// A thread nobody named keeps the name `nn::os` made up for it,
    /// `Thread_0x` and the address of its function. That address is worth
    /// more than the name: every such thread enters through the same SDK
    /// trampoline, so the function is the one thing telling them apart, and
    /// it is printed as `module+offset` like every other address here.
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

    /// The NUL-terminated printable text at `at`, when there is some that
    /// could be a name: at least two characters, at least one of them a
    /// letter, and shorter than [`NAME_MAX`].
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

    /// What thread `index` is doing, in words: for a blocked thread, what it
    /// is blocked on and, where it can be said, who holds it.
    fn describe_thread(&self, index: usize) -> String {
        let thread = &self.threads[index];
        let running = index == self.current_thread;
        let hz = u64::from(crate::cpu::power::CLOCK_RATES_HZ[0]).max(1);
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
                // The lock word holds its owner's handle, which says whose
                // turn it is to let go.
                let owner = self.mem.read_u32(addr).unwrap_or(0) & !super::MUTEX_HAS_LISTENERS;
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
                // Parked with its pc on the `svcWaitSynchronization`, so the
                // arguments are still in its saved registers: the handle list
                // and how long it is.
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
                format!("waiting on {}", waits.join(", "))
            }
        };
        if thread.paused {
            text.push_str(", paused");
        }
        text
    }

    /// A thread by its handle, as the report numbers it.
    fn thread_named_by(&self, handle: u64) -> String {
        match self.threads.iter().position(|t| t.handle == handle) {
            Some(index) => match self.thread_name(index) {
                Some(name) => format!("thread {index} ({})", self.name_text(&name)),
                None => format!("thread {index}"),
            },
            None => format!("handle {handle:#x}"),
        }
    }

    /// A thread as its lifecycle lines name it: its number, its name once it
    /// has one, and where it started.
    pub(super) fn thread_label(&self, handle: u64) -> String {
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
    fn a_thread_is_reported_from_creation_to_exit() {
        let mut cpu = Cpu::new();
        cpu.record_module_name(0x0800_0000, 0x0900_0000, "main");
        let handle = cpu.create_thread(0x0800_0100, 7, 0x1000_0000);
        assert!(cpu.start_thread(handle));

        let (threads, log, dropped) = cpu.take_thread_report();
        assert_eq!(dropped, 0);
        assert_eq!(threads.len(), 2, "{threads:?}");
        assert_eq!(threads[1].entry, "main+0x100");
        assert_eq!(threads[1].at, "at main+0x100, called from 0x20000100");
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

    /// The name is found by what points at it, not by where it sits, and a
    /// pointer that leaves the `ThreadType` is not taken for one: that is how
    /// a neighbouring thread's name, or any other string, stays out.
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

        let handle = cpu.create_thread(0x0800_0100, u64::from(TYPE), 0x1000_0000);
        let (threads, _, _) = cpu.take_thread_report();
        assert_eq!(threads[1].name.as_deref(), Some("\"LoadingThread\""));
        assert!(cpu.thread_label(handle).contains("\"LoadingThread\""));

        // The name the SDK gives a thread nobody named is its function's
        // address, which is the more useful thing to print.
        cpu.record_module_name(0x0800_4000, 0x0900_0000, "main");
        for (i, &b) in b"Thread_0x00000000086563A8\0".iter().enumerate() {
            cpu.mem.write_u8(TYPE + 0x188 + i as u32, b).unwrap();
        }
        let (threads, _, _) = cpu.take_thread_report();
        assert_eq!(
            threads[1].name.as_deref(),
            Some("unnamed, runs main+0x6523a8")
        );

        // A struct whose only inward pointer is to something that is not text
        // is nameless rather than named after garbage.
        cpu.mem.write_u8(TYPE + 0x188, 0x01).unwrap();
        let (threads, _, _) = cpu.take_thread_report();
        assert_eq!(threads[1].name, None);
    }
}
