//! Guest cores beyond the kernel's. Core 0 runs the kernel, the services and
//! the GPU; it lends runnable threads to the other cores, which run guest code
//! until it needs the kernel and then hand the thread back.

use crate::cpu::*;
use crate::kernel::thread::ThreadContext;
use crate::mem::AddressSpace;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// Instructions a borrowing core runs before its thread comes back to core 0's clock.
const QUANTUM: u64 = 1_000_000;

/// How long core 0, with nothing to run, waits for a borrowing core before looking again.
const IDLE_WAIT: Duration = Duration::from_micros(200);

/// One borrowing core's mailbox, shared with core 0.
#[derive(Debug)]
pub struct CoreSlot {
    core: usize,
    state: Mutex<Slot>,
    changed: Condvar,
    recall: Arc<AtomicBool>,
    clock: Arc<AtomicU64>,
    /// The priority of the thread lent to it, which a more urgent one recalls.
    held: AtomicU8,
}

#[derive(Debug)]
enum Slot {
    Idle,
    Lent(Box<Lent>),
    Running,
    Returned(Box<Returned>),
    Stop,
}

impl CoreSlot {
    fn lock(&self) -> MutexGuard<'_, Slot> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Run guest code for core 0 on this thread until [`Cpu::stop_cores`].
pub fn serve(slot: Arc<CoreSlot>) {
    let mut cpu = Cpu::borrower(slot.core);
    cpu.recall = slot.recall.clone();
    cpu.clock = slot.clock.clone();
    loop {
        let lent = {
            let mut state = slot.lock();
            loop {
                match std::mem::replace(&mut *state, Slot::Running) {
                    Slot::Lent(lent) => break lent,
                    Slot::Stop => return,
                    other => {
                        *state = other;
                        state = slot.changed.wait(state).unwrap_or_else(|e| e.into_inner());
                    }
                }
            }
        };
        let returned = cpu.run_lent(*lent);
        *slot.lock() = Slot::Returned(Box::new(returned));
        slot.changed.notify_all();
    }
}

/// A thread core 0 handed to another core.
#[derive(Debug)]
pub(crate) struct Lent {
    pub(crate) pid: u64,
    /// The thread's index in its process.
    pub(crate) index: usize,
    pub(crate) mem: AddressSpace,
    /// Its registers on the way out, and back.
    pub(crate) thread: ThreadContext,
    /// Instructions to run at most.
    pub(crate) quantum: u64,
}

/// Why a lent thread came back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Handback {
    /// On an `svc`, not yet executed: core 0 issues it.
    Svc,
    /// On a faulting instruction, which core 0 runs again to meet the fault.
    Fault,
    /// Its quantum ran out.
    Quantum,
    /// Core 0 asked for it.
    Recalled,
}

#[derive(Debug)]
pub(crate) struct Returned {
    pub(crate) lent: Lent,
    pub(crate) why: Handback,
    /// Instructions it retired here.
    pub(crate) retired: u64,
}

impl Cpu {
    /// Another core for [`serve`] to run on its own host thread.
    pub fn add_core(&mut self) -> Arc<CoreSlot> {
        let slot = Arc::new(CoreSlot {
            core: self.machine.cores.len() + 1,
            state: Mutex::new(Slot::Idle),
            changed: Condvar::new(),
            recall: Arc::default(),
            clock: self.clock.clone(),
            held: AtomicU8::new(LOWEST_PRIORITY),
        });
        self.machine.cores.push(slot.clone());
        slot
    }

    /// Take every lent thread back and end each core's [`serve`].
    pub fn stop_cores(&mut self) {
        self.quiesce_cores();
        for slot in std::mem::take(&mut self.machine.cores) {
            *slot.lock() = Slot::Stop;
            slot.changed.notify_all();
        }
    }

    /// Take back what the borrowing cores returned and lend the idle ones work.
    pub(crate) fn service_cores(&mut self) {
        for k in 0..self.machine.cores.len() {
            let slot = self.machine.cores[k].clone();
            let Ok(mut state) = slot.state.try_lock() else {
                continue;
            };
            if let Slot::Returned(_) = *state {
                let Slot::Returned(returned) = std::mem::replace(&mut *state, Slot::Idle) else {
                    unreachable!()
                };
                self.take_back(*returned);
            }
            match *state {
                Slot::Idle => {
                    if let Some(lent) = self.lend_to(slot.core) {
                        slot.recall.store(false, Relaxed);
                        slot.held.store(lent.thread.priority, Relaxed);
                        *state = Slot::Lent(Box::new(lent));
                        slot.changed.notify_all();
                    }
                }
                // A more urgent thread preempts the one there, as on the console.
                Slot::Lent(_) | Slot::Running
                    if self
                        .best_for(slot.core)
                        .is_some_and(|(_, _, priority)| priority < slot.held.load(Relaxed)) =>
                {
                    slot.recall.store(true, Relaxed);
                }
                _ => {}
            }
        }
    }

    /// The best thread core `core` may run that core 0 is not running, marked lent.
    fn lend_to(&mut self, core: usize) -> Option<Lent> {
        let (pid, index, _) = self.best_for(core)?;
        let process = self.machine.process_mut(pid);
        let thread = &mut process.threads[index];
        thread.lent = true;
        thread.core = core as u8;
        Some(Lent {
            pid,
            index,
            mem: process.mem.clone(),
            thread: thread.clone(),
            quantum: QUANTUM,
        })
    }

    /// The most urgent ready thread `core` may run that core 0 is not running:
    /// its pid, index and priority.
    fn best_for(&self, core: usize) -> Option<(u64, usize, u8)> {
        let mut best: Option<(u64, usize, u8)> = None;
        for (&pid, process) in &self.machine.processes {
            for (index, t) in process.threads.iter().enumerate() {
                let eligible = t.ready()
                    && !t.kernel_pending
                    && t.affinity & (1 << core) != 0
                    && (pid, index) != (self.pid, self.current_thread);
                if eligible && best.is_none_or(|(_, _, p)| t.priority < p) {
                    best = Some((pid, index, t.priority));
                }
            }
        }
        best
    }

    /// Put a returned thread's registers back. Its instructions ran beside
    /// core 0's, so they count as steps but not on the clock.
    fn take_back(&mut self, returned: Returned) {
        let Returned { lent, why, retired } = returned;
        self.steps += retired;
        self.machine.lent_steps += retired;
        let Some(thread) = self
            .machine
            .processes
            .get_mut(&lent.pid)
            .and_then(|p| p.threads.get_mut(lent.index))
        else {
            return;
        };
        thread.take_registers(&lent.thread);
        thread.lent = false;
        thread.kernel_pending = matches!(why, Handback::Svc | Handback::Fault);
        thread.ran += retired;
    }

    /// Instructions retired on the cores [`Cpu::add_core`] made, already in [`Cpu::steps`].
    pub fn other_core_steps(&self) -> u64 {
        self.machine.lent_steps
    }

    /// Core 0 is issuing `svc #imm` for the running thread: it may be lent
    /// again after, and what changes memory under guest code waits for
    /// the other cores to give their threads back.
    pub(crate) fn lend_again(&mut self, imm: u16) {
        let current = self.current_thread;
        if let Some(thread) = self.process_mut().threads.get_mut(current) {
            thread.kernel_pending = false;
        }
        // SetHeapSize, SetMemoryPermission, Map/UnmapMemory, ExitProcess,
        // Map/UnmapSharedMemory, Map/UnmapPhysicalMemory, the process memory
        // calls; and SetThreadActivity and GetThreadContext3, which must find a
        // paused thread stopped and its registers current (a garbage collector
        // stopping the world).
        if matches!(
            imm,
            0x01 | 0x02 | 0x04 | 0x05 | 0x07 | 0x13 | 0x14 | 0x2C | 0x2D | 0x32 | 0x33 | 0x73
                ..=0x78
        ) {
            self.quiesce_cores();
        }
    }

    /// Whether core 0 may run `thread`: it waits on the kernel, its mask
    /// names core 0, or no other core it names is here to run it.
    pub(crate) fn runs_here(&self, thread: &ThreadContext) -> bool {
        let others = ((1u64 << self.machine.cores.len()) - 1) << 1;
        thread.kernel_pending || thread.affinity & 1 != 0 || thread.affinity & others == 0
    }

    /// Whether any thread is out on another core.
    pub(crate) fn cores_busy(&self) -> bool {
        self.machine
            .cores
            .iter()
            .any(|slot| !matches!(*slot.lock(), Slot::Idle | Slot::Stop))
    }

    /// With nothing to run here, wait briefly for a borrowing core to return a thread.
    pub(crate) fn wait_for_cores(&mut self) {
        for slot in self.machine.cores.clone() {
            let state = slot.lock();
            if matches!(*state, Slot::Lent(_) | Slot::Running) {
                let _ = slot.changed.wait_timeout(state, IDLE_WAIT);
                break;
            }
        }
        self.service_cores();
    }

    /// Take every lent thread back, for what must not race guest code: freeing
    /// memory or changing what stores check.
    pub(crate) fn quiesce_cores(&mut self) {
        for slot in self.machine.cores.clone() {
            slot.recall.store(true, Relaxed);
            let mut state = slot.lock();
            loop {
                match std::mem::replace(&mut *state, Slot::Idle) {
                    Slot::Returned(returned) => {
                        self.take_back(*returned);
                        break;
                    }
                    Slot::Lent(lent) => {
                        self.take_back(Returned {
                            lent: *lent,
                            why: Handback::Recalled,
                            retired: 0,
                        });
                        break;
                    }
                    Slot::Running => {
                        *state = Slot::Running;
                        state = slot.changed.wait(state).unwrap_or_else(|e| e.into_inner());
                    }
                    other => {
                        *state = other;
                        break;
                    }
                }
            }
            slot.recall.store(false, Relaxed);
        }
    }

    /// A core that borrows threads; its `Machine` is never used.
    pub(crate) fn borrower(core: usize) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.core = core;
        cpu
    }

    /// Run a lent thread until it needs the kernel, faults, is recalled or
    /// uses its quantum, and give it back with its registers.
    pub(crate) fn run_lent(&mut self, mut lent: Lent) -> Returned {
        debug_assert_ne!(self.core, 0, "core 0 runs threads itself");
        if lent.pid != self.pid {
            self.switch_translations(lent.pid);
            self.pid = lent.pid;
        }
        self.mem = lent.mem.clone();
        self.load_registers(&lent.thread);
        // Timed waits are core 0's to sweep.
        self.next_expiry = u64::MAX;
        self.handback = None;
        let steps = self.steps;
        let why = match self.run_engine(lent.quantum) {
            Err(_) => Handback::Fault,
            Ok(_) => self.handback.take().unwrap_or(Handback::Quantum),
        };
        let mut retired = self.steps - steps;
        // Core 0 retires the `svc` or the faulting instruction itself.
        if matches!(why, Handback::Svc | Handback::Fault) {
            retired = retired.saturating_sub(1);
        }
        self.registers_into(&mut lent.thread);
        Returned { lent, why, retired }
    }

    /// Between blocks or instructions: core 0 preempts and sweeps timed
    /// waits (servicing the other cores with them); a borrowing core hands
    /// its thread back if recalled.
    #[inline(always)]
    pub(crate) fn preemption_point(&mut self) {
        if self.slice_used >= TIME_SLICE {
            self.slice_used = 0;
            if self.core == 0 {
                self.yield_thread();
            } else if self.recall.load(Relaxed) {
                self.handback = Some(Handback::Recalled);
            }
        }
        self.sweep_timed_waits();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAIN: [u32; 12] = [
        0xd2900002, // movz x2, #0x8000
        0xd37ff845, // lsl  x5, x2, #1
        0xc85f7c01, // ldxr x1, [x0]
        0x91000421, // add  x1, x1, #1
        0xc8037c01, // stxr w3, x1, [x0]
        0x35ffffa3, // cbnz w3, ldxr
        0xf1000442, // subs x2, x2, #1
        0x54ffff61, // b.ne ldxr
        0xc8dffc04, // ldar x4, [x0]
        0xeb05009f, // cmp  x4, x5
        0x54ffffc1, // b.ne ldar
        0xd4000001, // svc  #0
    ];
    const WORKER: [u32; 8] = [
        0xd2900002, // movz x2, #0x8000
        0xc85f7c01, // ldxr x1, [x0]
        0x91000421, // add  x1, x1, #1
        0xc8037c01, // stxr w3, x1, [x0]
        0x35ffffa3, // cbnz w3, ldxr
        0xf1000442, // subs x2, x2, #1
        0x54ffff61, // b.ne ldxr
        0xd4000141, // svc  #0xa (ExitThread)
    ];
    const COUNTER: u32 = 0x3000;

    fn words(code: &[u32]) -> Vec<u8> {
        code.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// A thread made for core 1 runs there, beside core 0's, and their
    /// exclusive increments of one counter all land.
    #[test]
    fn a_thread_for_core_1_runs_there_beside_core_0() {
        let mut cpu = Cpu::new();
        cpu.mem.map(0x1000, &words(&MAIN)).unwrap();
        cpu.mem.map(0x1100, &words(&WORKER)).unwrap();
        cpu.mem.map_zero(COUNTER, 0x1000).unwrap();
        cpu.mem.map_zero(0x10_0000, 0x1000).unwrap();
        cpu.set_pc(0x1000);
        cpu.set_reg(0, u64::from(COUNTER));
        let worker = cpu.create_thread(0x1100, u64::from(COUNTER), 0x10_1000, 44, 1);
        assert!(cpu.start_thread(worker));
        let slot = cpu.add_core();
        let core = std::thread::spawn(move || serve(slot));

        cpu.run(u64::MAX).unwrap();
        cpu.stop_cores();
        core.join().unwrap();
        assert!(cpu.halted);
        assert_eq!(
            cpu.mem.read_u64(u64::from(COUNTER) as u32).unwrap(),
            0x10000
        );
        assert!(
            cpu.other_core_steps() > 0x8000 * 6,
            "core 1 ran {} instructions",
            cpu.other_core_steps()
        );
    }
}
