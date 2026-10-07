//! Mutexes, condition variables and the address arbiter.

use crate::cpu::*;

/// How an `svcWaitForAddress` resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArbiterWait {
    /// The predicate held and the caller is now blocked.
    Blocked,
    /// The word did not hold the expected value; nothing to wait for.
    Mismatch,
    /// The predicate held but the timeout was zero.
    TimedOut,
}

/// Mutex word bit meaning unlock must go through `svcArbitrateUnlock`.
pub(crate) const MUTEX_HAS_LISTENERS: u32 = 0x4000_0000;

/// Written into a condvar's word while a thread waits; `nn::os` skips the signal syscall when it is zero.
const CONDVAR_HAS_WAITERS: u32 = 1;

impl Cpu {
    // ---- mutexes and condition variables ----
    //
    // The lock word holds the owner's handle plus MUTEX_HAS_LISTENERS when contended;
    // libnx re-reads it, so ownership must really move.

    /// `svcArbitrateLock`: block until the owner releases, unless the word already changed.
    pub(crate) fn arbitrate_lock(&mut self, owner: u32, addr: u32, _self_handle: u32) {
        self.ensure_main_thread();
        let word = self.mem.read_u32(addr).unwrap_or(0);
        if word & !MUTEX_HAS_LISTENERS != owner || owner == 0 {
            return; // stale request; the guest re-reads the word and retries
        }
        self.threads[self.current_thread].state = ThreadState::WaitMutex(addr);
        self.reschedule();
    }

    /// `svcArbitrateUnlock`: hand the mutex to a waiter, or clear it.
    pub(crate) fn arbitrate_unlock(&mut self, addr: u32) {
        self.ensure_main_thread();
        let waiters: Vec<usize> = (0..self.threads.len())
            .filter(|&i| self.threads[i].state == ThreadState::WaitMutex(addr))
            .collect();
        match waiters.first() {
            Some(&next) => {
                let mut handle = self.threads[next].handle as u32;
                if waiters.len() > 1 {
                    handle |= MUTEX_HAS_LISTENERS;
                }
                let _ = self.mem.write_u32(addr, handle);
                self.threads[next].state = ThreadState::Runnable;
            }
            None => {
                let _ = self.mem.write_u32(addr, 0);
            }
        }
    }

    /// `svcWaitProcessWideKeyAtomic`: release the mutex and block on the condvar,
    /// marking the condvar word so `nn::os` signals it.
    pub(crate) fn wait_process_wide_key(
        &mut self,
        mutex: u32,
        key: u32,
        _self_handle: u32,
        timeout: i64,
    ) {
        self.ensure_main_thread();
        let _ = self.mem.write_u32(key, CONDVAR_HAS_WAITERS);
        self.arbitrate_unlock(mutex);
        let deadline = self.wait_deadline(timeout);
        self.threads[self.current_thread].state = ThreadState::WaitKey {
            key,
            mutex,
            deadline,
        };
        self.reschedule();
    }

    /// Wake expired timed waits, at most once per [`TIME_SLICE`] cycles of the clock.
    #[inline(always)]
    pub(crate) fn sweep_timed_waits(&mut self) {
        if self.cycles < self.next_expiry {
            return;
        }
        self.next_expiry = self.cycles.wrapping_add(TIME_SLICE);
        self.expire_timed_waits();
    }

    /// Wake every timed wait whose deadline has passed; `nn::os` rechecks its predicate.
    pub(crate) fn expire_timed_waits(&mut self) {
        let now = self.cycles;
        for index in 0..self.threads.len() {
            let state = self.threads[index].state;
            let deadline = match state {
                ThreadState::WaitKey { deadline, .. }
                | ThreadState::WaitAddress { deadline, .. } => deadline,
                ThreadState::Sleeping { deadline } | ThreadState::WaitEvent { deadline } => {
                    Some(deadline)
                }
                _ => None,
            };
            if !deadline.is_some_and(|at| now >= at) {
                continue;
            }
            match state {
                ThreadState::WaitKey { mutex, .. } => self.wake_condvar_waiter(index, mutex),
                _ => self.threads[index].state = ThreadState::Runnable,
            }
        }
    }

    /// Dequeue a condvar waiter holding (or queued for) its mutex, as the kernel does on every wake.
    pub(crate) fn wake_condvar_waiter(&mut self, index: usize, mutex: u32) {
        let handle = self.threads[index].handle as u32;
        let owner = self.mem.read_u32(mutex).unwrap_or(0);
        if owner == 0 {
            let _ = self.mem.write_u32(mutex, handle);
            self.threads[index].state = ThreadState::Runnable;
        } else {
            // Contended: queue up and mark the word so the owner arbitrates its unlock.
            let _ = self.mem.write_u32(mutex, owner | MUTEX_HAS_LISTENERS);
            self.threads[index].state = ThreadState::WaitMutex(mutex);
        }
    }

    /// `svcSignalProcessWideKey`: wake up to `count` waiters (all if negative).
    pub(crate) fn signal_process_wide_key(&mut self, key: u32, count: i32) {
        self.ensure_main_thread();
        let mut woken = 0;
        for i in 0..self.threads.len() {
            if count >= 0 && woken >= count {
                break;
            }
            if let ThreadState::WaitKey {
                key: waiting,
                mutex,
                ..
            } = self.threads[i].state
            {
                if waiting != key {
                    continue;
                }
                self.wake_condvar_waiter(i, mutex);
                woken += 1;
            }
        }
        // Clear the word once the queue is empty.
        let queued = self.threads.iter().any(
            |t| matches!(t.state, ThreadState::WaitKey { key: waiting, .. } if waiting == key),
        );
        if !queued {
            let _ = self.mem.write_u32(key, 0);
        }
    }

    /// Deadline for a `timeout` in nanoseconds; negative waits forever.
    pub(crate) fn wait_deadline(&self, timeout: i64) -> Option<u64> {
        (timeout > 0).then(|| {
            let cycles = (timeout as u128) * u128::from(crate::services::power::CLOCK_RATES_HZ[0])
                / 1_000_000_000;
            self.cycles.wrapping_add(cycles as u64)
        })
    }

    // ---- the address arbiter ----
    //
    // The arbiter word carries no ownership; the kernel only compares it with the caller's value.

    /// `svcWaitForAddress`'s decision, separate from [`Cpu::block_on_address`] so X0 is
    /// written before switching threads.
    pub(crate) fn arbitrate_address(
        &mut self,
        addr: u32,
        arb_type: u32,
        value: i32,
        timeout: i64,
    ) -> ArbiterWait {
        self.ensure_main_thread();
        let Ok(current) = self.mem.read_u32(addr).map(|w| w as i32) else {
            return ArbiterWait::Mismatch;
        };
        let holds = match arb_type {
            // WaitIfLessThan, and its atomic-decrement variant.
            0 | 1 => current < value,
            // WaitIfEqual.
            2 => current == value,
            _ => return ArbiterWait::Mismatch,
        };
        if !holds {
            return ArbiterWait::Mismatch;
        }
        if arb_type == 1 {
            let _ = self.mem.write_u32(addr, current.wrapping_sub(1) as u32);
        }
        // A zero timeout is a poll.
        if timeout == 0 {
            return ArbiterWait::TimedOut;
        }
        ArbiterWait::Blocked
    }

    /// Park on the arbiter word at `addr`, after [`Cpu::arbitrate_address`] decided to wait.
    pub(crate) fn block_on_address(&mut self, addr: u32, timeout: i64) {
        let deadline = self.wait_deadline(timeout);
        self.threads[self.current_thread].state = ThreadState::WaitAddress { addr, deadline };
        self.reschedule();
    }

    /// `svcSignalToAddress`: wake up to `count` waiters (all if negative) after the
    /// signal type's compare-and-modify. Reports whether the word held `value`.
    pub(crate) fn signal_to_address(
        &mut self,
        addr: u32,
        signal_type: u32,
        value: i32,
        count: i32,
    ) -> bool {
        self.ensure_main_thread();
        let waiting = self
            .threads
            .iter()
            .filter(|t| matches!(t.state, ThreadState::WaitAddress { addr: a, .. } if a == addr))
            .count() as i32;
        if signal_type != 0 {
            let Ok(current) = self.mem.read_u32(addr).map(|w| w as i32) else {
                return false;
            };
            if current != value {
                return false;
            }
            let updated = match signal_type {
                // SignalAndIncrementIfEqual.
                1 => value.wrapping_add(1),
                // SignalAndModifyByWaitingCountIfEqual: the new word tells a semaphore whether waiters remain.
                _ => match (count > 0).then_some(waiting.cmp(&count)) {
                    Some(std::cmp::Ordering::Greater) => value.wrapping_sub(1),
                    Some(std::cmp::Ordering::Equal) => value,
                    Some(std::cmp::Ordering::Less) => value.wrapping_add(1),
                    None if waiting > 0 => value.wrapping_sub(1),
                    None => value.wrapping_add(1),
                },
            };
            let _ = self.mem.write_u32(addr, updated as u32);
        }
        let mut woken = 0;
        for i in 0..self.threads.len() {
            if count >= 0 && woken >= count {
                break;
            }
            if matches!(self.threads[i].state, ThreadState::WaitAddress { addr: a, .. } if a == addr)
            {
                self.threads[i].state = ThreadState::Runnable;
                woken += 1;
            }
        }
        true
    }

    /// The soonest timed-wait deadline: the furthest the clock may idle forward.
    pub(crate) fn earliest_deadline(&self) -> Option<u64> {
        self.threads
            .iter()
            .filter(|t| !t.paused)
            .filter_map(|t| match t.state {
                ThreadState::WaitKey { deadline, .. }
                | ThreadState::WaitAddress { deadline, .. } => deadline,
                ThreadState::Sleeping { deadline } | ThreadState::WaitEvent { deadline } => {
                    Some(deadline)
                }
                _ => None,
            })
            .min()
    }
}
