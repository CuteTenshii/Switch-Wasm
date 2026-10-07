//! Host-side diagnostics, tracing and backtraces.

use super::*;

impl Cpu {
    /// Index of the running thread, for host-side sampling profilers.
    pub fn current_thread_index(&self) -> usize {
        self.current_thread
    }

    /// Debugging lever: make every blocked thread runnable; returns the count.
    pub fn wake_all_blocked(&mut self) -> usize {
        let mut woken = 0;
        for index in 0..self.threads.len() {
            match self.threads[index].state {
                ThreadState::WaitKey { mutex, .. } => self.wake_condvar_waiter(index, mutex),
                ThreadState::WaitMutex(_) | ThreadState::WaitAddress { .. } => {
                    self.threads[index].state = ThreadState::Runnable;
                }
                _ => continue,
            }
            woken += 1;
        }
        woken
    }

    /// Every `(interface, command)` reported as unimplemented, sorted.
    pub fn unimplemented_ipc(&self) -> Vec<(String, Option<u32>)> {
        let mut all: Vec<(String, Option<u32>)> = self.unimplemented_ipc.iter().cloned().collect();
        all.sort();
        all
    }

    /// Every `(interface, command)` answered by a stub; see [`Cpu::warn_stub`].
    pub fn stubbed_ipc(&self) -> Vec<(String, Option<u32>)> {
        let mut all: Vec<(String, Option<u32>)> = self.stubbed_ipc.iter().cloned().collect();
        all.sort();
        all
    }

    /// Count one failed nvdrv ioctl towards the next [`Cpu::take_nv_errors`].
    pub(crate) fn count_nv_error(&mut self, node: &str, request: u32, error: u32) {
        /// Distinct failures held between readings.
        const CAP: usize = 64;
        if let Some(calls) = self
            .nv_errors
            .iter_mut()
            .find(|((n, r, e), _)| n == node && *r == request && *e == error)
            .map(|(_, calls)| calls)
        {
            *calls += 1;
        } else if self.nv_errors.len() < CAP {
            self.nv_errors.insert((node.to_owned(), request, error), 1);
        }
    }

    /// nvdrv ioctl failures since the last call: node, request, error, count.
    pub fn take_nv_errors(&mut self) -> Vec<(String, u32, u32, u64)> {
        std::mem::take(&mut self.nv_errors)
            .into_iter()
            .map(|((node, request, error), calls)| (node, request, error, calls))
            .collect()
    }

    /// Requested `HidNpadStyleTag` bits (0 before any) and the presented style.
    pub fn npad_styles(&self) -> (u32, u32) {
        (
            self.npad_style_set,
            npad_presentation_for(self.npad_style_set).style,
        )
    }

    /// Debugging lever: start every created-but-unstarted thread; returns the count.
    pub fn start_created_threads(&mut self) -> usize {
        let mut started = 0;
        for thread in &mut self.threads {
            if thread.state == ThreadState::Created {
                thread.state = ThreadState::Runnable;
                started += 1;
            }
        }
        started
    }

    pub fn thread_dump(&self) -> String {
        let mut out = String::new();
        // Report the implicit main thread for a program that never created one.
        if self.threads.is_empty() {
            out.push_str(&format!(
                "  [0]* handle={MAIN_THREAD_HANDLE:#x} state=Runnable paused=false pc={:#x}\n  \
                 (the main thread, which has no slot of its own until the guest creates a \
                 second)\n",
                self.pc
            ));
            return out;
        }
        for (index, thread) in self.threads.iter().enumerate() {
            let running = index == self.current_thread;
            out.push_str(&format!(
                "  [{index}]{} handle={:#x} priority={} state={:?} paused={} pc={:#x}\n",
                if running { "*" } else { " " },
                thread.handle,
                thread.priority,
                thread.state,
                thread.paused,
                if running { self.pc } else { thread.pc },
            ));
        }
        out
    }

    /// Record a user-facing diagnostic: to stderr on the host and to the trace buffer
    /// the browser drains, with a level the page colours by.
    pub fn diagnostic(&mut self, level: Level, line: &str) {
        // Not `traceln!`: that would feed the pending sink a second time.
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!("{line}");
        self.absorb_traces();
        self.trace_marked(level, line);
    }

    /// Fold in traces from parts without a `Cpu` (rasterizer, shader translator, texture
    /// decoder); called before diagnostics, faults and drains to keep ordering.
    pub fn absorb_traces(&mut self) {
        let pending = crate::trace::take_pending();
        if pending.is_empty() {
            return;
        }
        self.note_dropped_trace();
        self.trace.extend_from_slice(&pending);
        self.trim_trace();
    }

    /// Append an unmarked line, a continuation of the previous one.
    pub(crate) fn trace_line(&mut self, line: &str) {
        self.note_dropped_trace();
        self.trace.extend_from_slice(line.as_bytes());
        self.trim_trace();
    }

    /// Append a line at `level`, inherited by following unmarked lines.
    pub(crate) fn trace_marked(&mut self, level: Level, line: &str) {
        self.note_dropped_trace();
        self.trace.push(level.marker());
        self.trace.extend_from_slice(line.as_bytes());
        if !line.ends_with('\n') {
            self.trace.push(b'\n');
        }
        self.trim_trace();
    }

    /// Note lost text once per loss, where it was lost.
    fn note_dropped_trace(&mut self) {
        if !self.trace_dropped {
            return;
        }
        self.trace_dropped = false;
        self.trace.push(Level::Warn.marker());
        self.trace
            .extend_from_slice(b"[trace] the buffer filled; older lines above were dropped\n");
    }

    /// Drop the oldest quarter of the trace at a line boundary to get under the cap.
    fn trim_trace(&mut self) {
        if self.trace.len() <= self.trace_cap {
            return;
        }
        let least = self.trace.len() - self.trace_cap;
        let want = (least + self.trace_cap / 4).min(self.trace.len());
        let cut = match self.trace[want..].iter().position(|&b| b == b'\n') {
            Some(at) => want + at + 1,
            // A single line longer than the buffer: drop it all.
            None => self.trace.len(),
        };
        self.trace.drain(..cut);
        self.trace_dropped = true;
    }

    pub(crate) fn trace_regs(&mut self, pc: u32) {
        let dump = self.reg_dump();
        self.trace_line(&dump);
        let _ = pc;
    }

    /// Walk the frame-pointer chain and return return addresses, innermost first.
    /// Frames must lie above SP and addresses must follow a call, since not all code keeps x29.
    pub fn backtrace(&self, depth: usize) -> Vec<u32> {
        self.walk_frames(&self.regs, self.mode, depth)
    }

    /// [`Cpu::backtrace`] over any register file, live or saved.
    pub(crate) fn walk_frames(
        &self,
        regs: &[u64; REG_FILE],
        mode: ExecMode,
        depth: usize,
    ) -> Vec<u32> {
        // x29/x30 with 16-byte frames in A64, r11/r14 with 8-byte frames in AArch32.
        let (mut fp, lr, sp, width) = match mode {
            ExecMode::A64 => (regs[29] as u32, regs[30] as u32, regs[SP_SLOT] as u32, 8),
            ExecMode::A32 => (regs[11] as u32, regs[14] as u32, regs[13] as u32, 4),
        };
        let mut out = Vec::with_capacity(depth + 1);
        if self.is_return_address(lr, mode) {
            out.push(lr);
        }
        // The first frame is at or above SP; `next_fp <= fp` keeps later ones above it.
        if fp < sp {
            return out;
        }
        for _ in 0..depth {
            if !fp.is_multiple_of(width) {
                break;
            }
            let read = |at: u32| match mode {
                ExecMode::A64 => self.mem.read_u64(at).map(|v| v as u32),
                ExecMode::A32 => self.mem.read_u32(at),
            };
            let (next_fp, lr) = match (read(fp), read(fp.wrapping_add(width))) {
                (Ok(next_fp), Ok(lr)) => (next_fp, lr),
                _ => break,
            };
            if next_fp <= fp || !self.is_return_address(lr, mode) {
                break;
            }
            out.push(lr);
            fp = next_fp;
        }
        out
    }

    /// Whether `addr` follows a call instruction or is a thread/host return stub.
    fn is_return_address(&self, addr: u32, mode: ExecMode) -> bool {
        if addr == THREAD_EXIT_TRAMPOLINE || addr == SELF_RETURN_TRAMPOLINE {
            return true;
        }
        if addr < 4 || !addr.is_multiple_of(4) {
            return false;
        }
        let Ok(call) = self.mem.read_u32(addr - 4) else {
            return false;
        };
        match mode {
            // BL, then BLR.
            ExecMode::A64 => call & 0xFC00_0000 == 0x9400_0000 || call & 0xFFFF_FC1F == 0xD63F_0000,
            // BL (cond != 1111), BLX to an immediate, then BLX to a register.
            ExecMode::A32 => {
                (call & 0x0F00_0000 == 0x0B00_0000 && call >> 28 != 0xF)
                    || call & 0xFE00_0000 == 0xFA00_0000
                    || call & 0x0FFF_FFF0 == 0x012F_FF30
            }
        }
    }

    /// One general-purpose register, for host-side debuggers.
    pub fn reg(&self, i: usize) -> u64 {
        self.regs[i]
    }

    /// A register snapshot named for the current state (A64 or AArch32).
    pub fn reg_dump(&self) -> String {
        use std::fmt::Write;
        let mut s = String::with_capacity(1024);
        let n = (self.nzcv >> 31) & 1;
        let z = (self.nzcv >> 30) & 1;
        let c = (self.nzcv >> 29) & 1;
        let v = (self.nzcv >> 28) & 1;
        match self.mode {
            ExecMode::A64 => {
                let _ = writeln!(
                    s,
                    "pc={:#010x}  sp={:#018x}  nzcv=N:{n} Z:{z} C:{c} V:{v}",
                    self.pc, self.regs[SP_SLOT]
                );
                for i in 0..31 {
                    let _ = write!(s, "x{:<2}={:#018x}  ", i, self.regs[i]);
                    if i % 4 == 3 {
                        let _ = writeln!(s);
                    }
                }
            }
            ExecMode::A32 => {
                let _ = writeln!(
                    s,
                    "pc={:#010x}  nzcv=N:{n} Z:{z} C:{c} V:{v} Q:{} ge={:#06b}",
                    self.pc,
                    u8::from(self.cpsr_q),
                    self.cpsr_ge
                );
                for i in 0..15 {
                    let name = match i {
                        13 => "sp".to_string(),
                        14 => "lr".to_string(),
                        _ => format!("r{i}"),
                    };
                    let _ = write!(s, "{name:<3}={:#010x}  ", self.regs[i] as u32);
                    if i % 4 == 3 {
                        let _ = writeln!(s);
                    }
                }
                let _ = write!(s, "pc ={:#010x}  ", self.pc);
            }
        }
        let _ = writeln!(s);
        s
    }
}
