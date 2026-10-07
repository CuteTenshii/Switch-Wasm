//! Instruction stepping, the run loop and top-level decode.

use super::*;

impl Cpu {
    // ---- main execution ----

    /// Execute a single instruction.
    pub fn step(&mut self) -> Result<()> {
        self.step_inner()
    }

    /// Body of [`Cpu::step`], inlined into [`Cpu::run`]'s loop.
    #[inline(always)]
    fn step_inner(&mut self) -> Result<()> {
        if self.halted {
            return Err(Error::Cpu("attempted to step a halted CPU".into()));
        }
        // Preemption point; `yield_thread` is a no-op when nothing else can run.
        self.slice_used += 1;
        if self.slice_used >= TIME_SLICE {
            self.slice_used = 0;
            self.yield_thread();
        }
        self.sweep_timed_waits();
        let pc = self.pc;
        let insn = match self.mem.fetch(pc) {
            Ok(i) => i,
            Err(e) => {
                self.record_fault(&e, pc, 0);
                return Err(e);
            }
        };
        let next_pc = pc.wrapping_add(4);
        self.record_run(pc, 1);
        let result = match self.mode {
            ExecMode::A64 => self.execute(insn, next_pc),
            ExecMode::A32 => self.execute_a32(insn),
        };
        if self.trace_enabled {
            self.trace_line(&format!(
                "{:08x}: {:08x}  {}\n",
                pc,
                insn,
                self.disassemble_for_mode(insn)
            ));
        }
        if let Err(e) = &result {
            self.record_fault(e, pc, insn);
        }
        self.retire();
        result
    }

    pub(super) fn record_fault(&mut self, e: &Error, pc: u32, insn: u32) {
        // Traces from parts without a `Cpu` belong before the fault.
        self.absorb_traces();
        // Unmarked, so the separator does not grade the previous line.
        self.trace_line("\n");
        // The dump and trail below inherit the fault's level.
        self.trace_marked(
            Level::Error,
            &format!(
                "=== FAULT ===\n{}\n  at pc={:#010x} insn={:#010x}  {}",
                e,
                pc,
                insn,
                if insn == 0 {
                    String::new()
                } else {
                    self.disassemble_for_mode(insn)
                }
            ),
        );
        self.trace_regs(pc);
        self.trace_trail();
    }

    /// Trace the run-up to the current PC, expanding the recorded runs.
    pub(super) fn trace_trail(&mut self) {
        let trail = self.trail_text();
        if !trail.is_empty() {
            self.trace_line(&trail);
        }
    }

    /// The trail as text: a heading and one disassembled instruction per line.
    pub(super) fn trail_text(&self) -> String {
        let runs = self.recent_len.min(RECENT_LEN);
        if runs == 0 {
            return String::new();
        }
        let first = self.recent_len.wrapping_sub(runs) % RECENT_LEN;
        let mut trail: Vec<(u32, u32)> = Vec::new();
        for i in 0..runs {
            let (start, count) = self.recent[(first + i) % RECENT_LEN];
            for step in 0..count {
                let at = start.wrapping_add(4 * step);
                let word = self.mem.fetch(at).unwrap_or(0);
                trail.push((at, word));
            }
        }
        let shown = trail.len().min(RECENT_LEN);
        let mut text = format!("--- last {shown} instructions ---\n");
        for &(ipc, iinsn) in &trail[trail.len() - shown..] {
            text.push_str(&format!(
                "{:08x}: {:08x}  {}\n",
                ipc,
                iinsn,
                self.disassemble_for_mode(iinsn)
            ));
        }
        text
    }

    /// Run up to `max_steps` instructions, stopping early on halt or error. Uses the JIT
    /// when enabled, except with full tracing, which needs the interpreter.
    pub fn run(&mut self, max_steps: u64) -> Result<RunReport> {
        self.complete_pending_present();
        if self.jit_enabled && !self.trace_enabled && self.mode == ExecMode::A64 {
            return self.run_jit(max_steps);
        }
        let mut steps = 0u64;
        while steps < max_steps && !self.halted {
            self.step_inner()?;
            steps += 1;
        }
        Ok(RunReport {
            steps,
            halted: self.halted,
        })
    }

    #[inline]
    fn b_imm(&mut self, next_pc: &mut u32, imm: i64) {
        *next_pc = (self.pc as i64).wrapping_add(imm) as u32;
    }

    /// Route an instruction by its top-level group (bits 28:25) to that group's decoder
    /// first, falling back to [`Cpu::execute_chain`].
    pub(super) fn execute(&mut self, insn: u32, next_pc: u32) -> Result<()> {
        let mut pc = next_pc;
        match (insn >> 25) & 0xF {
            // Data processing -- immediate, PC-relative addressing included.
            0x8 | 0x9 => {
                if self.try_pc_relative(insn) || self.try_data_proc_imm(insn, &mut pc)? {
                    self.pc = pc;
                    return Ok(());
                }
            }
            // Data processing -- register.
            0x5 | 0xD => {
                if self.try_data_proc_reg(insn, &mut pc)? {
                    self.pc = pc;
                    return Ok(());
                }
            }
            // Loads and stores, the literal (PC-relative) forms included.
            0x4 | 0x6 | 0xC | 0xE => {
                if self.try_load_literal(insn)? || self.try_load_store(insn, &mut pc)? {
                    self.pc = pc;
                    return Ok(());
                }
            }
            // SIMD and FP: scalar FP top bytes are 0x1E/0x1F and 0x9E/0x9F; ask that decoder first.
            0x7 | 0xF => {
                let scalar_fp = matches!((insn >> 24) & 0xFF, 0x1E | 0x1F | 0x9E | 0x9F);
                #[allow(clippy::if_same_then_else)] // the order is the point
                let claimed = if scalar_fp {
                    self.try_fp(insn)? || self.try_simd(insn)?
                } else {
                    self.try_simd(insn)? || self.try_fp(insn)?
                };
                if claimed {
                    self.pc = pc;
                    return Ok(());
                }
            }
            // Branches, exception generation and system instructions.
            #[allow(clippy::collapsible_match)] // no fallible call in a guard
            0xA | 0xB => {
                if self.try_branch_or_system(insn, next_pc)? {
                    return Ok(());
                }
            }
            // Reserved and SVE groups, left to the chain.
            _ => {}
        }
        self.execute_chain(insn, next_pc)
    }

    /// ADR/ADRP: bits[28:24] == 10000; bits[30:29] are immlo.
    fn try_pc_relative(&mut self, insn: u32) -> bool {
        if ((insn >> 24) & 0x1F) != 0b10000 {
            return false;
        }
        let rd = (insn & 0x1F) as u8;
        let immhi = ((insn >> 5) & 0x7_FFFF) as u64;
        let immlo = ((insn >> 29) & 0b11) as u64;
        let imm = sext_u64((immhi << 2) | immlo, 21);
        let page = (insn >> 31) & 1 == 1;
        let target = if page {
            ((self.pc & !0xFFF) as u64).wrapping_add(imm.wrapping_shl(12))
        } else {
            (self.pc as u64).wrapping_add(imm)
        };
        self.write_zr(rd, target);
        true
    }

    /// `LDR Xt, label` and friends.
    fn try_load_literal(&mut self, insn: u32) -> Result<bool> {
        if ((insn >> 27) & 0b111) != 0b011
            || ((insn >> 26) & 1) != 0
            || ((insn >> 24) & 0b11) != 0b00
        {
            return Ok(false);
        }
        let rt = (insn & 0x1F) as u8;
        let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
        let addr = (self.pc as i64).wrapping_add(imm as i64) as u32;
        match (insn >> 30) & 0b11 {
            0b00 => {
                let val = self.mem.read_u32(addr)? as u64;
                self.write_zr(rt, val & u64::from(u32::MAX));
            }
            0b01 => {
                let val = self.mem.read_u64(addr)?;
                self.write_zr(rt, val);
            }
            0b10 => {
                let val = self.mem.read_u32(addr)? as u64;
                self.write_zr(rt, sext_u64(val, 32));
            }
            // PRFM: a prefetch hint.
            _ => {}
        }
        Ok(true)
    }

    /// Branches, exceptions and system (bits 28:25 = 101x), dispatched on the top byte
    /// in order of frequency. Returns whether handled; handlers set `self.pc`.
    fn try_branch_or_system(&mut self, insn: u32, mut next_pc: u32) -> Result<bool> {
        match (insn >> 24) & 0xFF {
            // B.cond
            0x54 => {
                let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
                let cond = (insn & 0xF) as u8;
                if self.condition_holds(cond) {
                    self.b_imm(&mut next_pc, imm as i64);
                }
                self.pc = next_pc;
                Ok(true)
            }
            // B #imm
            0x14..=0x17 => {
                let imm = sext_u64((insn & 0x3FF_FFFF) as u64, 26) << 2;
                self.b_imm(&mut next_pc, imm as i64);
                self.pc = next_pc;
                Ok(true)
            }
            // TBZ / TBNZ
            0x36 | 0x37 | 0xB6 | 0xB7 => {
                let rt = (insn & 0x1F) as u8;
                let nz = ((insn >> 24) & 1) == 1;
                let bit = ((insn >> 31) & 1) << 5 | ((insn >> 19) & 0x1F);
                let imm = sext_u64((insn >> 5) & 0x3FFF, 14) << 2;
                let bit_val = (self.read_zr(rt) >> bit) & 1 == 1;
                if bit_val == nz {
                    self.b_imm(&mut next_pc, imm as i64);
                }
                self.pc = next_pc;
                Ok(true)
            }
            // CBZ / CBNZ
            0x34 | 0x35 | 0xB4 | 0xB5 => {
                let rt = (insn & 0x1F) as u8;
                let nz = ((insn >> 24) & 1) == 1;
                let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
                let val = self.read_zr(rt);
                let is_zero = if (insn >> 31) & 1 == 1 {
                    val == 0
                } else {
                    (val as u32) == 0
                };
                if is_zero == !nz {
                    self.b_imm(&mut next_pc, imm as i64);
                }
                self.pc = next_pc;
                Ok(true)
            }
            // BL #imm
            0x94..=0x97 => {
                let imm = sext_u64((insn & 0x3FF_FFFF) as u64, 26) << 2;
                self.write_zr(30, next_pc as u64);
                self.b_imm(&mut next_pc, imm as i64);
                self.pc = next_pc;
                Ok(true)
            }
            // BR / BLR / RET
            0xD6 | 0xD7 => {
                let opc = (insn >> 21) & 0xF;
                let op2 = (insn >> 16) & 0x1F;
                let op3 = (insn >> 10) & 0x3F;
                if op2 != 0x1F || op3 != 0 {
                    return Ok(false);
                }
                let rn = ((insn >> 5) & 0x1F) as u8;
                match opc {
                    0b0000 => {
                        // BR
                        self.pc = self.read_zr(rn) as u32;
                        Ok(true)
                    }
                    0b0001 => {
                        // BLR: read the target before linking, since it may be x30.
                        let target = self.read_zr(rn) as u32;
                        self.write_zr(30, next_pc as u64);
                        self.pc = target;
                        Ok(true)
                    }
                    0b0010 => {
                        // RET to 0 is a homebrew exit path; redirect to the exit trampoline.
                        let tgt = self.read_zr(rn) as u32;
                        self.pc = if tgt == 0 {
                            SELF_RETURN_TRAMPOLINE
                        } else {
                            tgt
                        };
                        Ok(true)
                    }
                    _ => Err(Error::Cpu(format!(
                        "unimplemented branch-register opc {:#b} at {:#x}",
                        opc, self.pc
                    ))),
                }
            }
            // Exception generation: SVC/HVC/SMC and BRK.
            0xD4 => match (insn >> 21) & 0b111 {
                0b000 => {
                    if (insn & 0x1F) == 0b00001 {
                        let imm = ((insn >> 5) & 0xFFFF) as u16;
                        // Retire the SVC first: a thread switch installs the incoming thread's PC.
                        self.pc = next_pc;
                        self.syscall(imm)?;
                        Ok(true)
                    } else {
                        Err(Error::Cpu(format!(
                            "unimplemented HVC/SMC at {:#x}",
                            self.pc
                        )))
                    }
                }
                0b001 => {
                    let imm = ((insn >> 5) & 0xFFFF) as u16;
                    Err(Error::Cpu(format!("BRK #{} at {:#x}", imm, self.pc)))
                }
                _ => Err(Error::Cpu(format!(
                    "unimplemented exception instruction at {:#x}",
                    self.pc
                ))),
            },
            // MSR/MRS, barriers and hints.
            0xD5 => {
                if ((insn >> 22) & 0x3FF) != 0b1101010100 {
                    return Ok(false);
                }
                self.system(insn, next_pc)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Fallback over the whole encoding space, kept out of line to keep `execute` small.
    /// Order: branch/system, load literal, load/store, SIMD, scalar FP, PC-relative,
    /// DP immediate, DP register.
    #[cold]
    #[inline(never)]
    pub(super) fn execute_chain(&mut self, insn: u32, mut next_pc: u32) -> Result<()> {
        if self.try_branch_or_system(insn, next_pc)? {
            return Ok(());
        }

        if self.try_load_literal(insn)? {
            self.pc = next_pc;
            return Ok(());
        }

        if self.try_load_store(insn, &mut next_pc)? {
            self.pc = next_pc;
            return Ok(());
        }

        if self.try_simd(insn)? {
            self.pc = next_pc;
            return Ok(());
        }

        if self.try_fp(insn)? {
            self.pc = next_pc;
            return Ok(());
        }

        if self.try_pc_relative(insn) {
            self.pc = next_pc;
            return Ok(());
        }

        if self.try_data_proc_imm(insn, &mut next_pc)? {
            self.pc = next_pc;
            return Ok(());
        }

        if self.try_data_proc_reg(insn, &mut next_pc)? {
            self.pc = next_pc;
            return Ok(());
        }

        Err(Error::Cpu(format!(
            "unimplemented instruction 0x{:08x} at pc={:#x}",
            insn, self.pc
        )))
    }
}
