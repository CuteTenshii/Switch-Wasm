//! Running a block: the loop over its ops, the conditional exits it may
//! leave through, and the terminator it ends on.
//!
//! Every arm here does what the interpreter's decoder would have done once it
//! finished decoding — in most cases by calling the very same helper, so the
//! two engines are one computation with two front ends.

use super::cache::JitStats;
use super::decode::translate;
use super::ir::{Block, Exit, Op, Term};
use crate::cpu::bits::*;
use crate::cpu::{Cpu, Result, RunReport, SELF_RETURN_TRAMPOLINE, TIME_SLICE};
use std::rc::Rc;

/// The operand an addition needs to compute a subtraction. `carry` is 1
/// exactly when the instruction subtracts, so it doubles as the mask that
/// inverts the operand — no branch, and nothing left to decide at run time.
#[inline(always)]
fn invert_if(v: u64, carry: u8) -> u64 {
    v ^ 0u64.wrapping_sub(u64::from(carry))
}

impl Cpu {
    /// Whether [`Cpu::run`] goes through the translator.
    pub fn jit_enabled(&self) -> bool {
        self.jit_enabled
    }

    /// Turn the translator on or off. Turning it off drops the cache, so a
    /// caller that switches back gets a translator with no stale state.
    pub fn set_jit_enabled(&mut self, on: bool) {
        if !on {
            self.jit.clear();
        }
        self.jit_enabled = on;
    }

    /// What the translator has been doing since the CPU was created.
    pub fn jit_stats(&self) -> JitStats {
        self.jit.stats()
    }

    /// Drop every translated block. Callers that rewrite guest code behind
    /// [`crate::mem::Memory`]'s back need this; ordinary guest stores are
    /// noticed on their own.
    pub fn jit_flush(&mut self) {
        self.jit.clear();
    }

    /// [`Cpu::run`] over translated blocks.
    ///
    /// The step budget is honoured exactly. A block whose remaining
    /// instructions do not fit is entered anyway and left part-way through,
    /// with `pc` on the instruction that would have come next — the block is
    /// a cache, not a unit of execution, so stopping inside one is no
    /// different from stopping between two interpreted instructions.
    pub(in crate::cpu) fn run_jit(&mut self, max_steps: u64) -> Result<RunReport> {
        let mut steps = 0u64;
        // The block last executed, kept across iterations. A loop that branches
        // back to its own head — which is most of what a hot loop is — then
        // costs neither the cache lookup nor the reference count, because the
        // handle is moved out and back rather than cloned.
        let mut held: Option<Rc<Block>> = None;
        while steps < max_steps && !self.halted {
            // The scheduler's preemption point. The interpreter takes it
            // between any two instructions; here it is between blocks, which
            // is the same thing at the scale of a 20,000-instruction slice.
            if self.slice_used >= TIME_SLICE {
                self.slice_used = 0;
                self.yield_thread();
            }
            self.sweep_timed_waits();
            let block = match held.take() {
                Some(block) if block.start == self.pc && !self.mem.has_dirty_code() => {
                    self.jit.executed += 1;
                    block
                }
                _ => self.jit_block_at(self.pc),
            };
            let ran = self.exec_block(&block, max_steps - steps)?;
            held = Some(block);
            self.slice_used += ran;
            steps += ran;
        }
        Ok(RunReport {
            steps,
            halted: self.halted,
        })
    }

    /// The block entered at `pc`, translating it if this is the first visit.
    ///
    /// Drains the pages guest stores have landed on first, so a block is never
    /// handed out after the instructions behind it have changed.
    fn jit_block_at(&mut self, pc: u32) -> Rc<Block> {
        if self.mem.has_dirty_code() {
            let dirty = self.mem.dirty_code_pages();
            self.jit.invalidate(&dirty);
        }
        self.jit.executed += 1;
        if let Some(block) = self.jit.get(pc) {
            return block;
        }
        let block = Rc::new(translate(&self.mem, pc));
        self.mem.mark_code_page(block.start);
        self.jit.translated += 1;
        self.jit.insert(block.clone());
        block
    }

    /// Run at most `budget` of a block's instructions, reporting how many
    /// retired.
    ///
    /// `self.pc` tracks the instruction being executed throughout, exactly as
    /// it does in the interpreter, so a fault inside a block reports the same
    /// address and the same register state an interpreted one would.
    fn exec_block(&mut self, block: &Block, budget: u64) -> Result<u64> {
        let body = (block.ops.len() as u64).min(budget) as usize;
        let mut i = 0usize;
        let mut pc = block.start;
        let mut next_exit = 0usize;
        loop {
            // Run straight through to the next conditional branch, or to the
            // end of what the budget allows. Taking the segment as a slice
            // keeps this the same bounds-check-free walk it was when a block
            // had no interior exits at all.
            let stop = match block.exits.get(next_exit) {
                Some(&(at, _)) if (at as usize) < body => at as usize,
                _ => body,
            };
            for (k, &op) in block.ops[i..stop].iter().enumerate() {
                if let Err(e) = self.exec_op(op, pc) {
                    // The clock, the step counter, the trail and `pc` are all
                    // settled here rather than maintained per instruction:
                    // nothing inside a block reads any of them, and a fault is
                    // the only thing that ever does. The faulting instruction
                    // counts, exactly as it does in the interpreter.
                    let at = i + k;
                    self.retire_run(block.start, at as u64 + 1);
                    self.pc = pc;
                    self.record_fault(&e, pc, block.words[at]);
                    return Err(e);
                }
                pc = pc.wrapping_add(4);
            }
            i = stop;
            if stop == body {
                break;
            }
            let (_, exit) = block.exits[next_exit];
            let span = exit.span();
            if i + span > body {
                // The budget splits a fused pair. Run the compare's half of it
                // and stop on the branch, which is a valid entry point with the
                // flags already set. Doing nothing here instead would return no
                // progress at all when the pair starts the block, and
                // [`Cpu::run_jit`] would spin on it forever.
                if i < body {
                    self.apply_compare(exit);
                    i += 1;
                    pc = pc.wrapping_add(4);
                }
                break;
            }
            i += span;
            pc = pc.wrapping_add(4 * span as u32);
            if self.take_exit(exit) {
                // Taken: the branch is the last instruction of this visit, and
                // `take_exit` has already put the target in `pc`.
                self.retire_run(block.start, i as u64);
                return Ok(i as u64);
            }
            next_exit += 1;
        }
        self.retire_run(block.start, i as u64);
        let mut ran = i as u64;
        match block.term {
            Some(term) if i == block.ops.len() && ran < budget => {
                self.pc = pc;
                self.record_run(pc, 1);
                let result = self.exec_term(term, pc);
                // After the terminator, not before, and whether or not it
                // faulted — which is what `step_inner` does. An `SVC` is the
                // one instruction that reads the clock while it runs, so
                // retiring it early made the JIT hand every syscall a tick the
                // interpreter had not spent yet: sdl-hello ended 1 cycle apart
                // between the two engines, because a sleep deadline is
                // computed from the value the syscall saw.
                self.retire();
                if let Err(e) = result {
                    self.record_fault(&e, pc, block.words[i]);
                    return Err(e);
                }
                ran += 1;
            }
            // Either the budget ran out inside the block, or it covered the
            // body of a block that has no terminator. Both leave `pc` on the
            // next instruction to run.
            _ => self.pc = pc,
        }
        Ok(ran)
    }

    /// The flag-setting half of a fused compare-and-branch, for the one case
    /// that cannot run both: a step budget that ends between them.
    #[inline(always)]
    fn apply_compare(&mut self, exit: Exit) {
        let (a, b, carry, sf) = match exit {
            Exit::CmpImm {
                rn, rhs, carry, sf, ..
            } => (self.reg_at(rn) & Cpu::mask(sf), rhs, carry, sf),
            Exit::CmpReg {
                rn, rm, carry, sf, ..
            } => (
                self.reg_at(rn) & Cpu::mask(sf),
                invert_if(self.reg_at(rm) & Cpu::mask(sf), carry),
                carry,
                sf,
            ),
            // Nothing else spans two instructions.
            _ => return,
        };
        let (result, c, v) = Cpu::add_carry_overflow(a, b, u64::from(carry), sf);
        self.set_nzcv_from_alu(result, sf, c, v);
    }

    /// Evaluate a conditional branch inside a block. Returns whether it was
    /// taken — in which case `pc` is where control went and the block is over,
    /// and otherwise the block carries on at the following instruction with
    /// `pc` still to be settled by the caller.
    #[inline(always)]
    fn take_exit(&mut self, exit: Exit) -> bool {
        let (taken, target) = match exit {
            Exit::Cond { cond, target } => (self.condition_holds(cond), target),
            Exit::CmpImm {
                rn,
                rhs,
                carry,
                sf,
                cond,
                target,
            } => {
                let a = self.reg_at(rn) & Cpu::mask(sf);
                let (result, c, v) = Cpu::add_carry_overflow(a, rhs, u64::from(carry), sf);
                self.set_nzcv_from_alu(result, sf, c, v);
                (self.condition_holds(cond), target)
            }
            Exit::CmpReg {
                rn,
                rm,
                carry,
                sf,
                cond,
                target,
            } => {
                let a = self.reg_at(rn) & Cpu::mask(sf);
                let b = invert_if(self.reg_at(rm) & Cpu::mask(sf), carry);
                let (result, c, v) = Cpu::add_carry_overflow(a, b, u64::from(carry), sf);
                self.set_nzcv_from_alu(result, sf, c, v);
                (self.condition_holds(cond), target)
            }
            Exit::Cbz { rt, sf, nz, target } => {
                let val = self.read_zr(rt);
                let is_zero = if sf { val == 0 } else { (val as u32) == 0 };
                (is_zero == !nz, target)
            }
            Exit::Tbz {
                rt,
                bit,
                nz,
                target,
            } => {
                let set = (self.read_zr(rt) >> bit) & 1 == 1;
                (set == nz, target)
            }
        };
        if taken {
            self.pc = target;
        }
        taken
    }

    /// Execute one body op. Every arm does what the interpreter's decoder
    /// would have done once it finished decoding.
    #[inline(always)]
    fn exec_op(&mut self, op: Op, pc: u32) -> Result<()> {
        match op {
            Op::Nop => {}
            // The three arms that re-enter the interpreter are the only ones
            // that need `pc` in the register file: `execute` resolves
            // PC-relative forms from it, and every fault message names it.
            Op::Interpret { insn } => {
                self.pc = pc;
                self.jit.interpreted += 1;
                self.execute(insn, pc.wrapping_add(4))?;
            }
            Op::Fp { insn, scalar, form } => {
                self.pc = pc;
                let next = pc.wrapping_add(4);
                let claimed = if scalar {
                    self.run_fp(form, insn)? || self.try_simd(insn)?
                } else {
                    self.try_simd(insn)? || self.run_fp(form, insn)?
                };
                if claimed {
                    self.pc = next;
                } else {
                    self.execute_chain(insn, next)?;
                }
            }
            Op::System { insn } => {
                self.pc = pc;
                self.system(insn, pc.wrapping_add(4))?;
            }
            Op::Sys { op } => self.exec_sys(op)?,

            Op::MovConst { rd, val } => self.set_reg_at(rd, val),
            Op::MovK { rd, shift, val, sf } => self.movk(rd, shift, val, sf),

            Op::AddSubImm {
                rd,
                rn,
                rhs,
                carry,
                set_flags,
                sf,
            } => {
                self.add_sub_pre(rd, rn, rhs, carry, set_flags, sf);
            }
            Op::AddSubReg {
                rd,
                rn,
                rm,
                carry,
                set_flags,
                sf,
            } => {
                let v = self.reg_at(rm) & Cpu::mask(sf);
                self.add_sub_pre(rd, rn, invert_if(v, carry), carry, set_flags, sf);
            }
            Op::AddSubShifted {
                rd,
                rn,
                rm,
                st,
                sa,
                carry,
                set_flags,
                sf,
            } => {
                let v = shift_reg(
                    self.reg_at(rm) & Cpu::mask(sf),
                    u32::from(st),
                    u32::from(sa),
                    sf,
                );
                self.add_sub_pre(rd, rn, invert_if(v, carry), carry, set_flags, sf);
            }
            Op::AddSubExtended {
                rd,
                rn,
                rm,
                option,
                shift,
                carry,
                set_flags,
                sf,
            } => {
                let v = extend_reg(self.reg_at(rm), option, sf) & Cpu::mask(sf);
                let v = v.wrapping_shl(u32::from(shift)) & Cpu::mask(sf);
                self.add_sub_pre(rd, rn, invert_if(v, carry), carry, set_flags, sf);
            }

            Op::LogicalImm {
                rd,
                rn,
                imm,
                opc,
                sf,
            } => {
                // Rd == 31 is SP for AND/ORR/EOR and the zero register only
                // for ANDS — one of the few places the two differ, and one
                // the translator has already settled into `rd`.
                self.logical(rd, rn, imm, opc, sf);
            }
            Op::LogicalShifted {
                rd,
                rn,
                rm,
                st,
                sa,
                opc,
                invert,
                sf,
            } => {
                let b = shift_reg(
                    self.reg_at(rm) & Cpu::mask(sf),
                    u32::from(st),
                    u32::from(sa),
                    sf,
                );
                // `BIC`/`ORN`/`EON` invert the *shifted* operand, not the
                // register: `ir.Not(ShiftReg(...))` in dynarmic, and the same
                // order in the ARM ARM's pseudocode.
                let b = if invert { !b & Cpu::mask(sf) } else { b };
                self.logical(rd, rn, b, opc, sf);
            }
            Op::LogicalReg {
                rd,
                rn,
                rm,
                opc,
                invert,
                sf,
            } => {
                let b = self.reg_at(rm) & Cpu::mask(sf);
                let b = if invert { !b & Cpu::mask(sf) } else { b };
                self.logical(rd, rn, b, opc, sf);
            }

            Op::Bitfield {
                rd,
                rn,
                opc,
                immr,
                imms,
                sf,
            } => self.bitfield(rd, rn, opc, immr, imms, sf),
            Op::Extr {
                rd,
                rn,
                rm,
                imm,
                sf,
            } => self.extr(rd, rn, rm, imm, sf),

            Op::CondSel {
                rd,
                rn,
                rm,
                cond,
                else_inv,
                else_inc,
                sf,
            } => self.cond_sel(rd, rn, rm, cond, else_inv, else_inc, sf),
            Op::CondCmp {
                rn,
                rm,
                imm,
                cond,
                nzcv,
                sub,
                is_imm,
                sf,
            } => self.cond_cmp(rn, rm, imm, cond, nzcv, sub, is_imm, sf),

            Op::Madd {
                rd,
                rn,
                rm,
                ra,
                sub,
                sf,
            } => self.madd(rd, rn, rm, ra, sub, sf),
            Op::MaddLong {
                rd,
                rn,
                rm,
                ra,
                sub,
                signed,
            } => self.madd_long(rd, rn, rm, ra, sub, signed),
            Op::Mulh { rd, rn, rm, signed } => self.mulh(rd, rn, rm, signed),

            Op::LoadStoreImm {
                rt,
                rn,
                acc,
                wb,
                offset,
            } => {
                let base = self.reg_at(rn);
                let (addr, wb_val) = Self::indexed(base, offset, wb);
                self.access(addr as u32, rt, acc)?;
                if let Some(v) = wb_val {
                    self.set_reg_at(rn, v);
                }
            }
            Op::LoadStoreReg {
                rt,
                rn,
                rm,
                ext,
                shift,
                acc,
            } => {
                let offset = self.reg_offset(rm, ext, shift);
                let addr = (self.reg_at(rn) as i64).wrapping_add(offset) as u32;
                self.access(addr, rt, acc)?;
            }
            Op::Pair {
                rt,
                rt2,
                rn,
                offset,
                kind,
                wb,
            } => self.pair(rt, rt2, rn, offset, kind, wb)?,
            Op::LoadLiteral { rt, addr, acc } => self.access(addr, rt, acc)?,
        }
        Ok(())
    }

    /// Execute the instruction a block ends on, leaving `self.pc` wherever
    /// control goes next.
    #[inline(always)]
    fn exec_term(&mut self, term: Term, pc: u32) -> Result<()> {
        match term {
            Term::B { target } => self.pc = target,
            Term::Bl { target, ret_pc } => {
                self.write_zr(30, u64::from(ret_pc));
                self.pc = target;
            }
            Term::Br { rn } => self.pc = self.read_zr(rn) as u32,
            Term::Blr { rn, ret_pc } => {
                // Read the target before linking: `blr x30` is a
                // return-and-relink, and writing x30 first makes it jump to
                // itself.
                let target = self.read_zr(rn) as u32;
                self.write_zr(30, u64::from(ret_pc));
                self.pc = target;
            }
            Term::Ret { rn } => {
                // A return to address 0 is homebrew's exit path; the boot
                // model routes it through the exit trampoline instead of
                // fetching from NULL.
                let target = self.read_zr(rn) as u32;
                self.pc = if target == 0 {
                    SELF_RETURN_TRAMPOLINE
                } else {
                    target
                };
            }
            Term::Svc { imm, next } => {
                // Retire the SVC before dispatching it: a syscall that
                // switches threads installs the incoming thread's PC, and the
                // outgoing one has to resume after its own SVC.
                self.pc = next;
                self.syscall(imm)?;
            }
            Term::Interpret { insn, next } => {
                self.jit.interpreted += 1;
                self.execute(insn, next)?;
            }
            Term::Fetch => {
                let insn = self.mem.fetch(pc)?;
                self.jit.interpreted += 1;
                self.execute(insn, pc.wrapping_add(4))?;
            }
        }
        Ok(())
    }
}
