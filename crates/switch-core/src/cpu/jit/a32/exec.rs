//! Running a block: its ops through the interpreter's helpers, then its terminator.

use super::decode::translate;
use super::{Block, Kind, Offset, Op, Operand, Term, TermKind, ALWAYS};
use crate::cpu::a32::shift::{shift_by_register, shift_c};
use crate::cpu::{Cpu, ExecMode, Result, RunReport};
use std::rc::Rc;

impl Cpu {
    /// [`Cpu::run`] over translated A32 blocks. The step budget is exact: a
    /// block that does not fit is left part-way through. Returns early if the
    /// running thread is no longer A32.
    pub(in crate::cpu) fn run_a32_jit(&mut self, max_steps: u64) -> Result<RunReport> {
        let mut steps = 0u64;
        while steps < max_steps && !self.halted {
            self.preemption_point();
            if self.handback.is_some() || self.mode != ExecMode::A32 {
                break;
            }
            let block = self.a32_block_at(self.pc);
            let ran = self.exec_a32_block(&block, max_steps - steps)?;
            self.slice_used += ran;
            steps += ran;
        }
        Ok(RunReport {
            steps,
            halted: self.halted,
        })
    }

    /// The block entered at `pc`, translating it on first visit. Drains dirty
    /// code pages first so a stale block is never handed out.
    fn a32_block_at(&mut self, pc: u32) -> Rc<Block> {
        if self.mem.has_dirty_code(self.core) {
            match self.mem.dirty_code_pages(self.core) {
                Some(dirty) => self.jit.invalidate(&dirty),
                None => self.jit.clear(),
            }
        }
        if let Some(block) = self.jit.a32.get(pc) {
            self.jit.executed += 1;
            return block;
        }
        let block = Rc::new(translate(&self.mem, pc));
        for &page in &block.pages {
            self.mem.mark_code_page(page << crate::mem::PAGE_BITS);
        }
        self.jit.translated += 1;
        self.jit.executed += 1;
        self.jit.a32.insert(block.clone());
        block
    }

    /// Run at most `budget` of a block's instructions, returning how many
    /// retired. `self.pc` tracks the current instruction, so faults match the
    /// interpreter.
    fn exec_a32_block(&mut self, block: &Block, budget: u64) -> Result<u64> {
        let body = (block.ops.len() as u64).min(budget) as usize;
        let mut pc = block.start;
        for (i, op) in block.ops[..body].iter().enumerate() {
            if op.cond != ALWAYS && !self.condition_holds(op.cond) {
                pc = pc.wrapping_add(4);
                continue;
            }
            self.pc = pc;
            if let Err(e) = self.exec_a32_op(op) {
                // The faulting instruction counts as retired.
                self.retire_run(block.start, i + 1);
                self.record_fault(&e, pc, block.words[i]);
                return Err(e);
            }
            pc = pc.wrapping_add(4);
        }
        self.retire_run(block.start, body);
        let mut ran = body as u64;
        match block.term {
            Some(term) if body == block.ops.len() && ran < budget => {
                self.pc = pc;
                self.record_run(pc, 1);
                let result = self.exec_a32_term(&term, pc);
                // After the terminator, faulted or not, as `step_inner` does:
                // an `SVC` reads the clock.
                self.retire();
                if let Err(e) = result {
                    let word = block.words.get(body).copied().unwrap_or(0);
                    self.record_fault(&e, pc, word);
                    return Err(e);
                }
                ran += 1;
            }
            _ => self.pc = pc,
        }
        Ok(ran)
    }

    fn retire_run(&mut self, start: u32, ran: usize) {
        self.cycles += ran as u64;
        self.steps += ran as u64;
        if ran != 0 {
            self.push_run(start, ran as u32);
        }
    }

    #[inline(always)]
    fn exec_a32_op(&mut self, op: &Op) -> Result<()> {
        match op.kind {
            Kind::Nop => {}
            Kind::Interpret { insn } => {
                self.jit.note_interpreted(insn);
                return self.execute_a32(insn);
            }
            Kind::Dp {
                opcode,
                set_flags,
                rd,
                rn,
                operand,
            } => {
                let carry_in = self.carry_flag();
                let (operand, shifter_carry) = match operand {
                    Operand::Imm(value) => (value, value >> 31 != 0),
                    Operand::ImmKeep(value) => (value, carry_in),
                    Operand::RegImm { rm, ty, amount } => {
                        shift_c(self.r32(rm), ty, u32::from(amount), carry_in)
                    }
                    Operand::RegReg { rm, ty, rs } => {
                        shift_by_register(self.r32(rm), ty, self.r32(rs), carry_in)
                    }
                };
                let a = self.r32(rn);
                let result = self.a32_alu(
                    u32::from(opcode),
                    set_flags,
                    a,
                    operand,
                    carry_in,
                    shifter_carry,
                );
                if let Some(result) = result {
                    self.set_r32(rd, result);
                }
            }
            Kind::Single {
                access,
                byte,
                offset,
            } => {
                let offset = self.a32_offset(offset);
                return self.a32_single(access, byte, offset);
            }
            Kind::Extra { access, sh, offset } => {
                let offset = self.a32_offset(offset);
                return self.a32_extra(access, sh, offset);
            }
            Kind::Block { access, list } => return self.a32_block_transfer(access, list),
        }
        Ok(())
    }

    #[inline(always)]
    fn a32_offset(&self, offset: Offset) -> u32 {
        match offset {
            Offset::Imm(imm) => u32::from(imm),
            Offset::Reg { rm, ty, amount } => {
                shift_c(self.r32(rm), ty, u32::from(amount), self.carry_flag()).0
            }
        }
    }

    fn exec_a32_term(&mut self, term: &Term, pc: u32) -> Result<()> {
        match term.kind {
            TermKind::Interpret { insn } => {
                self.jit.note_interpreted(insn);
                return self.execute_a32(insn);
            }
            TermKind::Fetch => {
                let insn = self.mem.fetch(pc)?;
                self.jit.note_interpreted(insn);
                return self.execute_a32(insn);
            }
            _ => {}
        }
        if term.cond != ALWAYS && !self.condition_holds(term.cond) {
            self.pc = pc.wrapping_add(4);
            return Ok(());
        }
        match term.kind {
            TermKind::B { target } => self.pc = target,
            TermKind::Bl { target } => {
                self.regs[14] = u64::from(pc.wrapping_add(4));
                self.pc = target;
            }
            TermKind::Bx { rm } => {
                let target = self.r32(rm);
                self.a32_write_pc(target)?;
            }
            TermKind::Svc { imm } => {
                // Retire the SVC first: a thread switch installs the next thread's PC.
                self.pc = pc.wrapping_add(4);
                self.syscall(imm)?;
            }
            TermKind::Interpret { .. } | TermKind::Fetch => unreachable!("handled above"),
        }
        Ok(())
    }
}
