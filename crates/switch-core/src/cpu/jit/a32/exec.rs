//! Running a block: its ops through the interpreter's helpers, then its terminator.

use super::decode::translate;
use super::{Block, Kind, Offset, Op, Operand, Term, TermKind, ALWAYS};
use crate::cpu::a32::shift::{shift_by_register, shift_c};
use crate::cpu::jit::emit::{emit_a32, COUNT_MASK, LEFT};
use crate::cpu::jit::exec::{HOT, MAX_MISSES};
use crate::cpu::jit::host;
use crate::cpu::jit::ir::Code;
use crate::cpu::{Cpu, ExecMode, Layout, Result, RunReport};
use std::rc::Rc;

/// What a visit to an emitted block did.
struct Entered {
    /// Instructions the block that answered retired.
    retired: usize,
    /// Whether it left through a branch, with the target in `pc`.
    left: bool,
    /// Instructions retired by the blocks jumped out of before it.
    chained: u64,
}

impl Cpu {
    /// [`Cpu::run`] over translated A32 blocks. The step budget is exact: a
    /// block that does not fit is left part-way through. Returns early if the
    /// running thread is no longer A32.
    pub(in crate::cpu) fn run_a32_jit(&mut self, max_steps: u64) -> Result<RunReport> {
        let mut steps = 0u64;
        let watches = self.mem.watchpoints_armed();
        if watches != self.jit.watches {
            self.jit.clear();
            self.jit.watches = watches;
        }
        while steps < max_steps && !self.halted {
            self.preemption_point();
            if self.handback.is_some() || self.mode != ExecMode::A32 || self.thumb {
                break;
            }
            let mut block = self.a32_block_at(self.pc);
            let code = block.code.get();
            let ran = match self.exec_a32_block(&mut block, max_steps - steps) {
                Ok(ran) => ran,
                Err(e) => {
                    self.jit.retired += steps;
                    return Err(e);
                }
            };
            self.jit.count_walked(code, ran);
            self.slice_used += ran;
            steps += ran;
        }
        self.jit.retired += steps;
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
            self.mem
                .mark_code_page(u64::from(page << crate::mem::PAGE_BITS));
        }
        self.jit.translated += 1;
        self.jit.executed += 1;
        self.jit.insert_a32(block.clone());
        block
    }

    /// Run at most `budget` of a block's instructions, returning how many
    /// retired. `self.pc` tracks the current instruction, so faults match the
    /// interpreter. A chain of compiled blocks leaves `block` on the last one.
    fn exec_a32_block(&mut self, block: &mut Rc<Block>, budget: u64) -> Result<u64> {
        let entered = self.enter_a32_emitted(block, budget);
        // Blocks the visit left through a branch before the one that answered.
        self.cycles += entered.chained;
        self.steps += entered.chained;
        let budget = budget - entered.chained;
        let block: &Block = block;
        if entered.left {
            // The branch set `pc`; its terminator, if it had one, retired with it.
            self.retire_run(block.start, entered.retired);
            return Ok(entered.chained + entered.retired as u64);
        }
        let first = entered.retired;
        let body = (block.ops.len() as u64).min(budget) as usize;
        let mut pc = block.start.wrapping_add(4 * first as u32);
        for (i, op) in block.ops[..body].iter().enumerate().skip(first) {
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
        Ok(entered.chained + ran)
    }

    /// Run `block`'s emitted form, and the compiled blocks it jumps into,
    /// leaving `block` on the one that answered. The op walk carries on from
    /// what that one retired. A visit that retires nothing counts against it.
    fn enter_a32_emitted(&mut self, block: &mut Rc<Block>, budget: u64) -> Entered {
        const NOTHING: Entered = Entered {
            retired: 0,
            left: false,
            chained: 0,
        };
        let Code::Ready { entry, misses } = block.code.get() else {
            self.warm_a32(block);
            return NOTHING;
        };
        if block.emitted.get() as u64 > budget {
            return NOTHING;
        }
        self.jit.entered_emitted += 1;
        let fuel = self.chain_fuel(budget);
        self.jit.chain.fuel = fuel;
        self.jit.chain.origin = self.cycles.wrapping_add_signed(i64::from(fuel));
        self.jit.chain.hops = 0;
        let state = self as *mut Cpu;
        // SAFETY: `entry` came from `host::install` for this block and is only
        // released after it leaves the chain table; it was emitted against
        // `Layout::of_cpu`, as were the blocks it can jump to.
        let answer = unsafe { host::enter(entry, state) };
        let entered = Entered {
            retired: (answer & COUNT_MASK) as usize,
            left: answer & LEFT != 0,
            chained: (fuel - self.jit.chain.fuel) as u64,
        };
        self.jit.retired_emitted += entered.chained + entered.retired as u64;
        let hops = u64::from(self.jit.chain.hops);
        if hops != 0 {
            self.jit.chained += hops;
            self.jit.executed += hops;
            self.jit.entered_emitted += hops;
            *block = self
                .jit
                .a32
                .get(self.jit.chain.last)
                .expect("the chain table holds only cached blocks");
            if let Code::Ready { entry, misses } = block.code.get() {
                if misses != 0 {
                    block.code.set(Code::Ready { entry, misses: 0 });
                }
            }
            return entered;
        }
        if entered.retired == 0 && !entered.left {
            if misses + 1 >= MAX_MISSES {
                self.jit.unchain_a32(block);
                block.drop_code();
                self.jit.dropped += 1;
            } else {
                block.code.set(Code::Ready {
                    entry,
                    misses: misses + 1,
                });
            }
        } else if misses != 0 {
            block.code.set(Code::Ready { entry, misses: 0 });
        }
        entered
    }

    /// Count a visit to an unemitted block and emit it after [`HOT`] visits.
    fn warm_a32(&mut self, block: &Block) {
        let Code::Cold(seen) = block.code.get() else {
            return;
        };
        if seen + 1 < HOT {
            block.code.set(Code::Cold(seen + 1));
            return;
        }
        block.code.set(Code::Never);
        if !host::available() {
            return;
        }
        let chain = host::chains();
        let Ok(emitted) = emit_a32(block, Layout::of_cpu(), chain, self.jit.watches) else {
            return;
        };
        let entry = host::install(&emitted.module);
        if entry != 0 {
            block.emitted.set(emitted.retires);
            block.code.set(Code::Ready { entry, misses: 0 });
            if chain {
                self.jit.chain_to_a32(block.start, entry as u32);
            }
            self.jit.emitted += 1;
        }
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
            Kind::MoveWide { rd, imm, top } => {
                let value = if top {
                    (self.r32(rd) & 0xFFFF) | (u32::from(imm) << 16)
                } else {
                    u32::from(imm)
                };
                self.set_r32(rd, value);
            }
            Kind::Extend {
                rd,
                rn,
                rm,
                rotate,
                signed,
                halfword,
            } => {
                let value = self.r32(rm).rotate_right(u32::from(rotate));
                let extended = match (signed, halfword) {
                    (true, false) => value as u8 as i8 as i32 as u32,
                    (true, true) => value as u16 as i16 as i32 as u32,
                    (false, false) => value & 0xFF,
                    (false, true) => value & 0xFFFF,
                };
                let result = if rn == 15 {
                    extended
                } else {
                    self.r32(rn).wrapping_add(extended)
                };
                self.set_r32(rd, result);
            }
            Kind::Extend16 {
                rd,
                rn,
                rm,
                rotate,
                signed,
            } => {
                let value = self.r32(rm).rotate_right(u32::from(rotate));
                let lane = |byte: u32| -> u16 {
                    if signed {
                        byte as u8 as i8 as i16 as u16
                    } else {
                        byte as u8 as u16
                    }
                };
                let (mut low, mut high) = (lane(value), lane(value >> 16));
                if rn != 15 {
                    let addend = self.r32(rn);
                    low = low.wrapping_add(addend as u16);
                    high = high.wrapping_add((addend >> 16) as u16);
                }
                self.set_r32(rd, u32::from(low) | u32::from(high) << 16);
            }
            Kind::BitfieldInsert { rd, rn, lsb, mask } => {
                let source = if rn == 15 { 0 } else { self.r32(rn) << lsb };
                self.set_r32(rd, (self.r32(rd) & !mask) | (source & mask));
            }
            Kind::Multiply {
                op,
                set_flags,
                rd_hi,
                rd_lo,
                rs,
                rm,
            } => {
                let (rm, rs) = (self.r32(rm), self.r32(rs));
                if matches!(op, 0b000 | 0b001 | 0b011) {
                    let product = rm.wrapping_mul(rs);
                    let result = match op {
                        0b000 => product,
                        0b001 => product.wrapping_add(self.r32(rd_lo)),
                        _ => self.r32(rd_lo).wrapping_sub(product),
                    };
                    self.set_r32(rd_hi, result);
                    if set_flags {
                        self.set_nz32(result);
                    }
                } else {
                    let product = if op & 0b010 != 0 {
                        (i64::from(rm as i32) * i64::from(rs as i32)) as u64
                    } else {
                        u64::from(rm) * u64::from(rs)
                    };
                    let product = if op & 0b001 != 0 {
                        let acc = u64::from(self.r32(rd_hi)) << 32 | u64::from(self.r32(rd_lo));
                        product.wrapping_add(acc)
                    } else {
                        product
                    };
                    self.set_r32(rd_lo, product as u32);
                    self.set_r32(rd_hi, (product >> 32) as u32);
                    if set_flags {
                        let n = u32::from(product >> 63 != 0) << 31;
                        let z = u32::from(product == 0) << 30;
                        self.nzcv = (self.nzcv & 0x3000_0000) | n | z;
                    }
                }
            }
            Kind::ReadThreadPointer { rt, rw } => {
                let value = if rw { self.tpidr_rw } else { self.tpidr };
                self.set_r32(rt, value as u32);
            }
            Kind::Exclusive {
                load,
                size,
                rd,
                rn,
                rt,
            } => {
                let addr = self.r32(rn);
                if load {
                    let value = match size {
                        0 => u32::from(self.mem.read_u8(u64::from(addr))?),
                        1 => u32::from(self.mem.read_u16(u64::from(addr))?),
                        _ => self.mem.read_u32(u64::from(addr))?,
                    };
                    self.set_r32(rd, value);
                    self.exclusive.arm(u64::from(addr), [u64::from(value), 0]);
                } else {
                    let value = u64::from(self.r32(rt));
                    let stored = self.store_exclusive_at(u64::from(addr), 1 << size, [value, 0])?;
                    self.set_r32(rd, u32::from(!stored));
                }
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
            TermKind::Blx { rm } => {
                let target = self.r32(rm);
                self.regs[14] = u64::from(pc.wrapping_add(4));
                self.a32_write_pc(target)?;
            }
            TermKind::LdrPc { access, offset } => {
                let offset = self.a32_offset(offset);
                return self.a32_single(access, false, offset);
            }
            TermKind::LdmPc { access, list } => return self.a32_block_transfer(access, list),
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
