//! Running a block: its ops, its conditional exits, and its terminator. Each
//! arm does what the interpreter would, mostly through the same helpers.

use super::cache::JitStats;
use super::decode::translate;
use super::emit::{
    emit_region, term_retires, writes_term, Refused, COUNT_MASK, LEFT, MAX_WHOLE, MEMBER_SHIFT,
    PLT_STUB,
};
use super::host;
use super::ir::{Block, Code, Exit, Op, PackedImm, Term};
use crate::cpu::bits::*;
use crate::cpu::loadstore::{Acc, PairKind, Wb};
use crate::cpu::{Cpu, Layout, Result, RunReport, SELF_RETURN_TRAMPOLINE, TIME_SLICE};
use std::rc::Rc;
use std::sync::atomic::{fence, Ordering::SeqCst};

/// Entries before a block is emitted as wasm.
pub const HOT: u32 = 512;

/// Hand-backs at a block's first instruction before its emitted form is dropped.
const MAX_MISSES: u16 = 8;

/// The most blocks a region compiles together.
const MAX_REGION_BLOCKS: usize = 32;

/// Entries before a block is worth compiling into another's region.
const REGION_WARM: u32 = 16;

#[derive(Clone, Copy)]
enum Taken {
    No,
    /// Taken, out of the block; `self.pc` holds the target.
    Leave,
    /// A followed branch: the block's ops go on at the target.
    Follow(u32),
}

#[derive(Clone, Copy)]
struct Emitted {
    /// Of the block that answered.
    retired: usize,
    /// Whether it left through a taken branch, with the target in `pc`.
    left: bool,
    /// Jumps between compiled regions before the answer.
    hops: u32,
    /// Instructions retired by blocks left through a branch before the answer.
    chained: u64,
    /// Which block of the answering region answered; 0 is its entry block.
    member: usize,
}

/// Where the straight-line stretch being executed starts, in the block body
/// and in guest memory, so any op's address can be derived.
#[derive(Clone, Copy)]
struct Here {
    first: usize,
    pc: u32,
}

impl Here {
    #[inline(always)]
    fn new(ops: &[Op], pc: u32) -> Here {
        Here {
            first: ops.as_ptr() as usize,
            pc,
        }
    }

    #[inline(always)]
    fn pc_of(self, op: &Op) -> u32 {
        let index = (op as *const Op as usize - self.first) / std::mem::size_of::<Op>();
        self.pc.wrapping_add(4 * index as u32)
    }
}

/// The operand that makes an addition a subtraction; `carry` doubles as the
/// inversion mask.
#[inline(always)]
fn invert_if(v: u64, carry: u8) -> u64 {
    v ^ 0u64.wrapping_sub(u64::from(carry))
}

/// What an entered emitted block can retire, terminator included.
#[inline(always)]
fn whole(block: &Block) -> usize {
    block.ops.len()
        + block
            .term
            .as_ref()
            .filter(|term| writes_term(term))
            .map_or(0, term_retires)
}

impl Cpu {
    pub fn jit_enabled(&self) -> bool {
        self.jit_enabled
    }

    /// Turning the translator off drops the cache.
    pub fn set_jit_enabled(&mut self, on: bool) {
        if !on {
            self.jit_flush();
        }
        self.jit_enabled = on;
    }

    pub fn jit_stats(&self) -> JitStats {
        self.jit.stats()
    }

    /// Drop every translated block, for callers that rewrite guest code
    /// behind [`crate::mem::Memory`]'s back.
    pub fn jit_flush(&mut self) {
        self.jit.clear();
        self.parked_jits.clear();
    }

    /// [`Cpu::run`] over translated blocks. The step budget is exact: a block
    /// that does not fit is left part-way through.
    pub(in crate::cpu) fn run_jit(&mut self, max_steps: u64) -> Result<RunReport> {
        let mut steps = 0u64;
        // The last block, moved out and back so a self-loop skips lookup and refcount.
        let mut held: Option<Rc<Block>> = None;
        // The process `held` belongs to: another's code is not linked to.
        let mut held_pid = self.pid;
        // Block entries counted in locals and committed once, on the way out.
        let mut executed = 0u64;
        let mut linked = 0u64;
        let watches = self.mem.watchpoints_armed();
        if watches != self.jit.watches {
            self.jit.clear();
            self.jit.watches = watches;
        }
        while steps < max_steps && !self.halted {
            // Between blocks.
            self.preemption_point();
            if self.handback.is_some() {
                break;
            }
            if self.pid != held_pid {
                held = None;
                held_pid = self.pid;
            }
            let pc = self.pc;
            // A store to translated code disables the fast paths until
            // `jit_block_at` drains the dirty list.
            let stale = self.mem.has_dirty_code(self.core);
            let block = match held.take() {
                Some(block) if block.start == pc && !stale => block,
                // Where this block went last time.
                Some(previous) => match previous.successor(pc).filter(|_| !stale) {
                    Some(next) => {
                        linked += 1;
                        next
                    }
                    None => {
                        let next = self.jit_block_at(pc);
                        previous.link_to(pc, &next);
                        next
                    }
                },
                None => self.jit_block_at(pc),
            };
            executed += 1;
            // Straight on to a linked successor while nothing it checks has changed.
            let mut block = block;
            loop {
                let ran = match self.exec_block(&mut block, max_steps - steps) {
                    Ok(ran) => ran,
                    Err(e) => {
                        self.jit.executed += executed;
                        self.jit.linked += linked;
                        return Err(e);
                    }
                };
                self.slice_used += ran;
                steps += ran;
                let next = if ran == 0
                    || steps >= max_steps
                    || self.halted
                    || self.handback.is_some()
                    || self.slice_used >= TIME_SLICE
                    || self.mem.has_dirty_code(self.core)
                    || self.pid != held_pid
                {
                    None
                } else {
                    block.successor(self.pc)
                };
                let Some(next) = next else {
                    held = Some(block);
                    break;
                };
                self.sweep_timed_waits();
                executed += 1;
                linked += 1;
                block = next;
            }
        }
        self.jit.executed += executed;
        self.jit.linked += linked;
        Ok(RunReport {
            steps,
            halted: self.halted,
        })
    }

    /// The block entered at `pc`, translating it on first visit. Drains dirty
    /// code pages first so a stale block is never handed out.
    fn jit_block_at(&mut self, pc: u32) -> Rc<Block> {
        if self.mem.has_dirty_code(self.core) {
            match self.mem.dirty_code_pages(self.core) {
                Some(dirty) => self.jit.invalidate(&dirty),
                None => self.jit.clear(),
            }
        }
        if let Some(block) = self.jit.get(pc) {
            return block;
        }
        let block = Rc::new(translate(&self.mem, pc));
        for &page in &block.pages {
            self.mem.mark_code_page(page << crate::mem::PAGE_BITS);
        }
        self.jit.translated += 1;
        self.jit.insert(block.clone());
        block
    }

    /// Run at most `budget` of a block's instructions, returning how many
    /// retired. `self.pc` tracks the current instruction, so faults match the
    /// interpreter. A chain of compiled blocks leaves `block` on the last one.
    #[inline(always)]
    fn exec_block(&mut self, block: &mut Rc<Block>, budget: u64) -> Result<u64> {
        // The emitted form replaces only the op walk; the terminator and accounting are shared.
        if let Some(done) = self.enter_emitted(block, budget) {
            if done.hops != 0 || done.chained != 0 {
                return self.finish_chain(block, done, budget);
            }
            return self.finish_emitted(block, done.retired, done.left, budget);
        }
        let block: &Block = block;
        let body = (block.ops.len() as u64).min(budget) as usize;
        let mut i = 0usize;
        let mut pc = block.start;
        let mut next_exit = 0usize;
        // Start of the current straight-line run, by address and index.
        let mut run_pc = block.start;
        let mut run_i = 0usize;
        loop {
            // Run to the next conditional branch or the end of the budget.
            let stop = match block.exits.get(next_exit) {
                Some(branch) if (branch.at as usize) < body => branch.at as usize,
                _ => body,
            };
            let segment = &block.ops[i..stop];
            let here = Here::new(segment, pc);
            for op in segment {
                if let Err(e) = self.exec_op(op, here) {
                    // Clock, steps, trail and `pc` are settled only on a fault;
                    // the faulting instruction counts.
                    let pc = here.pc_of(op);
                    let at = run_i + (pc.wrapping_sub(run_pc) / 4) as usize;
                    self.retire_runs(run_pc, run_i, at + 1);
                    self.pc = pc;
                    self.record_fault(&e, pc, block.words[at]);
                    return Err(e);
                }
            }
            pc = pc.wrapping_add(4 * (stop - i) as u32);
            i = stop;
            if stop == body {
                break;
            }
            let branch = &block.exits[next_exit];
            let exit = &branch.exit;
            let span = branch.span as usize;
            if i + span > body {
                // The budget splits a fused exit: run what fits and stop, so
                // progress is never zero.
                if i < body {
                    let fit = body - i;
                    self.apply_compare(exit, fit);
                    i += fit;
                    pc = pc.wrapping_add(4 * fit as u32);
                }
                break;
            }
            i += span;
            pc = pc.wrapping_add(4 * span as u32);
            match self.take_exit(exit) {
                Taken::No => {}
                Taken::Leave => {
                    // `take_exit` has already put the target in `pc`.
                    self.retire_runs(run_pc, run_i, i);
                    return Ok(i as u64);
                }
                Taken::Follow(target) => {
                    // Leave as a terminator would, so `run_jit` notices a store
                    // to the translated code past here.
                    if self.mem.has_dirty_code(self.core) {
                        self.pc = target;
                        self.retire_runs(run_pc, run_i, i);
                        return Ok(i as u64);
                    }
                    // Record the ending run in the trail while its start is known.
                    self.push_run(run_pc, (i - run_i) as u32);
                    pc = target;
                    run_pc = target;
                    run_i = i;
                }
            }
            next_exit += 1;
        }
        self.finish(block, i, pc, (run_pc, run_i), budget)
    }

    /// Account for the first `i` instructions an emitted block retired, and run
    /// its terminator if it did not leave.
    #[inline(always)]
    fn finish_emitted(&mut self, block: &Block, i: usize, left: bool, budget: u64) -> Result<u64> {
        // A folded PLT call: its stub is a run of its own.
        if let Some(Term::BlPlt { stub, .. } | Term::BPlt { stub, .. }) = block.term {
            if left && i > block.ops.len() + 1 {
                let call = block.ops.len() + 1;
                let (_, run_pc, run_i) = self.follow_runs(block, call);
                self.retire_runs(run_pc, run_i, call);
                self.record_run(stub, PLT_STUB);
                let stub = u64::from(PLT_STUB);
                self.cycles += stub;
                self.steps += stub;
                return Ok(call as u64 + stub);
            }
        }
        let (pc, run_pc, run_i) = self.follow_runs(block, i);
        if left {
            // The branch set `pc`; the terminator does not run.
            self.retire_runs(run_pc, run_i, i);
            return Ok(i as u64);
        }
        self.finish(block, i, pc, (run_pc, run_i), budget)
    }

    /// Settle a visit that went on past the entered block, within its region
    /// or by jumps into others: the left blocks' clock and steps, then the
    /// block that answered.
    #[inline(never)]
    fn finish_chain(&mut self, block: &mut Rc<Block>, done: Emitted, budget: u64) -> Result<u64> {
        let chained = done.chained;
        self.cycles += chained;
        self.steps += chained;
        let hops = u64::from(done.hops);
        self.jit.chained += hops;
        self.jit.executed += hops;
        self.jit.entered_emitted += hops;
        if hops != 0 {
            *block = self
                .jit
                .get(self.jit.chain.last)
                .expect("the chain table holds only cached blocks");
            if let Code::Ready { entry, misses } = block.code.get() {
                if misses != 0 {
                    block.code.set(Code::Ready { entry, misses: 0 });
                }
            }
        }
        if done.member != 0 {
            let member = block.members.borrow()[done.member - 1].clone();
            *block = member;
        }
        // Stopped at a block's first instruction: the next visit runs it.
        if done.retired == 0 && !done.left {
            self.pc = block.start;
            return Ok(chained);
        }
        let retired = done.retired.min(whole(block));
        let ran = self.finish_emitted(block, retired, done.left, budget - chained)?;
        Ok(chained + ran)
    }

    /// Retire the first `i` instructions, whose last run starts at `run`, and
    /// run the terminator at `pc` if the block got that far.
    #[inline(always)]
    fn finish(
        &mut self,
        block: &Block,
        i: usize,
        pc: u32,
        (run_pc, run_i): (u32, usize),
        budget: u64,
    ) -> Result<u64> {
        self.retire_runs(run_pc, run_i, i);
        let mut ran = i as u64;
        match block.term {
            Some(ref term) if i == block.ops.len() && ran < budget => {
                self.pc = pc;
                self.record_run(pc, 1);
                let result = self.exec_term(term, pc, budget - ran);
                // After the terminator, faulted or not, as `step_inner` does:
                // an `SVC` reads the clock.
                self.retire();
                match result {
                    Ok(folded) => {
                        // Instructions of a folded PLT stub.
                        self.cycles += folded;
                        self.steps += folded;
                        ran += 1 + folded;
                    }
                    Err(e) => {
                        self.record_fault(&e, pc, block.words[i]);
                        return Err(e);
                    }
                }
            }
            // Budget ran out, or the block has no terminator; `pc` is next.
            _ => self.pc = pc,
        }
        Ok(ran)
    }

    /// Run `block`'s emitted form, returning how many leading instructions it
    /// retired, or `None` to hand this visit to the interpreter. Emitted code
    /// hands back before writing anything, and a visit that retires nothing
    /// is handed back so `run_jit` cannot spin.
    #[inline(always)]
    fn enter_emitted(&mut self, block: &Block, budget: u64) -> Option<Emitted> {
        let Code::Ready { entry, misses } = block.code.get() else {
            self.warm(block);
            return None;
        };
        // Only enter a block that fits the remaining budget.
        let whole = whole(block);
        if whole as u64 > budget {
            return None;
        }
        self.jit.entered_emitted += 1;
        let fuel = self.chain_fuel(budget);
        self.jit.chain.fuel = fuel;
        self.jit.chain.hops = 0;
        let state = self as *mut Cpu;
        // SAFETY: `entry` came from `host::install` for this block and is only
        // released by `Block::drop_code`; it was emitted against `Layout::of_cpu`.
        // So were the blocks it can jump to, which leave the chain table first.
        let answer = unsafe { host::enter(entry, state) };
        let left = answer & LEFT != 0;
        let retired = (answer & COUNT_MASK) as usize;
        let member = ((answer & !LEFT) >> MEMBER_SHIFT) as usize;
        let hops = self.jit.chain.hops;
        let chained = (fuel - self.jit.chain.fuel) as u64;
        if hops != 0 || chained != 0 {
            if misses != 0 {
                block.code.set(Code::Ready { entry, misses: 0 });
            }
            return Some(Emitted {
                retired,
                left,
                hops,
                chained,
                member,
            });
        }
        if retired == 0 && !left {
            let misses = misses + 1;
            if misses >= MAX_MISSES {
                self.jit.unchain(block);
                block.drop_code();
            } else {
                block.code.set(Code::Ready { entry, misses });
            }
            return None;
        }
        // Any progress resets the miss count.
        if misses != 0 {
            block.code.set(Code::Ready { entry, misses: 0 });
        }
        Some(Emitted {
            retired: retired.min(whole),
            left,
            hops: 0,
            chained: 0,
            member: 0,
        })
    }

    /// How many instructions compiled blocks may retire before a jump has to
    /// come back here: where `run_jit` would stop following blocks (budget,
    /// time slice, timed-wait sweep), less room for the next block to fit.
    #[inline(always)]
    fn chain_fuel(&self, budget: u64) -> i32 {
        let room = budget.min(i32::MAX as u64) as i64 - MAX_WHOLE;
        let slice = TIME_SLICE as i64 - 1 - self.slice_used as i64;
        let sweep = self
            .next_expiry
            .saturating_sub(self.cycles)
            .min(i32::MAX as u64) as i64
            - 1;
        room.min(slice).min(sweep).max(-1) as i32
    }

    /// The address of `block`'s `i`th instruction across followed `B`s, with
    /// the start of its run. Earlier runs go into the trail.
    #[inline(always)]
    fn follow_runs(&mut self, block: &Block, i: usize) -> (u32, u32, usize) {
        if !block.follows {
            return (block.start.wrapping_add(4 * i as u32), block.start, 0);
        }
        let (mut run_pc, mut run_i) = (block.start, 0usize);
        for branch in &block.exits {
            let Exit::Jump { target } = branch.exit else {
                continue;
            };
            let after = branch.at as usize + branch.span as usize;
            if after > i {
                break;
            }
            self.push_run(run_pc, (after - run_i) as u32);
            (run_pc, run_i) = (target, after);
        }
        (run_pc.wrapping_add(4 * (i - run_i) as u32), run_pc, run_i)
    }

    /// Count a visit to an unemitted block and emit it after [`HOT`] visits.
    #[inline(always)]
    fn warm(&mut self, block: &Block) {
        if let Code::Cold(seen) = block.code.get() {
            let seen = seen + 1;
            if seen >= HOT {
                self.install(block);
            } else {
                block.code.set(Code::Cold(seen));
            }
        }
    }

    /// Emit `block` as wasm, with the warm blocks it branches to as its
    /// region, and install it. Whatever the outcome, the block never asks again.
    #[cold]
    #[inline(never)]
    fn install(&mut self, block: &Block) {
        if !host::available() {
            block.code.set(Code::Never);
            return;
        }
        let chains = host::chains();
        let mut members = self.region_members(block);
        let code = loop {
            let blocks: Vec<&Block> = std::iter::once(block)
                .chain(members.iter().map(|m| &**m))
                .collect();
            let watches = self.jit.watches;
            match emit_region(
                &blocks,
                Layout::of_cpu(),
                chains,
                true,
                host::simd(),
                watches,
            ) {
                Ok(code) => break Some(code),
                Err((0, _)) => break None,
                Err((j, Refused::TooLong)) => members.truncate(j - 1),
                Err((j, _)) => {
                    let refused = members.remove(j - 1);
                    if let Code::Cold(_) = refused.code.get() {
                        refused.code.set(Code::Never);
                    }
                }
            }
        };
        let entry = code.map_or(0, |code| host::install(&code));
        if entry == 0 {
            block.code.set(Code::Never);
            return;
        }
        self.jit.emitted += 1;
        block.code.set(Code::Ready { entry, misses: 0 });
        *block.members.borrow_mut() = members;
        // A block no longer cached may still be running; nothing may jump to it.
        if self.jit.holds(block) {
            self.jit.cover(block);
            if chains {
                self.jit.chain_to(block, entry as u32);
            }
        }
    }

    /// The cached blocks reachable from `head` by known branches that have
    /// run often enough to be worth compiling with it, nearest first.
    fn region_members(&self, head: &Block) -> Vec<Rc<Block>> {
        let mut members: Vec<Rc<Block>> = Vec::new();
        // Pending stores could have left a cached block stale.
        if self.mem.has_dirty_code(self.core) {
            return members;
        }
        let mut seen = vec![head.start];
        let mut queue: std::collections::VecDeque<u32> = head.region_successors().into();
        while let Some(pc) = queue.pop_front() {
            if members.len() + 1 >= MAX_REGION_BLOCKS {
                break;
            }
            if seen.contains(&pc) {
                continue;
            }
            seen.push(pc);
            let Some(next) = self.jit.peek(pc) else {
                continue;
            };
            let warm = match next.code.get() {
                Code::Cold(seen) => seen >= REGION_WARM,
                Code::Ready { .. } => true,
                Code::Never => false,
            };
            if warm {
                queue.extend(next.region_successors());
                members.push(next);
            }
        }
        members
    }

    /// Account for `ran` retired instructions: clock and steps for all, trail
    /// for those in the run starting at `run_i`/`run_pc`.
    #[inline(always)]
    fn retire_runs(&mut self, run_pc: u32, run_i: usize, ran: usize) {
        self.cycles += ran as u64;
        self.steps += ran as u64;
        if ran > run_i {
            self.push_run(run_pc, (ran - run_i) as u32);
        }
    }

    /// The first `fit` instructions of a fused exit, when the budget ends inside it.
    #[inline(always)]
    fn apply_compare(&mut self, exit: &Exit, fit: usize) {
        let (a, b, carry, sf) = match *exit {
            Exit::UpdateCmpImm {
                rd,
                source,
                step,
                rn,
                imm,
                ..
            } => {
                self.apply_step(rd, source, step);
                if fit == 1 {
                    return;
                }
                let sf = imm.sf();
                let carry = imm.carry();
                (
                    self.reg_at(rn) & Cpu::mask(sf),
                    invert_if(imm.imm(), carry),
                    carry,
                    sf,
                )
            }
            Exit::CmpImm {
                rn, imm, carry, sf, ..
            } => (
                self.reg_at(rn) & Cpu::mask(sf),
                invert_if(u64::from(imm), carry),
                carry,
                sf,
            ),
            Exit::CmpReg {
                rn, rm, carry, sf, ..
            } => (
                self.reg_at(rn) & Cpu::mask(sf),
                invert_if(self.reg_at(rm) & Cpu::mask(sf), carry),
                carry,
                sf,
            ),
            _ => return,
        };
        let (result, c, v) = Cpu::add_carry_overflow(a, b, u64::from(carry), sf);
        self.set_nzcv_from_alu(result, sf, c, v);
    }

    /// The flagless `ADD`/`SUB` immediate [`Exit::UpdateCmpImm`] folds in.
    #[inline(always)]
    fn apply_step(&mut self, rd: u8, source: u8, step: PackedImm) {
        let carry = step.carry();
        let rhs = invert_if(step.imm(), carry);
        self.add_sub_pre(rd, source, rhs, carry, false, step.sf());
    }

    /// Evaluate a branch inside a block: fall through, leave with `self.pc` on
    /// the target, or continue at a followed target.
    #[inline(always)]
    fn take_exit(&mut self, exit: &Exit) -> Taken {
        let (taken, target) = match *exit {
            Exit::Cond { cond, target } => (self.condition_holds(cond), target),
            Exit::CmpImm {
                rn,
                imm,
                carry,
                sf,
                cond,
                target,
            } => {
                let a = self.reg_at(rn) & Cpu::mask(sf);
                let b = invert_if(u64::from(imm), carry);
                let (result, c, v) = Cpu::add_carry_overflow(a, b, u64::from(carry), sf);
                self.set_nzcv_from_alu(result, sf, c, v);
                (self.condition_holds(cond), target)
            }
            Exit::UpdateCmpImm { cond, target, .. } => {
                self.apply_compare(exit, 2);
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
            Exit::Jump { target } => return Taken::Follow(target),
        };
        if taken {
            self.pc = target;
            Taken::Leave
        } else {
            Taken::No
        }
    }

    /// Execute one body op. Takes `&Op` on purpose: by value, the compiler
    /// hoists every arm's field loads above the jump table.
    #[inline(always)]
    fn exec_op(&mut self, op: &Op, here: Here) -> Result<()> {
        match *op {
            Op::Nop => {}
            // Only arms that re-enter the interpreter need `pc` set.
            Op::Interpret { insn } => {
                let pc = here.pc_of(op);
                self.pc = pc;
                self.jit.note_interpreted(insn);
                self.execute(insn, pc.wrapping_add(4))?;
            }
            Op::Fp { insn, scalar, form } => {
                let pc = here.pc_of(op);
                self.pc = pc;
                let next = pc.wrapping_add(4);
                #[allow(clippy::if_same_then_else)] // the order is the point
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
                let pc = here.pc_of(op);
                self.pc = pc;
                self.system(insn, pc.wrapping_add(4))?;
            }
            Op::Sys { op } => self.exec_sys(op)?,
            Op::SimdLoadStore { insn } => {
                let pc = here.pc_of(op);
                self.pc = pc;
                if self.try_simd_load_store(insn)? {
                    self.pc = pc.wrapping_add(4);
                } else {
                    self.execute(insn, pc.wrapping_add(4))?;
                }
            }

            Op::MovConst { rd, val } => self.set_reg_at(rd, val),
            Op::Mov32 { rd, rn } => self.copy_reg(rd, rn, false),
            Op::Mov64 { rd, rn } => self.copy_reg(rd, rn, true),
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
                // Rd == 31 is SP here and XZR for ANDS; the translator settled it in `rd`.
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
                // `BIC`/`ORN`/`EON` invert the shifted operand.
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

            Op::Extract {
                rd,
                rn,
                extract,
                sf,
            } => self.extract(rd, rn, extract, sf),
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
            Op::ShiftVar {
                rd,
                rn,
                rm,
                kind,
                sf,
            } => self.shift_by_reg(rd, rn, rm, kind, sf),
            Op::Divide {
                rd,
                rn,
                rm,
                signed,
                sf,
            } => self.divide(rd, rn, rm, signed, sf),
            Op::Adc {
                rd,
                rn,
                rm,
                sub,
                set_flags,
                sf,
            } => self.adc(rd, rn, rm, sub, set_flags, sf),
            Op::OneSource { rd, rn, opcode, sf } => self.one_source(rd, rn, opcode, sf),
            Op::Crc {
                rd,
                rn,
                rm,
                sz,
                castagnoli,
            } => self.crc(rd, rn, rm, sz, castagnoli),

            Op::LoadStoreImm {
                rt,
                rn,
                acc,
                wb,
                offset,
            } => self.load_store_fast(op, rt, rn, acc, wb, offset)?,
            Op::Load64 { rt, rn, wb, offset } => {
                self.load_store_fast(op, rt, rn, Acc::Load64, wb, offset)?
            }
            Op::Store64 { rt, rn, wb, offset } => {
                self.load_store_fast(op, rt, rn, Acc::Store64, wb, offset)?
            }
            Op::Load32 { rt, rn, wb, offset } => {
                self.load_store_fast(op, rt, rn, Acc::Load32, wb, offset)?
            }
            Op::Store32 { rt, rn, wb, offset } => {
                self.load_store_fast(op, rt, rn, Acc::Store32, wb, offset)?
            }
            Op::Load8 { rt, rn, wb, offset } => {
                self.load_store_fast(op, rt, rn, Acc::Load8, wb, offset)?
            }
            Op::Store8 { rt, rn, wb, offset } => {
                self.load_store_fast(op, rt, rn, Acc::Store8, wb, offset)?
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
                if !self.access_fast(addr, rt, acc) {
                    return self.exec_op_slow(op);
                }
            }
            Op::Pair {
                rt,
                rt2,
                rn,
                offset,
                kind,
                wb,
            } => self.pair(rt, rt2, rn, offset, kind, wb)?,
            Op::PairLoad64 {
                rt,
                rt2,
                rn,
                offset,
                wb,
            } => {
                let (addr, wb_val) = Self::indexed(self.reg_at(rn), offset, wb);
                let Some(bytes) = self.mem.peek::<16>(addr as u32) else {
                    return self.exec_op_slow(op);
                };
                let (first, second) = bytes.split_at(8);
                self.set_reg_at(rt, u64::from_le_bytes(first.try_into().unwrap()));
                self.set_reg_at(rt2, u64::from_le_bytes(second.try_into().unwrap()));
                if let Some(v) = wb_val {
                    self.set_reg_at(rn, v);
                }
            }
            Op::PairStore64 {
                rt,
                rt2,
                rn,
                offset,
                wb,
            } => {
                let (addr, wb_val) = Self::indexed(self.reg_at(rn), offset, wb);
                let first = self.reg_at(rt).to_le_bytes();
                let second = self.reg_at(rt2).to_le_bytes();
                if !self.mem.poke_pair(addr as u32, first, second) {
                    return self.exec_op_slow(op);
                }
                if let Some(v) = wb_val {
                    self.set_reg_at(rn, v);
                }
            }
            Op::LoadLiteral { rt, addr, acc } => {
                if !self.access_fast(addr, rt, acc) {
                    return self.exec_op_slow(op);
                }
            }
            Op::LoadExclusive { rt, rn, sz } => self.load_exclusive(rt, rn, sz)?,
            Op::StoreExclusive { rs, rt, rn, sz } => self.store_exclusive(rs, rt, rn, sz)?,
            Op::Ordered { rt, rn, acc } => {
                fence(SeqCst);
                self.access(self.reg_at(rn) as u32, rt, acc)?;
                fence(SeqCst);
            }
        }
        Ok(())
    }

    /// [`Cpu::load_store_imm`] when the page table suffices, else [`Cpu::exec_op_slow`]
    /// reruns `op`. The fast path makes no calls and changes nothing before committing.
    #[inline(always)]
    fn load_store_fast(
        &mut self,
        op: &Op,
        rt: u8,
        rn: u8,
        acc: Acc,
        wb: Wb,
        offset: i64,
    ) -> Result<()> {
        let (addr, wb_val) = Self::indexed(self.reg_at(rn), offset, wb);
        if !self.access_fast(addr as u32, rt, acc) {
            return self.exec_op_slow(op);
        }
        if let Some(v) = wb_val {
            self.set_reg_at(rn, v);
        }
        Ok(())
    }

    /// [`Cpu::access`] via `peek`/`poke`; declining leaves all state unchanged.
    #[inline(always)]
    fn access_fast(&mut self, addr: u32, rt: u8, acc: Acc) -> bool {
        let mem = &self.mem;
        let loaded = match acc {
            Acc::Load8 => mem.peek(addr).map(|[b]: [u8; 1]| u64::from(b)),
            Acc::Load16 => mem.peek(addr).map(|b| u64::from(u16::from_le_bytes(b))),
            Acc::Load32 => mem.peek(addr).map(|b| u64::from(u32::from_le_bytes(b))),
            Acc::Load64 => mem.peek(addr).map(u64::from_le_bytes),
            Acc::LoadS8 => mem.peek(addr).map(|[b]: [u8; 1]| sext_u64(u64::from(b), 8)),
            Acc::LoadS16 => mem
                .peek(addr)
                .map(|b| sext_u64(u64::from(u16::from_le_bytes(b)), 16)),
            Acc::LoadS8To32 => mem
                .peek(addr)
                .map(|[b]: [u8; 1]| u64::from(sext_u64(u64::from(b), 8) as u32)),
            Acc::LoadS16To32 => mem
                .peek(addr)
                .map(|b| u64::from(sext_u64(u64::from(u16::from_le_bytes(b)), 16) as u32)),
            Acc::LoadS32 => mem
                .peek(addr)
                .map(|b| sext_u64(u64::from(u32::from_le_bytes(b)), 32)),
            Acc::Store8 => return self.mem.poke(addr, [self.reg_at(rt) as u8]),
            Acc::Store16 => return self.mem.poke(addr, (self.reg_at(rt) as u16).to_le_bytes()),
            Acc::Store32 => return self.mem.poke(addr, (self.reg_at(rt) as u32).to_le_bytes()),
            Acc::Store64 => return self.mem.poke(addr, self.reg_at(rt).to_le_bytes()),
            Acc::Prefetch => return true,
        };
        match loaded {
            Some(value) => {
                self.set_reg_at(rt, value);
                true
            }
            None => false,
        }
    }

    /// A load or store through the full memory path, for ops whose fast path
    /// declined before changing anything.
    #[cold]
    #[inline(never)]
    fn exec_op_slow(&mut self, op: &Op) -> Result<()> {
        match *op {
            Op::LoadStoreImm {
                rt,
                rn,
                acc,
                wb,
                offset,
            } => self.load_store_imm(rt, rn, acc, wb, offset),
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
                self.access(addr, rt, acc)
            }
            Op::LoadLiteral { rt, addr, acc } => self.access(addr, rt, acc),
            Op::Load64 { rt, rn, wb, offset } => {
                self.load_store_imm(rt, rn, Acc::Load64, wb, offset)
            }
            Op::Store64 { rt, rn, wb, offset } => {
                self.load_store_imm(rt, rn, Acc::Store64, wb, offset)
            }
            Op::Load32 { rt, rn, wb, offset } => {
                self.load_store_imm(rt, rn, Acc::Load32, wb, offset)
            }
            Op::Store32 { rt, rn, wb, offset } => {
                self.load_store_imm(rt, rn, Acc::Store32, wb, offset)
            }
            Op::Load8 { rt, rn, wb, offset } => self.load_store_imm(rt, rn, Acc::Load8, wb, offset),
            Op::Store8 { rt, rn, wb, offset } => {
                self.load_store_imm(rt, rn, Acc::Store8, wb, offset)
            }
            Op::PairLoad64 {
                rt,
                rt2,
                rn,
                offset,
                wb,
            } => self.pair(rt, rt2, rn, offset, PairKind::Load64, wb),
            Op::PairStore64 {
                rt,
                rt2,
                rn,
                offset,
                wb,
            } => self.pair(rt, rt2, rn, offset, PairKind::Store64, wb),
            _ => unreachable!("only the fast load and store arms fall back to the full path"),
        }
    }

    /// A single-register immediate-offset load or store; `acc` is often a constant.
    #[inline(always)]
    fn load_store_imm(&mut self, rt: u8, rn: u8, acc: Acc, wb: Wb, offset: i64) -> Result<()> {
        let base = self.reg_at(rn);
        let (addr, wb_val) = Self::indexed(base, offset, wb);
        self.access(addr as u32, rt, acc)?;
        if let Some(v) = wb_val {
            self.set_reg_at(rn, v);
        }
        Ok(())
    }

    /// Execute a block's terminator, leaving `self.pc` on the next target, and
    /// return extra instructions retired (a folded PLT stub). `room` is the
    /// remaining budget, terminator included.
    #[inline(always)]
    fn exec_term(&mut self, term: &Term, pc: u32, room: u64) -> Result<u64> {
        match *term {
            Term::BlPlt { got, stub, ret_pc } => {
                self.write_zr(30, u64::from(ret_pc));
                return Ok(self.through_plt(got, stub, room));
            }
            Term::BPlt { got, stub } => return Ok(self.through_plt(got, stub, room)),
            Term::B { target } => self.pc = target,
            Term::Bl { target, ret_pc } => {
                self.write_zr(30, u64::from(ret_pc));
                self.pc = target;
            }
            Term::Br { rn } => self.pc = self.read_zr(rn) as u32,
            Term::Blr { rn, ret_pc } => {
                // Read the target first: `blr x30` would otherwise jump to itself.
                let target = self.read_zr(rn) as u32;
                self.write_zr(30, u64::from(ret_pc));
                self.pc = target;
            }
            Term::Ret { rn } => {
                // A return to 0 is homebrew's exit; route it through the trampoline.
                let target = self.read_zr(rn) as u32;
                self.pc = if target == 0 {
                    SELF_RETURN_TRAMPOLINE
                } else {
                    target
                };
            }
            Term::Svc { imm, next } => {
                // Retire the SVC first: a thread switch installs the next thread's PC.
                self.pc = next;
                self.syscall(imm)?;
            }
            Term::Interpret { insn, next } => {
                self.jit.note_interpreted(insn);
                self.execute(insn, next)?;
            }
            Term::Fetch => {
                let insn = self.mem.fetch(pc)?;
                self.jit.note_interpreted(insn);
                self.execute(insn, pc.wrapping_add(4))?;
            }
        }
        Ok(0)
    }

    /// Run the PLT stub at `stub` (jumping through GOT slot `got`) and return
    /// its four instructions. If the budget or the slot read needs more than
    /// the fast path, leave `pc` on the stub to run as its own block.
    #[inline(always)]
    fn through_plt(&mut self, got: u32, stub: u32, room: u64) -> u64 {
        const STUB: u64 = PLT_STUB as u64;
        let target = match self.mem.peek::<8>(got) {
            Some(bytes) if room > STUB => u64::from_le_bytes(bytes),
            _ => {
                self.pc = stub;
                return 0;
            }
        };
        self.write_zr(16, u64::from(got));
        self.write_zr(17, target);
        self.record_run(stub, STUB as u32);
        self.pc = target as u32;
        STUB
    }
}

#[cfg(test)]
mod tests {
    use super::super::ir::{Block, Branch, Exit, Op};
    use crate::cpu::Cpu;

    #[test]
    fn a_block_places_each_instruction_where_exec_does() {
        // Two followed `B`s: runs at 0x1000, 0x2000 and 0x3000.
        let exits = vec![
            Branch::new(1, Exit::Jump { target: 0x2000 }),
            Branch::new(3, Exit::Jump { target: 0x3000 }),
        ];
        let block = Block::new(
            0x1000,
            vec![Op::Nop; 6],
            vec![0; 6],
            exits,
            None,
            vec![1, 2, 3],
        );
        for i in 0..=block.ops.len() {
            let (pc, _, _) = Cpu::new().follow_runs(&block, i);
            assert_eq!(block.pc_at(i), pc, "instruction {i}");
        }
        assert_eq!(block.successors().collect::<Vec<_>>(), vec![0x3008]);
    }
}
