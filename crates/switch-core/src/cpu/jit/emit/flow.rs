//! Block walk, branches, terminators and the region's entry and exit.

use super::*;

impl Emitter<'_> {
    /// `run`'s result for leaving `retired` instructions into the current block.
    pub(super) fn answer(&self, retired: usize) -> u32 {
        (self.member as u32) << MEMBER_SHIFT | retired as u32
    }

    /// Leave the region through the epilogue with `answer`.
    pub(super) fn exit_with(&mut self, answer: u32) {
        if answer & LEFT != 0 {
            self.leaves = true;
        }
        self.f.i32_const(answer as i32);
        self.f.local_set(L_RET);
        self.f.br(self.epilogue);
    }

    /// Continue at `target` after `retired` instructions of the current block:
    /// inside the region if it has the block, out through the epilogue if not.
    /// `fall` says the code emitted next is where the following block starts.
    pub(super) fn goto(&mut self, retired: usize, target: u32, fall: bool) {
        match self.inside(target) {
            Some(k) => self.edge(retired, k, fall),
            None => {
                self.f.i32_const(target as i32);
                self.f.local_set(L_TARGET);
                self.exit_with(self.answer(retired) | LEFT);
            }
        }
    }

    /// The region's block at `pc`, when there is a loop head to reach it by.
    pub(super) fn inside(&self, pc: u32) -> Option<usize> {
        self.top?;
        self.targets.iter().position(|&start| start == pc)
    }

    /// Branch to the region's block `k`, charging `retired` to the fuel; out
    /// of fuel, leave with nothing of block `k` retired. Block `k` reads NZCV
    /// from its local, so an owed compare's is built first.
    pub(super) fn edge(&mut self, retired: usize, k: usize, fall: bool) {
        let owed = self.pending;
        if let Some(sf) = owed {
            self.materialize_pending(sf);
        }
        self.jump_to_member(retired, k, fall);
        // Only this path paid it; the code after a conditional edge still owes it.
        self.pending = owed;
    }

    fn jump_to_member(&mut self, retired: usize, k: usize, fall: bool) {
        self.fuel_used = true;
        self.f.local_get(L_FUEL);
        self.f.i32_const(retired as i32);
        self.f.i32_sub();
        self.f.local_tee(L_FUEL);
        self.f.i32_const(0);
        self.f.i32_lt_s();
        self.f.if_void();
        self.exit_with((k as u32) << MEMBER_SHIFT);
        self.f.end();
        if fall && k == self.member + 1 {
            return;
        }
        if k > self.member {
            self.f.br(self.starts[k]);
            return;
        }
        self.f.i32_const(k as i32);
        self.f.local_set(L_NEXT);
        let top = self.top.expect("`inside` answers only with a loop head");
        self.f.br(top);
    }

    /// Leave through `term`, which follows `len` instructions of the block,
    /// linking `x30` for a call. A register target that is one of `observed`
    /// (or, for a `RET`, where one of the region's calls returns) and in the
    /// region is branched to inside it.
    pub(super) fn term(&mut self, term: &Term, len: usize, observed: &[u32]) {
        let retired = len + term_retires(term);
        match *term {
            Term::B { target } => self.goto(retired, target, true),
            Term::Bl { target, ret_pc } => {
                self.f.i64_const(i64::from(ret_pc));
                self.store_reg(30);
                self.goto(retired, target, true);
            }
            // Read before linking: `blr x30` jumps to the old x30.
            Term::Br { rn } | Term::Blr { rn, .. } | Term::Ret { rn } => {
                self.read_reg_raw(rn & 0x1F);
                self.f.local_tee(L_T);
                self.leave_unless_below_4gib(len);
                self.f.local_get(L_T);
                self.f.i32_wrap_i64();
                self.f.local_set(L_TARGET);
                // A return to 0 is homebrew's exit, routed through the trampoline.
                if let Term::Ret { .. } = term {
                    self.f.i32_const(SELF_RETURN_TRAMPOLINE as i32);
                    self.f.local_get(L_TARGET);
                    self.f.local_get(L_TARGET);
                    self.f.i32_eqz();
                    self.f.select();
                    self.f.local_set(L_TARGET);
                }
                if let Term::Blr { ret_pc, .. } = *term {
                    self.f.i64_const(i64::from(ret_pc));
                    self.store_reg(30);
                }
                let mut candidates = observed.to_vec();
                if let Term::Ret { .. } = term {
                    let returns = self.returns.iter().filter(|pc| !observed.contains(pc));
                    candidates.extend(returns);
                }
                self.predict(&candidates, retired);
                self.exit_with(self.answer(retired) | LEFT);
            }
            // The stub's `adrp`/`ldr`/`add`/`br` as `exec`'s fold does them: x16 the
            // slot, x17 what it holds. A slot the page table cannot answer hands the
            // call back before anything is written.
            Term::BlPlt { got, .. } | Term::BPlt { got, .. } => {
                self.f.i32_const(got as i32);
                self.set_addr32();
                self.page_for_read(8, len);
                self.in_page();
                self.f.i64_load(UNALIGNED, 0);
                self.f.local_tee(L_T);
                self.leave_unless_below_4gib(len);
                self.f.local_get(L_T);
                self.store_reg(17);
                self.f.local_get(L_T);
                self.f.i32_wrap_i64();
                self.f.local_set(L_TARGET);
                self.f.i64_const(i64::from(got));
                self.store_reg(16);
                if let Term::BlPlt { ret_pc, .. } = *term {
                    self.f.i64_const(i64::from(ret_pc));
                    self.store_reg(30);
                }
                self.predict(observed, retired);
                self.exit_with(self.answer(retired) | LEFT);
            }
            _ => unreachable!("writes_term accepts only branches"),
        }
    }

    /// Branch to whichever of `observed` the region has when `L_TARGET` is it.
    pub(super) fn predict(&mut self, observed: &[u32], retired: usize) {
        for &pc in observed {
            let Some(k) = self.inside(pc) else {
                continue;
            };
            self.f.local_get(L_TARGET);
            self.f.i32_const(pc as i32);
            self.f.i32_eq();
            self.f.if_void();
            self.edge(retired, k, false);
            self.f.end();
        }
    }

    /// The current block's body and how it ends.
    pub(super) fn block(&mut self, block: &Block) -> Result<(), Refused> {
        self.pending = None;
        let mut next_exit = 0usize;
        let mut i = 0usize;
        while i < block.ops.len() {
            // The same walk [`super::exec`] makes over ops and branch spans.
            match block.exits.get(next_exit) {
                Some(branch) if branch.at as usize == i => {
                    let retired = i + branch.span as usize;
                    self.flags_dead = matches!(
                        branch.exit,
                        Exit::CmpImm { .. } | Exit::CmpReg { .. } | Exit::UpdateCmpImm { .. }
                    ) && flags_overwritten(block, retired, next_exit + 1);
                    if !self.exit(&branch.exit, retired) {
                        return Err(Refused::ControlFlow);
                    }
                    i = retired;
                    next_exit += 1;
                }
                _ => {
                    self.at = i;
                    if !self.op(&block.ops[i], i) {
                        return Err(Refused::Op(block.words[i]));
                    }
                    i += 1;
                }
            }
            if self.f.len() > MAX_BODY_BYTES {
                return Err(Refused::TooLong);
            }
        }
        // An exit the walk did not reach would be silently dropped; refuse.
        if next_exit != block.exits.len() {
            return Err(Refused::ControlFlow);
        }
        let len = block.ops.len();
        match block.term {
            Some(ref term) if writes_term(term) => self.term(term, len, &block.observed()),
            // Off the end of a block cut at its length or page limit.
            None => self.goto_or_stop(len, block.pc_at(len)),
            // `exec` runs the terminator.
            Some(_) => self.exit_with(self.answer(len)),
        }
        Ok(())
    }

    /// Fall into `next` when the region has it; otherwise stop, `exec` finding
    /// the pc from the count.
    pub(super) fn goto_or_stop(&mut self, retired: usize, next: u32) {
        match self.inside(next) {
            Some(k) => self.edge(retired, k, true),
            None => self.exit_with(self.answer(retired)),
        }
    }

    /// Store what the region changed and answer `L_RET`; a region that left
    /// writes the target to `pc` and, chained, jumps into its compiled form.
    pub(super) fn epilogue(&mut self) {
        if self.pend_kind != 0 {
            for sf in [false, true] {
                if self.pend_forms & (1 << u8::from(sf)) != 0 {
                    self.f.local_get(self.pend_kind);
                    self.f.i32_const(1 + i32::from(sf));
                    self.f.i32_eq();
                    self.f.if_void();
                    self.materialize_pending(sf);
                    self.f.end();
                }
            }
        }
        let regs = self.layout.regs;
        for slot in 0..REG_SLOTS as u32 {
            if self.written & (1 << slot) != 0 {
                self.f.local_get(STATE);
                self.f.local_get(self.locals[slot as usize]);
                self.f.i64_store(ALIGN_8, regs + 8 * slot);
            }
        }
        for r in 0..32u32 {
            if self.vwritten & (1 << r) != 0 {
                self.f.local_get(STATE);
                self.f.local_get(self.vlocals[r as usize]);
                self.f
                    .simd_mem(V128_STORE, ALIGN_8, self.layout.vregs + 16 * r);
            }
        }
        if self.flags_written {
            self.f.local_get(STATE);
            self.f.local_get(L_NZCV);
            self.f.i32_store(ALIGN_4, self.layout.nzcv);
        }
        if self.fuel_used {
            self.f.local_get(STATE);
            self.f.local_get(L_FUEL);
            self.f.i32_store(ALIGN_4, self.layout.fuel);
        }
        if self.leaves {
            self.f.local_get(L_RET);
            self.f.i32_const(LEFT as i32);
            self.f.i32_and();
            self.f.if_void();
            self.f.local_get(STATE);
            self.f.local_get(L_TARGET);
            self.f.i32_store(ALIGN_4, self.layout.pc);
            if self.chain {
                self.jump_out();
            }
            self.f.end();
        }
        self.f.local_get(L_RET);
    }

    /// Jump into the compiled region at `L_TARGET` if the chain table has it
    /// and what the stopping block retired fits the fuel; otherwise fall through.
    pub(super) fn jump_out(&mut self) {
        let at = ChainLayout::OF_CPU;
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, self.layout.fuel);
        self.f.local_get(L_RET);
        self.f.i32_const(COUNT_MASK as i32);
        self.f.i32_and();
        self.f.i32_sub();
        self.f.local_tee(L_IDX);
        self.f.i32_const(0);
        self.f.i32_ge_s();
        self.f.if_void();
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, at.slots);
        self.f.local_get(L_TARGET);
        self.f.i32_const(1);
        self.f.i32_shl();
        self.f.i32_const(((CHAIN_SLOTS - 1) * 8) as i32);
        self.f.i32_and();
        self.f.i32_add();
        self.f.local_tee(L_PAGE);
        self.f.i32_load(ALIGN_4, 0);
        self.f.local_get(L_TARGET);
        self.f.i32_eq();
        self.f.if_void();
        self.f.local_get(STATE);
        self.f.local_get(L_IDX);
        self.f.i32_store(ALIGN_4, self.layout.fuel);
        self.f.local_get(STATE);
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, at.hops);
        self.f.i32_const(1);
        self.f.i32_add();
        self.f.i32_store(ALIGN_4, at.hops);
        self.f.local_get(STATE);
        self.f.local_get(L_TARGET);
        self.f.i32_store(ALIGN_4, at.last);
        self.f.local_get(STATE);
        self.f.local_get(L_PAGE);
        self.f.i32_load(ALIGN_4, 4);
        self.f.return_call_indirect(0);
        self.f.end();
        self.f.end();
    }

    /// Leave the region at its terminator, `retired` instructions done, if the branch target
    /// on the stack is above 4 GiB, where no code is loaded: the interpreter faults on it.
    fn leave_unless_below_4gib(&mut self, retired: usize) {
        self.f.i64_const(32);
        self.f.i64_shr_u();
        self.f.i64_const(0);
        self.f.i64_ne();
        self.deopt_if(retired);
    }

    /// Load what the region touches, ahead of everything else.
    pub(super) fn prologue(&self) -> Vec<u8> {
        let mut f = Func::new();
        for slot in 0..REG_SLOTS as u32 {
            if self.used & (1 << slot) != 0 {
                f.local_get(STATE);
                f.i64_load(ALIGN_8, self.layout.regs + 8 * slot);
                f.local_set(self.locals[slot as usize]);
            }
        }
        for r in 0..32u32 {
            if self.vused & (1 << r) != 0 {
                f.local_get(STATE);
                f.simd_mem(V128_LOAD, ALIGN_8, self.layout.vregs + 16 * r);
                f.local_set(self.vlocals[r as usize]);
            }
        }
        if self.flags_used {
            f.local_get(STATE);
            f.i32_load(ALIGN_4, self.layout.nzcv);
            f.local_set(L_NZCV);
        }
        if self.fuel_used {
            f.local_get(STATE);
            f.i32_load(ALIGN_4, self.layout.fuel);
            f.local_set(L_FUEL);
        }
        if self.env.iter().any(Option::is_some) {
            // `L_ADDR` is free until the body sets it.
            f.local_get(STATE);
            f.i32_load(ALIGN_4, self.layout.memory);
            f.local_set(L_ADDR);
        }
        for (slot, offset) in self.env.iter().enumerate() {
            if let Some(offset) = *offset {
                f.local_get(L_ADDR);
                f.i32_load(ALIGN_4, offset);
                f.local_set(L_ENV + slot as u32);
            }
        }
        f.code().to_vec()
    }

    /// Leave the region if the value on the stack is nonzero, reporting
    /// `retired` instructions of this block done; the interpreter resumes there.
    pub(super) fn deopt_if(&mut self, retired: usize) {
        self.f.if_void();
        self.exit_with(self.answer(retired));
        self.f.end();
    }

    /// Branch to `target` if the value on the stack is nonzero, with
    /// `retired` instructions (the branch included) done.
    pub(super) fn branch_if(&mut self, retired: usize, target: u32) {
        self.f.if_void();
        self.goto(retired, target, false);
        self.f.end();
    }

    /// The flag-setting half of a fused compare-and-branch, operands in `L_A`
    /// and `L_B`.
    pub(super) fn compare_flags(&mut self, carry: u8, sf: bool) {
        self.add_carry(carry, true, sf);
        self.pack_nzcv(sf, true);
        self.store_nzcv();
    }

    /// Emit the conditional branch `retired` instructions into the block, or
    /// answer `false`. The not-taken path is whatever is emitted next.
    pub(super) fn exit(&mut self, exit: &Exit, retired: usize) -> bool {
        let target = match *exit {
            Exit::Cond { cond, target } => {
                self.cond_holds(cond);
                target
            }
            // The compare runs whether or not the branch is taken.
            Exit::CmpImm {
                rn,
                imm,
                carry,
                sf,
                cond,
                target,
            } if carry != 0 && self.flags_dead && Self::direct_compare(cond).is_some() => {
                self.pending_locals();
                self.read_reg(rn, sf);
                self.f.local_set(self.pend_a);
                self.f.i64_const(i64::from(imm) & width_mask(sf));
                self.f.local_set(self.pend_b);
                self.lazy_branch(cond, sf, retired, target);
                return true;
            }
            Exit::CmpReg {
                rn,
                rm,
                carry,
                sf,
                cond,
                target,
            } if carry != 0 && self.flags_dead && Self::direct_compare(cond).is_some() => {
                self.pending_locals();
                self.read_reg(rn, sf);
                self.f.local_set(self.pend_a);
                self.read_reg(rm, sf);
                self.f.local_set(self.pend_b);
                self.lazy_branch(cond, sf, retired, target);
                return true;
            }
            Exit::CmpImm {
                rn,
                imm,
                carry,
                sf,
                cond,
                target,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                let rhs = invert_if_const(u64::from(imm), carry) & width_mask(sf) as u64;
                self.f.i64_const(rhs as i64);
                self.f.local_set(L_B);
                self.compare_flags(carry, sf);
                self.cond_holds(cond);
                target
            }
            // So does the update.
            Exit::UpdateCmpImm {
                rd,
                source,
                step,
                rn,
                imm,
                cond,
                target,
            } => {
                let update = Op::AddSubImm {
                    rd,
                    rn: source,
                    rhs: invert_if_const(step.imm(), step.carry()),
                    carry: step.carry(),
                    set_flags: false,
                    sf: step.sf(),
                };
                let compare = Exit::CmpImm {
                    rn,
                    imm: imm.imm() as u32,
                    carry: imm.carry(),
                    sf: imm.sf(),
                    cond,
                    target,
                };
                return self.op(&update, retired - 3) && self.exit(&compare, retired);
            }
            Exit::CmpReg {
                rn,
                rm,
                carry,
                sf,
                cond,
                target,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.read_reg(rm, sf);
                self.invert_if(carry, sf);
                self.f.local_set(L_B);
                self.compare_flags(carry, sf);
                self.cond_holds(cond);
                target
            }
            // Register 31 reads as the zero register, so the field is the slot.
            Exit::Cbz { rt, sf, nz, target } => {
                self.read_reg(rt & 0x1F, sf);
                self.is_zero(!nz);
                target
            }
            Exit::Tbz {
                rt,
                bit,
                nz,
                target,
            } => {
                self.read_reg_raw(rt & 0x1F);
                self.f.i64_const(i64::from(bit));
                self.f.i64_shr_u();
                self.f.i64_const(1);
                self.f.i64_and();
                self.is_zero(!nz);
                target
            }
            // A followed `B`: the following ops are already its target's.
            Exit::Jump { .. } => return true,
        };
        self.branch_if(retired, target);
        true
    }

    /// Replace the value on the stack with whether it is zero (or nonzero
    /// when `want_zero` is false).
    pub(super) fn is_zero(&mut self, want_zero: bool) {
        if want_zero {
            self.f.i64_eqz();
        } else {
            self.f.i64_const(0);
            self.f.i64_ne();
        }
    }
}

/// Whether [`emit_block`] writes `term` instead of leaving it to [`super::exec`].
/// What an instruction does to NZCV.
enum FlagEffect {
    Neutral,
    Overwrites,
    Live,
}

fn flag_effect(op: &Op) -> FlagEffect {
    match op {
        Op::AddSubImm { set_flags, .. }
        | Op::AddSubShifted { set_flags, .. }
        | Op::AddSubReg { set_flags, .. }
        | Op::AddSubExtended { set_flags, .. } => {
            if *set_flags {
                FlagEffect::Overwrites
            } else {
                FlagEffect::Neutral
            }
        }
        // `ANDS` keeps C and V, so it reads the flags.
        Op::LogicalImm { opc, .. }
        | Op::LogicalShifted { opc, .. }
        | Op::LogicalReg { opc, .. } => {
            if *opc == 0b11 {
                FlagEffect::Live
            } else {
                FlagEffect::Neutral
            }
        }
        Op::Nop
        | Op::MovConst { .. }
        | Op::Mov32 { .. }
        | Op::Mov64 { .. }
        | Op::MovK { .. }
        | Op::Extract { .. }
        | Op::Bitfield { .. }
        | Op::Extr { .. }
        | Op::Madd { .. }
        | Op::MaddLong { .. }
        | Op::Mulh { .. }
        | Op::ShiftVar { .. }
        | Op::Divide { .. }
        | Op::OneSource { .. }
        | Op::Crc { .. }
        | Op::LoadStoreImm { .. }
        | Op::Load64 { .. }
        | Op::Store64 { .. }
        | Op::Load32 { .. }
        | Op::Store32 { .. }
        | Op::Load8 { .. }
        | Op::Store8 { .. }
        | Op::LoadStoreReg { .. }
        | Op::Pair { .. }
        | Op::PairLoad64 { .. }
        | Op::PairStore64 { .. }
        | Op::LoadLiteral { .. }
        | Op::SimdLoadStore { .. } => FlagEffect::Neutral,
        _ => FlagEffect::Live,
    }
}

/// Whether a fused compare ending at op `from` (its next exit being `exit`)
/// has its flags overwritten by a later compare or flag-setting instruction
/// of the block before anything reads them or the block ends. Flags live at the
/// block's end are built on every way out, so deferring them would save nothing.
fn flags_overwritten(block: &Block, from: usize, mut exit: usize) -> bool {
    let mut i = from;
    while i < block.ops.len() {
        if let Some(branch) = block.exits.get(exit).filter(|b| b.at as usize == i) {
            match branch.exit {
                Exit::CmpImm { .. } | Exit::CmpReg { .. } | Exit::UpdateCmpImm { .. } => {
                    return true
                }
                // These test registers; a taken one into the region builds the owed NZCV first.
                Exit::Jump { .. } | Exit::Cbz { .. } | Exit::Tbz { .. } => {
                    i += branch.span as usize;
                    exit += 1;
                    continue;
                }
                Exit::Cond { .. } => return false,
            }
        }
        match flag_effect(&block.ops[i]) {
            FlagEffect::Neutral => {}
            FlagEffect::Overwrites => return true,
            FlagEffect::Live => return false,
        }
        i += 1;
    }
    false
}
