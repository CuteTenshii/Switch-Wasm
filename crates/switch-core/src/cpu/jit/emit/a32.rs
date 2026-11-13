//! A32 blocks as wasm: the leading ops of a block the emitter can write, then a
//! stop with the count retired, so `exec` carries on from the next one. Anything
//! that needs the interpreter's memory path hands the instruction back, as in A64.

use super::a32_neon::neon_writable;
use super::a32_vfp::vfp_writable;
use super::*;
use crate::cpu::a32::loadstore::Access;
use crate::cpu::jit::a32::decode::{decode, Decoded};
use crate::cpu::jit::a32::{
    Block as A32Block, Kind, Offset, Op as A32Op, Operand, Term as A32Term, TermKind, ALWAYS,
};

/// Registers hold 32-bit values zero-extended to 64.
const WORD: i64 = 0xFFFF_FFFF;

/// What [`emit_a32`] wrote.
pub(in crate::cpu::jit) struct EmittedA32 {
    pub(in crate::cpu::jit) module: Vec<u8>,
    /// The most instructions the module can retire: the leading ops it covers, and
    /// the terminator too when it leaves through a branch.
    pub(in crate::cpu::jit) retires: usize,
}

/// Emit as many of `block`'s leading ops as there are writers for, as a module
/// exporting `run`. A block written whole, with a terminator that branches or
/// none at all, leaves with [`LEFT`] and `pc` set; `chain` has it jump into the
/// next compiled block through the function table.
pub(in crate::cpu::jit) fn emit_a32(
    block: &A32Block,
    layout: Layout,
    chain: bool,
    watches: bool,
) -> Result<EmittedA32, Refused> {
    let mut f = scratch_func();
    let mut e = Emitter::new(&mut f, layout, chain, Vec::new());
    e.watches = watches;
    let mut done = 0;
    for (i, op) in block.ops.iter().enumerate() {
        let pc = block.start.wrapping_add(4 * i as u32);
        if !e.a32_op(op, i, pc) {
            break;
        }
        done = i + 1;
        if e.f.len() > MAX_BODY_BYTES {
            return Err(Refused::TooLong);
        }
    }
    let mut retires = done;
    if done == block.ops.len() {
        let end = block.start.wrapping_add(4 * done as u32);
        match block.term {
            Some(term) if e.a32_term(&term, done, end) => retires += 1,
            Some(_) => e.exit_with(e.answer(done)),
            None => e.goto(done, end, false),
        }
    } else {
        e.exit_with(e.answer(done));
    }
    if retires == 0 {
        return Err(Refused::Op(block.words.first().copied().unwrap_or(0)));
    }
    e.f.end();
    e.epilogue();
    let prologue = e.prologue();
    let registers = e.next_local - L_REGS;
    f.prepend(&prologue);
    f.locals(1, registers, I64);
    f.end();

    let mut m = Module::new();
    if chain {
        m.import_table();
    }
    let ty = m.add_type(vec![I32], vec![I32]);
    let idx = m.add_func(ty, f);
    m.export("run", idx);
    Ok(EmittedA32 {
        module: m.finish(),
        retires,
    })
}

/// Whether the A32 emitter has a writer for `insn`, as an op or as the branch that ends a block.
pub fn a32_emits(insn: u32) -> bool {
    const REPRESENTATIVE_PC: u32 = 0x0800_0000;
    match decode(insn, REPRESENTATIVE_PC) {
        Decoded::Op(op) => writable(&op),
        Decoded::Term(term) => term_writable(&term.kind),
    }
}

/// Whether [`Emitter::a32_term`] has a writer for a terminator of `kind`.
fn term_writable(kind: &TermKind) -> bool {
    match *kind {
        TermKind::B { .. }
        | TermKind::Bl { .. }
        | TermKind::Bx { .. }
        | TermKind::Blx { .. }
        | TermKind::LdrPc { .. } => true,
        TermKind::LdmPc { access, list } => block_writable(access, list),
        TermKind::Svc { .. } | TermKind::Interpret { .. } | TermKind::Fetch => false,
    }
}

/// `SBFX` and `UBFX` the emitter writes, as destination, source, least significant bit, width and signedness.
fn bitfield_extract(insn: u32) -> Option<(u8, u8, u32, u32, bool)> {
    let op1 = (insn >> 20) & 0x1F;
    let rd = ((insn >> 12) & 0xF) as u8;
    let rm = (insn & 0xF) as u8;
    let (lsb, width) = ((insn >> 7) & 0x1F, ((insn >> 16) & 0x1F) + 1);
    let media = insn >> 28 != 0xF && (insn >> 25) & 7 == 0b011 && insn & 0x10 != 0;
    (media
        && matches!(op1, 0x1A | 0x1B | 0x1E | 0x1F)
        && (insn >> 5) & 0b11 == 0b10
        && rd != 15
        && rm != 15
        && lsb + width <= 32)
        .then_some((rd, rm, lsb, width, op1 & 0b00100 == 0))
}

/// An `LDM` or `STM` the emitter can write.
fn block_writable(access: Access, list: u16) -> bool {
    // A load that names its base is not safe to hand back half-way.
    list != 0 && !(access.load && list & (1 << access.rn) != 0)
}

/// The data-processing opcodes whose flags come from the shifter's carry.
fn is_logical(opcode: u8) -> bool {
    matches!(opcode, 0x0 | 0x1 | 0x8 | 0x9 | 0xC | 0xD | 0xE | 0xF)
}

/// Whether [`Emitter::a32_op`] has a writer for `op`.
fn writable(op: &A32Op) -> bool {
    match op.kind {
        Kind::Nop
        | Kind::Single { .. }
        | Kind::MoveWide { .. }
        | Kind::Extend { .. }
        | Kind::Extend16 { .. }
        | Kind::BitfieldInsert { .. }
        | Kind::Multiply { .. }
        | Kind::Exclusive { .. }
        | Kind::ReadThreadPointer { .. } => true,
        Kind::Interpret { insn } => {
            vfp_writable(insn) || neon_writable(insn) || bitfield_extract(insn).is_some()
        }
        Kind::Dp { .. } => true,
        Kind::Extra { access, sh, .. } => match (sh, access.load) {
            (0b01, _) | (0b10 | 0b11, true) => true,
            // LDRD and STRD move `rt` and `rt + 1`, and a written-back base may not be either.
            _ => {
                access.rt % 2 == 0
                    && access.rt < 14
                    && !(access.writes_base()
                        && (access.rn == access.rt || access.rn == access.rt + 1))
            }
        },
        Kind::Block { access, list } => block_writable(access, list),
    }
}

impl Emitter<'_> {
    /// Write `op`, the `i`th of its block, at `pc`. `false`, with nothing
    /// written, when it needs the interpreter.
    fn a32_op(&mut self, op: &A32Op, i: usize, pc: u32) -> bool {
        if !writable(op) {
            return false;
        }
        let cond = match op.kind {
            Kind::Interpret { insn } if insn >> 28 != 0xF => (insn >> 28) as u8,
            _ => op.cond,
        };
        let predicated = cond != ALWAYS;
        if predicated {
            self.cond_holds(cond);
            self.f.if_void();
        }
        match op.kind {
            Kind::Nop => {}
            Kind::Interpret { insn } if vfp_writable(insn) => self.a32_vfp(insn, i, pc),
            Kind::Interpret { insn } if neon_writable(insn) => self.a32_neon(insn, i, pc),
            Kind::Interpret { insn } => {
                let (rd, rm, lsb, width, signed) =
                    bitfield_extract(insn).expect("`writable` was checked");
                self.read_reg_raw(rm);
                if signed {
                    self.f.i64_const(i64::from(64 - lsb - width));
                    self.f.i64_shl();
                    self.f.i64_const(i64::from(64 - width));
                    self.f.i64_shr_s();
                    self.mask_to(false);
                } else {
                    self.f.i64_const(i64::from(lsb));
                    self.f.i64_shr_u();
                    self.f.i64_const(i64::from(u32::MAX >> (32 - width)));
                    self.f.i64_and();
                }
                self.store_reg(rd);
            }
            Kind::Dp {
                opcode,
                set_flags,
                rd,
                rn,
                operand,
            } => self.a32_data_processing(opcode, set_flags, rd, rn, operand, pc),
            Kind::Single {
                access,
                byte,
                offset,
            } => {
                let acc = match (access.load, byte) {
                    (true, true) => Acc::Load8,
                    (true, false) => Acc::Load32,
                    (false, true) => Acc::Store8,
                    (false, false) => Acc::Store32,
                };
                self.a32_transfer(access, acc, offset, false, i, pc);
            }
            Kind::Extra { access, sh, offset } if sh >= 0b10 && !access.load => {
                self.a32_effective_address(access, offset, pc);
                if sh == 0b10 {
                    self.page_for_read(8, i);
                    self.in_page();
                    self.f.i64_load(UNALIGNED, 0);
                    self.f.local_tee(L_T);
                    self.mask_to(false);
                    self.store_reg(access.rt);
                    self.f.local_get(L_T);
                    self.f.i64_const(32);
                    self.f.i64_shr_u();
                    self.store_reg(access.rt + 1);
                } else {
                    self.page_for_write(8, i);
                    self.in_page();
                    self.a32_read(access.rt + 1, pc);
                    self.f.i64_const(32);
                    self.f.i64_shl();
                    self.a32_read(access.rt, pc);
                    self.f.i64_or();
                    self.f.i64_store(UNALIGNED, 0);
                }
                if access.writes_base() {
                    self.f.local_get(L_W);
                    self.store_reg(access.rn);
                }
            }
            Kind::Extra { access, sh, offset } => {
                let acc = match (sh, access.load) {
                    (0b01, true) => Acc::Load16,
                    (0b01, false) => Acc::Store16,
                    (0b10, true) => Acc::LoadS8To32,
                    _ => Acc::LoadS16To32,
                };
                self.a32_transfer(access, acc, offset, true, i, pc);
            }
            Kind::MoveWide { rd, imm, top } => {
                if top {
                    self.read_reg_raw(rd);
                    self.f.i64_const(0xFFFF);
                    self.f.i64_and();
                    self.f.i64_const(i64::from(imm) << 16);
                    self.f.i64_or();
                } else {
                    self.f.i64_const(i64::from(imm));
                }
                self.store_reg(rd);
            }
            Kind::Extend {
                rd,
                rn,
                rm,
                rotate,
                signed,
                halfword,
            } => {
                self.read_reg_raw(rm);
                if rotate != 0 {
                    self.f.i64_const(i64::from(rotate));
                    self.f.i64_shr_u();
                    self.read_reg_raw(rm);
                    self.f.i64_const(i64::from(32 - rotate));
                    self.f.i64_shl();
                    self.f.i64_or();
                }
                match (signed, halfword) {
                    (true, false) => self.f.i64_extend8_s(),
                    (true, true) => self.f.i64_extend16_s(),
                    (false, false) => {
                        self.f.i64_const(0xFF);
                        self.f.i64_and();
                    }
                    (false, true) => {
                        self.f.i64_const(0xFFFF);
                        self.f.i64_and();
                    }
                }
                if rn != 15 {
                    self.read_reg_raw(rn);
                    self.f.i64_add();
                }
                self.mask_to(false);
                self.store_reg(rd);
            }
            Kind::Extend16 {
                rd,
                rn,
                rm,
                rotate,
                signed,
            } => {
                self.read_reg_raw(rm);
                if rotate != 0 {
                    self.f.i64_const(i64::from(rotate));
                    self.f.i64_shr_u();
                    self.read_reg_raw(rm);
                    self.f.i64_const(i64::from(32 - rotate));
                    self.f.i64_shl();
                    self.f.i64_or();
                }
                self.f.local_set(L_T);
                if rn != 15 {
                    self.read_reg_raw(rn);
                    self.f.local_set(L_A);
                }
                // The low lane, in bits 15:0.
                self.f.local_get(L_T);
                if signed {
                    self.f.i64_extend8_s();
                } else {
                    self.f.i64_const(0xFF);
                    self.f.i64_and();
                }
                if rn != 15 {
                    self.f.local_get(L_A);
                    self.f.i64_add();
                }
                self.f.i64_const(0xFFFF);
                self.f.i64_and();
                // The high lane, in bits 31:16.
                self.f.local_get(L_T);
                if signed {
                    self.f.i64_const(16);
                    self.f.i64_shr_u();
                    self.f.i64_extend8_s();
                    self.f.i64_const(16);
                    self.f.i64_shl();
                } else {
                    self.f.i64_const(0xFF_0000);
                    self.f.i64_and();
                }
                if rn != 15 {
                    self.f.local_get(L_A);
                    self.f.i64_add();
                }
                self.f.i64_const(0xFFFF_0000);
                self.f.i64_and();
                self.f.i64_or();
                self.store_reg(rd);
            }
            Kind::BitfieldInsert { rd, rn, lsb, mask } => {
                self.read_reg_raw(rd);
                self.f.i64_const(i64::from(!mask));
                self.f.i64_and();
                if rn != 15 {
                    self.read_reg_raw(rn);
                    self.f.i64_const(i64::from(lsb));
                    self.f.i64_shl();
                    self.f.i64_const(i64::from(mask));
                    self.f.i64_and();
                    self.f.i64_or();
                }
                self.store_reg(rd);
            }
            Kind::Multiply {
                op,
                set_flags,
                rd_hi,
                rd_lo,
                rs,
                rm,
            } => self.a32_multiply(op, set_flags, rd_hi, rd_lo, rs, rm),
            Kind::ReadThreadPointer { rt, rw } => {
                self.f.local_get(STATE);
                let at = if rw {
                    self.layout.tpidr_rw
                } else {
                    self.layout.tpidr
                };
                self.f.i64_load(ALIGN_8, at);
                self.mask_to(false);
                self.store_reg(rt);
            }
            Kind::Exclusive {
                load,
                size,
                rd,
                rn,
                rt,
            } => {
                self.a32_read(rn, pc);
                self.f.i32_wrap_i64();
                self.f.local_set(L_ADDR);
                if load {
                    self.load_exclusive_at(rd, size, i);
                } else {
                    self.store_exclusive_at(rd, rt, size, i);
                }
            }
            Kind::Block { access, list } => self.a32_block_transfer(access, list, i, pc),
        }
        if predicated {
            self.f.end();
        }
        true
    }

    /// Write `term`, which follows `len` instructions and sits at `pc`, leaving
    /// through every path. `false`, with nothing written, when the host has to run it.
    fn a32_term(&mut self, term: &A32Term, len: usize, pc: u32) -> bool {
        if !term_writable(&term.kind) {
            return false;
        }
        let predicated = term.cond != ALWAYS;
        if predicated {
            self.cond_holds(term.cond);
            self.f.if_void();
        }
        let retired = len + 1;
        match term.kind {
            TermKind::B { target } => self.goto(retired, target, false),
            TermKind::Bl { target } => {
                self.f.i64_const(i64::from(pc.wrapping_add(4)));
                self.store_reg(14);
                self.goto(retired, target, false);
            }
            TermKind::Bx { rm } | TermKind::Blx { rm } => {
                self.a32_read(rm, pc);
                self.f.local_tee(L_T);
                self.f.i64_const(1);
                self.f.i64_and();
                self.f.i32_wrap_i64();
                // An odd target is a switch to Thumb, which the interpreter runs.
                self.deopt_if(len);
                if matches!(term.kind, TermKind::Blx { .. }) {
                    self.f.i64_const(i64::from(pc.wrapping_add(4)));
                    self.store_reg(14);
                }
                self.f.local_get(L_T);
                self.f.i32_wrap_i64();
                self.a32_leave_to_target(retired);
            }
            TermKind::LdrPc { access, offset } => {
                self.a32_transfer(access, Acc::Load32, offset, false, len, pc);
                self.f.local_get(L_TARGET);
                self.a32_leave_to_target(retired);
            }
            TermKind::LdmPc { access, list } => {
                self.a32_block_transfer(access, list, len, pc);
                self.f.local_get(L_TARGET);
                self.a32_leave_to_target(retired);
            }
            TermKind::Svc { .. } | TermKind::Interpret { .. } | TermKind::Fetch => unreachable!(),
        }
        if predicated {
            self.f.end();
            self.goto(retired, pc.wrapping_add(4), false);
        }
        true
    }

    /// Leave to the word-aligned branch target on the stack, an `i32`.
    fn a32_leave_to_target(&mut self, retired: usize) {
        self.f.i32_const(!3);
        self.f.i32_and();
        self.f.local_set(L_TARGET);
        self.exit_with(self.answer(retired) | LEFT);
    }

    /// Push `r`, with `r15` as the address of the instruction at `pc` plus 8.
    pub(super) fn a32_read(&mut self, r: u8, pc: u32) {
        if r == 15 {
            self.f.i64_const(i64::from(pc.wrapping_add(8)));
        } else {
            self.read_reg_raw(r);
        }
    }

    /// Push C as 0 or 1, an `i32`.
    fn a32_carry_flag(&mut self) {
        self.flags_used = true;
        self.f.local_get(L_NZCV);
        self.f.i32_const(29);
        self.f.i32_shr_u();
        self.f.i32_const(1);
        self.f.i32_and();
    }

    /// Push bit `bit` of `L_T` as 0 or 1, an `i32`.
    fn a32_bit_of_t(&mut self, bit: u32) {
        self.f.local_get(L_T);
        self.f.i64_const(i64::from(bit));
        self.f.i64_shr_u();
        self.f.i32_wrap_i64();
        self.f.i32_const(1);
        self.f.i32_and();
    }

    /// The second operand into `L_B`, and the shifter's carry into `L_C` when
    /// `carry_out`, as `shift_c` and `expand_imm_c` produce them.
    fn a32_operand(&mut self, operand: Operand, carry_out: bool, pc: u32) {
        match operand {
            Operand::Imm(value) | Operand::ImmKeep(value) => {
                self.f.i64_const(i64::from(value));
                self.f.local_set(L_B);
                if carry_out {
                    if matches!(operand, Operand::Imm(_)) {
                        self.f.i32_const((value >> 31) as i32);
                    } else {
                        self.a32_carry_flag();
                    }
                    self.f.local_set(L_C);
                }
            }
            Operand::RegImm { rm, ty, amount } => {
                self.a32_read(rm, pc);
                self.f.local_set(L_T);
                self.a32_shift(ty, u32::from(amount), carry_out);
            }
            Operand::RegReg { rm, ty, rs } => {
                self.a32_read(rm, pc);
                self.f.local_set(L_T);
                self.a32_read(rs, pc);
                self.f.i64_const(0xFF);
                self.f.i64_and();
                self.f.local_set(L_A);
                self.a32_shift_by_register(ty);
                if carry_out {
                    self.a32_register_shift_carry(ty);
                }
            }
        }
    }

    /// Push `L_A` limited to `most`.
    fn a32_amount_up_to(&mut self, most: i64) {
        self.f.local_get(L_A);
        self.f.i64_const(most);
        self.f.local_get(L_A);
        self.f.i64_const(most);
        self.f.i64_lt_u();
        self.f.select();
    }

    /// `L_T` shifted by `L_A` bits, a byte, into `L_B`; the carry out is not written.
    /// The shifter's carry out for the shift of `L_T` by `L_A` (a byte) into `L_C`: C itself
    /// for a shift of zero, else the last bit shifted out.
    fn a32_register_shift_carry(&mut self, ty: u8) {
        let last = |e: &mut Self| {
            e.f.local_get(L_A);
            e.f.i64_const(1);
            e.f.i64_sub();
        };
        // The candidate: the bit of `L_T` that leaves last, or zero past the width.
        match ty {
            0 | 1 => {
                self.f.local_get(L_T);
                last(self);
                if ty == 0 {
                    self.f.i64_shl();
                    self.f.i64_const(31);
                }
                self.f.i64_shr_u();
                self.f.i64_const(1);
                self.f.i64_and();
                self.f.i32_wrap_i64();
                self.f.i32_const(0);
                self.f.local_get(L_A);
                self.f.i64_const(33);
                self.f.i64_lt_u();
                self.f.select();
            }
            2 => {
                self.f.local_get(L_T);
                self.f.i64_extend32_s();
                last(self);
                self.f.i64_const(31);
                last(self);
                self.f.i64_const(31);
                self.f.i64_lt_u();
                self.f.select();
                self.f.i64_shr_s();
                self.f.i64_const(1);
                self.f.i64_and();
                self.f.i32_wrap_i64();
            }
            _ => {
                self.f.local_get(L_T);
                last(self);
                self.f.i64_const(31);
                self.f.i64_and();
                self.f.i64_shr_u();
                self.f.i64_const(1);
                self.f.i64_and();
                self.f.i32_wrap_i64();
            }
        }
        self.f.local_set(L_C);
        // `select` keeps its first operand when the condition holds.
        self.a32_carry_flag();
        self.f.local_get(L_C);
        self.f.local_get(L_A);
        self.f.i64_eqz();
        self.f.select();
        self.f.local_set(L_C);
    }

    fn a32_shift_by_register(&mut self, ty: u8) {
        match ty {
            0 | 1 => {
                self.f.local_get(L_T);
                self.a32_amount_up_to(32);
                if ty == 0 {
                    self.f.i64_shl();
                    self.f.i64_const(WORD);
                    self.f.i64_and();
                } else {
                    self.f.i64_shr_u();
                }
            }
            2 => {
                self.f.local_get(L_T);
                self.f.i64_extend32_s();
                self.a32_amount_up_to(31);
                self.f.i64_shr_s();
                self.f.i64_const(WORD);
                self.f.i64_and();
            }
            _ => {
                self.f.local_get(L_T);
                self.f.i64_const(32);
                self.f.i64_shl();
                self.f.local_get(L_T);
                self.f.i64_or();
                self.f.local_get(L_A);
                self.f.i64_const(31);
                self.f.i64_and();
                self.f.i64_shr_u();
                self.f.i64_const(WORD);
                self.f.i64_and();
            }
        }
        self.f.local_set(L_B);
    }

    /// `L_T` shifted by a constant into `L_B`, with the carry out in `L_C`
    /// when wanted. `ty` is 4 for `RRX`; amounts are as `decode_imm_shift` leaves them.
    fn a32_shift(&mut self, ty: u8, amount: u32, carry_out: bool) {
        match (ty, amount) {
            (0, 0) => {
                self.f.local_get(L_T);
                self.f.local_set(L_B);
                if carry_out {
                    self.a32_carry_flag();
                    self.f.local_set(L_C);
                }
            }
            (0, _) => {
                self.f.local_get(L_T);
                self.f.i64_const(i64::from(amount));
                self.f.i64_shl();
                self.f.i64_const(WORD);
                self.f.i64_and();
                self.f.local_set(L_B);
                if carry_out {
                    self.a32_bit_of_t(32 - amount);
                    self.f.local_set(L_C);
                }
            }
            (1, 32) => {
                self.f.i64_const(0);
                self.f.local_set(L_B);
                if carry_out {
                    self.a32_bit_of_t(31);
                    self.f.local_set(L_C);
                }
            }
            (1, _) => {
                self.f.local_get(L_T);
                self.f.i64_const(i64::from(amount));
                self.f.i64_shr_u();
                self.f.local_set(L_B);
                if carry_out {
                    self.a32_bit_of_t(amount - 1);
                    self.f.local_set(L_C);
                }
            }
            (2, _) => {
                self.f.local_get(L_T);
                self.f.i64_extend32_s();
                self.f.i64_const(i64::from(amount.min(31)));
                self.f.i64_shr_s();
                self.f.i64_const(WORD);
                self.f.i64_and();
                self.f.local_set(L_B);
                if carry_out {
                    self.a32_bit_of_t(amount - 1);
                    self.f.local_set(L_C);
                }
            }
            (3, _) => {
                self.f.local_get(L_T);
                self.rotate_right_const(amount % 32, false);
                self.f.local_set(L_B);
                if carry_out {
                    self.a32_bit_of_t(amount - 1);
                    self.f.local_set(L_C);
                }
            }
            _ => {
                self.a32_carry_flag();
                self.f.i64_extend_i32_u();
                self.f.i64_const(31);
                self.f.i64_shl();
                self.f.local_get(L_T);
                self.f.i64_const(1);
                self.f.i64_shr_u();
                self.f.i64_or();
                self.f.local_set(L_B);
                if carry_out {
                    self.a32_bit_of_t(0);
                    self.f.local_set(L_C);
                }
            }
        }
    }

    /// A data-processing instruction, as [`crate::cpu::Cpu::a32_alu`] computes it.
    fn a32_data_processing(
        &mut self,
        opcode: u8,
        set_flags: bool,
        rd: u8,
        rn: u8,
        operand: Operand,
        pc: u32,
    ) {
        let logical = is_logical(opcode);
        self.a32_operand(operand, logical && set_flags, pc);
        if !matches!(opcode, 0xD | 0xF) {
            self.a32_read(rn, pc);
            self.f.local_set(L_A);
        }
        if logical {
            self.a32_logical(opcode, set_flags);
        } else {
            self.a32_arithmetic(opcode, set_flags);
        }
        if !(0x8..=0xB).contains(&opcode) {
            self.write_reg_from_r(rd);
        }
    }

    /// The logical opcodes into `L_R`, with N, Z and C set when `set_flags` and V kept.
    fn a32_logical(&mut self, opcode: u8, set_flags: bool) {
        match opcode {
            0xD => self.f.local_get(L_B),
            0xF => {
                self.f.local_get(L_B);
                self.f.i64_const(WORD);
                self.f.i64_xor();
            }
            _ => {
                self.f.local_get(L_A);
                match opcode {
                    0xE => {
                        self.f.local_get(L_B);
                        self.f.i64_const(-1);
                        self.f.i64_xor();
                        self.f.i64_and();
                    }
                    0xC => {
                        self.f.local_get(L_B);
                        self.f.i64_or();
                    }
                    0x1 | 0x9 => {
                        self.f.local_get(L_B);
                        self.f.i64_xor();
                    }
                    _ => {
                        self.f.local_get(L_B);
                        self.f.i64_and();
                    }
                }
            }
        }
        self.f.local_set(L_R);
        if set_flags {
            self.a32_pack_nz(31);
            self.f.local_get(L_C);
            self.f.i32_const(29);
            self.f.i32_shl();
            self.f.i32_or();
            self.flags_used = true;
            self.f.local_get(L_NZCV);
            self.f.i32_const(0x1000_0000);
            self.f.i32_and();
            self.f.i32_or();
            self.store_nzcv();
        }
    }

    /// `MUL`, `MLA`, `MLS` and the long multiplies into `L_R`, with N and Z set when `set_flags`.
    fn a32_multiply(&mut self, op: u8, set_flags: bool, rd_hi: u8, rd_lo: u8, rs: u8, rm: u8) {
        let long = !matches!(op, 0b000 | 0b001 | 0b011);
        let signed = op & 0b010 != 0;
        if op == 0b011 {
            self.read_reg_raw(rd_lo);
        }
        for r in [rm, rs] {
            self.read_reg_raw(r);
            if long && signed {
                self.f.i64_extend32_s();
            }
        }
        self.f.i64_mul();
        match (long, op) {
            (false, 0b001) => {
                self.read_reg_raw(rd_lo);
                self.f.i64_add();
            }
            (false, 0b011) => self.f.i64_sub(),
            (true, _) if op & 0b001 != 0 => {
                self.read_reg_raw(rd_hi);
                self.f.i64_const(32);
                self.f.i64_shl();
                self.read_reg_raw(rd_lo);
                self.f.i64_or();
                self.f.i64_add();
            }
            _ => {}
        }
        if !long {
            self.mask_to(false);
        }
        self.f.local_set(L_R);
        if set_flags {
            self.a32_pack_nz(if long { 63 } else { 31 });
            self.flags_used = true;
            self.f.local_get(L_NZCV);
            self.f.i32_const(0x3000_0000);
            self.f.i32_and();
            self.f.i32_or();
            self.store_nzcv();
        }
        if long {
            self.f.local_get(L_R);
            self.mask_to(false);
            self.store_reg(rd_lo);
            self.f.local_get(L_R);
            self.f.i64_const(32);
            self.f.i64_shr_u();
            self.store_reg(rd_hi);
        } else {
            self.write_reg_from_r(rd_hi);
        }
    }

    /// Push N, from bit `sign` of `L_R`, and Z of `L_R` in their NZCV places, an `i32`.
    fn a32_pack_nz(&mut self, sign: i64) {
        self.f.local_get(L_R);
        self.f.i64_const(sign);
        self.f.i64_shr_u();
        self.f.i32_wrap_i64();
        self.f.i32_const(1);
        self.f.i32_and();
        self.f.i32_const(31);
        self.f.i32_shl();
        self.f.local_get(L_R);
        self.f.i64_eqz();
        self.f.i32_const(30);
        self.f.i32_shl();
        self.f.i32_or();
    }

    /// The adding opcodes into `L_R`; flags, when set, are all four from the adder.
    fn a32_arithmetic(&mut self, opcode: u8, set_flags: bool) {
        // Subtraction adds the inverted operand; reverse subtraction the inverted first.
        let (invert_a, invert_b) = match opcode {
            0x3 | 0x7 => (true, false),
            0x2 | 0x6 | 0xA => (false, true),
            _ => (false, false),
        };
        for (local, invert) in [(L_A, invert_a), (L_B, invert_b)] {
            if invert {
                self.f.local_get(local);
                self.f.i64_const(WORD);
                self.f.i64_xor();
                self.f.local_set(local);
            }
        }
        self.f.local_get(L_A);
        self.f.local_get(L_B);
        self.f.i64_add();
        match opcode {
            0x2 | 0x3 | 0xA => {
                self.f.i64_const(1);
                self.f.i64_add();
            }
            0x5..=0x7 => {
                self.a32_carry_flag();
                self.f.i64_extend_i32_u();
                self.f.i64_add();
            }
            _ => {}
        }
        self.f.local_tee(L_T);
        self.f.i64_const(WORD);
        self.f.i64_and();
        self.f.local_set(L_R);
        if !set_flags && !matches!(opcode, 0xA | 0xB) {
            return;
        }
        self.a32_pack_nz(31);
        self.a32_bit_of_t(32);
        self.f.i32_const(29);
        self.f.i32_shl();
        self.f.i32_or();
        // Overflow: both operands the same sign and the result another.
        self.f.local_get(L_A);
        self.f.local_get(L_R);
        self.f.i64_xor();
        self.f.local_get(L_B);
        self.f.local_get(L_R);
        self.f.i64_xor();
        self.f.i64_and();
        self.f.i64_const(31);
        self.f.i64_shr_u();
        self.f.i32_wrap_i64();
        self.f.i32_const(1);
        self.f.i32_and();
        self.f.i32_const(28);
        self.f.i32_shl();
        self.f.i32_or();
        self.store_nzcv();
    }

    /// The address of a single or extra transfer into `L_ADDR`, and the base's written-back value into `L_W`.
    fn a32_effective_address(&mut self, a: Access, offset: Offset, pc: u32) {
        match offset {
            Offset::Imm(imm) => {
                self.f.i64_const(i64::from(imm));
                self.f.local_set(L_B);
            }
            Offset::Reg { rm, ty, amount } => {
                self.a32_operand(Operand::RegImm { rm, ty, amount }, false, pc)
            }
        }
        self.a32_read(a.rn, pc);
        self.f.local_set(L_A);
        self.f.local_get(L_A);
        self.f.local_get(L_B);
        if a.add {
            self.f.i64_add();
        } else {
            self.f.i64_sub();
        }
        self.f.i64_const(WORD);
        self.f.i64_and();
        self.f.local_set(L_W);
        self.f.local_get(if a.pre { L_W } else { L_A });
        self.f.i32_wrap_i64();
        self.f.local_set(L_ADDR);
    }

    /// A single or extra load or store: the address into `L_ADDR` and the
    /// writeback value into `L_W`, then the access. A load into its own base
    /// keeps the loaded value, except that the extra forms write the base last.
    fn a32_transfer(
        &mut self,
        a: Access,
        acc: Acc,
        offset: Offset,
        base_last: bool,
        at: usize,
        pc: u32,
    ) {
        self.a32_effective_address(a, offset, pc);
        if a.load && a.rt == 15 {
            self.load_branch_target(at);
            if a.writes_base() {
                self.f.local_get(L_W);
                self.store_reg(a.rn);
            }
        } else if a.load {
            self.load(a.rt, acc, at);
            // `ldr r0, [r0], #4` keeps the loaded value.
            if a.writes_base() && (base_last || a.rt != a.rn) {
                self.f.local_get(L_W);
                self.store_reg(a.rn);
            }
        } else {
            self.a32_store(a.rt, acc, at, pc);
            if a.writes_base() {
                self.f.local_get(L_W);
                self.store_reg(a.rn);
            }
        }
    }

    /// A store of `rt`, which may be the PC, to `L_ADDR`.
    fn a32_store(&mut self, rt: u8, acc: Acc, at: usize, pc: u32) {
        self.page_for_write(access_bytes(acc), at);
        self.in_page();
        self.a32_read(rt, pc);
        match acc {
            Acc::Store8 => self.f.i64_store8(0),
            Acc::Store16 => self.f.i64_store16(UNALIGNED, 0),
            _ => self.f.i64_store32(UNALIGNED, 0),
        }
    }

    /// `LDM` or `STM`: each register at its own address from the lowest, the
    /// base written back last.
    fn a32_block_transfer(&mut self, a: Access, list: u16, at: usize, pc: u32) {
        let bytes = i64::from(list.count_ones()) * 4;
        let first = match (a.pre, a.add) {
            (false, true) => 0,
            (true, true) => 4,
            (false, false) => 4 - bytes,
            (true, false) => -bytes,
        };
        self.a32_read(a.rn, pc);
        self.f.local_set(L_A);
        if a.wb {
            self.f.local_get(L_A);
            self.f.i64_const(if a.add { bytes } else { -bytes });
            self.f.i64_add();
            self.f.i64_const(WORD);
            self.f.i64_and();
            self.f.local_set(L_W);
        }
        // Check every access before making any, so handing the instruction
        // back leaves no register or word half done.
        for check in [true, false] {
            let mut offset = first;
            for reg in (0..16u8).filter(|r| list & (1 << r) != 0) {
                self.f.local_get(L_A);
                self.f.i64_const(offset);
                self.f.i64_add();
                self.f.i32_wrap_i64();
                self.f.local_set(L_ADDR);
                match (check, a.load) {
                    (true, true) if reg == 15 => self.load_branch_target(at),
                    (false, true) if reg == 15 => {}
                    (true, true) => self.page_for_read(4, at),
                    (true, false) => self.page_for_write(4, at),
                    (false, true) => self.load(reg, Acc::Load32, at),
                    (false, false) => self.a32_store(reg, Acc::Store32, at, pc),
                }
                offset += 4;
            }
        }
        if a.wb {
            self.f.local_get(L_W);
            self.store_reg(a.rn);
        }
    }
}
