//! Integer arithmetic, flags, shifts and multiplies.

use super::*;

impl Emitter<'_> {
    /// Write the local `L_R` into `regs[slot]`.
    pub(super) fn write_reg_from_r(&mut self, slot: u8) {
        self.f.local_get(L_R);
        self.store_reg(slot);
    }

    /// Invert the value on the stack when the operation subtracts, as
    /// [`super::exec`]'s `invert_if` does, and narrow it again.
    pub(super) fn invert_if(&mut self, carry: u8, sf: bool) {
        if carry != 0 {
            self.f.i64_const(-1);
            self.f.i64_xor();
            self.mask_to(sf);
        }
    }

    /// Push 1 if `cond` holds under the current NZCV and 0 if it does not:
    /// its [`CONDITION_MASKS`] row shifted by the flags.
    pub(super) fn cond_holds(&mut self, cond: u8) {
        self.f
            .i32_const(i32::from(CONDITION_MASKS[(cond & 0xF) as usize]));
        self.flags_used = true;
        self.f.local_get(L_NZCV);
        self.f.i32_const(28);
        self.f.i32_shr_u();
        self.f.i32_shr_u();
        self.f.i32_const(1);
        self.f.i32_and();
    }

    /// [`crate::cpu::bits::shift_reg`] applied to the value on the stack. An
    /// out-of-range distance answers as the interpreter does.
    pub(super) fn shift_const(&mut self, st: u8, sa: u8, sf: bool) {
        let size = size_of(sf);
        let sa = u32::from(sa);
        match st {
            0 => {
                if sa >= size {
                    self.f.drop_value();
                    self.f.i64_const(0);
                } else if sa != 0 {
                    self.f.i64_const(i64::from(sa));
                    self.f.i64_shl();
                    self.mask_to(sf);
                }
            }
            1 => {
                if sa >= size {
                    self.f.drop_value();
                    self.f.i64_const(0);
                } else if sa != 0 {
                    self.f.i64_const(i64::from(sa));
                    self.f.i64_shr_u();
                }
            }
            2 => {
                if sa != 0 {
                    self.sext_to(sf);
                    self.f.i64_const(i64::from(sa.min(size - 1)));
                    self.f.i64_shr_s();
                    self.mask_to(sf);
                }
            }
            _ => {
                let sa = sa % size;
                if sa != 0 {
                    self.rotate_right_const(sa, sf);
                }
            }
        }
    }

    /// Rotate the value on the stack right by `sa`, nonzero and in range. The
    /// 32-bit form is written as two shifts, using `L_S`.
    pub(super) fn rotate_right_const(&mut self, sa: u32, sf: bool) {
        if sf {
            self.f.i64_const(i64::from(sa));
            self.f.i64_rotr();
            return;
        }
        self.f.local_tee(L_S);
        self.f.i64_const(i64::from(sa));
        self.f.i64_shr_u();
        self.f.local_get(L_S);
        self.f.i64_const(i64::from(32 - sa));
        self.f.i64_shl();
        self.f.i64_or();
        self.mask_to(false);
    }

    /// [`crate::cpu::bits::extend_reg`] applied to the value on the stack.
    pub(super) fn extend_const(&mut self, option: u8, sf: bool) {
        match option & 0b111 {
            0b000 => {
                self.f.i64_const(0xFF);
                self.f.i64_and();
            }
            0b001 => {
                self.f.i64_const(0xFFFF);
                self.f.i64_and();
            }
            0b010 => {
                self.f.i64_const(width_mask(false));
                self.f.i64_and();
            }
            0b100 => self.f.i64_extend8_s(),
            0b101 => self.f.i64_extend16_s(),
            0b110 => self.f.i64_extend32_s(),
            // UXTX and SXTX are the whole register.
            _ => {}
        }
        self.mask_to(sf);
    }

    /// `a + b + carry` from `L_A` and `L_B` into `L_R`, narrowed, with the
    /// carry out in `L_C` when flags are wanted. Only 64-bit operations need
    /// the explicit carry chain.
    pub(super) fn add_carry(&mut self, carry: u8, set_flags: bool, sf: bool) {
        let chain = set_flags && sf;

        self.f.local_get(L_A);
        self.f.local_get(L_B);
        self.f.i64_add();
        self.f.local_set(L_T);

        if chain {
            // c1 = t <u a, the carry out of the first add
            self.f.local_get(L_T);
            self.f.local_get(L_A);
            self.f.i64_lt_u();
            self.f.local_set(L_C);
        }

        if carry != 0 {
            if chain {
                // c = c1 | (t + 1 <u t), and the two can never both be set
                self.f.local_get(L_T);
                self.f.i64_const(1);
                self.f.i64_add();
                self.f.local_get(L_T);
                self.f.i64_lt_u();
                self.f.local_get(L_C);
                self.f.i32_or();
                self.f.local_set(L_C);
            }
            self.f.local_get(L_T);
            self.f.i64_const(1);
            self.f.i64_add();
            self.f.local_set(L_T);
        }

        self.f.local_get(L_T);
        self.mask_to(sf);
        self.f.local_set(L_R);

        if set_flags && !sf {
            self.f.local_get(L_T);
            self.f.i64_const(32);
            self.f.i64_shr_u();
            self.f.i32_wrap_i64();
            self.f.i32_const(1);
            self.f.i32_and();
            self.f.local_set(L_C);
        }
    }

    /// `ADD`/`SUB`/`ADDS`/`SUBS` with operands in `L_A` and `L_B`, as
    /// [`crate::cpu::Cpu::add_sub_pre`] receives them.
    pub(super) fn add_sub(&mut self, rd: u8, carry: u8, set_flags: bool, sf: bool) {
        self.add_carry(carry, set_flags, sf);
        if set_flags {
            self.pack_nzcv(sf, true);
            self.store_nzcv();
        }
        self.write_reg_from_r(rd);
    }

    /// Push the packed NZCV word from `L_R` and, when `arithmetic`, C and V
    /// from `L_A`, `L_B` and `L_C`; otherwise C and V are kept (`ANDS`).
    pub(super) fn pack_nzcv(&mut self, sf: bool, arithmetic: bool) {
        let shift = if sf { 63 } else { 31 };

        // N
        self.f.local_get(L_R);
        self.f.i64_const(shift);
        self.f.i64_shr_u();
        self.f.i32_wrap_i64();
        self.f.i32_const(1);
        self.f.i32_and();
        self.f.i32_const(31);
        self.f.i32_shl();

        // Z
        self.f.local_get(L_R);
        self.f.i64_eqz();
        self.f.i32_const(30);
        self.f.i32_shl();
        self.f.i32_or();

        if arithmetic {
            // C
            self.f.local_get(L_C);
            self.f.i32_const(29);
            self.f.i32_shl();
            self.f.i32_or();

            // V: both operands the same sign and the result a different one.
            self.f.local_get(L_A);
            self.f.local_get(L_B);
            self.f.i64_xor();
            self.f.i64_const(-1);
            self.f.i64_xor();
            self.f.local_get(L_A);
            self.f.local_get(L_R);
            self.f.i64_xor();
            self.f.i64_and();
            self.f.i64_const(sign_bit(sf));
            self.f.i64_and();
            self.f.i64_const(0);
            self.f.i64_ne();
            self.f.i32_const(28);
            self.f.i32_shl();
            self.f.i32_or();
        } else {
            // `ANDS` leaves C and V exactly as they were.
            self.flags_used = true;
            self.f.local_get(L_NZCV);
            self.f.i32_const(0x3000_0000);
            self.f.i32_and();
            self.f.i32_or();
        }
    }

    /// Store the packed word on the stack into NZCV.
    pub(super) fn store_nzcv(&mut self) {
        self.flags_used = true;
        self.flags_written = true;
        self.f.local_set(L_NZCV);
        if self.pend_kind != 0 {
            self.f.i32_const(0);
            self.f.local_set(self.pend_kind);
        }
    }

    /// Build NZCV from the compare held in the pending locals.
    pub(super) fn materialize_pending(&mut self, sf: bool) {
        self.f.local_get(self.pend_a);
        self.f.local_set(L_A);
        self.f.local_get(self.pend_b);
        self.invert_if(1, sf);
        self.f.local_set(L_B);
        self.add_carry(1, true, sf);
        self.pack_nzcv(sf, true);
        self.store_nzcv();
    }

    /// Whether `cond` after a `CMP` can be tested on the operands, and the
    /// wasm comparison that does it.
    pub(super) fn direct_compare(cond: u8) -> Option<(u8, bool)> {
        match cond & 0xF {
            0 => Some((0x51, false)),
            1 => Some((0x52, false)),
            2 => Some((I64_GE_U, false)),
            3 => Some((0x54, false)),
            8 => Some((I64_GT_U, false)),
            9 => Some((I64_LE_U, false)),
            10 => Some((I64_GE_S, true)),
            11 => Some((I64_LT_S, true)),
            12 => Some((I64_GT_S, true)),
            13 => Some((I64_LE_S, true)),
            _ => None,
        }
    }

    /// A fused `CMP` and branch whose flags nothing reads: the condition is
    /// tested on the operands, which the caller left in the pending locals,
    /// and NZCV is built only where the run could leave the compare behind.
    pub(super) fn lazy_branch(&mut self, cond: u8, sf: bool, retired: usize, target: u32) {
        let (compare, signed) =
            Self::direct_compare(cond).expect("the caller checked the condition");
        self.pend_forms |= 1 << u8::from(sf);
        self.f.i32_const(1 + i32::from(sf));
        self.f.local_set(self.pend_kind);
        self.flags_used = true;
        self.flags_written = true;
        for local in [self.pend_a, self.pend_b] {
            self.f.local_get(local);
            if signed && !sf {
                self.f.i64_extend32_s();
            }
        }
        self.f.op_raw(compare);
        self.f.if_void();
        if self.inside(target).is_some() {
            self.materialize_pending(sf);
        }
        self.goto(retired, target, false);
        self.f.end();
    }

    /// The locals a held compare's operands and form live in, assigned on first use.
    pub(super) fn pending_locals(&mut self) {
        if self.pend_kind == 0 {
            self.pend_a = self.fresh_local(I64);
            self.pend_b = self.fresh_local(I64);
            self.pend_kind = self.fresh_local(I32);
        }
    }

    /// `AND`/`ORR`/`EOR`/`ANDS` with the second operand already in `L_B`.
    pub(super) fn logical(&mut self, rd: u8, rn: u8, opc: u8, sf: bool) {
        self.read_reg(rn, sf);
        self.f.local_get(L_B);
        match opc {
            0b00 | 0b11 => self.f.i64_and(),
            0b01 => self.f.i64_or(),
            _ => self.f.i64_xor(),
        }
        self.f.local_set(L_R);
        if opc == 0b11 {
            self.pack_nzcv(sf, false);
            self.store_nzcv();
        }
        self.write_reg_from_r(rd);
    }

    /// `MADD`/`MSUB` and the widening `SMADDL`/`UMADDL` family. A 32x32
    /// product fits in 64 bits.
    pub(super) fn multiply_accumulate(
        &mut self,
        rd: u8,
        ra: u8,
        sub: bool,
        sf: bool,
        operands: impl Fn(&mut Self),
    ) {
        self.read_reg(ra, sf);
        operands(self);
        self.f.i64_mul();
        if sub {
            self.f.i64_sub();
        } else {
            self.f.i64_add();
        }
        self.mask_to(sf);
        self.store_reg(rd);
    }

    /// `UDIV`/`SDIV`, `L_A` by `L_B`. Division by zero answers zero and
    /// `i64::MIN / -1` wraps, guarded with real branches since wasm traps.
    pub(super) fn divide(&mut self, signed: bool, sf: bool) {
        self.f.local_get(L_B);
        self.f.i64_eqz();
        self.f.if_result(I64);
        self.f.i64_const(0);
        self.f.else_();
        if signed && sf {
            self.f.local_get(L_B);
            self.f.i64_const(-1);
            self.f.i64_eq();
            self.f.if_result(I64);
            // `x.wrapping_div(-1)` is `x.wrapping_neg()`.
            self.f.i64_const(0);
            self.f.local_get(L_A);
            self.f.i64_sub();
            self.f.else_();
            self.f.local_get(L_A);
            self.f.local_get(L_B);
            self.f.i64_div_s();
            self.f.end();
        } else {
            self.f.local_get(L_A);
            self.f.local_get(L_B);
            if signed {
                self.f.i64_div_s();
            } else {
                self.f.i64_div_u();
            }
        }
        self.f.end();
    }

    /// `SMULH`/`UMULH`: the high half of the 128-bit product, from 32-bit partial products.
    pub(super) fn mulh(&mut self, rd: u8, rn: u8, rm: u8, signed: bool) {
        const LOW: i64 = 0xFFFF_FFFF;
        self.read_reg_raw(rn);
        self.f.local_tee(L_A);
        self.f.i64_const(LOW);
        self.f.i64_and();
        self.f.local_set(L_T);
        self.f.local_get(L_A);
        self.f.i64_const(32);
        self.f.i64_shr_u();
        self.f.local_set(L_S);
        self.read_reg_raw(rm);
        self.f.local_tee(L_B);
        self.f.i64_const(LOW);
        self.f.i64_and();
        self.f.local_set(L_R);
        self.f.local_get(L_B);
        self.f.i64_const(32);
        self.f.i64_shr_u();
        self.f.local_set(L_W);
        // hi = a1*b1 + (a0*b1 >> 32) + (a1*b0 >> 32) + (mid >> 32)
        self.product(L_S, L_W);
        self.product(L_T, L_W);
        self.f.i64_const(32);
        self.f.i64_shr_u();
        self.f.i64_add();
        self.product(L_S, L_R);
        self.f.i64_const(32);
        self.f.i64_shr_u();
        self.f.i64_add();
        // mid = (a0*b0 >> 32) + (a0*b1 & low) + (a1*b0 & low)
        self.product(L_T, L_R);
        self.f.i64_const(32);
        self.f.i64_shr_u();
        self.product(L_T, L_W);
        self.f.i64_const(LOW);
        self.f.i64_and();
        self.f.i64_add();
        self.product(L_S, L_R);
        self.f.i64_const(LOW);
        self.f.i64_and();
        self.f.i64_add();
        self.f.i64_const(32);
        self.f.i64_shr_u();
        self.f.i64_add();
        if signed {
            // Subtract each operand where the other is negative.
            for (sign, other) in [(L_A, L_B), (L_B, L_A)] {
                self.f.local_get(sign);
                self.f.i64_const(63);
                self.f.i64_shr_s();
                self.f.local_get(other);
                self.f.i64_and();
                self.f.i64_sub();
            }
        }
        self.store_reg(rd);
    }

    /// Push the product of two locals.
    pub(super) fn product(&mut self, a: u32, b: u32) {
        self.f.local_get(a);
        self.f.local_get(b);
        self.f.i64_mul();
    }

    /// `RBIT`, `REV16`, `REV32`, `REV` and `CLZ`; false for the rest.
    pub(super) fn one_source(&mut self, rd: u8, rn: u8, opcode: u8, sf: bool) -> bool {
        // A 32-bit `REV` of a doubleword is not an encoding.
        if opcode > 4 || (opcode == 3 && !sf) {
            return false;
        }
        self.read_reg(rn, sf);
        if opcode == 4 {
            self.f.i64_clz();
            if !sf {
                self.f.i64_const(32);
                self.f.i64_sub();
            }
            self.store_reg(rd);
            return true;
        }
        self.f.local_set(L_T);
        // Swap `shift`-bit groups with their neighbours; the bit reversal is these down to
        // single bits, then the byte reversal.
        let swaps: &[(i64, i64)] = match opcode {
            0 => &[
                (0x5555_5555_5555_5555, 1),
                (0x3333_3333_3333_3333, 2),
                (0x0F0F_0F0F_0F0F_0F0F, 4),
                (0x00FF_00FF_00FF_00FF, 8),
                (0x0000_FFFF_0000_FFFF, 16),
            ],
            1 => &[(0x00FF_00FF_00FF_00FF, 8)],
            _ => &[(0x00FF_00FF_00FF_00FF, 8), (0x0000_FFFF_0000_FFFF, 16)],
        };
        for &(mask, shift) in swaps {
            self.f.local_get(L_T);
            self.f.i64_const(shift);
            self.f.i64_shr_u();
            self.f.i64_const(mask);
            self.f.i64_and();
            self.f.local_get(L_T);
            self.f.i64_const(mask);
            self.f.i64_and();
            self.f.i64_const(shift);
            self.f.i64_shl();
            self.f.i64_or();
            self.f.local_set(L_T);
        }
        self.f.local_get(L_T);
        // The last step swaps the halves of the doubleword: a bit reversal and `REV` need it.
        if opcode == 0 || opcode == 3 {
            self.f.i64_const(32);
            self.f.i64_rotr();
        }
        // A 32-bit bit reversal ends up in the upper half.
        if !sf && opcode == 0 {
            self.f.i64_const(32);
            self.f.i64_shr_u();
        }
        self.store_reg(rd);
        true
    }
}
