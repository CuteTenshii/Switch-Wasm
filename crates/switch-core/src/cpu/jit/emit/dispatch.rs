//! The per-instruction dispatch.

use super::*;

impl Emitter<'_> {
    /// Push the tick [`Cpu::system_tick`] answers at this instruction. Core 0
    /// reads its own cycles, which `run_jit` settles only when the chain ends.
    pub(super) fn system_tick(&mut self) {
        self.fuel_used = true;
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, self.layout.core);
        self.f.i32_eqz();
        self.f.if_result(I64);
        self.f.local_get(STATE);
        self.f.i64_load(ALIGN_8, self.layout.origin);
        self.f.local_get(L_FUEL);
        self.f.i64_extend_i32_s();
        self.f.i64_sub();
        self.f.i64_const(self.at as i64);
        self.f.i64_add();
        self.f.else_();
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, self.layout.clock_at);
        self.f.i64_atomic_load();
        self.f.end();
        // `cycles * TICK_HZ / CLOCK_HZ` without its 128-bit product, as `q * N + r * N / D`.
        let (n, d) = (i64::from(TICK_HZ), i64::from(CLOCK_RATES_HZ[0]));
        self.f.local_tee(L_T);
        self.f.i64_const(d);
        self.f.i64_div_u();
        self.f.i64_const(n);
        self.f.i64_mul();
        self.f.local_get(L_T);
        self.f.i64_const(d);
        self.f.i64_rem_u();
        self.f.i64_const(n);
        self.f.i64_mul();
        self.f.i64_const(d);
        self.f.i64_div_u();
        self.f.i64_add();
    }

    /// One op, or `false` if there is no way to write it yet.
    pub(super) fn op(&mut self, op: &Op, retired: usize) -> bool {
        match *op {
            Op::Nop => true,

            Op::MovConst { rd, val } => {
                self.f.i64_const(val as i64);
                self.store_reg(rd);
                true
            }

            Op::Mov32 { rd, rn } => {
                self.read_reg(rn, false);
                self.store_reg(rd);
                true
            }

            Op::Mov64 { rd, rn } => {
                self.read_reg(rn, true);
                self.store_reg(rd);
                true
            }

            Op::MovK { rd, shift, val, sf } => {
                // Replace the field, then narrow: a 32-bit MOVK zeroes the top half.
                let keep = !(0xFFFFu64 << shift) & (width_mask(sf) as u64);
                self.read_reg_raw(rd);
                self.f.i64_const(keep as i64);
                self.f.i64_and();
                self.f.i64_const((u64::from(val) << shift) as i64);
                self.f.i64_or();
                self.store_reg(rd);
                true
            }

            Op::AddSubImm {
                rd,
                rn,
                rhs,
                carry,
                set_flags,
                sf,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.f.i64_const((rhs & width_mask(sf) as u64) as i64);
                self.f.local_set(L_B);
                self.add_sub(rd, carry, set_flags, sf);
                true
            }

            Op::AddSubReg {
                rd,
                rn,
                rm,
                carry,
                set_flags,
                sf,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.read_reg(rm, sf);
                self.invert_if(carry, sf);
                self.f.local_set(L_B);
                self.add_sub(rd, carry, set_flags, sf);
                true
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
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.read_reg(rm, sf);
                self.shift_const(st, sa, sf);
                self.invert_if(carry, sf);
                self.f.local_set(L_B);
                self.add_sub(rd, carry, set_flags, sf);
                true
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
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.read_reg_raw(rm);
                self.extend_const(option, sf);
                if shift != 0 {
                    self.f.i64_const(i64::from(shift));
                    self.f.i64_shl();
                    self.mask_to(sf);
                }
                self.invert_if(carry, sf);
                self.f.local_set(L_B);
                self.add_sub(rd, carry, set_flags, sf);
                true
            }

            Op::LogicalImm {
                rd,
                rn,
                imm,
                opc,
                sf,
            } => {
                self.f.i64_const((imm & width_mask(sf) as u64) as i64);
                self.f.local_set(L_B);
                self.logical(rd, rn, opc, sf);
                true
            }

            Op::LogicalReg {
                rd,
                rn,
                rm,
                opc,
                invert,
                sf,
            } => {
                self.read_reg(rm, sf);
                self.invert_if(u8::from(invert), sf);
                self.f.local_set(L_B);
                self.logical(rd, rn, opc, sf);
                true
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
                self.read_reg(rm, sf);
                self.shift_const(st, sa, sf);
                // `BIC`/`ORN`/`EON` invert the shifted operand.
                self.invert_if(u8::from(invert), sf);
                self.f.local_set(L_B);
                self.logical(rd, rn, opc, sf);
                true
            }

            Op::Extract {
                rd,
                rn,
                extract,
                sf,
            } => {
                let (left, right, up, signed) = extract.parts();
                // `left` shifts the high bits out, so read the register whole.
                self.read_reg_raw(rn);
                if left != 0 {
                    self.f.i64_const(i64::from(left));
                    self.f.i64_shl();
                }
                if right != 0 {
                    self.f.i64_const(i64::from(right));
                    if signed {
                        self.f.i64_shr_s();
                    } else {
                        self.f.i64_shr_u();
                    }
                }
                if up != 0 {
                    self.f.i64_const(i64::from(up));
                    self.f.i64_shl();
                }
                self.mask_to(sf);
                self.store_reg(rd);
                true
            }

            Op::Bitfield {
                rd,
                rn,
                opc,
                immr,
                imms,
                sf,
            } => {
                // Only `BFM` reaches here with anything to do.
                if opc != 0b01 {
                    self.f.i64_const(0);
                    self.store_reg(rd);
                    return true;
                }
                let (lsb, msb) = (u32::from(immr), u32::from(imms));
                let (field, right, left) = if msb >= lsb {
                    (mask_of_width(msb - lsb + 1, sf), lsb, 0)
                } else {
                    let up = size_of(sf) - lsb;
                    (mask_of_width(msb + 1, sf).wrapping_shl(up), 0, up)
                };
                self.read_reg(rd, sf);
                self.f.i64_const(!field as i64);
                self.f.i64_and();
                self.read_reg(rn, sf);
                if right != 0 {
                    self.f.i64_const(i64::from(right));
                    self.f.i64_shr_u();
                }
                if left != 0 {
                    self.f.i64_const(i64::from(left));
                    self.f.i64_shl();
                }
                self.f.i64_const(field as i64);
                self.f.i64_and();
                self.f.i64_or();
                self.mask_to(sf);
                self.store_reg(rd);
                true
            }

            Op::Extr {
                rd,
                rn,
                rm,
                imm,
                sf,
            } => {
                let size = size_of(sf);
                let imm = u32::from(imm);
                if imm >= size {
                    return false;
                }
                self.read_reg(rm, sf);
                if imm != 0 {
                    // Rn is the *high* half of the pair being shifted down.
                    self.f.i64_const(i64::from(imm));
                    self.f.i64_shr_u();
                    self.read_reg(rn, sf);
                    self.f.i64_const(i64::from(size - imm));
                    self.f.i64_shl();
                    self.f.i64_or();
                    self.mask_to(sf);
                }
                self.store_reg(rd);
                true
            }

            Op::ShiftVar {
                rd,
                rn,
                rm,
                kind,
                sf,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.read_reg(rm, sf);
                if !sf {
                    // Modulo 32 for a 32-bit shift, not wasm's 64.
                    self.f.i64_const(31);
                    self.f.i64_and();
                }
                self.f.local_set(L_B);
                match kind {
                    0 => {
                        self.f.local_get(L_A);
                        self.f.local_get(L_B);
                        self.f.i64_shl();
                        self.mask_to(sf);
                    }
                    1 => {
                        self.f.local_get(L_A);
                        self.f.local_get(L_B);
                        self.f.i64_shr_u();
                    }
                    2 => {
                        self.f.local_get(L_A);
                        self.sext_to(sf);
                        self.f.local_get(L_B);
                        self.f.i64_shr_s();
                        self.mask_to(sf);
                    }
                    _ if sf => {
                        self.f.local_get(L_A);
                        self.f.local_get(L_B);
                        self.f.i64_rotr();
                    }
                    // A 32-bit variable rotate as two shifts; at zero the left
                    // shift by 32 is discarded by the final narrowing.
                    _ => {
                        self.f.local_get(L_A);
                        self.f.local_get(L_B);
                        self.f.i64_shr_u();
                        self.f.local_get(L_A);
                        self.f.i64_const(32);
                        self.f.local_get(L_B);
                        self.f.i64_sub();
                        self.f.i64_shl();
                        self.f.i64_or();
                        self.mask_to(false);
                    }
                }
                self.store_reg(rd);
                true
            }

            Op::Divide {
                rd,
                rn,
                rm,
                signed,
                sf,
            } => {
                self.read_reg(rn, sf);
                if signed {
                    self.sext_to(sf);
                }
                self.f.local_set(L_A);
                self.read_reg(rm, sf);
                if signed {
                    self.sext_to(sf);
                }
                self.f.local_set(L_B);
                self.divide(signed, sf);
                self.mask_to(sf);
                self.store_reg(rd);
                true
            }

            Op::Madd {
                rd,
                rn,
                rm,
                ra,
                sub,
                sf,
            } => {
                self.multiply_accumulate(rd, ra, sub, sf, |e| {
                    e.read_reg(rn, sf);
                    e.read_reg(rm, sf);
                });
                true
            }

            Op::MaddLong {
                rd,
                rn,
                rm,
                ra,
                sub,
                signed,
            } => {
                self.multiply_accumulate(rd, ra, sub, true, |e| {
                    for r in [rn, rm] {
                        e.read_reg_raw(r);
                        if signed {
                            // Fills the top with the sign, so it narrows too.
                            e.f.i64_extend32_s();
                        } else {
                            e.mask_to(false);
                        }
                    }
                });
                true
            }

            Op::CondSel {
                rd,
                rn,
                rm,
                cond,
                else_inv,
                else_inc,
                sf,
            } => {
                self.read_reg(rn, sf);
                self.read_reg(rm, sf);
                // The invert and increment apply to the else value only.
                if else_inv {
                    self.f.i64_const(-1);
                    self.f.i64_xor();
                }
                if else_inc {
                    self.f.i64_const(1);
                    self.f.i64_add();
                }
                self.cond_holds(cond);
                self.f.select();
                self.mask_to(sf);
                self.store_reg(rd);
                true
            }

            Op::CondCmp {
                rn,
                rm,
                imm,
                cond,
                nzcv,
                sub,
                is_imm,
                sf,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                if is_imm {
                    self.f.i64_const(i64::from(imm));
                } else {
                    // Narrowed, as [`crate::cpu::Cpu::add_carry_overflow`] does.
                    self.read_reg(rm, sf);
                }
                self.invert_if(u8::from(sub), sf);
                self.f.local_set(L_B);
                self.add_carry(u8::from(sub), true, sf);
                // Select between the compare's flags and the immediate ones.
                self.pack_nzcv(sf, true);
                self.f.i32_const((u32::from(nzcv) << 28) as i32);
                self.cond_holds(cond);
                self.f.select();
                self.store_nzcv();
                true
            }

            Op::LoadExclusive { rt, rn, sz } => {
                self.load_exclusive(rt, rn, sz, retired);
                true
            }
            Op::StoreExclusive { rs, rt, rn, sz } => {
                self.store_exclusive(rs, rt, rn, sz, retired);
                true
            }
            Op::LoadExclusivePair { rt, rt2, rn, wide } => {
                self.load_exclusive_pair(rt, rt2, rn, wide, retired);
                true
            }
            Op::StoreExclusivePair {
                rs,
                rt,
                rt2,
                rn,
                wide,
            } => {
                self.store_exclusive_pair(rs, rt, rt2, rn, wide, retired);
                true
            }

            Op::Load64 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.load(rt, Acc::Load64, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Load32 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.load(rt, Acc::Load32, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Load8 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.load(rt, Acc::Load8, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Store64 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.store(rt, Acc::Store64, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Store32 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.store(rt, Acc::Store32, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Store8 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.store(rt, Acc::Store8, retired);
                self.write_back(rn, wb);
                true
            }
            Op::LoadStoreImm {
                rt,
                rn,
                acc,
                wb,
                offset,
            } => {
                self.address(rn, offset, wb);
                self.access(rt, acc, retired);
                self.write_back(rn, wb);
                true
            }
            Op::LoadStoreReg {
                rt,
                rn,
                rm,
                ext,
                shift,
                acc,
            } => {
                self.address_reg(rn, rm, ext, shift);
                self.access(rt, acc, retired);
                true
            }
            Op::PairLoad64 {
                rt,
                rt2,
                rn,
                offset,
                wb,
            } => {
                self.address(rn, offset, wb);
                self.load_pair(rt, rt2, 8, false, retired);
                self.write_back(rn, wb);
                true
            }
            Op::PairStore64 {
                rt,
                rt2,
                rn,
                offset,
                wb,
            } => {
                self.address(rn, offset, wb);
                self.store_pair(rt, rt2, 8, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Pair {
                rt,
                rt2,
                rn,
                offset,
                kind,
                wb,
            } => {
                self.address(rn, offset, wb);
                match kind {
                    PairKind::Load64 => self.load_pair(rt, rt2, 8, false, retired),
                    PairKind::Load32 => self.load_pair(rt, rt2, 4, false, retired),
                    PairKind::Load32Sext => self.load_pair(rt, rt2, 4, true, retired),
                    PairKind::Store64 => self.store_pair(rt, rt2, 8, retired),
                    PairKind::Store32 => self.store_pair(rt, rt2, 4, retired),
                    PairKind::LoadQ => self.load_pair_q(rt, rt2, retired),
                    PairKind::StoreQ => self.store_pair_q(rt, rt2, retired),
                }
                self.write_back(rn, wb);
                true
            }
            Op::Fp { insn, scalar, form } => {
                if scalar {
                    self.fp(insn, form, retired)
                } else {
                    self.simd(insn)
                }
            }
            Op::SimdLoadStore { insn } | Op::VecImm { insn, .. } | Op::VecPair { insn, .. } => {
                self.simd_load_store(insn, retired)
            }
            Op::LoadLiteral { rt, addr, acc } => {
                self.f.i32_const(addr as i32);
                self.set_addr32();
                self.access(rt, acc, retired);
                true
            }
            Op::Ordered { rt, rn, acc } => {
                self.address(rn, 0, Wb::None);
                self.f.atomic_fence();
                self.access(rt, acc, retired);
                self.f.atomic_fence();
                true
            }
            Op::Sys { op: SysOp::Barrier } => {
                self.f.atomic_fence();
                true
            }
            Op::Sys {
                op: SysOp::ClearExclusive,
            } => {
                self.f.local_get(STATE);
                self.f.i32_const(0);
                self.f
                    .i32_store(ALIGN_4, self.layout.exclusive + MONITOR_ARMED);
                true
            }
            Op::Sys {
                op: SysOp::Mrs { rd, reg },
            } => match reg {
                SysReg::Tpidr | SysReg::TpidrRo => {
                    let at = if reg == SysReg::Tpidr {
                        self.layout.tpidr_rw
                    } else {
                        self.layout.tpidr
                    };
                    self.f.local_get(STATE);
                    self.f.i64_load(ALIGN_8, at);
                    self.store_reg(rd);
                    true
                }
                SysReg::Fixed(value) => {
                    self.f.i64_const(i64::from(value));
                    self.store_reg(rd);
                    true
                }
                SysReg::SystemTick => {
                    self.system_tick();
                    self.store_reg(rd);
                    true
                }
                _ => false,
            },
            Op::Mulh { rd, rn, rm, signed } => {
                self.mulh(rd, rn, rm, signed);
                true
            }
            Op::OneSource { rd, rn, opcode, sf } => self.one_source(rd, rn, opcode, sf),

            _ => false,
        }
    }
}
