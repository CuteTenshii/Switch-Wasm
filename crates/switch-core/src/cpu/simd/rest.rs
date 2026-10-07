//! The AdvSIMD decode that continues past `try_simd`.

use super::*;

impl Cpu {
    pub(super) fn try_simd_rest(&mut self, insn: u32) -> Result<bool> {
        // EXT: bytes from Vm:Vn starting at imm4.
        if ((insn >> 31) & 1) == 0
            && ((insn >> 24) & 0x3F) == 0b101110
            && ((insn >> 22) & 0b11) == 0
            && ((insn >> 21) & 1) == 0
            && ((insn >> 15) & 1) == 0
            && ((insn >> 10) & 1) == 0
        {
            let q = (insn >> 30) & 1 == 1;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let imm4 = (insn >> 11) & 0xF;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rd = (insn & 0x1F) as u8;
            if !q && imm4 & 0b1000 != 0 {
                return Ok(false); // an 8-byte result can't start 8+ bytes in
            }
            let shift = imm4 * 8;
            let n = self.vregs[rn as usize];
            let m = self.vregs[rm as usize];
            self.vregs[rd as usize] = if q {
                if shift == 0 {
                    n
                } else {
                    (n >> shift) | (m << (128 - shift))
                }
            } else {
                let low = elem_mask(64);
                (((n & low) | ((m & low) << 64)) >> shift) & low
            };
            return Ok(true);
        }

        // Two-register misc (vector).
        if ((insn >> 31) & 1) == 0
            && ((insn >> 24) & 0x1F) == 0b01110
            && ((insn >> 17) & 0x1F) == 0b10000
            && ((insn >> 10) & 0b11) == 0b10
        {
            return self.simd_two_reg_misc(insn, false);
        }
        // Scalar FP three-same.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 1) == 1
            && ((insn >> 11) & 0x1F) >= 0b11000
        {
            return self.simd_fp_three_same(insn, true);
        }
        // Scalar integer three-same: variable shifts.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 1) == 1
            && (0b01000..=0b01011).contains(&((insn >> 11) & 0x1F))
        {
            let op = (insn >> 11) & 0x1F;
            let size = (insn >> 22) & 0b11;
            let saturating = op & 1 == 1;
            if !saturating && size != 0b11 {
                return Ok(false); // SSHL/SRSHL have no narrow scalar form
            }
            let esize = 8u32 << size;
            let unsigned = (insn >> 29) & 1 == 1;
            let rounding = op & 0b10 != 0;
            let a = self.vregs[((insn >> 5) & 0x1F) as usize] as u64;
            let b = self.vregs[((insn >> 16) & 0x1F) as usize] as u64;
            let v = shift_by_reg(a, b, esize, unsigned, rounding, saturating);
            self.vregs[(insn & 0x1F) as usize] = u128::from(v);
            return Ok(true);
        }
        // Scalar pairwise: ADDP, FADDP, FMAXP, FMINP, FMAXNMP, FMINNMP.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 17) & 0x1F) == 0b11000
            && ((insn >> 10) & 0b11) == 0b10
        {
            let unsigned = (insn >> 29) & 1 == 1;
            let opcode = (insn >> 12) & 0x1F;
            let a = self.vregs[((insn >> 5) & 0x1F) as usize];
            let rd = (insn & 0x1F) as usize;
            if !unsigned {
                if opcode != 0b11011 || (insn >> 22) & 0b11 != 0b11 {
                    return Ok(false); // ADDP only, and only on doublewords
                }
                self.vregs[rd] = u128::from((a as u64).wrapping_add((a >> 64) as u64));
                return Ok(true);
            }
            let double = (insn >> 22) & 1 == 1;
            let min = (insn >> 23) & 1 == 1;
            let op = |x: f64, y: f64| -> Option<f64> {
                Some(match (opcode, min) {
                    (0b01100, false) => fp_maxnum(x, y),
                    (0b01100, true) => fp_minnum(x, y),
                    (0b01101, false) => x + y,
                    (0b01111, false) => fp_max(x, y),
                    (0b01111, true) => fp_min(x, y),
                    _ => return None,
                })
            };
            self.vregs[rd] = if double {
                let lane = |i: u32| f64::from_bits((a >> (64 * i)) as u64);
                let Some(r) = op(lane(0), lane(1)) else {
                    return Ok(false);
                };
                u128::from(r.to_bits())
            } else {
                let lane = |i: u32| f32::from_bits((a >> (32 * i)) as u32);
                // Computed via widening to match the vector pairwise forms.
                let Some(r) = op(f64::from(lane(0)), f64::from(lane(1))) else {
                    return Ok(false);
                };
                u128::from((r as f32).to_bits())
            };
            return Ok(true);
        }
        // Scalar integer three-same: compares and ADD/SUB on one doubleword.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 1) == 1
            && matches!((insn >> 11) & 0x1F, 0b00110 | 0b00111 | 0b10000 | 0b10001)
        {
            if (insn >> 22) & 0b11 != 0b11 {
                return Ok(false); // no byte, half or word scalar form
            }
            let unsigned = (insn >> 29) & 1 == 1;
            let a = self.vregs[((insn >> 5) & 0x1F) as usize] as u64;
            let b = self.vregs[((insn >> 16) & 0x1F) as usize] as u64;
            let all = |holds: bool| if holds { u64::MAX } else { 0 };
            let v = match ((insn >> 11) & 0x1F, unsigned) {
                (0b00110, false) => all((a as i64) > (b as i64)), // CMGT
                (0b00110, true) => all(a > b),                    // CMHI
                (0b00111, false) => all((a as i64) >= (b as i64)), // CMGE
                (0b00111, true) => all(a >= b),                   // CMHS
                (0b10000, false) => a.wrapping_add(b),            // ADD
                (0b10000, true) => a.wrapping_sub(b),             // SUB
                (0b10001, false) => all(a & b != 0),              // CMTST
                _ => all(a == b),                                 // CMEQ
            };
            self.vregs[(insn & 0x1F) as usize] = u128::from(v);
            return Ok(true);
        }
        // Two-register misc, scalar forms.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 17) & 0x1F) == 0b10000
            && ((insn >> 10) & 0b11) == 0b10
        {
            return self.simd_two_reg_misc(insn, true);
        }

        // Integer three-same / compare / logical (vector).
        let grp = (insn >> 24) & 0x1F;
        // Copy group (DUP/INS/UMOV/SMOV) is told apart from three-same by bit21 == 0.
        let copy_group = ((insn >> 21) & 0xFF) == 0b01110000 && ((insn >> 31) & 1) == 0;
        if ((insn >> 31) & 1) == 0 && grp == 0b01110 && !copy_group {
            let q = (insn >> 30) & 1 == 1;
            let rd = (insn & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let sz = (insn >> 22) & 0b11;
            let u = (insn >> 29) & 1; // 0 → 0x4e group, 1 → 0x6e group
            let op = (insn >> 11) & 0x1F;
            let b10 = (insn >> 10) & 1;
            let esize = match sz {
                0 => 8u32,
                1 => 16,
                2 => 32,
                _ => 64,
            };
            // Across lanes (SMAXV etc.): bits[21:17] = 11000, bit10 = 0.
            if b10 == 0 && ((insn >> 17) & 0x1F) == 0b11000 {
                return self.simd_across_lanes(insn);
            }
            // Opcodes from 0b11000 up are FP.
            if b10 == 1 && ((insn >> 21) & 1) == 1 && op >= 0b11000 {
                return self.simd_fp_three_same(insn, false);
            }
            if b10 == 1 {
                match op {
                    0b00000 => {
                        // SHADD / UHADD
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                ((a as i128 + b as i128) >> 1) as u64
                            } else {
                                a.wrapping_add(b) >> 1
                            }
                        });
                        return Ok(true);
                    }
                    0b00001 => {
                        // SQADD (signed group) / UQADD (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            saturating_add(a, b, esize, u != 0)
                        });
                        return Ok(true);
                    }
                    0b00010 => {
                        // SRHADD / URHADD: rounding halving add.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                ((a as i128 + b as i128 + 1) >> 1) as u64
                            } else {
                                a.wrapping_add(b).wrapping_add(1) >> 1
                            }
                        });
                        return Ok(true);
                    }
                    0b00100 => {
                        // SHSUB / UHSUB: halving subtract.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                ((a as i128 - b as i128) >> 1) as u64
                            } else {
                                a.wrapping_sub(b) >> 1
                            }
                        });
                        return Ok(true);
                    }
                    0b00101 => {
                        // SQSUB / UQSUB: saturating subtract.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            saturating_sub(a, b, esize, u != 0)
                        });
                        return Ok(true);
                    }
                    // Variable shifts: opcode low bits are the saturate and round flags.
                    0b01000..=0b01011 => {
                        let saturating = op & 1 == 1;
                        let rounding = op & 0b10 != 0;
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            shift_by_reg(a, b, esize, u != 0, rounding, saturating)
                        });
                        return Ok(true);
                    }
                    0b10000 => {
                        // ADD (signed group) / SUB (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                a.wrapping_add(b)
                            } else {
                                a.wrapping_sub(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b10001 => {
                        // CMTST (signed group) / CMEQ (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if a & b != 0 {
                                    u64::MAX
                                } else {
                                    0
                                }
                            } else if a == b {
                                u64::MAX
                            } else {
                                0
                            }
                        });
                        return Ok(true);
                    }
                    0b00111 => {
                        // CMGE (signed group) / CMHS (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            let ge = if u == 0 {
                                Self::sge(a, b, esize)
                            } else {
                                a >= b
                            };
                            if ge {
                                u64::MAX
                            } else {
                                0
                            }
                        });
                        return Ok(true);
                    }
                    0b00110 => {
                        // CMGT (signed group) / CMHI (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            let gt = if u == 0 {
                                Self::sge(a, b, esize) && a != b
                            } else {
                                a > b
                            };
                            if gt {
                                u64::MAX
                            } else {
                                0
                            }
                        });
                        return Ok(true);
                    }
                    0b01100 => {
                        // SMAX / UMAX.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if Self::sge(a, b, esize) {
                                    a
                                } else {
                                    b
                                }
                            } else {
                                a.max(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b01101 => {
                        // SMIN / UMIN.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if Self::sge(a, b, esize) {
                                    b
                                } else {
                                    a
                                }
                            } else {
                                a.min(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b01110 => {
                        // SABD / UABD: absolute difference.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            simd_abs_diff(a, b, esize, u != 0)
                        });
                        return Ok(true);
                    }
                    0b01111 => {
                        // SABA / UABA: accumulate the absolute difference.
                        self.simd_elem_acc(rd, rn, rm, q, esize, |a, b, d| {
                            d.wrapping_add(simd_abs_diff(a, b, esize, u != 0))
                        });
                        return Ok(true);
                    }
                    0b10010 => {
                        // MLA (signed group) / MLS (unsigned group).
                        self.simd_elem_acc(rd, rn, rm, q, esize, |a, b, d| {
                            let product = a.wrapping_mul(b);
                            if u == 0 {
                                d.wrapping_add(product)
                            } else {
                                d.wrapping_sub(product)
                            }
                        });
                        return Ok(true);
                    }
                    0b10011 if u == 0 => {
                        // MUL (PMUL, U=1, is 8-bit polynomial)
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| a.wrapping_mul(b));
                        return Ok(true);
                    }
                    0b10110 => {
                        // SQDMULH / SQRDMULH
                        let rounding = u == 1;
                        self.simd_elem(rd, rn, rm, q, esize, move |a, b| {
                            let a = sext_u64(a, esize) as i64 as i128;
                            let b = sext_u64(b, esize) as i64 as i128;
                            let mut product = 2 * a * b;
                            if rounding {
                                product += 1i128 << (esize - 1);
                            }
                            let max = (1i128 << (esize - 1)) - 1;
                            let min = -(1i128 << (esize - 1));
                            ((product >> esize).clamp(min, max)) as u64
                        });
                        return Ok(true);
                    }
                    0b10111 if u == 0 => {
                        // ADDP: pairwise addition.
                        self.simd_pairwise(rd, rn, rm, q, esize, |a, b| a.wrapping_add(b));
                        return Ok(true);
                    }
                    0b10100 => {
                        // SMAXP (signed group) / UMAXP (unsigned group).
                        self.simd_pairwise(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if Self::sge(a, b, esize) {
                                    a
                                } else {
                                    b
                                }
                            } else {
                                a.max(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b10101 => {
                        // SMINP (signed group) / UMINP (unsigned group).
                        self.simd_pairwise(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if Self::sge(a, b, esize) {
                                    b
                                } else {
                                    a
                                }
                            } else {
                                a.min(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b00011 => {
                        // Bitwise logicals: selector in bits[23:21].
                        let sub = (insn >> 21) & 0b111;
                        let a = self.vregs[rn as usize];
                        let b = self.vregs[rm as usize];
                        let full = if q { u128::MAX } else { (1u128 << 64) - 1 };
                        let d = self.vregs[rd as usize];
                        let r = match (u, sub) {
                            (0, 0b001) => a & b,  // AND
                            (0, 0b011) => a & !b, // BIC
                            (0, 0b101) => a | b,  // ORR
                            (0, 0b111) => a | !b, // ORN
                            (1, 0b001) => a ^ b,  // EOR
                            // BSL masks with Vd; BIT and BIF with Vm.
                            (1, 0b011) => (a & d) | (b & !d), // BSL
                            (1, 0b101) => (a & b) | (d & !b), // BIT
                            (1, 0b111) => (a & !b) | (d & b), // BIF
                            _ => return Ok(false),
                        };
                        self.vregs[rd as usize] = r & full;
                        return Ok(true);
                    }
                    _ => {}
                }
            } else if op == 0b10011 && rm == 0 && u == 0 {
                // CMEQ <Vd>.<T>, <Vn>.<T>, #0 (compare against zero).
                self.simd_elem(
                    rd,
                    rn,
                    rm,
                    q,
                    esize,
                    |a, _| {
                        if a == 0 {
                            u64::MAX
                        } else {
                            0
                        }
                    },
                );
                return Ok(true);
            }
            return Err(Error::Cpu(format!(
                "unimplemented SIMD three-same u={} op={:#b} sz={} at {:#x}",
                u, op, sz, self.pc
            )));
        }

        // Scalar DUP (element): lane to the bottom, rest zeroed. Matched before the vector copy group.
        if ((insn >> 21) & 0xFF) == 0b11110000
            && ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 10) & 0b111111) == 0b000001
        {
            let imm5 = (insn >> 16) & 0x1F;
            let lsb = imm5.trailing_zeros();
            // Reserved imm5 encodings.
            if imm5 == 0 || lsb > 3 {
                return Ok(false);
            }
            let esize = 8u32 << lsb;
            let index = imm5 >> (lsb + 1);
            let rd = (insn & 0x1F) as usize;
            let rn = ((insn >> 5) & 0x1F) as usize;
            let mask = (1u128 << esize) - 1;
            self.vregs[rd] = (self.vregs[rn] >> (index * esize)) & mask;
            return Ok(true);
        }

        // Copy / element moves and table lookup (bits[28:21] == 0111 0000).
        if ((insn >> 21) & 0xFF) != 0b01110000 || ((insn >> 31) & 1) != 0 {
            return Ok(false);
        }
        let op = (insn >> 29) & 1;

        // TBL / TBX: split off from the copy group by bits[11:10] = 00.
        if op == 0 && ((insn >> 15) & 1) == 0 && ((insn >> 10) & 0b11) == 0 {
            let q = (insn >> 30) & 1 == 1;
            let rd = (insn & 0x1F) as usize;
            let rn = ((insn >> 5) & 0x1F) as usize;
            let rm = ((insn >> 16) & 0x1F) as usize;
            let len = ((insn >> 13) & 0b11) as usize + 1;
            // Out-of-range index: TBL writes zero, TBX keeps Vd.
            let keep_on_miss = (insn >> 12) & 1 == 1;
            let indices = self.vregs[rm].to_le_bytes();
            let mut out = self.vregs[rd].to_le_bytes();
            let lanes = if q { 16 } else { 8 };
            for (i, slot) in out.iter_mut().enumerate().take(lanes) {
                let idx = indices[i] as usize;
                if idx < len * 16 {
                    // The table wraps past v31.
                    *slot = self.vregs[(rn + idx / 16) % 32].to_le_bytes()[idx % 16];
                } else if !keep_on_miss {
                    *slot = 0;
                }
            }
            out[lanes..].fill(0);
            self.vregs[rd] = u128::from_le_bytes(out);
            return Ok(true);
        }

        let q = (insn >> 30) & 1;
        let rd = (insn & 0x1F) as u8;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let imm5 = (insn >> 16) & 0x1F;

        if op == 1 {
            // INS (element): imm5 gives size and destination index, imm4 the source index.
            if q == 0 || imm5 == 0 || ((insn >> 15) & 1) != 0 || ((insn >> 10) & 1) == 0 {
                return Ok(false);
            }
            let lsb = imm5.trailing_zeros();
            if lsb > 3 {
                return Ok(false);
            }
            let esize = 8u32 << lsb;
            let dst_index = imm5 >> (lsb + 1);
            let src_index = ((insn >> 11) & 0xF) >> lsb;
            let mask = (1u128 << esize) - 1;
            let val = (self.vregs[rn as usize] >> (src_index * esize)) & mask;
            let shift = dst_index * esize;
            let v = self.vregs[rd as usize];
            // INS leaves other lanes, including the top half, untouched.
            self.vregs[rd as usize] = (v & !(mask << shift)) | (val << shift);
            return Ok(true);
        }

        match (insn >> 10) & 0b111111 {
            0b000111 => {
                // INS (general): esize = 8 << ctz(imm5), index = imm5 >> (ctz + 1).
                let lsb = imm5.trailing_zeros();
                if lsb > 3 {
                    return Ok(false);
                }
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let shift = index * esize;
                let mask = (1u128 << esize) - 1;
                let v = self.vregs[rd as usize];
                let val = (self.read_zr(rn) as u128) & mask;
                self.vregs[rd as usize] = (v & !(mask << shift)) | (val << shift);
                Ok(true)
            }
            0b000011 if imm5 != 0 => {
                // DUP (general): esize = 8 << ctz(imm5).
                let esize = 8u32 << imm5.trailing_zeros();
                let elements = if q == 1 { 128 / esize } else { 64 / esize };
                let val = (self.read_zr(rn) as u128) & ((1u128 << esize) - 1);
                let mut v: u128 = 0;
                for i in 0..elements {
                    v |= val << (i * esize);
                }
                self.vregs[rd as usize] = v;
                Ok(true)
            }
            0b000001 if imm5 != 0 => {
                // DUP (element)
                let lsb = imm5.trailing_zeros();
                if lsb > 3 {
                    return Ok(false);
                }
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let elements = if q == 1 { 128 / esize } else { 64 / esize };
                let mask = (1u128 << esize) - 1;
                let val = (self.vregs[rn as usize] >> (index * esize)) & mask;
                let mut v: u128 = 0;
                for i in 0..elements {
                    v |= val << (i * esize);
                }
                self.vregs[rd as usize] = v;
                Ok(true)
            }
            0b001111 => {
                let lsb = imm5.trailing_zeros();
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let shift = index * esize;
                let val = (self.vregs[rn as usize] >> shift) & ((1u128 << esize) - 1);
                self.write_zr(rd, val as u64);
                Ok(true)
            }
            0b001011 => {
                // SMOV: sign-extended lane extract.
                let lsb = imm5.trailing_zeros();
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let shift = index * esize;
                let val = (self.vregs[rn as usize] >> shift) & ((1u128 << esize) - 1);
                let val = sext_u64(val as u64, esize);
                self.write_zr(rd, val);
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}
