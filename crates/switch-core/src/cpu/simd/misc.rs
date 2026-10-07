//! Two-register misc and floating-point three-same.

use super::*;

impl Cpu {
    /// Two-register misc. FP forms are keyed by `(U, size<1>, opcode)`.
    pub(super) fn simd_two_reg_misc(&mut self, insn: u32, scalar: bool) -> Result<bool> {
        let q = (insn >> 30) & 1 == 1;
        let u = (insn >> 29) & 1;
        let size = (insn >> 22) & 0b11;
        let opcode = (insn >> 12) & 0x1F;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        let esize = 8u32 << size;
        let lanes = |esize: u32| {
            if scalar {
                1
            } else if q {
                128 / esize
            } else {
                64 / esize
            }
        };

        // Integer forms.
        match (u, opcode) {
            // REV64 / REV32 / REV16
            (0, 0b00000) | (1, 0b00000) | (0, 0b00001) => {
                if scalar {
                    return Ok(false);
                }
                let container = match (u, opcode) {
                    (0, 0b00000) => 64u32,
                    (1, 0b00000) => 32,
                    _ => 16,
                };
                if esize >= container {
                    return Ok(false);
                }
                let lanes = if q { 128 / esize } else { 64 / esize };
                let per_container = container / esize;
                let mask = (1u128 << esize) - 1;
                let a = self.vregs[rn as usize];
                let mut out: u128 = 0;
                for i in 0..lanes {
                    let group = i / per_container;
                    let within = i % per_container;
                    let src = group * per_container + (per_container - 1 - within);
                    out |= ((a >> (esize * src)) & mask) << (esize * i);
                }
                self.vregs[rd as usize] = out;
                return Ok(true);
            }
            // CLS / CLZ: count leading sign bits / leading zeros.
            (_, 0b00100) => {
                if scalar {
                    return Ok(false);
                }
                let signed = u == 0;
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    let shifted = v << (64 - esize);
                    if signed {
                        let inverted = if shifted >> 63 == 1 {
                            !shifted
                        } else {
                            shifted
                        };
                        u64::from((inverted << 1).leading_zeros().min(esize - 1))
                    } else {
                        u64::from(shifted.leading_zeros().min(esize))
                    }
                });
                return Ok(true);
            }
            // CNT (population count per byte) and NOT / RBIT.
            (0, 0b00101) => {
                if scalar {
                    return Ok(false);
                }
                if esize != 8 {
                    return Ok(false);
                }
                self.simd_lane_unary_n(rd, rn, lanes(8), 8, |v| u64::from(v.count_ones()));
                return Ok(true);
            }
            (1, 0b00101) => {
                if scalar {
                    return Ok(false);
                }
                let full = if q { u128::MAX } else { (1u128 << 64) - 1 };
                match size {
                    0b00 => self.vregs[rd as usize] = !self.vregs[rn as usize] & full,
                    0b01 => {
                        self.simd_lane_unary_n(rd, rn, lanes(8), 8, |v| {
                            u64::from((v as u8).reverse_bits())
                        });
                    }
                    _ => return Ok(false),
                }
                return Ok(true);
            }
            // SADDLP / UADDLP and SADALP / UADALP
            (_, 0b00010) | (_, 0b00110) => {
                if scalar {
                    return Ok(false);
                }
                let accumulate = opcode == 0b00110;
                let signed = u == 0;
                if esize == 64 {
                    return Ok(false);
                }
                let dest_esize = esize * 2;
                let lanes = if q { 128 / dest_esize } else { 64 / dest_esize };
                let src_mask = (1u128 << esize) - 1;
                let dest_mask = (1u128 << dest_esize) - 1;
                let a = self.vregs[rn as usize];
                let d = self.vregs[rd as usize];
                let mut out: u128 = 0;
                for i in 0..lanes {
                    let lo = ((a >> (esize * 2 * i)) & src_mask) as u64;
                    let hi = ((a >> (esize * (2 * i + 1))) & src_mask) as u64;
                    let sum = if signed {
                        (sext_u64(lo, esize) as i64).wrapping_add(sext_u64(hi, esize) as i64) as u64
                    } else {
                        lo.wrapping_add(hi)
                    };
                    let sum = if accumulate {
                        let acc = ((d >> (dest_esize * i)) & dest_mask) as u64;
                        sum.wrapping_add(acc)
                    } else {
                        sum
                    };
                    out |= (sum as u128 & dest_mask) << (dest_esize * i);
                }
                self.vregs[rd as usize] = out;
                return Ok(true);
            }
            // ABS / NEG, and the compares against zero.
            (0, 0b01011) => {
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    (sext_u64(v, esize) as i64).wrapping_abs() as u64
                });
                return Ok(true);
            }
            (1, 0b01011) => {
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, |v| {
                    (v as i64).wrapping_neg() as u64
                });
                return Ok(true);
            }
            (0, 0b01000) | (0, 0b01001) | (0, 0b01010) | (1, 0b01000) | (1, 0b01001) => {
                let kind = (u, opcode);
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    let signed = sext_u64(v, esize) as i64;
                    let holds = match kind {
                        (0, 0b01000) => signed > 0,  // CMGT #0
                        (0, 0b01001) => signed == 0, // CMEQ #0
                        (0, 0b01010) => signed < 0,  // CMLT #0
                        (1, 0b01000) => signed >= 0, // CMGE #0
                        _ => signed <= 0,            // CMLE #0
                    };
                    if holds {
                        u64::MAX
                    } else {
                        0
                    }
                });
                return Ok(true);
            }
            // XTN / SQXTN / UQXTN / SQXTUN: `size` is the destination width.
            (0, 0b10010) => {
                self.simd_shrn(rd, rn, q, esize, 0, false, false, false, false);
                return Ok(true);
            }
            (0, 0b10100) => {
                self.simd_shrn(rd, rn, q, esize, 0, false, true, false, true);
                return Ok(true);
            }
            (1, 0b10010) => {
                self.simd_shrn(rd, rn, q, esize, 0, false, true, true, true);
                return Ok(true);
            }
            (1, 0b10100) => {
                self.simd_shrn(rd, rn, q, esize, 0, false, false, false, true);
                return Ok(true);
            }
            // SHLL: widen each lane and shift it left by the element width.
            (1, 0b10011) => {
                if scalar {
                    return Ok(false);
                }
                if esize == 64 {
                    return Ok(false);
                }
                let dest_esize = esize * 2;
                let lanes = 64 / esize;
                let src_mask = (1u128 << esize) - 1;
                let src = self.vregs[rn as usize];
                let base = if q {
                    u128::from(lanes) * u128::from(esize)
                } else {
                    0
                };
                let mut out: u128 = 0;
                for i in 0..lanes {
                    let shift = esize * i + base as u32;
                    let v = (src >> shift) & src_mask;
                    out |= (v << esize) << (dest_esize * i);
                }
                self.vregs[rd as usize] = out;
                return Ok(true);
            }
            // FCVTL / FCVTN: half <-> single and single <-> double.
            (0, 0b10111) if size <= 0b01 => {
                if scalar {
                    return Ok(false);
                }
                let src = self.vregs[rn as usize];
                let base = if q { 64 } else { 0 };
                let mut out: u128 = 0;
                if size == 0b00 {
                    for i in 0..4u32 {
                        let h = ((src >> (base + 16 * i)) & 0xFFFF) as u16;
                        out |= u128::from(f16_to_f32(h).to_bits()) << (32 * i);
                    }
                } else {
                    for i in 0..2u32 {
                        let bits = ((src >> (base + 32 * i)) & 0xFFFF_FFFF) as u32;
                        let wide = f64::from(f32::from_bits(bits)).to_bits();
                        out |= u128::from(wide) << (64 * i);
                    }
                }
                self.vregs[rd as usize] = out;
                return Ok(true);
            }
            (0, 0b10110) if size <= 0b01 => {
                if scalar {
                    return Ok(false);
                }
                let src = self.vregs[rn as usize];
                let mut narrowed: u128 = 0;
                if size == 0b00 {
                    // Via double so the half is rounded once.
                    for i in 0..4u32 {
                        let bits = ((src >> (32 * i)) & 0xFFFF_FFFF) as u32;
                        let h = f64_to_f16(f64::from(f32::from_bits(bits)));
                        narrowed |= u128::from(h) << (16 * i);
                    }
                } else {
                    for i in 0..2u32 {
                        let bits = (src >> (64 * i)) as u64;
                        let narrow = (f64::from_bits(bits) as f32).to_bits();
                        narrowed |= u128::from(narrow) << (32 * i);
                    }
                }
                self.vregs[rd as usize] = if q {
                    (self.vregs[rd as usize] & ((1u128 << 64) - 1)) | (narrowed << 64)
                } else {
                    narrowed
                };
                return Ok(true);
            }
            _ => {}
        }

        // FP forms.
        let double = size & 1 == 1;
        if double && !q && !scalar {
            return Ok(false); // a single 64-bit lane isn't a vector form
        }
        let key = (u << 6) | ((size >> 1) << 5) | opcode;
        let esize = if double { 64 } else { 32 };
        if matches!(key, 0x2c | 0x2d | 0x2e | 0x6c | 0x6d) {
            self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                let a = if double {
                    f64::from_bits(v)
                } else {
                    f64::from(f32::from_bits(v as u32))
                };
                let holds = match key {
                    0x2c => a > 0.0,
                    0x2d => a == 0.0,
                    0x2e => a < 0.0,
                    0x6c => a >= 0.0,
                    _ => a <= 0.0,
                };
                if holds {
                    u64::MAX
                } else {
                    0
                }
            });
            return Ok(true);
        }
        // Float -> integer converts, with the rounding mode in the opcode.
        if matches!(
            key,
            0x1a | 0x1b | 0x1c | 0x3a | 0x3b | 0x5a | 0x5b | 0x5c | 0x7a | 0x7b
        ) {
            let signed = u == 0;
            let rounding = match key & 0x1F {
                0b11010 if size >> 1 == 0 => Rounding::TiesEven, // FCVTNS/FCVTNU
                0b11010 => Rounding::TowardPos,                  // FCVTPS/FCVTPU
                0b11011 if size >> 1 == 0 => Rounding::TowardNeg, // FCVTMS/FCVTMU
                0b11011 => Rounding::TowardZero,                 // FCVTZS/FCVTZU
                _ => Rounding::TiesAway,                         // FCVTAS/FCVTAU
            };
            self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                let a = if double {
                    f64::from_bits(v)
                } else {
                    f64::from(f32::from_bits(v as u32))
                };
                round_to_int_sized(a, rounding, signed, esize)
            });
            return Ok(true);
        }
        // Integer -> float, the FP unary ops and the reciprocal estimates.
        match key {
            // SCVTF / UCVTF
            0x1d | 0x5d => {
                let signed = u == 0;
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    if double {
                        let f = if signed { (v as i64) as f64 } else { v as f64 };
                        f.to_bits()
                    } else {
                        let f = if signed {
                            (v as i32) as f32
                        } else {
                            (v as u32) as f32
                        };
                        u64::from(f.to_bits())
                    }
                });
                Ok(true)
            }
            // FABS / FNEG: sign-bit operations, so no float round-trip.
            0x2f | 0x6f => {
                let negate = u == 1;
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    let sign = 1u64 << (esize - 1);
                    if negate {
                        v ^ sign
                    } else {
                        v & !sign
                    }
                });
                Ok(true)
            }
            // FSQRT, FRINTx and the reciprocal estimates.
            0x7f | 0x18 | 0x19 | 0x38 | 0x39 | 0x58 | 0x59 | 0x79 | 0x3d | 0x7d => {
                let mode = fpcr_rounding(self.fpcr);
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    if double {
                        let a = f64::from_bits(v);
                        let r = match key {
                            0x7f => a.sqrt(),
                            0x18 => a.round_ties_even(), // FRINTN
                            0x19 => a.floor(),           // FRINTM
                            0x38 => a.ceil(),            // FRINTP
                            0x39 => a.trunc(),           // FRINTZ
                            0x58 => a.round(),           // FRINTA
                            0x59 | 0x79 => round_to_integral(a, mode), // FRINTX / FRINTI
                            0x3d => {
                                return super::fp::recip_estimate_bits(v, 64);
                            }
                            0x7d => {
                                return super::fp::rsqrt_estimate_bits(v, 64);
                            }
                            _ => unreachable!("unhandled two-register misc key"),
                        };
                        r.to_bits()
                    } else {
                        let a = f32::from_bits(v as u32);
                        let r = match key {
                            0x7f => a.sqrt(),
                            0x18 => a.round_ties_even(),
                            0x19 => a.floor(),
                            0x38 => a.ceil(),
                            0x39 => a.trunc(),
                            0x58 => a.round(),
                            0x59 | 0x79 => round_to_integral(f64::from(a), mode) as f32,
                            0x3d => {
                                return super::fp::recip_estimate_bits(v & 0xFFFF_FFFF, 32);
                            }
                            0x7d => {
                                return super::fp::rsqrt_estimate_bits(v & 0xFFFF_FFFF, 32);
                            }
                            _ => unreachable!("unhandled two-register misc key"),
                        };
                        u64::from(r.to_bits())
                    }
                });
                Ok(true)
            }
            // URECPE / URSQRTE: the unsigned-integer estimates.
            0x3c | 0x7c => {
                let sqrt = key == 0x7c;
                self.simd_lane_unary_n(rd, rn, lanes(32), 32, move |v| {
                    let a = v as u32;
                    if a == 0 {
                        return u64::from(u32::MAX);
                    }
                    let f = a as f64 / 4_294_967_296.0;
                    let r = if sqrt { 1.0 / f.sqrt() } else { 1.0 / f };
                    (r * 4_294_967_296.0).min(f64::from(u32::MAX)) as u64
                });
                Ok(true)
            }
            _ => Err(Error::Cpu(format!(
                "unimplemented SIMD two-register misc u={} size={} opcode={:#07b} at {:#x}",
                u, size, opcode, self.pc
            ))),
        }
    }

    /// Three-same, floating point (opcode >= 0b11000).
    pub(super) fn simd_fp_three_same(&mut self, insn: u32, scalar: bool) -> Result<bool> {
        let q = (insn >> 30) & 1 == 1;
        let u = (insn >> 29) & 1;
        let a = (insn >> 23) & 1;
        let double = (insn >> 22) & 1 == 1;
        let rm = ((insn >> 16) & 0x1F) as u8;
        let opcode = (insn >> 11) & 0x1F;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        if double && !q && !scalar {
            return Ok(false); // a single 64-bit lane isn't a vector form
        }
        let esize = if double { 64u32 } else { 32 };
        let lanes = if scalar {
            1
        } else if q {
            128 / esize
        } else {
            64 / esize
        };
        let key = (u << 6) | (a << 5) | opcode;
        let arith = |x: f64, y: f64| -> f64 {
            match key {
                0x1a => x + y,            // FADD
                0x3a => x - y,            // FSUB
                0x5b => x * y,            // FMUL
                0x1b => fmulx(x, y),      // FMULX
                0x5f => x / y,            // FDIV
                0x7a => (x - y).abs(),    // FABD
                0x1e => fp_max(x, y),     // FMAX
                0x3e => fp_min(x, y),     // FMIN
                0x18 => fp_maxnum(x, y),  // FMAXNM
                0x38 => fp_minnum(x, y),  // FMINNM
                0x1f => 2.0 - x * y,      // FRECPS
                _ => (3.0 - x * y) / 2.0, // FRSQRTS
            }
        };
        if matches!(
            key,
            0x1a | 0x3a | 0x5b | 0x1b | 0x5f | 0x7a | 0x1e | 0x3e | 0x18 | 0x38 | 0x1f | 0x3f
        ) {
            if double {
                self.simd_elem_n(rd, rn, rm, lanes, esize, |x, y| {
                    arith(f64::from_bits(x), f64::from_bits(y)).to_bits()
                });
            } else {
                self.simd_elem_n(rd, rn, rm, lanes, esize, |x, y| {
                    let r = arith(
                        f64::from(f32::from_bits(x as u32)),
                        f64::from(f32::from_bits(y as u32)),
                    ) as f32;
                    u64::from(r.to_bits())
                });
            }
            return Ok(true);
        }
        match key {
            // FMLA / FMLS: fused multiply-accumulate into Vd.
            0x19 | 0x39 if !scalar => {
                let subtract = a == 1;
                self.simd_elem_acc(rd, rn, rm, q, esize, move |x, y, d| {
                    if double {
                        let n = f64::from_bits(x);
                        let n = if subtract { -n } else { n };
                        n.mul_add(f64::from_bits(y), f64::from_bits(d)).to_bits()
                    } else {
                        let n = f32::from_bits(x as u32);
                        let n = if subtract { -n } else { n };
                        u64::from(
                            n.mul_add(f32::from_bits(y as u32), f32::from_bits(d as u32))
                                .to_bits(),
                        )
                    }
                });
                Ok(true)
            }
            // The compares, including the absolute-value forms (FACGE/FACGT).
            0x1c | 0x5c | 0x7c | 0x5d | 0x7d => {
                self.simd_elem_n(rd, rn, rm, lanes, esize, move |x, y| {
                    let (mut fx, mut fy) = if double {
                        (f64::from_bits(x), f64::from_bits(y))
                    } else {
                        (
                            f64::from(f32::from_bits(x as u32)),
                            f64::from(f32::from_bits(y as u32)),
                        )
                    };
                    if key == 0x5d || key == 0x7d {
                        fx = fx.abs();
                        fy = fy.abs();
                    }
                    let holds = match key {
                        0x1c => fx == fy,        // FCMEQ
                        0x5c | 0x5d => fx >= fy, // FCMGE / FACGE
                        _ => fx > fy,            // FCMGT / FACGT
                    };
                    if holds {
                        u64::MAX
                    } else {
                        0
                    }
                });
                Ok(true)
            }
            // The pairwise reductions.
            0x5a | 0x5e | 0x7e | 0x58 | 0x78 if !scalar => {
                self.simd_pairwise(rd, rn, rm, q, esize, move |x, y| {
                    let (fx, fy) = if double {
                        (f64::from_bits(x), f64::from_bits(y))
                    } else {
                        (
                            f64::from(f32::from_bits(x as u32)),
                            f64::from(f32::from_bits(y as u32)),
                        )
                    };
                    let r = match key {
                        0x5a => fx + fy,           // FADDP
                        0x5e => fp_max(fx, fy),    // FMAXP
                        0x7e => fp_min(fx, fy),    // FMINP
                        0x58 => fp_maxnum(fx, fy), // FMAXNMP
                        _ => fp_minnum(fx, fy),    // FMINNMP
                    };
                    if double {
                        r.to_bits()
                    } else {
                        u64::from((r as f32).to_bits())
                    }
                });
                Ok(true)
            }
            _ => Err(Error::Cpu(format!(
                "unimplemented SIMD FP three-same u={} a={} opcode={:#07b} at {:#x}",
                u, a, opcode, self.pc
            ))),
        }
    }
}
