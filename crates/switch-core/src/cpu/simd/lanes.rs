//! Lane helpers, across-lanes reductions, shifts and fixed-point conversion.

use super::*;

impl Cpu {
    // Scalar floating point. FP exception flags are not modelled.

    /// Element-wise binary op over `esize`-bit lanes; `q` selects 128 vs 64 bits.
    pub(super) fn simd_elem<F: Fn(u64, u64) -> u64>(
        &mut self,
        rd: u8,
        rn: u8,
        rm: u8,
        q: bool,
        esize: u32,
        f: F,
    ) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            out = set_lane(out, esize, i, f(lane(a, esize, i), lane(b, esize, i)));
        }
        self.vregs[rd as usize] = out;
    }

    pub(super) fn simd_elem_n<F: Fn(u64, u64) -> u64>(
        &mut self,
        rd: u8,
        rn: u8,
        rm: u8,
        lanes: u32,
        esize: u32,
        f: F,
    ) {
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            out = set_lane(out, esize, i, f(lane(a, esize, i), lane(b, esize, i)));
        }
        self.vregs[rd as usize] = out;
    }

    /// ZIP1/ZIP2/UZP1/UZP2/TRN1/TRN2 over `esize`-bit lanes.
    pub(super) fn simd_permute(&mut self, rd: u8, rn: u8, rm: u8, q: bool, esize: u32, op: u32) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let half = lanes / 2;
        let mask = elem_mask(esize);
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let get = |r: u128, i: u32| (r >> (esize * i)) & mask;
        let mut out: u128 = 0;
        match op {
            // UZP1 (even elements) / UZP2 (odd) of Vn:Vm.
            0b000110 | 0b010110 => {
                let start = u32::from(op == 0b010110);
                for i in 0..lanes {
                    let index = start + 2 * (i % half);
                    let v = if i < half {
                        get(a, index)
                    } else {
                        get(b, index)
                    };
                    out |= v << (esize * i);
                }
            }
            // TRN1 (even elements) / TRN2 (odd) of both, interleaved.
            0b001010 | 0b011010 => {
                let odd = u32::from(op == 0b011010);
                for i in 0..half {
                    out |= get(a, 2 * i + odd) << (esize * 2 * i);
                    out |= get(b, 2 * i + odd) << (esize * (2 * i + 1));
                }
            }
            // ZIP1 (low halves) / ZIP2 (high halves), interleaved.
            _ => {
                let base = if op == 0b011110 { half } else { 0 };
                for i in 0..half {
                    out |= get(a, base + i) << (esize * 2 * i);
                    out |= get(b, base + i) << (esize * (2 * i + 1));
                }
            }
        }
        self.vregs[rd as usize] = out;
    }

    /// Lanewise op that also reads the destination lane (MLA/MLS, SABA/UABA).
    pub(super) fn simd_elem_acc<F: Fn(u64, u64, u64) -> u64>(
        &mut self,
        rd: u8,
        rn: u8,
        rm: u8,
        q: bool,
        esize: u32,
        f: F,
    ) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let d = self.vregs[rd as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            let position = esize * i;
            let va = ((a >> position) & mask) as u64;
            let vb = ((b >> position) & mask) as u64;
            let vd = ((d >> position) & mask) as u64;
            out |= (f(va, vb, vd) as u128 & mask) << position;
        }
        self.vregs[rd as usize] = out;
    }

    /// Pairwise op: low half pairs Vn's lanes, high half Vm's.
    pub(super) fn simd_pairwise<F: Fn(u64, u64) -> u64>(
        &mut self,
        rd: u8,
        rn: u8,
        rm: u8,
        q: bool,
        esize: u32,
        f: F,
    ) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let half = lanes / 2;
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let mut out: u128 = 0;
        for i in 0..half {
            let a0 = ((a >> (esize * 2 * i)) & mask) as u64;
            let a1 = ((a >> (esize * (2 * i + 1))) & mask) as u64;
            let b0 = ((b >> (esize * 2 * i)) & mask) as u64;
            let b1 = ((b >> (esize * (2 * i + 1))) & mask) as u64;
            out |= (f(a0, a1) as u128 & mask) << (esize * i);
            out |= (f(b0, b1) as u128 & mask) << (esize * (i + half));
        }
        self.vregs[rd as usize] = out;
    }
    pub(super) fn sge(a: u64, b: u64, bits: u32) -> bool {
        let shift = 64 - bits;
        ((a << shift) as i64) >= ((b << shift) as i64)
    }

    /// Across lanes, integer forms: reduce Vn into one scalar lane.
    pub(super) fn simd_across_lanes(&mut self, insn: u32) -> Result<bool> {
        let q = (insn >> 30) & 1 == 1;
        let u = (insn >> 29) & 1;
        let size = (insn >> 22) & 0b11;
        let opcode = (insn >> 12) & 0x1F;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        // FMAXNMV / FMINNMV / FMAXV / FMINV: reduced in pairs, (0 op 1) op (2 op 3).
        if u == 1 && matches!(opcode, 0b01100 | 0b01111) {
            if !q || (insn >> 22) & 1 != 0 {
                return Ok(false); // two lanes, or doubles: unallocated
            }
            let min = (insn >> 23) & 1 == 1;
            let op = |x: f64, y: f64| match (opcode == 0b01100, min) {
                (true, false) => fp_maxnum(x, y),
                (true, true) => fp_minnum(x, y),
                (false, false) => fp_max(x, y),
                (false, true) => fp_min(x, y),
            };
            let a = self.vregs[rn as usize];
            let lane = |i: u32| f64::from(f32::from_bits((a >> (32 * i)) as u32));
            let reduced = op(op(lane(0), lane(1)), op(lane(2), lane(3))) as f32;
            self.vregs[rd as usize] = u128::from(reduced.to_bits());
            return Ok(true);
        }
        if size == 0b11 || !matches!(opcode, 0b00011 | 0b01010 | 0b11010 | 0b11011) {
            return Ok(false);
        }
        let esize = 8u32 << size;
        let lanes = if q { 128 / esize } else { 64 / esize };
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let elem = |i: u32| ((a >> (esize * i)) & mask) as u64;
        let signed = u == 0;
        let result = match opcode {
            0b00011 => {
                // SADDLV / UADDLV: sum widened to double the element size.
                let mut sum: i128 = 0;
                for i in 0..lanes {
                    let v = elem(i);
                    sum += if signed {
                        sext_u64(v, esize) as i64 as i128
                    } else {
                        v as i128
                    };
                }
                (sum as u128) & ((1u128 << (esize * 2)) - 1)
            }
            0b01010 | 0b11010 => {
                // SMAXV / UMAXV (0b01010), SMINV / UMINV (0b11010).
                let want_max = opcode == 0b01010;
                let mut best = elem(0);
                for i in 1..lanes {
                    let v = elem(i);
                    let v_wins = if signed {
                        if want_max {
                            !Self::sge(best, v, esize)
                        } else {
                            Self::sge(best, v, esize) && best != v
                        }
                    } else if want_max {
                        v > best
                    } else {
                        v < best
                    };
                    if v_wins {
                        best = v;
                    }
                }
                best as u128
            }
            0b11011 => {
                // ADDV: wrapping sum, no widening.
                let mut sum: u128 = 0;
                for i in 0..lanes {
                    sum = sum.wrapping_add(elem(i) as u128);
                }
                sum & mask
            }
            _ => unreachable!(),
        };
        self.vregs[rd as usize] = result;
        Ok(true)
    }

    /// Fixed-point SCVTF/UCVTF (`to_int` false) and FCVTZS/FCVTZU (`to_int` true).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn simd_fixed_convert(
        &mut self,
        rd: u8,
        rn: u8,
        lanes: u32,
        esize: u32,
        fbits: u32,
        to_int: bool,
        unsigned: bool,
    ) -> Result<()> {
        if esize == 16 {
            return Err(Error::Cpu(
                "unimplemented half-precision SIMD fixed-point convert".to_owned(),
            ));
        }
        let mask = if esize == 64 {
            u64::MAX
        } else {
            (1u64 << esize) - 1
        };
        let scale = Self::pow2(fbits);
        let src = self.vregs[rn as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            let raw = (src >> (esize * i)) as u64 & mask;
            let value = if to_int {
                let f = if esize == 64 {
                    f64::from_bits(raw)
                } else {
                    f64::from(f32::from_bits(raw as u32))
                };
                round_to_int_sized(f * scale, Rounding::TowardZero, !unsigned, esize) & mask
            } else {
                let int = match (unsigned, esize == 64) {
                    (false, true) => raw as i64 as f64,
                    (false, false) => f64::from(raw as u32 as i32),
                    (true, true) => raw as f64,
                    (true, false) => f64::from(raw as u32),
                };
                if esize == 64 {
                    (int / scale).to_bits()
                } else {
                    u64::from(((int / scale) as f32).to_bits())
                }
            };
            out |= u128::from(value) << (esize * i);
        }
        self.vregs[rd as usize] = out;
        Ok(())
    }

    /// Shift by immediate: right shift is `2*esize - imm`, left is `imm - esize`.
    pub(super) fn simd_shift_imm(
        &mut self,
        rd: u8,
        rn: u8,
        q: bool,
        u: bool,
        opcode: u32,
        esize: u32,
        imm: u32,
    ) -> Result<()> {
        let mask: u128 = if esize >= 128 {
            u128::MAX
        } else {
            (1u128 << esize) - 1
        };
        let src = self.vregs[rn as usize];
        let dst = self.vregs[rd as usize];

        // SSHLL/USHLL (and SXTL/UXTL): widen from one half of Vn.
        if opcode == 0b10100 {
            let shift = imm - esize;
            let wide = 2 * esize;
            let wide_mask: u128 = if wide >= 128 {
                u128::MAX
            } else {
                (1u128 << wide) - 1
            };
            let lanes = 64 / esize;
            let base = if q { lanes } else { 0 };
            let mut out: u128 = 0;
            for i in 0..lanes {
                let raw = ((src >> (esize * (i + base))) & mask) as u64;
                let extended = if u { raw } else { sext_u64(raw, esize) };
                let value = ((extended as u128) << shift) & wide_mask;
                out |= value << (wide * i);
            }
            self.vregs[rd as usize] = out;
            return Ok(());
        }

        let lanes = if q { 128 / esize } else { 64 / esize };
        let left = matches!(opcode, 0b01010 | 0b01100 | 0b01110);
        let shift = if left { imm - esize } else { 2 * esize - imm };
        let mut out: u128 = 0;
        for i in 0..lanes {
            let position = esize * i;
            let raw = ((src >> position) & mask) as u64;
            let old = ((dst >> position) & mask) as u64;
            let value: u64 = match opcode {
                // SSHR / USHR, SSRA / USRA, SRSHR / URSHR, SRSRA / URSRA.
                0b00000 | 0b00010 | 0b00100 | 0b00110 => {
                    let rounding = opcode & 0b00100 != 0;
                    let round = if rounding && shift > 0 {
                        1u64 << (shift - 1)
                    } else {
                        0
                    };
                    let shifted = if u {
                        raw.wrapping_add(round) >> shift.min(63)
                    } else {
                        let signed = sext_u64(raw, esize) as i64;
                        (signed.wrapping_add(round as i64) >> shift.min(63)) as u64
                    };
                    if opcode & 0b00010 != 0 {
                        old.wrapping_add(shifted)
                    } else {
                        shifted
                    }
                }
                // SRI: shift right and insert, keeping the high bits of Vd.
                0b01000 => {
                    let keep = if shift >= esize {
                        mask as u64
                    } else {
                        !(mask as u64 >> shift)
                    };
                    (old & keep) | ((raw >> shift.min(63)) & !keep)
                }
                // SHL, or SLI when the insert variant is selected.
                0b01010 => {
                    let shifted = raw.wrapping_shl(shift);
                    if u {
                        let keep = if shift == 0 { 0 } else { (1u64 << shift) - 1 };
                        (old & keep) | (shifted & !keep)
                    } else {
                        shifted
                    }
                }
                other => {
                    return Err(Error::Cpu(format!(
                        "unimplemented SIMD shift-by-immediate opcode {:#07b}",
                        other
                    )))
                }
            };
            out |= ((value as u128) & mask) << position;
        }
        self.vregs[rd as usize] = out;
        Ok(())
    }

    /// Lanewise unary op on raw lane bits.
    pub(super) fn simd_lane_unary_n<F: Fn(u64) -> u64>(
        &mut self,
        rd: u8,
        rn: u8,
        lanes: u32,
        esize: u32,
        f: F,
    ) {
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            let v = ((a >> (esize * i)) & mask) as u64;
            out |= (f(v) as u128 & mask) << (esize * i);
        }
        self.vregs[rd as usize] = out;
    }

    /// SHRN/RSHRN and saturating forms; `Q=1` targets the high half.
    pub(super) fn simd_shrn(
        &mut self,
        rd: u8,
        rn: u8,
        q: bool,
        dest_esize: u32,
        shift: u32,
        rounding: bool,
        signed_src: bool,
        to_unsigned: bool,
        saturating: bool,
    ) {
        let src_esize = 2 * dest_esize;
        let src_elements = 128 / src_esize;
        let dest_mask = (1u128 << dest_esize) - 1;
        let src = self.vregs[rn as usize];
        let round_add = if rounding { 1i64 << (shift - 1) } else { 0 };
        let mut narrowed = [0u64; 16];
        for i in 0..src_elements {
            let raw = lane(src, src_esize, i);
            let shifted = if signed_src {
                let v = (raw as i64) << (64 - src_esize) >> (64 - src_esize);
                v.wrapping_add(round_add) >> shift
            } else {
                (raw.wrapping_add(round_add as u64) >> shift) as i64
            };
            let mut v = shifted;
            if saturating || to_unsigned {
                let (min, max) = if to_unsigned || !signed_src {
                    (0i64, dest_mask as i64)
                } else {
                    (-(1i64 << (dest_esize - 1)), (1i64 << (dest_esize - 1)) - 1)
                };
                v = v.clamp(min, max);
            }
            narrowed[i as usize] = (v as u64) & (dest_mask as u64);
        }
        let mut out: u128 = 0;
        for i in 0..src_elements {
            out = set_lane(out, dest_esize, i, narrowed[i as usize]);
        }
        if q {
            self.vregs[rd as usize] = (self.vregs[rd as usize] & ((1u128 << 64) - 1)) | (out << 64);
        } else {
            self.vregs[rd as usize] = out & ((1u128 << 64) - 1);
        }
    }
}
