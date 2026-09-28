//! Advanced SIMD (NEON), the AArch32 vector unit.
//!
//! Its registers are the same file again: `Qn` is `V(n)` whole, and `Dn` its
//! halves, so nothing here needs storage of its own.
//!
//! What is implemented is what Mario Kart 8 Deluxe's modules actually contain.
//! Canonicalising every NEON encoding in their 4.8M words, masking out the
//! register fields so one class collapses to one key: gives 97 distinct
//! classes, and a long tail: the float multiply-accumulate by element is 2,898
//! of them, the immediate zero 1,140, and by the fortieth the count is in the
//! tens. The classes below are that head. Anything else still says so by name
//! rather than being quietly approximated. The two-register miscellaneous
//! group is the exception, complete and checked against `qemu-arm`, since
//! its tail was what the title reached next each time.

use crate::cpu::Cpu;
use crate::{Error, Result};

/// One lane's worth of a vector, as raw bits.
type Lane = u64;

/// How many single-precision lanes a vector holds. NEON has no
/// double-precision vectors, so this is the only float lane count there is,
/// and in the three-registers-of-the-same-length group it cannot come from
/// `size`, where bit 21 selects subtract and bit 20 is `sz`.
#[inline]
fn f32_lanes(quad: bool) -> u32 {
    if quad {
        4
    } else {
        2
    }
}

/// Split `value` into `lanes` lanes of `esize` bits.
#[inline]
fn lanes_of(value: u128, esize: u32, lanes: u32) -> [Lane; 16] {
    debug_assert!(
        esize * lanes <= 128,
        "{lanes} lanes of {esize} bits overflow a vector"
    );
    let mut out = [0u64; 16];
    let mask = if esize == 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    for (i, slot) in out.iter_mut().enumerate().take(lanes as usize) {
        *slot = ((value >> (esize * i as u32)) as u64) & mask;
    }
    out
}

/// Reassemble lanes into a vector.
#[inline]
fn from_lanes(lanes_in: &[Lane; 16], esize: u32, lanes: u32) -> u128 {
    let mask = if esize == 64 {
        u128::from(u64::MAX)
    } else {
        (1u128 << esize) - 1
    };
    let mut out = 0u128;
    for (i, &lane) in lanes_in.iter().enumerate().take(lanes as usize) {
        out |= (u128::from(lane) & mask) << (esize * i as u32);
    }
    out
}

/// Sign-extend a lane to `i64`.
#[inline]
fn sext(value: Lane, esize: u32) -> i64 {
    if esize == 64 {
        value as i64
    } else {
        let shift = 64 - esize;
        ((value << shift) as i64) >> shift
    }
}

/// The `VMOV`/`VMVN` modified immediate, whose `cmode` says how the eight bits
/// are spread across a 64-bit pattern.
fn modified_immediate(cmode: u32, op: u32, imm8: u32) -> Option<u64> {
    let byte = u64::from(imm8);
    let replicate32 = |v: u64| v | (v << 32);
    let replicate16 = |v: u64| {
        let v = v | (v << 16);
        v | (v << 32)
    };
    Some(match (cmode >> 1, cmode & 1, op) {
        (0b000, _, _) => replicate32(byte),
        (0b001, _, _) => replicate32(byte << 8),
        (0b010, _, _) => replicate32(byte << 16),
        (0b011, _, _) => replicate32(byte << 24),
        (0b100, _, _) => replicate16(byte),
        (0b101, _, _) => replicate16(byte << 8),
        (0b110, 0, _) => replicate32((byte << 8) | 0xFF),
        (0b110, 1, _) => replicate32((byte << 16) | 0xFFFF),
        (0b111, 0, 0) => replicate16(byte | byte << 8),
        (0b111, 0, 1) => {
            // Each bit of the byte becomes a whole byte of the result.
            let mut out = 0u64;
            for i in 0..8 {
                if byte & (1 << i) != 0 {
                    out |= 0xFFu64 << (8 * i);
                }
            }
            out
        }
        (0b111, 1, 0) => {
            // A single-precision float built the VFP way, replicated.
            let bits = ((byte & 0x80) << 24)
                | ((!(byte >> 6) & 1) << 30)
                | (if byte & 0x40 != 0 { 0x1F } else { 0 } << 25)
                | ((byte & 0x3F) << 19);
            replicate32(u64::from(bits as u32))
        }
        _ => return None,
    })
}

impl Cpu {
    /// A `D` or `Q` register as one value. A quad's number is its low `D`'s
    /// halved, which is why the encoding's register field is always a `D`.
    #[inline]
    fn neon_get(&self, quad: bool, d: u8) -> u128 {
        if quad {
            self.vregs[((d >> 1) & 0xF) as usize]
        } else {
            u128::from(self.vfp_d(d))
        }
    }

    #[inline]
    fn neon_set(&mut self, quad: bool, d: u8, val: u128) {
        if quad {
            self.vregs[((d >> 1) & 0xF) as usize] = val;
        } else {
            self.set_vfp_d(d, val as u64);
        }
    }

    /// Advanced SIMD data processing, the `cond == 0xF`, bits 27:25 == 001
    /// encoding space.
    ///
    /// The groups are told apart by A (bits 23:19) and C (bits 7:4) together,
    /// as the manual's table does. With bit 4 set it is a modified immediate
    /// when bits 21:19 are clear and bit 7 is too, and otherwise a shift:
    /// bit 7 set is a 64-bit shift whatever A says, and a shift's `imm6`
    /// reaches every value of bits 21:19, the pattern the other groups use
    /// included. With bit 4 clear, bits 21:20 both set are `VEXT` or the
    /// two-register groups, and otherwise bit 6 separates the
    /// different-length forms from the by-scalar ones.
    pub(super) fn a32_neon_data(&mut self, insn: u32) -> Result<()> {
        let result = if (insn >> 23) & 1 == 0 {
            self.neon_three_same(insn)
        } else if (insn >> 4) & 1 != 0 {
            if (insn >> 7) & 1 == 0 && (insn >> 19) & 0b111 == 0 {
                self.neon_immediate(insn)
            } else {
                self.neon_shift_immediate(insn)
            }
        } else if (insn >> 20) & 0b11 == 0b11 {
            // Bit 24 is what tells `VEXT` from the two-register group: both
            // put 1011 in bits 23:20, and `VEXT`'s `imm4` freely reaches the
            // values the other group uses to name an operation.
            if (insn >> 24) & 1 == 0 {
                self.neon_ext(insn)
            } else {
                self.neon_two_reg(insn)
            }
        } else if (insn >> 6) & 1 == 0 {
            self.neon_three_different(insn)
        } else {
            self.neon_by_scalar(insn)
        };
        result.map(|()| self.pc = self.pc.wrapping_add(4))
    }

    fn neon_unimplemented(&self, insn: u32) -> Error {
        Error::Cpu(format!(
            "unimplemented NEON instruction {:#010x} at pc={:#010x}",
            insn, self.pc
        ))
    }

    /// The three-registers-of-the-same-length group, which is most of NEON.
    fn neon_three_same(&mut self, insn: u32) -> Result<()> {
        let unsigned = (insn >> 24) & 1 != 0;
        let size = (insn >> 20) & 0b11;
        let opc = (insn >> 8) & 0xF;
        let op = (insn >> 4) & 1 != 0;
        let quad = (insn >> 6) & 1 != 0;
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let vn = (((insn >> 7) & 1) as u8) << 4 | ((insn >> 16) & 0xF) as u8;
        let vm = (((insn >> 5) & 1) as u8) << 4 | (insn & 0xF) as u8;
        let a = self.neon_get(quad, vn);
        let b = self.neon_get(quad, vm);
        let esize = 8 << size;
        let count = if quad { 128 } else { 64 } / esize;

        let value = match (opc, op) {
            // The bitwise group, whose operation is in `size` rather than in
            // the opcode.
            (0x1, true) => match (unsigned, size) {
                (false, 0b00) => a & b,
                (false, 0b01) => a & !b,
                (false, 0b10) => a | b,
                (false, _) => a | !b,
                (true, 0b00) => a ^ b,
                // VBSL takes its mask from the destination; VBIT and VBIF
                // take theirs from one of the sources.
                (true, 0b01) => {
                    let d = self.neon_get(quad, vd);
                    (a & d) | (b & !d)
                }
                (true, 0b10) => {
                    let d = self.neon_get(quad, vd);
                    (a & b) | (d & !b)
                }
                (true, _) => {
                    let d = self.neon_get(quad, vd);
                    (d & b) | (a & !b)
                }
            },
            // Integer add and subtract, and the equality tests beside them.
            (0x8, false) => self.neon_lane_op(a, b, esize, count, |x, y| {
                if unsigned {
                    x.wrapping_sub(y)
                } else {
                    x.wrapping_add(y)
                }
            }),
            (0x8, true) => self.neon_lane_op(a, b, esize, count, |x, y| {
                let hit = if unsigned { x == y } else { x & y != 0 };
                if hit {
                    u64::MAX
                } else {
                    0
                }
            }),
            // Integer maximum and minimum, told apart by bit 4 (the float
            // forms use bit 21 instead).
            (0x6, _) => self.neon_lane_op(a, b, esize, count, |x, y| {
                let (x, y) = (int_lane(x, esize, unsigned), int_lane(y, esize, unsigned));
                (if op { x.min(y) } else { x.max(y) }) as u64
            }),
            // The halving and saturating adds and subtracts. Each lane is
            // worked at full width, which the halving and the saturation are
            // both taken from.
            (0x0 | 0x2, _) => self.neon_lane_op(a, b, esize, count, |x, y| {
                let (x, y) = (int_lane(x, esize, unsigned), int_lane(y, esize, unsigned));
                let full = if opc == 0x0 { x + y } else { x - y };
                if op {
                    saturate(full, esize, unsigned) as u64
                } else {
                    (full >> 1) as u64
                }
            }),
            (0x1, false) => self.neon_lane_op(a, b, esize, count, |x, y| {
                let (x, y) = (int_lane(x, esize, unsigned), int_lane(y, esize, unsigned));
                ((x + y + 1) >> 1) as u64
            }),
            // VCGT and VCGE.
            (0x3, _) => self.neon_lane_op(a, b, esize, count, |x, y| {
                let (x, y) = (int_lane(x, esize, unsigned), int_lane(y, esize, unsigned));
                if (op && x >= y) || (!op && x > y) {
                    u64::MAX
                } else {
                    0
                }
            }),
            // Shifts by a register: the signed low byte of each lane of Vn
            // shifts the same lane of Vm, left if positive and right if
            // negative. The operands are the other way round from every
            // other form here. VSHL, VQSHL, VRSHL and VQRSHL.
            (0x4 | 0x5, _) => self.neon_lane_op(b, a, esize, count, |x, y| {
                let value = int_lane(x, esize, unsigned);
                let shift = i32::from(y as u8 as i8);
                let shifted = shift_by(value, shift, opc == 0x5, esize);
                if op {
                    saturate(shifted, esize, unsigned) as u64
                } else {
                    shifted as u64
                }
            }),
            // VABD and VABA: the absolute difference, accumulated by VABA.
            (0x7, _) => {
                let d = self.neon_get(quad, vd);
                let diff = self.neon_lane_op(a, b, esize, count, |x, y| {
                    let (x, y) = (int_lane(x, esize, unsigned), int_lane(y, esize, unsigned));
                    (x - y).unsigned_abs() as u64
                });
                if op {
                    self.neon_lane_op(d, diff, esize, count, |x, y| x.wrapping_add(y))
                } else {
                    diff
                }
            }
            // VPMAX, VPMIN and VPADD: adjacent pairs of the first operand fill
            // the low half of the result, and of the second the high half.
            (0xA, _) | (0xB, true) => {
                let x = lanes_of(a, esize, count);
                let y = lanes_of(b, esize, count);
                let mut out = [0u64; 16];
                let half = count as usize / 2;
                for (i, slot) in out.iter_mut().enumerate().take(count as usize) {
                    let (src, j) = if i < half {
                        (&x, 2 * i)
                    } else {
                        (&y, 2 * (i - half))
                    };
                    let (p, q) = (
                        int_lane(src[j], esize, unsigned),
                        int_lane(src[j + 1], esize, unsigned),
                    );
                    *slot = match (opc, op) {
                        (0xA, false) => p.max(q),
                        (0xA, true) => p.min(q),
                        _ => p + q,
                    } as u64;
                }
                from_lanes(&out, esize, count)
            }
            // VQDMULH and VQRDMULH: the high half of twice the product,
            // rounded by the second, saturated.
            (0xB, false) => self.neon_lane_op(a, b, esize, count, |x, y| {
                let product = 2 * sext(x, esize) as i128 * sext(y, esize) as i128;
                let rounding = if unsigned { 1i128 << (esize - 1) } else { 0 };
                saturate((product + rounding) >> esize, esize, false) as u64
            }),
            // The three-register SHA steps, on Q registers: SHA1C, SHA1P,
            // SHA1M and SHA1SU0 by `size`, and with U set SHA256H, SHA256H2
            // and SHA256SU1.
            (0xC, false) => {
                let opcode = u32::from(unsigned) << 2 | size;
                let d = self.neon_get(true, vd);
                let n = self.neon_get(true, vn);
                let m = self.neon_get(true, vm);
                let Some(result) = crate::cpu::crypto::sha_three(opcode, d, n, m) else {
                    return Err(self.neon_unimplemented(insn));
                };
                self.neon_set(true, vd, result);
                return Ok(());
            }
            // Integer multiply, and multiply-accumulate.
            (0x9, false) => {
                let d = self.neon_get(quad, vd);
                let acc = lanes_of(d, esize, count);
                let mut out = [0u64; 16];
                let x = lanes_of(a, esize, count);
                let y = lanes_of(b, esize, count);
                for i in 0..count as usize {
                    let product = x[i].wrapping_mul(y[i]);
                    out[i] = if unsigned {
                        acc[i].wrapping_sub(product)
                    } else {
                        acc[i].wrapping_add(product)
                    };
                }
                from_lanes(&out, esize, count)
            }
            // VMUL, and with U set its polynomial form: a carry-less
            // multiply of bytes, keeping the low eight bits.
            (0x9, true) if unsigned => self.neon_lane_op(a, b, esize, count, |x, y| {
                crate::cpu::crypto::poly_mul(x, y, 8) as u64
            }),
            (0x9, true) => self.neon_lane_op(a, b, esize, count, |x, y| x.wrapping_mul(y)),
            // Floating point. NEON has no double-precision vectors, so every
            // one of these is F32.
            (0xD, false) => {
                // Bit 21 selects subtract, and U selects the pairwise forms.
                if unsigned {
                    if size & 0b10 != 0 {
                        self.neon_f32(a, b, quad, |x, y| (x - y).abs())
                    } else {
                        return self.neon_pairwise_f32(quad, vd, a, b);
                    }
                } else if size & 0b10 != 0 {
                    self.neon_f32(a, b, quad, |x, y| x - y)
                } else {
                    self.neon_f32(a, b, quad, |x, y| x + y)
                }
            }
            (0xD, true) => {
                if unsigned {
                    self.neon_f32(a, b, quad, |x, y| x * y)
                } else {
                    let d = self.neon_get(quad, vd);
                    let negate = size & 0b10 != 0;
                    let count = f32_lanes(quad);
                    let x = lanes_of(a, 32, count);
                    let y = lanes_of(b, 32, count);
                    let acc = lanes_of(d, 32, count);
                    let mut out = [0u64; 16];
                    for i in 0..count as usize {
                        let product = f32::from_bits(x[i] as u32) * f32::from_bits(y[i] as u32);
                        let base = f32::from_bits(acc[i] as u32);
                        let sum = if negate {
                            base - product
                        } else {
                            base + product
                        };
                        out[i] = u64::from(sum.to_bits());
                    }
                    from_lanes(&out, 32, count)
                }
            }
            // Floating-point comparisons, which produce a lane mask.
            (0xE, false) => self.neon_f32_bits(a, b, quad, |x, y| {
                let hit = if unsigned {
                    if size & 0b10 != 0 {
                        x > y
                    } else {
                        x >= y
                    }
                } else {
                    x == y
                };
                if hit {
                    u32::MAX
                } else {
                    0
                }
            }),
            // Floating-point maximum and minimum, and the Newton-Raphson
            // steps beside them.
            (0xF, false) => {
                let minimum = size & 0b10 != 0;
                if unsigned {
                    return self.neon_pairwise_minmax_f32(quad, vd, a, b, minimum);
                }
                self.neon_f32(a, b, quad, |x, y| if minimum { x.min(y) } else { x.max(y) })
            }
            (0xF, true) => {
                // VRECPS and VRSQRTS, the two iteration steps.
                let sqrt = size & 0b10 != 0;
                self.neon_f32(a, b, quad, move |x, y| {
                    if sqrt {
                        (3.0 - x * y) / 2.0
                    } else {
                        2.0 - x * y
                    }
                })
            }
            _ => return Err(self.neon_unimplemented(insn)),
        };
        self.neon_set(quad, vd, value);
        Ok(())
    }

    /// Apply `f` to each pair of lanes.
    #[inline]
    fn neon_lane_op(
        &self,
        a: u128,
        b: u128,
        esize: u32,
        count: u32,
        f: impl Fn(u64, u64) -> u64,
    ) -> u128 {
        let x = lanes_of(a, esize, count);
        let y = lanes_of(b, esize, count);
        let mut out = [0u64; 16];
        for i in 0..count as usize {
            out[i] = f(x[i], y[i]);
        }
        from_lanes(&out, esize, count)
    }

    /// The same over single-precision lanes.
    #[inline]
    fn neon_f32(&self, a: u128, b: u128, quad: bool, f: impl Fn(f32, f32) -> f32) -> u128 {
        self.neon_lane_op(a, b, 32, f32_lanes(quad), |x, y| {
            u64::from(f(f32::from_bits(x as u32), f32::from_bits(y as u32)).to_bits())
        })
    }

    /// The same, for the comparisons, whose result is a mask rather than a
    /// float.
    #[inline]
    fn neon_f32_bits(&self, a: u128, b: u128, quad: bool, f: impl Fn(f32, f32) -> u32) -> u128 {
        self.neon_lane_op(a, b, 32, f32_lanes(quad), |x, y| {
            u64::from(f(f32::from_bits(x as u32), f32::from_bits(y as u32)))
        })
    }

    /// `VPADD.F32`, which adds adjacent lanes within each source rather than
    /// across the two.
    fn neon_pairwise_f32(&mut self, quad: bool, vd: u8, a: u128, b: u128) -> Result<()> {
        let count = f32_lanes(quad);
        let x = lanes_of(a, 32, count);
        let y = lanes_of(b, 32, count);
        let mut out = [0u64; 16];
        let half = count as usize / 2;
        for i in 0..half {
            let sum = f32::from_bits(x[2 * i] as u32) + f32::from_bits(x[2 * i + 1] as u32);
            out[i] = u64::from(sum.to_bits());
            let sum = f32::from_bits(y[2 * i] as u32) + f32::from_bits(y[2 * i + 1] as u32);
            out[half + i] = u64::from(sum.to_bits());
        }
        self.neon_set(quad, vd, from_lanes(&out, 32, count));
        Ok(())
    }

    fn neon_pairwise_minmax_f32(
        &mut self,
        quad: bool,
        vd: u8,
        a: u128,
        b: u128,
        minimum: bool,
    ) -> Result<()> {
        let count = f32_lanes(quad);
        let x = lanes_of(a, 32, count);
        let y = lanes_of(b, 32, count);
        let mut out = [0u64; 16];
        let half = count as usize / 2;
        let pick = |p: f32, q: f32| if minimum { p.min(q) } else { p.max(q) };
        for i in 0..half {
            let v = pick(
                f32::from_bits(x[2 * i] as u32),
                f32::from_bits(x[2 * i + 1] as u32),
            );
            out[i] = u64::from(v.to_bits());
            let v = pick(
                f32::from_bits(y[2 * i] as u32),
                f32::from_bits(y[2 * i + 1] as u32),
            );
            out[half + i] = u64::from(v.to_bits());
        }
        self.neon_set(quad, vd, from_lanes(&out, 32, count));
        Ok(())
    }

    /// `VMOV`, `VMVN`, `VORR` and `VBIC` with a modified immediate.
    fn neon_immediate(&mut self, insn: u32) -> Result<()> {
        let quad = (insn >> 6) & 1 != 0;
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let cmode = (insn >> 8) & 0xF;
        let op = (insn >> 5) & 1;
        let imm8 = (((insn >> 24) & 1) << 7) | (((insn >> 16) & 0x7) << 4) | (insn & 0xF);
        let Some(pattern) = modified_immediate(cmode, op, imm8) else {
            return Err(self.neon_unimplemented(insn));
        };
        let wide = u128::from(pattern) | (u128::from(pattern) << 64);
        // An odd cmode below 1100 operates on the destination, VORR or VBIC
        // by `op`. Otherwise `op` inverts the constant, VMVN, except with
        // cmode 1110, where it chose the bit-per-byte expansion instead.
        let value = match (op, cmode) {
            (0, _) if cmode & 1 != 0 && cmode < 0b1100 => self.neon_get(quad, vd) | wide,
            (_, _) if cmode & 1 != 0 && cmode < 0b1100 => self.neon_get(quad, vd) & !wide,
            (1, 0b1110) | (0, _) => wide,
            _ => !wide,
        };
        self.neon_set(quad, vd, value);
        Ok(())
    }

    /// `VEXT`: a window sliding across the two sources.
    fn neon_ext(&mut self, insn: u32) -> Result<()> {
        let quad = (insn >> 6) & 1 != 0;
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let vn = (((insn >> 7) & 1) as u8) << 4 | ((insn >> 16) & 0xF) as u8;
        let vm = (((insn >> 5) & 1) as u8) << 4 | (insn & 0xF) as u8;
        let shift = ((insn >> 8) & 0xF) * 8;
        let a = self.neon_get(quad, vn);
        let b = self.neon_get(quad, vm);
        let width = if quad { 128 } else { 64 };
        let value = if shift == 0 {
            a
        } else {
            (a >> shift) | (b << (width - shift))
        };
        let value = if quad {
            value
        } else {
            value & u128::from(u64::MAX)
        };
        self.neon_set(quad, vd, value);
        Ok(())
    }

    /// The `1011` corner: the two-register miscellaneous operations, the table
    /// lookups, and `VDUP` from a lane.
    fn neon_two_reg(&mut self, insn: u32) -> Result<()> {
        // Bits 11:8 pick the sub-group: `10xx` is the table lookups, `1100`
        // is `VDUP` from a lane, and everything else is the miscellaneous
        // table. The register fields say nothing about which.
        match (insn >> 8) & 0xF {
            0b1000..=0b1011 => self.neon_table(insn),
            0b1100 => self.neon_dup_lane(insn),
            _ => self.neon_two_reg_misc(insn),
        }
    }

    /// The two-register miscellaneous group. Field A (bits 17:16) splits it
    /// into four tables and B (bits 10:6, `Q` at the bottom) names the
    /// operation within one; the same B means different things in different
    /// tables, so neither field can be read without the other.
    fn neon_two_reg_misc(&mut self, insn: u32) -> Result<()> {
        let quad = (insn >> 6) & 1 != 0;
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let vm = (((insn >> 5) & 1) as u8) << 4 | (insn & 0xF) as u8;
        let size = (insn >> 18) & 0b11;
        let table = (insn >> 16) & 0b11;
        let op = (insn >> 7) & 0b1111;
        let esize = 8 << size;
        let count = if quad { 128 } else { 64 } / esize;
        let m = self.neon_get(quad, vm);

        let value = match (table, op) {
            // VREV64, VREV32, VREV16: reverse the elements inside a container
            // whose width the opcode names.
            (0b00, 0b0000..=0b0010) => {
                let container = 64 >> op;
                let per = container / esize;
                let x = lanes_of(m, esize, count);
                let mut out = [0u64; 16];
                for i in 0..count as usize {
                    let group_base = (i / per as usize) * per as usize;
                    out[i] = x[group_base + (per as usize - 1 - (i % per as usize))];
                }
                from_lanes(&out, esize, count)
            }
            // VPADDL and VPADAL: adjacent pairs added into lanes twice as
            // wide, the second accumulating onto what the destination holds.
            (0b00, 0b0100 | 0b0101 | 0b1100 | 0b1101) => {
                let unsigned = op & 1 != 0;
                let x = lanes_of(m, esize, count);
                let wide = esize * 2;
                let prior = lanes_of(self.neon_get(quad, vd), wide, count / 2);
                let mut out = [0u64; 16];
                for i in 0..count as usize / 2 {
                    let widen = |v: u64| if unsigned { v } else { sext(v, esize) as u64 };
                    let sum = widen(x[2 * i]).wrapping_add(widen(x[2 * i + 1]));
                    out[i] = if op & 0b1000 != 0 {
                        sum.wrapping_add(prior[i])
                    } else {
                        sum
                    };
                }
                from_lanes(&out, wide, count / 2)
            }
            // The AES steps, always on Q registers: bit 6, which is `Q`
            // elsewhere in the group, picks between each pair.
            (0b00, 0b0110 | 0b0111) => {
                let opcode = 0b00100 | (op & 1) << 1 | (insn >> 6) & 1;
                let d = self.neon_get(true, vd);
                let n = self.neon_get(true, vm);
                let Some(result) = crate::cpu::crypto::aes(opcode, d, n) else {
                    return Err(self.neon_unimplemented(insn));
                };
                self.neon_set(true, vd, result);
                return Ok(());
            }
            // VCLS: the sign bit's copies below it. VCLZ: leading zeros.
            (0b00, 0b1000) => self.neon_lane_op(m, m, esize, count, |x, _| {
                let v = sext(x, esize);
                let flipped = if v < 0 { !v } else { v } as u64;
                u64::from((flipped << (64 - esize)).leading_zeros().min(esize)) - 1
            }),
            (0b00, 0b1001) => self.neon_lane_op(m, m, esize, count, |x, _| {
                u64::from((x << (64 - esize)).leading_zeros().min(esize))
            }),
            // VCNT: how many bits are set in each byte.
            (0b00, 0b1010) => {
                self.neon_lane_op(m, m, esize, count, |x, _| u64::from(x.count_ones()))
            }
            // VMVN.
            (0b00, 0b1011) => !m,
            // VQABS and VQNEG: the most negative value saturates rather than
            // wrapping back onto itself.
            (0b00, 0b1110 | 0b1111) => {
                let negate = op & 1 != 0;
                let (lo, hi) = (-(1i64 << (esize - 1)), (1i64 << (esize - 1)) - 1);
                self.neon_lane_op(m, m, esize, count, move |x, _| {
                    let v = sext(x, esize);
                    (if negate { -v } else { v.abs() }).clamp(lo, hi) as u64
                })
            }
            // The comparisons with zero, integer and floating point.
            (0b01, 0b0000..=0b0100 | 0b1000..=0b1100) => {
                let test = op & 0b111;
                let hit = move |ordering: Option<std::cmp::Ordering>| {
                    use std::cmp::Ordering::{Equal, Greater, Less};
                    match (test, ordering) {
                        (_, None) => false,
                        (0b000, Some(o)) => o == Greater,
                        (0b001, Some(o)) => o != Less,
                        (0b010, Some(o)) => o == Equal,
                        (0b011, Some(o)) => o != Greater,
                        (_, Some(o)) => o == Less,
                    }
                };
                if op & 0b1000 != 0 {
                    self.neon_lane_op(m, m, 32, f32_lanes(quad), move |x, _| {
                        let v = f32::from_bits(flush_input(x as u32));
                        if hit(v.partial_cmp(&0.0)) {
                            0xFFFF_FFFF
                        } else {
                            0
                        }
                    })
                } else {
                    let mask = u64::MAX >> (64 - esize);
                    self.neon_lane_op(m, m, esize, count, move |x, _| {
                        if hit(Some(sext(x, esize).cmp(&0))) {
                            mask
                        } else {
                            0
                        }
                    })
                }
            }
            // SHA1H, a scalar rotate, and in the next table SHA1SU1 and
            // SHA256SU0: the two-register SHA steps, on Q registers.
            (0b01, 0b0101) | (0b10, 0b0111) if table == 0b10 || (insn >> 6) & 1 != 0 => {
                let opcode = if table == 0b01 {
                    0
                } else {
                    1 + ((insn >> 6) & 1)
                };
                let d = self.neon_get(true, vd);
                let n = self.neon_get(true, vm);
                let Some(result) = crate::cpu::crypto::sha_two(opcode, d, n) else {
                    return Err(self.neon_unimplemented(insn));
                };
                self.neon_set(true, vd, result);
                return Ok(());
            }
            // VABS and VNEG, integer and floating point. The float forms only
            // touch the sign bit, NaNs and denormals included.
            (0b01, 0b0110 | 0b0111 | 0b1110 | 0b1111) => {
                let negate = op & 1 != 0;
                if op & 0b1000 != 0 {
                    self.neon_lane_op(m, m, 32, f32_lanes(quad), move |x, _| {
                        if negate {
                            x ^ 0x8000_0000
                        } else {
                            x & 0x7FFF_FFFF
                        }
                    })
                } else {
                    self.neon_lane_op(m, m, esize, count, move |x, _| {
                        let v = sext(x, esize);
                        (if negate {
                            v.wrapping_neg()
                        } else {
                            v.wrapping_abs()
                        }) as u64
                    })
                }
            }
            // VSWP, VTRN, VUZP and VZIP write both registers.
            (0b10, 0b0000..=0b0011) => {
                let d = self.neon_get(quad, vd);
                let (new_d, new_m) = permute(op, d, m, esize, count);
                self.neon_set(quad, vm, new_m);
                new_d
            }
            // The narrowing moves: a Q register's lanes into a D register's,
            // half as wide. VMOVN truncates; VQMOVUN saturates signed to
            // unsigned; VQMOVN saturates signed to signed or unsigned to
            // unsigned.
            (0b10, 0b0100 | 0b0101) => {
                let (bit6, narrow) = ((insn >> 6) & 1, esize);
                let wide = narrow * 2;
                let x = lanes_of(self.neon_get(true, vm), wide, 64 / narrow);
                let (smin, smax) = (-(1i64 << (narrow - 1)), (1i64 << (narrow - 1)) - 1);
                let umax = (1u64 << narrow) - 1;
                let mut out = [0u64; 16];
                for i in 0..(64 / narrow) as usize {
                    out[i] = match (op & 1, bit6) {
                        (0, 0) => x[i],
                        (0, _) => sext(x[i], wide).clamp(0, umax as i64) as u64,
                        (_, 0) => sext(x[i], wide).clamp(smin, smax) as u64,
                        _ => x[i].min(umax),
                    };
                }
                self.neon_set(false, vd, from_lanes(&out, narrow, 64 / narrow));
                return Ok(());
            }
            // VSHLL by the element width, the one shift the immediate form
            // cannot encode.
            (0b10, 0b0110) if (insn >> 6) & 1 == 0 => {
                let x = lanes_of(self.neon_get(false, vm), esize, 64 / esize);
                let mut out = [0u64; 16];
                for i in 0..(64 / esize) as usize {
                    out[i] = x[i] << esize;
                }
                self.neon_set(true, vd, from_lanes(&out, esize * 2, 64 / esize));
                return Ok(());
            }
            // VCVT between single and half precision, four lanes at a time.
            (0b10, 0b1100) if (insn >> 6) & 1 == 0 => {
                let x = lanes_of(self.neon_get(true, vm), 32, 4);
                let mut out = [0u64; 16];
                for i in 0..4 {
                    out[i] = u64::from(f32_to_f16_standard(x[i] as u32));
                }
                self.neon_set(false, vd, from_lanes(&out, 16, 4));
                return Ok(());
            }
            (0b10, 0b1110) if (insn >> 6) & 1 == 0 => {
                let x = lanes_of(self.neon_get(false, vm), 16, 4);
                let mut out = [0u64; 16];
                for i in 0..4 {
                    out[i] = u64::from(f16_to_f32_standard(x[i] as u16));
                }
                self.neon_set(true, vd, from_lanes(&out, 32, 4));
                return Ok(());
            }
            // VRECPE and VRSQRTE, floating point or unsigned fixed point.
            (0b11, 0b1000..=0b1011) => {
                let float = op & 0b0010 != 0;
                let sqrt = op & 1 != 0;
                self.neon_lane_op(m, m, 32, f32_lanes(quad), move |x, _| {
                    u64::from(match (float, sqrt) {
                        (true, false) => recip_estimate_f32(x as u32),
                        (true, true) => rsqrt_estimate_f32(x as u32),
                        (false, false) => recip_estimate_u32(x as u32),
                        (false, true) => rsqrt_estimate_u32(x as u32),
                    })
                })
            }
            // VCVT between single precision and 32-bit integers. To an
            // integer rounds towards zero and saturates, NaN giving zero.
            (0b11, 0b1100..=0b1111) => {
                let kind = op & 0b11;
                self.neon_lane_op(m, m, 32, f32_lanes(quad), move |x, _| {
                    let bits = x as u32;
                    u64::from(match kind {
                        0b00 => (bits as i32 as f32).to_bits(),
                        0b01 => (bits as f32).to_bits(),
                        0b10 => f32::from_bits(flush_input(bits)) as i32 as u32,
                        _ => f32::from_bits(flush_input(bits)) as u32,
                    })
                })
            }
            _ => return Err(self.neon_unimplemented(insn)),
        };
        self.neon_set(quad, vd, value);
        Ok(())
    }

    /// `VTBL` and `VTBX`: a byte-wise gather out of one to four consecutive
    /// `D` registers, with out-of-range indices reading zero (`VTBL`) or
    /// leaving the destination byte alone (`VTBX`).
    fn neon_table(&mut self, insn: u32) -> Result<()> {
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let vn = (((insn >> 7) & 1) as u8) << 4 | ((insn >> 16) & 0xF) as u8;
        let vm = (((insn >> 5) & 1) as u8) << 4 | (insn & 0xF) as u8;
        let length = ((insn >> 8) & 0b11) + 1;
        let extend = (insn >> 6) & 1 != 0;
        let mut table = [0u8; 32];
        for i in 0..length {
            let d = self.vfp_d((vn + i as u8) & 0x1F);
            table[i as usize * 8..(i as usize + 1) * 8].copy_from_slice(&d.to_le_bytes());
        }
        let indices = self.vfp_d(vm).to_le_bytes();
        let current = self.vfp_d(vd).to_le_bytes();
        let mut out = [0u8; 8];
        for (i, slot) in out.iter_mut().enumerate() {
            let index = indices[i] as usize;
            *slot = if index < length as usize * 8 {
                table[index]
            } else if extend {
                current[i]
            } else {
                0
            };
        }
        self.set_vfp_d(vd, u64::from_le_bytes(out));
        Ok(())
    }

    /// `VDUP` from one lane of a `D` register to every lane of the result.
    fn neon_dup_lane(&mut self, insn: u32) -> Result<()> {
        let quad = (insn >> 6) & 1 != 0;
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let vm = (((insn >> 5) & 1) as u8) << 4 | (insn & 0xF) as u8;
        let imm4 = (insn >> 16) & 0xF;
        // The lowest set bit of imm4 says the element size, and the bits above
        // it are the index.
        let (esize, index) = if imm4 & 1 != 0 {
            (8, imm4 >> 1)
        } else if imm4 & 0b10 != 0 {
            (16, imm4 >> 2)
        } else if imm4 & 0b100 != 0 {
            (32, imm4 >> 3)
        } else {
            return Err(self.neon_unimplemented(insn));
        };
        let source = u128::from(self.vfp_d(vm));
        let lane = lanes_of(source, esize, 64 / esize)[index as usize];
        let count = if quad { 128 } else { 64 } / esize;
        let mut out = [0u64; 16];
        for slot in out.iter_mut().take(count as usize) {
            *slot = lane;
        }
        let value = from_lanes(&out, esize, count);
        self.neon_set(quad, vd, value);
        Ok(())
    }

    /// The two-registers-and-a-scalar group: every lane multiplied by one lane
    /// of a second register. The float multiply-accumulate by element is the
    /// single most common NEON encoding in the title.
    fn neon_by_scalar(&mut self, insn: u32) -> Result<()> {
        let quad = (insn >> 24) & 1 != 0;
        let size = (insn >> 20) & 0b11;
        let opc = (insn >> 8) & 0xF;
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let vn = (((insn >> 7) & 1) as u8) << 4 | ((insn >> 16) & 0xF) as u8;
        // The scalar's register and lane are packed together, differently for
        // each element size.
        let (vm, index) = if size == 0b01 {
            // A 16-bit scalar lives in D0..D7 and needs two index bits.
            (
                (insn & 0x7) as u8,
                (((insn >> 5) & 1) << 1) | ((insn >> 3) & 1),
            )
        } else {
            ((insn & 0xF) as u8, (insn >> 5) & 1)
        };
        let esize = 8 << size;
        let count = if quad { 128 } else { 64 } / esize;
        let a = self.neon_get(quad, vn);
        let scalar = lanes_of(u128::from(self.vfp_d(vm)), esize, 64 / esize)[index as usize];

        let value = match opc {
            // VMUL, VMLA and VMLS by element, in floating point.
            0x9 | 0x1 | 0x5 => {
                let s = f32::from_bits(scalar as u32);
                let x = lanes_of(a, 32, count);
                let acc = lanes_of(self.neon_get(quad, vd), 32, count);
                let mut out = [0u64; 16];
                for i in 0..count as usize {
                    let product = f32::from_bits(x[i] as u32) * s;
                    let v = match opc {
                        0x9 => product,
                        0x1 => f32::from_bits(acc[i] as u32) + product,
                        _ => f32::from_bits(acc[i] as u32) - product,
                    };
                    out[i] = u64::from(v.to_bits());
                }
                from_lanes(&out, 32, count)
            }
            // The integer forms of the same three.
            0x8 | 0x0 | 0x4 => {
                let x = lanes_of(a, esize, count);
                let acc = lanes_of(self.neon_get(quad, vd), esize, count);
                let mut out = [0u64; 16];
                for i in 0..count as usize {
                    let product = x[i].wrapping_mul(scalar);
                    out[i] = match opc {
                        0x8 => product,
                        0x0 => acc[i].wrapping_add(product),
                        _ => acc[i].wrapping_sub(product),
                    };
                }
                from_lanes(&out, esize, count)
            }
            _ => return Err(self.neon_unimplemented(insn)),
        };
        self.neon_set(quad, vd, value);
        Ok(())
    }

    /// The two-registers-and-a-shift group: the immediate shifts.
    /// Three registers of different lengths: the long forms widen two `D`
    /// registers into a `Q`, the wide forms widen the second operand to
    /// meet a `Q`, and the narrow ones keep the high half of each lane of a
    /// `Q` result in a `D`. `size` names the narrow element.
    fn neon_three_different(&mut self, insn: u32) -> Result<()> {
        let unsigned = (insn >> 24) & 1 != 0;
        let size = (insn >> 20) & 0b11;
        let opc = (insn >> 8) & 0xF;
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let vn = (((insn >> 7) & 1) as u8) << 4 | ((insn >> 16) & 0xF) as u8;
        let vm = (((insn >> 5) & 1) as u8) << 4 | (insn & 0xF) as u8;
        let narrow = 8 << size;
        let wide = narrow * 2;
        let count = 64 / narrow;
        let mut out = [0u64; 16];

        // The narrowing forms: Qn op Qm, the high half of each lane kept.
        if matches!(opc, 0x4 | 0x6) {
            let x = lanes_of(self.neon_get(true, vn), wide, count);
            let y = lanes_of(self.neon_get(true, vm), wide, count);
            for (i, slot) in out.iter_mut().enumerate().take(count as usize) {
                let full = if opc == 0x4 {
                    x[i].wrapping_add(y[i])
                } else {
                    x[i].wrapping_sub(y[i])
                };
                let rounding = if unsigned { 1u64 << (narrow - 1) } else { 0 };
                *slot = full.wrapping_add(rounding) >> narrow;
            }
            self.neon_set(false, vd, from_lanes(&out, narrow, count));
            return Ok(());
        }

        // Everything else produces a Q register of wide lanes. The first
        // operand is a Q register for the wide forms and a D otherwise.
        let wide_n = matches!(opc, 0x1 | 0x3);
        let n = if wide_n {
            lanes_of(self.neon_get(true, vn), wide, count)
        } else {
            lanes_of(self.neon_get(false, vn), narrow, count)
        };
        let m = lanes_of(self.neon_get(false, vm), narrow, count);
        let acc = lanes_of(self.neon_get(true, vd), wide, count);
        // The doubling forms are signed whatever U is; U is not a flag there.
        let signed = !unsigned || matches!(opc, 0x9 | 0xB | 0xD);
        let widen = |v: u64, bits: u32| int_lane(v, bits, !signed);
        for i in 0..count as usize {
            let a = widen(n[i], if wide_n { wide } else { narrow });
            let b = widen(m[i], narrow);
            let accumulated = int_lane(acc[i], wide, !signed);
            let value = match opc {
                0x0 | 0x1 => a + b,
                0x2 | 0x3 => a - b,
                0x5 => accumulated + (a - b).abs(),
                0x7 => (a - b).abs(),
                0x8 => accumulated + a * b,
                0xA => accumulated - a * b,
                0xC => a * b,
                // VQDMLAL, VQDMLSL and VQDMULL: twice the product,
                // saturated, then (for the first two) accumulated with a
                // second saturation.
                0x9 | 0xB | 0xD => {
                    let doubled = saturate(2 * a * b, wide, false);
                    match opc {
                        0x9 => saturate(accumulated + doubled, wide, false),
                        0xB => saturate(accumulated - doubled, wide, false),
                        _ => doubled,
                    }
                }
                // VMULL.P8: a carry-less product of bytes into halfwords.
                0xE if size == 0 && !unsigned => {
                    i128::from(crate::cpu::crypto::poly_mul(n[i], m[i], 8) as u64)
                }
                _ => return Err(self.neon_unimplemented(insn)),
            };
            out[i] = value as u64;
        }
        self.neon_set(true, vd, from_lanes(&out, wide, count));
        Ok(())
    }

    /// The shifts by an immediate, and the narrowing, widening and
    /// fixed-point forms beside them.
    ///
    /// The element size is the highest set bit of `L:imm6`, and the amount is
    /// what lies under it: counted up from the element size for the left
    /// shifts and down from twice it for the right ones. The narrowing forms
    /// name the destination's size the same way, and `B` (bit 6, which is
    /// `Q` elsewhere) says whether they round.
    fn neon_shift_immediate(&mut self, insn: u32) -> Result<()> {
        let unsigned = (insn >> 24) & 1 != 0;
        let quad = (insn >> 6) & 1 != 0;
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let vm = (((insn >> 5) & 1) as u8) << 4 | (insn & 0xF) as u8;
        let opc = (insn >> 8) & 0xF;
        let imm6 = (insn >> 16) & 0x3F;
        let esize = if (insn >> 7) & 1 != 0 {
            64
        } else if imm6 & 0b10_0000 != 0 {
            32
        } else if imm6 & 0b01_0000 != 0 {
            16
        } else {
            8
        };
        let field = if esize == 64 {
            imm6
        } else {
            imm6 & (esize - 1)
        };
        let (left, right) = (field, esize - field);
        let count = if quad { 128 } else { 64 } / esize;
        let m = self.neon_get(quad, vm);
        let d = self.neon_get(quad, vd);
        let mask = u64::MAX >> (64 - esize);

        let value = match opc {
            // VSHR, VSRA, VRSHR and VRSRA: right shifts, rounded by bit 9
            // and accumulated by bit 8.
            0x0..=0x3 => {
                let round = opc & 0b10 != 0;
                let shifted = self.neon_lane_op(m, m, esize, count, move |x, _| {
                    shift_by(int_lane(x, esize, unsigned), -(right as i32), round, esize) as u64
                });
                if opc & 1 != 0 {
                    self.neon_lane_op(d, shifted, esize, count, |x, y| x.wrapping_add(y))
                } else {
                    shifted
                }
            }
            // VSRI: the shifted lane over the destination's top bits, which
            // it leaves alone.
            0x4 if unsigned => self.neon_lane_op(d, m, esize, count, move |x, y| {
                let kept = if right >= esize {
                    mask
                } else {
                    !(mask >> right) & mask
                };
                let inserted = if right >= esize { 0 } else { y >> right };
                x & kept | inserted
            }),
            // VSHL, and VSLI, which keeps the destination's bits below it.
            0x5 => self.neon_lane_op(d, m, esize, count, move |x, y| {
                let kept = if unsigned {
                    x & ((1u64 << left) - 1)
                } else {
                    0
                };
                (y << left) & mask | kept
            }),
            // VQSHL, and VQSHLU, which saturates a signed lane to unsigned.
            0x6 | 0x7 => {
                let (signed_in, unsigned_out) = match (opc, unsigned) {
                    (0x6, _) => (true, true),
                    (_, u) => (!u, u),
                };
                self.neon_lane_op(m, m, esize, count, move |x, _| {
                    let value = int_lane(x, esize, !signed_in);
                    saturate(
                        shift_by(value, left as i32, false, esize),
                        esize,
                        unsigned_out,
                    ) as u64
                })
            }
            // The narrowing right shifts: VSHRN and VRSHRN truncate;
            // VQSHRUN and VQRSHRUN saturate signed to unsigned; VQSHRN and
            // VQRSHRN saturate within one signedness.
            0x8 | 0x9 => {
                let round = quad;
                let wide = esize * 2;
                let x = lanes_of(self.neon_get(true, vm), wide, 64 / esize);
                let mut out = [0u64; 16];
                for (i, slot) in out.iter_mut().enumerate().take((64 / esize) as usize) {
                    let signed_in = !(opc == 0x9 && unsigned);
                    let shifted = shift_by(
                        int_lane(x[i], wide, !signed_in),
                        -(right as i32),
                        round,
                        wide,
                    );
                    *slot = match (opc, unsigned) {
                        (0x8, false) => shifted as u64,
                        (0x8, true) => saturate(shifted, esize, true) as u64,
                        (_, u) => saturate(shifted, esize, u) as u64,
                    };
                }
                self.neon_set(false, vd, from_lanes(&out, esize, 64 / esize));
                return Ok(());
            }
            // VSHLL and VMOVL: each lane of a D register widened, then
            // shifted into the wider lane.
            0xA if !quad => {
                let x = lanes_of(self.neon_get(false, vm), esize, 64 / esize);
                let mut out = [0u64; 16];
                for (i, slot) in out.iter_mut().enumerate().take((64 / esize) as usize) {
                    *slot = (int_lane(x[i], esize, unsigned) << left) as u64;
                }
                self.neon_set(true, vd, from_lanes(&out, esize * 2, 64 / esize));
                return Ok(());
            }
            // VCVT between single precision and fixed point, `64 - imm6`
            // fraction bits: to a float rounds to nearest, and to fixed
            // point rounds towards zero and saturates.
            0xE | 0xF if esize == 32 => {
                let fraction = 64 - imm6;
                let scale = (fraction as f64).exp2();
                self.neon_lane_op(m, m, 32, f32_lanes(quad), move |x, _| {
                    let bits = x as u32;
                    u64::from(if opc == 0xE {
                        let value = if unsigned {
                            f64::from(bits)
                        } else {
                            f64::from(bits as i32)
                        };
                        ((value / scale) as f32).to_bits()
                    } else {
                        let value = f64::from(f32::from_bits(flush_input(bits))) * scale;
                        if unsigned {
                            value as u32
                        } else {
                            value as i32 as u32
                        }
                    })
                })
            }
            _ => return Err(self.neon_unimplemented(insn)),
        };
        self.neon_set(quad, vd, value);
        Ok(())
    }
}

/// One integer lane as a number, sign-extended unless `unsigned`.
fn int_lane(lane: Lane, esize: u32, unsigned: bool) -> i128 {
    if unsigned {
        i128::from(lane)
    } else {
        i128::from(sext(lane, esize))
    }
}

/// `value` clamped to what an `esize`-bit lane of that signedness holds.
fn saturate(value: i128, esize: u32, unsigned: bool) -> i128 {
    if unsigned {
        value.clamp(0, (1i128 << esize) - 1)
    } else {
        value.clamp(-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1)
    }
}

/// `value` shifted left by `shift`, or right by its magnitude when it is
/// negative, rounding to nearest when `round` is set.
///
/// A left shift of the lane's width or more leaves nothing of the value in
/// the lane: that is zero for the plain forms, and for the saturating ones
/// an overflow in the value's direction, so a nonzero value comes back as
/// one lane-width past the range with its sign. A right shift past the lane
/// leaves only the sign, which the wide arithmetic gives on its own.
fn shift_by(value: i128, shift: i32, round: bool, esize: u32) -> i128 {
    if shift >= 0 {
        if shift as u32 >= esize {
            return value.signum() << esize;
        }
        value << shift
    } else {
        let amount = (-shift).min(esize as i32 + 1) as u32;
        let rounding = if round { 1i128 << (amount - 1) } else { 0 };
        (value + rounding) >> amount
    }
}

impl Cpu {
    /// The NEON element and structure load/store space, `1111 0100`.
    ///
    /// Mario Kart 8 Deluxe issues 14,168 of these and all but a handful are
    /// `VLD1`/`VST1` moving one or two `D` registers, the vector equivalent
    /// of a `memcpy` step. The interleaving forms (`VLD2`/`VLD3`/`VLD4`)
    /// share the space: member `k` of structure `e` is lane `e` of
    /// `D[d + r + k * inc]`.
    pub(super) fn a32_neon_load_store(&mut self, insn: u32) -> Result<()> {
        let single = (insn >> 23) & 1 != 0;
        let load = (insn >> 21) & 1 != 0;
        let rn = ((insn >> 16) & 0xF) as u8;
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let rm = (insn & 0xF) as u8;
        let base = self.r32(rn);
        let mut addr = base;

        let bytes = if !single {
            // Whole registers, `type` saying how many.
            // `type`: how many members a structure has, how far apart their
            // registers are, and how many registers each member fills.
            let (members, inc, rows) = match (insn >> 8) & 0xF {
                0b0111 => (1, 1, 1),
                0b1010 => (1, 1, 2),
                0b0110 => (1, 1, 3),
                0b0010 => (1, 1, 4),
                0b1000 => (2, 1, 1),
                0b1001 => (2, 2, 1),
                0b0011 => (2, 2, 2),
                0b0100 => (3, 1, 1),
                0b0101 => (3, 2, 1),
                0b0000 => (4, 1, 1),
                0b0001 => (4, 2, 1),
                _ => return Err(self.neon_unimplemented(insn)),
            };
            if members > 1 {
                let esize = 8u32 << ((insn >> 6) & 0b11);
                let ebytes = esize / 8;
                let lanes = 64 / esize;
                for row in 0..rows {
                    for e in 0..lanes {
                        for k in 0..members {
                            let d = (vd + row + k * inc) & 0x1F;
                            let shift = e * esize;
                            let mask = (u64::MAX >> (64 - esize)) << shift;
                            let current = self.vfp_d(d);
                            if load {
                                let value = match ebytes {
                                    1 => u64::from(self.mem.read_u8(addr)?),
                                    2 => u64::from(self.mem.read_u16(addr)?),
                                    4 => u64::from(self.mem.read_u32(addr)?),
                                    _ => {
                                        u64::from(self.mem.read_u32(addr)?)
                                            | u64::from(self.mem.read_u32(addr.wrapping_add(4))?)
                                                << 32
                                    }
                                };
                                self.set_vfp_d(d, current & !mask | (value << shift) & mask);
                            } else {
                                let value = (current & mask) >> shift;
                                match ebytes {
                                    1 => self.mem.write_u8(addr, value as u8)?,
                                    2 => self.mem.write_u16(addr, value as u16)?,
                                    4 => self.mem.write_u32(addr, value as u32)?,
                                    _ => {
                                        self.mem.write_u32(addr, value as u32)?;
                                        self.mem.write_u32(
                                            addr.wrapping_add(4),
                                            (value >> 32) as u32,
                                        )?;
                                    }
                                }
                            }
                            addr = addr.wrapping_add(ebytes);
                        }
                    }
                }
                u32::from(members * rows) * 8
            } else {
                let registers = rows;
                for i in 0..registers {
                    let d = (vd + i) & 0x1F;
                    if load {
                        let lo = self.mem.read_u32(addr)?;
                        let hi = self.mem.read_u32(addr.wrapping_add(4))?;
                        self.set_vfp_d(d, u64::from(lo) | (u64::from(hi) << 32));
                    } else {
                        let val = self.vfp_d(d);
                        self.mem.write_u32(addr, val as u32)?;
                        self.mem
                            .write_u32(addr.wrapping_add(4), (val >> 32) as u32)?;
                    }
                    addr = addr.wrapping_add(8);
                }
                u32::from(registers) * 8
            }
        } else if (insn >> 10) & 0b11 == 0b11 {
            // One element broadcast to every lane, optionally into two
            // registers.
            let size = 8 << ((insn >> 6) & 0b11);
            let registers = 1 + ((insn >> 5) & 1);
            if !load {
                return Err(self.neon_unimplemented(insn));
            }
            let value = match size {
                8 => u64::from(self.mem.read_u8(addr)?),
                16 => u64::from(self.mem.read_u16(addr)?),
                _ => u64::from(self.mem.read_u32(addr)?),
            };
            let mut lanes = [0u64; 16];
            for slot in lanes.iter_mut().take((64 / size) as usize) {
                *slot = value;
            }
            let broadcast = from_lanes(&lanes, size, 64 / size) as u64;
            for i in 0..registers as u8 {
                self.set_vfp_d((vd + i) & 0x1F, broadcast);
            }
            size / 8
        } else {
            // One lane of one register. The index and the alignment share a
            // field whose split depends on the element size.
            if (insn >> 8) & 0b11 != 0 {
                return Err(self.neon_unimplemented(insn));
            }
            let size = (insn >> 10) & 0b11;
            let index_align = (insn >> 4) & 0xF;
            let (esize, index) = match size {
                0b00 => (8, index_align >> 1),
                0b01 => (16, index_align >> 2),
                _ => (32, index_align >> 3),
            };
            let lanes = 64 / esize;
            let current = u128::from(self.vfp_d(vd));
            let mut split = lanes_of(current, esize, lanes);
            if load {
                split[index as usize] = match esize {
                    8 => u64::from(self.mem.read_u8(addr)?),
                    16 => u64::from(self.mem.read_u16(addr)?),
                    _ => u64::from(self.mem.read_u32(addr)?),
                };
                let value = from_lanes(&split, esize, lanes) as u64;
                self.set_vfp_d(vd, value);
            } else {
                let value = split[index as usize];
                match esize {
                    8 => self.mem.write_u8(addr, value as u8)?,
                    16 => self.mem.write_u16(addr, value as u16)?,
                    _ => self.mem.write_u32(addr, value as u32)?,
                }
            }
            esize / 8
        };

        // `Rm` is not always a register: 15 means leave the base alone, and 13
        // means advance it by what was transferred.
        match rm {
            0b1111 => {}
            0b1101 => self.set_r32(rn, base.wrapping_add(bytes)),
            _ => {
                let offset = self.r32(rm);
                self.set_r32(rn, base.wrapping_add(offset));
            }
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }
}

/// `VSWP`, `VTRN`, `VUZP` and `VZIP` (`op` 0 to 3) on `d` and `m`, giving the
/// new `d` and the new `m`.
fn permute(op: u32, d: u128, m: u128, esize: u32, count: u32) -> (u128, u128) {
    let (x, y) = (lanes_of(d, esize, count), lanes_of(m, esize, count));
    let n = count as usize;
    let (mut nd, mut nm) = ([0u64; 16], [0u64; 16]);
    match op {
        0b00 => return (m, d),
        // Transpose: each pair's odd lane of `d` trades with its even lane
        // of `m`.
        0b01 => {
            nd[..n].copy_from_slice(&x[..n]);
            nm[..n].copy_from_slice(&y[..n]);
            for i in (0..n).step_by(2) {
                nd[i + 1] = y[i];
                nm[i] = x[i + 1];
            }
        }
        // Unzip: the even lanes of `d` then `m` into `d`, the odd into `m`.
        0b10 => {
            let joined: Vec<u64> = x[..n].iter().chain(&y[..n]).copied().collect();
            for i in 0..n {
                nd[i] = joined[2 * i];
                nm[i] = joined[2 * i + 1];
            }
        }
        // Zip: the lanes of `d` and `m` interleaved, the first half into
        // `d` and the second into `m`.
        _ => {
            let joined: Vec<u64> = (0..n).flat_map(|i| [x[i], y[i]]).collect();
            nd[..n].copy_from_slice(&joined[..n]);
            nm[..n].copy_from_slice(&joined[n..]);
        }
    }
    (from_lanes(&nd, esize, count), from_lanes(&nm, esize, count))
}

/// A single-precision input as NEON reads it: the standard FPSCR flushes a
/// denormal to a zero of the same sign.
fn flush_input(bits: u32) -> u32 {
    if bits & 0x7F80_0000 == 0 {
        bits & 0x8000_0000
    } else {
        bits
    }
}

/// `VCVT.F16.F32` under the standard FPSCR: a denormal input is flushed and
/// a NaN becomes the default NaN.
fn f32_to_f16_standard(bits: u32) -> u16 {
    let bits = flush_input(bits);
    if f32::from_bits(bits).is_nan() {
        return 0x7E00;
    }
    crate::gpu::surface::f32_to_f16(f32::from_bits(bits))
}

/// `VCVT.F32.F16` under the standard FPSCR: exact, bar a NaN, which becomes
/// the default NaN.
fn f16_to_f32_standard(bits: u16) -> u32 {
    if bits & 0x7C00 == 0x7C00 && bits & 0x03FF != 0 {
        return 0x7FC0_0000;
    }
    crate::gpu::surface::f16_to_f32(bits).to_bits()
}

/// The ARM ARM's `RecipEstimate`: `a` in 256..512 standing for [0.5, 1),
/// and the result an 8-bit-accurate reciprocal in 256..512 standing for
/// (1, 2].
fn recip_table(a: u32) -> u32 {
    let a = a * 2 + 1;
    let b = (1 << 19) / a;
    // `(b + 1) DIV 2` in the manual: halving, rounded to nearest.
    b.div_ceil(2)
}

/// The ARM ARM's `RecipSqrtEstimate`: `a` in 128..512 standing for
/// [0.25, 1), the result in 256..512. Below a half the input keeps its
/// bottom bit, in units of 1/512; above, it is taken in units of 1/256.
/// Both are then rounded to the nearest 1/1024.
fn rsqrt_table(a: u32) -> u32 {
    let a = if a < 256 {
        a * 2 + 1
    } else {
        ((a >> 1 << 1) + 1) * 2
    };
    // The largest b with b < 2^14 / sqrt(a), found by counting up.
    let mut b = 512u32;
    while a * (b + 1) * (b + 1) < 1 << 28 {
        b += 1;
    }
    b.div_ceil(2)
}

/// `VRECPE.F32`, the ARMv7 single-precision reciprocal estimate under the
/// standard FPSCR. The input is scaled into [0.5, 1) and its top eight
/// fraction bits pick the estimate; the exponent is worked out apart.
fn recip_estimate_f32(bits: u32) -> u32 {
    let bits = flush_input(bits);
    let sign = bits & 0x8000_0000;
    let exp = (bits >> 23) & 0xFF;
    if f32::from_bits(bits).is_nan() {
        return 0x7FC0_0000;
    }
    if exp == 0xFF {
        return sign;
    }
    if exp == 0 {
        return sign | 0x7F80_0000;
    }
    // From 2^126 up the reciprocal is below the smallest normal, which the
    // standard FPSCR flushes to zero.
    if exp >= 253 {
        return sign;
    }
    let estimate = recip_table(0x100 | (bits >> 15) & 0xFF);
    sign | (253 - exp) << 23 | (estimate & 0xFF) << 15
}

/// `VRSQRTE.F32`, the ARMv7 single-precision reciprocal square-root estimate
/// under the standard FPSCR.
fn rsqrt_estimate_f32(bits: u32) -> u32 {
    let bits = flush_input(bits);
    let exp = (bits >> 23) & 0xFF;
    if f32::from_bits(bits).is_nan() {
        return 0x7FC0_0000;
    }
    if exp == 0 {
        return bits & 0x8000_0000 | 0x7F80_0000;
    }
    if bits & 0x8000_0000 != 0 {
        return 0x7FC0_0000;
    }
    if exp == 0xFF {
        return 0;
    }
    // The fraction keeps one more bit when the exponent is even, so that the
    // square root of the scale is exact.
    let fraction = if exp & 1 == 0 {
        (bits >> 15) & 0xFF | 0x100
    } else {
        (bits >> 16) & 0x7F | 0x80
    };
    let estimate = rsqrt_table(fraction);
    let result_exp = (380 - exp) / 2;
    result_exp << 23 | (estimate & 0xFF) << 15
}

/// `VRECPE.U32`: the estimate of an unsigned fraction in [0, 1), as the top
/// nine bits say it, and all ones below a half.
fn recip_estimate_u32(value: u32) -> u32 {
    if value >> 31 == 0 {
        return u32::MAX;
    }
    recip_table(value >> 23) << 23
}

/// `VRSQRTE.U32`: the same for the reciprocal square root, all ones below a
/// quarter.
fn rsqrt_estimate_u32(value: u32) -> u32 {
    if value >> 30 == 0 {
        return u32::MAX;
    }
    rsqrt_table(value >> 23) << 23
}
