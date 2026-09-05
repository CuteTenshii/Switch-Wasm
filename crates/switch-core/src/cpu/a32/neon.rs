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
//! rather than being quietly approximated.

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
        (0b111, 0, 0) => {
            // Each bit of the byte becomes a whole byte of the result.
            let mut out = 0u64;
            for i in 0..8 {
                if byte & (1 << i) != 0 {
                    out |= 0xFFu64 << (8 * i);
                }
            }
            out
        }
        (0b111, 0, 1) => {
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
    pub(super) fn a32_neon_data(&mut self, insn: u32) -> Result<()> {
        let result = if (insn >> 23) & 1 == 0 {
            self.neon_three_same(insn)
        } else if (insn >> 20) & 0b11 == 0b11 {
            // Bit 24 is what tells `VEXT` from the two-register group: both
            // put 1011 in bits 23:20, and `VEXT`'s `imm4` freely reaches the
            // values the other group uses to name an operation.
            if (insn >> 4) & 1 != 0 {
                Err(self.neon_unimplemented(insn))
            } else if (insn >> 24) & 1 == 0 {
                self.neon_ext(insn)
            } else {
                self.neon_two_reg(insn)
            }
        } else if (insn >> 4) & 1 == 0 {
            self.neon_by_scalar(insn)
        } else if (insn >> 19) & 0b111 == 0 {
            self.neon_immediate(insn)
        } else {
            self.neon_shift_immediate(insn)
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
            // Integer maximum and minimum.
            (0x6, false) => self.neon_lane_op(a, b, esize, count, |x, y| {
                if unsigned {
                    if op_max(insn) {
                        x.max(y)
                    } else {
                        x.min(y)
                    }
                } else if op_max(insn) {
                    sext(x, esize).max(sext(y, esize)) as u64
                } else {
                    sext(x, esize).min(sext(y, esize)) as u64
                }
            }),
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
        // `op` with a cmode that carries an operation means VORR or VBIC
        // against the destination rather than a plain move.
        let value = match (op, cmode & 0b1001) {
            (1, 0b0000 | 0b0001) => self.neon_get(quad, vd) & !wide,
            (0, 0b0000 | 0b0001) if cmode & 0b0001 != 0 => self.neon_get(quad, vd) | wide,
            (1, _) => !wide,
            _ => wide,
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

    fn neon_two_reg_misc(&mut self, insn: u32) -> Result<()> {
        let quad = (insn >> 6) & 1 != 0;
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let vm = (((insn >> 5) & 1) as u8) << 4 | (insn & 0xF) as u8;
        let size = (insn >> 18) & 0b11;
        let opc = (insn >> 7) & 0b1111;
        let esize = 8 << size;
        let count = if quad { 128 } else { 64 } / esize;
        let m = self.neon_get(quad, vm);

        let value = match opc {
            // VREV64, VREV32, VREV16: reverse the elements inside a container
            // whose width the opcode names.
            0b0000..=0b0010 => {
                let container = 64 >> opc;
                let per = container / esize;
                let x = lanes_of(m, esize, count);
                let mut out = [0u64; 16];
                for i in 0..count as usize {
                    let group_base = (i / per as usize) * per as usize;
                    out[i] = x[group_base + (per as usize - 1 - (i % per as usize))];
                }
                from_lanes(&out, esize, count)
            }
            // VCNT: how many bits are set in each byte.
            0b1010 => self.neon_lane_op(m, m, esize, count, |x, _| u64::from(x.count_ones())),
            // VMVN.
            0b1011 => !m,
            // VNEG and VABS, integer and floating point.
            0b0110 | 0b0111 | 0b1110 | 0b1111 => {
                let negate = opc & 1 != 0;
                if opc & 0b1000 != 0 {
                    self.neon_f32(m, m, quad, move |x, _| if negate { -x } else { x.abs() })
                } else {
                    self.neon_lane_op(m, m, esize, count, move |x, _| {
                        let v = sext(x, esize);
                        (if negate { -v } else { v.abs() }) as u64
                    })
                }
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
    fn neon_shift_immediate(&mut self, insn: u32) -> Result<()> {
        let unsigned = (insn >> 24) & 1 != 0;
        let quad = (insn >> 6) & 1 != 0;
        let vd = (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8;
        let vm = (((insn >> 5) & 1) as u8) << 4 | (insn & 0xF) as u8;
        let opc = (insn >> 8) & 0xF;
        let imm6 = (insn >> 16) & 0x3F;
        // The element size is the highest set bit of the immediate field, and
        // the amount is what is left under it.
        let (esize, amount) = if imm6 & 0b100000 != 0 {
            (64, imm6)
        } else if imm6 & 0b010000 != 0 {
            (32, imm6 & 0x1F)
        } else if imm6 & 0b001000 != 0 {
            (16, imm6 & 0xF)
        } else {
            (8, imm6 & 0x7)
        };
        let count = if quad { 128 } else { 64 } / esize;
        let m = self.neon_get(quad, vm);
        let value = match opc {
            // VSHR and VSRA: the amount counts down from the element size.
            0x0 | 0x1 => {
                let shift = esize - amount;
                self.neon_lane_op(m, m, esize, count, move |x, _| {
                    if unsigned {
                        x >> shift.min(63)
                    } else {
                        (sext(x, esize) >> shift.min(63)) as u64
                    }
                })
            }
            // VSHL, whose amount counts up.
            0x5 => self.neon_lane_op(m, m, esize, count, move |x, _| x << amount.min(63)),
            _ => return Err(self.neon_unimplemented(insn)),
        };
        self.neon_set(quad, vd, value);
        Ok(())
    }
}

/// Bit 21 selects maximum over minimum in the integer max/min encodings.
#[inline]
fn op_max(insn: u32) -> bool {
    (insn >> 21) & 1 == 0
}

impl Cpu {
    /// The NEON element and structure load/store space, `1111 0100`.
    ///
    /// Mario Kart 8 Deluxe issues 14,168 of these and all but a handful are
    /// `VLD1`/`VST1` moving one or two `D` registers, the vector equivalent
    /// of a `memcpy` step. The interleaving forms (`VLD2`/`VLD3`/`VLD4`) are
    /// in this space too and are not implemented.
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
            let registers = match (insn >> 8) & 0xF {
                0b0111 => 1,
                0b1010 => 2,
                0b0110 => 3,
                0b0010 => 4,
                _ => return Err(self.neon_unimplemented(insn)),
            };
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
