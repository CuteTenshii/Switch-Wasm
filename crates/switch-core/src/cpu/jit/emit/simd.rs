//! Advanced SIMD and SIMD&FP memory accesses as wasm SIMD128, decoded from
//! the instruction word. Anything not matched here answers `false` and keeps
//! its block on the interpreter.

use super::*;

/// Wasm SIMD opcodes, after the `0xFD` prefix.
mod op {
    pub const V128_LOAD: u32 = 0x00;
    /// `load{8,16,32,64}_splat` are consecutive.
    pub const LOAD8_SPLAT: u32 = 0x07;
    pub const V128_STORE: u32 = 0x0B;
    pub const I8X16_SPLAT: u32 = 0x0F;
    pub const I16X8_SPLAT: u32 = 0x10;
    pub const I32X4_SPLAT: u32 = 0x11;
    pub const I64X2_SPLAT: u32 = 0x12;
    pub const I8X16_EXTRACT_U: u32 = 0x16;
    pub const I8X16_EXTRACT_S: u32 = 0x15;
    pub const I8X16_REPLACE: u32 = 0x17;
    pub const I16X8_EXTRACT_S: u32 = 0x18;
    pub const I16X8_EXTRACT_U: u32 = 0x19;
    pub const I16X8_REPLACE: u32 = 0x1A;
    pub const I32X4_EXTRACT: u32 = 0x1B;
    pub const I32X4_REPLACE: u32 = 0x1C;
    pub const I64X2_EXTRACT: u32 = 0x1D;
    pub const I64X2_REPLACE: u32 = 0x1E;
    pub const NOT: u32 = 0x4D;
    pub const AND: u32 = 0x4E;
    pub const ANDNOT: u32 = 0x4F;
    pub const OR: u32 = 0x50;
    pub const XOR: u32 = 0x51;
    pub const BITSELECT: u32 = 0x52;
    /// `load{8,16,32,64}_lane`, and the stores after them, are consecutive.
    pub const LOAD8_LANE: u32 = 0x54;
    pub const STORE8_LANE: u32 = 0x58;
    pub const I8X16_ABS: u32 = 0x60;
    pub const I8X16_NEG: u32 = 0x61;
    pub const I8X16_POPCNT: u32 = 0x62;
    pub const I8X16_NARROW_S: u32 = 0x65;
    pub const I8X16_NARROW_U: u32 = 0x66;
    pub const I16X8_Q15MULR_SAT_S: u32 = 0x82;
    pub const I16X8_NARROW_S: u32 = 0x85;
    pub const I16X8_NARROW_U: u32 = 0x86;
    pub const I16X8_MUL: u32 = 0x95;
    pub const I16X8_MIN_U: u32 = 0x97;
    pub const I32X4_MIN_U: u32 = 0xB7;
    pub const I32X4_MUL: u32 = 0xB5;
}

/// Lane-size-indexed opcode families: 8-, 16-, 32- and 64-bit lanes.
const ABS: [u32; 4] = [op::I8X16_ABS, 0x80, 0xA0, 0xC0];
const NEG: [u32; 4] = [op::I8X16_NEG, 0x81, 0xA1, 0xC1];
const SHL: [u32; 4] = [0x6B, 0x8B, 0xAB, 0xCB];
const SHR_S: [u32; 4] = [0x6C, 0x8C, 0xAC, 0xCC];
const SHR_U: [u32; 4] = [0x6D, 0x8D, 0xAD, 0xCD];
const ADD: [u32; 4] = [0x6E, 0x8E, 0xAE, 0xCE];
const SUB: [u32; 4] = [0x71, 0x91, 0xB1, 0xD1];
const ADD_SAT_S: [u32; 2] = [0x6F, 0x8F];
const ADD_SAT_U: [u32; 2] = [0x70, 0x90];
const SUB_SAT_S: [u32; 2] = [0x72, 0x92];
const SUB_SAT_U: [u32; 2] = [0x73, 0x93];
const SPLAT: [u32; 4] = [
    op::I8X16_SPLAT,
    op::I16X8_SPLAT,
    op::I32X4_SPLAT,
    op::I64X2_SPLAT,
];
const EXTRACT_U: [u32; 4] = [
    op::I8X16_EXTRACT_U,
    op::I16X8_EXTRACT_U,
    op::I32X4_EXTRACT,
    op::I64X2_EXTRACT,
];
const REPLACE: [u32; 4] = [
    op::I8X16_REPLACE,
    op::I16X8_REPLACE,
    op::I32X4_REPLACE,
    op::I64X2_REPLACE,
];
/// `extend_{low,high}_{s,u}` into a lane of twice the size, by source size.
const EXTEND_LOW_S: [u32; 3] = [0x87, 0xA7, 0xC7];
const EXTEND_HIGH_S: [u32; 3] = [0x88, 0xA8, 0xC8];
const EXTEND_LOW_U: [u32; 3] = [0x89, 0xA9, 0xC9];
const EXTEND_HIGH_U: [u32; 3] = [0x8A, 0xAA, 0xCA];
/// `extmul_{low,high}_{s,u}`, by source size.
const EXTMUL_LOW_S: [u32; 3] = [0x9C, 0xBC, 0xDC];
const EXTMUL_HIGH_S: [u32; 3] = [0x9D, 0xBD, 0xDD];
const EXTMUL_LOW_U: [u32; 3] = [0x9E, 0xBE, 0xDE];
const EXTMUL_HIGH_U: [u32; 3] = [0x9F, 0xBF, 0xDF];

/// Byte indices that take the low byte(s) of each lane of `2 << dest` bytes,
/// leaving the narrowed lanes in the low 64 bits.
fn truncate_lanes(dest: u32) -> [u8; 16] {
    let (src, keep) = (2u32 << dest, 1u32 << dest);
    let mut lanes = [0u8; 16];
    for (i, lane) in lanes.iter_mut().take(8).enumerate() {
        let i = i as u32;
        *lane = ((i / keep) * src + i % keep) as u8;
    }
    lanes
}

impl Emitter<'_> {
    /// Push `V<r>`.
    pub(super) fn vget(&mut self, r: u32) {
        self.f.local_get(STATE);
        self.f
            .simd_mem(op::V128_LOAD, ALIGN_8, self.layout.vregs + 16 * r);
    }

    /// Pop into `V<r>`.
    pub(super) fn vset(&mut self, r: u32) {
        self.f.local_set(L_V0);
        self.f.local_get(STATE);
        self.f.local_get(L_V0);
        self.f
            .simd_mem(op::V128_STORE, ALIGN_8, self.layout.vregs + 16 * r);
    }

    /// Clear the high 64 bits of the vector on the stack when `q` is clear.
    fn vlow(&mut self, q: bool) {
        if !q {
            self.f.i64_const(0);
            self.f.simd_lane(op::I64X2_REPLACE, 1);
        }
    }

    /// Pop a narrowed result whose lanes sit in the low 64 bits into `V<rd>`:
    /// the low half with the high cleared, or for the `2` forms the high half
    /// with the low kept.
    fn vset_narrow(&mut self, rd: u32, q: bool) {
        if q {
            self.f.local_set(L_V1);
            self.vget(rd);
            self.f.local_get(L_V1);
            self.f
                .i8x16_shuffle([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]);
        } else {
            self.vlow(false);
        }
        self.vset(rd);
    }

    /// Push `value` in every lane of `8 << size` bits.
    fn splat(&mut self, size: u32, value: i64) {
        if size == 3 {
            self.f.i64_const(value);
        } else {
            self.f.i32_const(value as i32);
        }
        self.f.simd(SPLAT[size as usize]);
    }

    /// Shift every lane of the vector on the stack right by `shift` (1 to the
    /// lane width), rounding to nearest with ties up when `round`:
    /// `(x >> s) + ((x >> (s - 1)) & 1)` never leaves the lane.
    fn shift_right(&mut self, size: u32, shift: u32, signed: bool, round: bool) {
        let width = 8 << size;
        let shr = if signed { SHR_S } else { SHR_U }[size as usize];
        if shift >= width {
            // Wasm takes the count modulo the width: a whole-width shift is a fill.
            self.f.drop_value();
            self.f.v128_const([0; 16]);
            return;
        }
        if !round {
            self.f.i32_const(shift as i32);
            self.f.simd(shr);
            return;
        }
        self.f.local_tee(L_V2);
        self.f.i32_const(shift as i32);
        self.f.simd(shr);
        self.f.local_get(L_V2);
        self.f.i32_const(shift as i32 - 1);
        self.f.simd(SHR_U[size as usize]);
        self.splat(size, 1);
        self.f.simd(op::AND);
        self.f.simd(ADD[size as usize]);
    }

    /// One Advanced SIMD data-processing instruction, or `false`.
    pub(super) fn simd(&mut self, insn: u32) -> bool {
        if !self.simd {
            return false;
        }
        let q = (insn >> 30) & 1 == 1;
        let u = (insn >> 29) & 1 == 1;
        let size = (insn >> 22) & 3;
        let rm = (insn >> 16) & 0x1F;
        let rn = (insn >> 5) & 0x1F;
        let rd = insn & 0x1F;

        // Scalar SCVTF and UCVTF: one lane, converted straight to its width.
        if insn & 0xDFBF_FC00 == 0x5E21_D800 {
            let double = size & 1 == 1;
            self.vget(rn);
            self.f.simd_lane(
                if double {
                    op::I64X2_EXTRACT
                } else {
                    op::I32X4_EXTRACT
                },
                0,
            );
            self.f.op_raw(match (double, u) {
                (false, false) => 0xB2,
                (false, true) => 0xB3,
                (true, false) => 0xB9,
                (true, true) => 0xBA,
            });
            self.fset(rd, double);
            return true;
        }
        // Three registers of the same type.
        if insn & 0x9F20_0400 == 0x0E20_0400 {
            return self.three_same(insn, q, u, size, rm, rn, rd);
        }
        // Three registers of different types: the long and wide forms.
        if insn & 0x9F20_0C00 == 0x0E20_0000 {
            return self.three_different(insn, q, u, size, rm, rn, rd);
        }
        // Modified immediate, before the shifts it shares bits with (`immh` zero).
        if insn & 0x9FF8_0400 == 0x0F00_0400 {
            return self.modified_immediate(insn, q, rd);
        }
        // Shift by immediate.
        if insn & 0x9F80_0400 == 0x0F00_0400 {
            return self.shift_immediate(insn, q, u, rn, rd);
        }
        // Two-register miscellaneous.
        if insn & 0x9F3E_0C00 == 0x0E20_0800 {
            return self.two_misc(insn, q, u, size, rn, rd);
        }
        // Permute.
        if insn & 0xBF20_8C00 == 0x0E00_0800 {
            return self.permute(insn, q, size, rm, rn, rd);
        }
        // EXT.
        if insn & 0xBFE0_8400 == 0x2E00_0000 {
            let imm4 = (insn >> 11) & 0xF;
            if !q && imm4 >= 8 {
                return false;
            }
            let len = if q { 16 } else { 8 };
            let mut lanes = [0u8; 16];
            for (i, lane) in lanes.iter_mut().enumerate().take(len) {
                let k = imm4 as usize + i;
                *lane = if k < len { k } else { 16 + k - len } as u8;
            }
            self.vget(rn);
            self.vget(rm);
            self.f.i8x16_shuffle(lanes);
            self.vlow(q);
            self.vset(rd);
            return true;
        }
        // Copy: DUP, INS, SMOV, UMOV.
        if insn & 0x9FE0_8400 == 0x0E00_0400 {
            return self.copy(insn, q, u, rn, rd);
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn three_same(
        &mut self,
        insn: u32,
        q: bool,
        u: bool,
        size: u32,
        rm: u32,
        rn: u32,
        rd: u32,
    ) -> bool {
        let opcode = (insn >> 11) & 0x1F;
        if size == 3 && !q && opcode != 0b00011 {
            return false;
        }
        let i = size as usize;
        match (opcode, u) {
            (0b00011, _) => {
                // AND, BIC, ORR, ORN; EOR, BSL, BIT, BIF.
                match (u, size) {
                    (false, 0) => self.binary(rn, rm, op::AND),
                    (false, 1) => self.binary(rn, rm, op::ANDNOT),
                    (false, 2) => self.binary(rn, rm, op::OR),
                    (false, _) => {
                        self.vget(rn);
                        self.vget(rm);
                        self.f.simd(op::NOT);
                        self.f.simd(op::OR);
                    }
                    (true, 0) => self.binary(rn, rm, op::XOR),
                    (true, sel) => {
                        let (a, b, mask) = match sel {
                            1 => (rn, rm, rd),
                            2 => (rn, rd, rm),
                            _ => (rd, rn, rm),
                        };
                        self.vget(a);
                        self.vget(b);
                        self.vget(mask);
                        self.f.simd(op::BITSELECT);
                    }
                }
            }
            (0b00001, _) | (0b00101, _) => {
                if size > 1 {
                    return false;
                }
                let family = match (opcode, u) {
                    (0b00001, false) => ADD_SAT_S,
                    (0b00001, true) => ADD_SAT_U,
                    (_, false) => SUB_SAT_S,
                    (_, true) => SUB_SAT_U,
                };
                self.binary(rn, rm, family[i]);
            }
            (0b10000, false) => self.binary(rn, rm, ADD[i]),
            (0b10000, true) => self.binary(rn, rm, SUB[i]),
            (0b10011, false) => match size {
                1 => self.binary(rn, rm, op::I16X8_MUL),
                2 => self.binary(rn, rm, op::I32X4_MUL),
                _ => return false,
            },
            // SQDMULH: the doubled product's high half, saturated only at -1 * -1.
            (0b10110, false) if size == 1 => {
                self.vget(rn);
                self.vget(rm);
                self.f.simd(EXTMUL_LOW_S[1]);
                self.f.i32_const(15);
                self.f.simd(SHR_S[2]);
                self.vget(rn);
                self.vget(rm);
                self.f.simd(EXTMUL_HIGH_S[1]);
                self.f.i32_const(15);
                self.f.simd(SHR_S[2]);
                self.f.simd(op::I16X8_NARROW_S);
            }
            // SQRDMULH is wasm's Q15 rounding multiply.
            (0b10110, true) if size == 1 => self.binary(rn, rm, op::I16X8_Q15MULR_SAT_S),
            _ => return false,
        }
        self.vlow(q);
        self.vset(rd);
        true
    }

    /// Push `op(V<a>, V<b>)`.
    fn binary(&mut self, a: u32, b: u32, opcode: u32) {
        self.vget(a);
        self.vget(b);
        self.f.simd(opcode);
    }

    #[allow(clippy::too_many_arguments)]
    fn three_different(
        &mut self,
        insn: u32,
        q: bool,
        u: bool,
        size: u32,
        rm: u32,
        rn: u32,
        rd: u32,
    ) -> bool {
        if size == 3 {
            return false;
        }
        let i = size as usize;
        let wide = i + 1;
        let extend = match (u, q) {
            (false, false) => EXTEND_LOW_S[i],
            (false, true) => EXTEND_HIGH_S[i],
            (true, false) => EXTEND_LOW_U[i],
            (true, true) => EXTEND_HIGH_U[i],
        };
        let extmul = match (u, q) {
            (false, false) => EXTMUL_LOW_S[i],
            (false, true) => EXTMUL_HIGH_S[i],
            (true, false) => EXTMUL_LOW_U[i],
            (true, true) => EXTMUL_HIGH_U[i],
        };
        match (insn >> 12) & 0xF {
            // [SU]ADDL, [SU]SUBL.
            0b0000 | 0b0010 => {
                self.vget(rn);
                self.f.simd(extend);
                self.vget(rm);
                self.f.simd(extend);
                let sub = (insn >> 12) & 0xF == 0b0010;
                self.f.simd(if sub { SUB[wide] } else { ADD[wide] });
            }
            // [SU]ADDW, [SU]SUBW.
            0b0001 | 0b0011 => {
                self.vget(rn);
                self.vget(rm);
                self.f.simd(extend);
                let sub = (insn >> 12) & 0xF == 0b0011;
                self.f.simd(if sub { SUB[wide] } else { ADD[wide] });
            }
            // [SU]MLAL, [SU]MLSL.
            0b1000 | 0b1010 => {
                self.vget(rd);
                self.vget(rn);
                self.vget(rm);
                self.f.simd(extmul);
                let sub = (insn >> 12) & 0xF == 0b1010;
                self.f.simd(if sub { SUB[wide] } else { ADD[wide] });
            }
            // [SU]MULL.
            0b1100 => {
                self.vget(rn);
                self.vget(rm);
                self.f.simd(extmul);
            }
            _ => return false,
        }
        self.vset(rd);
        true
    }

    fn shift_immediate(&mut self, insn: u32, q: bool, u: bool, rn: u32, rd: u32) -> bool {
        let immh = (insn >> 19) & 0xF;
        let field = (insn >> 16) & 0x7F;
        let size = 31 - immh.leading_zeros();
        let width = 8u32 << size;
        let opcode = (insn >> 11) & 0x1F;
        if size == 3 && !q && !matches!(opcode, 0b10000..=0b10011) {
            return false;
        }
        match (opcode, u) {
            // [SU]SHR, [SU]RSHR: the shift is 1 to the lane width.
            (0b00000, _) | (0b00100, _) => {
                let shift = 2 * width - field;
                let round = opcode == 0b00100;
                if round && shift >= width {
                    return false;
                }
                self.vget(rn);
                if !u && shift >= width {
                    // SSHR by the whole width fills with the sign.
                    self.f.i32_const(width as i32 - 1);
                    self.f.simd(SHR_S[size as usize]);
                } else {
                    self.shift_right(size, shift, !u, round);
                }
                self.vlow(q);
                self.vset(rd);
            }
            // SHL.
            (0b01010, false) => {
                self.vget(rn);
                self.f.i32_const((field - width) as i32);
                self.f.simd(SHL[size as usize]);
                self.vlow(q);
                self.vset(rd);
            }
            // [SU]SHLL, with SXTL/UXTL at a zero shift.
            (0b10100, _) => {
                if size == 3 {
                    return false;
                }
                let i = size as usize;
                let extend = match (u, q) {
                    (false, false) => EXTEND_LOW_S[i],
                    (false, true) => EXTEND_HIGH_S[i],
                    (true, false) => EXTEND_LOW_U[i],
                    (true, true) => EXTEND_HIGH_U[i],
                };
                self.vget(rn);
                self.f.simd(extend);
                let shift = field - width;
                if shift != 0 {
                    self.f.i32_const(shift as i32);
                    self.f.simd(SHL[i + 1]);
                }
                self.vset(rd);
            }
            // The narrowing shifts: `size` is the destination's, the source twice it.
            (0b10000..=0b10011, _) => {
                if size > 1 {
                    return false;
                }
                let shift = 2 * width - field;
                let round = opcode & 1 == 1;
                // SHRN/RSHRN truncate; the rest saturate, unsigned or signed.
                let (signed_src, narrow) = match (u, opcode) {
                    (false, 0b10000) | (false, 0b10001) => (false, None),
                    (true, 0b10000) | (true, 0b10001) => (true, Some(true)),
                    (false, _) => (true, Some(false)),
                    (true, _) => (false, Some(true)),
                };
                self.vget(rn);
                self.shift_right(size + 1, shift, signed_src, round);
                match narrow {
                    None => {
                        self.f.local_tee(L_V2);
                        self.f.local_get(L_V2);
                        self.f.i8x16_shuffle(truncate_lanes(size));
                    }
                    Some(to_unsigned) => {
                        self.f.local_tee(L_V2);
                        self.f.local_get(L_V2);
                        self.f.simd(match (size, to_unsigned) {
                            (0, true) => op::I8X16_NARROW_U,
                            (0, false) => op::I8X16_NARROW_S,
                            (_, true) => op::I16X8_NARROW_U,
                            (_, false) => op::I16X8_NARROW_S,
                        });
                    }
                }
                self.vset_narrow(rd, q);
            }
            _ => return false,
        }
        true
    }

    fn two_misc(&mut self, insn: u32, q: bool, u: bool, size: u32, rn: u32, rd: u32) -> bool {
        let opcode = (insn >> 12) & 0x1F;
        match (opcode, u) {
            // XTN truncates; SQXTUN, SQXTN and UQXTN saturate.
            (0b10010, _) | (0b10100, _) => {
                if size > 1 {
                    return false;
                }
                self.vget(rn);
                // UQXTN has an unsigned source: clamp it, then truncate.
                if (opcode, u) == (0b10100, true) {
                    self.splat(size + 1, (1 << (8 << size)) - 1);
                    self.f.simd(if size == 0 {
                        op::I16X8_MIN_U
                    } else {
                        op::I32X4_MIN_U
                    });
                }
                self.f.local_tee(L_V2);
                self.f.local_get(L_V2);
                // XTN and the clamped UQXTN truncate; SQXTUN and SQXTN saturate.
                if (opcode == 0b10010) != u {
                    self.f.i8x16_shuffle(truncate_lanes(size));
                } else {
                    self.f.simd(match (size, opcode == 0b10010) {
                        (0, true) => op::I8X16_NARROW_U,
                        (_, true) => op::I16X8_NARROW_U,
                        (0, false) => op::I8X16_NARROW_S,
                        (_, false) => op::I16X8_NARROW_S,
                    });
                }
                self.vset_narrow(rd, q);
                true
            }
            // ABS, NEG.
            (0b01011, _) => {
                if size == 3 && !q {
                    return false;
                }
                self.vget(rn);
                self.f.simd(if u { NEG } else { ABS }[size as usize]);
                self.vlow(q);
                self.vset(rd);
                true
            }
            // CNT, NOT.
            (0b00101, _) if size == 0 => {
                self.vget(rn);
                self.f.simd(if u { op::NOT } else { op::I8X16_POPCNT });
                self.vlow(q);
                self.vset(rd);
                true
            }
            _ => false,
        }
    }

    fn permute(&mut self, insn: u32, q: bool, size: u32, rm: u32, rn: u32, rd: u32) -> bool {
        if size == 3 && !q {
            return false;
        }
        let opcode = (insn >> 12) & 7;
        let bytes = 1usize << size;
        let elements = if q { 16 } else { 8 } / bytes;
        // Element `e` of Vn is at byte `e * bytes`, of Vm at `16 + e * bytes`.
        let source = |from_m: bool, e: usize| if from_m { 16 } else { 0 } + e * bytes;
        let mut picks: Vec<usize> = Vec::with_capacity(elements);
        for i in 0..elements {
            let (pair, odd) = (i / 2, i % 2 == 1);
            let at = match opcode {
                // UZP1/UZP2: even/odd elements of Vn then Vm.
                0b001 | 0b101 => {
                    let e = 2 * i + usize::from(opcode == 0b101);
                    if e < elements {
                        source(false, e)
                    } else {
                        source(true, e - elements)
                    }
                }
                // TRN1/TRN2.
                0b010 | 0b110 => source(odd, 2 * pair + usize::from(opcode == 0b110)),
                // ZIP1/ZIP2.
                0b011 | 0b111 => {
                    let base = if opcode == 0b111 { elements / 2 } else { 0 };
                    source(odd, base + pair)
                }
                _ => return false,
            };
            picks.push(at);
        }
        let mut lanes = [0u8; 16];
        for (i, &at) in picks.iter().enumerate() {
            for b in 0..bytes {
                lanes[i * bytes + b] = (at + b) as u8;
            }
        }
        self.vget(rn);
        self.vget(rm);
        self.f.i8x16_shuffle(lanes);
        self.vlow(q);
        self.vset(rd);
        true
    }

    fn copy(&mut self, insn: u32, q: bool, op1: bool, rn: u32, rd: u32) -> bool {
        let imm5 = (insn >> 16) & 0x1F;
        let imm4 = (insn >> 11) & 0xF;
        if imm5 & 0xF == 0 {
            return false;
        }
        let size = imm5.trailing_zeros();
        let index = (imm5 >> (size + 1)) as u8;
        let s = size as usize;
        match (op1, imm4) {
            // DUP (element).
            (false, 0b0000) => {
                if size == 3 && !q {
                    return false;
                }
                self.vget(rn);
                self.f.simd_lane(EXTRACT_U[s], index);
                self.f.simd(SPLAT[s]);
                self.vlow(q);
                self.vset(rd);
            }
            // DUP (general).
            (false, 0b0001) => {
                if size == 3 && !q {
                    return false;
                }
                self.read_reg_raw(rn as u8);
                if size < 3 {
                    self.f.i32_wrap_i64();
                }
                self.f.simd(SPLAT[s]);
                self.vlow(q);
                self.vset(rd);
            }
            // INS (general).
            (false, 0b0011) if q => {
                self.vget(rd);
                self.read_reg_raw(rn as u8);
                if size < 3 {
                    self.f.i32_wrap_i64();
                }
                self.f.simd_lane(REPLACE[s], index);
                self.vset(rd);
            }
            // SMOV, UMOV.
            (false, 0b0101) | (false, 0b0111) => {
                let signed = imm4 == 0b0101;
                // SMOV to X needs Q; UMOV moves a D lane only to X (Q) and others only to W.
                if (signed && (size >= 2 + u32::from(q) || size == 3))
                    || (!signed && (size == 3) != q)
                {
                    return false;
                }
                self.vget(rn);
                match (size, signed) {
                    (3, _) => self.f.simd_lane(op::I64X2_EXTRACT, index),
                    (2, false) => {
                        self.f.simd_lane(op::I32X4_EXTRACT, index);
                        self.f.i64_extend_i32_u();
                    }
                    (2, true) => {
                        self.f.simd_lane(op::I32X4_EXTRACT, index);
                        self.f.i64_extend_i32_s();
                    }
                    (_, false) => {
                        self.f.simd_lane(EXTRACT_U[s], index);
                        self.f.i64_extend_i32_u();
                    }
                    (_, true) => {
                        let extract = if size == 0 {
                            op::I8X16_EXTRACT_S
                        } else {
                            op::I16X8_EXTRACT_S
                        };
                        self.f.simd_lane(extract, index);
                        self.f.i64_extend_i32_s();
                        if !q {
                            self.mask_to(false);
                        }
                    }
                }
                self.store_reg(gp_dst(rd));
            }
            // INS (element).
            (true, _) if q => {
                let from = (imm4 >> size) as u8;
                self.vget(rd);
                self.vget(rn);
                self.f.simd_lane(EXTRACT_U[s], from);
                self.f.simd_lane(REPLACE[s], index);
                self.vset(rd);
            }
            _ => return false,
        }
        true
    }

    /// MOVI, MVNI, and ORR/BIC (vector, immediate); not FMOV.
    fn modified_immediate(&mut self, insn: u32, q: bool, rd: u32) -> bool {
        let op1 = (insn >> 29) & 1 == 1;
        let cmode = (insn >> 12) & 0xF;
        if (insn >> 11) & 1 == 1 || cmode == 0b1111 {
            return false;
        }
        let imm8 = u64::from(((insn >> 11) & 0xE0) | ((insn >> 5) & 0x1F));
        let replicate = |unit: u64, bits: u32| {
            let mut out = 0u64;
            for k in 0..64 / bits {
                out |= unit << (k * bits);
            }
            out
        };
        // The 64-bit pattern, and whether it is OR'd or AND-NOT'd into Vd instead of written.
        let (pattern, modify) = match cmode {
            0b0000..=0b0111 => {
                let shifted = imm8 << (8 * ((cmode >> 1) & 3));
                (replicate(shifted, 32), cmode & 1 == 1)
            }
            0b1000..=0b1011 => {
                let shifted = imm8 << (8 * ((cmode >> 1) & 1));
                (replicate(shifted, 16), cmode & 1 == 1)
            }
            // Shifting ones in (MSL).
            0b1100 | 0b1101 => {
                let ones = if cmode & 1 == 1 { 0xFFFF } else { 0xFF };
                let unit = (imm8 << (8 * (1 + (cmode & 1)))) | ones;
                (replicate(unit, 32), false)
            }
            0b1110 if !op1 => (replicate(imm8, 8), false),
            // Each bit of imm8 a whole byte.
            _ => {
                let mut bytes = 0u64;
                for bit in 0..8 {
                    if imm8 >> bit & 1 == 1 {
                        bytes |= 0xFF << (8 * bit);
                    }
                }
                (bytes, false)
            }
        };
        // MVNI and BIC invert; the byte-mask form uses op for its own meaning.
        let invert = op1 && cmode != 0b1110;
        let value = if invert && !modify { !pattern } else { pattern };
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&value.to_le_bytes());
        bytes[8..].copy_from_slice(&value.to_le_bytes());
        if modify {
            self.vget(rd);
            self.f.v128_const(bytes);
            self.f.simd(if op1 { op::ANDNOT } else { op::OR });
        } else {
            self.f.v128_const(bytes);
        }
        self.vlow(q);
        self.vset(rd);
        true
    }

    /// A SIMD&FP load or store, or `false`.
    pub(super) fn simd_load_store(&mut self, insn: u32, retired: usize) -> bool {
        if !self.simd {
            return false;
        }
        if insn & 0xBF20_0000 == 0x0C00_0000 {
            return self.multiple_structures(insn, retired);
        }
        // LD1/ST1 (single structure) and LD1R, one register.
        if insn & 0xBF00_0000 == 0x0D00_0000 && (insn >> 21) & 1 == 0 {
            return self.single_structure(insn, retired);
        }
        // LDP/STP of S and D registers; Q pairs are `Op::Pair`.
        if insn & 0x3E00_0000 == 0x2C00_0000 {
            return self.simd_pair(insn, retired);
        }
        if insn & 0x3F00_0000 == 0x3C00_0000 || insn & 0x3F00_0000 == 0x3D00_0000 {
            return self.simd_register(insn, retired);
        }
        false
    }

    /// LDR/STR (and LDUR/STUR) of a B, H, S, D or Q register.
    fn simd_register(&mut self, insn: u32, retired: usize) -> bool {
        let size = insn >> 30;
        let opc = (insn >> 22) & 3;
        let quad = opc & 2 != 0;
        if quad && size != 0 {
            return false;
        }
        let load = opc & 1 == 1;
        let log2 = if quad { 4 } else { size };
        let bytes = 1u32 << log2;
        let rn = base_slot((insn >> 5) & 0x1F);
        let rt = insn & 0x1F;
        let wb = if insn & 0x3F00_0000 == 0x3D00_0000 {
            let offset = i64::from((insn >> 10) & 0xFFF) << log2;
            self.address(rn, offset, Wb::None);
            Wb::None
        } else {
            let imm9 = i64::from((((insn >> 12) & 0x1FF) as i32) << 23 >> 23);
            let wide = (insn >> 21) & 1 == 1;
            let wb = match ((insn >> 10) & 3, wide) {
                (0b00, false) => Wb::None,
                (0b01, false) => Wb::Post,
                (0b11, false) => Wb::Pre,
                (0b10, true) => {
                    let ext = match (insn >> 13) & 7 {
                        0b010 => Ext::Uxtw,
                        0b110 => Ext::Sxtw,
                        0b011 | 0b111 => Ext::None,
                        _ => return false,
                    };
                    let shift = if (insn >> 12) & 1 == 1 { log2 as u8 } else { 0 };
                    self.address_reg(rn, ((insn >> 16) & 0x1F) as u8, ext, shift);
                    Wb::None
                }
                _ => return false,
            };
            if !wide {
                self.address(rn, imm9, wb);
            }
            wb
        };
        if load {
            self.page_for_read(bytes, retired);
            if quad {
                self.in_page();
                self.f.simd_mem(op::V128_LOAD, UNALIGNED, 0);
            } else {
                self.f.v128_const([0; 16]);
                self.in_page();
                match bytes {
                    1 => self.f.i64_load8_u(0),
                    2 => self.f.i64_load16_u(UNALIGNED, 0),
                    4 => self.f.i64_load32_u(UNALIGNED, 0),
                    _ => self.f.i64_load(UNALIGNED, 0),
                }
                self.f.simd_lane(op::I64X2_REPLACE, 0);
            }
            self.vset(rt);
        } else {
            self.page_for_write(bytes, retired);
            self.in_page();
            self.vget(rt);
            if quad {
                self.f.simd_mem(op::V128_STORE, UNALIGNED, 0);
            } else {
                self.f.simd_lane(op::I64X2_EXTRACT, 0);
                match bytes {
                    1 => self.f.i64_store8(0),
                    2 => self.f.i64_store16(UNALIGNED, 0),
                    4 => self.f.i64_store32(UNALIGNED, 0),
                    _ => self.f.i64_store(UNALIGNED, 0),
                }
            }
        }
        self.write_back(rn, wb);
        true
    }

    /// LDP/STP of two S or D registers, checked as one access.
    fn simd_pair(&mut self, insn: u32, retired: usize) -> bool {
        let opc = insn >> 30;
        if opc > 1 {
            return false;
        }
        let bytes = 4u32 << opc;
        let wb = match (insn >> 23) & 3 {
            0b01 => Wb::Post,
            0b11 => Wb::Pre,
            _ => Wb::None,
        };
        let load = (insn >> 22) & 1 == 1;
        let imm7 = (((insn >> 15) & 0x7F) as i32) << 25 >> 25;
        let offset = i64::from(imm7) * i64::from(bytes);
        let rn = base_slot((insn >> 5) & 0x1F);
        let (rt, rt2) = (insn & 0x1F, (insn >> 10) & 0x1F);
        self.address(rn, offset, wb);
        if load {
            self.page_for_read(2 * bytes, retired);
            for (reg, at) in [(rt, 0), (rt2, bytes)] {
                self.f.v128_const([0; 16]);
                self.in_page();
                if bytes == 4 {
                    self.f.i64_load32_u(UNALIGNED, at);
                } else {
                    self.f.i64_load(UNALIGNED, at);
                }
                self.f.simd_lane(op::I64X2_REPLACE, 0);
                self.vset(reg);
            }
        } else {
            self.page_for_write(2 * bytes, retired);
            for (reg, at) in [(rt, 0), (rt2, bytes)] {
                self.in_page();
                self.vget(reg);
                self.f.simd_lane(op::I64X2_EXTRACT, 0);
                if bytes == 4 {
                    self.f.i64_store32(UNALIGNED, at);
                } else {
                    self.f.i64_store(UNALIGNED, at);
                }
            }
        }
        self.write_back(rn, wb);
        true
    }

    /// LD1 to LD4 and ST1 to ST4 of whole registers, checked as one access.
    fn multiple_structures(&mut self, insn: u32, retired: usize) -> bool {
        let q = (insn >> 30) & 1;
        let post = (insn >> 23) & 1 == 1;
        let load = (insn >> 22) & 1 == 1;
        let rm = (insn >> 16) & 0x1F;
        let size = (insn >> 10) & 3;
        let rn = base_slot((insn >> 5) & 0x1F);
        let rt = insn & 0x1F;
        if !post && rm != 0 {
            return false;
        }
        let (regs, members) = match (insn >> 12) & 0xF {
            0b0000 => (1, 4),
            0b0010 => (4, 1),
            0b0100 => (1, 3),
            0b0110 => (3, 1),
            0b0111 => (1, 1),
            0b1000 => (1, 2),
            0b1010 => (2, 1),
            _ => return false,
        };
        if size == 3 && q == 0 && members != 1 {
            return false;
        }
        let vector = 8u32 << q;
        let total = regs * members * vector;
        self.read_reg_raw(rn);
        self.f.local_tee(L_A);
        self.f.i32_wrap_i64();
        self.f.local_set(L_ADDR);
        if post {
            self.f.local_get(L_A);
            if rm == 31 {
                self.f.i64_const(i64::from(total));
            } else {
                self.read_reg_raw(rm as u8);
            }
            self.f.i64_add();
            self.f.local_set(L_W);
        }
        if load {
            self.page_for_read(total, retired);
        } else {
            self.page_for_write(total, retired);
        }
        if members == 1 {
            for i in 0..regs {
                let reg = (rt + i) % 32;
                let at = i * vector;
                if load {
                    if q == 1 {
                        self.in_page();
                        self.f.simd_mem(op::V128_LOAD, UNALIGNED, at);
                    } else {
                        self.f.v128_const([0; 16]);
                        self.in_page();
                        self.f.i64_load(UNALIGNED, at);
                        self.f.simd_lane(op::I64X2_REPLACE, 0);
                    }
                    self.vset(reg);
                } else {
                    self.in_page();
                    self.vget(reg);
                    if q == 1 {
                        self.f.simd_mem(op::V128_STORE, UNALIGNED, at);
                    } else {
                        self.f.simd_lane(op::I64X2_EXTRACT, 0);
                        self.f.i64_store(UNALIGNED, at);
                    }
                }
            }
        } else {
            let bytes = 1u32 << size;
            for member in 0..members {
                let reg = (rt + member) % 32;
                for lane in 0..vector / bytes {
                    let at = (lane * members + member) * bytes;
                    self.in_page();
                    if load {
                        if lane == 0 {
                            self.f.v128_const([0; 16]);
                        } else {
                            self.vget(reg);
                        }
                        self.f
                            .simd_mem_lane(op::LOAD8_LANE + size, UNALIGNED, at, lane as u8);
                        self.vset(reg);
                    } else {
                        self.vget(reg);
                        self.f
                            .simd_mem_lane(op::STORE8_LANE + size, UNALIGNED, at, lane as u8);
                    }
                }
            }
        }
        if post {
            self.write_back(rn, Wb::Post);
        }
        true
    }

    /// LD1/ST1 of one lane, and LD1R.
    fn single_structure(&mut self, insn: u32, retired: usize) -> bool {
        let q = (insn >> 30) & 1;
        let post = (insn >> 23) & 1 == 1;
        let load = (insn >> 22) & 1 == 1;
        let opcode = (insn >> 13) & 7;
        let s = (insn >> 12) & 1;
        let size = (insn >> 10) & 3;
        let rm = (insn >> 16) & 0x1F;
        let rn = base_slot((insn >> 5) & 0x1F);
        let rt = insn & 0x1F;
        if !post && rm != 0 {
            return false;
        }
        // Element size (log2 bytes) and lane, or `None` for LD1R.
        let (log2, lane) = match opcode {
            0b000 => (0, Some((q << 3) | (s << 2) | size)),
            0b010 if size & 1 == 0 => (1, Some((q << 2) | (s << 1) | (size >> 1))),
            0b100 if size == 0 => (2, Some((q << 1) | s)),
            0b100 if size == 1 && s == 0 => (3, Some(q)),
            0b110 if load && s == 0 => (size, None),
            _ => return false,
        };
        let bytes = 1u32 << log2;
        // The address, and the base's writeback by the transfer size or Xm.
        self.read_reg_raw(rn);
        self.f.local_tee(L_A);
        self.f.i32_wrap_i64();
        self.f.local_set(L_ADDR);
        if post {
            self.f.local_get(L_A);
            if rm == 31 {
                self.f.i64_const(i64::from(bytes));
            } else {
                self.read_reg_raw(rm as u8);
            }
            self.f.i64_add();
            self.f.local_set(L_W);
        }
        if load {
            self.page_for_read(bytes, retired);
        } else {
            self.page_for_write(bytes, retired);
        }
        match lane {
            Some(lane) => {
                self.in_page();
                if load {
                    self.vget(rt);
                    self.f
                        .simd_mem_lane(op::LOAD8_LANE + log2, UNALIGNED, 0, lane as u8);
                    self.vset(rt);
                } else {
                    self.vget(rt);
                    self.f
                        .simd_mem_lane(op::STORE8_LANE + log2, UNALIGNED, 0, lane as u8);
                }
            }
            None => {
                self.in_page();
                self.f.simd_mem(op::LOAD8_SPLAT + log2, UNALIGNED, 0);
                self.vlow(q == 1);
                self.vset(rt);
            }
        }
        if post {
            self.write_back(rn, Wb::Post);
        }
        true
    }
}
