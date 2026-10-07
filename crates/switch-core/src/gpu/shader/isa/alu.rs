//! The 0xfff8 ALU opcode group.

use super::*;

pub(super) fn decode_alu(insn: u64) -> Op {
    let un = Op::Unimplemented { raw: insn };

    // The three operand forms of a "normal" ALU op share a sub-opcode.
    let form = insn >> 48;
    let (rhs_int, rhs_float) = match form >> 8 {
        0x5c => (
            Operand::Reg(reg(insn, 20, 8)),
            Operand::Reg(reg(insn, 20, 8)),
        ),
        0x4c => (const_operand(insn), const_operand(insn)),
        0x38 | 0x39 => (Operand::Imm(imm20(insn)), Operand::Imm(imm20f(insn))),
        _ => return decode_alu_wide(insn),
    };
    let rhs_int = Some(rhs_int);
    let rhs_float = Some(rhs_float);
    let sub = form & 0x00f8;

    match sub {
        // ---- float ----
        // fadd: ftz 44, sat 50, a: neg 48/abs 46, b: neg 45/abs 49.
        0x58 => {
            let Some(b) = rhs_float else { return un };
            Op::Fadd {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                am: FMod {
                    neg: field(insn, 48, 1) != 0,
                    abs: field(insn, 46, 1) != 0,
                },
                b,
                bm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                ftz: field(insn, 44, 1) != 0,
                sat: field(insn, 50, 1) != 0,
            }
        }
        // fmul: ftz/fmz at 44..46, scale at 41..44, sat 50, b: neg 48.
        0x68 => {
            let Some(b) = rhs_float else { return un };
            let Some(scale) = FmulScale::decode(field(insn, 41, 3)) else {
                return un;
            };
            Op::Fmul {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                bm: FMod {
                    neg: field(insn, 48, 1) != 0,
                    abs: false,
                },
                ftz: field(insn, 44, 2) == 1,
                sat: field(insn, 50, 1) != 0,
                scale,
            }
        }
        // fmnmx: ftz 44, a: neg 48/abs 46, b: neg 45/abs 49, pred at 39.
        0x60 => {
            let Some(b) = rhs_float else { return un };
            Op::Fmnmx {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                am: FMod {
                    neg: field(insn, 48, 1) != 0,
                    abs: field(insn, 46, 1) != 0,
                },
                b,
                bm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                pred: src_pred(insn, 39, 42),
                ftz: field(insn, 44, 1) != 0,
            }
        }
        // r2p.
        0xf0 => {
            let Some(mask) = rhs_int else { return un };
            if field(insn, 40, 1) != 0 {
                return un; // the CC form
            }
            Op::R2p {
                src: reg(insn, 8, 8),
                mask,
                byte: field(insn, 41, 2) as u8,
            }
        }
        // ---- integer ----
        // iadd: sat 50, x 43, a: neg 49, b: neg 48.
        0x10 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 50, 1) != 0 {
                return un; // saturating add
            }
            Op::Iadd {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                aneg: field(insn, 49, 1) != 0,
                b,
                bneg: field(insn, 48, 1) != 0,
                cin: field(insn, 43, 1) != 0,
                cout: field(insn, 47, 1) != 0,
            }
        }
        // iscadd, shift at 39..44.
        0x18 => {
            let Some(b) = rhs_int else { return un };
            Op::Iscadd {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                aneg: field(insn, 49, 1) != 0,
                b,
                bneg: field(insn, 48, 1) != 0,
                shift: field(insn, 39, 5) as u8,
            }
        }
        // imnmx, signed 48, pred at 39.
        0x20 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 43, 2) != 0 {
                return un; // the xlo/xmed/xhi extended forms
            }
            Op::Imnmx {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                pred: src_pred(insn, 39, 42),
                signed: field(insn, 48, 1) != 0,
            }
        }
        // shr, signed 48, wrap 39, brev 40, x 44.
        0x28 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 40, 1) != 0 || field(insn, 44, 1) != 0 {
                return un; // bit-reverse / extended
            }
            Op::Shr {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                signed: field(insn, 48, 1) != 0,
                wrap: field(insn, 39, 1) != 0,
            }
        }
        // flo, signed 48, shift 41, inv 40.
        0x30 => {
            let Some(b) = rhs_int else { return un };
            Op::Flo {
                dst: reg(insn, 0, 8),
                b,
                signed: field(insn, 48, 1) != 0,
                shift: field(insn, 41, 1) != 0,
                inv: field(insn, 40, 1) != 0,
            }
        }
        // imul, hi 39, signedness at 41 (a) and 40 (b) in tab5c38_0/1.
        0x38 => {
            let Some(b) = rhs_int else { return un };
            Op::Imul {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                signed: field(insn, 41, 1) != 0,
                hi: field(insn, 39, 1) != 0,
            }
        }
        // lop, op at 41..43, inv 39 (a) / 40 (b), x 43.
        0x40 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 43, 1) != 0 {
                return un; // extended-carry form
            }
            let op = match field(insn, 41, 2) {
                0 => LogicOp::And,
                1 => LogicOp::Or,
                2 => LogicOp::Xor,
                _ => LogicOp::PassB,
            };
            let pred = match field(insn, 44, 2) {
                0 => None,
                1 => Some((reg(insn, 48, 3), LopTest::True)),
                2 => Some((reg(insn, 48, 3), LopTest::Zero)),
                _ => Some((reg(insn, 48, 3), LopTest::NonZero)),
            };
            Op::Lop {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                ainv: field(insn, 39, 1) != 0,
                b,
                binv: field(insn, 40, 1) != 0,
                op,
                pred,
            }
        }
        // shl, wrap 39, x 43.
        0x48 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 43, 1) != 0 {
                return un;
            }
            Op::Shl {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                wrap: field(insn, 39, 1) != 0,
            }
        }
        // bfe, signed 48, brev 40.
        0x00 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 40, 1) != 0 {
                return un;
            }
            Op::Bfe {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                signed: field(insn, 48, 1) != 0,
            }
        }
        // popc, inv 40.
        0x08 => {
            let Some(b) = rhs_int else { return un };
            Op::Popc {
                dst: reg(insn, 0, 8),
                b,
                inv: field(insn, 40, 1) != 0,
            }
        }
        // ---- moves and selects ----
        // mov: the 4-bit byte-enable mask at 39..43 must be "all".
        0x98 => {
            let Some(src) = rhs_int else { return un };
            if field(insn, 39, 4) != 0xf {
                return un;
            }
            Op::Mov {
                dst: reg(insn, 0, 8),
                src,
            }
        }
        // rro, the range-reduction operator that precedes `mufu`.
        0x90 => {
            let Some(src) = rhs_float else { return un };
            if field(insn, 50, 1) != 0 {
                return un;
            }
            Op::Rro {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
            }
        }
        // sel, pred at 39.
        0xa0 => {
            let Some(b) = rhs_int else { return un };
            Op::Sel {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                pred: src_pred(insn, 39, 42),
            }
        }
        // ---- conversions ----
        // i2f.
        0xb8 => {
            let Some(src) = rhs_int else { return un };
            if field(insn, 8, 2) != 2 {
                return un; // only f32 destinations
            }
            let bits = field(insn, 10, 2) | (field(insn, 13, 1) << 2);
            let Some((src_bytes, src_signed)) = int_type(bits) else {
                return un;
            };
            Op::I2f {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                src_bytes,
                src_signed,
                sel: field(insn, 41, 2) as u8,
            }
        }
        // f2i.
        0xb0 => {
            let Some(src) = rhs_float else { return un };
            if field(insn, 10, 2) != 2 {
                return un; // only f32 sources
            }
            let bits = field(insn, 8, 2) | (field(insn, 12, 1) << 2);
            let Some((dst_bytes, dst_signed)) = int_type(bits) else {
                return un;
            };
            Op::F2i {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                dst_bytes,
                dst_signed,
                round: fround(field(insn, 39, 2)),
                ftz: field(insn, 44, 1) != 0,
            }
        }
        // f2f, f16 and f32 in either direction, and the rounding a same-width conversion names.
        0xa8 => {
            let (Some(dst_bits), Some(src_bits)) = (
                float_width(field(insn, 8, 2)),
                float_width(field(insn, 10, 2)),
            ) else {
                return un; // f64, which nothing here models
            };
            let hi = field(insn, 41, 1) != 0;
            let src = if src_bits == 16 {
                match rhs_int {
                    Some(Operand::Imm(v)) => Operand::Imm((v & 0xffff) | (v << 16)),
                    Some(other) => other,
                    None => return un,
                }
            } else {
                let Some(src) = rhs_float else { return un };
                src
            };
            let round = if src_bits == dst_bits {
                // `RoundingOp` is bits 39, 40 and 42.
                match field(insn, 39, 2) | (field(insn, 42, 1) << 3) {
                    0 | 3 => None,
                    8..=11 => Some(fround(field(insn, 39, 2))),
                    _ => return un,
                }
            } else {
                if field(insn, 39, 2) != 0 {
                    return un;
                }
                None
            };
            Op::F2f {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                round,
                sat: field(insn, 50, 1) != 0,
                ftz: field(insn, 44, 1) != 0,
                src_bits,
                dst_bits,
                hi,
            }
        }
        // i2i, src type in tab5ce0_1, dst type in tab5ce0_0.
        0xe0 => {
            let Some(src) = rhs_int else { return un };
            let sbits = field(insn, 10, 2) | (field(insn, 13, 1) << 2);
            let dbits = field(insn, 8, 2) | (field(insn, 12, 1) << 2);
            let (Some((src_bytes, src_signed)), Some((_, dst_signed))) =
                (int_type(sbits), int_type(dbits))
            else {
                return un;
            };
            Op::I2i {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                src_bytes,
                src_signed,
                dst_signed,
                sat: field(insn, 50, 1) != 0,
                sel: field(insn, 41, 2) as u8,
                cc: field(insn, 47, 1) != 0,
            }
        }
        _ => decode_alu_wide(insn),
    }
}
