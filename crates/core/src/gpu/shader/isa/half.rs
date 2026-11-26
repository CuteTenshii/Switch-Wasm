//! The half-precision group.

use super::*;

/// The half pair an immediate form of a half-precision op carries.
pub(super) fn half_imm(insn: u64) -> u32 {
    let low = (field(insn, 20, 9) << 6) | (field(insn, 29, 1) << 15);
    let high = (field(insn, 30, 9) << 22) | (field(insn, 56, 1) << 31);
    (low | high) as u32
}

/// The second operand of a half op's constant-or-immediate pair, and where its two lanes come from.
fn half_cbuf_or_imm(insn: u64, cbuf: bool) -> (Operand, HSwizzle) {
    if cbuf {
        (const_operand(insn), HSwizzle::F32)
    } else {
        (Operand::Imm(half_imm(insn)), HSwizzle::H1H0)
    }
}

/// The half-precision group.
pub(super) fn decode_half(insn: u64) -> Option<Op> {
    let top = insn >> 48;
    let un = || Some(Op::Unimplemented { raw: insn });
    let dst = reg(insn, 0, 8);
    let a = reg(insn, 8, 8);
    let merge = HMerge::decode(field(insn, 49, 2));
    let asw = HSwizzle::decode(field(insn, 47, 2));
    let reg20 = Operand::Reg(reg(insn, 20, 8));
    let reg39 = Operand::Reg(reg(insn, 39, 8));
    let imm = Operand::Imm(half_imm(insn));
    let imm32 = Operand::Imm(field(insn, 20, 32) as u32);
    let bsw_reg = HSwizzle::decode(field(insn, 28, 2));
    let no_mod = FMod::NONE;

    // ---- hadd2 ----
    if top & 0xfff8 == 0x5d10 {
        return Some(Op::Hadd2 {
            dst,
            a,
            am: FMod {
                neg: field(insn, 43, 1) != 0,
                abs: field(insn, 44, 1) != 0,
            },
            asw,
            b: reg20,
            bm: FMod {
                neg: field(insn, 31, 1) != 0,
                abs: field(insn, 30, 1) != 0,
            },
            bsw: bsw_reg,
            merge,
            ftz: field(insn, 39, 1) != 0,
            sat: field(insn, 32, 1) != 0,
        });
    }
    if top & 0xfe80 == 0x7a80 || top & 0xfe80 == 0x7a00 {
        let cbuf = top & 0x0080 != 0;
        let (b, bsw) = half_cbuf_or_imm(insn, cbuf);
        return Some(Op::Hadd2 {
            dst,
            a,
            am: FMod {
                neg: field(insn, 43, 1) != 0,
                abs: field(insn, 44, 1) != 0,
            },
            asw,
            b,
            // An immediate form spends the bits a modifier would need on the pair's own two signs.
            bm: if cbuf {
                FMod {
                    neg: field(insn, 56, 1) != 0,
                    abs: field(insn, 54, 1) != 0,
                }
            } else {
                no_mod
            },
            bsw,
            merge,
            ftz: field(insn, 39, 1) != 0,
            sat: field(insn, 52, 1) != 0,
        });
    }
    // hadd2_32i: its own field positions, and the merge is fixed.
    if top & 0xfe00 == 0x2c00 {
        return Some(Op::Hadd2 {
            dst,
            a,
            am: FMod {
                neg: field(insn, 56, 1) != 0,
                abs: false,
            },
            asw: HSwizzle::decode(field(insn, 53, 2)),
            b: imm32,
            bm: no_mod,
            bsw: HSwizzle::H1H0,
            merge: HMerge::H1H0,
            ftz: field(insn, 55, 1) != 0,
            sat: field(insn, 52, 1) != 0,
        });
    }

    // ---- hmul2 ----
    if top & 0xfff8 == 0x5d08 {
        return Some(Op::Hmul2 {
            dst,
            a,
            am: FMod {
                neg: false,
                abs: field(insn, 44, 1) != 0,
            },
            asw,
            b: reg20,
            bm: FMod {
                neg: field(insn, 31, 1) != 0,
                abs: field(insn, 30, 1) != 0,
            },
            bsw: bsw_reg,
            merge,
            prec: HPrecision::decode(field(insn, 39, 2)),
            sat: field(insn, 32, 1) != 0,
        });
    }
    if top & 0xfe80 == 0x7880 || top & 0xfe80 == 0x7800 {
        let cbuf = top & 0x0080 != 0;
        let (b, bsw) = half_cbuf_or_imm(insn, cbuf);
        return Some(Op::Hmul2 {
            dst,
            a,
            am: FMod {
                neg: field(insn, 43, 1) != 0,
                abs: field(insn, 44, 1) != 0,
            },
            asw,
            b,
            bm: if cbuf {
                FMod {
                    neg: false,
                    abs: field(insn, 54, 1) != 0,
                }
            } else {
                no_mod
            },
            bsw,
            merge,
            prec: HPrecision::decode(field(insn, 39, 2)),
            sat: field(insn, 52, 1) != 0,
        });
    }
    if top & 0xfe00 == 0x2a00 {
        return Some(Op::Hmul2 {
            dst,
            a,
            am: no_mod,
            asw: HSwizzle::decode(field(insn, 53, 2)),
            b: imm32,
            bm: no_mod,
            bsw: HSwizzle::H1H0,
            merge: HMerge::H1H0,
            prec: HPrecision::decode(field(insn, 55, 2)),
            sat: field(insn, 52, 1) != 0,
        });
    }

    // ---- hfma2 ----
    if top & 0xfff8 == 0x5d00 {
        return Some(Op::Hfma2 {
            dst,
            a,
            asw,
            b: reg20,
            bneg: field(insn, 31, 1) != 0,
            bsw: bsw_reg,
            c: reg39,
            cneg: field(insn, 30, 1) != 0,
            csw: HSwizzle::decode(field(insn, 35, 2)),
            merge,
            prec: HPrecision::decode(field(insn, 37, 2)),
            sat: field(insn, 32, 1) != 0,
        });
    }
    if top & 0xf880 == 0x6080 || top & 0xf880 == 0x7080 || top & 0xf880 == 0x7000 {
        let (b, bsw, c, csw) = if top & 0xf880 == 0x6080 {
            (
                reg39,
                HSwizzle::decode(field(insn, 53, 2)),
                const_operand(insn),
                HSwizzle::F32,
            )
        } else if top & 0x0080 != 0 {
            (
                const_operand(insn),
                HSwizzle::F32,
                reg39,
                HSwizzle::decode(field(insn, 53, 2)),
            )
        } else {
            (
                imm,
                HSwizzle::H1H0,
                reg39,
                HSwizzle::decode(field(insn, 53, 2)),
            )
        };
        return Some(Op::Hfma2 {
            dst,
            a,
            asw,
            b,
            bneg: top & 0xf880 != 0x7000 && field(insn, 56, 1) != 0,
            bsw,
            c,
            cneg: field(insn, 51, 1) != 0,
            csw,
            merge,
            prec: HPrecision::decode(field(insn, 57, 2)),
            sat: field(insn, 52, 1) != 0,
        });
    }
    // hfma2_32i.
    if top & 0xfe00 == 0x2800 {
        return Some(Op::Hfma2 {
            dst,
            a,
            asw: HSwizzle::decode(field(insn, 53, 2)),
            b: imm32,
            bneg: false,
            bsw: HSwizzle::H1H0,
            c: Operand::Reg(dst),
            cneg: field(insn, 52, 1) != 0,
            csw: HSwizzle::H1H0,
            merge: HMerge::H1H0,
            prec: HPrecision::decode(field(insn, 55, 2)),
            sat: false,
        });
    }

    // ---- hset2 / hsetp2 ----
    let set_am = FMod {
        neg: field(insn, 43, 1) != 0,
        abs: field(insn, 44, 1) != 0,
    };
    let src = src_pred(insn, 39, 42);
    let is_set2 = top & 0xfff8 == 0x5d18 || top & 0xfe00 == 0x7c00;
    let is_setp2 = top & 0xfff8 == 0x5d20 || top & 0xfe00 == 0x7e00;
    if is_set2 || is_setp2 {
        let Some(bop) = bool_op(field(insn, 45, 2)) else {
            return un();
        };
        let register_form = top & 0xf000 == 0x5000;
        let cbuf = !register_form && top & 0x0080 != 0;
        let (b, bm, bsw) = if register_form {
            (
                reg20,
                FMod {
                    neg: field(insn, 31, 1) != 0,
                    abs: field(insn, 30, 1) != 0,
                },
                bsw_reg,
            )
        } else if cbuf {
            (
                const_operand(insn),
                FMod {
                    neg: field(insn, 56, 1) != 0,
                    abs: field(insn, 54, 1) != 0,
                },
                HSwizzle::F32,
            )
        } else {
            (imm, no_mod, HSwizzle::H1H0)
        };
        let cmp = fcmp(if register_form {
            field(insn, 35, 4)
        } else {
            field(insn, 49, 4)
        });
        let flag = field(insn, if register_form { 49 } else { 53 }, 1) != 0;
        if is_set2 {
            return Some(Op::Hset2 {
                dst,
                a,
                am: set_am,
                asw,
                b,
                bm,
                bsw,
                cmp,
                bop,
                src,
                bf: flag,
                ftz: field(insn, if register_form { 50 } else { 54 }, 1) != 0,
            });
        }
        return Some(Op::Hsetp2 {
            p0: reg(insn, 3, 3),
            p1: reg(insn, 0, 3),
            a,
            am: set_am,
            asw,
            b,
            bm,
            bsw,
            cmp,
            bop,
            src,
            and: flag,
            ftz: field(insn, 6, 1) != 0,
        });
    }

    None
}
