//! ALU ops outside the 0xfff8 group, plus `ffma` and `xmad` forms.

use super::*;

/// The ops whose opcode field is wider or narrower than the 0xfff8 group [`decode_alu`] handles.
pub(super) fn decode_alu_wide(insn: u64) -> Op {
    let un = Op::Unimplemented { raw: insn };
    let form = insn >> 48;

    // The half-precision group first.
    if let Some(op) = decode_half(insn) {
        return op;
    }

    // ---- 0xfff0-masked: fsetp/isetp/iset/icmp/prmt/lop3/bfi ----
    // Each immediate form is listed twice, as `0x36…`/`0x38…` and one above.
    match form >> 4 {
        // fsetp, cmp 48..52, ftz 47, bop 45..47.
        0x5bb | 0x4bb | 0x36b | 0x37b => {
            let b = match form >> 12 {
                0x5 => Operand::Reg(reg(insn, 20, 8)),
                0x4 => const_operand(insn),
                _ => Operand::Imm(imm20f(insn)),
            };
            let Some(bop) = bool_op(field(insn, 45, 2)) else {
                return un;
            };
            return Op::Fsetp {
                p0: reg(insn, 3, 3),
                p1: reg(insn, 0, 3),
                a: reg(insn, 8, 8),
                am: FMod {
                    neg: field(insn, 43, 1) != 0,
                    abs: field(insn, 7, 1) != 0,
                },
                b,
                bm: FMod {
                    neg: field(insn, 6, 1) != 0,
                    abs: field(insn, 44, 1) != 0,
                },
                cmp: fcmp(field(insn, 48, 4)),
                bop,
                src: src_pred(insn, 39, 42),
            };
        }
        // isetp, cmp 49..52, signed 48, bop 45..47, x 43.
        0x5b6 | 0x4b6 | 0x366 | 0x376 => {
            let b = match form >> 12 {
                0x5 => Operand::Reg(reg(insn, 20, 8)),
                0x4 => const_operand(insn),
                _ => Operand::Imm(imm20(insn)),
            };
            let Some(bop) = bool_op(field(insn, 45, 2)) else {
                return un;
            };
            if field(insn, 43, 1) != 0 {
                return un; // extended-carry compare
            }
            return Op::Isetp {
                p0: reg(insn, 3, 3),
                p1: reg(insn, 0, 3),
                a: reg(insn, 8, 8),
                b,
                cmp: icmp(field(insn, 49, 3)),
                signed: field(insn, 48, 1) != 0,
                bop,
                src: src_pred(insn, 39, 42),
            };
        }
        // iset, the register-writing form of isetp.
        0x5b5 | 0x4b5 | 0x365 | 0x375 => {
            let b = match form >> 12 {
                0x5 => Operand::Reg(reg(insn, 20, 8)),
                0x4 => const_operand(insn),
                _ => Operand::Imm(imm20(insn)),
            };
            let Some(bop) = bool_op(field(insn, 45, 2)) else {
                return un;
            };
            return Op::Iset {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                cmp: icmp(field(insn, 49, 3)),
                signed: field(insn, 48, 1) != 0,
                bop,
                src: src_pred(insn, 39, 42),
                bf: field(insn, 44, 1) != 0,
            };
        }
        // icmp.
        0x5b4 | 0x4b4 | 0x534 | 0x364 | 0x374 => {
            let (b, c) = match form >> 4 {
                0x5b4 => (Operand::Reg(reg(insn, 20, 8)), reg(insn, 39, 8)),
                0x364 | 0x374 => (Operand::Imm(imm20(insn)), reg(insn, 39, 8)),
                _ => (const_operand(insn), reg(insn, 39, 8)),
            };
            return Op::Icmp {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                c,
                cmp: icmp(field(insn, 49, 3)),
                signed: field(insn, 48, 1) != 0,
            };
        }
        // bfi.
        0x5bf | 0x4bf | 0x53f | 0x36f | 0x37f => {
            let (src, base) = match form >> 4 {
                0x5bf => (
                    Operand::Reg(reg(insn, 20, 8)),
                    Operand::Reg(reg(insn, 39, 8)),
                ),
                0x4bf => (const_operand(insn), Operand::Reg(reg(insn, 39, 8))),
                0x53f => (Operand::Reg(reg(insn, 39, 8)), const_operand(insn)),
                _ => (Operand::Imm(imm20(insn)), Operand::Reg(reg(insn, 39, 8))),
            };
            return Op::Bfi {
                dst: reg(insn, 0, 8),
                insert: reg(insn, 8, 8),
                src,
                base,
            };
        }
        // iadd3: three-way add, negation per source.
        0x5cc | 0x4cc | 0x38c | 0x39c => {
            let (b, c) = match form >> 12 {
                0x5 => (
                    Operand::Reg(reg(insn, 20, 8)),
                    Operand::Reg(reg(insn, 39, 8)),
                ),
                0x4 => (const_operand(insn), Operand::Reg(reg(insn, 39, 8))),
                _ => (Operand::Imm(imm20(insn)), Operand::Reg(reg(insn, 39, 8))),
            };
            if field(insn, 48, 1) != 0 {
                return un; // extended-carry
            }
            return Op::Iadd3 {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                aneg: field(insn, 51, 1) != 0,
                b,
                bneg: field(insn, 50, 1) != 0,
                c,
                cneg: field(insn, 49, 1) != 0,
            };
        }
        // psetp, a pure predicate op.
        _ => {}
    }

    if insn & 0xfff8_0000_0000_0000 == 0x5090_0000_0000_0000 {
        let (Some(op1), Some(op2)) = (bool_op(field(insn, 24, 2)), bool_op(field(insn, 45, 2)))
        else {
            return un;
        };
        return Op::Psetp {
            p0: reg(insn, 3, 3),
            p1: reg(insn, 0, 3),
            a: src_pred(insn, 12, 15),
            b: src_pred(insn, 29, 32),
            c: src_pred(insn, 39, 42),
            op1,
            op2,
        };
    }

    // csetp, 0x50a0/0xfff8: Eden's `CSETP`.
    if insn & 0xfff8_0000_0000_0000 == 0x50a0_0000_0000_0000 {
        let Some(op) = bool_op(field(insn, 45, 2)) else {
            return un;
        };
        return Op::Csetp {
            p0: reg(insn, 3, 3),
            p1: reg(insn, 0, 3),
            test: field(insn, 8, 5) as u8,
            src: src_pred(insn, 39, 42),
            op,
        };
    }

    // nop, 0x50b0/0xfff8.
    if insn & 0xfff8_0000_0000_0000 == 0x50b0_0000_0000_0000 {
        return Op::Nop;
    }

    // vote.vtg.
    if insn & 0xfff8_0000_0000_0000 == 0x50e0_0000_0000_0000 {
        return Op::Nop;
    }

    // vote.
    if insn & 0xfff8_0000_0000_0000 == 0x50d8_0000_0000_0000 {
        let mode = match field(insn, 48, 2) {
            0 => VoteMode::All,
            1 => VoteMode::Any,
            2 => VoteMode::Eq,
            _ => return Op::Unimplemented { raw: insn },
        };
        return Op::Vote {
            dst: reg(insn, 0, 8),
            pred: reg(insn, 45, 3),
            src: src_pred(insn, 39, 42),
            mode,
        };
    }

    // fswzadd.
    if insn & 0xfff8_0000_0000_0000 == 0x50f8_0000_0000_0000 {
        if field(insn, 47, 1) != 0 || field(insn, 39, 2) != 0 {
            return un;
        }
        return Op::Fswzadd {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b: reg(insn, 20, 8),
            swizzle: field(insn, 28, 8) as u8,
            ftz: field(insn, 44, 1) != 0,
        };
    }

    // mufu: subop at 20..24, sat 50, src: neg 48 / abs 46.
    if insn & 0xfff8_0000_0000_0000 == 0x5080_0000_0000_0000 {
        let mufu = match field(insn, 20, 4) {
            0 => MufuOp::Cos,
            1 => MufuOp::Sin,
            2 => MufuOp::Ex2,
            3 => MufuOp::Lg2,
            4 => MufuOp::Rcp,
            5 => MufuOp::Rsq,
            8 => MufuOp::Sqrt,
            _ => return un,
        };
        return Op::Mufu {
            dst: reg(insn, 0, 8),
            src: reg(insn, 8, 8),
            sm: FMod {
                neg: field(insn, 48, 1) != 0,
                abs: field(insn, 46, 1) != 0,
            },
            op: mufu,
            sat: field(insn, 50, 1) != 0,
        };
    }

    // lop3: the LUT byte sits in a different field in each form.
    if insn & 0xfff8_0000_0000_0000 == 0x5be0_0000_0000_0000 {
        if field(insn, 38, 1) != 0 || field(insn, 36, 2) != 0 {
            return un;
        }
        return Op::Lop3 {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b: Operand::Reg(reg(insn, 20, 8)),
            c: Operand::Reg(reg(insn, 39, 8)),
            lut: field(insn, 28, 8) as u8,
        };
    }
    // vmnmx, 0x3a00/0xfe00.
    if insn & 0xfe00_0000_0000_0000 == 0x3a00_0000_0000_0000 {
        const WORD: u64 = 3;
        const MIN: u64 = 5;
        const MAX: u64 = 6;
        let then = field(insn, 51, 3);
        let whole_words = field(insn, 37, 2) == WORD && field(insn, 29, 2) == WORD;
        let signed = field(insn, 48, 1) != 0;
        if !whole_words
            || field(insn, 50, 1) == 0
            || signed != (field(insn, 49, 1) != 0)
            || field(insn, 47, 1) != 0
            || field(insn, 55, 1) != 0
            || !matches!(then, MIN | MAX)
        {
            return Op::Unimplemented { raw: insn };
        }
        return Op::Vmnmx {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b: reg(insn, 20, 8),
            c: reg(insn, 39, 8),
            max: field(insn, 56, 1) != 0,
            then_max: then == MAX,
            signed,
            then_signed: field(insn, 54, 1) != 0,
        };
    }
    if insn & 0xfc00_0000_0000_0000 == 0x3c00_0000_0000_0000 {
        return Op::Lop3 {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b: Operand::Imm(imm20(insn)),
            c: Operand::Reg(RZ),
            lut: field(insn, 48, 8) as u8,
        };
    }

    // ffma, three operand orders across four opcodes.
    if insn & 0xff80_0000_0000_0000 == 0x5980_0000_0000_0000 {
        return decode_ffma(
            insn,
            Operand::Reg(reg(insn, 20, 8)),
            Operand::Reg(reg(insn, 39, 8)),
        );
    }
    if insn & 0xff80_0000_0000_0000 == 0x4980_0000_0000_0000 {
        return decode_ffma(insn, const_operand(insn), Operand::Reg(reg(insn, 39, 8)));
    }
    if insn & 0xff80_0000_0000_0000 == 0x5180_0000_0000_0000 {
        // The register/constant operands are the other way round here.
        return decode_ffma(insn, Operand::Reg(reg(insn, 39, 8)), const_operand(insn));
    }
    if insn & 0xfe80_0000_0000_0000 == 0x3280_0000_0000_0000 {
        return decode_ffma(
            insn,
            Operand::Imm(imm20f(insn)),
            Operand::Reg(reg(insn, 39, 8)),
        );
    }

    // xmad, 16x16 multiply-accumulate, in each of its four operand forms.
    if insn & 0xffc0_0000_0000_0000 == 0x5b00_0000_0000_0000 {
        return decode_xmad(
            insn,
            XmadForm {
                b: Operand::Reg(reg(insn, 20, 8)),
                c: Operand::Reg(reg(insn, 39, 8)),
                bh: field(insn, 35, 1) != 0,
                mode: field(insn, 50, 3),
                x: field(insn, 38, 1) != 0,
                psl: field(insn, 36, 1) != 0,
                mrg: field(insn, 37, 1) != 0,
            },
        );
    }
    // The `rc` form multiplies by the register and adds the bank; `cr` is the other way round.
    if insn & 0xff80_0000_0000_0000 == 0x5100_0000_0000_0000 {
        return decode_xmad(
            insn,
            XmadForm {
                b: Operand::Reg(reg(insn, 39, 8)),
                c: const_operand(insn),
                bh: field(insn, 52, 1) != 0,
                mode: field(insn, 50, 2),
                x: field(insn, 54, 1) != 0,
                psl: false,
                mrg: false,
            },
        );
    }
    if insn & 0xfe00_0000_0000_0000 == 0x4e00_0000_0000_0000 {
        return decode_xmad(
            insn,
            XmadForm {
                b: const_operand(insn),
                c: Operand::Reg(reg(insn, 39, 8)),
                bh: field(insn, 52, 1) != 0,
                mode: field(insn, 50, 2),
                x: field(insn, 54, 1) != 0,
                psl: field(insn, 55, 1) != 0,
                mrg: field(insn, 56, 1) != 0,
            },
        );
    }
    // The immediate form carries a **16-bit** `b` at 20..36 and multiplies by its low half always.
    if insn & 0xfec0_0000_0000_0000 == 0x3600_0000_0000_0000 {
        return decode_xmad(
            insn,
            XmadForm {
                b: Operand::Imm(field(insn, 20, 16) as u32),
                c: Operand::Reg(reg(insn, 39, 8)),
                bh: false,
                mode: field(insn, 50, 3),
                x: field(insn, 38, 1) != 0,
                psl: field(insn, 36, 1) != 0,
                mrg: field(insn, 37, 1) != 0,
            },
        );
    }

    // fset, the register-writing form of fsetp.
    if insn & 0xff00_0000_0000_0000 == 0x5800_0000_0000_0000
        || insn & 0xfe00_0000_0000_0000 == 0x4800_0000_0000_0000
        || insn & 0xfe00_0000_0000_0000 == 0x3000_0000_0000_0000
    {
        let b = match insn >> 57 {
            0x2c => Operand::Reg(reg(insn, 20, 8)),
            0x24 => const_operand(insn),
            _ => Operand::Imm(imm20f(insn)),
        };
        let Some(bop) = bool_op(field(insn, 45, 2)) else {
            return un;
        };
        return Op::Fset {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            am: FMod {
                neg: field(insn, 43, 1) != 0,
                abs: field(insn, 54, 1) != 0,
            },
            b,
            bm: FMod {
                neg: field(insn, 53, 1) != 0,
                abs: field(insn, 44, 1) != 0,
            },
            cmp: fcmp(field(insn, 48, 4)),
            bop,
            src: src_pred(insn, 39, 42),
            bf: field(insn, 52, 1) != 0,
        };
    }

    // ipa, a[]-relative, non-indexed.
    if insn & 0xff00_0040_0000_ff00 == 0xe000_0000_0000_ff00 {
        // The interpolation mode (bits 54..56).
        let mode = field(insn, 54, 2);
        // The sample mode (`SampleMode` in Eden's decode).
        let sample = field(insn, 52, 2);
        if sample > 1 {
            return un;
        }
        let multiply = mode == 1;
        return Op::Ipa {
            dst: reg(insn, 0, 8),
            offset: field(insn, 28, 10) as u16,
            mul: opt_reg(reg(insn, 20, 8)).filter(|_| multiply),
            perspective: multiply,
            sat: field(insn, 51, 1) != 0,
            centroid: sample == 1,
        };
    }

    // texs.
    if insn & 0xf600_0000_0000_0000 == 0xd000_0000_0000_0000 {
        let dst = reg(insn, 0, 8);
        let dst2 = reg(insn, 28, 8);
        let (a, b) = (reg(insn, 8, 8), reg(insn, 20, 8));
        if let (Some((dim, coords, dref)), Some(mask)) = (
            texs_encoding(field(insn, 53, 4), a, b),
            decode_tex_mask(field(insn, 50, 3), dst, dst2),
        ) {
            return Op::Texs {
                dst,
                dst2,
                coords,
                dref,
                handle: field(insn, 36, 13) as u16,
                dim,
                mask,
                f16: field(insn, 59, 1) == 0,
            };
        }
        return un;
    }

    // tex.b: the bindless sample, 0xdeb8/0xfff8.
    if insn & 0xfff8_0000_0000_0000 == 0xdeb8_0000_0000_0000 {
        return decode_tex(insn, true);
    }
    // txq: a bound texture's size, 0xdf48/0xfff8.
    if insn & 0xfff8_0000_0000_0000 == 0xdf48_0000_0000_0000 {
        return decode_txq(insn);
    }
    // tld4: the gather, `110010` at the top and `111` at [51, 54).
    if insn >> 58 == 0b11_0010 && field(insn, 51, 3) == 0b111 {
        return decode_tld4(insn);
    }
    // tex.
    if insn & 0xf800_0000_0000_0000 == 0xc000_0000_0000_0000 {
        return decode_tex(insn, false);
    }

    // The 32-bit-immediate forms.
    if insn & 0xfff0_0000_0000_0000 == 0x0100_0000_0000_0000 {
        return Op::Mov32i {
            dst: reg(insn, 0, 8),
            imm: field(insn, 20, 32) as u32,
        };
    }
    if insn & 0xfc00_0000_0000_0000 == 0x0800_0000_0000_0000 {
        // fadd32i
        return Op::Fadd {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            am: FMod {
                neg: field(insn, 56, 1) != 0,
                abs: field(insn, 54, 1) != 0,
            },
            b: Operand::Imm(field(insn, 20, 32) as u32),
            bm: FMod::NONE,
            ftz: field(insn, 55, 1) != 0,
            sat: false,
        };
    }
    if insn & 0xff00_0000_0000_0000 == 0x1e00_0000_0000_0000 {
        // fmul32i
        return Op::Fmul {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b: Operand::Imm(field(insn, 20, 32) as u32),
            bm: FMod::NONE,
            ftz: field(insn, 55, 1) != 0,
            sat: field(insn, 54, 1) != 0,
            scale: FmulScale::None,
        };
    }
    if insn & 0xfe80_0000_0000_0000 == 0x1c00_0000_0000_0000 {
        // iadd32i
        return Op::Iadd {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            aneg: field(insn, 56, 1) != 0,
            b: Operand::Imm(field(insn, 20, 32) as u32),
            bneg: false,
            cin: false,
            // `iadd32i` writes the carry from bit 52.
            cout: field(insn, 52, 1) != 0,
        };
    }
    if insn & 0xfc00_0000_0000_0000 == 0x0400_0000_0000_0000 {
        // lop32i
        let op = match field(insn, 53, 2) {
            0 => LogicOp::And,
            1 => LogicOp::Or,
            2 => LogicOp::Xor,
            _ => LogicOp::PassB,
        };
        return Op::Lop {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            ainv: field(insn, 55, 1) != 0,
            b: Operand::Imm(field(insn, 20, 32) as u32),
            binv: field(insn, 56, 1) != 0,
            op,
            pred: None,
        };
    }

    un
}

fn decode_ffma(insn: u64, b: Operand, c: Operand) -> Op {
    if field(insn, 51, 2) != 0 {
        return Op::Unimplemented { raw: insn }; // explicit rounding modes
    }
    Op::Ffma {
        dst: reg(insn, 0, 8),
        a: reg(insn, 8, 8),
        b,
        bneg: field(insn, 48, 1) != 0,
        c,
        cneg: field(insn, 49, 1) != 0,
        ftz: field(insn, 53, 2) == 1,
        sat: field(insn, 50, 1) != 0,
    }
}

/// What one `xmad` form supplies.
struct XmadForm {
    b: Operand,
    c: Operand,
    bh: bool,
    mode: u64,
    x: bool,
    psl: bool,
    mrg: bool,
}

fn decode_xmad(insn: u64, form: XmadForm) -> Op {
    let un = Op::Unimplemented { raw: insn };
    if form.x {
        return un; // the extended-carry form
    }
    let cmode = match form.mode {
        0 => XmadC::Full,
        1 => XmadC::Lo,
        2 => XmadC::Hi,
        4 => XmadC::Bcc,
        // `csfu` folds a sign into an unsigned product.
        _ => return un,
    };
    let sign = field(insn, 48, 2);
    Op::Xmad {
        dst: reg(insn, 0, 8),
        a: reg(insn, 8, 8),
        ah: field(insn, 53, 1) != 0,
        asigned: sign == 1 || sign == 3,
        b: form.b,
        bh: form.bh,
        bsigned: sign == 2 || sign == 3,
        c: form.c,
        cmode,
        psl: form.psl,
        mrg: form.mrg,
    }
}
