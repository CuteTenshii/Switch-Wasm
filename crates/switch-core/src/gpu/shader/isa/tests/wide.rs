use super::*;

// ---- the wider instruction set ----

/// Assemble one instruction.
#[test]
fn xmad_immediate_keeps_its_modifiers_where_the_register_form_does() {
    // xmad R1, R2, 0x7, RZ
    let lo = asm(0x3600, &[(0, 8, 1), (8, 8, 2), (20, 15, 7), (39, 8, 255)]);
    match op(lo) {
        Op::Xmad {
            dst,
            a,
            ah,
            b,
            c,
            psl,
            mrg,
            ..
        } => {
            assert_eq!((dst, a), (1, 2));
            assert_eq!(b, Operand::Imm(7));
            assert_eq!(c, Operand::Reg(255));
            assert!(!ah && !psl && !mrg);
        }
        other => panic!("expected xmad, got {other:?}"),
    }
    // xmad.psl R1, R2.h1, 0x7, R0, the same constant, one bit up.
    let hi = asm(
        0x3600,
        &[
            (0, 8, 1),
            (8, 8, 2),
            (20, 15, 7),
            (36, 1, 1),
            (39, 8, 0),
            (53, 1, 1),
        ],
    );
    match op(hi) {
        Op::Xmad { b, c, ah, psl, .. } => {
            assert_eq!(b, Operand::Imm(7), "the immediate absorbed the psl bit");
            assert_eq!(c, Operand::Reg(0));
            assert!(ah, "a.h1 not decoded");
            assert!(psl, "psl not decoded");
        }
        other => panic!("expected xmad.psl, got {other:?}"),
    }
}

#[test]
fn decodes_the_xmad_select_modes_and_operand_forms() {
    let cmode = |raw| match op(raw) {
        Op::Xmad { cmode, .. } => Some(cmode),
        Op::Unimplemented { .. } => None,
        other => panic!("expected xmad, got {other:?}"),
    };
    for (mode, want) in [
        (0, Some(XmadC::Full)),
        (1, Some(XmadC::Lo)),
        (2, Some(XmadC::Hi)),
        // `csfu`, which Eden does not implement either.
        (3, None),
        (4, Some(XmadC::Bcc)),
    ] {
        assert_eq!(cmode(asm(0x5b00, &[(50, 3, mode)])), want, "mode {mode}");
    }
    // A Short Hike's own.
    assert!(matches!(
        op(0x36247f9000180303),
        Op::Xmad {
            cmode: XmadC::Lo,
            ..
        }
    ));
    assert!(matches!(
        op(0x5b30041800970704),
        Op::Xmad {
            cmode: XmadC::Bcc,
            ..
        }
    ));

    // `rc` multiplies by the register and adds the bank; `cr` is the other way round.
    let operands = |raw| match op(raw) {
        Op::Xmad { b, c, .. } => (b, c),
        other => panic!("expected xmad, got {other:?}"),
    };
    let (b, c) = operands(asm(0x5100, &[(39, 8, 5)]));
    assert_eq!(b, Operand::Reg(5));
    assert!(matches!(c, Operand::Const { .. }));
    let (b, c) = operands(asm(0x4e00, &[(39, 8, 5)]));
    assert!(matches!(b, Operand::Const { .. }));
    assert_eq!(c, Operand::Reg(5));
}

const FLOW_TEST_T: u64 = 0xF;

fn asm(opcode: u16, fields: &[(u32, u32, u64)]) -> u64 {
    let mut w = (opcode as u64) << 48;
    w |= 0x7 << 16; // PT, not negated
    for &(pos, len, value) in fields {
        w |= (value & ((1u64 << len) - 1)) << pos;
    }
    w
}

#[test]
fn a_guard_predicate_is_decoded_rather_than_rejected() {
    // The same `exit`, guarded by `!p1`.
    let raw = 0xe3000000_0007000f & !(0xf << 16) | (1 << 16) | (1 << 19);
    let insn = decode(raw);
    assert_eq!(insn.op, Op::Exit);
    assert_eq!(
        insn.pred,
        Pred {
            reg: 1,
            negate: true
        }
    );
    assert!(!insn.pred.is_always());
    assert!(decode(0xe3000000_0007000f).pred.is_always());
}

#[test]
fn decodes_source_modifiers_on_fadd() {
    // fadd $r0, -|$r1|, $r2: neg 48 / abs 46 on a, both clear on b.
    let raw = asm(
        0x5c58,
        &[(0, 8, 0), (8, 8, 1), (20, 8, 2), (48, 1, 1), (46, 1, 1)],
    );
    assert_eq!(
        op(raw),
        Op::Fadd {
            dst: 0,
            a: 1,
            am: FMod {
                neg: true,
                abs: true
            },
            b: Operand::Reg(2),
            bm: FMod::NONE,
            ftz: false,
            sat: false,
        }
    );
}

#[test]
fn decodes_isetp_and_its_predicate_destinations() {
    let raw = asm(
        0x5b60,
        &[
            (0, 3, 7),
            (3, 3, 0),
            (8, 8, 1),
            (20, 8, 2),
            (39, 3, 7),
            (48, 1, 1),
            (49, 3, 1),
        ],
    );
    assert_eq!(
        op(raw),
        Op::Isetp {
            p0: 0,
            p1: 7,
            a: 1,
            b: Operand::Reg(2),
            cmp: ICmp::Lt,
            signed: true,
            bop: BoolOp::And,
            src: Pred::ALWAYS,
        }
    );
}

#[test]
fn decodes_a_relative_branch_to_an_absolute_offset() {
    let raw = asm(0xe240, &[(20, 24, (-0x10i64) as u64), (0, 5, FLOW_TEST_T)]);
    assert_eq!(decode_at(raw, 0x18).op, Op::Bra { target: 0x10 });
}

#[test]
fn decodes_the_reconvergence_ops() {
    assert_eq!(
        decode_at(asm(0xe290, &[(20, 24, 0x18)]), 0).op,
        Op::Ssy { target: 0x28 }
    );
    // And one that lands on a real slot is left alone.
    assert_eq!(
        decode_at(asm(0xe290, &[(20, 24, 0x20)]), 0).op,
        Op::Ssy { target: 0x28 }
    );
    assert_eq!(op(asm(0xf0f8, &[])), Op::Sync);
    assert_eq!(op(asm(0xe340, &[])), Op::Brk);
    assert_eq!(op(asm(0x50b0, &[])), Op::Nop);
}

#[test]
fn a_vertex_stage_vote_is_a_nop() {
    assert_eq!(op(0x50e2_4321_1117_0000), Op::Nop);
    // The whole 0x50e0 group, not just that encoding.
    for low in 0..8u64 {
        assert_eq!(op(0x50e0_0000_0000_0000 | (low << 48)), Op::Nop);
    }
}

#[test]
fn a_range_reduction_keeps_its_modifiers() {
    let neg = FMod {
        neg: true,
        abs: false,
    };
    assert_eq!(
        op(0x5c90_2000_0077_001b),
        Op::Rro {
            dst: 27,
            src: Operand::Reg(7),
            sm: neg,
        }
    );
    assert_eq!(
        op(0x5c90_2000_01f7_0018),
        Op::Rro {
            dst: 24,
            src: Operand::Reg(31),
            sm: neg,
        }
    );
}

/// A negative immediate sets bit 56, which moves the opcode up one.
#[test]
fn a_negative_immediate_does_not_change_the_opcode() {
    assert!(matches!(
        op(0x3764_03ff_fff7_0207),
        Op::Isetp {
            a: 2,
            b: Operand::Imm(u32::MAX),
            cmp: ICmp::Eq,
            signed: false,
            ..
        }
    ));
    assert!(matches!(
        op(0x37b3_03bf_8007_0407),
        Op::Fsetp {
            a: 4,
            b: Operand::Imm(0xbf80_0000),
            ..
        }
    ));
    assert!(matches!(
        op(0x3754_03ff_fff7_0403),
        Op::Iset {
            dst: 3,
            b: Operand::Imm(u32::MAX),
            ..
        }
    ));
    assert!(matches!(
        op(0x39c0_0300_0057_0402),
        Op::Iadd3 {
            dst: 2,
            b: Operand::Imm(0xfff8_0005),
            ..
        }
    ));
}

#[test]
fn surface_accesses_decode_as_envydis_reads_them() {
    assert_eq!(
        op(0xeb20_1386_00f7_0018),
        Op::Sust {
            src: 24,
            coords: 0,
            handle: 0,
            handle_reg: Some(39),
            dim: SurfaceDim::D2,
            data: SurfaceData::Formatted([true; 4]),
        }
    );
    assert_eq!(
        op(0xeb18_0406_0047_0804),
        Op::Suld {
            dst: 4,
            coords: 8,
            handle: 0x40,
            handle_reg: None,
            dim: SurfaceDim::D2,
            data: SurfaceData::Raw(SurfaceSize::B32),
        }
    );
    assert_eq!(
        op(0xeb00_1388_0097_0804),
        Op::Suld {
            dst: 4,
            coords: 8,
            handle: 0,
            handle_reg: Some(39),
            dim: SurfaceDim::Array2d,
            data: SurfaceData::Formatted([true, false, false, true]),
        }
    );
    assert_eq!(
        op(0xeb18_0030_0007_0100),
        Op::Suld {
            dst: 0,
            coords: 1,
            handle: 3,
            handle_reg: None,
            dim: SurfaceDim::D1,
            data: SurfaceData::Raw(SurfaceSize::U8),
        }
    );
    assert!(matches!(
        op(0xeb20_1386_0077_0018),
        Op::Unimplemented { .. }
    ));
    assert!(matches!(
        op(0xeb20_1386_00f7_0018 | 2 << 49),
        Op::Unimplemented { .. }
    ));
}

#[test]
fn a_vote_decodes_its_mode_and_both_predicates() {
    assert_eq!(
        op(0x50d8_e380_0007_0002),
        Op::Vote {
            dst: 2,
            pred: Pred::PT,
            src: Pred::ALWAYS,
            mode: VoteMode::All,
        }
    );
    assert_eq!(
        op(0x50d9_a200_0007_0005),
        Op::Vote {
            dst: 5,
            pred: 5,
            src: Pred {
                reg: 4,
                negate: false
            },
            mode: VoteMode::Any,
        }
    );
    assert_eq!(
        op(0x50da_4400_0007_0003),
        Op::Vote {
            dst: 3,
            pred: 2,
            src: Pred {
                reg: 0,
                negate: true
            },
            mode: VoteMode::Eq,
        }
    );
}

#[test]
fn a_reconvergence_push_is_never_predicated() {
    for opcode in [0xe290u16, 0xe2a0, 0xe2b0] {
        let raw = asm(opcode, &[(20, 24, 0x20)]);
        assert!(decode_at(raw, 0).pred.is_always(), "opcode {opcode:#x}");
    }
    // `bra` in the same group *is* predicated, and keeps its guard.
    let bra = (asm(0xe240, &[(20, 24, 0x20), (0, 5, FLOW_TEST_T)]) & !(0x7 << 16)) | (3 << 16);
    assert_eq!(
        decode_at(bra, 0).pred,
        Pred {
            reg: 3,
            negate: false
        }
    );
}

#[test]
fn decodes_integer_alu() {
    // iadd r0, r1, -r2
    assert_eq!(
        op(asm(0x5c10, &[(0, 8, 0), (8, 8, 1), (20, 8, 2), (48, 1, 1)])),
        Op::Iadd {
            dst: 0,
            a: 1,
            aneg: false,
            b: Operand::Reg(2),
            bneg: true,
            cin: false,
            cout: false
        }
    );
    // shl r3, r4, 0x2 (immediate form)
    assert_eq!(
        op(asm(0x3848, &[(0, 8, 3), (8, 8, 4), (20, 19, 2)])),
        Op::Shl {
            dst: 3,
            a: 4,
            b: Operand::Imm(2),
            wrap: false
        }
    );
    // lop.and r0, r1, r2
    assert_eq!(
        op(asm(0x5c40, &[(0, 8, 0), (8, 8, 1), (20, 8, 2)])),
        Op::Lop {
            dst: 0,
            a: 1,
            ainv: false,
            b: Operand::Reg(2),
            binv: false,
            op: LogicOp::And,
            pred: None,
        }
    );
    // mov r5, r6: the byte-enable mask must be "all four".
    assert_eq!(
        op(asm(0x5c98, &[(0, 8, 5), (20, 8, 6), (39, 4, 0xf)])),
        Op::Mov {
            dst: 5,
            src: Operand::Reg(6)
        }
    );
}

#[test]
fn decodes_conversions() {
    // i2f.f32.s32 r0, r1
    assert_eq!(
        op(asm(
            0x5cb8,
            &[(0, 8, 0), (20, 8, 1), (8, 2, 2), (10, 2, 2), (13, 1, 1)]
        )),
        Op::I2f {
            dst: 0,
            src: Operand::Reg(1),
            sm: FMod::NONE,
            src_bytes: 4,
            src_signed: true,
            sel: 0,
        }
    );
    // f2i.s32.f32.trunc r2, r3
    assert_eq!(
        op(asm(
            0x5cb0,
            &[
                (0, 8, 2),
                (20, 8, 3),
                (10, 2, 2),
                (8, 2, 2),
                (12, 1, 1),
                (39, 2, 3)
            ]
        )),
        Op::F2i {
            dst: 2,
            src: Operand::Reg(3),
            sm: FMod::NONE,
            dst_bytes: 4,
            dst_signed: true,
            round: FRound::Trunc,
            ftz: false,
        }
    );
}

#[test]
fn the_half_instructions_a_unity_shader_issues() {
    // hadd2.f32 $r12 $r12 $r17.
    assert_eq!(
        op(0x5d12800011170c0c),
        Op::Hadd2 {
            dst: 12,
            a: 12,
            am: FMod::NONE,
            asw: HSwizzle::F32,
            b: Operand::Reg(17),
            bm: FMod::NONE,
            bsw: HSwizzle::F32,
            merge: HMerge::F32,
            ftz: false,
            sat: false,
        }
    );
    // hmul2.f32 $r0 $r9 $r4
    assert_eq!(
        op(0x5d0a800010470900),
        Op::Hmul2 {
            dst: 0,
            a: 9,
            am: FMod::NONE,
            asw: HSwizzle::F32,
            b: Operand::Reg(4),
            bm: FMod::NONE,
            bsw: HSwizzle::F32,
            merge: HMerge::F32,
            prec: HPrecision::None,
            sat: false,
        }
    );
    // hmul2.f32 $r4 $r5 c1[0xc].
    assert_eq!(
        op(0x7882800400370504),
        Op::Hmul2 {
            dst: 4,
            a: 5,
            am: FMod::NONE,
            asw: HSwizzle::F32,
            b: Operand::Const {
                bank: 1,
                offset: 0xc
            },
            bm: FMod::NONE,
            bsw: HSwizzle::F32,
            merge: HMerge::F32,
            prec: HPrecision::None,
            sat: false,
        }
    );
    assert_eq!(
        op(0x7a02883c0f070900),
        Op::Hadd2 {
            dst: 0,
            a: 9,
            am: FMod {
                neg: true,
                abs: false
            },
            asw: HSwizzle::F32,
            b: Operand::Imm(0x3C00_3C00),
            bm: FMod::NONE,
            bsw: HSwizzle::H1H0,
            merge: HMerge::F32,
            ftz: false,
            sat: false,
        }
    );
    // hadd2 $r8.h0 -$rZ.h0_h0 c1[0x0].
    assert_eq!(
        op(0x7a8508040007ff08),
        Op::Hadd2 {
            dst: 8,
            a: RZ,
            am: FMod {
                neg: true,
                abs: false
            },
            asw: HSwizzle::H0H0,
            b: Operand::Const { bank: 1, offset: 0 },
            bm: FMod::NONE,
            bsw: HSwizzle::F32,
            merge: HMerge::MrgH0,
            ftz: false,
            sat: false,
        }
    );
    assert_eq!(
        op(0x7e85038c0507ff07),
        Op::Hsetp2 {
            p0: 0,
            p1: Pred::PT,
            a: RZ,
            am: FMod::NONE,
            asw: HSwizzle::H0H0,
            b: Operand::Const {
                bank: 3,
                offset: 0x140
            },
            bm: FMod::NONE,
            bsw: HSwizzle::F32,
            cmp: FCmp::Eq,
            bop: BoolOp::And,
            src: Pred::ALWAYS,
            and: false,
            ftz: false,
        }
    );
}

#[test]
fn every_half_operand_form_reaches_its_op() {
    // hfma2 $r1 $r2 $r3 $r4, register form.
    let hfma_reg = asm(0x5d00, &[(0, 8, 1), (8, 8, 2), (20, 8, 3), (39, 8, 4)]);
    assert!(matches!(
        op(hfma_reg),
        Op::Hfma2 {
            dst: 1,
            a: 2,
            b: Operand::Reg(3),
            c: Operand::Reg(4),
            ..
        }
    ));
    // hfma2 with the constant bank as `b` (`cr`) and as `c` (`rc`).
    let hfma_cr = asm(
        0x7080,
        &[(0, 8, 1), (8, 8, 2), (39, 8, 4), (20, 14, 3), (34, 5, 2)],
    );
    assert!(matches!(
        op(hfma_cr),
        Op::Hfma2 {
            b: Operand::Const {
                bank: 2,
                offset: 0xc
            },
            c: Operand::Reg(4),
            ..
        }
    ));
    let hfma_rc = asm(
        0x6080,
        &[(0, 8, 1), (8, 8, 2), (39, 8, 4), (20, 14, 3), (34, 5, 2)],
    );
    assert!(matches!(
        op(hfma_rc),
        Op::Hfma2 {
            b: Operand::Reg(4),
            c: Operand::Const {
                bank: 2,
                offset: 0xc
            },
            ..
        }
    ));
    assert!(matches!(
        op(asm(0x2c00, &[(0, 8, 1), (8, 8, 2), (20, 32, 0x3c00_3c00)])),
        Op::Hadd2 {
            dst: 1,
            a: 2,
            b: Operand::Imm(0x3c00_3c00),
            merge: HMerge::H1H0,
            ..
        }
    ));
    assert!(matches!(
        op(asm(0x2a00, &[(0, 8, 1), (8, 8, 2), (20, 32, 0x3c00_3c00)])),
        Op::Hmul2 {
            dst: 1,
            b: Operand::Imm(0x3c00_3c00),
            merge: HMerge::H1H0,
            ..
        }
    ));
    assert!(matches!(
        op(asm(0x2800, &[(0, 8, 1), (8, 8, 2), (20, 32, 0x3c00_3c00)])),
        Op::Hfma2 {
            dst: 1,
            c: Operand::Reg(1),
            merge: HMerge::H1H0,
            ..
        }
    ));
    assert!(matches!(
        op(asm(
            0x5d18,
            &[(0, 8, 1), (8, 8, 2), (20, 8, 3), (35, 4, 4), (49, 1, 1)]
        )),
        Op::Hset2 {
            dst: 1,
            a: 2,
            b: Operand::Reg(3),
            cmp: FCmp::Gt,
            bf: true,
            ..
        }
    ));
    assert!(matches!(
        op(asm(
            0x7c80,
            &[(0, 8, 1), (8, 8, 2), (49, 4, 4), (20, 14, 3), (34, 5, 2)]
        )),
        Op::Hset2 {
            cmp: FCmp::Gt,
            b: Operand::Const {
                bank: 2,
                offset: 0xc
            },
            ..
        }
    ));
    // hsetp2's `.h_and` collapses both lanes into one predicate.
    assert!(matches!(
        op(asm(
            0x5d20,
            &[
                (3, 3, 1),
                (0, 3, 2),
                (8, 8, 2),
                (20, 8, 3),
                (35, 4, 1),
                (49, 1, 1)
            ]
        )),
        Op::Hsetp2 {
            p0: 1,
            p1: 2,
            cmp: FCmp::Lt,
            and: true,
            ..
        }
    ));
}

#[test]
fn an_immediate_half_pair_reassembles_both_signs() {
    // -1.0 in the low half (0xbc00) and +2.0 in the high (0x4000).
    let insn = asm(
        0x7a00,
        &[(20, 9, 0xf0), (29, 1, 1), (30, 9, 0x100), (56, 1, 0)],
    );
    assert_eq!(half_imm(insn), 0x4000_bc00);
}
