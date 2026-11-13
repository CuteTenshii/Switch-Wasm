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
            x: false,
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
        op(asm(0x5c40, &[(0, 8, 0), (8, 8, 1), (20, 8, 2), (48, 3, 7)])),
        Op::Lop {
            dst: 0,
            a: 1,
            ainv: false,
            b: Operand::Reg(2),
            binv: false,
            op: LogicOp::And,
            pred: None,
            cc: false,
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
            dst_bits: 32,
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
            src_bits: 32,
            hi: false,
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

#[test]
fn decodes_every_lea_form() {
    // lea.hi r4, r4, r3, rz, 0x1f: Echoes of Wisdom's.
    assert_eq!(
        op(0x5bdf7f81f0370404),
        Op::Lea {
            dst: 4,
            a: 4,
            hi: Some(255),
            b: Operand::Reg(3),
            neg: false,
            shift: 31
        }
    );
    // lea r0, -r1, r2, 0x4
    assert_eq!(
        op(asm(
            0x5bd7,
            &[(0, 8, 0), (8, 8, 1), (20, 8, 2), (39, 5, 4), (45, 1, 1)]
        )),
        Op::Lea {
            dst: 0,
            a: 1,
            hi: None,
            b: Operand::Reg(2),
            neg: true,
            shift: 4
        }
    );
    // lea r0, r1, c[0x3][0x10], 0x2
    assert_eq!(
        op(asm(
            0x4bd7,
            &[(0, 8, 0), (8, 8, 1), (20, 14, 4), (34, 5, 3), (39, 5, 2)]
        )),
        Op::Lea {
            dst: 0,
            a: 1,
            hi: None,
            b: Operand::Const {
                bank: 3,
                offset: 0x10
            },
            neg: false,
            shift: 2
        }
    );
    // lea r0, r1, -0x8, 0x3: the immediate's sign is the opcode's low bit.
    assert_eq!(
        op(asm(
            0x37d7,
            &[(0, 8, 0), (8, 8, 1), (20, 19, 0x7fff8), (39, 5, 3)]
        )),
        Op::Lea {
            dst: 0,
            a: 1,
            hi: None,
            b: Operand::Imm(-8i32 as u32),
            neg: false,
            shift: 3
        }
    );
    // lea.hi r0, -r1, c[0x3][0x10], r5, 0x1e
    assert_eq!(
        op(asm(
            0x1807,
            &[
                (0, 8, 0),
                (8, 8, 1),
                (20, 14, 4),
                (34, 5, 3),
                (39, 8, 5),
                (51, 5, 30),
                (56, 1, 1)
            ]
        )),
        Op::Lea {
            dst: 0,
            a: 1,
            hi: Some(5),
            b: Operand::Const {
                bank: 3,
                offset: 0x10
            },
            neg: true,
            shift: 30
        }
    );
}

#[test]
fn a_lea_that_writes_a_carry_or_predicate_is_refused() {
    const WORD: u64 = 0x5bdf7f81f0370404;
    for word in [
        WORD | 1 << 38,            // .x
        WORD | 1 << 47,            // .cc
        WORD & !(7 << 48),         // writes p0
        asm(0x36df, &[(0, 8, 0)]), // no immediate .hi
    ] {
        assert!(matches!(op(word), Op::Unimplemented { .. }), "{word:#x}");
    }
}

#[test]
fn f2i_reads_a_half_source_from_the_half_its_selector_names() {
    // The instruction Just Dance 2023's vertex shader stopped the translator on.
    let Op::F2i {
        src_bits,
        hi,
        dst_bytes,
        dst_signed,
        ..
    } = crate::gpu::shader::isa::decode(0x5cb0_0180_0047_0602).op
    else {
        panic!("not decoded as f2i");
    };
    assert_eq!((src_bits, hi, dst_bytes, dst_signed), (16, false, 4, false));
    let high = crate::gpu::shader::isa::decode(0x5cb0_0180_0047_0602 | 1 << 41).op;
    assert!(matches!(high, Op::F2i { hi: true, .. }));
}

#[test]
fn pset_keeps_its_three_predicates_and_picks_the_true_value() {
    // The instruction after `f2i` that Just Dance 2023's vertex shader needs.
    let Op::Pset { dst, a, float, .. } = crate::gpu::shader::isa::decode(0x5088_0380_e007_1006).op
    else {
        panic!("not decoded as pset");
    };
    assert_eq!((dst, a.reg, a.negate, float), (6, 1, false, false));
    assert!(matches!(
        crate::gpu::shader::isa::decode(0x5088_0380_e007_1006 | 1 << 44).op,
        Op::Pset { float: true, .. }
    ));
    assert!(matches!(
        crate::gpu::shader::isa::decode(0x5088_0380_e007_1006 | 1 << 47).op,
        Op::Pset { cc: true, .. }
    ));
}

#[test]
fn fcmp_immediate_selects_between_a_register_and_the_immediate() {
    // The instruction Just Dance 2023's fragment shader needs after its handle.
    let Op::Fcmp {
        dst,
        a,
        b,
        test,
        ftz,
        ..
    } = crate::gpu::shader::isa::decode(0x36a4_81bf_8007_0308).op
    else {
        panic!("not decoded as fcmp");
    };
    assert_eq!((dst, a, ftz), (8, 3, true));
    assert_eq!(b, Operand::Imm(0x3f80_0000));
    assert_eq!(test, Operand::Reg(3));
}

fn unimplemented(word: u64) -> bool {
    matches!(op(word), Op::Unimplemented { .. })
}

#[test]
fn iadd3_decodes_its_extended_and_condition_code_bits() {
    let word = |extra: u64| asm(0x5cc0, &[(0, 8, 4), (8, 8, 1), (20, 8, 2), (39, 8, 3)]) | extra;
    let flags = |word| match op(word) {
        Op::Iadd3 { x, cc, .. } => (x, cc),
        other => panic!("not iadd3: {other:?}"),
    };
    assert_eq!(flags(word(0)), (false, false));
    assert_eq!(flags(word(1 << 48)), (true, false));
    assert_eq!(flags(word(1 << 47)), (false, true));
    // Eden leaves the flags of an extended add unimplemented.
    assert!(unimplemented(word(1 << 48 | 1 << 47)));
}

#[test]
fn isetp_and_iset_decode_the_extended_bit_and_iset_its_flags() {
    let isetp = |extra: u64| asm(0x5b60, &[(0, 3, 1), (3, 3, 0), (8, 8, 1), (20, 8, 2)]) | extra;
    assert!(matches!(op(isetp(0)), Op::Isetp { x: false, .. }));
    assert!(matches!(op(isetp(1 << 43)), Op::Isetp { x: true, .. }));

    let iset = |extra: u64| asm(0x5b50, &[(0, 8, 3), (8, 8, 1), (20, 8, 2)]) | extra;
    assert!(matches!(
        op(iset(0)),
        Op::Iset {
            x: false,
            cc: false,
            ..
        }
    ));
    assert!(matches!(
        op(iset(1 << 43)),
        Op::Iset {
            x: true,
            cc: false,
            ..
        }
    ));
    assert!(matches!(
        op(iset(1 << 47)),
        Op::Iset {
            x: false,
            cc: true,
            ..
        }
    ));
    assert!(unimplemented(iset(1 << 43 | 1 << 47)));
}

#[test]
fn the_logic_and_field_ops_decode_their_condition_code_bit() {
    let lop = asm(0x5c40, &[(0, 8, 3), (8, 8, 1), (20, 8, 2)]);
    assert!(matches!(op(lop), Op::Lop { cc: false, .. }));
    assert!(matches!(op(lop | 1 << 47), Op::Lop { cc: true, .. }));
    // `.x` stays unimplemented, as in Eden.
    assert!(unimplemented(lop | 1 << 43));

    let lop32i = asm(0x0400, &[(0, 8, 3), (8, 8, 1), (20, 32, 0xff)]);
    assert!(matches!(op(lop32i), Op::Lop { cc: false, .. }));
    assert!(matches!(op(lop32i | 1 << 52), Op::Lop { cc: true, .. }));

    let bfe = asm(0x5c00, &[(0, 8, 3), (8, 8, 1), (20, 8, 2)]);
    assert!(matches!(op(bfe), Op::Bfe { cc: false, .. }));
    assert!(matches!(op(bfe | 1 << 47), Op::Bfe { cc: true, .. }));

    let fset = asm(0x5800, &[(0, 8, 3), (8, 8, 1), (20, 8, 2)]);
    assert!(matches!(op(fset), Op::Fset { cc: false, .. }));
    assert!(matches!(op(fset | 1 << 47), Op::Fset { cc: true, .. }));
}

#[test]
fn lop3_decodes_the_predicate_its_register_form_writes() {
    let lop3 = |extra: u64| {
        asm(
            0x5be0,
            &[(0, 8, 4), (8, 8, 1), (20, 8, 2), (39, 8, 3), (28, 8, 0xc0)],
        ) | extra
    };
    let pred = |word| match op(word) {
        Op::Lop3 { pred, .. } => pred,
        other => panic!("not lop3: {other:?}"),
    };
    assert_eq!(pred(lop3(5 << 48)), Some((5, LopTest::False)));
    assert_eq!(pred(lop3(7 << 48)), None);
    assert_eq!(pred(lop3(1 << 36 | 5 << 48)), Some((5, LopTest::True)));
    assert_eq!(pred(lop3(2 << 36 | 5 << 48)), Some((5, LopTest::Zero)));
    assert_eq!(pred(lop3(3 << 36 | 2 << 48)), Some((2, LopTest::NonZero)));
    assert!(unimplemented(lop3(1 << 38)));
    assert!(unimplemented(lop3(1 << 47)));
}

#[test]
fn r2p_decodes_whether_it_writes_predicates_or_flags() {
    let r2p = asm(0x5cf0, &[(8, 8, 1), (20, 8, 2), (41, 2, 1)]);
    assert!(matches!(
        op(r2p),
        Op::R2p {
            byte: 1,
            flags: false,
            ..
        }
    ));
    assert!(matches!(
        op(r2p | 1 << 40),
        Op::R2p {
            byte: 1,
            flags: true,
            ..
        }
    ));
}

#[test]
fn shr_decodes_bit_reverse_and_still_rejects_the_extended_forms() {
    let shr = asm(0x5c28, &[(0, 8, 3), (8, 8, 1), (20, 8, 2)]);
    assert!(matches!(op(shr), Op::Shr { brev: false, .. }));
    assert!(matches!(op(shr | 1 << 40), Op::Shr { brev: true, .. }));
    assert!(unimplemented(shr | 1 << 43));
}

#[test]
fn ffma_refuses_an_explicit_rounding_mode_and_the_condition_codes() {
    let ffma = asm(0x5980, &[(0, 8, 3), (8, 8, 1), (20, 8, 2), (39, 8, 4)]);
    assert!(matches!(op(ffma), Op::Ffma { .. }));
    for rounding in 1..4u64 {
        assert!(unimplemented(ffma | rounding << 51), "rounding {rounding}");
    }
    assert!(unimplemented(ffma | 1 << 47));
    assert!(unimplemented(ffma | 3 << 53));
}

#[test]
fn lop3_decodes_its_constant_bank_form() {
    // lop3.lut r4, r1, c[0x3][0x10], r5, 0xc0
    let word = asm(
        0x0200,
        &[
            (0, 8, 4),
            (8, 8, 1),
            (20, 14, 4),
            (34, 5, 3),
            (39, 8, 5),
            (48, 8, 0xc0),
        ],
    );
    assert_eq!(
        op(word),
        Op::Lop3 {
            dst: 4,
            a: 1,
            b: Operand::Const {
                bank: 3,
                offset: 0x10
            },
            c: Operand::Reg(5),
            lut: 0xc0,
            pred: None,
        }
    );
}

#[test]
fn iscadd32i_decodes_its_scale_and_carry_out() {
    // iscadd32i r4, r1, 0x1234, 3
    let word = asm(
        0x1400,
        &[(0, 8, 4), (8, 8, 1), (20, 32, 0x1234), (53, 5, 3)],
    );
    assert_eq!(
        op(word),
        Op::Iscadd {
            dst: 4,
            a: 1,
            aneg: false,
            b: Operand::Imm(0x1234),
            bneg: false,
            shift: 3,
            cout: false,
        }
    );
    assert!(matches!(op(word | 1 << 52), Op::Iscadd { cout: true, .. }));
}

#[test]
fn i2f_decodes_a_half_destination_and_rejects_a_double() {
    // i2f.f16.s32 r0, r1
    let word = |float_format| {
        asm(
            0x5cb8,
            &[
                (0, 8, 0),
                (20, 8, 1),
                (8, 2, float_format),
                (10, 2, 2),
                (13, 1, 1),
            ],
        )
    };
    assert!(matches!(op(word(1)), Op::I2f { dst_bits: 16, .. }));
    assert!(matches!(op(word(2)), Op::I2f { dst_bits: 32, .. }));
    assert!(unimplemented(word(3)));
}

/// Eden throws on `.cc` for each of these, so a decode must not ignore the bit.
#[test]
fn the_condition_code_forms_eden_leaves_unimplemented_are_refused() {
    let regs = &[(0, 8, 3), (8, 8, 1), (20, 8, 2)];
    let conversion = &[(0, 8, 3), (8, 2, 2), (10, 2, 2), (20, 8, 2)];
    let forms: [(&str, u64, u32); 14] = [
        ("fadd", asm(0x5c58, regs), 47),
        ("fmul", asm(0x5c68, regs), 47),
        ("fmnmx", asm(0x5c60, regs), 47),
        ("imnmx", asm(0x5c20, regs), 47),
        ("shr", asm(0x5c28, regs), 47),
        ("shl", asm(0x5c48, regs), 47),
        ("i2f", asm(0x5cb8, conversion), 47),
        ("f2i", asm(0x5cb0, conversion), 47),
        ("f2f", asm(0x5ca8, conversion), 47),
        ("lop3 immediate", asm(0x3c00, regs), 47),
        ("lop3 constant bank", asm(0x0200, regs), 47),
        ("ffma", asm(0x5980, regs), 47),
        ("fadd32i", asm(0x0800, regs), 52),
        ("fmul32i", asm(0x1e00, regs), 52),
    ];
    for (name, word, cc) in forms {
        assert!(!unimplemented(word), "{name} decodes without .cc");
        assert!(unimplemented(word | 1 << cc), "{name} with .cc");
    }
}

#[test]
fn fmul32i_keeps_its_mode_at_53_and_saturation_at_55() {
    let word = asm(0x1e00, &[(0, 8, 3), (8, 8, 1), (20, 32, 0x3f80_0000)]);
    let flags = |word| match op(word) {
        Op::Fmul { ftz, fmz, sat, .. } => (ftz, fmz, sat),
        other => panic!("not fmul: {other:?}"),
    };
    assert_eq!(flags(word), (false, false, false));
    assert_eq!(flags(word | 1 << 53), (true, false, false));
    assert_eq!(flags(word | 2 << 53), (false, true, false));
    assert_eq!(flags(word | 1 << 55), (false, false, true));
    assert_eq!(flags(word | 2 << 53 | 1 << 55), (false, true, true));
    assert!(unimplemented(word | 3 << 53));
}

#[test]
fn fadd32i_reads_the_negation_and_absolute_value_of_its_immediate() {
    let word = asm(0x0800, &[(0, 8, 3), (8, 8, 1), (20, 32, 0x3f80_0000)]);
    let modifier = |word| match op(word) {
        Op::Fadd { bm, .. } => bm,
        other => panic!("not fadd: {other:?}"),
    };
    assert_eq!(modifier(word), FMod::NONE);
    assert_eq!(
        modifier(word | 1 << 53),
        FMod {
            neg: true,
            abs: false
        }
    );
    assert_eq!(
        modifier(word | 1 << 57),
        FMod {
            neg: false,
            abs: true
        }
    );
}

#[test]
fn iadd3_refuses_the_register_form_half_selects_and_shift() {
    let word = asm(0x5cc0, &[(0, 8, 4), (8, 8, 1), (20, 8, 2), (39, 8, 3)]);
    assert!(!unimplemented(word));
    for bit in [31, 32, 33, 34, 35, 36, 37, 38] {
        assert!(unimplemented(word | 1 << bit), "bit {bit}");
    }
}
