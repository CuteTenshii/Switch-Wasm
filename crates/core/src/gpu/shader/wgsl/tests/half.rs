//! Half-precision translation tests.

use super::*;
use crate::gpu::shader::isa::{HMerge, HPrecision, HSwizzle};

/// Every half-precision op, in the modes that change what is emitted.
fn half_opcodes() -> Vec<Op> {
    let pair = |asw, bsw, merge| Op::Hadd2 {
        dst: 1,
        a: 2,
        am: NO_MOD,
        asw,
        b: Operand::Reg(3),
        bm: NO_MOD,
        bsw,
        merge,
        ftz: true,
        sat: true,
    };
    vec![
        pair(HSwizzle::H1H0, HSwizzle::H1H0, HMerge::H1H0),
        pair(HSwizzle::H0H0, HSwizzle::F32, HMerge::F32),
        pair(HSwizzle::H1H1, HSwizzle::H1H0, HMerge::MrgH0),
        pair(HSwizzle::H1H0, HSwizzle::H1H0, HMerge::MrgH1),
        Op::Hmul2 {
            dst: 1,
            a: 2,
            am: FMod {
                neg: true,
                abs: true,
            },
            asw: HSwizzle::H1H0,
            b: Operand::Const {
                bank: 1,
                offset: 0x10,
            },
            bm: NO_MOD,
            bsw: HSwizzle::F32,
            merge: HMerge::H1H0,
            prec: HPrecision::Fmz,
            sat: false,
        },
        Op::Hfma2 {
            dst: 1,
            a: 2,
            asw: HSwizzle::H1H0,
            b: Operand::Reg(3),
            bneg: true,
            bsw: HSwizzle::H1H0,
            c: Operand::Reg(4),
            cneg: false,
            csw: HSwizzle::H1H0,
            merge: HMerge::H1H0,
            prec: HPrecision::Fmz,
            sat: false,
        },
        Op::Hfma2 {
            dst: 1,
            a: 2,
            asw: HSwizzle::H1H0,
            b: Operand::Imm(0x3c00_3c00),
            bneg: false,
            bsw: HSwizzle::H1H0,
            c: Operand::Reg(1),
            cneg: true,
            csw: HSwizzle::H1H0,
            merge: HMerge::H1H0,
            prec: HPrecision::Ftz,
            sat: true,
        },
        Op::Hset2 {
            dst: 1,
            a: 2,
            am: NO_MOD,
            asw: HSwizzle::H1H0,
            b: Operand::Reg(3),
            bm: NO_MOD,
            bsw: HSwizzle::H1H0,
            cmp: FCmp::Gt,
            bop: BoolOp::And,
            src: ALWAYS,
            bf: true,
            ftz: false,
        },
        Op::Hsetp2 {
            p0: 0,
            p1: 1,
            a: 2,
            am: NO_MOD,
            asw: HSwizzle::H1H0,
            b: Operand::Reg(3),
            bm: NO_MOD,
            bsw: HSwizzle::H1H0,
            cmp: FCmp::Lt,
            bop: BoolOp::Or,
            src: IF_P0,
            and: true,
            ftz: true,
        },
        Op::Hsetp2 {
            p0: 0,
            p1: 1,
            a: 2,
            am: NO_MOD,
            asw: HSwizzle::H1H0,
            b: Operand::Reg(3),
            bm: NO_MOD,
            bsw: HSwizzle::H1H0,
            cmp: FCmp::Ne,
            bop: BoolOp::And,
            src: ALWAYS,
            and: false,
            ftz: false,
        },
    ]
}

#[test]
fn every_half_opcode_translates() {
    for op in half_opcodes() {
        let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
        let wgsl = translate(&p)
            .unwrap_or_else(|e| panic!("{op:?}: {e}"))
            .source;
        assert!(braces_balance(&wgsl), "{op:?} left a block open:\n{wgsl}");
    }
}

/// Halves round as `f32_to_f16` does.
#[test]
fn a_half_op_unpacks_its_lanes_and_a_merge_keeps_the_other_one() {
    let ops: Vec<(Op, Pred)> = half_opcodes()
        .into_iter()
        .map(|op| (op, ALWAYS))
        .chain([(Op::Exit, ALWAYS)])
        .collect();
    let wgsl = translate(&program(&ops)).unwrap().source;
    assert!(
        wgsl.contains("pack2x16float(fsat2((hftz(unpack2x16float(r2))"),
        "{wgsl}"
    );
    assert!(wgsl.contains("unpack2x16float(r2).xx"), "{wgsl}");
    assert!(wgsl.contains("unpack2x16float(r2).yy"), "{wgsl}");
    assert!(wgsl.contains("vec2<f32>(bitcast<f32>(r3))"), "{wgsl}");
    assert!(wgsl.contains("(r1 & 0xffff0000u) |"), "{wgsl}");
    assert!(wgsl.contains("(r1 & 0x0000ffffu) |"), "{wgsl}");
    assert!(wgsl.contains("== vec2<f32>(0.0)) | ("), "{wgsl}");
    assert!(wgsl.contains("ftz2(vec2<f32>(bitcast<f32>(r3)))"), "{wgsl}");
}
