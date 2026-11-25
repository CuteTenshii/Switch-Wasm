use super::*;
use crate::gpu::shader::isa::LopTest;

#[test]
fn a_hand_written_alu_program_produces_the_expected_registers() {
    // r2 = r0 * r1; r3 = r2 * r1 + r0.
    let program = prog(&[
        Op::Fmul {
            dst: 2,
            a: 0,
            b: Operand::Reg(1),
            bm: FMod::NONE,
            ftz: true,
            fmz: false,
            sat: false,
            scale: FmulScale::None,
        },
        Op::Ffma {
            dst: 3,
            a: 2,
            b: Operand::Reg(1),
            bneg: false,
            c: Operand::Reg(0),
            cneg: false,
            ftz: true,
            fmz: false,
            sat: false,
        },
        Op::Exit,
    ]);
    let mut inv = Invocation::new();
    inv.set_reg_f32(0, 2.0);
    inv.set_reg_f32(1, 3.0);

    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();

    assert_eq!(inv.reg_f32(2), 6.0);
    assert_eq!(inv.reg_f32(3), 20.0);
}

/// Z-A's tick difference: a 64-bit pair to a double, then to a single, each to nearest even.
#[test]
fn an_integer_pair_narrows_to_a_single_through_a_double() {
    let program = prog(&[
        Op::I2d {
            dst: 4,
            src: 2,
            src_bytes: 8,
            src_signed: false,
        },
        Op::D2f {
            dst: 6,
            src: 4,
            ftz: false,
        },
        Op::I2d {
            dst: 8,
            src: 1,
            src_bytes: 4,
            src_signed: true,
        },
        Op::Exit,
    ]);
    // 2^53 + 1 ties to 2^53 in the double; the single holds it exactly.
    let ticks = (1u64 << 53) + 1;
    let mut inv = Invocation::new();
    inv.set_reg(2, ticks as u32);
    inv.set_reg(3, (ticks >> 32) as u32);
    inv.set_reg(1, (-3i32) as u32);

    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();

    let double = u64::from(inv.reg(4)) | u64::from(inv.reg(5)) << 32;
    assert_eq!(f64::from_bits(double), 9_007_199_254_740_992.0);
    assert_eq!(inv.reg_f32(6), 9_007_199_254_740_992.0);
    let negative = u64::from(inv.reg(8)) | u64::from(inv.reg(9)) << 32;
    assert_eq!(f64::from_bits(negative), -3.0);
}

#[test]
fn ffma32i_accumulates_into_its_destination() {
    // Taisei's "ffma32i ftz $r14 $r9 0x3e4ccccd $r14": r14 = r9 * 0.2 + r14.
    let decoded = crate::gpu::shader::isa::decode(0x0c23e4ccccd7090e).op;
    let program = prog(&[decoded, Op::Exit]);
    let mut inv = Invocation::new();
    inv.set_reg_f32(9, 10.0);
    inv.set_reg_f32(14, 1.5);

    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();

    assert_eq!(
        inv.reg_f32(14),
        10.0f32.mul_add(f32::from_bits(0x3e4ccccd), 1.5)
    );
}

/// A program at real byte offsets so branch targets resolve; slot 0 of each block is `sched`.
fn prog_at(entries: &[(Op, Pred)]) -> Compiled {
    let mut p = crate::gpu::shader::Program::default();
    let mut offset = crate::gpu::shader::ENTRY_OFFSET;
    for &(op, pred) in entries {
        p.insns.push(Instruction { pred, op });
        p.offsets.push(offset);
        offset = crate::gpu::shader::next_slot(offset);
    }
    Compiled::new(&p)
}

#[test]
fn a_guard_predicate_skips_the_instruction() {
    // r1 = 1.0 always; r2 = 2.0 only if p0; r3 = 3.0 only if !p0.
    let program = prog_at(&[
        (
            Op::Mov32i {
                dst: 1,
                imm: 1.0f32.to_bits(),
            },
            Pred::ALWAYS,
        ),
        (
            Op::Mov32i {
                dst: 2,
                imm: 2.0f32.to_bits(),
            },
            Pred {
                reg: 0,
                negate: false,
            },
        ),
        (
            Op::Mov32i {
                dst: 3,
                imm: 3.0f32.to_bits(),
            },
            Pred {
                reg: 0,
                negate: true,
            },
        ),
        (Op::Exit, Pred::ALWAYS),
    ]);
    let mut inv = Invocation::new();
    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();
    assert_eq!(inv.reg_f32(1), 1.0);
    assert_eq!(inv.reg(2), 0, "a false guard must skip the write");
    assert_eq!(inv.reg_f32(3), 3.0);
}

#[test]
fn isetp_then_a_predicated_branch_takes_the_right_path() {
    // if (r0 < r1) r2 = 10 else r2 = 20
    let program = prog_at(&[
        (
            Op::Isetp {
                p0: 0,
                p1: 7,
                a: 0,
                b: Operand::Reg(1),
                cmp: ICmp::Lt,
                signed: true,
                bop: BoolOp::And,
                src: Pred::ALWAYS,
                x: false,
            },
            Pred::ALWAYS,
        ),
        // @!p0 bra else
        (
            Op::Bra { target: 0x30 },
            Pred {
                reg: 0,
                negate: true,
            },
        ),
        (Op::Mov32i { dst: 2, imm: 10 }, Pred::ALWAYS),
        (Op::Bra { target: 0x38 }, Pred::ALWAYS), // skip the else
        (Op::Mov32i { dst: 2, imm: 20 }, Pred::ALWAYS), // else, at 0x30
        (Op::Exit, Pred::ALWAYS),                 // at 0x38
    ]);
    // Offset 0x20 is a `sched` word.
    let offsets: Vec<u32> = (0..program.len()).map(|i| program.offset(i)).collect();
    assert_eq!(offsets, vec![0x08, 0x10, 0x18, 0x28, 0x30, 0x38]);

    let mut taken = Invocation::new();
    taken.set_reg(0, 1);
    taken.set_reg(1, 2);
    taken
        .execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();
    assert_eq!(taken.reg(2), 10);

    let mut not_taken = Invocation::new();
    not_taken.set_reg(0, 5);
    not_taken.set_reg(1, 2);
    not_taken
        .execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();
    assert_eq!(not_taken.reg(2), 20);
}

#[test]
fn a_backward_branch_runs_a_real_loop() {
    // r1 = 0; do { r1 += 1 } while (r1 < 4)
    let program = prog_at(&[
        (Op::Mov32i { dst: 1, imm: 0 }, Pred::ALWAYS),
        // loop body, at 0x10
        (
            Op::Iadd {
                dst: 1,
                a: 1,
                aneg: false,
                b: Operand::Imm(1),
                bneg: false,
                cin: false,
                cout: false,
            },
            Pred::ALWAYS,
        ),
        (
            Op::Isetp {
                p0: 0,
                p1: 7,
                a: 1,
                b: Operand::Imm(4),
                cmp: ICmp::Lt,
                signed: true,
                bop: BoolOp::And,
                src: Pred::ALWAYS,
                x: false,
            },
            Pred::ALWAYS,
        ),
        (
            Op::Bra { target: 0x10 },
            Pred {
                reg: 0,
                negate: false,
            },
        ),
        (Op::Exit, Pred::ALWAYS),
    ]);
    let mut inv = Invocation::new();
    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();
    assert_eq!(inv.reg(1), 4);
}

#[test]
fn ssy_and_sync_reconverge() {
    let program = prog_at(&[
        (Op::Ssy { target: 0x30 }, Pred::ALWAYS),
        (Op::Mov32i { dst: 1, imm: 7 }, Pred::ALWAYS),
        (Op::Sync, Pred::ALWAYS),
        (Op::Mov32i { dst: 3, imm: 5 }, Pred::ALWAYS), // skipped by the sync
        (Op::Mov32i { dst: 2, imm: 9 }, Pred::ALWAYS), // at 0x30
        (Op::Exit, Pred::ALWAYS),
    ]);
    let mut inv = Invocation::new();
    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();
    assert_eq!(inv.reg(1), 7);
    assert_eq!(inv.reg(3), 0);
    assert_eq!(inv.reg(2), 9);
}

#[test]
fn a_program_that_never_exits_fails_instead_of_hanging() {
    let program = prog_at(&[(Op::Bra { target: 0x08 }, Pred::ALWAYS)]);
    let err = Invocation::new()
        .execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap_err();
    assert!(err.to_string().contains("did not terminate"), "got {err}");
}

#[test]
fn kil_discards_the_fragment() {
    let program = prog_at(&[(Op::Kil, Pred::ALWAYS), (Op::Exit, Pred::ALWAYS)]);
    let mut inv = Invocation::new();
    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();
    assert!(inv.discarded);
}

#[test]
fn integer_ops_use_the_registers_as_integers_not_floats() {
    // An integer op must not round-trip through f32.
    let program = prog_at(&[
        (
            Op::Mov32i {
                dst: 0,
                imm: 0x1234_5678,
            },
            Pred::ALWAYS,
        ),
        (
            Op::Shr {
                dst: 1,
                a: 0,
                b: Operand::Imm(16),
                signed: false,
                wrap: false,
                brev: false,
            },
            Pred::ALWAYS,
        ),
        (
            Op::Lop {
                dst: 2,
                a: 0,
                ainv: false,
                b: Operand::Imm(0xffff),
                binv: false,
                op: LogicOp::And,
                pred: None,
                cc: false,
            },
            Pred::ALWAYS,
        ),
        (
            Op::Iadd {
                dst: 3,
                a: 1,
                aneg: false,
                b: Operand::Reg(2),
                bneg: false,
                cin: false,
                cout: false,
            },
            Pred::ALWAYS,
        ),
        (Op::Exit, Pred::ALWAYS),
    ]);
    let mut inv = Invocation::new();
    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();
    assert_eq!(inv.reg(1), 0x1234);
    assert_eq!(inv.reg(2), 0x5678);
    assert_eq!(inv.reg(3), 0x1234 + 0x5678);
}

#[test]
fn lop3_evaluates_its_truth_table() {
    // lut 0xe8 is majority(a, b, c).
    assert_eq!(lop3(0b1100, 0b1010, 0b0110, 0xe8), 0b1110);
    // lut 0xf0 is a, 0xcc b, 0xaa c.
    assert_eq!(lop3(0xdead, 0xbeef, 0x1234, 0xf0), 0xdead);
    assert_eq!(lop3(0xdead, 0xbeef, 0x1234, 0xcc), 0xbeef);
    assert_eq!(lop3(0xdead, 0xbeef, 0x1234, 0xaa), 0x1234);
}

#[test]
fn f2i_pset_and_fcmp_run_the_way_just_dance_2023_uses_them() {
    let imm = |dst, imm| (Op::Mov32i { dst, imm }, Pred::ALWAYS);
    let program = prog_at(&[
        // 2.5 and 7.0 as f16, in the low and high halves.
        imm(0, 0x4700_4100),
        (
            Op::F2i {
                dst: 1,
                src: Operand::Reg(0),
                sm: FMod::NONE,
                dst_bytes: 4,
                dst_signed: false,
                round: FRound::Trunc,
                ftz: false,
                src_bits: 16,
                hi: false,
            },
            Pred::ALWAYS,
        ),
        (
            Op::F2i {
                dst: 2,
                src: Operand::Reg(0),
                sm: FMod::NONE,
                dst_bytes: 4,
                dst_signed: false,
                round: FRound::Trunc,
                ftz: false,
                src_bits: 16,
                hi: true,
            },
            Pred::ALWAYS,
        ),
        (
            Op::Pset {
                dst: 3,
                a: Pred::ALWAYS,
                b: Pred::ALWAYS,
                c: Pred::ALWAYS,
                op1: BoolOp::And,
                op2: BoolOp::And,
                float: false,
                cc: false,
            },
            Pred::ALWAYS,
        ),
        (
            Op::Pset {
                dst: 4,
                a: Pred::ALWAYS,
                b: Pred::ALWAYS,
                c: Pred::ALWAYS,
                op1: BoolOp::And,
                op2: BoolOp::And,
                float: true,
                cc: false,
            },
            Pred::ALWAYS,
        ),
        imm(5, 0xaaaa),
        imm(6, (-1.0f32).to_bits()),
        (
            Op::Fcmp {
                dst: 7,
                a: 5,
                b: Operand::Imm(0xbbbb),
                test: Operand::Reg(6),
                cmp: FCmp::Lt,
                ftz: false,
            },
            Pred::ALWAYS,
        ),
        (
            Op::Fcmp {
                dst: 8,
                a: 5,
                b: Operand::Imm(0xbbbb),
                test: Operand::Reg(6),
                cmp: FCmp::Gt,
                ftz: false,
            },
            Pred::ALWAYS,
        ),
        (Op::Exit, Pred::ALWAYS),
    ]);
    let mut inv = Invocation::new();
    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();
    assert_eq!((inv.reg(1), inv.reg(2)), (2, 0x7000 >> 12));
    assert_eq!((inv.reg(3), inv.reg(4)), (u32::MAX, 1f32.to_bits()));
    assert_eq!((inv.reg(7), inv.reg(8)), (0xaaaa, 0xbbbb));
}

#[test]
fn conversions_round_the_way_the_instruction_asks() {
    let program = prog_at(&[
        (
            Op::Mov32i {
                dst: 0,
                imm: (-2.5f32).to_bits(),
            },
            Pred::ALWAYS,
        ),
        (
            Op::F2i {
                dst: 1,
                src: Operand::Reg(0),
                sm: FMod::NONE,
                dst_bytes: 4,
                dst_signed: true,
                round: FRound::Trunc,
                ftz: false,
                src_bits: 32,
                hi: false,
            },
            Pred::ALWAYS,
        ),
        (
            Op::F2i {
                dst: 2,
                src: Operand::Reg(0),
                sm: FMod::NONE,
                dst_bytes: 4,
                dst_signed: true,
                round: FRound::Floor,
                ftz: false,
                src_bits: 32,
                hi: false,
            },
            Pred::ALWAYS,
        ),
        (
            Op::I2f {
                dst: 3,
                src: Operand::Reg(1),
                sm: FMod::NONE,
                src_bytes: 4,
                src_signed: true,
                sel: 0,
                dst_bits: 32,
            },
            Pred::ALWAYS,
        ),
        (Op::Exit, Pred::ALWAYS),
    ]);
    let mut inv = Invocation::new();
    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();
    assert_eq!(inv.reg(1) as i32, -2);
    assert_eq!(inv.reg(2) as i32, -3);
    assert_eq!(inv.reg_f32(3), -2.0);
}

#[test]
fn rz_reads_as_zero_and_discards_writes() {
    let program = prog(&[
        Op::Fmul {
            dst: 0xff,
            a: 0,
            b: Operand::Reg(1),
            bm: FMod::NONE,
            ftz: true,
            fmz: false,
            sat: false,
            scale: FmulScale::None,
        },
        Op::Ffma {
            dst: 2,
            a: 0xff,
            b: Operand::Reg(1),
            bneg: false,
            c: Operand::Reg(5),
            cneg: false,
            ftz: true,
            fmz: false,
            sat: false,
        },
        Op::Exit,
    ]);
    let mut inv = Invocation::new();
    inv.set_reg_f32(0, 99.0);
    inv.set_reg_f32(1, 3.0);
    inv.set_reg_f32(5, 7.0);

    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();

    // dst=RZ discards the write.
    assert_eq!(inv.reg_f32(2), 0.0 * 3.0 + 7.0);
}

#[test]
fn a_half_op_computes_both_lanes_at_once() {
    let inv = run_half(
        &[(1, halves(1.0, 2.0)), (2, halves(0.5, -4.0))],
        &[hadd2(0, 1, 2, HSwizzle::H1H0, HSwizzle::H1H0, HMerge::H1H0)],
    );
    assert_eq!(lanes(inv.reg(0)), [1.5, -2.0]);
}

/// Every swizzle but `H1_H0` reads one lane twice; `F32` reads one float.
#[test]
fn a_half_swizzle_chooses_which_lanes_a_source_offers() {
    let a = halves(1.0, 2.0);
    let b = halves(10.0, 20.0);
    let inv = run_half(
        &[(1, a), (2, b)],
        &[
            hadd2(3, 1, 2, HSwizzle::H0H0, HSwizzle::H1H0, HMerge::H1H0),
            hadd2(4, 1, 2, HSwizzle::H1H1, HSwizzle::H1H0, HMerge::H1H0),
            hadd2(5, 1, 2, HSwizzle::H1H0, HSwizzle::H0H0, HMerge::H1H0),
        ],
    );
    assert_eq!(lanes(inv.reg(3)), [11.0, 21.0]);
    assert_eq!(lanes(inv.reg(4)), [12.0, 22.0]);
    assert_eq!(lanes(inv.reg(5)), [11.0, 12.0]);
}

/// `hadd2.f32` is a plain float add on the half unit.
#[test]
fn a_half_op_in_f32_mode_is_an_ordinary_float_op() {
    let inv = run_half(
        &[(1, 1.5f32.to_bits()), (2, 2.25f32.to_bits())],
        &[hadd2(0, 1, 2, HSwizzle::F32, HSwizzle::F32, HMerge::F32)],
    );
    assert_eq!(f32::from_bits(inv.reg(0)), 3.75);
}

/// A merging write keeps the other half and reads the destination.
#[test]
fn a_merging_half_op_keeps_the_half_it_does_not_write() {
    let inv = run_half(
        &[
            (0, halves(7.0, 9.0)),
            (1, halves(1.0, 2.0)),
            (2, halves(0.5, -4.0)),
        ],
        &[hadd2(
            0,
            1,
            2,
            HSwizzle::H1H0,
            HSwizzle::H1H0,
            HMerge::MrgH0,
        )],
    );
    assert_eq!(lanes(inv.reg(0)), [1.5, 9.0]);

    let inv = run_half(
        &[
            (0, halves(7.0, 9.0)),
            (1, halves(1.0, 2.0)),
            (2, halves(0.5, -4.0)),
        ],
        &[hadd2(
            0,
            1,
            2,
            HSwizzle::H1H0,
            HSwizzle::H1H0,
            HMerge::MrgH1,
        )],
    );
    assert_eq!(lanes(inv.reg(0)), [7.0, -2.0]);

    let merging = hadd2(3, 1, 2, HSwizzle::H1H0, HSwizzle::H1H0, HMerge::MrgH1);
    assert!(
        reads(&merging).contains(&3),
        "a merge reads its destination back"
    );
    let whole = hadd2(3, 1, 2, HSwizzle::H1H0, HSwizzle::H1H0, HMerge::H1H0);
    assert!(!reads(&whole).contains(&3), "a full write does not");
}

#[test]
fn a_half_multiply_add_runs_per_lane() {
    let inv = run_half(
        &[
            (1, halves(2.0, 3.0)),
            (2, halves(4.0, 5.0)),
            (3, halves(1.0, -1.0)),
        ],
        &[Op::Hfma2 {
            dst: 0,
            a: 1,
            asw: HSwizzle::H1H0,
            b: Operand::Reg(2),
            bneg: false,
            bsw: HSwizzle::H1H0,
            c: Operand::Reg(3),
            cneg: false,
            csw: HSwizzle::H1H0,
            merge: HMerge::H1H0,
            prec: HPrecision::None,
            sat: false,
        }],
    );
    assert_eq!(lanes(inv.reg(0)), [9.0, 14.0]);
}

/// `.fmz`: anything times zero is zero, including infinity and NaN.
#[test]
fn fmz_makes_anything_times_zero_zero() {
    let hmul = |prec| Op::Hmul2 {
        dst: 0,
        a: 1,
        am: FMod::NONE,
        asw: HSwizzle::H1H0,
        b: Operand::Reg(2),
        bm: FMod::NONE,
        bsw: HSwizzle::H1H0,
        merge: HMerge::H1H0,
        prec,
        sat: false,
    };
    let operands = [(1, halves(0.0, 2.0)), (2, halves(f32::INFINITY, 3.0))];
    let plain = run_half(&operands, &[hmul(HPrecision::None)]);
    assert!(lanes(plain.reg(0))[0].is_nan());
    let fmz = run_half(&operands, &[hmul(HPrecision::Fmz)]);
    assert_eq!(lanes(fmz.reg(0)), [0.0, 6.0]);
}

/// `hsetp2` writes one predicate per lane; `.h_and` ands them and writes the inverse.
#[test]
fn hsetp2_writes_one_predicate_per_lane_until_h_and() {
    let setp = |and| Op::Hsetp2 {
        p0: 0,
        p1: 1,
        a: 1,
        am: FMod::NONE,
        asw: HSwizzle::H1H0,
        b: Operand::Reg(2),
        bm: FMod::NONE,
        bsw: HSwizzle::H1H0,
        cmp: FCmp::Gt,
        bop: BoolOp::And,
        src: Pred::ALWAYS,
        and,
        ftz: false,
    };
    let operands = [(1, halves(5.0, 1.0)), (2, halves(2.0, 8.0))];
    let split = run_half(&operands, &[setp(false)]);
    assert!(split.pred(0) && !split.pred(1));
    let anded = run_half(&operands, &[setp(true)]);
    assert!(!anded.pred(0) && anded.pred(1));
}

#[test]
fn hset2_fills_each_half_with_its_own_lane() {
    let set = |bf| Op::Hset2 {
        dst: 0,
        a: 1,
        am: FMod::NONE,
        asw: HSwizzle::H1H0,
        b: Operand::Reg(2),
        bm: FMod::NONE,
        bsw: HSwizzle::H1H0,
        cmp: FCmp::Gt,
        bop: BoolOp::And,
        src: Pred::ALWAYS,
        bf,
        ftz: false,
    };
    let operands = [(1, halves(5.0, 1.0)), (2, halves(2.0, 8.0))];
    assert_eq!(run_half(&operands, &[set(false)]).reg(0), 0x0000_FFFF);
    // `.bf` answers 1.0h.
    assert_eq!(run_half(&operands, &[set(true)]).reg(0), halves(1.0, 0.0));
}

/// `vmnmx` minimum of three words, signed and unsigned.
#[test]
fn vmnmx_takes_the_minimum_then_compares_with_the_third_operand() {
    let run = |word: u64, regs: [(u8, u32); 3]| run_half(&regs, &[isa::decode(word).op]).reg(7);
    const WORD: u64 = 0x3a2c03e060c70907;
    assert_eq!(run(WORD, [(9, 5), (12, 9), (7, 3)]), 3);
    assert_eq!(run(WORD, [(9, 5), (12, 9), (7, 7)]), 5);
    let signed = WORD | 1 << 48 | 1 << 49 | 1 << 54;
    assert_eq!(run(WORD, [(9, u32::MAX), (12, 2), (7, 9)]), 2);
    assert_eq!(run(signed, [(9, u32::MAX), (12, 2), (7, 9)]), u32::MAX);
}

/// A half `.ftz` flushes at the half threshold, only for half lanes.
#[test]
fn ftz_flushes_a_subnormal_half_but_not_a_small_float() {
    let subnormal = f16_to_f32(0x0001);
    let mut add = hadd2(0, 1, 2, HSwizzle::H1H0, HSwizzle::H1H0, HMerge::H1H0);
    let Op::Hadd2 { ftz, .. } = &mut add else {
        unreachable!()
    };
    *ftz = true;
    let inv = run_half(
        &[(1, halves(subnormal, 1.0)), (2, halves(0.0, 0.0))],
        &[add],
    );
    assert_eq!(lanes(inv.reg(0)), [0.0, 1.0]);

    // An f32 lane is left alone.
    let mut add = hadd2(0, 1, 2, HSwizzle::F32, HSwizzle::F32, HMerge::F32);
    let Op::Hadd2 { ftz, .. } = &mut add else {
        unreachable!()
    };
    *ftz = true;
    let inv = run_half(&[(1, subnormal.to_bits()), (2, 0.0f32.to_bits())], &[add]);
    assert_eq!(f32::from_bits(inv.reg(0)), subnormal);
}

/// `lea.hi` adds the low word of a 64-bit shift right; Echoes of Wisdom's halves r4 into r3.
#[test]
fn lea_hi_adds_the_low_word_of_a_wide_shift() {
    let run = |word: u64, regs: &[(u8, u32)]| run_half(regs, &[isa::decode(word).op]).reg(4);
    const WORD: u64 = 0x5bdf7f81f0370404;
    assert_eq!(run(WORD, &[(4, 0x8000_0001), (3, 0x10)]), 0x4000_0010);

    let with_hi = WORD & !(0xff << 39) | 5 << 39;
    assert_eq!(
        run(with_hi, &[(4, 0x8000_0001), (5, 3), (3, 0x10)]),
        0xc000_0010
    );
    // Negation is of the whole 64 bits, so a zero low word borrows nothing.
    let negated = with_hi | 1 << 37;
    assert_eq!(run(negated, &[(4, 2), (5, 0), (3, 0x10)]), 0xf);
    assert_eq!(run(negated, &[(4, 0), (5, 1), (3, 0x10)]), 0x8000_0010);
    // A shift of zero takes the high word whole.
    let unshifted = with_hi & !(0x1f << 28);
    assert_eq!(run(unshifted, &[(4, 0x1234), (5, 7), (3, 1)]), 8);
}

#[test]
fn lea_adds_a_shifted_and_optionally_negated_register() {
    // lea r4, r1, r2, 0x4
    let word = 0x5bd7u64 << 48 | 7 << 16 | 4 | 1 << 8 | 2 << 20 | 4 << 39;
    let run = |word: u64| run_half(&[(1, 3), (2, 0x100)], &[isa::decode(word).op]).reg(4);
    assert_eq!(run(word), 0x130);
    assert_eq!(run(word | 1 << 45), 0xd0);
}

fn run_fmz(fmz: bool, sat: bool, a: f32, b: f32, c: f32) -> (f32, f32) {
    let program = prog(&[
        Op::Fmul {
            dst: 3,
            a: 0,
            b: Operand::Reg(1),
            bm: FMod::NONE,
            ftz: false,
            fmz,
            sat,
            scale: FmulScale::None,
        },
        Op::Ffma {
            dst: 4,
            a: 0,
            b: Operand::Reg(1),
            bneg: false,
            c: Operand::Reg(2),
            cneg: false,
            ftz: false,
            fmz,
            sat,
        },
        Op::Exit,
    ]);
    let mut inv = Invocation::new();
    inv.set_reg_f32(0, a);
    inv.set_reg_f32(1, b);
    inv.set_reg_f32(2, c);
    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();
    (inv.reg_f32(3), inv.reg_f32(4))
}

/// `.fmz` on `fmul` and `ffma`: a zero operand beats infinity and NaN, and `ffma` keeps its addend.
#[test]
fn fmz_floats_zero_the_product_of_a_zero_operand() {
    let (mul, fma) = run_fmz(false, false, 0.0, f32::INFINITY, 5.0);
    assert!(mul.is_nan() && fma.is_nan());
    let (mul, fma) = run_fmz(true, false, 0.0, f32::INFINITY, 5.0);
    assert_eq!((mul, fma), (0.0, 5.0));
    let (mul, fma) = run_fmz(true, false, f32::NAN, -0.0, 5.0);
    assert_eq!((mul, fma), (0.0, 5.0));
    let (mul, fma) = run_fmz(true, false, 2.0, 3.0, 5.0);
    assert_eq!((mul, fma), (6.0, 11.0));
}

/// Eden skips `.fmz` under `.sat`: `0 * inf + 0.25` is NaN, which saturates to 0, not the addend.
#[test]
fn fmz_with_sat_is_the_saturated_plain_operation() {
    let (mul, fma) = run_fmz(true, true, 0.0, f32::INFINITY, 0.25);
    assert_eq!((mul, fma), (0.0, 0.0));
    let (mul, fma) = run_fmz(true, false, 0.0, f32::INFINITY, 0.25);
    assert_eq!((mul, fma), (0.0, 0.25));
    let (mul, fma) = run_fmz(true, true, 2.0, 3.0, 0.25);
    assert_eq!((mul, fma), (1.0, 1.0));
}

const Z: u32 = 1;
const S: u32 = 2;
const C: u32 = 4;
const O: u32 = 8;

/// Seeds the flags from `bits` (Z, S, C, O from bit 0) with `r2p.cc` from r9, then runs `ops`.
fn run_with_flags(bits: u32, setup: &[(u8, u32)], ops: &[Op]) -> Invocation {
    let mut program = vec![Op::R2p {
        src: 9,
        mask: Operand::Imm(0xf),
        byte: 0,
        flags: true,
    }];
    program.extend_from_slice(ops);
    let mut regs = vec![(9, bits)];
    regs.extend_from_slice(setup);
    run_half(&regs, &program)
}

fn flags_of(inv: &Invocation) -> u32 {
    [
        (inv.zero, Z),
        (inv.sign, S),
        (inv.carry, C),
        (inv.overflow, O),
    ]
    .into_iter()
    .filter(|&(set, _)| set)
    .fold(0, |bits, (_, flag)| bits | flag)
}

#[test]
fn r2p_cc_writes_only_the_flags_its_mask_names() {
    let r2p = |mask, byte| Op::R2p {
        src: 1,
        mask: Operand::Imm(mask),
        byte,
        flags: true,
    };
    let from = |start: u32, mask, byte, bits: u32| {
        flags_of(&run_with_flags(start, &[(1, bits)], &[r2p(mask, byte)]))
    };
    assert_eq!(from(0, 0xf, 0, 0b0101), Z | C);
    assert_eq!(from(Z | S | C | O, 0xf, 0, 0), 0);
    assert_eq!(from(Z | O, 0b0110, 0, 0b0110), Z | S | C | O);
    assert_eq!(from(0, 0xf, 2, 0x0003_0000), Z | S);
    assert_eq!(from(C, 0b0011, 0, 0b1111), Z | S | C);
    // Bits 4 and up of the mask name predicates, which `.cc` leaves alone.
    let inv = run_with_flags(0, &[(1, 0xff)], &[r2p(0x70, 0)]);
    assert_eq!((flags_of(&inv), inv.pred(4)), (0, false));
}

#[test]
fn lop_cc_takes_z_and_s_from_the_result_and_clears_c_and_o() {
    let lop = |op, cc| Op::Lop {
        dst: 3,
        a: 1,
        ainv: false,
        b: Operand::Reg(2),
        binv: false,
        op,
        pred: None,
        cc,
    };
    let run = |op, a, b| flags_of(&run_with_flags(C | O, &[(1, a), (2, b)], &[lop(op, true)]));
    assert_eq!(run(LogicOp::And, 0xf0, 0x0f), Z);
    assert_eq!(run(LogicOp::Or, 0x8000_0000, 1), S);
    assert_eq!(run(LogicOp::Xor, 3, 1), 0);
    assert_eq!(run(LogicOp::PassB, 9, 0), Z);
    // Without `.cc` the flags are left as they were.
    let plain = lop(LogicOp::And, false);
    let inv = run_with_flags(S | C, &[(1, 1), (2, 1)], &[plain]);
    assert_eq!(flags_of(&inv), S | C);
}

#[test]
fn iadd_cc_writes_all_four_flags_of_the_sum() {
    let iadd = |cin, cout| Op::Iadd {
        dst: 3,
        a: 1,
        aneg: false,
        b: Operand::Reg(2),
        bneg: false,
        cin,
        cout,
    };
    let run = |a, b, op| flags_of(&run_with_flags(Z | S | O, &[(1, a), (2, b)], &[op]));
    assert_eq!(run(1, 2, iadd(false, true)), 0);
    assert_eq!(run(u32::MAX, 1, iadd(false, true)), Z | C);
    assert_eq!(run(0x7fff_ffff, 1, iadd(false, true)), S | O);
    assert_eq!(run(0x8000_0000, 0x8000_0000, iadd(false, true)), Z | C | O);
    // Without `.cc` the flags stay as they were.
    assert_eq!(run(u32::MAX, 1, iadd(false, false)), Z | S | O);
    // `.x.cc` is Eden's TODO: only the carry is written.
    assert_eq!(run(u32::MAX, 1, iadd(true, true)), Z | S | C | O);
}

#[test]
fn iscadd_cc_writes_all_four_flags_of_the_scaled_sum() {
    let iscadd = |aneg, shift| Op::Iscadd {
        dst: 3,
        a: 1,
        aneg,
        b: Operand::Reg(2),
        bneg: false,
        shift,
        cout: true,
    };
    let run = |a, b, op| flags_of(&run_with_flags(S | O, &[(1, a), (2, b)], &[op]));
    assert_eq!(run(1, 2, iscadd(false, 2)), 0);
    assert_eq!(run(1, 0xffff_fffc, iscadd(false, 2)), Z | C);
    assert_eq!(run(0x4000_0000, 0x8000_0000, iscadd(false, 1)), Z | C | O);
    assert_eq!(run(1, 0x7fff_ffff, iscadd(false, 0)), S | O);
    assert_eq!(run(3, 12, iscadd(true, 2)), Z | C);
    // iscadd32i.cc r3, r1, 0x80000000, 1
    let wide =
        isa::decode(0x1400u64 << 48 | 7 << 16 | 3 | 1 << 8 | 0x8000_0000 << 20 | 1 << 52 | 1 << 53)
            .op;
    assert_eq!(run(0x4000_0000, 0, wide), Z | C | O);
}

#[test]
fn lop3_clears_its_predicate_for_the_false_test() {
    let op = Op::Lop3 {
        dst: 4,
        a: 1,
        b: Operand::Reg(2),
        c: Operand::Reg(3),
        lut: 0xf0,
        pred: Some((2, LopTest::False)),
    };
    let mut program = vec![Op::Isetp {
        p0: 2,
        p1: 7,
        a: 0,
        b: Operand::Reg(0),
        cmp: ICmp::Eq,
        signed: false,
        bop: BoolOp::And,
        src: Pred::ALWAYS,
        x: false,
    }];
    program.push(op);
    let inv = run_half(&[(1, 5), (2, 0), (3, 0)], &program);
    assert!(!inv.pred(2));
    assert_eq!(inv.reg(4), 5);
}

#[test]
fn lop3_writes_its_predicate_from_the_result() {
    let run = |test, lut, a| {
        let op = Op::Lop3 {
            dst: 4,
            a: 1,
            b: Operand::Reg(2),
            c: Operand::Reg(3),
            lut,
            pred: Some((2, test)),
        };
        run_half(&[(1, a), (2, 0xf0f0), (3, 0)], &[op])
    };
    // lut 0xf0 is `a`.
    let inv = run(LopTest::Zero, 0xf0, 0);
    assert_eq!((inv.reg(4), inv.pred(2)), (0, true));
    let inv = run(LopTest::Zero, 0xf0, 5);
    assert_eq!((inv.reg(4), inv.pred(2)), (5, false));
    let inv = run(LopTest::NonZero, 0xf0, 5);
    assert!(inv.pred(2));
    let inv = run(LopTest::NonZero, 0xf0, 0);
    assert!(!inv.pred(2));
    let inv = run(LopTest::True, 0xf0, 0);
    assert!(inv.pred(2));
}

#[test]
fn bfe_cc_flags_follow_the_extracted_field() {
    let bfe = Op::Bfe {
        dst: 3,
        a: 1,
        b: Operand::Imm(8 << 8 | 4),
        signed: true,
        brev: false,
        cc: true,
    };
    let run = |a| {
        let inv = run_with_flags(C | O, &[(1, a)], &[bfe]);
        (inv.reg(3), flags_of(&inv))
    };
    assert_eq!(run(0x0000_0ff0), (0xffff_ffff, S));
    assert_eq!(run(0x0000_0070), (0x7, 0));
    assert_eq!(run(0x0000_000f), (0, Z));
}

#[test]
fn iset_pset_and_fset_cc_flag_the_value_they_write() {
    let iset = |bf| Op::Iset {
        dst: 3,
        a: 1,
        b: Operand::Reg(2),
        cmp: ICmp::Eq,
        signed: true,
        bop: BoolOp::And,
        src: Pred::ALWAYS,
        bf,
        x: false,
        cc: true,
    };
    let run = |op, setup: &[(u8, u32)]| {
        let inv = run_with_flags(C | O, setup, &[op]);
        (inv.reg(3), flags_of(&inv))
    };
    assert_eq!(run(iset(false), &[(1, 4), (2, 4)]), (u32::MAX, S));
    assert_eq!(run(iset(false), &[(1, 4), (2, 5)]), (0, Z));
    assert_eq!(run(iset(true), &[(1, 4), (2, 4)]), (1f32.to_bits(), 0));

    let pset = |float| Op::Pset {
        dst: 3,
        a: Pred::ALWAYS,
        b: Pred::ALWAYS,
        c: Pred::ALWAYS,
        op1: BoolOp::And,
        op2: BoolOp::And,
        float,
        cc: true,
    };
    assert_eq!(run(pset(false), &[]), (u32::MAX, S));
    assert_eq!(run(pset(true), &[]), (1f32.to_bits(), 0));

    let fset = |cmp| Op::Fset {
        dst: 3,
        a: 1,
        am: FMod::NONE,
        b: Operand::Reg(2),
        bm: FMod::NONE,
        cmp,
        bop: BoolOp::And,
        src: Pred::ALWAYS,
        bf: false,
        cc: true,
    };
    let (one, two) = (1f32.to_bits(), 2f32.to_bits());
    assert_eq!(run(fset(FCmp::Lt), &[(1, one), (2, two)]), (u32::MAX, S));
    assert_eq!(run(fset(FCmp::Gt), &[(1, one), (2, two)]), (0, Z));
}

#[test]
fn iadd3_cc_derives_its_flags_as_eden_does() {
    let iadd3 = |cc, x| Op::Iadd3 {
        dst: 4,
        a: 1,
        aneg: false,
        b: Operand::Reg(2),
        bneg: false,
        c: Operand::Reg(3),
        cneg: false,
        x,
        cc,
        halves: [Half16::All; 3],
        shift: Iadd3Shift::None,
    };
    let run = |a, b, c| {
        let inv = run_half(&[(1, a), (2, b), (3, c)], &[iadd3(true, false)]);
        (inv.reg(4), flags_of(&inv))
    };
    // C is the carry of the final add only; the carry of `a + b` lands in O.
    assert_eq!(run(u32::MAX, 1, 0), (0, Z | O));
    assert_eq!(run(u32::MAX, 0, 1), (0, Z | C));
    // Signed overflow of the final add sets O and the sign.
    assert_eq!(run(0x7fff_ffff, 0, 1), (0x8000_0000, S | O));
    assert_eq!(run(1, 2, 3), (6, 0));
    // Without `.cc` the flags stay as they were.
    let inv = run_with_flags(C, &[(1, 1), (2, 2), (3, 3)], &[iadd3(false, false)]);
    assert_eq!((inv.reg(4), flags_of(&inv)), (6, C));
}

#[test]
fn iadd3_x_adds_the_carry_into_the_first_sum() {
    let iadd3 = Op::Iadd3 {
        dst: 4,
        a: 1,
        aneg: false,
        b: Operand::Reg(2),
        bneg: false,
        c: Operand::Reg(3),
        cneg: false,
        x: true,
        cc: false,
        halves: [Half16::All; 3],
        shift: Iadd3Shift::None,
    };
    let run = |bits| run_with_flags(bits, &[(1, 10), (2, 20), (3, 100)], &[iadd3]).reg(4);
    assert_eq!(run(0), 130);
    assert_eq!(run(C), 131);
    assert_eq!(run(Z | S | O), 130);
}

#[test]
fn iadd3_picks_halves_before_negating_and_shifts_the_first_sum() {
    use Half16::{All, Lower, Upper};
    let run = |aneg, halves, shift, regs: [u32; 3]| {
        let iadd3 = Op::Iadd3 {
            dst: 4,
            a: 1,
            aneg,
            b: Operand::Reg(2),
            bneg: false,
            c: Operand::Reg(3),
            cneg: false,
            x: false,
            cc: false,
            halves,
            shift,
        };
        let setup = [(1, regs[0]), (2, regs[1]), (3, regs[2])];
        run_half(&setup, &[iadd3]).reg(4)
    };
    let regs = [0xffff_0001, 0x0001_8000, 0x1234_5678];
    assert_eq!(
        run(false, [Upper, Lower, Upper], Iadd3Shift::None, regs),
        0xffff + 0x8000 + 0x1234
    );
    // The carry out of `a + b` is the 33rd bit, and a right shift keeps it as bit 16.
    assert_eq!(
        run(false, [All; 3], Iadd3Shift::Right, regs),
        0x1_0000 + 0x1234_5678
    );
    assert_eq!(
        run(
            false,
            [All; 3],
            Iadd3Shift::Right,
            [0x1234_5678, 0x0001_8000, 1]
        ),
        0x1235 + 1
    );
    assert_eq!(
        run(false, [All; 3], Iadd3Shift::Left, regs),
        0x8001_0000 + 0x1234_5678
    );
    // The low half of 1, negated, is all ones: `-1 + 0x8000` carries.
    assert_eq!(
        run(true, [Lower, Lower, Lower], Iadd3Shift::Right, regs),
        0x1_0000 + 0x5678
    );
}

/// Eden's `ExtendedIntegerCompare`: the hi-word compare of a 64-bit compare whose low word set C and Z.
#[test]
fn isetp_x_continues_a_compare_through_the_carry_and_zero_flags() {
    let run = |cmp, signed, a, b, bits| {
        let isetp = Op::Isetp {
            p0: 0,
            p1: 1,
            a: 1,
            b: Operand::Reg(2),
            cmp,
            signed,
            bop: BoolOp::And,
            src: Pred::ALWAYS,
            x: true,
        };
        let inv = run_with_flags(bits, &[(1, a), (2, b)], &[isetp]);
        assert_eq!(inv.pred(0), !inv.pred(1), "{cmp:?}");
        inv.pred(0)
    };
    let all = [ICmp::Lt, ICmp::Eq, ICmp::Le, ICmp::Gt, ICmp::Ne, ICmp::Ge];
    let row = |signed, a, b, bits| all.map(|cmp| run(cmp, signed, a, b, bits));
    // Equal high words: the low-word compare (Z, and C as "no borrow") decides.
    assert_eq!(
        row(true, 5, 5, C | Z),
        [false, true, true, false, false, true]
    );
    assert_eq!(row(true, 5, 5, C), [false, false, false, true, true, true]);
    assert_eq!(row(true, 5, 5, 0), [true, false, true, false, true, false]);
    // Different high words decide alone.
    assert_eq!(
        row(true, 3, 5, C | Z),
        [true, false, true, false, true, false]
    );
    assert_eq!(row(true, 7, 5, 0), [false, false, false, true, true, true]);
    // Unsigned, with the sign bits differing.
    assert_eq!(
        row(false, 0xffff_ffff, 1, C),
        [false, false, false, true, true, true]
    );
    assert_eq!(
        row(false, 1, 0xffff_ffff, C),
        [true, false, true, false, true, false]
    );
    assert!(run(ICmp::Always, true, 1, 2, 0) && !run(ICmp::Never, true, 1, 2, 0));
}

#[test]
fn iset_x_materialises_the_extended_compare() {
    let iset = Op::Iset {
        dst: 3,
        a: 1,
        b: Operand::Reg(2),
        cmp: ICmp::Eq,
        signed: false,
        bop: BoolOp::And,
        src: Pred::ALWAYS,
        bf: false,
        x: true,
        cc: false,
    };
    let run = |bits| run_with_flags(bits, &[(1, 8), (2, 8)], &[iset]).reg(3);
    assert_eq!(run(C | Z), u32::MAX);
    assert_eq!(run(C), 0);
}

#[test]
fn shr_brev_shifts_the_bit_reversed_source() {
    let shr = |signed| Op::Shr {
        dst: 3,
        a: 1,
        b: Operand::Imm(4),
        signed,
        wrap: false,
        brev: true,
    };
    let run = |signed, a| run_half(&[(1, a)], &[shr(signed)]).reg(3);
    assert_eq!(run(false, 1), 0x0800_0000);
    assert_eq!(run(true, 1), 0xf800_0000);
    assert_eq!(run(true, 0xffff_fffe), 0x07ff_ffff);
}

#[test]
fn i2f_to_half_leaves_the_rounded_half_in_the_low_bits() {
    let i2f = |sm| Op::I2f {
        dst: 3,
        src: Operand::Reg(1),
        sm,
        src_bytes: 4,
        src_signed: true,
        sel: 0,
        dst_bits: 16,
    };
    let run = |sm, v: i32| run_half(&[(1, v as u32), (3, 0xdead_beef)], &[i2f(sm)]).reg(3);
    assert_eq!(run(FMod::NONE, 3), 0x4200);
    assert_eq!(run(FMod::NONE, -3), 0xc200);
    assert_eq!(
        run(
            FMod {
                neg: true,
                abs: false
            },
            3
        ),
        0xc200
    );
    // 2049 is not representable in half precision and rounds to even.
    assert_eq!(run(FMod::NONE, 2049), 0x6800);
    assert_eq!(run(FMod::NONE, 100_000), 0x7c00);
}
