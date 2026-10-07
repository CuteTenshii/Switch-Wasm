use super::*;

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
    let mut inv = Invocation::new();
    assert!(inv
        .execute(&program, &Env::new(&no_consts(), &NoTextures))
        .is_err());
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
