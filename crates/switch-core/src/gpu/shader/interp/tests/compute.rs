use super::*;

#[test]
fn sr_y_direction_reads_a_sign_and_never_a_zero() {
    let up = SpecialRegs {
        y_negate: false,
        ..SpecialRegs::default()
    };
    assert_eq!(f32::from_bits(up.read(0x12).unwrap()), 1.0);
    let down = SpecialRegs {
        y_negate: true,
        ..SpecialRegs::default()
    };
    assert_eq!(f32::from_bits(down.read(0x12).unwrap()), -1.0);
}

#[test]
fn s2r_answers_the_thread_and_cta_registers_a_dispatch_set() {
    let consts = no_consts();
    let mut env = Env::new(&consts, &NoTextures);
    env.special.tid = [3, 4, 5];
    env.special.ctaid = [6, 7, 8];
    env.special.shared_size = 0x400;
    let mut inv = Invocation::new();
    inv.execute(
        &prog(&[
            Op::S2r { dst: 0, sr: 0x21 },
            Op::S2r { dst: 1, sr: 0x23 },
            Op::S2r { dst: 2, sr: 0x25 },
            Op::S2r { dst: 3, sr: 0x27 },
            Op::S2r { dst: 4, sr: 0x32 },
            // Unmodelled: zero, not an error.
            Op::S2r { dst: 5, sr: 0x1d },
            Op::Exit,
        ]),
        &env,
    )
    .unwrap();
    assert_eq!([inv.reg(0), inv.reg(1)], [3, 5]);
    assert_eq!([inv.reg(2), inv.reg(3)], [6, 8]);
    assert_eq!(inv.reg(4), 0x400);
    assert_eq!(inv.reg(5), 0);
}

#[test]
fn the_packed_thread_register_is_refused_rather_than_guessed_at() {
    let consts = no_consts();
    let env = Env::new(&consts, &NoTextures);
    let err = Invocation::new()
        .execute(&prog(&[Op::S2r { dst: 0, sr: 0x20 }, Op::Exit]), &env)
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("packed special register"),
        "got {err:?}"
    );
}

#[test]
fn a_global_store_narrower_than_a_word_touches_only_its_own_bytes() {
    // Byte and halfword loads and stores must not touch neighbours.
    let memory = FlatMemory::with(16);
    memory.write_u32(0, 0xAABB_CCDD).unwrap();
    let consts = no_consts();
    let mut env = Env::new(&consts, &NoTextures);
    env.memory = Some(&memory);

    let mut inv = Invocation::new();
    inv.set_reg(4, 1);
    inv.set_reg(5, 0);
    inv.set_reg(6, 0x77);
    inv.execute(
        &prog(&[
            Op::Stg {
                addr: 4,
                offset: 0,
                src: 6,
                size: MemSize::U8,
            },
            Op::Ldg {
                dst: 0,
                addr: 4,
                offset: 0,
                size: MemSize::U8,
            },
            Op::Ldg {
                dst: 1,
                addr: 4,
                offset: 0,
                size: MemSize::S8,
            },
            Op::Exit,
        ]),
        &env,
    )
    .unwrap();
    assert_eq!(memory.read_u32(0).unwrap(), 0xAABB_77DD);
    assert_eq!(inv.reg(0), 0x77);
    assert_eq!(inv.reg(1), 0x77);
}

#[test]
fn a_signed_narrow_load_extends_its_sign() {
    let memory = FlatMemory::with(16);
    memory.write_u32(0, 0x0000_FF80).unwrap();
    let consts = no_consts();
    let mut env = Env::new(&consts, &NoTextures);
    env.memory = Some(&memory);

    let mut inv = Invocation::new();
    inv.execute(
        &prog(&[
            Op::Ldg {
                dst: 0,
                addr: 4,
                offset: 0,
                size: MemSize::S16,
            },
            Op::Ldg {
                dst: 1,
                addr: 4,
                offset: 0,
                size: MemSize::U16,
            },
            Op::Exit,
        ]),
        &env,
    )
    .unwrap();
    assert_eq!(inv.reg(0), 0xFFFF_FF80);
    assert_eq!(inv.reg(1), 0x0000_FF80);
}

#[test]
fn a_global_atomic_returns_the_old_value_and_leaves_the_new_one() {
    let memory = FlatMemory::with(16);
    memory.write_u32(0, 10).unwrap();
    let consts = no_consts();
    let mut env = Env::new(&consts, &NoTextures);
    env.memory = Some(&memory);

    let mut inv = Invocation::new();
    inv.set_reg(6, 7);
    inv.execute(
        &prog(&[
            Op::Atom {
                dst: 0,
                addr: 4,
                offset: 0,
                src: 6,
                op: AtomOp::Add,
                ty: AtomType::U32,
                space: AtomSpace::Global,
            },
            Op::Exit,
        ]),
        &env,
    )
    .unwrap();
    assert_eq!(inv.reg(0), 10, "the old value");
    assert_eq!(memory.read_u32(0).unwrap(), 17);
}

#[test]
fn every_atomic_operation_computes_what_its_name_says() {
    use AtomOp::*;
    let u32s = AtomType::U32;
    assert_eq!(atom_apply(Add, u32s, 5, 3, 0).unwrap(), 8);
    assert_eq!(atom_apply(Min, u32s, 5, 3, 0).unwrap(), 3);
    assert_eq!(atom_apply(Max, u32s, 5, 3, 0).unwrap(), 5);
    assert_eq!(atom_apply(And, u32s, 0b110, 0b011, 0).unwrap(), 0b010);
    assert_eq!(atom_apply(Or, u32s, 0b110, 0b011, 0).unwrap(), 0b111);
    assert_eq!(atom_apply(Xor, u32s, 0b110, 0b011, 0).unwrap(), 0b101);
    assert_eq!(atom_apply(Exch, u32s, 5, 3, 0).unwrap(), 3);
    let negative = (-4i32) as u32 as u64;
    assert_eq!(
        atom_apply(Min, AtomType::S32, negative, 3, 0).unwrap(),
        negative
    );
    assert_eq!(atom_apply(Min, u32s, negative, 3, 0).unwrap(), 3);
    assert_eq!(atom_apply(Inc, u32s, 2, 4, 0).unwrap(), 3);
    assert_eq!(atom_apply(Inc, u32s, 4, 4, 0).unwrap(), 0);
    assert_eq!(atom_apply(Dec, u32s, 0, 4, 0).unwrap(), 4);
    assert_eq!(atom_apply(Dec, u32s, 3, 4, 0).unwrap(), 2);
    // `cas` stores the register after its comparand, only on a match.
    assert_eq!(atom_apply(Cas, u32s, 5, 5, 9).unwrap(), 9);
    assert_eq!(atom_apply(Cas, u32s, 5, 4, 9).unwrap(), 5);
    let one = 1.0f32.to_bits().into();
    let two = 2.0f32.to_bits().into();
    assert_eq!(
        atom_apply(Add, AtomType::F32, one, two, 0).unwrap(),
        3.0f32.to_bits().into()
    );
}

#[test]
fn a_barrier_suspends_where_it_stands_and_resumes_after_it() {
    let consts = no_consts();
    let env = Env::new(&consts, &NoTextures);
    let program = prog(&[
        Op::Mov32i { dst: 0, imm: 1 },
        Op::Bar {
            mode: BarMode::Sync,
        },
        Op::Mov32i { dst: 1, imm: 2 },
        Op::Exit,
    ]);

    let mut inv = Invocation::new();
    inv.begin();
    assert_eq!(inv.resume(&program, &env).unwrap(), Halt::Barrier);
    assert_eq!(inv.reg(0), 1);
    assert_eq!(inv.reg(1), 0, "nothing past the barrier has run");
    assert_eq!(inv.resume(&program, &env).unwrap(), Halt::Exited);
    assert_eq!(inv.reg(1), 2);
}

#[test]
fn a_barrier_in_a_draw_is_an_error_rather_than_a_silent_no_op() {
    // Outside a CTA a barrier is an error.
    let consts = no_consts();
    let env = Env::new(&consts, &NoTextures);
    let err = Invocation::new()
        .execute(
            &prog(&[
                Op::Bar {
                    mode: BarMode::Sync,
                },
                Op::Exit,
            ]),
            &env,
        )
        .unwrap_err();
    assert!(format!("{err:?}").contains("no CTA"), "got {err:?}");
}

/// `dFdx`'s fetch: each lane reads its horizontal neighbour.
#[test]
fn a_shuffle_suspends_until_the_rest_of_its_warp_can_answer_it() {
    let consts = no_consts();
    let env = Env::new(&consts, &NoTextures);
    let program = prog(&[
        Op::Shfl {
            dst: 1,
            pred: 0,
            src: 0,
            index: Operand::Imm(1),
            mask: Operand::Imm(0x1c),
            mode: ShflMode::Bfly,
        },
        Op::Exit,
    ]);

    // `begin`, not `reset`, to keep the seeded registers.
    let mut warp: [Invocation; 4] = std::array::from_fn(|_| Invocation::new());
    for (lane, invocation) in warp.iter_mut().enumerate() {
        invocation.set_reg(0, 10 + lane as u32);
        invocation.begin();
    }

    for invocation in warp.iter_mut() {
        assert_eq!(invocation.resume(&program, &env).unwrap(), Halt::Warp);
        assert_eq!(invocation.reg(1), 0, "nothing has been exchanged yet");
    }
    resolve_warp(&mut warp);
    for invocation in warp.iter_mut() {
        assert_eq!(invocation.resume(&program, &env).unwrap(), Halt::Exited);
    }

    // `bfly 1` pairs lanes differing in the low bit.
    assert_eq!(warp.each_ref().map(|lane| lane.reg(1)), [11, 10, 13, 12]);
    assert!(
        warp.iter().all(|lane| lane.pred(0)),
        "every lane was in bounds"
    );
}

/// An out-of-clamp lane keeps its own value and clears the predicate.
#[test]
fn a_shuffle_that_reaches_past_its_segment_keeps_the_lane_s_own_value() {
    let consts = no_consts();
    let env = Env::new(&consts, &NoTextures);
    let program = prog(&[
        Op::Shfl {
            dst: 1,
            pred: 0,
            src: 0,
            index: Operand::Imm(1),
            mask: Operand::Imm(0),
            mode: ShflMode::Up,
        },
        Op::Exit,
    ]);

    let mut warp: [Invocation; 2] = std::array::from_fn(|_| Invocation::new());
    for (lane, invocation) in warp.iter_mut().enumerate() {
        invocation.set_reg(0, 10 + lane as u32);
        invocation.begin();
    }
    for invocation in warp.iter_mut() {
        assert_eq!(invocation.resume(&program, &env).unwrap(), Halt::Warp);
    }
    resolve_warp(&mut warp);

    assert_eq!(warp[0].reg(1), 10, "lane 0 has nothing below it to read");
    assert!(!warp[0].pred(0));
    assert_eq!(warp[1].reg(1), 10);
    assert!(warp[1].pred(0));
}

/// `rro` applies modifiers, so a negated zero keeps its sign through `mufu.sin`.
#[test]
fn a_range_reduction_applies_its_negate_and_absolute_value() {
    let consts = no_consts();
    let env = Env::new(&consts, &NoTextures);
    let neg = FMod {
        neg: true,
        abs: false,
    };
    let abs_then_neg = FMod {
        neg: true,
        abs: true,
    };
    let program = prog(&[
        Op::Rro {
            dst: 1,
            src: Operand::Reg(0),
            sm: neg,
        },
        Op::Mufu {
            dst: 2,
            src: 1,
            sm: FMod::default(),
            op: MufuOp::Sin,
            sat: false,
        },
        Op::Rro {
            dst: 3,
            src: Operand::Reg(0),
            sm: abs_then_neg,
        },
        Op::Rro {
            dst: 4,
            src: Operand::Reg(RZ),
            sm: neg,
        },
        Op::Exit,
    ]);
    let mut invocation = Invocation::new();
    invocation.set_reg_f32(0, -0.5);
    invocation.execute(&program, &env).unwrap();
    assert_eq!(invocation.reg_f32(1), 0.5);
    assert_eq!(invocation.reg_f32(2), 0.5f32.sin());
    assert_eq!(invocation.reg_f32(3), -0.5);
    assert_eq!(invocation.reg(4), (-0.0f32).to_bits());
}

/// Three of four lanes hold the predicate; an exited lane isn't counted.
#[test]
fn a_vote_answers_from_the_lanes_that_reached_it() {
    let consts = no_consts();
    let env = Env::new(&consts, &NoTextures);
    for (mode, verdict) in [
        (VoteMode::All, false),
        (VoteMode::Any, true),
        (VoteMode::Eq, false),
    ] {
        let program = prog(&[
            Op::Vote {
                dst: 1,
                pred: 2,
                src: Pred {
                    reg: 0,
                    negate: false,
                },
                mode,
            },
            Op::Exit,
        ]);
        let mut warp: [Invocation; 4] = std::array::from_fn(|_| Invocation::new());
        for (lane, invocation) in warp.iter_mut().enumerate() {
            invocation.set_pred(0, lane != 1);
            invocation.begin();
        }
        for invocation in warp.iter_mut() {
            assert_eq!(invocation.resume(&program, &env).unwrap(), Halt::Warp);
        }
        resolve_warp(&mut warp);
        for invocation in warp.iter_mut() {
            assert_eq!(invocation.resume(&program, &env).unwrap(), Halt::Exited);
        }
        assert_eq!(warp.each_ref().map(|lane| lane.reg(1)), [0b1101; 4]);
        assert!(warp.iter().all(|lane| lane.pred(2) == verdict), "{mode:?}");
    }

    // Only lanes 0 and 2 reach this vote.
    let program = prog(&[
        Op::Vote {
            dst: 1,
            pred: 2,
            src: Pred {
                reg: 0,
                negate: false,
            },
            mode: VoteMode::All,
        },
        Op::Exit,
    ]);
    let mut warp: [Invocation; 4] = std::array::from_fn(|_| Invocation::new());
    for (lane, invocation) in warp.iter_mut().enumerate() {
        invocation.set_pred(0, true);
        invocation.begin();
        if lane % 2 == 0 {
            assert_eq!(invocation.resume(&program, &env).unwrap(), Halt::Warp);
        }
    }
    resolve_warp(&mut warp);
    assert_eq!(warp[0].reg(1), 0b101);
    assert!(warp[0].pred(2) && warp[2].pred(2));
}

#[test]
fn a_shuffle_outside_a_warp_is_an_error_rather_than_a_lane_reading_itself() {
    let consts = no_consts();
    let env = Env::new(&consts, &NoTextures);
    let program = prog(&[
        Op::Shfl {
            dst: 1,
            pred: 0,
            src: 0,
            index: Operand::Imm(1),
            mask: Operand::Imm(0x1c),
            mode: ShflMode::Bfly,
        },
        Op::Exit,
    ]);
    let err = Invocation::new().execute(&program, &env).unwrap_err();
    assert!(format!("{err:?}").contains("another lane"), "got {err:?}");
}

/// Each lane's `fswzadd` sign depends on its quad position.
#[test]
fn the_per_lane_add_takes_its_signs_from_the_lane_it_runs_on() {
    let consts = no_consts();
    let program = prog(&[
        Op::S2r { dst: 5, sr: 0x00 },
        // 0xe4 is the identity swizzle.
        Op::Fswzadd {
            dst: 0,
            a: 1,
            b: 2,
            swizzle: 0xe4,
            ftz: false,
        },
        Op::Exit,
    ]);

    // (-a - b), (a - b), (-a + b), (0 - b).
    for (lane, expected) in [-4.0f32, 2.0, -2.0, -1.0].into_iter().enumerate() {
        let mut env = Env::new(&consts, &NoTextures);
        env.special.lane = lane as u32;
        let mut invocation = Invocation::new();
        invocation.set_reg_f32(1, 3.0);
        invocation.set_reg_f32(2, 1.0);
        invocation.execute(&program, &env).unwrap();
        assert_eq!(invocation.reg_f32(0), expected, "lane {lane}");
        assert_eq!(invocation.reg(5), lane as u32);
    }
}

#[test]
fn shared_memory_is_addressed_in_bytes_and_shared_between_invocations() {
    let consts = no_consts();
    let shared: SharedMemory = RefCell::new(vec![0u8; 64]);
    let mut env = Env::new(&consts, &NoTextures);
    env.shared = Some(&shared);

    let mut writer = Invocation::new();
    writer.set_reg(4, 8);
    writer.set_reg(5, 0xDEAD);
    writer
        .execute(
            &prog(&[
                Op::Sts {
                    addr: 4,
                    offset: 4,
                    src: 5,
                    size: MemSize::B32,
                },
                Op::Exit,
            ]),
            &env,
        )
        .unwrap();

    let mut reader = Invocation::new();
    reader
        .execute(
            &prog(&[
                Op::Lds {
                    dst: 0,
                    addr: RZ,
                    offset: 12,
                    size: MemSize::B32,
                },
                Op::Exit,
            ]),
            &env,
        )
        .unwrap();
    assert_eq!(reader.reg(0), 0xDEAD);
}
