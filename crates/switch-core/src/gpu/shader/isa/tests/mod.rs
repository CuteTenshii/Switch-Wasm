use super::*;

/// The guard-predicate field holding `PT`.
const PT: u64 = 7 << 16;

#[test]
fn decodes_the_shared_memory_pair() {
    // ld/st s[] sit one nibble above their local counterparts and are encoded identically.
    assert_eq!(
        decode((0xef48u64 | 4) << 48 | PT | 0x20 << 20 | 5 << 8 | 3).op,
        Op::Lds {
            dst: 3,
            addr: 5,
            offset: 0x20,
            size: MemSize::B32
        }
    );
    assert_eq!(
        decode((0xef58u64 | 5) << 48 | PT | 6 << 8 | 2).op,
        Op::Sts {
            addr: 6,
            offset: 0,
            src: 2,
            size: MemSize::B64
        }
    );
    // Still the local pair, not the shared one.
    assert_eq!(
        decode((0xef40u64 | 4) << 48 | PT | 5 << 8 | 3).op,
        Op::Ldl {
            dst: 3,
            addr: 5,
            offset: 0,
            size: MemSize::B32
        }
    );
}

#[test]
fn a_negative_shared_offset_stays_negative() {
    let offset = (-8i64 as u64) & 0xFF_FFFF;
    assert_eq!(
        decode((0xef48u64 | 4) << 48 | PT | offset << 20 | 5 << 8 | 3).op,
        Op::Lds {
            dst: 3,
            addr: 5,
            offset: -8,
            size: MemSize::B32
        }
    );
}

#[test]
fn decodes_each_barrier_form() {
    // The mode's bits are not contiguous.
    let bar = |mode: u64| decode(0xf0a8u64 << 48 | mode << 32 | PT).op;
    assert_eq!(
        bar(0x80),
        Op::Bar {
            mode: BarMode::Sync
        }
    );
    assert_eq!(
        bar(0x81),
        Op::Bar {
            mode: BarMode::Arrive
        }
    );
    assert_eq!(
        bar(0x02),
        Op::Bar {
            mode: BarMode::RedPopc
        }
    );
    assert_eq!(
        bar(0x03),
        Op::Bar {
            mode: BarMode::Scan
        }
    );
    assert_eq!(
        bar(0x0a),
        Op::Bar {
            mode: BarMode::RedAnd
        }
    );
    assert_eq!(
        bar(0x12),
        Op::Bar {
            mode: BarMode::RedOr
        }
    );
    // membar and depbar are still the no-ops they were.
    assert_eq!(decode(0xef98u64 << 48 | PT).op, Op::Inert);
    assert_eq!(decode(0xf0f0u64 << 48 | PT).op, Op::Inert);
}

#[test]
fn decodes_a_warp_shuffle_in_each_mode_and_operand_form() {
    // shfl.<mode> p0, r3, r4, 0x1, 0x1c
    let immediate = |mode: u64| {
        decode(
            0xef10u64 << 48
                | 0x1c << 34
                | mode << 30
                | 1 << 29
                | 1 << 28
                | 1 << 20
                | PT
                | 4 << 8
                | 3,
        )
        .op
    };
    for (bits, mode) in [
        (0, ShflMode::Idx),
        (1, ShflMode::Up),
        (2, ShflMode::Down),
        (3, ShflMode::Bfly),
    ] {
        assert_eq!(
            immediate(bits),
            Op::Shfl {
                dst: 3,
                pred: 0,
                src: 4,
                index: Operand::Imm(1),
                mask: Operand::Imm(0x1c),
                mode,
            }
        );
    }

    // The same instruction with both operands in registers.
    assert_eq!(
        decode(0xef10u64 << 48 | 3 << 30 | 6 << 39 | 5 << 20 | PT | 4 << 8 | 3 | 2 << 48).op,
        Op::Shfl {
            dst: 3,
            pred: 2,
            src: 4,
            index: Operand::Reg(5),
            mask: Operand::Reg(6),
            mode: ShflMode::Bfly,
        }
    );
}

#[test]
fn decodes_the_per_lane_add_a_derivative_ends_with() {
    // fswzadd r3, r1, r2, 0xe4
    let fswzadd =
        |extra: u64| decode(0x50f8u64 << 48 | extra | 0xe4 << 28 | 2 << 20 | PT | 1 << 8 | 3).op;
    assert_eq!(
        fswzadd(0),
        Op::Fswzadd {
            dst: 3,
            a: 1,
            b: 2,
            swizzle: 0xe4,
            ftz: false
        }
    );
    assert_eq!(
        fswzadd(1 << 44),
        Op::Fswzadd {
            dst: 3,
            a: 1,
            b: 2,
            swizzle: 0xe4,
            ftz: true
        }
    );
    for extra in [1u64 << 47, 1 << 39, 2 << 39] {
        assert!(
            matches!(fswzadd(extra), Op::Unimplemented { .. }),
            "{extra:#x}"
        );
    }
}

#[test]
fn decodes_a_global_atomic_with_its_operation_and_type() {
    // atom.max.s32 r3, [r5 + -8], r7
    let offset = (-8i64 as u64) & 0xF_FFFF;
    assert_eq!(
        decode(0xed00u64 << 48 | 2 << 52 | 1 << 49 | offset << 28 | 7 << 20 | PT | 5 << 8 | 3).op,
        Op::Atom {
            dst: 3,
            addr: 5,
            offset: -8,
            src: 7,
            op: AtomOp::Max,
            ty: AtomType::S32,
            space: AtomSpace::Global,
        }
    );
}

#[test]
fn a_shared_atomic_counts_its_offset_in_dwords() {
    // The one place the two atomic encodings genuinely differ.
    assert_eq!(
        decode(0xec00u64 << 48 | 8 << 52 | 3 << 30 | 7 << 20 | PT | 5 << 8 | 3).op,
        Op::Atom {
            dst: 3,
            addr: 5,
            offset: 12,
            src: 7,
            op: AtomOp::Exch,
            ty: AtomType::U32,
            space: AtomSpace::Shared,
        }
    );
}

#[test]
fn red_is_an_atomic_that_discards_its_old_value() {
    // Which is exactly RZ as the destination, so the interpreter needs no second path for it.
    assert_eq!(
        decode(0xebf8u64 << 48 | 2 << 20 | 4 << 28 | PT | 5 << 8 | 3).op,
        Op::Atom {
            dst: RZ,
            addr: 5,
            offset: 4,
            src: 3,
            op: AtomOp::Add,
            ty: AtomType::U64,
            space: AtomSpace::Global,
        }
    );
}

#[test]
fn decodes_compare_and_swap_in_both_address_spaces() {
    assert_eq!(
        decode(0xeef0u64 << 48 | 7 << 20 | PT | 5 << 8 | 3).op,
        Op::Atom {
            dst: 3,
            addr: 5,
            offset: 0,
            src: 7,
            op: AtomOp::Cas,
            ty: AtomType::U32,
            space: AtomSpace::Global,
        }
    );
    assert_eq!(
        decode(0xee00u64 << 48 | 2 << 53 | 1 << 52 | 7 << 20 | PT | 5 << 8 | 3).op,
        Op::Atom {
            dst: 3,
            addr: 5,
            offset: 0,
            src: 7,
            op: AtomOp::Cas,
            ty: AtomType::U64,
            space: AtomSpace::Shared,
        }
    );
}

#[test]
fn an_atomic_operation_this_decoder_has_no_name_for_is_not_invented() {
    assert!(matches!(
        decode(0xed00u64 << 48 | 9 << 52 | PT).op,
        Op::Unimplemented { .. }
    ));
}

#[test]
fn an_op_still_fits_in_thirty_two_bytes() {
    assert_eq!(std::mem::size_of::<Op>(), 32);
}

fn op(word: u64) -> Op {
    decode(word).op
}

#[test]
fn decodes_ipa_pass_then_mufu_rcp() {
    // solid.frag: "ipa pass $r0 a[0x7c] 0x0 0x0 0x1"
    assert_eq!(
        op(0xe003ff87cff7ff00),
        Op::Ipa {
            dst: 0,
            offset: 0x7c,
            mul: None,
            perspective: false,
            sat: false,
            centroid: false
        }
    );
    // "mufu rcp $r3 $r0"
    assert_eq!(
        op(0x5080000000470003),
        Op::Mufu {
            dst: 3,
            src: 0,
            sm: FMod::NONE,
            op: MufuOp::Rcp,
            sat: false
        }
    );
    // "ipa $r0 a[0x80] $r3 0x0 0x1"
    assert_eq!(
        op(0xe043ff880037ff00),
        Op::Ipa {
            dst: 0,
            offset: 0x80,
            mul: Some(3),
            perspective: true,
            sat: false,
            centroid: false
        }
    );
}

#[test]
fn decodes_the_ipa_sample_modes() {
    assert_eq!(
        op(0xe013ff87cff7ff06),
        Op::Ipa {
            dst: 6,
            offset: 0x7c,
            mul: None,
            perspective: false,
            sat: false,
            centroid: true
        }
    );
    // The same instruction with sample mode 2 (offset) and 3.
    assert!(matches!(op(0xe023ff87cff7ff06), Op::Unimplemented { .. }));
    assert!(matches!(op(0xe033ff87cff7ff06), Op::Unimplemented { .. }));
}

/// Bits 54..56 are the interpolation mode.
#[test]
fn decodes_the_ipa_interpolation_modes() {
    let ipa = |raw| match op(raw) {
        Op::Ipa {
            mul, perspective, ..
        } => (mul, perspective),
        other => panic!("not an ipa: {other:?}"),
    };
    assert_eq!(ipa(0xe003ff8800_37ff00), (None, false));
    assert_eq!(ipa(0xe043ff8800_37ff00), (Some(3), true));
    assert_eq!(ipa(0xe083ff8800_37ff00), (None, false));
    assert_eq!(ipa(0xe0c3ff8800_37ff00), (None, false));
    // One of A Short Hike's own, whose multiplier field is already RZ.
    assert_eq!(
        op(0xe083ff890ff7ff00),
        Op::Ipa {
            dst: 0,
            offset: 0x90,
            mul: None,
            perspective: false,
            sat: false,
            centroid: false,
        }
    );
}

mod encodings;
mod wide;
