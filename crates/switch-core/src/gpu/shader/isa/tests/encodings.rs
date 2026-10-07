use super::*;

/// `f2f` between the two float widths.
#[test]
fn decodes_f2f_between_half_and_single() {
    // Two of the title's own: `f2f.f16.f16.floor` off H0 and off H1.
    assert_eq!(
        op(0x5ca8048000370503),
        Op::F2f {
            dst: 3,
            src: Operand::Reg(3),
            sm: FMod::NONE,
            round: Some(FRound::Floor),
            sat: false,
            ftz: false,
            src_bits: 16,
            dst_bits: 16,
            hi: false,
        }
    );
    assert!(matches!(
        op(0x5ca8068000370500),
        Op::F2f {
            round: Some(FRound::Floor),
            src_bits: 16,
            dst_bits: 16,
            hi: true,
            ..
        }
    ));
    let widths = |dst: u64, src: u64| {
        let raw = 0x5ca8_0000_0000_0000 | (dst << 8) | (src << 10);
        match op(raw) {
            Op::F2f {
                src_bits,
                dst_bits,
                round,
                ..
            } => (src_bits, dst_bits, round),
            other => panic!("not an f2f: {other:?}"),
        }
    };
    assert_eq!(widths(2, 1), (16, 32, None));
    assert_eq!(widths(1, 2), (32, 16, None));
    assert_eq!(widths(2, 2), (32, 32, None));
    // f64 is not modelled, and neither is a cast that rounds any way but to nearest.
    assert!(matches!(
        op(0x5ca8_0000_0000_0300),
        Op::Unimplemented { .. }
    ));
    assert!(matches!(
        op(0x5ca8_0080_0000_0200),
        Op::Unimplemented { .. }
    ));
}

/// `fset` against a 20-bit float immediate.
#[test]
fn decodes_fset_against_an_immediate() {
    assert_eq!(
        op(0x309303bf00070301),
        Op::Fset {
            dst: 1,
            a: 3,
            am: FMod::NONE,
            b: Operand::Imm(0.5f32.to_bits()),
            bm: FMod::NONE,
            cmp: FCmp::Le,
            bop: BoolOp::And,
            src: Pred {
                reg: 7,
                negate: false
            },
            bf: true,
        }
    );
    // The same instruction in its register and constant-bank forms.
    assert!(matches!(
        op(0x5800_0000_0000_0000),
        Op::Fset {
            b: Operand::Reg(_),
            ..
        }
    ));
    assert!(matches!(
        op(0x4800_0000_0000_0000),
        Op::Fset {
            b: Operand::Const { .. },
            ..
        }
    ));
}

/// The depth-compare encodings, which name a reference register beside their coordinates.
#[test]
fn decodes_the_texs_depth_compare_forms() {
    let texs = |enc: u64| {
        let raw = 0xd000_0000_0000_0000u64
            | (1 << 59)
            | (enc << 53)
            | (0x20 << 36)
            | (8 << 8)
            | (20 << 20)
            | 2;
        match op(raw) {
            Op::Texs {
                coords, dref, dim, ..
            } => (dim, coords, dref),
            other => panic!("expected texs, got {other:?}"),
        }
    };
    // The plain 2D sample it is otherwise identical to.
    assert_eq!(texs(1), (TexDim::T2d, [8, 20, RZ], None));
    // 2d.dc, 2d.lz.dc: the reference is `b`.
    assert_eq!(texs(4), (TexDim::T2d, [8, 9, RZ], Some(20)));
    assert_eq!(texs(6), (TexDim::T2d, [8, 9, RZ], Some(20)));
    // 2d.ll.dc: `b` is the level, so the reference follows it.
    assert_eq!(texs(5), (TexDim::T2d, [8, 9, RZ], Some(21)));
    // array_2d.lz.dc: the layer still comes from `a`.
    assert_eq!(texs(9), (TexDim::T2dArray, [9, 20, 8], Some(21)));

    // A Short Hike's own three, all `2d.lz.dc`.
    for raw in [
        0xd0c200aff0670404u64,
        0xd8c200aff1270001,
        0xd8c200aff1270004,
    ] {
        assert!(
            matches!(op(raw), Op::Texs { dref: Some(_), .. }),
            "{raw:#018x} is a shadow sample"
        );
    }
}

#[test]
fn an_exit_its_flow_test_can_never_satisfy_is_not_an_exit() {
    // `exit` carries a condition-code test beside its predicate, and both have to hold.
    assert_eq!(op(0xe3000000_0007001c), Op::Nop); // FCSM_TR
    assert_eq!(op(0xe3000000_00070000), Op::Nop); // F
    assert_eq!(op(0xe3000000_0007000f), Op::Exit); // T

    // `kil` carries the same field.
    assert_eq!(op(0xe3300000_00070000), Op::Nop);
    assert_eq!(op(0xe3300000_0007000f), Op::Kil);
    assert_eq!(op(0xe2400fffff870000), Op::Nop);
    assert!(matches!(op(0xe2400fffff87000f), Op::Bra { .. }));
}

#[test]
fn decodes_ld_st_b128_attribute_space() {
    // mvp.vert: "ld b128 $r0 a[0x80] 0x0"
    assert_eq!(
        op(0xefd9ff80_0807ff00),
        Op::Ld {
            dst: 0,
            offset: 0x80,
            idx: RZ,
            size: MemSize::B128
        }
    );
    // "st b128 a[0x70] $r0 0x0"
    assert_eq!(
        op(0xeff1ff80_0707ff00),
        Op::St {
            offset: 0x70,
            idx: RZ,
            src: 0,
            size: MemSize::B128
        }
    );
}

#[test]
fn decodes_fmul_constant_bank_and_register_forms() {
    // mvp.vert: "fmul ftz $r4 $r0 c2[0x0]"
    assert_eq!(
        op(0x4c681008_00070004),
        Op::Fmul {
            dst: 4,
            a: 0,
            scale: FmulScale::None,
            b: Operand::Const {
                bank: 2,
                offset: 0x0
            },
            bm: FMod::NONE,
            ftz: true,
            sat: false,
        }
    );
    // mvp.vert: "fmul ftz $r5 $r0 c2[0x4]"
    assert_eq!(
        op(0x4c681008_00170005),
        Op::Fmul {
            dst: 5,
            a: 0,
            scale: FmulScale::None,
            b: Operand::Const {
                bank: 2,
                offset: 0x4
            },
            bm: FMod::NONE,
            ftz: true,
            sat: false,
        }
    );
    // tex.frag: "fmul ftz $r0 $r0 $r5"
    assert_eq!(
        op(0x5c681000_00570000),
        Op::Fmul {
            dst: 0,
            a: 0,
            b: Operand::Reg(5),
            bm: FMod::NONE,
            ftz: true,
            sat: false,
            scale: FmulScale::None,
        }
    );
}

#[test]
fn decodes_fadd_constant_bank_form() {
    assert_eq!(
        op(0x4c58100000c70204),
        Op::Fadd {
            dst: 4,
            a: 2,
            am: FMod::NONE,
            b: Operand::Const {
                bank: 0,
                offset: 0x30
            },
            bm: FMod::NONE,
            ftz: true,
            sat: false,
        }
    );
}

#[test]
fn decodes_mov32i() {
    // Captured from a live JKSV run.
    assert_eq!(
        op(0x0103f8000007f000),
        Op::Mov32i {
            dst: 0,
            imm: 0x3f800000
        }
    );
}

#[test]
fn decodes_ffma_constant_bank_chain() {
    // mvp.vert: "ffma ftz $r4 $r1 c2[0x10] $r4"
    assert_eq!(
        op(0x49a00208_00470104),
        Op::Ffma {
            dst: 4,
            a: 1,
            b: Operand::Const {
                bank: 2,
                offset: 0x10
            },
            bneg: false,
            c: Operand::Reg(4),
            cneg: false,
            ftz: true,
            sat: false,
        }
    );
    // "ffma ftz $r0 $r3 c2[0x30] $r1"
    assert_eq!(
        op(0x49a00088_00c70300),
        Op::Ffma {
            dst: 0,
            a: 3,
            b: Operand::Const {
                bank: 2,
                offset: 0x30
            },
            bneg: false,
            c: Operand::Reg(1),
            cneg: false,
            ftz: true,
            sat: false,
        }
    );
}

#[test]
fn decodes_tex() {
    assert_eq!(
        op(0xc07a0080a0770401),
        Op::Tex {
            dst: 1,
            coords: [4, 5, 6],
            layer: None,
            dref: None,
            offset: Some(7),
            lod: None,
            handle: 8,
            handle_reg: None,
            dim: TexDim::T2d,
            mask: [true, false, false, false],
        }
    );
    assert_eq!(
        op(0xc0f80083a0b70e08),
        Op::Tex {
            dst: 8,
            coords: [14, 15, 16],
            layer: None,
            dref: None,
            offset: Some(11),
            lod: None,
            handle: 8,
            handle_reg: None,
            dim: TexDim::T2d,
            mask: [true, true, true, false],
        }
    );
    // `.LL` takes a level out of the meta register, so everything after it moves along one.
    let ll = 0xc07a0080a0770401 | 3 << 55;
    assert!(matches!(
        op(ll),
        Op::Tex {
            lod: Some(7),
            offset: Some(8),
            ..
        }
    ));
    assert!(matches!(op(ll | 1 << 58), Op::Unimplemented { .. }));
    assert!(matches!(
        op(0xc07a0080a0770401 & !(0xf << 31)),
        Op::Unimplemented { .. }
    ));
    assert!(matches!(
        op(0xc07a0080a0770401 | u64::from(RZ)),
        Op::Unimplemented { .. }
    ));
}

#[test]
fn decodes_an_i2i_cc_and_a_csetp() {
    assert!(matches!(
        decode(0x5ce0800000170aff).op,
        Op::I2i {
            dst: RZ,
            cc: true,
            ..
        }
    ));
    assert_eq!(
        decode(0x50a0038000070d07).op,
        Op::Csetp {
            p0: 0,
            p1: 7,
            test: 13,
            src: Pred::ALWAYS,
            op: BoolOp::And,
        }
    );
}

#[test]
fn decodes_a_txq_and_a_tld4() {
    assert_eq!(
        decode(0xdf48008180470800).op,
        Op::Txq {
            dst: 0,
            lod: 8,
            handle: 8,
            mask: [true, true, false, false],
        }
    );
    assert_eq!(
        decode(0xc83a0086aff70208).op,
        Op::Tld4 {
            dst: 8,
            coords: [2, 3, 4],
            layer: None,
            offset: None,
            handle: 8,
            dim: TexDim::T2d,
            component: 0,
            mask: [true, false, true, true],
        }
    );
    // The green channel, and a shadow gather, which is not decoded.
    let green = 0xc83a0086aff70208u64 | 1 << 56;
    assert!(matches!(decode(green).op, Op::Tld4 { component: 1, .. }));
    let shadow = 0xc83a0086aff70208u64 | 1 << 50;
    assert!(matches!(decode(shadow).op, Op::Unimplemented { .. }));
}

#[test]
fn decodes_a_bindless_tex() {
    // Tomodachi Life's `tex.b`.
    assert_eq!(
        op(0xdeba0007a0270000),
        Op::Tex {
            dst: 0,
            coords: [0, 1, 2],
            layer: None,
            dref: None,
            offset: None,
            lod: None,
            handle: 0,
            handle_reg: Some(2),
            dim: TexDim::T2d,
            mask: [true; 4],
        }
    );
    assert!(matches!(
        op(0xdeba0007a0270000 | 3 << 37),
        Op::Tex {
            handle_reg: Some(2),
            lod: Some(3),
            ..
        }
    ));
    // `.LC` is at bit 40 in this form.
    assert!(matches!(
        op(0xdeba0007a0270000 | 1 << 40),
        Op::Unimplemented { .. }
    ));
}

#[test]
fn decodes_a_cube_array_tex() {
    // Tomodachi Life's, the two a draw fell back on before cube arrays decoded.
    assert_eq!(
        op(0xc03a0087fff70400),
        Op::Tex {
            dst: 0,
            coords: [5, 6, 7],
            layer: Some(4),
            dref: None,
            offset: None,
            lod: None,
            handle: 8,
            handle_reg: None,
            dim: TexDim::TCubeArray,
            mask: [true; 4],
        }
    );
    assert!(matches!(
        op(0xc1ba0087f0970400),
        Op::Tex {
            layer: Some(4),
            lod: Some(9),
            dim: TexDim::TCubeArray,
            ..
        }
    ));
}

#[test]
fn decodes_vmnmx_only_where_every_field_is_one_it_models() {
    // Tomodachi Life's `vmnmx r7, r9, r12, r7`.
    const WORD: u64 = 0x3a2c03e060c70907;
    assert_eq!(
        op(WORD),
        Op::Vmnmx {
            dst: 7,
            a: 9,
            b: 12,
            c: 7,
            max: false,
            then_max: false,
            signed: false,
            then_signed: false,
        }
    );
    assert!(matches!(
        op(WORD & !(7 << 51) | 6 << 51 | 1 << 56),
        Op::Vmnmx {
            max: true,
            then_max: true,
            ..
        }
    ));
    for (why, word) in [
        ("an immediate b", WORD & !(1 << 50)),
        ("a byte of a", WORD & !(1 << 38)),
        ("operands of different signedness", WORD | 1 << 48),
        ("saturation", WORD | 1 << 55),
        ("condition codes", WORD | 1 << 47),
        ("an accumulate", WORD & !(7 << 51) | 4 << 51),
    ] {
        assert!(
            matches!(op(word), Op::Unimplemented { .. }),
            "{why} is not modelled"
        );
    }
}

#[test]
fn decodes_texs() {
    // tex.frag.
    assert_eq!(
        op(0xd8301a40_20170000),
        Op::Texs {
            dst: 0,
            dst2: 2,
            coords: [0, 1, RZ],
            dref: None,
            handle: 0x1a4,
            dim: TexDim::T2d,
            mask: [true, true, true, true],
            f16: false,
        }
    );
    assert_eq!(
        texs_destinations(4, 2, [true, true, true, true], false),
        vec![
            (4, TexsStore::Float(0)),
            (5, TexsStore::Float(1)),
            (2, TexsStore::Float(2)),
            (3, TexsStore::Float(3)),
        ]
    );
}

#[test]
fn an_f16_texs_packs_two_channels_into_each_destination() {
    // Bit 59 halves the register count.
    assert_eq!(
        texs_destinations(1, 0, [true, true, true, true], true),
        vec![
            (1, TexsStore::Halves(0, Some(1))),
            (0, TexsStore::Halves(2, Some(3)))
        ]
    );
    // An odd count pads the unused half with zero rather than spilling into another register.
    assert_eq!(
        texs_destinations(4, 6, [true, true, true, false], true),
        vec![
            (4, TexsStore::Halves(0, Some(1))),
            (6, TexsStore::Halves(2, None))
        ]
    );
    assert_eq!(
        texs_destinations(4, RZ, [false, false, false, true], true),
        vec![(4, TexsStore::Halves(3, None))]
    );
}

#[test]
fn the_precision_bit_is_decoded_and_its_polarity_is_backwards() {
    // `Precision` numbers F16 as 0 and F32 as 1, so a set bit is the *unpacked* form.
    assert!(matches!(
        op(0xd8301a40_20170000),
        Op::Texs { f16: false, .. }
    ));
    assert!(matches!(
        op(0xd8301a40_20170000 & !(1 << 59)),
        Op::Texs { f16: true, .. }
    ));
}

#[test]
fn a_one_destination_texs_reads_the_single_and_double_channel_masks() {
    assert_eq!(decode_tex_mask(0, 0, 2), Some([true, true, true, false]));
    assert_eq!(decode_tex_mask(0, 0, RZ), Some([true, false, false, false]));
    assert_eq!(decode_tex_mask(3, 0, RZ), Some([false, false, false, true]));
    assert_eq!(decode_tex_mask(7, 0, RZ), Some([false, false, true, true]));
    // Both destinations, but a selector past the four this decoder knows.
    assert_eq!(decode_tex_mask(5, 0, 2), None);
    // Nowhere to put the result at all.
    assert_eq!(decode_tex_mask(0, RZ, RZ), None);
}

#[test]
fn a_two_channel_texs_fills_only_the_first_destination() {
    // `ga` into $r4: two channels, so $r2 is never touched.
    assert_eq!(
        texs_destinations(4, RZ, [false, true, false, true], false),
        vec![(4, TexsStore::Float(1)), (5, TexsStore::Float(3))]
    );
}

#[test]
fn unrecognised_bits_are_unimplemented_not_a_panic() {
    assert_eq!(op(0), Op::Unimplemented { raw: 0 });
    assert_eq!(op(u64::MAX), Op::Unimplemented { raw: u64::MAX });
}
