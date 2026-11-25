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
            cc: false,
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
fn decodes_an_unaligned_128_bit_global_load_as_four_words() {
    // "ldg.e.u.128 r8, [r0 + 0x120]": size 7 in bits 48..51 of the `0xeed0` opcode.
    let ldg = 0xeed7u64 << 48 | 0x120 << 20 | 7 << 16 | 8;
    assert_eq!(
        op(ldg),
        Op::Ldg {
            dst: 8,
            addr: 0,
            offset: 0x120,
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
            fmz: false,
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
            fmz: false,
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
            fmz: false,
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
            fmz: false,
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
            fmz: false,
            sat: false,
        }
    );
}

#[test]
fn decodes_ffma32i_into_its_own_destination() {
    // Taisei's, which envydis reads as "ffma32i ftz $r34 $r30 0x3d707f87 $r34".
    assert_eq!(
        op(0x0c23d707f8771e22),
        Op::Ffma {
            dst: 34,
            a: 30,
            b: Operand::Imm(0x3d707f87),
            bneg: false,
            c: Operand::Reg(34),
            cneg: false,
            ftz: true,
            fmz: false,
            sat: false,
        }
    );
    // envydis: "ffma32i sat $r34 neg $r30 0x3d707f87 neg $r34".
    assert_eq!(
        op(0x0f83d707f8771e22),
        Op::Ffma {
            dst: 34,
            a: 30,
            b: Operand::Imm(0x3d707f87),
            bneg: true,
            c: Operand::Reg(34),
            cneg: true,
            ftz: false,
            fmz: false,
            sat: true,
        }
    );
    // `.cc` has no `Ffma` form; Eden leaves it unimplemented.
    assert!(matches!(op(0x0c13d707f8771e22), Op::Unimplemented { .. }));
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
            handle_reg: None,
            mask: [true, true, false, false],
        }
    );
    assert_eq!(
        decode(0xdf50008180470800).op,
        Op::Txq {
            dst: 0,
            lod: 9,
            handle: 0,
            handle_reg: Some(8),
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
            ptp: None,
            dref: None,
            handle: 8,
            handle_reg: None,
            sparse: 7,
            dim: TexDim::T2d,
            component: 0,
            mask: [true, false, true, true],
        }
    );
    let green = 0xc83a0086aff70208u64 | 1 << 56;
    assert!(matches!(decode(green).op, Op::Tld4 { component: 1, .. }));
}

/// `tld4`'s forms as Eden's `texture_gather.cpp` reads them: `base` gathers a 2D image with r6
/// where the handle, offsets and reference chain starts.
#[test]
fn a_tld4_reads_its_offsets_and_reference_from_the_register_chain() {
    let base = 0xc83a0086aff70208u64 & !(0xff << 20) | 6 << 20;
    let gather = |insn: u64| match decode(insn).op {
        Op::Tld4 {
            coords,
            layer,
            offset,
            ptp,
            dref,
            handle_reg,
            sparse: 7,
            dim,
            ..
        } => (dim, coords, layer, offset, ptp, dref, handle_reg),
        other => panic!("{insn:#x}: {other:?}"),
    };
    let dim = |dim: u64| base & !(0b111 << 28) | dim << 28;
    let offsets = |insn: u64, kind: u64| insn & !(0b11 << 54) | kind << 54;
    let dc = 1 << 50;

    assert_eq!(
        gather(base | dc),
        (TexDim::T2d, [2, 3, 4], None, None, None, Some(6), None)
    );
    assert_eq!(
        gather(offsets(base, 1)),
        (TexDim::T2d, [2, 3, 4], None, Some(6), None, None, None)
    );
    assert_eq!(
        gather(offsets(base, 1) | dc),
        (TexDim::T2d, [2, 3, 4], None, Some(6), None, Some(7), None)
    );
    assert_eq!(
        gather(offsets(base, 2) | dc),
        (
            TexDim::T2d,
            [2, 3, 4],
            None,
            None,
            Some([6, 7]),
            Some(8),
            None
        )
    );
    // An array's layer comes first; a cube's three coordinates are a direction.
    assert_eq!(
        gather(dim(3)),
        (TexDim::T2dArray, [3, 4, 5], Some(2), None, None, None, None)
    );
    assert_eq!(
        gather(dim(6) | dc),
        (TexDim::TCube, [2, 3, 4], None, None, None, Some(6), None)
    );
    assert_eq!(
        gather(dim(7) | dc),
        (
            TexDim::TCubeArray,
            [3, 4, 5],
            Some(2),
            None,
            None,
            Some(6),
            None
        )
    );
    // A cube takes no offset, offset kind 3 is not one, and 1D and 3D have no gather.
    for refused in [
        offsets(dim(6), 1),
        offsets(dim(7), 2),
        offsets(base, 3),
        dim(0),
        dim(1),
        dim(4),
        dim(5),
    ] {
        assert!(
            matches!(decode(refused).op, Op::Unimplemented { .. }),
            "{refused:#x}"
        );
    }
}

/// A gather's sparse predicate is the three bits above its channel mask; `PT` is none.
#[test]
fn a_tld4_names_the_predicate_it_clears_for_a_resident_texel() {
    let base = 0xc83a0086aff70208u64;
    let sparse = |insn: u64| match decode(insn).op {
        Op::Tld4 { sparse, .. } => sparse,
        other => panic!("{insn:#x}: {other:?}"),
    };
    assert_eq!(sparse(base), 7);
    assert_eq!(sparse(base & !(0b111 << 51) | 3 << 51), 3);
}

/// `tld4.b` has the handle in the first register of the chain and its fields lower down.
#[test]
fn a_bindless_tld4_takes_its_handle_from_a_register() {
    let insn = 0xdec0_0000_0000_0000u64
        | 0b111 << 51
        | 0b1101 << 31
        | 2 << 28
        | 6 << 20
        | 2 << 8
        | 8
        | 1 << 36
        | 2 << 38
        | 1 << 50;
    assert_eq!(
        decode(insn).op,
        Op::Tld4 {
            dst: 8,
            coords: [2, 3, 4],
            layer: None,
            offset: Some(7),
            ptp: None,
            dref: Some(8),
            handle: 0,
            handle_reg: Some(6),
            sparse: 7,
            dim: TexDim::T2d,
            component: 2,
            mask: [true, false, true, true],
        }
    );
}

#[test]
fn decodes_a_tld4s_into_texss_registers() {
    // Mario Kart 8 Deluxe's movie shader: the red gather of a bound texture.
    assert_eq!(
        decode(0xdf020080c0a7010e).op,
        Op::Tld4s {
            dst: 14,
            dst2: 12,
            coords: [1, 10],
            offset: None,
            dref: None,
            handle: 8,
            component: 0,
            f16: false,
        }
    );
    // `.AOFFI` takes the second coordinate from the next register and the offset from the other.
    let aoffi = 0xdf020080c0a7010eu64 | 1 << 51;
    assert!(matches!(
        decode(aoffi).op,
        Op::Tld4s {
            coords: [1, 2],
            offset: Some(10),
            ..
        }
    ));
    let green_f16 = 0xdf020080c0a7010eu64 | 1 << 52 | 1 << 55;
    assert!(matches!(
        decode(green_f16).op,
        Op::Tld4s {
            component: 1,
            f16: true,
            ..
        }
    ));
    // `.DC` takes its reference from the other register, or after the offset with `.AOFFI`.
    let shadow = 0xdf020080c0a7010eu64 | 1 << 50;
    assert!(matches!(
        decode(shadow).op,
        Op::Tld4s {
            coords: [1, 2],
            offset: None,
            dref: Some(10),
            ..
        }
    ));
    assert!(matches!(
        decode(shadow | 1 << 51).op,
        Op::Tld4s {
            coords: [1, 2],
            offset: Some(10),
            dref: Some(11),
            ..
        }
    ));
    // Persona 5 Royal's shadow gather.
    assert_eq!(
        decode(0xdf0601e040480202).op,
        Op::Tld4s {
            dst: 2,
            dst2: 4,
            coords: [2, 3],
            offset: None,
            dref: Some(4),
            handle: 30,
            component: 0,
            f16: false,
        }
    );
    assert!(matches!(decode(0xdf48008180470800).op, Op::Txq { .. }));
}

#[test]
fn decodes_a_tmml_asking_only_for_the_level_a_sample_reads() {
    // Breath of the Wild's `tmml.b`, mask 0x2.
    for (insn, dst) in [
        (0xdf60_0001_2037_0403u64, 3),
        (0xdf60_0001_2077_0407, 7),
        (0xdf60_0001_20b7_080b, 11),
    ] {
        assert_eq!(decode(insn).op, Op::Tmml { dst }, "{insn:#x}");
    }
    // The bound form, 0xdf58.
    let bound = 0xdf58_0001_2037_0403u64;
    assert_eq!(decode(bound).op, Op::Tmml { dst: 3 });
    // The raw LOD, alone (0x1) or beside the level (0x3), is not decoded.
    let mask = |insn: u64, bits: u64| insn & !(0xf << 31) | bits << 31;
    for bits in [0b0001, 0b0011] {
        assert!(matches!(
            decode(mask(bound, bits)).op,
            Op::Unimplemented { .. }
        ));
    }
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

#[test]
fn decodes_a_tlds_fetch_into_texss_registers() {
    // Persona 5 Royal: the green channel of the texel at (r4, r13), the plain 2D form.
    let plain = 0xda4600aff0d70404u64;
    assert_eq!(
        decode(plain).op,
        Op::Tlds {
            dst: 4,
            dst2: RZ,
            coords: [4, 13, RZ],
            layer: None,
            lod: None,
            offset: None,
            handle: 10,
            dim: TexDim::T2d,
            mask: [false, true, false, false],
            f16: false,
        }
    );
}

#[test]
fn a_tlds_encoding_picks_where_its_registers_go() {
    // The same word with each encoding of Eden's `texture_load_swizzled.cpp`; a is r4, b is r12.
    let with = |encoding: u64| match decode(0xda4600aff0c70404 & !(0xf << 53) | encoding << 53).op {
        Op::Tlds {
            dim,
            coords,
            layer,
            lod,
            offset,
            ..
        } => (dim, coords, layer, lod, offset),
        other => panic!("encoding {encoding}: {other:?}"),
    };
    assert_eq!(with(0), (TexDim::T2d, [4, RZ, RZ], None, None, None));
    assert_eq!(with(1), (TexDim::T2d, [4, RZ, RZ], None, Some(12), None));
    assert_eq!(with(4), (TexDim::T2d, [4, 5, RZ], None, None, Some(12)));
    assert_eq!(with(5), (TexDim::T2d, [4, 5, RZ], None, Some(12), None));
    assert_eq!(with(7), (TexDim::T3d, [4, 5, 12], None, None, None));
    assert_eq!(
        with(8),
        (TexDim::T2dArray, [12, 13, RZ], Some(4), None, None)
    );
    assert_eq!(
        with(12),
        (TexDim::T2d, [4, 5, RZ], None, Some(12), Some(13))
    );
    // 6 reads a multisample index, and 3 and the rest are not encodings.
    for encoding in [3, 6, 9, 10, 11, 13, 14, 15] {
        assert!(
            matches!(
                decode(0xda4600aff0c70404 & !(0xf << 53) | encoding << 53).op,
                Op::Unimplemented { .. }
            ),
            "encoding {encoding}"
        );
    }
}

#[test]
fn a_tld_takes_its_coordinates_from_its_type_and_its_meta_registers_in_order() {
    // tld r0, r4, r12, handle 10, rgba, the predicate PT: each form only changes type, .LL and .AOFFI.
    const BASE: u64 = 0xdc << 56 | 7 << 51 | 10 << 36 | 0xf << 31 | 12 << 20 | 4 << 8;
    let with = |form: u64| match decode(BASE | form).op {
        Op::Tld {
            dim,
            coords,
            layer,
            lod,
            offset,
            handle,
            handle_reg,
            ..
        } => (dim, coords, layer, lod, offset, handle, handle_reg),
        other => panic!("form {form:#x}: {other:?}"),
    };
    let (lod, aoffi) = (1 << 55, 1 << 35);
    assert_eq!(
        with(0),
        (TexDim::T2d, [4, RZ, RZ], None, None, None, 10, None)
    );
    assert_eq!(
        with(1 << 28 | lod),
        (
            TexDim::T2dArray,
            [5, RZ, RZ],
            Some(4),
            Some(12),
            None,
            10,
            None
        )
    );
    assert_eq!(
        with(2 << 28 | aoffi),
        (TexDim::T2d, [4, 5, RZ], None, None, Some(12), 10, None)
    );
    assert_eq!(
        with(3 << 28 | lod | aoffi),
        (
            TexDim::T2dArray,
            [5, 6, RZ],
            Some(4),
            Some(12),
            Some(13),
            10,
            None
        )
    );
    assert_eq!(
        with(4 << 28),
        (TexDim::T3d, [4, 5, 6], None, None, None, 10, None)
    );
    // The bindless form reads its handle from the first meta register.
    let bindless = BASE & !(0xff << 56) | 0xdd << 56 | 2 << 28 | lod;
    assert!(matches!(
        decode(bindless).op,
        Op::Tld {
            handle_reg: Some(12),
            lod: Some(13),
            sparse: 7,
            ..
        }
    ));
    // A cube, a clamp, a multisample index and an empty mask are not fetched.
    for unsupported in [5 << 28, 1 << 54, 1 << 50] {
        assert!(matches!(
            decode(BASE | unsupported).op,
            Op::Unimplemented { .. }
        ));
    }
    assert!(matches!(
        decode(BASE & !(0xf << 31)).op,
        Op::Unimplemented { .. }
    ));
}

/// Eden throws on a register pair that starts on an odd register, so the decoder refuses it.
#[test]
fn a_tlds_pair_on_an_odd_register_is_unimplemented() {
    const BASE: u64 = 0xda4600aff0c70404;
    let decoded = |encoding: u64, a: u64, b: u64| {
        let insn = BASE & !(0xf << 53) & !(0xff << 8) & !(0xff << 20);
        !matches!(
            decode(insn | encoding << 53 | a << 8 | b << 20).op,
            Op::Unimplemented { .. }
        )
    };
    // reg_a pairs: encodings 4, 5, 7 and 12.
    for encoding in [4, 5, 7, 12] {
        assert!(decoded(encoding, 4, 12), "encoding {encoding}");
        assert!(!decoded(encoding, 5, 12), "encoding {encoding}");
    }
    // 8 pairs reg_b and reads the layer from reg_a, which may be odd.
    assert!(decoded(8, 5, 12));
    assert!(!decoded(8, 4, 13));
    assert!(!decoded(12, 4, 13));
    // The unpaired forms take any register, and RZ counts as aligned.
    for encoding in [0, 1, 2] {
        assert!(decoded(encoding, 5, 13), "encoding {encoding}");
    }
    assert!(decoded(8, 4, RZ.into()));
}

/// Two or more f32 channels span a pair from `dst`, and four a second pair from `dst2`.
#[test]
fn a_tlds_destination_pair_on_an_odd_register_is_unimplemented() {
    const BASE: u64 = 0xda4600aff0c70404 & !(0x7 << 50) & !(0xff << 28) & !0xff & !(1 << 59);
    let decoded = |swizzle: u64, dst: u64, dst2: u64, f16: bool| {
        let precision = if f16 { 0 } else { 1 << 59 };
        let insn = BASE | precision | swizzle << 50 | dst2 << 28 | dst;
        !matches!(decode(insn).op, Op::Unimplemented { .. })
    };
    let rz = u64::from(RZ);
    assert!(decoded(0, 5, rz, false), "one channel takes any register");
    assert!(decoded(4, 4, rz, false), "two channels, red and green");
    assert!(!decoded(4, 5, rz, false));
    assert!(
        decoded(0, 4, 7, false),
        "three channels, the second pair is unused"
    );
    assert!(!decoded(0, 5, 7, false));
    assert!(decoded(4, 4, 6, false), "four channels");
    assert!(!decoded(4, 4, 7, false));
    assert!(decoded(4, 5, 7, true), "halves pack, so there is no pair");
}

/// `.fmz` is mode 2 of the two-bit float mode: at 44 for `fmul`, at 53 for `ffma` and `ffma32i`.
#[test]
fn decodes_fmz_on_fmul_ffma_and_ffma32i() {
    let modes = |word: u64| match op(word) {
        Op::Fmul { ftz, fmz, .. } | Op::Ffma { ftz, fmz, .. } => (ftz, fmz),
        other => panic!("not a float multiply: {other:?}"),
    };
    for (base, at) in [
        (0x5c68_0000_0000_0000u64, 44),
        (0x4c68_0000_0000_0000, 44),
        (0x5980_0000_0000_0000, 53),
        (0x0c00_0000_0000_0000, 53),
    ] {
        assert_eq!(modes(base), (false, false), "{base:#x}");
        assert_eq!(modes(base | 1 << at), (true, false), "{base:#x}");
        assert_eq!(modes(base | 2 << at), (false, true), "{base:#x}");
    }
    assert!(matches!(
        op(0x0c00_0000_0000_0000 | 3 << 53),
        Op::Unimplemented { .. }
    ));
    assert!(matches!(
        op(0x0c00_0000_0000_0000 | 1 << 52),
        Op::Unimplemented { .. }
    ));
}
