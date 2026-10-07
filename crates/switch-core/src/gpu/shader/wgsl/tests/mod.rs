use super::*;
use crate::gpu::shader::isa::{
    BoolOp, FCmp, FMod, FRound, ICmp, LogicOp, LopTest, MufuOp, Op, Operand, Pred, TexDim, RZ,
};
use crate::gpu::shader::isa::{FmulScale, Instruction, MemSize, ShflMode};
use crate::gpu::shader::{next_slot, Program, ENTRY_OFFSET};
use std::collections::BTreeMap;

mod half;
mod module;

const ALWAYS: Pred = Pred::ALWAYS;
/// `@p0`: the guard a two-armed branch is built out of.
const IF_P0: Pred = Pred {
    reg: 0,
    negate: false,
};
const NO_MOD: FMod = FMod::NONE;

/// The byte offset of instruction `index` in a 32-byte-block layout.
fn at(index: usize) -> u32 {
    let mut offset = ENTRY_OFFSET;
    for _ in 0..index {
        offset = next_slot(offset);
    }
    offset
}

fn program(entries: &[(Op, Pred)]) -> Compiled {
    build(entries, BTreeMap::new())
}

fn build(entries: &[(Op, Pred)], indirect: BTreeMap<u32, Vec<u32>>) -> Compiled {
    let mut p = Program {
        indirect,
        ..Program::default()
    };
    for (index, &(op, pred)) in entries.iter().enumerate() {
        p.insns.push(Instruction { pred, op });
        p.offsets.push(at(index));
    }
    Compiled::new(&p)
}

/// Non-finite immediates go through a `let`.
#[test]
fn a_non_finite_immediate_is_converted_at_run_time() {
    let fadd = |bits: u32| Op::Fadd {
        dst: 1,
        a: 2,
        am: NO_MOD,
        b: Operand::Imm(bits),
        bm: NO_MOD,
        ftz: false,
        sat: false,
    };
    let source = |op: Op| {
        translate(&program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]))
            .unwrap()
            .source
    };
    for bits in [0x7f80_0000u32, 0xff80_0000, 0x7fc0_0000] {
        let wgsl = source(fadd(bits));
        assert!(!wgsl.contains(&format!("bitcast<f32>({bits}u)")), "{wgsl}");
        assert!(wgsl.contains(&format!(" = {bits}u;")), "{wgsl}");
    }
    let one = source(fadd(0x3f80_0000));
    assert!(one.contains("bitcast<f32>(1065353216u)"), "{one}");
}

/// Whether braces balance.
fn braces_balance(source: &str) -> bool {
    let mut depth = 0i32;
    for c in source.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        if depth < 0 {
            return false;
        }
    }
    depth == 0
}

/// One of every opcode the Home Menu's shaders use.
fn home_menu_opcodes() -> Vec<Op> {
    vec![
        Op::Ffma {
            dst: 1,
            a: 2,
            b: Operand::Reg(3),
            bneg: false,
            c: Operand::Imm(0),
            cneg: false,
            ftz: true,
            sat: false,
        },
        Op::Fadd {
            dst: 1,
            a: 2,
            am: NO_MOD,
            b: Operand::Reg(3),
            bm: NO_MOD,
            ftz: true,
            sat: false,
        },
        Op::Fmul {
            dst: 1,
            a: 2,
            b: Operand::Reg(3),
            bm: NO_MOD,
            ftz: true,
            sat: false,
            scale: FmulScale::None,
        },
        Op::Mov {
            dst: 1,
            src: Operand::Reg(2),
        },
        Op::Fsetp {
            p0: 0,
            p1: 7,
            a: 1,
            am: NO_MOD,
            b: Operand::Reg(2),
            bm: NO_MOD,
            cmp: FCmp::Lt,
            bop: BoolOp::And,
            src: ALWAYS,
        },
        Op::Isetp {
            p0: 0,
            p1: 7,
            a: 1,
            b: Operand::Imm(3),
            cmp: ICmp::Eq,
            signed: true,
            bop: BoolOp::And,
            src: ALWAYS,
        },
        Op::Mov32i {
            dst: 1,
            imm: 0x3f80_0000,
        },
        Op::Iadd {
            dst: 1,
            a: 2,
            aneg: false,
            b: Operand::Imm(1),
            bneg: false,
            cin: false,
            cout: true,
        },
        Op::Lop {
            dst: 1,
            a: 2,
            ainv: false,
            b: Operand::Imm(0xff),
            binv: false,
            op: LogicOp::And,
            pred: Some((1, LopTest::NonZero)),
        },
        Op::Mufu {
            dst: 1,
            src: 2,
            sm: NO_MOD,
            op: MufuOp::Rcp,
            sat: false,
        },
        Op::Shr {
            dst: 1,
            a: 2,
            b: Operand::Imm(4),
            signed: false,
            wrap: false,
        },
        Op::F2i {
            dst: 1,
            src: Operand::Reg(2),
            sm: NO_MOD,
            dst_bytes: 4,
            dst_signed: true,
            round: FRound::Trunc,
            ftz: true,
        },
        Op::Iscadd {
            dst: 1,
            a: 2,
            aneg: false,
            b: Operand::Reg(3),
            bneg: false,
            shift: 2,
        },
        Op::Iset {
            dst: 1,
            a: 2,
            b: Operand::Imm(3),
            cmp: ICmp::Eq,
            signed: true,
            bop: BoolOp::And,
            src: ALWAYS,
            bf: false,
        },
        Op::Ipa {
            dst: 1,
            offset: 0x80,
            mul: Some(2),
            perspective: true,
            sat: false,
            centroid: false,
        },
        Op::Fmnmx {
            dst: 1,
            a: 2,
            am: NO_MOD,
            b: Operand::Reg(3),
            bm: NO_MOD,
            pred: ALWAYS,
            ftz: true,
        },
        Op::Ldc {
            dst: 1,
            bank: 1,
            offset: 0x14,
            idx: 2,
            size: MemSize::B32,
        },
        Op::St {
            offset: 0x70,
            idx: RZ,
            src: 1,
            size: MemSize::B32,
        },
        Op::I2f {
            dst: 1,
            src: Operand::Reg(2),
            sm: NO_MOD,
            src_bytes: 4,
            src_signed: true,
            sel: 0,
        },
        Op::Shl {
            dst: 1,
            a: 2,
            b: Operand::Imm(2),
            wrap: false,
        },
        Op::Bfi {
            dst: 1,
            insert: 2,
            src: Operand::Reg(3),
            base: Operand::Reg(4),
        },
        Op::Imnmx {
            dst: 1,
            a: 2,
            b: Operand::Imm(4),
            pred: ALWAYS,
            signed: false,
        },
        Op::Fset {
            dst: 1,
            a: 2,
            am: NO_MOD,
            b: Operand::Reg(3),
            bm: NO_MOD,
            cmp: FCmp::Ge,
            bop: BoolOp::And,
            src: ALWAYS,
            bf: true,
        },
        Op::R2p {
            src: 1,
            mask: Operand::Imm(0x7f),
            byte: 0,
        },
        Op::Ld {
            offset: 0x80,
            idx: RZ,
            dst: 1,
            size: MemSize::B32,
        },
        Op::Texs {
            dst: 1,
            dst2: 3,
            coords: [4, 5, RZ],
            dref: None,
            handle: 0x1a4,
            dim: TexDim::T2d,
            mask: [true, true, true, true],
            f16: false,
        },
        Op::Icmp {
            dst: 1,
            a: 2,
            b: Operand::Reg(3),
            c: 4,
            cmp: ICmp::Ne,
            signed: true,
        },
        Op::Iadd3 {
            dst: 1,
            a: 2,
            aneg: false,
            b: Operand::Reg(3),
            bneg: false,
            c: Operand::Reg(4),
            cneg: false,
        },
        Op::Bfe {
            dst: 1,
            a: 2,
            b: Operand::Imm(0x0810),
            signed: false,
        },
    ]
}

#[test]
fn every_opcode_the_home_menu_uses_translates() {
    // Control-flow opcodes are covered by the tests below.
    for op in home_menu_opcodes() {
        let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
        let wgsl = translate(&p)
            .unwrap_or_else(|e| panic!("{op:?}: {e}"))
            .source;
        assert!(braces_balance(&wgsl), "{op:?} left a block open:\n{wgsl}");
    }
}

#[test]
fn a_guard_becomes_a_conditional_rather_than_a_dropped_instruction() {
    let p = program(&[
        (
            Op::Mov {
                dst: 1,
                src: Operand::Imm(7),
            },
            IF_P0,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let wgsl = translate(&p).unwrap().source;
    assert!(wgsl.contains("if (p0) {"), "{wgsl}");
    assert!(wgsl.contains("r1 = 7u;"), "{wgsl}");
}

#[test]
fn a_guarded_branch_says_where_control_goes_when_it_is_not_taken() {
    // Without the `else`, the block would loop forever.
    let p = program(&[
        (Op::Bra { target: at(2) }, IF_P0),
        (Op::Nop, ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    let wgsl = translate(&p).unwrap().source;
    assert!(wgsl.contains("pc = 2u;"), "the taken edge:\n{wgsl}");
    assert!(wgsl.contains("} else {"), "the not-taken edge:\n{wgsl}");
    assert!(wgsl.contains("pc = 1u;"), "which falls through:\n{wgsl}");
}

#[test]
fn reconvergence_becomes_an_explicit_stack() {
    let p = program(&[
        (Op::Ssy { target: at(3) }, ALWAYS),
        (Op::Nop, ALWAYS),
        (Op::Sync, ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    let wgsl = translate(&p).unwrap().source;
    assert!(wgsl.contains("stack[sp] = 3u;"), "the push:\n{wgsl}");
    assert!(wgsl.contains("sp = sp - 1;"), "the pop:\n{wgsl}");
    assert!(
        wgsl.contains("pc = stack[sp];"),
        "and where it goes:\n{wgsl}"
    );
}

#[test]
fn a_brx_becomes_a_switch_over_the_arms_its_table_names() {
    // Byte offsets in, block indices out.
    let arms = vec![at(3), at(4)];
    let mut indirect = BTreeMap::new();
    indirect.insert(at(0), arms.clone());
    let p = build(
        &[
            (Op::Brx { base: 0, reg: 16 }, ALWAYS),
            (Op::Nop, ALWAYS),
            (Op::Nop, ALWAYS),
            (Op::Exit, ALWAYS),
            (Op::Exit, ALWAYS),
        ],
        indirect,
    );
    let wgsl = translate(&p).unwrap().source;
    assert!(wgsl.contains("0u + r16"), "the computed address:\n{wgsl}");
    assert!(
        wgsl.contains("& 31u) == 0u"),
        "rounded onto a slot:\n{wgsl}"
    );
    assert!(
        wgsl.contains(&format!("case {}u: {{ pc = 3u; }}", at(3))),
        "arm 0:\n{wgsl}"
    );
    assert!(
        wgsl.contains(&format!("case {}u: {{ pc = 4u; }}", at(4))),
        "arm 1:\n{wgsl}"
    );
}

#[test]
fn a_brx_with_no_known_arms_is_reported_rather_than_guessed() {
    let p = program(&[(Op::Brx { base: 0, reg: 16 }, ALWAYS), (Op::Exit, ALWAYS)]);
    assert_eq!(
        translate(&p).unwrap_err(),
        Unsupported::IndirectBranch { at: 0 }
    );
}

#[test]
fn global_memory_is_reported_rather_than_mistranslated() {
    // `ldg` without a traceable descriptor is unsupported.
    let op = Op::Ldg {
        dst: 1,
        addr: 2,
        offset: 0,
        size: MemSize::B32,
    };
    let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
    assert_eq!(translate(&p).unwrap_err(), Unsupported::Op { at: 0, op });
}

/// A `ldg` from a constant bank descriptor plus an index.
#[test]
fn a_global_load_through_a_descriptor_binds_the_memory_it_names() {
    let base = |dst, a, offset, cin, cout| Op::Iadd {
        dst,
        a,
        aneg: false,
        b: Operand::Const { bank: 0, offset },
        bneg: false,
        cin,
        cout,
    };
    let p = program(&[
        // r10 = r7 + c0[0x110], carrying out; r11 = RZ + c0[0x114] with it.
        (base(10, 7, 0x110, false, true), ALWAYS),
        (base(11, RZ, 0x114, true, false), ALWAYS),
        (
            Op::Ldg {
                dst: 10,
                addr: 10,
                offset: 0,
                size: MemSize::B32,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    assert_eq!(translated.globals, vec![(0, 0x110)]);
    assert!(
        translated.source.contains("gRead(0u,"),
        "{}",
        translated.source
    );
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(
        source.contains("var<storage, read> g0: array<u32>"),
        "{source}"
    );
    assert!(
        source.contains("case 0u: { return g0[offset >> 2u]; }"),
        "{source}"
    );

    // Any other address is unsupported.
    let op = Op::Ldg {
        dst: 1,
        addr: 2,
        offset: 0,
        size: MemSize::B32,
    };
    let loose = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
    assert_eq!(
        translate(&loose).unwrap_err(),
        Unsupported::Op { at: 0, op }
    );
}

#[test]
fn a_warp_shuffle_is_a_quad_operation_where_the_device_has_them() {
    // `shfl` maps onto quad swaps.
    let op = Op::Shfl {
        dst: 1,
        pred: 0,
        src: 2,
        index: Operand::Imm(1),
        mask: Operand::Imm(0x1c),
        mode: ShflMode::Bfly,
    };
    let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
    let translated = translate_for(
        &p,
        Caps {
            subgroups: true,
            subgroup_enable: true,
        },
    )
    .unwrap();
    assert_eq!(translated.quad, Some(0));
    assert_eq!(translated.quad_swap, Some(0));
    for wanted in ["quadSwapX(", "quadSwapY(", "quadSwapDiagonal("] {
        assert!(
            translated.source.contains(wanted),
            "{wanted} missing from {}",
            translated.source
        );
    }
    // With device quad operations, only the enable and lane are added.
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(source.starts_with("enable subgroups;"), "{source}");
    assert!(!source.contains("dpdxFine"), "{source}");
    assert!(
        source.contains("quad_lane = (u32(input.position.y)"),
        "{source}"
    );
}

#[test]
fn a_warp_shuffle_without_quad_operations_is_a_fine_derivative() {
    // Without them the module defines the swaps from derivatives.
    let op = Op::Shfl {
        dst: 1,
        pred: 0,
        src: 2,
        index: Operand::Imm(1),
        mask: Operand::Imm(0x1c),
        mode: ShflMode::Bfly,
    };
    let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
    let translated = translate(&p).unwrap();
    assert_eq!(translated.quad_swap, Some(0));

    let layout = Layout::of(&translated, Stage::Fragment);
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(
        source.starts_with("diagnostic(off, derivative_uniformity);"),
        "{source}"
    );
    assert!(!source.contains("enable subgroups"), "{source}");
    for wanted in ["fn quadSwapX(", "dpdxFine(", "dpdyFine("] {
        assert!(source.contains(wanted), "{wanted} missing from {source}");
    }

    // Only fragment shaders have derivatives.
    let layout = Layout::of(&translated, Stage::Vertex);
    assert_eq!(
        module(&translated, Stage::Vertex, &layout).unwrap_err(),
        Unsupported::Quad { at: 0 }
    );
}

#[test]
fn fswzadd_asks_which_lane_it_is_and_nothing_of_the_device() {
    // `fswzadd` reads no other lane, so needs no device support.
    let op = Op::Fswzadd {
        dst: 1,
        a: 2,
        b: 3,
        swizzle: 0x99,
        ftz: false,
    };
    let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
    let translated = translate(&p).unwrap();
    assert_eq!(translated.quad, Some(0));
    assert_eq!(translated.quad_swap, None);

    let layout = Layout::of(&translated, Stage::Fragment);
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(source.contains("fn quadLane()"), "{source}");
    assert!(!source.contains("fn quadSwapX("), "{source}");

    // A vertex shader has no lane index.
    let layout = Layout::of(&translated, Stage::Vertex);
    assert_eq!(
        module(&translated, Stage::Vertex, &layout).unwrap_err(),
        Unsupported::Quad { at: 0 }
    );
}

#[test]
fn an_undecoded_branch_target_is_reported_before_anything_is_emitted() {
    let p = program(&[(Op::Bra { target: 0x9999 }, ALWAYS), (Op::Exit, ALWAYS)]);
    assert_eq!(
        translate(&p).unwrap_err(),
        Unsupported::UndecodedTarget { at: 0 }
    );
}

#[test]
fn a_block_starts_at_every_branch_target() {
    // A branch-only target gets its own case.
    let p = program(&[
        (Op::Bra { target: at(2) }, ALWAYS),
        (Op::Nop, ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    let wgsl = translate(&p).unwrap().source;
    for leader in ["case 0u: {", "case 1u: {", "case 2u: {"] {
        assert!(wgsl.contains(leader), "missing {leader}:\n{wgsl}");
    }
}

#[test]
fn only_the_registers_a_program_touches_are_declared() {
    let p = program(&[
        (
            Op::Mov {
                dst: 9,
                src: Operand::Reg(4),
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let wgsl = translate(&p).unwrap().source;
    assert!(wgsl.contains("var<private> r4: u32 = 0u;"), "{wgsl}");
    assert!(wgsl.contains("var<private> r9: u32 = 0u;"), "{wgsl}");
    assert!(
        !wgsl.contains(" r5:"),
        "declared a register nothing uses:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("var carry"),
        "declared a carry nothing sets:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("var stack"),
        "declared a stack nothing pushes:\n{wgsl}"
    );
}

#[test]
fn a_fragment_shaders_colour_is_readable_after_the_call() {
    // Fragment colour is `r0`..`r3`, so registers must outlive `run`.
    let p = program(&[
        (
            Op::Mov {
                dst: 0,
                src: Operand::Imm(0x3f80_0000),
            },
            ALWAYS,
        ),
        (
            Op::Mov {
                dst: 3,
                src: Operand::Imm(0),
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    assert_eq!(translated.registers, vec![0, 3]);
    for reg in &translated.registers {
        assert!(
            translated
                .source
                .contains(&format!("var<private> r{reg}: u32")),
            "r{reg} does not outlive the call:\n{}",
            translated.source
        );
    }
}

#[test]
fn the_zero_register_reads_as_zero_and_discards_what_is_written_to_it() {
    let p = program(&[
        (
            Op::Mov {
                dst: 1,
                src: Operand::Reg(RZ),
            },
            ALWAYS,
        ),
        (
            Op::Mov {
                dst: RZ,
                src: Operand::Reg(2),
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let wgsl = translate(&p).unwrap().source;
    assert!(!wgsl.contains("r255"), "RZ is not a register:\n{wgsl}");
    assert!(wgsl.contains("r1 = 0u;"), "RZ reads as zero:\n{wgsl}");
    assert!(
        !wgsl.contains("= r2;"),
        "a write to RZ is discarded:\n{wgsl}"
    );
}

#[test]
fn the_function_returns_on_every_path() {
    // WGSL requires a final return.
    let p = program(&[(Op::Exit, ALWAYS)]);
    let wgsl = translate(&p).unwrap().source;
    assert!(wgsl.trim_end().ends_with("return false;\n}"), "{wgsl}");
}

#[test]
fn only_the_helpers_a_program_reaches_are_emitted() {
    let p = program(&[
        (
            Op::Shl {
                dst: 1,
                a: 2,
                b: Operand::Imm(2),
                wrap: false,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let wgsl = translate(&p).unwrap().source;
    assert!(wgsl.contains("fn shl32("), "{wgsl}");
    assert!(
        !wgsl.contains("fn lop3("),
        "carried a helper it never calls:\n{wgsl}"
    );
}

#[test]
fn a_helper_never_arrives_without_the_one_it_calls() {
    // `mulhi_s` depends on `mulhi_u`.
    let p = program(&[
        (
            Op::Imul {
                dst: 1,
                a: 2,
                b: Operand::Reg(3),
                signed: true,
                hi: true,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let wgsl = translate(&p).unwrap().source;
    assert!(wgsl.contains("fn mulhi_u("), "{wgsl}");
    assert!(
        wgsl.find("fn mulhi_u(") < wgsl.find("fn mulhi_s("),
        "a helper must be defined before it is called:\n{wgsl}"
    );
}

#[test]
fn the_host_interface_is_what_the_emitted_text_calls() {
    // Every hook the emitter calls must be in `HOST_INTERFACE`.
    let mut wgsl = String::new();
    for op in home_menu_opcodes() {
        let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
        wgsl.push_str(&translate(&p).unwrap().source);
    }
    for hook in ["attrIn(", "attrOut(", "cbRead(", "texSample("] {
        assert!(wgsl.contains(hook), "nothing emits a call to {hook}");
        assert!(
            HOST_INTERFACE.contains(hook),
            "{hook} is not in HOST_INTERFACE"
        );
    }
}
