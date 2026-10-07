//! The block translator against the interpreter: run the same program both
//! ways and compare all guest-visible state. The corpus is `llvm-mc` + `ld.lld`
//! output linked at `CODE`.

use switch_core::cpu::Cpu;

const CODE: u32 = 0x1000;
const DATA: u32 = 0x8000;
/// Bytes of `DATA` the corpus can reach.
const DATA_LEN: usize = 0x200;

/// The corpus, assembled at [`CODE`]: runs to the `spin` self-branch, then the literal pool.
#[rustfmt::skip]
const CORPUS: &[u32] = &[
    0xd2824680, 0xf2b579a0, 0xf2e1e1e0, 0x9281e1e1,
    0x52933322, 0x10001163, 0x90000004, 0x91000084,
    0x91048c05, 0xb1400406, 0xd1001c07, 0xeb000028,
    0x8b010c09, 0xcb81140a, 0xab41080b, 0x8b21480c,
    0xcb21c40d, 0x0b01040e, 0xcb0003ef, 0x92781c10,
    0xb200cc11, 0xd2403c12, 0xf2401c13, 0x8a011014,
    0xaa412015, 0xcac10c16, 0x8a210017, 0xaa210018,
    0xea810819, 0xd3443c1a, 0x93483c1b, 0xb37c1c1c,
    0xd37d201d, 0xd37df002, 0xd345fc03, 0x9347fc24,
    0x531e7405, 0x93c14406, 0x93401c07, 0x53003c08,
    0xeb01001f, 0x9a810009, 0x9a81140a, 0xda81b00b,
    0xda81a40c, 0x9a9f97ed, 0x9a80840e, 0xfa430804,
    0xba411002, 0x7a414000, 0x9b01080f, 0x9b018810,
    0x9b017c11, 0x9b01fc12, 0x9b210813, 0x9ba10814,
    0x9b218815, 0x9ba18816, 0x9b417c17, 0x9bc17c18,
    0x1b010819, 0x9ac1081a, 0x9ac10c1b, 0x9ac1201c,
    0x9ac12c1d, 0xdac00002, 0xdac00c03, 0xdac01004,
    0xdac01405, 0x9a010006, 0xfa010007, 0xd503201f,
    0xd5033bbf, 0xd5033fdf, 0xd53bd068, 0xd51bd040,
    0xd53b4209, 0xd2900002, 0xf9000040, 0xb9000841,
    0x39003040, 0x79001c40, 0xf9400043, 0xb9400844,
    0x39403045, 0x39803046, 0x79401c47, 0x79801c48,
    0xb9800849, 0xf8014040, 0xf841404a, 0xf8020c40,
    0xf840844b, 0xd280008c, 0xf86c784d, 0xf82c5840,
    0xf86ce84e, 0x386c684f, 0x382c6840, 0x580005b0,
    0x180005d1, 0x980005b2, 0xd2902002, 0xa9030440,
    0xa9435053, 0x29080440, 0x29485855, 0x69486057,
    0xa9810440, 0xa8c16859, 0xa93e0440, 0xa97e705b,
    0x9e670000, 0x9e670021, 0x9e620002, 0x1e622843,
    0x1e620864, 0x9e78009d, 0x4e081c05, 0x4e083ca2,
    0xd2800143, 0xd2800004, 0x8b030084, 0xf1000463,
    0x54ffffc1, 0xb4000043, 0xd29bd5a5, 0xb5000040,
    0xd297dde5, 0x36000040, 0xd2800026, 0x37100040,
    0xd2800046, 0x94000005, 0x10000087, 0xd63f00e0,
    0x100000a8, 0xd61f0100, 0x91000529, 0xca09014a,
    0xd65f03c0, 0xd2800aab, 0x14000000, 0xd503201f,
    0x89abcdef, 0x01234567, 0x89abcdef, 0x00000000,
];

struct State {
    /// X0..=X30 then SP.
    regs: [u64; 32],
    pc: u32,
    nzcv: u32,
    vregs: [u128; 32],
    cycles: u64,
    halted: bool,
    data: Vec<u8>,
}

fn snapshot(cpu: &Cpu) -> State {
    let mut regs = [0u64; 32];
    for (i, r) in regs.iter_mut().enumerate() {
        *r = cpu.read_x(i as u8);
    }
    let mut vregs = [0u128; 32];
    for (i, v) in vregs.iter_mut().enumerate() {
        *v = cpu.read_vreg(i as u8);
    }
    State {
        regs,
        pc: cpu.get_pc(),
        nzcv: cpu.nzcv(),
        vregs,
        cycles: cpu.cycles,
        halted: cpu.halted,
        data: cpu.mem.dump(DATA, DATA_LEN).unwrap_or_default(),
    }
}

/// Compare two runs field by field, naming what differs.
fn assert_same(interpreted: &State, translated: &State, what: &str) {
    for i in 0..32 {
        let name = if i == 31 {
            String::from("sp")
        } else {
            format!("x{i}")
        };
        assert_eq!(
            interpreted.regs[i], translated.regs[i],
            "{what}: {name} differs ({:#x} interpreted, {:#x} translated)",
            interpreted.regs[i], translated.regs[i]
        );
    }
    for i in 0..32 {
        assert_eq!(
            interpreted.vregs[i], translated.vregs[i],
            "{what}: v{i} differs ({:#x} interpreted, {:#x} translated)",
            interpreted.vregs[i], translated.vregs[i]
        );
    }
    assert_eq!(interpreted.pc, translated.pc, "{what}: pc differs");
    assert_eq!(interpreted.nzcv, translated.nzcv, "{what}: nzcv differs");
    assert_eq!(
        interpreted.cycles, translated.cycles,
        "{what}: cycle count differs"
    );
    assert_eq!(
        interpreted.halted, translated.halted,
        "{what}: halt state differs"
    );
    assert_eq!(
        interpreted.data, translated.data,
        "{what}: guest memory differs"
    );
}

/// A CPU with `code` at [`CODE`], data at [`DATA`], and the translator on or off.
fn loaded(code: &[u32], jit: bool) -> Cpu {
    let mut cpu = Cpu::new();
    cpu.set_jit_enabled(jit);
    cpu.mem.map_zero(CODE, 0x1000).unwrap();
    cpu.mem.map_zero(DATA, 0x1000).unwrap();
    let mut bytes = Vec::with_capacity(code.len() * 4);
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(CODE, &bytes).unwrap();
    cpu.set_pc(CODE);
    cpu
}

fn compare(code: &[u32], steps: u64, what: &str) {
    let mut interpreted = loaded(code, false);
    let mut translated = loaded(code, true);
    let a = interpreted.run(steps).unwrap();
    let b = translated.run(steps).unwrap();
    assert_eq!(a, b, "{what}: run reports differ");
    assert_same(&snapshot(&interpreted), &snapshot(&translated), what);
    assert!(
        translated.jit_stats().translated > 0,
        "{what}: nothing was translated, so the comparison proved nothing"
    );
}

#[test]
fn the_interpreter_runs_the_whole_corpus_without_faulting() {
    // Guards against both runs faulting on the first encoding and still matching.
    let mut cpu = loaded(CORPUS, false);
    let report = cpu.run(400).unwrap();
    assert_eq!(report.steps, 400);
    assert!(!report.halted);
}

#[test]
fn a_translated_run_matches_an_interpreted_one() {
    compare(CORPUS, 400, "corpus");
}

#[test]
fn a_translated_run_matches_at_every_step_budget() {
    // A budget ending mid-block stops on the next instruction with an exact retired count.
    for steps in 1..=CORPUS.len() as u64 + 8 {
        compare(CORPUS, steps, &format!("corpus, {steps} steps"));
    }
}

#[test]
fn a_translated_run_matches_when_resumed_repeatedly() {
    // Blocks are entered, left part-way and re-entered across many calls.
    let mut interpreted = loaded(CORPUS, false);
    let mut translated = loaded(CORPUS, true);
    for chunk in [1u64, 3, 5, 7, 11, 13, 17, 64, 65, 63, 128] {
        interpreted.run(chunk).unwrap();
        translated.run(chunk).unwrap();
        assert_same(
            &snapshot(&interpreted),
            &snapshot(&translated),
            &format!("corpus resumed in {chunk}-step chunks"),
        );
    }
}

#[test]
fn exclusives_and_sign_filled_bitfields_run_as_ops_and_match() {
    // llvm-mc:
    //   movz x2, #0x8000 ; movz x3, #0xdef0 ; movk x3, #0x9abc, lsl #16
    //   ldaxr w8, [x2] ; add w8, w8, #5 ; stlxr w9, w8, [x2]
    //   stlxr w10, w3, [x2]          // monitor already consumed: fails
    //   ldxr x11, [x2] ; stxr w12, x3, [x2]
    //   ldar w13, [x2] ; stlr x13, [x2]
    //   ldaxrb w14, [x2] ; stlxrb w15, w3, [x2] ; ldar x16, [x2]
    //   sbfiz w17, w3, #4, #8 ; sbfiz x18, x3, #12, #8 ; b .
    let code = [
        0xd2900002, 0xd29bde03, 0xf2b35783, 0x885ffc48, 0x11001508, 0x8809fc48, 0x880afc43,
        0xc85f7c4b, 0xc80c7c43, 0x88dffc4d, 0xc89ffc4d, 0x085ffc4e, 0x080ffc43, 0xc8dffc50,
        0x131c1c71, 0x93741c72, 0x14000000,
    ];
    compare(&code, code.len() as u64, "exclusives and bitfields");
    let mut cpu = loaded(&code, true);
    cpu.run(code.len() as u64).unwrap();
    assert_eq!(cpu.read_x(9), 0, "a store against a live monitor succeeds");
    assert_eq!(cpu.read_x(10), 1, "a second store has no monitor left");
    assert_eq!(cpu.read_x(11), 5, "the failed store wrote nothing");
    assert_eq!(cpu.read_x(12), 0);
    assert_eq!(cpu.read_x(13), 0x9abc_def0);
    assert_eq!(cpu.read_x(14), 0xf0);
    assert_eq!(cpu.read_x(15), 0);
    assert_eq!(cpu.read_x(16), 0x9abc_def0);
    // The field is 0xF0, so the sign fills everything above it.
    assert_eq!(cpu.read_x(17), 0xffff_ff00);
    assert_eq!(cpu.read_x(18), 0xffff_ffff_ffff_0000);
    assert_eq!(
        cpu.jit_stats().interpreted,
        0,
        "every instruction here has an op"
    );
}

#[test]
fn a_fault_inside_a_block_reports_what_the_interpreter_would() {
    // LDR x0, [x1] with x1 zero; page zero is unmapped.
    let code = [0xf9400020u32];
    let mut interpreted = loaded(&code, false);
    let mut translated = loaded(&code, true);
    let a = interpreted.run(1).unwrap_err();
    let b = translated.run(1).unwrap_err();
    assert_eq!(a.to_string(), b.to_string(), "fault messages differ");
    assert_eq!(
        interpreted.get_pc(),
        translated.get_pc(),
        "a fault left the pc somewhere else"
    );
    assert_eq!(
        interpreted.get_pc(),
        CODE,
        "the pc should be on the faulting load"
    );
}

#[test]
fn a_host_write_into_translated_code_is_noticed() {
    let mut cpu = loaded(&[0xd2800020, 0x14000000], true); // movz x0, #1 ; b .
    cpu.run(8).unwrap();
    assert_eq!(cpu.read_x(0), 1);
    let before = cpu.jit_stats().translated;

    cpu.mem.write_u32(CODE, 0xd2800040).unwrap(); // movz x0, #2
    cpu.set_pc(CODE);
    cpu.run(8).unwrap();
    assert_eq!(
        cpu.read_x(0),
        2,
        "the block was re-run from the old instruction"
    );
    assert!(
        cpu.jit_stats().translated > before,
        "the patched block was never translated again"
    );
    assert!(cpu.jit_stats().invalidated > 0, "nothing was invalidated");
}

/// Calls a subroutine, overwrites its first instruction, calls it again:
///
/// ```text
///         movz  x6, #0
/// top:    bl    patch
///         cbnz  x6, done
///         movz  x6, #1
///         adr   x1, patch
///         movz  w2, #0x4445
///         movk  w2, #0xd284, lsl #16   // together: movz x5, #0x2222
///         str   w2, [x1]
///         b     top
/// done:   b     done
/// patch:  movz  x5, #0x1111
///         ret
/// ```
#[rustfmt::skip]
const SELF_MODIFYING: &[u32] = &[
    0xd2800006, 0x94000009, 0xb50000e6, 0xd2800026,
    0x100000c1, 0x528888a2, 0x72ba5082, 0xb9000022,
    0x17fffff9, 0x14000000, 0xd2822225, 0xd65f03c0,
];

#[test]
fn guest_code_that_rewrites_itself_runs_the_new_instruction() {
    for jit in [false, true] {
        let mut cpu = loaded(SELF_MODIFYING, jit);
        cpu.run(64).unwrap();
        assert_eq!(
            cpu.read_x(5),
            0x2222,
            "with the translator {}, the patched instruction never took effect",
            if jit { "on" } else { "off" }
        );
    }
    compare(SELF_MODIFYING, 64, "self-modifying code");
}

/// A back edge into the middle of a cached block, then two `svc`s:
///
/// ```text
///         movz  x0, #0
///         movz  x1, #0
/// mid:    add   x1, x1, #7
///         add   x0, x0, #1
///         cmp   x0, #3
///         b.lt  mid
///         svc   #0xb          // svcSleepThread
///         add   x2, x1, #1
///         svc   #0xb
/// spin:   b     spin
/// ```
#[rustfmt::skip]
const REENTRY: &[u32] = &[
    0xd2800000, 0xd2800001, 0x91001c21, 0x91000400, 0xf1000c1f,
    0x54ffffab, 0xd4000161, 0x91000422, 0xd4000161, 0x14000000,
];

#[test]
fn control_landing_inside_a_translated_block_re_enters_it() {
    // The back edge targets the middle of a cached block; nothing may be skipped or re-run.
    compare(REENTRY, 32, "re-entry into a translated block");
    let mut cpu = loaded(REENTRY, true);
    cpu.run(32).unwrap();
    // Three passes of +7; svcSleepThread overwrote x0 with its result.
    assert_eq!(
        cpu.read_x(1),
        21,
        "the mid-block target was entered the wrong number of times"
    );
    assert_eq!(
        cpu.read_x(2),
        22,
        "execution did not continue past the syscall"
    );
    assert_eq!(
        cpu.read_x(0),
        0,
        "the syscall did not leave its result in x0"
    );
}

#[test]
fn a_syscall_terminates_a_block_and_resumes_after_it() {
    // `Term::Svc` retires the `svc` first; both engines must agree on the pc.
    for steps in 1..=REENTRY.len() as u64 + 4 {
        compare(REENTRY, steps, &format!("syscall block, {steps} steps"));
    }
}

#[test]
fn writing_the_zero_register_never_makes_it_read_back() {
    // XZR discards writes in every destination shape and reads as zero:
    //   movz xzr, #0x1234        ; wide move
    //   cmn  x0, #1              ; ADDS immediate, i.e. Rd=31 as ZR not SP
    //   add  xzr, x0, x0         ; shifted register
    //   orr  xzr, x0, x0         ; logical shifted register
    //   madd xzr, x0, x0, x0     ; multiply
    //   ubfx xzr, x0, #4, #8     ; bitfield, which reads Rd as well
    //   csel xzr, x0, x0, eq     ; conditional select
    //   adr  x1, _start          ; a base that is mapped without any setup
    //   ldr  xzr, [x1]           ; a load whose destination is ZR
    //   ldp  xzr, x3, [x1]       ; and half of a pair
    //   movz x2, #0
    //   add  x2, x2, xzr         ; read it back through Rm
    #[rustfmt::skip]
    let code = [
        0xd282469fu32, 0xb100041f, 0x8b00001f, 0xaa00001f,
        0x9b00001f, 0xd3442c1f, 0x9a80001f, 0x10ffff21,
        0xf940003f, 0xa9400c3f, 0xd2800002, 0x8b1f0042,
    ];
    for jit in [false, true] {
        let mut cpu = loaded(&code, jit);
        cpu.run(code.len() as u64).unwrap();
        assert_eq!(
            cpu.read_x(2),
            0,
            "the zero register read back non-zero with the translator {}",
            if jit { "on" } else { "off" }
        );
    }
    compare(&code, code.len() as u64, "writes to the zero register");
}

#[test]
fn a_block_across_two_pages_is_dropped_by_a_store_to_the_second() {
    // A block spanning two pages is invalidated by a store to either.
    let mut cpu = Cpu::new();
    cpu.set_jit_enabled(true);
    cpu.mem.map_zero(0x1000, 0x2000).unwrap();
    let start = 0x1FF8;
    for (i, insn) in [0xd2800020u32, 0xd2800041, 0xd2800062, 0x14000000]
        .iter()
        .enumerate()
    {
        cpu.mem
            .map(start + 4 * i as u32, &insn.to_le_bytes())
            .unwrap();
    }
    cpu.set_pc(start);
    cpu.run(8).unwrap();
    assert_eq!(cpu.read_x(0), 1);
    assert_eq!(cpu.read_x(1), 2);
    assert_eq!(cpu.read_x(2), 3);

    cpu.mem.write_u32(0x2000, 0xd2800122).unwrap(); // movz x2, #9
    cpu.set_pc(start);
    cpu.run(8).unwrap();
    assert_eq!(
        cpu.read_x(2),
        9,
        "the block ran the instruction the second page no longer holds"
    );
    assert!(cpu.jit_stats().invalidated > 0, "nothing was invalidated");
}

/// `B`s are followed within a block; the `BL` still ends one:
///
/// ```text
///         movz  x0, #1
///         b     one
///         movz  x0, #99
///         movz  x0, #98
/// one:    bl    fn
///         add   x0, x0, #1
///         b     two
///         movz  x0, #97
/// fn:     movz  x1, #5
///         ret
/// two:    add   x1, x1, x0
///         b     .
/// ```
#[rustfmt::skip]
const FOLLOWED: &[u32] = &[
    0xd2800020, 0x14000003, 0xd2800c60, 0xd2800c40,
    0x94000004, 0x91000400, 0x14000004, 0xd2800c20,
    0xd28000a1, 0xd65f03c0, 0x8b000021, 0x14000000,
];

#[test]
fn followed_branches_match_the_interpreter_at_every_step_budget() {
    // Budgets ending before, on, and past each followed branch.
    for steps in 1..=16 {
        compare(
            FOLLOWED,
            steps,
            &format!("followed branches, {steps} steps"),
        );
    }
    let mut cpu = loaded(FOLLOWED, true);
    cpu.run(16).unwrap();
    assert_eq!(cpu.read_x(0), 2, "a skipped instruction ran");
    assert_eq!(cpu.read_x(1), 7);
    assert_eq!(cpu.read_x(30), u64::from(CODE + 0x14), "BL did not link");
    // Four blocks; six would mean the `B`s ended blocks.
    assert_eq!(
        cpu.jit_stats().translated,
        4,
        "the branches were not followed"
    );
}

#[test]
fn a_fault_past_a_followed_branch_names_the_right_instruction() {
    // The fault is placed from the branch target, not the instruction index.
    let code = [0x14000002u32, 0xd503201f, 0xf9400020];
    let mut interpreted = loaded(&code, false);
    let mut translated = loaded(&code, true);
    let a = interpreted.run(4).unwrap_err();
    let b = translated.run(4).unwrap_err();
    assert_eq!(a.to_string(), b.to_string(), "fault messages differ");
    assert_eq!(translated.get_pc(), CODE + 8, "the pc is not on the load");
    assert_eq!(
        interpreted.cycles, translated.cycles,
        "retired counts differ"
    );
}

/// Two calls through a PLT stub whose GOT slot at `DATA + 0x10` the test fills:
///
/// ```text
///         bl    stub
///         add   x0, x0, #1
///         bl    stub
///         b     .
///         nop ; nop ; nop ; nop
/// stub:   adrp  x16, DATA
///         ldr   x17, [x16, #0x10]
///         add   x16, x16, #0x10
///         br    x17
/// five:   movz  x1, #5
///         ret
/// seven:  movz  x1, #7
///         ret
/// ```
#[rustfmt::skip]
const THROUGH_PLT: &[u32] = &[
    0x94000008, 0x91000400, 0x94000006, 0x14000000,
    0xd503201f, 0xd503201f, 0xd503201f, 0xd503201f,
    0xf0000030, 0xf9400a11, 0x91004210, 0xd61f0220,
    0xd28000a1, 0xd65f03c0, 0xd28000e1, 0xd65f03c0,
];
const PLT_SLOT: u32 = DATA + 0x10;
const FIVE: u64 = CODE as u64 + 0x30;
const SEVEN: u64 = CODE as u64 + 0x38;

fn bound(code: &[u32], jit: bool, target: u64) -> Cpu {
    let mut cpu = loaded(code, jit);
    cpu.mem.write_u64(PLT_SLOT, target).unwrap();
    cpu
}

#[test]
fn a_call_through_a_plt_stub_matches_the_interpreter_at_every_step_budget() {
    // Budgets ending between the `BL` and the folded stub, inside it, or past it.
    for steps in 1..=20 {
        let mut interpreted = bound(THROUGH_PLT, false, FIVE);
        let mut translated = bound(THROUGH_PLT, true, FIVE);
        let a = interpreted.run(steps).unwrap();
        let b = translated.run(steps).unwrap();
        let what = format!("through a PLT stub, {steps} steps");
        assert_eq!(a, b, "{what}: run reports differ");
        assert_same(&snapshot(&interpreted), &snapshot(&translated), &what);
    }
    let mut cpu = bound(THROUGH_PLT, true, FIVE);
    cpu.run(20).unwrap();
    assert_eq!(cpu.read_x(1), 5);
    assert_eq!(cpu.read_x(16), u64::from(PLT_SLOT));
    assert_eq!(cpu.read_x(17), FIVE);
    // A fifth block would be the stub, unfolded.
    assert_eq!(
        cpu.jit_stats().translated,
        4,
        "the stub was not folded into the call"
    );
}

#[test]
fn a_rebound_plt_slot_is_followed_on_the_next_call() {
    // A rewritten GOT slot redirects the next call.
    let mut cpu = bound(THROUGH_PLT, true, FIVE);
    // `bl`, the stub's four, `movz` and `ret`.
    cpu.run(7).unwrap();
    assert_eq!(cpu.read_x(1), 5);
    assert_eq!(cpu.get_pc(), CODE + 4);
    cpu.mem.write_u64(PLT_SLOT, SEVEN).unwrap();
    cpu.run(13).unwrap();
    assert_eq!(cpu.read_x(1), 7, "the second call went to the old target");
}

#[test]
fn a_plt_slot_that_cannot_be_read_faults_where_the_interpreter_does() {
    // A GOT slot on an unmapped page: the folded call falls back and faults on the stub's `ldr`.
    let mut code = THROUGH_PLT.to_vec();
    code[8] = 0x90000050;
    let mut interpreted = loaded(&code, false);
    let mut translated = loaded(&code, true);
    let a = interpreted.run(20).unwrap_err();
    let b = translated.run(20).unwrap_err();
    assert_eq!(a.to_string(), b.to_string(), "fault messages differ");
    assert_eq!(
        translated.get_pc(),
        CODE + 0x24,
        "the pc is not on the load"
    );
    assert_same(
        &snapshot(&interpreted),
        &snapshot(&translated),
        "an unreadable PLT slot",
    );
}

#[test]
fn a_hot_loop_is_translated_once_and_entered_many_times() {
    // A loop body is decoded once:
    //     movz x0, #1000
    //     back: subs x0, x0, #1
    //     b.ne  back
    //     b     .
    let code = [0xd2807d00u32, 0xf1000400, 0x54ffffe1, 0x14000000];
    let mut cpu = loaded(&code, true);
    cpu.run(4000).unwrap();
    assert_eq!(cpu.read_x(0), 0, "the loop did not run to completion");
    let stats = cpu.jit_stats();
    assert!(
        stats.translated <= 4,
        "a three-instruction loop was translated {} times",
        stats.translated
    );
    assert!(
        stats.executed > 900,
        "the loop body was only entered {} times",
        stats.executed
    );
}

#[test]
fn turning_the_translator_off_drops_what_it_had_cached() {
    let mut cpu = loaded(CORPUS, true);
    cpu.run(200).unwrap();
    assert!(cpu.jit_stats().blocks > 0);
    cpu.set_jit_enabled(false);
    assert!(!cpu.jit_enabled());
    assert_eq!(
        cpu.jit_stats().blocks,
        0,
        "the cache outlived the translator"
    );
}

#[test]
fn tracing_a_run_still_produces_a_line_per_instruction() {
    // Full tracing takes the interpreter path even with the translator on.
    let mut cpu = loaded(CORPUS, true);
    cpu.trace_enabled = true;
    cpu.run(32).unwrap();
    let trace = String::from_utf8_lossy(&cpu.trace);
    assert_eq!(
        trace.lines().count(),
        32,
        "one line per instruction was expected"
    );
}

// Reserved two-source opcodes and invalid CRC widths take the interpreter's error path.
#[test]
fn invalid_two_source_forms_report_the_interpreters_error() {
    for sf in [0u32, 1] {
        for opcode in 0x0C..=0x17 {
            if opcode >= 0x10 && ((opcode & 3) == 3) == (sf == 1) {
                continue;
            }
            let insn = 0x1AC00000 | (sf << 31) | (opcode << 10);
            let mut interpreted = loaded(&[insn], false);
            let mut translated = loaded(&[insn], true);
            let expected = interpreted.run(1).unwrap_err().to_string();
            let actual = translated.run(1).unwrap_err().to_string();
            assert_eq!(actual, expected, "encoding {insn:#010x}");
            assert_same(
                &snapshot(&interpreted),
                &snapshot(&translated),
                "invalid two-source",
            );
        }
    }
}

#[test]
fn register_copies_mask_the_32_bit_form_and_read_the_zero_register() {
    // (sf, Rm, Rd, expected Rd)
    let cases = [
        (0u32, 0u32, 1u32, 0x7654_3210),
        (1, 0, 1, 0xFEDC_BA98_7654_3210),
        (1, 31, 1, 0),
    ];
    for (sf, rm, rd, expected) in cases {
        // ORR Rd, ZR, Rm, which is MOV.
        let insn = 0x2A0003E0 | (sf << 31) | (rm << 16) | rd;
        let mut interpreted = loaded(&[insn], false);
        let mut translated = loaded(&[insn], true);
        for cpu in [&mut interpreted, &mut translated] {
            cpu.set_reg(0, 0xFEDC_BA98_7654_3210);
            cpu.set_reg(1, u64::MAX);
            cpu.run(1).unwrap();
        }
        assert_same(&snapshot(&interpreted), &snapshot(&translated), "mov");
        assert_eq!(translated.read_x(rd as u8), expected, "{insn:#010x}");
    }
}

#[test]
fn fused_updates_match_at_partial_budgets_and_on_resume() {
    // add w3, w3, #1; cmp w3, #720; b.ne back to the add; b .
    let code = [0x11000463, 0x710b407f, 0x54ffffc1, 0x14000000];
    for initial in [718, 719] {
        // Budgets ending after the update, the compare, the branch, and past the fused exit.
        for budget in [1, 2, 3, 8] {
            let mut interpreted = loaded(&code, false);
            let mut translated = loaded(&code, true);
            for cpu in [&mut interpreted, &mut translated] {
                cpu.set_reg(3, initial);
                cpu.run(budget).unwrap();
            }
            assert_same(&snapshot(&interpreted), &snapshot(&translated), "partial");
            interpreted.run(7).unwrap();
            translated.run(7).unwrap();
            assert_same(&snapshot(&interpreted), &snapshot(&translated), "resumed");
        }
    }
}

#[test]
fn simd_loads_and_stores_match_the_interpreter() {
    let code = [
        0xd2900002, // mov x2, #DATA
        0x3d800040, // str q0, [x2]
        0x3dc00041, // ldr q1, [x2]
        0xfd000843, // str d3, [x2, #16]
        0x14000000, // b   .
    ];
    let mut interpreted = loaded(&code, false);
    let mut translated = loaded(&code, true);
    for cpu in [&mut interpreted, &mut translated] {
        cpu.set_vreg(0, 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF);
        cpu.set_vreg(3, 0x0123_4567_89AB_CDEF);
        cpu.run(4).unwrap();
    }
    assert_same(&snapshot(&interpreted), &snapshot(&translated), "simd");
    assert_eq!(translated.read_vreg(1), translated.read_vreg(0));
    assert_eq!(
        translated.mem.read_u64(0x8010).unwrap(),
        0x0123_4567_89AB_CDEF
    );
}
