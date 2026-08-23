//! The block translator against the interpreter.
//!
//! The translator's contract is that a translated run and an interpreted one
//! are the same computation, so almost everything here is differential: run
//! the same program both ways and compare every piece of state a guest can
//! observe — the register file, the flags, the vector registers, the program
//! counter, the retired instruction count and the memory it wrote.
//!
//! The corpus is real assembler output (`llvm-mc` + `ld.lld`, linked at
//! `CODE`), chosen to reach every class of operation the translator has an op
//! for, plus a sample of the ones it deliberately hands back: divides,
//! variable shifts, bit counts, ADC/SBC, the system registers, and scalar
//! floating point.

use switch_core::cpu::Cpu;

const CODE: u32 = 0x1000;
const DATA: u32 = 0x8000;
/// Bytes of `DATA` the corpus can reach, and so what a comparison has to
/// cover.
const DATA_LEN: usize = 0x200;

/// The corpus, assembled at [`CODE`]. Instructions run to `spin`, a
/// self-branch, followed by the literal pool the `LDR <t>, label` forms read.
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

/// Everything a program can observe about itself.
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

/// Compare two runs field by field, naming what differs rather than dumping
/// two opaque structs.
fn assert_same(interpreted: &State, translated: &State, what: &str) {
    for i in 0..32 {
        let name = if i == 31 { String::from("sp") } else { format!("x{i}") };
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
    assert_eq!(interpreted.cycles, translated.cycles, "{what}: cycle count differs");
    assert_eq!(interpreted.halted, translated.halted, "{what}: halt state differs");
    assert_eq!(interpreted.data, translated.data, "{what}: guest memory differs");
}

/// A CPU with `code` mapped at [`CODE`], a data region at [`DATA`], and the
/// translator in the requested state.
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

/// Run `code` both ways for `steps` instructions and compare.
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
    // The differential tests below compare two runs, which would pass just as
    // well if both faulted on the first unimplemented encoding. This is what
    // says the corpus is really being executed.
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
    // A budget that runs out part-way through a block has to leave the CPU
    // exactly where the interpreter would have: on the next instruction, with
    // the retired count matching to the instruction.
    for steps in 1..=CORPUS.len() as u64 + 8 {
        compare(CORPUS, steps, &format!("corpus, {steps} steps"));
    }
}

#[test]
fn a_translated_run_matches_when_resumed_repeatedly() {
    // The frontend runs a frame's worth of instructions per call, so blocks
    // are entered, left part-way through and re-entered constantly.
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
fn a_fault_inside_a_block_reports_what_the_interpreter_would() {
    // LDR x0, [x1] with x1 still zero: page zero is not mapped here.
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
    assert_eq!(interpreted.get_pc(), CODE, "the pc should be on the faulting load");
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
    assert_eq!(cpu.read_x(0), 2, "the block was re-run from the old instruction");
    assert!(
        cpu.jit_stats().translated > before,
        "the patched block was never translated again"
    );
    assert!(cpu.jit_stats().invalidated > 0, "nothing was invalidated");
}

/// A program that calls a subroutine, overwrites its first instruction, and
/// calls it again. Assembled from:
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

/// A loop whose back edge lands in the middle of the block it is already in,
/// then two `svc`s. Assembled from:
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
    // The first block runs from the entry to the `b.lt`, so the back edge
    // targets an address half way through a block that is already cached.
    // Nothing may be skipped and nothing re-run: a syscall that parks a thread
    // rewinds the pc onto its own `svc` for exactly this reason, and the `svc`
    // is never a block entry until it does.
    compare(REENTRY, 32, "re-entry into a translated block");
    let mut cpu = loaded(REENTRY, true);
    cpu.run(32).unwrap();
    // Three passes over the back edge, each adding 7. x0 counted them but the
    // first `svc` overwrote it: svcSleepThread writes its result code into x0.
    assert_eq!(cpu.read_x(1), 21, "the mid-block target was entered the wrong number of times");
    assert_eq!(cpu.read_x(2), 22, "execution did not continue past the syscall");
    assert_eq!(cpu.read_x(0), 0, "the syscall did not leave its result in x0");
}

#[test]
fn a_syscall_terminates_a_block_and_resumes_after_it() {
    // `Term::Svc` retires the instruction before dispatching it, because a
    // syscall that switches threads installs the incoming thread's pc and the
    // outgoing one has to resume after its own `svc`. Both engines have to
    // leave the pc in the same place.
    for steps in 1..=REENTRY.len() as u64 + 4 {
        compare(REENTRY, steps, &format!("syscall block, {steps} steps"));
    }
}

#[test]
fn a_block_stops_at_the_end_of_its_page() {
    // A block that spanned two pages could not be invalidated by one page's
    // worth of dirt, so the translator never lets one. Straight-line code
    // across the boundary still has to run.
    let mut cpu = Cpu::new();
    cpu.set_jit_enabled(true);
    cpu.mem.map_zero(0x1000, 0x2000).unwrap();
    let start = 0x1FF8;
    for (i, insn) in [0xd2800020u32, 0xd2800041, 0xd2800062, 0x14000000]
        .iter()
        .enumerate()
    {
        cpu.mem.map(start + 4 * i as u32, &insn.to_le_bytes()).unwrap();
    }
    cpu.set_pc(start);
    cpu.run(8).unwrap();
    assert_eq!(cpu.read_x(0), 1);
    assert_eq!(cpu.read_x(1), 2);
    assert_eq!(cpu.read_x(2), 3);
    assert!(
        cpu.jit_stats().translated >= 2,
        "the run across the page boundary was translated as one block"
    );
}

#[test]
fn a_hot_loop_is_translated_once_and_entered_many_times() {
    // The whole point: a loop body pays for its decode on the first pass and
    // never again.
    //
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
    assert_eq!(cpu.jit_stats().blocks, 0, "the cache outlived the translator");
}

#[test]
fn tracing_a_run_still_produces_a_line_per_instruction() {
    // Full tracing needs a disassembly of every instruction, which only the
    // interpreter emits, so `run` has to take that path even with the
    // translator enabled.
    let mut cpu = loaded(CORPUS, true);
    cpu.trace_enabled = true;
    cpu.run(32).unwrap();
    let trace = String::from_utf8_lossy(&cpu.trace);
    assert_eq!(trace.lines().count(), 32, "one line per instruction was expected");
}
