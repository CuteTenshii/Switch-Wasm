//! The A32 block translator against the interpreter: run the same program
//! both ways and compare all guest-visible state. The corpus is `llvm-mc`
//! (`-triple=armv7-none-eabi -mcpu=cortex-a15`) output, position independent.

mod a32;

use a32::{cpu, BASE, SCRATCH};
use switch_core::cpu::Cpu;

/// Every addressing form, shifter case, condition, block transfer and PC write
/// the translator has an op or terminator for, then a halt.
#[rustfmt::skip]
const CORPUS: &[u32] = &[
    0xe3a00102, 0xe3011234, 0xe34a1bcd, 0xe1b02080, 0xe2a33000, 0xe0914221, 0xe05153e1,
    0xe2716000, 0xe1e071c1, 0xe1a08061, 0xe1b09111, 0xe001a004, 0xe231bcff, 0xe381c001,
    0xe3ccc001, 0xe1510004, 0x12822001, 0x02822002, 0xe3110001, 0xe1310001, 0xe1710004,
    0xe2f13005, 0xe0c33001, 0xe3080000, 0xe5901000, 0xe5b02004, 0xe4903004, 0xe5504001,
    0xe3a01004, 0xe7905001, 0xe5801008, 0xe5c01009, 0xe1d060b2, 0xe15070f2, 0xe15080d3, 0xe1c010bc,
    0xe140a0d8, 0xe1c0a1f0, 0xe1a0900d, 0xe92d400f, 0xe8bd00f0, 0xe89d0003, 0xe59f0008,
    0xe28f1004, 0xe0020190, 0xe16f3f10, 0xe3a04005, 0xe2544001, 0x1afffffd, 0xeb00000a,
    0xe28f5004, 0xe12fff15, 0xe320f000, 0xe3007077, 0xe28f6008, 0xe52d6004, 0xe49df004,
    0xe320f000, 0xe28f6018, 0xe92d0050, 0xe8bd8010, 0xe2888001, 0xe3b02000, 0x02888008,
    0x112fff1e, 0xe12fff1e, 0xe1580008, 0x11a0f000, 0xef000000,
];

/// A conditional branch, link, return and `SVC`, taken and skipped.
const CONDITIONAL: &[u32] = &[
    0xe3a00000, 0xe3500000, 0x02800001, 0x12800064, 0x0a000000, 0xe3a00032, 0x1a000004, 0xe2800002,
    0x1b000002, 0x112fff1e, 0x1f000007, 0x0f000000, 0xe3a0004d,
];

/// Patches its own code and runs it again, so the second pass needs a new translation.
const SELF_MODIFYING: &[u32] = &[
    0xe3a05002, 0xe3a00000, 0xeaffffff, 0xe2800001, 0xe24f100c, 0xe28f2010, 0xe5922000, 0xe5812000,
    0xe2555001, 0x1afffff7, 0xef000000, 0xe280000a,
];

/// Loads from an unmapped address after two instructions that must still retire.
const FAULTING: &[u32] = &[
    0xe3a01601, 0xe3a00003, 0xe2800004, 0xe5912000, 0xe3a00009, 0xef000000,
];

struct State {
    regs: [u64; 15],
    pc: u32,
    nzcv: u32,
    cycles: u64,
    steps: u64,
    halted: bool,
    memory: Vec<u8>,
}

fn snapshot(cpu: &Cpu) -> State {
    State {
        regs: std::array::from_fn(|i| cpu.read_x(i as u8)),
        pc: cpu.get_pc(),
        nzcv: cpu.nzcv(),
        cycles: cpu.cycles,
        steps: cpu.steps,
        halted: cpu.halted,
        memory: cpu.mem.dump(BASE, 0x1000).unwrap(),
    }
}

fn assert_same(interpreted: &State, translated: &State, what: &str) {
    for i in 0..15 {
        assert_eq!(
            interpreted.regs[i], translated.regs[i],
            "{what}: r{i} differs ({:#x} interpreted, {:#x} translated)",
            interpreted.regs[i], translated.regs[i]
        );
    }
    assert_eq!(interpreted.pc, translated.pc, "{what}: pc differs");
    assert_eq!(interpreted.nzcv, translated.nzcv, "{what}: nzcv differs");
    assert_eq!(
        interpreted.cycles, translated.cycles,
        "{what}: cycles differ"
    );
    assert_eq!(interpreted.steps, translated.steps, "{what}: steps differ");
    assert_eq!(
        interpreted.halted, translated.halted,
        "{what}: halt state differs"
    );
    assert!(
        interpreted.memory == translated.memory,
        "{what}: code memory differs"
    );
}

/// An A32 core with `code` at [`BASE`], the data words in scratch, and the translator on or off.
fn loaded(code: &[u32], jit: bool) -> Cpu {
    let mut cpu = cpu();
    cpu.set_jit_enabled(jit);
    cpu.set_pc_and_sp(BASE, u64::from(SCRATCH) + 0x800);
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    cpu.mem.write_bytes(BASE, &bytes).unwrap();
    for (i, word) in [0x1122_3344u32, 0x5566_7788, 0x99aa_bbcc, 0xddee_ff00]
        .into_iter()
        .enumerate()
    {
        cpu.mem.write_u32(SCRATCH + 4 * i as u32, word).unwrap();
    }
    cpu
}

fn scratch(cpu: &Cpu) -> Vec<u8> {
    cpu.mem.dump(SCRATCH, 0x1000).unwrap()
}

fn compare(code: &[u32], steps: u64, what: &str) {
    let mut interpreted = loaded(code, false);
    let mut translated = loaded(code, true);
    let a = interpreted.run(steps).unwrap();
    let b = translated.run(steps).unwrap();
    assert_eq!(a, b, "{what}: run reports differ");
    assert_same(&snapshot(&interpreted), &snapshot(&translated), what);
    assert!(
        scratch(&interpreted) == scratch(&translated),
        "{what}: scratch memory differs"
    );
    assert!(
        translated.jit_stats().translated > 0,
        "{what}: nothing was translated, so the comparison proved nothing"
    );
}

#[test]
fn the_interpreter_halts_on_every_program() {
    // Guards against both runs faulting on the first encoding and still matching.
    for (what, code) in [
        ("corpus", CORPUS),
        ("conditional", CONDITIONAL),
        ("self-modifying", SELF_MODIFYING),
    ] {
        let mut cpu = loaded(code, false);
        let report = cpu.run(1000).unwrap();
        assert!(report.halted, "{what}");
    }
}

#[test]
fn a_translated_run_matches_an_interpreted_one() {
    for (what, code) in [
        ("corpus", CORPUS),
        ("conditional", CONDITIONAL),
        ("self-modifying", SELF_MODIFYING),
    ] {
        compare(code, 1000, what);
    }
}

#[test]
fn a_translated_run_matches_at_every_step_budget() {
    // A budget ending mid-block stops on the next instruction with an exact retired count.
    for steps in 1..=200 {
        compare(CORPUS, steps, &format!("corpus, {steps} steps"));
    }
    for steps in 1..=CONDITIONAL.len() as u64 + 2 {
        compare(CONDITIONAL, steps, &format!("conditional, {steps} steps"));
    }
}

#[test]
fn a_translated_run_matches_when_resumed_repeatedly() {
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
fn a_store_to_translated_code_is_seen_by_the_next_entry() {
    let mut cpu = loaded(SELF_MODIFYING, true);
    cpu.run(1000).unwrap();
    assert_eq!(cpu.read_x(0), 11, "the second pass runs the patched add");
    assert!(cpu.jit_stats().invalidated > 0);
}

#[test]
fn a_fault_counts_the_faulting_instruction_and_matches() {
    let mut interpreted = loaded(FAULTING, false);
    let mut translated = loaded(FAULTING, true);
    let a = interpreted.run(100).unwrap_err().to_string();
    let b = translated.run(100).unwrap_err().to_string();
    assert_eq!(a, b);
    assert_same(&snapshot(&interpreted), &snapshot(&translated), "fault");
    assert_eq!(translated.get_pc(), BASE + 12);
    assert_eq!(translated.read_x(0), 7);
}
