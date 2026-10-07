//! Runs blocks from their emitted form on the host. A [`JitHost`] whose `install`
//! returns a Rust function address stands in for the browser's compiler; the fakes
//! do exactly what the block's first `retired` ops do, so runs stay differential
//! against the interpreter.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use switch_core::cpu::{set_jit_host, Cpu, Entry, JitHost, Layout, HOT, LEFT};

const CODE: u32 = 0x1000;

/// A loop whose body is one emittable, memory-free block: three ops and a `RET`.
#[rustfmt::skip]
const LOOP: &[u32] = &[
    0xD282001E, // movz x30, #0x1000
    0xD2824680, // movz x0,  #0x1234
    0xD503201F, // nop
    0xD65F03C0, // ret  x30
];

/// A loop whose back edge is a `CBNZ` as the block's last instruction, always
/// taken, so the `RET` below it must never run.
#[rustfmt::skip]
const BRANCH_LOOP: &[u32] = &[
    0xD282001E, // movz x30, #0x1000
    0xD1000400, // sub  x0, x0, #1
    0xB5FFFFC0, // cbnz x0, #-8
    0xD65F03C0, // ret  x30
];

/// A block after a `B`, so op addresses are not `start + 4 * n`. Assembled with clang.
#[rustfmt::skip]
const JUMP_LOOP: &[u32] = &[
    0xD282001E, // movz x30, #0x1000
    0x14000002, // b    #+8
    0xD2800C60, // movz x0,  #0x63, jumped over
    0xD2824680, // movz x0,  #0x1234
    0xD503201F, // nop
    0xD65F03C0, // ret  x30
];

/// Instructions per trip round [`BRANCH_LOOP`]; the `RET` never runs.
const BRANCH_TRIP: u64 = 3;

/// Instructions per trip round [`LOOP`], including the `RET`.
const TRIP: u64 = 4;
/// Ops in its block, what a fully retired entry reports.
const OPS: u32 = 3;

/// Enough trips to turn [`HOT`] plus a hundred more, ending inside a block.
const STEPS: u64 = (HOT as u64 + 100) * TRIP + 1;

/// What `fake_run` reports retiring.
static RETIRE: AtomicU32 = AtomicU32::new(OPS);
static ENTERED: AtomicU32 = AtomicU32::new(0);
/// Entry points released by dropped blocks.
static RELEASED: AtomicU32 = AtomicU32::new(0);
/// Size of the last module passed to `install`.
static LAST_MODULE: AtomicUsize = AtomicUsize::new(0);

/// Stands in for [`LOOP`]'s block: does its first `retired` ops and reports that many.
extern "C" fn fake_run(state: usize) -> u32 {
    ENTERED.fetch_add(1, Ordering::SeqCst);
    let retired = RETIRE.load(Ordering::SeqCst);
    let regs = state + Layout::of_cpu().regs as usize;
    // SAFETY: `state` is a live `Cpu` passed by the core; the writes are
    // register slots at the layout's offsets.
    unsafe {
        if retired >= 1 {
            *((regs + 8 * 30) as *mut u64) = u64::from(CODE);
        }
        if retired >= 2 {
            *(regs as *mut u64) = 0x1234;
        }
    }
    retired
}

/// Stands in for [`BRANCH_LOOP`]'s block: all three ops, then leaves via the `CBNZ`
/// target with [`LEFT`].
extern "C" fn fake_branch(state: usize) -> u32 {
    ENTERED.fetch_add(1, Ordering::SeqCst);
    let layout = Layout::of_cpu();
    let regs = state + layout.regs as usize;
    // SAFETY: as `fake_run`; `pc` is a field of the same `Cpu`.
    unsafe {
        *((regs + 8 * 30) as *mut u64) = u64::from(CODE);
        let x0 = (regs as *mut u64).read();
        (regs as *mut u64).write(x0.wrapping_sub(1));
        *((state + layout.pc as usize) as *mut u32) = CODE;
    }
    BRANCH_TRIP as u32 | LEFT
}

/// Stands in for [`JUMP_LOOP`]'s block, retiring the first [`RETIRE`] of its four ops.
extern "C" fn fake_jump(state: usize) -> u32 {
    ENTERED.fetch_add(1, Ordering::SeqCst);
    let retired = RETIRE.load(Ordering::SeqCst);
    let regs = state + Layout::of_cpu().regs as usize;
    // SAFETY: as `fake_run`.
    unsafe {
        if retired >= 1 {
            *((regs + 8 * 30) as *mut u64) = u64::from(CODE);
        }
        if retired >= 3 {
            *(regs as *mut u64) = 0x1234;
        }
    }
    retired
}

/// Which fake the next `install` returns.
static EMIT_BRANCH: AtomicBool = AtomicBool::new(false);
static EMIT_JUMP: AtomicBool = AtomicBool::new(false);

fn fake_for_this_block() -> Entry {
    if EMIT_JUMP.load(Ordering::SeqCst) {
        return fake_jump as *const () as Entry;
    }
    match EMIT_BRANCH.load(Ordering::SeqCst) {
        true => fake_branch as *const () as Entry,
        false => fake_run as *const () as Entry,
    }
}

fn install(code: &[u8]) -> Entry {
    LAST_MODULE.store(code.len(), Ordering::SeqCst);
    assert_eq!(
        &code[..4],
        &[0x00, 0x61, 0x73, 0x6D],
        "what reached the host is not a wasm module"
    );
    fake_for_this_block()
}

fn release(entry: Entry) {
    // Either fake: the last block is released after the test body ends.
    assert!(
        [
            fake_run as *const (),
            fake_branch as *const (),
            fake_jump as *const ()
        ]
        .iter()
        .any(|&fake| entry == fake as Entry),
        "an entry point came back that was never handed out"
    );
    RELEASED.fetch_add(1, Ordering::SeqCst);
}

/// The counters and host are process-wide, so tests take turns. Ignore poisoning.
fn exclusive() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    set_jit_host(JitHost { install, release });
    ENTERED.store(0, Ordering::SeqCst);
    RELEASED.store(0, Ordering::SeqCst);
    LAST_MODULE.store(0, Ordering::SeqCst);
    RETIRE.store(OPS, Ordering::SeqCst);
    EMIT_BRANCH.store(false, Ordering::SeqCst);
    EMIT_JUMP.store(false, Ordering::SeqCst);
    guard
}

fn loaded(jit: bool) -> Cpu {
    running(jit, LOOP)
}

fn running(jit: bool, program: &[u32]) -> Cpu {
    let mut cpu = Cpu::new();
    cpu.set_jit_enabled(jit);
    cpu.mem.map_zero(CODE, 0x1000).unwrap();
    let mut bytes = Vec::with_capacity(program.len() * 4);
    for insn in program {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(CODE, &bytes).unwrap();
    cpu.set_pc(CODE);
    cpu
}

fn snapshot(cpu: &Cpu) -> (u64, u64, u32, u64, u64) {
    (
        cpu.read_x(0),
        cpu.read_x(30),
        cpu.get_pc(),
        cpu.cycles,
        cpu.steps,
    )
}

/// Run the loop both ways and require identical state and clock. `steps` ends
/// inside a block to check the budget.
fn compare(steps: u64, what: &str) {
    compare_running(LOOP, steps, what);
}

fn compare_running(program: &[u32], steps: u64, what: &str) {
    let mut interpreted = running(false, program);
    let mut translated = running(true, program);
    let a = interpreted.run(steps).unwrap();
    let b = translated.run(steps).unwrap();
    assert_eq!(a.steps, steps, "{what}: the interpreter missed the budget");
    assert_eq!(b.steps, steps, "{what}: the emitted run missed the budget");
    assert_eq!(a, b, "{what}: the two runs report differently");
    assert_eq!(
        snapshot(&interpreted),
        snapshot(&translated),
        "{what}: (x0, x30, pc, cycles, steps) differ"
    );
}

/// A hot block runs its emitted form with the same results.
#[test]
fn a_hot_block_is_emitted_and_then_run_from_its_emitted_form() {
    let _guard = exclusive();
    compare(STEPS, "a fully retiring block");

    // A fresh run, since the counters are process-wide.
    ENTERED.store(0, Ordering::SeqCst);
    let mut cpu = loaded(true);
    cpu.run(STEPS).unwrap();
    let stats = cpu.jit_stats();
    assert_eq!(
        stats.emitted, 1,
        "the loop's block was not emitted exactly once"
    );
    assert!(
        LAST_MODULE.load(Ordering::SeqCst) > 8,
        "the module handed over was only a header"
    );
    assert_eq!(
        stats.entered_emitted, 100,
        "every entry after the first HOT reaches emitted code, of {}",
        stats.executed
    );
    assert_eq!(
        stats.entered_emitted,
        u64::from(ENTERED.load(Ordering::SeqCst)),
        "the counter and the block disagree about how often it was entered"
    );
}

/// A partially retired block leaves state where those instructions left it,
/// and the interpreter continues from there.
#[test]
fn a_block_that_stops_early_hands_the_rest_back() {
    let _guard = exclusive();
    RETIRE.store(1, Ordering::SeqCst);
    compare(STEPS, "a block that retires one instruction");
    assert!(
        ENTERED.load(Ordering::SeqCst) > 0,
        "the emitted form was never entered, so nothing stopped early"
    );
}

/// A block that retires nothing goes to the interpreter; one that keeps doing
/// it is dropped and releases its entry point.
#[test]
fn a_block_that_retires_nothing_stops_being_entered() {
    let _guard = exclusive();
    RETIRE.store(0, Ordering::SeqCst);
    compare(STEPS, "a block that retires nothing");

    let entered = ENTERED.load(Ordering::SeqCst);
    assert!(entered > 0, "the emitted form was never entered at all");
    assert!(
        entered < 20,
        "{entered} wasted entries: the block was never given up on"
    );
    assert_eq!(
        RELEASED.load(Ordering::SeqCst),
        1,
        "the dropped block did not give its entry point back"
    );
}

/// Dropping the cache releases installed entry points.
#[test]
fn flushing_the_cache_releases_what_was_emitted() {
    let _guard = exclusive();
    let mut cpu = loaded(true);
    cpu.run(STEPS).unwrap();
    assert_eq!(cpu.jit_stats().emitted, 1, "nothing was emitted to release");
    assert_eq!(RELEASED.load(Ordering::SeqCst), 0, "released too early");

    cpu.jit_flush();
    assert_eq!(
        RELEASED.load(Ordering::SeqCst),
        1,
        "the flushed block kept its entry point"
    );
}

/// Self-modifying code drops emitted blocks built from the old bytes.
#[test]
fn overwriting_the_code_releases_its_emitted_form() {
    let _guard = exclusive();
    let mut cpu = loaded(true);
    cpu.run(STEPS).unwrap();
    assert_eq!(
        cpu.jit_stats().emitted,
        1,
        "nothing was emitted to invalidate"
    );

    // Rewrite the `nop`, invalidating the translated page.
    cpu.mem.write_u32(CODE + 8, 0xD2800021).unwrap(); // movz x1, #1
    cpu.run(TRIP * 4).unwrap();
    assert_eq!(
        RELEASED.load(Ordering::SeqCst),
        1,
        "the invalidated block kept its entry point"
    );
    assert_eq!(cpu.read_x(1), 1, "the new instruction did not run");
}

/// A taken branch leaves through its target, the terminator doesn't run, and the
/// clock counts what did.
#[test]
fn a_block_left_through_a_taken_branch_skips_its_terminator() {
    let _guard = exclusive();
    EMIT_BRANCH.store(true, Ordering::SeqCst);
    compare_running(BRANCH_LOOP, STEPS, "a block left at its branch");

    ENTERED.store(0, Ordering::SeqCst);
    let mut cpu = running(true, BRANCH_LOOP);
    cpu.run(STEPS).unwrap();

    let stats = cpu.jit_stats();
    assert_eq!(stats.emitted, 1, "the loop's block was not emitted");
    assert_eq!(
        stats.entered_emitted,
        STEPS / BRANCH_TRIP - HOT as u64,
        "every whole trip after the first HOT reaches emitted code, of {}",
        stats.executed
    );
    // One `sub` per three-instruction trip.
    let subs = STEPS / BRANCH_TRIP + u64::from(STEPS % BRANCH_TRIP >= 2);
    assert_eq!(
        cpu.read_x(0),
        0u64.wrapping_sub(subs),
        "the loop did not go round as many times as the budget allows"
    );
}

/// A block after a `B` hands back at the right address wherever it stops.
#[test]
fn a_block_across_a_followed_branch_hands_back_where_it_stopped() {
    const JUMP_TRIP: u64 = 5;
    let _guard = exclusive();
    EMIT_JUMP.store(true, Ordering::SeqCst);
    for retired in [1, 2, 4] {
        RETIRE.store(retired, Ordering::SeqCst);
        ENTERED.store(0, Ordering::SeqCst);
        let steps = (HOT as u64 + 100) * JUMP_TRIP + 1;
        compare_running(
            JUMP_LOOP,
            steps,
            &format!("a block retiring {retired} of 4"),
        );
        assert!(
            ENTERED.load(Ordering::SeqCst) > 0,
            "the emitted form was never entered"
        );
    }
}
