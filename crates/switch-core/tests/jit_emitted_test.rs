//! Running a block from its emitted form rather than by walking its ops.
//!
//! The browser compiles emitted wasm and the core calls it; neither half of
//! that exists on the host, so what is under test here is everything in
//! between: when a block is written out, how what it reports is accounted for,
//! what happens when it hands work back, and that the step budget survives it.
//!
//! A [`JitHost`] whose `install` answers the address of an ordinary Rust
//! function stands in for the compiler. That is not a mock of the interface,
//! it is the interface: an entry point is a code address in whatever sense the
//! target has one, a table index under `wasm32` and a function address here,
//! and the core calls it the same way either way.
//!
//! `fake_run` does exactly what the block's first `retired` ops do, so the run
//! is still differential against the interpreter, whatever it reports. That is
//! the property the whole handover rests on: a block that stops early leaves
//! guest state as if only those instructions had run.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use switch_core::cpu::{set_jit_host, Cpu, Entry, JitHost, Layout};

const CODE: u32 = 0x1000;

/// A loop whose body is one block: three ops and a `RET` that comes back
/// here. `x30` is set by the block itself, so the loop needs no set-up and
/// every iteration is the same four instructions.
///
/// Every op in it is one the emitter can write, which is what makes the block
/// emittable at all, and none of them reads memory, so nothing here can hand
/// back for a reason of its own.
#[rustfmt::skip]
const LOOP: &[u32] = &[
    0xD282001E, // movz x30, #0x1000
    0xD2824680, // movz x0,  #0x1234
    0xD503201F, // nop
    0xD65F03C0, // ret  x30
];

/// Instructions in one trip round [`LOOP`], the `RET` included.
const TRIP: u64 = 4;
/// Ops in its block, which is the trip without the terminator, and so what a
/// fully retired entry reports.
const OPS: u32 = 3;

/// What `fake_run` reports, standing for a block that ran all of itself, part of
/// itself, or none of it.
static RETIRE: AtomicU32 = AtomicU32::new(OPS);
/// How many times it has been entered.
static ENTERED: AtomicU32 = AtomicU32::new(0);
/// How many entry points have been given back, which is what says a dropped
/// block released the one it held.
static RELEASED: AtomicU32 = AtomicU32::new(0);
/// The size of the last module handed to `install`, so a test can tell that
/// the core emitted something rather than nothing.
static LAST_MODULE: AtomicUsize = AtomicUsize::new(0);

/// Stands in for [`LOOP`]'s compiled block: does what its first `retired` ops
/// do, by the same offsets emitted code would, and reports that many.
extern "C" fn fake_run(state: usize) -> u32 {
    ENTERED.fetch_add(1, Ordering::SeqCst);
    let retired = RETIRE.load(Ordering::SeqCst);
    let regs = state + Layout::of_cpu().regs as usize;
    // SAFETY: `state` is the address of a live `Cpu`, because that is what the
    // core passes and this is only ever reached from there. The two writes are
    // register-file slots at the layout's own offset, which is the whole of
    // what this block touches.
    unsafe {
        if retired >= 1 {
            *((regs + 8 * 30) as *mut u64) = u64::from(CODE);
        }
        if retired >= 2 {
            *(regs as *mut u64) = 0x1234;
        }
        // The third op is a `nop`, so there is nothing to do for it.
    }
    retired
}

fn install(code: &[u8]) -> Entry {
    LAST_MODULE.store(code.len(), Ordering::SeqCst);
    assert_eq!(
        &code[..4],
        &[0x00, 0x61, 0x73, 0x6D],
        "what reached the host is not a wasm module"
    );
    fake_run as *const () as Entry
}

fn release(entry: Entry) {
    assert_eq!(
        entry, fake_run as *const () as Entry,
        "an entry point came back wrong"
    );
    RELEASED.fetch_add(1, Ordering::SeqCst);
}

/// The counters above are one set for the whole binary and the host is
/// installed once, so the tests take turns. A poisoned lock is a test that
/// already failed; the rest have nothing to gain by failing too.
fn exclusive() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    set_jit_host(JitHost { install, release });
    ENTERED.store(0, Ordering::SeqCst);
    RELEASED.store(0, Ordering::SeqCst);
    LAST_MODULE.store(0, Ordering::SeqCst);
    RETIRE.store(OPS, Ordering::SeqCst);
    guard
}

fn loaded(jit: bool) -> Cpu {
    let mut cpu = Cpu::new();
    cpu.set_jit_enabled(jit);
    cpu.mem.map_zero(CODE, 0x1000).unwrap();
    let mut bytes = Vec::with_capacity(LOOP.len() * 4);
    for insn in LOOP {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(CODE, &bytes).unwrap();
    cpu.set_pc(CODE);
    cpu
}

/// Everything the loop can be observed to have done.
fn snapshot(cpu: &Cpu) -> (u64, u64, u32, u64, u64) {
    (
        cpu.read_x(0),
        cpu.read_x(30),
        cpu.get_pc(),
        cpu.cycles,
        cpu.steps,
    )
}

/// Run the loop both ways and insist the two agree down to the clock. `steps`
/// is deliberately not a multiple of [`TRIP`], so the run ends inside a block
/// and the budget has to survive the emitted path as exactly as it does the
/// interpreted one.
fn compare(steps: u64, what: &str) {
    let mut interpreted = loaded(false);
    let mut translated = loaded(true);
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

/// A block entered enough times is written out, and from then on entering it
/// runs the emitted form: the same computation, the same clock, the same
/// budget.
#[test]
fn a_hot_block_is_emitted_and_then_run_from_its_emitted_form() {
    let _guard = exclusive();
    compare(401, "a fully retiring block");

    // A second run of its own, because the comparison above already drove the
    // loop and the counters below are the binary's, not this machine's.
    ENTERED.store(0, Ordering::SeqCst);
    let mut cpu = loaded(true);
    cpu.run(401).unwrap();
    let stats = cpu.jit_stats();
    assert_eq!(
        stats.emitted, 1,
        "the loop's block was not emitted exactly once"
    );
    assert!(
        LAST_MODULE.load(Ordering::SeqCst) > 8,
        "the module handed over was only a header"
    );
    assert!(
        stats.entered_emitted > 50,
        "only {} of {} entries reached emitted code",
        stats.entered_emitted,
        stats.executed
    );
    assert_eq!(
        stats.entered_emitted,
        u64::from(ENTERED.load(Ordering::SeqCst)),
        "the counter and the block disagree about how often it was entered"
    );
}

/// A block that retires only part of itself leaves guest state where those
/// instructions left it, and the interpreter carries on from the instruction
/// it stopped at. That is what an emitted access does when the page table
/// cannot answer it alone.
#[test]
fn a_block_that_stops_early_hands_the_rest_back() {
    let _guard = exclusive();
    RETIRE.store(1, Ordering::SeqCst);
    compare(401, "a block that retires one instruction");
    assert!(
        ENTERED.load(Ordering::SeqCst) > 0,
        "the emitted form was never entered, so nothing stopped early"
    );
}

/// A block that retires *nothing* has done nothing, so the visit has to go to
/// the interpreter: reporting no progress would leave `run_jit` entering the
/// same block at the same pc for ever. One that keeps doing it is dropped, and
/// gives its entry point back.
#[test]
fn a_block_that_retires_nothing_stops_being_entered() {
    let _guard = exclusive();
    RETIRE.store(0, Ordering::SeqCst);
    compare(401, "a block that retires nothing");

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

/// Dropping the cache drops what was installed with it. Otherwise a table
/// slot would be held for every block a long run ever compiled.
#[test]
fn flushing_the_cache_releases_what_was_emitted() {
    let _guard = exclusive();
    let mut cpu = loaded(true);
    cpu.run(401).unwrap();
    assert_eq!(cpu.jit_stats().emitted, 1, "nothing was emitted to release");
    assert_eq!(RELEASED.load(Ordering::SeqCst), 0, "released too early");

    cpu.jit_flush();
    assert_eq!(
        RELEASED.load(Ordering::SeqCst),
        1,
        "the flushed block kept its entry point"
    );
}

/// Guest code that rewrites itself drops the blocks translated from it, and an
/// emitted block is code that has already been compiled from the old bytes: it
/// has to go with them, or the guest runs instructions it has overwritten.
#[test]
fn overwriting_the_code_releases_its_emitted_form() {
    let _guard = exclusive();
    let mut cpu = loaded(true);
    cpu.run(401).unwrap();
    assert_eq!(
        cpu.jit_stats().emitted,
        1,
        "nothing was emitted to invalidate"
    );

    // Rewrite the `nop` and run on. The store lands on a translated page, so
    // the block goes, and with it the emitted form built from what was there.
    cpu.mem.write_u32(CODE + 8, 0xD2800021).unwrap(); // movz x1, #1
    cpu.run(TRIP * 4).unwrap();
    assert_eq!(
        RELEASED.load(Ordering::SeqCst),
        1,
        "the invalidated block kept its entry point"
    );
    assert_eq!(cpu.read_x(1), 1, "the new instruction did not run");
}
