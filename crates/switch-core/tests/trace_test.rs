//! The diagnostic channel: what the emulator says about itself, how much each
//! line matters, and what survives a buffer that has filled.
//!
//! Its own binary, not a section of `cpu_a64_test`. The sink that code with no
//! `Cpu` in reach traces into is process-global, and any test that faults
//! drains it, so a test asserting on what is in the sink has to be the only
//! thing in the process that could take from it.

use switch_core::cpu::Cpu;
use switch_core::trace::{self, Level};

/// The sink is process-global and *taken* rather than copied, so any test that
/// absorbs it (which is every test that raises a diagnostic or faults) can
/// take a line another test was about to assert on. Held for the length of
/// each test below, which makes this file's tests serial among themselves.
static SINK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The lock, surviving a test that failed while holding it: the failure has
/// already been reported, and the next test still needs to run.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    SINK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The trace buffer is a ring, and it drops from the front.
///
/// It used to stop appending once it was full, which threw away exactly the
/// part worth keeping: a fault writes its block, its register dump and its
/// instruction trail *after* everything that led up to them, so a run with
/// tracing on lost its entire crash report and kept half a megabyte of
/// ordinary disassembly.
#[test]
fn a_full_trace_loses_its_oldest_lines_and_keeps_the_fault() {
    let _sink = exclusive();
    let mut cpu = Cpu::new();
    cpu.diagnostic(Level::Info, "[test] the very first thing that happened");
    // Well past the 512 KiB cap, so the front has to go several times over.
    for i in 0..40_000 {
        cpu.diagnostic(
            Level::Info,
            &format!("[test] filler line {i} ................."),
        );
    }

    cpu.mem.map_zero(0x1000, 0x20).unwrap();
    cpu.mem.map(0x1000, &0x0000_0000u32.to_le_bytes()).unwrap();
    cpu.set_pc(0x1000);
    assert!(cpu.run(10).is_err());

    let trace = String::from_utf8_lossy(&cpu.trace).to_string();
    assert!(
        trace.contains("=== FAULT ==="),
        "the fault is what the buffer exists to carry:\n{trace:.600}"
    );
    assert!(
        trace.contains("pc="),
        "and the register dump under it:\n{trace:.600}"
    );
    assert!(
        !trace.contains("the very first thing that happened"),
        "the oldest line is what a full buffer gives up"
    );
    assert!(
        trace.contains("[trace] the buffer filled"),
        "and the loss has to be admitted where it happened:\n{trace:.600}"
    );
    assert!(
        trace.len() <= 512 * 1024 + 64 * 1024,
        "cap not honoured: {}",
        trace.len()
    );
}

/// A diagnostic carries how much it matters, and the lines under it inherit
/// that. Without it a title's fatal abort and a stubbed-out command arrive as
/// the same grey text, and the one that explains the failure has to be found
/// by reading.
#[test]
fn a_diagnostic_carries_its_level_and_the_lines_under_it_inherit_it() {
    let _sink = exclusive();
    let mut cpu = Cpu::new();
    cpu.diagnostic(Level::Warn, "[test] answered with nothing behind it");
    cpu.diagnostic(Level::Error, "[test] gave up");
    let trace = String::from_utf8_lossy(&cpu.trace).to_string();

    let lines: Vec<&str> = trace.lines().collect();
    assert_eq!(lines[0].as_bytes()[0], Level::Warn.marker());
    assert_eq!(lines[1].as_bytes()[0], Level::Error.marker());
    assert!(lines[0][1..].starts_with("[test] answered"));

    // A fault's register dump and instruction trail are written unmarked, so
    // they read as part of the fault rather than reverting to trace text.
    cpu.mem.map_zero(0x1000, 0x20).unwrap();
    cpu.mem.map(0x1000, &0x0000_0000u32.to_le_bytes()).unwrap();
    cpu.set_pc(0x1000);
    assert!(cpu.run(10).is_err());
    let trace = String::from_utf8_lossy(&cpu.trace).to_string();
    let fault = trace.find("=== FAULT ===").expect("a fault was recorded");
    assert_eq!(
        trace.as_bytes()[fault - 1],
        Level::Error.marker(),
        "the fault block heads at error level"
    );
    let after: Vec<&str> = trace[fault..].lines().skip(1).collect();
    assert!(
        after
            .iter()
            .all(|l| l.is_empty() || l.as_bytes()[0] >= b' '),
        "nothing under a fault carries a marker of its own"
    );
}

/// What the parts of the emulator with no `Cpu` in reach trace, the
/// rasterizer, the shader translator, the texture decoder: reaches the same
/// buffer the host drains. Natively they go to stderr; in a browser there is
/// no stderr at all, which is where they used to stop.
#[test]
fn a_trace_from_code_with_no_cpu_still_reaches_the_host() {
    let _sink = exclusive();
    let mut cpu = Cpu::new();
    // Emitted directly rather than through the mask, so this test says nothing
    // about which channels another test left on.
    trace::emit("[test] something the rasterizer said");
    cpu.diagnostic(Level::Info, "[test] and then the cpu said this");

    let trace = String::from_utf8_lossy(&cpu.trace).to_string();
    let said = trace.find("something the rasterizer said");
    let then = trace.find("and then the cpu said this");
    assert!(said.is_some(), "the sink has to be folded in:\n{trace}");
    assert!(said < then, "and in the order it was said:\n{trace}");
}
