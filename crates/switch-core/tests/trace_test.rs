//! The diagnostic channel: levels, the host sink, and buffer overflow.
//!
//! Its own binary because the sink is process-global.

use switch_core::cpu::Cpu;
use switch_core::trace::{self, Level};

/// Serializes tests that take from the process-global sink.
static SINK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The lock, recovered if a previous test panicked while holding it.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    SINK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The trace buffer is a ring that drops from the front.
#[test]
fn a_full_trace_loses_its_oldest_lines_and_keeps_the_fault() {
    let _sink = exclusive();
    let mut cpu = Cpu::new();
    cpu.diagnostic(Level::Info, "[test] the very first thing that happened");
    // Well past the 512 KiB cap.
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

/// A diagnostic carries its level, and the lines under it inherit it.
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

    // Unmarked lines after a fault inherit its level.
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

/// Traces from code with no `Cpu` in reach reach the host buffer.
#[test]
fn a_trace_from_code_with_no_cpu_still_reaches_the_host() {
    let _sink = exclusive();
    let mut cpu = Cpu::new();
    // Emitted directly, independent of the channel mask.
    trace::emit("[test] something the rasterizer said");
    cpu.diagnostic(Level::Info, "[test] and then the cpu said this");

    let trace = String::from_utf8_lossy(&cpu.trace).to_string();
    let said = trace.find("something the rasterizer said");
    let then = trace.find("and then the cpu said this");
    assert!(said.is_some(), "the sink has to be folded in:\n{trace}");
    assert!(said < then, "and in the order it was said:\n{trace}");
}
