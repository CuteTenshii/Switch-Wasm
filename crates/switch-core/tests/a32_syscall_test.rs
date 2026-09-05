//! The AArch32 syscall ABI.
//!
//! Horizon numbers its syscalls the same in both execution states and passes
//! most arguments in the same low registers, so most of the syscall layer
//! needs no translation at all. What does need it is every argument and result
//! that is 64 bits wide, because AArch32 has no register that holds one: the
//! kernel splits each across a pair, and the pairs are not always adjacent.
//!
//! The expected mappings are Eden's `SvcWrap_*64From32` wrappers in
//! `core/hle/kernel/svc.cpp`, which are generated from the kernel's own
//! definitions.

mod a32;

use a32::{cpu, r, BASE};

use switch_core::cpu::Cpu;

/// Run one syscall with `r0..r4` preloaded.
fn syscall(imm: u32, regs: [u32; 5]) -> Cpu {
    let mut cpu = cpu();
    cpu.bootstrap();
    cpu.set_pc_and_sp(BASE, 0x9000);
    for (i, v) in regs.iter().enumerate() {
        cpu.set_reg(i as u8, u64::from(*v));
    }
    cpu.mem
        .map(BASE, &(0xEF00_0000 | imm).to_le_bytes())
        .unwrap();
    cpu.run(1).unwrap();
    cpu
}

/// `svcGetSystemTick` answers with a 64-bit count and no result code, so in
/// AArch32 the whole of it is the pair `r0:r1`.
#[test]
fn get_system_tick_comes_back_in_a_register_pair() {
    // r1 starts with a sentinel: an implementation that answered in r0 alone
    // would leave it there, and the caller's wrapper would store it as the
    // top half of the count.
    let cpu = syscall(0x1E, [0, 0xDEAD_BEEF, 0, 0, 0]);
    assert_eq!(r(&cpu, 1), 0, "the top half was written, not left stale");
    let ticks = u64::from(r(&cpu, 0)) | (u64::from(r(&cpu, 1)) << 32);
    assert_eq!(u64::from(r(&cpu, 0)), ticks);
}

/// `svcGetThreadId`'s id is 64 bits, so it occupies `r1:r2`, not `r1` alone.
/// A wrapper that stores both halves through its out pointer would otherwise
/// write a stale `r2` into the top of the caller's `u64`.
#[test]
fn a_thread_id_fills_both_halves_of_its_pair() {
    let cpu = syscall(0x25, [0, 0, 0xDEAD_BEEF, 0, 0]);
    assert_eq!(r(&cpu, 0), 0, "Result");
    assert_eq!(r(&cpu, 1), 1, "the id's low half");
    assert_eq!(r(&cpu, 2), 0, "and its top half, which was not left stale");
}

/// The same for `svcGetProcessId`.
#[test]
fn a_process_id_fills_both_halves_of_its_pair() {
    let cpu = syscall(0x24, [0, 0, 0xDEAD_BEEF, 0, 0]);
    assert_eq!(r(&cpu, 0), 0);
    assert_eq!(r(&cpu, 1), 1);
    assert_eq!(r(&cpu, 2), 0);
}

/// `svcGetInfo` takes its sub-value in the *non-adjacent* pair `r0:r3`, the
/// info type stays in r1 and the handle in r2, and answers in `r1:r2`.
#[test]
fn get_info_takes_a_split_subvalue_and_answers_in_a_pair() {
    // InfoType 1 is the priority mask, whose value has bits above 32: a
    // decoder that answered in one register would drop the top half.
    let cpu = syscall(0x29, [0, 1, 0, 0, 0]);
    assert_eq!(r(&cpu, 0), 0, "Result");
    assert_eq!(r(&cpu, 1), 0xF000_0000, "the mask's low half");
    assert_eq!(r(&cpu, 2), 0x0FFF_FFFF, "and its high half");
}

/// InfoType 11 is RandomEntropy, whose sub-value selects the word. It arrives
/// in `r0:r3`, so a reader that took `r3` alone gets a different word.
#[test]
fn get_info_reads_its_subvalue_from_r0_and_r3() {
    let with_zero = syscall(0x29, [0, 11, 0, 0, 0]);
    let with_one = syscall(0x29, [1, 11, 0, 0, 0]);
    assert_ne!(
        (r(&with_zero, 1), r(&with_zero, 2)),
        (r(&with_one, 1), r(&with_one, 2)),
        "the low half of the sub-value has to reach the syscall"
    );
}

/// `svcSleepThread`'s duration is the pair `r0:r1`. Horizon spends the
/// negative values on yield modes, so reading only `r0` turns a -1 yield into
/// a sleep of 4.29 seconds.
#[test]
fn a_sleep_duration_spans_r0_and_r1() {
    // -1 is the "yield with load balancing" mode: all ones in both halves.
    let cpu = syscall(0x0B, [0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0, 0]);
    assert!(!cpu.halted);
    // A yield leaves the thread runnable rather than parking it on a deadline.
    assert!(
        cpu.thread_dump().contains("Runnable") || cpu.thread_count() <= 1,
        "a negative duration is a yield, not a sleep: {}",
        cpu.thread_dump()
    );
}

/// `svcWaitSynchronization`'s timeout is the pair `r0:r3`, while its handle
/// list stays in `r1` and its count in `r2`. Reading the timeout from `r3`
/// alone loses the low half.
#[test]
fn a_wait_timeout_spans_r0_and_r3() {
    // A wait on no handles with a zero timeout returns TimedOut rather than
    // parking. Both halves of the timeout are zero here.
    let cpu = syscall(0x18, [0, 0, 0, 0, 0]);
    assert!(!cpu.halted);
    // The syscall retired rather than rewinding onto itself, which is what a
    // park does.
    assert_eq!(cpu.get_pc(), BASE + 4, "the wait was answered, not parked");
}
