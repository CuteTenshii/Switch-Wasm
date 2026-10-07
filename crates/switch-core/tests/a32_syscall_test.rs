//! The AArch32 syscall ABI: 64-bit arguments and results split across register
//! pairs, per Eden's `SvcWrap_*64From32` wrappers in `core/hle/kernel/svc.cpp`.

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

/// `svcGetSystemTick` answers in `r0:r1`.
#[test]
fn get_system_tick_comes_back_in_a_register_pair() {
    // A sentinel in r1 catches an answer in r0 alone.
    let cpu = syscall(0x1E, [0, 0xDEAD_BEEF, 0, 0, 0]);
    assert_eq!(r(&cpu, 1), 0, "the top half was written, not left stale");
}

/// `svcGetThreadId` answers in `r1:r2`; the pseudo handle is the main thread, id 1.
#[test]
fn a_thread_id_fills_both_halves_of_its_pair() {
    let cpu = syscall(0x25, [0, 0xFFFF_8000, 0xDEAD_BEEF, 0, 0]);
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

/// `svcGetInfo`'s sub-value is the non-adjacent pair `r0:r3`; the answer is `r1:r2`.
#[test]
fn get_info_takes_a_split_subvalue_and_answers_in_a_pair() {
    // InfoType 1 (priority mask) has bits above 32.
    let cpu = syscall(0x29, [0, 1, 0, 0, 0]);
    assert_eq!(r(&cpu, 0), 0, "Result");
    assert_eq!(r(&cpu, 1), 0xF000_0000, "the mask's low half");
    assert_eq!(r(&cpu, 2), 0x0FFF_FFFF, "and its high half");
}

/// InfoType 11 (RandomEntropy) selects its word by the `r0:r3` sub-value.
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

/// `svcSleepThread`'s duration is `r0:r1`; reading `r0` alone turns a -1 yield
/// into a 4.29 s sleep.
#[test]
fn a_sleep_duration_spans_r0_and_r1() {
    // -1: yield with load balancing.
    let cpu = syscall(0x0B, [0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0, 0]);
    assert!(!cpu.halted);
    assert!(
        cpu.cycles < 1_000_000,
        "a negative duration is a yield, not a sleep: cycles = {}",
        cpu.cycles
    );
}

/// `svcWaitSynchronization`'s timeout is `r0:r3`; handles in `r1`, count in `r2`.
#[test]
fn a_wait_timeout_spans_r0_and_r3() {
    // A zero timeout on no handles is a poll: TimedOut, not parked.
    let polled = syscall(0x18, [0, 0, 0, 0, 0]);
    assert_eq!(polled.get_pc(), BASE + 4, "the poll was answered");
    assert_eq!(r(&polled, 0), 0xEA01, "TimedOut");
    // A timeout only in r0 is non-zero, so the wait rewinds onto the svc.
    let waited = syscall(0x18, [1, 0, 0, 0, 0]);
    assert!(!waited.halted);
    assert_eq!(waited.get_pc(), BASE, "the wait was parked, not answered");
}
