//! AArch32 execution state: register layout, subroutine calls, CP15 thread
//! pointer, and the Thumb boundary.

mod a32;

use a32::{cpu, r, run, run_failing, BASE, HALT};

use switch_core::cpu::{Cpu, ExecMode};

/// AArch32 keeps the stack pointer in `r13`, so the mode switch must move it.
#[test]
fn the_mode_switch_moves_the_stack_pointer_and_link_register() {
    let mut cpu = Cpu::new();
    cpu.set_pc_and_sp(0, 0x9000);
    cpu.set_reg(30, 0x1234);
    cpu.set_mode(ExecMode::A32);
    assert_eq!(cpu.sp(), 0x9000);
    assert_eq!(r(&cpu, 13), 0x9000);
    assert_eq!(r(&cpu, 14), 0x1234);
}

#[test]
fn bl_links_and_bx_returns() {
    let cpu = run(&[
        0xEB00_0001, // bl   the callee below
        0xE3A0_2002, // mov  r2, #2      <- returned to
        HALT,
        0xE3A0_1001, // mov  r1, #1      <- the callee
        0xE12F_FF1E, // bx   lr
    ]);
    assert_eq!(r(&cpu, 1), 1);
    assert_eq!(r(&cpu, 2), 2);
}

/// `bx lr` is selected by bit 7, not bit 4.
#[test]
fn bx_is_decoded_as_a_branch_and_not_as_a_multiply() {
    let cpu = run(&[
        0xE3A0_0A01, // mov r0, #0x1000
        0xE280_0014, // add r0, r0, #20
        0xE12F_FF10, // bx  r0
        0xE3A0_2001, // mov r2, #1        (skipped)
        HALT,        //                   (skipped)
        0xE3A0_4004, // mov r4, #4        <- landed on
    ]);
    assert_eq!(r(&cpu, 4), 4);
    assert_eq!(r(&cpu, 2), 0, "the branch skipped this");
}

/// Branching to Thumb is unimplemented and must fault at the branch.
#[test]
fn an_interworking_branch_to_thumb_is_reported_where_it_happens() {
    let err = run_failing(&[
        0xE3A0_0A08, // mov r0, #0x8000
        0xE280_0001, // add r0, r0, #1
        0xE12F_FF10, // bx  r0
    ]);
    assert!(err.contains("T32 is not implemented"), "got {err}");
    assert!(err.contains("0x00008001"), "and says where: {err}");
}

/// `TPIDRURO` and `TPIDRURW` are distinct registers.
#[test]
fn the_thread_pointer_comes_from_cp15_c13() {
    let mut cpu = cpu();
    cpu.bootstrap();
    cpu.set_pc_and_sp(BASE, 0x9000);
    let code: [u32; 6] = [
        0xEE1D_0F70, // mrc p15, 0, r0, c13, c0, 3   (TPIDRURO)
        0xE3A0_10FF, // mov r1, #0xff
        0xEE0D_1F50, // mcr p15, 0, r1, c13, c0, 2   (TPIDRURW)
        0xEE1D_2F50, // mrc p15, 0, r2, c13, c0, 2
        0xEE1D_3F70, // mrc p15, 0, r3, c13, c0, 3   (TPIDRURO)
        HALT,
    ];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(BASE, &bytes).unwrap();
    cpu.run(code.len() as u64).unwrap();
    assert_eq!(
        r(&cpu, 0),
        switch_core::cpu::MAIN_THREAD_TLS_BASE,
        "the read-only thread pointer is the one bootstrap set"
    );
    assert_eq!(r(&cpu, 2), 0xFF, "and the writable one is the guest's own");
    assert_eq!(
        r(&cpu, 3),
        switch_core::cpu::MAIN_THREAD_TLS_BASE,
        "writing the guest's own did not move the kernel's"
    );
}

/// Barriers and preload hints live in the `cond == 0xF` encoding space.
#[test]
fn the_barriers_and_preloads_retire() {
    let cpu = run(&[
        0xF57F_F05F, // dmb sy
        0xF57F_F04F, // dsb sy
        0xF57F_F06F, // isb sy
        0xF5D0_F000, // pld [r0]
        0xE3A0_0001, // mov r0, #1
    ]);
    assert_eq!(r(&cpu, 0), 1);
}

/// `svc` retires before dispatching, so the caller resumes after it.
#[test]
fn a_syscall_retires_before_it_dispatches() {
    let mut cpu = cpu();
    let code: [u32; 2] = [
        0xE3A0_0001, // mov r0, #1
        HALT,        // svc #0
    ];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(BASE, &bytes).unwrap();
    cpu.run(2).unwrap();
    assert!(cpu.halted);
    assert_eq!(cpu.get_pc(), BASE + 8, "past the svc, not on it");
}

/// A32 fault traces use the A32 decoder.
#[test]
fn a_fault_names_a32_mnemonics() {
    let mut cpu = cpu();
    cpu.trace_enabled = true;
    let code: [u32; 2] = [
        0xE3A0_0001, // mov r0, #1
        0xE7F0_00F0, // udf, an encoding nothing here claims
    ];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(BASE, &bytes).unwrap();
    assert!(cpu.run(2).is_err());
    let trace = String::from_utf8_lossy(&cpu.trace);
    assert!(trace.contains("mov r0, #0x1"), "{trace}");
    assert!(
        !trace.contains("movz"),
        "the A64 decoder annotated an A32 trace: {trace}"
    );
    assert!(trace.contains("lr ="), "{trace}");
    assert!(!trace.contains("x30"), "{trace}");
}
