//! The AArch32 execution state itself: the register-file layout the mode
//! switch installs, control flow between subroutines, CP15's thread pointer,
//! and the Thumb boundary this core deliberately refuses to cross.

mod a32;

use a32::{cpu, r, run, run_failing, BASE, HALT};

use switch_core::cpu::{Cpu, ExecMode};

/// A64 keeps the stack pointer in a slot of its own; AArch32 keeps it in
/// `r13`. The switch has to move it, or the first push writes to zero.
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

/// `bx lr` sits in the miscellaneous group, which bit 7 selects — not bit 4.
/// Reading bit 4 sends every return to the halfword multiplier, which falls
/// through into whatever follows it.
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

/// A branch to an odd address would switch to Thumb, which is not
/// implemented; it has to say so where it happens rather than decode ARM
/// words at a Thumb address and fault somewhere else entirely.
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

/// The thread pointer is CP15 c13, and AArch32 keeps the same two registers
/// apart that A64 does: `TPIDRURO` is the kernel's, `TPIDRURW` the guest's.
/// Aliasing them lets a guest's own write stomp the IPC buffer pointer.
#[test]
fn the_thread_pointer_comes_from_cp15_c13() {
    let mut cpu = cpu();
    cpu.bootstrap();
    cpu.set_pc_and_sp(BASE, 0x9000);
    let code: [u32; 5] = [
        0xEE1D_0F70, // mrc p15, 0, r0, c13, c0, 3   (TPIDRURO)
        0xE3A0_10FF, // mov r1, #0xff
        0xEE0D_1F50, // mcr p15, 0, r1, c13, c0, 2   (TPIDRURW)
        0xEE1D_2F50, // mrc p15, 0, r2, c13, c0, 2
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
        r(&cpu, 0),
        switch_core::cpu::MAIN_THREAD_TLS_BASE,
        "writing the guest's own did not move the kernel's"
    );
}

/// The barriers and preload hints are architectural no-ops here, but they are
/// in the unconditional encoding space, so a decoder that only handles
/// `cond != 0xF` faults on them.
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

/// `svc` retires before it dispatches, so a syscall that switches threads
/// leaves the outgoing one resuming after its own `svc` rather than on it.
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

/// An A32 fault trace has to be annotated by the A32 decoder; the A64 one
/// names entirely different instructions for the same words.
#[test]
fn a_fault_names_a32_mnemonics() {
    let mut cpu = cpu();
    cpu.trace_enabled = true;
    let code: [u32; 2] = [
        0xE3A0_0001, // mov r0, #1
        0xE7F0_00F0, // udf — an encoding nothing here claims
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
    // And the register dump names the state's own registers rather than
    // sixteen it does not have.
    assert!(trace.contains("lr ="), "{trace}");
    assert!(!trace.contains("x30"), "{trace}");
}
