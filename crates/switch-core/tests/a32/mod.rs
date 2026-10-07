//! Shared AArch32 test harness. Encodings come from `llvm-mc -triple=armv7-none-eabi`.
#![allow(dead_code)]

use switch_core::cpu::{Cpu, ExecMode};

pub const BASE: u32 = 0x1000;
pub const SCRATCH: u32 = 0x8000;
/// `svc #0`, reserved by the emulator as a halt trap.
pub const HALT: u32 = 0xEF00_0000;

/// An AArch32 core with program, scratch and stack mapped.
pub fn cpu() -> Cpu {
    let mut cpu = Cpu::new();
    cpu.mem.map_zero(BASE, 0x1000).unwrap();
    cpu.mem.map_zero(SCRATCH, 0x1000).unwrap();
    cpu.set_mode(ExecMode::A32);
    cpu.set_pc_and_sp(BASE, 0x9000);
    cpu
}

/// Assemble `code` at [`BASE`] and run it to the appended halt.
pub fn run(code: &[u32]) -> Cpu {
    let mut cpu = cpu();
    load(&mut cpu, code);
    cpu.run(code.len() as u64 + 1).unwrap();
    cpu
}

/// The same, for a program expected to fault.
pub fn run_failing(code: &[u32]) -> String {
    let mut cpu = cpu();
    load(&mut cpu, code);
    format!("{}", cpu.run(code.len() as u64 + 1).unwrap_err())
}

/// Assemble `code` at [`BASE`] with a halt appended, without running it.
pub fn load(cpu: &mut Cpu, code: &[u32]) {
    let mut bytes = Vec::with_capacity(code.len() * 4 + 4);
    for insn in code.iter().chain(std::iter::once(&HALT)) {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(BASE, &bytes).unwrap();
}

pub fn r(cpu: &Cpu, n: u8) -> u32 {
    cpu.read_x(n) as u32
}
