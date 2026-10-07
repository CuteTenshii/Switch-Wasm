//! Runs a raw AArch64 instruction stream and dumps the vector registers, for
//! comparison against qemu-aarch64 by `tools/difftest.py`.
mod common;

use common::{Flow, Pace};
use switch_core::cpu::Cpu;

const USAGE: &str = "difftest <code.bin> <inputs.bin> <out.bin> <inputs-address-hex>";

fn main() {
    let code = common::read(common::arg(1, USAGE));
    let inputs = common::read(common::arg(2, USAGE));
    let out = common::arg(3, USAGE);
    const CODE: u32 = 0x1000;
    // Must match the test ELF's .data address.
    let inputs_addr = common::hex(&common::arg(4, USAGE));
    const OUTPUT: u32 = 0x2_0000;
    let mut cpu = Cpu::new();
    cpu.mem.map_zero(CODE, code.len() + 16).unwrap();
    cpu.mem.map(CODE, &code).unwrap();
    cpu.mem.map_zero(inputs_addr, inputs.len() + 4096).unwrap();
    cpu.mem.map(inputs_addr, &inputs).unwrap();
    // One dump per instruction under test.
    let capacity = (code.len() / 4 + 1) * 512;
    cpu.mem.map_zero(OUTPUT, capacity).unwrap();
    cpu.set_reg(0, inputs_addr as u64);
    cpu.set_reg(1, OUTPUT as u64);
    cpu.set_pc(CODE);
    cpu.set_reg(9, OUTPUT as u64);
    // The scalar harness uses x26..x28 instead, so seed both.
    cpu.set_reg(26, OUTPUT as u64);
    cpu.set_reg(27, inputs_addr as u64);
    cpu.set_reg(28, OUTPUT as u64);
    // Stepwise so the high-water mark can be read between instructions.
    let mut high_water = OUTPUT as u64;
    let run = common::drive(
        &mut cpu,
        Pace::Instructions,
        code.len() as u64 / 4,
        |cpu, _| {
            high_water = high_water.max(cpu.read_reg(1)).max(cpu.read_reg(28));
            Flow::Continue
        },
    );
    if let Some(fault) = &run.fault {
        println!(
            "FAULT at step {} pc={:#x}: {fault}",
            run.steps,
            cpu.get_pc()
        );
    }
    let written = (high_water as u32)
        .saturating_sub(OUTPUT)
        .min(capacity as u32);
    let dump = cpu.mem.dump(OUTPUT, written as usize).unwrap();
    std::fs::write(&out, &dump).unwrap();
    println!(
        "ran {} instructions, dumped {written} bytes (halted={})",
        run.steps, cpu.halted
    );
}
