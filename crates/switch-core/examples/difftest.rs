//! Run a raw AArch64 instruction stream and dump the vector registers, so the
//! same bytes can be compared against real ARM semantics under qemu-aarch64.
//!
//! `tools/difftest.py` drives this: it assembles a list of instructions, runs
//! them under qemu, runs the identical bytes here, and reports the first
//! register that differs. This is how the TRN1/TRN2 lane mix-up behind
//! hbmenu's JPEG decode was found.
//!
//! Usage: `difftest <code.bin> <inputs.bin> <out.bin> <inputs-address-hex>`
use std::fs;
use switch_core::cpu::Cpu;

fn main() {
    let code = fs::read(std::env::args().nth(1).expect("code.bin")).expect("code");
    let inputs = fs::read(std::env::args().nth(2).expect("inputs.bin")).expect("inputs");
    let out = std::env::args().nth(3).expect("out.bin");
    const CODE: u32 = 0x1000;
    // Must match the test ELF's .data address: the program computes its own
    // scratch pointer with adrp/add against it.
    let inputs_addr = u32::from_str_radix(
        std::env::args().nth(4).expect("inputs address").trim_start_matches("0x"),
        16,
    )
    .expect("hex address");
    const OUTPUT: u32 = 0x2_0000;
    let mut cpu = Cpu::new();
    cpu.mem.map_zero(CODE, code.len() + 16).unwrap();
    cpu.mem.map(CODE, &code).unwrap();
    cpu.mem.map_zero(inputs_addr, inputs.len() + 4096).unwrap();
    cpu.mem.map(inputs_addr, &inputs).unwrap();
    cpu.mem.map_zero(OUTPUT, 512 * 129).unwrap();
    cpu.set_reg(0, inputs_addr as u64);
    cpu.set_reg(1, OUTPUT as u64);
    cpu.set_pc(CODE);
    cpu.set_reg(9, OUTPUT as u64);
    // The scalar harness keeps its pointers in x26..x28 instead, so seed both;
    // whichever the program uses, the other set is simply unread.
    cpu.set_reg(26, OUTPUT as u64);
    cpu.set_reg(27, inputs_addr as u64);
    cpu.set_reg(28, OUTPUT as u64);
    let mut steps = 0;
    let mut high_water = OUTPUT as u64;
    while !cpu.halted && steps < code.len() as u64 / 4 {
        high_water = high_water.max(cpu.read_reg(1)).max(cpu.read_reg(28));
        let pc = cpu.get_pc();
        if let Err(e) = cpu.step() {
            println!("FAULT at {pc:#x} step {steps}: {e}");
            break;
        }
        steps += 1;
    }
    let written = (high_water as u32).saturating_sub(OUTPUT).min(512 * 128);
    println!("dumped {written} bytes after {steps} instructions (halted={})", cpu.halted);
    let dump = cpu.mem.dump(OUTPUT, written as usize).unwrap();
    fs::write(&out, &dump).unwrap();
    println!("ran {steps} instructions, dumped {written} bytes");
}
