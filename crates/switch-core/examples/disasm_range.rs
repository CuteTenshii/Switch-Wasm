//! Disassemble a run of instructions from a loaded NRO:
//! `disasm_range <path.nro> <addr> [count]`.
mod common;

use switch_core::disasm::disassemble;
use switch_core::mem::Memory;
use switch_core::nro::load_nro;

const USAGE: &str = "disasm_range <path.nro> <addr> [count]";

fn main() {
    let data = common::read(common::arg(1, USAGE));
    let start = common::hex(&common::arg(2, USAGE));
    let count = common::opt_num(3).unwrap_or(64) as u32;
    let mut mem = Memory::new();
    load_nro(&mut mem, &data).expect("load nro");
    for i in 0..count {
        let at = start.wrapping_add(i * 4);
        let insn = mem.fetch(at).unwrap_or(0);
        println!("{at:#010x}: {insn:#010x} {}", disassemble(insn));
    }
}
