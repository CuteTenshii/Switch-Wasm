//! Disassemble the instructions around an address in an NRO: `disasm_around <path.nro> <pc>`.
mod common;

use switch_core::disasm::disassemble;
use switch_core::mem::Memory;
use switch_core::nro::load_nro;

const USAGE: &str = "disasm_around <path.nro> <pc>";
const CONTEXT: u32 = 32;

fn main() {
    let data = common::read(common::arg(1, USAGE));
    let pc = common::hex(&common::arg(2, USAGE));
    let mut mem = Memory::new();
    let loaded = load_nro(&mut mem, &data).expect("load nro");
    println!(
        "base={:#x} text_off={:#x} text_size={:#x}",
        loaded.base, loaded.text.file_offset, loaded.text.file_size
    );
    for at in (pc.wrapping_sub(CONTEXT)..=pc.wrapping_add(CONTEXT)).step_by(4) {
        let insn = mem.fetch(at).unwrap_or(0);
        let here = if at == pc { " <--" } else { "" };
        println!("{at:#010x}: {insn:#010x} {}{here}", disassemble(insn));
    }
}
