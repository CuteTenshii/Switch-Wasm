//! Disassemble a single A64 instruction word: `disasm_insn <word>`.
mod common;

use switch_core::disasm::disassemble;

fn main() {
    let insn = common::hex(&common::arg(1, "disasm_insn <word>"));
    println!("{insn:#010x}: {}", disassemble(insn));
}
