//! Disassemble a flat image dumped by `dump_exefs` at its real load address:
//! `disasm_flat <file.bin> <base> <addr> [count]`.
mod common;

use switch_core::disasm::disassemble;

const USAGE: &str = "disasm_flat <file.bin> <base> <addr> [count]";

fn main() {
    let data = common::read(common::arg(1, USAGE));
    let base = common::hex(&common::arg(2, USAGE));
    let addr = common::hex(&common::arg(3, USAGE));
    let count = common::opt_num(4).unwrap_or(48) as u32;
    for i in 0..count {
        let va = addr.wrapping_add(i * 4);
        let Some(word) = data
            .get((va.wrapping_sub(base)) as usize..)
            .and_then(|s| s.first_chunk::<4>())
        else {
            break;
        };
        let insn = u32::from_le_bytes(*word);
        println!("{va:#010x}: {insn:08x}  {}", disassemble(insn));
    }
}
