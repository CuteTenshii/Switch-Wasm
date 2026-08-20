//! Disassemble a flat image dumped by `dump_exefs` at its real load address:
//! `disasm_flat <file.bin> <base> <addr> [count]`.
use std::fs;
use switch_core::disasm::disassemble;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = fs::read(&a[1]).expect("read");
    let p = |s: &String| u32::from_str_radix(s.trim_start_matches("0x"), 16).expect("hex");
    let base = p(&a[2]);
    let addr = p(&a[3]);
    let count: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(48);
    for i in 0..count {
        let va = addr.wrapping_add(i * 4);
        let off = (va - base) as usize;
        if off + 4 > data.len() { break; }
        let insn = u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        println!("{va:#010x}: {insn:08x}  {}", disassemble(insn));
    }
}
