use std::fs;
use switch_core::disasm::disassemble;
use switch_core::mem::Memory;
use switch_core::nro::load_nro;

fn main() {
    let path = std::env::args().nth(1).expect("nro path");
    let start = u32::from_str_radix(
        std::env::args().nth(2).expect("start").trim_start_matches("0x"),
        16,
    ).expect("start");
    let count = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let data = fs::read(&path).expect("read");
    let mut mem = Memory::new();
    let _loaded = load_nro(&mut mem, &data).expect("load");
    for i in 0..count {
        let a = start.wrapping_add(i * 4);
        let insn = mem.fetch(a).unwrap_or(0);
        println!("{:#010x}: {:#010x} {}", a, insn, disassemble(insn));
    }
}
