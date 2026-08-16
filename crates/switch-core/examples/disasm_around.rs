use std::fs;
use switch_core::disasm::disassemble;
use switch_core::mem::Memory;
use switch_core::nro::load_nro;

fn main() {
    let path = std::env::args().nth(1).expect("nro path");
    let pc = u32::from_str_radix(
        std::env::args().nth(2).expect("pc").trim_start_matches("0x"),
        16,
    )
    .expect("pc");
    let data = fs::read(&path).expect("read");
    let mut mem = Memory::new();
    let loaded = load_nro(&mut mem, &data).expect("load");
    println!("base={:#x} text_off={:#x} text_size={:#x}", loaded.base, loaded.text.file_offset, loaded.text.file_size);
    let start = pc.wrapping_sub(32);
    for a in (start..=pc.wrapping_add(32)).step_by(4) {
        let insn = mem.fetch(a).unwrap_or(0);
        println!("{:#010x}: {:#010x} {}", a, insn, disassemble(insn));
    }
}
