use switch_core::disasm::disassemble;

fn main() {
    let insn = u32::from_str_radix(
        std::env::args().nth(1).expect("insn").trim_start_matches("0x"),
        16,
    )
    .expect("insn");
    println!("{}: {}", insn, disassemble(insn));
}
