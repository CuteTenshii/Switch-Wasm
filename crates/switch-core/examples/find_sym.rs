use std::fs;
use switch_core::nro::symbol_value;

fn main() {
    let path = std::env::args().nth(1).expect("usage: find_sym <nro> <name>");
    let name = std::env::args().nth(2).expect("name");
    let data = fs::read(&path).expect("read nro");
    let base = 0x8000000u64;
    if let Some(v) = symbol_value(&data, &name) {
        println!("{} = {:#x}", name, base + v);
    } else {
        println!("{} = not found", name);
    }
}
