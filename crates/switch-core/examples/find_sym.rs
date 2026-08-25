//! Look a symbol up in an NRO and print its load address:
//! `find_sym <path.nro> <name>`.
mod common;

use switch_core::nro::symbol_value;

const USAGE: &str = "find_sym <path.nro> <name>";
/// Where `load_nro` puts an NRO's first byte.
const BASE: u64 = 0x0800_0000;

fn main() {
    let data = common::read(common::arg(1, USAGE));
    let name = common::arg(2, USAGE);
    match symbol_value(&data, &name) {
        Some(value) => println!("{name} = {:#x}", BASE + value),
        None => println!("{name} = not found"),
    }
}
