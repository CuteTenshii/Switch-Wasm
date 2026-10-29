//! Look symbols up in an NRO and print their load addresses:
//! `find_sym <path.nro> <name>...`.
mod common;

use switch_core::nro::symbol_value;

const USAGE: &str = "find_sym <path.nro> <name>...";
/// Where `load_nro` puts an NRO's first byte.
const BASE: u64 = 0x0800_0000;

fn main() {
    let data = common::read(common::arg(1, USAGE));
    let names: Vec<String> = std::env::args().skip(2).collect();
    if names.is_empty() {
        common::usage(USAGE);
    }
    for name in names {
        match symbol_value(&data, &name) {
            Some(value) => println!("{name} = {:#x}", BASE + value),
            None => println!("{name} = not found"),
        }
    }
}
