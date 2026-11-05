//! Look symbols up in an NRO and print their load addresses:
//! `find_sym <path.nro> <name>...`.
mod common;

use clap::Parser;
use switch_core::nro::symbol_value;

/// Where `load_nro` puts an NRO's first byte.
const BASE: u64 = 0x0800_0000;

#[derive(Parser)]
struct Cli {
    nro: String,
    #[arg(required = true)]
    names: Vec<String>,
}

fn main() {
    let cli = Cli::parse();
    let data = common::read(&cli.nro);
    for name in cli.names {
        match symbol_value(&data, &name) {
            Some(value) => println!("{name} = {:#x}", BASE + value),
            None => println!("{name} = not found"),
        }
    }
}
