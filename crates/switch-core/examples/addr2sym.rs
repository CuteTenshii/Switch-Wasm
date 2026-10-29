//! Map an address back to the symbol it falls in, using an NRO's own dynamic
//! symbol table: `addr2sym <path.nro> <addr>`.
mod common;

/// Where `load_nro` puts an NRO's first byte.
const BASE: u64 = 0x0800_0000;

fn main() {
    const USAGE: &str = "addr2sym <path.nro> <addr>";
    let data = common::read(common::arg(1, USAGE));
    let target = u64::from(common::hex(&common::arg(2, USAGE)));
    let symbols = common::symbols::read(&data);
    if symbols.is_empty() {
        println!("no dynamic symbols (statically linked homebrew exports none)");
        return;
    }
    // Symbol values are relative to the image, so accept either form of address.
    let offset = target.checked_sub(BASE).unwrap_or(target);
    if let Some(sym) = symbols
        .iter()
        .find(|s| offset >= s.value && offset < s.value + s.size)
    {
        println!(
            "{target:#x} is in {}+{:#x} (size {:#x})",
            sym.name,
            offset - sym.value,
            sym.size
        );
        return;
    }
    match symbols
        .iter()
        .filter(|s| s.value < offset)
        .max_by_key(|s| s.value)
    {
        Some(sym) => println!("{target:#x} after {}+{:#x}", sym.name, offset - sym.value),
        None => println!("{target:#x} not found"),
    }
}
