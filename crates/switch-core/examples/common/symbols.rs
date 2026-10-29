/// "MOD0", the module header crt0 points at.
const MOD0_MAGIC: u32 = 0x3044_4f4d;
const DT_HASH: u64 = 4;
const DT_STRTAB: u64 = 5;
const DT_SYMTAB: u64 = 6;
const SYM_SIZE: usize = 24;

/// One defined symbol; `value` is relative to the image's first byte.
pub struct Symbol {
    pub value: u64,
    pub size: u64,
    pub name: String,
}

/// The defined symbols of an image laid out as loaded: an NRO file, or a
/// module flattened at its load address. Empty without a dynamic symbol table.
pub fn read(image: &[u8]) -> Vec<Symbol> {
    let tables = (|| {
        let mod0 = find_mod0(image)?;
        let dynamic = mod0.checked_add_signed(word(image, mod0 + 4)? as i32 as isize)?;
        let (mut symtab, mut strtab, mut hash) = (0, 0, 0);
        for entry in image.get(dynamic..)?.as_chunks::<16>().0 {
            let (tag, value) = entry.split_at(8);
            let tag = u64::from_le_bytes(tag.try_into().ok()?);
            let value = u64::from_le_bytes(value.try_into().ok()?) as usize;
            match tag {
                0 => break,
                DT_SYMTAB => symtab = value,
                DT_STRTAB => strtab = value,
                DT_HASH => hash = value,
                _ => {}
            }
        }
        (symtab != 0 && strtab != 0 && hash != 0).then_some((symtab, strtab, hash))
    })();
    let Some((symtab, strtab, hash)) = tables else {
        return Vec::new();
    };
    // DT_HASH's chain count is the symbol count.
    let count = word(image, hash + 4).unwrap_or(0) as usize;
    (0..count)
        .map_while(|i| image.get(symtab + i * SYM_SIZE..symtab + (i + 1) * SYM_SIZE))
        .filter_map(|sym| {
            let name = u32::from_le_bytes(sym[..4].try_into().ok()?) as usize;
            let value = u64::from_le_bytes(sym[8..16].try_into().ok()?);
            let size = u64::from_le_bytes(sym[16..].try_into().ok()?);
            (name != 0 && value != 0).then(|| Symbol {
                value,
                size,
                name: cstr(image, strtab + name),
            })
        })
        .collect()
}

/// crt0's second word points at MOD0; search for the magic if it does not.
fn find_mod0(image: &[u8]) -> Option<usize> {
    let magic = MOD0_MAGIC.to_le_bytes();
    let pointed = word(image, 4).map(|at| at as usize);
    match pointed {
        Some(at) if image.get(at..at + 4) == Some(&magic[..]) => Some(at),
        _ => image.windows(4).position(|w| w == magic),
    }
}

fn word(image: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        image.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn cstr(image: &[u8], at: usize) -> String {
    let bytes = image.get(at..).unwrap_or_default();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}
