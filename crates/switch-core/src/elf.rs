//! Minimal ELF loader: little-endian `ET_EXEC`/`ET_DYN` for AArch64 (64-bit) or ARM
//! (32-bit), with `PT_LOAD` segments mapped at `p_vaddr` and their BSS tails zero-filled.

use crate::mem::Memory;
use crate::{Error, Result};

pub const EM_AARCH64: u16 = 183;
pub const EM_ARM: u16 = 40;
pub const PT_LOAD: u32 = 1;
pub const ELF64_HEADER_SIZE: usize = 64;
pub const ELF32_HEADER_SIZE: usize = 52;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedElf {
    pub entry: u64,
    pub segments: Vec<LoadSegment>,
    /// A 32-bit ARM image, which runs in the A32 state.
    pub is_32_bit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadSegment {
    pub vaddr: u64,
    pub offset: u64,
    pub filesz: u64,
    pub memsz: u64,
    pub flags: u32,
}

pub fn parse_elf(data: &[u8]) -> Result<LoadedElf> {
    let header = |expected: usize| Error::Truncated {
        what: "ELF header".into(),
        expected,
        got: data.len(),
    };
    if data.len() < ELF32_HEADER_SIZE {
        return Err(header(ELF32_HEADER_SIZE));
    }
    if &data[0..4] != b"\x7fELF" {
        return Err(Error::Elf("not an ELF file".into()));
    }
    if data[5] != 1 {
        return Err(Error::Elf("not little-endian ELF".into()));
    }
    let machine = u16::from_le_bytes([data[18], data[19]]);
    let is_32_bit = match (data[4], machine) {
        (2, EM_AARCH64) => false,
        (1, EM_ARM) => true,
        _ => {
            return Err(Error::Elf(format!(
                "not an AArch64 or 32-bit ARM ELF (class {}, machine {})",
                data[4], machine
            )))
        }
    };
    if !is_32_bit && data.len() < ELF64_HEADER_SIZE {
        return Err(header(ELF64_HEADER_SIZE));
    }
    let e_type = u16::from_le_bytes([data[16], data[17]]);
    if e_type != 2 && e_type != 3 {
        return Err(Error::Elf(format!(
            "unsupported ELF type {} (want EXEC or DYN)",
            e_type
        )));
    }
    let word = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
    let half = |at: usize| usize::from(u16::from_le_bytes([data[at], data[at + 1]]));
    let dword = |at: usize| u64::from_le_bytes(data[at..at + 8].try_into().unwrap());
    let (entry, phoff, phentsize, phnum, min_phentsize) = if is_32_bit {
        (
            u64::from(word(0x18)),
            word(0x1C) as usize,
            half(0x2A),
            half(0x2C),
            32,
        )
    } else {
        (
            dword(0x18),
            dword(0x20) as usize,
            half(0x36),
            half(0x38),
            56,
        )
    };
    if phentsize < min_phentsize {
        return Err(Error::Elf(format!(
            "program header size {} too small",
            phentsize
        )));
    }

    let mut segments = Vec::new();
    for i in 0..phnum {
        let off = phoff.checked_add(i * phentsize).ok_or(Error::Overflow)?;
        let ph = data
            .get(off..off + phentsize)
            .ok_or_else(|| Error::Elf(format!("program header {} outside file", i)))?;
        if u32::from_le_bytes(ph[0..4].try_into().unwrap()) != PT_LOAD {
            continue;
        }
        let word = |at: usize| u64::from(u32::from_le_bytes(ph[at..at + 4].try_into().unwrap()));
        let dword = |at: usize| u64::from_le_bytes(ph[at..at + 8].try_into().unwrap());
        let segment = if is_32_bit {
            LoadSegment {
                vaddr: word(0x08),
                offset: word(0x04),
                filesz: word(0x10),
                memsz: word(0x14),
                flags: word(0x18) as u32,
            }
        } else {
            LoadSegment {
                vaddr: dword(0x10),
                offset: dword(0x08),
                filesz: dword(0x20),
                memsz: dword(0x28),
                flags: word(0x04) as u32,
            }
        };
        if segment.filesz as usize > data.len().saturating_sub(segment.offset as usize) {
            return Err(Error::Elf(format!("segment {} filesz exceeds file", i)));
        }
        segments.push(segment);
    }

    Ok(LoadedElf {
        entry,
        segments,
        is_32_bit,
    })
}

/// Map an ELF's `PT_LOAD` segments into `mem`.
pub fn load_elf(mem: &Memory, data: &[u8]) -> Result<LoadedElf> {
    let elf = parse_elf(data)?;
    for seg in &elf.segments {
        if seg.filesz > 0 {
            let start = seg.offset as usize;
            mem.map(seg.vaddr as u32, &data[start..start + seg.filesz as usize])?;
        }
        if seg.memsz > seg.filesz {
            mem.map_zero(
                (seg.vaddr + seg.filesz) as u32,
                (seg.memsz - seg.filesz) as usize,
            )?;
        }
    }
    Ok(elf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One `PT_LOAD` segment of `payload` at `vaddr`, in either class.
    fn image(is_32_bit: bool, entry: u64, vaddr: u64, payload: &[u8]) -> Vec<u8> {
        let (header, phentsize) = if is_32_bit { (52, 32) } else { (64, 56) };
        let mut out = vec![0u8; header + phentsize];
        out[..4].copy_from_slice(b"\x7fELF");
        out[4] = if is_32_bit { 1 } else { 2 };
        out[5] = 1;
        out[16] = 2;
        let machine = if is_32_bit { EM_ARM } else { EM_AARCH64 };
        out[18..20].copy_from_slice(&machine.to_le_bytes());
        let offset = (header + phentsize) as u64;
        let put = |out: &mut Vec<u8>, at: usize, v: u64, wide: bool| {
            let bytes = v.to_le_bytes();
            out[at..at + if wide { 8 } else { 4 }]
                .copy_from_slice(&bytes[..if wide { 8 } else { 4 }]);
        };
        if is_32_bit {
            put(&mut out, 0x18, entry, false);
            put(&mut out, 0x1C, header as u64, false);
            out[0x2A..0x2C].copy_from_slice(&(phentsize as u16).to_le_bytes());
            out[0x2C..0x2E].copy_from_slice(&1u16.to_le_bytes());
            let ph = header;
            put(&mut out, ph, u64::from(PT_LOAD), false);
            put(&mut out, ph + 0x04, offset, false);
            put(&mut out, ph + 0x08, vaddr, false);
            put(&mut out, ph + 0x10, payload.len() as u64, false);
            put(&mut out, ph + 0x14, payload.len() as u64 + 0x100, false);
            put(&mut out, ph + 0x18, 5, false);
        } else {
            put(&mut out, 0x18, entry, true);
            put(&mut out, 0x20, header as u64, true);
            out[0x36..0x38].copy_from_slice(&(phentsize as u16).to_le_bytes());
            out[0x38..0x3A].copy_from_slice(&1u16.to_le_bytes());
            let ph = header;
            put(&mut out, ph, u64::from(PT_LOAD), false);
            put(&mut out, ph + 0x04, 5, false);
            put(&mut out, ph + 0x08, offset, true);
            put(&mut out, ph + 0x10, vaddr, true);
            put(&mut out, ph + 0x20, payload.len() as u64, true);
            put(&mut out, ph + 0x28, payload.len() as u64 + 0x100, true);
        }
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn both_classes_give_the_same_segment() {
        for is_32_bit in [false, true] {
            let elf =
                parse_elf(&image(is_32_bit, 0x4000_0004, 0x4000_0000, &[1, 2, 3, 4])).unwrap();
            assert_eq!(elf.is_32_bit, is_32_bit);
            assert_eq!(elf.entry, 0x4000_0004);
            assert_eq!(
                elf.segments,
                vec![LoadSegment {
                    vaddr: 0x4000_0000,
                    offset: if is_32_bit { 84 } else { 120 },
                    filesz: 4,
                    memsz: 0x104,
                    flags: 5,
                }]
            );
        }
    }

    #[test]
    fn another_machine_is_refused() {
        let mut elf = image(true, 0, 0x1000, &[0; 4]);
        elf[18] = 3;
        assert!(parse_elf(&elf).is_err());
    }
}
