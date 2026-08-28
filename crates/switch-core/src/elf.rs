//! Minimal AArch64 ELF loader.
//!
//! Supports the common case for a bare-metal homebrew ELF: 64-bit,
//! little-endian, `EM_AARCH64`, `ET_EXEC` or `ET_DYN`, with `PT_LOAD`
//! segments mapped at their `p_vaddr` and `p_memsz > p_filesz` regions
//! zero-filled. Entry point is `e_entry`.

use crate::mem::Memory;
use crate::{Error, Result};

pub const EM_AARCH64: u16 = 183;
pub const PT_LOAD: u32 = 1;
pub const ELF64_HEADER_SIZE: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedElf {
    pub entry: u64,
    pub segments: Vec<LoadSegment>,
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
    if data.len() < ELF64_HEADER_SIZE {
        return Err(Error::Truncated {
            what: "ELF header".into(),
            expected: ELF64_HEADER_SIZE,
            got: data.len(),
        });
    }
    if &data[0..4] != b"\x7fELF" {
        return Err(Error::Elf("not an ELF file".into()));
    }
    if data[4] != 2 {
        return Err(Error::Elf("not a 64-bit ELF".into()));
    }
    if data[5] != 1 {
        return Err(Error::Elf("not little-endian ELF".into()));
    }
    let machine = u16::from_le_bytes([data[18], data[19]]);
    if machine != EM_AARCH64 {
        return Err(Error::Elf(format!(
            "not an AArch64 ELF (machine {})",
            machine
        )));
    }
    let e_type = u16::from_le_bytes([data[16], data[17]]);
    if e_type != 2 && e_type != 3 {
        return Err(Error::Elf(format!(
            "unsupported ELF type {} (want EXEC or DYN)",
            e_type
        )));
    }
    let entry = u64::from_le_bytes(data[0x18..0x20].try_into().unwrap());
    let phoff = u64::from_le_bytes(data[0x20..0x28].try_into().unwrap()) as usize;
    let phentsize = u16::from_le_bytes([data[0x36], data[0x37]]) as usize;
    let phnum = u16::from_le_bytes([data[0x38], data[0x39]]) as usize;
    if phentsize < 56 {
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
        let p_type = u32::from_le_bytes(ph[0..4].try_into().unwrap());
        if p_type != PT_LOAD {
            continue;
        }
        let p_offset = u64::from_le_bytes(ph[0x08..0x10].try_into().unwrap()) as usize;
        let p_vaddr = u64::from_le_bytes(ph[0x10..0x18].try_into().unwrap());
        let p_filesz = u64::from_le_bytes(ph[0x20..0x28].try_into().unwrap());
        let p_memsz = u64::from_le_bytes(ph[0x28..0x30].try_into().unwrap());
        let p_flags = u32::from_le_bytes(ph[0x04..0x08].try_into().unwrap());
        if p_filesz as usize > data.len().saturating_sub(p_offset) {
            return Err(Error::Elf(format!("segment {} filesz exceeds file", i)));
        }
        segments.push(LoadSegment {
            vaddr: p_vaddr,
            offset: p_offset as u64,
            filesz: p_filesz,
            memsz: p_memsz,
            flags: p_flags,
        });
    }

    Ok(LoadedElf { entry, segments })
}

/// Map an ELF's `PT_LOAD` segments into `mem`.
pub fn load_elf(mem: &mut Memory, data: &[u8]) -> Result<LoadedElf> {
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
