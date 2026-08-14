//! Nintendo homebrew NRO loader.
//!
//! NRO is the binary format produced by devkitA64 for Switch homebrew. It is
//! a simple container: a 0x40/0x50 byte header followed by three segments
//! (`.text`, `.rodata`, `.data`) that are loaded contiguously, plus a BSS
//! region zero-filled at the end.
//!
//! Header layout (offsets relative to the file start):
//!
//! ```text
//! 0x00  magic "NRO0"
//! 0x04  version (0, 1 or 2)
//! 0x08  total NRO size
//! 0x0C  flags
//! 0x10  text: u32 offset, u32 size
//! 0x18  ro:   u32 offset, u32 size
//! 0x20  data: u32 offset, u32 size
//! 0x28  bss size (u32)
//! 0x2C  reserved
//! 0x30  build id [0x20]
//! 0x50  (version 2 only) "NRO2" header
//! ```

use crate::mem::Memory;
use crate::{Error, Result};

pub const NRO0_MAGIC: u32 = 0x304f524e; // "NRO0"
pub const NRO2_MAGIC: u32 = 0x32524f4e; // "NRO2"
/// Base address where the NRO image is mapped.
///
/// Real homebrew (devkitA64/libnx) is linked against `0x08000000`, the load
/// address the Homebrew Loader (HBL) uses, so baked-in absolute pointers in
/// the image assume that base. Loading anywhere else would make those
/// pointers dangle.
pub const NRO_BASE: u32 = 0x0800_0000;
const HEADER_MIN: usize = 0x50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NroHeader {
    /// File offset where the "NRO0" magic lives (may be nonzero after a boot
    /// stub preamble).
    pub magic_offset: u32,
    pub version: u32,
    pub nro_size: u32,
    pub flags: u32,
    pub text_offset: u32,
    pub text_size: u32,
    pub ro_offset: u32,
    pub ro_size: u32,
    pub data_offset: u32,
    pub data_size: u32,
    pub bss_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadedNro {
    pub base: u32,
    pub entry: u32,
    pub text: Segment,
    pub ro: Segment,
    pub data: Segment,
    pub bss_size: u32,
    pub build_id: [u8; 0x20],
    pub is_64bit: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub file_offset: u32,
    pub file_size: u32,
    pub mem_addr: u32,
}

impl NroHeader {
    /// Parse an NRO header. The `NRO0` magic may appear after a small
    /// preamble (some builds prepend a boot stub, e.g. hbmenu); we scan the
    /// first 0x100 bytes for it.
    pub fn parse(data: &[u8]) -> Result<NroHeader> {
        if data.len() < HEADER_MIN {
            return Err(Error::Truncated {
                what: "NRO header".into(),
                expected: HEADER_MIN,
                got: data.len(),
            });
        }
        let magic_at = find_magic(data, NRO0_MAGIC).ok_or(Error::BadMagic {
            what: "NRO".into(),
            found: crate::nsp::read_u32(data, 0),
        })?;
        let h = magic_at;
        let magic = crate::nsp::read_u32(data, h);
        if magic != NRO0_MAGIC {
            return Err(Error::BadMagic {
                what: "NRO".into(),
                found: magic,
            });
        }
        let version = crate::nsp::read_u32(data, h + 0x04);
        if version > 2 {
            return Err(Error::Nro(format!("unsupported NRO version {}", version)));
        }
        let text_offset = crate::nsp::read_u32(data, h + 0x10);
        let text_size = crate::nsp::read_u32(data, h + 0x14);
        let ro_offset = crate::nsp::read_u32(data, h + 0x18);
        let ro_size = crate::nsp::read_u32(data, h + 0x1C);
        let data_offset = crate::nsp::read_u32(data, h + 0x20);
        let data_size = crate::nsp::read_u32(data, h + 0x24);

        let nro_size = crate::nsp::read_u32(data, h + 0x08);
        if nro_size as usize > data.len() {
            return Err(Error::Nro(format!(
                "header claims {} bytes but file has {}",
                nro_size,
                data.len()
            )));
        }
        for (name, off, size) in [
            ("text", text_offset, text_size),
            ("ro", ro_offset, ro_size),
            ("data", data_offset, data_size),
        ] {
            if off as usize + size as usize > nro_size as usize {
                return Err(Error::Nro(format!(
                    "{} segment [{:#x}, {:#x}) out of bounds (nro_size {:#x})",
                    name, off, off + size, nro_size
                )));
            }
        }

        Ok(NroHeader {
            magic_offset: h as u32,
            version,
            nro_size,
            flags: crate::nsp::read_u32(data, h + 0x0C),
            text_offset,
            text_size,
            ro_offset,
            ro_size,
            data_offset,
            data_size,
            bss_size: crate::nsp::read_u32(data, h + 0x28),
        })
    }

    pub fn is_64bit(&self, data: &[u8]) -> bool {
        if self.version >= 2 && data.len() >= self.magic_offset as usize + 0x64 {
            let is_64 = data[self.magic_offset as usize + 0x60];
            if is_64 == 0 {
                return false;
            }
        }
        true
    }

    pub fn build_id(&self, data: &[u8]) -> [u8; 0x20] {
        let mut id = [0u8; 0x20];
        let start = self.magic_offset as usize + 0x30;
        let end = (start + 0x20).min(data.len());
        id[..end - start].copy_from_slice(&data[start..end]);
        id
    }
}

fn find_magic(data: &[u8], magic: u32) -> Option<usize> {
    let magic_le = magic.to_le_bytes();
    data[..data.len().min(0x100)]
        .windows(4)
        .position(|w| w == &magic_le[..])
}

/// Load an NRO into `mem` at [`NRO_BASE`], returning its entry point.
pub fn load_nro(mem: &mut Memory, data: &[u8]) -> Result<LoadedNro> {
    let h = NroHeader::parse(data)?;
    let build_id = h.build_id(data);
    let is_64bit = h.is_64bit(data);

    let base = NRO_BASE;
    let text_addr = base;
    let ro_addr = text_addr.wrapping_add(h.text_size);
    let data_addr = ro_addr.wrapping_add(h.ro_size);
    let end_addr = data_addr.wrapping_add(h.data_size);

    copy_segment(mem, data, h.text_offset, h.text_size, text_addr)?;
    copy_segment(mem, data, h.ro_offset, h.ro_size, ro_addr)?;
    copy_segment(mem, data, h.data_offset, h.data_size, data_addr)?;
    mem.map_zero(end_addr, h.bss_size as usize)?;

    Ok(LoadedNro {
        base,
        entry: text_addr,
        text: Segment {
            file_offset: h.text_offset,
            file_size: h.text_size,
            mem_addr: text_addr,
        },
        ro: Segment {
            file_offset: h.ro_offset,
            file_size: h.ro_size,
            mem_addr: ro_addr,
        },
        data: Segment {
            file_offset: h.data_offset,
            file_size: h.data_size,
            mem_addr: data_addr,
        },
        bss_size: h.bss_size,
        build_id,
        is_64bit,
    })
}

fn copy_segment(mem: &mut Memory, data: &[u8], off: u32, size: u32, addr: u32) -> Result<()> {
    if size == 0 {
        return Ok(());
    }
    let start = off as usize;
    let end = start + size as usize;
    if end > data.len() {
        return Err(Error::Nro("segment exceeds input data".into()));
    }
    mem.map(addr, &data[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_nro(text: &[u8], data: &[u8]) -> Vec<u8> {
        let text_size = align4(text.len());
        let data_off = 0x50 + text_size;
        let data_size = align4(data.len());
        let mut out = vec![0u8; data_off + data_size];
        out[0..4].copy_from_slice(&NRO0_MAGIC.to_le_bytes());
        out[4..8].copy_from_slice(&1u32.to_le_bytes()); // version 1
        let total = out.len() as u32;
        out[8..12].copy_from_slice(&total.to_le_bytes());
        out[0x10..0x14].copy_from_slice(&0x50u32.to_le_bytes());
        out[0x14..0x18].copy_from_slice(&(text_size as u32).to_le_bytes());
        out[0x18..0x1C].copy_from_slice(&0u32.to_le_bytes());
        out[0x1C..0x20].copy_from_slice(&0u32.to_le_bytes());
        out[0x20..0x24].copy_from_slice(&(data_off as u32).to_le_bytes());
        out[0x24..0x28].copy_from_slice(&(data_size as u32).to_le_bytes());
        out[0x28..0x2C].copy_from_slice(&0x100u32.to_le_bytes()); // bss
        out[0x50..0x50 + text.len()].copy_from_slice(text);
        out[data_off..data_off + data.len()].copy_from_slice(data);
        out
    }

    fn align4(n: usize) -> usize {
        (n + 3) & !3
    }

    #[test]
    fn loads_and_lays_out_segments() {
        let text = [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00];
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        let nro = build_nro(&text, &data);
        let mut mem = Memory::new();
        let loaded = load_nro(&mut mem, &nro).unwrap();
        assert_eq!(loaded.entry, NRO_BASE);
        assert_eq!(mem.read_u32(NRO_BASE).unwrap(), 0x01);
        assert_eq!(mem.read_u32(NRO_BASE + 4).unwrap(), 0x02);
        assert_eq!(
            mem.read_u32(loaded.data.mem_addr).unwrap(),
            0xEFBE_ADDE
        );
        // bss zero-filled
        assert_eq!(
            mem.read_u8(loaded.data.mem_addr + loaded.data.file_size).unwrap(),
            0
        );
        assert!(loaded.is_64bit);
        assert_eq!(loaded.build_id, [0u8; 0x20]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut nro = build_nro(&[0u8; 4], &[]);
        nro[0] = b'X';
        assert!(matches!(NroHeader::parse(&nro), Err(Error::BadMagic { .. })));
    }

    #[test]
    fn rejects_truncated() {
        assert!(matches!(
            NroHeader::parse(&[0u8; 0x20]),
            Err(Error::Truncated { .. })
        ));
    }

    #[test]
    fn rejects_segment_out_of_bounds() {
        let mut nro = build_nro(&[0u8; 4], &[]);
        nro[0x14..0x18].copy_from_slice(&0xFFFFu32.to_le_bytes());
        assert!(matches!(
            NroHeader::parse(&nro),
            Err(Error::Nro(_))
        ));
    }
}
