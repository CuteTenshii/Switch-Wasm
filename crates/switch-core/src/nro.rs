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
use crate::nsp::read_u32;
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

fn read_u64(data: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    let n = (data.len().saturating_sub(off)).min(8);
    b[..n].copy_from_slice(&data[off..off + n]);
    u64::from_le_bytes(b)
}

/// Find the MOD0 header that follows the NRO0 header (it carries the dynamic
/// section offset the loader needs for relocations).
fn find_mod0(data: &[u8]) -> Option<usize> {
    let magic = 0x30444f4du32.to_le_bytes(); // "MOD0"
    data[..data.len().min(0x1000)]
        .windows(4)
        .position(|w| w == &magic[..])
}

/// Apply RELR packed relative relocations. Every 64-bit entry is either an
/// address (bit 0 clear) or a bitmap (bit 0 set); each relocated word gets the
/// load base added, turning stored file offsets into runtime addresses.
///
/// The chain is strictly monotonic (addresses only increase), so processing
/// stops as soon as an entry would move backwards or past `end_addr` — this
/// also guards against trailing garbage after the real RELR data.
fn apply_relr(mem: &mut Memory, base: u32, end_addr: u32, relr: &[u8]) -> Result<()> {
    let mut addr: u32 = 0;
    let mut last: u32 = 0;
    for chunk in relr.chunks(8) {
        let mut bytes = [0u8; 8];
        bytes[..chunk.len()].copy_from_slice(chunk);
        let word = u64::from_le_bytes(bytes);
        if word & 1 == 0 {
            addr = base.wrapping_add(word as u32);
            if addr < last || addr >= end_addr {
                return Ok(());
            }
            last = addr;
            let cur = mem.read_u64(addr)?;
            mem.write_u64(addr, cur.wrapping_add(base as u64))?;
        } else {
            if addr >= end_addr {
                return Ok(());
            }
            for bit in 1..64u32 {
                if word & (1u64 << bit) != 0 {
                    let a = addr.wrapping_add(8 * (bit - 1));
                    if a < last || a >= end_addr {
                        return Ok(());
                    }
                    last = a;
                    let cur = mem.read_u64(a)?;
                    mem.write_u64(a, cur.wrapping_add(base as u64))?;
                }
            }
            addr = addr.wrapping_add(8 * 63);
        }
    }
    Ok(())
}

/// Apply the RELR relocations described by the image's MOD0/dynamic headers,
/// if present. NROs are linked against `NRO_BASE`, so absolute pointers in
/// .data/.rodata only become valid once the base is added.
fn apply_nro_relocations(mem: &mut Memory, data: &[u8], image_end: u32) -> Result<()> {
    let mod0 = match find_mod0(data) {
        Some(m) => m,
        None => return Ok(()),
    };
    let dyn_off = mod0.wrapping_add(read_u32(data, mod0 + 4) as usize);
    if dyn_off + 16 > data.len() {
        return Ok(());
    }
    let mut off = dyn_off;
    let mut relr_off = 0u32;
    let mut relr_size = 0u32;
    let mut relr_count = 0u64;
    loop {
        if off + 16 > data.len() {
            break;
        }
        let tag = read_u64(data, off);
        let val = read_u64(data, off + 8);
        off += 16;
        if tag == 0 {
            break;
        }
        match tag {
            0x24 => relr_off = val as u32,   // DT_RELR
            0x23 => relr_count = val,        // DT_RELRCOUNT
            0x25 => relr_size = val as u32,  // DT_RELRSZ
            _ => {}
        }
    }
    // Some builds report a bogus DT_RELRSZ (e.g. hbmenu writes the byte size
    // into DT_RELRCOUNT instead); take whichever field is larger, in bytes.
    let size = relr_size.max(relr_count as u32);
    let start = relr_off as usize;
    let end = start.saturating_add(size as usize);
    if size > 0 && end <= data.len() {
        apply_relr(mem, NRO_BASE, image_end, &data[start..end])?;
    }
    Ok(())
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
    let image_end = end_addr.wrapping_add(h.bss_size);
    apply_nro_relocations(mem, data, image_end)?;

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
    fn apply_relr_relocates_and_stops_at_backward_jump() {
        let mut mem = Memory::new();
        mem.map_zero(NRO_BASE, 0x2000).unwrap();
        mem.write_u64(NRO_BASE + 0x1000, 0x1234).unwrap();
        mem.write_u64(NRO_BASE + 0x1008, 0x5678).unwrap();
        // RELR: address entry [0x1000], a bitmap flagging the next slot, then
        // a backward address (the trailing garbage hbmenu's section has).
        let mut relr = Vec::new();
        relr.extend_from_slice(&0x1000u64.to_le_bytes());
        relr.extend_from_slice(&0b101u64.to_le_bytes()); // bitmap: bit0=1, bit2 → +8
        relr.extend_from_slice(&0u64.to_le_bytes());     // address 0 → backwards → stop
        apply_relr(&mut mem, NRO_BASE, NRO_BASE + 0x2000, &relr).unwrap();
        assert_eq!(mem.read_u64(NRO_BASE + 0x1000).unwrap(), 0x1234 + NRO_BASE as u64);
        assert_eq!(mem.read_u64(NRO_BASE + 0x1008).unwrap(), 0x5678 + NRO_BASE as u64);
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
