//! Nintendo Switch NSO0 executable loader.
//!
//! NSO is the format Nintendo's SDK links a game's `main`/`subsdk*`/`sdk`
//! executables into — what an NCA's ExeFS (PFS0) actually contains. Like NRO
//! it's three segments (`.text`, `.rodata`, `.data`) loaded contiguously plus
//! a zero-filled BSS, but each segment may be individually LZ4-compressed on
//! disk, and there's no HBL-style external loader: the linked-in crt0
//! (Nintendo's `rtld`) processes its own relocations and BSS zeroing itself
//! before calling `main`, the same way a self-relocating homebrew NRO does.
//! So this loader only places the decompressed bytes and hands off control —
//! no MOD0/RELR handling needed here.
//!
//! For a regular SDK-linked module (`main`/`subsdk*`/`sdk`), execution does
//! **not** start at `.text`+0: the first bytes there are a `ModulePtr` (`u32`
//! reserved, `u32` offset to the `MOD0` header, then `MOD0` itself — 0x1C
//! bytes — followed by 0xC bytes of padding to a 16-byte boundary). Confirmed
//! against a real title's decompiled `.text` (via the project's own
//! disassembler): everything up to offset [`NSO_ENTRY_OFFSET`] disassembles
//! as garbage (it's data, not code), and at exactly that offset a textbook
//! crt0 prologue begins (`sub sp, sp, #0x90` / `stp x29, x30, [...]`)
//! followed by the same constructor-array-calling pattern (`blr` in a loop)
//! homebrew's own crt0 runs.
//!
//! `rtld` itself is the exception: it has no `ModulePtr`/`MOD0` header at
//! all — its `.text`+0 is real code, a `b` that jumps over an inline
//! PC-relative literal used by its own base-address bootstrap (it must
//! establish where it was loaded before it can do anything else, including
//! locating its own `MOD0`). Jumping straight to `.text`+[`NSO_ENTRY_OFFSET`]
//! for `rtld` skips that bootstrap, leaving its registers unset and
//! corrupting later computations that assume it ran (confirmed by tracing a
//! real title's `rtld` module: `x0` — its own base address — stays `0`,
//! which turns a `bss_end - base` size computation into a bogus ~4GB byte
//! count fed to a self-corrupting `memset` loop). [`entry_offset`] tells the
//! two cases apart by checking for the `ModulePtr` + `MOD0` signature rather
//! than assuming it's always present; `entry` in [`LoadedNso`] already
//! accounts for this.
//!
//! Header layout (0x100 bytes):
//!
//! ```text
//! 0x00  magic "NSO0"
//! 0x04  version
//! 0x08  reserved
//! 0x0C  flags: bit0/1/2 = text/rodata/data compressed
//! 0x10  .text: u32 file_offset, u32 mem_offset, u32 decompressed_size
//! 0x1C  (module name offset — unused here)
//! 0x20  .rodata: u32 file_offset, u32 mem_offset, u32 decompressed_size
//! 0x2C  (module name size — unused here)
//! 0x30  .data: u32 file_offset, u32 mem_offset, u32 decompressed_size
//! 0x3C  .bss size (u32)
//! 0x40  module id / build id [0x20]
//! 0x60  .text compressed (on-disk) size (u32)
//! 0x64  .rodata compressed size (u32)
//! 0x68  .data compressed size (u32)
//! ```

use crate::mem::Memory;
use crate::nro::{Segment, NRO_BASE};
use crate::nsp::read_u32;
use crate::{Error, Result};

pub const NSO0_MAGIC: u32 = 0x304f_534e; // "NSO0"
/// Base address the image is mapped at. Reuses the homebrew NRO's base: the
/// two loaders are mutually exclusive (a session runs one image at a time),
/// and every other fixed address in this emulator (stack, TLS, env block) is
/// already pinned relative to this scheme.
pub const NSO_BASE: u32 = NRO_BASE;
const HEADER_SIZE: usize = 0x100;

/// Distance from `.text` start to the real crt0 entry point when a
/// `ModulePtr`/`MOD0` header is present: past the `ModulePtr` (`u32`
/// reserved + `u32` MOD0-offset, 8 bytes), the `MOD0` header itself (5 `u32`
/// fields after the magic, 0x1C bytes total from the `ModulePtr` start), and
/// padding out to a 16-byte boundary.
pub const NSO_ENTRY_OFFSET: u32 = 0x30;

const MOD0_MAGIC: u32 = 0x3044_4f4d; // "MOD0"

/// Distance from `.text` start to the real entry point, for either module
/// layout `.text` can have. See the module doc comment.
fn entry_offset(text: &[u8]) -> u32 {
    if text.len() < 8 {
        return 0;
    }
    let reserved = read_u32(text, 0);
    let mod0_offset = read_u32(text, 4) as usize;
    let has_mod0 = reserved == 0
        && mod0_offset + 4 <= text.len()
        && read_u32(text, mod0_offset) == MOD0_MAGIC;
    if has_mod0 {
        NSO_ENTRY_OFFSET
    } else {
        0
    }
}

const FLAG_TEXT_COMPRESSED: u32 = 1 << 0;
const FLAG_RODATA_COMPRESSED: u32 = 1 << 1;
const FLAG_DATA_COMPRESSED: u32 = 1 << 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadedNso {
    pub base: u32,
    /// `.text` start + [`NSO_ENTRY_OFFSET`] — *not* `.text` start itself
    /// (unlike a homebrew NRO). See the module doc comment.
    pub entry: u32,
    pub text: Segment,
    pub ro: Segment,
    pub data: Segment,
    pub bss_size: u32,
    pub build_id: [u8; 0x20],
}

struct RawSegment {
    file_offset: u32,
    mem_offset: u32,
    decompressed_size: u32,
}

fn read_segment(data: &[u8], at: usize) -> RawSegment {
    RawSegment {
        file_offset: read_u32(data, at),
        mem_offset: read_u32(data, at + 4),
        decompressed_size: read_u32(data, at + 8),
    }
}

/// Load an NSO0 image into `mem` at `base`, decompressing any segment
/// flagged LZ4-compressed. Returns the loaded segment layout and entry point.
///
/// A retail title is multiple NSO modules (`rtld`, `main`, `subsdk*`, `sdk`)
/// sharing one address space — `base` lets a caller lay them out
/// sequentially instead of every module claiming [`NSO_BASE`] for itself.
pub fn load_nso(mem: &mut Memory, data: &[u8], base: u32) -> Result<LoadedNso> {
    if data.len() < HEADER_SIZE {
        return Err(Error::Truncated {
            what: "NSO header".into(),
            expected: HEADER_SIZE,
            got: data.len(),
        });
    }
    if read_u32(data, 0) != NSO0_MAGIC {
        return Err(Error::BadMagic {
            what: "NSO".into(),
            found: read_u32(data, 0),
        });
    }
    let flags = read_u32(data, 0x0C);
    let text = read_segment(data, 0x10);
    let rodata = read_segment(data, 0x20);
    let raw_data = read_segment(data, 0x30);
    let bss_size = read_u32(data, 0x3C);
    let mut build_id = [0u8; 0x20];
    build_id.copy_from_slice(&data[0x40..0x60]);
    let text_compressed_size = read_u32(data, 0x60);
    let rodata_compressed_size = read_u32(data, 0x64);
    let data_compressed_size = read_u32(data, 0x68);

    let text_addr = base.wrapping_add(text.mem_offset);
    let ro_addr = base.wrapping_add(rodata.mem_offset);
    let data_addr = base.wrapping_add(raw_data.mem_offset);

    let text_bytes = extract_segment(
        data,
        &text,
        text_compressed_size,
        flags & FLAG_TEXT_COMPRESSED != 0,
        ".text",
    )?;
    let ro_bytes = extract_segment(
        data,
        &rodata,
        rodata_compressed_size,
        flags & FLAG_RODATA_COMPRESSED != 0,
        ".rodata",
    )?;
    let data_bytes = extract_segment(
        data,
        &raw_data,
        data_compressed_size,
        flags & FLAG_DATA_COMPRESSED != 0,
        ".data",
    )?;

    mem.map(text_addr, &text_bytes)?;
    mem.map(ro_addr, &ro_bytes)?;
    mem.map(data_addr, &data_bytes)?;
    let bss_addr = data_addr.wrapping_add(raw_data.decompressed_size);
    mem.map_zero(bss_addr, bss_size as usize)?;

    // .text is never a legitimate relocation target — lock it down the same
    // way the NRO loader does, so a wild guest write faults immediately.
    mem.mark_readonly(text_addr, ro_addr);

    Ok(LoadedNso {
        base,
        entry: text_addr.wrapping_add(entry_offset(&text_bytes)),
        text: Segment {
            file_offset: text.file_offset,
            file_size: text.decompressed_size,
            mem_addr: text_addr,
        },
        ro: Segment {
            file_offset: rodata.file_offset,
            file_size: rodata.decompressed_size,
            mem_addr: ro_addr,
        },
        data: Segment {
            file_offset: raw_data.file_offset,
            file_size: raw_data.decompressed_size,
            mem_addr: data_addr,
        },
        bss_size,
        build_id,
    })
}

fn extract_segment(
    data: &[u8],
    seg: &RawSegment,
    compressed_size: u32,
    is_compressed: bool,
    name: &str,
) -> Result<Vec<u8>> {
    let on_disk_size = if is_compressed {
        compressed_size
    } else {
        seg.decompressed_size
    } as usize;
    let start = seg.file_offset as usize;
    let end = start
        .checked_add(on_disk_size)
        .ok_or(Error::Overflow)?;
    if end > data.len() {
        return Err(Error::Nso(format!(
            "{} segment [{:#x}, {:#x}) exceeds input data ({} bytes)",
            name, start, end, data.len()
        )));
    }
    let raw = &data[start..end];
    if is_compressed {
        crate::lz4::decompress_block(raw, seg.decompressed_size as usize)
            .map_err(|e| Error::Nso(format!("{} LZ4 decompress failed: {}", name, e)))
    } else {
        Ok(raw.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_nso(text: &[u8], rodata: &[u8], data: &[u8], bss: u32) -> Vec<u8> {
        let text_off = HEADER_SIZE;
        let ro_off = text_off + text.len();
        let data_off = ro_off + rodata.len();
        let mut out = vec![0u8; data_off + data.len()];
        out[0..4].copy_from_slice(&NSO0_MAGIC.to_le_bytes());
        out[0x0C..0x10].copy_from_slice(&0u32.to_le_bytes()); // flags: nothing compressed
        out[0x10..0x14].copy_from_slice(&(text_off as u32).to_le_bytes());
        out[0x14..0x18].copy_from_slice(&0u32.to_le_bytes()); // text mem offset
        out[0x18..0x1C].copy_from_slice(&(text.len() as u32).to_le_bytes());
        out[0x20..0x24].copy_from_slice(&(ro_off as u32).to_le_bytes());
        out[0x24..0x28].copy_from_slice(&(text.len() as u32).to_le_bytes()); // ro mem offset
        out[0x28..0x2C].copy_from_slice(&(rodata.len() as u32).to_le_bytes());
        out[0x30..0x34].copy_from_slice(&(data_off as u32).to_le_bytes());
        out[0x34..0x38].copy_from_slice(&((text.len() + rodata.len()) as u32).to_le_bytes()); // data mem offset
        out[0x38..0x3C].copy_from_slice(&(data.len() as u32).to_le_bytes());
        out[0x3C..0x40].copy_from_slice(&bss.to_le_bytes());
        out[0x60..0x64].copy_from_slice(&(text.len() as u32).to_le_bytes());
        out[0x64..0x68].copy_from_slice(&(rodata.len() as u32).to_le_bytes());
        out[0x68..0x6C].copy_from_slice(&(data.len() as u32).to_le_bytes());
        out[text_off..text_off + text.len()].copy_from_slice(text);
        out[ro_off..ro_off + rodata.len()].copy_from_slice(rodata);
        out[data_off..data_off + data.len()].copy_from_slice(data);
        out
    }

    #[test]
    fn loads_uncompressed_segments() {
        let text = [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00];
        let rodata = [0xAA, 0xBB, 0xCC, 0xDD];
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        let nso = build_nso(&text, &rodata, &data, 0x100);
        let mut mem = Memory::new();
        let loaded = load_nso(&mut mem, &nso, NSO_BASE).unwrap();
        // No ModulePtr/MOD0 signature in this synthetic .text: entry is the
        // raw start, same as rtld's real-world layout.
        assert_eq!(loaded.entry, NSO_BASE);
        assert_eq!(mem.read_u32(NSO_BASE).unwrap(), 0x01);
        assert_eq!(mem.read_u32(loaded.ro.mem_addr).unwrap(), 0xDDCCBBAA);
        assert_eq!(mem.read_u32(loaded.data.mem_addr).unwrap(), 0xEFBEADDE);
        // BSS zero-filled right after .data.
        assert_eq!(
            mem.read_u8(loaded.data.mem_addr + loaded.data.file_size).unwrap(),
            0
        );
    }

    #[test]
    fn text_is_read_only() {
        let text = [0x01, 0x00, 0x00, 0x00];
        let nso = build_nso(&text, &[], &[0xEFu8], 0);
        let mut mem = Memory::new();
        let loaded = load_nso(&mut mem, &nso, NSO_BASE).unwrap();
        assert!(mem.write_u32(loaded.text.mem_addr, 0xDEAD_BEEF).is_err());
    }

    #[test]
    fn decompresses_lz4_segments() {
        // A trivially "compressed" .text: one literal-only LZ4 sequence.
        let plain_text: Vec<u8> = (0..40u8).collect();
        let mut compressed = vec![0xf0u8, (plain_text.len() - 15) as u8];
        compressed.extend_from_slice(&plain_text);

        let text_off = HEADER_SIZE;
        let mut out = vec![0u8; text_off + compressed.len()];
        out[0..4].copy_from_slice(&NSO0_MAGIC.to_le_bytes());
        out[0x0C..0x10].copy_from_slice(&FLAG_TEXT_COMPRESSED.to_le_bytes());
        out[0x10..0x14].copy_from_slice(&(text_off as u32).to_le_bytes());
        out[0x14..0x18].copy_from_slice(&0u32.to_le_bytes());
        out[0x18..0x1C].copy_from_slice(&(plain_text.len() as u32).to_le_bytes());
        out[0x60..0x64].copy_from_slice(&(compressed.len() as u32).to_le_bytes());
        out[text_off..].copy_from_slice(&compressed);

        let mut mem = Memory::new();
        let loaded = load_nso(&mut mem, &out, NSO_BASE).unwrap();
        for (i, &b) in plain_text.iter().enumerate() {
            assert_eq!(mem.read_u8(loaded.text.mem_addr + i as u32).unwrap(), b);
        }
    }

    #[test]
    fn entry_skips_modptr_mod0_header_when_present() {
        let mut text = vec![0u8; 0x40];
        text[4..8].copy_from_slice(&8u32.to_le_bytes()); // ModulePtr -> MOD0 at +8
        text[8..12].copy_from_slice(&MOD0_MAGIC.to_le_bytes());
        let nso = build_nso(&text, &[], &[], 0);
        let mut mem = Memory::new();
        let loaded = load_nso(&mut mem, &nso, NSO_BASE).unwrap();
        assert_eq!(loaded.entry, NSO_BASE + NSO_ENTRY_OFFSET);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut nso = build_nso(&[0u8; 4], &[], &[], 0);
        nso[0] = b'X';
        assert!(matches!(load_nso(&mut Memory::new(), &nso, NSO_BASE), Err(Error::BadMagic { .. })));
    }

    #[test]
    fn rejects_truncated() {
        assert!(matches!(
            load_nso(&mut Memory::new(), &[0u8; 0x20], NSO_BASE),
            Err(Error::Truncated { .. })
        ));
    }
}
