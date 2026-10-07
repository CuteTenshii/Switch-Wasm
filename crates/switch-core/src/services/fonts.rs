//! Shared fonts served through `pl:u`.

use crate::cpu::*;
use crate::IdMap;

/// One shared font in pl's shared memory; `offset` points past the 8-byte header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FontRegion {
    pub offset: u32,
    pub size: u32,
}

/// Shared fonts in `PlSharedFontType` order (archive id, file name), plus
/// `nintendo_ext2_003` seventh, matching Eden's `SHARED_FONTS`.
const SHARED_FONTS: [(u64, &str); 7] = [
    (0x0100_0000_0000_0811, "/nintendo_udsg-r_std_003.bfttf"),
    (
        0x0100_0000_0000_0814,
        "/nintendo_udsg-r_org_zh-cn_003.bfttf",
    ),
    (
        0x0100_0000_0000_0814,
        "/nintendo_udsg-r_ext_zh-cn_003.bfttf",
    ),
    (0x0100_0000_0000_0813, "/nintendo_udjxh-db_zh-tw_003.bfttf"),
    (0x0100_0000_0000_0812, "/nintendo_udsg-r_ko_003.bfttf"),
    (0x0100_0000_0000_0810, "/nintendo_ext_003.bfttf"),
    (0x0100_0000_0000_0810, "/nintendo_ext2_003.bfttf"),
];

/// A `.bfttf`'s first four bytes; the xor key is derived from them.
const BFTTF_MAGIC: [u8; 4] = [0x36, 0xf8, 0x1a, 0x1e];
const BFTTF_KEY: [u8; 4] = [0x49, 0x62, 0x18, 0x06];

const BFTTF_HEADER: usize = 8;

/// Decode a `.bfttf` into header plus TrueType file. The size field stays byte-reversed, as on a console.
pub fn decode_bfttf(file: &[u8]) -> Option<Vec<u8>> {
    let len = file.len() / 4 * 4;
    if len < BFTTF_HEADER || file[..4] != BFTTF_MAGIC {
        return None;
    }
    let mut out: Vec<u8> = file[..len]
        .iter()
        .zip(BFTTF_KEY.iter().cycle())
        .map(|(b, k)| b ^ k)
        .collect();
    out[4..8].copy_from_slice(&[file[7], file[6], file[5], file[4]]);
    Some(out)
}

/// Wrap a TrueType file as a `.bfttf`, padding (not trimming) to whole words.
pub fn encode_bfttf(ttf: &[u8]) -> Vec<u8> {
    let len = ttf.len().next_multiple_of(4);
    let mut out = Vec::with_capacity(len + BFTTF_HEADER);
    out.extend_from_slice(&[0x7f, 0x9a, 0x02, 0x18]);
    out.extend_from_slice(&(len as u32).to_be_bytes());
    out.extend_from_slice(ttf);
    out.resize(len + BFTTF_HEADER, 0);
    for (i, b) in out.iter_mut().enumerate() {
        *b ^= BFTTF_KEY[i % 4];
    }
    out
}

impl Cpu {
    /// Set the font `pl:u` serves for every shared font type (TrueType/OpenType).
    pub fn set_shared_font(&mut self, font: Vec<u8>) {
        self.shared_font = font;
        self.pl_shmem_image.clear();
        self.shared_font_regions.clear();
        // Refill in place if the guest already mapped the region.
        if self.pl_shmem_addr != 0 {
            self.write_shared_font(self.pl_shmem_addr);
        }
    }

    pub fn shared_font_len(&self) -> usize {
        self.shared_font.len()
    }

    /// Assemble pl's shared memory lazily: firmware `.bfttf` fonts, or the host font
    /// wrapped the same way in every slot.
    pub(crate) fn build_shared_fonts(&mut self) {
        if !self.shared_font_regions.is_empty() {
            return;
        }
        // Read each archive at most once: two hold two fonts.
        let mut archives: IdMap<u64, Vec<u8>> = IdMap::default();
        for (id, _) in SHARED_FONTS {
            if archives.contains_key(&id) {
                continue;
            }
            let Some(src) = self.data_archives.get(&id) else {
                continue;
            };
            let mut image = vec![0u8; src.len() as usize];
            if src.read_at(0, &mut image).is_err() {
                continue;
            }
            archives.insert(id, image);
        }

        for (id, name) in SHARED_FONTS {
            let font = archives
                .get(&id)
                .and_then(|image| crate::romfs::RomFs::parse(image).ok()?.read_path(name))
                .and_then(decode_bfttf);
            let Some(font) = font else { continue };
            self.push_shared_font(&font);
        }

        if self.shared_font_regions.is_empty() && !self.shared_font.is_empty() {
            // No firmware fonts: the host font stands in for every type.
            let font = decode_bfttf(&encode_bfttf(&self.shared_font.clone()));
            if let Some(font) = font {
                for _ in 0..SHARED_FONTS.len() {
                    self.push_shared_font(&font);
                }
            }
        }
        if crate::trace::enabled(crate::trace::Trace::Font) {
            crate::traceln!("[pl] {} bytes of shared font", self.pl_shmem_image.len());
            for (i, region) in self.shared_font_regions.iter().enumerate() {
                crate::traceln!(
                    "[pl]  type {i}: offset={:#x} size={:#x}",
                    region.offset,
                    region.size
                );
            }
        }
    }

    /// Append a decoded font; one that does not fit is dropped, not truncated.
    fn push_shared_font(&mut self, font: &[u8]) {
        let offset = self.pl_shmem_image.len();
        if offset + font.len() > PL_SHMEM_SIZE as usize {
            return;
        }
        self.pl_shmem_image.extend_from_slice(font);
        self.shared_font_regions.push(FontRegion {
            offset: (offset + BFTTF_HEADER) as u32,
            size: (font.len() - BFTTF_HEADER) as u32,
        });
    }

    #[cfg(test)]
    pub(crate) fn shared_font_image(&mut self) -> &[u8] {
        self.build_shared_fonts();
        &self.pl_shmem_image
    }

    pub(crate) fn shared_font_regions(&mut self) -> &[FontRegion] {
        self.build_shared_fonts();
        &self.shared_font_regions
    }

    /// Copy the shared fonts into pl's shared memory at `addr`.
    pub(crate) fn write_shared_font(&mut self, addr: u32) {
        self.build_shared_fonts();
        let image = std::mem::take(&mut self.pl_shmem_image);
        let _ = self.mem.map(addr, &image);
        self.pl_shmem_image = image;
    }
}
