//! NCA container header reader.
//!
//! An NCA file is Nintendo's encryption wrapper around a games's filesystem
//! images. Phase 0 only reads the *header*, which is stored in the clear and
//! describes the file's metadata, content type and section layout. Decrypting
//! the body is intentionally out of scope (it requires console keys we do not
//! handle).
//!
//! Header layout (relative to the NCA start):
//!
//! ```text
//! 0x200  magic "NCA3" (u32)
//! 0x204  distribution type (u8)
//! 0x205  content type (u8)
//! 0x210  title id (u64)
//! 0x214  sdk version (u32)
//! 0x218  crypto type (u8)
//! 0x219  ...key area / tables...
//! 0x240  section table header entry 0 (16 bytes)
//! 0x250  section table header entry 1 (16 bytes)
//! 0x260  section table header entry 2 (16 bytes)
//! 0x270  section table header entry 3 (16 bytes)
//! 0x340  file size (u64)
//! 0x348  program id (u64) / title id
//! ```
//!
//! Section header entries describe the backing filesystem image: its offset
//! into the NCA, total size, partition index and type (PFS0/ROMFS/...).

use crate::Error;

pub const NCA_MAGIC: u32 = 0x3341_434e; // "NCA3"
pub const NCA_HEADER_OFFSET: usize = 0x200;
pub const SECTION_HEADER_COUNT: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ContentType {
    Program = 0,
    Meta = 1,
    Control = 2,
    Manual = 3,
    Data = 4,
    Unknown(u8),
}

impl ContentType {
    pub fn from_u8(v: u8) -> ContentType {
        match v {
            0 => ContentType::Program,
            1 => ContentType::Meta,
            2 => ContentType::Control,
            3 => ContentType::Manual,
            4 => ContentType::Data,
            other => ContentType::Unknown(other),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ContentType::Program => "Program",
            ContentType::Meta => "Meta",
            ContentType::Control => "Control",
            ContentType::Manual => "Manual",
            ContentType::Data => "Data",
            ContentType::Unknown(_) => "Unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionHeader {
    /// Offset of the section's filesystem image within the NCA.
    pub media_offset: u64,
    /// Total section size.
    pub media_size: u64,
    /// Which of the 4 possible partitions the image belongs to.
    pub partition_index: u8,
    /// Whether the filesystem type is PFS0 (0) or ROMFS (1).
    pub fs_type: u8,
    /// Hash region size (bytes).
    pub hash_region_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nca {
    pub distribution_type: u8,
    pub content_type: ContentType,
    pub content_type_raw: u8,
    pub title_id: u64,
    pub sdk_version: u32,
    pub crypto_type: u8,
    pub sections: Vec<SectionHeader>,
    pub file_size: u64,
    pub program_id: u64,
}

impl Nca {
    /// Parse an NCA header from a full NCA buffer.
    pub fn parse(data: &[u8]) -> Result<Nca, Error> {
        const HEADER_SIZE: usize = 0x400;
        if data.len() < HEADER_SIZE {
            return Err(Error::Truncated {
                what: "NCA header".into(),
                expected: HEADER_SIZE,
                got: data.len(),
            });
        }
        let h = NCA_HEADER_OFFSET;
        let magic = crate::nsp::read_u32(data, h);
        if magic != NCA_MAGIC {
            return Err(Error::BadMagic {
                what: "NCA".into(),
                found: magic,
            });
        }

        let content_type_raw = data[h + 0x05];
        let mut sections = Vec::with_capacity(SECTION_HEADER_COUNT);
        for i in 0..SECTION_HEADER_COUNT {
            let at = h + 0x40 + i * 0x10;
            sections.push(SectionHeader {
                media_offset: crate::nsp::read_u64(data, at),
                media_size: crate::nsp::read_u64(data, at + 8),
                partition_index: data[at + 16] & 3,
                fs_type: data[at + 16] >> 2,
                hash_region_size: 0x4000, // reserved region at the start of each FS image
            });
        }

        Ok(Nca {
            distribution_type: data[h + 0x04],
            content_type: ContentType::from_u8(content_type_raw),
            content_type_raw,
            title_id: crate::nsp::read_u64(data, h + 0x10),
            sdk_version: crate::nsp::read_u32(data, h + 0x18),
            crypto_type: data[h + 0x1C],
            sections,
            file_size: crate::nsp::read_u64(data, h + 0x140),
            program_id: crate::nsp::read_u64(data, h + 0x148),
        })
    }

    /// Whether the file body is encrypted. In practice every retail NCA is,
    /// and the key data required to decrypt lives in the header's key area.
    pub fn is_encrypted(&self) -> bool {
        self.crypto_type != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nca() -> Vec<u8> {
        let mut data = vec![0u8; 0x400];
        let h = NCA_HEADER_OFFSET;
        data[h..h + 4].copy_from_slice(&NCA_MAGIC.to_le_bytes());
        data[h + 0x04] = 0; // distribution type: downloadable
        data[h + 0x05] = 0; // content type: program
        data[h + 0x10..h + 0x18].copy_from_slice(&0x0100_0000_0010_5A00u64.to_le_bytes());
        data[h + 0x18..h + 0x1C].copy_from_slice(&0x0001_000Au32.to_le_bytes());
        data[h + 0x1C] = 0x01; // crypto type
        // section 0: a PFS0 image starting at 0x0, 0x2000 bytes
        data[h + 0x40..h + 0x48].copy_from_slice(&0u64.to_le_bytes());
        data[h + 0x48..h + 0x50].copy_from_slice(&0x2000u64.to_le_bytes());
        data[h + 0x50] = 0; // partition 0, fs_type PFS0
        data[h + 0x140..h + 0x148].copy_from_slice(&0x12345u64.to_le_bytes());
        data[h + 0x148..h + 0x150].copy_from_slice(&0x0100_0000_0010_5A00u64.to_le_bytes());
        data
    }

    #[test]
    fn parses_nca_header() {
        let nca = Nca::parse(&make_nca()).unwrap();
        assert_eq!(nca.content_type, ContentType::Program);
        assert_eq!(nca.title_id, 0x0100_0000_0010_5A00);
        assert_eq!(nca.sdk_version, 0x0001_000A);
        assert_eq!(nca.crypto_type, 0x01);
        assert!(nca.is_encrypted());
        assert_eq!(nca.file_size, 0x12345);
        assert_eq!(nca.sections.len(), 4);
        assert_eq!(nca.sections[0].media_offset, 0);
        assert_eq!(nca.sections[0].media_size, 0x2000);
        assert_eq!(nca.sections[0].fs_type, 0); // PFS0
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = make_nca();
        data[NCA_HEADER_OFFSET] = b'X';
        assert!(matches!(Nca::parse(&data), Err(Error::BadMagic { .. })));
    }

    #[test]
    fn rejects_short_buffer() {
        assert!(matches!(
            Nca::parse(&[0u8; 0x300]),
            Err(Error::Truncated { .. })
        ));
    }
}
