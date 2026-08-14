//! PFS0 container format, the on-disk layout behind `.nsp` files.
//!
//! A PFS0 image begins with a header:
//!
//! ```text
//! offset  size  field
//! 0x00    4     magic "PFS0" (0x30534650)
//! 0x04    4     number of files
//! 0x08    4     size of the string table
//! 0x0C    4     reserved (must be 0)
//! 0x10    -     FileEntry[file_count]   (16 bytes each)
//! -       -     string table
//! -       -     file data
//! ```
//!
//! Each `FileEntry` is 16 bytes: `u64 offset`, `u64 size`, `u32 name_offset`,
//! `u32 padding`. `offset`/`size` reference the file payload, `name_offset`
//! references a NUL-terminated string in the string table. All offsets are
//! relative to the start of the PFS0 header.

use crate::Error;

pub const PFS0_MAGIC: u32 = 0x3053_4650; // "PFS0"
/// Each entry: u64 offset, u64 size, u32 name_offset, u32 padding.
pub const FILE_ENTRY_SIZE: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pfs0File {
    /// Byte offset of the file payload, relative to the PFS0 header start.
    pub offset: u64,
    /// Payload size in bytes.
    pub size: u64,
    /// File name (without the trailing NUL).
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pfs0 {
    pub files: Vec<Pfs0File>,
    /// Total size of the image in bytes (used for bounds checks).
    pub image_size: usize,
}

impl Pfs0 {
    /// Parse a PFS0 image from raw bytes.
    ///
    /// Returns an error if the magic does not match, the image is too small
    /// for the declared header, or any file entry falls outside the image.
    pub fn parse(data: &[u8]) -> Result<Pfs0, Error> {
        const HEADER_SIZE: usize = 0x10;
        if data.len() < HEADER_SIZE {
            return Err(Error::Truncated {
                what: "PFS0 header".into(),
                expected: HEADER_SIZE,
                got: data.len(),
            });
        }
        if read_u32(data, 0) != PFS0_MAGIC {
            return Err(Error::BadMagic {
                what: "PFS0".into(),
                found: read_u32(data, 0),
            });
        }

        let file_count = read_u32(data, 0x04) as usize;
        let string_table_size = read_u32(data, 0x08) as usize;

        // Header + entries + string table must fit within the image.
        let table_start = HEADER_SIZE;
        let table_end = table_start
            .checked_add(file_count.saturating_mul(FILE_ENTRY_SIZE))
            .ok_or(Error::Overflow)?;
        let strings_start = table_end;
        let strings_end = strings_start
            .checked_add(string_table_size)
            .ok_or(Error::Overflow)?;
        if strings_end > data.len() {
            return Err(Error::Truncated {
                what: "PFS0 string table".into(),
                expected: strings_end,
                got: data.len(),
            });
        }

        let mut files = Vec::with_capacity(file_count);
        for i in 0..file_count {
            let entry = table_start + i * FILE_ENTRY_SIZE;
            let offset = read_u64(data, entry);
            let size = read_u64(data, entry + 8);
            let name_off = read_u32(data, entry + 16) as usize;
            let name = read_cstr(data, strings_start + name_off)
                .ok_or(Error::BadStringTable {
                    index: i,
                    offset: name_off,
                })?
                .to_string();

            let end = offset
                .checked_add(size)
                .ok_or(Error::Overflow)?;
            if end as usize > data.len() {
                return Err(Error::FileOutOfBounds {
                    index: i,
                    name: name.clone(),
                    offset,
                    size,
                    image_size: data.len(),
                });
            }
            files.push(Pfs0File { offset, size, name });
        }

        Ok(Pfs0 {
            files,
            image_size: data.len(),
        })
    }

    /// Find a file by exact name.
    pub fn find(&self, name: &str) -> Option<&Pfs0File> {
        self.files.iter().find(|f| f.name == name)
    }

    /// Find a file whose name ends with `suffix` (case-insensitive).
    pub fn find_with_suffix(&self, suffix: &str) -> Option<&Pfs0File> {
        let suffix = suffix.to_ascii_lowercase();
        self.files
            .iter()
            .find(|f| f.name.to_ascii_lowercase().ends_with(&suffix))
    }
}

pub(crate) fn read_u32(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([
        data[at],
        data[at + 1],
        data[at + 2],
        data[at + 3],
    ])
}

pub(crate) fn read_u64(data: &[u8], at: usize) -> u64 {
    u64::from_le_bytes([
        data[at],
        data[at + 1],
        data[at + 2],
        data[at + 3],
        data[at + 4],
        data[at + 5],
        data[at + 6],
        data[at + 7],
    ])
}

/// Read a NUL-terminated string, returning `None` if no NUL is found within
/// the remaining buffer.
pub(crate) fn read_cstr(data: &[u8], at: usize) -> Option<&str> {
    if at >= data.len() {
        return None;
    }
    let end = data[at..].iter().position(|&b| b == 0).map(|p| at + p)?;
    std::str::from_utf8(&data[at..end]).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_pfs0(files: &[(&str, &[u8])]) -> Vec<u8> {
        // Layout: header (0x10) + entries + string table, then payloads.
        let file_count = files.len();
        let mut names = String::new();
        let mut entries = Vec::new();
        for (i, (name, _)) in files.iter().enumerate() {
            // PFS0 name offsets are relative to the start of the string table.
            let name_offset = names.len();
            names.push_str(name);
            names.push('\0');
            entries.push((name_offset, i));
        }
        let strings_start = 0x10 + file_count * FILE_ENTRY_SIZE;
        let mut image = vec![0u8; strings_start + names.len()];
        image[0..4].copy_from_slice(&PFS0_MAGIC.to_le_bytes());
        image[4..8].copy_from_slice(&(file_count as u32).to_le_bytes());
        image[8..12].copy_from_slice(&(names.len() as u32).to_le_bytes());
        image[strings_start..].copy_from_slice(names.as_bytes());

        let mut payload_start = image.len();
        for (i, (_, payload)) in files.iter().enumerate() {
            let (name_offset, _) = entries[i];
            let entry = 0x10 + i * FILE_ENTRY_SIZE;
            image[entry..entry + 8].copy_from_slice(&(payload_start as u64).to_le_bytes());
            image[entry + 8..entry + 16]
                .copy_from_slice(&(payload.len() as u64).to_le_bytes());
            image[entry + 16..entry + 20].copy_from_slice(&(name_offset as u32).to_le_bytes());
            image.extend_from_slice(payload);
            payload_start += payload.len();
        }
        image
    }

    #[test]
    fn parses_empty_container() {
        let data = build_pfs0(&[]);
        let pfs0 = Pfs0::parse(&data).unwrap();
        assert!(pfs0.files.is_empty());
    }

    #[test]
    fn parses_files_with_offsets() {
        let data = build_pfs0(&[("main.nca", &[1, 2, 3]), ("a.bin", &[9])]);
        let pfs0 = Pfs0::parse(&data).unwrap();
        assert_eq!(pfs0.files.len(), 2);
        assert_eq!(pfs0.files[0].name, "main.nca");
        assert_eq!(pfs0.files[0].size, 3);
        assert_eq!(&data[pfs0.files[0].offset as usize..][..3], &[1, 2, 3]);
        assert_eq!(pfs0.files[1].name, "a.bin");
        assert_eq!(&data[pfs0.files[1].offset as usize..][..1], &[9]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = build_pfs0(&[]);
        data[0] = b'X';
        assert!(matches!(Pfs0::parse(&data), Err(Error::BadMagic { .. })));
    }

    #[test]
    fn rejects_truncated_string_table() {
        let mut data = build_pfs0(&[("main.nca", &[1])]);
        data.truncate(30);
        assert!(matches!(Pfs0::parse(&data), Err(Error::Truncated { .. })));
    }

    #[test]
    fn rejects_file_out_of_bounds() {
        let mut data = build_pfs0(&[("main.nca", &[1, 2, 3])]);
        // Claim the file is much larger than the image.
        data[0x18..0x20].copy_from_slice(&0xFFFFu64.to_le_bytes());
        assert!(matches!(
            Pfs0::parse(&data),
            Err(Error::FileOutOfBounds { .. })
        ));
    }

    #[test]
    fn find_and_suffix_lookup() {
        let data = build_pfs0(&[
            ("main.nca", &[1]),
            ("update.nca", &[2]),
            ("readme.txt", &[3]),
        ]);
        let pfs0 = Pfs0::parse(&data).unwrap();
        assert_eq!(pfs0.find("update.nca").unwrap().name, "update.nca");
        assert_eq!(pfs0.find("nope").is_none(), true);
        assert_eq!(pfs0.find_with_suffix(".NCA").unwrap().name, "main.nca");
    }
}
