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

use crate::source::{ByteSource, SliceSource};
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
    /// Total size of the image in bytes (used for bounds checks). `u64`, not
    /// `usize`: a retail `.nsp` is routinely larger than a wasm32 address
    /// space, and truncating its size here let entries past the 4 GiB mark
    /// pass a bounds check they should have failed.
    pub image_size: u64,
}

impl Pfs0 {
    /// Parse a PFS0 image from raw bytes.
    ///
    /// Returns an error if the magic does not match, the image is too small
    /// for the declared header, or any file entry falls outside the image.
    pub fn parse(data: &[u8]) -> Result<Pfs0, Error> {
        Pfs0::read_from(&SliceSource(data))
    }

    /// Parse a PFS0 image out of a [`ByteSource`], reading only its header.
    ///
    /// This is the form a multi-gigabyte `.nsp` is read with: the header
    /// (magic, entry table and string table) is a few kilobytes at the front
    /// of the file, and nothing else has to be in memory to know what the
    /// container holds or where each file starts.
    pub fn read_from<S: ByteSource>(src: &S) -> Result<Pfs0, Error> {
        const HEADER_SIZE: usize = 0x10;
        let mut head = [0u8; HEADER_SIZE];
        let got = src.read_at(0, &mut head)?;
        if got < HEADER_SIZE {
            return Err(Error::Truncated {
                what: "PFS0 header".into(),
                expected: HEADER_SIZE,
                got,
            });
        }
        if read_u32(&head, 0) != PFS0_MAGIC {
            return Err(Error::BadMagic {
                what: "PFS0".into(),
                found: read_u32(&head, 0),
            });
        }

        let file_count = read_u32(&head, 0x04) as u64;
        let string_table_size = read_u32(&head, 0x08) as u64;

        // Header + entries + string table must fit within the image. Every
        // term comes from a `u32`, so the `u64` arithmetic cannot overflow —
        // which is the point of doing it in `u64` on a 32-bit target, where
        // `file_count * FILE_ENTRY_SIZE` alone can exceed `usize`.
        let table_start = HEADER_SIZE as u64;
        let strings_start = table_start + file_count * FILE_ENTRY_SIZE as u64;
        let strings_end = strings_start + string_table_size;
        if strings_end > src.len() {
            return Err(Error::Truncated {
                what: "PFS0 string table".into(),
                expected: usize::try_from(strings_end).unwrap_or(usize::MAX),
                got: usize::try_from(src.len()).unwrap_or(usize::MAX),
            });
        }
        // Only the header region: for a retail container the rest is
        // gigabytes, and none of it is needed to build the file table.
        let header = src.read_vec(0, strings_end)?;
        let strings_start = strings_start as usize;

        // Not `with_capacity(file_count)`: the count is whatever the file
        // says, and reserving for a corrupt one is an allocation the target
        // aborts on rather than an error anyone can read.
        let mut files = Vec::new();
        // Most PFS0 images store file offsets relative to the file start, but
        // some repack tools emit offsets relative to the end of the header +
        // string table (the payload area). No file can legitimately overlap
        // the header, so if an entry points inside it, rebase every offset by
        // the payload start.
        let payload_base = strings_end;
        let mut rebase = false;
        for i in 0..file_count as usize {
            let entry = HEADER_SIZE + i * FILE_ENTRY_SIZE;
            let offset = read_u64(&header, entry);
            let size = read_u64(&header, entry + 8);
            let name_off = read_u32(&header, entry + 16) as usize;
            if offset < payload_base {
                rebase = true;
            }
            // The name has to be NUL-terminated inside the string table —
            // `header` ends where the table does, so a name offset pointing
            // past it fails here instead of running into the payload.
            let name = read_cstr(&header, strings_start.saturating_add(name_off))
                .ok_or(Error::BadStringTable {
                    index: i,
                    offset: name_off,
                })?
                .to_string();
            files.push(Pfs0File {
                offset,
                size,
                name,
            });
        }
        if rebase {
            for f in files.iter_mut() {
                f.offset = f.offset.saturating_add(payload_base);
            }
        }
        for (i, f) in files.iter().enumerate() {
            let end = f
                .offset
                .checked_add(f.size)
                .ok_or(Error::Overflow)?;
            if end > src.len() {
                return Err(Error::FileOutOfBounds {
                    index: i,
                    name: f.name.clone(),
                    offset: f.offset,
                    size: f.size,
                    image_size: src.len(),
                });
            }
        }

        Ok(Pfs0 {
            files,
            image_size: src.len(),
        })
    }

    /// A [`ByteSource`] over file `index`'s payload, addressed from 0.
    ///
    /// This is how an NCA inside a container is read without extracting it:
    /// the window is a view of the container source, not a copy.
    pub fn file_source<S: ByteSource>(
        &self,
        src: S,
        index: usize,
    ) -> Result<crate::source::Window<S>, Error> {
        let f = self
            .files
            .get(index)
            .ok_or_else(|| Error::Nca(format!("no file at index {} in this PFS0", index)))?;
        crate::source::Window::new(src, f.offset, f.size, &f.name)
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

    #[test]
    fn rebases_offsets_relative_to_payload() {
        // Some repack tools emit PFS0 entries whose offsets are relative to
        // the end of the string table (the payload area) instead of the file
        // start. Build one: entries claim offset 0/4, but the real payloads
        // live after the header + string table.
        let mut image = Vec::new();
        image.extend_from_slice(&PFS0_MAGIC.to_le_bytes());
        image.extend_from_slice(&1u32.to_le_bytes()); // file count
        image.extend_from_slice(&4u32.to_le_bytes()); // string table size
        image.extend_from_slice(&0u32.to_le_bytes());
        image.extend_from_slice(&0u64.to_le_bytes()); // offset (relative)
        image.extend_from_slice(&4u64.to_le_bytes()); // size
        image.extend_from_slice(&0u32.to_le_bytes()); // name offset
        image.extend_from_slice(&0u32.to_le_bytes()); // padding
        image.extend_from_slice(b"x\0\0\0");
        let payload_base = image.len(); // header + entry + string table
        image.extend_from_slice(b"DATA");
        let pfs0 = Pfs0::parse(&image).unwrap();
        assert_eq!(pfs0.files.len(), 1);
        assert_eq!(pfs0.files[0].offset, payload_base as u64);
        assert_eq!(&image[pfs0.files[0].offset as usize..][..4], b"DATA");
    }

    /// A container far larger than a wasm32 address space, without needing
    /// one: it answers `len()` with 5 GiB and serves the header from a real
    /// buffer, so the entry table can point past the 4 GiB mark.
    #[derive(Debug)]
    struct HugeSource {
        header: Vec<u8>,
        len: u64,
    }

    impl ByteSource for HugeSource {
        fn len(&self) -> u64 {
            self.len
        }
        fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
            SliceSource(&self.header).read_at(offset, out)
        }
    }

    fn huge_container(entry_offset: u64, entry_size: u64, len: u64) -> HugeSource {
        let mut header = Vec::new();
        header.extend_from_slice(&PFS0_MAGIC.to_le_bytes());
        header.extend_from_slice(&1u32.to_le_bytes()); // file count
        header.extend_from_slice(&9u32.to_le_bytes()); // string table size
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&entry_offset.to_le_bytes());
        header.extend_from_slice(&entry_size.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes()); // name offset
        header.extend_from_slice(&0u32.to_le_bytes()); // padding
        header.extend_from_slice(b"main.nca\0");
        HugeSource { header, len }
    }

    #[test]
    fn entries_past_the_four_gib_mark_keep_their_offsets() {
        // The offset a retail container's program NCA actually lives at. Held
        // in a `usize` (as this parser used to), it truncates to 0x1000 and
        // every read lands on the wrong bytes; the bounds check below passes
        // for the same reason, so nothing catches it.
        const PAST_4GIB: u64 = 0x1_0000_1000;
        let src = huge_container(PAST_4GIB, 0x2000, 5 << 30);
        let pfs0 = Pfs0::read_from(&src).unwrap();
        assert_eq!(pfs0.image_size, 5 << 30);
        assert_eq!(pfs0.files[0].name, "main.nca");
        assert_eq!(pfs0.files[0].offset, PAST_4GIB);
        assert_eq!(pfs0.files[0].size, 0x2000);
    }

    #[test]
    fn an_entry_running_past_the_end_is_still_caught_past_four_gib() {
        // Same shape, but the extent ends one byte past the container. The
        // truncating version of this check compared wrapped values and let it
        // through.
        let len = 5u64 << 30;
        let src = huge_container(len - 0x1000, 0x1001, len);
        assert!(matches!(
            Pfs0::read_from(&src),
            Err(Error::FileOutOfBounds { .. })
        ));
    }
}
