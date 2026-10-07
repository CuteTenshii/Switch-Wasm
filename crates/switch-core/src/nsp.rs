//! PFS0 container format, the on-disk layout behind `.nsp` files.
//!
//! Header layout:
//!
//! ```text
//! offset  size  field
//! 0x00    4     magic "PFS0" (0x30534650)
//! 0x04    4     number of files
//! 0x08    4     size of the string table
//! 0x0C    4     reserved (must be 0)
//! 0x10    -     FileEntry[file_count]   (24 bytes each)
//! -       -     string table
//! -       -     file data
//! ```
//!
//! Entry offsets are counted from the end of the string table. XCI partitions
//! ([`crate::xci`]) use the same table under the magic "HFS0" with 64-byte entries.

use crate::source::{ByteSource, SliceSource};
use crate::Error;

pub const PFS0_MAGIC: u32 = 0x3053_4650; // "PFS0"
/// Each entry: u64 offset, u64 size, u32 name_offset, u32 padding.
pub const FILE_ENTRY_SIZE: usize = 24;
pub const HFS0_MAGIC: u32 = 0x3053_4648; // "HFS0"
/// PFS0 entry fields plus `u32 hashed_region_size`, 8 reserved bytes and a SHA-256.
pub const HFS0_ENTRY_SIZE: usize = 0x40;

/// `PFS0` (`.nsp`, ExeFS) or `HFS0` (XCI); only the magic and entry stride differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionKind {
    Pfs0,
    Hfs0,
}

impl PartitionKind {
    pub fn magic(self) -> u32 {
        match self {
            PartitionKind::Pfs0 => PFS0_MAGIC,
            PartitionKind::Hfs0 => HFS0_MAGIC,
        }
    }

    pub fn entry_size(self) -> usize {
        match self {
            PartitionKind::Pfs0 => FILE_ENTRY_SIZE,
            PartitionKind::Hfs0 => HFS0_ENTRY_SIZE,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PartitionKind::Pfs0 => "PFS0",
            PartitionKind::Hfs0 => "HFS0",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pfs0File {
    /// Offset of the payload from the start of the image.
    pub offset: u64,
    pub size: u64,
    /// Without the trailing NUL.
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pfs0 {
    pub files: Vec<Pfs0File>,
    /// `u64`: a retail `.nsp` can exceed the wasm32 address space.
    pub image_size: u64,
}

impl Pfs0 {
    pub fn parse(data: &[u8]) -> Result<Pfs0, Error> {
        Pfs0::read_from(&SliceSource(data))
    }

    /// Reads only the header.
    pub fn read_from<S: ByteSource>(src: &S) -> Result<Pfs0, Error> {
        Pfs0::read_partition_at(src, 0, PartitionKind::Pfs0)
    }

    /// Parse the partition table at `at`; file offsets come out absolute within `src`.
    pub fn read_partition_at<S: ByteSource>(
        src: &S,
        at: u64,
        kind: PartitionKind,
    ) -> Result<Pfs0, Error> {
        const HEADER_SIZE: usize = 0x10;
        let entry_size = kind.entry_size();
        let mut head = [0u8; HEADER_SIZE];
        let got = src.read_at(at, &mut head)?;
        if got < HEADER_SIZE {
            return Err(Error::Truncated {
                what: format!("{} header", kind.name()),
                expected: HEADER_SIZE,
                got,
            });
        }
        if read_u32(&head, 0) != kind.magic() {
            return Err(Error::BadMagic {
                what: kind.name().into(),
                found: read_u32(&head, 0),
            });
        }

        let file_count = read_u32(&head, 0x04) as u64;
        let string_table_size = read_u32(&head, 0x08) as u64;

        // `u64` so the sum cannot overflow `usize` on wasm32.
        let strings_start = HEADER_SIZE as u64 + file_count * entry_size as u64;
        let header_len = strings_start + string_table_size;
        // Entry offsets are counted from the payload area.
        let payload_base = at.checked_add(header_len).ok_or(Error::Overflow)?;
        if payload_base > src.len() {
            return Err(Error::Truncated {
                what: format!("{} string table", kind.name()),
                expected: usize::try_from(payload_base).unwrap_or(usize::MAX),
                got: usize::try_from(src.len()).unwrap_or(usize::MAX),
            });
        }
        // Only the header region; the rest can be gigabytes.
        let header = src.read_vec(at, header_len)?;
        let strings_start = strings_start as usize;

        // No `with_capacity`: the count is untrusted.
        let mut files = Vec::new();
        for i in 0..file_count as usize {
            let entry = HEADER_SIZE + i * entry_size;
            let offset = read_u64(&header, entry);
            let size = read_u64(&header, entry + 8);
            let name_off = read_u32(&header, entry + 16) as usize;
            // `header` ends with the string table, so out-of-range names fail here.
            let name = read_cstr(&header, strings_start.saturating_add(name_off))
                .ok_or(Error::BadStringTable {
                    index: i,
                    offset: name_off,
                })?
                .to_string();
            files.push(Pfs0File { offset, size, name });
        }
        // Some repacks write absolute offsets; take them as such only for a top-level table
        // where the relative reading overruns the image and the absolute one does not.
        let relative_fits = extents_fit(&files, payload_base, src.len());
        let past_the_header = files.iter().all(|f| f.offset >= payload_base);
        let absolute_fits = at == 0 && past_the_header && extents_fit(&files, 0, src.len());
        // If neither fits, report the error against the relative reading.
        let base = if absolute_fits && !relative_fits {
            0
        } else {
            payload_base
        };
        for f in files.iter_mut() {
            f.offset = f.offset.saturating_add(base);
        }
        for (i, f) in files.iter().enumerate() {
            let end = f.offset.checked_add(f.size).ok_or(Error::Overflow)?;
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

    /// A view of file `index` within the container, addressed from 0.
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

    pub fn find(&self, name: &str) -> Option<&Pfs0File> {
        self.files.iter().find(|f| f.name == name)
    }

    /// Case-insensitive.
    pub fn find_with_suffix(&self, suffix: &str) -> Option<&Pfs0File> {
        let suffix = suffix.to_ascii_lowercase();
        self.files
            .iter()
            .find(|f| f.name.to_ascii_lowercase().ends_with(&suffix))
    }
}

/// Whether every file fits in the image once its offset is counted from `base`.
fn extents_fit(files: &[Pfs0File], base: u64, image_size: u64) -> bool {
    files.iter().all(|f| {
        let Some(start) = f.offset.checked_add(base) else {
            return false;
        };
        let Some(end) = start.checked_add(f.size) else {
            return false;
        };
        end <= image_size
    })
}

pub(crate) fn read_u32(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
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

/// `None` if no NUL is found.
pub(crate) fn read_cstr(data: &[u8], at: usize) -> Option<&str> {
    if at >= data.len() {
        return None;
    }
    let end = data[at..].iter().position(|&b| b == 0).map(|p| at + p)?;
    std::str::from_utf8(&data[at..end]).ok()
}

/// Container fixtures; not `#[cfg(test)]` so `switch-wasm` can use them.
pub mod testing {
    use super::*;

    /// Header, entries, string table, then payloads, with offsets relative to the payload area.
    pub fn partition_fs(kind: PartitionKind, files: &[(&str, &[u8])]) -> Vec<u8> {
        let entry_size = kind.entry_size();
        let mut names = Vec::new();
        let mut name_offsets = Vec::new();
        for (name, _) in files {
            name_offsets.push(names.len() as u32);
            names.extend_from_slice(name.as_bytes());
            names.push(0);
        }

        let mut image = Vec::new();
        image.extend_from_slice(&kind.magic().to_le_bytes());
        image.extend_from_slice(&(files.len() as u32).to_le_bytes());
        image.extend_from_slice(&(names.len() as u32).to_le_bytes());
        image.extend_from_slice(&0u32.to_le_bytes());
        let mut at = 0u64;
        for (i, (_, payload)) in files.iter().enumerate() {
            let entry = image.len();
            image.resize(entry + entry_size, 0);
            image[entry..entry + 8].copy_from_slice(&at.to_le_bytes());
            image[entry + 8..entry + 16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
            image[entry + 16..entry + 20].copy_from_slice(&name_offsets[i].to_le_bytes());
            at += payload.len() as u64;
        }
        image.extend_from_slice(&names);
        for (_, payload) in files {
            image.extend_from_slice(payload);
        }
        image
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes absolute offsets, which exercises the absolute-offset fallback.
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
            image[entry + 8..entry + 16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
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
        assert!(pfs0.find("nope").is_none());
        assert_eq!(pfs0.find_with_suffix(".NCA").unwrap().name, "main.nca");
    }

    #[test]
    fn rebases_offsets_relative_to_payload() {
        // Offset 0 means the first byte after the string table.
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

    #[test]
    fn a_padded_payload_area_is_still_read_relative_to_the_header() {
        // Relative offsets with a padded payload area, as retail repacks write.
        const PAYLOAD_AT: usize = 0x80;
        let mut image = Vec::new();
        image.extend_from_slice(&PFS0_MAGIC.to_le_bytes());
        image.extend_from_slice(&1u32.to_le_bytes()); // file count
        image.extend_from_slice(&4u32.to_le_bytes()); // string table size
        image.extend_from_slice(&0u32.to_le_bytes());
        let payload_base = (0x10 + FILE_ENTRY_SIZE + 4) as u64;
        image.extend_from_slice(&(PAYLOAD_AT as u64 - payload_base).to_le_bytes());
        image.extend_from_slice(&4u64.to_le_bytes()); // size
        image.extend_from_slice(&0u32.to_le_bytes()); // name offset
        image.extend_from_slice(&0u32.to_le_bytes()); // padding
        image.extend_from_slice(b"x\0\0\0");
        assert_eq!(image.len() as u64, payload_base);
        image.resize(PAYLOAD_AT, 0);
        image.extend_from_slice(b"DATA");

        let pfs0 = Pfs0::parse(&image).unwrap();
        assert_eq!(pfs0.files[0].offset, PAYLOAD_AT as u64);
        assert_eq!(&image[pfs0.files[0].offset as usize..][..4], b"DATA");
    }

    /// A nested partition's offsets come out against the whole image.
    #[test]
    fn a_partition_is_read_where_the_image_keeps_it() {
        const AT: usize = 0x1000;
        let partition = testing::partition_fs(
            PartitionKind::Hfs0,
            &[("a.nca", b"first"), ("b.nca", b"second")],
        );
        let mut image = vec![0u8; AT];
        image.extend_from_slice(&partition);
        image.resize(0x4000, 0);

        let hfs0 =
            Pfs0::read_partition_at(&SliceSource(&image), AT as u64, PartitionKind::Hfs0).unwrap();
        assert_eq!(hfs0.files.len(), 2);
        assert_eq!(hfs0.image_size, image.len() as u64);
        for (file, want) in hfs0.files.iter().zip([&b"first"[..], b"second"]) {
            assert_eq!(&image[file.offset as usize..][..file.size as usize], want);
        }
        // Read as a PFS0 it is not one, and it says so under its own name.
        assert!(matches!(
            Pfs0::read_partition_at(&SliceSource(&image), AT as u64, PartitionKind::Pfs0),
            Err(Error::BadMagic { .. })
        ));
    }

    /// The absolute-offset fallback does not apply to nested partitions.
    #[test]
    fn a_nested_partition_is_never_reread_as_absolute_offsets() {
        const AT: u64 = 0x1000;
        let mut partition = testing::partition_fs(PartitionKind::Hfs0, &[("a.nca", b"first")]);
        let payload_base = AT + (0x10 + HFS0_ENTRY_SIZE + 6) as u64;
        // Fits measured from byte 0, but not from the payload area.
        partition[0x10..0x18].copy_from_slice(&0x1100u64.to_le_bytes());
        partition[0x18..0x20].copy_from_slice(&0x100u64.to_le_bytes());
        assert!(0x1100 > payload_base);
        let mut image = vec![0u8; AT as usize];
        image.extend_from_slice(&partition);
        image.resize(0x2000, 0);

        assert!(matches!(
            Pfs0::read_partition_at(&SliceSource(&image), AT, PartitionKind::Hfs0),
            Err(Error::FileOutOfBounds { .. })
        ));
    }

    /// Reports a 5 GiB length while serving the header from a real buffer.
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

    /// Returns the container and its payload area offset.
    fn huge_container(entry_offset: u64, entry_size: u64, len: u64) -> (HugeSource, u64) {
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
        let payload_base = header.len() as u64;
        (HugeSource { header, len }, payload_base)
    }

    #[test]
    fn entries_past_the_four_gib_mark_keep_their_offsets() {
        // Truncated to `usize` on wasm32, this would read from 0x1000.
        const PAST_4GIB: u64 = 0x1_0000_1000;
        let (src, payload_base) = huge_container(PAST_4GIB, 0x2000, 5 << 30);
        let pfs0 = Pfs0::read_from(&src).unwrap();
        assert_eq!(pfs0.image_size, 5 << 30);
        assert_eq!(pfs0.files[0].name, "main.nca");
        assert_eq!(pfs0.files[0].offset, PAST_4GIB + payload_base);
        assert_eq!(pfs0.files[0].size, 0x2000);
    }

    #[test]
    fn an_entry_running_past_the_end_is_still_caught_past_four_gib() {
        // The extent ends one byte past the container.
        let len = 5u64 << 30;
        let (src, _) = huge_container(len - 0x1000, 0x1001, len);
        assert!(matches!(
            Pfs0::read_from(&src),
            Err(Error::FileOutOfBounds { .. })
        ));
    }
}
