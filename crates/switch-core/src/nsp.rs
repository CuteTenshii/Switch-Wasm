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
//! 0x10    -     FileEntry[file_count]   (24 bytes each)
//! -       -     string table
//! -       -     file data
//! ```
//!
//! Each `FileEntry` is 24 bytes: `u64 offset`, `u64 size`, `u32 name_offset`,
//! `u32 padding`. `offset`/`size` reference the file payload and are counted
//! from the end of the string table, where the payload area begins;
//! `name_offset` references a NUL-terminated string in the string table.
//!
//! An XCI's partitions ([`crate::xci`]) are the same table under the magic
//! "HFS0", with a 64-byte entry that adds a hash of the file's first bytes
//! after the four fields above. Both are read by [`Pfs0::read_partition_at`],
//! so a cartridge partition and an `.nsp` present the rest of the stack the
//! same file table.

use crate::source::{ByteSource, SliceSource};
use crate::Error;

pub const PFS0_MAGIC: u32 = 0x3053_4650; // "PFS0"
/// Each entry: u64 offset, u64 size, u32 name_offset, u32 padding.
pub const FILE_ENTRY_SIZE: usize = 24;
pub const HFS0_MAGIC: u32 = 0x3053_4648; // "HFS0"
/// An HFS0 entry adds `u32 hashed_region_size`, 8 reserved bytes and a
/// SHA-256 of that region to the four fields a PFS0 entry has.
pub const HFS0_ENTRY_SIZE: usize = 0x40;

/// The two spellings of one partition table: `PFS0` in an `.nsp` and in an
/// NCA's ExeFS, `HFS0` in the root and partitions of an XCI.
///
/// Nothing but the magic and the entry stride separates them — the header and
/// the first four fields of an entry are laid out identically — so they are
/// read by one parser rather than two.
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
    /// Byte offset of the file payload from the start of the image — the
    /// entry's own offset already resolved against the payload area, so it
    /// can be read without knowing how long the header was.
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
        Pfs0::read_partition_at(src, 0, PartitionKind::Pfs0)
    }

    /// Parse the partition table that starts at `at`, in either spelling.
    ///
    /// An XCI keeps its partitions at offsets of their own inside the image
    /// and writes them as HFS0; that offset and the entry stride are the
    /// whole of the difference. File offsets come out absolute within `src`
    /// either way, so what a cartridge partition hands the rest of the stack
    /// is a file table indistinguishable from an `.nsp`'s.
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

        // Header + entries + string table must fit within the image. Every
        // term comes from a `u32`, so the `u64` arithmetic cannot overflow —
        // which is the point of doing it in `u64` on a 32-bit target, where
        // `file_count * FILE_ENTRY_SIZE` alone can exceed `usize`.
        let strings_start = HEADER_SIZE as u64 + file_count * entry_size as u64;
        let header_len = strings_start + string_table_size;
        // The payload area starts where the header — entry table and string
        // table — ends, and that is what an entry's offset is counted from.
        let payload_base = at.checked_add(header_len).ok_or(Error::Overflow)?;
        if payload_base > src.len() {
            return Err(Error::Truncated {
                what: format!("{} string table", kind.name()),
                expected: usize::try_from(payload_base).unwrap_or(usize::MAX),
                got: usize::try_from(src.len()).unwrap_or(usize::MAX),
            });
        }
        // Only the header region: for a retail container the rest is
        // gigabytes, and none of it is needed to build the file table.
        let header = src.read_vec(at, header_len)?;
        let strings_start = strings_start as usize;

        // Not `with_capacity(file_count)`: the count is whatever the file
        // says, and reserving for a corrupt one is an allocation the target
        // aborts on rather than an error anyone can read.
        let mut files = Vec::new();
        for i in 0..file_count as usize {
            let entry = HEADER_SIZE + i * entry_size;
            let offset = read_u64(&header, entry);
            let size = read_u64(&header, entry + 8);
            let name_off = read_u32(&header, entry + 16) as usize;
            // The name has to be NUL-terminated inside the string table —
            // `header` ends where the table does, so a name offset pointing
            // past it fails here instead of running into the payload.
            let name = read_cstr(&header, strings_start.saturating_add(name_off))
                .ok_or(Error::BadStringTable {
                    index: i,
                    offset: name_off,
                })?
                .to_string();
            files.push(Pfs0File { offset, size, name });
        }
        // Some repack tools emit offsets already counted from the start of
        // the image instead. Both readings are tried, the format's own
        // first, and the entries are taken as absolute only when rebasing
        // them would run a file past the end of the image while leaving them
        // alone would not.
        //
        // This is decided from the extents rather than from "does any entry
        // point inside the header", which is what it used to ask. A repack
        // that pads its payload area — aligning the first file to 0x8000,
        // say — has no entry at offset 0 to give the relative reading away,
        // so every offset was left short by the header length and every NCA
        // in it was read from the wrong place. A 7 GiB Just Dance 2022 `.nsp`
        // whose first entry starts at 0x7e30 (0x8000 once rebased) is one.
        //
        // Only for a table that starts the image. A partition nested inside
        // one is written by the cartridge master, not by a repacker, and
        // "absolute" there would mean an offset into the image rather than
        // into the partition — a reading no producer intends and one that
        // happens to fit often enough to be dangerous.
        let relative_fits = extents_fit(&files, payload_base, src.len());
        // An absolute offset cannot point into the header its own entry
        // lives in, so a reading that puts one there is not one.
        let past_the_header = files.iter().all(|f| f.offset >= payload_base);
        let absolute_fits = at == 0 && past_the_header && extents_fit(&files, 0, src.len());
        // A container neither reading fits is malformed, and rebasing it
        // anyway is what reports the out-of-bounds below against the format's
        // own reading rather than against the fallback.
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

/// Whether every file still lies inside an `image_size`-byte image once its
/// entry offset is counted from `base`.
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

/// Read a NUL-terminated string, returning `None` if no NUL is found within
/// the remaining buffer.
pub(crate) fn read_cstr(data: &[u8], at: usize) -> Option<&str> {
    if at >= data.len() {
        return None;
    }
    let end = data[at..].iter().position(|&b| b == 0).map(|p| at + p)?;
    std::str::from_utf8(&data[at..end]).ok()
}

/// Fixtures that write the on-disk form this module reads.
///
/// Not `#[cfg(test)]`: `switch-wasm` is a separate crate and cannot reach a
/// test module in this one, and the container path it exports is worth
/// testing against a real image — the same reason [`crate::gpu::testing`]
/// exists.
pub mod testing {
    use super::*;

    /// A partition table in either spelling, laid out the way a producer
    /// writes one: header, entry table, string table, then the payloads, with
    /// each entry's offset counted from the payload area.
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

    /// Entry offsets here are written from the start of the image, not from
    /// the payload area — so every test built on this one is also what keeps
    /// the absolute-offset fallback honest.
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
        assert_eq!(pfs0.find("nope").is_none(), true);
        assert_eq!(pfs0.find_with_suffix(".NCA").unwrap().name, "main.nca");
    }

    #[test]
    fn rebases_offsets_relative_to_payload() {
        // The ordinary case, and what the format says: an entry counts from
        // the payload area, so the one claiming offset 0 means the first byte
        // after the header + string table, not the first byte of the image.
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
        // The shape a retail repack has: relative offsets, but the payload
        // area padded so the first file lands on an alignment boundary. No
        // entry sits at offset 0, which is what the old "does any entry point
        // inside the header" test needed to notice the offsets were relative
        // at all — so every file in one was read a header-length short.
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

    /// An XCI keeps its partitions at offsets of their own, so the table has
    /// to be readable somewhere other than the front of the image — and what
    /// comes out has to be addressed against the image, not against the
    /// partition, or every layer above would need to know it was nested.
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

    /// The absolute-offset fallback is for a repacked `.nsp` and stops there.
    /// Inside an image, an offset that "fits" measured from byte 0 is a
    /// coincidence — the entry belongs to a partition, and reading it that way
    /// would hand back a range from somewhere else entirely.
    #[test]
    fn a_nested_partition_is_never_reread_as_absolute_offsets() {
        const AT: u64 = 0x1000;
        let mut partition = testing::partition_fs(PartitionKind::Hfs0, &[("a.nca", b"first")]);
        let payload_base = AT + (0x10 + HFS0_ENTRY_SIZE + 6) as u64;
        // Past this partition's header, and inside the image measured from
        // byte 0 — but past the end of it once counted from the payload area,
        // which is the only reading a cartridge ever means.
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

    /// The container, and where its payload area starts — which is what the
    /// entry offset in it is counted from.
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
        // The offset a retail container's program NCA actually lives at. Held
        // in a `usize` (as this parser used to), it truncates to 0x1000 and
        // every read lands on the wrong bytes; the bounds check below passes
        // for the same reason, so nothing catches it.
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
        // Same shape, but the extent ends one byte past the container. The
        // truncating version of this check compared wrapped values and let it
        // through.
        let len = 5u64 << 30;
        let (src, _) = huge_container(len - 0x1000, 0x1001, len);
        assert!(matches!(
            Pfs0::read_from(&src),
            Err(Error::FileOutOfBounds { .. })
        ));
    }
}
