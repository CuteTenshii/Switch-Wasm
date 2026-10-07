//! XCI cartridge images: an HFS0 root of partitions, each an HFS0 of NCAs,
//! flattened into the same file table an `.nsp` gives.
//!
//! Header fields read: 0x100 magic "HEAD", 0x110 package id, 0x118 last data
//! page, 0x130/0x138 root partition header offset/size.

use crate::nsp::{PartitionKind, Pfs0, Pfs0File};
use crate::source::ByteSource;
use crate::Error;

pub const XCI_MAGIC: u32 = 0x4441_4548;
pub const MAGIC_OFFSET: u64 = 0x100;
pub const HEADER_SIZE: u64 = 0x200;
/// Gamecard addresses count 0x200-byte pages.
pub const MEDIA_UNIT: u64 = 0x200;

pub const UPDATE_PARTITION: &str = "update";
pub const SECURE_PARTITION: &str = "secure";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    /// Offsets are absolute within the image.
    pub files: Vec<Pfs0File>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xci {
    pub package_id: u64,
    /// Bytes the cartridge wrote; dumps are usually trimmed to this.
    pub valid_data_size: u64,
    pub partitions: Vec<Partition>,
    pub image_size: u64,
}

impl Xci {
    /// Reads only the headers.
    pub fn read_from<S: ByteSource>(src: &S) -> Result<Xci, Error> {
        let head = read_header(src)?;
        let package_id = read_u64(&head, 0x110);
        let valid_data_end = read_u32(&head, 0x118) as u64;
        let root_offset = read_u64(&head, 0x130);
        let mut partitions = Vec::new();
        for entry in Pfs0::read_partition_at(src, root_offset, PartitionKind::Hfs0)?.files {
            let files = Pfs0::read_partition_at(src, entry.offset, PartitionKind::Hfs0)
                .map_err(|e| Error::Xci(format!("the {} partition: {e}", entry.name)))?;
            partitions.push(Partition {
                name: entry.name,
                offset: entry.offset,
                size: entry.size,
                files: files.files,
            });
        }
        Ok(Xci {
            package_id,
            // Page address of the last page with data.
            valid_data_size: (valid_data_end + 1) * MEDIA_UNIT,
            partitions,
            image_size: src.len(),
        })
    }

    pub fn is_xci<S: ByteSource>(src: &S) -> bool {
        read_header(src).is_ok()
    }

    pub fn partition(&self, name: &str) -> Option<&Partition> {
        self.partitions
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
    }

    /// The title's content as one file table. `update` (a firmware bundle) is left
    /// out so content-type scans find the game first, and `secure` goes last
    /// because the Program NCA scan keeps the last match.
    pub fn content(&self) -> Pfs0 {
        let mut ordered: Vec<&Partition> = self
            .partitions
            .iter()
            .filter(|p| !p.name.eq_ignore_ascii_case(UPDATE_PARTITION))
            .collect();
        ordered.sort_by_key(|p| p.name.eq_ignore_ascii_case(SECURE_PARTITION));
        Pfs0 {
            files: ordered
                .iter()
                .flat_map(|p| p.files.iter().cloned())
                .collect(),
            image_size: self.image_size,
        }
    }
}

/// The file table of a PFS0 (`.nsp`) or of a cartridge image's partitions.
pub fn read_container<S: ByteSource>(src: &S) -> Result<Pfs0, Error> {
    // Only a container that is not a PFS0 falls back to the cartridge reader.
    match Pfs0::read_from(src) {
        Err(Error::BadMagic { .. }) if Xci::is_xci(src) => Ok(Xci::read_from(src)?.content()),
        other => other,
    }
}

fn read_header<S: ByteSource>(src: &S) -> Result<Vec<u8>, Error> {
    let head = src.read_vec(0, HEADER_SIZE.min(src.len()))?;
    if head.len() < HEADER_SIZE as usize {
        return Err(Error::Truncated {
            what: "XCI header".into(),
            expected: HEADER_SIZE as usize,
            got: head.len(),
        });
    }
    let magic = read_u32(&head, MAGIC_OFFSET as usize);
    if magic != XCI_MAGIC {
        return Err(Error::BadMagic {
            what: "XCI".into(),
            found: magic,
        });
    }
    Ok(head)
}

fn read_u32(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

fn read_u64(data: &[u8], at: usize) -> u64 {
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

/// Fixtures for the on-disk format; not `#[cfg(test)]` (see [`crate::nsp::testing`]).
pub mod testing {
    use super::*;
    use crate::nsp::testing::partition_fs;

    /// A trimmed cartridge image holding `partitions`, each already an HFS0.
    pub fn cartridge(partitions: &[(&str, &[u8])]) -> Vec<u8> {
        let mut image = vec![0u8; HEADER_SIZE as usize];
        image[MAGIC_OFFSET as usize..MAGIC_OFFSET as usize + 4]
            .copy_from_slice(&XCI_MAGIC.to_le_bytes());
        image[0x110..0x118].copy_from_slice(&0x0123_4567_89ab_cdefu64.to_le_bytes());
        image[0x130..0x138].copy_from_slice(&HEADER_SIZE.to_le_bytes());
        let root = partition_fs(PartitionKind::Hfs0, partitions);
        image[0x138..0x140].copy_from_slice(&(root.len() as u64).to_le_bytes());
        image.extend_from_slice(&root);
        // Pad to a whole page, then say which page the data ends on.
        image.resize(image.len().next_multiple_of(MEDIA_UNIT as usize), 0);
        let last_page = (image.len() as u64 / MEDIA_UNIT) - 1;
        image[0x118..0x11c].copy_from_slice(&(last_page as u32).to_le_bytes());
        image
    }
}

#[cfg(test)]
mod tests {
    use super::testing::cartridge;
    use super::*;
    use crate::nsp::testing::partition_fs;
    use crate::source::SliceSource;

    const PROGRAM: &[u8] = b"the program nca";
    const CONTROL: &[u8] = b"the control nca";
    const SYSTEM: &[u8] = b"a system update nca";

    /// `update`, `secure` and `normal` partitions, like a retail cartridge.
    fn retail_shaped() -> Vec<u8> {
        let update = partition_fs(PartitionKind::Hfs0, &[("sys.nca", SYSTEM)]);
        let normal = partition_fs(PartitionKind::Hfs0, &[("cert", b"\xff\xff\xff\xff")]);
        let secure = partition_fs(
            PartitionKind::Hfs0,
            &[("program.nca", PROGRAM), ("control.nca", CONTROL)],
        );
        cartridge(&[
            ("update", &update),
            ("normal", &normal),
            ("secure", &secure),
        ])
    }

    #[test]
    fn a_files_offset_is_absolute_within_the_image() {
        let image = retail_shaped();
        let xci = Xci::read_from(&SliceSource(&image)).unwrap();
        assert_eq!(xci.package_id, 0x0123_4567_89ab_cdef);
        assert_eq!(
            xci.partitions
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["update", "normal", "secure"]
        );
        let secure = xci.partition("SECURE").unwrap();
        let program = &secure.files[0];
        assert_eq!(program.name, "program.nca");
        assert_eq!(
            &image[program.offset as usize..][..program.size as usize],
            PROGRAM
        );
    }

    #[test]
    fn the_content_table_leaves_the_system_update_out() {
        let image = retail_shaped();
        let xci = Xci::read_from(&SliceSource(&image)).unwrap();
        let content = xci.content();
        assert_eq!(
            content
                .files
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            ["cert", "program.nca", "control.nca"]
        );
        assert_eq!(content.image_size, image.len() as u64);
        assert!(xci.partition("update").is_some());
    }

    #[test]
    fn the_titles_own_partition_comes_last() {
        let secure = partition_fs(PartitionKind::Hfs0, &[("program.nca", PROGRAM)]);
        let normal = partition_fs(PartitionKind::Hfs0, &[("other.nca", CONTROL)]);
        let image = cartridge(&[("secure", &secure), ("normal", &normal)]);
        let content = Xci::read_from(&SliceSource(&image)).unwrap().content();
        assert_eq!(content.files.last().unwrap().name, "program.nca");
    }

    #[test]
    fn a_trimmed_dump_reports_the_data_it_holds() {
        let image = retail_shaped();
        let xci = Xci::read_from(&SliceSource(&image)).unwrap();
        assert_eq!(xci.valid_data_size, image.len() as u64);
        assert_eq!(xci.image_size, image.len() as u64);
    }

    #[test]
    fn a_container_is_read_as_whichever_kind_it_is() {
        let image = retail_shaped();
        assert!(Xci::is_xci(&SliceSource(&image)));
        assert_eq!(
            read_container(&SliceSource(&image)).unwrap(),
            Xci::read_from(&SliceSource(&image)).unwrap().content()
        );

        let nsp = partition_fs(PartitionKind::Pfs0, &[("program.nca", PROGRAM)]);
        assert!(!Xci::is_xci(&SliceSource(&nsp)));
        let files = read_container(&SliceSource(&nsp)).unwrap().files;
        assert_eq!(files.len(), 1);
        assert_eq!(
            &nsp[files[0].offset as usize..][..files[0].size as usize],
            PROGRAM
        );
    }

    #[test]
    fn something_that_is_neither_is_reported_as_a_container() {
        let junk = vec![0u8; 0x400];
        assert!(matches!(
            read_container(&SliceSource(&junk)),
            Err(Error::BadMagic { what, .. }) if what == "PFS0"
        ));
        // Truncated, with the magic intact.
        let mut cut = retail_shaped();
        cut.truncate(0x180);
        assert!(matches!(
            Xci::read_from(&SliceSource(&cut)),
            Err(Error::Truncated { .. })
        ));
    }

    #[test]
    fn a_partition_that_does_not_parse_says_which_one() {
        let secure = partition_fs(PartitionKind::Hfs0, &[("program.nca", PROGRAM)]);
        let mut image = cartridge(&[("secure", &secure)]);
        // The root entry's payload offset, made to point at the padding.
        let root = HEADER_SIZE as usize;
        let entry = root + 0x10;
        image[entry..entry + 8].copy_from_slice(&0x100u64.to_le_bytes());
        let e = Xci::read_from(&SliceSource(&image)).unwrap_err();
        assert!(
            matches!(&e, Error::Xci(msg) if msg.contains("secure")),
            "{e}"
        );
    }
}
