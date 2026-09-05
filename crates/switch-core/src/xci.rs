//! XCI: the image of a Switch game cartridge.
//!
//! ```text
//! offset  size   field
//! 0x000   0x100  RSA-2048 signature over the header
//! 0x100   4      magic "HEAD" (0x44414548)
//! 0x104   4      first page of the ROM area
//! 0x108   4      first page of the backup area
//! 0x10C   1      key index
//! 0x10D   1      cartridge capacity
//! 0x10E   1      header version
//! 0x10F   1      flags
//! 0x110   8      package id
//! 0x118   4      the last page holding data
//! 0x120   0x10   gamecard info IV
//! 0x130   8      offset of the root partition header
//! 0x138   8      size of the root partition header
//! 0x140   0x20   SHA-256 of the root partition header
//! ```
//!
//! The root is an HFS0 whose entries are the cartridge's partitions,
//! `update`, `normal`, `secure`, and `logo` on later carts, and each of
//! those is an HFS0 in turn, holding the NCAs. Nothing between the image and
//! the NCAs is encrypted in a dump, so reading a cartridge is
//! [`Pfs0::read_partition_at`] twice and then the ordinary NCA reader: what
//! [`Xci::content`] hands back is a file table an `.nsp`'s is
//! indistinguishable from, and every layer above this one is unchanged.

use crate::nsp::{PartitionKind, Pfs0, Pfs0File};
use crate::source::ByteSource;
use crate::Error;

/// "HEAD", little-endian, at [`MAGIC_OFFSET`]: the signature that opens the
/// image comes first.
pub const XCI_MAGIC: u32 = 0x4441_4548;
pub const MAGIC_OFFSET: u64 = 0x100;
/// The header the magic sits in, signature included.
pub const HEADER_SIZE: u64 = 0x200;
/// A gamecard address is a page count, and a page is 0x200 bytes, the same
/// media unit an NCA measures its sections in.
pub const MEDIA_UNIT: u64 = 0x200;

/// The partition holding a system update rather than the title, and the one
/// partition [`Xci::content`] leaves out. See there for why.
pub const UPDATE_PARTITION: &str = "update";
/// The partition holding the title's own NCAs.
pub const SECURE_PARTITION: &str = "secure";

/// One partition of a cartridge: an HFS0 somewhere in the image, and the
/// files in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    /// `update`, `normal`, `secure` or `logo`.
    pub name: String,
    /// Where the partition's own HFS0 header starts in the image.
    pub offset: u64,
    pub size: u64,
    /// Its files, with offsets already absolute within the image.
    pub files: Vec<Pfs0File>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xci {
    /// Identifies the cartridge master; two dumps of one cartridge share it.
    pub package_id: u64,
    /// How much of the image the cartridge actually wrote. A dump is
    /// routinely *trimmed* to this, so it is usually the image size and is
    /// larger than it only for an untrimmed one.
    pub valid_data_size: u64,
    pub partitions: Vec<Partition>,
    pub image_size: u64,
}

impl Xci {
    /// Read a cartridge image's partitions, and only its headers: a retail
    /// XCI is up to 32 GB, and what this returns is a handful of file tables
    /// pointing into it.
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
            // The page address of the last page with data in it, so the
            // page after it is where a trimmed dump ends.
            valid_data_size: (valid_data_end + 1) * MEDIA_UNIT,
            partitions,
            image_size: src.len(),
        })
    }

    /// Whether `src` opens with a cartridge header. Cheap enough to ask of
    /// any container before parsing it as one.
    pub fn is_xci<S: ByteSource>(src: &S) -> bool {
        read_header(src).is_ok()
    }

    pub fn partition(&self, name: &str) -> Option<&Partition> {
        self.partitions
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
    }

    /// The title's content, as the one flat file table the rest of the stack
    /// reads a container through.
    ///
    /// The `update` partition is left out. It is a firmware bundle, dozens
    /// of system NCAs, several of them Program content, and every search
    /// that follows ("the Program NCA", "the Control NCA") is a scan of this
    /// table by content type, so a cartridge's system update would answer
    /// them before the game does.
    ///
    /// `secure` goes last for the other half of that rule: the scan for a
    /// Program NCA keeps the *last* match, because a container carrying two
    /// is an update over a base, and the game's own content lives here.
    pub fn content(&self) -> Pfs0 {
        let mut ordered: Vec<&Partition> = self
            .partitions
            .iter()
            .filter(|p| !p.name.eq_ignore_ascii_case(UPDATE_PARTITION))
            .collect();
        // A stable sort on "is this the secure partition", so everything else
        // keeps the order the cartridge wrote it in.
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

/// The file table of whichever container this is: a PFS0 (`.nsp`) read
/// straight through, or a cartridge image's partitions flattened into the
/// same table.
///
/// Every reader above this, the Program NCA scan, the Control NCA, the
/// bundled ticket, the browser's file list: works from a [`Pfs0`] and a
/// source, so this one function is the whole of what an XCI needed from them.
pub fn read_container<S: ByteSource>(src: &S) -> Result<Pfs0, Error> {
    // The PFS0 magic is at offset 0 and settles the question; a cartridge's
    // "HEAD" is 0x100 bytes in, where an `.nsp`'s entry table and string
    // table can spell anything a repacker put in a file name. So the
    // cartridge reading is what a container that is not a PFS0 falls back to,
    // and every other PFS0 failure is still reported as itself.
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

/// Fixtures that write the on-disk form this module reads: see
/// [`crate::nsp::testing`] for why they are not `#[cfg(test)]`.
pub mod testing {
    use super::*;
    use crate::nsp::testing::partition_fs;

    /// A cartridge image holding `partitions`, each already an HFS0 (build
    /// one with `partition_fs(PartitionKind::Hfs0, ..)`).
    ///
    /// Trimmed, as a dump usually is: the image ends where its data does, and
    /// the header says so.
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

    /// The shape a retail cartridge has: a firmware bundle in `update`, the
    /// title's own content in `secure`, and a `normal` partition beside them.
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

    /// What every reader above this one needs: an offset it can hand to the
    /// source it opened the image with, without knowing there were two
    /// partition headers between the two.
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

    /// A cartridge's `update` partition is a firmware bundle, and the scans
    /// that follow (the Program NCA, the Control NCA) would find its system
    /// titles before the game's own content.
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

    /// `secure` last, because the Program NCA scan keeps the last match it
    /// finds, a rule that exists for an update stacked over a base.
    #[test]
    fn the_titles_own_partition_comes_last() {
        let secure = partition_fs(PartitionKind::Hfs0, &[("program.nca", PROGRAM)]);
        let normal = partition_fs(PartitionKind::Hfs0, &[("other.nca", CONTROL)]);
        let image = cartridge(&[("secure", &secure), ("normal", &normal)]);
        let content = Xci::read_from(&SliceSource(&image)).unwrap().content();
        assert_eq!(content.files.last().unwrap().name, "program.nca");
    }

    /// A dump is trimmed to the data the cartridge wrote, and the header says
    /// where that ends, so a short image is not evidence of a short read.
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
        // Too short to hold a header at all, and the magic is where it would
        // be: the failure a truncated download gives.
        let mut cut = retail_shaped();
        cut.truncate(0x180);
        assert!(matches!(
            Xci::read_from(&SliceSource(&cut)),
            Err(Error::Truncated { .. })
        ));
    }

    /// A partition header that points outside the image is the cartridge's
    /// error, and it names the partition it came from rather than failing as
    /// an anonymous bad magic.
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
