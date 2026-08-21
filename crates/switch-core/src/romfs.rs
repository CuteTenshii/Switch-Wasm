//! RomFS reader: the directory tree inside an NCA's RomFS section.
//!
//! [`crate::nca::Nca::decrypt_romfs_section`] hands back the raw image; this
//! turns it into a list of paths and payload extents. The image starts with a
//! 0x50-byte header of `u64` offset/size pairs:
//!
//! ```text
//! 0x00  header size (always 0x50 — RomFS has no magic, this is the check)
//! 0x08  directory hash table offset
//! 0x10  directory hash table size
//! 0x18  directory metadata table offset
//! 0x20  directory metadata table size
//! 0x28  file hash table offset
//! 0x30  file hash table size
//! 0x38  file metadata table offset
//! 0x40  file metadata table size
//! 0x48  file data offset
//! ```
//!
//! The two hash tables only accelerate lookup by name; the metadata tables on
//! their own describe the whole tree, so this walks those and ignores the
//! hashes. A directory entry is
//!
//! ```text
//! 0x00  u32 parent directory offset
//! 0x04  u32 next sibling directory offset
//! 0x08  u32 first child directory offset
//! 0x0C  u32 first file offset
//! 0x10  u32 next directory in this hash bucket
//! 0x14  u32 name length
//! 0x18  name (UTF-8, padded to a 4-byte boundary)
//! ```
//!
//! and a file entry is
//!
//! ```text
//! 0x00  u32 parent directory offset
//! 0x04  u32 next sibling file offset
//! 0x08  u64 payload offset, relative to the file data offset
//! 0x10  u64 payload size
//! 0x18  u32 next file in this hash bucket
//! 0x1C  u32 name length
//! 0x20  name (UTF-8, padded to a 4-byte boundary)
//! ```
//!
//! `0xFFFFFFFF` ends a sibling chain. The root directory is the entry at
//! offset 0 of the directory metadata table and its own name is empty.

use crate::Error;

/// The header's declared size. RomFS carries no magic number, so this
/// doubles as the format check.
pub const HEADER_SIZE: u64 = 0x50;

/// End-of-chain marker for the `next`/`first` links between entries.
const INVALID_OFFSET: u32 = 0xFFFF_FFFF;
/// Fixed part of a directory metadata entry, before its name.
const DIR_ENTRY_SIZE: usize = 0x18;
/// Fixed part of a file metadata entry, before its name.
const FILE_ENTRY_SIZE: usize = 0x20;

/// One file in the image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RomFsFile {
    /// Absolute path within the image, with a leading slash (`/control.nacp`).
    pub path: String,
    /// Payload offset, relative to the image's file data region.
    pub offset: u64,
    /// Payload size in bytes.
    pub size: u64,
}

/// A parsed RomFS image, borrowing the decrypted section it was read from.
#[derive(Debug, Clone)]
pub struct RomFs<'a> {
    image: &'a [u8],
    data_offset: u64,
    files: Vec<RomFsFile>,
}

impl<'a> RomFs<'a> {
    /// Walk the metadata tables and collect every file in the image.
    pub fn parse(image: &'a [u8]) -> Result<RomFs<'a>, Error> {
        if (image.len() as u64) < HEADER_SIZE {
            return Err(Error::Truncated {
                what: "RomFS header".into(),
                expected: HEADER_SIZE as usize,
                got: image.len(),
            });
        }
        let header_size = crate::nsp::read_u64(image, 0);
        if header_size != HEADER_SIZE {
            return Err(Error::RomFs(format!(
                "header size is {:#x}, expected {:#x} — not a RomFS image",
                header_size, HEADER_SIZE
            )));
        }
        let dir_table = table(image, crate::nsp::read_u64(image, 0x18), crate::nsp::read_u64(image, 0x20), "RomFS directory metadata table")?;
        let file_table = table(image, crate::nsp::read_u64(image, 0x38), crate::nsp::read_u64(image, 0x40), "RomFS file metadata table")?;
        let data_offset = crate::nsp::read_u64(image, 0x48);

        // Chains are just offsets into the tables, so a corrupt image can
        // point an entry back at itself. The walk is bounded by the number of
        // entries the tables could possibly hold: every entry is at least its
        // fixed part long, so dividing each table by that is an upper bound on
        // what it can contain. A directory is read twice — once when its
        // parent lists it, once when it is walked itself — so it gets two.
        let mut budget = 2 * (dir_table.len() / DIR_ENTRY_SIZE)
            + file_table.len() / FILE_ENTRY_SIZE
            + 2;
        let spend = |budget: &mut usize| -> Result<(), Error> {
            *budget = budget.checked_sub(1).ok_or_else(|| {
                Error::RomFs("entry chain doesn't terminate — corrupt image".into())
            })?;
            Ok(())
        };
        let mut files = Vec::new();
        let mut pending = vec![(0u32, String::new())];
        while let Some((dir_offset, prefix)) = pending.pop() {
            spend(&mut budget)?;
            let dir = entry(dir_table, dir_offset, DIR_ENTRY_SIZE, "directory")?;

            let mut file_offset = crate::nsp::read_u32(dir, 0x0C);
            while file_offset != INVALID_OFFSET {
                spend(&mut budget)?;
                let file = entry(file_table, file_offset, FILE_ENTRY_SIZE, "file")?;
                files.push(RomFsFile {
                    path: format!("{}/{}", prefix, name(file, FILE_ENTRY_SIZE, "file")?),
                    offset: crate::nsp::read_u64(file, 0x08),
                    size: crate::nsp::read_u64(file, 0x10),
                });
                file_offset = crate::nsp::read_u32(file, 0x04);
            }

            let mut child_offset = crate::nsp::read_u32(dir, 0x08);
            while child_offset != INVALID_OFFSET {
                spend(&mut budget)?;
                let child = entry(dir_table, child_offset, DIR_ENTRY_SIZE, "directory")?;
                pending.push((
                    child_offset,
                    format!("{}/{}", prefix, name(child, DIR_ENTRY_SIZE, "directory")?),
                ));
                child_offset = crate::nsp::read_u32(child, 0x04);
            }
        }

        Ok(RomFs { image, data_offset, files })
    }

    /// Every file in the image, in no particular order.
    pub fn files(&self) -> &[RomFsFile] {
        &self.files
    }

    /// Look up a file by absolute path (`/control.nacp`), case-insensitively —
    /// RomFS itself is case-sensitive, but the names this reader looks for are
    /// SDK-generated and have been spelled inconsistently by repack tools.
    pub fn find(&self, path: &str) -> Option<&RomFsFile> {
        self.files.iter().find(|f| f.path.eq_ignore_ascii_case(path))
    }

    /// The payload bytes of `file`, or `None` if its extent falls outside the
    /// image.
    pub fn read(&self, file: &RomFsFile) -> Option<&'a [u8]> {
        let start = self.data_offset.checked_add(file.offset)? as usize;
        let end = start.checked_add(file.size as usize)?;
        self.image.get(start..end)
    }

    /// The payload bytes of the file at `path`.
    pub fn read_path(&self, path: &str) -> Option<&'a [u8]> {
        self.read(self.find(path)?)
    }
}

/// Slice one of the header's declared tables out of the image.
fn table<'a>(image: &'a [u8], offset: u64, size: u64, what: &str) -> Result<&'a [u8], Error> {
    let start = offset as usize;
    let end = start
        .checked_add(size as usize)
        .ok_or(Error::Overflow)?;
    image.get(start..end).ok_or_else(|| Error::Truncated {
        what: what.to_owned(),
        expected: end,
        got: image.len(),
    })
}

/// The bytes of one metadata entry: its fixed part plus its name, which the
/// entry's own `name length` field sizes.
fn entry<'a>(table: &'a [u8], offset: u32, fixed_size: usize, what: &str) -> Result<&'a [u8], Error> {
    let start = offset as usize;
    let fixed_end = start.checked_add(fixed_size).ok_or(Error::Overflow)?;
    if fixed_end > table.len() {
        return Err(Error::RomFs(format!(
            "RomFS {} entry at {:#x} is outside its metadata table",
            what, offset
        )));
    }
    let name_len = crate::nsp::read_u32(table, fixed_end - 4) as usize;
    let end = fixed_end.checked_add(name_len).ok_or(Error::Overflow)?;
    if end > table.len() {
        return Err(Error::RomFs(format!(
            "RomFS {} entry at {:#x} has a name that runs past its metadata table",
            what, offset
        )));
    }
    Ok(&table[start..end])
}

/// The name of an entry produced by [`entry`].
fn name(entry: &[u8], fixed_size: usize, what: &str) -> Result<String, Error> {
    std::str::from_utf8(&entry[fixed_size..])
        .map(|s| s.to_owned())
        .map_err(|_| Error::RomFs(format!("RomFS {} entry has a non-UTF-8 name", what)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a RomFS image with a root directory holding `root_files`, and one
    /// subdirectory `sub` holding `sub_files`.
    fn build(root_files: &[(&str, &[u8])], sub: Option<(&str, &[(&str, &[u8])])>) -> Vec<u8> {
        fn push_padded(table: &mut Vec<u8>, name: &str) {
            table.extend_from_slice(name.as_bytes());
            while table.len() % 4 != 0 {
                table.push(0);
            }
        }

        let mut dir_table: Vec<u8> = Vec::new();
        let mut file_table: Vec<u8> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        // Files are laid out root-first, then the subdirectory's, so each
        // chain is a run of consecutive entries.
        let chain = |table: &mut Vec<u8>, data: &mut Vec<u8>, files: &[(&str, &[u8])]| -> u32 {
            if files.is_empty() {
                return INVALID_OFFSET;
            }
            let first = table.len() as u32;
            for (i, (name, payload)) in files.iter().enumerate() {
                let mut entry = Vec::new();
                entry.extend_from_slice(&0u32.to_le_bytes()); // parent
                let next = if i + 1 == files.len() {
                    INVALID_OFFSET
                } else {
                    // Every entry in this chain has the same length only if
                    // the names do; compute the next offset after the fact.
                    0
                };
                entry.extend_from_slice(&next.to_le_bytes());
                entry.extend_from_slice(&(data.len() as u64).to_le_bytes());
                entry.extend_from_slice(&(payload.len() as u64).to_le_bytes());
                entry.extend_from_slice(&INVALID_OFFSET.to_le_bytes());
                entry.extend_from_slice(&(name.len() as u32).to_le_bytes());
                let at = table.len();
                table.extend_from_slice(&entry);
                push_padded(table, name);
                if i + 1 != files.len() {
                    let next_offset = table.len() as u32;
                    table[at + 4..at + 8].copy_from_slice(&next_offset.to_le_bytes());
                }
                data.extend_from_slice(payload);
            }
            first
        };

        let root_first_file = chain(&mut file_table, &mut data, root_files);
        let (sub_name, sub_files) = sub.unwrap_or(("", &[]));
        let sub_first_file = if sub.is_some() {
            chain(&mut file_table, &mut data, sub_files)
        } else {
            INVALID_OFFSET
        };

        // Root directory entry, then the subdirectory's.
        dir_table.extend_from_slice(&0u32.to_le_bytes()); // parent
        dir_table.extend_from_slice(&INVALID_OFFSET.to_le_bytes()); // sibling
        let child_slot = dir_table.len();
        dir_table.extend_from_slice(&INVALID_OFFSET.to_le_bytes()); // first child
        dir_table.extend_from_slice(&root_first_file.to_le_bytes());
        dir_table.extend_from_slice(&INVALID_OFFSET.to_le_bytes()); // hash chain
        dir_table.extend_from_slice(&0u32.to_le_bytes()); // name length
        if sub.is_some() {
            let child_offset = dir_table.len() as u32;
            dir_table[child_slot..child_slot + 4].copy_from_slice(&child_offset.to_le_bytes());
            dir_table.extend_from_slice(&0u32.to_le_bytes()); // parent
            dir_table.extend_from_slice(&INVALID_OFFSET.to_le_bytes()); // sibling
            dir_table.extend_from_slice(&INVALID_OFFSET.to_le_bytes()); // first child
            dir_table.extend_from_slice(&sub_first_file.to_le_bytes());
            dir_table.extend_from_slice(&INVALID_OFFSET.to_le_bytes()); // hash chain
            dir_table.extend_from_slice(&(sub_name.len() as u32).to_le_bytes());
            push_padded(&mut dir_table, sub_name);
        }

        let mut image = vec![0u8; HEADER_SIZE as usize];
        let dir_offset = image.len() as u64;
        image.extend_from_slice(&dir_table);
        let file_offset = image.len() as u64;
        image.extend_from_slice(&file_table);
        let data_offset = image.len() as u64;
        image.extend_from_slice(&data);

        image[0x00..0x08].copy_from_slice(&HEADER_SIZE.to_le_bytes());
        image[0x18..0x20].copy_from_slice(&dir_offset.to_le_bytes());
        image[0x20..0x28].copy_from_slice(&(dir_table.len() as u64).to_le_bytes());
        image[0x38..0x40].copy_from_slice(&file_offset.to_le_bytes());
        image[0x40..0x48].copy_from_slice(&(file_table.len() as u64).to_le_bytes());
        image[0x48..0x50].copy_from_slice(&data_offset.to_le_bytes());
        image
    }

    #[test]
    fn reads_root_files() {
        let image = build(&[("control.nacp", b"NACP"), ("icon_AmericanEnglish.dat", b"JPEG")], None);
        let romfs = RomFs::parse(&image).unwrap();
        assert_eq!(romfs.files().len(), 2);
        assert_eq!(romfs.read_path("/control.nacp").unwrap(), b"NACP");
        assert_eq!(romfs.read_path("/icon_AmericanEnglish.dat").unwrap(), b"JPEG");
    }

    #[test]
    fn lookup_ignores_case() {
        let image = build(&[("control.nacp", b"NACP")], None);
        let romfs = RomFs::parse(&image).unwrap();
        assert_eq!(romfs.read_path("/Control.NACP").unwrap(), b"NACP");
        assert!(romfs.find("/missing.bin").is_none());
    }

    #[test]
    fn walks_subdirectories() {
        let image = build(&[("a.bin", b"A")], Some(("sub", &[("b.bin", b"BB")])));
        let romfs = RomFs::parse(&image).unwrap();
        assert_eq!(romfs.files().len(), 2);
        assert_eq!(romfs.read_path("/a.bin").unwrap(), b"A");
        assert_eq!(romfs.read_path("/sub/b.bin").unwrap(), b"BB");
    }

    /// A chain of `depth` nested directories, only the deepest holding a file.
    fn build_nested(depth: usize) -> Vec<u8> {
        /// The root's name is empty, so its entry is just the fixed part.
        const ROOT_SIZE: usize = DIR_ENTRY_SIZE;
        /// Every level below it is named "d", one byte padded to four.
        const LEVEL_SIZE: usize = DIR_ENTRY_SIZE + 4;
        let level_offset = |level: usize| (ROOT_SIZE + (level - 1) * LEVEL_SIZE) as u32;

        let mut dir_table: Vec<u8> = Vec::new();
        for level in 0..=depth {
            let first_child = if level < depth { level_offset(level + 1) } else { INVALID_OFFSET };
            let first_file = if level == depth { 0 } else { INVALID_OFFSET };
            dir_table.extend_from_slice(&0u32.to_le_bytes()); // parent
            dir_table.extend_from_slice(&INVALID_OFFSET.to_le_bytes()); // sibling
            dir_table.extend_from_slice(&first_child.to_le_bytes());
            dir_table.extend_from_slice(&first_file.to_le_bytes());
            dir_table.extend_from_slice(&INVALID_OFFSET.to_le_bytes()); // hash chain
            let name = if level == 0 { "" } else { "d" };
            dir_table.extend_from_slice(&(name.len() as u32).to_le_bytes());
            dir_table.extend_from_slice(name.as_bytes());
            while dir_table.len() % 4 != 0 {
                dir_table.push(0);
            }
        }

        let payload = b"DEEP";
        let mut file_table: Vec<u8> = Vec::new();
        file_table.extend_from_slice(&0u32.to_le_bytes()); // parent
        file_table.extend_from_slice(&INVALID_OFFSET.to_le_bytes()); // sibling
        file_table.extend_from_slice(&0u64.to_le_bytes()); // payload offset
        file_table.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        file_table.extend_from_slice(&INVALID_OFFSET.to_le_bytes()); // hash chain
        file_table.extend_from_slice(&5u32.to_le_bytes()); // name length
        file_table.extend_from_slice(b"f.bin");
        while file_table.len() % 4 != 0 {
            file_table.push(0);
        }

        let mut image = vec![0u8; HEADER_SIZE as usize];
        let dir_offset = image.len() as u64;
        image.extend_from_slice(&dir_table);
        let file_offset = image.len() as u64;
        image.extend_from_slice(&file_table);
        let data_offset = image.len() as u64;
        image.extend_from_slice(payload);

        image[0x00..0x08].copy_from_slice(&HEADER_SIZE.to_le_bytes());
        image[0x18..0x20].copy_from_slice(&dir_offset.to_le_bytes());
        image[0x20..0x28].copy_from_slice(&(dir_table.len() as u64).to_le_bytes());
        image[0x38..0x40].copy_from_slice(&file_offset.to_le_bytes());
        image[0x40..0x48].copy_from_slice(&(file_table.len() as u64).to_le_bytes());
        image[0x48..0x50].copy_from_slice(&data_offset.to_le_bytes());
        image
    }

    /// The cycle guard has to leave room for a legitimately deep tree: every
    /// directory is read twice, so budgeting one read each rejected a valid
    /// image the moment it nested at all.
    #[test]
    fn walks_a_deeply_nested_tree() {
        let image = build_nested(16);
        let romfs = RomFs::parse(&image).unwrap();
        let path = format!("{}/f.bin", "/d".repeat(16));
        assert_eq!(romfs.files().len(), 1);
        assert_eq!(romfs.read_path(&path).unwrap(), b"DEEP");
    }

    #[test]
    fn rejects_a_non_romfs_image() {
        let mut image = build(&[("a.bin", b"A")], None);
        image[0] = 0x40;
        assert!(matches!(RomFs::parse(&image), Err(Error::RomFs(_))));
    }

    #[test]
    fn rejects_a_truncated_image() {
        let image = build(&[("a.bin", b"A")], None);
        assert!(matches!(RomFs::parse(&image[..0x20]), Err(Error::Truncated { .. })));
    }

    #[test]
    fn rejects_a_self_referential_file_chain() {
        let mut image = build(&[("a.bin", b"A"), ("b.bin", b"B")], None);
        // Point the first file entry's sibling link back at itself.
        let file_table = crate::nsp::read_u64(&image, 0x38) as usize;
        image[file_table + 4..file_table + 8].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(RomFs::parse(&image), Err(Error::RomFs(_))));
    }

    #[test]
    fn rejects_a_file_entry_outside_the_table() {
        let mut image = build(&[("a.bin", b"A")], None);
        let dir_table = crate::nsp::read_u64(&image, 0x18) as usize;
        image[dir_table + 0x0C..dir_table + 0x10].copy_from_slice(&0x7000u32.to_le_bytes());
        assert!(matches!(RomFs::parse(&image), Err(Error::RomFs(_))));
    }

    #[test]
    fn read_rejects_an_extent_past_the_image() {
        let image = build(&[("a.bin", b"A")], None);
        let romfs = RomFs::parse(&image).unwrap();
        let mut file = romfs.find("/a.bin").unwrap().clone();
        file.size = 0x1000;
        assert!(romfs.read(&file).is_none());
    }
}
