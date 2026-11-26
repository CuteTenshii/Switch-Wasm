//! A flat RomFS image writer, for archives synthesized in memory.

use super::{FILE_ENTRY_SIZE, HEADER_SIZE, INVALID_OFFSET};
use crate::Error;

/// Where [`build`] places the file data region, as Nintendo's builder does.
const FILE_DATA_OFFSET: u64 = 0x200;
/// Longest entry name [`build`] accepts: `FS_MAX_PATH` less the terminator.
const MAX_NAME: usize = 0x300;

/// Build a RomFS image holding `files` (name, contents) in its root directory,
/// laid out as Nintendo's builder does, hash tables included: the guest's `fs`
/// finds entries through those, not by walking the tables.
pub fn build(files: &[(&str, &[u8])]) -> Result<Vec<u8>, Error> {
    let mut files: Vec<(&str, &[u8])> = files.to_vec();
    files.sort_unstable_by(|a, b| a.0.cmp(b.0));
    for (name, _) in &files {
        if name.is_empty() || name.contains('/') || name.len() > MAX_NAME {
            return Err(Error::RomFs(format!(
                "{name:?} is not a valid RomFS entry name"
            )));
        }
    }
    if let Some(pair) = files.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(Error::RomFs(format!(
            "{:?} appears twice in the image",
            pair[0].0
        )));
    }

    const ROOT: u32 = 0;
    let mut data: Vec<u8> = Vec::new();
    let mut file_table: Vec<u8> = Vec::new();
    let mut file_hashes = vec![INVALID_OFFSET; hash_table_len(files.len())];
    for (i, (name, payload)) in files.iter().enumerate() {
        data.resize(data.len().next_multiple_of(16), 0);
        let entry_offset = file_table.len() as u32;
        let next = if i + 1 == files.len() {
            INVALID_OFFSET
        } else {
            entry_offset + (FILE_ENTRY_SIZE + name.len().next_multiple_of(4)) as u32
        };
        let bucket = path_hash(ROOT, name.as_bytes()) as usize % file_hashes.len();
        file_table.extend_from_slice(&ROOT.to_le_bytes());
        file_table.extend_from_slice(&next.to_le_bytes());
        file_table.extend_from_slice(&(data.len() as u64).to_le_bytes());
        file_table.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        file_table.extend_from_slice(&file_hashes[bucket].to_le_bytes());
        file_table.extend_from_slice(&(name.len() as u32).to_le_bytes());
        file_table.extend_from_slice(name.as_bytes());
        file_table.resize(file_table.len().next_multiple_of(4), 0);
        file_hashes[bucket] = entry_offset;
        data.extend_from_slice(payload);
    }

    let mut dir_hashes = vec![INVALID_OFFSET; hash_table_len(1)];
    let root_bucket = path_hash(ROOT, b"") as usize % dir_hashes.len();
    let first_file = if files.is_empty() { INVALID_OFFSET } else { 0 };
    let mut dir_table: Vec<u8> = Vec::new();
    for field in [
        ROOT,
        INVALID_OFFSET,
        INVALID_OFFSET,
        first_file,
        INVALID_OFFSET,
        0,
    ] {
        dir_table.extend_from_slice(&field.to_le_bytes());
    }
    dir_hashes[root_bucket] = ROOT;

    let words = |table: &[u32]| -> Vec<u8> { table.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let dir_hashes = words(&dir_hashes);
    let file_hashes = words(&file_hashes);

    let mut image = vec![0u8; FILE_DATA_OFFSET as usize];
    image.extend_from_slice(&data);
    image.resize(image.len().next_multiple_of(4), 0);
    let mut header = [0u64; 10];
    header[0] = HEADER_SIZE;
    for (slot, table) in [&dir_hashes, &dir_table, &file_hashes, &file_table]
        .into_iter()
        .enumerate()
    {
        header[1 + 2 * slot] = image.len() as u64;
        header[2 + 2 * slot] = table.len() as u64;
        image.extend_from_slice(table);
    }
    header[9] = FILE_DATA_OFFSET;
    for (i, field) in header.iter().enumerate() {
        image[i * 8..i * 8 + 8].copy_from_slice(&field.to_le_bytes());
    }
    Ok(image)
}

/// The hash the guest's `fs` buckets an entry by: its parent's entry offset and its name.
fn path_hash(parent: u32, name: &[u8]) -> u32 {
    name.iter().fold(parent ^ 123_456_789, |hash, &byte| {
        hash.rotate_right(5) ^ u32::from(byte)
    })
}

/// Bucket count for a hash table over `entries` entries, as Nintendo's builder picks it.
fn hash_table_len(entries: usize) -> usize {
    match entries {
        0..3 => 3,
        3..19 => entries | 1,
        _ => {
            let mut count = entries;
            while [2, 3, 5, 7, 11, 13, 17]
                .iter()
                .any(|&p| count.is_multiple_of(p))
            {
                count += 1;
            }
            count
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::romfs::{RomFs, RomFsIndex};

    #[test]
    fn a_built_image_reads_back_through_the_parser_and_the_index() {
        let files: [(&str, &[u8]); 3] = [
            ("version.dat", &[0, 0, 0, 0x1A]),
            ("a.bin", b"seventeen bytes!!"),
            ("empty", b""),
        ];
        let image = super::build(&files).unwrap();
        let romfs = RomFs::parse(&image).unwrap();
        assert_eq!(romfs.files().len(), files.len());
        for (name, payload) in files {
            assert_eq!(romfs.read_path(&format!("/{name}")).unwrap(), payload);
        }

        let index = RomFsIndex::read(&crate::source::SliceSource(&image)).unwrap();
        let mut indexed: Vec<(u64, u64, &str)> = index.files().collect();
        indexed.retain(|&(start, end, _)| start != end);
        let data = super::FILE_DATA_OFFSET;
        // Sorted by name, each payload 16-aligned.
        assert_eq!(
            indexed,
            vec![
                (data, data + 17, "/a.bin"),
                (data + 32, data + 36, "/version.dat"),
            ]
        );
    }

    /// Find `name` in the root the way the guest's `fs` does: hash, bucket, chain.
    fn lookup_by_hash(image: &[u8], name: &str) -> Option<(u64, u64)> {
        let at = |offset: u64| crate::nsp::read_u64(image, offset as usize);
        let (dir_hashes, dir_hashes_len) = (at(0x08) as usize, at(0x10) as usize / 4);
        let (dir_table, file_hashes) = (at(0x18) as usize, at(0x28) as usize);
        let (file_hashes_len, file_table) = (at(0x30) as usize / 4, at(0x38) as usize);
        let word = |offset: usize| crate::nsp::read_u32(image, offset);

        let root_bucket = super::path_hash(0, b"") as usize % dir_hashes_len;
        let root = word(dir_hashes + 4 * root_bucket);
        assert_eq!(root, 0, "the root is in its bucket");
        assert_eq!(word(dir_table + root as usize + 0x14), 0, "with no name");

        let bucket = super::path_hash(root, name.as_bytes()) as usize % file_hashes_len;
        let mut entry = word(file_hashes + 4 * bucket);
        while entry != INVALID_OFFSET {
            let e = file_table + entry as usize;
            let len = word(e + 0x1C) as usize;
            if word(e) == root && &image[e + 0x20..e + 0x20 + len] == name.as_bytes() {
                return Some((at(e as u64 + 8), at(e as u64 + 0x10)));
            }
            entry = word(e + 0x18);
        }
        None
    }

    #[test]
    fn every_built_file_is_found_through_the_hash_tables() {
        // Enough files for the prime-sized table and for buckets to collide.
        let names: Vec<String> = (0..40).map(|i| format!("file_{i}")).collect();
        let payloads: Vec<Vec<u8>> = (0..40u8).map(|i| vec![i; usize::from(i)]).collect();
        let files: Vec<(&str, &[u8])> = names
            .iter()
            .zip(&payloads)
            .map(|(n, p)| (n.as_str(), p.as_slice()))
            .collect();
        let image = super::build(&files).unwrap();
        assert_eq!(super::hash_table_len(40), 41);
        for (name, payload) in files {
            let (offset, size) = lookup_by_hash(&image, name).unwrap();
            let start = (super::FILE_DATA_OFFSET + offset) as usize;
            assert_eq!(&image[start..start + size as usize], payload, "{name}");
        }
        assert!(lookup_by_hash(&image, "file_40").is_none());
    }

    #[test]
    fn build_refuses_names_the_root_cannot_hold() {
        for files in [
            vec![("", &b""[..])],
            vec![("sub/a.bin", &b""[..])],
            vec![("a.bin", &b"1"[..]), ("a.bin", &b"2"[..])],
        ] {
            assert!(
                matches!(super::build(&files), Err(Error::RomFs(_))),
                "{files:?}"
            );
        }
        let image = super::build(&[]).unwrap();
        assert!(RomFs::parse(&image).unwrap().files().is_empty());
    }
}
