//! Byte ranges of a RomFS attributed to the files they hold.

use super::{walk, HEADER_SIZE};
use crate::source::ByteSource;
use crate::Error;

/// The largest metadata table [`RomFsIndex::read`] accepts before calling the
/// header corrupt.
const MAX_TABLE: u64 = 64 << 20;

/// Maps byte ranges of a RomFS to file names, from the metadata tables alone,
/// so raw `IStorage` reads can be attributed to files.
#[derive(Debug, Clone, Default)]
pub struct RomFsIndex {
    /// `(start, end, path)`, absolute within the image and sorted by `start`
    /// (payloads never overlap, so `end` is sorted too).
    extents: Vec<(u64, u64, String)>,
}

impl RomFsIndex {
    /// Read the header and the two metadata tables out of `src`.
    pub fn read(src: &dyn ByteSource) -> Result<RomFsIndex, Error> {
        let mut header = [0u8; HEADER_SIZE as usize];
        src.read_exact_at(0, &mut header)?;
        if crate::nsp::read_u64(&header, 0) != HEADER_SIZE {
            return Err(Error::RomFs("not a RomFS image".into()));
        }
        let table = |offset_at: usize, what: &str| -> Result<Vec<u8>, Error> {
            let offset = crate::nsp::read_u64(&header, offset_at);
            let size = crate::nsp::read_u64(&header, offset_at + 8);
            if size > MAX_TABLE {
                return Err(Error::RomFs(format!("{what} claims {size:#x} bytes")));
            }
            src.read_vec(offset, size)
        };
        let dir_table = table(0x18, "RomFS directory metadata table")?;
        let file_table = table(0x38, "RomFS file metadata table")?;
        let data_offset = crate::nsp::read_u64(&header, 0x48);
        let mut extents: Vec<(u64, u64, String)> = walk(&dir_table, &file_table)?
            .into_iter()
            .filter_map(|f| {
                let start = data_offset.checked_add(f.offset)?;
                Some((start, start.checked_add(f.size)?, f.path))
            })
            .collect();
        extents.sort_unstable_by_key(|&(start, _, _)| start);
        Ok(RomFsIndex { extents })
    }

    /// Every file as `(start, end, path)`, absolute within the image, in image order.
    pub fn files(&self) -> impl Iterator<Item = (u64, u64, &str)> {
        self.extents
            .iter()
            .map(|(start, end, path)| (*start, *end, path.as_str()))
    }

    /// Every file the range `offset..offset + len` touches, with how many of
    /// its bytes fall inside that file, in image order.
    pub fn files_in(&self, offset: u64, len: u64) -> impl Iterator<Item = (&str, u64)> {
        let end = offset.saturating_add(len);
        let first = self.extents.partition_point(|&(_, e, _)| e <= offset);
        self.extents[first..]
            .iter()
            .take_while(move |&&(start, _, _)| start < end)
            .filter_map(move |(start, e, path)| {
                let overlap = end.min(*e).saturating_sub(offset.max(*start));
                (overlap > 0).then_some((path.as_str(), overlap))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_index_names_the_files_a_range_touches() {
        let image = crate::romfs::tests::build_tree(
            &[("a.bin", b"AAAA")],
            Some(("sub", &[("b.bin", b"BB")])),
        );
        let index = RomFsIndex::read(&crate::source::SliceSource(&image)).unwrap();
        let data = crate::nsp::read_u64(&image, 0x48);
        let hits = |offset: u64, len: u64| -> Vec<(String, u64)> {
            index
                .files_in(offset, len)
                .map(|(path, n)| (path.to_owned(), n))
                .collect()
        };
        assert_eq!(hits(data + 1, 2), vec![("/a.bin".to_owned(), 2)]);
        // Across the boundary: three bytes of one file, one of the next.
        assert_eq!(
            hits(data + 1, 4),
            vec![("/a.bin".to_owned(), 3), ("/sub/b.bin".to_owned(), 1)]
        );
        // The header and the tables are no file's.
        assert!(hits(0, HEADER_SIZE).is_empty());
        assert!(hits(data + 6, 16).is_empty());
    }
}
