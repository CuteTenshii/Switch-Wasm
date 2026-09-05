//! Reading an NCA section that is not all in the file.
//!
//! A sparse section stores only the ranges that hold anything. A
//! [`crate::bucket`] tree says which range came from where: storage 0 is the
//! bytes that were kept, storage 1 is a hole. What the section is *declared*
//! to be (the size in the NCA's section table) is the size after the holes
//! are put back.
//!
//! `SparseInfo` (FS header 0x148) carries the table, then two fields nothing
//! else in an FS header has:
//!
//! ```text
//! 0x20  physical offset: where the stored body really is in the NCA,
//!       which is not where the section table says the section is
//! 0x28  generation (u16): replaces the counter's generation word for the
//!       table's own bytes, and for those bytes only
//! ```
//!
//! **The layer sits underneath decryption**, which is the part that reads
//! backwards: `nn::fssystem` builds the sparse storage over the raw NCA body
//! and puts the AES-CTR layer on top of *that*, counting from the section's
//! ordinary offset. So the stored bytes were encrypted at the position they
//! occupy in the reassembled section, not the one they are stored at, and a
//! hole decrypts to the keystream rather than to zeroes. That is what real
//! `fs` produces; a hole is a range no reader is expected to look at.

use crate::bucket;
use crate::nca::{BktrTable, BKTR_MAGIC};
use crate::source::{ByteSource, SliceSource};
use crate::Error;

/// The range was kept, at `phys` in the stored body.
pub(crate) const STORAGE_DATA: u32 = 0;
/// The range is a hole.
pub(crate) const STORAGE_HOLE: u32 = 1;

/// One range of the reassembled section, and where its bytes are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    virt: u64,
    phys: u64,
    storage: u32,
}

impl bucket::Entry for Entry {
    const NODE_SIZE: u64 = 0x4000;
    /// `s64 virt, s64 phys, s32 storage index`, 0x14, not padded to 0x18.
    const SIZE: u64 = 0x14;

    fn parse(raw: &[u8]) -> Entry {
        Entry {
            virt: crate::nsp::read_u64(raw, 0),
            phys: crate::nsp::read_u64(raw, 8),
            storage: crate::nsp::read_u32(raw, 0x10),
        }
    }

    fn virt(&self) -> u64 {
        self.virt
    }
}

/// Where a section's stored body lives, and how to put the holes back.
#[derive(Debug, Clone)]
pub struct SparseTable {
    entries: Vec<Entry>,
    /// Size of the stored body, which every entry has to point inside.
    body_len: u64,
}

impl SparseTable {
    /// Read the table out of the stored body.
    ///
    /// `body` is the NCA at [`FsHeader::sparse_physical_offset`], `table` the
    /// `SparseInfo` bucket header, and `meta` the already-decrypted table
    /// bytes: the caller decrypts them because they use a counter of their
    /// own ([`crate::nca::FsHeader::sparse_counter`]) that nothing else in
    /// the section does.
    pub fn parse(meta: &[u8], table: BktrTable, body_len: u64) -> Result<SparseTable, Error> {
        if table.magic != BKTR_MAGIC {
            return Err(Error::Nca(format!(
                "sparse table has magic {:#010x}, not BKTR",
                table.magic
            )));
        }
        let (entries, _) = bucket::read::<Entry, _>(&SliceSource(meta), table.entries, "sparse")?;
        let sparse = SparseTable { entries, body_len };
        sparse.validate()?;
        Ok(sparse)
    }

    /// Check every entry once, so a read never has to.
    fn validate(&self) -> Result<(), Error> {
        for (i, entry) in self.entries.iter().enumerate() {
            match entry.storage {
                STORAGE_HOLE => continue,
                STORAGE_DATA => {}
                other => {
                    return Err(Error::Nca(format!(
                    "sparse entry {i} names storage {other}, which is neither the body nor a hole"
                )))
                }
            }
            if entry.phys > self.body_len {
                return Err(Error::OutOfRange {
                    what: format!("sparse entry {i}"),
                    start: entry.phys,
                    end: entry.phys,
                    available: self.body_len,
                });
            }
        }
        Ok(())
    }

    /// Read `out` bytes of the reassembled section at `offset`, still
    /// encrypted: the caller decrypts, because the counter is numbered from
    /// this offset and not from where the bytes were stored.
    ///
    /// `len` is the section's declared size, which is what the last entry
    /// runs to; the table's own end offset covers only the stored body.
    pub fn read_raw<S: ByteSource>(
        &self,
        body: &S,
        len: u64,
        offset: u64,
        out: &mut [u8],
    ) -> Result<usize, Error> {
        if offset >= len {
            return Ok(0);
        }
        let want = ((out.len() as u64).min(len - offset)) as usize;
        let mut done = 0;
        while done < want {
            let at = offset + done as u64;
            let index = bucket::index_of(&self.entries, at);
            let entry = self.entries[index];
            let next = self.entries.get(index + 1).map_or(len, |e| e.virt);
            if next <= at {
                break;
            }
            let take = (want - done).min((next - at) as usize);
            let into = &mut out[done..done + take];
            if entry.storage == STORAGE_HOLE {
                into.fill(0);
            } else {
                let got = body.read_at(entry.phys + (at - entry.virt), into)?;
                if got < take {
                    return Ok(done + got);
                }
            }
            done += take;
        }
        Ok(done)
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::bucket::Entry as _;

    /// Lay out a sparse table over `(virtual offset, physical offset,
    /// storage)` triples, for a fixture that places the body itself.
    pub(crate) fn write_table(entries: &[(u64, u64, u32)], end: u64) -> Vec<u8> {
        let rows: Vec<Vec<u8>> = entries
            .iter()
            .map(|&(virt, phys, storage)| {
                let mut row = vec![0u8; Entry::SIZE as usize];
                row[..8].copy_from_slice(&virt.to_le_bytes());
                row[8..16].copy_from_slice(&phys.to_le_bytes());
                row[0x10..0x14].copy_from_slice(&storage.to_le_bytes());
                row
            })
            .collect();
        crate::bucket::testing::write_table::<Entry>(&rows, end)
    }

    /// What one entry of the fixture describes: bytes that were kept, or a
    /// hole of that many bytes.
    pub(crate) enum Range {
        Data(Vec<u8>),
        Hole(usize),
    }

    /// Build a stored body: the kept bytes, then the table describing them.
    /// Returns the body, the section it reassembles to, and the `SparseInfo`
    /// bucket header.
    pub(crate) fn build(ranges: &[Range]) -> (Vec<u8>, Vec<u8>, BktrTable) {
        let mut body = Vec::new();
        let mut whole = Vec::new();
        let mut entries: Vec<(u64, u64, u32)> = Vec::new();
        for range in ranges {
            let virt = whole.len() as u64;
            let (storage, phys) = match range {
                Range::Data(bytes) => {
                    let phys = body.len() as u64;
                    body.extend_from_slice(bytes);
                    whole.extend_from_slice(bytes);
                    (STORAGE_DATA, phys)
                }
                Range::Hole(len) => {
                    whole.resize(whole.len() + len, 0);
                    (STORAGE_HOLE, 0)
                }
            };
            entries.push((virt, phys, storage));
        }
        // The table follows the kept bytes, which is what `GetPhysicalSize`
        // means by "bucket offset plus bucket size".
        let table_offset = body.len() as u64;
        let meta = write_table(&entries, whole.len() as u64);
        body.extend_from_slice(&meta);
        let table = BktrTable {
            offset: table_offset,
            size: meta.len() as u64,
            magic: BKTR_MAGIC,
            entries: entries.len() as u32,
        };
        (body, whole, table)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{build, Range};
    use super::*;

    fn sample() -> (Vec<u8>, Vec<u8>, BktrTable, SparseTable) {
        let (body, whole, table) = build(&[
            Range::Data((0..0x40u8).collect()),
            Range::Hole(0x100),
            Range::Data((0..0x80u8).map(|b| b ^ 0x33).collect()),
            Range::Hole(0x20),
        ]);
        let meta = body[table.offset as usize..].to_vec();
        let sparse = SparseTable::parse(&meta, table, table.offset).expect("parse");
        (body, whole, table, sparse)
    }

    #[test]
    fn puts_the_holes_back_where_the_table_says() {
        let (body, whole, table, sparse) = sample();
        let len = whole.len() as u64;
        let stored = crate::source::SliceSource(&body[..table.offset as usize]);

        let mut all = vec![0xAAu8; whole.len()];
        assert_eq!(
            sparse.read_raw(&stored, len, 0, &mut all).unwrap(),
            whole.len()
        );
        assert_eq!(all, whole);

        // Ranges inside each kind of range and across every boundary.
        for &(offset, size) in &[
            (0u64, 1usize),
            (0x3f, 2),    // the last kept byte and the first of the hole
            (0x80, 0x40), // inside the hole
            (0x13f, 4),   // out of the hole and into the next kept range
            (0x140, 0x80),
            (0x1bf, 2), // into the trailing hole
            (0x10, 0x1d0),
        ] {
            let mut out = vec![0xAAu8; size];
            assert_eq!(
                sparse.read_raw(&stored, len, offset, &mut out).unwrap(),
                size,
                "short read at {offset:#x}+{size:#x}"
            );
            assert_eq!(
                out,
                &whole[offset as usize..offset as usize + size],
                "wrong bytes at {offset:#x}+{size:#x}"
            );
        }
    }

    /// The section is longer than what was stored: that is the whole point,
    /// so a read is bounded by the declared size, not by the body.
    #[test]
    fn the_section_is_longer_than_the_body_that_stores_it() {
        let (body, whole, table, sparse) = sample();
        assert!(table.offset < whole.len() as u64, "body is the smaller one");
        let stored = crate::source::SliceSource(&body[..table.offset as usize]);
        let len = whole.len() as u64;

        let mut out = vec![0u8; 0x40];
        assert_eq!(
            sparse.read_raw(&stored, len, len - 0x10, &mut out).unwrap(),
            0x10
        );
        assert_eq!(sparse.read_raw(&stored, len, len, &mut out).unwrap(), 0);
    }

    #[test]
    fn rejects_a_table_that_is_not_one() {
        let (body, _, table, _) = sample();
        let meta = body[table.offset as usize..].to_vec();

        let mut wrong = table;
        wrong.magic = 0;
        assert!(matches!(
            SparseTable::parse(&meta, wrong, table.offset),
            Err(Error::Nca(_))
        ));

        // A storage index that is neither the body nor a hole.
        let mut bad = meta.clone();
        let set = bucket::node_storage_size::<Entry>(table.entries) as usize;
        bad[set + bucket::NODE_HEADER_SIZE as usize + 0x10] = 2;
        assert!(matches!(
            SparseTable::parse(&bad, table, table.offset),
            Err(Error::Nca(_))
        ));

        // An entry pointing past the stored body.
        let mut past = meta.clone();
        let at = set + bucket::NODE_HEADER_SIZE as usize;
        past[at + 8..at + 16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            SparseTable::parse(&past, table, table.offset),
            Err(Error::OutOfRange { .. })
        ));
    }
}
