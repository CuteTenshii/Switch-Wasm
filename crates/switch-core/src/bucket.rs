//! The bucket tree an NCA writes its storage tables as.
//!
//! A section that is not simply "these bytes, in this order" carries a table
//! saying what is where: which LZ4 block covers which range
//! ([`crate::compressed`]), which ranges are stored at all
//! ([`crate::sparse`]), which ranges an update replaced ([`crate::bktr`]).
//! All of them are the same structure, differing only in node size and in
//! what one entry holds.
//!
//! ```text
//! + 0x0000   L1 node: u32 index, u32 count, u64 end offset, then `count`
//!            u64 offsets — one per entry set, or one per L2 node when there
//!            are more entry sets than a node has room for offsets
//! then       one node per entry set: the same header, then `count` entries
//! ```
//!
//! Only the entry sets are read. The index nodes exist to find one entry set
//! without holding the whole table, which is exactly what holding the whole
//! table makes unnecessary — a retail table is a couple of megabytes against
//! a container of gigabytes, and a binary search over it costs less than
//! walking a tree per read.

use crate::source::ByteSource;
use crate::Error;

/// `u32 index, u32 count, u64 end offset`, at the head of every node.
pub(crate) const NODE_HEADER_SIZE: u64 = 0x10;

/// The most table any of this will read in. Echoes of Wisdom's compression
/// table — the largest in any container to hand — is 2.3 MiB over 98,846
/// entries; this only exists so a corrupt header cannot ask the browser for
/// an allocation it has no way to make.
pub(crate) const MAX_TABLE: u64 = 64 << 20;

/// One kind of table entry: how big it is, how they are paged, and how to
/// read one.
pub(crate) trait Entry: Sized {
    /// Node size, which the format fixes per table rather than storing.
    const NODE_SIZE: u64;
    /// On-disk size of one entry, padding included.
    const SIZE: u64;
    fn parse(raw: &[u8]) -> Self;
    /// Where in the virtual image this entry's range starts. Entries are
    /// sorted by it, and it is what a lookup searches on.
    fn virt(&self) -> u64;
}

/// Entries per entry-set node, and offsets per index node: both are however
/// many fit in a node once its header is out.
pub(crate) fn entries_per_node<E: Entry>() -> u64 {
    (E::NODE_SIZE - NODE_HEADER_SIZE) / E::SIZE
}

pub(crate) fn offsets_per_node<E: Entry>() -> u64 {
    (E::NODE_SIZE - NODE_HEADER_SIZE) / 8
}

pub(crate) fn entry_set_count<E: Entry>(entries: u32) -> u64 {
    u64::from(entries).div_ceil(entries_per_node::<E>())
}

/// The index nodes ahead of the entries: one L1 node, plus a row of L2 nodes
/// when there are more entry sets than L1 can hold offsets for.
pub(crate) fn node_storage_size<E: Entry>(entries: u32) -> u64 {
    let sets = entry_set_count::<E>(entries);
    let per_node = offsets_per_node::<E>();
    let l2 = if sets <= per_node {
        0
    } else {
        let count = sets.div_ceil(per_node);
        (sets - (per_node - (count - 1))).div_ceil(per_node)
    };
    (1 + l2) * E::NODE_SIZE
}

pub(crate) fn entry_storage_size<E: Entry>(entries: u32) -> u64 {
    entry_set_count::<E>(entries) * E::NODE_SIZE
}

/// Read a table's entries into one sorted list, and the virtual offset it
/// ends at.
///
/// `src` must cover exactly the table: the index nodes, then the entry sets.
pub(crate) fn read<E: Entry, S: ByteSource>(
    src: &S,
    entries: u32,
    what: &str,
) -> Result<(Vec<E>, u64), Error> {
    if entries == 0 {
        return Err(Error::Nca(format!("{what} table has no entries")));
    }
    let node_bytes = node_storage_size::<E>(entries);
    let entry_bytes = entry_storage_size::<E>(entries);
    if node_bytes.saturating_add(entry_bytes) > src.len() {
        return Err(Error::Nca(format!(
            "{what} table is {:#x} bytes, too small for {entries} entries",
            src.len()
        )));
    }
    if entry_bytes > MAX_TABLE {
        return Err(Error::TooLarge {
            what: format!("{what} table"),
            len: entry_bytes,
            max: MAX_TABLE,
        });
    }
    let raw = src.read_vec(node_bytes, entry_bytes)?;

    let mut out = Vec::new();
    out.try_reserve_exact(entries as usize)
        .map_err(|_| Error::TooLarge {
            what: format!("{what} entries"),
            len: u64::from(entries) * E::SIZE,
            max: MAX_TABLE,
        })?;
    for set in 0..entry_set_count::<E>(entries) {
        let node = (set * E::NODE_SIZE) as usize;
        let index = crate::nsp::read_u32(&raw, node);
        let count = u64::from(crate::nsp::read_u32(&raw, node + 4));
        if u64::from(index) != set {
            return Err(Error::Nca(format!(
                "{what} entry set {set} is labelled {index}"
            )));
        }
        if count == 0 || count > entries_per_node::<E>() {
            return Err(Error::Nca(format!(
                "{what} entry set {set} holds {count} entries"
            )));
        }
        for i in 0..count {
            let at = node + (NODE_HEADER_SIZE + i * E::SIZE) as usize;
            let entry = E::parse(&raw[at..at + E::SIZE as usize]);
            if out.last().is_some_and(|last: &E| last.virt() >= entry.virt()) {
                return Err(Error::Nca(format!(
                    "{what} entries are not in ascending order"
                )));
            }
            out.push(entry);
        }
    }
    if out.len() != entries as usize {
        return Err(Error::Nca(format!(
            "{what} table declares {entries} entries and holds {}",
            out.len()
        )));
    }
    if out[0].virt() != 0 {
        return Err(Error::Nca(format!(
            "{what} table starts at {:#x}, not 0",
            out[0].virt()
        )));
    }
    // The end offset the last entry set carries is the size of the image the
    // whole table describes.
    let last = ((entry_set_count::<E>(entries) - 1) * E::NODE_SIZE) as usize;
    let end = crate::nsp::read_u64(&raw, last + 8);
    if end == 0 {
        return Err(Error::Nca(format!("{what} table ends at offset 0")));
    }
    Ok((out, end))
}

/// The entry covering `at`, given a list [`read`] produced. Every list starts
/// at virtual offset 0, so there is always one.
pub(crate) fn index_of<E: Entry>(entries: &[E], at: u64) -> usize {
    entries.partition_point(|e| e.virt() <= at) - 1
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// Lay out `entries` as one entry set, preceded by the index nodes the
    /// format requires but this reader never looks at. Returns the table
    /// bytes; `end` is the virtual offset the image runs to.
    pub(crate) fn write_table<E: Entry>(entries: &[Vec<u8>], end: u64) -> Vec<u8> {
        let count = entries.len() as u32;
        let mut out = vec![0u8; node_storage_size::<E>(count) as usize];
        let set = out.len();
        out.resize(set + E::NODE_SIZE as usize, 0);
        out[set..set + 4].copy_from_slice(&0u32.to_le_bytes());
        out[set + 4..set + 8].copy_from_slice(&count.to_le_bytes());
        out[set + 8..set + 16].copy_from_slice(&end.to_le_bytes());
        for (i, entry) in entries.iter().enumerate() {
            let at = set + (NODE_HEADER_SIZE + i as u64 * E::SIZE) as usize;
            out[at..at + entry.len()].copy_from_slice(entry);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SliceSource;

    /// A stand-in with the compression table's geometry.
    struct Fake {
        virt: u64,
    }

    impl Entry for Fake {
        const NODE_SIZE: u64 = 0x4000;
        const SIZE: u64 = 0x18;
        fn parse(raw: &[u8]) -> Fake {
            Fake {
                virt: crate::nsp::read_u64(raw, 0),
            }
        }
        fn virt(&self) -> u64 {
            self.virt
        }
    }

    fn entry(virt: u64) -> Vec<u8> {
        let mut e = vec![0u8; Fake::SIZE as usize];
        e[..8].copy_from_slice(&virt.to_le_bytes());
        e
    }

    /// The figures `nn::fssystem` derives for a real title's table: Echoes of
    /// Wisdom's is 98,846 entries in 0x268000 bytes, of which 0x4000 is the
    /// one index node and 0x244000 the 145 entry sets.
    #[test]
    fn sizes_a_table_the_way_the_sdk_does() {
        assert_eq!(entries_per_node::<Fake>(), 682);
        assert_eq!(offsets_per_node::<Fake>(), 2046);
        assert_eq!(entry_set_count::<Fake>(98_846), 145);
        assert_eq!(node_storage_size::<Fake>(98_846), 0x4000);
        assert_eq!(entry_storage_size::<Fake>(98_846), 0x244000);
        // Past one node of offsets, a row of L2 nodes appears.
        assert_eq!(node_storage_size::<Fake>(2047 * 682), 0x8000);
    }

    #[test]
    fn reads_entries_in_order_and_the_end_they_run_to() {
        let raw = testing::write_table::<Fake>(&[entry(0), entry(0x100), entry(0x180)], 0x400);
        let (entries, end) = read::<Fake, _>(&SliceSource(&raw), 3, "test").expect("read");
        assert_eq!(entries.len(), 3);
        assert_eq!(end, 0x400);
        assert_eq!(index_of(&entries, 0), 0);
        assert_eq!(index_of(&entries, 0xFF), 0);
        assert_eq!(index_of(&entries, 0x100), 1);
        assert_eq!(index_of(&entries, 0x3FF), 2);
    }

    #[test]
    fn rejects_a_table_that_does_not_describe_itself() {
        let ordered = testing::write_table::<Fake>(&[entry(0), entry(0x100)], 0x400);
        assert!(read::<Fake, _>(&SliceSource(&ordered), 0, "test").is_err(), "no entries");
        // A count the table cannot hold, and one it does not match.
        assert!(read::<Fake, _>(&SliceSource(&ordered[..0x100]), 2, "test").is_err());
        assert!(read::<Fake, _>(&SliceSource(&ordered), 3, "test").is_err());

        let unordered = testing::write_table::<Fake>(&[entry(0), entry(0x100), entry(0x80)], 0x400);
        assert!(read::<Fake, _>(&SliceSource(&unordered), 3, "test").is_err());

        let offset = testing::write_table::<Fake>(&[entry(0x40), entry(0x100)], 0x400);
        assert!(read::<Fake, _>(&SliceSource(&offset), 2, "test").is_err(), "must start at 0");

        let zero_end = testing::write_table::<Fake>(&[entry(0)], 0);
        assert!(read::<Fake, _>(&SliceSource(&zero_end), 1, "test").is_err());
    }
}
