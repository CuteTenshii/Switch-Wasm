//! An NCA section stored as LZ4 blocks plus a [`crate::bucket`] table.
//!
//! Offsets are relative to the image the hash layer exposes, not the section.
//! An entry is `u64 virtual offset, u64 physical offset, u8 kind, u32
//! physical size`; the table is located by `CompressionInfo` at 0x178.

use crate::bucket;
use crate::nca::{BktrTable, BKTR_MAGIC};
use crate::source::ByteSource;
use crate::Error;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

const KIND_NONE: u8 = 0;
/// Stored not at all: the range reads as zeroes.
const KIND_ZEROS: u8 = 1;
const KIND_LZ4: u8 = 3;

/// Decompressed blocks kept, since the guest reads RomFS in small pieces.
const CACHED_BLOCKS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    virt: u64,
    phys: u64,
    phys_size: u32,
    kind: u8,
}

impl bucket::Entry for Entry {
    const NODE_SIZE: u64 = 0x4000;
    const SIZE: u64 = 0x18;

    fn parse(raw: &[u8]) -> Entry {
        Entry {
            virt: crate::nsp::read_u64(raw, 0),
            phys: crate::nsp::read_u64(raw, 8),
            kind: raw[0x10],
            phys_size: crate::nsp::read_u32(raw, 0x14),
        }
    }

    fn virt(&self) -> u64 {
        self.virt
    }
}

pub struct CompressedStorage<S> {
    /// The image as stored: compressed data first, then the table.
    inner: S,
    entries: Vec<Entry>,
    data_end: u64,
    len: u64,
    cache: RefCell<VecDeque<(usize, Rc<[u8]>)>>,
}

impl<S: ByteSource> CompressedStorage<S> {
    /// Read the table out of `inner` (the hash layer's image) and index it.
    pub fn new(inner: S, table: BktrTable) -> Result<CompressedStorage<S>, Error> {
        if table.magic != BKTR_MAGIC {
            return Err(Error::Nca(format!(
                "compression table has magic {:#010x}, not BKTR",
                table.magic
            )));
        }
        let end = table
            .offset
            .checked_add(table.size)
            .ok_or(Error::Overflow)?;
        if end > inner.len() {
            return Err(Error::OutOfRange {
                what: "compression table".into(),
                start: table.offset,
                end,
                available: inner.len(),
            });
        }
        let region = crate::source::Window::new(&inner, table.offset, table.size, "compression")?;
        let (entries, len) = bucket::read::<Entry, _>(&region, table.entries, "compression")?;
        let storage = CompressedStorage {
            inner,
            entries,
            data_end: table.offset,
            len,
            cache: RefCell::new(VecDeque::with_capacity(CACHED_BLOCKS)),
        };
        storage.validate()?;
        Ok(storage)
    }

    fn validate(&self) -> Result<(), Error> {
        for (i, entry) in self.entries.iter().enumerate() {
            let next = self.entries.get(i + 1).map_or(self.len, |e| e.virt);
            if next <= entry.virt {
                return Err(Error::Nca(format!(
                    "compression entry {i} covers nothing: {:#x}..{next:#x}",
                    entry.virt
                )));
            }
            let physical = match entry.kind {
                KIND_ZEROS => continue,
                KIND_NONE => next - entry.virt,
                KIND_LZ4 => u64::from(entry.phys_size),
                other => {
                    return Err(Error::Nca(format!(
                        "compression entry {i} has unsupported kind {other}"
                    )))
                }
            };
            let end = entry.phys.checked_add(physical).ok_or(Error::Overflow)?;
            if end > self.data_end {
                return Err(Error::OutOfRange {
                    what: format!("compression entry {i}"),
                    start: entry.phys,
                    end,
                    available: self.data_end,
                });
            }
        }
        Ok(())
    }

    fn block(&self, index: usize, virtual_size: usize) -> Result<Rc<[u8]>, Error> {
        if let Some((_, block)) = self.cache.borrow().iter().find(|(i, _)| *i == index) {
            return Ok(Rc::clone(block));
        }
        let entry = self.entries[index];
        let compressed = self
            .inner
            .read_vec(entry.phys, u64::from(entry.phys_size))?;
        let plain = crate::lz4::decompress_block(&compressed, virtual_size).map_err(|e| {
            Error::Nca(format!(
                "compressed block at {:#x} ({:#x} bytes): {e}",
                entry.virt, entry.phys_size
            ))
        })?;
        let block: Rc<[u8]> = Rc::from(plain);
        let mut cache = self.cache.borrow_mut();
        if cache.len() == CACHED_BLOCKS {
            cache.pop_back();
        }
        cache.push_front((index, Rc::clone(&block)));
        Ok(block)
    }
}

impl<S: ByteSource> ByteSource for CompressedStorage<S> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        if offset >= self.len {
            return Ok(0);
        }
        let want = ((out.len() as u64).min(self.len - offset)) as usize;
        let mut done = 0;
        while done < want {
            let at = offset + done as u64;
            let index = bucket::index_of(&self.entries, at);
            let entry = self.entries[index];
            let next = self.entries.get(index + 1).map_or(self.len, |e| e.virt);
            let within = (at - entry.virt) as usize;
            let take = (want - done).min((next - at) as usize);
            let into = &mut out[done..done + take];
            match entry.kind {
                KIND_ZEROS => into.fill(0),
                KIND_NONE => {
                    let got = self.inner.read_at(entry.phys + within as u64, into)?;
                    if got < take {
                        return Ok(done + got);
                    }
                }
                _ => {
                    let block = self.block(index, (next - entry.virt) as usize)?;
                    into.copy_from_slice(&block[within..within + take]);
                }
            }
            done += take;
        }
        Ok(done)
    }
}

impl<S> std::fmt::Debug for CompressedStorage<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompressedStorage")
            .field("entries", &self.entries.len())
            .field("data_end", &self.data_end)
            .field("len", &self.len)
            .finish()
    }
}

/// Fixture writing the on-disk form, also used by [`crate::nca`]'s tests.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::bucket::Entry as _;

    /// An all-literals LZ4 block, writable without a compressor.
    pub(crate) fn lz4_literals(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        if data.len() < 15 {
            out.push((data.len() as u8) << 4);
        } else {
            out.push(0xF0);
            let mut left = data.len() - 15;
            while left >= 255 {
                out.push(255);
                left -= 255;
            }
            out.push(left as u8);
        }
        out.extend_from_slice(data);
        out
    }

    pub(crate) enum Block {
        Raw(Vec<u8>),
        Zeros(usize),
        Lz4(Vec<u8>),
    }

    /// Returns the stored form, the image it stands for, and its `CompressionInfo`.
    pub(crate) fn build(blocks: &[Block]) -> (Vec<u8>, Vec<u8>, BktrTable) {
        let mut data = Vec::new();
        let mut plain = Vec::new();
        let mut entries: Vec<Vec<u8>> = Vec::new();
        for block in blocks {
            let virt = plain.len() as u64;
            let (kind, phys_size) = match block {
                Block::Raw(bytes) => {
                    plain.extend_from_slice(bytes);
                    data.extend_from_slice(bytes);
                    (KIND_NONE, bytes.len() as u32)
                }
                Block::Zeros(len) => {
                    plain.resize(plain.len() + len, 0);
                    (KIND_ZEROS, 0)
                }
                Block::Lz4(bytes) => {
                    plain.extend_from_slice(bytes);
                    let compressed = lz4_literals(bytes);
                    data.extend_from_slice(&compressed);
                    (KIND_LZ4, compressed.len() as u32)
                }
            };
            let phys = match block {
                Block::Zeros(_) => 0,
                _ => (data.len() - phys_size as usize) as u64,
            };
            let mut entry = vec![0u8; Entry::SIZE as usize];
            entry[..8].copy_from_slice(&virt.to_le_bytes());
            entry[8..16].copy_from_slice(&phys.to_le_bytes());
            entry[0x10] = kind;
            entry[0x14..0x18].copy_from_slice(&phys_size.to_le_bytes());
            entries.push(entry);
            // The physical stream stays 0x10-aligned, as the format requires.
            let aligned = data.len().next_multiple_of(0x10);
            data.resize(aligned, 0);
        }

        let table_offset = data.len() as u64;
        let mut image = data;
        image.extend_from_slice(&crate::bucket::testing::write_table::<Entry>(
            &entries,
            plain.len() as u64,
        ));
        let table = BktrTable {
            offset: table_offset,
            size: image.len() as u64 - table_offset,
            magic: BKTR_MAGIC,
            entries: entries.len() as u32,
        };
        (image, plain, table)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{build, Block};
    use super::*;
    use crate::bucket::Entry as _;
    use crate::source::SliceSource;

    fn sample() -> (Vec<u8>, Vec<u8>, BktrTable) {
        build(&[
            Block::Raw((0..0x40u8).collect()),
            Block::Lz4((0..=0xFFu8).map(|b| b ^ 0x5A).collect()),
            Block::Zeros(0x30),
            Block::Lz4(vec![0xC3; 0x200]),
        ])
    }

    #[test]
    fn serves_every_kind_of_block_and_reads_across_them() {
        let (image, plain, table) = sample();
        let storage = CompressedStorage::new(SliceSource(&image), table).expect("open");
        assert_eq!(storage.len(), plain.len() as u64);

        let mut all = vec![0u8; plain.len()];
        assert_eq!(storage.read_at(0, &mut all).unwrap(), plain.len());
        assert_eq!(all, plain);

        for &(offset, len) in &[
            (0u64, 1usize),
            (0x3f, 2),    // the last raw byte and the first compressed one
            (0x40, 0x10), // inside one LZ4 block
            (0x41, 0x7f), // unaligned inside it
            (0x13f, 3),   // across into the zeroes
            (0x150, 0x30),
            (0x16f, 0x20), // out of the zeroes and into the next block
            (0x100, 0x200),
        ] {
            let mut out = vec![0u8; len];
            assert_eq!(
                storage.read_at(offset, &mut out).unwrap(),
                len,
                "short read at {offset:#x}+{len:#x}"
            );
            assert_eq!(
                out,
                &plain[offset as usize..offset as usize + len],
                "wrong bytes at {offset:#x}+{len:#x}"
            );
        }
    }

    #[test]
    fn a_read_past_the_end_stops_at_it() {
        let (image, plain, table) = sample();
        let storage = CompressedStorage::new(SliceSource(&image), table).expect("open");
        let mut out = vec![0u8; 0x40];
        let at = plain.len() as u64 - 0x10;
        assert_eq!(storage.read_at(at, &mut out).unwrap(), 0x10);
        assert_eq!(storage.read_at(plain.len() as u64, &mut out).unwrap(), 0);
    }

    #[test]
    fn re_reading_a_block_does_not_decompress_it_again() {
        let (image, plain, table) = sample();
        let storage = CompressedStorage::new(SliceSource(&image), table).expect("open");
        let mut out = [0u8; 4];
        for at in 0x40..0x60u64 {
            storage.read_at(at, &mut out).unwrap();
            assert_eq!(out, plain[at as usize..at as usize + 4]);
        }
        // Every read falls in the second entry, so one block is held.
        assert_eq!(storage.cache.borrow().len(), 1);
    }

    #[test]
    fn rejects_a_table_that_is_not_one() {
        let (image, _, table) = sample();
        let mut wrong = table;
        wrong.magic = 0;
        assert!(matches!(
            CompressedStorage::new(SliceSource(&image), wrong),
            Err(Error::Nca(_))
        ));

        let mut short = table;
        short.size = Entry::NODE_SIZE - 1;
        assert!(matches!(
            CompressedStorage::new(SliceSource(&image), short),
            Err(Error::Nca(_))
        ));

        let mut past = table;
        past.offset = image.len() as u64;
        assert!(matches!(
            CompressedStorage::new(SliceSource(&image), past),
            Err(Error::OutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_an_entry_whose_data_is_not_in_the_image() {
        let (mut image, _, table) = sample();
        // The first entry set's second entry, pointed past the data region.
        let set = (table.offset + bucket::node_storage_size::<Entry>(table.entries)) as usize;
        let at = set + (bucket::NODE_HEADER_SIZE + Entry::SIZE) as usize;
        image[at + 8..at + 16].copy_from_slice(&table.offset.to_le_bytes());
        assert!(matches!(
            CompressedStorage::new(SliceSource(&image), table),
            Err(Error::OutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_a_compression_kind_it_cannot_read() {
        let (mut image, _, table) = sample();
        let set = (table.offset + bucket::node_storage_size::<Entry>(table.entries)) as usize;
        image[set + bucket::NODE_HEADER_SIZE as usize + 0x10] = 2;
        assert!(matches!(
            CompressedStorage::new(SliceSource(&image), table),
            Err(Error::Nca(_))
        ));
    }
}
