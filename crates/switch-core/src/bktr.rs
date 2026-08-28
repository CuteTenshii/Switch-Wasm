//! Reading a title's RomFS through the update that replaced parts of it.
//!
//! An update NSP does not contain the game. Its Program NCA carries a full
//! ExeFS — the patched executables, which replace the base title's outright —
//! but its RomFS section holds only the ranges the update changed, together
//! with two tables that say how to put the two halves back together:
//!
//! * the **relocation table**, which maps each range of the patched RomFS to
//!   either this section or the base title's, and
//! * the **subsection table**, which says which AES-CTR counter each range of
//!   this section's own bytes was encrypted with (this is what makes a patch
//!   section `AesCtrEx` rather than plain `AesCtr`: the counter's top word
//!   changes from region to region instead of being the section's throughout).
//!
//! So reading an update's data is a two-container operation, and neither
//! container is ever held in memory: [`patched_romfs_source`] returns a
//! [`ByteSource`] that resolves each read to a range of one file or the other
//! and decrypts only that range. The browser hands it two `File`s it never
//! reads through; the guest asks for a few hundred bytes at a time through
//! `IStorage` either way.
//!
//! Table layout (hactool's `bktr_relocation_block_t`/`bktr_subsection_block_t`,
//! both of them a page of header followed by a page per bucket):
//!
//! ```text
//! 0x0000  u32 _, u32 bucket count, u64 total size, u64 first key per bucket
//! 0x4000  bucket 0: u32 _, u32 entry count, u64 end key, then its entries
//! 0x8000  bucket 1: the same
//! ```
//!
//! A relocation entry is `u64 virtual offset, u64 physical offset, u32 from
//! the patch`; a subsection entry is `u64 offset, u32 _, u32 counter`. Both
//! are sorted, and both tables are flattened into one list here — the bucket
//! split is a paging detail of the on-disk form, not something a lookup needs.

use crate::keys::KeySet;
use crate::nca::{BktrTable, Nca, SectionSource, ENCRYPTION_AES_CTR_EX};
use crate::source::{ByteSource, Window};
use crate::Error;

/// "BKTR", on both of a patch section's table headers.
const BKTR_MAGIC: u32 = 0x5254_4b42;

/// Both tables are paged: one page of header, then one page per bucket.
const BUCKET_SIZE: u64 = 0x4000;

/// Where a bucket's entries start, past its own count and end key.
const BUCKET_HEADER: usize = 0x10;

/// The most table this will read in. Both tables together are a fraction of a
/// per-cent of an update (a few hundred KiB against JD2017's 28 MB), so a
/// figure this far above the real ones only exists to keep a corrupt header
/// from asking for an allocation the browser cannot make.
const MAX_TABLE: u64 = 64 << 20;

/// One range of the patched RomFS, and where its bytes actually are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Relocation {
    /// Where the range starts in the patched (virtual) section.
    virt: u64,
    /// Where it starts in whichever section holds it.
    phys: u64,
    from_patch: bool,
}

/// One range of the patch section's own bytes, and the counter it was
/// encrypted with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Subsection {
    phys: u64,
    ctr_val: u32,
}

/// The base title's RomFS section with an update's patch section over it,
/// addressed as the one section the two describe together.
///
/// This is the whole section — IVFC hash levels and all, since that is what
/// the relocation table's offsets are in terms of. [`patched_romfs_source`]
/// windows it down to the RomFS image the guest actually mounts.
#[derive(Debug)]
pub struct PatchedSection<P: ByteSource, B: ByteSource> {
    patch: SectionSource<P>,
    base: SectionSource<B>,
    relocations: Vec<Relocation>,
    subsections: Vec<Subsection>,
    len: u64,
}

impl<P: ByteSource, B: ByteSource> PatchedSection<P, B> {
    /// Read a range of the patch section's own bytes, splitting it at every
    /// subsection boundary so each piece is decrypted with its own counter.
    fn read_patch(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        let mut done = 0;
        while done < out.len() {
            let at = offset + done as u64;
            let i = index_before(&self.subsections, at, |s| s.phys);
            let end = self
                .subsections
                .get(i + 1)
                .map_or(self.patch.len(), |s| s.phys);
            if end <= at {
                break;
            }
            let take = (out.len() - done).min((end - at) as usize);
            let got = self.patch.read_region(
                at,
                &mut out[done..done + take],
                self.subsections[i].ctr_val,
            )?;
            done += got;
            if got < take {
                break;
            }
        }
        Ok(done)
    }
}

impl<P: ByteSource, B: ByteSource> ByteSource for PatchedSection<P, B> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        if offset >= self.len {
            return Ok(0);
        }
        let want = ((out.len() as u64).min(self.len - offset)) as usize;
        let mut done = 0;
        // One read can span any number of relocation entries — a guest asking
        // for a file that the update rewrote the middle of gets base bytes,
        // patch bytes and base bytes again out of a single call.
        while done < want {
            let at = offset + done as u64;
            let i = index_before(&self.relocations, at, |r| r.virt);
            let entry = self.relocations[i];
            let end = self.relocations.get(i + 1).map_or(self.len, |r| r.virt);
            if end <= at {
                break;
            }
            let take = (want - done).min((end - at) as usize);
            let phys = entry.phys + (at - entry.virt);
            let buf = &mut out[done..done + take];
            let got = if entry.from_patch {
                self.read_patch(phys, buf)?
            } else {
                self.base.read_at(phys, buf)?
            };
            done += got;
            if got < take {
                break;
            }
        }
        Ok(done)
    }
}

/// The index of the last entry starting at or before `at`.
///
/// Both tables start at 0 and cover their whole section, so there is always
/// one; a table that did not is rejected when it is read.
fn index_before<T: Copy>(entries: &[T], at: u64, key: impl Fn(&T) -> u64) -> usize {
    entries.partition_point(|e| key(e) <= at).saturating_sub(1)
}

/// A [`ByteSource`] over the RomFS image an update and its base title
/// describe together: the base's, with everything the update changed in place
/// of the original.
///
/// `patch` is the update container's Program NCA and `base` the base game's.
/// Both sources are the whole (still-encrypted) NCA, since a section's
/// counter is numbered from its position in the file.
///
/// Nothing is decrypted up front beyond the two tables, which are a fraction
/// of a per-cent of the update.
pub fn patched_romfs_source<P: ByteSource, B: ByteSource>(
    patch: &Nca,
    patch_src: P,
    base: &Nca,
    base_src: B,
    keys: &KeySet,
) -> Result<Window<PatchedSection<P, B>>, Error> {
    let patch_index = patch
        .romfs_section_index()
        .ok_or_else(|| Error::Nca("the update's Program NCA has no RomFS section".into()))?;
    let fs = patch.fs_headers[patch_index]
        .ok_or_else(|| Error::Nca("no FS header for the update's RomFS section".into()))?;
    if fs.encryption_type != ENCRYPTION_AES_CTR_EX {
        return Err(Error::Nca(
            "this container's RomFS is not a patch — it is a title in its own right, \
             and boots without a base game"
                .into(),
        ));
    }
    let base_index = base
        .romfs_section_index()
        .ok_or_else(|| Error::Nca("the base title's Program NCA has no RomFS section".into()))?;
    if base.is_update() {
        return Err(Error::Nca(
            "the base container is itself an update — updates do not stack, \
             both halves have to be the same title's base game and one update"
                .into(),
        ));
    }
    if patch.program_id != base.program_id {
        return Err(Error::Nca(format!(
            "this update is for title {:016x}, but the base game is {:016x}",
            patch.program_id, base.program_id
        )));
    }

    let patch_section = patch.section_source(patch_src, keys, patch_index)?;
    let base_section = base.section_source(base_src, keys, base_index)?;

    let (relocations, virtual_size) = read_relocations(&patch_section, fs.relocation)?;
    let (subsections, subsection_total) = read_subsections(&patch_section, fs.subsection)?;
    // hactool's own consistency check, and the one that catches a table read
    // with the wrong key before any of it is believed: the subsection table
    // covers the section's data, and starts where that data ends.
    if subsection_total != fs.subsection.offset {
        return Err(Error::Nca(format!(
            "patch subsection table covers {:#x} bytes but starts at {:#x} — wrong keys or a corrupt update",
            subsection_total, fs.subsection.offset
        )));
    }
    if fs.romfs_data_offset >= virtual_size {
        return Err(Error::Nca(
            "the update's RomFS data offset is past the end of the patched section".into(),
        ));
    }

    let section = PatchedSection {
        patch: patch_section,
        base: base_section,
        relocations,
        subsections,
        len: virtual_size,
    };
    let romfs = Window::new(
        section,
        fs.romfs_data_offset,
        virtual_size - fs.romfs_data_offset,
        "patched RomFS image",
    )?;
    // The same header check [`Nca::romfs_source`] makes, and it means more
    // here: it is the one place where the base game and the update are read
    // through together, so a mismatched pair shows up as a bad header rather
    // than as a title that boots and then cannot find its files.
    let mut header_size = [0u8; 8];
    romfs.read_exact_at(0, &mut header_size)?;
    const ROMFS_HEADER_SIZE: u64 = 0x50;
    if u64::from_le_bytes(header_size) != ROMFS_HEADER_SIZE {
        return Err(Error::Nca(
            "the patched RomFS doesn't start with a valid RomFS header — \
             wrong keys, or this update does not belong to this base game"
                .into(),
        ));
    }
    Ok(romfs)
}

/// Read a table's pages out of the patch section, returning the bucket pages
/// and the total size the header claims.
///
/// The two tables differ only in what their entries hold, so everything up to
/// the entries is read here once.
fn read_table<S: ByteSource>(
    section: &SectionSource<S>,
    table: BktrTable,
    what: &str,
) -> Result<(Vec<u8>, u64), Error> {
    if table.magic != BKTR_MAGIC {
        return Err(Error::Nca(format!(
            "{what} table: magic is {:#010x}, not BKTR — this is not a patch section",
            table.magic
        )));
    }
    if table.size > MAX_TABLE {
        return Err(Error::TooLarge {
            what: format!("{what} table"),
            len: table.size,
            max: MAX_TABLE,
        });
    }
    if table.size < BUCKET_SIZE * 2 || table.size % BUCKET_SIZE != 0 {
        return Err(Error::Nca(format!(
            "{what} table: {:#x} bytes is not a header page plus whole bucket pages",
            table.size
        )));
    }
    let bytes = section.read_vec(table.offset, table.size)?;
    let buckets = u64::from(crate::nsp::read_u32(&bytes, 4));
    let total = crate::nsp::read_u64(&bytes, 8);
    if buckets == 0 || (buckets + 1) * BUCKET_SIZE > table.size {
        return Err(Error::Nca(format!(
            "{what} table: {buckets} buckets do not fit in {:#x} bytes",
            table.size
        )));
    }
    Ok((bytes, total))
}

/// Walk a table's buckets, handing each entry's bytes to `parse`.
fn each_entry(
    bytes: &[u8],
    entry_size: usize,
    what: &str,
    mut parse: impl FnMut(&[u8]),
) -> Result<(), Error> {
    let buckets = crate::nsp::read_u32(bytes, 4) as usize;
    for bucket in 0..buckets {
        let at = (bucket + 1) * BUCKET_SIZE as usize;
        let count = crate::nsp::read_u32(bytes, at + 4) as usize;
        let end = BUCKET_HEADER + count * entry_size;
        if end > BUCKET_SIZE as usize {
            return Err(Error::Nca(format!(
                "{what} table: bucket {bucket} claims {count} entries, more than a page holds"
            )));
        }
        for i in 0..count {
            let e = at + BUCKET_HEADER + i * entry_size;
            parse(&bytes[e..e + entry_size]);
        }
    }
    Ok(())
}

/// The relocation table, flattened and in virtual-offset order, plus the size
/// of the patched section it describes.
fn read_relocations<S: ByteSource>(
    section: &SectionSource<S>,
    table: BktrTable,
) -> Result<(Vec<Relocation>, u64), Error> {
    let (bytes, total) = read_table(section, table, "patch relocation")?;
    let mut out = Vec::new();
    each_entry(&bytes, 0x14, "patch relocation", |e| {
        out.push(Relocation {
            virt: crate::nsp::read_u64(e, 0),
            phys: crate::nsp::read_u64(e, 8),
            from_patch: crate::nsp::read_u32(e, 0x10) != 0,
        })
    })?;
    check_covers(out.first().map(|r| r.virt), out.len(), "patch relocation")?;
    Ok((out, total))
}

/// The subsection table, flattened and in offset order, plus the size of the
/// patch section's data region.
fn read_subsections<S: ByteSource>(
    section: &SectionSource<S>,
    table: BktrTable,
) -> Result<(Vec<Subsection>, u64), Error> {
    let (bytes, total) = read_table(section, table, "patch subsection")?;
    let mut out = Vec::new();
    each_entry(&bytes, 0x10, "patch subsection", |e| {
        out.push(Subsection {
            phys: crate::nsp::read_u64(e, 0),
            ctr_val: crate::nsp::read_u32(e, 0xC),
        })
    })?;
    check_covers(out.first().map(|s| s.phys), out.len(), "patch subsection")?;
    Ok((out, total))
}

/// Both lookups take "the last entry at or before this offset" and index the
/// result, so both tables have to be non-empty and start at 0.
fn check_covers(first: Option<u64>, count: usize, what: &str) -> Result<(), Error> {
    match first {
        Some(0) => Ok(()),
        Some(other) => Err(Error::Nca(format!(
            "{what} table starts at {other:#x}, not 0 — it does not cover the section"
        ))),
        None => Err(Error::Nca(format!("{what} table has no entries ({count})"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nca::FsHeader;

    /// A lookup lands on the entry that covers the offset, not the one after
    /// it — and an offset inside the first entry finds the first entry rather
    /// than underflowing.
    #[test]
    fn a_lookup_finds_the_entry_that_covers_an_offset() {
        let keys = [0u64, 0x100, 0x180, 0x400];
        let at = |o| index_before(&keys, o, |k| *k);
        assert_eq!(at(0), 0);
        assert_eq!(at(0xFF), 0);
        assert_eq!(at(0x100), 1);
        assert_eq!(at(0x17F), 1);
        assert_eq!(at(0x180), 2);
        assert_eq!(at(0x9999), 3);
    }

    /// A patch section's counter differs from its section's in exactly one
    /// word — the generation. The secure value above it identifies the
    /// section and the block index below it is the position, so a region
    /// counter that disturbs either is a different keystream entirely.
    #[test]
    fn a_patch_counter_replaces_only_the_generation() {
        let fs = FsHeader::parse(&{
            let mut raw = [0u8; 0x200];
            raw[0x140..0x144].copy_from_slice(&0x1234_5678u32.to_le_bytes());
            raw[0x144..0x148].copy_from_slice(&0x9ABC_DEF0u32.to_le_bytes());
            raw
        });
        let plain = fs.initial_counter(0x1_0000);
        let patched = fs.patch_counter(0x1_0000, 0x0BAD_F00D);
        assert_eq!(&patched[4..8], &0x0BAD_F00Du32.to_be_bytes());
        assert_eq!(&patched[0..4], &plain[0..4]);
        assert_eq!(&patched[8..16], &plain[8..16]);
        // A region carrying the section's own generation is the section's own
        // counter, which is what makes the tables readable before any of this
        // is known.
        assert_eq!(fs.patch_counter(0x1_0000, fs.generation), plain);
    }

    fn table(entry_size: usize, entries: &[Vec<u8>]) -> Vec<u8> {
        let mut bytes = vec![0u8; (BUCKET_SIZE * 2) as usize];
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        bytes[8..16].copy_from_slice(&0x2000u64.to_le_bytes());
        let bucket = BUCKET_SIZE as usize;
        bytes[bucket + 4..bucket + 8].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        for (i, e) in entries.iter().enumerate() {
            let at = bucket + BUCKET_HEADER + i * entry_size;
            bytes[at..at + e.len()].copy_from_slice(e);
        }
        bytes
    }

    fn relocation(virt: u64, phys: u64, from_patch: bool) -> Vec<u8> {
        let mut e = vec![0u8; 0x14];
        e[0..8].copy_from_slice(&virt.to_le_bytes());
        e[8..16].copy_from_slice(&phys.to_le_bytes());
        e[16..20].copy_from_slice(&u32::from(from_patch).to_le_bytes());
        e
    }

    /// The buckets are a paging detail of the on-disk table: what a lookup
    /// sees is one flat list, in order.
    #[test]
    fn a_tables_buckets_flatten_into_one_list() {
        let bytes = table(
            0x14,
            &[
                relocation(0, 0, false),
                relocation(0x1000, 0x40, true),
                relocation(0x1400, 0x1400, false),
            ],
        );
        let mut out = Vec::new();
        each_entry(&bytes, 0x14, "test", |e| {
            out.push((
                crate::nsp::read_u64(e, 0),
                crate::nsp::read_u64(e, 8),
                crate::nsp::read_u32(e, 0x10) != 0,
            ))
        })
        .unwrap();
        assert_eq!(
            out,
            vec![(0, 0, false), (0x1000, 0x40, true), (0x1400, 0x1400, false)]
        );
    }

    /// The two containers a patched RomFS is made of, built small enough to
    /// check byte by byte: a base section of `(i)` and a patch section of
    /// `(0x80 | i)`, with the tables a real patch section carries.
    ///
    /// Both are left unencrypted, which is a legal `encryption_type` and
    /// keeps the test about the composition rather than about AES.
    struct Pair {
        base_nca: Nca,
        base_bytes: Vec<u8>,
        patch_nca: Nca,
        /// The patch section as it is stored: encrypted.
        patch_bytes: Vec<u8>,
        /// The same bytes in the clear, to assert a composed read against.
        patch_plain: Vec<u8>,
    }

    const BASE_LEN: u64 = 0x200;
    const PATCH_DATA: u64 = 0x100;
    const VIRTUAL_LEN: u64 = 0x200;
    const ROMFS_AT: u64 = 0x40;
    /// The patch section's key, handed to the keyset as a ticket's would be.
    const PATCH_KEY: [u8; 16] = *b"a patch aes key!";
    const RIGHTS_ID: [u8; 16] = [0x11; 16];
    /// Deliberately not the section's own generation (0): a region counter
    /// that happened to match would let the data decrypt with either, and the
    /// point of the test is that each half is read with its own.
    const CTR_VAL: u32 = 7;

    fn fs_header(encryption: u8, tables: Option<(u64, u64)>) -> crate::nca::FsHeader {
        let mut raw = [0u8; 0x200];
        raw[2] = 0; // RomFs partition
        raw[3] = 3; // HierarchicalIntegrity
        raw[4] = encryption;
        // IVFC level 5's logical offset, which is where the RomFS image starts.
        raw[0x90..0x98].copy_from_slice(&ROMFS_AT.to_le_bytes());
        if let Some((relocation, subsection)) = tables {
            for (at, offset) in [(0x100usize, relocation), (0x120, subsection)] {
                raw[at..at + 8].copy_from_slice(&offset.to_le_bytes());
                raw[at + 8..at + 16].copy_from_slice(&(BUCKET_SIZE * 2).to_le_bytes());
                raw[at + 0x10..at + 0x14].copy_from_slice(&BKTR_MAGIC.to_le_bytes());
            }
        }
        crate::nca::FsHeader::parse(&raw)
    }

    fn one_section_nca(len: u64, fs: crate::nca::FsHeader) -> Nca {
        let encrypted = fs.encryption_type != crate::nca::ENCRYPTION_NONE;
        Nca {
            distribution_type: 0,
            content_type: crate::nca::ContentType::Program,
            content_type_raw: 0,
            title_id: 0x0100_0000_0000_1000,
            sdk_version: 0,
            crypto_type: 0,
            sections: vec![crate::nca::SectionHeader {
                media_offset: 0,
                media_size: len,
                partition_index: 0,
            }],
            file_size: len,
            program_id: 0x0100_0000_0000_1000,
            rights_id: if encrypted { RIGHTS_ID } else { [0; 16] },
            key_index: 0,
            key_generation_old: 0,
            key_generation_new: 0,
            encrypted_key_area: [0; 0x40],
            fs_headers: [Some(fs), None, None, None],
        }
    }

    /// A table page pair — header page then one bucket — holding `entries`.
    fn table_pages(total: u64, entry_size: usize, entries: &[Vec<u8>], end_key: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; (BUCKET_SIZE * 2) as usize];
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        bytes[8..16].copy_from_slice(&total.to_le_bytes());
        let bucket = BUCKET_SIZE as usize;
        bytes[bucket + 4..bucket + 8].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        bytes[bucket + 8..bucket + 16].copy_from_slice(&end_key.to_le_bytes());
        for (i, e) in entries.iter().enumerate() {
            let at = bucket + BUCKET_HEADER + i * entry_size;
            bytes[at..at + e.len()].copy_from_slice(e);
        }
        bytes
    }

    fn pair() -> Pair {
        let base_bytes: Vec<u8> = (0..BASE_LEN).map(|i| i as u8).collect();
        let mut patch_bytes: Vec<u8> = (0..PATCH_DATA).map(|i| 0x80 | i as u8).collect();
        // The RomFS header the composed image has to start with, planted where
        // the relocation table sends the image's first bytes.
        patch_bytes[0x40..0x48].copy_from_slice(&0x50u64.to_le_bytes());

        let relocations = [
            relocation(0x000, 0x000, true),
            relocation(ROMFS_AT, ROMFS_AT, true),
            relocation(0x080, 0x080, false),
            relocation(0x100, 0x020, true),
        ];
        patch_bytes.extend(table_pages(VIRTUAL_LEN, 0x14, &relocations, VIRTUAL_LEN));
        let subsection_end = PATCH_DATA + BUCKET_SIZE * 2;
        let mut subsection = vec![0u8; 0x10];
        subsection[0xC..0x10].copy_from_slice(&CTR_VAL.to_le_bytes());
        patch_bytes.extend(table_pages(
            subsection_end,
            0x10,
            &[subsection],
            subsection_end,
        ));

        let patch_len = patch_bytes.len() as u64;
        let patch_nca = one_section_nca(
            patch_len,
            fs_header(ENCRYPTION_AES_CTR_EX, Some((PATCH_DATA, subsection_end))),
        );
        // Encrypt the patch section the way a real one is: its data under the
        // subsection's counter, its tables under the section's own. Reading it
        // back is the assertion that each half is read with the right one.
        let patch_plain = patch_bytes.clone();
        let fs = patch_nca.fs_headers[0].unwrap();
        crate::crypto::aes128_ctr_xor_in_place(
            &PATCH_KEY,
            &fs.patch_counter(0, CTR_VAL),
            &mut patch_bytes[..PATCH_DATA as usize],
        );
        crate::crypto::aes128_ctr_xor_in_place(
            &PATCH_KEY,
            &fs.initial_counter(PATCH_DATA),
            &mut patch_bytes[PATCH_DATA as usize..],
        );
        Pair {
            base_nca: one_section_nca(BASE_LEN, fs_header(crate::nca::ENCRYPTION_NONE, None)),
            base_bytes,
            patch_nca,
            patch_bytes,
            patch_plain,
        }
    }

    /// A keyset holding the patch section's key, the way a container's own
    /// ticket supplies it: wrapped under the `titlekek` for the synthetic
    /// NCA's key generation, which is 0.
    fn patch_keys() -> KeySet {
        const TITLEKEK: [u8; 16] = [0x5a; 16];
        let mut keys = KeySet::default();
        keys.titlekek[0] = Some(TITLEKEK);
        keys.add_title_key(
            RIGHTS_ID,
            crate::crypto::aes128_encrypt_block(&TITLEKEK, &PATCH_KEY),
        );
        keys
    }

    /// Every byte of the composed image comes from the container the
    /// relocation table names, including a read that crosses from one to the
    /// other in the middle.
    #[test]
    fn a_patched_image_reads_from_both_containers() {
        let p = pair();
        let keys = patch_keys();
        let romfs = patched_romfs_source(
            &p.patch_nca,
            crate::source::SliceSource(&p.patch_bytes),
            &p.base_nca,
            crate::source::SliceSource(&p.base_bytes),
            &keys,
        )
        .expect("compose the patched romfs");
        // The window starts at the IVFC data offset, so virtual `ROMFS_AT + w`
        // is image offset `w`.
        assert_eq!(romfs.len(), VIRTUAL_LEN - ROMFS_AT);
        assert_eq!(romfs.read_vec(0, 8).unwrap(), 0x50u64.to_le_bytes());
        // 0x70..0x90 spans the patch's range and the base's: eight bytes of
        // `0x80 | i` and then eight of `i`.
        let across = romfs.read_vec(0x70 - ROMFS_AT, 0x20).unwrap();
        assert_eq!(&across[..0x10], &p.patch_plain[0x70..0x80]);
        assert_eq!(&across[0x10..], &p.base_bytes[0x80..0x90]);
        // The last entry sends the image back into the patch, at a different
        // offset from the one it is mapped to.
        let moved = romfs.read_vec(0x100 - ROMFS_AT, 0x10).unwrap();
        assert_eq!(moved, p.patch_plain[0x20..0x30]);
        // And the image ends where the relocation table says it does.
        assert_eq!(
            romfs
                .read_at(VIRTUAL_LEN - ROMFS_AT, &mut [0u8; 16])
                .unwrap(),
            0
        );
    }

    /// A container that is not an update, and an update that is not this
    /// title's, are both refused before anything is read through them.
    #[test]
    fn only_this_titles_update_composes() {
        let p = pair();
        let keys = patch_keys();
        // The base in the patch's place: its RomFS is its own, not a patch.
        assert!(matches!(
            patched_romfs_source(
                &p.base_nca,
                crate::source::SliceSource(&p.base_bytes),
                &p.base_nca,
                crate::source::SliceSource(&p.base_bytes),
                &keys,
            ),
            Err(Error::Nca(_))
        ));
        let mut other = p.base_nca.clone();
        other.program_id = 0x0100_0000_0000_2000;
        assert!(matches!(
            patched_romfs_source(
                &p.patch_nca,
                crate::source::SliceSource(&p.patch_bytes),
                &other,
                crate::source::SliceSource(&p.base_bytes),
                &keys,
            ),
            Err(Error::Nca(_))
        ));
    }

    /// An entry count that would run off the end of its page is a corrupt
    /// table, not a page-and-a-half of entries.
    #[test]
    fn a_bucket_cannot_claim_more_entries_than_a_page_holds() {
        let mut bytes = table(0x14, &[relocation(0, 0, false)]);
        let bucket = BUCKET_SIZE as usize;
        bytes[bucket + 4..bucket + 8].copy_from_slice(&10_000u32.to_le_bytes());
        assert!(matches!(
            each_entry(&bytes, 0x14, "test", |_| {}),
            Err(Error::Nca(_))
        ));
    }
}
