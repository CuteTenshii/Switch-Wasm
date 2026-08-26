//! NCA container reader and body decryption.
//!
//! An NCA file is Nintendo's encryption wrapper around a game's filesystem
//! images. The base header (0x400 bytes) is stored AES-128-XTS encrypted with
//! the console-family-wide `header_key` and describes the file's metadata,
//! content type and section layout; each of up to 4 sections has its own
//! 0x200-byte FS header (immediately after the base header, same XTS key,
//! continuing sector numbers) describing how that section's body is hashed
//! and encrypted.
//!
//! Section bodies are AES-128-CTR encrypted with a key that lives in the base
//! header's own encrypted key area (unlocked with one of the three
//! `key_area_key_<application|ocean|system>_XX` keys, selected by the header's
//! key index and generation) — or, for titles distributed with a rights id,
//! with the matching entry from `title.keys` directly.
//!
//! Header layout (relative to the NCA start):
//!
//! ```text
//! 0x200  magic "NCA3" (u32)
//! 0x204  distribution type (u8)
//! 0x205  content type (u8)
//! 0x206  key generation, old field (u8)
//! 0x207  key area key index (u8)
//! 0x208  content size (u64)
//! 0x210  program id / title id (u64)
//! 0x218  sdk version (u32)
//! 0x21C  crypto type (u8) — 0 for title-key crypto; check rights id instead
//! 0x220  key generation (u8)
//! 0x230  rights id (16 bytes) — nonzero means title-key crypto
//! 0x240  section table header entry 0 (16 bytes)
//! 0x250  section table header entry 1 (16 bytes)
//! 0x260  section table header entry 2 (16 bytes)
//! 0x270  section table header entry 3 (16 bytes)
//! 0x300  encrypted key area (4 x 16 bytes)
//! 0x400  FS header 0 (0x200 bytes, itself header_key-XTS encrypted)
//! 0x600  FS header 1
//! 0x800  FS header 2
//! 0xA00  FS header 3
//! ```
//!
//! Section table entries describe the backing filesystem image: its offset
//! into the NCA, total size, partition index and type (PFS0/ROMFS/...). The
//! per-section FS header (parsed separately, since it needs the full 0xC00
//! byte header rather than just the base 0x400) carries the hash and
//! encryption metadata needed to actually decrypt and verify the section.

use crate::keys::KeySet;
use crate::nsp::Pfs0File;
use crate::source::{ByteSource, SliceSource, Window};
use crate::Error;

pub const NCA_MAGIC: u32 = 0x3341_434e; // "NCA3"
pub const NCA_HEADER_OFFSET: usize = 0x200;
pub const SECTION_HEADER_COUNT: usize = 4;
/// Size of the base header plus all 4 FS headers. `Nca::parse_with_keys`
/// needs at least this much data to populate `fs_headers` (and therefore to
/// decrypt section bodies); the lightweight "inspect this NCA" path in the
/// frontend only reads [`NCA_HEADER_OFFSET`] + 0x400 bytes and gets metadata
/// only, which is fine for display.
pub const NCA_FULL_HEADER_SIZE: usize = 0xC00;

/// Hash type byte in an FS header: the section is hashed as a two-layer
/// `HierarchicalSha256` (PFS0/ExeFS sections use this).
pub const HASH_TYPE_SHA256: u8 = 2;
/// Hash type byte for `HierarchicalIntegrity` (IVFC/RomFS sections). Not
/// verified here — RomFS mounting is future work.
pub const HASH_TYPE_IVFC: u8 = 3;
/// Encryption type byte in an FS header: no encryption.
pub const ENCRYPTION_NONE: u8 = 1;
/// Encryption type byte in an FS header: AES-128-CTR, the form used by
/// standard-crypto Program NCA sections (ExeFS/RomFS).
pub const ENCRYPTION_AES_CTR: u8 = 3;
/// Encryption type byte in an FS header: AES-128-CTR again, but with the
/// counter's top word chosen per region from the section's own subsection
/// table rather than fixed for the whole section. Only an update's patch
/// RomFS is stored this way, alongside the relocation table that says which
/// of its ranges come from the base title — see [`crate::bktr`].
pub const ENCRYPTION_AES_CTR_EX: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ContentType {
    Program = 0,
    Meta = 1,
    Control = 2,
    Manual = 3,
    Data = 4,
    /// Content shared between titles rather than owned by one — the system's
    /// Mii and amiibo models, the bad-word lists. Mounted by data id through
    /// `OpenDataStorageByDataId` exactly as `Data` is.
    PublicData = 5,
    Unknown(u8),
}

impl ContentType {
    pub fn from_u8(v: u8) -> ContentType {
        match v {
            0 => ContentType::Program,
            1 => ContentType::Meta,
            2 => ContentType::Control,
            3 => ContentType::Manual,
            4 => ContentType::Data,
            5 => ContentType::PublicData,
            other => ContentType::Unknown(other),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ContentType::Program => "Program",
            ContentType::Meta => "Meta",
            ContentType::Control => "Control",
            ContentType::Manual => "Manual",
            ContentType::Data => "Data",
            ContentType::PublicData => "PublicData",
            ContentType::Unknown(_) => "Unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionHeader {
    /// Byte offset of the section's filesystem image within the NCA. The raw
    /// entry stores this as a `u32` count of 0x200-byte media units, not a
    /// byte offset directly.
    pub media_offset: u64,
    /// Total section size, in bytes (derived the same way).
    pub media_size: u64,
    /// Which of the 4 possible partitions the image belongs to (this is just
    /// the entry's index — the entry itself carries no partition id).
    pub partition_index: u8,
}

/// A section's FS header (0x200 bytes, decrypted from immediately after the
/// base header). Field names/offsets below are cross-checked against
/// hactool's `nca_fs_header_t`/`ivfc_hdr_t` (a real reference implementation,
/// not just the public wiki write-up) — `partition_type`/`fs_type` in
/// particular are named the way hactool names them, which turned out to
/// differ from this project's first guess (harmlessly — the byte
/// *positions* were already right, only the semantic labels were swapped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsHeader {
    pub version: u16,
    /// 0 = RomFs, 1 = Pfs0 (byte 2 of the header).
    pub partition_type: u8,
    /// 2 = Pfs0 (`HierarchicalSha256`), 3 = RomFs (`HierarchicalIntegrity`) —
    /// byte 3. This one value doubles as what earlier revisions of this code
    /// called `hash_type`: there's no separate hash-type byte, the two are
    /// the same field.
    pub fs_type: u8,
    pub encryption_type: u8,
    /// `HierarchicalSha256` superblock: SHA-256 of the hash-table region.
    pub master_hash: [u8; 32],
    /// `HierarchicalSha256` superblock: how much data each hash in the table
    /// covers. The table is one SHA-256 per `block_size` bytes of the data
    /// region, the last one over whatever is left.
    pub hash_block_size: u32,
    /// `HierarchicalSha256` superblock: where the per-block hash table lives
    /// within the decrypted section.
    pub hash_table_offset: u64,
    pub hash_table_size: u64,
    /// `HierarchicalSha256` superblock: where the actual PFS0 image starts
    /// within the decrypted section (after the hash table).
    pub data_offset: u64,
    pub data_size: u64,
    /// `HierarchicalIntegrity` (IVFC) superblock: where the actual RomFS
    /// image starts within the decrypted section — the *last* IVFC level's
    /// `logical_offset` (levels 0..N-2 are progressively coarser hash
    /// tables; the last level is the real data). Getting this wrong looks
    /// exactly like a decryption failure: byte 0 of an IVFC section is
    /// Level 0's hash table, not RomFS's own header, so checking for RomFS's
    /// `header_size` magic at section offset 0 fails even with perfectly
    /// correct decryption.
    pub romfs_data_offset: u64,
    /// A patch (`AesCtrEx`) section's relocation table: which ranges of the
    /// patched RomFS come from this section and which from the base title's.
    /// Zeroed on every other kind of section.
    pub relocation: BktrTable,
    /// A patch section's subsection table: which counter each range of this
    /// section's own bytes was encrypted with.
    pub subsection: BktrTable,
    /// AES-CTR IV components (hactool's `section_ctr`): the low 8 bytes of
    /// the counter are the block index and start at 0 for the section start.
    pub generation: u32,
    pub secure_value: u32,
}

/// One of a patch section's two table headers (hactool's `bktr_header_t`),
/// as the FS header carries it: where the table lives inside the section, how
/// long it is, and how many entries it holds in total across its buckets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BktrTable {
    pub offset: u64,
    pub size: u64,
    /// "BKTR" on a real patch section, and nothing at all on any other kind —
    /// which is what [`crate::bktr`] checks before believing the rest.
    pub magic: u32,
    pub entries: u32,
}

impl BktrTable {
    fn parse(fs: &[u8], at: usize) -> BktrTable {
        BktrTable {
            offset: crate::nsp::read_u64(fs, at),
            size: crate::nsp::read_u64(fs, at + 0x08),
            magic: crate::nsp::read_u32(fs, at + 0x10),
            entries: crate::nsp::read_u32(fs, at + 0x18),
        }
    }
}

impl FsHeader {
    /// Parse a decrypted 0x200-byte FS header.
    pub fn parse(fs: &[u8]) -> FsHeader {
        let fs_type = fs[3];
        let mut romfs_data_offset = 0u64;
        if fs_type == HASH_TYPE_IVFC {
            // ivfc_hdr_t (at fs_header+0x08): magic(4) id(4) master_hash_size(4)
            // num_levels(4) then level_headers[6] (24 bytes each: u64
            // logical_offset, u64 hash_data_size, u32 block_size, u32
            // reserved). hactool's own `nca_save_section` always reads
            // `level_headers[IVFC_MAX_LEVEL - 1]` (fixed index 5) as the real
            // RomFS data level and ignores `num_levels` for this — on a real
            // file `num_levels` reads as 7 while the array only holds 6
            // entries, so deriving the index from it (as this code did at
            // first) reads out of bounds into the trailing padding.
            const IVFC_MAX_LEVEL: usize = 6;
            let entry_off = 0x18 + (IVFC_MAX_LEVEL - 1) * 24;
            romfs_data_offset = crate::nsp::read_u64(fs, entry_off);
        }
        FsHeader {
            version: u16::from_le_bytes([fs[0], fs[1]]),
            partition_type: fs[2],
            fs_type,
            encryption_type: fs[4],
            master_hash: fs[0x08..0x28].try_into().unwrap(),
            hash_block_size: crate::nsp::read_u32(fs, 0x28),
            hash_table_offset: crate::nsp::read_u64(fs, 0x30),
            hash_table_size: crate::nsp::read_u64(fs, 0x38),
            data_offset: crate::nsp::read_u64(fs, 0x40),
            data_size: crate::nsp::read_u64(fs, 0x48),
            romfs_data_offset,
            // The BKTR superblock overlays the IVFC one from 0x8, putting its
            // two table headers at 0x100 and 0x120.
            relocation: BktrTable::parse(fs, 0x100),
            subsection: BktrTable::parse(fs, 0x120),
            generation: crate::nsp::read_u32(fs, 0x140),
            secure_value: crate::nsp::read_u32(fs, 0x144),
        }
    }

    /// The AES-CTR counter block for the very start of the section.
    /// `aes128_ctr_xor` increments it correctly from there for every
    /// subsequent 16-byte block. The low 8 bytes are the block index — which
    /// is the section's *absolute* position in the NCA file divided by 16,
    /// not 0: confirmed empirically against a real title (Nintendo's own
    /// `nca_calculate_section_ctr` runs the same counter across the whole
    /// file rather than resetting it per section, and using 0 here decrypted
    /// to garbage that failed the master-hash check on real content, even
    /// with the correct key).
    pub fn initial_counter(&self, media_offset: u64) -> [u8; 16] {
        let mut ctr = [0u8; 16];
        ctr[0..4].copy_from_slice(&self.secure_value.to_be_bytes());
        ctr[4..8].copy_from_slice(&self.generation.to_be_bytes());
        ctr[8..16].copy_from_slice(&(media_offset >> 4).to_be_bytes());
        ctr
    }

    /// The counter block for one region of a *patch* (`AesCtrEx`) section:
    /// the same counter, with the *generation* word replaced by the `ctr_val`
    /// the section's subsection table gives that region. The secure value
    /// identifies the section and stays put; the generation is what the
    /// regions vary.
    ///
    /// The tables themselves are written before any of this applies and
    /// decrypt with [`FsHeader::initial_counter`] — which is also why a
    /// region whose `ctr_val` is the section's own generation reads
    /// identically either way.
    pub fn patch_counter(&self, media_offset: u64, ctr_val: u32) -> [u8; 16] {
        let mut ctr = self.initial_counter(media_offset);
        ctr[4..8].copy_from_slice(&ctr_val.to_be_bytes());
        ctr
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nca {
    pub distribution_type: u8,
    pub content_type: ContentType,
    pub content_type_raw: u8,
    pub title_id: u64,
    pub sdk_version: u32,
    pub crypto_type: u8,
    pub sections: Vec<SectionHeader>,
    /// The NCA's total content size, in bytes.
    pub file_size: u64,
    /// Same field as `title_id` (the header stores "program ID" once, at
    /// 0x210) — kept as a separate field for API clarity even though the
    /// values are always identical.
    pub program_id: u64,
    /// Nonzero when the title uses title-key crypto: the section key comes
    /// from `title.keys` (looked up by this id) instead of the header's own
    /// encrypted key area.
    pub rights_id: [u8; 16],
    /// Selects which `key_area_key_<kind>` family unlocks the encrypted key
    /// area (0 = Application, 1 = Ocean, 2 = System).
    pub key_index: u8,
    /// The two key-generation fields (old and current), combined into the
    /// master-key revision used to pick a `key_area_key_*_XX` generation.
    pub key_generation_old: u8,
    pub key_generation_new: u8,
    /// The header's own encrypted key area: 4 x 16-byte AES keys. Slot 2 is
    /// the one used as the AES-CTR section key for standard-crypto Program
    /// NCAs; the others are unused by anything this emulator does.
    pub encrypted_key_area: [u8; 0x40],
    /// Per-section FS headers, populated only when `parse_with_keys` was
    /// given the full [`NCA_FULL_HEADER_SIZE`]-byte header (or more) and a
    /// usable header key.
    pub fs_headers: [Option<FsHeader>; SECTION_HEADER_COUNT],
}

impl Nca {
    /// Parse an NCA header from a full NCA buffer (header must be cleartext).
    pub fn parse(data: &[u8]) -> Result<Nca, Error> {
        Self::parse_with_keys(data, None)
    }

    /// Parse an NCA header, transparently decrypting a CDN-encrypted header
    /// with the supplied keyset when the magic doesn't match. The header is
    /// AES-128-XTS over two 0x200-byte sectors with the global `header_key`
    /// (hactool `nca_decrypt_header`); the per-title key isn't needed for the
    /// header itself.
    pub fn parse_with_keys(raw: &[u8], keys: Option<&crate::keys::KeySet>) -> Result<Nca, Error> {
        const HEADER_SIZE: usize = 0x400;
        if raw.len() < HEADER_SIZE {
            return Err(Error::Truncated {
                what: "NCA header".into(),
                expected: HEADER_SIZE,
                got: raw.len(),
            });
        }
        let h = NCA_HEADER_OFFSET;
        let magic = crate::nsp::read_u32(raw, h);
        let header_key = keys.and_then(|k| k.effective_header_key());
        let buf: Vec<u8>;
        let data = if magic == NCA_MAGIC {
            raw
        } else if let Some(key) = header_key {
            buf = crate::crypto::aes128_xts_decrypt(&key, &raw[..HEADER_SIZE], 0, 0x200);
            if crate::nsp::read_u32(&buf, h) != NCA_MAGIC {
                return Err(Error::BadMagic {
                    what: "NCA".into(),
                    found: magic,
                });
            }
            buf.as_slice()
        } else {
            return Err(Error::BadMagic {
                what: "NCA".into(),
                found: magic,
            });
        };

        let content_type_raw = data[h + 0x05];
        let mut sections = Vec::with_capacity(SECTION_HEADER_COUNT);
        // Each entry is `u32 start_offset; u32 end_offset; u8 reserved[8]`,
        // both offsets counted in 0x200-byte media units — NOT a byte
        // offset/size pair. (A real Program NCA's section 0 decoded as a
        // multi-terabyte offset with a 1-byte size before this was fixed.)
        const MEDIA_UNIT: u64 = 0x200;
        for i in 0..SECTION_HEADER_COUNT {
            let at = h + 0x40 + i * 0x10;
            let start = crate::nsp::read_u32(data, at) as u64;
            let end = crate::nsp::read_u32(data, at + 4) as u64;
            sections.push(SectionHeader {
                media_offset: start * MEDIA_UNIT,
                media_size: end.saturating_sub(start) * MEDIA_UNIT,
                partition_index: i as u8,
            });
        }

        let mut encrypted_key_area = [0u8; 0x40];
        encrypted_key_area.copy_from_slice(&data[h + 0x100..h + 0x140]);

        // FS headers live right after the base header, still XTS-encrypted
        // with the same header_key, continuing the sector count (sectors 0-1
        // are the base header, so FS header `i` is sector 2+i). They need the
        // *raw* file bytes regardless of whether the base header itself
        // needed decrypting, and enough of the file to reach them — the
        // lightweight header-only inspection path doesn't provide that, so
        // this is skipped (all `None`) rather than erroring.
        let mut fs_headers: [Option<FsHeader>; SECTION_HEADER_COUNT] = Default::default();
        if raw.len() >= NCA_FULL_HEADER_SIZE {
            if let Some(key) = header_key {
                for (i, slot) in fs_headers.iter_mut().enumerate() {
                    let start = 0x400 + i * 0x200;
                    let sector = 2 + i as u64;
                    let plain =
                        crate::crypto::aes128_xts_decrypt(&key, &raw[start..start + 0x200], sector, 0x200);
                    *slot = Some(FsHeader::parse(&plain));
                }
            }
        }

        Ok(Nca {
            distribution_type: data[h + 0x04],
            content_type: ContentType::from_u8(content_type_raw),
            content_type_raw,
            title_id: crate::nsp::read_u64(data, h + 0x10),
            sdk_version: crate::nsp::read_u32(data, h + 0x18),
            crypto_type: data[h + 0x1C],
            sections,
            file_size: crate::nsp::read_u64(data, h + 0x08),
            program_id: crate::nsp::read_u64(data, h + 0x10),
            rights_id: data[h + 0x30..h + 0x40].try_into().unwrap(),
            key_index: data[h + 0x07],
            key_generation_old: data[h + 0x06],
            key_generation_new: data[h + 0x20],
            encrypted_key_area,
            fs_headers,
        })
    }

    /// Parse an NCA header out of a [`ByteSource`], reading only the
    /// [`NCA_FULL_HEADER_SIZE`] bytes the header occupies.
    ///
    /// This is how an NCA inside a multi-gigabyte container is opened: the
    /// header is 3 KiB at the front of it, and the sections it describes are
    /// then read through [`Nca::section_source`] rather than extracted.
    pub fn parse_source<S: ByteSource>(
        src: &S,
        keys: Option<&crate::keys::KeySet>,
    ) -> Result<Nca, Error> {
        let want = src.len().min(NCA_FULL_HEADER_SIZE as u64);
        let header = src.read_vec(0, want)?;
        Nca::parse_with_keys(&header, keys)
    }

    /// Whether the file body is encrypted. In practice every retail NCA is,
    /// and the key data required to decrypt lives in the header's key area.
    pub fn is_encrypted(&self) -> bool {
        self.crypto_type != 0 || self.has_rights_id()
    }

    /// Whether this title uses title-key crypto (a nonzero rights id).
    pub fn has_rights_id(&self) -> bool {
        self.rights_id != [0u8; 16]
    }

    /// The master-key revision selecting a `key_area_key_*_XX` generation:
    /// the higher of the two key-generation fields, then shifted down by one
    /// the way hactool's `crypto_type == 0 ? 0 : crypto_type - 1` does.
    fn master_key_revision(&self) -> u8 {
        let crypto_type = self.key_generation_old.max(self.key_generation_new);
        crypto_type.saturating_sub(1)
    }

    /// The AES-128 key that decrypts this NCA's sections: either the title
    /// key (rights-id crypto) or key-area slot 2, unlocked with the matching
    /// `key_area_key_<kind>_<generation>`.
    ///
    /// A `title.keys` entry is not itself that key: the file stores a
    /// ticket's key block verbatim, still wrapped under `titlekek_XX`, so it
    /// has to be unwrapped with this NCA's key generation first. Used raw it
    /// decrypts to noise that only surfaces later as a section hash
    /// mismatch, which reads as "wrong keys" when the keys were fine.
    pub fn section_key(&self, keys: &crate::keys::KeySet) -> Result<[u8; 16], Error> {
        if self.has_rights_id() {
            let generation = self.master_key_revision();
            if let Some(key) = keys.title_key(&self.rights_id, generation) {
                return Ok(key);
            }
            // Which half is missing decides what the user has to go fetch.
            return Err(if keys.wrapped_title_key(&self.rights_id).is_none() {
                Error::Nca("no title key loaded for this title's rights id".into())
            } else {
                Error::Nca(format!(
                    "missing titlekek_{:02x} in prod.keys — needed to unwrap this title's key",
                    generation
                ))
            });
        }
        let kind = crate::keys::KeyAreaKind::from_index(self.key_index)
            .ok_or_else(|| Error::Nca(format!("unknown key area index {}", self.key_index)))?;
        let generation = self.master_key_revision();
        let kek = keys.key_area_key(kind, generation).ok_or_else(|| {
            Error::Nca(format!(
                "missing key_area_key_{:?}_{:02x} in prod.keys",
                kind, generation
            ))
        })?;
        let mut block = [0u8; 16];
        block.copy_from_slice(&self.encrypted_key_area[0x20..0x30]);
        Ok(crate::crypto::aes128_decrypt_block(&kek, &block))
    }

    /// A decrypting [`ByteSource`] over section `index`'s body.
    ///
    /// Nothing is read here: the returned source decrypts on demand, so a
    /// section far larger than memory (a retail RomFS is the whole game) can
    /// be served range by range as the guest asks for it. `nca` is a source
    /// over the *whole* NCA, since a section's AES-CTR counter is derived
    /// from its absolute position in the file.
    pub fn section_source<S: ByteSource>(
        &self,
        nca: S,
        keys: &crate::keys::KeySet,
        index: usize,
    ) -> Result<SectionSource<S>, Error> {
        let sec = self
            .sections
            .get(index)
            .ok_or_else(|| Error::Nca(format!("no section {}", index)))?;
        let fs = self
            .fs_headers
            .get(index)
            .and_then(|o| o.as_ref())
            .copied()
            .ok_or_else(|| {
                Error::Nca(
                    "missing FS header — pass the full NCA (>= 0xC00 bytes) with a loaded header_key"
                        .into(),
                )
            })?;
        let key = match fs.encryption_type {
            // A patch section shares the key and takes the same counter; only
            // its top word varies by region, and the relocation and
            // subsection tables that say how are themselves written under the
            // section's own counter, so reading them needs exactly this.
            ENCRYPTION_AES_CTR | ENCRYPTION_AES_CTR_EX => Some(self.section_key(keys)?),
            ENCRYPTION_NONE => None,
            other => {
                return Err(Error::Nca(format!(
                    "unsupported section encryption type {}",
                    other
                )))
            }
        };
        let body = Window::new(
            nca,
            sec.media_offset,
            sec.media_size,
            &format!("NCA section {}", index),
        )?;
        Ok(SectionSource {
            body,
            key,
            fs,
            nca_offset: sec.media_offset,
        })
    }

    /// Check a decrypted section against the FS header's master hash, for the
    /// `HierarchicalSha256` sections that have one (PFS0/ExeFS).
    ///
    /// A wrong key is otherwise undetectable: AES-CTR with the wrong key
    /// still "succeeds", it just XORs a different keystream, so this is the
    /// only signal that what came out is the real thing.
    fn verify_section_hash(&self, plain: &[u8], index: usize) -> Result<(), Error> {
        let fs = match self.fs_headers.get(index).and_then(|o| o.as_ref()) {
            Some(fs) if fs.fs_type == HASH_TYPE_SHA256 => fs,
            _ => return Ok(()),
        };
        let ht_start = fs.hash_table_offset as usize;
        let ht_end = ht_start
            .checked_add(fs.hash_table_size as usize)
            .ok_or(Error::Overflow)?;
        if ht_end > plain.len() {
            return Err(Error::Nca("hash table region exceeds decrypted section".into()));
        }
        if crate::crypto::sha256(&plain[ht_start..ht_end]) != fs.master_hash {
            return Err(Error::Nca(
                "decrypted section hash mismatch — wrong keys or a corrupt file".into(),
            ));
        }
        self.verify_data_blocks(plain, fs)
    }

    /// Check the data region against the per-block hashes in the table the
    /// master hash just vouched for.
    ///
    /// The master hash only says the *table* is intact — every byte the
    /// emulator goes on to execute is covered by the table, not by it. Left
    /// unchecked, a single wrong byte anywhere in an ExeFS boots: what
    /// follows is a fault somewhere inside the title's own crt0, reported
    /// against whatever garbage the relocations produced, with nothing to
    /// connect it back to a bad read. This is the check that says so
    /// directly, and it is also what proves a streamed read was byte-exact.
    ///
    /// Skipped unless the geometry is exactly what a two-layer
    /// `HierarchicalSha256` implies, since an unrecognized layout must not
    /// turn into a spurious failure.
    fn verify_data_blocks(&self, plain: &[u8], fs: &FsHeader) -> Result<(), Error> {
        let Some((block, blocks)) = hash_coverage(fs) else {
            return Ok(());
        };
        let block = block as u64;
        let data_start = fs.data_offset as usize;
        let data_end = data_start
            .checked_add(fs.data_size as usize)
            .ok_or(Error::Overflow)?;
        let table_start = fs.hash_table_offset as usize;
        if data_end > plain.len() {
            return Err(Error::Nca("data region exceeds decrypted section".into()));
        }
        let data = &plain[data_start..data_end];
        for (i, chunk) in data.chunks(block as usize).enumerate() {
            let at = table_start + i * 32;
            if crate::crypto::sha256(chunk) != plain[at..at + 32] {
                return Err(Error::Nca(format!(
                    "block {} of {} does not match its hash — the {:#x} bytes at section offset {:#x} are not what this NCA says they are",
                    i,
                    blocks,
                    chunk.len(),
                    data_start + i * block as usize
                )));
            }
        }
        Ok(())
    }

    /// Decrypt section `index`'s body from the *raw* (still-encrypted) NCA
    /// bytes, verifying it against the FS header's master hash when the
    /// section is `HierarchicalSha256`-hashed (PFS0/ExeFS).
    ///
    /// Holds the whole section in memory, so it is only for sections known to
    /// be small (an ExeFS) or for a host that has the file mapped anyway; the
    /// browser reads RomFS through [`Nca::romfs_source`] instead.
    pub fn decrypt_section(&self, raw: &[u8], keys: &crate::keys::KeySet, index: usize) -> Result<Vec<u8>, Error> {
        let section = self.section_source(SliceSource(raw), keys, index)?;
        let plain = section.read_vec(0, section.len())?;
        self.verify_section_hash(&plain, index)?;
        Ok(plain)
    }

    /// Read section `index` as an ExeFS: decrypt it, verify it, and return
    /// just the PFS0 payload (after the hash table), ready for `Pfs0::parse`.
    /// Only valid for `HierarchicalSha256`-hashed sections.
    ///
    /// An ExeFS is a title's executables — tens of megabytes at the outside —
    /// so unlike its RomFS this is read in full.
    pub fn read_pfs0_section<S: ByteSource>(
        &self,
        nca: S,
        keys: &crate::keys::KeySet,
        index: usize,
    ) -> Result<Vec<u8>, Error> {
        let fs = self
            .fs_headers
            .get(index)
            .and_then(|o| o.as_ref())
            .copied()
            .ok_or_else(|| Error::Nca(format!("no FS header for section {}", index)))?;
        let section = self.section_source(nca, keys, index)?;
        let mut plain = section.read_vec(0, section.len())?;
        self.verify_section_hash(&plain, index)?;
        let start = crate::source::alloc_len(fs.data_offset, "PFS0 region offset")?;
        let end = start
            .checked_add(crate::source::alloc_len(fs.data_size, "PFS0 region")?)
            .ok_or(Error::Overflow)?;
        if end > plain.len() {
            return Err(Error::Nca("PFS0 region exceeds decrypted section".into()));
        }
        // Trim in place rather than copying the payload out: the section is
        // the largest thing this path holds, and a second copy of it is the
        // difference between loading a title and running out of memory.
        plain.truncate(end);
        plain.drain(..start);
        Ok(plain)
    }

    /// Decrypt section `index` and slice out just the PFS0/ExeFS payload
    /// (after the hash table), ready for `Pfs0::parse`. Only valid for
    /// `HierarchicalSha256`-hashed sections.
    pub fn decrypt_pfs0_section(
        &self,
        raw: &[u8],
        keys: &crate::keys::KeySet,
        index: usize,
    ) -> Result<Vec<u8>, Error> {
        self.read_pfs0_section(SliceSource(raw), keys, index)
    }

    /// How much of section `index`'s data the hash table actually covers:
    /// `(block size, block count)`, or `None` when the geometry is not the
    /// two-layer `HierarchicalSha256` [`Nca::read_pfs0_section`] can verify.
    ///
    /// A loader reports this rather than assuming it: "the ExeFS decrypted"
    /// and "every byte of the ExeFS is what this NCA says it is" are
    /// different claims, and only the second one makes a fault further in
    /// worth chasing anywhere but the reader.
    pub fn pfs0_hash_coverage(&self, index: usize) -> Option<(u32, u64)> {
        hash_coverage(self.fs_headers.get(index).and_then(|o| o.as_ref())?)
    }

    /// The index of this NCA's PFS0 (ExeFS) section, if any — `partition_type`
    /// is 1 for PartitionFS, 0 for RomFS.
    pub fn exefs_section_index(&self) -> Option<usize> {
        self.fs_headers.iter().position(|fs| matches!(fs, Some(h) if h.partition_type == 1))
    }

    /// The index of this NCA's RomFS section, if any.
    pub fn romfs_section_index(&self) -> Option<usize> {
        self.fs_headers
            .iter()
            .position(|fs| matches!(fs, Some(h) if h.partition_type == 0 && h.fs_type == HASH_TYPE_IVFC))
    }

    /// Whether this is an update's Program NCA: one whose RomFS section holds
    /// a patch over some base title's, rather than a RomFS of its own.
    ///
    /// Nothing else distinguishes it. An update's Program NCA carries the
    /// *base* title id (the `...800` update id appears only on the container's
    /// Meta NCA), and its ExeFS is a complete replacement set of modules — so
    /// the patch RomFS is what says the container cannot be booted on its own.
    pub fn is_update(&self) -> bool {
        self.romfs_section_index()
            .and_then(|i| self.fs_headers[i])
            .is_some_and(|fs| fs.encryption_type == ENCRYPTION_AES_CTR_EX)
    }

    /// A [`ByteSource`] over section `index`'s RomFS image: the decrypting
    /// section view, windowed past the IVFC hash-tree levels via
    /// [`FsHeader::romfs_data_offset`].
    ///
    /// Nothing is decrypted up front. This is the only way a modern title's
    /// RomFS can be served at all — it is the bulk of a container that
    /// already does not fit in memory — and the guest reads it a range at a
    /// time through `IStorage` anyway.
    ///
    /// Sanity-checks the image against RomFS's own `header_size` field
    /// (always 0x50), which is what catches a wrong key: full multi-level
    /// IVFC hash verification (the way [`Nca::read_pfs0_section`] verifies
    /// `HierarchicalSha256`) isn't implemented, so this catches "wrong key"
    /// but not a subtler corruption deep in the hash tree.
    pub fn romfs_source<S: ByteSource>(
        &self,
        nca: S,
        keys: &crate::keys::KeySet,
        index: usize,
    ) -> Result<Window<SectionSource<S>>, Error> {
        let fs = self
            .fs_headers
            .get(index)
            .and_then(|o| o.as_ref())
            .copied()
            .ok_or_else(|| Error::Nca(format!("no FS header for section {}", index)))?;
        if fs.encryption_type == ENCRYPTION_AES_CTR_EX {
            return Err(Error::Nca(
                "this section is an update's patch RomFS — it holds only what the update changed, \
                 and has to be read over the base title's RomFS (see `bktr::patched_romfs_source`)"
                    .into(),
            ));
        }
        let section = self.section_source(nca, keys, index)?;
        if fs.romfs_data_offset >= section.len() {
            return Err(Error::Nca("RomFS data offset exceeds the decrypted section".into()));
        }
        let len = section.len() - fs.romfs_data_offset;
        let romfs = Window::new(section, fs.romfs_data_offset, len, "RomFS image")?;
        // RomFS's own header starts with its size, always 0x50. With the
        // wrong key the section decrypts to noise, and this is what says so —
        // there is no per-block hash to check an IVFC section against.
        let mut header_size = [0u8; 8];
        romfs.read_exact_at(0, &mut header_size)?;
        const ROMFS_HEADER_SIZE: u64 = 0x50;
        if u64::from_le_bytes(header_size) != ROMFS_HEADER_SIZE {
            return Err(Error::Nca(
                "decrypted RomFS section doesn't start with a valid RomFS header — wrong keys or a corrupt file".into(),
            ));
        }
        Ok(romfs)
    }

    /// Decrypt section `index` as a RomFS image and return the whole thing.
    ///
    /// Only for a host with memory to spare (the native examples): the
    /// browser composes [`Nca::romfs_source`] into the CPU instead, since a
    /// retail RomFS is gigabytes.
    pub fn decrypt_romfs_section(
        &self,
        raw: &[u8],
        keys: &crate::keys::KeySet,
        index: usize,
    ) -> Result<Vec<u8>, Error> {
        let romfs = self.romfs_source(SliceSource(raw), keys, index)?;
        romfs.read_vec(0, romfs.len())
    }
}

/// The block size and block count a `HierarchicalSha256` section's hash table
/// covers, when the table is exactly one SHA-256 per block of the data region
/// and nothing else. Anything that doesn't add up is a layout this code does
/// not know, and must not be turned into a verification failure.
fn hash_coverage(fs: &FsHeader) -> Option<(u32, u64)> {
    if fs.fs_type != HASH_TYPE_SHA256 || fs.hash_block_size == 0 {
        return None;
    }
    let blocks = fs.data_size.div_ceil(fs.hash_block_size as u64);
    (blocks.checked_mul(32) == Some(fs.hash_table_size)).then_some((fs.hash_block_size, blocks))
}

/// A decrypting view of one NCA section: [`Nca::section_source`] builds it,
/// and reads through it come back in the clear.
///
/// AES-CTR is seekable — the keystream block a byte gets is decided by its
/// own position — so a range out of the middle of a section costs exactly
/// that range, which is what lets a RomFS larger than memory be read at all.
/// Reads are aligned down to the 16-byte cipher block internally; callers see
/// plain byte addressing.
#[derive(Debug, Clone)]
pub struct SectionSource<S> {
    /// The section body, still encrypted, addressed from the section start.
    body: Window<S>,
    /// The AES-128-CTR section key, or `None` for an unencrypted section.
    key: Option<[u8; 16]>,
    fs: FsHeader,
    /// The section's absolute offset within the NCA, which is what its
    /// counter blocks are numbered from.
    nca_offset: u64,
}

impl<S: ByteSource> SectionSource<S> {
    /// The section's FS header (hash type, encryption, IVFC layout).
    pub fn fs_header(&self) -> &FsHeader {
        &self.fs
    }

    /// Decrypt `buf`, which must hold the section's bytes starting at the
    /// 16-byte-aligned section offset `at`. `ctr_val` is the counter's top
    /// word for a patch section's region, or `None` for the section's own.
    fn decrypt_at(&self, at: u64, buf: &mut [u8], ctr_val: Option<u32>) {
        if let Some(key) = self.key {
            let media = self.nca_offset + at;
            let ctr = match ctr_val {
                Some(v) => self.fs.patch_counter(media, v),
                None => self.fs.initial_counter(media),
            };
            crate::crypto::aes128_ctr_xor_in_place(&key, &ctr, buf);
        }
    }

    /// Read a range of a patch (`AesCtrEx`) section that lies within one
    /// subsection, whose counter top word is `ctr_val`.
    ///
    /// The caller splits at subsection boundaries; this is the piece between
    /// two of them. See [`crate::bktr`].
    pub(crate) fn read_region(
        &self,
        offset: u64,
        out: &mut [u8],
        ctr_val: u32,
    ) -> Result<usize, Error> {
        self.read_decrypting(offset, out, Some(ctr_val))
    }

    /// The body of [`ByteSource::read_at`], with the counter left open so a
    /// patch section can supply its region's own.
    fn read_decrypting(
        &self,
        offset: u64,
        out: &mut [u8],
        ctr_val: Option<u32>,
    ) -> Result<usize, Error> {
        if offset >= self.len() {
            return Ok(0);
        }
        let want = ((out.len() as u64).min(self.len() - offset)) as usize;
        let aligned = offset & !0xF;
        let head = (offset - aligned) as usize;
        let mut done = 0;
        // A read that starts mid-cipher-block needs that whole block to
        // decrypt it, so the first one goes through a scratch block and the
        // rest is decrypted straight into the caller's buffer.
        if head != 0 {
            let mut block = [0u8; 16];
            let got = self.body.read_at(aligned, &mut block)?;
            self.decrypt_at(aligned, &mut block[..got], ctr_val);
            let take = (16 - head).min(want).min(got.saturating_sub(head));
            out[..take].copy_from_slice(&block[head..head + take]);
            done = take;
            // Short here means the section ended inside this very block, so
            // there is nothing past it to go on with.
            if take < (16 - head).min(want) {
                return Ok(done);
            }
        }
        if done < want {
            let at = aligned + if head != 0 { 16 } else { 0 };
            let rest = &mut out[done..want];
            let got = self.body.read_at(at, rest)?;
            self.decrypt_at(at, &mut rest[..got], ctr_val);
            done += got;
        }
        Ok(done)
    }
}

impl<S: ByteSource> ByteSource for SectionSource<S> {
    fn len(&self) -> u64 {
        self.body.len()
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        self.read_decrypting(offset, out, None)
    }
}

/// Find the first NCA of content type `want` in a PFS0 container's file
/// table, returning its index and parsed header.
///
/// Every `.nca` in the container has to be opened to answer: the file names
/// are content hashes, so what a file *is* shows only in its (possibly
/// encrypted) header. `keys` therefore needs the header key, or nothing here
/// is readable at all and this finds nothing.
///
/// This is how a title's own executable is picked out of a container. A
/// retail NSP holds the Program NCA next to a `cnmt` metadata record, the
/// Control NCA that carries the icon, and often a manual and a legal-info
/// NCA - all with hash names, so nothing but the content type distinguishes
/// the one worth booting.
pub fn find_nca_by_type<S: ByteSource>(
    files: &[Pfs0File],
    src: &S,
    keys: &KeySet,
    want: ContentType,
) -> Option<(usize, Nca)> {
    files.iter().enumerate().find_map(|(index, f)| {
        if !f.name.to_ascii_lowercase().ends_with(".nca") {
            return None;
        }
        let window = Window::new(src, f.offset, f.size, &f.name).ok()?;
        let nca = Nca::parse_source(&window, Some(keys)).ok()?;
        (nca.content_type == want).then_some((index, nca))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nca() -> Vec<u8> {
        let mut data = vec![0u8; 0x400];
        let h = NCA_HEADER_OFFSET;
        data[h..h + 4].copy_from_slice(&NCA_MAGIC.to_le_bytes());
        data[h + 0x04] = 0; // distribution type: downloadable
        data[h + 0x05] = 0; // content type: program
        data[h + 0x08..h + 0x10].copy_from_slice(&0x12345u64.to_le_bytes()); // content size
        data[h + 0x10..h + 0x18].copy_from_slice(&0x0100_0000_0010_5A00u64.to_le_bytes()); // program/title id
        data[h + 0x18..h + 0x1C].copy_from_slice(&0x0001_000Au32.to_le_bytes()); // sdk version
        data[h + 0x1C] = 0x01; // crypto type
        // section 0: a PFS0 image starting at media unit 0, 0x10 units
        // (0x2000 bytes) long — the entry is `u32 start; u32 end`, both in
        // 0x200-byte media units, not a byte offset/size pair.
        data[h + 0x40..h + 0x44].copy_from_slice(&0u32.to_le_bytes());
        data[h + 0x44..h + 0x48].copy_from_slice(&0x10u32.to_le_bytes());
        data
    }

    #[test]
    fn parses_nca_header() {
        let nca = Nca::parse(&make_nca()).unwrap();
        assert_eq!(nca.content_type, ContentType::Program);
        assert_eq!(nca.title_id, 0x0100_0000_0010_5A00);
        assert_eq!(nca.sdk_version, 0x0001_000A);
        assert_eq!(nca.crypto_type, 0x01);
        assert!(nca.is_encrypted());
        assert_eq!(nca.file_size, 0x12345);
        assert_eq!(nca.sections.len(), 4);
        assert_eq!(nca.sections[0].media_offset, 0);
        assert_eq!(nca.sections[0].media_size, 0x2000);
        assert_eq!(nca.sections[0].partition_index, 0);
    }

    /// A rights-id title's key comes out of `title.keys` still wrapped under
    /// `titlekek_<key generation - 1>`; `section_key` is where that gets
    /// undone. Real-world symptom of skipping it: every section decrypts to
    /// noise and fails its hash, with the keys reported as wrong.
    #[test]
    fn section_key_unwraps_a_title_keys_entry() {
        let mut data = make_nca();
        let h = NCA_HEADER_OFFSET;
        data[h + 0x06] = 2; // key generation (old), pinned at 2 once >= 3
        data[h + 0x20] = 0x0e; // key generation (new)
        let rights_id = [0x77u8; 16];
        data[h + 0x30..h + 0x40].copy_from_slice(&rights_id);
        let nca = Nca::parse(&data).unwrap();
        assert!(nca.has_rights_id());

        let plain = [0x11u8; 16];
        let kek = [0x22u8; 16];
        let mut keys = crate::keys::KeySet::default();
        keys.titlekek[0x0d] = Some(kek);
        keys.title_keys = vec![(rights_id, crate::crypto::aes128_encrypt_block(&kek, &plain))];
        assert_eq!(nca.section_key(&keys).unwrap(), plain);

        // Without the titlekek there is no usable key, and saying so beats
        // handing back the wrapped bytes for the hash check to reject.
        keys.titlekek[0x0d] = None;
        assert!(nca.section_key(&keys).is_err());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = make_nca();
        data[NCA_HEADER_OFFSET] = b'X';
        assert!(matches!(Nca::parse(&data), Err(Error::BadMagic { .. })));
    }

    #[test]
    fn rejects_short_buffer() {
        assert!(matches!(
            Nca::parse(&[0u8; 0x300]),
            Err(Error::Truncated { .. })
        ));
    }

    /// A cleartext NCA header, long enough for the base header to parse.
    fn nca_header(content_type: u8) -> Vec<u8> {
        let mut hdr = vec![0u8; NCA_HEADER_OFFSET + 0x200];
        hdr[0x200..0x204].copy_from_slice(&NCA_MAGIC.to_le_bytes());
        hdr[0x204] = 2; // distribution type
        hdr[0x205] = content_type;
        hdr[0x210..0x218].copy_from_slice(&0x0100_0000_0000_1000u64.to_le_bytes());
        hdr
    }

    #[test]
    fn finds_an_nca_in_a_container_by_content_type() {
        // What a container actually looks like: hash-named NCAs whose names
        // say nothing about what they hold, next to metadata that is not an
        // NCA at all. Only the content type in each header distinguishes them.
        let parts: [(&str, Vec<u8>); 4] = [
            ("0100000000001000.cnmt.xml", vec![0u8; 0x40]),
            ("aaaa.nca", nca_header(1)), // Meta
            ("bbbb.nca", nca_header(2)), // Control
            ("cccc.nca", nca_header(0)), // Program
        ];
        let mut image: Vec<u8> = Vec::new();
        let mut files = Vec::new();
        for (name, bytes) in &parts {
            files.push(crate::nsp::Pfs0File {
                offset: image.len() as u64,
                size: bytes.len() as u64,
                name: (*name).to_string(),
            });
            image.extend_from_slice(bytes);
        }
        let src = SliceSource(&image);
        let keys = KeySet::default();

        let (index, nca) =
            find_nca_by_type(&files, &src, &keys, ContentType::Program).expect("program nca");
        assert_eq!(index, 3);
        assert_eq!(nca.content_type, ContentType::Program);
        assert_eq!(
            find_nca_by_type(&files, &src, &keys, ContentType::Control).map(|(i, _)| i),
            Some(2)
        );
        // A type the container does not hold finds nothing, rather than
        // falling back to the first header that happened to parse.
        assert!(find_nca_by_type(&files, &src, &keys, ContentType::Manual).is_none());
    }
}

#[cfg(test)]
mod decrypt_tests {
    use super::*;
    use crate::crypto::{aes128_encrypt_block, aes128_xts_decrypt};
    use crate::keys::KeySet;

    fn encrypt_xts(key: &[u8; 32], data: &[u8], sector: u64, sector_size: usize) -> Vec<u8> {
        // XTS encrypt is decrypt of the "ciphertext" — not needed; instead we
        // encrypt manually: standard XTS encrypt (E(K1, P^T) ^ T).
        let mut key1 = [0u8; 16];
        let mut key2 = [0u8; 16];
        key1.copy_from_slice(&key[..16]);
        key2.copy_from_slice(&key[16..]);
        let mut out = Vec::with_capacity(data.len());
        let mut s = sector;
        for chunk in data.chunks(sector_size) {
            let mut tweak = [0u8; 16];
            let mut sv = s;
            for i in (0..16).rev() {
                tweak[i] = (sv & 0xff) as u8;
                sv >>= 8;
            }
            s += 1;
            let mut tweak = aes128_encrypt_block(&key2, &tweak);
            for blk in chunk.chunks(16) {
                let mut p = [0u8; 16];
                p[..blk.len()].copy_from_slice(blk);
                let mut x = [0u8; 16];
                for i in 0..16 {
                    x[i] = p[i] ^ tweak[i];
                }
                let c = aes128_encrypt_block(&key1, &x);
                for i in 0..16 {
                    out.push(c[i] ^ tweak[i]);
                }
                // multiply tweak by x (little-endian left shift)
                let carry = tweak[15] & 0x80;
                for i in (0..15).rev() {
                    tweak[i + 1] = (tweak[i + 1] << 1) | (tweak[i] >> 7);
                }
                tweak[0] <<= 1;
                if carry != 0 {
                    tweak[0] ^= 0x87;
                }
            }
        }
        out
    }

    #[test]
    fn decrypts_encrypted_header_with_header_key() {
        // Build a cleartext NCA header (NCA3 magic), encrypt the first 0x400
        // bytes with a known header key, then parse_with_keys must decrypt and
        // succeed.
        let mut hdr = [0u8; 0x400];
        hdr[0x200..0x204].copy_from_slice(&NCA_MAGIC.to_le_bytes());
        hdr[0x204] = 2; // distribution type
        hdr[0x205] = 0; // content type: Program
        hdr[0x210..0x218].copy_from_slice(&0x010075600ae96800u64.to_le_bytes()); // title id
        hdr[0x218..0x21C].copy_from_slice(&0x00090007u32.to_le_bytes()); // sdk version
        hdr[0x21C] = 0; // crypto type

        let mut key = [0u8; 32];
        for i in 0..32 {
            key[i] = i as u8;
        }
        let encrypted = encrypt_xts(&key, &hdr, 0, 0x200);

        let mut keys = KeySet::default();
        keys.header_key = Some(key);
        let nca = Nca::parse_with_keys(&encrypted, Some(&keys)).expect("decrypt+parse");
        assert_eq!(nca.title_id, 0x010075600ae96800);
        assert_eq!(nca.content_type, ContentType::Program);

        // Without keys it must fail with bad magic.
        assert!(matches!(
            Nca::parse_with_keys(&encrypted, None),
            Err(Error::BadMagic { .. })
        ));
    }

    #[test]
    fn xts_roundtrip_consistency() {
        let mut key = [0u8; 32];
        for i in 0..32 {
            key[i] = i as u8;
        }
        let mut data = [0u8; 0x400];
        for (i, b) in data.iter_mut().enumerate() {
            *b = i as u8;
        }
        let encrypted = encrypt_xts(&key, &data, 0, 0x200);
        let decrypted = aes128_xts_decrypt(&key, &encrypted, 0, 0x200);
        assert_eq!(decrypted.as_slice(), &data[..]);
    }

    /// Build a minimal PFS0 image with one file ("main", the given bytes).
    fn build_pfs0(name: &str, payload: &[u8]) -> Vec<u8> {
        let strings = format!("{}\0", name);
        let strings_padded_len = strings.len();
        let header_len = 0x10 + 1 * crate::nsp::FILE_ENTRY_SIZE;
        let payload_off = header_len + strings_padded_len;
        let mut out = vec![0u8; payload_off + payload.len()];
        out[0..4].copy_from_slice(&crate::nsp::PFS0_MAGIC.to_le_bytes());
        out[4..8].copy_from_slice(&1u32.to_le_bytes());
        out[8..12].copy_from_slice(&(strings_padded_len as u32).to_le_bytes());
        let entry = 0x10;
        // File offsets are relative to the end of the header+string table.
        out[entry..entry + 8].copy_from_slice(&0u64.to_le_bytes());
        out[entry + 8..entry + 16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        out[entry + 16..entry + 20].copy_from_slice(&0u32.to_le_bytes());
        out[header_len..header_len + strings.len()].copy_from_slice(strings.as_bytes());
        out[payload_off..].copy_from_slice(payload);
        out
    }

    /// The block size a real ExeFS's `HierarchicalSha256` table hashes over.
    const HASH_BLOCK_SIZE: u32 = 0x1_0000;

    /// Build a synthetic *encrypted* Program NCA with an AES-CTR ExeFS
    /// section, its `HierarchicalSha256` hash table, and a master hash over
    /// that table — returns the raw NCA bytes, the keyset that unlocks them,
    /// and the PFS0 payload that should come back out.
    fn build_exefs_nca() -> (Vec<u8>, KeySet, Vec<u8>) {
        use crate::crypto::{aes128_ctr_xor, sha256};

        let header_key = {
            let mut k = [0u8; 32];
            for i in 0..32 {
                k[i] = i as u8;
            }
            k
        };
        let kek = {
            let mut k = [0u8; 16];
            for i in 0..16 {
                k[i] = 0xA0 + i as u8;
            }
            k
        };
        let section_key = {
            let mut k = [0u8; 16];
            for i in 0..16 {
                k[i] = 0xB0 + i as u8;
            }
            k
        };

        // Base header + 4 FS headers, cleartext for now.
        let mut header = vec![0u8; NCA_FULL_HEADER_SIZE];
        let h = NCA_HEADER_OFFSET;
        header[h..h + 4].copy_from_slice(&NCA_MAGIC.to_le_bytes());
        header[h + 0x05] = 0; // content type: Program
        header[h + 0x06] = 0; // key generation (old)
        header[h + 0x07] = 2; // key index: System
        header[h + 0x10..h + 0x18].copy_from_slice(&0x0100_dead_beef_0000u64.to_le_bytes());
        header[h + 0x1C] = 1; // crypto type: encrypted
        header[h + 0x20] = 0; // key generation (new)

        // Section 0 entry: `u32 start; u32 end`, both in 0x200-byte media
        // units (not a byte offset/size pair — that was the real-world bug
        // this test caught).
        const SECTION_OFFSET: usize = 0x1000;
        let pfs0 = build_pfs0("main", b"fake NSO bytes for the test");
        // The real thing: one SHA-256 per HASH_BLOCK_SIZE bytes of the data
        // region, which for a payload this small is a single hash.
        let hash_table = sha256(&pfs0).to_vec();
        let plain_section = [hash_table.clone(), pfs0.clone()].concat();

        let at = h + 0x40;
        let start_units = (SECTION_OFFSET / 0x200) as u32;
        let size_units = ((plain_section.len() + 0x1ff) / 0x200) as u32;
        let end_units = start_units + size_units;
        header[at..at + 4].copy_from_slice(&start_units.to_le_bytes());
        header[at + 4..at + 8].copy_from_slice(&end_units.to_le_bytes());

        // Encrypted key area: slot 2 (System) holds `section_key`, ECB
        // "encrypted" with `kek` — `section_key()` decrypts it back.
        let encrypted_slot2 = crate::crypto::aes128_encrypt_block(&kek, &section_key);
        header[h + 0x120..h + 0x130].copy_from_slice(&encrypted_slot2);

        // FS header 0: PartitionFS, HierarchicalSha256, AES-CTR.
        let fs0 = h + 0x400 - h; // == 0x400, offset within `header`
        let generation: u32 = 0x01;
        let secure_value: u32 = 0x1122_3344;
        header[fs0 + 0x02] = 1; // partition_type: Pfs0
        header[fs0 + 0x03] = HASH_TYPE_SHA256;
        header[fs0 + 0x04] = ENCRYPTION_AES_CTR;
        header[fs0 + 0x28..fs0 + 0x2C].copy_from_slice(&HASH_BLOCK_SIZE.to_le_bytes());
        header[fs0 + 0x2C..fs0 + 0x30].copy_from_slice(&2u32.to_le_bytes()); // layer count
        header[fs0 + 0x30..fs0 + 0x38].copy_from_slice(&0u64.to_le_bytes()); // hash_table_offset
        header[fs0 + 0x38..fs0 + 0x40].copy_from_slice(&(hash_table.len() as u64).to_le_bytes());
        header[fs0 + 0x40..fs0 + 0x48].copy_from_slice(&(hash_table.len() as u64).to_le_bytes()); // data_offset
        header[fs0 + 0x48..fs0 + 0x50].copy_from_slice(&(pfs0.len() as u64).to_le_bytes());
        header[fs0 + 0x140..fs0 + 0x144].copy_from_slice(&generation.to_le_bytes());
        header[fs0 + 0x144..fs0 + 0x148].copy_from_slice(&secure_value.to_le_bytes());
        let master_hash = sha256(&hash_table);
        header[fs0 + 0x08..fs0 + 0x28].copy_from_slice(&master_hash);

        let mut ctr = [0u8; 16];
        ctr[0..4].copy_from_slice(&secure_value.to_be_bytes());
        ctr[4..8].copy_from_slice(&generation.to_be_bytes());
        ctr[8..16].copy_from_slice(&((SECTION_OFFSET as u64) >> 4).to_be_bytes());
        let encrypted_section = aes128_ctr_xor(&section_key, &ctr, &plain_section);

        let encrypted_header = encrypt_xts(&header_key, &header, 0, 0x200);
        let media_size_bytes = size_units as usize * 0x200;
        let mut raw = vec![0u8; SECTION_OFFSET + media_size_bytes];
        raw[..encrypted_header.len()].copy_from_slice(&encrypted_header);
        raw[SECTION_OFFSET..SECTION_OFFSET + encrypted_section.len()].copy_from_slice(&encrypted_section);

        let mut keys = KeySet::default();
        keys.header_key = Some(header_key);
        keys.key_area_key_system[0] = Some(kek);
        (raw, keys, pfs0)
    }

    /// End-to-end: decrypt and extract a synthetic ExeFS the way a real
    /// loader would.
    ///
    /// This proves the plumbing (XTS header decrypt → key-area unlock →
    /// AES-CTR section decrypt → hash verification → PFS0 extraction) is
    /// internally consistent. It cannot prove the exact FS-header field
    /// offsets or CTR IV layout match a real retail NCA — there is no
    /// legally includable fixture for that — so treat a real title's
    /// decryption as unverified until tried against real keys.
    #[test]
    fn decrypts_and_extracts_a_synthetic_exefs_section() {
        let (raw, keys, pfs0) = build_exefs_nca();

        let nca = Nca::parse_with_keys(&raw, Some(&keys)).expect("parse");
        assert!(nca.fs_headers[0].is_some());
        assert_eq!(nca.exefs_section_index(), Some(0));

        let extracted = nca
            .decrypt_pfs0_section(&raw, &keys, 0)
            .expect("decrypt + hash-verify");
        assert_eq!(extracted, pfs0);

        let inner = crate::nsp::Pfs0::parse(&extracted).expect("valid PFS0");
        let main = inner.find("main").expect("main entry");
        assert_eq!(
            &extracted[main.offset as usize..][..main.size as usize],
            b"fake NSO bytes for the test"
        );

        // A wrong key-area key decrypts to garbage and must be caught by the
        // master-hash check rather than silently "succeeding".
        let mut wrong_keys = keys.clone();
        wrong_keys.key_area_key_system[0] = Some([0u8; 16]);
        assert!(matches!(
            nca.decrypt_pfs0_section(&raw, &wrong_keys, 0),
            Err(Error::Nca(_))
        ));    }

    /// One wrong byte in the data region has to be caught. The master hash
    /// covers the hash *table* only, so before the per-block check this was
    /// invisible: the ExeFS extracted "successfully" and the title booted on
    /// top of a corrupted executable, faulting later somewhere inside its own
    /// crt0 with nothing pointing back at the read that caused it.
    #[test]
    fn a_single_wrong_byte_in_the_data_region_is_caught() {
        let (mut raw, keys, _) = build_exefs_nca();
        let nca = Nca::parse_with_keys(&raw, Some(&keys)).expect("parse");
        let data_start = 0x1000 + nca.fs_headers[0].unwrap().data_offset as usize;
        raw[data_start + 9] ^= 0x01;

        let err = nca.decrypt_pfs0_section(&raw, &keys, 0).unwrap_err();
        let Error::Nca(msg) = &err else {
            panic!("wrong error: {err}");
        };
        assert!(msg.contains("block 0"), "{msg}");
    }

    /// Same shape as the ExeFS test above, but for a RomFS (`HierarchicalIntegrity`/IVFC)
    /// section: no `data_offset` sub-slice, no master-hash check — just the
    /// section decrypting to something starting with a valid RomFS header.
    /// Build a synthetic *encrypted* Data NCA whose single section is a
    /// RomFS (`HierarchicalIntegrity`/IVFC): returns the raw NCA bytes, the
    /// keyset that unlocks them, the section in the clear, and the offset the
    /// real RomFS image starts at within it.
    fn build_romfs_nca() -> (Vec<u8>, KeySet, Vec<u8>, u64) {
        use crate::crypto::aes128_ctr_xor;

        let header_key = {
            let mut k = [0u8; 32];
            for i in 0..32 {
                k[i] = (0x40 + i) as u8;
            }
            k
        };
        let kek = {
            let mut k = [0u8; 16];
            for i in 0..16 {
                k[i] = 0xC0 + i as u8;
            }
            k
        };
        let section_key = {
            let mut k = [0u8; 16];
            for i in 0..16 {
                k[i] = 0xD0 + i as u8;
            }
            k
        };

        let mut header = vec![0u8; NCA_FULL_HEADER_SIZE];
        let h = NCA_HEADER_OFFSET;
        header[h..h + 4].copy_from_slice(&NCA_MAGIC.to_le_bytes());
        header[h + 0x05] = 4; // content type: Data
        header[h + 0x07] = 2; // key index: System
        header[h + 0x1C] = 1; // crypto type: encrypted

        const SECTION_OFFSET: usize = 0x1000;
        // The real RomFS data always lives at IVFC level index 5 (hactool
        // reads `level_headers[IVFC_MAX_LEVEL - 1]` unconditionally) — bytes
        // before that are the (unverified here) hash-tree levels. This
        // exercises the actual bug this fixture caught against real content:
        // byte 0 of the section is NOT the RomFS header for a real,
        // multi-level IVFC section.
        const LEVEL5_OFFSET: u64 = 0x40;
        let mut plain_section = vec![0xAAu8; LEVEL5_OFFSET as usize]; // levels 0..4 "hash tables"
        plain_section.extend_from_slice(&0x50u64.to_le_bytes()); // RomFS header_size
        // Every byte past the header is a function of its own offset, so a
        // partial read can be checked for having landed where it claims.
        while plain_section.len() < 0x200 {
            plain_section.push((plain_section.len() as u8) ^ 0x5A);
        }

        let at = h + 0x40;
        let start_units = (SECTION_OFFSET / 0x200) as u32;
        let size_units = ((plain_section.len() + 0x1ff) / 0x200) as u32;
        header[at..at + 4].copy_from_slice(&start_units.to_le_bytes());
        header[at + 4..at + 8].copy_from_slice(&(start_units + size_units).to_le_bytes());

        let encrypted_slot2 = crate::crypto::aes128_encrypt_block(&kek, &section_key);
        header[h + 0x120..h + 0x130].copy_from_slice(&encrypted_slot2);

        let fs0 = 0x400;
        let generation: u32 = 0x03;
        let secure_value: u32 = 0x5566_7788;
        header[fs0 + 0x02] = 0; // partition_type: RomFs
        header[fs0 + 0x03] = HASH_TYPE_IVFC;
        header[fs0 + 0x04] = ENCRYPTION_AES_CTR;
        header[fs0 + 0x18 + 5 * 24..fs0 + 0x18 + 5 * 24 + 8].copy_from_slice(&LEVEL5_OFFSET.to_le_bytes()); // level[5].logical_offset
        header[fs0 + 0x140..fs0 + 0x144].copy_from_slice(&generation.to_le_bytes());
        header[fs0 + 0x144..fs0 + 0x148].copy_from_slice(&secure_value.to_le_bytes());

        let mut ctr = [0u8; 16];
        ctr[0..4].copy_from_slice(&secure_value.to_be_bytes());
        ctr[4..8].copy_from_slice(&generation.to_be_bytes());
        ctr[8..16].copy_from_slice(&((SECTION_OFFSET as u64) >> 4).to_be_bytes());
        let encrypted_section = aes128_ctr_xor(&section_key, &ctr, &plain_section);

        let encrypted_header = encrypt_xts(&header_key, &header, 0, 0x200);
        let media_size_bytes = size_units as usize * 0x200;
        let mut raw = vec![0u8; SECTION_OFFSET + media_size_bytes];
        raw[..encrypted_header.len()].copy_from_slice(&encrypted_header);
        raw[SECTION_OFFSET..SECTION_OFFSET + encrypted_section.len()].copy_from_slice(&encrypted_section);

        let mut keys = KeySet::default();
        keys.header_key = Some(header_key);
        keys.key_area_key_system[0] = Some(kek);
        (raw, keys, plain_section, LEVEL5_OFFSET)
    }

    /// Same shape as the ExeFS test above, but for a RomFS
    /// (`HierarchicalIntegrity`/IVFC) section: no `data_offset` sub-slice, no
    /// master-hash check — just the section decrypting to something starting
    /// with a valid RomFS header.
    #[test]
    fn decrypts_a_synthetic_romfs_section() {
        let (raw, keys, plain_section, level5) = build_romfs_nca();

        let nca = Nca::parse_with_keys(&raw, Some(&keys)).expect("parse");
        assert_eq!(nca.romfs_section_index(), Some(0));
        assert_eq!(nca.exefs_section_index(), None);
        assert_eq!(nca.fs_headers[0].unwrap().romfs_data_offset, level5);

        let extracted = nca.decrypt_romfs_section(&raw, &keys, 0).expect("decrypt romfs");
        assert_eq!(extracted, &plain_section[level5 as usize..]);

        // A wrong key decrypts to garbage, caught by the header_size check
        // (there's no per-block hash to verify against, unlike PFS0).
        let mut wrong_keys = keys.clone();
        wrong_keys.key_area_key_system[0] = Some([0u8; 16]);
        assert!(matches!(
            nca.decrypt_romfs_section(&raw, &wrong_keys, 0),
            Err(Error::Nca(_))
        ));
    }

    /// The path a real title's RomFS is actually served through: no full
    /// decryption anywhere, just the ranges the guest asked for.
    ///
    /// The ranges deliberately do not line up with anything. AES-CTR numbers
    /// its keystream blocks by position, so a read starting mid-block has to
    /// be aligned down to the cipher block, decrypted, and then trimmed —
    /// get that wrong and only reads that happen to start on a multiple of 16
    /// come back correct, which most of a RomFS mount's do.
    #[test]
    fn a_romfs_source_serves_unaligned_ranges_without_decrypting_the_section() {
        let (raw, keys, plain_section, level5) = build_romfs_nca();
        let nca = Nca::parse_with_keys(&raw, Some(&keys)).expect("parse");
        let romfs = nca
            .romfs_source(SliceSource(&raw), &keys, 0)
            .expect("romfs source");

        let want = &plain_section[level5 as usize..];
        assert_eq!(romfs.len(), want.len() as u64);

        for &(offset, len) in &[
            (0u64, 8usize),   // the RomFS header itself
            (1, 1),           // one byte, mid-block
            (3, 13),          // up to the first block boundary
            (3, 14),          // and one byte past it
            (15, 2),          // straddling a block boundary
            (16, 32),         // exactly aligned, whole blocks
            (17, 100),        // unaligned start, unaligned end
            (0x3f, 0x81),     // spanning many blocks
        ] {
            let mut out = vec![0u8; len];
            let got = romfs.read_at(offset, &mut out).unwrap();
            assert_eq!(got, len, "short read at {:#x}+{:#x}", offset, len);
            assert_eq!(
                out,
                &want[offset as usize..offset as usize + len],
                "wrong bytes at {:#x}+{:#x}",
                offset,
                len
            );
        }

        // Reads that run off the end report what they filled rather than
        // failing, and past it, nothing.
        let mut out = vec![0u8; 64];
        let last = romfs.len() - 10;
        assert_eq!(romfs.read_at(last, &mut out).unwrap(), 10);
        assert_eq!(&out[..10], &want[last as usize..]);
        assert_eq!(romfs.read_at(romfs.len(), &mut out).unwrap(), 0);
    }
}
