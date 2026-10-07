//! NCA container reader and body decryption: AES-128-XTS headers and
//! AES-128-CTR section bodies.

use crate::keys::KeySet;
use crate::nsp::Pfs0File;
use crate::source::{ByteSource, SliceSource, Window};
use crate::Error;

pub const NCA_MAGIC: u32 = 0x3341_434e; // "NCA3"
pub const NCA_HEADER_OFFSET: usize = 0x200;
pub const SECTION_HEADER_COUNT: usize = 4;
/// Size of the base header plus all 4 FS headers.
pub const NCA_FULL_HEADER_SIZE: usize = 0xC00;

/// FS header hash type: `HierarchicalSha256` (PFS0/ExeFS).
pub const HASH_TYPE_SHA256: u8 = 2;
/// FS header hash type: `HierarchicalIntegrity` (IVFC/RomFS).
pub const HASH_TYPE_IVFC: u8 = 3;
/// FS header encryption type: none.
pub const ENCRYPTION_NONE: u8 = 1;
/// FS header encryption type: AES-128-CTR.
pub const ENCRYPTION_AES_CTR: u8 = 3;
/// FS header encryption type: AES-128-CTR with per-region counters (patch RomFS).
pub const ENCRYPTION_AES_CTR_EX: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ContentType {
    Program = 0,
    Meta = 1,
    Control = 2,
    Manual = 3,
    Data = 4,
    /// Content shared between titles (Mii models, bad-word lists), mounted by data id.
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
    /// Byte offset of the section image within the NCA (stored in 0x200-byte media units).
    pub media_offset: u64,
    /// Total section size, in bytes.
    pub media_size: u64,
    /// The entry's index; the entry carries no partition id.
    pub partition_index: u8,
}

/// A section's FS header (0x200 bytes after the base header).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsHeader {
    pub version: u16,
    /// 0 = RomFs, 1 = Pfs0 (byte 2 of the header).
    pub partition_type: u8,
    /// 2 = Pfs0 (`HierarchicalSha256`), 3 = RomFs (`HierarchicalIntegrity`), byte 3.
    pub fs_type: u8,
    pub encryption_type: u8,
    /// `HierarchicalSha256` superblock: SHA-256 of the hash-table region.
    pub master_hash: [u8; 32],
    /// `HierarchicalSha256` superblock: bytes covered by each hash in the table.
    pub hash_block_size: u32,
    /// `HierarchicalSha256` superblock: hash table location in the decrypted section.
    pub hash_table_offset: u64,
    pub hash_table_size: u64,
    /// `HierarchicalSha256` superblock: PFS0 image location in the decrypted section.
    pub data_offset: u64,
    pub data_size: u64,
    /// IVFC superblock: RomFS image offset (the last IVFC level's `logical_offset`).
    pub romfs_data_offset: u64,
    /// Exact RomFS image size from the last IVFC level; 0 means unstated.
    pub romfs_data_size: u64,
    /// Patch (`AesCtrEx`) relocation table; zeroed on other sections.
    pub relocation: BktrTable,
    /// Patch section subsection table: the counter for each range.
    pub subsection: BktrTable,
    /// Sparse layer table (`SparseInfo`, 0x148). See [`crate::sparse`].
    pub sparse: BktrTable,
    /// Physical offset of a sparse section's stored body (`SparseInfo` + 0x20).
    pub sparse_physical_offset: u64,
    /// Generation word the sparse table is encrypted under (`SparseInfo` + 0x28).
    pub sparse_generation: u32,
    /// Compression layer table (`CompressionInfo`, 0x178). See [`crate::compressed`].
    pub compression: BktrTable,
    /// AES-CTR IV components (hactool's `section_ctr`).
    pub generation: u32,
    pub secure_value: u32,
}

/// Magic on every bucket-tree header in an FS header.
pub const BKTR_MAGIC: u32 = 0x5254_4b42;

/// A bucket-tree table header (hactool's `bktr_header_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BktrTable {
    pub offset: u64,
    pub size: u64,
    /// [`BKTR_MAGIC`] when the table is present.
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
        let mut romfs_data_size = 0u64;
        if fs_type == HASH_TYPE_IVFC {
            // ivfc_hdr_t at +0x08. The RomFS data level is always level_headers[5],
            // whatever num_levels says.
            const IVFC_MAX_LEVEL: usize = 6;
            let entry_off = 0x18 + (IVFC_MAX_LEVEL - 1) * 24;
            romfs_data_offset = crate::nsp::read_u64(fs, entry_off);
            romfs_data_size = crate::nsp::read_u64(fs, entry_off + 8);
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
            romfs_data_size,
            // The BKTR superblock overlays IVFC from 0x8; sparse and compression tables follow it.
            relocation: BktrTable::parse(fs, 0x100),
            subsection: BktrTable::parse(fs, 0x120),
            sparse: BktrTable::parse(fs, 0x148),
            sparse_physical_offset: crate::nsp::read_u64(fs, 0x148 + 0x20),
            sparse_generation: u32::from(u16::from_le_bytes([fs[0x148 + 0x28], fs[0x148 + 0x29]])),
            compression: BktrTable::parse(fs, 0x178),
            generation: crate::nsp::read_u32(fs, 0x140),
            secure_value: crate::nsp::read_u32(fs, 0x144),
        }
    }

    /// AES-CTR counter for the section start; the block index is the absolute NCA offset / 16.
    pub fn initial_counter(&self, media_offset: u64) -> [u8; 16] {
        let mut ctr = [0u8; 16];
        ctr[0..4].copy_from_slice(&self.secure_value.to_be_bytes());
        ctr[4..8].copy_from_slice(&self.generation.to_be_bytes());
        ctr[8..16].copy_from_slice(&(media_offset >> 4).to_be_bytes());
        ctr
    }

    /// Counter for a patch section region: the generation word replaced by `ctr_val`.
    pub fn patch_counter(&self, media_offset: u64, ctr_val: u32) -> [u8; 16] {
        let mut ctr = self.initial_counter(media_offset);
        ctr[4..8].copy_from_slice(&ctr_val.to_be_bytes());
        ctr
    }

    /// Counter for a sparse section's table, using `SparseInfo`'s own generation.
    pub fn sparse_counter(&self, media_offset: u64) -> [u8; 16] {
        let mut ctr = self.initial_counter(media_offset);
        ctr[4..8].copy_from_slice(&(self.sparse_generation << 16).to_be_bytes());
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
    /// Same value as `title_id`.
    pub program_id: u64,
    /// Nonzero for title-key crypto.
    pub rights_id: [u8; 16],
    /// Selects the `key_area_key_<kind>` family (0 = Application, 1 = Ocean, 2 = System).
    pub key_index: u8,
    /// Old and current key-generation fields.
    pub key_generation_old: u8,
    pub key_generation_new: u8,
    /// Encrypted key area: 4 x 16-byte keys; slot 2 is the AES-CTR section key.
    pub encrypted_key_area: [u8; 0x40],
    /// Populated only when `parse_with_keys` got the full header and a header key.
    pub fs_headers: [Option<FsHeader>; SECTION_HEADER_COUNT],
}

impl Nca {
    /// Parse an NCA header from a full NCA buffer (header must be cleartext).
    pub fn parse(data: &[u8]) -> Result<Nca, Error> {
        Self::parse_with_keys(data, None)
    }

    /// Parse an NCA header, XTS-decrypting it with `header_key` when the magic doesn't match.
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
        // Entries are `u32 start; u32 end` in 0x200-byte media units.
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

        // FS header `i` is XTS sector 2+i; skipped when the buffer is too short.
        let mut fs_headers: [Option<FsHeader>; SECTION_HEADER_COUNT] = Default::default();
        if raw.len() >= NCA_FULL_HEADER_SIZE {
            if let Some(key) = header_key {
                for (i, slot) in fs_headers.iter_mut().enumerate() {
                    let start = 0x400 + i * 0x200;
                    let sector = 2 + i as u64;
                    let plain = crate::crypto::aes128_xts_decrypt(
                        &key,
                        &raw[start..start + 0x200],
                        sector,
                        0x200,
                    );
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

    /// Parse an NCA header from a [`ByteSource`], reading only the header bytes.
    pub fn parse_source<S: ByteSource>(
        src: &S,
        keys: Option<&crate::keys::KeySet>,
    ) -> Result<Nca, Error> {
        let want = src.len().min(NCA_FULL_HEADER_SIZE as u64);
        let header = src.read_vec(0, want)?;
        Nca::parse_with_keys(&header, keys)
    }

    pub fn is_encrypted(&self) -> bool {
        self.crypto_type != 0 || self.has_rights_id()
    }

    /// Whether this title uses title-key crypto (a nonzero rights id).
    pub fn has_rights_id(&self) -> bool {
        self.rights_id != [0u8; 16]
    }

    /// Master-key revision: the higher key-generation field, minus one.
    fn master_key_revision(&self) -> u8 {
        let crypto_type = self.key_generation_old.max(self.key_generation_new);
        crypto_type.saturating_sub(1)
    }

    /// The section AES key: the title key (unwrapped with this NCA's `titlekek`) or
    /// key-area slot 2.
    pub fn section_key(&self, keys: &crate::keys::KeySet) -> Result<[u8; 16], Error> {
        if self.has_rights_id() {
            let generation = self.master_key_revision();
            if let Some(key) = keys.title_key(&self.rights_id, generation) {
                return Ok(key);
            }
            return Err(if keys.wrapped_title_key(&self.rights_id).is_none() {
                Error::Nca("no title key loaded for this title's rights id".into())
            } else {
                Error::Nca(format!(
                    "missing titlekek_{:02x} in prod.keys, needed to unwrap this title's key",
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

    /// A decrypting [`ByteSource`] over section `index`'s body; `nca` is the whole NCA.
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
                    "missing FS header; pass the full NCA (>= 0xC00 bytes) with a loaded header_key"
                        .into(),
                )
            })?;
        let key = match fs.encryption_type {
            // Patch tables are encrypted under the section's own counter.
            ENCRYPTION_AES_CTR | ENCRYPTION_AES_CTR_EX => Some(self.section_key(keys)?),
            ENCRYPTION_NONE => None,
            other => {
                return Err(Error::Nca(format!(
                    "unsupported section encryption type {}",
                    other
                )))
            }
        };
        // A sparse section reads from its stored body, not the section table extent.
        let (body, sparse) = if fs.sparse.magic == BKTR_MAGIC {
            let stored = fs
                .sparse
                .offset
                .checked_add(fs.sparse.size)
                .ok_or(Error::Overflow)?;
            let body = Window::new(
                nca,
                fs.sparse_physical_offset,
                stored,
                &format!("NCA section {} sparse body", index),
            )?;
            let table = read_sparse_table(&body, &fs, key)?;
            (body, Some(table))
        } else {
            let body = Window::new(
                nca,
                sec.media_offset,
                sec.media_size,
                &format!("NCA section {}", index),
            )?;
            (body, None)
        };
        Ok(SectionSource {
            body,
            key,
            fs,
            nca_offset: sec.media_offset,
            len: sec.media_size,
            sparse,
        })
    }

    /// Check a `HierarchicalSha256` section against the FS header's master hash.
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
            return Err(Error::Nca(
                "hash table region exceeds decrypted section".into(),
            ));
        }
        if crate::crypto::sha256(&plain[ht_start..ht_end]) != fs.master_hash {
            return Err(Error::Nca(
                "decrypted section hash mismatch; wrong keys or a corrupt file".into(),
            ));
        }
        self.verify_data_blocks(plain, fs)
    }

    /// Check the data region against the per-block hash table.
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
                    "block {} of {} does not match its hash; the {:#x} bytes at section offset {:#x} are not what this NCA says they are",
                    i,
                    blocks,
                    chunk.len(),
                    data_start + i * block as usize
                )));
            }
        }
        Ok(())
    }

    /// Decrypt section `index` in memory, verifying its master hash when present.
    pub fn decrypt_section(
        &self,
        raw: &[u8],
        keys: &crate::keys::KeySet,
        index: usize,
    ) -> Result<Vec<u8>, Error> {
        let section = self.section_source(SliceSource(raw), keys, index)?;
        let plain = section.read_vec(0, section.len())?;
        self.verify_section_hash(&plain, index)?;
        Ok(plain)
    }

    /// Read section `index` as an ExeFS and return the verified PFS0 payload.
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
        // Hashes cover the stored (compressed) bytes, so verify before decompressing.
        if fs.compression.magic == BKTR_MAGIC {
            let stored = SliceSource(&plain[start..end]);
            let image = crate::compressed::CompressedStorage::new(stored, fs.compression)?;
            return image.read_vec(0, image.len());
        }
        // Trim in place to avoid a second copy of the section.
        plain.truncate(end);
        plain.drain(..start);
        Ok(plain)
    }

    /// Decrypt section `index` and slice out its PFS0 payload.
    pub fn decrypt_pfs0_section(
        &self,
        raw: &[u8],
        keys: &crate::keys::KeySet,
        index: usize,
    ) -> Result<Vec<u8>, Error> {
        self.read_pfs0_section(SliceSource(raw), keys, index)
    }

    /// `(block size, block count)` covered by section `index`'s hash table, if verifiable.
    pub fn pfs0_hash_coverage(&self, index: usize) -> Option<(u32, u64)> {
        hash_coverage(self.fs_headers.get(index).and_then(|o| o.as_ref())?)
    }

    /// The index of this NCA's PFS0 (ExeFS) section, if any.
    pub fn exefs_section_index(&self) -> Option<usize> {
        self.fs_headers
            .iter()
            .position(|fs| matches!(fs, Some(h) if h.partition_type == 1))
    }

    /// The index of this NCA's RomFS section, if any.
    pub fn romfs_section_index(&self) -> Option<usize> {
        self.fs_headers.iter().position(
            |fs| matches!(fs, Some(h) if h.partition_type == 0 && h.fs_type == HASH_TYPE_IVFC),
        )
    }

    /// Whether this is an update's Program NCA, whose RomFS is a patch.
    pub fn is_update(&self) -> bool {
        self.romfs_section_index()
            .and_then(|i| self.fs_headers[i])
            .is_some_and(|fs| fs.encryption_type == ENCRYPTION_AES_CTR_EX)
    }

    /// A [`ByteSource`] over section `index`'s RomFS image, sanity-checked by its header size.
    pub fn romfs_source<S: ByteSource>(
        &self,
        nca: S,
        keys: &crate::keys::KeySet,
        index: usize,
    ) -> Result<RomFsImage<Window<SectionSource<S>>>, Error> {
        let fs = self
            .fs_headers
            .get(index)
            .and_then(|o| o.as_ref())
            .copied()
            .ok_or_else(|| Error::Nca(format!("no FS header for section {}", index)))?;
        if fs.encryption_type == ENCRYPTION_AES_CTR_EX {
            return Err(Error::Nca(
                "this section is an update's patch RomFS; it holds only what the update changed, \
                 and has to be read over the base title's RomFS (see `bktr::patched_romfs_source`)"
                    .into(),
            ));
        }
        // Sparse sections are reassembled in `section_source`.
        let section = self.section_source(nca, keys, index)?;
        if fs.romfs_data_offset >= section.len() {
            return Err(Error::Nca(
                "RomFS data offset exceeds the decrypted section".into(),
            ));
        }
        // Prefer the IVFC level size; the section size is rounded up to a media unit.
        let available = section.len() - fs.romfs_data_offset;
        let len = match fs.romfs_data_size {
            0 => available,
            stated => stated.min(available),
        };
        let stored = Window::new(section, fs.romfs_data_offset, len, "RomFS image")?;
        let romfs = RomFsImage::open(stored, fs.compression)?;
        // RomFS header_size is always 0x50; anything else means a wrong key.
        let mut header_size = [0u8; 8];
        romfs.read_exact_at(0, &mut header_size)?;
        const ROMFS_HEADER_SIZE: u64 = 0x50;
        if u64::from_le_bytes(header_size) != ROMFS_HEADER_SIZE {
            return Err(Error::Nca(
                "decrypted RomFS section doesn't start with a valid RomFS header; wrong keys or a corrupt file".into(),
            ));
        }
        Ok(romfs)
    }

    /// Decrypt section `index` as a whole RomFS image in memory.
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

/// Block size and count of a `HierarchicalSha256` hash table, if the layout is recognised.
fn hash_coverage(fs: &FsHeader) -> Option<(u32, u64)> {
    if fs.fs_type != HASH_TYPE_SHA256 || fs.hash_block_size == 0 {
        return None;
    }
    let blocks = fs.data_size.div_ceil(fs.hash_block_size as u64);
    (blocks.checked_mul(32) == Some(fs.hash_table_size)).then_some((fs.hash_block_size, blocks))
}

/// A decrypting view of one NCA section.
#[derive(Debug, Clone)]
pub struct SectionSource<S> {
    /// The section body, still encrypted, addressed from the section start.
    body: Window<S>,
    /// The AES-128-CTR section key, or `None` for an unencrypted section.
    key: Option<[u8; 16]>,
    fs: FsHeader,
    /// The section's absolute NCA offset, which numbers its counter blocks.
    nca_offset: u64,
    /// Declared section size; larger than the body when sparse.
    len: u64,
    /// How to put a sparse section's holes back, when it has any.
    sparse: Option<crate::sparse::SparseTable>,
}

impl<S: ByteSource> SectionSource<S> {
    /// The section's FS header (hash type, encryption, IVFC layout).
    pub fn fs_header(&self) -> &FsHeader {
        &self.fs
    }

    /// Decrypt `buf` at aligned section offset `at`; `ctr_val` overrides the counter top word.
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

    /// Read a range of a patch section within one subsection.
    pub(crate) fn read_region(
        &self,
        offset: u64,
        out: &mut [u8],
        ctr_val: u32,
    ) -> Result<usize, Error> {
        self.read_decrypting(offset, out, Some(ctr_val))
    }

    /// The section's encrypted bytes, reassembled first when sparse.
    fn read_raw(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        match &self.sparse {
            None => self.body.read_at(offset, out),
            Some(table) => table.read_raw(&self.body, self.len, offset, out),
        }
    }

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
        // A mid-block start decrypts its first block through a scratch block.
        if head != 0 {
            let mut block = [0u8; 16];
            let got = self.read_raw(aligned, &mut block)?;
            self.decrypt_at(aligned, &mut block[..got], ctr_val);
            let take = (16 - head).min(want).min(got.saturating_sub(head));
            out[..take].copy_from_slice(&block[head..head + take]);
            done = take;
            // Short read: the section ended inside this block.
            if take < (16 - head).min(want) {
                return Ok(done);
            }
        }
        if done < want {
            let at = aligned + if head != 0 { 16 } else { 0 };
            let rest = &mut out[done..want];
            let got = self.read_raw(at, rest)?;
            self.decrypt_at(at, &mut rest[..got], ctr_val);
            done += got;
        }
        Ok(done)
    }
}

impl<S: ByteSource> ByteSource for SectionSource<S> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        self.read_decrypting(offset, out, None)
    }
}

/// Read and decrypt a sparse section's table out of its stored body.
fn read_sparse_table<S: ByteSource>(
    body: &S,
    fs: &FsHeader,
    key: Option<[u8; 16]>,
) -> Result<crate::sparse::SparseTable, Error> {
    if !fs.sparse.offset.is_multiple_of(16) {
        return Err(Error::Nca(format!(
            "sparse table starts at {:#x}, which is not a counter block boundary",
            fs.sparse.offset
        )));
    }
    let mut meta = body.read_vec(fs.sparse.offset, fs.sparse.size)?;
    if let Some(key) = key {
        let at = fs
            .sparse_physical_offset
            .checked_add(fs.sparse.offset)
            .ok_or(Error::Overflow)?;
        crate::crypto::aes128_ctr_xor_in_place(&key, &fs.sparse_counter(at), &mut meta);
    }
    crate::sparse::SparseTable::parse(&meta, fs.sparse, fs.sparse.offset)
}

/// A title's RomFS image, plain or compressed.
#[derive(Debug)]
pub enum RomFsImage<S: ByteSource> {
    /// Uncompressed image.
    Plain(S),
    /// A run of LZ4 blocks.
    Compressed(crate::compressed::CompressedStorage<S>),
}

impl<S: ByteSource> RomFsImage<S> {
    /// Put the compression layer over `stored` when `compression` declares one.
    pub fn open(stored: S, compression: BktrTable) -> Result<RomFsImage<S>, Error> {
        if compression.magic == BKTR_MAGIC {
            Ok(RomFsImage::Compressed(
                crate::compressed::CompressedStorage::new(stored, compression)?,
            ))
        } else {
            Ok(RomFsImage::Plain(stored))
        }
    }
}

impl<S: ByteSource> ByteSource for RomFsImage<S> {
    fn len(&self) -> u64 {
        match self {
            RomFsImage::Plain(s) => s.len(),
            RomFsImage::Compressed(s) => s.len(),
        }
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        match self {
            RomFsImage::Plain(s) => s.read_at(offset, out),
            RomFsImage::Compressed(s) => s.read_at(offset, out),
        }
    }
}

/// Find the first NCA of content type `want` in a PFS0 container.
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
                               // Section 0: PFS0 at media unit 0, 0x10 units long.
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

    /// A rights-id title key must be unwrapped with `titlekek`.
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
        assert!(find_nca_by_type(&files, &src, &keys, ContentType::Manual).is_none());
    }
}

#[cfg(test)]
mod decrypt_tests {
    use super::*;
    use crate::crypto::{aes128_encrypt_block, aes128_xts_decrypt};
    use crate::keys::{KeySet, KEY_GENERATION_COUNT};

    fn encrypt_xts(key: &[u8; 32], data: &[u8], sector: u64, sector_size: usize) -> Vec<u8> {
        // Standard XTS encrypt: E(K1, P ^ T) ^ T.
        let mut key1 = [0u8; 16];
        let mut key2 = [0u8; 16];
        key1.copy_from_slice(&key[..16]);
        key2.copy_from_slice(&key[16..]);
        let mut out = Vec::with_capacity(data.len());
        for (s, chunk) in (sector..).zip(data.chunks(sector_size)) {
            let mut tweak = [0u8; 16];
            let mut sv = s;
            for i in (0..16).rev() {
                tweak[i] = (sv & 0xff) as u8;
                sv >>= 8;
            }
            let mut tweak = aes128_encrypt_block(&key2, &tweak);
            for blk in chunk.chunks(16) {
                let mut p = [0u8; 16];
                p[..blk.len()].copy_from_slice(blk);
                let mut x = [0u8; 16];
                for (i, b) in x.iter_mut().enumerate() {
                    *b = p[i] ^ tweak[i];
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
        // Encrypt a cleartext NCA header; parse_with_keys must decrypt it.
        let mut hdr = [0u8; 0x400];
        hdr[0x200..0x204].copy_from_slice(&NCA_MAGIC.to_le_bytes());
        hdr[0x204] = 2; // distribution type
        hdr[0x205] = 0; // content type: Program
        hdr[0x210..0x218].copy_from_slice(&0x010075600ae96800u64.to_le_bytes()); // title id
        hdr[0x218..0x21C].copy_from_slice(&0x00090007u32.to_le_bytes()); // sdk version
        hdr[0x21C] = 0; // crypto type

        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
        }
        let encrypted = encrypt_xts(&key, &hdr, 0, 0x200);

        let keys = KeySet {
            header_key: Some(key),
            ..Default::default()
        };
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
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
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

    const HASH_BLOCK_SIZE: u32 = 0x1_0000;

    /// Build a synthetic encrypted Program NCA with an AES-CTR ExeFS section.
    fn build_exefs_nca() -> (Vec<u8>, KeySet, Vec<u8>) {
        build_exefs_nca_with(false)
    }

    /// Same, optionally storing the ExeFS compressed.
    fn build_exefs_nca_with(compressed: bool) -> (Vec<u8>, KeySet, Vec<u8>) {
        use crate::crypto::{aes128_ctr_xor, sha256};

        let header_key = {
            let mut k = [0u8; 32];
            for (i, b) in k.iter_mut().enumerate() {
                *b = i as u8;
            }
            k
        };
        let kek = {
            let mut k = [0u8; 16];
            for (i, b) in k.iter_mut().enumerate() {
                *b = 0xA0 + i as u8;
            }
            k
        };
        let section_key = {
            let mut k = [0u8; 16];
            for (i, b) in k.iter_mut().enumerate() {
                *b = 0xB0 + i as u8;
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

        // Section 0 entry: `u32 start; u32 end` in media units.
        const SECTION_OFFSET: usize = 0x1000;
        let pfs0 = build_pfs0("main", b"fake NSO bytes for the test");
        let (stored, compression) = if compressed {
            use crate::compressed::testing::{build, Block};
            let (image, plain, table) = build(&[Block::Lz4(pfs0.clone())]);
            assert_eq!(plain, pfs0, "the fixture must compress the real payload");
            (image, Some(table))
        } else {
            (pfs0.clone(), None)
        };
        let hash_table = sha256(&stored).to_vec();
        let plain_section = [hash_table.clone(), stored.clone()].concat();

        let at = h + 0x40;
        let start_units = (SECTION_OFFSET / 0x200) as u32;
        let size_units = plain_section.len().div_ceil(0x200) as u32;
        let end_units = start_units + size_units;
        header[at..at + 4].copy_from_slice(&start_units.to_le_bytes());
        header[at + 4..at + 8].copy_from_slice(&end_units.to_le_bytes());

        // Key area slot 2 holds `section_key`, ECB-encrypted with `kek`.
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
        header[fs0 + 0x48..fs0 + 0x50].copy_from_slice(&(stored.len() as u64).to_le_bytes());
        header[fs0 + 0x140..fs0 + 0x144].copy_from_slice(&generation.to_le_bytes());
        header[fs0 + 0x144..fs0 + 0x148].copy_from_slice(&secure_value.to_le_bytes());
        if let Some(table) = compression {
            let at = fs0 + 0x178;
            header[at..at + 8].copy_from_slice(&table.offset.to_le_bytes());
            header[at + 8..at + 16].copy_from_slice(&table.size.to_le_bytes());
            header[at + 0x10..at + 0x14].copy_from_slice(&table.magic.to_le_bytes());
            header[at + 0x14..at + 0x18].copy_from_slice(&1u32.to_le_bytes()); // version
            header[at + 0x18..at + 0x1C].copy_from_slice(&table.entries.to_le_bytes());
        }
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
        raw[SECTION_OFFSET..SECTION_OFFSET + encrypted_section.len()]
            .copy_from_slice(&encrypted_section);

        let keys = KeySet {
            header_key: Some(header_key),
            key_area_key_system: {
                let mut slots = [None; KEY_GENERATION_COUNT];
                slots[0] = Some(kek);
                slots
            },
            ..Default::default()
        };
        (raw, keys, pfs0)
    }

    /// End-to-end decrypt and extract of a synthetic ExeFS.
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

        // A wrong key-area key must fail the master-hash check.
        let mut wrong_keys = keys.clone();
        wrong_keys.key_area_key_system[0] = Some([0u8; 16]);
        assert!(matches!(
            nca.decrypt_pfs0_section(&raw, &wrong_keys, 0),
            Err(Error::Nca(_))
        ));
    }

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

    /// A compressed RomFS section reads as its decompressed image.
    #[test]
    fn a_compressed_romfs_section_is_served_decompressed() {
        use crate::compressed::testing::{build, Block};

        let mut header = vec![0u8; 0x50];
        header[..8].copy_from_slice(&0x50u64.to_le_bytes());
        let (stored, plain, table) = build(&[
            Block::Raw(header),
            Block::Lz4((0..=0xFFu8).map(|b| b ^ 0x33).collect()),
            Block::Zeros(0x40),
            Block::Lz4(vec![0x11; 0x180]),
        ]);
        assert!(stored.len() < plain.len() + table.size as usize);

        let (raw, keys, _, _) = build_romfs_nca_with(stored, Some(table));
        let nca = Nca::parse_with_keys(&raw, Some(&keys)).expect("parse");
        assert_eq!(nca.fs_headers[0].unwrap().compression.magic, BKTR_MAGIC);

        let romfs = nca
            .romfs_source(SliceSource(&raw), &keys, 0)
            .expect("romfs source");
        assert!(matches!(romfs, RomFsImage::Compressed(_)));
        // The decompressed size, not the stored one.
        assert_eq!(romfs.len(), plain.len() as u64);
        assert_eq!(romfs.read_vec(0, romfs.len()).unwrap(), plain);

        // Ranges that start and end inside a block, and across boundaries.
        for &(offset, len) in &[(0u64, 8usize), (0x4f, 2), (0x51, 0x7f), (0x14f, 0x60)] {
            let mut out = vec![0u8; len];
            assert_eq!(romfs.read_at(offset, &mut out).unwrap(), len);
            assert_eq!(out, &plain[offset as usize..offset as usize + len]);
        }

        // A wrong key fails the header check.
        let mut wrong_keys = keys.clone();
        wrong_keys.key_area_key_system[0] = Some([0u8; 16]);
        assert!(matches!(
            nca.romfs_source(SliceSource(&raw), &wrong_keys, 0),
            Err(Error::Nca(_))
        ));
    }

    /// A section claiming a sparse layer without describing one is refused.
    #[test]
    fn a_sparse_layer_that_describes_nothing_is_refused() {
        let (mut raw, keys, _, _) = build_romfs_nca();
        // Re-encrypt the XTS header around the edited field.
        let header_key = keys.header_key.expect("header key");
        let mut header = aes128_xts_decrypt(&header_key, &raw[..NCA_FULL_HEADER_SIZE], 0, 0x200);
        header[0x400 + 0x148 + 0x10..0x400 + 0x148 + 0x14]
            .copy_from_slice(&BKTR_MAGIC.to_le_bytes());
        let encrypted = encrypt_xts(&header_key, &header, 0, 0x200);
        raw[..NCA_FULL_HEADER_SIZE].copy_from_slice(&encrypted);

        let nca = Nca::parse_with_keys(&raw, Some(&keys)).expect("parse");
        assert_eq!(nca.fs_headers[0].unwrap().sparse.magic, BKTR_MAGIC);
        let err = nca
            .romfs_source(SliceSource(&raw), &keys, 0)
            .expect_err("an empty sparse table describes no section");
        assert!(format!("{err}").contains("sparse"), "{err}");
    }

    /// Compressed ExeFS hashes cover the compressed bytes.
    #[test]
    fn a_compressed_exefs_section_is_extracted_decompressed() {
        let (raw, keys, pfs0) = build_exefs_nca_with(true);
        let nca = Nca::parse_with_keys(&raw, Some(&keys)).expect("parse");
        assert_eq!(nca.fs_headers[0].unwrap().compression.magic, BKTR_MAGIC);

        let extracted = nca
            .decrypt_pfs0_section(&raw, &keys, 0)
            .expect("decrypt + hash-verify + decompress");
        assert_eq!(extracted, pfs0);

        let inner = crate::nsp::Pfs0::parse(&extracted).expect("valid PFS0");
        let main = inner.find("main").expect("main entry");
        assert_eq!(
            &extracted[main.offset as usize..][..main.size as usize],
            b"fake NSO bytes for the test"
        );

        let mut corrupt = raw.clone();
        let data_start = 0x1000 + nca.fs_headers[0].unwrap().data_offset as usize;
        corrupt[data_start + 9] ^= 0x01;
        assert!(matches!(
            nca.decrypt_pfs0_section(&corrupt, &keys, 0),
            Err(Error::Nca(_))
        ));
    }

    /// Build a synthetic encrypted NCA with a sparse RomFS section whose declared
    /// extent lies past EOF.
    fn build_sparse_romfs_nca() -> (Vec<u8>, KeySet, Vec<u8>) {
        use crate::crypto::aes128_ctr_xor;
        use crate::sparse::{STORAGE_DATA, STORAGE_HOLE};

        let mut header_key = [0u8; 32];
        for (i, b) in header_key.iter_mut().enumerate() {
            *b = (0x10 + i) as u8;
        }
        let mut kek = [0u8; 16];
        for (i, b) in kek.iter_mut().enumerate() {
            *b = 0xE0 + i as u8;
        }
        let mut section_key = [0u8; 16];
        for (i, b) in section_key.iter_mut().enumerate() {
            *b = 0xF0 + i as u8;
        }

        const SECTION_OFFSET: u64 = 0x20000;
        const BODY_OFFSET: u64 = 0x1000;
        const SECTION_SIZE: u64 = 0x400;
        const LEVEL5_OFFSET: u64 = 0x40;
        // [0, 0x100) is kept, [0x100, 0x200) is a hole, [0x200, 0x400) is kept.
        const HOLE: std::ops::Range<usize> = 0x100..0x200;
        const TABLE_AT: u64 = 0x300;

        let generation: u32 = 0x05;
        let secure_value: u32 = 0x1122_3344;
        let sparse_generation: u16 = 0x0007;

        let counter = |at: u64| {
            let mut ctr = [0u8; 16];
            ctr[0..4].copy_from_slice(&secure_value.to_be_bytes());
            ctr[4..8].copy_from_slice(&generation.to_be_bytes());
            ctr[8..16].copy_from_slice(&(at >> 4).to_be_bytes());
            ctr
        };

        // A hole decrypts to the keystream; see `crate::sparse`.
        let mut plain: Vec<u8> = (0..SECTION_SIZE).map(|i| (i as u8) ^ 0x3C).collect();
        plain[LEVEL5_OFFSET as usize..LEVEL5_OFFSET as usize + 8]
            .copy_from_slice(&0x50u64.to_le_bytes()); // RomFS header_size
        let keystream = aes128_ctr_xor(
            &section_key,
            &counter(SECTION_OFFSET + HOLE.start as u64),
            &vec![0u8; HOLE.len()],
        );
        plain[HOLE].copy_from_slice(&keystream);

        let cipher = aes128_ctr_xor(&section_key, &counter(SECTION_OFFSET), &plain);
        assert!(
            cipher[HOLE].iter().all(|&b| b == 0),
            "a hole stores nothing"
        );

        let mut body = Vec::new();
        body.extend_from_slice(&cipher[..HOLE.start]);
        body.extend_from_slice(&cipher[HOLE.end..]);
        let entries = [
            (0u64, 0u64, STORAGE_DATA),
            (HOLE.start as u64, 0, STORAGE_HOLE),
            (HOLE.end as u64, HOLE.start as u64, STORAGE_DATA),
        ];
        let meta = crate::sparse::testing::write_table(&entries, SECTION_SIZE);
        body.resize(TABLE_AT as usize, 0);
        // The table is encrypted at its stored offset.
        let mut sparse_ctr = counter(BODY_OFFSET + TABLE_AT);
        sparse_ctr[4..8].copy_from_slice(&(u32::from(sparse_generation) << 16).to_be_bytes());
        body.extend_from_slice(&aes128_ctr_xor(&section_key, &sparse_ctr, &meta));

        let mut header = vec![0u8; NCA_FULL_HEADER_SIZE];
        let h = NCA_HEADER_OFFSET;
        header[h..h + 4].copy_from_slice(&NCA_MAGIC.to_le_bytes());
        header[h + 0x05] = 4; // content type: Data
        header[h + 0x07] = 2; // key index: System
        header[h + 0x1C] = 1; // crypto type: encrypted
        let at = h + 0x40;
        header[at..at + 4].copy_from_slice(&((SECTION_OFFSET / 0x200) as u32).to_le_bytes());
        header[at + 4..at + 8]
            .copy_from_slice(&(((SECTION_OFFSET + SECTION_SIZE) / 0x200) as u32).to_le_bytes());
        header[h + 0x120..h + 0x130]
            .copy_from_slice(&crate::crypto::aes128_encrypt_block(&kek, &section_key));

        let fs0 = 0x400;
        header[fs0 + 0x02] = 0; // partition_type: RomFs
        header[fs0 + 0x03] = HASH_TYPE_IVFC;
        header[fs0 + 0x04] = ENCRYPTION_AES_CTR;
        header[fs0 + 0x18 + 5 * 24..fs0 + 0x18 + 5 * 24 + 8]
            .copy_from_slice(&LEVEL5_OFFSET.to_le_bytes());
        header[fs0 + 0x18 + 5 * 24 + 8..fs0 + 0x18 + 5 * 24 + 16]
            .copy_from_slice(&(SECTION_SIZE - LEVEL5_OFFSET).to_le_bytes());
        header[fs0 + 0x140..fs0 + 0x144].copy_from_slice(&generation.to_le_bytes());
        header[fs0 + 0x144..fs0 + 0x148].copy_from_slice(&secure_value.to_le_bytes());
        let sp = fs0 + 0x148;
        header[sp..sp + 8].copy_from_slice(&TABLE_AT.to_le_bytes());
        header[sp + 8..sp + 16].copy_from_slice(&(meta.len() as u64).to_le_bytes());
        header[sp + 0x10..sp + 0x14].copy_from_slice(&BKTR_MAGIC.to_le_bytes());
        header[sp + 0x14..sp + 0x18].copy_from_slice(&1u32.to_le_bytes()); // version
        header[sp + 0x18..sp + 0x1C].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        header[sp + 0x20..sp + 0x28].copy_from_slice(&BODY_OFFSET.to_le_bytes());
        header[sp + 0x28..sp + 0x2A].copy_from_slice(&sparse_generation.to_le_bytes());

        let mut raw = vec![0u8; BODY_OFFSET as usize];
        raw[..NCA_FULL_HEADER_SIZE].copy_from_slice(&encrypt_xts(&header_key, &header, 0, 0x200));
        raw.extend_from_slice(&body);
        assert!(
            (raw.len() as u64) < SECTION_OFFSET,
            "the declared extent must not be in the file"
        );

        let keys = KeySet {
            header_key: Some(header_key),
            key_area_key_system: {
                let mut slots = [None; KEY_GENERATION_COUNT];
                slots[0] = Some(kek);
                slots
            },
            ..Default::default()
        };
        (raw, keys, plain)
    }

    /// A sparse section reads as the reassembled section.
    #[test]
    fn a_sparse_section_is_reassembled_before_it_is_decrypted() {
        let (raw, keys, plain) = build_sparse_romfs_nca();
        let nca = Nca::parse_with_keys(&raw, Some(&keys)).expect("parse");
        let fs = nca.fs_headers[0].expect("fs header");
        assert_eq!(fs.sparse.magic, BKTR_MAGIC);
        assert_eq!(fs.sparse_physical_offset, 0x1000);
        assert_eq!(fs.sparse_generation, 7);

        let section = nca
            .section_source(SliceSource(&raw), &keys, 0)
            .expect("section source");
        assert_eq!(section.len(), plain.len() as u64);
        assert_eq!(section.len(), 0x400);
        assert_eq!(section.read_vec(0, section.len()).unwrap(), plain);

        // Unaligned ranges across every boundary.
        for &(offset, len) in &[
            (0u64, 1usize),
            (0xff, 2),     // the last kept byte and the first of the hole
            (0x101, 0x7f), // unaligned inside the hole
            (0x1ff, 3),    // out of the hole and into the kept range past it
            (0x37, 0x200),
        ] {
            let mut out = vec![0u8; len];
            assert_eq!(
                section.read_at(offset, &mut out).unwrap(),
                len,
                "short read at {offset:#x}+{len:#x}"
            );
            assert_eq!(
                out,
                &plain[offset as usize..offset as usize + len],
                "wrong bytes at {offset:#x}+{len:#x}"
            );
        }

        let romfs = nca
            .romfs_source(SliceSource(&raw), &keys, 0)
            .expect("romfs source");
        assert_eq!(romfs.len(), plain.len() as u64 - 0x40);
        assert_eq!(romfs.read_vec(0, romfs.len()).unwrap(), &plain[0x40..]);
    }

    /// Build a synthetic encrypted Data NCA with a single IVFC RomFS section.
    fn build_romfs_nca() -> (Vec<u8>, KeySet, Vec<u8>, u64) {
        let mut image = vec![0u8; 0x1C0];
        image[..8].copy_from_slice(&0x50u64.to_le_bytes()); // RomFS header_size
                                                            // Each byte past the header encodes its offset.
        for (i, byte) in image.iter_mut().enumerate().skip(8) {
            *byte = (i as u8).wrapping_add(0x40) ^ 0x5A;
        }
        build_romfs_nca_with(image, None)
    }

    /// Same, with a caller-chosen image and compression table.
    fn build_romfs_nca_with(
        image: Vec<u8>,
        compression: Option<BktrTable>,
    ) -> (Vec<u8>, KeySet, Vec<u8>, u64) {
        use crate::crypto::aes128_ctr_xor;

        let header_key = {
            let mut k = [0u8; 32];
            for (i, b) in k.iter_mut().enumerate() {
                *b = (0x40 + i) as u8;
            }
            k
        };
        let kek = {
            let mut k = [0u8; 16];
            for (i, b) in k.iter_mut().enumerate() {
                *b = 0xC0 + i as u8;
            }
            k
        };
        let section_key = {
            let mut k = [0u8; 16];
            for (i, b) in k.iter_mut().enumerate() {
                *b = 0xD0 + i as u8;
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
        // Real RomFS data lives at IVFC level 5, not at section offset 0.
        const LEVEL5_OFFSET: u64 = 0x40;
        let level5_size = image.len() as u64;
        let mut plain_section = vec![0xAAu8; LEVEL5_OFFSET as usize]; // levels 0..4 "hash tables"
        plain_section.extend_from_slice(&image);
        // The image ends before the section does.
        let padded = plain_section.len().next_multiple_of(0x200);
        plain_section.resize(padded, 0xEE);

        let at = h + 0x40;
        let start_units = (SECTION_OFFSET / 0x200) as u32;
        let size_units = plain_section.len().div_ceil(0x200) as u32;
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
        header[fs0 + 0x18 + 5 * 24..fs0 + 0x18 + 5 * 24 + 8]
            .copy_from_slice(&LEVEL5_OFFSET.to_le_bytes()); // level[5].logical_offset
        header[fs0 + 0x18 + 5 * 24 + 8..fs0 + 0x18 + 5 * 24 + 16]
            .copy_from_slice(&level5_size.to_le_bytes()); // level[5].hash_data_size
        if let Some(table) = compression {
            let at = fs0 + 0x178;
            header[at..at + 8].copy_from_slice(&table.offset.to_le_bytes());
            header[at + 8..at + 16].copy_from_slice(&table.size.to_le_bytes());
            header[at + 0x10..at + 0x14].copy_from_slice(&table.magic.to_le_bytes());
            header[at + 0x14..at + 0x18].copy_from_slice(&1u32.to_le_bytes()); // version
            header[at + 0x18..at + 0x1C].copy_from_slice(&table.entries.to_le_bytes());
        }
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
        raw[SECTION_OFFSET..SECTION_OFFSET + encrypted_section.len()]
            .copy_from_slice(&encrypted_section);

        let keys = KeySet {
            header_key: Some(header_key),
            key_area_key_system: {
                let mut slots = [None; KEY_GENERATION_COUNT];
                slots[0] = Some(kek);
                slots
            },
            ..Default::default()
        };
        (raw, keys, plain_section, LEVEL5_OFFSET)
    }

    #[test]
    fn decrypts_a_synthetic_romfs_section() {
        let (raw, keys, plain_section, level5) = build_romfs_nca();

        let nca = Nca::parse_with_keys(&raw, Some(&keys)).expect("parse");
        assert_eq!(nca.romfs_section_index(), Some(0));
        assert_eq!(nca.exefs_section_index(), None);
        assert_eq!(nca.fs_headers[0].unwrap().romfs_data_offset, level5);

        let extracted = nca
            .decrypt_romfs_section(&raw, &keys, 0)
            .expect("decrypt romfs");
        assert_eq!(extracted, &plain_section[level5 as usize..]);

        // A wrong key is caught by the header_size check.
        let mut wrong_keys = keys.clone();
        wrong_keys.key_area_key_system[0] = Some([0u8; 16]);
        assert!(matches!(
            nca.decrypt_romfs_section(&raw, &wrong_keys, 0),
            Err(Error::Nca(_))
        ));
    }

    /// Unaligned range reads through `romfs_source`.
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
            (0u64, 8usize), // the RomFS header itself
            (1, 1),         // one byte, mid-block
            (3, 13),        // up to the first block boundary
            (3, 14),        // and one byte past it
            (15, 2),        // straddling a block boundary
            (16, 32),       // exactly aligned, whole blocks
            (17, 100),      // unaligned start, unaligned end
            (0x3f, 0x81),   // spanning many blocks
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

        // Reads past the end return what they filled, then nothing.
        let mut out = vec![0u8; 64];
        let last = romfs.len() - 10;
        assert_eq!(romfs.read_at(last, &mut out).unwrap(), 10);
        assert_eq!(&out[..10], &want[last as usize..]);
        assert_eq!(romfs.read_at(romfs.len(), &mut out).unwrap(), 0);
    }
}
