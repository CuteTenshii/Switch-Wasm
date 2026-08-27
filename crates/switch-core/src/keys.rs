//! Nintendo key-file parsing (`prod.keys` / `title.keys`) and derivation of
//! the NCA header key.
//!
//! Key files are `name = hex` lines (the format used by lockpick/hactool).
//! To decrypt an NCA header we only need the global `header_key`: use it
//! directly if the file provides it, otherwise derive it from the sources via
//! the master-key chain (hactool `pki.c`).

use crate::crypto::aes128_ecb_decrypt;

/// Number of key generations (`_00`.._1f`) prod.keys dumps carry per key kind.
pub const KEY_GENERATION_COUNT: usize = 0x20;

/// Which of the three "key area" key families decrypts an NCA's embedded key
/// area, selected by the NCA header's key index byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAreaKind {
    Application,
    Ocean,
    System,
}

impl KeyAreaKind {
    pub fn from_index(index: u8) -> Option<KeyAreaKind> {
        match index {
            0 => Some(KeyAreaKind::Application),
            1 => Some(KeyAreaKind::Ocean),
            2 => Some(KeyAreaKind::System),
            _ => None,
        }
    }
}

/// Parsed keysets with everything needed to decrypt NCA headers and bodies.
#[derive(Debug, Default, Clone)]
pub struct KeySet {
    /// 32-byte NCA header key (directly, if provided).
    pub header_key: Option<[u8; 32]>,
    /// Title keys by rights id (16 bytes each) exactly as `title.keys`
    /// stores them: each one is still wrapped, AES-128-ECB, under the
    /// `titlekek_XX` for its title's key generation. Lockpick_RCM copies a
    /// ticket's key block into that file verbatim, so an entry here is the
    /// ciphertext, not a usable NCA section key — [`KeySet::title_key`]
    /// unwraps it.
    ///
    /// A ticket bundled in a container lands here too, in the same wrapped
    /// form (see [`crate::ticket::load_bundled_title_key`]): this is the only
    /// representation of a title key the keyset holds, so the generation that
    /// unwraps one is always the NCA's, never a ticket field's.
    pub title_keys: Vec<([u8; 16], [u8; 16])>,
    // Sources for deriving the header key (prod.keys).
    pub header_key_source: Option<[u8; 32]>,
    pub header_kek_source: Option<[u8; 16]>,
    pub master_key_00: Option<[u8; 16]>,
    pub aes_kek_generation_source: Option<[u8; 16]>,
    pub aes_key_generation_source: Option<[u8; 16]>,
    /// `key_area_key_application_XX` / `_ocean_XX` / `_system_XX`, indexed by
    /// key generation. Like `header_key`, these are stored directly rather
    /// than derived — that's what prod.keys dumps (Lockpick_RCM) provide, and
    /// deriving them would need Nintendo's secret seed constants, which this
    /// project does not embed.
    pub key_area_key_application: [Option<[u8; 16]>; KEY_GENERATION_COUNT],
    pub key_area_key_ocean: [Option<[u8; 16]>; KEY_GENERATION_COUNT],
    pub key_area_key_system: [Option<[u8; 16]>; KEY_GENERATION_COUNT],
    /// `titlekek_XX`, indexed by key generation — decrypts a "Common"-crypto
    /// ticket's title-key block (see `ticket.rs`). Stored directly, like the
    /// key-area keys above.
    pub titlekek: [Option<[u8; 16]>; KEY_GENERATION_COUNT],
}

impl KeySet {
    /// Look up a still-`titlekek`-wrapped title key by rights id, as
    /// `title.keys` and a bundled ticket both store it.
    pub fn wrapped_title_key(&self, rights_id: &[u8; 16]) -> Option<[u8; 16]> {
        find_key(&self.title_keys, rights_id)
    }

    /// Whether this keyset carries a title key for `rights_id` at all,
    /// wrapped or not — which is a different question from whether the
    /// `titlekek` that unwraps it is present.
    pub fn has_title_key(&self, rights_id: &[u8; 16]) -> bool {
        self.wrapped_title_key(rights_id).is_some()
    }

    /// The usable AES-128 title key for `rights_id`: the stored key block
    /// unwrapped with `titlekek_<generation>`, where `generation` is the
    /// NCA's key generation.
    pub fn title_key(&self, rights_id: &[u8; 16], generation: u8) -> Option<[u8; 16]> {
        let wrapped = self.wrapped_title_key(rights_id)?;
        let kek = self.titlekek(generation)?;
        Some(crate::crypto::aes128_decrypt_block(&kek, &wrapped))
    }

    /// Record a title key in its stored, `titlekek`-wrapped form, replacing
    /// any entry this keyset already had for the same title.
    pub fn add_title_key(&mut self, rights_id: [u8; 16], wrapped: [u8; 16]) {
        match self.title_keys.iter_mut().find(|(id, _)| *id == rights_id) {
            Some(slot) => slot.1 = wrapped,
            None => self.title_keys.push((rights_id, wrapped)),
        }
    }

    /// Look up a key-area key by kind and generation (the NCA header's key
    /// index and key generation byte).
    pub fn key_area_key(&self, kind: KeyAreaKind, generation: u8) -> Option<[u8; 16]> {
        let table = match kind {
            KeyAreaKind::Application => &self.key_area_key_application,
            KeyAreaKind::Ocean => &self.key_area_key_ocean,
            KeyAreaKind::System => &self.key_area_key_system,
        };
        table.get(generation as usize).copied().flatten()
    }

    /// Look up `titlekek_<generation>`.
    pub fn titlekek(&self, generation: u8) -> Option<[u8; 16]> {
        self.titlekek.get(generation as usize).copied().flatten()
    }

    /// The 32-byte header key, either provided directly or derived from the
    /// prod.keys sources (hactool `pki.c`):
    /// `header_key = AESECBDecrypt(header_key_source, header_kek)` where
    /// `header_kek` is derived from `header_kek_source` via the master key.
    pub fn effective_header_key(&self) -> Option<[u8; 32]> {
        if let Some(k) = self.header_key {
            return Some(k);
        }
        let src = self.header_key_source?;
        let kek = self.derive_header_kek()?;
        let mut out = [0u8; 32];
        out.copy_from_slice(&aes128_ecb_decrypt(&kek, &src));
        Some(out)
    }

    fn derive_header_kek(&self) -> Option<[u8; 16]> {
        let master = self.master_key_00?;
        let kek_seed = self.aes_kek_generation_source?;
        let key_seed = self.aes_key_generation_source?;
        let header_kek_source = self.header_kek_source?;
        // generate_kek (hactool pki.c):
        //   kek = AESECBDecrypt(master, kek_seed)
        //   src_kek = AESECBDecrypt(kek, header_kek_source)
        //   header_kek = AESECBDecrypt(src_kek, key_seed)
        let mut kek = [0u8; 16];
        kek.copy_from_slice(&aes128_ecb_decrypt(&master, &kek_seed)[..16]);
        let mut src_kek = [0u8; 16];
        src_kek.copy_from_slice(&aes128_ecb_decrypt(&kek, &header_kek_source)[..16]);
        let mut out = [0u8; 16];
        out.copy_from_slice(&aes128_ecb_decrypt(&src_kek, &key_seed)[..16]);
        Some(out)
    }
}

fn find_key(table: &[([u8; 16], [u8; 16])], rights_id: &[u8; 16]) -> Option<[u8; 16]> {
    table
        .iter()
        .find(|(id, _)| id == rights_id)
        .map(|(_, k)| *k)
}

/// Parse a `prod.keys` / `title.keys` file: `name = hexdigits` lines, `#`
/// comments, blank lines ignored. Duplicate keys overwrite.
pub fn parse_keys_file(text: &str) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(eq) = line.find('=') else { continue };
        let name = line[..eq].trim();
        let value = line[eq + 1..].trim();
        // Allow "0x" prefixes and inline comments.
        let value = value.split(['#', ';']).next().unwrap_or("").trim();
        let value = value.strip_prefix("0x").unwrap_or(value);
        let value = value.replace([' ', '_', '-'], "");
        if value.len() % 2 != 0 || value.is_empty() {
            continue;
        }
        let mut bytes = Vec::with_capacity(value.len() / 2);
        let mut ok = true;
        for i in (0..value.len()).step_by(2) {
            match u8::from_str_radix(&value[i..i + 2], 16) {
                Ok(b) => bytes.push(b),
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            out.push((name.to_string(), bytes));
        }
    }
    out
}

/// Build a [`KeySet`] from parsed `prod.keys` entries.
pub fn keyset_from_prod(entries: &[(String, Vec<u8>)]) -> KeySet {
    let mut ks = KeySet::default();
    for (name, val) in entries {
        match (name.as_str(), val.len()) {
            ("header_key", 32) => {
                let mut k = [0u8; 32];
                k.copy_from_slice(val);
                ks.header_key = Some(k);
            }
            ("header_key_source", 32) => {
                let mut k = [0u8; 32];
                k.copy_from_slice(val);
                ks.header_key_source = Some(k);
            }
            ("header_kek_source", 16) => {
                let mut k = [0u8; 16];
                k.copy_from_slice(val);
                ks.header_kek_source = Some(k);
            }
            ("master_key_00", 16) => {
                let mut k = [0u8; 16];
                k.copy_from_slice(val);
                ks.master_key_00 = Some(k);
            }
            ("aes_kek_generation_source", 16) => {
                let mut k = [0u8; 16];
                k.copy_from_slice(val);
                ks.aes_kek_generation_source = Some(k);
            }
            ("aes_key_generation_source", 16) => {
                let mut k = [0u8; 16];
                k.copy_from_slice(val);
                ks.aes_key_generation_source = Some(k);
            }
            _ => {
                if val.len() == 16 {
                    if let Some((table, gen)) = key_area_table_and_generation(&mut ks, name) {
                        let mut k = [0u8; 16];
                        k.copy_from_slice(val);
                        table[gen] = Some(k);
                    }
                }
            }
        }
    }
    ks
}

/// Match `key_area_key_<application|ocean|system>_<XX>` or `titlekek_<XX>`
/// and return the matching table slot and generation index, if `name` fits
/// one of those shapes.
fn key_area_table_and_generation<'a>(
    ks: &'a mut KeySet,
    name: &str,
) -> Option<(&'a mut [Option<[u8; 16]>; KEY_GENERATION_COUNT], usize)> {
    let suffix = name.strip_prefix("key_area_key_application_").map(|s| (s, 0));
    let suffix = suffix.or_else(|| name.strip_prefix("key_area_key_ocean_").map(|s| (s, 1)));
    let suffix = suffix.or_else(|| name.strip_prefix("key_area_key_system_").map(|s| (s, 2)));
    let suffix = suffix.or_else(|| name.strip_prefix("titlekek_").map(|s| (s, 3)));
    let (gen_hex, kind) = suffix?;
    let gen = usize::from_str_radix(gen_hex, 16).ok()?;
    if gen >= KEY_GENERATION_COUNT {
        return None;
    }
    let table = match kind {
        0 => &mut ks.key_area_key_application,
        1 => &mut ks.key_area_key_ocean,
        2 => &mut ks.key_area_key_system,
        _ => &mut ks.titlekek,
    };
    Some((table, gen))
}

/// Build a [`KeySet`] title-key list from parsed `title.keys` entries. Keys
/// are either `titlekey_<rights_id> = hex` or `rights_id = hex` (16-byte),
/// and each value is still `titlekek`-wrapped — the file stores a ticket's
/// key block as-is. Assign the result to [`KeySet::title_keys`], which is
/// where that wrapping is accounted for.
pub fn keyset_from_title(entries: &[(String, Vec<u8>)]) -> Vec<([u8; 16], [u8; 16])> {
    let mut out = Vec::new();
    for (name, val) in entries {
        if val.len() != 16 {
            continue;
        }
        let id_hex: String = name
            .strip_prefix("titlekey_")
            .map(|s| s.to_string())
            .unwrap_or_else(|| name.clone());
        let id_hex = id_hex.replace([' ', '_', '-'], "");
        if id_hex.len() != 32 {
            continue;
        }
        let mut id = [0u8; 16];
        let mut ok = true;
        for i in 0..16 {
            match u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16) {
                Ok(b) => id[i] = b,
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let mut key = [0u8; 16];
        key.copy_from_slice(val);
        out.push((id, key));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prod_keys() {
        let text = "# comment\nheader_key = 0x00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff\nmaster_key_00 = 0102030405060708090a0b0c0d0e0f10\n\nbad line\n";
        let entries = parse_keys_file(text);
        assert_eq!(entries.len(), 2);
        let ks = keyset_from_prod(&entries);
        assert_eq!(ks.header_key.unwrap()[0], 0x00);
        assert_eq!(ks.master_key_00.unwrap()[15], 0x10);
    }

    #[test]
    fn parses_key_area_keys_by_generation() {
        let text = "key_area_key_application_00 = 00000000000000000000000000000000\n\
                     key_area_key_application_01 = 11111111111111111111111111111111\n\
                     key_area_key_ocean_1f = 22222222222222222222222222222222\n\
                     key_area_key_system_05 = 33333333333333333333333333333333\n";
        let entries = parse_keys_file(text);
        let ks = keyset_from_prod(&entries);
        assert_eq!(
            ks.key_area_key(KeyAreaKind::Application, 0),
            Some([0u8; 16])
        );
        assert_eq!(
            ks.key_area_key(KeyAreaKind::Application, 1),
            Some([0x11u8; 16])
        );
        assert_eq!(ks.key_area_key(KeyAreaKind::Ocean, 0x1f), Some([0x22u8; 16]));
        assert_eq!(ks.key_area_key(KeyAreaKind::System, 5), Some([0x33u8; 16]));
        // Unset generations and the wrong kind both miss.
        assert_eq!(ks.key_area_key(KeyAreaKind::Application, 2), None);
        assert_eq!(ks.key_area_key(KeyAreaKind::System, 0), None);
        // Out-of-range generation index doesn't panic.
        assert_eq!(ks.key_area_key(KeyAreaKind::Application, 0xff), None);
    }

    #[test]
    fn derives_header_key_from_sources() {
        // Constructed sources: header_kek = AESECBDecrypt(header_kek_source,
        // ...), so pick a header_key_source and check the derivation runs and
        // reproduces a direct key when the sources are consistent.
        let mut ks = KeySet::default();
        let mut direct = [0u8; 32];
        for i in 0..32 {
            direct[i] = i as u8;
        }
        ks.header_key = Some(direct);
        assert_eq!(ks.effective_header_key(), Some(direct));
        // Without a direct key and without sources → None.
        let ks2 = KeySet::default();
        assert_eq!(ks2.effective_header_key(), None);
    }

    #[test]
    fn parses_title_keys() {
        let text = "titlekey_010075600ae968000000000000000005 = 0102030405060708090a0b0c0d0e0f10\n";
        let entries = parse_keys_file(text);
        let tks = keyset_from_title(&entries);
        assert_eq!(tks.len(), 1);
        assert_eq!(tks[0].0[0], 0x01);
        assert_eq!(tks[0].1[15], 0x10);
    }

    /// A `title.keys` entry is the ticket's key block, still wrapped: using
    /// it as-is is what made a real title fail its section hash check with a
    /// perfectly good key file.
    #[test]
    fn unwraps_a_title_keys_entry_with_the_titlekek() {
        let rights_id = [0xaau8; 16];
        let plain = [0x11u8; 16];
        let kek = [0x22u8; 16];
        let mut ks = KeySet::default();
        ks.titlekek[0x0d] = Some(kek);
        ks.title_keys = vec![(rights_id, crate::crypto::aes128_encrypt_block(&kek, &plain))];
        assert_eq!(ks.title_key(&rights_id, 0x0d), Some(plain));
        // The stored form is not the usable key, and a generation with no
        // titlekek can't produce one rather than producing the wrong one.
        assert_ne!(ks.wrapped_title_key(&rights_id), Some(plain));
        assert_eq!(ks.title_key(&rights_id, 0x0c), None);
        assert_eq!(ks.title_key(&[0xbbu8; 16], 0x0d), None);
    }

    /// The ticket shipped with the content describes that content; a
    /// `title.keys` entry for the same title is a guess from elsewhere.
    #[test]
    fn a_ticket_key_replaces_a_title_keys_entry() {
        let rights_id = [0xaau8; 16];
        let kek = [0x22u8; 16];
        let from_ticket = [0x33u8; 16];
        let mut ks = KeySet::default();
        ks.titlekek[0x0d] = Some(kek);
        ks.title_keys = vec![(rights_id, [0x44u8; 16])];
        ks.add_title_key(rights_id, crate::crypto::aes128_encrypt_block(&kek, &from_ticket));
        assert_eq!(ks.title_key(&rights_id, 0x0d), Some(from_ticket));
        // Recording the same title twice replaces it instead of stacking a
        // second entry the first would shadow forever.
        ks.add_title_key(rights_id, [0x55u8; 16]);
        assert_eq!(ks.title_keys.len(), 1);
        assert_eq!(ks.wrapped_title_key(&rights_id), Some([0x55u8; 16]));
    }
}
