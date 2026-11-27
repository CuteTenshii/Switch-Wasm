//! Key-file parsing (`prod.keys` / `title.keys`) and NCA header key derivation.

use crate::crypto::aes128_ecb_decrypt;

pub const KEY_GENERATION_COUNT: usize = 0x20;

/// Key area key family, selected by the NCA header's key index byte.
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

#[derive(Debug, Default, Clone)]
pub struct KeySet {
    pub header_key: Option<[u8; 32]>,
    /// `titlekek`-wrapped title keys by rights id.
    pub title_keys: Vec<([u8; 16], [u8; 16])>,
    // Sources for deriving the header key (prod.keys).
    pub header_key_source: Option<[u8; 32]>,
    pub header_kek_source: Option<[u8; 16]>,
    pub master_key_00: Option<[u8; 16]>,
    pub aes_kek_generation_source: Option<[u8; 16]>,
    pub aes_key_generation_source: Option<[u8; 16]>,
    pub key_area_key_application: [Option<[u8; 16]>; KEY_GENERATION_COUNT],
    pub key_area_key_ocean: [Option<[u8; 16]>; KEY_GENERATION_COUNT],
    pub key_area_key_system: [Option<[u8; 16]>; KEY_GENERATION_COUNT],
    pub titlekek: [Option<[u8; 16]>; KEY_GENERATION_COUNT],
}

impl KeySet {
    pub fn wrapped_title_key(&self, rights_id: &[u8; 16]) -> Option<[u8; 16]> {
        find_key(&self.title_keys, rights_id)
    }

    pub fn has_title_key(&self, rights_id: &[u8; 16]) -> bool {
        self.wrapped_title_key(rights_id).is_some()
    }

    /// Title key for `rights_id`, unwrapped with the NCA generation's `titlekek`.
    pub fn title_key(&self, rights_id: &[u8; 16], generation: u8) -> Option<[u8; 16]> {
        let wrapped = self.wrapped_title_key(rights_id)?;
        let kek = self.titlekek(generation)?;
        Some(crate::crypto::aes128_decrypt_block(&kek, &wrapped))
    }

    pub fn add_title_key(&mut self, rights_id: [u8; 16], wrapped: [u8; 16]) {
        match self.title_keys.iter_mut().find(|(id, _)| *id == rights_id) {
            Some(slot) => slot.1 = wrapped,
            None => self.title_keys.push((rights_id, wrapped)),
        }
    }

    pub fn key_area_key(&self, kind: KeyAreaKind, generation: u8) -> Option<[u8; 16]> {
        let table = match kind {
            KeyAreaKind::Application => &self.key_area_key_application,
            KeyAreaKind::Ocean => &self.key_area_key_ocean,
            KeyAreaKind::System => &self.key_area_key_system,
        };
        table.get(generation as usize).copied().flatten()
    }

    pub fn titlekek(&self, generation: u8) -> Option<[u8; 16]> {
        self.titlekek.get(generation as usize).copied().flatten()
    }

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

/// Parse `name = hex` lines; `#` comments and blank lines are ignored.
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

fn key_area_table_and_generation<'a>(
    ks: &'a mut KeySet,
    name: &str,
) -> Option<(&'a mut [Option<[u8; 16]>; KEY_GENERATION_COUNT], usize)> {
    let suffix = name
        .strip_prefix("key_area_key_application_")
        .map(|s| (s, 0));
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

/// Parse `title.keys` entries into (rights id, wrapped key) pairs.
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
        let mut header_key = [0u8; 32];
        for (i, b) in header_key.iter_mut().enumerate() {
            *b = (i as u8 % 16) * 0x11;
        }
        assert_eq!(ks.header_key, Some(header_key));
        let master: [u8; 16] = std::array::from_fn(|i| i as u8 + 1);
        assert_eq!(ks.master_key_00, Some(master));
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
        assert_eq!(
            ks.key_area_key(KeyAreaKind::Ocean, 0x1f),
            Some([0x22u8; 16])
        );
        assert_eq!(ks.key_area_key(KeyAreaKind::System, 5), Some([0x33u8; 16]));
        assert_eq!(ks.key_area_key(KeyAreaKind::Application, 2), None);
        assert_eq!(ks.key_area_key(KeyAreaKind::System, 0), None);
        assert_eq!(ks.key_area_key(KeyAreaKind::Application, 0xff), None);
    }

    #[test]
    fn a_direct_header_key_is_used_as_is_and_none_is_made_up_without_one() {
        let mut ks = KeySet::default();
        let mut direct = [0u8; 32];
        for (i, b) in direct.iter_mut().enumerate() {
            *b = i as u8;
        }
        ks.header_key = Some(direct);
        assert_eq!(ks.effective_header_key(), Some(direct));
        let ks2 = KeySet::default();
        assert_eq!(ks2.effective_header_key(), None);
    }

    #[test]
    fn parses_title_keys() {
        let text = "titlekey_010075600ae968000000000000000005 = 0102030405060708090a0b0c0d0e0f10\n";
        let entries = parse_keys_file(text);
        let tks = keyset_from_title(&entries);
        let rights_id = [
            0x01, 0x00, 0x75, 0x60, 0x0a, 0xe9, 0x68, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x05,
        ];
        let key: [u8; 16] = std::array::from_fn(|i| i as u8 + 1);
        assert_eq!(tks, vec![(rights_id, key)]);
    }

    #[test]
    fn unwraps_a_title_keys_entry_with_the_titlekek() {
        let rights_id = [0xaau8; 16];
        let plain = [0x11u8; 16];
        let kek = [0x22u8; 16];
        let mut ks = KeySet::default();
        ks.titlekek[0x0d] = Some(kek);
        ks.title_keys = vec![(rights_id, crate::crypto::aes128_encrypt_block(&kek, &plain))];
        assert_eq!(ks.title_key(&rights_id, 0x0d), Some(plain));
        assert_ne!(ks.wrapped_title_key(&rights_id), Some(plain));
        assert_eq!(ks.title_key(&rights_id, 0x0c), None);
        assert_eq!(ks.title_key(&[0xbbu8; 16], 0x0d), None);
    }

    #[test]
    fn a_ticket_key_replaces_a_title_keys_entry() {
        let rights_id = [0xaau8; 16];
        let kek = [0x22u8; 16];
        let from_ticket = [0x33u8; 16];
        let mut ks = KeySet::default();
        ks.titlekek[0x0d] = Some(kek);
        ks.title_keys = vec![(rights_id, [0x44u8; 16])];
        ks.add_title_key(
            rights_id,
            crate::crypto::aes128_encrypt_block(&kek, &from_ticket),
        );
        assert_eq!(ks.title_key(&rights_id, 0x0d), Some(from_ticket));
        ks.add_title_key(rights_id, [0x55u8; 16]);
        assert_eq!(ks.title_keys.len(), 1);
        assert_eq!(ks.wrapped_title_key(&rights_id), Some([0x55u8; 16]));
    }
}
