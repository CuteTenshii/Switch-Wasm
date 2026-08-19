//! ES ticket (`.tik`) parsing and title-key decryption.
//!
//! A ticket carries the AES-128 key that decrypts a title-key-crypto NCA's
//! sections (a nonzero `rights_id` in the NCA header selects this path
//! instead of the header's own key area). Scene NSP releases bundle the
//! ticket right next to the content, so the title key is available without a
//! separate personal `title.keys` dump — as long as the ticket uses "Common"
//! crypto (the overwhelming majority do), decrypting it only needs a public
//! `titlekek_XX` from `prod.keys`.
//!
//! Layout: a fixed-size signature block (size depends on the signature type)
//! followed by the ticket body. Offsets below are relative to the body, and
//! were cross-checked against a real ticket's bytes field-by-field (magic,
//! rights_id, titlekey_type, common_key_id all landed exactly where expected
//! for a `RSA2048_SHA256` ticket):
//!
//! ```text
//! body + 0x00  issuer [0x40]
//! body + 0x40  title key block [0x100] (Common crypto: first 16 bytes are
//!              the AES-128-ECB-encrypted title key; Personalized crypto
//!              RSA-wraps it instead, which needs a console's ETicket key —
//!              out of scope here)
//! body + 0x140 format_version (u8)
//! body + 0x141 titlekey_type (u8): 0 = Common, 1 = Personalized
//! body + 0x142 ticket_version (u16)
//! body + 0x144 license_type (u8)
//! body + 0x145 common_key_id (u8): which `titlekek_XX` decrypts the key block
//! body + 0x160 rights_id [0x10]
//! ```

use crate::crypto::aes128_decrypt_block;
use crate::keys::KeySet;
use crate::Error;

const BODY_OFFSET_TITLEKEY_BLOCK: usize = 0x40;
const BODY_OFFSET_TITLEKEY_TYPE: usize = 0x141;
const BODY_OFFSET_COMMON_KEY_ID: usize = 0x145;
const BODY_OFFSET_RIGHTS_ID: usize = 0x160;
const BODY_MIN_SIZE: usize = BODY_OFFSET_RIGHTS_ID + 0x10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket {
    pub rights_id: [u8; 16],
    /// 0 = Common (a public `titlekek` decrypts it), 1 = Personalized (needs
    /// a console's ETicket RSA key — not supported).
    pub titlekey_type: u8,
    /// Selects `titlekek_<common_key_id>`.
    pub common_key_id: u8,
    /// First 16 bytes of the title-key block: the AES-128-ECB-encrypted
    /// title key (Common crypto only).
    pub encrypted_title_key: [u8; 16],
}

/// Body offset for each ES signature type (signature size, then padded to a
/// fixed alignment): ECDSA at 0x80, RSA-2048 at 0x140, RSA-4096 at 0x240.
fn body_offset(sig_type: u32) -> Option<usize> {
    match sig_type {
        0x010000 | 0x010003 => Some(0x240), // RSA-4096, SHA-1 / SHA-256
        0x010001 | 0x010004 => Some(0x140), // RSA-2048, SHA-1 / SHA-256
        0x010002 | 0x010005 => Some(0x80),  // ECDSA, SHA-1 / SHA-256
        _ => None,
    }
}

impl Ticket {
    pub fn parse(data: &[u8]) -> Result<Ticket, Error> {
        if data.len() < 4 {
            return Err(Error::Truncated {
                what: "ticket signature type".into(),
                expected: 4,
                got: data.len(),
            });
        }
        let sig_type = crate::nsp::read_u32(data, 0);
        let body = body_offset(sig_type)
            .ok_or_else(|| Error::Ticket(format!("unknown signature type {:#x}", sig_type)))?;
        let need = body + BODY_MIN_SIZE;
        if data.len() < need {
            return Err(Error::Truncated {
                what: "ticket body".into(),
                expected: need,
                got: data.len(),
            });
        }

        let mut encrypted_title_key = [0u8; 16];
        encrypted_title_key.copy_from_slice(&data[body + BODY_OFFSET_TITLEKEY_BLOCK..body + BODY_OFFSET_TITLEKEY_BLOCK + 16]);
        let titlekey_type = data[body + BODY_OFFSET_TITLEKEY_TYPE];
        let common_key_id = data[body + BODY_OFFSET_COMMON_KEY_ID];
        let mut rights_id = [0u8; 16];
        rights_id.copy_from_slice(&data[body + BODY_OFFSET_RIGHTS_ID..body + BODY_OFFSET_RIGHTS_ID + 16]);

        Ok(Ticket {
            rights_id,
            titlekey_type,
            common_key_id,
            encrypted_title_key,
        })
    }

    /// The master-key revision `common_key_id` selects: like the NCA header's
    /// key-area generation, the stored value is one more than the actual
    /// `titlekek_XX` index (confirmed against a real ticket — `common_key_id`
    /// 0x0b decrypts with `titlekek_0a`, not `titlekek_0b`), except that 0
    /// stays 0.
    fn master_key_revision(&self) -> u8 {
        self.common_key_id.saturating_sub(1)
    }

    /// Decrypt the title key (Common crypto only).
    pub fn decrypt_title_key(&self, keys: &KeySet) -> Result<[u8; 16], Error> {
        if self.titlekey_type != 0 {
            return Err(Error::Ticket(
                "personalized ticket — its title key is RSA-wrapped with a console's ETicket key, which this emulator doesn't have".into(),
            ));
        }
        let generation = self.master_key_revision();
        let kek = keys.titlekek(generation).ok_or_else(|| {
            Error::Ticket(format!("missing titlekek_{:02x} in prod.keys", generation))
        })?;
        Ok(aes128_decrypt_block(&kek, &self.encrypted_title_key))
    }
}

/// Find `<rights_id-hex>.tik` among an NSP's files (scene releases bundle the
/// ticket right next to the content it unlocks) and decrypt its title key.
/// `nsp_data` is the full NSP buffer; `files` its parsed PFS0 file table.
pub fn find_and_decrypt_title_key(
    rights_id: &[u8; 16],
    files: &[crate::nsp::Pfs0File],
    nsp_data: &[u8],
    keys: &KeySet,
) -> Result<[u8; 16], Error> {
    let want_name = format!("{}.tik", hex_lower(rights_id));
    let f = files
        .iter()
        .find(|f| f.name.eq_ignore_ascii_case(&want_name))
        .ok_or_else(|| Error::Ticket(format!("no {} in this NSP", want_name)))?;
    let start = f.offset as usize;
    let end = start
        .checked_add(f.size as usize)
        .filter(|&e| e <= nsp_data.len())
        .ok_or_else(|| Error::Ticket("ticket entry exceeds the NSP".into()))?;
    Ticket::parse(&nsp_data[start..end])?.decrypt_title_key(keys)
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_ticket(rights_id: [u8; 16], titlekey_type: u8, common_key_id: u8, encrypted_key: [u8; 16]) -> Vec<u8> {
        let mut data = vec![0u8; 0x140 + 0x170];
        data[0..4].copy_from_slice(&0x010004u32.to_le_bytes()); // RSA-2048 SHA-256
        let body = 0x140;
        data[body + BODY_OFFSET_TITLEKEY_BLOCK..body + BODY_OFFSET_TITLEKEY_BLOCK + 16]
            .copy_from_slice(&encrypted_key);
        data[body + BODY_OFFSET_TITLEKEY_TYPE] = titlekey_type;
        data[body + BODY_OFFSET_COMMON_KEY_ID] = common_key_id;
        data[body + BODY_OFFSET_RIGHTS_ID..body + BODY_OFFSET_RIGHTS_ID + 16].copy_from_slice(&rights_id);
        data
    }

    #[test]
    fn parses_a_real_shaped_rsa2048_ticket() {
        // Byte layout cross-checked against a real "A Short Hike" ticket:
        // rights_id/titlekey_type/common_key_id all land where this parser
        // expects for signature type 0x010004.
        let rights_id = *b"\x01\x00\x48\x90\x11\x7b\x20\x00\x00\x00\x00\x00\x00\x00\x00\x0b";
        let enc_key = [0xAAu8; 16];
        let data = build_ticket(rights_id, 0, 0x0b, enc_key);
        let tik = Ticket::parse(&data).unwrap();
        assert_eq!(tik.rights_id, rights_id);
        assert_eq!(tik.titlekey_type, 0);
        assert_eq!(tik.common_key_id, 0x0b);
        assert_eq!(tik.encrypted_title_key, enc_key);
    }

    #[test]
    fn decrypts_common_crypto_title_key() {
        let kek = [0x11u8; 16];
        let title_key = [0x22u8; 16];
        let encrypted = crate::crypto::aes128_encrypt_block(&kek, &title_key);
        let data = build_ticket([0u8; 16], 0, 0x05, encrypted);
        let tik = Ticket::parse(&data).unwrap();

        // common_key_id 0x05 selects titlekek_04 (the stored value is one
        // more than the actual generation, except 0 stays 0).
        let mut keys = KeySet::default();
        keys.titlekek[0x04] = Some(kek);
        assert_eq!(tik.decrypt_title_key(&keys).unwrap(), title_key);
    }

    #[test]
    fn common_key_id_zero_stays_zero() {
        let kek = [0x33u8; 16];
        let title_key = [0x44u8; 16];
        let encrypted = crate::crypto::aes128_encrypt_block(&kek, &title_key);
        let data = build_ticket([0u8; 16], 0, 0, encrypted);
        let tik = Ticket::parse(&data).unwrap();

        let mut keys = KeySet::default();
        keys.titlekek[0] = Some(kek);
        assert_eq!(tik.decrypt_title_key(&keys).unwrap(), title_key);
    }

    #[test]
    fn rejects_personalized_crypto() {
        let data = build_ticket([0u8; 16], 1, 0, [0u8; 16]);
        let tik = Ticket::parse(&data).unwrap();
        assert!(matches!(tik.decrypt_title_key(&KeySet::default()), Err(Error::Ticket(_))));
    }

    #[test]
    fn reports_missing_titlekek() {
        let data = build_ticket([0u8; 16], 0, 0x1f, [0u8; 16]);
        let tik = Ticket::parse(&data).unwrap();
        assert!(matches!(tik.decrypt_title_key(&KeySet::default()), Err(Error::Ticket(_))));
    }

    #[test]
    fn rejects_unknown_signature_type() {
        let mut data = vec![0u8; 0x300];
        data[0..4].copy_from_slice(&0xdeadbeefu32.to_le_bytes());
        assert!(matches!(Ticket::parse(&data), Err(Error::Ticket(_))));
    }

    #[test]
    fn rejects_truncated_ticket() {
        assert!(matches!(Ticket::parse(&[0u8; 8]), Err(Error::Ticket(_))));
        // A valid signature type but a body too short to hold rights_id.
        let mut data = vec![0u8; 0x140 + 0x10];
        data[0..4].copy_from_slice(&0x010004u32.to_le_bytes());
        assert!(matches!(Ticket::parse(&data), Err(Error::Truncated { .. })));
    }

    #[test]
    fn finds_and_decrypts_ticket_from_nsp_file_list() {
        let rights_id = [0x01u8, 0x00, 0x48, 0x90, 0x11, 0x7b, 0x20, 0, 0, 0, 0, 0, 0, 0, 0, 0x0b];
        let kek = [0x77u8; 16];
        let title_key = [0x88u8; 16];
        let encrypted = crate::crypto::aes128_encrypt_block(&kek, &title_key);
        let tik_bytes = build_ticket(rights_id, 0, 0x09, encrypted);
        // Lay out a fake NSP buffer: some padding, then the ticket.
        let tik_offset = 0x1000;
        let mut nsp_data = vec![0u8; tik_offset + tik_bytes.len()];
        nsp_data[tik_offset..].copy_from_slice(&tik_bytes);
        let files = vec![crate::nsp::Pfs0File {
            offset: tik_offset as u64,
            size: tik_bytes.len() as u64,
            name: format!("{}.tik", hex_lower(&rights_id)),
        }];

        let mut keys = KeySet::default();
        keys.titlekek[0x08] = Some(kek); // common_key_id 0x09 -> titlekek_08
        let resolved = find_and_decrypt_title_key(&rights_id, &files, &nsp_data, &keys).unwrap();
        assert_eq!(resolved, title_key);

        // A rights id with no matching ticket file reports a clear error.
        assert!(find_and_decrypt_title_key(&[0xffu8; 16], &files, &nsp_data, &keys).is_err());
    }
}
