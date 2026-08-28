//! ES ticket (`.tik`) parsing and title-key extraction.
//!
//! A ticket carries the AES-128 key that decrypts a title-key-crypto NCA's
//! sections (a nonzero `rights_id` in the NCA header selects this path
//! instead of the header's own key area). Scene NSP releases bundle the
//! ticket right next to the content, so the title key is available without a
//! separate personal `title.keys` dump — as long as the ticket uses "Common"
//! crypto (the overwhelming majority do), unwrapping it only needs a public
//! `titlekek_XX` from `prod.keys`.
//!
//! What a ticket yields here is the key block *still wrapped*, exactly as
//! `title.keys` stores it. Which `titlekek_XX` unwraps it is the NCA's key
//! generation, not anything the ticket says — so unwrapping belongs to
//! [`crate::nca::Nca::section_key`], which is the only place that knows it.
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
//!              the AES-128-ECB-wrapped title key; Personalized crypto
//!              RSA-wraps it instead, which needs a console's ETicket key —
//!              out of scope here)
//! body + 0x140 format_version (u8)
//! body + 0x141 titlekey_type (u8): 0 = Common, 1 = Personalized
//! body + 0x142 ticket_version (u16)
//! body + 0x144 license_type (u8)
//! body + 0x145 common_key_id (u8)
//! body + 0x160 rights_id [0x10]
//! ```

use crate::keys::KeySet;
use crate::source::ByteSource;
use crate::Error;

const BODY_OFFSET_TITLEKEY_BLOCK: usize = 0x40;
const BODY_OFFSET_TITLEKEY_TYPE: usize = 0x141;
const BODY_OFFSET_COMMON_KEY_ID: usize = 0x145;
const BODY_OFFSET_RIGHTS_ID: usize = 0x160;
const BODY_MIN_SIZE: usize = BODY_OFFSET_RIGHTS_ID + 0x10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket {
    pub rights_id: [u8; 16],
    /// 0 = Common (a public `titlekek` unwraps it), 1 = Personalized (needs
    /// a console's ETicket RSA key — not supported).
    pub titlekey_type: u8,
    /// The ticket's own idea of which `titlekek_XX` its key block belongs to.
    /// Reported for diagnostics only: a retail ticket may leave this 0 while
    /// the content it unlocks needs a much later generation (Asphalt 9's
    /// reads 0 against a `titlekek_07` title), so nothing selects a key with
    /// it. The NCA's key generation is what does.
    pub common_key_id: u8,
    /// First 16 bytes of the title-key block: the title key, still
    /// AES-128-ECB-wrapped under a `titlekek` (Common crypto only).
    pub wrapped_title_key: [u8; 16],
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

        let mut wrapped_title_key = [0u8; 16];
        wrapped_title_key.copy_from_slice(
            &data[body + BODY_OFFSET_TITLEKEY_BLOCK..body + BODY_OFFSET_TITLEKEY_BLOCK + 16],
        );
        let titlekey_type = data[body + BODY_OFFSET_TITLEKEY_TYPE];
        let common_key_id = data[body + BODY_OFFSET_COMMON_KEY_ID];
        let mut rights_id = [0u8; 16];
        rights_id.copy_from_slice(
            &data[body + BODY_OFFSET_RIGHTS_ID..body + BODY_OFFSET_RIGHTS_ID + 16],
        );

        Ok(Ticket {
            rights_id,
            titlekey_type,
            common_key_id,
            wrapped_title_key,
        })
    }

    /// The still-`titlekek`-wrapped title key, for the keyset to hold
    /// alongside `title.keys`' entries — Common crypto only.
    pub fn title_key_block(&self) -> Result<[u8; 16], Error> {
        if self.titlekey_type != 0 {
            return Err(Error::Ticket(
                "personalized ticket — its title key is RSA-wrapped with a console's ETicket key, which this emulator doesn't have".into(),
            ));
        }
        Ok(self.wrapped_title_key)
    }
}

/// Find `<rights_id-hex>.tik` among a container's files (scene releases
/// bundle the ticket right next to the content it unlocks) and return its
/// still-wrapped title key.
pub fn find_wrapped_title_key<S: ByteSource>(
    rights_id: &[u8; 16],
    files: &[crate::nsp::Pfs0File],
    src: &S,
) -> Result<[u8; 16], Error> {
    let want_name = format!("{}.tik", hex_lower(rights_id));
    let f = files
        .iter()
        .find(|f| f.name.eq_ignore_ascii_case(&want_name))
        .ok_or_else(|| Error::Ticket(format!("no {} in this NSP", want_name)))?;
    let end = f.offset.saturating_add(f.size);
    if end > src.len() {
        return Err(Error::Ticket("ticket entry exceeds the NSP".into()));
    }
    // A ticket is a signature block plus a 0x2c0-byte body; the largest
    // signature type puts the end of the body at 0x3b0. Anything a container
    // claims past that is padding, and reading it would only be a way for a
    // malformed entry to ask for an arbitrary allocation.
    const MAX_TICKET: u64 = 0x1000;
    let data = src.read_vec(f.offset, f.size.min(MAX_TICKET))?;
    Ticket::parse(&data)?.title_key_block()
}

/// Give `keys` the title key `nca` needs, from a ticket bundled in the same
/// container. Answers whether one was adopted: `false` when the NCA doesn't
/// use title-key crypto at all, or when the container has no ticket but the
/// keyset already carries this title's key from elsewhere.
///
/// The container's own ticket wins over a `title.keys` entry for the same
/// rights id: it ships with the content it unlocks, where `title.keys` is a
/// dump that may be for a different revision of it.
pub fn load_bundled_title_key<S: ByteSource>(
    keys: &mut KeySet,
    nca: &crate::nca::Nca,
    files: &[crate::nsp::Pfs0File],
    src: &S,
) -> Result<bool, Error> {
    if !nca.has_rights_id() {
        return Ok(false);
    }
    match find_wrapped_title_key(&nca.rights_id, files, src) {
        Ok(wrapped) => {
            keys.add_title_key(nca.rights_id, wrapped);
            Ok(true)
        }
        // No ticket here is only a problem when nothing else has the key.
        Err(_) if keys.has_title_key(&nca.rights_id) => Ok(false),
        Err(e) => Err(e),
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SliceSource;

    fn build_ticket(
        rights_id: [u8; 16],
        titlekey_type: u8,
        common_key_id: u8,
        wrapped_key: [u8; 16],
    ) -> Vec<u8> {
        let mut data = vec![0u8; 0x140 + 0x170];
        data[0..4].copy_from_slice(&0x010004u32.to_le_bytes()); // RSA-2048 SHA-256
        let body = 0x140;
        data[body + BODY_OFFSET_TITLEKEY_BLOCK..body + BODY_OFFSET_TITLEKEY_BLOCK + 16]
            .copy_from_slice(&wrapped_key);
        data[body + BODY_OFFSET_TITLEKEY_TYPE] = titlekey_type;
        data[body + BODY_OFFSET_COMMON_KEY_ID] = common_key_id;
        data[body + BODY_OFFSET_RIGHTS_ID..body + BODY_OFFSET_RIGHTS_ID + 16]
            .copy_from_slice(&rights_id);
        data
    }

    #[test]
    fn parses_a_real_shaped_rsa2048_ticket() {
        // Byte layout cross-checked against a real "A Short Hike" ticket:
        // rights_id/titlekey_type/common_key_id all land where this parser
        // expects for signature type 0x010004.
        let rights_id = *b"\x01\x00\x48\x90\x11\x7b\x20\x00\x00\x00\x00\x00\x00\x00\x00\x0b";
        let wrapped = [0xAAu8; 16];
        let data = build_ticket(rights_id, 0, 0x0b, wrapped);
        let tik = Ticket::parse(&data).unwrap();
        assert_eq!(tik.rights_id, rights_id);
        assert_eq!(tik.titlekey_type, 0);
        assert_eq!(tik.common_key_id, 0x0b);
        assert_eq!(tik.wrapped_title_key, wrapped);
    }

    /// Asphalt 9's ticket says `common_key_id` 0 and its content needs
    /// `titlekek_07`. A ticket hands over the key block untouched precisely
    /// so that field can never pick the generation.
    #[test]
    fn a_ticket_hands_over_the_key_block_unwrapped_by_itself() {
        let wrapped = [0x22u8; 16];
        for common_key_id in [0u8, 0x07, 0x0b] {
            let data = build_ticket([0u8; 16], 0, common_key_id, wrapped);
            assert_eq!(
                Ticket::parse(&data).unwrap().title_key_block().unwrap(),
                wrapped
            );
        }
    }

    #[test]
    fn rejects_personalized_crypto() {
        let data = build_ticket([0u8; 16], 1, 0, [0u8; 16]);
        let tik = Ticket::parse(&data).unwrap();
        assert!(matches!(tik.title_key_block(), Err(Error::Ticket(_))));
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

    /// Lay out a fake NSP buffer: some padding, then the ticket.
    fn nsp_with_ticket(
        tik_bytes: &[u8],
        rights_id: &[u8; 16],
    ) -> (Vec<u8>, Vec<crate::nsp::Pfs0File>) {
        let tik_offset = 0x1000;
        let mut nsp_data = vec![0u8; tik_offset + tik_bytes.len()];
        nsp_data[tik_offset..].copy_from_slice(tik_bytes);
        let files = vec![crate::nsp::Pfs0File {
            offset: tik_offset as u64,
            size: tik_bytes.len() as u64,
            name: format!("{}.tik", hex_lower(rights_id)),
        }];
        (nsp_data, files)
    }

    #[test]
    fn finds_a_ticket_from_an_nsp_file_list() {
        let rights_id = [
            0x01u8, 0x00, 0x48, 0x90, 0x11, 0x7b, 0x20, 0, 0, 0, 0, 0, 0, 0, 0, 0x0b,
        ];
        let wrapped = [0x88u8; 16];
        let (nsp_data, files) =
            nsp_with_ticket(&build_ticket(rights_id, 0, 0x09, wrapped), &rights_id);

        let src = SliceSource(&nsp_data);
        assert_eq!(
            find_wrapped_title_key(&rights_id, &files, &src).unwrap(),
            wrapped
        );
        // A rights id with no matching ticket file reports a clear error.
        assert!(find_wrapped_title_key(&[0xffu8; 16], &files, &src).is_err());
    }

    /// An NCA that carries nothing but the two fields this path reads.
    fn rights_id_nca(rights_id: [u8; 16], key_generation: u8) -> crate::nca::Nca {
        crate::nca::Nca {
            distribution_type: 0,
            content_type: crate::nca::ContentType::Program,
            content_type_raw: 0,
            title_id: 0,
            sdk_version: 0,
            crypto_type: 0,
            sections: Vec::new(),
            file_size: 0,
            program_id: 0,
            rights_id,
            key_index: 0,
            key_generation_old: 0,
            key_generation_new: key_generation,
            encrypted_key_area: [0; 0x40],
            fs_headers: Default::default(),
        }
    }

    /// The whole point of loading a ticket: the key it yields is unwrapped
    /// with the *NCA's* generation, and the section key comes out right even
    /// though the ticket's own `common_key_id` names a different one.
    #[test]
    fn a_bundled_ticket_unwraps_with_the_ncas_generation() {
        let rights_id = [0x33u8; 16];
        let kek = [0x44u8; 16];
        let title_key = [0x55u8; 16];
        let wrapped = crate::crypto::aes128_encrypt_block(&kek, &title_key);
        // `common_key_id` 0 against content whose key generation is 8 —
        // Asphalt 9's shape exactly.
        let (nsp_data, files) =
            nsp_with_ticket(&build_ticket(rights_id, 0, 0, wrapped), &rights_id);
        let nca = rights_id_nca(rights_id, 8);

        let mut keys = KeySet::default();
        keys.titlekek[0x07] = Some(kek);
        assert!(load_bundled_title_key(&mut keys, &nca, &files, &SliceSource(&nsp_data)).unwrap());
        assert_eq!(nca.section_key(&keys).unwrap(), title_key);
    }

    /// A container with no ticket is only a failure when nothing else has
    /// the key — and a bundled ticket replaces a `title.keys` entry rather
    /// than stacking behind it.
    #[test]
    fn a_missing_ticket_defers_to_the_keyset() {
        let rights_id = [0x33u8; 16];
        let nca = rights_id_nca(rights_id, 1);

        let mut keys = KeySet::default();
        assert!(load_bundled_title_key(&mut keys, &nca, &[], &SliceSource(&[])).is_err());
        keys.add_title_key(rights_id, [0x66u8; 16]);
        assert!(!load_bundled_title_key(&mut keys, &nca, &[], &SliceSource(&[])).unwrap());

        let wrapped = [0x77u8; 16];
        let (nsp_data, files) =
            nsp_with_ticket(&build_ticket(rights_id, 0, 0, wrapped), &rights_id);
        assert!(load_bundled_title_key(&mut keys, &nca, &files, &SliceSource(&nsp_data)).unwrap());
        assert_eq!(keys.wrapped_title_key(&rights_id), Some(wrapped));
        assert_eq!(keys.title_keys.len(), 1);
    }
}
