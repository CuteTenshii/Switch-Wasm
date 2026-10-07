//! `mii`: the Mii database, empty but with the six built-in default Miis.

use crate::cpu::Cpu;
use crate::Result;

/// `nn::mii::CharInfo`: create id, nickname, then one byte per feature.
const MII_CHAR_INFO_LEN: usize = 0x58;

const MII_CREATE_ID_LEN: usize = 0x10;

/// Create id tags for default and `BuildRandom` Miis; distinct so they never collide.
const MII_DEFAULT_CREATE_ID_TAG: &[u8; MII_CREATE_ID_LEN] = b"switch-wasm mii\0";
const MII_RANDOM_CREATE_ID_TAG: &[u8; MII_CREATE_ID_LEN] = b"switch-wasm rnd\0";

const MII_CHAR_INFO_NICKNAME: usize = 0x10;

/// Features start past the nickname and its terminator.
const MII_CHAR_INFO_FEATURES: usize = 0x26;

/// Nickname length in UTF-16 units, excluding the terminator.
const MII_NICKNAME_LEN: usize = 10;

const MII_DEFAULT_NICKNAME: &str = "no name";

/// Module 126, description 1: index out of range.
const MII_INVALID_ARGUMENT: u32 = 126 | (1 << 9);

const MII_GENDER_ALL: u8 = 2;

/// 3DS/Wii U hair, eyebrow and beard colours widened to the Switch palette.
const MII_HAIR_COLORS: [u8; 8] = [8, 1, 2, 3, 4, 5, 6, 7];

const MII_EYE_COLORS: [u8; 6] = [8, 9, 10, 11, 12, 13];

/// Features that differ between the six default Miis.
struct DefaultMii {
    faceline_color: u8,
    hair_type: u8,
    hair_color: u8,
    eye_type: u8,
    eye_color: u8,
    eye_rotate: u8,
    eyebrow_type: u8,
    eyebrow_color: u8,
    /// 0 male, 1 female.
    gender: u8,
    favorite_color: u8,
}

/// The six default Miis, in `BuildDefault` order.
const DEFAULT_MIIS: [DefaultMii; 6] = [
    DefaultMii {
        faceline_color: 4,
        hair_type: 68,
        hair_color: 0,
        eye_type: 2,
        eye_color: 0,
        eye_rotate: 4,
        eyebrow_type: 6,
        eyebrow_color: 0,
        gender: 0,
        favorite_color: 4,
    },
    DefaultMii {
        faceline_color: 0,
        hair_type: 55,
        hair_color: 6,
        eye_type: 2,
        eye_color: 4,
        eye_rotate: 4,
        eyebrow_type: 6,
        eyebrow_color: 6,
        gender: 0,
        favorite_color: 5,
    },
    DefaultMii {
        faceline_color: 1,
        hair_type: 33,
        hair_color: 1,
        eye_type: 2,
        eye_color: 0,
        eye_rotate: 4,
        eyebrow_type: 6,
        eyebrow_color: 1,
        gender: 0,
        favorite_color: 0,
    },
    DefaultMii {
        faceline_color: 2,
        hair_type: 24,
        hair_color: 0,
        eye_type: 4,
        eye_color: 0,
        eye_rotate: 3,
        eyebrow_type: 0,
        eyebrow_color: 0,
        gender: 1,
        favorite_color: 2,
    },
    DefaultMii {
        faceline_color: 0,
        hair_type: 14,
        hair_color: 7,
        eye_type: 4,
        eye_color: 5,
        eye_rotate: 3,
        eyebrow_type: 0,
        eyebrow_color: 7,
        gender: 1,
        favorite_color: 6,
    },
    DefaultMii {
        faceline_color: 0,
        hair_type: 12,
        hair_color: 1,
        eye_type: 4,
        eye_color: 0,
        eye_rotate: 3,
        eyebrow_type: 0,
        eyebrow_color: 1,
        gender: 1,
        favorite_color: 7,
    },
];

/// The `index`th default Mii as a `CharInfo`; built, not read from the database.
fn default_mii_char_info(index: u32) -> Option<[u8; MII_CHAR_INFO_LEN]> {
    let mii = DEFAULT_MIIS.get(index as usize)?;
    let mut info = [0u8; MII_CHAR_INFO_LEN];
    info[..MII_CREATE_ID_LEN].copy_from_slice(&mii_create_id(MII_DEFAULT_CREATE_ID_TAG, index));
    let name = MII_DEFAULT_NICKNAME.encode_utf16().take(MII_NICKNAME_LEN);
    for (position, unit) in name.enumerate() {
        let at = MII_CHAR_INFO_NICKNAME + position * 2;
        info[at..at + 2].copy_from_slice(&unit.to_le_bytes());
    }
    info[MII_CHAR_INFO_FEATURES..].copy_from_slice(&[
        0, // font_region: standard
        mii.favorite_color,
        mii.gender,
        64, // height
        64, // build
        0,  // type: a Mii of this console's own, not a foreign one
        0,  // region_move: it may be copied anywhere
        0,  // faceline_type
        mii.faceline_color,
        0, // faceline_wrinkle
        0, // faceline_make
        mii.hair_type,
        MII_HAIR_COLORS[mii.hair_color as usize],
        0, // hair_flip
        mii.eye_type,
        MII_EYE_COLORS[mii.eye_color as usize],
        4, // eye_scale
        3, // eye_aspect
        mii.eye_rotate,
        2,  // eye_x
        12, // eye_y
        mii.eyebrow_type,
        MII_HAIR_COLORS[mii.eyebrow_color as usize],
        4,                  // eyebrow_scale
        3,                  // eyebrow_aspect
        6,                  // eyebrow_rotate
        2,                  // eyebrow_x
        10,                 // eyebrow_y
        1,                  // nose_type
        4,                  // nose_scale
        9,                  // nose_y
        23,                 // mouth_type
        0x13,               // mouth_color, already translated
        4,                  // mouth_scale
        3,                  // mouth_aspect
        13,                 // mouth_y
        MII_HAIR_COLORS[0], // beard_color
        0,                  // beard_type: none
        0,                  // mustache_type: none either
        4,                  // mustache_scale
        10,                 // mustache_y
        0,                  // glass_type: none
        8,                  // glass_color
        4,                  // glass_scale
        10,                 // glass_y
        0,                  // mole_type: none
        4,                  // mole_scale
        2,                  // mole_x
        20,                 // mole_y
        0,                  // padding
    ]);
    Some(info)
}

/// Deterministic create id: `tag` with `sequence` in its last byte.
fn mii_create_id(tag: &[u8; MII_CREATE_ID_LEN], sequence: u32) -> [u8; MII_CREATE_ID_LEN] {
    let mut id = *tag;
    id[MII_CREATE_ID_LEN - 1] = sequence as u8;
    // RFC 4122 version 4 and variant fields.
    id[6] = (id[6] & 0x0F) | 0x40;
    id[8] = (id[8] & 0x3F) | 0x80;
    id
}

/// Built-in Mii for the `sequence`th `BuildRandom` of `gender`, walking the matches.
fn random_mii_index(gender: u8, sequence: u32) -> Option<u32> {
    let matching: Vec<u32> = (0..DEFAULT_MIIS.len() as u32)
        .filter(|&index| gender >= MII_GENDER_ALL || DEFAULT_MIIS[index as usize].gender == gender)
        .collect();
    matching
        .get((sequence as usize).checked_rem(matching.len())?)
        .copied()
}

impl Cpu {
    /// `mii:e`/`mii:u`: an empty database.
    pub(crate) fn mii_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                // ConvertCurrentObjectToDomain; `nnSdk` needs the object id back.
                Some(0) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "mii:static");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("mii:static")
                .to_string()
        } else {
            "mii:static".to_string()
        };
        match iface.as_str() {
            // GetDatabaseService(u32 key) -> IDatabaseService.
            "mii:static" => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "mii:database")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "mii:database" => match cmd_id {
                // IsUpdated(SourceFlag) -> bool.
                Some(0) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // IsFullDatabase -> bool.
                Some(1) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // GetCount(SourceFlag) -> u32.
                Some(2) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // Get through Get3: list reads, empty.
                Some(3) | Some(4) | Some(8) | Some(9) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // BuildRandom(Age, Gender, Race): only Gender narrows; each gets its own create id.
                Some(6) => {
                    let data = self.ipc_request_data(tls);
                    let gender = self
                        .mem
                        .read_u8(data.wrapping_add(1))
                        .unwrap_or(MII_GENDER_ALL);
                    let sequence = self.mii_random_sequence;
                    match random_mii_index(gender, sequence).and_then(default_mii_char_info) {
                        Some(mut info) => {
                            self.mii_random_sequence = sequence.wrapping_add(1);
                            info[..MII_CREATE_ID_LEN].copy_from_slice(&mii_create_id(
                                MII_RANDOM_CREATE_ID_TAG,
                                sequence,
                            ));
                            self.write_ipc_response(tls, 0, &[], &info, &[])
                        }
                        None => self.write_ipc_response(tls, MII_INVALID_ARGUMENT, &[], &[], &[]),
                    }
                }
                // BuildDefault(u32 index) -> CharInfo.
                Some(7) => {
                    let index = self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
                    match default_mii_char_info(index) {
                        Some(info) => self.write_ipc_response(tls, 0, &[], &info, &[]),
                        None => self.write_ipc_response(tls, MII_INVALID_ARGUMENT, &[], &[], &[]),
                    }
                }
                // IsBrokenDatabaseWithClearFlag -> bool.
                Some(20) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // SetInterfaceVersion(u32).
                Some(22) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// `miiimg`: the rendered Mii image database, empty.
    pub(crate) fn miiimg_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        match cmd_id {
            // Initialize / Reload.
            Some(0) | Some(10) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetCount -> u32.
            Some(11) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // IsEmpty -> bool.
            Some(12) => self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[]),
            // IsFull -> bool.
            Some(13) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
            _ => self.unimplemented_command(tls, "miiimg", cmd_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::kernel::ipc::testing::*;

    #[test]
    fn mii_has_six_default_faces_and_no_seventh() {
        let mut create_ids = Vec::new();
        for index in 0..6u32 {
            let info = super::default_mii_char_info(index).expect("a default Mii");

            let create_id = &info[..super::MII_CREATE_ID_LEN];
            assert_ne!(
                create_id,
                [0u8; super::MII_CREATE_ID_LEN],
                "a zero id is no id"
            );
            assert_eq!(create_id[6] & 0xF0, 0x40, "RFC 4122 version 4");
            assert_eq!(create_id[8] & 0xC0, 0x80, "RFC 4122 variant");
            assert!(
                !create_ids.contains(&create_id.to_vec()),
                "two Miis, one identity"
            );
            create_ids.push(create_id.to_vec());

            let name: Vec<u16> = info[super::MII_CHAR_INFO_NICKNAME..super::MII_CHAR_INFO_FEATURES]
                .chunks(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            let end = name
                .iter()
                .position(|&unit| unit == 0)
                .expect("a terminator");
            assert_eq!(String::from_utf16(&name[..end]).unwrap(), "no name");

            // Three male Miis, then three female.
            assert_eq!(u32::from(info[0x28]), u32::from(index >= 3), "gender");
        }
        assert!(
            super::default_mii_char_info(6).is_none(),
            "there is no seventh"
        );

        // Hair colour 0 in the old palette is 8 in the new one.
        let first = super::default_mii_char_info(0).unwrap();
        assert_eq!(first[0x32], super::MII_HAIR_COLORS[0], "hair_color");
        assert_eq!(first[0x35], super::MII_EYE_COLORS[0], "eye_color");
    }

    #[test]
    fn mii_build_random_walks_the_faces_and_gives_each_its_own_identity() {
        let mut cpu = super::Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.record_domain_object(9, 7, "mii:database");

        let mut faces = Vec::new();
        let mut create_ids = Vec::new();
        for _ in 0..4 {
            marshal(&mut cpu, true, 6, &[3, 1, 3]);
            cpu.mii_request(TLS, 9, Some(6)).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), 0, "result");
            let mut info = [0u8; super::MII_CHAR_INFO_LEN];
            for (offset, byte) in info.iter_mut().enumerate() {
                *byte = cpu.mem.read_u8(TLS + 0x30 + offset as u32).unwrap();
            }
            assert_eq!(info[0x28], 1, "a female Mii was asked for");
            create_ids.push(info[..super::MII_CREATE_ID_LEN].to_vec());
            faces.push(info[super::MII_CHAR_INFO_FEATURES..].to_vec());
        }

        // Three Miis match; four calls walk them and wrap.
        assert_ne!(faces[0], faces[1]);
        assert_ne!(faces[1], faces[2]);
        assert_ne!(faces[0], faces[2]);
        assert_eq!(faces[0], faces[3], "the fourth comes back round");

        // Every built Mii has a distinct create id.
        for (position, id) in create_ids.iter().enumerate() {
            assert!(
                !create_ids[..position].contains(id),
                "two Miis, one identity"
            );
        }

        // None takes a built-in Mii's create id.
        let built_in: Vec<Vec<u8>> = (0..super::DEFAULT_MIIS.len() as u32)
            .map(|index| {
                super::default_mii_char_info(index).unwrap()[..super::MII_CREATE_ID_LEN].to_vec()
            })
            .collect();
        for id in &create_ids {
            assert!(!built_in.contains(id), "a new Mii took a built-in one's id");
        }
    }

    #[test]
    fn every_list_read_reports_an_empty_database() {
        for cmd in [3, 4, 8, 9] {
            let mut cpu = request(true, cmd, &1u32.to_le_bytes());
            cpu.record_domain_object(9, 7, "mii:database");
            cpu.mii_request(TLS, 9, Some(cmd)).unwrap();
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x28).unwrap(),
                0,
                "Get {cmd}: result"
            );
            assert_eq!(cpu.mem.read_u32(TLS + 0x30).unwrap(), 0, "Get {cmd}: count");
        }
    }

    #[test]
    fn mii_reports_a_database_that_is_intact_and_empty() {
        // Unanswered, the editor reads garbage and offers to wipe the database.
        let mut cpu = request(true, 20, &[]);
        cpu.record_domain_object(9, 7, "mii:database");
        cpu.mii_request(TLS, 9, Some(20)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u8(TLS + 0x30).unwrap(), 0, "not broken");
    }

    #[test]
    fn mii_build_default_answers_over_a_domain() {
        // The domain header shifts the payload 0x10 further.
        let mut cpu = request(true, 7, &3u32.to_le_bytes());
        cpu.record_domain_object(9, 7, "mii:database");
        cpu.mii_request(TLS, 9, Some(7)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u8(TLS + 0x30 + 0x28).unwrap(), 1, "gender");
        let expected = super::default_mii_char_info(3).unwrap();
        for (offset, &byte) in expected.iter().enumerate() {
            let at = TLS + 0x30 + offset as u32;
            assert_eq!(
                cpu.mem.read_u8(at).unwrap(),
                byte,
                "CharInfo byte {offset:#x}"
            );
        }

        let mut cpu = request(true, 7, &6u32.to_le_bytes());
        cpu.record_domain_object(9, 7, "mii:database");
        cpu.mii_request(TLS, 9, Some(7)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x28).unwrap(),
            super::MII_INVALID_ARGUMENT
        );
    }
}
