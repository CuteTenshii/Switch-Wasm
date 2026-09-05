//! `mii`: the console's Mii database.
//!
//! A console with no Miis on it is a console every Mii picker refuses to open,
//! so the database ships with a small set of built-in characters rather than
//! being empty.

use super::Cpu;
use crate::Result;

/// `nn::mii::CharInfo`: one Mii as everything that reads the database is
/// handed it, a create id, a nickname, then one byte per feature.
const MII_CHAR_INFO_LEN: usize = 0x58;

/// A Mii's `CreateId`: the UUID that names it wherever it is copied to, and
/// the first thing in a `CharInfo`.
const MII_CREATE_ID_LEN: usize = 0x10;

/// The tags [`mii_create_id`] stamps the six Miis `nn::mii` carries with, and
/// the ones `BuildRandom` invents. They differ so that a Mii a title made can
/// never be handed the identity of a built-in one.
const MII_DEFAULT_CREATE_ID_TAG: &[u8; MII_CREATE_ID_LEN] = b"switch-wasm mii\0";
const MII_RANDOM_CREATE_ID_TAG: &[u8; MII_CREATE_ID_LEN] = b"switch-wasm rnd\0";

/// Where the nickname sits, past that id.
const MII_CHAR_INFO_NICKNAME: usize = 0x10;

/// Where the features start, past the nickname and its terminator.
const MII_CHAR_INFO_FEATURES: usize = 0x26;

/// How long a Mii's name may be, in UTF-16 code units. The field holds one
/// more, for the terminator a name of the full length still carries.
const MII_NICKNAME_LEN: usize = 10;

/// The name `nn::mii` gives a Mii nobody has named. The default Miis all carry
/// it: naming one is the first thing the editor asks whoever picks it.
const MII_DEFAULT_NICKNAME: &str = "no name";

/// mii's "that argument is out of range" (module 126, description 1), which is
/// what `BuildDefault` answers an index past the Miis it has.
const MII_INVALID_ARGUMENT: u32 = 126 | (1 << 9);

/// `nn::mii::Gender::All`: `BuildRandom`'s default, and the value past the two
/// real genders that narrows nothing.
const MII_GENDER_ALL: u8 = 2;

/// A Mii's colours are numbered in the palette the 3DS and Wii U used, which
/// is the palette the default Miis are written in, and are widened on the way
/// out to the Switch's larger one. Hair, eyebrows and beards share this table;
/// eyes have [`MII_EYE_COLORS`]; a faceline colour is the same number in both.
const MII_HAIR_COLORS: [u8; 8] = [8, 1, 2, 3, 4, 5, 6, 7];

/// The same translation for eye colours.
const MII_EYE_COLORS: [u8; 6] = [8, 9, 10, 11, 12, 13];

/// What differs between the six Miis `nn::mii` carries in its own image. Every
/// other feature is the same in all six and is written by
/// [`default_mii_char_info`], which is also where the colours here are carried
/// over into the palette a `CharInfo` is read in.
struct DefaultMii {
    faceline_color: u8,
    hair_type: u8,
    hair_color: u8,
    eye_type: u8,
    eye_color: u8,
    eye_rotate: u8,
    eyebrow_type: u8,
    eyebrow_color: u8,
    /// 0 is a male Mii, 1 a female one; the first three of these are male.
    gender: u8,
    favorite_color: u8,
}

/// The six default Miis, in the order `BuildDefault` indexes them.
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

/// The `index`th default Mii as a `CharInfo`, or `None` when there is no such
/// Mii, which is the only way `BuildDefault` can fail.
///
/// These are built, not looked up: they live in `nn::mii`'s own image rather
/// than in the database, which is why a console with no Miis on it still has
/// all six to offer. An editor asks for them to fill the row of faces it opens
/// on, so answering the database's count with none and this with nothing are
/// not the same answer.
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
        0, // font_region: the standard set, not the Chinese, Korean or Taiwanese one
        mii.favorite_color,
        mii.gender,
        64, // height: the middle of the range, as is the build below it
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
        0x13,               // mouth_color: the one the default Miis use, already translated
        4,                  // mouth_scale
        3,                  // mouth_aspect
        13,                 // mouth_y
        MII_HAIR_COLORS[0], // beard_color
        0,                  // beard_type: none, so the colour above never shows
        0,                  // mustache_type: none either
        4,                  // mustache_scale
        10,                 // mustache_y
        0,                  // glass_type: none
        8,                  // glass_color: the glasses palette's first, as unseen as the beard
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

/// The create id a Mii is stamped with, built from a tag and a count.
///
/// A real one is an RFC 4122 version 4 UUID drawn at random when the Mii is
/// built, and it is the Mii's identity: a database keyed on it treats two Miis
/// sharing one as the same Mii. There is no database here to collide in, so
/// these are counted rather than drawn: the same Mii gets the same id every
/// run, which is what makes one boot's trace comparable with the next's.
///
/// The count occupies the tag's last byte, so a tag has 256 ids in it. That
/// outlasts the hundred Miis a console's database holds, which is the only
/// population these ever have to stay distinct across.
fn mii_create_id(tag: &[u8; MII_CREATE_ID_LEN], sequence: u32) -> [u8; MII_CREATE_ID_LEN] {
    let mut id = *tag;
    id[MII_CREATE_ID_LEN - 1] = sequence as u8;
    // The version (4, "random") and variant (RFC 4122) fields, which sit in
    // the middle of the id rather than at either end of it.
    id[6] = (id[6] & 0x0F) | 0x40;
    id[8] = (id[8] & 0x3F) | 0x80;
    id
}

/// Which built-in Mii `BuildRandom` answers with, given the gender asked for
/// and how many random Miis have already been built.
///
/// Successive calls walk the matching Miis rather than repeating one, because
/// an editor fills a row of faces by calling this once per face, answering
/// them all with the same Mii offers a choice of one.
///
/// `None` means no built-in Mii has the requested gender, which cannot happen
/// with the six below and is `MII_INVALID_ARGUMENT` if the table ever changes.
fn random_mii_index(gender: u8, sequence: u32) -> Option<u32> {
    let matching: Vec<u32> = (0..DEFAULT_MIIS.len() as u32)
        .filter(|&index| gender >= MII_GENDER_ALL || DEFAULT_MIIS[index as usize].gender == gender)
        .collect();
    matching
        .get((sequence as usize).checked_rem(matching.len())?)
        .copied()
}

impl Cpu {
    /// `mii:e`/`mii:u`: the console's Mii database.
    ///
    /// There are no Miis on this console and no NAND to keep them on, so the
    /// database is real but empty. That is a truthful answer rather than a
    /// convenient one: an editor asks how many exist before it decides
    /// whether to open on the list or on "create a new one", and both are
    /// valid states of a real console.
    ///
    /// Empty is not the same as having nothing to offer, though: the six
    /// default Miis come out of `nn::mii`'s own image rather than the
    /// database, and [`default_mii_char_info`] builds them.
    pub(super) fn mii_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                // QueryPointerBufferSize.
                // ConvertCurrentObjectToDomain -> the id the session itself
                // takes in its new domain. `nnSdk` converts this one before
                // asking for the database, so answering without an object id
                // leaves every later request addressed to nothing.
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
            // IStaticService::GetDatabaseService(u32 key) -> IDatabaseService.
            "mii:static" => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "mii:database")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "mii:database" => match cmd_id {
                // IsUpdated(SourceFlag) -> bool. Nothing writes the database,
                // so it has not changed since the caller last looked.
                Some(0) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // IsFullDatabase -> bool: an empty one is not full.
                Some(1) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // GetCount(SourceFlag) -> u32.
                Some(2) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // The list reads: each fills a caller-provided buffer with
                // as many Mii records as it holds and reports how many that
                // was. An empty database writes nothing and reports none,
                // which is a state a real console is in until someone makes
                // their first Mii.
                Some(4) | Some(8) | Some(9) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // BuildRandom(Age, Gender, Race) -> CharInfo: a Mii nobody
                // made, which is what an editor offers whoever has not made
                // one yet. The three arguments are one byte each and narrow
                // what may come back; only Gender narrows anything here,
                // because gender is the only one of the three the six
                // built-in Miis differ in. Filtering on an age or a race they
                // all share would be choosing between faces on a property
                // that is not in them.
                //
                // Random by hardware's definition, counted by this one: the
                // Mii is picked in sequence and stamped with a create id of
                // its own, which is what a database that keeps it tells it
                // apart from every other Mii by. Reusing the built-in id
                // would file each new Mii on top of the one it was built
                // from.
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
                // BuildDefault(u32 index) -> CharInfo: one of the six Miis
                // `nn::mii` carries in its own image. This is the one read
                // that does not go through the database, and the editor makes
                // it before it has asked for anything else: it is where the
                // faces it opens on come from when nobody has made a Mii yet.
                Some(7) => {
                    let index = self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
                    match default_mii_char_info(index) {
                        Some(info) => self.write_ipc_response(tls, 0, &[], &info, &[]),
                        None => self.write_ipc_response(tls, MII_INVALID_ARGUMENT, &[], &[], &[]),
                    }
                }
                // IsBrokenDatabaseWithClearFlag -> bool, and clears the
                // flag it reports. This database is synthesized rather than
                // read off a filesystem, so it has never been corrupted and
                // there is no flag behind the answer to clear.
                Some(20) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // SetInterfaceVersion(u32): which revision of the Mii
                // structures the caller speaks. Nothing here reads them, and
                // an empty database is the same shape in every revision.
                Some(22) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// `miiimg`: the database of *rendered* Mii images, kept alongside the Mii
    /// data itself so the menu can show faces without rendering them.
    ///
    /// Empty, for the same reason [`Cpu::mii_request`]'s is. Answering its
    /// count with a fabricated object id, which is what the generic
    /// no-implementation reply did: left the editor reading a garbage count
    /// and asking for the attributes of images that were never there, half a
    /// million times over, which is what a "running but drawing nothing"
    /// applet turned out to be.
    pub(super) fn miiimg_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
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
    use crate::cpu::ipc::testing::*;

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

            // Three Miis with a male build, then three with a female one.
            assert_eq!(u32::from(info[0x28]), u32::from(index >= 3), "gender");
        }
        assert!(
            super::default_mii_char_info(6).is_none(),
            "there is no seventh"
        );

        // The table is written in the older colour palette, so a hair colour
        // of 0 has to come out as the newer palette's 8. Handing the raw
        // number over is the mistake this catches.
        let first = super::default_mii_char_info(0).unwrap();
        assert_eq!(first[0x32], super::MII_HAIR_COLORS[0], "hair_color");
        assert_eq!(first[0x35], super::MII_EYE_COLORS[0], "eye_color");
    }

    #[test]
    fn mii_build_random_walks_the_faces_and_gives_each_its_own_identity() {
        // BuildRandom takes Age, Gender and Race as a byte each. Gender is
        // the only one of the three the six built-in Miis differ in, so it is
        // the only one that narrows what comes back: Female here, which is
        // the last three of them.
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

        // Three Miis match, and four calls walk them and come back round. An
        // editor fills its row of faces by calling this once per face, so a
        // pick that does not move offers a choice of one.
        assert_ne!(faces[0], faces[1]);
        assert_ne!(faces[1], faces[2]);
        assert_ne!(faces[0], faces[2]);
        assert_eq!(faces[0], faces[3], "the fourth comes back round");

        // Each is still its own Mii, the two built from one face included:
        // a database keyed on create ids files two Miis sharing one on top of
        // each other.
        for (position, id) in create_ids.iter().enumerate() {
            assert!(
                !create_ids[..position].contains(id),
                "two Miis, one identity"
            );
        }

        // And none of them may take a built-in Mii's identity, which is what
        // the separate tag keeps apart.
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
    fn mii_reports_a_database_that_is_intact_and_empty() {
        // IsBrokenDatabaseWithClearFlag. Answered with nothing, the editor
        // read its own stack for the flag; a nonzero read there is a database
        // it will offer to wipe before it will show a face.
        let mut cpu = request(true, 20, &[]);
        cpu.record_domain_object(9, 7, "mii:database");
        cpu.mii_request(TLS, 9, Some(20)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u8(TLS + 0x30).unwrap(), 0, "not broken");
    }

    #[test]
    fn mii_build_default_answers_over_a_domain() {
        // `nnSdk` converts the mii session to a domain before it asks for the
        // database, so the index arrives (and the CharInfo goes back) 0x10
        // further into the buffer than a plain request's payload would.
        let mut cpu = request(true, 7, &3u32.to_le_bytes());
        cpu.record_domain_object(9, 7, "mii:database");
        cpu.mii_request(TLS, 9, Some(7)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), 0, "result");
        // The fourth default Mii, read out of the reply where the caller
        // reads it rather than out of what built it.
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

        // An index past the six is the caller's mistake, and the one failure
        // this command has. Answering it with a success leaves whoever asked
        // reading a Mii out of a zeroed reply.
        let mut cpu = request(true, 7, &6u32.to_le_bytes());
        cpu.record_domain_object(9, 7, "mii:database");
        cpu.mii_request(TLS, 9, Some(7)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x28).unwrap(),
            super::MII_INVALID_ARGUMENT
        );
    }
}
