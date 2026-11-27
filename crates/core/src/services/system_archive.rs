//! Stand-ins for system data archives the firmware does not provide, holding
//! the files titles open, with empty contents, so those titles run without firmware.

/// A system data archive that can be built when none is registered.
struct Synthesized {
    data_id: u64,
    name: &'static str,
    files: fn() -> Vec<(String, Vec<u8>)>,
}

const SYNTHESIZED: &[Synthesized] = &[Synthesized {
    data_id: 0x0100_0000_0000_0823,
    name: "NgWord2",
    files: ng_word2,
}];

/// The name and RomFS image of the stand-in for `data_id`, if there is one.
pub(crate) fn synthesize(data_id: u64) -> Option<(&'static str, Vec<u8>)> {
    let archive = SYNTHESIZED.iter().find(|a| a.data_id == data_id)?;
    let files = (archive.files)();
    let files: Vec<(&str, &[u8])> = files
        .iter()
        .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
        .collect();
    Some((archive.name, crate::romfs::build(&files).ok()?))
}

/// 21.2.0 ships sets 0 to 17; a title asks for the set of its language.
const NG_WORD2_LANGUAGE_SETS: usize = 18;
/// 21.2.0's, the closest real archive to the firmware version `set:sys` reports.
const NG_WORD2_VERSION: u32 = 0x41;

/// NgWord2 as `nn::ngc::ProfanityFilter` reads it: an Aho-Corasick automaton per
/// language set and match kind, plus `version.dat`. Firmware's other files (the
/// similar-form tables, the tries) are read by no title's SDK.
fn ng_word2() -> Vec<(String, Vec<u8>)> {
    let automaton = gzip(&empty_aho_corasick());
    let sets = (0..NG_WORD2_LANGUAGE_SETS)
        .map(|i| i.to_string())
        .chain(["common".to_owned()]);
    let mut files: Vec<(String, Vec<u8>)> = sets
        .flat_map(|set| ["b1", "b2", "not_b"].map(|kind| format!("ac_{set}_{kind}_nx")))
        .map(|name| (name, automaton.clone()))
        .collect();
    files.push((
        "version.dat".to_owned(),
        NG_WORD2_VERSION.to_be_bytes().to_vec(),
    ));
    files
}

/// An automaton with no words, in the little-endian layout
/// `nn::ngc::detail::AhoCorasick::Import` reads (the same in SDK 3.6 and 13.3).
/// Every count is zero, so no array follows one, and a rank over zero bits reads
/// nothing.
fn empty_aho_corasick() -> Vec<u8> {
    const COUNT: [u8; 4] = [0; 4];
    // BitVector32 and PlainArray32 are each a count and their words.
    let set = COUNT.to_vec();
    let select = [COUNT, COUNT, COUNT].concat();
    let sbv = [set.as_slice(), &select].concat();
    // Three u64 and three u32 header fields, the last the length of an array.
    let sparse_set = [&[0u8; 3 * 8 + 3 * 4][..], &sbv].concat();
    // Two counted u32 arrays, then a u8 count of a third.
    let compressed_array = [&COUNT[..], &COUNT, &[0]].concat();
    // A zero flag means no nested tree follows.
    let bp = [set.as_slice(), &[0], &select].concat();
    [
        compressed_array.as_slice(),
        &sparse_set,
        &compressed_array,
        &sparse_set,
        &COUNT,
        &sparse_set,
        &bp,
    ]
    .concat()
}

/// `data` as a gzip member of one stored deflate block, which the guest's zlib
/// inflates like any other.
fn gzip(data: &[u8]) -> Vec<u8> {
    let len = u16::try_from(data.len()).expect("a stored block holds at most 64 KiB");
    let mut out = vec![0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 0xFF];
    out.push(1);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// The gzip trailer's CRC-32 (reflected, polynomial 0xEDB88320).
fn crc32(data: &[u8]) -> u32 {
    !data.iter().fold(!0u32, |crc, &byte| {
        (0..8).fold(crc ^ u32::from(byte), |crc, _| {
            (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::romfs::RomFs;

    /// The payload of a gzip member written by [`gzip`], checked as zlib would.
    fn gunzip_stored(member: &[u8]) -> Vec<u8> {
        assert_eq!(&member[..4], [0x1F, 0x8B, 8, 0], "gzip, deflate, no extras");
        assert_eq!(member[10], 1, "one final stored block");
        let len = u16::from_le_bytes([member[11], member[12]]);
        assert_eq!(u16::from_le_bytes([member[13], member[14]]), !len);
        let data = &member[15..15 + usize::from(len)];
        let trailer = &member[15 + usize::from(len)..];
        assert_eq!(trailer.len(), 8);
        assert_eq!(trailer[..4], crc32(data).to_le_bytes());
        assert_eq!(trailer[4..], (data.len() as u32).to_le_bytes());
        data.to_vec()
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn an_empty_automaton_is_195_zero_bytes() {
        // 9 + 52 + 9 + 52 + 4 + 52 + 17, per the structures `Import` reads.
        let automaton = empty_aho_corasick();
        assert_eq!(automaton.len(), 195);
        assert!(automaton.iter().all(|&b| b == 0));
        assert_eq!(gunzip_stored(&gzip(&automaton)), automaton);
    }

    #[test]
    fn ng_word2_holds_every_file_the_profanity_filter_opens() {
        let (name, image) = synthesize(0x0100_0000_0000_0823).unwrap();
        assert_eq!(name, "NgWord2");
        let romfs = RomFs::parse(&image).unwrap();
        let mut expected: Vec<String> = (0..18)
            .map(|i| i.to_string())
            .chain(["common".to_owned()])
            .flat_map(|set| {
                [
                    format!("/ac_{set}_b1_nx"),
                    format!("/ac_{set}_b2_nx"),
                    format!("/ac_{set}_not_b_nx"),
                ]
            })
            .chain(["/version.dat".to_owned()])
            .collect();
        expected.sort();
        let mut listed: Vec<String> = romfs.files().iter().map(|f| f.path.clone()).collect();
        listed.sort();
        assert_eq!(listed, expected);

        for path in listed.iter().filter(|p| p.starts_with("/ac_")) {
            let automaton = gunzip_stored(romfs.read_path(path).unwrap());
            assert_eq!(automaton, empty_aho_corasick(), "{path}");
        }
        assert_eq!(romfs.read_path("/version.dat").unwrap(), [0, 0, 0, 0x41]);
    }

    #[test]
    fn other_ids_have_no_stand_in() {
        assert!(synthesize(0x0100_0000_0000_0806).is_none(), "NgWord1");
        assert!(synthesize(0).is_none());
    }
}
