//! 16-byte UUIDs, as `nn::util::Uuid` holds them, and the RFC 9562 version and variant bits that
//! make an id a UUID and not just 16 bytes.

pub type Uuid = [u8; 16];

/// Put `version` and the RFC 9562 variant into `bytes`, which keep every other bit.
pub const fn stamp(mut bytes: Uuid, version: u8) -> Uuid {
    bytes[6] = (bytes[6] & 0x0F) | (version << 4);
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    bytes
}

/// A version 4 UUID from 16 random bytes.
pub const fn v4(random: Uuid) -> Uuid {
    stamp(random, 4)
}

/// A version 8 UUID, the one RFC 9562 leaves to the vendor: for an id derived from other data.
pub const fn v8(custom: Uuid) -> Uuid {
    stamp(custom, 8)
}

pub const fn version(uuid: &Uuid) -> u8 {
    uuid[6] >> 4
}

/// Whether `uuid` has the RFC 9562 variant and a version it defines (1 to 8), which the nil UUID
/// and 16 bytes of text do not.
pub const fn is_valid(uuid: &Uuid) -> bool {
    uuid[8] >> 6 == 0b10 && matches!(version(uuid), 1..=8)
}

const fn hex_digit(digit: u8) -> u8 {
    match digit {
        b'0'..=b'9' => digit - b'0',
        b'a'..=b'f' => digit - b'a' + 10,
        b'A'..=b'F' => digit - b'A' + 10,
        _ => panic!("a UUID's digits are hexadecimal"),
    }
}

/// The UUID written `8-4-4-4-12`. Used for a constant, a malformed one fails the build.
pub const fn parse(text: &str) -> Uuid {
    let text = text.as_bytes();
    assert!(text.len() == 36, "a UUID is 36 characters");
    let mut uuid = [0u8; 16];
    let (mut at, mut out) = (0, 0);
    while at < 36 {
        if at == 8 || at == 13 || at == 18 || at == 23 {
            assert!(
                text[at] == b'-',
                "a UUID has hyphens after 8, 4, 4 and 4 digits"
            );
            at += 1;
        }
        uuid[out] = hex_digit(text[at]) << 4 | hex_digit(text[at + 1]);
        at += 2;
        out += 1;
    }
    uuid
}

/// The lowercase `8-4-4-4-12` text, for logs and messages.
pub fn format(uuid: &Uuid) -> String {
    let mut text = String::with_capacity(36);
    for (index, byte) in uuid.iter().enumerate() {
        if matches!(index, 4 | 6 | 8 | 10) {
            text.push('-');
        }
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "531e8558-b186-4424-ad40-1d1d21006036";

    #[test]
    fn text_and_bytes_convert_both_ways() {
        let uuid = parse(TEXT);
        assert_eq!(uuid[..4], [0x53, 0x1e, 0x85, 0x58]);
        assert_eq!(uuid[15], 0x36);
        assert_eq!(format(&uuid), TEXT);
        assert_eq!(parse(&TEXT.to_uppercase()), uuid, "case does not matter");
    }

    #[test]
    fn a_constant_can_be_written_as_text() {
        const ID: Uuid = parse("2c14f919-ede2-4ab3-94d3-dec441af3396");
        assert_eq!(format(&ID), "2c14f919-ede2-4ab3-94d3-dec441af3396");
        assert!(is_valid(&ID));
    }

    #[test]
    #[should_panic(expected = "36 characters")]
    fn text_of_the_wrong_length_is_refused() {
        parse("531e8558-b186-4424-ad40-1d1d2100603");
    }

    #[test]
    #[should_panic(expected = "hexadecimal")]
    fn a_character_that_is_not_a_digit_is_refused() {
        parse("531e8558-b186-4424-ad40-1d1d2100603g");
    }

    #[test]
    #[should_panic(expected = "hyphens")]
    fn hyphens_belong_where_the_format_puts_them() {
        parse("531e8558b-186-4424-ad40-1d1d21006036");
    }

    #[test]
    fn stamping_sets_the_version_and_the_variant_and_nothing_else() {
        let stamped = stamp([0xFF; 16], 4);
        assert_eq!(version(&stamped), 4);
        assert_eq!(stamped[6], 0x4F, "the low nibble of that byte is data");
        assert_eq!(stamped[8], 0xBF, "10 then six data bits");
        for (index, &byte) in stamped.iter().enumerate() {
            if index != 6 && index != 8 {
                assert_eq!(byte, 0xFF);
            }
        }
        assert_eq!(version(&v8([0; 16])), 8);
        assert_eq!(version(&v4([0; 16])), 4);
    }

    #[test]
    fn only_a_stamped_id_is_valid() {
        assert!(
            is_valid(&v4([0; 16])),
            "all-zero random bytes still make a v4"
        );
        assert!(is_valid(&v8([0xAB; 16])));
        assert!(is_valid(&parse(TEXT)));
        assert!(!is_valid(&[0; 16]), "the nil UUID");
        assert!(!is_valid(b"switch-wasm user"), "text is not a UUID");
        assert!(!is_valid(&stamp([0; 16], 9)), "version 9 is not defined");
        let mut wrong_variant = parse(TEXT);
        wrong_variant[8] &= 0x3F;
        assert!(!is_valid(&wrong_variant));
    }
}
