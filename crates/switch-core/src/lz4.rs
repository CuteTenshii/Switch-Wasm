//! LZ4 "legacy"/raw block decompression — no frame headers, just the token
//! stream. This is the format NSO0 embeds for compressed `.text`/`.rodata`/
//! `.data` segments: the decompressed size is already known from the NSO
//! header, so no length prefix is needed either.
//!
//! Block format (see the public `lz4_Block_format.md` spec):
//!
//! ```text
//! sequence: token(1) [literal_length_extra] literals[literal_length]
//!           offset(2, LE) [match_length_extra]
//! token: high nibble = literal_length (0-15), low nibble = match_length (0-15)
//! ```
//!
//! A length nibble of 15 means "read more": additional bytes each add 0-255,
//! and the sum keeps growing while a byte reads 255. The final sequence in a
//! block is literals-only (no offset/match) since there's nothing left to
//! match against.

/// Decompress an LZ4 block into a buffer of exactly `decompressed_size`
/// bytes. Returns an error string (not [`crate::Error`] — this module has no
/// NSO-specific context) on truncated input, a match offset of 0, or a match
/// that would read before the start of the output.
pub fn decompress_block(input: &[u8], decompressed_size: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(decompressed_size);
    let mut ip = 0usize;

    while out.len() < decompressed_size {
        let token = *input.get(ip).ok_or("truncated LZ4 block: missing token")?;
        ip += 1;

        let mut literal_len = (token >> 4) as usize;
        if literal_len == 15 {
            literal_len += read_extra_length(input, &mut ip)?;
        }
        let lit_end = ip
            .checked_add(literal_len)
            .ok_or("truncated LZ4 block: literal length overflow")?;
        if lit_end > input.len() {
            return Err("truncated LZ4 block: literals exceed input".into());
        }
        out.extend_from_slice(&input[ip..lit_end]);
        ip = lit_end;

        if out.len() >= decompressed_size {
            break;
        }
        if ip >= input.len() {
            return Err("truncated LZ4 block: missing match offset".into());
        }

        let off_lo = *input.get(ip).ok_or("truncated LZ4 block: offset")?;
        let off_hi = *input.get(ip + 1).ok_or("truncated LZ4 block: offset")?;
        ip += 2;
        let offset = u16::from_le_bytes([off_lo, off_hi]) as usize;
        if offset == 0 || offset > out.len() {
            return Err("invalid LZ4 match offset".into());
        }

        let mut match_len = (token & 0x0f) as usize;
        if match_len == 15 {
            match_len += read_extra_length(input, &mut ip)?;
        }
        match_len += 4; // minmatch

        let mut src = out.len() - offset;
        for _ in 0..match_len {
            let b = out[src];
            out.push(b);
            src += 1;
        }
    }

    out.truncate(decompressed_size);
    Ok(out)
}

fn read_extra_length(input: &[u8], ip: &mut usize) -> Result<usize, String> {
    let mut extra = 0usize;
    loop {
        let b = *input.get(*ip).ok_or("truncated LZ4 block: length byte")?;
        *ip += 1;
        extra += b as usize;
        if b != 0xff {
            break;
        }
    }
    Ok(extra)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_only_block() {
        // token 0x50 = 5 literals, 0 match; final sequence, no offset needed.
        let input = [0x50, b'h', b'e', b'l', b'l', b'o'];
        let out = decompress_block(&input, 5).unwrap();
        assert_eq!(out, b"hello");
    }

    #[test]
    fn extended_literal_length() {
        // literal nibble 15 + extra byte 10 => 25 literals.
        let mut input = vec![0xf0, 10];
        let literals: Vec<u8> = (0..25u8).collect();
        input.extend_from_slice(&literals);
        let out = decompress_block(&input, 25).unwrap();
        assert_eq!(out, literals);
    }

    #[test]
    fn back_reference_repeats_a_run() {
        // 4 literals "abcd", then a match copying those same 4 bytes via
        // offset 4, match_length nibble 0 -> minmatch 4. Token high=4 (lits),
        // low=0 (match len nibble) -> 0x40.
        let mut input = vec![0x40, b'a', b'b', b'c', b'd'];
        input.extend_from_slice(&4u16.to_le_bytes()); // offset
        // No trailing match-length extra needed (nibble != 15).
        let out = decompress_block(&input, 8).unwrap();
        assert_eq!(out, b"abcdabcd");
    }

    #[test]
    fn overlapping_match_runs_a_single_byte() {
        // 1 literal "a", offset 1, match length nibble 15 + extra 251 =>
        // 255 + 4 minmatch = wait: nibble 15 means "read more", so
        // match_len = 15 + extra, +4 minmatch. Use extra=5 -> match_len=20+4=24.
        // This copies byte-by-byte from 1 back, so it just repeats 'a' 24 times.
        let mut input = vec![0x1f, b'a'];
        input.extend_from_slice(&1u16.to_le_bytes());
        input.push(5); // extra for match length (< 0xff, stops immediately)
        let out = decompress_block(&input, 25).unwrap();
        let mut expected = vec![b'a'];
        expected.extend(std::iter::repeat(b'a').take(24));
        assert_eq!(out, expected);
    }

    #[test]
    fn rejects_bad_offset() {
        // 1 literal, then offset 0 (invalid).
        let mut input = vec![0x10, b'a'];
        input.extend_from_slice(&0u16.to_le_bytes());
        assert!(decompress_block(&input, 10).is_err());
    }

    #[test]
    fn rejects_truncated_input() {
        assert!(decompress_block(&[0x50, b'h', b'i'], 5).is_err());
    }
}
