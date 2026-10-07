//! Raw LZ4 block decompression (no frame header), as used by NSO segments.

/// Errors on truncated input, a zero offset, or a match before the output start.
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

        for src in (out.len() - offset..).take(match_len) {
            let b = out[src];
            out.push(b);
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
        // token 0x50 = 5 literals, final sequence.
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
        // 4 literals "abcd", then a minimum-length match at offset 4.
        let mut input = vec![0x40, b'a', b'b', b'c', b'd'];
        input.extend_from_slice(&4u16.to_le_bytes()); // offset
        let out = decompress_block(&input, 8).unwrap();
        assert_eq!(out, b"abcdabcd");
    }

    #[test]
    fn overlapping_match_runs_a_single_byte() {
        // 1 literal "a", then match length 15 + 5 + 4 = 24 at offset 1.
        let mut input = vec![0x1f, b'a'];
        input.extend_from_slice(&1u16.to_le_bytes());
        input.push(5); // extra for match length (< 0xff, stops immediately)
        let out = decompress_block(&input, 25).unwrap();
        let mut expected = vec![b'a'];
        expected.extend(std::iter::repeat_n(b'a', 24));
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
