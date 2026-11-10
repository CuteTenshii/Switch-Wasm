//! A VP9 decoder: profile 0 (8-bit 4:2:0), bit-exact with libvpx.

// Ported from the reference decoder's index arithmetic, which these lints would obscure.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_memcpy,
    dead_code
)]

mod bool_decoder;
mod idct;
mod intra;
mod probs;
mod tables;
mod transforms_1d;
mod types;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Truncated,
    Corrupt(&'static str),
    Unsupported(&'static str),
}

#[cfg(test)]
mod tests {
    use super::idct::inverse_transform_add;
    use super::idct::inverse_wht_add;

    /// Checks the transforms against vectors from libvpx's C code: the file named by
    /// `VP9_IDCT_VECTORS`, written by a harness linked to libvpx.
    #[test]
    fn transforms_match_libvpx_vectors() {
        let Some(path) = std::env::var_os("VP9_IDCT_VECTORS") else {
            return;
        };
        let data = std::fs::read(path).unwrap();
        let mut at = 0;
        let mut cases = 0;
        while at < data.len() {
            let (size, kind, lossless) = (data[at], data[at + 1], data[at + 2] != 0);
            at += 4;
            let n = 4usize << size;
            let count = n * n;
            let coeffs: Vec<i32> = data[at..at + count * 4]
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            at += count * 4;
            let mut dest = data[at..at + count].to_vec();
            at += count;
            let expect = &data[at..at + count];
            at += count;
            if lossless {
                inverse_wht_add(&coeffs, &mut dest, n);
            } else {
                inverse_transform_add(size, kind, &coeffs, &mut dest, n);
            }
            let first = dest.iter().zip(expect).position(|(a, b)| a != b);
            assert!(first.is_none(), "case {cases}: size {size}, type {kind}, lossless {lossless}, first mismatch at {first:?}");
            cases += 1;
        }
        assert!(cases > 0);
    }
}
