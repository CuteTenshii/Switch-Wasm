//! A VP9 decoder: profile 0 (8-bit 4:2:0), bit-exact with libvpx.

// Ported from the reference decoder's index arithmetic, which these lints would obscure.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_memcpy,
    dead_code
)]

mod block;
mod bool_decoder;
mod decoder;
mod frame;
mod header;
mod idct;
mod inter;
mod intra;
mod loopfilter;
mod modeinfo;
pub mod nvdec;
mod probs;
mod tables;
mod tokens;
mod transforms_1d;
mod types;

pub use decoder::Decoder;
pub use frame::{Frame, Plane};

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
    use super::Decoder;

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

    /// Decodes the IVF clips in the directory named by `VP9_CLIPS` and compares each frame with the
    /// `.yuv` ffmpeg wrote beside it.
    #[test]
    fn clips_decode_like_ffmpeg() {
        let Some(dir) = std::env::var_os("VP9_CLIPS") else {
            return;
        };
        let mut names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "ivf"))
            .collect();
        names.sort();
        assert!(!names.is_empty());
        for path in names {
            let ivf = std::fs::read(&path).unwrap();
            let expect = std::fs::read(path.with_extension("yuv")).unwrap();
            let (width, height) = (
                u16::from_le_bytes([ivf[12], ivf[13]]) as usize,
                u16::from_le_bytes([ivf[14], ivf[15]]) as usize,
            );
            let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
            let frame_bytes = width * height + 2 * cw * ch;
            let mut decoder = Decoder::new();
            let mut at = 32;
            let mut shown = 0;
            while at + 12 <= ivf.len() {
                let size = u32::from_le_bytes(ivf[at..at + 4].try_into().unwrap()) as usize;
                at += 12;
                let packet = &ivf[at..at + size];
                at += size;
                for frame in decoder.decode_packet(packet).unwrap() {
                    let reference = &expect[shown * frame_bytes..(shown + 1) * frame_bytes];
                    let mut got = Vec::with_capacity(frame_bytes);
                    for (plane, (w, h)) in
                        frame
                            .planes
                            .iter()
                            .zip([(width, height), (cw, ch), (cw, ch)])
                    {
                        for y in 0..h {
                            got.extend_from_slice(
                                &plane.data[y * plane.stride..y * plane.stride + w],
                            );
                        }
                    }
                    if let Some(i) = got.iter().zip(reference).position(|(a, b)| a != b) {
                        let (p, off) = if i < width * height {
                            (0, i)
                        } else if i < width * height + cw * ch {
                            (1, i - width * height)
                        } else {
                            (2, i - width * height - cw * ch)
                        };
                        let w = if p == 0 { width } else { cw };
                        panic!(
                            "{}: frame {shown}, plane {p}, pixel ({}, {}): got {}, want {}",
                            path.display(),
                            off % w,
                            off / w,
                            got[i],
                            reference[i]
                        );
                    }
                    shown += 1;
                }
            }
            assert_eq!(
                shown * frame_bytes,
                expect.len(),
                "{}: frame count",
                path.display()
            );
        }
    }
}
