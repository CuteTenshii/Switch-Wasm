//! Compare the BC decoders against an independent implementation:
//! `bcn_difftest <vectors.bin> [blocks] [astc_vectors.bin]`.
//!
//! The codecs in `gpu/bcn.rs` are transcriptions of a specification, and the
//! failure mode of a transcription is a table entry that is wrong in a way no
//! rendered frame localises. This checks them the only way that really
//! settles it: against a second implementation, over thousands of random
//! blocks.
//!
//! The fixture is produced by a throwaway C harness around `bcdec.h`
//! (<https://github.com/iOrange/bcdec>, public domain), which writes one
//! record per block — sixteen input bytes then the reference's decoded
//! texels — for each codec in turn:
//!
//! ```c
//! #define BCDEC_IMPLEMENTATION
//! #include "bcdec.h"
//! // for each codec: fill 16 random bytes, decode, fwrite(input, 16) then
//! // fwrite(output, texel_bytes * 16), 4000 blocks each, in the order
//! // BC1, BC2, BC3, BC7 (RGBA), then BC4 (R), then BC5 (RG).
//! ```
//!
//! BC7 is expected to agree exactly. BC1, BC2 and BC3 are expected to agree
//! to within one part in 255: their 5:6:5 endpoints can be expanded to eight
//! bits either by replicating the high bits, which is what this decoder and
//! the BPTC endpoint rule both do, or by rounding, which is what `bcdec`
//! does for those three. Both readings sit inside the tolerance S3TC allows,
//! and no two vendors agreed on it either.
use std::env;
use std::fs;
use switch_core::gpu::bcn::{decode, decode_into, Codec};

fn main() {
    let path = env::args().nth(1).expect("usage: bcn_difftest <vectors.bin>");
    let data = fs::read(&path).unwrap();
    // Codec, how many channels the reference wrote per texel, and how this
    // decoder's channels line up with them. BC6H's are floats; the rest bytes.
    struct Group {
        codec: Codec,
        channels: usize,
        mapping: &'static [usize],
        signed_bit: bool,
    }
    let groups = [
        Group { codec: Codec::Bc1, channels: 4, mapping: &[0, 1, 2, 3], signed_bit: false },
        Group { codec: Codec::Bc2, channels: 4, mapping: &[0, 1, 2, 3], signed_bit: false },
        Group { codec: Codec::Bc3, channels: 4, mapping: &[0, 1, 2, 3], signed_bit: false },
        Group { codec: Codec::Bc7, channels: 4, mapping: &[0, 1, 2, 3], signed_bit: false },
        Group { codec: Codec::Bc4Unorm, channels: 1, mapping: &[0], signed_bit: false },
        Group { codec: Codec::Bc5Unorm, channels: 2, mapping: &[0, 1], signed_bit: false },
        Group { codec: Codec::Bc6hUf16, channels: 3, mapping: &[0, 1, 2], signed_bit: true },
        Group { codec: Codec::Bc6hSf16, channels: 3, mapping: &[0, 1, 2], signed_bit: true },
    ];
    let blocks: usize = env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(4000);

    let mut at = 0usize;
    let mut any_out_of_tolerance = false;
    for group in groups {
        // The HDR codecs are compared as floats, the rest as 8-bit channels.
        let float_record = group.signed_bit;
        let element = if float_record { 4 } else { 1 };
        let record = 16 + group.channels * 16 * element;
        let mut worst_int = 0i32;
        let mut worst_rel = 0f64;
        let mut differing = 0u32;
        for _ in 0..blocks {
            if at + record > data.len() {
                break;
            }
            let input = &data[at..at + 16];
            let expected = &data[at + 16..at + record];
            at += record;
            let bytes = group.codec.bytes_per_block() as usize;
            let got = decode(group.codec, &input[..bytes]).unwrap();
            let mut differs = false;
            for (texel, rgba) in got.iter().enumerate() {
                for (slot, &channel) in group.mapping.iter().enumerate() {
                    let index = texel * group.channels + slot;
                    if float_record {
                        let raw = &expected[index * 4..index * 4 + 4];
                        let theirs =
                            f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as f64;
                        let mine = rgba[channel] as f64;
                        if mine == theirs || (mine.is_nan() && theirs.is_nan()) {
                            continue;
                        }
                        let scale = theirs.abs().max(mine.abs()).max(1e-6);
                        let relative = (mine - theirs).abs() / scale;
                        worst_rel = worst_rel.max(relative);
                        differs = true;
                    } else {
                        let mine = (rgba[channel] * 255.0 + 0.5) as i32;
                        let theirs = expected[index] as i32;
                        let delta = (mine - theirs).abs();
                        if delta > 0 {
                            worst_int = worst_int.max(delta);
                            differs = true;
                        }
                    }
                }
            }
            if differs {
                differing += 1;
                if env::var("BCN_DUMP").is_ok() && differing == 1 {
                    println!("  first mismatch, input {input:02x?}");
                    let first_bad = (0..16).find(|&texel| {
                        group.mapping.iter().enumerate().any(|(slot, &c)| {
                            let index = texel * group.channels + slot;
                            let theirs = if float_record {
                                let r = &expected[index * 4..index * 4 + 4];
                                f32::from_le_bytes([r[0], r[1], r[2], r[3]])
                            } else {
                                expected[index] as f32 / 255.0
                            };
                            (got[texel][c] - theirs).abs() > 1e-9
                        })
                    });
                    println!("    first differing texel: {first_bad:?}");
                    for texel in first_bad.into_iter() {
                        let mine: Vec<f32> =
                            group.mapping.iter().map(|&c| got[texel][c]).collect();
                        let theirs: Vec<f32> = (0..group.channels)
                            .map(|slot| {
                                let index = texel * group.channels + slot;
                                if float_record {
                                    let r = &expected[index * 4..index * 4 + 4];
                                    f32::from_le_bytes([r[0], r[1], r[2], r[3]])
                                } else {
                                    expected[index] as f32 / 255.0
                                }
                            })
                            .collect();
                        println!("    texel {texel}: mine {mine:?} theirs {theirs:?}");
                    }
                }
            }
        }
        let verdict = if float_record {
            if worst_rel == 0.0 {
                "exact".to_string()
            } else {
                any_out_of_tolerance = true;
                format!("worst relative error {worst_rel:e}")
            }
        } else {
            match worst_int {
                0 => "exact".to_string(),
                1 => "within 1/255".to_string(),
                other => {
                    any_out_of_tolerance = true;
                    format!("OUT OF TOLERANCE by {other}")
                }
            }
        };
        println!("{:?}: {differing}/{blocks} blocks differ [{verdict}]", group.codec);
    }
    if let Some(path) = env::args().nth(3) {
        any_out_of_tolerance |= !compare_astc(&fs::read(&path).unwrap());
    }
    if any_out_of_tolerance {
        println!("\nsomething is outside tolerance — investigate");
    } else {
        println!("\nevery codec agrees with the reference");
    }
}

/// The ASTC fixture is grouped by footprint: each group opens with its block
/// width, height and count as three little-endian i32, then that many records
/// of sixteen input bytes and one RGBA8 texel per texel of the footprint.
///
/// The reference emits bytes from its own float pipeline as
/// `clamp(v * 65536 + 0.5, 0, 65535) >> 8`, so the comparison applies exactly
/// that to this decoder's output rather than rounding some other way.
fn compare_astc(data: &[u8]) -> bool {
    let word = |at: usize| i32::from_le_bytes(data[at..at + 4].try_into().unwrap());
    let mut at = 0usize;
    let mut all_exact = true;
    while at + 12 <= data.len() {
        let (bw, bh, count) = (word(at) as usize, word(at + 4) as usize, word(at + 8) as usize);
        at += 12;
        let texels = bw * bh;
        let record = 16 + texels * 4;
        let codec = Codec::Astc { width: bw as u8, height: bh as u8 };
        let mut differing = 0u32;
        let mut worst = 0i32;
        for _ in 0..count {
            if at + record > data.len() {
                break;
            }
            let input = &data[at..at + 16];
            let expected = &data[at + 16..at + record];
            at += record;
            let mut got = [[0.0f32; 4]; switch_core::gpu::bcn::MAX_TEXELS];
            decode_into(codec, input, &mut got).unwrap();
            let mut differs = false;
            for texel in 0..texels {
                for channel in 0..4 {
                    let mine = ((got[texel][channel] * 65536.0 + 0.5) as i32).clamp(0, 65535) >> 8;
                    let theirs = expected[texel * 4 + channel] as i32;
                    if mine != theirs {
                        differs = true;
                        worst = worst.max((mine - theirs).abs());
                    }
                }
            }
            if differs {
                differing += 1;
            }
        }
        let verdict = if worst == 0 { "exact".to_string() } else { format!("worst delta {worst}") };
        println!("ASTC {bw}x{bh}: {differing}/{count} blocks differ [{verdict}]");
        all_exact &= worst == 0;
    }
    all_exact
}
