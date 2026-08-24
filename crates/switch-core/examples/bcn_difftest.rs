//! Compare the BC decoders against an independent implementation:
//! `bcn_difftest <vectors.bin>`.
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
use switch_core::gpu::bcn::{decode, Codec};

fn main() {
    let path = env::args().nth(1).expect("usage: bcn_difftest <vectors.bin>");
    let data = fs::read(&path).unwrap();
    // Codec, channels the reference wrote per texel, and how this decoder's
    // channels line up with them.
    let groups: [(Codec, usize, &[usize]); 6] = [
        (Codec::Bc1, 4, &[0, 1, 2, 3]),
        (Codec::Bc2, 4, &[0, 1, 2, 3]),
        (Codec::Bc3, 4, &[0, 1, 2, 3]),
        (Codec::Bc7, 4, &[0, 1, 2, 3]),
        (Codec::Bc4Unorm, 1, &[0]),
        (Codec::Bc5Unorm, 2, &[0, 1]),
    ];
    let blocks: usize = env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4000);

    let mut at = 0usize;
    let mut worst_overall = 0i32;
    for (codec, channels, mapping) in groups {
        let record = 16 + channels * 16;
        let mut worst = 0i32;
        let mut differing = 0u32;
        for _ in 0..blocks {
            if at + record > data.len() {
                break;
            }
            let input = &data[at..at + 16];
            let expected = &data[at + 16..at + record];
            at += record;
            let bytes = codec.bytes_per_block() as usize;
            let got = decode(codec, &input[..bytes]).unwrap();
            let mut block_worst = 0i32;
            for (texel, rgba) in got.iter().enumerate() {
                for (slot, &channel) in mapping.iter().enumerate() {
                    let mine = (rgba[channel] * 255.0 + 0.5) as i32;
                    let theirs = expected[texel * channels + slot] as i32;
                    block_worst = block_worst.max((mine - theirs).abs());
                }
            }
            if block_worst > 0 {
                differing += 1;
            }
            worst = worst.max(block_worst);
        }
        worst_overall = worst_overall.max(worst);
        let verdict = match worst {
            0 => "exact",
            1 => "within 1/255",
            _ => "OUT OF TOLERANCE",
        };
        println!("{codec:?}: {differing}/{blocks} blocks differ, worst channel delta {worst} [{verdict}]");
    }
    println!(
        "\nworst delta across every codec: {worst_overall} ({})",
        if worst_overall <= 1 { "acceptable" } else { "INVESTIGATE" }
    );
}
