//! Compare this crate's Opus decoder against a reference decode, frame by
//! frame.
//!
//! Usage: `cargo run --release -p switch-core --example opus_difftest -- <dir>`
//!
//! `<dir>` holds pairs of files per test case. `<name>.opus` is a header of
//! `{output rate, channels, 48 kHz frame size, streams, coupled streams}`
//! and eight bytes of channel mapping — a stream count of zero means an
//! ordinary single-stream decoder — followed, per frame, by a record of
//! `{packet length, samples decoded, final range}` and the packet's bytes.
//! `<name>.pcm` is the reference decode as interleaved `f32`. A length of
//! zero is a dropped packet, which both decoders conceal.
//!
//! Two things are checked, and they fail differently. The **final range** is
//! the range coder's state after the frame: it has to match exactly, because
//! it proves both decoders read the same symbols in the same order — a
//! mismatch means the bitstream was misparsed and everything after it is
//! meaningless. The **samples** need only be close: CELT is specified in
//! floating point and the reference's own arithmetic is not reproducible bit
//! for bit, so what matters is that the error stays far below the
//! quantisation noise the codec already has.

use std::env;
use std::fs;
use std::path::Path;
use switch_core::opus::{Decoder, MultiStreamDecoder};

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn main() {
    let dir = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: opus_difftest <vector directory>");
        std::process::exit(2);
    });
    let mut names: Vec<String> = fs::read_dir(Path::new(&dir))
        .expect("read vector directory")
        .filter_map(|e| {
            let path = e.ok()?.path();
            (path.extension()? == "opus").then(|| path.file_stem()?.to_str().map(str::to_owned))?
        })
        .collect();
    names.sort();

    let mut failures = 0usize;
    for name in &names {
        let packets = fs::read(format!("{dir}/{name}.opus")).expect("read packets");
        let reference = fs::read(format!("{dir}/{name}.pcm")).expect("read reference pcm");

        let rate = read_u32(&packets, 0);
        let channels = read_u32(&packets, 4) as usize;
        let frame48 = read_u32(&packets, 8) as usize;
        let streams = read_u32(&packets, 12) as usize;
        let coupled = read_u32(&packets, 16) as usize;
        let mapping = &packets[20..28];
        let frame_size = frame48 * rate as usize / 48000;

        let mut single = (streams == 0).then(|| Decoder::new(rate, channels).expect("decoder"));
        let mut multi = (streams != 0).then(|| {
            MultiStreamDecoder::new(rate, channels, streams, coupled, mapping).expect("decoder")
        });
        let mut pcm = vec![0.0f32; frame_size * channels * 2];
        let mut at = 28usize;
        let mut ref_at = 0usize;
        let mut frame = 0usize;
        let mut max_error = 0.0f32;
        let mut max_error_frame = 0usize;
        let mut max_error_lost = false;
        let mut sum_sq = 0.0f64;
        let mut sum_ref_sq = 0.0f64;
        let mut counted = 0usize;
        let mut range_mismatch: Option<(usize, u32, u32)> = None;
        let mut wrong_length: Option<(usize, usize, usize)> = None;

        while at + 12 <= packets.len() {
            let len = read_u32(&packets, at) as usize;
            let expected_samples = read_u32(&packets, at + 4) as usize;
            let expected_range = read_u32(&packets, at + 8);
            at += 12;
            let packet = (len > 0).then(|| &packets[at..at + len]);
            at += len;

            let decoded = match (single.as_mut(), multi.as_mut()) {
                (Some(decoder), _) => decoder.decode_float(packet, &mut pcm, frame_size),
                (_, Some(decoder)) => decoder.decode_float(packet, &mut pcm, frame_size),
                _ => unreachable!(),
            };
            let final_range = match (single.as_ref(), multi.as_ref()) {
                (Some(decoder), _) => decoder.final_range(),
                (_, Some(decoder)) => decoder.final_range(),
                _ => unreachable!(),
            };
            let got = match decoded {
                Ok(got) => got,
                Err(err) => {
                    println!("{name:<24} frame {frame}: decode failed: {err:?}");
                    failures += 1;
                    break;
                }
            };
            if got != expected_samples && wrong_length.is_none() {
                wrong_length = Some((frame, got, expected_samples));
            }
            if final_range != expected_range && range_mismatch.is_none() {
                range_mismatch = Some((frame, final_range, expected_range));
            }

            for &sample in &pcm[..got * channels] {
                let expected = f32::from_le_bytes([
                    reference[ref_at],
                    reference[ref_at + 1],
                    reference[ref_at + 2],
                    reference[ref_at + 3],
                ]);
                ref_at += 4;
                let diff = sample - expected;
                if diff.abs() > max_error {
                    max_error = diff.abs();
                    max_error_frame = frame;
                    max_error_lost = len == 0;
                }
                sum_sq += f64::from(diff) * f64::from(diff);
                sum_ref_sq += f64::from(expected) * f64::from(expected);
                counted += 1;
            }
            frame += 1;
        }

        let rms = (sum_sq / counted.max(1) as f64).sqrt();
        let ref_rms = (sum_ref_sq / counted.max(1) as f64).sqrt();
        let snr = if rms > 0.0 {
            20.0 * (ref_rms / rms).log10()
        } else {
            f64::INFINITY
        };
        let mut verdict = "ok";
        if let Some((frame, got, want)) = wrong_length {
            println!("{name:<24} frame {frame}: {got} samples, expected {want}");
            verdict = "FAIL";
        }
        if let Some((frame, got, want)) = range_mismatch {
            println!("{name:<24} frame {frame}: final range {got:#010x}, expected {want:#010x}");
            verdict = "FAIL";
        }
        if verdict == "FAIL" {
            failures += 1;
        }
        println!("{name:<24} {verdict:<4} frames={frame:<5} max_err={max_error:.6} (frame {max_error_frame}, lost={max_error_lost}) snr={snr:6.1} dB");
    }

    if failures > 0 {
        eprintln!("\n{failures} of {} cases failed", names.len());
        std::process::exit(1);
    }
    println!("\nall {} cases match", names.len());
}
