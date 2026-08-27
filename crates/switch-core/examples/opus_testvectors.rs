//! Decode the official Opus test vectors and write the result out for
//! `opus_compare`.
//!
//! Usage: `cargo run --release -p switch-core --example opus_testvectors -- <dir> [out dir]`
//!
//! `<dir>` is an unpacked `opus_testvectors` from opus-codec.org. Each
//! `testvectorNN.bit` is the `opus_demo` container: per packet, a 32-bit
//! big-endian length, the 32-bit big-endian range coder state the encoder
//! ended that packet with, and the packet itself. The matching `.dec` is the
//! reference decode at 48 kHz stereo.
//!
//! Each vector is decoded twice, once to stereo and once to mono — the
//! second exercises the downmix a stereo stream takes on its way to a mono
//! output, which nothing else does.
//!
//! Two results come out of this. The **final range** must match on every
//! packet — that is the format's own proof that a decoder read the same
//! symbols the encoder wrote, and RFC 6716 makes it normative. The decoded
//! samples are written to `<out dir>/testvectorNN.rs.dec` and
//! `…NNm.rs.dec`, which is what `opus_compare` scores; a floating-point
//! decoder is not expected to be bit identical, only close enough that
//! `opus_compare` passes.

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use switch_core::opus::Decoder;

/// Longest frame Opus can carry: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5760;

fn main() {
    let mut args = env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| {
        eprintln!("usage: opus_testvectors <vector directory> [output directory]");
        std::process::exit(2);
    });
    let out_dir = args.next().unwrap_or_else(|| dir.clone());

    let mut names: Vec<String> = fs::read_dir(Path::new(&dir))
        .expect("read vector directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            (path.extension()? == "bit").then(|| path.file_stem()?.to_str().map(str::to_owned))?
        })
        .collect();
    names.sort();

    let mut failures = 0usize;
    let mut decoded_samples = 0u64;
    let mut decode_time = std::time::Duration::ZERO;
    for name in &names {
        let bits = fs::read(format!("{dir}/{name}.bit")).expect("read bitstream");
        for channels in [2usize, 1] {
        let mut decoder = Decoder::new(48000, channels).expect("decoder");
        let mut pcm = vec![0.0f32; MAX_FRAME * channels];
        let mut out = Vec::with_capacity(bits.len() * 16);
        let mut at = 0usize;
        let mut packet = 0usize;
        let mut mismatch: Option<(usize, u32, u32)> = None;
        let mut error: Option<String> = None;

        while at + 8 <= bits.len() {
            let len = u32::from_be_bytes([bits[at], bits[at + 1], bits[at + 2], bits[at + 3]]) as usize;
            let expected_range =
                u32::from_be_bytes([bits[at + 4], bits[at + 5], bits[at + 6], bits[at + 7]]);
            at += 8;
            if len > bits.len() - at {
                break;
            }
            let started = Instant::now();
            let result = decoder.decode_float(Some(&bits[at..at + len]), &mut pcm, MAX_FRAME);
            decode_time += started.elapsed();
            match result {
                Ok(got) => {
                    decoded_samples += got as u64;
                    for &sample in &pcm[..got * channels] {
                        let scaled = (sample * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
                        out.extend_from_slice(&scaled.to_le_bytes());
                    }
                }
                Err(err) => {
                    error = Some(format!("{err:?}"));
                    break;
                }
            }
            if decoder.final_range() != expected_range && mismatch.is_none() {
                mismatch = Some((packet, decoder.final_range(), expected_range));
            }
            at += len;
            packet += 1;
        }

        let suffix = if channels == 1 { "m" } else { "" };
        let path = format!("{out_dir}/{name}{suffix}.rs.dec");
        fs::File::create(&path).expect("create output").write_all(&out).expect("write output");

        match (&error, mismatch) {
            (Some(err), _) => {
                println!("{name}{suffix} FAIL packet {packet}: {err}");
                failures += 1;
            }
            (None, Some((packet, got, want))) => {
                println!("{name}{suffix} FAIL packet {packet}: final range {got:#010x}, expected {want:#010x}");
                failures += 1;
            }
            (None, None) => {
                println!("{name}{suffix} ok   {packet} packets, {} samples -> {path}", out.len() / (2 * channels))
            }
        }
        }
    }

    // What this costs, against the audio it produced. A decoder that cannot
    // stay ahead of its own output is one the emulator cannot use.
    let audio = decoded_samples as f64 / 48000.0;
    let elapsed = decode_time.as_secs_f64();
    println!("\ndecoded {audio:.1} s of audio in {elapsed:.2} s ({:.0}x real time)", audio / elapsed);

    if failures > 0 {
        eprintln!("\n{failures} of {} vectors failed", names.len());
        std::process::exit(1);
    }
}
