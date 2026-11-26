//! Time the VP9 decoder on an IVF clip: `vp9_bench <clip.ivf> [--loops <n>]`.
use clap::Parser;
use std::time::Instant;
use switch_vp9::Decoder;

#[derive(Parser)]
struct Cli {
    clip: std::path::PathBuf,
    #[arg(long, default_value_t = 3)]
    loops: u32,
}

fn main() {
    let cli = Cli::parse();
    let ivf = std::fs::read(&cli.clip).expect("read clip");
    for _ in 0..cli.loops {
        let mut decoder = Decoder::new();
        let (mut at, mut frames) = (32, 0);
        let start = Instant::now();
        while at + 12 <= ivf.len() {
            let size = u32::from_le_bytes(ivf[at..at + 4].try_into().unwrap()) as usize;
            at += 12;
            frames += decoder
                .decode_packet(&ivf[at..at + size])
                .expect("decode")
                .len();
            at += size;
        }
        let secs = start.elapsed().as_secs_f64();
        println!(
            "{frames} frames in {secs:.3}s, {:.1} fps",
            frames as f64 / secs
        );
    }
}
