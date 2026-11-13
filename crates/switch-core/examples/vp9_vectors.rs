//! Check the transforms and intra predictors against libvpx's C output, from a file written by a
//! small harness linked to `libvpx.a`: `vp9_vectors idct|intra <file>`.
//! `idct`: random blocks through `vp9_iht*_add_c`, `vpx_idct32x32_1024_add_c` and `vpx_iwht4x4_16_add_c`.
//! `intra`: `vpx_*_predictor_*_c`.
use clap::{Parser, ValueEnum};
use switch_core::vp9::idct::{inverse_transform_add, inverse_wht_add};
use switch_core::vp9::intra::{edges_from, predict_from};

#[derive(Clone, ValueEnum)]
enum Kind {
    Idct,
    Intra,
}

#[derive(Parser)]
struct Cli {
    kind: Kind,
    file: std::path::PathBuf,
}

fn main() {
    let cli = Cli::parse();
    let data = std::fs::read(&cli.file).expect("read vectors");
    let mut at = 0;
    let mut cases = 0;
    while at < data.len() {
        match cli.kind {
            Kind::Idct => {
                let (size, kind, lossless) = (data[at], data[at + 1], data[at + 2] != 0);
                at += 4;
                let n = 4usize << size;
                let count = n * n;
                let coeffs: Vec<i32> = data[at..at + count * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| i32::from_le_bytes(*b))
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
                assert!(
                    first.is_none(),
                    "case {cases}: size {size}, type {kind}, lossless {lossless}, first mismatch at {first:?}"
                );
            }
            Kind::Intra => {
                let (mode, size, have_left, have_top) =
                    (data[at], data[at + 1], data[at + 2] != 0, data[at + 3] != 0);
                at += 4;
                let bs = 4usize << size;
                let above = &data[at..at + 1 + 2 * bs];
                at += 1 + 2 * bs;
                let left = &data[at..at + bs];
                at += bs;
                let expect = &data[at..at + bs * bs];
                at += bs * bs;
                let mut plane = vec![0u8; bs * bs];
                predict_from(
                    &mut plane,
                    0,
                    bs,
                    mode,
                    size,
                    &edges_from(above, left),
                    have_left,
                    have_top,
                );
                let first = plane.iter().zip(expect).position(|(a, b)| a != b);
                assert!(
                    first.is_none(),
                    "case {cases}: mode {mode}, size {size}, left {have_left}, top {have_top}, first mismatch at {first:?}"
                );
            }
        }
        cases += 1;
    }
    assert!(cases > 0, "empty vector file");
    println!("{cases} cases match");
}
