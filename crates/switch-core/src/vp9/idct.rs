//! 2-D inverse transforms: rows, then columns, then the rounding shift and clipped add.

use super::transforms_1d::{iadst16, iadst4, iadst8, idct16, idct32, idct4, idct8};

type Transform1d = fn(&[i32], &mut [i32]);

pub const DCT_DCT: u8 = 0;
pub const ADST_DCT: u8 = 1;
pub const DCT_ADST: u8 = 2;
pub const ADST_ADST: u8 = 3;

fn clip_add(dest: &mut u8, residual: i32) {
    *dest = (i32::from(*dest) + residual).clamp(0, 255) as u8;
}

fn round_shift(x: i32, bits: u32) -> i32 {
    (x + (1 << (bits - 1))) >> bits
}

fn inverse<const N: usize>(
    input: &[i32],
    dest: &mut [u8],
    stride: usize,
    rows: Transform1d,
    cols: Transform1d,
    shift: u32,
    skip_zero_rows: bool,
) {
    let mut out = vec![0i32; N * N];
    for i in 0..N {
        let row = &input[i * N..(i + 1) * N];
        if skip_zero_rows && row.iter().all(|&c| c as i16 == 0) {
            continue;
        }
        rows(row, &mut out[i * N..(i + 1) * N]);
    }
    let mut column = [0i32; 32];
    let mut result = [0i32; 32];
    for i in 0..N {
        for j in 0..N {
            column[j] = out[j * N + i];
        }
        cols(&column[..N], &mut result[..N]);
        for j in 0..N {
            clip_add(&mut dest[j * stride + i], round_shift(result[j], shift));
        }
    }
}

/// Adds the inverse transform of `coeffs` (raster order) to the `size`-square block at `dest`.
pub fn inverse_transform_add(
    tx_size: u8,
    tx_type: u8,
    coeffs: &[i32],
    dest: &mut [u8],
    stride: usize,
) {
    match tx_size {
        0 => {
            let (cols, rows): (Transform1d, Transform1d) = match tx_type {
                ADST_DCT => (iadst4, idct4),
                DCT_ADST => (idct4, iadst4),
                ADST_ADST => (iadst4, iadst4),
                _ => (idct4, idct4),
            };
            inverse::<4>(coeffs, dest, stride, rows, cols, 4, false);
        }
        1 => {
            let (cols, rows): (Transform1d, Transform1d) = match tx_type {
                ADST_DCT => (iadst8, idct8),
                DCT_ADST => (idct8, iadst8),
                ADST_ADST => (iadst8, iadst8),
                _ => (idct8, idct8),
            };
            inverse::<8>(coeffs, dest, stride, rows, cols, 5, false);
        }
        2 => {
            let (cols, rows): (Transform1d, Transform1d) = match tx_type {
                ADST_DCT => (iadst16, idct16),
                DCT_ADST => (idct16, iadst16),
                ADST_ADST => (iadst16, iadst16),
                _ => (idct16, idct16),
            };
            inverse::<16>(coeffs, dest, stride, rows, cols, 6, false);
        }
        _ => inverse::<32>(coeffs, dest, stride, idct32, idct32, 6, true),
    }
}

/// The lossless 4x4 inverse Walsh-Hadamard transform.
pub fn inverse_wht_add(coeffs: &[i32], dest: &mut [u8], stride: usize) {
    let mut out = [0i32; 16];
    for i in 0..4 {
        let ip = &coeffs[i * 4..i * 4 + 4];
        let mut a1 = ip[0] >> 2;
        let mut c1 = ip[1] >> 2;
        let mut d1 = ip[2] >> 2;
        let mut b1 = ip[3] >> 2;
        a1 += c1;
        d1 -= b1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        out[i * 4..i * 4 + 4].copy_from_slice(&[a1, b1, c1, d1]);
    }
    for i in 0..4 {
        let mut a1 = out[i];
        let mut c1 = out[4 + i];
        let mut d1 = out[8 + i];
        let mut b1 = out[12 + i];
        a1 += c1;
        d1 -= b1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        for (j, v) in [a1, b1, c1, d1].into_iter().enumerate() {
            clip_add(&mut dest[j * stride + i], v);
        }
    }
}
