//! PVQ codewords and the shape of one coded vector.

use crate::opus::range::RangeDecoder;
use crate::opus::tables_celt::*;
use super::SPREAD_NONE;

/// `U(n, k)`: PVQ codewords with `k` pulses in `n` dimensions and a positive
/// first pulse. Symmetric, stored as ragged rows.
fn pvq_u(n: usize, k: usize) -> u32 {
    let (lo, hi) = if n < k { (n, k) } else { (k, n) };
    PVQ_U_DATA[PVQ_U_ROW[lo] + hi]
}

fn pvq_v(n: usize, k: usize) -> u32 {
    pvq_u(n, k).wrapping_add(pvq_u(n, k + 1))
}

/// Decode a codeword index into its pulse vector, returning its squared norm.
fn cwrsi(mut n: usize, mut k: usize, mut i: u32, y: &mut [i32]) -> f32 {
    let mut yy = 0.0f32;
    let mut at = 0usize;
    while n > 2 {
        let (p, s, k0);
        if k >= n {
            let row = PVQ_U_ROW[n];
            let pv = PVQ_U_DATA[row + k + 1];
            s = if i >= pv { -1i32 } else { 0 };
            i -= pv & (s as u32);
            k0 = k;
            let q = PVQ_U_DATA[row + n];
            if q > i {
                k = n;
                loop {
                    k -= 1;
                    if PVQ_U_DATA[PVQ_U_ROW[k] + n] <= i {
                        break;
                    }
                }
                p = PVQ_U_DATA[PVQ_U_ROW[k] + n];
            } else {
                let mut pp = PVQ_U_DATA[row + k];
                while pp > i {
                    k -= 1;
                    pp = PVQ_U_DATA[row + k];
                }
                p = pp;
            }
            i -= p;
        } else {
            let pv = pvq_u(k, n);
            let q = pvq_u(k + 1, n);
            if pv <= i && i < q {
                i -= pv;
                y[at] = 0;
                at += 1;
                n -= 1;
                continue;
            }
            s = if i >= q { -1i32 } else { 0 };
            i -= q & (s as u32);
            k0 = k;
            loop {
                k -= 1;
                if pvq_u(k, n) <= i {
                    break;
                }
            }
            p = pvq_u(k, n);
            i -= p;
        }
        let val = ((k0 - k) as i32 + s) ^ s;
        y[at] = val;
        at += 1;
        yy += (val * val) as f32;
        n -= 1;
    }
    // n == 2: the ranking is linear from here.
    let p = 2 * k as u32 + 1;
    let s = if i >= p { -1i32 } else { 0 };
    i -= p & (s as u32);
    let k0 = k;
    k = ((i + 1) >> 1) as usize;
    if k != 0 {
        i -= 2 * k as u32 - 1;
    }
    let val = ((k0 - k) as i32 + s) ^ s;
    y[at] = val;
    at += 1;
    yy += (val * val) as f32;
    // n == 1: whatever is left, and its sign.
    let s = -(i as i32);
    let val = (k as i32 + s) ^ s;
    y[at] = val;
    yy += (val * val) as f32;
    yy
}

fn decode_pulses(y: &mut [i32], n: usize, k: usize, dec: &mut RangeDecoder) -> f32 {
    let index = dec.decode_uint(pvq_v(n, k));
    cwrsi(n, k, index, y)
}

/// Spread a few pulses across a band; the inverse of the encoder's rotation.
fn exp_rotation(x: &mut [f32], len: usize, stride: usize, k: i32, spread: usize) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    if 2 * k >= len as i32 || spread == SPREAD_NONE {
        return;
    }
    let factor = SPREAD_FACTOR[spread - 1];
    let gain = len as f32 / (len as i32 + factor * k) as f32;
    let theta = 0.5 * gain * gain;
    let c = (0.5 * core::f32::consts::PI * theta).cos();
    let s = (0.5 * core::f32::consts::PI * (1.0 - theta)).cos();

    let mut stride2 = 0usize;
    if len >= 8 * stride {
        stride2 = 1;
        // sqrt(len/stride), rounded: grow while (stride2+0.5)^2 fits.
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }
    let block = len / stride;
    for i in 0..stride {
        let part = &mut x[i * block..(i + 1) * block];
        if stride2 != 0 {
            exp_rotation1(part, block, stride2, s, c);
        }
        exp_rotation1(part, block, 1, c, s);
    }
}

fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    let ms = -s;
    for i in 0..len - stride {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }
    for i in (0..len.saturating_sub(2 * stride)).rev() {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }
}

/// Which sub-blocks got a pulse; empty ones are refilled by [`anti_collapse`].
fn extract_collapse_mask(y: &[i32], n: usize, blocks: usize) -> u32 {
    if blocks <= 1 {
        return 1;
    }
    let size = n / blocks;
    let mut mask = 0u32;
    for i in 0..blocks {
        let any = y[i * size..(i + 1) * size].iter().any(|&v| v != 0);
        mask |= u32::from(any) << i;
    }
    mask
}

pub(super) fn renormalise_vector(x: &mut [f32], gain: f32) {
    let energy: f32 = 1e-15 + x.iter().map(|&v| v * v).sum::<f32>();
    let g = gain / energy.sqrt();
    for v in x.iter_mut() {
        *v *= g;
    }
}

/// Decode a band's PVQ shape, scaled to `gain`.
pub(super) fn alg_unquant(
    x: &mut [f32],
    n: usize,
    k: i32,
    spread: usize,
    blocks: usize,
    dec: &mut RangeDecoder,
    gain: f32,
) -> u32 {
    let mut iy = vec![0i32; n];
    let ryy = decode_pulses(&mut iy, n, k as usize, dec);
    let g = gain / ryy.sqrt();
    for i in 0..n {
        x[i] = g * iy[i] as f32;
    }
    exp_rotation(x, n, blocks, k, spread);
    extract_collapse_mask(&iy, n, blocks)
}
