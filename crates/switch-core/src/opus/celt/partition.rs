//! One band of one channel, split into partitions.

use crate::opus::range::{ilog, RangeDecoder, BITRES};
use crate::opus::tables_celt::*;
use super::allocation::{bits2pulses, cache_row, get_pulses, pulses2bits};
use super::vq::{alg_unquant, renormalise_vector};
use super::{QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE};

/// A bit-exact cosine; the bit allocation depends on it.
fn bitexact_cos(x: i16) -> i32 {
    let tmp = (4096 + i32::from(x) * i32::from(x)) >> 13;
    let x2 = (32767 - tmp) + frac_mul16(tmp, -7651 + frac_mul16(tmp, 8277 + frac_mul16(-626, tmp)));
    1 + x2
}

fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ilog(icos as u32) as i32;
    let ls = ilog(isin as u32) as i32;
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// Rounded Q15 multiply of 16-bit-truncated operands, as the reference does.
fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + (a as i16 as i32) * (b as i16 as i32)) >> 15
}

/// CELT's LCG for filling empty bands; its sequence is part of the format.
pub(super) fn lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1664525).wrapping_add(1013904223)
}

/// One Haar step, trading frequency for time resolution in a band.
fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    const SQRT_HALF: f32 = 0.70710678;
    let half = n0 >> 1;
    for i in 0..stride {
        for j in 0..half {
            let a = SQRT_HALF * x[stride * 2 * j + i];
            let b = SQRT_HALF * x[stride * (2 * j + 1) + i];
            x[stride * 2 * j + i] = a + b;
            x[stride * (2 * j + 1) + i] = a - b;
        }
    }
}

fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    for i in 0..stride {
        let dest = if hadamard {
            ORDERY_TABLE[stride - 2 + i]
        } else {
            i
        };
        for j in 0..n0 {
            tmp[dest * n0 + j] = x[j * stride + i];
        }
    }
    x[..n].copy_from_slice(&tmp);
}

fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    for i in 0..stride {
        let src = if hadamard {
            ORDERY_TABLE[stride - 2 + i]
        } else {
            i
        };
        for j in 0..n0 {
            tmp[j * stride + i] = x[src * n0 + j];
        }
    }
    x[..n].copy_from_slice(&tmp);
}

/// How finely the mid/side angle may be coded, given the bits available.
fn compute_qn(n: usize, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    let mut n2 = 2 * n as i32 - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    // The cap stops a hard-over stereo angle leaving the side with no bits.
    let mut qb = (b + n2 * offset) / n2;
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES) >> 1 {
        1
    } else {
        let qn = EXP2_TABLE8[(qb & 0x7) as usize] >> (14 - (qb >> BITRES));
        (qn + 1) >> 1 << 1
    }
}

pub(super) struct SplitCtx {
    pub(super) inv: bool,
    pub(super) imid: i32,
    pub(super) iside: i32,
    pub(super) delta: i32,
    pub(super) itheta: i32,
    pub(super) qalloc: i32,
}

pub(super) struct BandCtx<'a, 'p> {
    pub(super) dec: &'a mut RangeDecoder<'p>,
    pub(super) band: usize,
    pub(super) intensity: usize,
    pub(super) spread: usize,
    pub(super) tf_change: i32,
    pub(super) remaining_bits: i32,
    pub(super) seed: u32,
    pub(super) disable_inv: bool,
}

/// Decode the split angle and divide the bits between the halves.
pub(super) fn compute_theta(
    ctx: &mut BandCtx,
    n: usize,
    b: &mut i32,
    blocks: usize,
    b0: usize,
    lm: i32,
    stereo: bool,
    fill: &mut i32,
) -> SplitCtx {
    let pulse_cap = i32::from(LOG_N400[ctx.band]) + lm * (1 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);
    if stereo && ctx.band >= ctx.intensity {
        qn = 1;
    }
    let tell = ctx.dec.tell_frac() as i32;
    let mut itheta = 0i32;
    let mut inv = false;

    if qn != 1 {
        // Uniform pdf for time splits, step for stereo, triangular otherwise.
        if stereo && n > 2 {
            let p0 = 3u32;
            let x0 = (qn / 2) as u32;
            let ft = p0 * (x0 + 1) + x0;
            let fs = ctx.dec.decode(ft);
            let value = if fs < (x0 + 1) * p0 {
                fs / p0
            } else {
                x0 + 1 + (fs - (x0 + 1) * p0)
            };
            let (fl, fh) = if value <= x0 {
                (p0 * value, p0 * (value + 1))
            } else {
                (
                    (value - 1 - x0) + (x0 + 1) * p0,
                    (value - x0) + (x0 + 1) * p0,
                )
            };
            ctx.dec.update(fl, fh, ft);
            itheta = value as i32;
        } else if b0 > 1 || stereo {
            itheta = ctx.dec.decode_uint(qn as u32 + 1) as i32;
        } else {
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            let fm = ctx.dec.decode(ft as u32) as i32;
            let (fl, fs);
            if fm < ((qn >> 1) * ((qn >> 1) + 1)) >> 1 {
                itheta = (isqrt32(8 * fm as u32 + 1) as i32 - 1) >> 1;
                fs = itheta + 1;
                fl = (itheta * (itheta + 1)) >> 1;
            } else {
                itheta = (2 * (qn + 1) - isqrt32(8 * (ft - fm - 1) as u32 + 1) as i32) >> 1;
                fs = qn + 1 - itheta;
                fl = ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
            }
            ctx.dec.update(fl as u32, (fl + fs) as u32, ft as u32);
        }
        debug_assert!(itheta >= 0);
        itheta = (itheta * 16384) / qn;
    } else if stereo {
        // With no angle to code, the side may still have been inverted.
        inv = if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            ctx.dec.decode_bit_logp(2)
        } else {
            false
        };
        if ctx.disable_inv {
            inv = false;
        }
        itheta = 0;
    }
    let qalloc = ctx.dec.tell_frac() as i32 - tell;
    *b -= qalloc;

    let (imid, iside, delta);
    if itheta == 0 {
        imid = 32767;
        iside = 0;
        *fill &= (1 << blocks) - 1;
        delta = -16384;
    } else if itheta == 16384 {
        imid = 0;
        iside = 32767;
        *fill &= ((1 << blocks) - 1) << blocks;
        delta = 16384;
    } else {
        imid = bitexact_cos(itheta as i16);
        iside = bitexact_cos((16384 - itheta) as i16);
        // The mid/side bit split that minimises squared error in this band.
        delta = frac_mul16((n as i32 - 1) << 7, bitexact_log2tan(iside, imid));
    }
    SplitCtx {
        inv,
        imid,
        iside,
        delta,
        itheta,
        qalloc,
    }
}

/// Integer square root matching the reference exactly.
fn isqrt32(mut val: u32) -> u32 {
    let mut g = 0u32;
    let mut bshift = (ilog(val) as i32 - 1) >> 1;
    let mut b = 1u32 << bshift;
    loop {
        let t = ((g << 1) + b) << bshift;
        if t <= val {
            g += b;
            val -= t;
        }
        b >>= 1;
        bshift -= 1;
        if bshift < 0 {
            break;
        }
    }
    g
}

pub(super) fn quant_band_n1(
    ctx: &mut BandCtx,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    lowband_out: Option<&mut [f32]>,
) -> u32 {
    let mut channels: [&mut [f32]; 2];
    let count;
    match y {
        Some(y) => {
            channels = [x, y];
            count = 2;
        }
        None => {
            channels = [x, &mut []];
            count = 1;
        }
    }
    for ch in channels.iter_mut().take(count) {
        let mut sign = 0u32;
        if ctx.remaining_bits >= 1 << BITRES {
            sign = ctx.dec.decode_bits(1);
            ctx.remaining_bits -= 1 << BITRES;
        }
        ch[0] = if sign != 0 { -1.0 } else { 1.0 };
    }
    if let Some(out) = lowband_out {
        out[0] = channels[0][0];
    }
    1
}

/// Decode one partition, splitting it in two when one codeword would cost
/// more bits than the band has.
fn quant_partition(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    blocks: usize,
    lowband: Option<&mut [f32]>,
    lm: i32,
    gain: f32,
    fill: i32,
) -> u32 {
    let b0 = blocks;
    let mut fill = fill;

    let splittable = if lm != -1 && n > 2 {
        let cache = cache_row(ctx.band, lm);
        b > i32::from(cache[cache[0] as usize]) + 12
    } else {
        false
    };

    if splittable {
        let half = n >> 1;
        let lm = lm - 1;
        if blocks == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        let blocks = (blocks + 1) >> 1;

        let mut b = b;
        let sctx = compute_theta(ctx, half, &mut b, blocks, b0, lm, false, &mut fill);
        let mid = sctx.imid as f32 * (1.0 / 32768.0);
        let side = sctx.iside as f32 * (1.0 / 32768.0);
        let mut delta = sctx.delta;

        // Bias short low-energy blocks toward the quieter half to avoid pre-echo.
        if b0 > 1 && (sctx.itheta & 0x3fff) != 0 {
            if sctx.itheta > 8192 {
                delta -= delta >> (4 - lm);
            } else {
                delta = 0.min(delta + ((half as i32) << BITRES >> (5 - lm)));
            }
        }
        let mut mbits = 0.max(b.min((b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= sctx.qalloc;

        let (xl, xr) = x.split_at_mut(half);
        let (mut lb_lo, mut lb_hi) = match lowband {
            Some(lb) => {
                let (a, c) = lb.split_at_mut(half);
                (Some(a), Some(c))
            }
            None => (None, None),
        };

        let rebalance = ctx.remaining_bits;
        let cm;
        if mbits >= sbits {
            cm = quant_partition(
                ctx,
                xl,
                half,
                mbits,
                blocks,
                lb_lo.take(),
                lm,
                gain * mid,
                fill,
            );
            let spent = mbits - (rebalance - ctx.remaining_bits);
            if spent > 3 << BITRES && sctx.itheta != 0 {
                sbits += spent - (3 << BITRES);
            }
            cm | quant_partition(
                ctx,
                xr,
                half,
                sbits,
                blocks,
                lb_hi.take(),
                lm,
                gain * side,
                fill >> blocks,
            ) << (b0 >> 1)
        } else {
            let cm2 = quant_partition(
                ctx,
                xr,
                half,
                sbits,
                blocks,
                lb_hi.take(),
                lm,
                gain * side,
                fill >> blocks,
            ) << (b0 >> 1);
            let spent = sbits - (rebalance - ctx.remaining_bits);
            if spent > 3 << BITRES && sctx.itheta != 16384 {
                mbits += spent - (3 << BITRES);
            }
            cm2 | quant_partition(
                ctx,
                xl,
                half,
                mbits,
                blocks,
                lb_lo.take(),
                lm,
                gain * mid,
                fill,
            )
        }
    } else {
        let mut q = bits2pulses(ctx.band, lm, b);
        let mut curr_bits = pulses2bits(ctx.band, lm, q);
        ctx.remaining_bits -= curr_bits;
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(ctx.band, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            alg_unquant(x, n, get_pulses(q), ctx.spread, blocks, ctx.dec, gain)
        } else {
            // No pulses: fold or fill rather than leave the band empty.
            let mask = ((1u32 << blocks) - 1) as i32;
            fill &= mask;
            if fill == 0 {
                x[..n].fill(0.0);
                0
            } else {
                match lowband {
                    None => {
                        for v in x.iter_mut().take(n) {
                            ctx.seed = lcg_rand(ctx.seed);
                            *v = ((ctx.seed as i32) >> 20) as f32;
                        }
                        renormalise_vector(&mut x[..n], gain);
                        mask as u32
                    }
                    Some(lb) => {
                        // Folded spectrum, dithered ~48 dB below normal folding.
                        for j in 0..n {
                            ctx.seed = lcg_rand(ctx.seed);
                            let tmp = if ctx.seed & 0x8000 != 0 {
                                1.0 / 256.0
                            } else {
                                -1.0 / 256.0
                            };
                            x[j] = lb[j] + tmp;
                        }
                        renormalise_vector(&mut x[..n], gain);
                        fill as u32
                    }
                }
            }
        }
    }
}

/// Decode one band of one channel, with any time-frequency change around it.
#[allow(clippy::too_many_arguments)]
pub(super) fn quant_band(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    blocks: usize,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    gain: f32,
    fill: i32,
) -> u32 {
    let n0 = n;
    let mut n_b = n;
    let mut blocks = blocks;
    let b0 = blocks;
    let mut time_divide = 0;
    let mut recombine = 0;
    let long_blocks = b0 == 1;
    let mut fill = fill;
    let mut tf_change = ctx.tf_change;

    n_b /= blocks;

    if n == 1 {
        return quant_band_n1(ctx, x, None, lowband_out);
    }

    if tf_change > 0 {
        recombine = tf_change;
    }

    // A private copy: the transforms below rewrite it, and later bands fold from the original.
    let mut lowband = lowband;

    for k in 0..recombine {
        if let Some(lb) = lowband.as_deref_mut() {
            haar1(lb, n >> k, 1 << k);
        }
        fill = i32::from(BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize])
            | i32::from(BIT_INTERLEAVE_TABLE[(fill >> 4) as usize]) << 2;
    }
    blocks >>= recombine;
    n_b <<= recombine;

    while (n_b & 1) == 0 && tf_change < 0 {
        if let Some(lb) = lowband.as_deref_mut() {
            haar1(lb, n_b, blocks);
        }
        fill |= fill << blocks;
        blocks <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tf_change += 1;
    }
    let b0 = blocks;
    let n_b0 = n_b;

    if b0 > 1 {
        if let Some(lb) = lowband.as_deref_mut() {
            deinterleave_hadamard(lb, n_b >> recombine, b0 << recombine, long_blocks);
        }
    }

    let mut cm = quant_partition(ctx, x, n, b, blocks, lowband, lm, gain, fill);

    if b0 > 1 {
        interleave_hadamard(x, n_b >> recombine, b0 << recombine, long_blocks);
    }

    let mut n_b = n_b0;
    let mut blocks = b0;
    for _ in 0..time_divide {
        blocks >>= 1;
        n_b <<= 1;
        cm |= cm >> blocks;
        haar1(x, n_b, blocks);
    }
    for k in 0..recombine {
        cm = u32::from(BIT_DEINTERLEAVE_TABLE[(cm & 0xF) as usize]);
        haar1(x, n0 >> k, 1 << k);
    }
    blocks <<= recombine;

    // Scale for later folding, as if a full-length spectrum.
    if let Some(out) = lowband_out {
        let scale = (n0 as f32).sqrt();
        for j in 0..n0 {
            out[j] = scale * x[j];
        }
    }
    cm & ((1 << blocks) - 1)
}
