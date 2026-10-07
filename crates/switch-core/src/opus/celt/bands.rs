//! Every band of a frame: stereo, folding, time-frequency changes and anti-collapse.

use crate::opus::range::{RangeDecoder, BITRES};
use crate::opus::tables_celt::*;
use super::partition::{compute_theta, lcg_rand, quant_band, quant_band_n1, BandCtx};
use super::vq::renormalise_vector;
use super::{NB_EBANDS, SHORT_MDCT_SIZE, SPREAD_AGGRESSIVE};

/// Decode one band of both channels by coding the angle between them.
#[allow(clippy::too_many_arguments)]
fn quant_band_stereo(
    ctx: &mut BandCtx,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    b: i32,
    blocks: usize,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    fill: i32,
) -> u32 {
    if n == 1 {
        return quant_band_n1(ctx, x, Some(y), lowband_out);
    }
    let orig_fill = fill;
    let mut fill = fill;
    let mut b = b;

    let sctx = compute_theta(ctx, n, &mut b, blocks, blocks, lm, true, &mut fill);
    let mid = sctx.imid as f32 * (1.0 / 32768.0);
    let side = sctx.iside as f32 * (1.0 / 32768.0);

    let cm;
    if n == 2 {
        let mut mbits = b;
        let mut sbits = 0;
        if sctx.itheta != 0 && sctx.itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits -= sbits;
        let swapped = sctx.itheta > 8192;
        ctx.remaining_bits -= sctx.qalloc + sbits;

        let sign = if sbits != 0 {
            ctx.dec.decode_bits(1)
        } else {
            0
        };
        let sign = 1.0 - 2.0 * sign as f32;

        let (x2, y2) = if swapped {
            (&mut *y, &mut *x)
        } else {
            (&mut *x, &mut *y)
        };
        // `orig_fill`: the side is folded even when the angle cleared the low bits.
        cm = quant_band(
            ctx,
            x2,
            n,
            mbits,
            blocks,
            lowband,
            lm,
            lowband_out,
            1.0,
            orig_fill,
        );
        y2[0] = -sign * x2[1];
        y2[1] = sign * x2[0];

        x[0] *= mid;
        x[1] *= mid;
        y[0] *= side;
        y[1] *= side;
        let tmp = x[0];
        x[0] = tmp - y[0];
        y[0] = tmp + y[0];
        let tmp = x[1];
        x[1] = tmp - y[1];
        y[1] = tmp + y[1];
    } else {
        let mut mbits = 0.max(b.min((b - sctx.delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= sctx.qalloc;
        let rebalance = ctx.remaining_bits;

        if mbits >= sbits {
            // The mid keeps unit gain because later bands fold from it.
            let cm0 = quant_band(
                ctx,
                x,
                n,
                mbits,
                blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                fill,
            );
            let spent = mbits - (rebalance - ctx.remaining_bits);
            if spent > 3 << BITRES && sctx.itheta != 0 {
                sbits += spent - (3 << BITRES);
            }
            cm = cm0
                | quant_band(
                    ctx,
                    y,
                    n,
                    sbits,
                    blocks,
                    None,
                    lm,
                    None,
                    side,
                    fill >> blocks,
                );
        } else {
            let cm0 = quant_band(
                ctx,
                y,
                n,
                sbits,
                blocks,
                None,
                lm,
                None,
                side,
                fill >> blocks,
            );
            let spent = sbits - (rebalance - ctx.remaining_bits);
            if spent > 3 << BITRES && sctx.itheta != 16384 {
                mbits += spent - (3 << BITRES);
            }
            cm = cm0
                | quant_band(
                    ctx,
                    x,
                    n,
                    mbits,
                    blocks,
                    lowband,
                    lm,
                    lowband_out,
                    1.0,
                    fill,
                );
        }
    }

    if n != 2 {
        stereo_merge(x, y, mid, n);
    }
    if sctx.inv {
        for v in y.iter_mut().take(n) {
            *v = -*v;
        }
    }
    cm
}

/// Mid/side back to left/right, normalised to the angle's energy.
fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
    let mut xp = 0.0f32;
    let mut side = 0.0f32;
    for j in 0..n {
        xp += x[j] * y[j];
        side += y[j] * y[j];
    }
    xp *= mid;
    let mid2 = mid;
    let el = mid2 * mid2 + side - 2.0 * xp;
    let er = mid2 * mid2 + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y[..n].copy_from_slice(&x[..n]);
        return;
    }
    let lgain = 1.0 / el.sqrt();
    let rgain = 1.0 / er.sqrt();
    for j in 0..n {
        let l = mid * x[j];
        let r = y[j];
        x[j] = lgain * (l - r);
        y[j] = rgain * (l + r);
    }
}

/// Hybrid frames: copy folding data so the second band has something to fold from.
fn special_hybrid_folding(norm: &mut [f32], norm2: Option<&mut [f32]>, start: usize, m: usize) {
    let n1 = m * (EBAND_5MS[start + 1] - EBAND_5MS[start]) as usize;
    let n2 = m * (EBAND_5MS[start + 2] - EBAND_5MS[start + 1]) as usize;
    if n2 <= n1 {
        return;
    }
    norm.copy_within(2 * n1 - n2..n1, n1);
    if let Some(norm2) = norm2 {
        norm2.copy_within(2 * n1 - n2..n1, n1);
    }
}

/// Decode every band, tracking the encoder's running bit balance.
#[allow(clippy::too_many_arguments)]
pub(super) fn quant_all_bands(
    start: usize,
    end: usize,
    x: &mut [f32],
    channels: usize,
    collapse_masks: &mut [u8],
    pulses: &[i32],
    short_blocks: bool,
    spread: usize,
    mut dual_stereo: bool,
    intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    mut balance: i32,
    dec: &mut RangeDecoder,
    lm: usize,
    coded_bands: usize,
    seed: &mut u32,
    disable_inv: bool,
) {
    let m = 1usize << lm;
    let blocks = if short_blocks { m } else { 1 };
    let norm_offset = m * EBAND_5MS[start] as usize;
    let norm_len = m * EBAND_5MS[NB_EBANDS - 1] as usize - norm_offset;
    let n_total = m * SHORT_MDCT_SIZE;

    let mut norm = vec![0.0f32; norm_len];
    let mut norm2 = vec![0.0f32; if channels == 2 { norm_len } else { 0 }];

    let (xch, ych) = x.split_at_mut(n_total);
    let mut lowband_offset = 0usize;
    let mut update_lowband = true;

    let mut ctx = BandCtx {
        dec,
        band: start,
        intensity,
        spread,
        tf_change: 0,
        remaining_bits: 0,
        seed: *seed,
        disable_inv,
    };

    for i in start..end {
        ctx.band = i;
        let last = i == end - 1;
        let lo = m * EBAND_5MS[i] as usize;
        let hi = m * EBAND_5MS[i + 1] as usize;
        let n = hi - lo;
        let tell = ctx.dec.tell_frac() as i32;

        if i != start {
            balance -= tell;
        }
        let remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        let b = if i <= coded_bands - 1 {
            let curr_balance = balance / 3.min(coded_bands - i) as i32;
            0.max(16383.min((remaining_bits + 1).min(pulses[i] + curr_balance)))
        } else {
            0
        };

        if (lo >= norm_offset + n || i == start + 1) && (update_lowband || lowband_offset == 0) {
            lowband_offset = i;
        }
        if i == start + 1 {
            let norm2_ref = if channels == 2 {
                Some(norm2.as_mut_slice())
            } else {
                None
            };
            special_hybrid_folding(&mut norm, norm2_ref, start, m);
        }

        ctx.tf_change = tf_res[i];

        // Estimate which folding-source blocks carry energy, as the encoder does.
        let mut effective_lowband: Option<usize> = None;
        let mut x_cm;
        let mut y_cm;
        if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || blocks > 1 || ctx.tf_change < 0) {
            let eff = (m * EBAND_5MS[lowband_offset] as usize).saturating_sub(norm_offset + n);
            effective_lowband = Some(eff);
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if m * EBAND_5MS[fold_start] as usize <= eff + norm_offset {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if fold_end >= i || m * EBAND_5MS[fold_end] as usize >= eff + norm_offset + n {
                    break;
                }
            }
            x_cm = 0u32;
            y_cm = 0u32;
            for fold_i in fold_start..fold_end {
                x_cm |= u32::from(collapse_masks[fold_i * channels]);
                y_cm |= u32::from(collapse_masks[fold_i * channels + channels - 1]);
            }
        } else {
            // Nothing to fold from: the LCG fills every block.
            x_cm = (1u32 << blocks) - 1;
            y_cm = x_cm;
        }

        if dual_stereo && i == intensity {
            // Intensity coding merges the two folding histories.
            dual_stereo = false;
            for j in 0..lo - norm_offset {
                norm[j] = 0.5 * (norm[j] + norm2[j]);
            }
        }

        let split = lo - norm_offset;
        // A private copy: the source may overlap this band's output in hybrid frames.
        let mut lowband = effective_lowband.map(|o| norm[o..o + n].to_vec());
        let mut lowband2 = if dual_stereo {
            effective_lowband.map(|o| norm2[o..o + n].to_vec())
        } else {
            None
        };

        if dual_stereo {
            x_cm = quant_band(
                &mut ctx,
                &mut xch[lo..hi],
                n,
                b / 2,
                blocks,
                lowband.as_deref_mut(),
                lm as i32,
                if last {
                    None
                } else {
                    Some(&mut norm[split..split + n])
                },
                1.0,
                x_cm as i32,
            );
            y_cm = quant_band(
                &mut ctx,
                &mut ych[lo..hi],
                n,
                b / 2,
                blocks,
                lowband2.as_deref_mut(),
                lm as i32,
                if last {
                    None
                } else {
                    Some(&mut norm2[split..split + n])
                },
                1.0,
                y_cm as i32,
            );
        } else if channels == 2 {
            x_cm = quant_band_stereo(
                &mut ctx,
                &mut xch[lo..hi],
                &mut ych[lo..hi],
                n,
                b,
                blocks,
                lowband.as_deref_mut(),
                lm as i32,
                if last {
                    None
                } else {
                    Some(&mut norm[split..split + n])
                },
                (x_cm | y_cm) as i32,
            );
            y_cm = x_cm;
        } else {
            x_cm = quant_band(
                &mut ctx,
                &mut xch[lo..hi],
                n,
                b,
                blocks,
                lowband.as_deref_mut(),
                lm as i32,
                if last {
                    None
                } else {
                    Some(&mut norm[split..split + n])
                },
                1.0,
                (x_cm | y_cm) as i32,
            );
            y_cm = x_cm;
        }
        collapse_masks[i * channels] = x_cm as u8;
        collapse_masks[i * channels + channels - 1] = y_cm as u8;
        balance += pulses[i] + tell;

        // Advance the folding source only while it has a bit per sample.
        update_lowband = b > (n as i32) << BITRES;
    }
    *seed = ctx.seed;
}

/// Per-band time-frequency changes and the bit selecting their interpretation.
pub(super) fn tf_decode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: usize,
    dec: &mut RangeDecoder,
    total_bytes: usize,
) {
    let mut budget = (total_bytes * 8) as i32;
    let mut tell = dec.tell();
    let mut logp = if is_transient { 2 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp + 1 <= budget;
    budget -= i32::from(tf_select_rsv);
    let mut curr = 0i32;
    let mut tf_changed = 0i32;
    for i in start..end {
        if tell + logp <= budget {
            curr ^= i32::from(dec.decode_bit_logp(logp as u32));
            tell = dec.tell();
            tf_changed |= curr;
        }
        tf_res[i] = curr;
        logp = if is_transient { 4 } else { 5 };
    }
    let row = &TF_SELECT_TABLE[lm * 8..][..8];
    let base = 4 * usize::from(is_transient);
    let mut tf_select = 0usize;
    if tf_select_rsv && row[base + tf_changed as usize] != row[base + 2 + tf_changed as usize] {
        tf_select = usize::from(dec.decode_bit_logp(1));
    }
    for i in start..end {
        tf_res[i] = i32::from(row[base + 2 * tf_select + tf_res[i] as usize]);
    }
}

/// Scale each unit-norm shape to its coded energy.
pub(super) fn denormalise_bands(
    x: &[f32],
    freq: &mut [f32],
    band_loge: &[f32],
    start: usize,
    end: usize,
    m: usize,
    downsample: usize,
    silence: bool,
) {
    let n = m * SHORT_MDCT_SIZE;
    let mut bound = m * EBAND_5MS[end] as usize;
    if downsample != 1 {
        bound = bound.min(n / downsample);
    }
    let (start, end) = if silence {
        bound = 0;
        (0, 0)
    } else {
        (start, end)
    };
    for f in freq.iter_mut().take(m * EBAND_5MS[start] as usize) {
        *f = 0.0;
    }
    for i in start..end {
        let g = exp2_approx((band_loge[i] + E_MEANS[i]).min(32.0));
        for j in m * EBAND_5MS[i] as usize..m * EBAND_5MS[i + 1] as usize {
            freq[j] = x[j] * g;
        }
    }
    for f in freq[bound..n].iter_mut() {
        *f = 0.0;
    }
}

/// `2^x` for base-2 log energies, safe on corrupt bands.
fn exp2_approx(x: f32) -> f32 {
    if x <= -128.0 {
        0.0
    } else {
        (x * core::f32::consts::LN_2).exp()
    }
}

/// Refill transient blocks left with no pulses, which would otherwise rattle.
#[allow(clippy::too_many_arguments)]
pub(super) fn anti_collapse(
    x: &mut [f32],
    collapse_masks: &[u8],
    lm: usize,
    channels: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[f32],
    prev1_log_e: &[f32],
    prev2_log_e: &[f32],
    pulses: &[i32],
    mut seed: u32,
) {
    for i in start..end {
        let n0 = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as usize;
        let depth = ((1 + pulses[i]) / (EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32) >> lm;
        let thresh = 0.5 * exp2_approx(-0.125 * depth as f32);
        let sqrt_1 = 1.0 / ((n0 << lm) as f32).sqrt();

        for c in 0..channels {
            let mut prev1 = prev1_log_e[c * NB_EBANDS + i];
            let mut prev2 = prev2_log_e[c * NB_EBANDS + i];
            if channels == 1 {
                prev1 = prev1.max(prev1_log_e[NB_EBANDS + i]);
                prev2 = prev2.max(prev2_log_e[NB_EBANDS + i]);
            }
            let ediff = (log_e[c * NB_EBANDS + i] - prev1.min(prev2)).max(0.0);
            // Scale up noise replacing a collapsed short block.
            let mut r = 2.0 * exp2_approx(-ediff);
            if lm == 3 {
                r *= 1.41421356;
            }
            r = r.min(thresh) * sqrt_1;

            let base = c * size + ((EBAND_5MS[i] as usize) << lm);
            let mut renormalize = false;
            for k in 0..1usize << lm {
                if collapse_masks[i * channels + c] & (1 << k) == 0 {
                    for j in 0..n0 {
                        seed = lcg_rand(seed);
                        x[base + (j << lm) + k] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                renormalise_vector(&mut x[base..base + (n0 << lm)], 1.0);
            }
        }
    }
}
