//! NLSF dequantisation and conversion to a stable LPC filter.

use crate::opus::tables_silk::*;
use super::{
    clz32, inverse32_varq, rshift_round, rshift_round64, sat16, smlawb, smmul, smulbb, smulww,
    sub_sat32, NlsfCodebook, MAX_LPC_ORDER, NLSF_QUANT_MAX_AMPLITUDE,
};

/// Entropy table and backward predictor per NLSF coefficient, packed per pair.
pub(super) fn nlsf_unpack(
    cb: &NlsfCodebook,
    cb1_index: usize,
) -> ([usize; MAX_LPC_ORDER], [u8; MAX_LPC_ORDER]) {
    let mut ec_ix = [0usize; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    let base = cb1_index * cb.order / 2;
    for i in (0..cb.order).step_by(2) {
        let entry = cb.ec_sel[base + i / 2];
        ec_ix[i] = usize::from((entry >> 1) & 7) * (2 * NLSF_QUANT_MAX_AMPLITUDE as usize + 1);
        pred_q8[i] = cb.pred_q8[i + usize::from(entry & 1) * (cb.order - 1)];
        ec_ix[i + 1] = usize::from((entry >> 5) & 7) * (2 * NLSF_QUANT_MAX_AMPLITUDE as usize + 1);
        pred_q8[i + 1] = cb.pred_q8[i + usize::from((entry >> 4) & 1) * (cb.order - 1) + 1];
    }
    (ec_ix, pred_q8)
}

/// The second-stage residual, dequantised backwards so each predictor sees
/// the coefficient above it.
fn nlsf_residual_dequant(
    x_q10: &mut [i16],
    indices: &[i8],
    pred_coef_q8: &[u8],
    quant_step_size_q16: i32,
    order: usize,
) {
    let mut out_q10 = 0i32;
    for i in (0..order).rev() {
        let pred_q10 = smulbb(out_q10, i32::from(pred_coef_q8[i])) >> 8;
        out_q10 = i32::from(indices[i]) << 10;
        // The dead zone around zero the quantiser left.
        if out_q10 > 0 {
            out_q10 -= 102;
        } else if out_q10 < 0 {
            out_q10 += 102;
        }
        out_q10 = smlawb(pred_q10, out_q10, quant_step_size_q16);
        x_q10[i] = out_q10 as i16;
    }
}

/// Pull the NLSFs apart until every pair is at least its minimum distance
/// apart; crossing frequencies give an unstable filter.
fn nlsf_stabilize(nlsf_q15: &mut [i16], ndelta_min_q15: &[i16], l: usize) {
    const MAX_LOOPS: usize = 20;
    let mut loops = 0;
    while loops < MAX_LOOPS {
        let mut min_diff_q15 = i32::from(nlsf_q15[0]) - i32::from(ndelta_min_q15[0]);
        let mut idx = 0usize;
        for i in 1..l {
            let diff = i32::from(nlsf_q15[i])
                - (i32::from(nlsf_q15[i - 1]) + i32::from(ndelta_min_q15[i]));
            if diff < min_diff_q15 {
                min_diff_q15 = diff;
                idx = i;
            }
        }
        let diff = (1 << 15) - (i32::from(nlsf_q15[l - 1]) + i32::from(ndelta_min_q15[l]));
        if diff < min_diff_q15 {
            min_diff_q15 = diff;
            idx = l;
        }
        if min_diff_q15 >= 0 {
            return;
        }
        if idx == 0 {
            nlsf_q15[0] = ndelta_min_q15[0];
        } else if idx == l {
            nlsf_q15[l - 1] = ((1 << 15) - i32::from(ndelta_min_q15[l])) as i16;
        } else {
            // Move the pair apart around its centre, within the neighbours' room.
            let mut min_center_q15 = 0i32;
            for k in 0..idx {
                min_center_q15 += i32::from(ndelta_min_q15[k]);
            }
            min_center_q15 += i32::from(ndelta_min_q15[idx]) >> 1;
            let mut max_center_q15 = 1i32 << 15;
            for k in (idx + 1..=l).rev() {
                max_center_q15 -= i32::from(ndelta_min_q15[k]);
            }
            max_center_q15 -= i32::from(ndelta_min_q15[idx]) >> 1;
            let center = rshift_round(i32::from(nlsf_q15[idx - 1]) + i32::from(nlsf_q15[idx]), 1)
                .clamp(min_center_q15, max_center_q15);
            nlsf_q15[idx - 1] = (center - (i32::from(ndelta_min_q15[idx]) >> 1)) as i16;
            nlsf_q15[idx] = (i32::from(nlsf_q15[idx - 1]) + i32::from(ndelta_min_q15[idx])) as i16;
        }
        loops += 1;
    }
    // Fallback for corrupt streams: sort, then force the spacing from both ends.
    nlsf_q15[..l].sort_unstable();
    nlsf_q15[0] = nlsf_q15[0].max(ndelta_min_q15[0]);
    for i in 1..l {
        nlsf_q15[i] = nlsf_q15[i].max(nlsf_q15[i - 1].saturating_add(ndelta_min_q15[i]));
    }
    nlsf_q15[l - 1] = nlsf_q15[l - 1].min(((1 << 15) - i32::from(ndelta_min_q15[l])) as i16);
    for i in (0..l - 1).rev() {
        nlsf_q15[i] = nlsf_q15[i].min(nlsf_q15[i + 1] - ndelta_min_q15[i + 1]);
    }
}

/// Turn the coded codebook path back into a normalised NLSF vector.
pub(super) fn nlsf_decode(nlsf_q15: &mut [i16], indices: &[i8], cb: &NlsfCodebook) {
    let cb1 = indices[0] as usize;
    let (_, pred_q8) = nlsf_unpack(cb, cb1);
    let mut res_q10 = [0i16; MAX_LPC_ORDER];
    nlsf_residual_dequant(
        &mut res_q10,
        &indices[1..],
        &pred_q8,
        cb.quant_step_size_q16,
        cb.order,
    );

    let element = &cb.cb1_nlsf_q8[cb1 * cb.order..];
    let wght = &cb.cb1_wght_q9[cb1 * cb.order..];
    for i in 0..cb.order {
        // The first-stage weights are inverse square roots, so divide.
        let tmp =
            ((i32::from(res_q10[i]) << 14) / i32::from(wght[i])) + (i32::from(element[i]) << 7);
        nlsf_q15[i] = tmp.clamp(0, 32767) as i16;
    }
    nlsf_stabilize(nlsf_q15, cb.delta_min_q15, cb.order);
}

/// The Q domain the NLSF-to-LPC conversion works in.
const NLSF2A_QA: u32 = 16;

/// Build one of the two symmetric polynomials whose roots are the LSFs, one
/// root pair at a time.
fn nlsf2a_find_poly(out: &mut [i64], c_lsf_qa: &[i32], dd: usize) {
    out[0] = 1i64 << NLSF2A_QA;
    out[1] = -i64::from(c_lsf_qa[0]);
    for k in 1..dd {
        let ftmp = i64::from(c_lsf_qa[2 * k]);
        out[k + 1] = (out[k - 1] << 1) - rshift_round64(ftmp * out[k], NLSF2A_QA);
        for n in (2..=k).rev() {
            out[n] += out[n - 2] - rshift_round64(ftmp * out[n - 1], NLSF2A_QA);
        }
        out[1] -= ftmp;
    }
}

/// Clamp LPC coefficients into 16 bits, by bandwidth expansion rather than clipping.
fn lpc_fit(a_qout: &mut [i16], a_qin: &mut [i32], qout: u32, qin: u32, d: usize) {
    let mut i = 0;
    while i < 10 {
        let mut maxabs = 0i32;
        let mut idx = 0usize;
        for k in 0..d {
            let absval = a_qin[k].wrapping_abs();
            if absval > maxabs {
                maxabs = absval;
                idx = k;
            }
        }
        maxabs = rshift_round(maxabs, qin - qout);
        if maxabs > 32767 {
            maxabs = maxabs.min(163838);
            let chirp_q16 =
                65471 - (((maxabs - 32767) << 14) / ((maxabs.wrapping_mul(idx as i32 + 1)) >> 2));
            bwexpander_32(a_qin, d, chirp_q16);
        } else {
            break;
        }
        i += 1;
    }
    if i == 10 {
        for k in 0..d {
            a_qout[k] = sat16(rshift_round(a_qin[k], qin - qout));
            a_qin[k] = i32::from(a_qout[k]) << (qin - qout);
        }
    } else {
        for k in 0..d {
            a_qout[k] = rshift_round(a_qin[k], qin - qout) as i16;
        }
    }
}

pub(super) fn bwexpander(ar: &mut [i16], d: usize, mut chirp_q16: i32) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    for i in 0..d - 1 {
        // Not `smulwb`: its bias accumulates into instability over repeated expansions.
        ar[i] = rshift_round(chirp_q16.wrapping_mul(i32::from(ar[i])), 16) as i16;
        chirp_q16 += rshift_round(chirp_q16.wrapping_mul(chirp_minus_one_q16), 16);
    }
    ar[d - 1] = rshift_round(chirp_q16.wrapping_mul(i32::from(ar[d - 1])), 16) as i16;
}

fn bwexpander_32(ar: &mut [i32], d: usize, mut chirp_q16: i32) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    for i in 0..d - 1 {
        ar[i] = smulww(chirp_q16, ar[i]);
        chirp_q16 += rshift_round(chirp_q16.wrapping_mul(chirp_minus_one_q16), 16);
    }
    ar[d - 1] = smulww(chirp_q16, ar[d - 1]);
}

/// The Q domain the stability check works in.
const INV_GAIN_QA: u32 = 24;

/// One over the prediction gain, in Q30, or zero if the filter is unstable
/// (a backwards Levinson recursion checking every reflection coefficient).
fn lpc_inverse_pred_gain_qa(a_qa: &mut [i32], order: usize) -> i32 {
    const A_LIMIT_QA: i32 = 16773022; // 0.99975 in Q24
    let mut inv_gain_q30 = 1i32 << 30;
    let mut k = order - 1;
    while k > 0 {
        if a_qa[k] > A_LIMIT_QA || a_qa[k] < -A_LIMIT_QA {
            return 0;
        }
        let rc_q31 = -(a_qa[k] << (31 - INV_GAIN_QA));
        let rc_mult1_q30 = (1i32 << 30).wrapping_sub(smmul(rc_q31, rc_q31));
        inv_gain_q30 = smmul(inv_gain_q30, rc_mult1_q30) << 2;
        // More than 40 dB of prediction gain means a corrupt stream.
        if inv_gain_q30 < 107374 {
            return 0;
        }
        let mult2q = (32 - clz32(rc_mult1_q30.wrapping_abs())) as u32;
        let rc_mult2 = inverse32_varq(rc_mult1_q30, mult2q + 30);
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a_qa[n];
            let tmp2 = a_qa[k - n - 1];
            let t = rshift_round64(
                i64::from(sub_sat32(
                    tmp1,
                    rshift_round64(i64::from(tmp2) * i64::from(rc_q31), 31) as i32,
                )) * i64::from(rc_mult2),
                mult2q,
            );
            if t > i64::from(i32::MAX) || t < i64::from(i32::MIN) {
                return 0;
            }
            a_qa[n] = t as i32;
            let t = rshift_round64(
                i64::from(sub_sat32(
                    tmp2,
                    rshift_round64(i64::from(tmp1) * i64::from(rc_q31), 31) as i32,
                )) * i64::from(rc_mult2),
                mult2q,
            );
            if t > i64::from(i32::MAX) || t < i64::from(i32::MIN) {
                return 0;
            }
            a_qa[k - n - 1] = t as i32;
        }
        k -= 1;
    }
    if a_qa[0] > A_LIMIT_QA || a_qa[0] < -A_LIMIT_QA {
        return 0;
    }
    let rc_q31 = -(a_qa[0] << (31 - INV_GAIN_QA));
    let rc_mult1_q30 = (1i32 << 30).wrapping_sub(smmul(rc_q31, rc_q31));
    inv_gain_q30 = smmul(inv_gain_q30, rc_mult1_q30) << 2;
    if inv_gain_q30 < 107374 {
        return 0;
    }
    inv_gain_q30
}

pub(super) fn lpc_inverse_pred_gain(a_q12: &[i16], order: usize) -> i32 {
    let mut atmp_qa = [0i32; MAX_LPC_ORDER];
    let mut dc_resp = 0i32;
    for k in 0..order {
        dc_resp += i32::from(a_q12[k]);
        atmp_qa[k] = i32::from(a_q12[k]) << (INV_GAIN_QA - 12);
    }
    // A pole at DC means unstable, without running the full recursion.
    if dc_resp >= 4096 {
        return 0;
    }
    lpc_inverse_pred_gain_qa(&mut atmp_qa, order)
}

/// Convert normalised LSFs into the LPC filter they describe.
pub(super) fn nlsf2a(a_q12: &mut [i16], nlsf: &[i16], d: usize) {
    // This ordering keeps intermediate coefficients small enough for fixed point.
    const ORDERING16: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
    const ORDERING10: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];
    let ordering: &[usize] = if d == 16 { &ORDERING16 } else { &ORDERING10 };

    let mut cos_lsf_qa = [0i32; MAX_LPC_ORDER];
    for k in 0..d {
        // A piecewise-linear cosine off a 128-entry table.
        let f_int = i32::from(nlsf[k]) >> (15 - 7);
        let f_frac = i32::from(nlsf[k]) - (f_int << (15 - 7));
        let cos_val = i32::from(LSF_COS_TAB_Q12[f_int as usize]);
        let delta = i32::from(LSF_COS_TAB_Q12[f_int as usize + 1]) - cos_val;
        cos_lsf_qa[ordering[k]] = rshift_round((cos_val << 8) + delta * f_frac, 20 - NLSF2A_QA);
    }

    let dd = d >> 1;
    let mut p = [0i64; MAX_LPC_ORDER / 2 + 1];
    let mut q = [0i64; MAX_LPC_ORDER / 2 + 1];
    nlsf2a_find_poly(&mut p, &cos_lsf_qa, dd);
    nlsf2a_find_poly(&mut q, &cos_lsf_qa[1..], dd);

    let mut a32_qa1 = [0i32; MAX_LPC_ORDER];
    for k in 0..dd {
        let ptmp = p[k + 1] + p[k];
        let qtmp = q[k + 1] - q[k];
        a32_qa1[k] = (-qtmp - ptmp) as i32;
        a32_qa1[d - k - 1] = (qtmp - ptmp) as i32;
    }

    lpc_fit(a_q12, &mut a32_qa1, 12, NLSF2A_QA + 1, d);

    // Expand the bandwidth until stable: the synthesis is an IIR.
    let mut i = 0;
    while lpc_inverse_pred_gain(a_q12, d) == 0 && i < 16 {
        bwexpander_32(&mut a32_qa1, d, 65536 - (2 << i));
        for k in 0..d {
            a_q12[k] = rshift_round(a32_qa1[k], NLSF2A_QA + 1 - 12) as i16;
        }
        i += 1;
    }
}
