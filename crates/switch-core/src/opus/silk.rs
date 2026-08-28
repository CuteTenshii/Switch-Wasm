//! The SILK layer: the linear-prediction half of Opus, and all of a
//! speech-rate packet.
//!
//! SILK models the signal the way a vocal tract makes it. A short-term LPC
//! filter stands for the resonances of the mouth, a long-term predictor
//! stands for the pitch, and what is left — the excitation — is what actually
//! gets coded, as pulses. Synthesis runs that backwards: pulses, through the
//! pitch predictor, through the LPC filter, scaled by a per-subframe gain.
//!
//! **This layer is integer arithmetic throughout, and that is not an
//! optimisation.** SILK's decoder is specified in fixed point: the filter
//! coefficients come out of a codebook through a chain of Q-format
//! multiplications, and a decoder that used floats would produce a slightly
//! different filter, which the next frame then predicts from. So every
//! multiply here is the reference's multiply, in its Q domain, rounding the
//! way it rounds.
//!
//! Internally SILK runs at 8, 12 or 16 kHz and is resampled to whatever the
//! caller asked for on the way out; in hybrid mode CELT carries everything
//! above 8 kHz and the two are summed.

use super::range::RangeDecoder;
use super::tables_silk::*;

/// Longest frame SILK codes: 20 ms at 16 kHz.
const MAX_FRAME_LENGTH: usize = 320;
const MAX_NB_SUBFR: usize = 4;
const MAX_LPC_ORDER: usize = 16;
const MIN_LPC_ORDER: usize = 10;
const LTP_ORDER: usize = 5;
const SUB_FRAME_LENGTH_MS: usize = 5;
const LTP_MEM_LENGTH_MS: usize = 20;
const SHELL_CODEC_FRAME_LENGTH: usize = 16;
const LOG2_SHELL_CODEC_FRAME_LENGTH: usize = 4;
const MAX_NB_SHELL_BLOCKS: usize = MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH;
const SILK_MAX_PULSES: i32 = 16;
const N_RATE_LEVELS: usize = 10;
const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;
const QUANT_LEVEL_ADJUST_Q10: i32 = 80;
const N_LEVELS_QGAIN: i32 = 64;
const MIN_DELTA_GAIN_QUANT: i32 = -4;
const MAX_DELTA_GAIN_QUANT: i32 = 36;
/// `(MIN_QGAIN_DB * 128) / 6 + 16 * 128`, the gain table's zero point.
const GAIN_OFFSET: i32 = (2 * 128) / 6 + 16 * 128;
/// `(65536 * ((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6) / (N_LEVELS_QGAIN - 1)`.
const GAIN_INV_SCALE_Q16: i32 = (65536 * (((88 - 2) * 128) / 6)) / (N_LEVELS_QGAIN - 1);
const STEREO_INTERP_LEN_MS: usize = 8;
const BWE_AFTER_LOSS_Q16: i32 = 63570;
const CNG_BUF_MASK_MAX: i32 = 255;
const CNG_GAIN_SMTH_Q16: i32 = 4634;
const CNG_GAIN_SMTH_THRESHOLD_Q16: i32 = 46396;
const CNG_NLSF_SMTH_Q16: i32 = 16348;
const RAND_BUF_SIZE: usize = 128;
const RAND_BUF_MASK: i32 = RAND_BUF_SIZE as i32 - 1;
const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469;
const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565;
const MAX_PITCH_LAG_MS: i32 = 18;
const LOG2_INV_LPC_GAIN_HIGH_THRES: i32 = 3;
const LOG2_INV_LPC_GAIN_LOW_THRES: i32 = 8;
const PITCH_DRIFT_FAC_Q16: i32 = 655;
/// `0.99` in Q16 — the bandwidth expansion concealment applies to the last
/// good filter, so a repeated frame loses resonance rather than ringing.
const BWE_COEF_Q16: i32 = 64881;
const PE_MAX_LAG_MS: i32 = 18;
const PE_MIN_LAG_MS: i32 = 2;

const TYPE_NO_VOICE_ACTIVITY: i32 = 0;
const TYPE_VOICED: i32 = 2;

/// How the first gain and the LTP scaling of a frame are coded, which depends
/// on whether the frame before it is available to predict from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CondCoding {
    Independently,
    Conditionally,
    IndependentlyNoLtpScaling,
}

/// Whether the caller wants a normal decode, concealment, or the redundant
/// low-bitrate copy of the *previous* frame this packet may carry.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LostFlag {
    Normal,
    PacketLost,
    DecodeLbrr,
}

// The fixed-point primitives. Each is the reference macro of the same name;
// the Q domain of every intermediate below depends on these rounding exactly
// as written.
/// `(a * (i16)b) >> 16`.
fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

/// `a + ((b * (i16)c) >> 16)`.
fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((i64::from(b) * i64::from(c as i16)) >> 16) as i32)
}

/// `(a * b) >> 16`, both operands full width.
fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

/// `a + ((b * c) >> 16)`.
fn smlaww(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((i64::from(b) * i64::from(c)) >> 16) as i32)
}

/// `(i16)a * (i16)b`.
fn smulbb(a: i32, b: i32) -> i32 {
    i32::from(a as i16).wrapping_mul(i32::from(b as i16))
}

/// `a + (i16)b * (i16)c`.
fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(i32::from(b as i16).wrapping_mul(i32::from(c as i16)))
}

/// `(a >> 16) * (b >> 16)`.
fn smultt(a: i32, b: i32) -> i32 {
    (a >> 16).wrapping_mul(b >> 16)
}

/// `(a * b) >> 32`.
fn smmul(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

/// Right shift with rounding to nearest.
fn rshift_round(a: i32, shift: u32) -> i32 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}

fn rshift_round64(a: i64, shift: u32) -> i64 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}

fn sat16(a: i32) -> i16 {
    a.clamp(-32768, 32767) as i16
}

fn add_sat32(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

fn sub_sat32(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}

/// Left shift, clamping the input first so the result cannot overflow.
fn lshift_sat32(a: i32, shift: u32) -> i32 {
    a.clamp(i32::MIN >> shift, i32::MAX >> shift) << shift
}

fn clz32(x: i32) -> i32 {
    if x == 0 {
        32
    } else {
        x.leading_zeros() as i32
    }
}

/// The linear congruential generator SILK seeds its excitation sign and its
/// concealment noise from. Its exact sequence is part of the bitstream.
fn silk_rand(seed: i32) -> i32 {
    907633515i32.wrapping_add(seed.wrapping_mul(196314165))
}

/// Leading zeros, and the seven bits just below the leading one.
fn clz_frac(x: i32) -> (i32, i32) {
    let lzeros = clz32(x);
    // A rotate, not a shift: the reference's `silk_ROR32` takes a negative
    // count as a rotate the other way, which is what happens for a value
    // with more than 24 leading zeros.
    let rot = ((24 - lzeros) as u32) & 31;
    (lzeros, ((x as u32).rotate_right(rot) as i32) & 0x7f)
}

/// Square root to about 2.5%, which is all the concealment gain needs.
fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let (lz, frac_q7) = clz_frac(x);
    let mut y = if lz & 1 != 0 { 32768 } else { 46214 };
    y >>= lz >> 1;
    smlawb(y, y, smulbb(213, frac_q7))
}

/// `(a << qres) / b`, to about 14 bits.
fn div32_varq(a32: i32, b32: i32, qres: u32) -> i32 {
    let a_headrm = clz32(a32.wrapping_abs()) - 1;
    let mut a32_nrm = a32 << a_headrm;
    let b_headrm = clz32(b32.wrapping_abs()) - 1;
    let b32_nrm = b32 << b_headrm;
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    let mut result = smulwb(a32_nrm, b32_inv);
    // The residual is deliberately allowed to wrap: what is left of it after
    // the refinement is always small.
    a32_nrm = a32_nrm.wrapping_sub(((smmul(b32_nrm, result) as u32) << 3) as i32);
    result = smlawb(result, a32_nrm, b32_inv);
    let lshift = 29 + a_headrm - b_headrm - qres as i32;
    if lshift < 0 {
        lshift_sat32(result, (-lshift) as u32)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// `(1 << qres) / b`, to about 14 bits.
fn inverse32_varq(b32: i32, qres: u32) -> i32 {
    let b_headrm = clz32(b32.wrapping_abs()) - 1;
    let b32_nrm = b32 << b_headrm;
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    let mut result = b32_inv << 16;
    let err_q32 = (((1i32 << 29) - smulwb(b32_nrm, b32_inv)) as u32).wrapping_shl(3) as i32;
    result = smlaww(result, err_q32, b32_inv);
    let lshift = 61 - b_headrm - qres as i32;
    if lshift <= 0 {
        lshift_sat32(result, (-lshift) as u32)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// `2^(x/128)`, the inverse of the log-domain gain quantiser.
fn log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    } else if in_log_q7 >= 3967 {
        return i32::MAX;
    }
    let mut out = 1i32 << (in_log_q7 >> 7);
    let frac_q7 = in_log_q7 & 0x7F;
    let adj = smlawb(frac_q7, smulbb(frac_q7, 128 - frac_q7), -174);
    if in_log_q7 < 2048 {
        out = out.wrapping_add(out.wrapping_mul(adj) >> 7);
    } else {
        out = out.wrapping_add((out >> 7).wrapping_mul(adj));
    }
    out
}

/// Sum of squares, shifted right just far enough to keep two bits of
/// headroom, with the shift reported alongside.
fn sum_sqr_shift(x: &[i16]) -> (i32, u32) {
    let len = x.len() as i32;
    let mut shft = (31 - clz32(len)) as u32;
    let mut nrg = len;
    for pair in x.chunks(2) {
        let mut tmp = smulbb(i32::from(pair[0]), i32::from(pair[0])) as u32;
        if pair.len() == 2 {
            tmp = tmp.wrapping_add(smulbb(i32::from(pair[1]), i32::from(pair[1])) as u32);
        }
        nrg = (nrg as u32).wrapping_add(tmp >> shft) as i32;
    }
    shft = 0.max(shft as i32 + 3 - clz32(nrg)) as u32;
    nrg = 0;
    for pair in x.chunks(2) {
        let mut tmp = smulbb(i32::from(pair[0]), i32::from(pair[0])) as u32;
        if pair.len() == 2 {
            tmp = tmp.wrapping_add(smulbb(i32::from(pair[1]), i32::from(pair[1])) as u32);
        }
        nrg = (nrg as u32).wrapping_add(tmp >> shft) as i32;
    }
    (nrg, shft)
}

/// One of the two NLSF codebooks — narrow/medium band at order 10, wideband
/// at order 16 — gathered so the decoder can hold a reference to whichever
/// the current bandwidth selects.
struct NlsfCodebook {
    n_vectors: usize,
    order: usize,
    quant_step_size_q16: i32,
    cb1_nlsf_q8: &'static [u8],
    cb1_wght_q9: &'static [i16],
    cb1_icdf: &'static [u8],
    pred_q8: &'static [u8],
    ec_sel: &'static [u8],
    ec_icdf: &'static [u8],
    delta_min_q15: &'static [i16],
}

static NLSF_CB_NB_MB: NlsfCodebook = NlsfCodebook {
    n_vectors: 32,
    order: 10,
    quant_step_size_q16: 11796,
    cb1_nlsf_q8: &NLSF_CB1_NB_MB_Q8,
    cb1_wght_q9: &NLSF_CB1_WGHT_NB_MB_Q9,
    cb1_icdf: &NLSF_CB1_ICDF_NB_MB,
    pred_q8: &NLSF_PRED_NB_MB_Q8,
    ec_sel: &NLSF_CB2_SELECT_NB_MB,
    ec_icdf: &NLSF_CB2_ICDF_NB_MB,
    delta_min_q15: &NLSF_DELTA_MIN_NB_MB_Q15,
};

static NLSF_CB_WB: NlsfCodebook = NlsfCodebook {
    n_vectors: 32,
    order: 16,
    quant_step_size_q16: 9830,
    cb1_nlsf_q8: &NLSF_CB1_WB_Q8,
    cb1_wght_q9: &NLSF_CB1_WGHT_WB_Q9,
    cb1_icdf: &NLSF_CB1_ICDF_WB,
    pred_q8: &NLSF_PRED_WB_Q8,
    ec_sel: &NLSF_CB2_SELECT_WB,
    ec_icdf: &NLSF_CB2_ICDF_WB,
    delta_min_q15: &NLSF_DELTA_MIN_WB_Q15,
};

/// The three LTP gain codebooks, by periodicity index.
fn ltp_gain_vq(index: usize) -> &'static [i8] {
    match index {
        0 => &LTP_GAIN_VQ_0,
        1 => &LTP_GAIN_VQ_1,
        _ => &LTP_GAIN_VQ_2,
    }
}

fn ltp_gain_icdf(index: usize) -> &'static [u8] {
    match index {
        0 => &LTP_GAIN_ICDF_0,
        1 => &LTP_GAIN_ICDF_1,
        _ => &LTP_GAIN_ICDF_2,
    }
}

/// Everything the entropy decoder reads out of one frame's side information,
/// before any of it is turned into filters.
#[derive(Clone, Copy)]
struct Indices {
    gains: [i8; MAX_NB_SUBFR],
    ltp: [i8; MAX_NB_SUBFR],
    nlsf: [i8; MAX_LPC_ORDER + 1],
    lag_index: i16,
    contour_index: i8,
    signal_type: i32,
    quant_offset_type: i32,
    nlsf_interp_coef_q2: i32,
    per_index: usize,
    ltp_scale_index: usize,
    seed: i32,
}

impl Default for Indices {
    fn default() -> Self {
        Indices {
            gains: [0; MAX_NB_SUBFR],
            ltp: [0; MAX_NB_SUBFR],
            nlsf: [0; MAX_LPC_ORDER + 1],
            lag_index: 0,
            contour_index: 0,
            signal_type: 0,
            quant_offset_type: 0,
            nlsf_interp_coef_q2: 0,
            per_index: 0,
            ltp_scale_index: 0,
            seed: 0,
        }
    }
}

/// What one frame's side information becomes: the filters and gains the
/// synthesis actually runs.
struct FrameControl {
    /// Two LPC filters — the first half of the frame may interpolate towards
    /// the second, which is what `nlsf_interp_coef_q2` selects.
    pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    ltp_scale_q14: i32,
    pitch_l: [i32; MAX_NB_SUBFR],
    gains_q16: [i32; MAX_NB_SUBFR],
}

impl Default for FrameControl {
    fn default() -> Self {
        FrameControl {
            pred_coef_q12: [[0; MAX_LPC_ORDER]; 2],
            ltp_coef_q14: [0; LTP_ORDER * MAX_NB_SUBFR],
            ltp_scale_q14: 0,
            pitch_l: [0; MAX_NB_SUBFR],
            gains_q16: [0; MAX_NB_SUBFR],
        }
    }
}

/// Comfort noise: what a decoder plays where the encoder sent nothing.
///
/// Digital silence between talk spurts is heard as the line going dead, so
/// the decoder keeps a smoothed spectrum and gain of the background and
/// synthesises noise with it.
struct CngState {
    exc_buf_q14: [i32; MAX_FRAME_LENGTH],
    smth_nlsf_q15: [i16; MAX_LPC_ORDER],
    synth_state: [i32; MAX_LPC_ORDER],
    smth_gain_q16: i32,
    rand_seed: i32,
    fs_khz: i32,
}

impl Default for CngState {
    fn default() -> Self {
        CngState {
            exc_buf_q14: [0; MAX_FRAME_LENGTH],
            smth_nlsf_q15: [0; MAX_LPC_ORDER],
            synth_state: [0; MAX_LPC_ORDER],
            smth_gain_q16: 0,
            rand_seed: 3176576,
            fs_khz: 0,
        }
    }
}

/// What concealment needs from the last frame that arrived.
#[derive(Default)]
struct PlcState {
    ltp_coef_q14: [i16; LTP_ORDER],
    prev_lpc_q12: [i16; MAX_LPC_ORDER],
    last_frame_lost: bool,
    rand_seed: i32,
    rand_scale_q14: i32,
    conc_energy: i32,
    conc_energy_shift: u32,
    prev_ltp_scale_q14: i32,
    prev_gain_q16: [i32; 2],
    fs_khz: i32,
    nb_subfr: usize,
    subfr_length: usize,
    pitch_l_q8: i32,
}

/// Which entropy table and which backward predictor each NLSF coefficient
/// uses, both packed into one byte per pair by the first-stage index.
fn nlsf_unpack(
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

/// The second-stage residual, dequantised backwards so each coefficient's
/// predictor sees the one above it.
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
/// apart. Two line spectral frequencies that cross produce an unstable
/// filter, and the coded values are only guaranteed to be ordered before
/// quantisation.
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
            // Move the offending pair apart, keeping the centre frequency
            // where it was and inside the room the neighbours leave.
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
    // A stream corrupt enough to defeat the loop above gets the blunt fix:
    // sort, then force the minimum spacing from both ends.
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

/// Turn the coded codebook path back into a normalised line spectral
/// frequency vector.
fn nlsf_decode(nlsf_q15: &mut [i16], indices: &[i8], cb: &NlsfCodebook) {
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
        // The first-stage weights are inverse square-rooted, so the residual
        // is divided by them rather than multiplied.
        let tmp =
            ((i32::from(res_q10[i]) << 14) / i32::from(wght[i])) + (i32::from(element[i]) << 7);
        nlsf_q15[i] = tmp.clamp(0, 32767) as i16;
    }
    nlsf_stabilize(nlsf_q15, cb.delta_min_q15, cb.order);
}

/// The Q domain the NLSF-to-LPC conversion works in.
const NLSF2A_QA: u32 = 16;

/// Build one of the two symmetric polynomials whose roots are the line
/// spectral frequencies, by convolving in one root pair at a time.
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

/// Clamp a set of LPC coefficients into 16 bits, applying bandwidth
/// expansion rather than clipping while any of them is too large.
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

fn bwexpander(ar: &mut [i16], d: usize, mut chirp_q16: i32) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    for i in 0..d - 1 {
        // Not `smulwb` here: its bias would accumulate into an unstable
        // filter over the repeated expansions concealment does.
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

/// One over the prediction gain, in Q30, or zero if the filter is unstable.
///
/// This is a Levinson recursion run backwards: it recovers the reflection
/// coefficients, and a filter is stable exactly when all of them are inside
/// the unit circle.
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
        // 40 dB of prediction gain is past anything a real filter needs, and
        // a stream claiming more is corrupt.
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

fn lpc_inverse_pred_gain(a_q12: &[i16], order: usize) -> i32 {
    let mut atmp_qa = [0i32; MAX_LPC_ORDER];
    let mut dc_resp = 0i32;
    for k in 0..order {
        dc_resp += i32::from(a_q12[k]);
        atmp_qa[k] = i32::from(a_q12[k]) << (INV_GAIN_QA - 12);
    }
    // A filter whose DC response says it has a pole at zero frequency cannot
    // be stable, and the full recursion need not be run to find out.
    if dc_resp >= 4096 {
        return 0;
    }
    lpc_inverse_pred_gain_qa(&mut atmp_qa, order)
}

/// Convert normalised line spectral frequencies into the LPC filter they
/// describe.
fn nlsf2a(a_q12: &mut [i16], nlsf: &[i16], d: usize) {
    // This ordering is not cosmetic: it keeps the intermediate polynomial
    // coefficients small, which is what makes the fixed-point convolution
    // above accurate enough.
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

    // If the result is unstable, expand its bandwidth until it is not. A
    // stable filter is a hard requirement: the synthesis below is an IIR.
    let mut i = 0;
    while lpc_inverse_pred_gain(a_q12, d) == 0 && i < 16 {
        bwexpander_32(&mut a32_qa1, d, 65536 - (2 << i));
        for k in 0..d {
            a_q12[k] = rshift_round(a32_qa1[k], NLSF2A_QA + 1 - 12) as i16;
        }
        i += 1;
    }
}

/// Turn the coded gain indices back into linear per-subframe gains.
///
/// The first gain of a frame is either absolute or a delta from the previous
/// frame's last; the rest are always deltas, with a coarser step once they
/// run past the top of the table.
fn gains_dequant(
    gain_q16: &mut [i32],
    ind: &[i8],
    prev_ind: &mut i8,
    conditional: bool,
    nb_subfr: usize,
) {
    for k in 0..nb_subfr {
        if k == 0 && !conditional {
            // A gain may not fall more than 16 steps, about 21.8 dB, in one
            // jump — that would be a click, not a fade.
            *prev_ind = i8::max(ind[k], prev_ind.saturating_sub(16));
        } else {
            let ind_tmp = i32::from(ind[k]) + MIN_DELTA_GAIN_QUANT;
            let double_step_size_threshold =
                2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + i32::from(*prev_ind);
            let next = if ind_tmp > double_step_size_threshold {
                i32::from(*prev_ind) + (ind_tmp << 1) - double_step_size_threshold
            } else {
                i32::from(*prev_ind) + ind_tmp
            };
            *prev_ind = next.clamp(-128, 127) as i8;
        }
        *prev_ind = (i32::from(*prev_ind)).clamp(0, N_LEVELS_QGAIN - 1) as i8;
        gain_q16[k] =
            log2lin((smulwb(GAIN_INV_SCALE_Q16, i32::from(*prev_ind)) + GAIN_OFFSET).min(3967));
    }
}

/// The per-subframe pitch lags, as a base lag plus a coded contour.
fn decode_pitch(
    lag_index: i16,
    contour_index: i8,
    pitch_lags: &mut [i32],
    fs_khz: i32,
    nb_subfr: usize,
) {
    let (cb, cbk_size): (&[i8], usize) = if fs_khz == 8 {
        if nb_subfr == MAX_NB_SUBFR {
            (&CB_LAGS_STAGE2, 11)
        } else {
            (&CB_LAGS_STAGE2_10MS, 3)
        }
    } else if nb_subfr == MAX_NB_SUBFR {
        (&CB_LAGS_STAGE3, 34)
    } else {
        (&CB_LAGS_STAGE3_10MS, 12)
    };
    let min_lag = PE_MIN_LAG_MS * fs_khz;
    let max_lag = PE_MAX_LAG_MS * fs_khz;
    let lag = min_lag + i32::from(lag_index);
    for k in 0..nb_subfr {
        pitch_lags[k] =
            (lag + i32::from(cb[k * cbk_size + contour_index as usize])).clamp(min_lag, max_lag);
    }
}

/// One node of the shell code: split `p` pulses between two halves.
fn decode_split(dec: &mut RangeDecoder, p: i32, table: &[u8]) -> (i32, i32) {
    if p > 0 {
        let child1 =
            dec.decode_icdf(&table[SHELL_CODE_TABLE_OFFSETS[p as usize] as usize..], 8) as i32;
        (child1, p - child1)
    } else {
        (0, 0)
    }
}

/// Distribute one block's pulse count over its sixteen positions, splitting
/// in half four times.
fn shell_decoder(pulses0: &mut [i16], dec: &mut RangeDecoder, pulses4: i32) {
    let mut pulses3 = [0i32; 2];
    let mut pulses2 = [0i32; 4];
    let mut pulses1 = [0i32; 8];

    let (a, b) = decode_split(dec, pulses4, &SHELL_CODE_TABLE3);
    pulses3[0] = a;
    pulses3[1] = b;
    for (i, &parent) in [pulses3[0], pulses3[1]].iter().enumerate() {
        let (a, b) = decode_split(dec, parent, &SHELL_CODE_TABLE2);
        pulses2[2 * i] = a;
        pulses2[2 * i + 1] = b;
        for j in 0..2 {
            let (a, b) = decode_split(dec, pulses2[2 * i + j], &SHELL_CODE_TABLE1);
            pulses1[4 * i + 2 * j] = a;
            pulses1[4 * i + 2 * j + 1] = b;
            for k in 0..2 {
                let (a, b) = decode_split(dec, pulses1[4 * i + 2 * j + k], &SHELL_CODE_TABLE0);
                pulses0[8 * i + 4 * j + 2 * k] = a as i16;
                pulses0[8 * i + 4 * j + 2 * k + 1] = b as i16;
            }
        }
    }
}

/// Attach a sign to every non-zero pulse. The sign's probability depends on
/// how many pulses the block holds, because a dense block is more likely to
/// be noise than a sparse one.
fn decode_signs(
    dec: &mut RangeDecoder,
    pulses: &mut [i16],
    length: usize,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    let base = 7 * ((quant_offset_type + (signal_type << 1)) as usize);
    let blocks = (length + SHELL_CODEC_FRAME_LENGTH / 2) >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    for i in 0..blocks {
        let p = sum_pulses[i];
        if p > 0 {
            let icdf = [SIGN_ICDF[base + (p & 0x1F).min(6) as usize], 0];
            for j in 0..SHELL_CODEC_FRAME_LENGTH {
                let at = i * SHELL_CODEC_FRAME_LENGTH + j;
                if pulses[at] > 0 {
                    pulses[at] *= (dec.decode_icdf(&icdf, 8) as i16) * 2 - 1;
                }
            }
        }
    }
}

/// Decode the whole excitation: a rate level, a pulse count per 16-sample
/// block, the shell code that places them, any extra low bits, and signs.
fn decode_pulses(
    dec: &mut RangeDecoder,
    pulses: &mut [i16],
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: usize,
) {
    let rate_level = dec.decode_icdf(&RATE_LEVELS_ICDF[(signal_type >> 1) as usize * 9..], 8);

    let mut iter = frame_length >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    if iter * SHELL_CODEC_FRAME_LENGTH < frame_length {
        // Only 10 ms at 12 kHz, whose 120 samples are not a whole number of
        // shell blocks.
        iter += 1;
    }

    let mut sum_pulses = [0i32; MAX_NB_SHELL_BLOCKS];
    let mut n_lshifts = [0i32; MAX_NB_SHELL_BLOCKS];
    for i in 0..iter {
        sum_pulses[i] = dec.decode_icdf(&PULSES_PER_BLOCK_ICDF[rate_level * 18..], 8) as i32;
        // A block too loud for the table codes its low bits separately, one
        // shift at a time.
        while sum_pulses[i] == SILK_MAX_PULSES + 1 {
            n_lshifts[i] += 1;
            let table = &PULSES_PER_BLOCK_ICDF
                [(N_RATE_LEVELS - 1) * 18 + usize::from(n_lshifts[i] == 10)..];
            sum_pulses[i] = dec.decode_icdf(table, 8) as i32;
        }
    }

    for i in 0..iter {
        let at = i * SHELL_CODEC_FRAME_LENGTH;
        if sum_pulses[i] > 0 {
            shell_decoder(
                &mut pulses[at..at + SHELL_CODEC_FRAME_LENGTH],
                dec,
                sum_pulses[i],
            );
        } else {
            pulses[at..at + SHELL_CODEC_FRAME_LENGTH].fill(0);
        }
    }

    for i in 0..iter {
        if n_lshifts[i] > 0 {
            let n_ls = n_lshifts[i];
            let at = i * SHELL_CODEC_FRAME_LENGTH;
            for k in 0..SHELL_CODEC_FRAME_LENGTH {
                let mut abs_q = i32::from(pulses[at + k]);
                for _ in 0..n_ls {
                    abs_q <<= 1;
                    abs_q += dec.decode_icdf(&LSB_ICDF, 8) as i32;
                }
                pulses[at + k] = abs_q as i16;
            }
            // Mark the block as non-empty for the sign decoder, which keys
            // off the same count.
            sum_pulses[i] |= n_ls << 5;
        }
    }

    decode_signs(
        dec,
        pulses,
        frame_length,
        signal_type,
        quant_offset_type,
        &sum_pulses,
    );
}

/// The LPC analysis filter: run the signal through `1 - A(z)` to recover the
/// excitation that produced it. Concealment and re-whitening both need the
/// history in the excitation domain rather than the signal domain.
fn lpc_analysis_filter(out: &mut [i16], input: &[i16], b_q12: &[i16], len: usize, d: usize) {
    for ix in d..len {
        let mut acc = 0i32;
        for j in 0..d {
            acc = smlabb(acc, i32::from(input[ix - 1 - j]), i32::from(b_q12[j]));
        }
        // Allowed to wrap: two wraps cancel, and only an invalid stream can
        // get here at all.
        let residual = (i32::from(input[ix]) << 12).wrapping_sub(acc);
        out[ix] = sat16(rshift_round(residual, 12));
    }
    out[..d].fill(0);
}

/// One coded channel's decoder. A stereo stream has two of these; the second
/// carries the side signal, and may be absent for frames the encoder decided
/// were effectively mono.
struct ChannelState {
    fs_khz: i32,
    fs_api_hz: u32,
    nb_subfr: usize,
    frame_length: usize,
    subfr_length: usize,
    ltp_mem_length: usize,
    lpc_order: usize,
    nlsf_cb: &'static NlsfCodebook,
    pitch_lag_low_bits_icdf: &'static [u8],
    pitch_contour_icdf: &'static [u8],

    prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    first_frame_after_reset: bool,
    ec_prev_signal_type: i32,
    ec_prev_lag_index: i16,

    vad_flags: [bool; 3],
    lbrr_flag: bool,
    lbrr_flags: [bool; 3],
    n_frames_per_packet: usize,
    n_frames_decoded: usize,

    indices: Indices,
    exc_q14: [i32; MAX_FRAME_LENGTH],
    s_lpc_q14_buf: [i32; MAX_LPC_ORDER],
    out_buf: [i16; MAX_FRAME_LENGTH * 2],
    lag_prev: i32,
    last_gain_index: i8,
    prev_signal_type: i32,
    loss_cnt: i32,
    prev_gain_q16: i32,

    cng: CngState,
    plc: PlcState,
    resampler: Resampler,
}

impl ChannelState {
    fn new() -> Self {
        let mut state = ChannelState {
            fs_khz: 0,
            fs_api_hz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            lpc_order: MIN_LPC_ORDER,
            nlsf_cb: &NLSF_CB_NB_MB,
            pitch_lag_low_bits_icdf: &UNIFORM8_ICDF,
            pitch_contour_icdf: &PITCH_CONTOUR_ICDF,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            first_frame_after_reset: true,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            vad_flags: [false; 3],
            lbrr_flag: false,
            lbrr_flags: [false; 3],
            n_frames_per_packet: 0,
            n_frames_decoded: 0,
            indices: Indices::default(),
            exc_q14: [0; MAX_FRAME_LENGTH],
            s_lpc_q14_buf: [0; MAX_LPC_ORDER],
            out_buf: [0; MAX_FRAME_LENGTH * 2],
            lag_prev: 0,
            last_gain_index: 0,
            prev_signal_type: 0,
            loss_cnt: 0,
            prev_gain_q16: 65536,
            cng: CngState::default(),
            plc: PlcState::default(),
            resampler: Resampler::default(),
        };
        state.cng_reset();
        state.plc_reset();
        state
    }

    fn cng_reset(&mut self) {
        let step_q15 = 32767 / (self.lpc_order as i32 + 1);
        let mut acc = 0i32;
        for i in 0..self.lpc_order {
            acc += step_q15;
            self.cng.smth_nlsf_q15[i] = acc as i16;
        }
        self.cng.smth_gain_q16 = 0;
        self.cng.rand_seed = 3176576;
    }

    fn plc_reset(&mut self) {
        self.plc.pitch_l_q8 = (self.frame_length as i32) << 7;
        self.plc.prev_gain_q16 = [65536, 65536];
        self.plc.subfr_length = 20;
        self.plc.nb_subfr = 2;
    }

    /// Re-derive everything that depends on the internal or output rate. A
    /// change of either resets the filter history: the old state describes a
    /// signal at a different rate and would be read as a discontinuity.
    fn set_fs(&mut self, fs_khz: i32, fs_api_hz: u32) {
        self.subfr_length = SUB_FRAME_LENGTH_MS * fs_khz as usize;
        let frame_length = self.nb_subfr * self.subfr_length;

        if self.fs_khz != fs_khz || self.fs_api_hz != fs_api_hz {
            self.resampler = Resampler::new(fs_khz as u32 * 1000, fs_api_hz);
            self.fs_api_hz = fs_api_hz;
        }

        if self.fs_khz != fs_khz || frame_length != self.frame_length {
            self.pitch_contour_icdf = if fs_khz == 8 {
                if self.nb_subfr == MAX_NB_SUBFR {
                    &PITCH_CONTOUR_NB_ICDF
                } else {
                    &PITCH_CONTOUR_10MS_NB_ICDF
                }
            } else if self.nb_subfr == MAX_NB_SUBFR {
                &PITCH_CONTOUR_ICDF
            } else {
                &PITCH_CONTOUR_10MS_ICDF
            };
            if self.fs_khz != fs_khz {
                self.ltp_mem_length = LTP_MEM_LENGTH_MS * fs_khz as usize;
                if fs_khz == 8 || fs_khz == 12 {
                    self.lpc_order = MIN_LPC_ORDER;
                    self.nlsf_cb = &NLSF_CB_NB_MB;
                } else {
                    self.lpc_order = MAX_LPC_ORDER;
                    self.nlsf_cb = &NLSF_CB_WB;
                }
                self.pitch_lag_low_bits_icdf = match fs_khz {
                    16 => &UNIFORM8_ICDF,
                    12 => &UNIFORM6_ICDF,
                    _ => &UNIFORM4_ICDF,
                };
                self.first_frame_after_reset = true;
                self.lag_prev = 100;
                self.last_gain_index = 10;
                self.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
                self.out_buf.fill(0);
                self.s_lpc_q14_buf.fill(0);
            }
            self.fs_khz = fs_khz;
            self.frame_length = frame_length;
        }
    }

    /// Read one frame's side information.
    fn decode_indices(
        &mut self,
        dec: &mut RangeDecoder,
        frame_index: usize,
        decode_lbrr: bool,
        cond_coding: CondCoding,
    ) {
        let ix = if decode_lbrr || self.vad_flags[frame_index] {
            dec.decode_icdf(&TYPE_OFFSET_VAD_ICDF, 8) as i32 + 2
        } else {
            dec.decode_icdf(&TYPE_OFFSET_NO_VAD_ICDF, 8) as i32
        };
        self.indices.signal_type = ix >> 1;
        self.indices.quant_offset_type = ix & 1;

        if cond_coding == CondCoding::Conditionally {
            self.indices.gains[0] = dec.decode_icdf(&DELTA_GAIN_ICDF, 8) as i8;
        } else {
            // Independent coding: three MSBs against a signal-type-dependent
            // model, then three raw LSBs.
            let msb =
                dec.decode_icdf(&GAIN_ICDF[self.indices.signal_type as usize * 8..], 8) as i32;
            let lsb = dec.decode_icdf(&UNIFORM8_ICDF, 8) as i32;
            self.indices.gains[0] = ((msb << 3) + lsb) as i8;
        }
        for i in 1..self.nb_subfr {
            self.indices.gains[i] = dec.decode_icdf(&DELTA_GAIN_ICDF, 8) as i8;
        }

        let cb = self.nlsf_cb;
        self.indices.nlsf[0] = dec.decode_icdf(
            &cb.cb1_icdf[(self.indices.signal_type >> 1) as usize * cb.n_vectors..],
            8,
        ) as i8;
        let (ec_ix, _) = nlsf_unpack(cb, self.indices.nlsf[0] as usize);
        for i in 0..cb.order {
            let mut value = dec.decode_icdf(&cb.ec_icdf[ec_ix[i]..], 8) as i32;
            // The ends of the residual alphabet are escapes into a
            // geometric tail, so a large deviation stays codeable.
            if value == 0 {
                value -= dec.decode_icdf(&NLSF_EXT_ICDF, 8) as i32;
            } else if value == 2 * NLSF_QUANT_MAX_AMPLITUDE {
                value += dec.decode_icdf(&NLSF_EXT_ICDF, 8) as i32;
            }
            self.indices.nlsf[i + 1] = (value - NLSF_QUANT_MAX_AMPLITUDE) as i8;
        }

        self.indices.nlsf_interp_coef_q2 = if self.nb_subfr == MAX_NB_SUBFR {
            dec.decode_icdf(&NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i32
        } else {
            4
        };

        if self.indices.signal_type == TYPE_VOICED {
            let mut decode_absolute = true;
            if cond_coding == CondCoding::Conditionally && self.ec_prev_signal_type == TYPE_VOICED {
                let delta = dec.decode_icdf(&PITCH_DELTA_ICDF, 8) as i32;
                if delta > 0 {
                    self.indices.lag_index = (i32::from(self.ec_prev_lag_index) + delta - 9) as i16;
                    decode_absolute = false;
                }
            }
            if decode_absolute {
                let high = dec.decode_icdf(&PITCH_LAG_ICDF, 8) as i32 * (self.fs_khz >> 1);
                let low = dec.decode_icdf(self.pitch_lag_low_bits_icdf, 8) as i32;
                self.indices.lag_index = (high + low) as i16;
            }
            self.ec_prev_lag_index = self.indices.lag_index;

            self.indices.contour_index = dec.decode_icdf(self.pitch_contour_icdf, 8) as i8;
            self.indices.per_index = dec.decode_icdf(&LTP_PER_INDEX_ICDF, 8);
            for k in 0..self.nb_subfr {
                self.indices.ltp[k] =
                    dec.decode_icdf(ltp_gain_icdf(self.indices.per_index), 8) as i8;
            }
            self.indices.ltp_scale_index = if cond_coding == CondCoding::Independently {
                dec.decode_icdf(&LTP_SCALE_ICDF, 8)
            } else {
                0
            };
        }
        self.ec_prev_signal_type = self.indices.signal_type;
        self.indices.seed = dec.decode_icdf(&UNIFORM4_ICDF, 8) as i32;
    }

    /// Turn the side information into the filters and gains synthesis runs.
    fn decode_parameters(&mut self, ctrl: &mut FrameControl, cond_coding: CondCoding) {
        gains_dequant(
            &mut ctrl.gains_q16,
            &self.indices.gains,
            &mut self.last_gain_index,
            cond_coding == CondCoding::Conditionally,
            self.nb_subfr,
        );

        let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
        nlsf_decode(&mut nlsf_q15, &self.indices.nlsf, self.nlsf_cb);
        nlsf2a(&mut ctrl.pred_coef_q12[1], &nlsf_q15, self.lpc_order);

        // A decoder that has just reset has no previous NLSFs to interpolate
        // from, and using the zeroed ones would ring.
        if self.first_frame_after_reset {
            self.indices.nlsf_interp_coef_q2 = 4;
        }
        if self.indices.nlsf_interp_coef_q2 < 4 {
            let mut nlsf0_q15 = [0i16; MAX_LPC_ORDER];
            for i in 0..self.lpc_order {
                nlsf0_q15[i] = (i32::from(self.prev_nlsf_q15[i])
                    + ((self.indices.nlsf_interp_coef_q2
                        * (i32::from(nlsf_q15[i]) - i32::from(self.prev_nlsf_q15[i])))
                        >> 2)) as i16;
            }
            nlsf2a(&mut ctrl.pred_coef_q12[0], &nlsf0_q15, self.lpc_order);
        } else {
            ctrl.pred_coef_q12[0] = ctrl.pred_coef_q12[1];
        }
        self.prev_nlsf_q15[..self.lpc_order].copy_from_slice(&nlsf_q15[..self.lpc_order]);

        // After a loss the filter is a guess; widening it keeps the guess
        // from ringing when the real signal comes back.
        if self.loss_cnt != 0 {
            bwexpander(
                &mut ctrl.pred_coef_q12[0],
                self.lpc_order,
                BWE_AFTER_LOSS_Q16,
            );
            bwexpander(
                &mut ctrl.pred_coef_q12[1],
                self.lpc_order,
                BWE_AFTER_LOSS_Q16,
            );
        }

        if self.indices.signal_type == TYPE_VOICED {
            decode_pitch(
                self.indices.lag_index,
                self.indices.contour_index,
                &mut ctrl.pitch_l,
                self.fs_khz,
                self.nb_subfr,
            );
            let cbk = ltp_gain_vq(self.indices.per_index);
            for k in 0..self.nb_subfr {
                let ix = self.indices.ltp[k] as usize;
                for i in 0..LTP_ORDER {
                    ctrl.ltp_coef_q14[k * LTP_ORDER + i] = i16::from(cbk[ix * LTP_ORDER + i]) << 7;
                }
            }
            ctrl.ltp_scale_q14 = i32::from(LTP_SCALES_Q14[self.indices.ltp_scale_index]);
        } else {
            ctrl.pitch_l = [0; MAX_NB_SUBFR];
            ctrl.ltp_coef_q14 = [0; LTP_ORDER * MAX_NB_SUBFR];
            self.indices.per_index = 0;
            ctrl.ltp_scale_q14 = 0;
        }
    }
}

impl ChannelState {
    /// Synthesis: pulses become excitation, excitation goes through the pitch
    /// predictor and then the LPC filter, and the result is scaled by the
    /// subframe gain.
    fn decode_core(&mut self, ctrl: &mut FrameControl, xq: &mut [i16], pulses: &[i16]) {
        let offset_q10 = i32::from(
            QUANTIZATION_OFFSETS_Q10[(self.indices.signal_type >> 1) as usize * 2
                + self.indices.quant_offset_type as usize],
        );
        let nlsf_interpolation_flag = self.indices.nlsf_interp_coef_q2 < 4;

        // The excitation. Each pulse is pulled towards zero by the dead zone
        // the quantiser left, pushed away by the frame's offset, and given a
        // pseudo-random sign — the sign is not coded, only its seed.
        let mut rand_seed = self.indices.seed;
        for i in 0..self.frame_length {
            rand_seed = silk_rand(rand_seed);
            let mut e = i32::from(pulses[i]) << 14;
            if e > 0 {
                e -= QUANT_LEVEL_ADJUST_Q10 << 4;
            } else if e < 0 {
                e += QUANT_LEVEL_ADJUST_Q10 << 4;
            }
            e += offset_q10 << 4;
            if rand_seed < 0 {
                e = e.wrapping_neg();
            }
            self.exc_q14[i] = e;
            rand_seed = rand_seed.wrapping_add(i32::from(pulses[i]));
        }

        let mut s_lpc_q14 = vec![0i32; self.subfr_length + MAX_LPC_ORDER];
        s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&self.s_lpc_q14_buf);
        let mut s_ltp = vec![0i16; self.ltp_mem_length];
        let mut s_ltp_q15 = vec![0i32; self.ltp_mem_length + self.frame_length];
        let mut res_q14 = vec![0i32; self.subfr_length];
        let mut s_ltp_buf_idx = self.ltp_mem_length;
        let mut lag = 0usize;

        for k in 0..self.nb_subfr {
            let a_q12 = ctrl.pred_coef_q12[k >> 1];
            let mut b_q14 = [0i16; LTP_ORDER];
            b_q14.copy_from_slice(&ctrl.ltp_coef_q14[k * LTP_ORDER..k * LTP_ORDER + LTP_ORDER]);
            let mut signal_type = self.indices.signal_type;

            let gain_q10 = ctrl.gains_q16[k] >> 6;
            let mut inv_gain_q31 = inverse32_varq(ctrl.gains_q16[k], 47);

            // A gain change rescales the filter state rather than being
            // applied to the output, so the filter itself never sees a step.
            let gain_adj_q16 = if ctrl.gains_q16[k] != self.prev_gain_q16 {
                let adj = div32_varq(self.prev_gain_q16, ctrl.gains_q16[k], 16);
                for v in s_lpc_q14[..MAX_LPC_ORDER].iter_mut() {
                    *v = smulww(adj, *v);
                }
                adj
            } else {
                1 << 16
            };
            self.prev_gain_q16 = ctrl.gains_q16[k];

            // Going straight from concealed voiced audio to real unvoiced
            // audio drops the pitch abruptly, which is heard as a click.
            if self.loss_cnt != 0
                && self.prev_signal_type == TYPE_VOICED
                && self.indices.signal_type != TYPE_VOICED
                && k < MAX_NB_SUBFR / 2
            {
                b_q14 = [0; LTP_ORDER];
                b_q14[LTP_ORDER / 2] = 4096;
                signal_type = TYPE_VOICED;
                ctrl.pitch_l[k] = self.lag_prev;
            }

            if signal_type == TYPE_VOICED {
                lag = ctrl.pitch_l[k] as usize;
                if k == 0 || (k == 2 && nlsf_interpolation_flag) {
                    // The LPC filter just changed, so the pitch history has
                    // to be re-whitened through the new one.
                    let start_idx = self.ltp_mem_length - lag - self.lpc_order - LTP_ORDER / 2;
                    if k == 2 {
                        self.out_buf
                            [self.ltp_mem_length..self.ltp_mem_length + 2 * self.subfr_length]
                            .copy_from_slice(&xq[..2 * self.subfr_length]);
                    }
                    let src_start = start_idx + k * self.subfr_length;
                    let mut whitened = vec![0i16; self.ltp_mem_length - start_idx];
                    lpc_analysis_filter(
                        &mut whitened,
                        &self.out_buf[src_start..src_start + self.ltp_mem_length - start_idx],
                        &a_q12,
                        self.ltp_mem_length - start_idx,
                        self.lpc_order,
                    );
                    s_ltp[start_idx..self.ltp_mem_length].copy_from_slice(&whitened);

                    if k == 0 {
                        // Scale the pitch history down so this frame depends
                        // less on the last one — which is what makes a lost
                        // packet recoverable rather than permanent.
                        inv_gain_q31 = smulwb(inv_gain_q31, ctrl.ltp_scale_q14) << 2;
                    }
                    for i in 0..lag + LTP_ORDER / 2 {
                        s_ltp_q15[s_ltp_buf_idx - i - 1] =
                            smulwb(inv_gain_q31, i32::from(s_ltp[self.ltp_mem_length - i - 1]));
                    }
                } else if gain_adj_q16 != 1 << 16 {
                    for i in 0..lag + LTP_ORDER / 2 {
                        s_ltp_q15[s_ltp_buf_idx - i - 1] =
                            smulww(gain_adj_q16, s_ltp_q15[s_ltp_buf_idx - i - 1]);
                    }
                }
            }

            let pexc = k * self.subfr_length;
            if signal_type == TYPE_VOICED {
                let mut pred_lag = s_ltp_buf_idx - lag + LTP_ORDER / 2;
                for i in 0..self.subfr_length {
                    // The 2 is a rounding offset: `smlawb` truncates towards
                    // negative infinity, and five of them would bias the
                    // prediction down.
                    let mut ltp_pred_q13 = 2i32;
                    for j in 0..LTP_ORDER {
                        ltp_pred_q13 =
                            smlawb(ltp_pred_q13, s_ltp_q15[pred_lag - j], i32::from(b_q14[j]));
                    }
                    pred_lag += 1;
                    res_q14[i] = self.exc_q14[pexc + i].wrapping_add(ltp_pred_q13 << 1);
                    s_ltp_q15[s_ltp_buf_idx] = res_q14[i] << 1;
                    s_ltp_buf_idx += 1;
                }
            }

            for i in 0..self.subfr_length {
                let mut lpc_pred_q10 = (self.lpc_order >> 1) as i32;
                for j in 0..self.lpc_order {
                    lpc_pred_q10 = smlawb(
                        lpc_pred_q10,
                        s_lpc_q14[MAX_LPC_ORDER + i - 1 - j],
                        i32::from(a_q12[j]),
                    );
                }
                let excitation = if signal_type == TYPE_VOICED {
                    res_q14[i]
                } else {
                    self.exc_q14[pexc + i]
                };
                s_lpc_q14[MAX_LPC_ORDER + i] = add_sat32(excitation, lshift_sat32(lpc_pred_q10, 4));
                xq[k * self.subfr_length + i] = sat16(rshift_round(
                    smulww(s_lpc_q14[MAX_LPC_ORDER + i], gain_q10),
                    8,
                ));
            }
            s_lpc_q14.copy_within(self.subfr_length..self.subfr_length + MAX_LPC_ORDER, 0);
        }
        self.s_lpc_q14_buf
            .copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
    }
}

impl ChannelState {
    /// Keep the state concealment would need, from a frame that arrived.
    fn plc_update(&mut self, ctrl: &FrameControl) {
        self.prev_signal_type = self.indices.signal_type;
        let mut ltp_gain_q14 = 0i32;
        if self.indices.signal_type == TYPE_VOICED {
            // Use the last subframe that actually contains a pitch pulse:
            // one that does not has no gain worth carrying forward.
            let mut j = 0usize;
            while j * self.subfr_length < ctrl.pitch_l[self.nb_subfr - 1] as usize {
                if j == self.nb_subfr {
                    break;
                }
                let base = (self.nb_subfr - 1 - j) * LTP_ORDER;
                let temp: i32 = ctrl.ltp_coef_q14[base..base + LTP_ORDER]
                    .iter()
                    .map(|&v| i32::from(v))
                    .sum();
                if temp > ltp_gain_q14 {
                    ltp_gain_q14 = temp;
                    self.plc
                        .ltp_coef_q14
                        .copy_from_slice(&ctrl.ltp_coef_q14[base..base + LTP_ORDER]);
                    self.plc.pitch_l_q8 = ctrl.pitch_l[self.nb_subfr - 1 - j] << 8;
                }
                j += 1;
            }
            self.plc.ltp_coef_q14 = [0; LTP_ORDER];
            self.plc.ltp_coef_q14[LTP_ORDER / 2] = ltp_gain_q14 as i16;

            // Hold the concealment's pitch gain in a range that neither dies
            // out immediately nor rings forever.
            if ltp_gain_q14 < V_PITCH_GAIN_START_MIN_Q14 {
                let scale_q10 = (V_PITCH_GAIN_START_MIN_Q14 << 10) / ltp_gain_q14.max(1);
                for v in self.plc.ltp_coef_q14.iter_mut() {
                    *v = (smulbb(i32::from(*v), scale_q10) >> 10) as i16;
                }
            } else if ltp_gain_q14 > V_PITCH_GAIN_START_MAX_Q14 {
                let scale_q14 = (V_PITCH_GAIN_START_MAX_Q14 << 14) / ltp_gain_q14.max(1);
                for v in self.plc.ltp_coef_q14.iter_mut() {
                    *v = (smulbb(i32::from(*v), scale_q14) >> 14) as i16;
                }
            }
        } else {
            self.plc.pitch_l_q8 = smulbb(self.fs_khz, 18) << 8;
            self.plc.ltp_coef_q14 = [0; LTP_ORDER];
        }

        self.plc.prev_lpc_q12[..self.lpc_order]
            .copy_from_slice(&ctrl.pred_coef_q12[1][..self.lpc_order]);
        self.plc.prev_ltp_scale_q14 = ctrl.ltp_scale_q14;
        self.plc.prev_gain_q16[0] = ctrl.gains_q16[self.nb_subfr - 2];
        self.plc.prev_gain_q16[1] = ctrl.gains_q16[self.nb_subfr - 1];
        self.plc.subfr_length = self.subfr_length;
        self.plc.nb_subfr = self.nb_subfr;
    }

    /// Extrapolate one lost frame: keep running the pitch predictor and the
    /// LPC filter, driven by noise taken from the last frame's own
    /// excitation, and fade everything down.
    fn plc_conceal(&mut self, ctrl: &mut FrameControl, frame: &mut [i16]) {
        let prev_gain_q10 = [
            self.plc.prev_gain_q16[0] >> 6,
            self.plc.prev_gain_q16[1] >> 6,
        ];
        if self.first_frame_after_reset {
            self.plc.prev_lpc_q12 = [0; MAX_LPC_ORDER];
        }

        // Drive the concealment from whichever of the last two subframes was
        // quieter, so a decaying signal is not held up by its own onset.
        let mut exc_buf = vec![0i16; 2 * self.subfr_length];
        for k in 0..2 {
            for i in 0..self.subfr_length {
                exc_buf[k * self.subfr_length + i] = sat16(
                    smulww(
                        self.exc_q14[i + (k + self.nb_subfr - 2) * self.subfr_length],
                        prev_gain_q10[k],
                    ) >> 8,
                );
            }
        }
        let (energy1, shift1) = sum_sqr_shift(&exc_buf[..self.subfr_length]);
        let (energy2, shift2) = sum_sqr_shift(&exc_buf[self.subfr_length..]);
        let rand_base = if (energy1 >> shift2) < (energy2 >> shift1) {
            0.max(
                (self.plc.nb_subfr as i32 - 1) * self.plc.subfr_length as i32
                    - RAND_BUF_SIZE as i32,
            ) as usize
        } else {
            0.max(self.plc.nb_subfr as i32 * self.plc.subfr_length as i32 - RAND_BUF_SIZE as i32)
                as usize
        };

        let mut rand_scale_q14 = self.plc.rand_scale_q14;
        let harm_gain_q15 = PLC_HARM_ATT_Q15[(self.loss_cnt as usize).min(1)];
        let mut rand_gain_q15 = if self.prev_signal_type == TYPE_VOICED {
            PLC_RAND_ATTENUATE_V_Q15[(self.loss_cnt as usize).min(1)]
        } else {
            PLC_RAND_ATTENUATE_UV_Q15[(self.loss_cnt as usize).min(1)]
        };

        bwexpander(&mut self.plc.prev_lpc_q12, self.lpc_order, BWE_COEF_Q16);
        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        a_q12[..self.lpc_order].copy_from_slice(&self.plc.prev_lpc_q12[..self.lpc_order]);

        if self.loss_cnt == 0 {
            rand_scale_q14 = 1 << 14;
            if self.prev_signal_type == TYPE_VOICED {
                for &v in self.plc.ltp_coef_q14.iter() {
                    rand_scale_q14 -= i32::from(v);
                }
                rand_scale_q14 = rand_scale_q14.max(3277);
                rand_scale_q14 = smulbb(rand_scale_q14, self.plc.prev_ltp_scale_q14) >> 14;
            } else {
                // An unvoiced frame under a resonant filter needs less noise
                // driving it, or the concealment rings.
                let inv_gain_q30 = lpc_inverse_pred_gain(&self.plc.prev_lpc_q12, self.lpc_order);
                let mut down_scale_q30 =
                    ((1i32 << 30) >> LOG2_INV_LPC_GAIN_HIGH_THRES).min(inv_gain_q30);
                down_scale_q30 = down_scale_q30.max((1i32 << 30) >> LOG2_INV_LPC_GAIN_LOW_THRES);
                down_scale_q30 <<= LOG2_INV_LPC_GAIN_HIGH_THRES;
                rand_gain_q15 = smulwb(down_scale_q30, rand_gain_q15) >> 14;
            }
        }

        let mut rand_seed = self.plc.rand_seed;
        let mut lag = rshift_round(self.plc.pitch_l_q8, 8) as usize;
        let mut s_ltp_buf_idx = self.ltp_mem_length;
        let mut b_q14 = self.plc.ltp_coef_q14;

        // Re-whiten the pitch history through the concealment filter.
        let idx = self.ltp_mem_length - lag - self.lpc_order - LTP_ORDER / 2;
        let mut s_ltp = vec![0i16; self.ltp_mem_length];
        let mut whitened = vec![0i16; self.ltp_mem_length - idx];
        lpc_analysis_filter(
            &mut whitened,
            &self.out_buf[idx..self.ltp_mem_length],
            &a_q12,
            self.ltp_mem_length - idx,
            self.lpc_order,
        );
        s_ltp[idx..self.ltp_mem_length].copy_from_slice(&whitened);

        let mut s_ltp_q14 = vec![0i32; self.ltp_mem_length + self.frame_length];
        let inv_gain_q30 = inverse32_varq(self.plc.prev_gain_q16[1], 46).min(i32::MAX >> 1);
        for i in idx + self.lpc_order..self.ltp_mem_length {
            s_ltp_q14[i] = smulwb(inv_gain_q30, i32::from(s_ltp[i]));
        }

        for _ in 0..self.nb_subfr {
            let mut pred_lag = s_ltp_buf_idx - lag + LTP_ORDER / 2;
            for _ in 0..self.subfr_length {
                let mut ltp_pred_q12 = 2i32;
                for j in 0..LTP_ORDER {
                    ltp_pred_q12 =
                        smlawb(ltp_pred_q12, s_ltp_q14[pred_lag - j], i32::from(b_q14[j]));
                }
                pred_lag += 1;
                rand_seed = silk_rand(rand_seed);
                let noise = ((rand_seed >> 25) & RAND_BUF_MASK) as usize;
                s_ltp_q14[s_ltp_buf_idx] = smlawb(
                    ltp_pred_q12,
                    self.exc_q14[rand_base + noise],
                    rand_scale_q14,
                ) << 2;
                s_ltp_buf_idx += 1;
            }
            // Fade the pitch and the noise, and let the lag drift, so a long
            // loss ends in noise rather than in a held tone.
            for v in b_q14.iter_mut() {
                *v = (smulbb(harm_gain_q15, i32::from(*v)) >> 15) as i16;
            }
            rand_scale_q14 = smulbb(rand_scale_q14, rand_gain_q15) >> 15;
            self.plc.pitch_l_q8 = smlawb(
                self.plc.pitch_l_q8,
                self.plc.pitch_l_q8,
                PITCH_DRIFT_FAC_Q16,
            );
            self.plc.pitch_l_q8 = self
                .plc
                .pitch_l_q8
                .min(smulbb(MAX_PITCH_LAG_MS, self.fs_khz) << 8);
            lag = rshift_round(self.plc.pitch_l_q8, 8) as usize;
        }

        let lpc_base = self.ltp_mem_length - MAX_LPC_ORDER;
        s_ltp_q14[lpc_base..lpc_base + MAX_LPC_ORDER].copy_from_slice(&self.s_lpc_q14_buf);
        for i in 0..self.frame_length {
            let at = lpc_base + MAX_LPC_ORDER + i;
            let mut lpc_pred_q10 = (self.lpc_order >> 1) as i32;
            for j in 0..self.lpc_order {
                lpc_pred_q10 = smlawb(lpc_pred_q10, s_ltp_q14[at - 1 - j], i32::from(a_q12[j]));
            }
            s_ltp_q14[at] = add_sat32(s_ltp_q14[at], lshift_sat32(lpc_pred_q10, 4));
            frame[i] = sat16(rshift_round(smulww(s_ltp_q14[at], prev_gain_q10[1]), 8));
        }
        self.s_lpc_q14_buf.copy_from_slice(
            &s_ltp_q14[lpc_base + MAX_LPC_ORDER + self.frame_length - MAX_LPC_ORDER..]
                [..MAX_LPC_ORDER],
        );

        self.plc.rand_seed = rand_seed;
        self.plc.rand_scale_q14 = rand_scale_q14;
        for v in ctrl.pitch_l.iter_mut() {
            *v = lag as i32;
        }
    }

    fn plc(&mut self, ctrl: &mut FrameControl, frame: &mut [i16], lost: bool) {
        if self.fs_khz != self.plc.fs_khz {
            self.plc_reset();
            self.plc.fs_khz = self.fs_khz;
        }
        if lost {
            self.plc_conceal(ctrl, frame);
            self.loss_cnt += 1;
        } else {
            self.plc_update(ctrl);
        }
    }

    /// Fade a good frame in after a concealed one, so the energy does not
    /// step back up.
    fn plc_glue_frames(&mut self, frame: &mut [i16], length: usize) {
        if self.loss_cnt != 0 {
            let (energy, shift) = sum_sqr_shift(&frame[..length]);
            self.plc.conc_energy = energy;
            self.plc.conc_energy_shift = shift;
            self.plc.last_frame_lost = true;
            return;
        }
        if self.plc.last_frame_lost {
            let (mut energy, energy_shift) = sum_sqr_shift(&frame[..length]);
            if energy_shift > self.plc.conc_energy_shift {
                self.plc.conc_energy >>= energy_shift - self.plc.conc_energy_shift;
            } else if energy_shift < self.plc.conc_energy_shift {
                energy >>= self.plc.conc_energy_shift - energy_shift;
            }
            if energy > self.plc.conc_energy {
                let lz = clz32(self.plc.conc_energy) - 1;
                self.plc.conc_energy <<= lz;
                energy >>= 0.max(24 - lz);
                let frac_q24 = self.plc.conc_energy / energy.max(1);
                let mut gain_q16 = sqrt_approx(frac_q24) << 4;
                // Four times as steep as a plain ramp, so an onset right
                // after a loss is not swallowed by the fade.
                let slope_q16 = (((1i32 << 16) - gain_q16) / length as i32) << 2;
                for v in frame[..length].iter_mut() {
                    *v = smulwb(gain_q16, i32::from(*v)) as i16;
                    gain_q16 += slope_q16;
                    if gain_q16 > 1 << 16 {
                        break;
                    }
                }
            }
        }
        self.plc.last_frame_lost = false;
    }

    /// Comfort noise: track the background while the signal is silent, and
    /// play it back over concealed frames.
    fn cng(&mut self, ctrl: &FrameControl, frame: &mut [i16], length: usize) {
        if self.fs_khz != self.cng.fs_khz {
            self.cng_reset();
            self.cng.fs_khz = self.fs_khz;
        }
        if self.loss_cnt == 0 && self.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
            for i in 0..self.lpc_order {
                let diff = i32::from(self.prev_nlsf_q15[i]) - i32::from(self.cng.smth_nlsf_q15[i]);
                self.cng.smth_nlsf_q15[i] =
                    (i32::from(self.cng.smth_nlsf_q15[i]) + smulwb(diff, CNG_NLSF_SMTH_Q16)) as i16;
            }
            let mut max_gain_q16 = 0i32;
            let mut subfr = 0usize;
            for i in 0..self.nb_subfr {
                if ctrl.gains_q16[i] > max_gain_q16 {
                    max_gain_q16 = ctrl.gains_q16[i];
                    subfr = i;
                }
            }
            self.cng.exc_buf_q14.copy_within(
                0..(self.nb_subfr - 1) * self.subfr_length,
                self.subfr_length,
            );
            self.cng.exc_buf_q14[..self.subfr_length].copy_from_slice(
                &self.exc_q14[subfr * self.subfr_length..(subfr + 1) * self.subfr_length],
            );

            for i in 0..self.nb_subfr {
                self.cng.smth_gain_q16 += smulwb(
                    ctrl.gains_q16[i] - self.cng.smth_gain_q16,
                    CNG_GAIN_SMTH_Q16,
                );
                // Track a fall faster than a rise: noise that stays too loud
                // is more audible than noise that stays too quiet.
                if smulww(self.cng.smth_gain_q16, CNG_GAIN_SMTH_THRESHOLD_Q16) > ctrl.gains_q16[i] {
                    self.cng.smth_gain_q16 = ctrl.gains_q16[i];
                }
            }
        }

        if self.loss_cnt == 0 {
            self.cng.synth_state[..self.lpc_order].fill(0);
            return;
        }

        let mut gain_q16 = smulww(self.plc.rand_scale_q14, self.plc.prev_gain_q16[1]);
        if gain_q16 >= (1 << 21) || self.cng.smth_gain_q16 > (1 << 23) {
            gain_q16 = smultt(gain_q16, gain_q16);
            gain_q16 =
                smultt(self.cng.smth_gain_q16, self.cng.smth_gain_q16).wrapping_sub(gain_q16 << 5);
            gain_q16 = sqrt_approx(gain_q16) << 16;
        } else {
            gain_q16 = smulww(gain_q16, gain_q16);
            gain_q16 =
                smulww(self.cng.smth_gain_q16, self.cng.smth_gain_q16).wrapping_sub(gain_q16 << 5);
            gain_q16 = sqrt_approx(gain_q16) << 8;
        }
        let gain_q10 = gain_q16 >> 6;

        let mut cng_sig_q14 = vec![0i32; length + MAX_LPC_ORDER];
        let mut exc_mask = CNG_BUF_MASK_MAX;
        while exc_mask > length as i32 {
            exc_mask >>= 1;
        }
        let mut seed = self.cng.rand_seed;
        for i in 0..length {
            seed = silk_rand(seed);
            let idx = ((seed >> 24) & exc_mask) as usize;
            cng_sig_q14[MAX_LPC_ORDER + i] = self.cng.exc_buf_q14[idx];
        }
        self.cng.rand_seed = seed;

        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        nlsf2a(&mut a_q12, &self.cng.smth_nlsf_q15, self.lpc_order);
        cng_sig_q14[..MAX_LPC_ORDER].copy_from_slice(&self.cng.synth_state);
        for i in 0..length {
            let mut lpc_pred_q10 = (self.lpc_order >> 1) as i32;
            for j in 0..self.lpc_order {
                lpc_pred_q10 = smlawb(
                    lpc_pred_q10,
                    cng_sig_q14[MAX_LPC_ORDER + i - 1 - j],
                    i32::from(a_q12[j]),
                );
            }
            cng_sig_q14[MAX_LPC_ORDER + i] = add_sat32(
                cng_sig_q14[MAX_LPC_ORDER + i],
                lshift_sat32(lpc_pred_q10, 4),
            );
            frame[i] = sat16(
                i32::from(frame[i])
                    + i32::from(sat16(rshift_round(
                        smulww(cng_sig_q14[MAX_LPC_ORDER + i], gain_q10),
                        8,
                    ))),
            );
        }
        self.cng
            .synth_state
            .copy_from_slice(&cng_sig_q14[length..length + MAX_LPC_ORDER]);
    }
}

/// Which of the four resampling paths a rate pair needs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ResampleMode {
    Copy,
    Up2Hq,
    IirFir,
    DownFir,
}

/// SILK's own resampler, between its internal 8/12/16 kHz and whatever rate
/// the caller asked for.
///
/// Each path is padded to the same total delay (`input_delay`), so a stream
/// that changes bandwidth mid-call does not jump forward or backward in time
/// at the switch.
#[derive(Clone)]
struct Resampler {
    fs_in_khz: usize,
    fs_out_khz: usize,
    batch_size: usize,
    inv_ratio_q16: i32,
    input_delay: usize,
    delay_buf: [i16; 48],
    s_iir: [i32; 6],
    s_fir_i16: [i16; 8],
    s_fir_i32: [i32; 36],
    mode: ResampleMode,
    fir_order: usize,
    fir_fracs: usize,
    coefs: &'static [i16],
}

impl Default for Resampler {
    fn default() -> Self {
        Resampler {
            fs_in_khz: 0,
            fs_out_khz: 0,
            batch_size: 0,
            inv_ratio_q16: 0,
            input_delay: 0,
            delay_buf: [0; 48],
            s_iir: [0; 6],
            s_fir_i16: [0; 8],
            s_fir_i32: [0; 36],
            mode: ResampleMode::Copy,
            fir_order: 0,
            fir_fracs: 0,
            coefs: &[],
        }
    }
}

/// `[8000, 12000, 16000, 24000, 48000]` to `0..=4`.
fn rate_id(rate: u32) -> usize {
    ((((rate >> 12) - u32::from(rate > 16000)) >> u32::from(rate > 24000)) - 1) as usize
}

impl Resampler {
    fn new(fs_in: u32, fs_out: u32) -> Self {
        let mut s = Resampler {
            input_delay: RESAMPLER_DELAY_MATRIX_DEC[rate_id(fs_in)][rate_id(fs_out)] as usize,
            fs_in_khz: (fs_in / 1000) as usize,
            fs_out_khz: (fs_out / 1000) as usize,
            ..Resampler::default()
        };
        s.batch_size = s.fs_in_khz * 10;

        let mut up2x = 0u32;
        if fs_out > fs_in {
            if fs_out == fs_in * 2 {
                s.mode = ResampleMode::Up2Hq;
            } else {
                s.mode = ResampleMode::IirFir;
                up2x = 1;
            }
        } else if fs_out < fs_in {
            s.mode = ResampleMode::DownFir;
            let (fracs, order, coefs): (usize, usize, &'static [i16]) = if fs_out * 4 == fs_in * 3 {
                (3, 18, &RESAMPLER_3_4_COEFS)
            } else if fs_out * 3 == fs_in * 2 {
                (2, 18, &RESAMPLER_2_3_COEFS)
            } else if fs_out * 2 == fs_in {
                (1, 24, &RESAMPLER_1_2_COEFS)
            } else if fs_out * 3 == fs_in {
                (1, 36, &RESAMPLER_1_3_COEFS)
            } else if fs_out * 4 == fs_in {
                (1, 36, &RESAMPLER_1_4_COEFS)
            } else {
                (1, 36, &RESAMPLER_1_6_COEFS)
            };
            s.fir_fracs = fracs;
            s.fir_order = order;
            s.coefs = coefs;
        }

        s.inv_ratio_q16 = (((fs_in << (14 + up2x)) / fs_out) << 2) as i32;
        // Round the ratio up, so the last output sample of a batch never
        // reads past the input it was given.
        while smulww(s.inv_ratio_q16, fs_out as i32) < (fs_in << up2x) as i32 {
            s.inv_ratio_q16 += 1;
        }
        s
    }

    /// Interpolating 2x upsampler: two all-pass chains, one per output
    /// phase, which is cheaper than a symmetric FIR for the same stopband.
    fn up2_hq(&mut self, out: &mut [i16], input: &[i16]) {
        for (k, &sample) in input.iter().enumerate() {
            let in32 = i32::from(sample) << 10;
            let mut out32_1;
            let mut out32_2;

            let y = in32 - self.s_iir[0];
            let x = smulwb(y, i32::from(RESAMPLER_UP2_HQ_0[0]));
            out32_1 = self.s_iir[0] + x;
            self.s_iir[0] = in32 + x;
            let y = out32_1 - self.s_iir[1];
            let x = smulwb(y, i32::from(RESAMPLER_UP2_HQ_0[1]));
            out32_2 = self.s_iir[1] + x;
            self.s_iir[1] = out32_1 + x;
            let y = out32_2 - self.s_iir[2];
            let x = smlawb(y, y, i32::from(RESAMPLER_UP2_HQ_0[2]));
            out32_1 = self.s_iir[2] + x;
            self.s_iir[2] = out32_2 + x;
            out[2 * k] = sat16(rshift_round(out32_1, 10));

            let y = in32 - self.s_iir[3];
            let x = smulwb(y, i32::from(RESAMPLER_UP2_HQ_1[0]));
            out32_1 = self.s_iir[3] + x;
            self.s_iir[3] = in32 + x;
            let y = out32_1 - self.s_iir[4];
            let x = smulwb(y, i32::from(RESAMPLER_UP2_HQ_1[1]));
            out32_2 = self.s_iir[4] + x;
            self.s_iir[4] = out32_1 + x;
            let y = out32_2 - self.s_iir[5];
            let x = smlawb(y, y, i32::from(RESAMPLER_UP2_HQ_1[2]));
            out32_1 = self.s_iir[5] + x;
            self.s_iir[5] = out32_2 + x;
            out[2 * k + 1] = sat16(rshift_round(out32_1, 10));
        }
    }

    /// 2x upsample, then interpolate between the doubled samples with a
    /// 12-phase fractional FIR — the general upsampling path.
    fn iir_fir(&mut self, out: &mut [i16], input: &[i16]) {
        let mut buf = vec![0i16; 2 * self.batch_size + 8];
        buf[..8].copy_from_slice(&self.s_fir_i16);
        let mut written = 0usize;
        let mut at = 0usize;
        let mut remaining = input.len();
        let mut n_samples_in;
        loop {
            n_samples_in = remaining.min(self.batch_size);
            let mut doubled = vec![0i16; 2 * n_samples_in];
            self.up2_hq(&mut doubled, &input[at..at + n_samples_in]);
            buf[8..8 + 2 * n_samples_in].copy_from_slice(&doubled);

            let max_index_q16 = (n_samples_in as i32) << 17;
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let table_index = smulwb(index_q16 & 0xFFFF, 12) as usize;
                let b = (index_q16 >> 16) as usize;
                let near = &RESAMPLER_FRAC_FIR_12[table_index * 4..table_index * 4 + 4];
                let far =
                    &RESAMPLER_FRAC_FIR_12[(11 - table_index) * 4..(11 - table_index) * 4 + 4];
                let mut res_q15 = smulbb(i32::from(buf[b]), i32::from(near[0]));
                res_q15 = smlabb(res_q15, i32::from(buf[b + 1]), i32::from(near[1]));
                res_q15 = smlabb(res_q15, i32::from(buf[b + 2]), i32::from(near[2]));
                res_q15 = smlabb(res_q15, i32::from(buf[b + 3]), i32::from(near[3]));
                res_q15 = smlabb(res_q15, i32::from(buf[b + 4]), i32::from(far[3]));
                res_q15 = smlabb(res_q15, i32::from(buf[b + 5]), i32::from(far[2]));
                res_q15 = smlabb(res_q15, i32::from(buf[b + 6]), i32::from(far[1]));
                res_q15 = smlabb(res_q15, i32::from(buf[b + 7]), i32::from(far[0]));
                out[written] = sat16(rshift_round(res_q15, 15));
                written += 1;
                index_q16 += self.inv_ratio_q16;
            }
            at += n_samples_in;
            remaining -= n_samples_in;
            if remaining == 0 {
                break;
            }
            buf.copy_within(n_samples_in << 1..(n_samples_in << 1) + 8, 0);
        }
        self.s_fir_i16
            .copy_from_slice(&buf[n_samples_in << 1..(n_samples_in << 1) + 8]);
    }

    /// Anti-alias with a second-order AR filter, then decimate with a
    /// polyphase FIR — the general downsampling path.
    fn down_fir(&mut self, out: &mut [i16], input: &[i16]) {
        let mut buf = vec![0i32; self.batch_size + self.fir_order];
        buf[..self.fir_order].copy_from_slice(&self.s_fir_i32[..self.fir_order]);
        let fir_coefs = &self.coefs[2..];
        let mut written = 0usize;
        let mut at = 0usize;
        let mut remaining = input.len();
        let mut n_samples_in;
        loop {
            n_samples_in = remaining.min(self.batch_size);
            for k in 0..n_samples_in {
                let out32 = self.s_iir[0] + (i32::from(input[at + k]) << 8);
                buf[self.fir_order + k] = out32;
                let scaled = out32 << 2;
                self.s_iir[0] = smlawb(self.s_iir[1], scaled, i32::from(self.coefs[0]));
                self.s_iir[1] = smulwb(scaled, i32::from(self.coefs[1]));
            }

            let max_index_q16 = (n_samples_in as i32) << 16;
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let b = (index_q16 >> 16) as usize;
                let res_q6 = match self.fir_order {
                    18 => {
                        let interpol_ind =
                            smulwb(index_q16 & 0xFFFF, self.fir_fracs as i32) as usize;
                        let near = &fir_coefs[9 * interpol_ind..];
                        let far = &fir_coefs[9 * (self.fir_fracs - 1 - interpol_ind)..];
                        let mut acc = smulwb(buf[b], i32::from(near[0]));
                        for j in 1..9 {
                            acc = smlawb(acc, buf[b + j], i32::from(near[j]));
                        }
                        for j in 0..9 {
                            acc = smlawb(acc, buf[b + 17 - j], i32::from(far[j]));
                        }
                        acc
                    }
                    order => {
                        let half = order / 2;
                        let mut acc = smulwb(buf[b] + buf[b + order - 1], i32::from(fir_coefs[0]));
                        for j in 1..half {
                            acc = smlawb(
                                acc,
                                buf[b + j] + buf[b + order - 1 - j],
                                i32::from(fir_coefs[j]),
                            );
                        }
                        acc
                    }
                };
                out[written] = sat16(rshift_round(res_q6, 6));
                written += 1;
                index_q16 += self.inv_ratio_q16;
            }
            at += n_samples_in;
            remaining -= n_samples_in;
            if remaining <= 1 {
                break;
            }
            buf.copy_within(n_samples_in..n_samples_in + self.fir_order, 0);
        }
        self.s_fir_i32[..self.fir_order]
            .copy_from_slice(&buf[n_samples_in..n_samples_in + self.fir_order]);
    }

    fn process(&mut self, out: &mut [i16], input: &[i16]) {
        match self.mode {
            ResampleMode::Copy => out[..input.len()].copy_from_slice(input),
            ResampleMode::Up2Hq => self.up2_hq(out, input),
            ResampleMode::IirFir => self.iir_fir(out, input),
            ResampleMode::DownFir => self.down_fir(out, input),
        }
    }

    /// Resample `input` into `out`, holding back `input_delay` samples for
    /// the next call.
    fn resample(&mut self, out: &mut [i16], input: &[i16]) {
        let in_len = input.len();
        let n_samples = self.fs_in_khz - self.input_delay;
        self.delay_buf[self.input_delay..self.input_delay + n_samples]
            .copy_from_slice(&input[..n_samples]);
        let head: Vec<i16> = self.delay_buf[..self.fs_in_khz].to_vec();
        self.process(out, &head);
        // The tail starts where the delay buffer's copy ended but stops a
        // whole millisecond short: those last `input_delay` samples are what
        // the next call will start from.
        let (_, tail) = out.split_at_mut(self.fs_out_khz);
        self.process(tail, &input[n_samples..n_samples + in_len - self.fs_in_khz]);
        self.delay_buf[..self.input_delay].copy_from_slice(&input[in_len - self.input_delay..]);
    }
}

/// The mid/side prediction state a stereo stream carries between frames.
#[derive(Default)]
struct StereoState {
    pred_prev_q13: [i32; 2],
    s_mid: [i16; 2],
    s_side: [i16; 2],
}

/// Read the two mid/side prediction weights.
fn stereo_decode_pred(dec: &mut RangeDecoder) -> [i32; 2] {
    let mut ix = [[0i32; 3]; 2];
    let n = dec.decode_icdf(&STEREO_PRED_JOINT_ICDF, 8) as i32;
    ix[0][2] = n / 5;
    ix[1][2] = n - 5 * ix[0][2];
    for row in ix.iter_mut() {
        row[0] = dec.decode_icdf(&UNIFORM3_ICDF, 8) as i32;
        row[1] = dec.decode_icdf(&UNIFORM5_ICDF, 8) as i32;
    }
    let mut pred_q13 = [0i32; 2];
    for n in 0..2 {
        ix[n][0] += 3 * ix[n][2];
        let low_q13 = i32::from(STEREO_PRED_QUANT_Q13[ix[n][0] as usize]);
        let step_q13 = smulwb(
            i32::from(STEREO_PRED_QUANT_Q13[ix[n][0] as usize + 1]) - low_q13,
            6554,
        );
        pred_q13[n] = smlabb(low_q13, step_q13, 2 * ix[n][1] + 1);
    }
    // The first weight is stored relative to the second, which is the form
    // the synthesis below wants.
    pred_q13[0] -= pred_q13[1];
    pred_q13
}

/// Turn a decoded mid/side pair back into left and right, ramping the
/// prediction weights over the first 8 ms so a change between frames is not
/// heard as a step in the stereo image.
fn stereo_ms_to_lr(
    state: &mut StereoState,
    x1: &mut [i16],
    x2: &mut [i16],
    pred_q13: &[i32; 2],
    fs_khz: usize,
    frame_length: usize,
) {
    x1[0] = state.s_mid[0];
    x1[1] = state.s_mid[1];
    x2[0] = state.s_side[0];
    x2[1] = state.s_side[1];
    state
        .s_mid
        .copy_from_slice(&x1[frame_length..frame_length + 2]);
    state
        .s_side
        .copy_from_slice(&x2[frame_length..frame_length + 2]);

    let mut pred0_q13 = state.pred_prev_q13[0];
    let mut pred1_q13 = state.pred_prev_q13[1];
    let ramp = STEREO_INTERP_LEN_MS * fs_khz;
    let denom_q16 = (1i32 << 16) / ramp as i32;
    let delta0_q13 = rshift_round(smulbb(pred_q13[0] - state.pred_prev_q13[0], denom_q16), 16);
    let delta1_q13 = rshift_round(smulbb(pred_q13[1] - state.pred_prev_q13[1], denom_q16), 16);
    for n in 0..frame_length {
        if n < ramp {
            pred0_q13 += delta0_q13;
            pred1_q13 += delta1_q13;
        } else if n == ramp {
            pred0_q13 = pred_q13[0];
            pred1_q13 = pred_q13[1];
        }
        // A three-tap smoothing of mid feeds the first predictor; mid itself
        // feeds the second.
        let mut sum = (i32::from(x1[n]) + i32::from(x1[n + 2]) + (i32::from(x1[n + 1]) << 1)) << 9;
        sum = smlawb(i32::from(x2[n + 1]) << 8, sum, pred0_q13);
        sum = smlawb(sum, i32::from(x1[n + 1]) << 11, pred1_q13);
        x2[n + 1] = sat16(rshift_round(sum, 8));
    }
    state.pred_prev_q13 = *pred_q13;

    for n in 0..frame_length {
        let sum = i32::from(x1[n + 1]) + i32::from(x2[n + 1]);
        let diff = i32::from(x1[n + 1]) - i32::from(x2[n + 1]);
        x1[n + 1] = sat16(sum);
        x2[n + 1] = sat16(diff);
    }
}

/// What the caller tells SILK about the stream it is decoding.
pub(super) struct Control {
    pub api_sample_rate: u32,
    pub channels_api: usize,
    pub channels_internal: usize,
    pub internal_sample_rate: u32,
    pub payload_size_ms: usize,
}

/// SILK could not decode the frame — a duration or rate it does not have.
#[derive(Debug)]
pub(super) struct SilkError;

/// One SILK stream's decoder: up to two coded channels plus the mid/side
/// state that joins them.
pub(super) struct SilkDecoder {
    channels: [ChannelState; 2],
    n_channels_api: usize,
    n_channels_internal: usize,
    prev_decode_only_middle: bool,
    stereo: StereoState,
}

impl SilkDecoder {
    pub(super) fn new() -> Self {
        SilkDecoder {
            channels: [ChannelState::new(), ChannelState::new()],
            n_channels_api: 0,
            n_channels_internal: 0,
            prev_decode_only_middle: false,
            stereo: StereoState::default(),
        }
    }

    /// Decode one SILK frame into `pcm`, interleaved at the API rate, and
    /// report how many samples per channel it produced.
    ///
    /// A packet may hold several SILK frames; this is called once per frame,
    /// with `first_frame` set on the first, which is where the per-packet
    /// flags are read.
    pub(super) fn decode(
        &mut self,
        control: &Control,
        lost: bool,
        first_frame: bool,
        mut dec: Option<&mut RangeDecoder>,
        pcm: &mut [i16],
    ) -> Result<usize, SilkError> {
        let lost_flag = if lost {
            LostFlag::PacketLost
        } else {
            LostFlag::Normal
        };
        let internal = control.channels_internal;

        if first_frame {
            for ch in self.channels.iter_mut() {
                ch.n_frames_decoded = 0;
            }
        }
        // A stream that turns stereo mid-call starts its side channel from
        // nothing rather than from whatever the last stereo stream left.
        if internal > self.n_channels_internal {
            self.channels[1] = ChannelState::new();
        }
        let stereo_to_mono = internal == 1
            && self.n_channels_internal == 2
            && control.internal_sample_rate == 1000 * self.channels[0].fs_khz as u32;

        if self.channels[0].n_frames_decoded == 0 {
            for n in 0..internal {
                let (frames_per_packet, nb_subfr) = match control.payload_size_ms {
                    0 | 10 => (1, 2),
                    20 => (1, 4),
                    40 => (2, 4),
                    60 => (3, 4),
                    _ => return Err(SilkError),
                };
                self.channels[n].n_frames_per_packet = frames_per_packet;
                self.channels[n].nb_subfr = nb_subfr;
                let fs_khz_dec = (control.internal_sample_rate >> 10) + 1;
                if !matches!(fs_khz_dec, 8 | 12 | 16) {
                    return Err(SilkError);
                }
                self.channels[n].set_fs(fs_khz_dec as i32, control.api_sample_rate);
            }
        }

        if control.channels_api == 2
            && internal == 2
            && (self.n_channels_api == 1 || self.n_channels_internal == 1)
        {
            self.stereo.pred_prev_q13 = [0; 2];
            self.stereo.s_side = [0; 2];
            let resampler = self.channels[0].resampler.clone();
            self.channels[1].resampler = resampler;
        }
        self.n_channels_api = control.channels_api;
        self.n_channels_internal = internal;

        let mut ms_pred_q13 = [0i32; 2];
        let mut decode_only_middle = false;

        if let Some(dec) = dec.as_deref_mut() {
            if lost_flag != LostFlag::PacketLost && self.channels[0].n_frames_decoded == 0 {
                // The per-packet header: a voice-activity flag per frame and
                // one low-bitrate-redundancy flag, for each coded channel.
                for n in 0..internal {
                    for i in 0..self.channels[n].n_frames_per_packet {
                        self.channels[n].vad_flags[i] = dec.decode_bit_logp(1);
                    }
                    self.channels[n].lbrr_flag = dec.decode_bit_logp(1);
                }
                for n in 0..internal {
                    self.channels[n].lbrr_flags = [false; 3];
                    if self.channels[n].lbrr_flag {
                        if self.channels[n].n_frames_per_packet == 1 {
                            self.channels[n].lbrr_flags[0] = true;
                        } else {
                            let table: &[u8] = if self.channels[n].n_frames_per_packet == 2 {
                                &LBRR_FLAGS_2_ICDF
                            } else {
                                &LBRR_FLAGS_3_ICDF
                            };
                            let symbol = dec.decode_icdf(table, 8) as i32 + 1;
                            for i in 0..self.channels[n].n_frames_per_packet {
                                self.channels[n].lbrr_flags[i] = (symbol >> i) & 1 != 0;
                            }
                        }
                    }
                }
                // The redundant copies are not played here, but they are in
                // the bitstream and have to be stepped over exactly.
                for i in 0..self.channels[0].n_frames_per_packet {
                    for n in 0..internal {
                        if self.channels[n].lbrr_flags[i] {
                            if internal == 2 && n == 0 {
                                stereo_decode_pred(dec);
                                if !self.channels[1].lbrr_flags[i] {
                                    dec.decode_icdf(&STEREO_ONLY_CODE_MID_ICDF, 8);
                                }
                            }
                            let cond = if i > 0 && self.channels[n].lbrr_flags[i - 1] {
                                CondCoding::Conditionally
                            } else {
                                CondCoding::Independently
                            };
                            self.channels[n].decode_indices(dec, i, true, cond);
                            let mut pulses = [0i16; MAX_FRAME_LENGTH + SHELL_CODEC_FRAME_LENGTH];
                            let (signal_type, offset_type, length) = (
                                self.channels[n].indices.signal_type,
                                self.channels[n].indices.quant_offset_type,
                                self.channels[n].frame_length,
                            );
                            decode_pulses(dec, &mut pulses, signal_type, offset_type, length);
                        }
                    }
                }
            }

            if internal == 2 {
                if lost_flag == LostFlag::Normal {
                    ms_pred_q13 = stereo_decode_pred(dec);
                    if !self.channels[1].vad_flags[self.channels[0].n_frames_decoded] {
                        decode_only_middle = dec.decode_icdf(&STEREO_ONLY_CODE_MID_ICDF, 8) != 0;
                    }
                } else {
                    ms_pred_q13 = self.stereo.pred_prev_q13;
                }
            }
        } else if internal == 2 {
            ms_pred_q13 = self.stereo.pred_prev_q13;
        }

        // The side channel's own prediction memory describes a signal that
        // was not coded last time, so it has to start again.
        if internal == 2 && !decode_only_middle && self.prev_decode_only_middle {
            self.channels[1].out_buf.fill(0);
            self.channels[1].s_lpc_q14_buf.fill(0);
            self.channels[1].lag_prev = 100;
            self.channels[1].last_gain_index = 10;
            self.channels[1].prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            self.channels[1].first_frame_after_reset = true;
        }

        let frame_length = self.channels[0].frame_length;
        let mut tmp: [Vec<i16>; 2] = [vec![0i16; frame_length + 2], vec![0i16; frame_length + 2]];

        let has_side = if lost_flag == LostFlag::Normal {
            !decode_only_middle
        } else {
            !self.prev_decode_only_middle
        };

        for n in 0..internal {
            if n == 0 || has_side {
                let frame_index = self.channels[0].n_frames_decoded - n;
                let cond = if frame_index == 0 {
                    CondCoding::Independently
                } else if n > 0 && self.prev_decode_only_middle {
                    // A skipped side frame leaves the LTP state well defined,
                    // so it needs no scaling to recover from.
                    CondCoding::IndependentlyNoLtpScaling
                } else {
                    CondCoding::Conditionally
                };
                let (a, b) = self.channels.split_at_mut(1);
                let channel = if n == 0 { &mut a[0] } else { &mut b[0] };
                channel.decode_frame(dec.as_deref_mut(), &mut tmp[n][2..], lost_flag, cond);
            } else {
                tmp[n][2..2 + frame_length].fill(0);
            }
            self.channels[n].n_frames_decoded += 1;
        }

        if control.channels_api == 2 && internal == 2 {
            let (a, b) = tmp.split_at_mut(1);
            stereo_ms_to_lr(
                &mut self.stereo,
                &mut a[0],
                &mut b[0],
                &ms_pred_q13,
                self.channels[0].fs_khz as usize,
                frame_length,
            );
        } else {
            tmp[0][0] = self.stereo.s_mid[0];
            tmp[0][1] = self.stereo.s_mid[1];
            self.stereo
                .s_mid
                .copy_from_slice(&tmp[0][frame_length..frame_length + 2]);
        }

        let n_samples_out = (frame_length * control.api_sample_rate as usize)
            / (self.channels[0].fs_khz as usize * 1000);
        let mut resampled = vec![0i16; n_samples_out];
        for n in 0..control.channels_api.min(internal) {
            self.channels[n]
                .resampler
                .resample(&mut resampled, &tmp[n][1..1 + frame_length]);
            if control.channels_api == 2 {
                for i in 0..n_samples_out {
                    pcm[n + 2 * i] = resampled[i];
                }
            } else {
                pcm[..n_samples_out].copy_from_slice(&resampled);
            }
        }

        if control.channels_api == 2 && internal == 1 {
            if stereo_to_mono {
                // The right channel's resampler has been idle; run it over
                // the same signal so it is warm if the stream goes back to
                // stereo.
                self.channels[1]
                    .resampler
                    .resample(&mut resampled, &tmp[0][1..1 + frame_length]);
                for i in 0..n_samples_out {
                    pcm[1 + 2 * i] = resampled[i];
                }
            } else {
                for i in 0..n_samples_out {
                    pcm[1 + 2 * i] = pcm[2 * i];
                }
            }
        }

        if lost_flag == LostFlag::PacketLost {
            // Drop the gain clamp: with the energy already falling, holding
            // it would make it bounce back when the stream resumes.
            for ch in self.channels.iter_mut() {
                ch.last_gain_index = 10;
            }
        } else {
            self.prev_decode_only_middle = decode_only_middle;
        }
        Ok(n_samples_out)
    }
}

impl ChannelState {
    /// Decode one SILK frame of one channel into `out`.
    fn decode_frame(
        &mut self,
        dec: Option<&mut RangeDecoder>,
        out: &mut [i16],
        lost_flag: LostFlag,
        cond_coding: CondCoding,
    ) {
        let length = self.frame_length;
        let mut ctrl = FrameControl::default();

        let decodable = lost_flag == LostFlag::Normal
            || (lost_flag == LostFlag::DecodeLbrr && self.lbrr_flags[self.n_frames_decoded]);
        match (decodable, dec) {
            (true, Some(dec)) => {
                let mut pulses = [0i16; MAX_FRAME_LENGTH + SHELL_CODEC_FRAME_LENGTH];
                self.decode_indices(dec, self.n_frames_decoded, false, cond_coding);
                let (signal_type, offset_type) =
                    (self.indices.signal_type, self.indices.quant_offset_type);
                decode_pulses(dec, &mut pulses, signal_type, offset_type, length);
                self.decode_parameters(&mut ctrl, cond_coding);
                self.decode_core(&mut ctrl, out, &pulses);
                self.plc(&mut ctrl, out, false);
                self.loss_cnt = 0;
                self.prev_signal_type = self.indices.signal_type;
                self.first_frame_after_reset = false;
            }
            _ => self.plc(&mut ctrl, out, true),
        }

        let mv_len = self.ltp_mem_length - length;
        self.out_buf.copy_within(length..length + mv_len, 0);
        self.out_buf[mv_len..mv_len + length].copy_from_slice(&out[..length]);

        self.cng(&ctrl, out, length);
        self.plc_glue_frames(out, length);
        self.lag_prev = ctrl.pitch_l[self.nb_subfr - 1];
    }
}
