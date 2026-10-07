//! Mid/side stereo prediction.

use crate::opus::range::RangeDecoder;
use crate::opus::tables_silk::*;
use super::{rshift_round, sat16, smlabb, smlawb, smulbb, smulwb, STEREO_INTERP_LEN_MS};

/// The mid/side prediction state a stereo stream carries between frames.
#[derive(Default)]
pub(super) struct StereoState {
    pub(super) pred_prev_q13: [i32; 2],
    pub(super) s_mid: [i16; 2],
    pub(super) s_side: [i16; 2],
}

/// Read the two mid/side prediction weights.
pub(super) fn stereo_decode_pred(dec: &mut RangeDecoder) -> [i32; 2] {
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
    // The first weight is stored relative to the second.
    pred_q13[0] -= pred_q13[1];
    pred_q13
}

/// Turn a decoded mid/side pair back into left and right, ramping the
/// prediction weights over the first 8 ms.
pub(super) fn stereo_ms_to_lr(
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
        // Smoothed mid feeds the first predictor; mid itself the second.
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
