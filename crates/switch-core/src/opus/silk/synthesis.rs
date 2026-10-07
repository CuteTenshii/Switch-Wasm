//! Excitation, long-term prediction and LPC synthesis.

use crate::opus::tables_silk::*;
use super::{
    add_sat32, div32_varq, inverse32_varq, lshift_sat32, rshift_round, sat16, silk_rand, smlabb,
    smlawb, smulwb, smulww, ChannelState, FrameControl, LTP_ORDER, MAX_LPC_ORDER, MAX_NB_SUBFR,
    QUANT_LEVEL_ADJUST_Q10, TYPE_VOICED,
};

/// The LPC analysis filter `1 - A(z)`: recovers the excitation from the signal.
pub(super) fn lpc_analysis_filter(out: &mut [i16], input: &[i16], b_q12: &[i16], len: usize, d: usize) {
    for ix in d..len {
        let mut acc = 0i32;
        for j in 0..d {
            acc = smlabb(acc, i32::from(input[ix - 1 - j]), i32::from(b_q12[j]));
        }
        // Allowed to wrap; only an invalid stream gets here.
        let residual = (i32::from(input[ix]) << 12).wrapping_sub(acc);
        out[ix] = sat16(rshift_round(residual, 12));
    }
    out[..d].fill(0);
}

impl ChannelState {
    /// Synthesis: excitation through the pitch predictor and LPC filter,
    /// scaled by the subframe gain.
    pub(super) fn decode_core(&mut self, ctrl: &mut FrameControl, xq: &mut [i16], pulses: &[i16]) {
        let offset_q10 = i32::from(
            QUANTIZATION_OFFSETS_Q10[(self.indices.signal_type >> 1) as usize * 2
                + self.indices.quant_offset_type as usize],
        );
        let nlsf_interpolation_flag = self.indices.nlsf_interp_coef_q2 < 4;

        // Excitation: pulses with the dead zone removed, the frame's offset
        // added, and a pseudo-random sign (only its seed is coded).
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

            // A gain change rescales the filter state, so the filter never sees a step.
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

            // Concealed voiced to real unvoiced drops the pitch abruptly (a click).
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
                    // The LPC filter changed: re-whiten the pitch history.
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
                        // Scale the pitch history down so a lost packet is recoverable.
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
                    // Rounding offset: five truncating `smlawb`s would bias downward.
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
