//! Packet loss concealment and comfort noise.

use crate::opus::tables_silk::*;
use super::nlsf::{bwexpander, lpc_inverse_pred_gain, nlsf2a};
use super::synthesis::lpc_analysis_filter;
use super::{
    add_sat32, clz32, inverse32_varq, lshift_sat32, rshift_round, sat16, silk_rand, smlawb, smulbb,
    smultt, smulwb, smulww, sqrt_approx, sum_sqr_shift, ChannelState, FrameControl, BWE_COEF_Q16,
    CNG_BUF_MASK_MAX, CNG_GAIN_SMTH_Q16, CNG_GAIN_SMTH_THRESHOLD_Q16, CNG_NLSF_SMTH_Q16,
    LOG2_INV_LPC_GAIN_HIGH_THRES, LOG2_INV_LPC_GAIN_LOW_THRES, LTP_ORDER, MAX_LPC_ORDER,
    MAX_PITCH_LAG_MS, PITCH_DRIFT_FAC_Q16, RAND_BUF_MASK, RAND_BUF_SIZE, TYPE_NO_VOICE_ACTIVITY,
    TYPE_VOICED, V_PITCH_GAIN_START_MAX_Q14, V_PITCH_GAIN_START_MIN_Q14,
};

impl ChannelState {
    /// Keep the state concealment would need, from a frame that arrived.
    fn plc_update(&mut self, ctrl: &FrameControl) {
        self.prev_signal_type = self.indices.signal_type;
        let mut ltp_gain_q14 = 0i32;
        if self.indices.signal_type == TYPE_VOICED {
            // The last subframe that contains a pitch pulse.
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

    /// Extrapolate one lost frame: run the pitch predictor and LPC filter on
    /// noise from the last frame's excitation, fading down.
    fn plc_conceal(&mut self, ctrl: &mut FrameControl, frame: &mut [i16]) {
        let prev_gain_q10 = [
            self.plc.prev_gain_q16[0] >> 6,
            self.plc.prev_gain_q16[1] >> 6,
        ];
        if self.first_frame_after_reset {
            self.plc.prev_lpc_q12 = [0; MAX_LPC_ORDER];
        }

        // Use the quieter of the last two subframes, so an onset doesn't hold up a decay.
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
                // Less noise under a resonant filter, or the concealment rings.
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
            // Fade pitch and noise and let the lag drift, so long losses end in noise.
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

    pub(super) fn plc(&mut self, ctrl: &mut FrameControl, frame: &mut [i16], lost: bool) {
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

    /// Fade a good frame in after a concealed one.
    pub(super) fn plc_glue_frames(&mut self, frame: &mut [i16], length: usize) {
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
                // Four times steeper than a plain ramp, so an onset is not swallowed.
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

    /// Comfort noise: track the background while silent, play it over concealed frames.
    pub(super) fn cng(&mut self, ctrl: &FrameControl, frame: &mut [i16], length: usize) {
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
                // Track falls faster than rises.
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
