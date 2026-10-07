//! A channel's state, side information and frame decode.

use crate::opus::range::RangeDecoder;
use crate::opus::tables_silk::*;
use super::nlsf::{bwexpander, nlsf2a, nlsf_decode, nlsf_unpack};
use super::pulses::decode_pulses;
use super::resampler::Resampler;
use super::{
    log2lin, ltp_gain_icdf, ltp_gain_vq, smulwb, ChannelState, CngState, CondCoding, FrameControl,
    Indices, LostFlag, PlcState, BWE_AFTER_LOSS_Q16, GAIN_INV_SCALE_Q16, GAIN_OFFSET,
    LTP_MEM_LENGTH_MS, LTP_ORDER, MAX_DELTA_GAIN_QUANT, MAX_FRAME_LENGTH, MAX_LPC_ORDER,
    MAX_NB_SUBFR, MIN_DELTA_GAIN_QUANT, MIN_LPC_ORDER, NLSF_CB_NB_MB, NLSF_CB_WB,
    NLSF_QUANT_MAX_AMPLITUDE, N_LEVELS_QGAIN, PE_MAX_LAG_MS, PE_MIN_LAG_MS,
    SHELL_CODEC_FRAME_LENGTH, SUB_FRAME_LENGTH_MS, TYPE_NO_VOICE_ACTIVITY, TYPE_VOICED,
};

/// Turn the coded gain indices back into linear per-subframe gains. The first
/// is absolute or a delta from the previous frame; the rest are deltas.
fn gains_dequant(
    gain_q16: &mut [i32],
    ind: &[i8],
    prev_ind: &mut i8,
    conditional: bool,
    nb_subfr: usize,
) {
    for k in 0..nb_subfr {
        if k == 0 && !conditional {
            // A gain may not fall more than 16 steps (about 21.8 dB) at once.
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

impl ChannelState {
    pub(super) fn new() -> Self {
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

    pub(super) fn cng_reset(&mut self) {
        let step_q15 = 32767 / (self.lpc_order as i32 + 1);
        let mut acc = 0i32;
        for i in 0..self.lpc_order {
            acc += step_q15;
            self.cng.smth_nlsf_q15[i] = acc as i16;
        }
        self.cng.smth_gain_q16 = 0;
        self.cng.rand_seed = 3176576;
    }

    pub(super) fn plc_reset(&mut self) {
        self.plc.pitch_l_q8 = (self.frame_length as i32) << 7;
        self.plc.prev_gain_q16 = [65536, 65536];
        self.plc.subfr_length = 20;
        self.plc.nb_subfr = 2;
    }

    /// Re-derive everything that depends on the internal or output rate. A
    /// change of either resets the filter history.
    pub(super) fn set_fs(&mut self, fs_khz: i32, fs_api_hz: u32) {
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
    pub(super) fn decode_indices(
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
            // Independent coding: three MSBs against a signal-type model, three raw LSBs.
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
            // The alphabet's ends escape into a geometric tail.
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

        // No interpolation right after a reset; the zeroed NLSFs would ring.
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

        // Widen the guessed filter after a loss so it does not ring.
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
    /// Decode one SILK frame of one channel into `out`.
    pub(super) fn decode_frame(
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
