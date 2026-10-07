//! The resampler to the output rate.

use crate::opus::tables_silk::*;
use super::{rshift_round, sat16, smlabb, smlawb, smulbb, smulwb, smulww};

/// Which of the four resampling paths a rate pair needs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ResampleMode {
    Copy,
    Up2Hq,
    IirFir,
    DownFir,
}

/// SILK's resampler between its internal rate and the output rate. Every path
/// is padded to the same delay so bandwidth switches do not shift time.
#[derive(Clone)]
pub(super) struct Resampler {
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
    pub(super) fn new(fs_in: u32, fs_out: u32) -> Self {
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
        // Round up so the last output sample never reads past the input.
        while smulww(s.inv_ratio_q16, fs_out as i32) < (fs_in << up2x) as i32 {
            s.inv_ratio_q16 += 1;
        }
        s
    }

    /// Interpolating 2x upsampler: two all-pass chains, one per output phase.
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

    /// 2x upsample, then a 12-phase fractional FIR: the general upsampling path.
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

    /// Second-order AR anti-alias, then a polyphase FIR: the general downsampling path.
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

    /// Resample `input` into `out`, holding back `input_delay` samples for the next call.
    pub(super) fn resample(&mut self, out: &mut [i16], input: &[i16]) {
        let in_len = input.len();
        let n_samples = self.fs_in_khz - self.input_delay;
        self.delay_buf[self.input_delay..self.input_delay + n_samples]
            .copy_from_slice(&input[..n_samples]);
        let head: Vec<i16> = self.delay_buf[..self.fs_in_khz].to_vec();
        self.process(out, &head);
        // Stop `input_delay` samples short: the next call starts from them.
        let (_, tail) = out.split_at_mut(self.fs_out_khz);
        self.process(tail, &input[n_samples..n_samples + in_len - self.fs_in_khz]);
        self.delay_buf[..self.input_delay].copy_from_slice(&input[in_len - self.input_delay..]);
    }
}
