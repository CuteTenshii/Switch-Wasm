//! The CELT layer of Opus (RFC 6716 §4.3): per-band energy plus a PVQ-coded
//! unit-norm shape. The bit allocation is implicit, so every count here must
//! match the encoder's integer arithmetic bit for bit.

mod allocation;
mod bands;
mod energy;
mod partition;
mod plc;
mod synthesis;
mod vq;

use allocation::{compute_allocation, init_caps};
use bands::{anti_collapse, quant_all_bands, tf_decode};
use energy::{unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy};
use partition::lcg_rand;
use plc::{celt_autocorr, celt_fir, celt_iir, celt_lpc, pitch_downsample, pitch_search};
use synthesis::{celt_synthesis, comb_filter, comb_filter_const, deemphasis};
use vq::renormalise_vector;

use super::mdct::Mdct;
use super::range::{RangeDecoder, BITRES};
use super::tables_celt::*;

pub(super) const NB_EBANDS: usize = 21;
const EFF_EBANDS: usize = 21;
pub(super) const OVERLAP: usize = 120;
const MAX_LM: usize = 3;
const SHORT_MDCT_SIZE: usize = 120;
const NB_ALLOC_VECTORS: usize = 11;
const MDCT_SIZE: usize = 1920;

/// The decoder's history, long enough for the pitch predictor's longest lag.
const DECODE_BUFFER_SIZE: usize = 2048;
const LPC_ORDER: usize = 24;
const MAX_PERIOD: usize = 1024;
const COMBFILTER_MINPERIOD: usize = 15;
const PLC_PITCH_LAG_MAX: usize = 720;
const PLC_PITCH_LAG_MIN: usize = 100;

const LOG_MAX_PSEUDO: i32 = 6;
const MAX_FINE_BITS: i32 = 8;
const FINE_OFFSET: i32 = 21;
const QTHETA_OFFSET: i32 = 4;
const QTHETA_OFFSET_TWOPHASE: i32 = 16;
const ALLOC_STEPS: i32 = 6;

const SPREAD_NONE: usize = 0;
const SPREAD_NORMAL: usize = 2;
const SPREAD_AGGRESSIVE: usize = 3;

/// The pre-emphasis the encoder applied, which synthesis has to undo.
const PREEMPH: f32 = 0.85000610;

/// One unit is one 16-bit sample step.
pub(super) const SIG_SCALE: f32 = 32768.0;

/// The 48 kHz / 960 mode's computed (not tabulated) values.
pub(super) struct Mode {
    mdct: Mdct,
}

impl Mode {
    pub(super) fn new() -> Self {
        Mode {
            mdct: Mdct::new(MDCT_SIZE, MAX_LM),
        }
    }
}

/// One MDCT window tap, also used by mode-transition cross-fades.
pub(super) fn window_at(i: usize) -> f32 {
    WINDOW120[i]
}

/// One CELT decoder: mode tables and inter-frame state.
pub(super) struct CeltDecoder {
    mode: Mode,
    channels: usize,
    /// Channels this frame codes, which may be fewer.
    pub(super) stream_channels: usize,
    /// 48000 / output rate.
    pub(super) downsample: usize,
    pub(super) start: usize,
    pub(super) end: usize,
    disable_inv: bool,
    pub(super) rng: u32,
    last_pitch_index: usize,
    loss_duration: i32,
    skip_plc: bool,
    postfilter_period: usize,
    postfilter_period_old: usize,
    postfilter_gain: f32,
    postfilter_gain_old: f32,
    postfilter_tapset: usize,
    postfilter_tapset_old: usize,
    preemph_mem: [f32; 2],
    decode_mem: Vec<Vec<f32>>,
    lpc: Vec<[f32; LPC_ORDER]>,
    old_bande: Vec<f32>,
    old_loge: Vec<f32>,
    old_loge2: Vec<f32>,
    background_loge: Vec<f32>,
}

impl CeltDecoder {
    pub(super) fn new(channels: usize) -> Self {
        let mut dec = CeltDecoder {
            mode: Mode::new(),
            channels,
            stream_channels: channels,
            downsample: 1,
            start: 0,
            end: EFF_EBANDS,
            disable_inv: channels == 1,
            rng: 0,
            last_pitch_index: 0,
            loss_duration: 0,
            skip_plc: true,
            postfilter_period: 0,
            postfilter_period_old: 0,
            postfilter_gain: 0.0,
            postfilter_gain_old: 0.0,
            postfilter_tapset: 0,
            postfilter_tapset_old: 0,
            preemph_mem: [0.0; 2],
            decode_mem: vec![vec![0.0; DECODE_BUFFER_SIZE + OVERLAP]; channels],
            lpc: vec![[0.0; LPC_ORDER]; channels],
            old_bande: vec![0.0; 2 * NB_EBANDS],
            old_loge: vec![-28.0; 2 * NB_EBANDS],
            old_loge2: vec![-28.0; 2 * NB_EBANDS],
            background_loge: vec![0.0; 2 * NB_EBANDS],
        };
        dec.reset();
        dec
    }

    pub(super) fn reset(&mut self) {
        self.rng = 0;
        self.last_pitch_index = 0;
        self.loss_duration = 0;
        self.skip_plc = true;
        self.postfilter_period = 0;
        self.postfilter_period_old = 0;
        self.postfilter_gain = 0.0;
        self.postfilter_gain_old = 0.0;
        self.postfilter_tapset = 0;
        self.postfilter_tapset_old = 0;
        self.preemph_mem = [0.0; 2];
        for mem in self.decode_mem.iter_mut() {
            mem.fill(0.0);
        }
        for lpc in self.lpc.iter_mut() {
            lpc.fill(0.0);
        }
        self.old_bande.fill(0.0);
        self.old_loge.fill(-28.0);
        self.old_loge2.fill(-28.0);
        self.background_loge.fill(0.0);
    }

    pub(super) fn set_channels(&mut self, stream_channels: usize) {
        self.stream_channels = stream_channels;
    }

    /// Conceal one lost frame: repeat the last pitch period through LPC synthesis,
    /// or for long losses fill bands with noise at the decaying energy.
    fn decode_lost(&mut self, n: usize, lm: usize) {
        let cc = self.channels;
        let at = DECODE_BUFFER_SIZE - n;
        let loss_duration = self.loss_duration;
        let start = self.start;
        let noise_based = loss_duration >= 40 || start != 0 || self.skip_plc;

        if noise_based {
            let end = self.end;
            let eff_end = start.max(end.min(EFF_EBANDS));
            let mut x = vec![0.0f32; cc * n];
            for mem in self.decode_mem.iter_mut() {
                mem.copy_within(n..DECODE_BUFFER_SIZE + (OVERLAP >> 1), 0);
            }
            let decay = if loss_duration == 0 { 1.5 } else { 0.5 };
            for c in 0..cc {
                for i in start..end {
                    let idx = c * NB_EBANDS + i;
                    self.old_bande[idx] =
                        self.background_loge[idx].max(self.old_bande[idx] - decay);
                }
            }
            let mut seed = self.rng;
            for c in 0..cc {
                for i in start..eff_end {
                    let boffs = n * c + ((EBAND_5MS[i] as usize) << lm);
                    let blen = ((EBAND_5MS[i + 1] - EBAND_5MS[i]) as usize) << lm;
                    for j in 0..blen {
                        seed = lcg_rand(seed);
                        x[boffs + j] = ((seed as i32) >> 20) as f32;
                    }
                    renormalise_vector(&mut x[boffs..boffs + blen], 1.0);
                }
            }
            self.rng = seed;
            celt_synthesis(
                &mut self.mode.mdct,
                &x,
                &mut self.decode_mem,
                at,
                &self.old_bande,
                start,
                eff_end,
                cc,
                cc,
                false,
                lm,
                self.downsample,
                false,
            );
        } else {
            let pitch_index = if loss_duration == 0 {
                let mut lp = vec![0.0f32; DECODE_BUFFER_SIZE >> 1];
                pitch_downsample(&self.decode_mem, &mut lp, DECODE_BUFFER_SIZE, cc);
                let found = pitch_search(
                    &lp[PLC_PITCH_LAG_MAX >> 1..],
                    &lp,
                    DECODE_BUFFER_SIZE - PLC_PITCH_LAG_MAX,
                    PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN,
                );
                self.last_pitch_index = PLC_PITCH_LAG_MAX - found;
                self.last_pitch_index
            } else {
                self.last_pitch_index
            };
            let fade = if loss_duration == 0 { 1.0f32 } else { 0.8 };
            // Two pitch periods, to detect decay.
            let exc_length = (2 * pitch_index).min(MAX_PERIOD);

            for c in 0..cc {
                let mut exc = vec![0.0f32; MAX_PERIOD + LPC_ORDER];
                for i in 0..MAX_PERIOD + LPC_ORDER {
                    exc[i] = self.decode_mem[c][DECODE_BUFFER_SIZE - MAX_PERIOD - LPC_ORDER + i];
                }
                if loss_duration == 0 {
                    let mut ac = [0.0f32; LPC_ORDER + 1];
                    celt_autocorr(
                        &exc[LPC_ORDER..],
                        &mut ac,
                        Some(&WINDOW120),
                        OVERLAP,
                        LPC_ORDER,
                        MAX_PERIOD,
                    );
                    ac[0] *= 1.0001;
                    for i in 1..=LPC_ORDER {
                        ac[i] -= ac[i] * (0.008 * 0.008) * (i * i) as f32;
                    }
                    celt_lpc(&mut self.lpc[c], &ac, LPC_ORDER);
                }
                let mut fir_tmp = vec![0.0f32; exc_length];
                celt_fir(
                    &exc[LPC_ORDER + MAX_PERIOD - exc_length - LPC_ORDER..],
                    &self.lpc[c],
                    &mut fir_tmp,
                    exc_length,
                    LPC_ORDER,
                );
                exc[LPC_ORDER + MAX_PERIOD - exc_length..LPC_ORDER + MAX_PERIOD]
                    .copy_from_slice(&fir_tmp);

                let decay_length = exc_length >> 1;
                let mut e1 = 1.0f32;
                let mut e2 = 1.0f32;
                for i in 0..decay_length {
                    let a = exc[LPC_ORDER + MAX_PERIOD - decay_length + i];
                    let b = exc[LPC_ORDER + MAX_PERIOD - 2 * decay_length + i];
                    e1 += a * a;
                    e2 += b * b;
                }
                let decay = (e1.min(e2) / e2).sqrt();

                self.decode_mem[c].copy_within(n..DECODE_BUFFER_SIZE, 0);

                let extrapolation_offset = MAX_PERIOD - pitch_index;
                let extrapolation_len = n + OVERLAP;
                let mut attenuation = fade * decay;
                let mut s1 = 0.0f32;
                let mut j = 0usize;
                for i in 0..extrapolation_len {
                    if j >= pitch_index {
                        j -= pitch_index;
                        attenuation *= decay;
                    }
                    self.decode_mem[c][at + i] =
                        attenuation * exc[LPC_ORDER + extrapolation_offset + j];
                    let tmp = self.decode_mem[c]
                        [DECODE_BUFFER_SIZE - MAX_PERIOD - n + extrapolation_offset + j];
                    s1 += tmp * tmp;
                    j += 1;
                }
                {
                    let mut lpc_mem = [0.0f32; LPC_ORDER];
                    for i in 0..LPC_ORDER {
                        lpc_mem[i] = self.decode_mem[c][DECODE_BUFFER_SIZE - n - 1 - i];
                    }
                    let input: Vec<f32> = self.decode_mem[c][at..at + extrapolation_len].to_vec();
                    let mut out = vec![0.0f32; extrapolation_len];
                    celt_iir(
                        &input,
                        &self.lpc[c],
                        &mut out,
                        extrapolation_len,
                        LPC_ORDER,
                        &mut lpc_mem,
                    );
                    self.decode_mem[c][at..at + extrapolation_len].copy_from_slice(&out);
                }

                // The synthesis filter can ring; the negated comparison also silences NaN.
                let mut s2 = 0.0f32;
                for i in 0..extrapolation_len {
                    let tmp = self.decode_mem[c][at + i];
                    s2 += tmp * tmp;
                }
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(s1 > 0.2 * s2) {
                    for i in 0..extrapolation_len {
                        self.decode_mem[c][at + i] = 0.0;
                    }
                } else if s1 < s2 {
                    let ratio = ((s1 + 1.0) / (s2 + 1.0)).sqrt();
                    for i in 0..OVERLAP {
                        let g = 1.0 - WINDOW120[i] * (1.0 - ratio);
                        self.decode_mem[c][at + i] *= g;
                    }
                    for i in OVERLAP..extrapolation_len {
                        self.decode_mem[c][at + i] *= ratio;
                    }
                }

                // Pre-filter the overlap the next frame will post-filter.
                let mut etmp = vec![0.0f32; OVERLAP];
                comb_filter_const(
                    &mut etmp,
                    &self.decode_mem[c],
                    DECODE_BUFFER_SIZE,
                    self.postfilter_period,
                    OVERLAP,
                    -self.postfilter_gain,
                    self.postfilter_tapset,
                );
                for i in 0..OVERLAP / 2 {
                    self.decode_mem[c][DECODE_BUFFER_SIZE + i] =
                        WINDOW120[i] * etmp[OVERLAP - 1 - i] + WINDOW120[OVERLAP - i - 1] * etmp[i];
                }
            }
        }
        self.loss_duration = 10000.min(loss_duration + (1 << lm));
    }

    pub(super) fn decode(
        &mut self,
        dec: Option<&mut RangeDecoder>,
        len: usize,
        pcm: &mut [f32],
        frame_size: usize,
    ) -> usize {
        let cc = self.channels;
        let frame_size = frame_size * self.downsample;
        let mut lm = 0usize;
        while lm <= MAX_LM {
            if SHORT_MDCT_SIZE << lm == frame_size {
                break;
            }
            lm += 1;
        }
        let n = SHORT_MDCT_SIZE << lm;
        let at = DECODE_BUFFER_SIZE - n;
        let start = self.start;
        let end = self.end;
        let eff_end = end.min(EFF_EBANDS);

        let dec = match dec {
            Some(dec) if len > 1 => dec,
            _ => {
                self.decode_lost(n, lm);
                deemphasis(
                    &self.decode_mem,
                    at,
                    pcm,
                    n,
                    cc,
                    self.downsample,
                    &mut self.preemph_mem,
                );
                return frame_size / self.downsample;
            }
        };

        // Pitch concealment only after two consecutive packets.
        self.skip_plc = self.loss_duration != 0;

        let c = self.stream_channels;
        if c == 1 {
            for i in 0..NB_EBANDS {
                self.old_bande[i] = self.old_bande[i].max(self.old_bande[NB_EBANDS + i]);
            }
        }

        let total_bits = (len * 8) as i32;
        let mut tell = dec.tell();
        let silence = if tell >= total_bits {
            true
        } else if tell == 1 {
            dec.decode_bit_logp(15)
        } else {
            false
        };
        if silence {
            dec.skip_to_end(len);
            tell = total_bits;
        }

        let mut postfilter_gain = 0.0f32;
        let mut postfilter_pitch = 0usize;
        let mut postfilter_tapset = 0usize;
        if start == 0 && tell + 16 <= total_bits {
            if dec.decode_bit_logp(1) {
                let octave = dec.decode_uint(6);
                postfilter_pitch = ((16 << octave) + dec.decode_bits(4 + octave) - 1) as usize;
                let qg = dec.decode_bits(3);
                if dec.tell() + 2 <= total_bits {
                    postfilter_tapset = dec.decode_icdf(&TAPSET_ICDF, 2);
                }
                postfilter_gain = 0.09375 * (qg + 1) as f32;
            }
            tell = dec.tell();
        }

        let is_transient = if lm > 0 && tell + 3 <= total_bits {
            let t = dec.decode_bit_logp(3);
            tell = dec.tell();
            t
        } else {
            false
        };

        let intra_ener = tell + 3 <= total_bits && dec.decode_bit_logp(3);
        unquant_coarse_energy(start, end, &mut self.old_bande, intra_ener, dec, c, lm, len);

        let mut tf_res = [0i32; NB_EBANDS];
        tf_decode(start, end, is_transient, &mut tf_res, lm, dec, len);

        let mut spread = SPREAD_NORMAL;
        if dec.tell() + 4 <= total_bits {
            spread = dec.decode_icdf(&SPREAD_ICDF, 5);
        }

        let mut cap = [0i32; NB_EBANDS];
        init_caps(&mut cap, lm, c);

        // Dynamic allocation boosts, coded as increasingly cheap flags.
        let mut offsets = [0i32; NB_EBANDS];
        let mut dynalloc_logp = 6i32;
        let mut total_bits_frac = total_bits << BITRES;
        let mut tell_frac = dec.tell_frac() as i32;
        for i in start..end {
            let width = (c * (EBAND_5MS[i + 1] - EBAND_5MS[i]) as usize) << lm;
            // Six bits at a time, capped at one bit per sample and floored at an eighth.
            let quanta = ((width << BITRES) as i32).min((6 << BITRES).max(width as i32));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            while tell_frac + (dynalloc_loop_logp << BITRES) < total_bits_frac && boost < cap[i] {
                let flag = dec.decode_bit_logp(dynalloc_loop_logp as u32);
                tell_frac = dec.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_bits_frac -= quanta;
                dynalloc_loop_logp = 1;
            }
            offsets[i] = boost;
            if boost > 0 {
                dynalloc_logp = 2.max(dynalloc_logp - 1);
            }
        }

        let alloc_trim = if tell_frac + (6 << BITRES) <= total_bits_frac {
            dec.decode_icdf(&TRIM_ICDF, 7) as i32
        } else {
            5
        };

        let mut bits = ((len as i32 * 8) << BITRES) - dec.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm as i32 + 2) << BITRES) {
            1 << BITRES
        } else {
            0
        };
        bits -= anti_collapse_rsv;

        let mut pulses = [0i32; NB_EBANDS];
        let mut fine_quant = [0i32; NB_EBANDS];
        let mut fine_priority = [0i32; NB_EBANDS];
        let mut intensity = 0usize;
        let mut dual_stereo = false;
        let mut balance = 0i32;
        let coded_bands = compute_allocation(
            start,
            end,
            &offsets,
            &cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            &mut pulses,
            &mut fine_quant,
            &mut fine_priority,
            c,
            lm,
            dec,
        );

        unquant_fine_energy(start, end, &mut self.old_bande, &fine_quant, dec, c);

        for mem in self.decode_mem.iter_mut() {
            mem.copy_within(n..DECODE_BUFFER_SIZE + OVERLAP / 2, 0);
        }

        let mut collapse_masks = vec![0u8; c * NB_EBANDS];
        let mut x = vec![0.0f32; c * n];
        let mut seed = self.rng;
        quant_all_bands(
            start,
            end,
            &mut x,
            c,
            &mut collapse_masks,
            &pulses,
            is_transient,
            spread,
            dual_stereo,
            intensity,
            &tf_res,
            (len as i32 * (8 << BITRES)) - anti_collapse_rsv,
            balance,
            dec,
            lm,
            coded_bands,
            &mut seed,
            self.disable_inv,
        );
        self.rng = seed;

        let anti_collapse_on = anti_collapse_rsv > 0 && dec.decode_bits(1) != 0;

        unquant_energy_finalise(
            start,
            end,
            &mut self.old_bande,
            &fine_quant,
            &fine_priority,
            len as i32 * 8 - dec.tell(),
            dec,
            c,
        );

        if anti_collapse_on {
            anti_collapse(
                &mut x,
                &collapse_masks,
                lm,
                c,
                n,
                start,
                end,
                &self.old_bande,
                &self.old_loge,
                &self.old_loge2,
                &pulses,
                self.rng,
            );
        }
        if silence {
            for v in self.old_bande.iter_mut() {
                *v = -28.0;
            }
        }

        celt_synthesis(
            &mut self.mode.mdct,
            &x,
            &mut self.decode_mem,
            at,
            &self.old_bande,
            start,
            eff_end,
            c,
            cc,
            is_transient,
            lm,
            self.downsample,
            silence,
        );

        for ch in 0..cc {
            self.postfilter_period = self.postfilter_period.max(COMBFILTER_MINPERIOD);
            self.postfilter_period_old = self.postfilter_period_old.max(COMBFILTER_MINPERIOD);
            comb_filter(
                &mut self.decode_mem[ch],
                at,
                self.postfilter_period_old,
                self.postfilter_period,
                SHORT_MDCT_SIZE,
                self.postfilter_gain_old,
                self.postfilter_gain,
                self.postfilter_tapset_old,
                self.postfilter_tapset,
                Some(&WINDOW120),
                OVERLAP,
            );
            if lm != 0 {
                comb_filter(
                    &mut self.decode_mem[ch],
                    at + SHORT_MDCT_SIZE,
                    self.postfilter_period,
                    postfilter_pitch,
                    n - SHORT_MDCT_SIZE,
                    self.postfilter_gain,
                    postfilter_gain,
                    self.postfilter_tapset,
                    postfilter_tapset,
                    Some(&WINDOW120),
                    OVERLAP,
                );
            }
        }
        self.postfilter_period_old = self.postfilter_period;
        self.postfilter_gain_old = self.postfilter_gain;
        self.postfilter_tapset_old = self.postfilter_tapset;
        self.postfilter_period = postfilter_pitch;
        self.postfilter_gain = postfilter_gain;
        self.postfilter_tapset = postfilter_tapset;
        if lm != 0 {
            self.postfilter_period_old = self.postfilter_period;
            self.postfilter_gain_old = self.postfilter_gain;
            self.postfilter_tapset_old = self.postfilter_tapset;
        }

        if c == 1 {
            let (lo, hi) = self.old_bande.split_at_mut(NB_EBANDS);
            hi[..NB_EBANDS].copy_from_slice(lo);
        }
        if !is_transient {
            self.old_loge2.copy_from_slice(&self.old_loge);
            self.old_loge.copy_from_slice(&self.old_bande);
        } else {
            for i in 0..2 * NB_EBANDS {
                self.old_loge[i] = self.old_loge[i].min(self.old_bande[i]);
            }
        }
        // The noise floor rises 2.4 dB/s, or catches up at once after losses.
        let max_background_increase = 160.min(self.loss_duration + (1 << lm)) as f32 * 0.001;
        for i in 0..2 * NB_EBANDS {
            self.background_loge[i] =
                (self.background_loge[i] + max_background_increase).min(self.old_bande[i]);
        }
        for ch in 0..2 {
            for i in 0..start {
                self.old_bande[ch * NB_EBANDS + i] = 0.0;
                self.old_loge[ch * NB_EBANDS + i] = -28.0;
                self.old_loge2[ch * NB_EBANDS + i] = -28.0;
            }
            for i in end..NB_EBANDS {
                self.old_bande[ch * NB_EBANDS + i] = 0.0;
                self.old_loge[ch * NB_EBANDS + i] = -28.0;
                self.old_loge2[ch * NB_EBANDS + i] = -28.0;
            }
        }
        self.rng = dec.rng();

        deemphasis(
            &self.decode_mem,
            at,
            pcm,
            n,
            cc,
            self.downsample,
            &mut self.preemph_mem,
        );
        self.loss_duration = 0;
        frame_size / self.downsample
    }
}
