//! The SILK layer of Opus: LPC and long-term pitch prediction driven by coded
//! pulses, at an internal 8, 12 or 16 kHz resampled to the output rate.
//! Integer arithmetic throughout, matching the reference's fixed-point
//! rounding bit for bit, since each frame predicts from the last.

mod channel;
mod nlsf;
mod plc;
mod pulses;
mod resampler;
mod stereo;
mod synthesis;

use pulses::decode_pulses;
use resampler::Resampler;
use stereo::{stereo_decode_pred, stereo_ms_to_lr, StereoState};

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
/// `0.99` in Q16, the bandwidth expansion concealment applies to the last good filter.
const BWE_COEF_Q16: i32 = 64881;
const PE_MAX_LAG_MS: i32 = 18;
const PE_MIN_LAG_MS: i32 = 2;

const TYPE_NO_VOICE_ACTIVITY: i32 = 0;
const TYPE_VOICED: i32 = 2;

/// How a frame's first gain and LTP scaling are coded, depending on whether
/// the previous frame is available to predict from.
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

// Fixed-point primitives, each the reference macro of the same name.
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

/// SILK's LCG for excitation signs and concealment noise; part of the bitstream.
fn silk_rand(seed: i32) -> i32 {
    907633515i32.wrapping_add(seed.wrapping_mul(196314165))
}

/// Leading zeros, and the seven bits just below the leading one.
fn clz_frac(x: i32) -> (i32, i32) {
    let lzeros = clz32(x);
    // A rotate, as `silk_ROR32` does for more than 24 leading zeros.
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
    // The residual may wrap; it is small after refinement.
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

/// Sum of squares, shifted right to keep two bits of headroom; returns the shift too.
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

/// One of the two NLSF codebooks: order 10 (narrow/medium band) or 16 (wideband).
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

/// One frame's side information as read from the entropy decoder.
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

/// The filters and gains synthesis runs for one frame.
struct FrameControl {
    /// Two LPC filters; `nlsf_interp_coef_q2` selects how the first half
    /// interpolates towards the second.
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

/// Comfort noise: a smoothed background spectrum and gain to synthesise
/// noise from where the encoder sent nothing.
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

/// One coded channel's decoder. In stereo the second carries the side signal
/// and may be absent for effectively mono frames.
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

/// What the caller tells SILK about the stream it is decoding.
pub(super) struct Control {
    pub api_sample_rate: u32,
    pub channels_api: usize,
    pub channels_internal: usize,
    pub internal_sample_rate: u32,
    pub payload_size_ms: usize,
}

/// SILK could not decode the frame: a duration or rate it does not have.
#[derive(Debug)]
pub(super) struct SilkError;

/// One SILK stream's decoder: up to two coded channels and their mid/side state.
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
    /// report samples per channel. `first_frame` reads the per-packet flags.
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
        // A stream turning stereo starts its side channel fresh.
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
                // Per-packet header: VAD flags per frame and an LBRR flag, per channel.
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
                // Redundant copies are not played, but must be stepped over.
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

        // The side channel's prediction memory is stale; restart it.
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
                    // A skipped side frame leaves the LTP state well defined.
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
                // Keep the idle right resampler warm for a return to stereo.
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
            // Drop the gain clamp so the energy does not bounce back on resume.
            for ch in self.channels.iter_mut() {
                ch.last_gain_index = 10;
            }
        } else {
            self.prev_decode_only_middle = decode_only_middle;
        }
        Ok(n_samples_out)
    }
}
