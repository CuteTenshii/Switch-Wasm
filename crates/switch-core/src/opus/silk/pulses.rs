//! The shell-coded excitation pulses.

use crate::opus::range::RangeDecoder;
use crate::opus::tables_silk::*;
use super::{
    LOG2_SHELL_CODEC_FRAME_LENGTH, MAX_NB_SHELL_BLOCKS, N_RATE_LEVELS, SHELL_CODEC_FRAME_LENGTH,
    SILK_MAX_PULSES,
};

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

/// Distribute one block's pulse count over its sixteen positions by halving four times.
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

/// Attach a sign to every non-zero pulse; the probability depends on the
/// block's pulse count.
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

/// Decode the excitation: rate level, pulse counts per 16-sample block, shell
/// code, extra low bits, and signs.
pub(super) fn decode_pulses(
    dec: &mut RangeDecoder,
    pulses: &mut [i16],
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: usize,
) {
    let rate_level = dec.decode_icdf(&RATE_LEVELS_ICDF[(signal_type >> 1) as usize * 9..], 8);

    let mut iter = frame_length >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    if iter * SHELL_CODEC_FRAME_LENGTH < frame_length {
        // 10 ms at 12 kHz: 120 samples is not a whole number of shell blocks.
        iter += 1;
    }

    let mut sum_pulses = [0i32; MAX_NB_SHELL_BLOCKS];
    let mut n_lshifts = [0i32; MAX_NB_SHELL_BLOCKS];
    for i in 0..iter {
        sum_pulses[i] = dec.decode_icdf(&PULSES_PER_BLOCK_ICDF[rate_level * 18..], 8) as i32;
        // A block too loud for the table codes its low bits separately.
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
            // Mark the block non-empty for the sign decoder.
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
