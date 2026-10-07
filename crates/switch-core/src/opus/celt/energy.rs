//! Coarse, fine and leftover band energy.

use crate::opus::range::RangeDecoder;
use crate::opus::tables_celt::*;
use super::{MAX_FINE_BITS, NB_EBANDS};

/// A Laplace-distributed coarse energy delta.
fn laplace_decode(dec: &mut RangeDecoder, fs0: u32, decay: i32) -> i32 {
    /// The floor probability of any delta, out of 32768.
    const MINP: u32 = 1;
    /// How many deltas either side are guaranteed representable.
    const NMIN: u32 = 16;

    let fm = dec.decode_bin(15);
    let mut fl = 0u32;
    let mut fs = fs0;
    let mut val = 0i32;
    if fm >= fs {
        val += 1;
        fl = fs;
        let ft = 32768 - MINP * (2 * NMIN) - fs0;
        fs = ((ft * (16384 - decay) as u32) >> 15) + MINP;
        while fs > MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = (((fs - 2 * MINP) * decay as u32) >> 15) + MINP;
            val += 1;
        }
        if fs <= MINP {
            let di = (fm - fl) >> 1;
            val += di as i32;
            fl += 2 * di * MINP;
        }
        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }
    dec.update(fl, (fl + fs).min(32768), 32768);
    val
}

/// Coarse energy, predicted from the band below and the previous frame
/// (`intra` breaks the inter-frame chain).
pub(super) fn unquant_coarse_energy(
    start: usize,
    end: usize,
    old_bande: &mut [f32],
    intra: bool,
    dec: &mut RangeDecoder,
    channels: usize,
    lm: usize,
    total_bytes: usize,
) {
    const PRED_COEF: [f32; 4] = [29440.0 / 32768.0, 26112.0 / 32768.0, 21248.0 / 32768.0, 0.5];
    const BETA_COEF: [f32; 4] = [
        30147.0 / 32768.0,
        22282.0 / 32768.0,
        12124.0 / 32768.0,
        6554.0 / 32768.0,
    ];
    const BETA_INTRA: f32 = 4915.0 / 32768.0;

    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm], BETA_COEF[lm])
    };
    let budget = (total_bytes * 8) as i32;
    let model = &E_PROB_MODEL[(lm * 2 + usize::from(intra)) * 42..][..42];
    let mut prev = [0.0f32; 2];

    for i in start..end {
        for c in 0..channels {
            let tell = dec.tell();
            let qi = if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                laplace_decode(
                    dec,
                    u32::from(model[pi]) << 7,
                    i32::from(model[pi + 1]) << 6,
                )
            } else if budget - tell >= 2 {
                let qi = dec.decode_icdf(&SMALL_ENERGY_ICDF, 2) as i32;
                (qi >> 1) ^ -(qi & 1)
            } else if budget - tell >= 1 {
                -i32::from(dec.decode_bit_logp(1))
            } else {
                -1
            };
            let q = qi as f32;
            let slot = &mut old_bande[c * NB_EBANDS + i];
            *slot = slot.max(-9.0);
            *slot = coef * *slot + prev[c] + q;
            prev[c] = prev[c] + q - beta * q;
        }
    }
}

/// Fine energy: a uniform fraction of the coarse step.
pub(super) fn unquant_fine_energy(
    start: usize,
    end: usize,
    old_bande: &mut [f32],
    fine_quant: &[i32],
    dec: &mut RangeDecoder,
    channels: usize,
) {
    for i in start..end {
        if fine_quant[i] <= 0 {
            continue;
        }
        for c in 0..channels {
            let q2 = dec.decode_bits(fine_quant[i] as u32);
            let offset = (q2 as f32 + 0.5) * ((1 << (14 - fine_quant[i])) as f32) / 16384.0 - 0.5;
            old_bande[c * NB_EBANDS + i] += offset;
        }
    }
}

/// Leftover bits, one at a time, to bands the allocator rounded down.
pub(super) fn unquant_energy_finalise(
    start: usize,
    end: usize,
    old_bande: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    dec: &mut RangeDecoder,
    channels: usize,
) {
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= channels as i32 {
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                i += 1;
                continue;
            }
            for c in 0..channels {
                let q2 = dec.decode_bits(1);
                let offset = (q2 as f32 - 0.5) * ((1 << (14 - fine_quant[i] - 1)) as f32) / 16384.0;
                old_bande[c * NB_EBANDS + i] += offset;
                bits_left -= 1;
            }
            i += 1;
        }
    }
}
