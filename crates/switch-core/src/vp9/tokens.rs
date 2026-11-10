//! Coefficient token decoding for one transform block.

use super::modeinfo::TileDecoder;
use super::tables::*;

const CAT1_MIN: i32 = 5;
const CAT2_MIN: i32 = 7;
const CAT3_MIN: i32 = 11;
const CAT4_MIN: i32 = 19;
const CAT5_MIN: i32 = 35;
const CAT6_MIN: i32 = 67;

/// The scan order and neighbour table of a transform size and type.
pub fn scan_order(tx_size: u8, tx_type: u8) -> (&'static [i16], &'static [i16]) {
    // Transform types: 0 default, 1 row scan, 2 column scan, 3 default.
    match (tx_size, tx_type) {
        (0, 1) => (&ROW_SCAN_4X4, &ROW_NEIGHBORS_4X4),
        (0, 2) => (&COL_SCAN_4X4, &COL_NEIGHBORS_4X4),
        (0, _) => (&DEFAULT_SCAN_4X4, &DEFAULT_NEIGHBORS_4X4),
        (1, 1) => (&ROW_SCAN_8X8, &ROW_NEIGHBORS_8X8),
        (1, 2) => (&COL_SCAN_8X8, &COL_NEIGHBORS_8X8),
        (1, _) => (&DEFAULT_SCAN_8X8, &DEFAULT_NEIGHBORS_8X8),
        (2, 1) => (&ROW_SCAN_16X16, &ROW_NEIGHBORS_16X16),
        (2, 2) => (&COL_SCAN_16X16, &COL_NEIGHBORS_16X16),
        (2, _) => (&DEFAULT_SCAN_16X16, &DEFAULT_NEIGHBORS_16X16),
        _ => (&DEFAULT_SCAN_32X32, &DEFAULT_NEIGHBORS_32X32),
    }
}

impl TileDecoder<'_> {
    fn read_coeff(&mut self, probs: &[u8]) -> i32 {
        probs
            .iter()
            .fold(0, |value, &p| (value << 1) | i32::from(self.r.read(p)))
    }

    /// Decodes one block's tokens into `self.dqcoeff` (dequantized, raster order) and returns the
    /// end-of-block position.
    pub fn decode_coefs(
        &mut self,
        plane_type: usize,
        tx_size: u8,
        inter: bool,
        dq: [i32; 2],
        mut ctx: usize,
        scan: &[i16],
        nb: &[i16],
    ) -> usize {
        let tx = usize::from(tx_size);
        let ref_type = usize::from(inter);
        let max_eob = 16usize << (tx << 1);
        let dq_shift = u32::from(tx_size == 3);
        let band_table: &[u8] = if tx_size == 0 {
            &COEFBAND_4X4
        } else {
            &COEFBAND_8X8PLUS
        };
        let probs = self.st.fc.coef[tx][plane_type][ref_type];
        let mut token_cache = [0u8; 1024];
        let mut dqv = dq[0];
        let mut c = 0usize;
        let counting = self.counting();

        macro_rules! count_token {
            ($band:expr, $ctx:expr, $token:expr) => {
                if counting {
                    self.st.counts.as_mut().unwrap().coef[tx][plane_type][ref_type][$band][$ctx]
                        [$token] += 1;
                }
            };
        }
        let context = |token_cache: &[u8; 1024], c: usize| -> usize {
            (1 + usize::from(token_cache[nb[2 * c] as usize])
                + usize::from(token_cache[nb[2 * c + 1] as usize]))
                >> 1
        };

        while c < max_eob {
            let mut band = usize::from(band_table[c]);
            let mut prob = probs[band][ctx];
            if counting {
                self.st.counts.as_mut().unwrap().eob_branch[tx][plane_type][ref_type][band][ctx] +=
                    1;
            }
            if !self.r.read(prob[0]) {
                count_token!(band, ctx, 3);
                break;
            }
            while !self.r.read(prob[1]) {
                count_token!(band, ctx, 0);
                dqv = dq[1];
                token_cache[scan[c] as usize] = 0;
                c += 1;
                if c >= max_eob {
                    return c;
                }
                ctx = context(&token_cache, c);
                band = usize::from(band_table[c]);
                prob = probs[band][ctx];
            }
            let value: i32;
            if self.r.read(prob[2]) {
                let p = &PARETO8_FULL[usize::from(prob[2]) - 1];
                count_token!(band, ctx, 2);
                if self.r.read(p[0]) {
                    let val;
                    if self.r.read(p[3]) {
                        token_cache[scan[c] as usize] = 5;
                        if self.r.read(p[5]) {
                            if self.r.read(p[7]) {
                                val = CAT6_MIN + self.read_coeff(&CAT6_PROB[..14]);
                            } else {
                                val = CAT5_MIN + self.read_coeff(&CAT5_PROB);
                            }
                        } else if self.r.read(p[6]) {
                            val = CAT4_MIN + self.read_coeff(&CAT4_PROB);
                        } else {
                            val = CAT3_MIN + self.read_coeff(&CAT3_PROB);
                        }
                    } else {
                        token_cache[scan[c] as usize] = 4;
                        if self.r.read(p[4]) {
                            val = CAT2_MIN + self.read_coeff(&CAT2_PROB);
                        } else {
                            val = CAT1_MIN + self.read_coeff(&CAT1_PROB);
                        }
                    }
                    value = (val * dqv) >> dq_shift;
                } else if self.r.read(p[1]) {
                    token_cache[scan[c] as usize] = 3;
                    let extra = i32::from(self.r.read(p[2]));
                    value = ((3 + extra) * dqv) >> dq_shift;
                } else {
                    token_cache[scan[c] as usize] = 2;
                    value = (2 * dqv) >> dq_shift;
                }
            } else {
                count_token!(band, ctx, 1);
                token_cache[scan[c] as usize] = 1;
                value = dqv >> dq_shift;
            }
            self.dqcoeff[scan[c] as usize] = if self.r.read_bit() { -value } else { value };
            c += 1;
            ctx = if c < max_eob {
                context(&token_cache, c)
            } else {
                0
            };
            dqv = dq[1];
        }
        c
    }
}
