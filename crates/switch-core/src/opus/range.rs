//! The range decoder every Opus frame is read through (RFC 6716 §4.1).
//!
//! One frame carries two interleaved streams in the same bytes: symbols coded
//! against a probability model, read forwards from the start, and *raw* bits
//! of uniform probability, read backwards from the end. Both advance the same
//! bit counter, which is what lets the two ends meet exactly in the middle —
//! and what makes [`RangeDecoder::tell_frac`] a usable budget rather than an
//! estimate. CELT spends the whole frame deciding how many bits a band may
//! have from that counter, so it has to agree with the encoder's to the
//! eighth of a bit.
//!
//! Reads past the end of the buffer return zero rather than failing. That is
//! the specified behaviour, not leniency: a decoder must stay in step with an
//! encoder that stopped writing once the remaining bits were implied.

/// Bits emitted at a time — the coder's base is a byte.
const SYM_BITS: u32 = 8;

/// Width of `rng` and `val`.
const CODE_BITS: u32 = 32;

const SYM_MAX: u32 = (1 << SYM_BITS) - 1;

const CODE_TOP: u32 = 1 << (CODE_BITS - 1);

const CODE_BOT: u32 = CODE_TOP >> SYM_BITS;

/// Bits left for the last, partial symbol of the code field.
const CODE_EXTRA: u32 = (CODE_BITS - 2) % SYM_BITS + 1;

/// Bits of an unsigned integer that are range-coded; the rest are raw.
const UINT_BITS: u32 = 8;

/// Fractional resolution of [`RangeDecoder::tell_frac`]: eighths of a bit.
pub(super) const BITRES: u32 = 3;

pub(super) struct RangeDecoder<'a> {
    buf: &'a [u8],
    /// How far in from the *end* the raw-bit reader has consumed.
    end_offs: u32,
    /// Raw bits read from the end and not yet handed out, and how many.
    end_window: u32,
    nend_bits: i32,
    /// Whole bits consumed by both streams. Partial bits still inside the
    /// range are subtracted by `tell`.
    nbits_total: i32,
    /// How far in from the start the symbol reader has consumed.
    offs: u32,
    rng: u32,
    val: u32,
    /// The normalization factor `decode` saved for the `update` that follows.
    ext: u32,
    /// The byte straddling the boundary between two normalization steps.
    rem: u32,
    /// Set when a decoded value was outside the range its own coding allows.
    /// The frame is corrupt from here on; the caller decides whether to keep
    /// what it has or conceal.
    pub error: bool,
}

impl<'a> RangeDecoder<'a> {
    pub(super) fn new(buf: &'a [u8]) -> Self {
        let mut dec = RangeDecoder {
            buf,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            // The offset `tell` subtracts partial bits from. The encoder adds
            // bits this side has not read yet, so the count starts ahead by
            // exactly what the normalization below is about to consume.
            nbits_total: (CODE_BITS + 1 - ((CODE_BITS - CODE_EXTRA) / SYM_BITS) * SYM_BITS) as i32,
            offs: 0,
            rng: 1 << CODE_EXTRA,
            val: 0,
            ext: 0,
            rem: 0,
            error: false,
        };
        dec.rem = dec.read_byte();
        dec.val = dec.rng - 1 - (dec.rem >> (SYM_BITS - CODE_EXTRA));
        dec.normalize();
        dec
    }

    fn read_byte(&mut self) -> u32 {
        let byte = self.buf.get(self.offs as usize).copied().unwrap_or(0);
        if (self.offs as usize) < self.buf.len() {
            self.offs += 1;
        }
        u32::from(byte)
    }

    fn read_byte_from_end(&mut self) -> u32 {
        if (self.end_offs as usize) < self.buf.len() {
            self.end_offs += 1;
            u32::from(self.buf[self.buf.len() - self.end_offs as usize])
        } else {
            0
        }
    }

    /// Rescale so the range again occupies the high-order symbol, pulling in
    /// input bytes to make up the difference.
    fn normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.nbits_total += SYM_BITS as i32;
            self.rng <<= SYM_BITS;
            let carried = self.rem;
            self.rem = self.read_byte();
            let sym = (carried << SYM_BITS | self.rem) >> (SYM_BITS - CODE_EXTRA);
            self.val = ((self.val << SYM_BITS) + (SYM_MAX & !sym)) & (CODE_TOP - 1);
        }
    }

    /// The symbol's position in a cumulative frequency table totalling `ft`.
    /// The caller looks it up in its own table and reports the bounds back
    /// through [`RangeDecoder::update`], which is what actually consumes it.
    pub(super) fn decode(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// [`RangeDecoder::decode`] against a power-of-two total, which is every
    /// table CELT codes against.
    pub(super) fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        (1 << bits) - (s + 1).min(1 << bits)
    }

    /// Consume the symbol whose cumulative frequency runs `[fl, fh)` out of
    /// `ft`.
    pub(super) fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext.wrapping_mul(ft - fh);
        self.val = self.val.wrapping_sub(s);
        self.rng = if fl > 0 { self.ext.wrapping_mul(fh - fl) } else { self.rng - s };
        self.normalize();
    }

    /// A bit whose probability of being one is `1/(1 << logp)`.
    pub(super) fn decode_bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let bit = d < s;
        if !bit {
            self.val = d - s;
        }
        self.rng = if bit { s } else { r - s };
        self.normalize();
        bit
    }

    /// A symbol from an *inverse* cumulative table: `icdf[i]` is the total
    /// minus the cumulative frequency through symbol `i`, scaled to
    /// `1 << ftb`, and the last entry is 0. Storing the complement is what
    /// lets the table be bytes.
    pub(super) fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut t;
        let mut ret = 0usize;
        loop {
            t = s;
            s = r.wrapping_mul(u32::from(icdf[ret]));
            if d >= s {
                break;
            }
            ret += 1;
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret
    }

    /// A uniformly distributed integer in `0..ft`. Values wider than
    /// [`UINT_BITS`] are split: the high bits are range-coded, the low bits
    /// are raw.
    pub(super) fn decode_uint(&mut self, ft: u32) -> u32 {
        debug_assert!(ft > 1);
        let ft = ft - 1;
        let ftb = ilog(ft);
        if ftb > UINT_BITS {
            let ftb = ftb - UINT_BITS;
            let split = (ft >> ftb) + 1;
            let s = self.decode(split);
            self.update(s, s + 1, split);
            let value = (s << ftb) | self.decode_bits(ftb);
            if value <= ft {
                return value;
            }
            self.error = true;
            ft
        } else {
            let s = self.decode(ft + 1);
            self.update(s, s + 1, ft + 1);
            s
        }
    }

    /// Raw bits, taken from the end of the frame inwards.
    pub(super) fn decode_bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if (available as u32) < bits {
            loop {
                window |= self.read_byte_from_end() << available;
                available += SYM_BITS as i32;
                if available > (CODE_BITS - SYM_BITS) as i32 {
                    break;
                }
            }
        }
        let value = window & ((1u32 << bits) - 1);
        self.end_window = window >> bits;
        self.nend_bits = available - bits as i32;
        self.nbits_total += bits as i32;
        value
    }

    /// Drop `bytes` from the end of the frame. A packet that carries a
    /// redundant CELT frame puts it there, and this frame's raw bits — which
    /// are read backwards from the end — must stop short of it.
    pub(super) fn shrink(&mut self, bytes: usize) {
        self.buf = &self.buf[..self.buf.len().saturating_sub(bytes)];
    }

    /// Declare the rest of the frame consumed. A frame flagged as silence
    /// carries nothing after that flag, and both ends have to agree on where
    /// the bit counter ends up.
    pub(super) fn skip_to_end(&mut self, len: usize) {
        let target = (len * 8) as i32;
        self.nbits_total += target - self.tell();
    }

    /// Whole bits consumed so far, rounded up.
    pub(super) fn tell(&self) -> i32 {
        self.nbits_total - ilog(self.rng) as i32
    }

    /// The same count in eighths of a bit. The linear term plus this table is
    /// exact at this resolution — the transition thresholds are where
    /// `r*r >> 15` would carry.
    pub(super) fn tell_frac(&self) -> u32 {
        const CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
        let l = ilog(self.rng);
        let r = self.rng >> (l - 16);
        let mut b = (r >> 12) - 8;
        b += u32::from(r > CORRECTION[b as usize]);
        ((self.nbits_total as u32) << BITRES) - ((l << 3) + b)
    }

    /// The decoder's range register, which is what a caller compares against
    /// the encoder's `final_range` to prove the two stayed in step.
    pub(super) fn rng(&self) -> u32 {
        self.rng
    }
}

/// `EC_ILOG`: one plus the index of the highest set bit, and 0 for 0.
pub(super) fn ilog(v: u32) -> u32 {
    32 - v.leading_zeros()
}
