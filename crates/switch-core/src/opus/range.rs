//! Opus range decoder (RFC 6716 §4.1): symbols read forwards, raw bits backwards.

const SYM_BITS: u32 = 8;

const CODE_BITS: u32 = 32;

const SYM_MAX: u32 = (1 << SYM_BITS) - 1;

const CODE_TOP: u32 = 1 << (CODE_BITS - 1);

const CODE_BOT: u32 = CODE_TOP >> SYM_BITS;

const CODE_EXTRA: u32 = (CODE_BITS - 2) % SYM_BITS + 1;

const UINT_BITS: u32 = 8;

/// [`RangeDecoder::tell_frac`] resolution: eighths of a bit.
pub(super) const BITRES: u32 = 3;

pub(super) struct RangeDecoder<'a> {
    buf: &'a [u8],
    end_offs: u32,
    end_window: u32,
    nend_bits: i32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: u32,
    /// Set when a decoded value was out of range; the frame is corrupt.
    pub error: bool,
}

impl<'a> RangeDecoder<'a> {
    pub(super) fn new(buf: &'a [u8]) -> Self {
        let mut dec = RangeDecoder {
            buf,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
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

    /// Symbol position in a cumulative table totalling `ft`; consume it with [`RangeDecoder::update`].
    pub(super) fn decode(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    pub(super) fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        (1 << bits) - (s + 1).min(1 << bits)
    }

    pub(super) fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext.wrapping_mul(ft - fh);
        self.val = self.val.wrapping_sub(s);
        self.rng = if fl > 0 {
            self.ext.wrapping_mul(fh - fl)
        } else {
            self.rng - s
        };
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

    /// A symbol from an inverse cumulative table scaled to `1 << ftb`.
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

    /// A uniform integer in `0..ft`; bits above [`UINT_BITS`] are raw.
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

    /// Raw bits, read from the end of the frame inwards.
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

    /// Drop `bytes` (a redundant CELT frame) from the end of the frame.
    pub(super) fn shrink(&mut self, bytes: usize) {
        self.buf = &self.buf[..self.buf.len().saturating_sub(bytes)];
    }

    pub(super) fn skip_to_end(&mut self, len: usize) {
        let target = (len * 8) as i32;
        self.nbits_total += target - self.tell();
    }

    /// Whole bits consumed so far, rounded up.
    pub(super) fn tell(&self) -> i32 {
        self.nbits_total - ilog(self.rng) as i32
    }

    /// Bits consumed, in eighths of a bit.
    pub(super) fn tell_frac(&self) -> u32 {
        const CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
        let l = ilog(self.rng);
        let r = self.rng >> (l - 16);
        let mut b = (r >> 12) - 8;
        b += u32::from(r > CORRECTION[b as usize]);
        ((self.nbits_total as u32) << BITRES) - ((l << 3) + b)
    }

    pub(super) fn rng(&self) -> u32 {
        self.rng
    }
}

/// `EC_ILOG`: one plus the index of the highest set bit, and 0 for 0.
pub(super) fn ilog(v: u32) -> u32 {
    32 - v.leading_zeros()
}
