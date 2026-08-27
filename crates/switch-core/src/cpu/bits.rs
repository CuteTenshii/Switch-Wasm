//! Bit-level helpers shared by the instruction groups: the ARM sign/zero
//! extension, shift and rotate primitives, the bitmask-immediate and
//! bitfield decoders, saturating arithmetic and the float rounding modes.

/// Mask of a vector element `bits` wide (`bits` <= 64).
#[inline]
pub(crate) fn elem_mask(bits: u32) -> u128 {
    (1u128 << bits) - 1
}

/// The FPCR bits this core observes: RMode (23:22), plus the FZ/DN/AH
/// controls it stores so a guest reads back what it wrote.
pub(crate) const FPCR_MASK: u32 = 0x07FF_9F00;
/// FPSR: the cumulative exception flags (IDC, IXC, UFC, OFC, DZC, IOC) and
/// QC, the sticky saturation flag.
pub(crate) const FPSR_MASK: u32 = 0x0800_009F;

/// FPSR cumulative exception flags.
/// Only the two the core actually raises are named. Overflow, underflow and
/// QC are storage the guest can write and read back, not signals we set.
pub(crate) const FPSR_IOC: u32 = 1 << 0;
pub(crate) const FPSR_DZC: u32 = 1 << 1;
pub(crate) const FPSR_IXC: u32 = 1 << 4;

/// The rounding mode FPCR.RMode selects, for the instructions the
/// architecture defines as rounding "to the current mode".
pub(crate) fn fpcr_rounding(fpcr: u32) -> Rounding {
    match (fpcr >> 22) & 0b11 {
        0b00 => Rounding::TiesEven,
        0b01 => Rounding::TowardPos,
        0b10 => Rounding::TowardNeg,
        _ => Rounding::TowardZero,
    }
}

/// Round a float to an integral float value in the given mode — FRINTX and
/// FRINTI, which take their mode from FPCR rather than from the opcode.
pub(crate) fn round_to_integral(v: f64, r: Rounding) -> f64 {
    if !v.is_finite() {
        return v;
    }
    match r {
        Rounding::TiesEven => v.round_ties_even(),
        Rounding::TowardPos => v.ceil(),
        Rounding::TowardNeg => v.floor(),
        Rounding::TowardZero => v.trunc(),
        Rounding::TiesAway => v.round(),
    }
}

/// Rounding mode for the float-to-integer conversion instructions.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rounding {
    /// Round to nearest, ties to even.
    TiesEven,
    /// Round toward +infinity.
    TowardPos,
    /// Round toward -infinity.
    TowardNeg,
    /// Round to nearest, ties away from zero.
    TiesAway,
    /// Round toward zero (truncate).
    TowardZero,
}

/// Convert a float to a (possibly signed) integer using an explicit rounding
/// mode, then truncate to the destination size. NaN → 0, out-of-range results
/// saturate, matching the default FPCR behavior the emulator assumes.
pub(crate) fn round_to_int(f: f64, r: Rounding, signed: bool) -> u64 {
    if f.is_nan() {
        return 0;
    }
    let rounded = match r {
        Rounding::TiesEven => f.round_ties_even(),
        Rounding::TowardPos => f.ceil(),
        Rounding::TowardNeg => f.floor(),
        Rounding::TiesAway => f.round(),
        Rounding::TowardZero => f.trunc(),
    };
    let clipped = rounded.clamp(
        i64::MIN as f64,
        if signed { i64::MAX as f64 } else { u64::MAX as f64 },
    );
    if signed {
        (clipped as i64) as u64
    } else {
        clipped.max(0.0) as u64
    }
}

/// [`round_to_int`] into a `bits`-wide destination: out-of-range values
/// saturate at that width rather than wrapping, which is what the vector
/// converts (`fcvtzs v0.4s, v1.4s`) need for their 32-bit lanes.
pub(crate) fn round_to_int_sized(f: f64, r: Rounding, signed: bool, bits: u32) -> u64 {
    if bits >= 64 {
        return round_to_int(f, r, signed);
    }
    if f.is_nan() {
        return 0;
    }
    let rounded = match r {
        Rounding::TiesEven => f.round_ties_even(),
        Rounding::TowardPos => f.ceil(),
        Rounding::TowardNeg => f.floor(),
        Rounding::TiesAway => f.round(),
        Rounding::TowardZero => f.trunc(),
    };
    let mask = (1u64 << bits) - 1;
    if signed {
        let max = (1i64 << (bits - 1)) - 1;
        let min = -(1i64 << (bits - 1));
        (rounded.clamp(min as f64, max as f64) as i64 as u64) & mask
    } else {
        rounded.clamp(0.0, mask as f64) as u64
    }
}

/// Saturating add of two `bits`-wide lanes (`signed` selects SQADD/UQADD).
pub(crate) fn saturating_add(a: u64, b: u64, bits: u32, signed: bool) -> u64 {
    let sum = (a as i128) + (b as i128);
    if signed {
        let (min, max) = (i64::MIN >> (64 - bits), (1i64 << (bits - 1)) - 1);
        sum.clamp(min as i128, max as i128) as u64
    } else {
        let max = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
        (sum as u128).min(max as u128) as u64
    }
}

/// Saturating subtract of two `bits`-wide lanes (`signed` selects SQSUB/UQSUB).
pub(crate) fn saturating_sub(a: u64, b: u64, bits: u32, signed: bool) -> u64 {
    let diff = (a as i128) - (b as i128);
    if signed {
        let (min, max) = (i64::MIN >> (64 - bits), (1i64 << (bits - 1)) - 1);
        diff.clamp(min as i128, max as i128) as u64
    } else {
        let max = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
        diff.clamp(0, max as i128) as u64
    }
}

/// Saturate a wide intermediate back into a `bits`-wide lane.
pub(crate) fn saturate_to(v: i128, bits: u32, unsigned: bool) -> u64 {
    if unsigned {
        let max = if bits == 64 { i128::from(u64::MAX) } else { (1i128 << bits) - 1 };
        v.clamp(0, max) as u64
    } else {
        let max = (1i128 << (bits - 1)) - 1;
        v.clamp(-(1i128 << (bits - 1)), max) as u64 & (elem_mask(bits) as u64)
    }
}

/// The variable-shift family: SSHL/USHL, plus the saturating (SQSHL/UQSHL),
/// rounding (SRSHL/URSHL) and both (SQRSHL/UQRSHL) forms. A negative amount
/// shifts right.
///
/// The amount is the low **8 bits** of `b` sign-extended, not the whole lane:
/// masking it to the element width made a negative amount impossible below
/// 64 bits, so `sshl v0.4s, v1.4s, v2.4s` could only ever shift left.
pub(crate) fn shift_by_reg(
    a: u64,
    b: u64,
    bits: u32,
    unsigned: bool,
    rounding: bool,
    saturating: bool,
) -> u64 {
    let amount = sext_u64(b & 0xFF, 8) as i64;
    let value = if unsigned {
        i128::from(a & (elem_mask(bits) as u64))
    } else {
        i128::from(sext_u64(a, bits) as i64)
    };
    let shifted = if amount >= 0 {
        let sh = amount as u32;
        // Any lane is at most 64 bits, so a shift that far always overflows
        // and the saturated answer only depends on the sign.
        if sh >= 64 {
            if value == 0 {
                0
            } else if !saturating {
                return 0;
            } else {
                return saturate_to(
                    if value > 0 { i128::MAX } else { i128::MIN },
                    bits,
                    unsigned,
                );
            }
        } else {
            value << sh
        }
    } else {
        // 127 rather than the true 128: the rounding constant would overflow,
        // and both shift every bit of a 64-bit lane away regardless.
        let sh = (-amount as u32).min(127);
        let rounded = if rounding { value + (1i128 << (sh - 1)) } else { value };
        rounded >> sh
    };
    if saturating {
        saturate_to(shifted, bits, unsigned)
    } else {
        (shifted as u64) & (elem_mask(bits) as u64)
    }
}

/// FP max/min with ARM semantics: if either operand is NaN the NaN operand is
/// returned (Rust's `f64::max` would discard it).
/// `FMULX`: an ordinary multiply, except that zero times infinity is 2.0 with
/// the sign of the product rather than a NaN. That is the whole reason the
/// instruction exists -- it is what makes `FRECPS`/`FRSQRTS` behave at the
/// extremes of Newton-Raphson refinement, where a reciprocal estimate of
/// infinity has to multiply back to a finite number.
pub(crate) fn fmulx(x: f64, y: f64) -> f64 {
    if (x == 0.0 && y.is_infinite()) || (x.is_infinite() && y == 0.0) {
        return if x.is_sign_negative() != y.is_sign_negative() { -2.0 } else { 2.0 };
    }
    x * y
}

pub(crate) fn fp_max(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else {
        a.max(b)
    }
}

pub(crate) fn fp_min(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else {
        a.min(b)
    }
}

/// FMAXNM/FMINNM: same NaN handling as the plain max/min.
pub(crate) fn fp_maxnum(a: f64, b: f64) -> f64 {
    fp_max(a, b)
}

pub(crate) fn fp_minnum(a: f64, b: f64) -> f64 {
    fp_min(a, b)
}

#[inline(always)]
pub(crate) fn sext_u64<T: Into<u64>>(v: T, bits: u32) -> u64 {
    let v = v.into();
    if bits >= 64 {
        return v;
    }
    let sign = 1u64 << (bits - 1);
    let mask = (1u64 << bits) - 1;
    let v = v & mask;
    if v & sign != 0 {
        v | !mask
    } else {
        v
    }
}

/// Shift `v` left/right logically or arithmetically, or rotate, by `sa`.
#[inline(always)]
pub(crate) fn shift_reg(v: u64, st: u32, sa: u32, sf: bool) -> u64 {
    let size = if sf { 64 } else { 32 };
    let mask = if sf { u64::MAX } else { u32::MAX as u64 };
    let v = v & mask;
    match st {
        0 => {
            // LSL
            if sa >= size {
                0
            } else if sa == 0 {
                v
            } else {
                (v << sa) & mask
            }
        }
        1 => {
            // LSR
            if sa >= size {
                0
            } else if sa == 0 {
                v
            } else {
                v >> sa
            }
        }
        2 => {
            // ASR. The operand was masked to its own width above, so it has to
            // be sign-extended from *that* width before shifting — shifting the
            // masked value as a positive i64 turned `asr w0, w0, w1` on a
            // negative word into a small positive number, which is how
            // libjpeg-turbo's HUFF_EXTEND lost the sign of every DC difference.
            if sa == 0 {
                v
            } else if sa >= size {
                if v & (1 << (size - 1)) != 0 {
                    mask
                } else {
                    0
                }
            } else {
                let signed = sext_u64(v, size) as i64;
                ((signed >> sa) as u64) & mask
            }
        }
        _ => {
            // ROR
            if sa == 0 {
                v
            } else if sf {
                v.rotate_right(sa % 64)
            } else {
                ((v as u32).rotate_right(sa % 32)) as u64
            }
        }
    }
}

/// Variable shift by register amount (LSLV/LSRV/ASRV).
pub(crate) fn shift_var(v: u64, amt: u64, kind: u32, sf: bool) -> u64 {
    let size = if sf { 64 } else { 32 };
    let amt = (amt % size) as u32;
    shift_reg(v, kind, amt, sf)
}

/// Extend a register value for the ADD/SUB extended-register form.
#[inline(always)]
pub(crate) fn extend_reg(v: u64, option: u8, sf: bool) -> u64 {
    match option {
        0b000 => v as u8 as u64,        // UXTB
        0b001 => v as u16 as u64,       // UXTH
        0b010 => v as u32 as u64,       // UXTW
        0b011 => v,                     // UXTX / LSL
        0b100 => sext_u64(v, 8),        // SXTB
        0b101 => sext_u64(v, 16),       // SXTH
        0b110 => sext_u64(v, 32),       // SXTW
        0b111 => v,                     // SXTX
        _ => v,
    }
    .min(if sf { u64::MAX } else { u32::MAX as u64 })
}

/// Decode the rotated-element bitmask of the logical-immediate encoding.
/// Decode a logical-immediate (AND/ORR/EOR/ANDS) bitmask, per ARM ARM
/// `DecodeBitMasks`. Matches QEMU `logic_imm_decode_wmask`: the element size
/// is derived from `N:NOT(imms)` and bits of `imms` above the element size
/// are ignored (e.g. `mov w20, #0x80808080`).
/// Expand a MOVI/MVNI 8-bit immediate per ARM `AdvSIMDExpandImm` (mirrors
/// QEMU's `asimd_imm_const`). Returns the 64-bit lane value; the caller
/// replicates it over the 128-bit register for Q=1.
pub(crate) fn simd_imm_const(imm: u32, cmode: u32, op: u32) -> u64 {
    let mut imm = imm;
    match cmode {
        0 | 1 => {}
        2 | 3 => imm <<= 8,
        4 | 5 => imm <<= 16,
        6 | 7 => imm <<= 24,
        8 | 9 => imm |= imm << 16,
        10 | 11 => imm = (imm << 8) | (imm << 24),
        12 => imm = (imm << 8) | 0xff,
        13 => imm = (imm << 16) | 0xffff,
        14 => {
            if op == 1 {
                // Byte-mask form: imm's set bits select 0xff bytes.
                let mut imm64 = 0u64;
                for n in 0..8 {
                    if imm & (1 << n) != 0 {
                        imm64 |= 0xffu64 << (n * 8);
                    }
                }
                return imm64;
            }
            imm |= (imm << 8) | (imm << 16) | (imm << 24);
        }
        15 => {
            if op == 1 {
                // 64-bit float immediate (valid for AArch64).
                let mut imm64 = ((imm & 0x3f) as u64) << 48;
                if imm & 0x80 != 0 {
                    imm64 |= 0x8000_0000_0000_0000;
                }
                if imm & 0x40 != 0 {
                    imm64 |= 0x3fc0_0000_0000_0000;
                } else {
                    imm64 |= 0x4000_0000_0000_0000;
                }
                return imm64;
            }
            imm = ((imm & 0x80) << 24)
                | ((imm & 0x3f) << 19)
                | if imm & 0x40 != 0 { 0x1f << 25 } else { 1 << 30 };
        }
        _ => {}
    }
    if op != 0 {
        imm = !imm;
    }
    (imm as u64) | ((imm as u64) << 32)
}

pub(crate) fn decode_bit_mask(sf: bool, n: u32, immr: u32, imms: u32) -> Option<u64> {
    if !sf && n != 0 {
        return None;
    }
    let combined = ((n & 1) << 6) | ((!imms) & 0x3F);
    if combined == 0 {
        return None;
    }
    let len = 32 - combined.leading_zeros() - 1;
    let e = 1u64 << len;
    let levels = e - 1;
    let s = imms as u64 & levels;
    let r = immr as u64 & levels;
    if s == levels {
        return None;
    }
    let mut welem = (1u64 << (s + 1)) - 1;
    if r != 0 {
        welem = (welem >> r) | (welem << (e - r));
        if e < 64 {
            welem &= (1u64 << e) - 1;
        }
    }
    let datasize = if sf { 64 } else { 32 };
    let mut wmask = 0u64;
    let mut shift = 0u32;
    while shift < datasize {
        wmask |= welem.wrapping_shl(shift);
        shift += e as u32;
    }
    Some(wmask)
}

/// SBFM / BFM / UBFM semantics.
///
/// The result is truncated to the operand width: a write to a W register zeroes
/// bits 63:32, and SBFM's sign extension would otherwise fill them (`asr w0,
/// w0, #31` produced `0xFFFF_FFFF_FFFF_FFFF`, so any later 64-bit use of that
/// register saw a huge value).
pub(crate) fn bitfield_apply(opc: u32, val: u64, cur: u64, immr: u32, imms: u32, sf: bool) -> u64 {
    let width = if sf { u64::MAX } else { u64::from(u32::MAX) };
    bitfield_value(opc, val, cur, immr, imms, sf) & width
}

fn bitfield_value(opc: u32, val: u64, cur: u64, immr: u32, imms: u32, sf: bool) -> u64 {
    let datasize = if sf { 64 } else { 32 };
    let lsb = immr as u64;
    let msb = imms as u64;

    match opc {
        // UBFM
        0b10 => {
            if msb >= lsb {
                let width = (msb - lsb + 1) as u32;
                (val >> lsb) & mask_of_width(width, sf)
            } else {
                // UBFIZ: field at the bottom, shifted up
                let shift = datasize - lsb;
                ((val & mask_of_width((msb + 1) as u32, sf)).wrapping_shl(shift as u32))
                    & mask_of_width(64, sf)
            }
        }
        // SBFM
        0b00 => {
            if msb >= lsb {
                let width = (msb - lsb + 1) as u32;
                sext_u64(val >> lsb, width)
            } else {
                let shift = datasize - lsb;
                let field = val & mask_of_width((msb + 1) as u32, sf);
                let shifted = field.wrapping_shl(shift as u32);
                // sign extend from bit (msb) after the shift
                let sign_bit = msb as u32;
                if shifted & (1u64 << sign_bit) != 0 {
                    shifted | !mask_of_width((shift + msb + 1) as u32, sf)
                } else {
                    shifted & mask_of_width((shift + msb + 1) as u32, sf)
                }
            }
        }
        // BFM — merges Rn into the ORIGINAL Rd (BFI / BFXIL). The old decoder
        // used `cur = val` (Rn) and never read the destination register, so
        // `bfi` zeroed the bits it was meant to preserve. libtransistor's
        // squashfs `swab_super` relies on this.
        0b01 => {
            if msb >= lsb {
                let width = (msb - lsb + 1) as u32;
                let field = (val >> lsb) & mask_of_width(width, sf);
                (cur & !mask_of_width(width, sf)) | field
            } else {
                let field = val & mask_of_width((msb + 1) as u32, sf);
                let shift = (datasize - lsb) as u32;
                let m = mask_of_width((msb + 1) as u32, sf).wrapping_shl(shift);
                (cur & !m) | (field << shift)
            }
        }
        _ => 0,
    }
}

pub(crate) fn mask_of_width(width: u32, _sf: bool) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

pub(crate) fn reverse_bits(v: u64, size: u32) -> u64 {
    let r = v.reverse_bits();
    if size == 64 {
        r
    } else {
        (r >> 32) as u32 as u64
    }
}

pub(crate) fn reverse_16_lanes(v: u64, size: u32) -> u64 {
    let mut out = 0u64;
    let lanes = size / 16;
    for i in 0..lanes {
        let lane = ((v >> (i * 16)) & 0xFFFF) as u16;
        out |= ((lane.swap_bytes() as u64) & 0xFFFF) << (i * 16);
    }
    out
}

pub(crate) fn reverse_32_lanes(v: u64, size: u32) -> u64 {
    let mut out = 0u64;
    let lanes = size / 32;
    for i in 0..lanes {
        let lane = ((v >> (i * 32)) & 0xFFFF_FFFF) as u32;
        out |= ((lane.swap_bytes() as u64) & 0xFFFF_FFFF) << (i * 32);
    }
    out
}

pub(crate) fn clz(v: u64, size: u32) -> u64 {
    let v = if size == 32 { (v as u32) as u64 } else { v };
    (if size == 64 {
        v.leading_zeros()
    } else {
        (v as u32).leading_zeros()
    }) as u64
}

/// CLS: how many bits after the sign bit match it (so 31/63 for 0 and -1).
pub(crate) fn cls(v: u64, size: u32) -> u64 {
    if size == 32 {
        let v = v as u32 as i32;
        let magnitude = if v < 0 { !v as u32 } else { v as u32 };
        u64::from(magnitude.leading_zeros() - 1)
    } else {
        let v = v as i64;
        let magnitude = if v < 0 { !v as u64 } else { v as u64 };
        u64::from(magnitude.leading_zeros() - 1)
    }
}

pub(crate) fn ctz(v: u64, size: u32) -> u64 {
    let v = if size == 32 { (v as u32) as u64 } else { v };
    (if size == 64 {
        v.trailing_zeros()
    } else {
        (v as u32).trailing_zeros()
    }) as u64
}


/// Absolute difference of two `bits`-wide lanes (SABD/UABD).
pub(crate) fn simd_abs_diff(a: u64, b: u64, bits: u32, unsigned: bool) -> u64 {
    if unsigned {
        if a >= b { a - b } else { b - a }
    } else {
        let sa = sext_u64(a, bits) as i64;
        let sb = sext_u64(b, bits) as i64;
        (sa - sb).unsigned_abs()
    }
}

/// CRC32/CRC32C accumulate over the low `size` bits of `val`.
///
/// ARM specifies these in terms of the bit-reversed accumulator and a
/// polynomial division, which is exactly the classic reflected CRC loop over
/// the bytes of `val` from least significant upwards.
pub(crate) fn crc32(acc: u32, val: u64, size: u32, castagnoli: bool) -> u32 {
    let poly: u32 = if castagnoli { 0x82F6_3B78 } else { 0xEDB8_8320 };
    let mut crc = acc;
    for i in 0..size / 8 {
        crc ^= ((val >> (i * 8)) & 0xFF) as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (poly & 0u32.wrapping_sub(crc & 1));
        }
    }
    crc
}

/// Widen an IEEE-754 half to a single. Exact for every input, so the callers
/// that want a double can promote the result without rounding twice.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1F);
    let mant = u32::from(h & 0x3FF);
    if exp == 0x1F {
        // Infinity, or a NaN whose payload carries over quieted.
        let bits = if mant == 0 {
            sign | 0x7F80_0000
        } else {
            sign | 0x7FC0_0000 | (mant << 13)
        };
        return f32::from_bits(bits);
    }
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal halves are normal singles: shift the leading one up into
        // the implicit position and pay for it in the exponent.
        let mut m = mant;
        let mut e: i32 = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        let bits = sign | (((e + 127) as u32) << 23) | ((m & 0x3FF) << 13);
        return f32::from_bits(bits);
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// Narrow a double to an IEEE-754 half, rounding to nearest-even once.
///
/// Singles come through here promoted (which is exact), so `fcvt h, s` rounds
/// once rather than once into a double and again into the half.
pub(crate) fn f64_to_f16(v: f64) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 48) as u16) & 0x8000;
    if v.is_nan() {
        // Keep the top of the payload and force it quiet, so the result cannot
        // decay into an infinity.
        return sign | 0x7C00 | 0x200 | (((bits >> 42) as u16) & 0x1FF);
    }
    if v.is_infinite() {
        return sign | 0x7C00;
    }
    let exp = (((bits >> 52) & 0x7FF) as i32) - 1023;
    // Beyond the half's range in either direction the answer is fixed: 2^-25 is
    // the largest magnitude that still ties down to zero.
    if v == 0.0 || exp < -25 {
        return sign;
    }
    if exp > 15 {
        return sign | 0x7C00;
    }
    // Round the 53-bit significand down to the 11 bits a normal half keeps, or
    // to whatever fewer bits the fixed 2^-24 subnormal step leaves.
    let sig = (1u64 << 52) | (bits & 0x000F_FFFF_FFFF_FFFF);
    let shift = if exp >= -14 { 42 } else { (28 - exp) as u32 };
    let truncated = sig >> shift;
    let rem = sig & ((1u64 << shift) - 1);
    let halfway = 1u64 << (shift - 1);
    let rounded = if rem > halfway || (rem == halfway && truncated & 1 == 1) {
        truncated + 1
    } else {
        truncated
    };
    if exp < -14 {
        // A subnormal that rounds up to 0x400 lands on the smallest normal,
        // which is what that bit pattern already means.
        return sign | (rounded as u16);
    }
    let mut half_exp = exp + 15;
    let mut mant = rounded;
    if mant == 0x800 {
        mant = 0x400;
        half_exp += 1;
    }
    sign | ((half_exp as u16) << 10) | ((mant as u16) & 0x3FF)
}
