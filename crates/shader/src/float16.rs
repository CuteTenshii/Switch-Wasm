//! IEEE half-precision conversions, as Maxwell's fp16 ALU and texture formats use them.

/// Convert to a half, round to nearest even (as Maxwell's fp16 ALU and WGSL do).
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    // A NaN keeps a set mantissa so it stays a NaN.
    if exp == 0xFF {
        return sign | 0x7C00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let exp = exp - 127 + 15;
    if exp >= 0x1F {
        return sign | 0x7C00;
    }
    if exp <= 0 {
        // Subnormal result; rounding may carry into the smallest normal.
        let shift = (1 - exp) as u32 + 13;
        if shift > 25 {
            return sign;
        }
        let significand = mantissa | 0x80_0000;
        return sign | (round_to_nearest_even(significand, shift) as u16);
    }
    // Rounded as one number so a mantissa carry steps the exponent, up to infinity.
    sign | (round_to_nearest_even(((exp as u32) << 23) | mantissa, 13) as u16)
}

/// `value >> shift`, rounded to nearest with ties to even.
fn round_to_nearest_even(value: u32, shift: u32) -> u32 {
    let kept = value >> shift;
    let dropped = value & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    if dropped > halfway || (dropped == halfway && kept & 1 != 0) {
        kept + 1
    } else {
        kept
    }
}

pub fn f16_to_f32(v: u16) -> f32 {
    let sign = ((v as u32) & 0x8000) << 16;
    let exp = ((v as u32) >> 10) & 0x1F;
    let mantissa = (v as u32) & 0x3FF;
    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: shift the top set bit to bit 10.
        let shift = mantissa.leading_zeros() - 22;
        let exp = 127 - 15 - shift;
        let mantissa = (mantissa << (shift + 1)) & 0x3FF;
        f32::from_bits(sign | (exp << 23) | (mantissa << 13))
    } else if exp == 0x1F {
        f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13))
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mantissa << 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_float_roundtrip() {
        for v in [0.0f32, 1.0, 0.5, -2.5, 65504.0] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v, "{}", v);
        }
    }

    /// Encoding rounds to nearest even and produces subnormals.
    #[test]
    fn halves_round_to_nearest_even() {
        // 1 + 2^-11 is halfway between 0x3C00 and 0x3C01: ties to even gives 0x3C00.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00);
        // Halfway between 0x3C01 and 0x3C02 goes to 0x3C02.
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
        // Just over halfway rounds up whatever the parity.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11) * 1.01), 0x3C01);
        // Subnormals are m * 2^-24; the largest is adjacent to the smallest normal.
        assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16(-2f32.powi(-24)), 0x8001);
        assert_eq!(f32_to_f16(0x3FF as f32 * 2f32.powi(-24)), 0x03FF);
        assert_eq!(f32_to_f16(2f32.powi(-14)), 0x0400);
        // Half the smallest subnormal ties to zero.
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_f16(2f32.powi(-30)), 0x0000);
        // Out of range in both directions.
        assert_eq!(f32_to_f16(70000.0), 0x7C00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    /// Subnormal halves decode exactly.
    #[test]
    fn subnormal_halves_decode_to_their_true_value() {
        // A subnormal's value is its mantissa times 2^-24, exactly.
        for mantissa in [1u16, 2, 3, 0x155, 0x200, 0x3FF] {
            let expected = mantissa as f32 * 2.0f32.powi(-24);
            assert_eq!(f16_to_f32(mantissa), expected, "half {mantissa:#06x}");
            assert_eq!(
                f16_to_f32(mantissa | 0x8000),
                -expected,
                "negative {mantissa:#06x}"
            );
        }
        // The largest subnormal and the smallest normal are adjacent.
        assert_eq!(f16_to_f32(0x0400), 2.0f32.powi(-14));
        assert!(f16_to_f32(0x03FF) < f16_to_f32(0x0400));
    }
}
