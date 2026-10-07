//! The A32 barrel shifter and the data-processing immediate, both producing a carry.

/// `RRX`: `ROR` with an immediate amount of zero.
const SHIFT_RRX: u8 = 4;

/// Resolve an immediate shift's type and amount, where an amount of zero is special.
#[inline]
pub(super) fn decode_imm_shift(ty: u8, imm5: u8) -> (u8, u32) {
    match ty {
        0 => (0, u32::from(imm5)),
        1 | 2 if imm5 == 0 => (ty, 32),
        1 | 2 => (ty, u32::from(imm5)),
        _ if imm5 == 0 => (SHIFT_RRX, 1),
        _ => (3, u32::from(imm5)),
    }
}

/// The barrel shifter, returning the carry it produces as well as the value.
#[inline]
pub(super) fn shift_c(value: u32, ty: u8, amount: u32, carry_in: bool) -> (u32, bool) {
    match ty {
        0 => match amount {
            0 => (value, carry_in),
            1..=31 => (value << amount, (value >> (32 - amount)) & 1 != 0),
            32 => (0, value & 1 != 0),
            _ => (0, false),
        },
        1 => match amount {
            0 => (value, carry_in),
            1..=31 => (value >> amount, (value >> (amount - 1)) & 1 != 0),
            32 => (0, value >> 31 != 0),
            _ => (0, false),
        },
        2 => match amount {
            0 => (value, carry_in),
            1..=31 => (
                ((value as i32) >> amount) as u32,
                (value >> (amount - 1)) & 1 != 0,
            ),
            _ => {
                let sign = (value as i32) >> 31;
                (sign as u32, sign != 0)
            }
        },
        3 => {
            if amount == 0 {
                return (value, carry_in);
            }
            let by = amount % 32;
            if by == 0 {
                (value, value >> 31 != 0)
            } else {
                (value.rotate_right(by), (value >> (by - 1)) & 1 != 0)
            }
        }
        _ => ((u32::from(carry_in) << 31) | (value >> 1), value & 1 != 0),
    }
}

/// The 12-bit data-processing immediate: an 8-bit value rotated right by twice a 4-bit field.
#[inline]
pub(super) fn expand_imm_c(imm12: u32, carry_in: bool) -> (u32, bool) {
    let rot = (imm12 >> 8) & 0xF;
    let val = imm12 & 0xFF;
    if rot == 0 {
        (val, carry_in)
    } else {
        let out = val.rotate_right(rot * 2);
        (out, out >> 31 != 0)
    }
}
