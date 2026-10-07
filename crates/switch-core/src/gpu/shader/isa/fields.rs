//! Bit-field extraction and the field tables decoders share.

use super::*;

pub(super) fn field(insn: u64, pos: u32, len: u32) -> u64 {
    (insn >> pos) & ((1u64 << len) - 1)
}

/// Whether a control-flow instruction's condition-code test (bits 0..5) can ever be true.
pub(super) fn flow_test_can_hold(insn: u64) -> bool {
    const NEVER: u64 = 0;
    const FCSM_TR: u64 = 28;
    !matches!(field(insn, 0, 5), NEVER | FCSM_TR)
}

pub(super) fn sfield(insn: u64, pos: u32, len: u32) -> i64 {
    let v = field(insn, pos, len);
    let sign = 1u64 << (len - 1);
    if v & sign != 0 {
        (v | !((1u64 << len) - 1)) as i64
    } else {
        v as i64
    }
}

pub(super) fn reg(insn: u64, pos: u32, len: u32) -> u8 {
    field(insn, pos, len) as u8
}

/// `RZ`, the register that reads as zero and discards writes.
pub const RZ: u8 = 0xff;

pub(super) fn opt_reg(r: u8) -> Option<u8> {
    if r == RZ {
        None
    } else {
        Some(r)
    }
}

/// The guard predicate every instruction carries.
pub(super) fn guard(insn: u64) -> Pred {
    Pred {
        reg: reg(insn, 16, 3),
        negate: field(insn, 19, 1) != 0,
    }
}

/// A source predicate at `[pos, pos+3)` with its negate flag at `not`.
pub(super) fn src_pred(insn: u64, pos: u32, not: u32) -> Pred {
    Pred {
        reg: reg(insn, pos, 3),
        negate: field(insn, not, 1) != 0,
    }
}

/// `C34_RZ_O14_20`.
pub(super) fn const_operand(insn: u64) -> Operand {
    Operand::Const {
        bank: reg(insn, 34, 5),
        offset: (sfield(insn, 20, 14) << 2) as u16,
    }
}

/// `S20_20`: 19 bits at 20 plus a sign bit at 56.
pub(super) fn imm20(insn: u64) -> u32 {
    let v = field(insn, 20, 19) | (field(insn, 56, 1) << 19);
    // Sign-extend the 20-bit field to 32 bits.
    if v & (1 << 19) != 0 {
        (v | !0xf_ffff) as u32
    } else {
        v as u32
    }
}

/// `F20_20`: the same 20 bits, but they are the *top* 20 bits of an f32.
pub(super) fn imm20f(insn: u64) -> u32 {
    ((field(insn, 20, 19) | (field(insn, 56, 1) << 19)) << 12) as u32
}

pub(super) fn fcmp(bits: u64) -> FCmp {
    match bits {
        0 => FCmp::Never,
        1 => FCmp::Lt,
        2 => FCmp::Eq,
        3 => FCmp::Le,
        4 => FCmp::Gt,
        5 => FCmp::Ne,
        6 => FCmp::Ge,
        7 => FCmp::Num,
        8 => FCmp::Nan,
        9 => FCmp::LtU,
        10 => FCmp::EqU,
        11 => FCmp::LeU,
        12 => FCmp::GtU,
        13 => FCmp::NeU,
        14 => FCmp::GeU,
        _ => FCmp::Always,
    }
}

pub(super) fn icmp(bits: u64) -> ICmp {
    match bits {
        0 => ICmp::Never,
        1 => ICmp::Lt,
        2 => ICmp::Eq,
        3 => ICmp::Le,
        4 => ICmp::Gt,
        5 => ICmp::Ne,
        6 => ICmp::Ge,
        _ => ICmp::Always,
    }
}

pub trait FlowLogic<T> {
    fn constant(&self, value: bool) -> T;
    fn not(&self, a: T) -> T;
    fn and(&self, a: T, b: T) -> T;
    fn or(&self, a: T, b: T) -> T;
    fn xor(&self, a: T, b: T) -> T;
}

pub fn flow_test<T: Clone>(test: u8, [z, s, c, o]: [T; 4], l: &impl FlowLogic<T>) -> Option<T> {
    Some(match test {
        0 => l.constant(false),                    // F
        1 => l.xor(l.and(s.clone(), l.not(z)), o), // LT
        2 => l.and(l.not(s), z),                   // EQ
        3 => l.xor(s, l.or(z, o)),                 // LE
        4 => l.and(l.xor(l.not(s), o), l.not(z)),  // GT
        5 => l.not(z),                             // NE
        6 => l.not(l.xor(s, o)),                   // GE
        7 => l.or(l.not(s), l.not(z)),             // NUM
        8 => l.and(s, z),                          // NaN
        9 => l.xor(s, o),                          // LTU
        10 => z,                                   // EQU
        11 => l.or(l.xor(s, o), z),                // LEU
        12 => l.xor(l.not(s), l.or(z, o)),         // GTU
        13 => l.or(s, l.not(z)),                   // NEU
        14 => l.xor(l.or(l.not(s), z), o),         // GEU
        15 => l.constant(true),                    // T
        16 => l.not(o),                            // OFF
        17 => l.not(c),                            // LO
        18 => l.not(s),                            // SFF
        19 => l.or(z, l.not(c)),                   // LS
        20 => l.and(c, l.not(z)),                  // HI
        21 => s,                                   // SFT
        22 => c,                                   // HS
        23 => o,                                   // OFT
        30 => l.or(s, z),                          // RLE
        31 => l.and(l.not(s), l.not(z)),           // RGT
        _ => return None,
    })
}

pub(super) fn bool_op(bits: u64) -> Option<BoolOp> {
    match bits {
        0 => Some(BoolOp::And),
        1 => Some(BoolOp::Or),
        2 => Some(BoolOp::Xor),
        _ => None,
    }
}

/// `tab5cb8_1`/`tab5ce0_1`-style integer type: `(bytes, signed)`.
pub(super) fn int_type(bits: u64) -> Option<(u8, bool)> {
    match bits {
        0 => Some((1, false)),
        1 => Some((2, false)),
        2 => Some((4, false)),
        4 => Some((1, true)),
        5 => Some((2, true)),
        6 => Some((4, true)),
        _ => None,
    }
}

/// The `FloatFormat` a conversion names, in bits.
pub(super) fn float_width(bits: u64) -> Option<u8> {
    match bits {
        1 => Some(16),
        2 => Some(32),
        _ => None,
    }
}

pub(super) fn fround(bits: u64) -> FRound {
    match bits {
        0 => FRound::Nearest,
        1 => FRound::Floor,
        2 => FRound::Ceil,
        _ => FRound::Trunc,
    }
}

/// `tab8000_0`/`tabeed0sz`-style transfer size.
pub(super) fn mem_size(bits: u64) -> MemSize {
    match bits {
        0 => MemSize::U8,
        1 => MemSize::S8,
        2 => MemSize::U16,
        3 => MemSize::S16,
        4 => MemSize::B32,
        5 => MemSize::B64,
        6 => MemSize::B128,
        _ => MemSize::B32,
    }
}

/// `tabeff0_0`: `ld`/`st` in attribute space only carry the wide sizes.
pub(super) fn attr_size(bits: u64) -> MemSize {
    match bits {
        0 => MemSize::B32,
        1 => MemSize::B64,
        2 => MemSize::B96,
        _ => MemSize::B128,
    }
}
