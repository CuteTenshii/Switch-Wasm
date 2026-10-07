//! Scalar arithmetic, comparison and scratch helpers.

use super::*;

/// [`isa::flow_test`] over the flags as they stand.
pub(super) struct BoolLogic;

impl isa::FlowLogic<bool> for BoolLogic {
    fn constant(&self, value: bool) -> bool {
        value
    }
    fn not(&self, a: bool) -> bool {
        !a
    }
    fn and(&self, a: bool, b: bool) -> bool {
        a && b
    }
    fn or(&self, a: bool, b: bool) -> bool {
        a || b
    }
    fn xor(&self, a: bool, b: bool) -> bool {
        a != b
    }
}

/// `fswzadd`'s multipliers per swizzle code (Eden's `FSWZ_A`/`FSWZ_B`).
pub(super) const FSWZ_SIGNS: [(f32, f32); 4] =
    [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (0.0, -1.0)];

pub(super) fn flush(v: f32, ftz: bool) -> f32 {
    if ftz && v.is_subnormal() {
        0.0f32.copysign(v)
    } else {
        v
    }
}

pub(super) fn saturate(v: f32, sat: bool) -> f32 {
    if sat {
        if v.is_nan() {
            0.0
        } else {
            v.clamp(0.0, 1.0)
        }
    } else {
        v
    }
}

/// The smallest normal half, the `.ftz` threshold.
const SMALLEST_NORMAL_HALF: f32 = 6.103_515_6e-5;

/// One half-op source: its two lanes, flushed, then modified.
pub(super) fn half_source(bits: u32, m: FMod, sw: HSwizzle, ftz: bool) -> [f32; 2] {
    let mut lanes = half_lanes(bits, sw);
    for lane in lanes.iter_mut() {
        *lane = m.apply(half_flush(*lane, sw, ftz));
    }
    lanes
}

pub(super) fn half_lanes(bits: u32, sw: HSwizzle) -> [f32; 2] {
    let low = f16_to_f32(bits as u16);
    let high = f16_to_f32((bits >> 16) as u16);
    match sw {
        HSwizzle::H1H0 => [low, high],
        HSwizzle::H0H0 => [low, low],
        HSwizzle::H1H1 => [high, high],
        // One f32 that both lanes read.
        HSwizzle::F32 => [f32::from_bits(bits); 2],
    }
}

/// `.ftz` at the threshold of the lane's actual precision.
fn half_flush(v: f32, sw: HSwizzle, ftz: bool) -> f32 {
    if !ftz {
        return v;
    }
    if sw == HSwizzle::F32 {
        return flush(v, true);
    }
    if v != 0.0 && v.abs() < SMALLEST_NORMAL_HALF {
        0.0f32.copysign(v)
    } else {
        v
    }
}

pub(super) fn half_pack(dst: u32, lanes: [f32; 2], merge: HMerge) -> u32 {
    let half = |v: f32| u32::from(f32_to_f16(v));
    match merge {
        HMerge::H1H0 => half(lanes[0]) | (half(lanes[1]) << 16),
        HMerge::F32 => lanes[0].to_bits(),
        HMerge::MrgH0 => (dst & 0xFFFF_0000) | half(lanes[0]),
        HMerge::MrgH1 => (dst & 0x0000_FFFF) | (half(lanes[1]) << 16),
    }
}

/// Whether `.fmz` zeroes this lane's product (D3D9: anything times zero is zero).
pub(super) fn fmz_zeroes(prec: HPrecision, sat: bool, a: f32, b: f32) -> bool {
    prec.zeroes_products(sat) && (a == 0.0 || b == 0.0)
}

pub(super) fn neg_if(v: f32, neg: bool) -> f32 {
    if neg {
        -v
    } else {
        v
    }
}

pub(super) fn ineg_if(v: u32, neg: bool) -> u32 {
    if neg {
        (v as i32).wrapping_neg() as u32
    } else {
        v
    }
}

pub(super) fn inv_if(v: u32, inv: bool) -> u32 {
    if inv {
        !v
    } else {
        v
    }
}

pub(super) fn apply_round(v: f32, round: FRound) -> f32 {
    match round {
        FRound::Nearest => v.round_ties_even(),
        FRound::Floor => v.floor(),
        FRound::Ceil => v.ceil(),
        FRound::Trunc => v.trunc(),
    }
}

/// A `set` result: all-ones, or 1.0f with `.bf`.
pub(super) fn set_result(r: bool, bf: bool) -> u32 {
    match (r, bf) {
        (false, _) => 0,
        (true, true) => 1.0f32.to_bits(),
        (true, false) => u32::MAX,
    }
}

pub(super) fn combine(op: BoolOp, a: bool, b: bool) -> bool {
    match op {
        BoolOp::And => a && b,
        BoolOp::Or => a || b,
        BoolOp::Xor => a != b,
    }
}

pub(super) fn float_compare(cmp: FCmp, a: f32, b: f32) -> bool {
    let unordered = a.is_nan() || b.is_nan();
    match cmp {
        FCmp::Never => false,
        FCmp::Lt => a < b,
        FCmp::Eq => a == b,
        FCmp::Le => a <= b,
        FCmp::Gt => a > b,
        FCmp::Ne => !unordered && a != b,
        FCmp::Ge => a >= b,
        FCmp::Num => !unordered,
        FCmp::Nan => unordered,
        FCmp::LtU => unordered || a < b,
        FCmp::EqU => unordered || a == b,
        FCmp::LeU => unordered || a <= b,
        FCmp::GtU => unordered || a > b,
        FCmp::NeU => unordered || a != b,
        FCmp::GeU => unordered || a >= b,
        FCmp::Always => true,
    }
}

pub(super) fn int_compare(cmp: ICmp, a: u32, b: u32, signed: bool) -> bool {
    let ord = if signed {
        (a as i32).cmp(&(b as i32))
    } else {
        a.cmp(&b)
    };
    match cmp {
        ICmp::Never => false,
        ICmp::Lt => ord.is_lt(),
        ICmp::Eq => ord.is_eq(),
        ICmp::Le => ord.is_le(),
        ICmp::Gt => ord.is_gt(),
        ICmp::Ne => ord.is_ne(),
        ICmp::Ge => ord.is_ge(),
        ICmp::Always => true,
    }
}

/// `lop3`'s truth table: bit `n` of `lut` is the result for inputs `(a, b, c)` = n.
pub(super) fn lop3(a: u32, b: u32, c: u32, lut: u8) -> u32 {
    let mut out = 0u32;
    for i in 0..8u32 {
        if lut & (1 << i) == 0 {
            continue;
        }
        let mask = mask_for(a, i & 4 != 0) & mask_for(b, i & 2 != 0) & mask_for(c, i & 1 != 0);
        out |= mask;
    }
    out
}

fn mask_for(v: u32, want_set: bool) -> u32 {
    if want_set {
        v
    } else {
        !v
    }
}

pub(super) fn bitfield_extract(v: u32, start: u32, width: u32, signed: bool) -> u32 {
    if width == 0 {
        return 0;
    }
    let start = start.min(31);
    let width = width.min(32 - start);
    let raw = (v >> start) & (u32::MAX >> (32 - width));
    if signed && width < 32 && raw & (1 << (width - 1)) != 0 {
        raw | !(u32::MAX >> (32 - width))
    } else {
        raw
    }
}

pub(super) fn sign_extend(v: u32, bytes: u8) -> u32 {
    match bytes {
        1 => v as u8 as i8 as i32 as u32,
        2 => v as u16 as i16 as i32 as u32,
        _ => v,
    }
}

pub(super) fn truncate(v: u32, bytes: u8) -> u32 {
    match bytes {
        1 => v & 0xff,
        2 => v & 0xffff,
        _ => v,
    }
}

pub(super) fn half(v: u32, high: bool, signed: bool) -> u32 {
    let h = if high { v >> 16 } else { v & 0xffff };
    if signed {
        h as u16 as i16 as i32 as u32
    } else {
        h
    }
}
/// A sub-register load, sign-extended for signed forms; `None` for whole registers.
pub(super) fn narrow_load(
    size: MemSize,
    mut byte: impl FnMut(usize) -> ShaderResult<u8>,
) -> ShaderResult<Option<u32>> {
    let width = size.bytes() as usize;
    if size.regs() != 1 || width >= 4 {
        return Ok(None);
    }
    let mut raw = 0u32;
    for i in 0..width {
        raw |= u32::from(byte(i)?) << (i * 8);
    }
    let signed = matches!(size, MemSize::S8 | MemSize::S16);
    Ok(Some(if signed {
        sign_extend(raw, size.bytes() as u8)
    } else {
        raw
    }))
}

/// Registers from byte-addressed scratch; past the end reads zero.
pub(super) fn read_scratch(bytes: &[u8], base: usize, size: MemSize) -> [u32; 4] {
    let mut out = [0u32; 4];
    if let Ok(Some(raw)) = narrow_load(size, |i| Ok(bytes.get(base + i).copied().unwrap_or(0))) {
        out[0] = raw;
        return out;
    }
    for (i, word) in out.iter_mut().enumerate().take(size.regs() as usize) {
        let mut raw = [0u8; 4];
        for (j, b) in raw.iter_mut().enumerate() {
            *b = bytes.get(base + i * 4 + j).copied().unwrap_or(0);
        }
        *word = u32::from_le_bytes(raw);
    }
    out
}

/// Write to scratch, growing to `cap`; stores past it are dropped.
pub(super) fn write_scratch(bytes: &mut Vec<u8>, cap: usize, base: usize, value: &[u8]) {
    if bytes.len() < cap {
        bytes.resize(cap, 0);
    }
    let end = base + value.len();
    if end <= bytes.len() {
        bytes[base..end].copy_from_slice(value);
    }
}

pub(super) fn wide_size(width: usize) -> MemSize {
    if width == 8 {
        MemSize::B64
    } else {
        MemSize::B32
    }
}

pub(super) fn pack(words: &[u32; 4], width: usize) -> u64 {
    if width == 8 {
        u64::from(words[0]) | (u64::from(words[1]) << 32)
    } else {
        u64::from(words[0])
    }
}

pub(super) fn unpack(value: u64, width: usize) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&(value as u32).to_le_bytes());
    if width == 8 {
        out[4..].copy_from_slice(&((value >> 32) as u32).to_le_bytes());
    }
    out
}

pub(super) fn atom_apply(
    op: AtomOp,
    ty: AtomType,
    old: u64,
    b: u64,
    stored: u64,
) -> ShaderResult<u64> {
    let wide = matches!(ty, AtomType::U64 | AtomType::S64);
    let trim = |v: u64| if wide { v } else { v & 0xFFFF_FFFF };
    let float = matches!(ty, AtomType::F32);
    Ok(match op {
        AtomOp::Add | AtomOp::SafeAdd if float => (f32::from_bits(old as u32)
            + f32::from_bits(b as u32))
        .to_bits()
        .into(),
        AtomOp::Add | AtomOp::SafeAdd => trim(old.wrapping_add(b)),
        AtomOp::Min => {
            if atom_less(ty, b, old) {
                b
            } else {
                old
            }
        }
        AtomOp::Max => {
            if atom_less(ty, old, b) {
                b
            } else {
                old
            }
        }
        // `inc` wraps to zero at the operand, `dec` back up to it; both unsigned.
        AtomOp::Inc => {
            if old >= b {
                0
            } else {
                trim(old + 1)
            }
        }
        AtomOp::Dec => {
            if old == 0 || old > b {
                b
            } else {
                old - 1
            }
        }
        AtomOp::And => old & b,
        AtomOp::Or => old | b,
        AtomOp::Xor => old ^ b,
        AtomOp::Exch => b,
        AtomOp::Cas => {
            if old == b {
                stored
            } else {
                old
            }
        }
    })
}

/// `x < y` under the atomic's type.
fn atom_less(ty: AtomType, x: u64, y: u64) -> bool {
    match ty {
        AtomType::F32 => f32::from_bits(x as u32) < f32::from_bits(y as u32),
        AtomType::S32 => (x as u32 as i32) < (y as u32 as i32),
        AtomType::S64 => (x as i64) < (y as i64),
        _ => x < y,
    }
}
