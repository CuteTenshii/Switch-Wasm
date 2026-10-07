//! WGSL helper functions emitted on demand.

/// WGSL helpers, in dependency order; only reached ones are emitted.
pub(super) const HELPERS: &[(&str, &str)] = &[
    // Local memory `l[]`, a byte-addressed private array; out-of-range bytes read zero.
    (
        "local",
        "\
var<private> local_mem: array<u32, 256>;

fn localByte(i: u32) -> u32 {
  if (i >= 1024u) { return 0u; }
  return (local_mem[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

fn localWord(i: u32) -> u32 {
  if (i >= 1024u) { return 0u; }
  return localByte(i) | (localByte(i + 1u) << 8u) | (localByte(i + 2u) << 16u)
    | (localByte(i + 3u) << 24u);
}

fn setLocalByte(i: u32, v: u32) {
  let shift = (i & 3u) * 8u;
  local_mem[i >> 2u] = (local_mem[i >> 2u] & ~(0xffu << shift)) | ((v & 0xffu) << shift);
}",
    ),
    (
        "ftz",
        "\
fn ftz(v: f32) -> f32 {
  // `.ftz` flushes a subnormal to a zero of the same sign.
  if (v != 0.0 && abs(v) < 1.1754943508222875e-38) {
    return bitcast<f32>(bitcast<u32>(v) & 0x80000000u);
  }
  return v;
}",
    ),
    (
        "fsat",
        "\
fn fsat(v: f32) -> f32 {
  // A saturating instruction answers 0 for NaN rather than propagating it.
  if (v != v) { return 0.0; }
  return clamp(v, 0.0, 1.0);
}",
    ),
    (
        "hftz",
        "\
fn hftz(v: vec2<f32>) -> vec2<f32> {
  // A half instruction's `.ftz` flushes a subnormal *half* to a zero of the
  // same sign, which starts four orders of magnitude above an f32's.
  let signs = bitcast<vec2<f32>>(bitcast<vec2<u32>>(v) & vec2<u32>(0x80000000u));
  return select(v, signs, abs(v) < vec2<f32>(6.103515625e-5));
}",
    ),
    (
        "ftz2",
        "\
fn ftz2(v: vec2<f32>) -> vec2<f32> {
  let signs = bitcast<vec2<f32>>(bitcast<vec2<u32>>(v) & vec2<u32>(0x80000000u));
  return select(v, signs, abs(v) < vec2<f32>(1.1754943508222875e-38));
}",
    ),
    (
        "fsat2",
        "\
fn fsat2(v: vec2<f32>) -> vec2<f32> {
  // A saturating instruction answers 0 for NaN rather than propagating it.
  return clamp(select(v, vec2<f32>(0.0), v != v), vec2<f32>(0.0), vec2<f32>(1.0));
}",
    ),
    (
        "shl32",
        "\
fn shl32(a: u32, n: u32) -> u32 {
  if (n >= 32u) { return 0u; }
  return a << n;
}",
    ),
    (
        "shr32",
        "\
fn shr32(a: u32, n: u32) -> u32 {
  if (n >= 32u) { return 0u; }
  return a >> n;
}",
    ),
    (
        "sar32",
        "\
fn sar32(a: u32, n: u32) -> u32 {
  let x = bitcast<i32>(a);
  if (n >= 32u) { return bitcast<u32>(x >> 31u); }
  return bitcast<u32>(x >> n);
}",
    ),
    (
        "shf",
        "\
fn shf(lo: u32, hi: u32, count: u32, left: bool, hi_out: bool) -> u32 {
  // A 64-bit shift of a register pair, in a language with no 64-bit integer.
  let n = count & 63u;
  var rlo = lo;
  var rhi = hi;
  if (left) {
    if (n >= 32u) { rhi = shl32(lo, n - 32u); rlo = 0u; }
    else if (n > 0u) { rhi = (hi << n) | (lo >> (32u - n)); rlo = lo << n; }
  } else {
    if (n >= 32u) { rlo = shr32(hi, n - 32u); rhi = 0u; }
    else if (n > 0u) { rlo = (lo >> n) | (hi << (32u - n)); rhi = hi >> n; }
  }
  if (hi_out) { return rhi; }
  return rlo;
}",
    ),
    (
        "bfe",
        "\
fn bfe(v: u32, start0: u32, width0: u32, signed: bool) -> u32 {
  if (width0 == 0u) { return 0u; }
  let start = min(start0, 31u);
  let width = min(width0, 32u - start);
  let raw = (v >> start) & (0xffffffffu >> (32u - width));
  if (signed && width < 32u && (raw & (1u << (width - 1u))) != 0u) {
    return raw | ~(0xffffffffu >> (32u - width));
  }
  return raw;
}",
    ),
    (
        "bfi",
        "\
fn bfi(insert: u32, src: u32, base: u32) -> u32 {
  // `src` carries the field's offset in its low byte and its width in the
  // next. An offset past the word leaves the base alone, and a width that
  // would run off the end is clamped to what is left.
  let offset = src & 0xffu;
  if (offset >= 32u) { return base; }
  let count = min((src >> 8u) & 0xffu, 32u - offset);
  var mask = 0xffffffffu;
  if (count < 32u) { mask = ((1u << count) - 1u) << offset; }
  return (base & ~mask) | ((insert << offset) & mask);
}",
    ),
    (
        "lop3",
        "\
fn lop3(a: u32, b: u32, c: u32, lut: u32) -> u32 {
  // Bit n of the truth table is the result for the input combination whose
  // bits are (a, b, c) read as a three-bit number.
  var out = 0u;
  for (var i = 0u; i < 8u; i = i + 1u) {
    if ((lut & (1u << i)) == 0u) { continue; }
    var m = 0xffffffffu;
    if ((i & 4u) != 0u) { m = m & a; } else { m = m & ~a; }
    if ((i & 2u) != 0u) { m = m & b; } else { m = m & ~b; }
    if ((i & 1u) != 0u) { m = m & c; } else { m = m & ~c; }
    out = out | m;
  }
  return out;
}",
    ),
    (
        "flo",
        "\
fn flo(v0: u32, signed: bool, shift: bool) -> u32 {
  // The highest set bit, counting from bit 0; a signed search ignores the
  // sign bits at the top.
  var v = v0;
  if (signed && bitcast<i32>(v) < 0) { v = ~v; }
  if (v == 0u) { return 0xffffffffu; }
  let index = 31u - countLeadingZeros(v);
  if (shift) { return 31u - index; }
  return index;
}",
    ),
    (
        "mulhi_u",
        "\
fn mulhi_u(a: u32, b: u32) -> u32 {
  let a0 = a & 0xffffu; let a1 = a >> 16u;
  let b0 = b & 0xffffu; let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
  return a1 * b1 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
}",
    ),
    (
        "mulhi_s",
        "\
fn mulhi_s(a: u32, b: u32) -> u32 {
  var hi = mulhi_u(a, b);
  if (bitcast<i32>(a) < 0) { hi = hi - b; }
  if (bitcast<i32>(b) < 0) { hi = hi - a; }
  return hi;
}",
    ),
    (
        "sext",
        "\
fn sext(v: u32, bytes: u32) -> u32 {
  if (bytes == 1u) { return bitcast<u32>(bitcast<i32>(v << 24u) >> 24u); }
  if (bytes == 2u) { return bitcast<u32>(bitcast<i32>(v << 16u) >> 16u); }
  return v;
}",
    ),
    (
        "truncw",
        "\
fn truncw(v: u32, bytes: u32) -> u32 {
  if (bytes == 1u) { return v & 0xffu; }
  if (bytes == 2u) { return v & 0xffffu; }
  return v;
}",
    ),
    (
        "f2i_s",
        "\
fn f2i_s(v: f32, bytes: u32) -> u32 {
  // Out of range saturates and NaN is zero, and the result is the value
  // sign-extended to 32 bits however narrow the destination was.
  if (v != v) { return 0u; }
  let bits = bytes * 8u;
  let limit = exp2(f32(bits - 1u));
  if (v >= limit) { return (1u << (bits - 1u)) - 1u; }
  if (v <= -limit) { return bitcast<u32>(-(1i << (bits - 1u))); }
  return bitcast<u32>(i32(v));
}",
    ),
    (
        "f2i_u",
        "\
fn f2i_u(v: f32, bytes: u32) -> u32 {
  if (v != v) { return 0u; }
  if (v <= 0.0) { return 0u; }
  let bits = bytes * 8u;
  if (v >= exp2(f32(bits))) {
    if (bits >= 32u) { return 0xffffffffu; }
    return (1u << bits) - 1u;
  }
  return u32(v);
}",
    ),
];
