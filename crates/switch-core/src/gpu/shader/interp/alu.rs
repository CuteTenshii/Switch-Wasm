//! Executing ALU and data-movement ops.

use super::*;

impl Invocation {
    pub(super) fn run_alu(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
            // ---- attribute space ----
            Op::Ld {
                dst,
                offset,
                idx,
                size,
            } => {
                let base = offset.wrapping_add(self.attr_index(idx));
                for i in 0..size.regs() {
                    let v = self.attr_in.get(base + i as u16 * 4);
                    self.set_reg_f32(dst.wrapping_add(i), v);
                }
            }
            Op::St {
                offset,
                idx,
                src,
                size,
            } => {
                let base = offset.wrapping_add(self.attr_index(idx));
                for i in 0..size.regs() {
                    let v = self.reg_f32(src.wrapping_add(i));
                    self.attr_out.set(base + i as u16 * 4, v);
                }
            }
            // `centroid` is exact as-is: shading is per covered sample at its centre.
            Op::Ipa {
                dst,
                offset,
                mul,
                perspective,
                sat,
                centroid: _,
            } => {
                let mut v = self.attr_in.get(offset);
                if perspective {
                    if let Some(m) = mul {
                        v *= self.reg_f32(m);
                    }
                }
                if sat {
                    v = v.clamp(0.0, 1.0);
                }
                self.set_reg_f32(dst, v);
            }

            // ---- float ----
            Op::Rro { dst, src, sm } => {
                let x = sm.apply(self.operand_f32(src, env)?);
                self.set_reg_f32(dst, x);
            }
            Op::Fadd {
                dst,
                a,
                am,
                b,
                bm,
                ftz,
                sat,
            } => {
                let x = am.apply(flush(self.reg_f32(a), ftz));
                let y = bm.apply(flush(self.operand_f32(b, env)?, ftz));
                self.set_reg_f32(dst, saturate(x + y, sat));
            }
            Op::Fmul {
                dst,
                a,
                b,
                bm,
                ftz,
                fmz,
                sat,
                scale,
            } => {
                // The pre-scale applies to the first operand.
                let x = flush(self.reg_f32(a), ftz) * scale.factor();
                let y = bm.apply(flush(self.operand_f32(b, env)?, ftz));
                let product = if fmz && !sat && (x == 0.0 || y == 0.0) {
                    0.0
                } else {
                    x * y
                };
                self.set_reg_f32(dst, saturate(product, sat));
            }
            Op::Ffma {
                dst,
                a,
                b,
                bneg,
                c,
                cneg,
                ftz,
                fmz,
                sat,
            } => {
                let x = flush(self.reg_f32(a), ftz);
                let y = neg_if(flush(self.operand_f32(b, env)?, ftz), bneg);
                let z = neg_if(flush(self.operand_f32(c, env)?, ftz), cneg);
                let fused = if fmz && !sat && (x == 0.0 || y == 0.0) {
                    z
                } else {
                    x.mul_add(y, z)
                };
                self.set_reg_f32(dst, saturate(fused, sat));
            }
            Op::Fmnmx {
                dst,
                a,
                am,
                b,
                bm,
                pred,
                ftz,
            } => {
                let x = am.apply(flush(self.reg_f32(a), ftz));
                let y = bm.apply(flush(self.operand_f32(b, env)?, ftz));
                // True picks the minimum, so `fmnmx ... !pt` is `max`.
                let v = if self.holds(pred) { x.min(y) } else { x.max(y) };
                self.set_reg_f32(dst, v);
            }
            Op::Mufu {
                dst,
                src,
                sm,
                op,
                sat,
            } => {
                let x = sm.apply(self.reg_f32(src));
                let v = match op {
                    MufuOp::Cos => x.cos(),
                    MufuOp::Sin => x.sin(),
                    MufuOp::Ex2 => x.exp2(),
                    MufuOp::Lg2 => x.log2(),
                    MufuOp::Rcp => 1.0 / x,
                    MufuOp::Rsq => 1.0 / x.sqrt(),
                    MufuOp::Sqrt => x.sqrt(),
                };
                self.set_reg_f32(dst, saturate(v, sat));
            }
            // Signs depend on the lane's position in the quad.
            Op::Fswzadd {
                dst,
                a,
                b,
                swizzle,
                ftz,
            } => {
                let code = (swizzle >> ((env.special.lane & 3) * 2)) & 3;
                let x = flush(self.reg_f32(a), ftz);
                let y = flush(self.reg_f32(b), ftz);
                let (ka, kb) = FSWZ_SIGNS[code as usize];
                self.set_reg_f32(dst, ka * x + kb * y);
            }
            Op::Fsetp {
                p0,
                p1,
                a,
                am,
                b,
                bm,
                cmp,
                bop,
                src,
            } => {
                let x = am.apply(self.reg_f32(a));
                let y = bm.apply(self.operand_f32(b, env)?);
                let r = float_compare(cmp, x, y);
                let s = self.holds(src);
                self.set_pred(p0, combine(bop, r, s));
                self.set_pred(p1, combine(bop, !r, s));
            }
            Op::Fset {
                dst,
                a,
                am,
                b,
                bm,
                cmp,
                bop,
                src,
                bf,
                cc,
            } => {
                let x = am.apply(self.reg_f32(a));
                let y = bm.apply(self.operand_f32(b, env)?);
                let r = combine(bop, float_compare(cmp, x, y), self.holds(src));
                let v = set_result(r, bf);
                self.set_reg(dst, v);
                if cc {
                    self.set_flags(v);
                }
            }

            // ---- half-precision ----
            // Lanes are computed in f32 and rounded at the merge; [`HMerge::F32`] never rounds.
            Op::Hadd2 {
                dst,
                a,
                am,
                asw,
                b,
                bm,
                bsw,
                merge,
                ftz,
                sat,
            } => {
                let x = half_source(self.reg(a), am, asw, ftz);
                let y = half_source(self.operand(b, env)?, bm, bsw, ftz);
                let lanes = [saturate(x[0] + y[0], sat), saturate(x[1] + y[1], sat)];
                let v = half_pack(self.reg(dst), lanes, merge);
                self.set_reg(dst, v);
            }
            Op::Hmul2 {
                dst,
                a,
                am,
                asw,
                b,
                bm,
                bsw,
                merge,
                prec,
                sat,
            } => {
                let ftz = prec == HPrecision::Ftz;
                let x = half_source(self.reg(a), am, asw, ftz);
                let y = half_source(self.operand(b, env)?, bm, bsw, ftz);
                let mut lanes = [x[0] * y[0], x[1] * y[1]];
                for lane in 0..2 {
                    if fmz_zeroes(prec, sat, x[lane], y[lane]) {
                        lanes[lane] = 0.0;
                    }
                    lanes[lane] = saturate(lanes[lane], sat);
                }
                let v = half_pack(self.reg(dst), lanes, merge);
                self.set_reg(dst, v);
            }
            Op::Hfma2 {
                dst,
                a,
                asw,
                b,
                bneg,
                bsw,
                c,
                cneg,
                csw,
                merge,
                prec,
                sat,
            } => {
                let ftz = prec == HPrecision::Ftz;
                let x = half_source(self.reg(a), FMod::NONE, asw, ftz);
                let y = half_source(
                    self.operand(b, env)?,
                    FMod {
                        neg: bneg,
                        abs: false,
                    },
                    bsw,
                    ftz,
                );
                let z = half_source(
                    self.operand(c, env)?,
                    FMod {
                        neg: cneg,
                        abs: false,
                    },
                    csw,
                    ftz,
                );
                let mut lanes = [x[0].mul_add(y[0], z[0]), x[1].mul_add(y[1], z[1])];
                for lane in 0..2 {
                    if fmz_zeroes(prec, sat, x[lane], y[lane]) {
                        lanes[lane] = z[lane];
                    }
                    lanes[lane] = saturate(lanes[lane], sat);
                }
                let v = half_pack(self.reg(dst), lanes, merge);
                self.set_reg(dst, v);
            }
            Op::Hset2 {
                dst,
                a,
                am,
                asw,
                b,
                bm,
                bsw,
                cmp,
                bop,
                src,
                bf,
                ftz,
            } => {
                let x = half_source(self.reg(a), am, asw, ftz);
                let y = half_source(self.operand(b, env)?, bm, bsw, ftz);
                let s = self.holds(src);
                // 1.0h with `.bf`, all ones without, per half.
                let taken = if bf { 0x3C00u32 } else { 0xFFFF };
                let mut out = 0u32;
                if combine(bop, float_compare(cmp, x[0], y[0]), s) {
                    out |= taken;
                }
                if combine(bop, float_compare(cmp, x[1], y[1]), s) {
                    out |= taken << 16;
                }
                self.set_reg(dst, out);
            }
            Op::Hsetp2 {
                p0,
                p1,
                a,
                am,
                asw,
                b,
                bm,
                bsw,
                cmp,
                bop,
                src,
                and,
                ftz,
            } => {
                let x = half_source(self.reg(a), am, asw, ftz);
                let y = half_source(self.operand(b, env)?, bm, bsw, ftz);
                let s = self.holds(src);
                let low = combine(bop, float_compare(cmp, x[0], y[0]), s);
                let high = combine(bop, float_compare(cmp, x[1], y[1]), s);
                if and {
                    self.set_pred(p0, low && high);
                    self.set_pred(p1, !(low && high));
                } else {
                    self.set_pred(p0, low);
                    self.set_pred(p1, high);
                }
            }

            // ---- integer ----
            Op::Iadd {
                dst,
                a,
                aneg,
                b,
                bneg,
                cin,
                cout,
            } => {
                let x = ineg_if(self.reg(a), aneg);
                let y = ineg_if(self.operand(b, env)?, bneg);
                // Widened so the carry is bit 32. A negated operand is two's complement,
                // so subtraction carries when it doesn't borrow, as `iadd.x` expects.
                let sum = u64::from(x) + u64::from(y) + u64::from(cin && self.carry);
                self.set_reg(dst, sum as u32);
                if cout && cin {
                    self.carry = sum > u64::from(u32::MAX);
                } else if cout {
                    self.set_add_flags(x, y);
                }
            }
            Op::Iadd3 {
                dst,
                a,
                aneg,
                b,
                bneg,
                c,
                cneg,
                x: extended,
                cc,
                halves,
                shift,
            } => {
                let x = ineg_if(half16(self.reg(a), halves[0]), aneg);
                let y = ineg_if(half16(self.operand(b, env)?, halves[1]), bneg);
                let z = ineg_if(half16(self.operand(c, env)?, halves[2]), cneg);
                let (sum, first_carry) = x.overflowing_add(y);
                let lhs = sum.wrapping_add(u32::from(extended && self.carry));
                let lhs = match shift {
                    Iadd3Shift::None => lhs,
                    Iadd3Shift::Right => (lhs >> 16) | u32::from(first_carry) << 16,
                    Iadd3Shift::Left => lhs << 16,
                };
                let (result, carry) = lhs.overflowing_add(z);
                self.set_reg(dst, result);
                if cc {
                    self.zero = result == 0;
                    self.sign = (result as i32) < 0;
                    self.carry = carry;
                    let (_, signed_overflow) = (lhs as i32).overflowing_add(z as i32);
                    self.overflow = signed_overflow || first_carry;
                }
            }
            Op::Iscadd {
                dst,
                a,
                aneg,
                b,
                bneg,
                shift,
                cout,
            } => {
                let x = ineg_if(self.reg(a), aneg).wrapping_shl(shift as u32);
                let y = ineg_if(self.operand(b, env)?, bneg);
                self.set_reg(dst, x.wrapping_add(y));
                if cout {
                    self.set_add_flags(x, y);
                }
            }
            Op::Lea {
                dst,
                a,
                hi,
                b,
                neg,
                shift,
            } => {
                let offset = match hi {
                    None => ineg_if(self.reg(a), neg).wrapping_shl(u32::from(shift)),
                    Some(hi) => {
                        let wide = (u64::from(self.reg(hi)) << 32) | u64::from(self.reg(a));
                        let wide = if neg { wide.wrapping_neg() } else { wide };
                        (wide >> (32 - u32::from(shift))) as u32
                    }
                };
                let base = self.operand(b, env)?;
                self.set_reg(dst, base.wrapping_add(offset));
            }
            Op::Vmnmx {
                dst,
                a,
                b,
                c,
                max,
                then_max,
                signed,
                then_signed,
            } => {
                let pick = |x: u32, y: u32, max: bool, signed: bool| match (max, signed) {
                    (false, false) => x.min(y),
                    (true, false) => x.max(y),
                    (false, true) => (x as i32).min(y as i32) as u32,
                    (true, true) => (x as i32).max(y as i32) as u32,
                };
                let first = pick(self.reg(a), self.reg(b), max, signed);
                self.set_reg(dst, pick(first, self.reg(c), then_max, then_signed));
            }
            Op::Imnmx {
                dst,
                a,
                b,
                pred,
                signed,
            } => {
                let x = self.reg(a);
                let y = self.operand(b, env)?;
                let take_min = self.holds(pred);
                let v = if signed {
                    let (x, y) = (x as i32, y as i32);
                    (if take_min { x.min(y) } else { x.max(y) }) as u32
                } else if take_min {
                    x.min(y)
                } else {
                    x.max(y)
                };
                self.set_reg(dst, v);
            }
            Op::Imul {
                dst,
                a,
                b,
                signed,
                hi,
            } => {
                let x = self.reg(a);
                let y = self.operand(b, env)?;
                let full = if signed {
                    ((x as i32 as i64) * (y as i32 as i64)) as u64
                } else {
                    (x as u64) * (y as u64)
                };
                self.set_reg(dst, if hi { (full >> 32) as u32 } else { full as u32 });
            }
            Op::Xmad {
                dst,
                a,
                ah,
                asigned,
                b,
                bh,
                bsigned,
                c,
                cmode,
                psl,
                mrg,
            } => {
                let raw_b = self.operand(b, env)?;
                let av = half(self.reg(a), ah, asigned);
                let bv = half(raw_b, bh, bsigned);
                let mut product = av.wrapping_mul(bv);
                if psl {
                    product <<= 16;
                }
                let raw_c = self.operand(c, env)?;
                let cv = match cmode {
                    XmadC::Full => raw_c,
                    XmadC::Lo => raw_c & 0xffff,
                    XmadC::Hi => raw_c >> 16,
                    XmadC::Bcc => (raw_b << 16).wrapping_add(raw_c),
                };
                let mut v = product.wrapping_add(cv);
                if mrg {
                    // `.mrg` replaces the high half with `b`'s low half.
                    v = (v & 0xffff) | (raw_b << 16);
                }
                self.set_reg(dst, v);
            }
            Op::Isetp {
                p0,
                p1,
                a,
                b,
                cmp,
                signed,
                bop,
                src,
                x,
            } => {
                let rhs = self.operand(b, env)?;
                let r = if x {
                    self.extended_compare(cmp, self.reg(a), rhs, signed)
                } else {
                    int_compare(cmp, self.reg(a), rhs, signed)
                };
                let s = self.holds(src);
                self.set_pred(p0, combine(bop, r, s));
                self.set_pred(p1, combine(bop, !r, s));
            }
            Op::Iset {
                dst,
                a,
                b,
                cmp,
                signed,
                bop,
                src,
                bf,
                x,
                cc,
            } => {
                let rhs = self.operand(b, env)?;
                let r = if x {
                    self.extended_compare(cmp, self.reg(a), rhs, signed)
                } else {
                    int_compare(cmp, self.reg(a), rhs, signed)
                };
                let r = combine(bop, r, self.holds(src));
                let v = set_result(r, bf);
                self.set_reg(dst, v);
                if cc {
                    self.set_flags(v);
                }
            }
            Op::Icmp {
                dst,
                a,
                b,
                c,
                cmp,
                signed,
            } => {
                // `icmp dst, a, b, c` is "dst = compare(c, 0) ? a : b".
                let taken = int_compare(cmp, self.reg(c), 0, signed);
                let v = if taken {
                    self.reg(a)
                } else {
                    self.operand(b, env)?
                };
                self.set_reg(dst, v);
            }
            Op::Fcmp {
                dst,
                a,
                b,
                test,
                cmp,
                ftz,
            } => {
                let test = flush(self.operand_f32(test, env)?, ftz);
                let v = if float_compare(cmp, test, 0.0) {
                    self.reg(a)
                } else {
                    self.operand(b, env)?
                };
                self.set_reg(dst, v);
            }
            Op::Bfi {
                dst,
                insert,
                src,
                base,
            } => {
                let src = self.operand(src, env)?;
                let base = self.operand(base, env)?;
                let offset = src & 0xff;
                let count = (src >> 8) & 0xff;
                // An offset past the word leaves the base alone; an overlong width is clamped.
                let v = if offset >= 32 {
                    base
                } else {
                    let count = count.min(32 - offset);
                    let mask = if count >= 32 {
                        !0
                    } else {
                        ((1u32 << count) - 1) << offset
                    };
                    (base & !mask) | ((self.reg(insert) << offset) & mask)
                };
                self.set_reg(dst, v);
            }
            Op::R2p {
                src,
                mask,
                byte,
                flags,
            } => {
                let bits = self.reg(src) >> (u32::from(byte) * 8);
                let mask = self.operand(mask, env)?;
                for index in 0..if flags { 4 } else { 7u8 } {
                    if mask & (1 << index) != 0 {
                        let bit = bits & (1 << index) != 0;
                        match (flags, index) {
                            (false, _) => self.set_pred(index, bit),
                            (true, 0) => self.zero = bit,
                            (true, 1) => self.sign = bit,
                            (true, 2) => self.carry = bit,
                            (true, _) => self.overflow = bit,
                        }
                    }
                }
            }
            Op::Lop {
                dst,
                a,
                ainv,
                b,
                binv,
                op,
                pred,
                cc,
            } => {
                let x = inv_if(self.reg(a), ainv);
                let y = inv_if(self.operand(b, env)?, binv);
                let v = match op {
                    LogicOp::And => x & y,
                    LogicOp::Or => x | y,
                    LogicOp::Xor => x ^ y,
                    LogicOp::PassB => y,
                };
                self.set_reg(dst, v);
                if let Some((p, test)) = pred {
                    self.set_pred(p, test.holds(v));
                }
                if cc {
                    self.set_flags(v);
                }
            }
            Op::Lop3 {
                dst,
                a,
                b,
                c,
                lut,
                pred,
            } => {
                let x = self.reg(a);
                let y = self.operand(b, env)?;
                let z = self.operand(c, env)?;
                let v = lop3(x, y, z, lut);
                self.set_reg(dst, v);
                if let Some((p, test)) = pred {
                    self.set_pred(p, test.holds(v));
                }
            }
            Op::Shl { dst, a, b, wrap } => {
                let n = self.operand(b, env)?;
                let n = if wrap { n & 31 } else { n };
                self.set_reg(dst, if n >= 32 { 0 } else { self.reg(a) << n });
            }
            Op::Shr {
                dst,
                a,
                b,
                signed,
                wrap,
                brev,
            } => {
                let n = self.operand(b, env)?;
                let n = if wrap { n & 31 } else { n };
                let x = if brev {
                    self.reg(a).reverse_bits()
                } else {
                    self.reg(a)
                };
                let v = if signed {
                    if n >= 32 {
                        ((x as i32) >> 31) as u32
                    } else {
                        ((x as i32) >> n) as u32
                    }
                } else if n >= 32 {
                    0
                } else {
                    x >> n
                };
                self.set_reg(dst, v);
            }
            Op::Shf {
                dst,
                lo,
                shift,
                hi,
                left,
                wrap,
                max,
            } => {
                let n = self.operand(shift, env)?;
                let limit = max.limit();
                let n = if wrap { n & (limit - 1) } else { n.min(limit) };
                let pair = (u64::from(self.reg(hi)) << 32) | u64::from(self.reg(lo));
                let word = if left {
                    ((pair << n) >> 32) as u32
                } else if max == ShfMax::S64 {
                    ((pair as i64) >> n) as u32
                } else {
                    (pair >> n) as u32
                };
                self.set_reg(dst, word);
            }
            Op::Bfe {
                dst,
                a,
                b,
                signed,
                brev,
                cc,
            } => {
                let desc = self.operand(b, env)?;
                let start = desc & 0xff;
                let width = (desc >> 8) & 0xff;
                let source = if brev {
                    self.reg(a).reverse_bits()
                } else {
                    self.reg(a)
                };
                let v = bitfield_extract(source, start, width, signed);
                self.set_reg(dst, v);
                if cc {
                    self.set_flags(v);
                }
            }
            Op::Popc { dst, b, inv } => {
                let v = inv_if(self.operand(b, env)?, inv);
                self.set_reg(dst, v.count_ones());
            }
            Op::Flo {
                dst,
                b,
                signed,
                shift,
                inv,
            } => {
                let v = inv_if(self.operand(b, env)?, inv);
                // Highest set bit; a signed search ignores leading sign bits.
                let v = if signed && (v as i32) < 0 { !v } else { v };
                let idx = if v == 0 {
                    0xffff_ffff
                } else {
                    31 - v.leading_zeros()
                };
                self.set_reg(dst, if shift && v != 0 { 31 - idx } else { idx });
            }
            Op::Sel { dst, a, b, pred } => {
                let v = if self.holds(pred) {
                    self.reg(a)
                } else {
                    self.operand(b, env)?
                };
                self.set_reg(dst, v);
            }

            // ---- conversions ----
            Op::I2d {
                dst,
                src,
                src_bytes,
                src_signed,
            } => {
                let raw = if src_bytes == 8 {
                    self.reg64(src)
                } else if src_signed {
                    self.reg(src) as i32 as i64 as u64
                } else {
                    u64::from(self.reg(src))
                };
                let bits = int_to_f64_bits(raw, src_signed);
                self.set_reg(dst, bits as u32);
                self.set_reg(dst.wrapping_add(1), (bits >> 32) as u32);
            }
            Op::D2f { dst, src, ftz } => {
                let bits = f64_to_f32_bits(self.reg64(src), ftz);
                self.set_reg(dst, bits);
            }
            Op::I2f {
                dst,
                src,
                sm,
                src_bytes,
                src_signed,
                sel,
                dst_bits,
            } => {
                let raw = self.operand(src, env)?;
                let raw = raw >> (sel as u32 * 8);
                let v = if src_signed {
                    sign_extend(raw, src_bytes) as i32 as f32
                } else {
                    truncate(raw, src_bytes) as f32
                };
                let v = sm.apply(v);
                if dst_bits == 16 {
                    self.set_reg(dst, u32::from(f32_to_f16(v)));
                } else {
                    self.set_reg_f32(dst, v);
                }
            }
            Op::F2i {
                dst,
                src,
                sm,
                dst_bytes,
                dst_signed,
                round,
                ftz,
                src_bits,
                hi,
            } => {
                let x = if src_bits == 16 {
                    let raw = self.operand(src, env)?;
                    f16_to_f32((raw >> if hi { 16 } else { 0 }) as u16)
                } else {
                    self.operand_f32(src, env)?
                };
                let x = sm.apply(flush(x, ftz));
                let r = apply_round(x, round);
                let v = if dst_signed {
                    let lo = -(2f64.powi(dst_bytes as i32 * 8 - 1)) as f32;
                    let hi = (2f64.powi(dst_bytes as i32 * 8 - 1) - 1.0) as f32;
                    if r.is_nan() {
                        0
                    } else {
                        r.clamp(lo, hi) as i32 as u32
                    }
                } else {
                    let hi = (2f64.powi(dst_bytes as i32 * 8) - 1.0) as f32;
                    if r.is_nan() {
                        0
                    } else {
                        r.clamp(0.0, hi) as u32
                    }
                };
                self.set_reg(dst, v);
            }
            Op::F2f {
                dst,
                src,
                sm,
                round,
                sat,
                ftz,
                src_bits,
                dst_bits,
                hi,
            } => {
                let x = if src_bits == 16 {
                    let raw = self.operand(src, env)?;
                    f16_to_f32((raw >> if hi { 16 } else { 0 }) as u16)
                } else {
                    self.operand_f32(src, env)?
                };
                let x = sm.apply(flush(x, ftz));
                let x = match round {
                    Some(round) => apply_round(x, round),
                    None => x,
                };
                let x = saturate(x, sat);
                if dst_bits == 16 {
                    // The half lands in the low half, the rest cleared.
                    self.set_reg(dst, u32::from(f32_to_f16(x)));
                } else {
                    self.set_reg_f32(dst, x);
                }
            }
            Op::I2i {
                dst,
                src,
                sm,
                src_bytes,
                src_signed,
                dst_signed,
                sat,
                sel,
                cc,
            } => {
                let raw = self.operand(src, env)? >> (sel as u32 * 8);
                let mut v = if src_signed {
                    sign_extend(raw, src_bytes)
                } else {
                    truncate(raw, src_bytes)
                };
                if sm.neg {
                    v = (v as i32).wrapping_neg() as u32;
                }
                if sm.abs {
                    v = (v as i32).unsigned_abs();
                }
                if sat && !dst_signed {
                    v = (v as i32).max(0) as u32;
                }
                self.set_reg(dst, v);
                if cc {
                    self.set_flags(v);
                }
            }

            // ---- moves ----
            Op::Mov { dst, src } => {
                let v = self.operand(src, env)?;
                self.set_reg(dst, v);
            }
            Op::Mov32i { dst, imm } => self.set_reg(dst, imm),
            Op::Tmml { dst } => self.set_reg(dst, 0),
            Op::S2r { dst, sr } => {
                if SpecialRegs::PACKED.contains(&sr) {
                    return Err(fault(format!(
                        "shader: s2r of the packed special register {sr:#x}, whose field \
                         layout is not confirmed"
                    )));
                }
                // An unmodelled register reads zero.
                self.set_reg(dst, env.special.read(sr).unwrap_or(0));
            }
            Op::Psetp {
                p0,
                p1,
                a,
                b,
                c,
                op1,
                op2,
            } => {
                let first = combine(op1, self.holds(a), self.holds(b));
                let r = combine(op2, first, self.holds(c));
                self.set_pred(p0, r);
                self.set_pred(p1, !r);
            }
            Op::Pset {
                dst,
                a,
                b,
                c,
                op1,
                op2,
                float,
                cc,
            } => {
                let first = combine(op1, self.holds(a), self.holds(b));
                let v = set_result(combine(op2, first, self.holds(c)), float);
                self.set_reg(dst, v);
                if cc {
                    self.set_flags(v);
                }
            }
            Op::Csetp {
                p0,
                p1,
                test,
                src,
                op,
            } => {
                let flags = [self.zero, self.sign, self.carry, self.overflow];
                let passed = isa::flow_test(test, flags, &BoolLogic).ok_or_else(|| {
                    fault(format!(
                        "shader: csetp condition-code test {test} is not modelled"
                    ))
                })?;
                let src = self.holds(src);
                self.set_pred(p0, combine(op, passed, src));
                self.set_pred(p1, combine(op, !passed, src));
            }

            // ---- memory ----
            Op::Ldc {
                dst,
                bank,
                offset,
                idx,
                size,
            } => {
                let base = offset.wrapping_add(self.reg(idx) as i32);
                for i in 0..size.regs() {
                    let at = base.wrapping_add(i as i32 * 4);
                    let v = env.consts.read_const(bank, at as u16)?;
                    self.set_reg(dst.wrapping_add(i), v);
                }
            }
            Op::Ldg {
                dst,
                addr,
                offset,
                size,
            } => {
                let mem = env
                    .memory
                    .ok_or_else(|| Error::Gpu("shader: ldg with no global memory bound".into()))?;
                let base = (self.reg64(addr) as i64).wrapping_add(offset as i64) as u64;
                if let Some(raw) = narrow_load(size, |i| mem.read_u8(base + i as u64))? {
                    self.set_reg(dst, raw);
                } else {
                    for i in 0..size.regs() {
                        let v = mem.read_u32(base.wrapping_add(u64::from(i) * 4))?;
                        self.set_reg(dst.wrapping_add(i), v);
                    }
                }
            }
            Op::Stg {
                addr,
                offset,
                src,
                size,
            } => {
                let mem = env
                    .memory
                    .ok_or_else(|| Error::Gpu("shader: stg with no global memory bound".into()))?;
                let base = (self.reg64(addr) as i64).wrapping_add(offset as i64) as u64;
                let (bytes, len) = self.store_value(src, size);
                if len < 4 {
                    for (i, byte) in bytes[..len].iter().enumerate() {
                        mem.write_u8(base + i as u64, *byte)?;
                    }
                } else {
                    for i in 0..size.regs() {
                        let v = self.reg(src.wrapping_add(i));
                        mem.write_u32(base.wrapping_add(u64::from(i) * 4), v)?;
                    }
                }
            }
            Op::Ldl {
                dst,
                addr,
                offset,
                size,
            } => {
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let values = read_scratch(&self.local, base, size);
                for i in 0..size.regs() {
                    self.set_reg(dst.wrapping_add(i), values[i as usize]);
                }
            }
            Op::Stl {
                addr,
                offset,
                src,
                size,
            } => {
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let (bytes, len) = self.store_value(src, size);
                let cap = self.local_bytes;
                write_scratch(&mut self.local, cap, base, &bytes[..len]);
            }
            Op::Lds {
                dst,
                addr,
                offset,
                size,
            } => {
                let shared = env
                    .shared
                    .ok_or_else(|| Error::Gpu("shader: lds with no shared memory bound".into()))?;
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let values = read_scratch(&shared.borrow(), base, size);
                for i in 0..size.regs() {
                    self.set_reg(dst.wrapping_add(i), values[i as usize]);
                }
            }
            Op::Sts {
                addr,
                offset,
                src,
                size,
            } => {
                let shared = env
                    .shared
                    .ok_or_else(|| Error::Gpu("shader: sts with no shared memory bound".into()))?;
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let (bytes, len) = self.store_value(src, size);
                let mut block = shared.borrow_mut();
                let cap = block.len();
                write_scratch(&mut block, cap, base, &bytes[..len]);
            }
            Op::Atom {
                dst,
                addr,
                offset,
                src,
                op,
                ty,
                space,
            } => {
                self.run_atom(dst, addr, offset, src, op, ty, space, env)?;
            }

            Op::Unimplemented { raw } => {
                return Err(fault(format!(
                    "shader: unimplemented instruction {raw:#018x}"
                )))
            }
            // Handled by `execute`.
            Op::Exit
            | Op::Kil
            | Op::Nop
            | Op::Inert
            | Op::Bra { .. }
            | Op::Brx { .. }
            | Op::Ssy { .. }
            | Op::Pbk { .. }
            | Op::Pcnt { .. }
            | Op::Sync
            | Op::Brk
            | Op::Cont
            | Op::Bar { .. }
            | Op::Shfl { .. }
            | Op::Vote { .. }
            | Op::Texs { .. }
            | Op::Tlds { .. }
            | Op::Tex { .. }
            | Op::Txq { .. }
            | Op::Tld4 { .. }
            | Op::Tld4s { .. }
            | Op::Suld { .. }
            | Op::Sust { .. }
            | Op::Sured { .. } => unreachable!("control flow is dispatched in execute"),
        }
        Ok(())
    }
}
