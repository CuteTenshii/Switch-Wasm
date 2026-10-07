//! Translation of everything that is not control flow or texturing.

use super::emitter::Emitter;
use super::layout::generic_slot;
use super::Unsupported;
use crate::gpu::shader::isa::{
    FMod, HPrecision, LogicOp, LopTest, MemSize, MufuOp, Op, Operand, ShflMode, TexDim, XmadC,
};
use crate::gpu::texture::TextureSlot;

impl Emitter<'_> {
    /// Everything that is not control flow.
    pub(super) fn emit_alu(&mut self, at: usize, op: Op) -> Result<(), Unsupported> {
        match op {
            // ---- attribute space ----
            Op::Ld {
                dst,
                offset,
                idx,
                size,
            } => {
                self.loads.extend(generic_slot(offset));
                let base = self.attr_base(offset, idx);
                for i in 0..size.regs() {
                    let word = i as u32 * 4;
                    self.set_f(dst.wrapping_add(i), &format!("attrIn({base} + {word}u)"));
                }
            }
            Op::St {
                offset,
                idx,
                src,
                size,
            } => {
                self.stores.extend(generic_slot(offset));
                let base = self.attr_base(offset, idx);
                for i in 0..size.regs() {
                    let word = i as u32 * 4;
                    let value = self.f(src.wrapping_add(i));
                    self.line(&format!("attrOut({base} + {word}u, {value});"));
                }
            }
            Op::Ipa {
                dst,
                offset,
                mul,
                perspective,
                sat,
                centroid,
            } => {
                self.loads.extend(generic_slot(offset));
                if centroid {
                    self.centroid_loads.extend(generic_slot(offset));
                }
                let mut value = format!("attrIn({offset}u)");
                if perspective {
                    if let Some(mul) = mul {
                        let factor = self.f(mul);
                        value = format!("({value} * {factor})");
                    }
                }
                let value = self.saturate(sat, value);
                self.set_f(dst, &value);
            }

            // ---- float ----
            Op::Rro { dst, src, sm } => {
                let x = self.operand_f(src);
                let x = self.fmod(sm, x);
                self.set_f(dst, &x);
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
                let x = self.f(a);
                let x = self.flush(ftz, x);
                let x = self.fmod(am, x);
                let y = self.operand_f(b);
                let y = self.flush(ftz, y);
                let y = self.fmod(bm, y);
                let value = self.saturate(sat, format!("({x} + {y})"));
                self.set_f(dst, &value);
            }
            Op::Fmul {
                dst,
                a,
                b,
                bm,
                ftz,
                sat,
                scale,
            } => {
                // The pre-scale multiplies the first operand.
                let x = self.f(a);
                let x = self.flush(ftz, x);
                let factor = scale.factor();
                let x = if factor == 1.0 {
                    x
                } else {
                    format!("({x} * {factor:?})")
                };
                let y = self.operand_f(b);
                let y = self.flush(ftz, y);
                let y = self.fmod(bm, y);
                let value = self.saturate(sat, format!("({x} * {y})"));
                self.set_f(dst, &value);
            }
            Op::Ffma {
                dst,
                a,
                b,
                bneg,
                c,
                cneg,
                ftz,
                sat,
            } => {
                let x = self.f(a);
                let x = self.flush(ftz, x);
                let y = self.operand_f(b);
                let y = self.flush(ftz, y);
                let y = if bneg { format!("-({y})") } else { y };
                let z = self.operand_f(c);
                let z = self.flush(ftz, z);
                let z = if cneg { format!("-({z})") } else { z };
                let value = self.saturate(sat, format!("fma({x}, {y}, {z})"));
                self.set_f(dst, &value);
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
                let x = self.f(a);
                let x = self.flush(ftz, x);
                let x = self.fmod(am, x);
                let y = self.operand_f(b);
                let y = self.flush(ftz, y);
                let y = self.fmod(bm, y);
                // True picks the minimum. NaN handling matches the interpreter.
                let take_min = self.holds(pred);
                let value = format!("select(max({x}, {y}), min({x}, {y}), {take_min})");
                self.set_f(dst, &value);
            }
            Op::Mufu {
                dst,
                src,
                sm,
                op: mufu,
                sat,
            } => {
                let x = self.f(src);
                let x = self.fmod(sm, x);
                let value = match mufu {
                    MufuOp::Cos => format!("cos({x})"),
                    MufuOp::Sin => format!("sin({x})"),
                    MufuOp::Ex2 => format!("exp2({x})"),
                    MufuOp::Lg2 => format!("log2({x})"),
                    MufuOp::Rcp => format!("(1.0 / {x})"),
                    MufuOp::Rsq => format!("(1.0 / sqrt({x}))"),
                    MufuOp::Sqrt => format!("sqrt({x})"),
                };
                let value = self.saturate(sat, value);
                self.set_f(dst, &value);
            }
            // ---- half-precision ----
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
                let x = self.r(a);
                let x = self.half_source(x, am, asw, ftz);
                let y = self.operand(b);
                let y = self.half_source(y, bm, bsw, ftz);
                let lanes = self.half_saturate(sat, format!("({x} + {y})"));
                let value = self.half_merge(dst, &lanes, merge);
                self.set_r(dst, &value);
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
                let x = self.r(a);
                let x = self.half_source(x, am, asw, ftz);
                let y = self.operand(b);
                let y = self.half_source(y, bm, bsw, ftz);
                let lanes = if prec.zeroes_products(sat) {
                    let x = self.bind(&x);
                    let y = self.bind(&y);
                    let zeroed = self.half_zeroed(&x, &y);
                    format!("select({x} * {y}, vec2<f32>(0.0), {zeroed})")
                } else {
                    format!("({x} * {y})")
                };
                let lanes = self.half_saturate(sat, lanes);
                let value = self.half_merge(dst, &lanes, merge);
                self.set_r(dst, &value);
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
                let x = self.r(a);
                let x = self.half_source(x, FMod::NONE, asw, ftz);
                let y = self.operand(b);
                let y = self.half_source(
                    y,
                    FMod {
                        neg: bneg,
                        abs: false,
                    },
                    bsw,
                    ftz,
                );
                let z = self.operand(c);
                let z = self.half_source(
                    z,
                    FMod {
                        neg: cneg,
                        abs: false,
                    },
                    csw,
                    ftz,
                );
                let lanes = if prec.zeroes_products(sat) {
                    let x = self.bind(&x);
                    let y = self.bind(&y);
                    let z = self.bind(&z);
                    let zeroed = self.half_zeroed(&x, &y);
                    format!("select(fma({x}, {y}, {z}), {z}, {zeroed})")
                } else {
                    format!("fma({x}, {y}, {z})")
                };
                let lanes = self.half_saturate(sat, lanes);
                let value = self.half_merge(dst, &lanes, merge);
                self.set_r(dst, &value);
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
                let (low, high) = self.half_compare(a, am, asw, b, bm, bsw, cmp, bop, src, ftz);
                // Each lane fills its half: 1.0h with `.bf`, all ones without.
                let taken = if bf { "0x3c00u" } else { "0x0000ffffu" };
                self.set_r(
                    dst,
                    &format!("(select(0u, {taken}, {low}) | select(0u, {taken} << 16u, {high}))"),
                );
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
                let (low, high) = self.half_compare(a, am, asw, b, bm, bsw, cmp, bop, src, ftz);
                if and {
                    let both = self.bind(&format!("({low} && {high})"));
                    self.set_p(p0, &both);
                    self.set_p(p1, &format!("!{both}"));
                } else {
                    self.set_p(p0, &low);
                    self.set_p(p1, &high);
                }
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
                let x = self.f(a);
                let x = self.fmod(am, x);
                let y = self.operand_f(b);
                let y = self.fmod(bm, y);
                let taken = self.float_compare(cmp, &x, &y);
                let taken = self.bind(&taken);
                let guard = self.holds(src);
                let guard = self.bind(&guard);
                let set = self.combine(bop, &taken, &guard);
                self.set_p(p0, &set);
                let clear = self.combine(bop, &format!("!{taken}"), &guard);
                self.set_p(p1, &clear);
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
            } => {
                let x = self.f(a);
                let x = self.fmod(am, x);
                let y = self.operand_f(b);
                let y = self.fmod(bm, y);
                let taken = self.float_compare(cmp, &x, &y);
                let guard = self.holds(src);
                let taken = self.combine(bop, &taken, &guard);
                let value = self.set_result(&taken, bf);
                self.set_r(dst, &value);
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
                let x = self.r(a);
                let x = self.ineg(aneg, x);
                let x = self.bind(&x);
                let y = self.operand(b);
                let y = self.ineg(bneg, y);
                // Two adds: the carry is whether either wrapped.
                let sum = self.bind(&format!("{x} + ({y})"));
                let carry_in = if cin {
                    self.uses_carry = true;
                    "select(0u, 1u, carry)".to_string()
                } else {
                    "0u".to_string()
                };
                let total = self.bind(&format!("{sum} + {carry_in}"));
                self.set_r(dst, &total);
                if cout {
                    self.uses_carry = true;
                    self.line(&format!("carry = ({sum} < {x}) || ({total} < {sum});"));
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
            } => {
                let x = self.r(a);
                let x = self.ineg(aneg, x);
                let y = self.operand(b);
                let y = self.ineg(bneg, y);
                let z = self.operand(c);
                let z = self.ineg(cneg, z);
                self.set_r(dst, &format!("{x} + ({y}) + ({z})"));
            }
            Op::Iscadd {
                dst,
                a,
                aneg,
                b,
                bneg,
                shift,
            } => {
                let x = self.r(a);
                let x = self.ineg(aneg, x);
                let y = self.operand(b);
                let y = self.ineg(bneg, y);
                let shift = u32::from(shift) & 31;
                self.set_r(dst, &format!("(({x}) << {shift}u) + ({y})"));
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
                let pick = |x: &str, y: &str, max: bool, signed: bool| {
                    let op = if max { "max" } else { "min" };
                    if signed {
                        format!("bitcast<u32>({op}(bitcast<i32>({x}), bitcast<i32>({y})))")
                    } else {
                        format!("{op}({x}, {y})")
                    }
                };
                let (x, y, z) = (self.r(a), self.r(b), self.r(c));
                let first = self.bind(&pick(&x, &y, max, signed));
                self.set_r(dst, &pick(&first, &z, then_max, then_signed));
            }
            Op::Imnmx {
                dst,
                a,
                b,
                pred,
                signed,
            } => {
                let x = self.r(a);
                let y = self.operand(b);
                let take_min = self.holds(pred);
                let value = if signed {
                    format!(
                        "bitcast<u32>(select(max(bitcast<i32>({x}), bitcast<i32>({y})), \
                         min(bitcast<i32>({x}), bitcast<i32>({y})), {take_min}))"
                    )
                } else {
                    format!("select(max({x}, {y}), min({x}, {y}), {take_min})")
                };
                self.set_r(dst, &value);
            }
            Op::Imul {
                dst,
                a,
                b,
                signed,
                hi,
            } => {
                let x = self.r(a);
                let y = self.operand(b);
                let value = match (hi, signed) {
                    (false, _) => format!("{x} * ({y})"),
                    (true, true) => {
                        self.need("mulhi_s");
                        format!("mulhi_s({x}, {y})")
                    }
                    (true, false) => {
                        self.need("mulhi_u");
                        format!("mulhi_u({x}, {y})")
                    }
                };
                self.set_r(dst, &value);
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
                let x = self.r(a);
                let x = self.half(&x, ah, asigned);
                let raw_b = self.operand(b);
                let raw_b = self.bind(&raw_b);
                let y = self.half(&raw_b, bh, bsigned);
                let product = self.bind(&format!("({x}) * ({y})"));
                let product = if psl {
                    self.bind(&format!("{product} << 16u"))
                } else {
                    product
                };
                let raw_c = self.operand(c);
                let z = match cmode {
                    XmadC::Full => raw_c,
                    XmadC::Lo => format!("(({raw_c}) & 0xffffu)"),
                    XmadC::Hi => format!("(({raw_c}) >> 16u)"),
                    XmadC::Bcc => format!("(({raw_b} << 16u) + ({raw_c}))"),
                };
                let sum = self.bind(&format!("{product} + ({z})"));
                // `.mrg` replaces the high half with `b`'s low half.
                let value = if mrg {
                    format!("({sum} & 0xffffu) | ({raw_b} << 16u)")
                } else {
                    sum
                };
                self.set_r(dst, &value);
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
            } => {
                let x = self.r(a);
                let y = self.operand(b);
                let taken = self.int_compare(cmp, &x, &y, signed);
                let taken = self.bind(&taken);
                let guard = self.holds(src);
                let guard = self.bind(&guard);
                let set = self.combine(bop, &taken, &guard);
                self.set_p(p0, &set);
                let clear = self.combine(bop, &format!("!{taken}"), &guard);
                self.set_p(p1, &clear);
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
            } => {
                let x = self.r(a);
                let y = self.operand(b);
                let taken = self.int_compare(cmp, &x, &y, signed);
                let guard = self.holds(src);
                let taken = self.combine(bop, &taken, &guard);
                let value = self.set_result(&taken, bf);
                self.set_r(dst, &value);
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
                let selector = self.r(c);
                let taken = self.int_compare(cmp, &selector, "0u", signed);
                let x = self.r(a);
                let y = self.operand(b);
                self.set_r(dst, &format!("select({y}, {x}, {taken})"));
            }
            Op::Bfi {
                dst,
                insert,
                src,
                base,
            } => {
                self.need("bfi");
                let insert = self.r(insert);
                let src = self.operand(src);
                let base = self.operand(base);
                self.set_r(dst, &format!("bfi({insert}, {src}, {base})"));
            }
            Op::R2p { src, mask, byte } => {
                let bits = self.r(src);
                let shift = u32::from(byte) * 8;
                let bits = self.bind(&format!("{bits} >> {shift}u"));
                let mask = self.operand(mask);
                let mask = self.bind(&mask);
                for index in 0..7u8 {
                    let bit = 1u32 << index;
                    let value = format!("(({bits} & {bit}u) != 0u)");
                    self.line(&format!("if (({mask} & {bit}u) != 0u) {{"));
                    self.indent += 1;
                    self.set_p(index, &value);
                    self.indent -= 1;
                    self.line("}");
                }
            }
            Op::Lop {
                dst,
                a,
                ainv,
                b,
                binv,
                op: logic,
                pred,
            } => {
                let x = self.r(a);
                let x = self.inv(ainv, x);
                let y = self.operand(b);
                let y = self.inv(binv, y);
                let value = match logic {
                    LogicOp::And => format!("({x}) & ({y})"),
                    LogicOp::Or => format!("({x}) | ({y})"),
                    LogicOp::Xor => format!("({x}) ^ ({y})"),
                    LogicOp::PassB => y,
                };
                let value = self.bind(&value);
                self.set_r(dst, &value);
                if let Some((p, test)) = pred {
                    let bit = match test {
                        LopTest::True => "true".to_string(),
                        LopTest::Zero => format!("({value} == 0u)"),
                        LopTest::NonZero => format!("({value} != 0u)"),
                    };
                    self.set_p(p, &bit);
                }
            }
            Op::Lop3 { dst, a, b, c, lut } => {
                self.need("lop3");
                let x = self.r(a);
                let y = self.operand(b);
                let z = self.operand(c);
                self.set_r(dst, &format!("lop3({x}, {y}, {z}, {lut}u)"));
            }
            Op::Shl { dst, a, b, wrap } => {
                self.need("shl32");
                let x = self.r(a);
                let n = self.shift_count(b, wrap);
                self.set_r(dst, &format!("shl32({x}, {n})"));
            }
            Op::Shr {
                dst,
                a,
                b,
                signed,
                wrap,
            } => {
                let x = self.r(a);
                let n = self.shift_count(b, wrap);
                if signed {
                    self.need("sar32");
                    self.set_r(dst, &format!("sar32({x}, {n})"));
                } else {
                    self.need("shr32");
                    self.set_r(dst, &format!("shr32({x}, {n})"));
                }
            }
            Op::Shf {
                dst,
                lo,
                shift,
                hi,
                left,
                wrap,
                hi_out,
            } => {
                self.need("shf");
                let low = self.r(lo);
                let high = self.r(hi);
                let count = self.operand(shift);
                let count = if wrap {
                    format!("(({count}) & 63u)")
                } else {
                    count
                };
                self.set_r(
                    dst,
                    &format!("shf({low}, {high}, {count}, {left}, {hi_out})"),
                );
            }
            Op::Bfe { dst, a, b, signed } => {
                self.need("bfe");
                let x = self.r(a);
                let desc = self.operand(b);
                let desc = self.bind(&desc);
                self.set_r(
                    dst,
                    &format!("bfe({x}, {desc} & 0xffu, ({desc} >> 8u) & 0xffu, {signed})"),
                );
            }
            Op::Popc { dst, b, inv } => {
                let value = self.operand(b);
                let value = self.inv(inv, value);
                self.set_r(dst, &format!("countOneBits({value})"));
            }
            Op::Flo {
                dst,
                b,
                signed,
                shift,
                inv,
            } => {
                self.need("flo");
                let value = self.operand(b);
                let value = self.inv(inv, value);
                self.set_r(dst, &format!("flo({value}, {signed}, {shift})"));
            }
            Op::Sel { dst, a, b, pred } => {
                let x = self.r(a);
                let y = self.operand(b);
                let taken = self.holds(pred);
                self.set_r(dst, &format!("select({y}, {x}, {taken})"));
            }

            // ---- conversions ----
            Op::I2f {
                dst,
                src,
                sm,
                src_bytes,
                src_signed,
                sel,
            } => {
                let raw = self.operand(src);
                let raw = self.narrow(&raw, sel, src_bytes, src_signed);
                let value = if src_signed {
                    format!("f32(bitcast<i32>({raw}))")
                } else {
                    format!("f32({raw})")
                };
                let value = self.fmod(sm, value);
                self.set_f(dst, &value);
            }
            Op::F2i {
                dst,
                src,
                sm,
                dst_bytes,
                dst_signed,
                round,
                ftz,
            } => {
                let x = self.operand_f(src);
                let x = self.flush(ftz, x);
                let x = self.fmod(sm, x);
                let x = self.round(round, x);
                let value = if dst_signed {
                    self.need("f2i_s");
                    format!("f2i_s({x}, {dst_bytes}u)")
                } else {
                    self.need("f2i_u");
                    format!("f2i_u({x}, {dst_bytes}u)")
                };
                self.set_r(dst, &value);
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
                    let raw = self.operand(src);
                    let lane = if hi { "y" } else { "x" };
                    format!("unpack2x16float({raw}).{lane}")
                } else {
                    self.operand_f(src)
                };
                let x = self.flush(ftz, x);
                let x = self.fmod(sm, x);
                let x = match round {
                    Some(round) => self.round(round, x),
                    None => x,
                };
                let value = self.saturate(sat, x);
                if dst_bits == 16 {
                    // Rounds as `f32_to_f16` does.
                    let packed = format!("pack2x16float(vec2<f32>({value}, 0.0))");
                    self.set_r(dst, &packed);
                } else {
                    self.set_f(dst, &value);
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
                let raw = self.operand(src);
                let value = self.narrow(&raw, sel, src_bytes, src_signed);
                let value = if sm.neg {
                    format!("(0u - ({value}))")
                } else {
                    value
                };
                let value = if sm.abs {
                    let bound = self.bind(&value);
                    format!("select({bound}, 0u - {bound}, bitcast<i32>({bound}) < 0)")
                } else {
                    value
                };
                let value = if sat && !dst_signed {
                    let bound = self.bind(&value);
                    format!("select({bound}, 0u, bitcast<i32>({bound}) < 0)")
                } else {
                    value
                };
                // Bound first: the flags read it and `dst` may be `RZ`.
                let value = self.bind(&value);
                self.set_r(dst, &value);
                if cc {
                    self.uses_carry = true;
                    self.uses_flags = true;
                    self.line(&format!("ccZ = ({value} == 0u);"));
                    self.line(&format!("ccS = (bitcast<i32>({value}) < 0);"));
                    self.line("carry = false;");
                    self.line("ccO = false;");
                }
            }

            // ---- moves ----
            Op::Mov { dst, src } => {
                let value = self.operand(src);
                self.set_r(dst, &value);
            }
            Op::Mov32i { dst, imm } => self.set_r(dst, &format!("{imm}u")),
            Op::S2r { dst, .. } => {
                // Lane and thread identities are zero, as in the interpreter.
                self.set_r(dst, "0u");
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
                let x = self.holds(a);
                let y = self.holds(b);
                let first = self.combine(op1, &x, &y);
                let z = self.holds(c);
                let value = self.combine(op2, &first, &z);
                let value = self.bind(&value);
                self.set_p(p0, &value);
                self.set_p(p1, &format!("!{value}"));
            }
            Op::Csetp {
                p0,
                p1,
                test,
                src,
                op: bop,
            } => {
                self.uses_carry = true;
                self.uses_flags = true;
                let flags = ["ccZ", "ccS", "carry", "ccO"].map(str::to_string);
                let Some(passed) = super::isa::flow_test(test, flags, &TextLogic) else {
                    return Err(Unsupported::Op { at, op });
                };
                let passed = self.bind(&passed);
                let source = self.holds(src);
                let a = self.combine(bop, &passed, &source);
                let b = self.combine(bop, &format!("!{passed}"), &source);
                self.set_p(p0, &a);
                self.set_p(p1, &b);
            }

            // ---- memory ----
            Op::Ldc {
                dst,
                bank,
                offset,
                idx,
                size,
            } => {
                self.banks.insert(bank);
                let index = self.r(idx);
                let base = self.bind(&format!("{}u + {index}", offset as u32));
                for i in 0..size.regs() {
                    let word = i as u32 * 4;
                    self.set_r(
                        dst.wrapping_add(i),
                        &format!("cbRead({bank}u, ({base} + {word}u) & 0xffffu)"),
                    );
                }
            }

            // ---- texture ----
            // A `texs` array keeps its layer in the third coordinate register.
            Op::Texs {
                coords,
                dref,
                handle,
                dim,
                ..
            } => {
                let layer = (dim == TexDim::T2dArray).then_some(coords[2]);
                let slot = TextureSlot::Bound(handle);
                self.sample_texture(at, slot, dim, dref, coords, layer)?;
            }
            // `tex` keeps the layer before the coordinates; `.LL`/`.LB` sample the one level.
            Op::Tex {
                coords,
                layer,
                dref,
                offset,
                handle,
                handle_reg,
                dim,
                ..
            } => {
                let slot = match handle_reg {
                    None => TextureSlot::Bound(handle),
                    Some(reg) => self
                        .bindless_slot(at, reg)
                        .ok_or(Unsupported::UntracedHandle { at })?,
                };
                match offset {
                    None => self.sample_texture(at, slot, dim, dref, coords, layer)?,
                    Some(reg) => {
                        self.sample_offset(at, slot, dim, dref, coords, layer, reg, op)?;
                    }
                }
            }
            Op::Txq { lod, handle, .. } => {
                self.query_texture(at, TextureSlot::Bound(handle), lod);
            }
            // WGSL needs a constant gather offset.
            Op::Tld4 {
                coords,
                layer,
                offset: None,
                handle,
                dim,
                component,
                ..
            } => self.gather_texture(
                at,
                TextureSlot::Bound(handle),
                dim,
                coords,
                layer,
                component,
            )?,

            // `shfl` maps onto `quadSwapX`/`Y`/`Diagonal`, mirroring `interp::shuffle_source`.
            Op::Shfl {
                dst,
                pred,
                src,
                index,
                mask,
                mode,
            } => {
                self.quad.get_or_insert(at);
                self.quad_swap.get_or_insert(at);
                let value = self.r(src);
                let here = self.bind(&value);
                let x = self.bind(&format!("quadSwapX({here})"));
                let y = self.bind(&format!("quadSwapY({here})"));
                let d = self.bind(&format!("quadSwapDiagonal({here})"));
                let lane = self.bind("i32(quadLane())");
                let index = self.operand(index);
                let index = self.bind(&format!("i32({index})"));
                let mask = self.operand(mask);
                let mask = self.bind(&format!("i32({mask})"));
                let clamp = self.bind(&format!("({mask} & 31)"));
                let segment = self.bind(&format!("(({mask} >> 8) & 31)"));
                let floor = self.bind(&format!("({lane} & {segment})"));
                let ceiling = self.bind(&format!("({floor} | ({clamp} & ~{segment}))"));
                let from = match mode {
                    ShflMode::Idx => format!("(({index} & ~{segment}) | {floor})"),
                    ShflMode::Up => format!("({lane} - {index})"),
                    ShflMode::Down => format!("({lane} + {index})"),
                    ShflMode::Bfly => format!("({lane} ^ {index})"),
                };
                let from = self.bind(&from);
                // `up` is the one mode whose bound holds from below.
                let within = match mode {
                    ShflMode::Up => format!("({from} >= {ceiling})"),
                    _ => format!("({from} <= {ceiling})"),
                };
                let ok = self.bind(&format!("({within} && {from} >= 0)"));
                let sel = self.bind(&format!("u32(({from} ^ {lane}) & 3)"));
                let peer = format!(
                    "select(select(select({here}, {x}, {sel} == 1u), {y}, {sel} == 2u), {d}, {sel} == 3u)"
                );
                // Out-of-quad lanes keep their own value.
                let reachable = format!("({ok} && {from} < 4)");
                self.set_r(dst, &format!("select({here}, {peer}, {reachable})"));
                self.set_p(pred, &ok);
            }

            // `fswzadd` needs only this lane's index, not another lane's value.
            Op::Fswzadd {
                dst,
                a,
                b,
                swizzle,
                ftz,
            } => {
                self.quad.get_or_insert(at);
                let x = self.r(a);
                let x = self.flush(ftz, format!("bitcast<f32>({x})"));
                let x = self.bind(&x);
                let y = self.r(b);
                let y = self.flush(ftz, format!("bitcast<f32>({y})"));
                let y = self.bind(&y);
                let code = self.bind(&format!(
                    "((({swizzle}u) >> ((quadLane() & 3u) * 2u)) & 3u)"
                ));
                // `FSWZ_SIGNS` in `super::interp`, arm for arm.
                let ka = self.bind(&format!(
                    "select(select(-1.0, 1.0, {code} == 1u), 0.0, {code} == 3u)"
                ));
                let kb = self.bind(&format!("select(-1.0, 1.0, {code} == 2u)"));
                self.set_f(dst, &format!("{ka} * {x} + {kb} * {y}"));
            }

            Op::Nop | Op::Inert => {}

            // `ldg` from a constant bank descriptor binds a buffer; other global
            // and shared memory, and barriers, are unsupported.
            Op::Ldg {
                dst,
                addr,
                offset,
                size,
            } => {
                let Some((bank, at, index)) = self.global_base(at, addr) else {
                    return Err(Unsupported::Op { at, op });
                };
                let slot = match self.globals.iter().position(|g| *g == (bank, at)) {
                    Some(slot) => slot,
                    None => {
                        self.globals.push((bank, at));
                        self.globals.len() - 1
                    }
                };
                let index = self.r(index);
                let base = self.bind(&format!("({index} + {offset}u)"));
                for word in 0..size.regs() {
                    let byte = u32::from(word) * 4;
                    self.set_r(
                        dst.wrapping_add(word),
                        &format!("gRead({slot}u, {base} + {byte}u)"),
                    );
                }
            }

            Op::Ldl {
                dst,
                addr,
                offset,
                size,
            } => {
                self.helpers.insert("local");
                let base = self.local_address(addr, offset);
                let value = |word: u32| format!("localWord({base} + {}u)", word * 4);
                match size {
                    MemSize::U8 => self.set_r(dst, &format!("localByte({base})")),
                    MemSize::S8 => self.set_r(
                        dst,
                        &format!(
                            "bitcast<u32>(extractBits(bitcast<i32>(localByte({base})), 0u, 8u))"
                        ),
                    ),
                    MemSize::U16 | MemSize::S16 => {
                        let half = format!("(localByte({base}) | (localByte({base} + 1u) << 8u))");
                        let half = if size == MemSize::S16 {
                            format!("bitcast<u32>(extractBits(bitcast<i32>({half}), 0u, 16u))")
                        } else {
                            half
                        };
                        self.set_r(dst, &half);
                    }
                    _ => {
                        for word in 0..u32::from(size.regs()) {
                            self.set_r(dst.wrapping_add(word as u8), &value(word));
                        }
                    }
                }
            }
            Op::Stl {
                addr,
                offset,
                src,
                size,
            } => {
                self.helpers.insert("local");
                let base = self.local_address(addr, offset);
                let len = size.bytes();
                let words: Vec<String> = (0..size.regs())
                    .map(|i| self.r(src.wrapping_add(i)))
                    .collect();
                // Out of range drops the whole store, as the interpreter does.
                self.line(&format!("if ({base} + {len}u <= 1024u) {{"));
                self.indent += 1;
                for byte in 0..len {
                    let word = &words[(byte / 4) as usize];
                    let shift = (byte % 4) * 8;
                    self.line(&format!(
                        "setLocalByte({base} + {byte}u, {word} >> {shift}u);"
                    ));
                }
                self.indent -= 1;
                self.line("}");
            }
            Op::Stg { .. }
            | Op::Lds { .. }
            | Op::Sts { .. }
            | Op::Atom { .. }
            | Op::Bar { .. }
            // A ballot needs the warp, which is only a quad here.
            | Op::Vote { .. }
            | Op::Suld { .. }
            | Op::Sust { .. }
            | Op::Unimplemented { .. } => return Err(Unsupported::Op { at, op }),

            // Handled by `emit_terminator` and `emit_instruction`.
            Op::Bra { .. }
            | Op::Brx { .. }
            | Op::Ssy { .. }
            | Op::Pbk { .. }
            | Op::Pcnt { .. }
            | Op::Sync
            | Op::Brk
            | Op::Cont
            | Op::Exit
            | Op::Kil => unreachable!("control flow is emitted by emit_instruction"),
            // `tex.aoffi` with a non-constant offset is the rasterizer's.
            Op::Tld4 { .. } => return Err(Unsupported::Op { at, op }),
        }
        Ok(())
    }

    /// `a[offset + Rn]`'s byte address, wrapping at 16 bits.
    fn attr_base(&mut self, offset: u16, idx: u8) -> String {
        let index = self.r(idx);
        self.bind(&format!("({offset}u + ({index} & 0xffffu)) & 0xffffu"))
    }

    /// One 16-bit half of a register, as `xmad` reads it.
    fn half(&mut self, value: &str, high: bool, signed: bool) -> String {
        let half = if high {
            format!("(({value}) >> 16u)")
        } else {
            format!("(({value}) & 0xffffu)")
        };
        if signed {
            self.need("sext");
            format!("sext({half}, 2u)")
        } else {
            half
        }
    }

    /// A shift instruction's count, masked when the encoding says to wrap.
    fn shift_count(&mut self, operand: Operand, wrap: bool) -> String {
        let count = self.operand(operand);
        if wrap {
            format!("(({count}) & 31u)")
        } else {
            count
        }
    }

    /// A conversion's source byte lane, narrowed and extended back to 32 bits.
    fn narrow(&mut self, raw: &str, sel: u8, bytes: u8, signed: bool) -> String {
        let shift = u32::from(sel) * 8;
        let shifted = if shift == 0 {
            raw.to_string()
        } else {
            format!("(({raw}) >> {shift}u)")
        };
        if signed {
            self.need("sext");
            format!("sext({shifted}, {bytes}u)")
        } else {
            self.need("truncw");
            format!("truncw({shifted}, {bytes}u)")
        }
    }
}

/// [`super::isa::flow_test`] as WGSL.
struct TextLogic;

impl super::isa::FlowLogic<String> for TextLogic {
    fn constant(&self, value: bool) -> String {
        value.to_string()
    }
    fn not(&self, a: String) -> String {
        format!("!({a})")
    }
    fn and(&self, a: String, b: String) -> String {
        format!("({a} && {b})")
    }
    fn or(&self, a: String, b: String) -> String {
        format!("({a} || {b})")
    }
    fn xor(&self, a: String, b: String) -> String {
        format!("({a} != {b})")
    }
}
