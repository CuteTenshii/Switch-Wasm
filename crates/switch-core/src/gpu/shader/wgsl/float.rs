//! Float and half-precision arithmetic and comparison.

use super::emitter::Emitter;
use super::Unsupported;
use crate::gpu::shader::isa::{FMod, HPrecision, MufuOp, Op};

impl Emitter<'_> {
    pub(super) fn emit_float(&mut self, op: Op) -> Result<(), Unsupported> {
        match op {
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
            _ => unreachable!("emit_alu sorts ops by group"),
        }
        Ok(())
    }
}
