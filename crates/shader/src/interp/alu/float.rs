//! Single- and half-precision float ops.

use super::*;

impl Invocation {
    #[inline(always)]
    pub(super) fn run_float(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
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
            _ => unreachable!("routed by run_alu"),
        }
        Ok(())
    }

    /// Lanes are computed in f32 and rounded at the merge; [`HMerge::F32`] never rounds.
    #[inline(always)]
    pub(super) fn run_half(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
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
            _ => unreachable!("routed by run_alu"),
        }
        Ok(())
    }
}
