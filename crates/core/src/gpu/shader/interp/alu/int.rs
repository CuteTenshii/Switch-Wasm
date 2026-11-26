//! Integer arithmetic and comparisons.

use super::*;

impl Invocation {
    #[inline(always)]
    pub(super) fn run_int_add(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
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
            Op::Vmad { dst, a, b, c } => {
                let x = a.extract(self.operand(a.src, env)?);
                let y = b.extract(self.operand(b.src, env)?);
                self.set_reg(dst, x.wrapping_mul(y).wrapping_add(self.reg(c)));
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
            _ => unreachable!("routed by run_alu"),
        }
        Ok(())
    }

    #[inline(always)]
    pub(super) fn run_int_compare(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
            Op::Vsetp {
                p0,
                p1,
                a,
                b,
                cmp,
                signed,
                bop,
                src,
            } => {
                let x = a.extract(self.operand(a.src, env)?);
                let y = b.extract(self.operand(b.src, env)?);
                let r = int_compare(cmp, x, y, signed);
                let s = self.holds(src);
                self.set_pred(p0, combine(bop, r, s));
                self.set_pred(p1, combine(bop, !r, s));
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
            Op::Sel { dst, a, b, pred } => {
                let v = if self.holds(pred) {
                    self.reg(a)
                } else {
                    self.operand(b, env)?
                };
                self.set_reg(dst, v);
            }
            _ => unreachable!("routed by run_alu"),
        }
        Ok(())
    }
}
