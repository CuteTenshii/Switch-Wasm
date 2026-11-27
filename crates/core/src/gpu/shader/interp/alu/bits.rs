//! Bitwise logic, shifts and bitfields.

use super::*;

impl Invocation {
    #[inline(always)]
    pub(super) fn run_logic(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
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
            _ => unreachable!("routed by run_alu"),
        }
        Ok(())
    }

    #[inline(always)]
    pub(super) fn run_shift(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
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
            _ => unreachable!("routed by run_alu"),
        }
        Ok(())
    }
}
