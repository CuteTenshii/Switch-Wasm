//! Conversions between integer and float formats.

use super::*;

impl Invocation {
    #[inline(always)]
    pub(super) fn run_convert(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
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
            _ => unreachable!("routed by run_alu"),
        }
        Ok(())
    }
}
