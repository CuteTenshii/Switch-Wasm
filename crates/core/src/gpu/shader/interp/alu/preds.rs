//! Predicate logic and moving predicates and condition codes through registers.

use super::*;

impl Invocation {
    #[inline(always)]
    pub(super) fn run_predicate(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
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
            Op::P2r {
                dst,
                src,
                mask,
                byte,
                flags,
            } => {
                let mask = self.operand(mask, env)?;
                let offset = u32::from(byte) * 8;
                let mut bits = 0u32;
                for index in 0..if flags { 4 } else { 7u8 } {
                    if mask & (1 << index) == 0 {
                        continue;
                    }
                    let set = match (flags, index) {
                        (false, _) => self.holds(Pred {
                            reg: index,
                            negate: false,
                        }),
                        (true, 0) => self.zero,
                        (true, 1) => self.sign,
                        (true, 2) => self.carry,
                        (true, _) => self.overflow,
                    };
                    bits |= u32::from(set) << (u32::from(index) + offset);
                }
                let kept = self.reg(src) & !(mask << offset);
                self.set_reg(dst, kept | bits);
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
            Op::Cset {
                dst,
                test,
                src,
                op,
                bf,
                cc,
            } => {
                let flags = [self.zero, self.sign, self.carry, self.overflow];
                let passed = isa::flow_test(test, flags, &BoolLogic).ok_or_else(|| {
                    fault(format!(
                        "shader: cset condition-code test {test} is not modelled"
                    ))
                })?;
                let v = set_result(combine(op, passed, self.holds(src)), bf);
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
            _ => unreachable!("routed by run_alu"),
        }
        Ok(())
    }
}
