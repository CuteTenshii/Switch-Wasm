//! Integer arithmetic, logic, shifts and bit fields.

use super::emitter::Emitter;
use super::Unsupported;
use crate::isa::{Half16, Iadd3Shift, LogicOp, LopTest, Op, Pred, ShfMax, VideoOperand, XmadC};

impl Emitter<'_> {
    pub(super) fn emit_integer(&mut self, op: Op) -> Result<(), Unsupported> {
        match op {
            Op::Iadd { .. } | Op::Iadd3 { .. } | Op::Iscadd { .. } | Op::Lea { .. } => {
                self.emit_add(op)
            }
            Op::Vmad { .. }
            | Op::Vmnmx { .. }
            | Op::Imnmx { .. }
            | Op::Imul { .. }
            | Op::Xmad { .. } => self.emit_multiply(op),
            Op::Vsetp { .. }
            | Op::Isetp { .. }
            | Op::Iset { .. }
            | Op::Icmp { .. }
            | Op::Sel { .. } => self.emit_int_compare(op),
            Op::P2r { .. } | Op::R2p { .. } => self.emit_predicate_move(op),
            Op::Lop { .. } | Op::Lop3 { .. } => self.emit_logic(op),
            Op::Bfi { .. }
            | Op::Shl { .. }
            | Op::Shr { .. }
            | Op::Shf { .. }
            | Op::Bfe { .. }
            | Op::Popc { .. }
            | Op::Flo { .. } => self.emit_bits(op),
            _ => unreachable!("emit_alu sorts ops by group"),
        }
        Ok(())
    }

    /// Additions, with their carries and overflow.
    fn emit_add(&mut self, op: Op) {
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
                let x = self.r(a);
                let x = self.ineg(aneg, x);
                let x = self.bind(&x);
                let y = self.operand(b);
                let y = self.ineg(bneg, y);
                let y = if cout && !cin { self.bind(&y) } else { y };
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
                    if !cin {
                        self.set_zero_sign(&total);
                        self.line(&format!(
                            "ccO = bitcast<i32>(({x} ^ {total}) & ({y} ^ {total})) < 0;"
                        ));
                    }
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
                x: extended,
                cc,
                halves,
                shift,
            } => {
                let x = half16(&self.r(a), halves[0]);
                let x = self.ineg(aneg, x);
                let x = self.bind(&x);
                let y = half16(&self.operand(b), halves[1]);
                let y = self.ineg(bneg, y);
                let z = half16(&self.operand(c), halves[2]);
                let z = self.ineg(cneg, z);
                if extended || cc || shift != Iadd3Shift::None {
                    let z = self.bind(&z);
                    let first = self.bind(&format!("{x} + ({y})"));
                    let summed = if extended {
                        self.uses_carry = true;
                        self.bind(&format!("{first} + select(0u, 1u, carry)"))
                    } else {
                        first.clone()
                    };
                    let lhs = match shift {
                        Iadd3Shift::None => summed,
                        Iadd3Shift::Right => self.bind(&format!(
                            "({summed} >> 16u) | select(0u, 0x10000u, {first} < {x})"
                        )),
                        Iadd3Shift::Left => self.bind(&format!("{summed} << 16u")),
                    };
                    let total = self.bind(&format!("{lhs} + {z}"));
                    self.set_r(dst, &total);
                    if cc {
                        self.set_zero_sign(&total);
                        self.line(&format!("carry = {total} < {lhs};"));
                        let overflow =
                            format!("bitcast<i32>(({lhs} ^ {total}) & ({z} ^ {total})) < 0");
                        self.line(&format!("ccO = ({first} < {x}) || ({overflow});"));
                    }
                } else {
                    self.set_r(dst, &format!("{x} + ({y}) + ({z})"));
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
                let x = self.r(a);
                let x = self.ineg(aneg, x);
                let y = self.operand(b);
                let y = self.ineg(bneg, y);
                let y = if cout { self.bind(&y) } else { y };
                let shift = u32::from(shift) & 31;
                let scaled = self.bind(&format!("({x}) << {shift}u"));
                let sum = self.bind(&format!("{scaled} + ({y})"));
                self.set_r(dst, &sum);
                if cout {
                    self.set_zero_sign(&sum);
                    self.line(&format!("carry = {sum} < {scaled};"));
                    self.line(&format!(
                        "ccO = bitcast<i32>(({scaled} ^ {sum}) & ({y} ^ {sum})) < 0;"
                    ));
                }
            }
            Op::Lea {
                dst,
                a,
                hi: None,
                b,
                neg,
                shift,
            } => {
                let x = self.r(a);
                let x = self.ineg(neg, x);
                let y = self.operand(b);
                let shift = u32::from(shift) & 31;
                self.set_r(dst, &format!("(({x}) << {shift}u) + ({y})"));
            }
            // WGSL has no u64, so the 64-bit negate and shift are done on the two words.
            Op::Lea {
                dst,
                a,
                hi: Some(hi),
                b,
                neg,
                shift,
            } => {
                let mut lo = self.r(a);
                let mut hi = self.r(hi);
                if neg {
                    let old_lo = self.bind(&lo);
                    lo = self.bind(&format!("0u - {old_lo}"));
                    hi = self.bind(&format!("~({hi}) + select(0u, 1u, {old_lo} == 0u)"));
                }
                let offset = match u32::from(shift) & 31 {
                    0 => hi,
                    shift => format!("(({lo} >> {}u) | (({hi}) << {shift}u))", 32 - shift),
                };
                let y = self.operand(b);
                self.set_r(dst, &format!("({offset}) + ({y})"));
            }
            _ => unreachable!("emit_integer sorts ops by family"),
        }
    }

    /// Video-operand arithmetic, minimum and maximum, and multiplication.
    fn emit_multiply(&mut self, op: Op) {
        match op {
            Op::Vmad { dst, a, b, c } => {
                let x = self.video(a);
                let y = self.video(b);
                let z = self.r(c);
                self.set_r(dst, &format!("(({x}) * ({y}) + {z})"));
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
            _ => unreachable!("emit_integer sorts ops by family"),
        }
    }

    /// Comparisons, and the selects and predicates they feed.
    fn emit_int_compare(&mut self, op: Op) {
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
                let x = self.video(a);
                let y = self.video(b);
                let taken = self.int_compare(cmp, &x, &y, signed);
                let taken = self.bind(&taken);
                let guard = self.holds(src);
                let guard = self.bind(&guard);
                let set = self.combine(bop, &taken, &guard);
                self.set_p(p0, &set);
                let clear = self.combine(bop, &format!("!{taken}"), &guard);
                self.set_p(p1, &clear);
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
                x: extended,
            } => {
                let x = self.r(a);
                let y = self.operand(b);
                let taken = if extended {
                    self.extended_compare(cmp, &x, &y, signed)
                } else {
                    self.int_compare(cmp, &x, &y, signed)
                };
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
                x: extended,
                cc,
            } => {
                let x = self.r(a);
                let y = self.operand(b);
                let taken = if extended {
                    self.extended_compare(cmp, &x, &y, signed)
                } else {
                    self.int_compare(cmp, &x, &y, signed)
                };
                let guard = self.holds(src);
                let taken = self.combine(bop, &taken, &guard);
                let value = self.set_result(&taken, bf);
                let value = self.bind(&value);
                self.set_r(dst, &value);
                if cc {
                    self.set_flags(&value);
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
                let selector = self.r(c);
                let taken = self.int_compare(cmp, &selector, "0u", signed);
                let x = self.r(a);
                let y = self.operand(b);
                self.set_r(dst, &format!("select({y}, {x}, {taken})"));
            }
            Op::Sel { dst, a, b, pred } => {
                let x = self.r(a);
                let y = self.operand(b);
                let taken = self.holds(pred);
                self.set_r(dst, &format!("select({y}, {x}, {taken})"));
            }
            _ => unreachable!("emit_integer sorts ops by family"),
        }
    }

    /// Moves between predicates or condition codes and a register.
    fn emit_predicate_move(&mut self, op: Op) {
        match op {
            Op::P2r {
                dst,
                src,
                mask,
                byte,
                flags,
            } => {
                let kept = self.r(src);
                let mask = self.operand(mask);
                let mask = self.bind(&mask);
                let offset = u32::from(byte) * 8;
                if flags {
                    self.uses_carry = true;
                    self.uses_flags = true;
                }
                let mut value = format!("({kept} & ~({mask} << {offset}u))");
                for index in 0..if flags { 4 } else { 7u8 } {
                    let set = if flags {
                        ["ccZ", "ccS", "carry", "ccO"][usize::from(index)].to_string()
                    } else {
                        self.holds(Pred {
                            reg: index,
                            negate: false,
                        })
                    };
                    let bit = 1u32 << (u32::from(index) + offset);
                    value.push_str(&format!(
                        " | select(0u, {bit}u, (({mask} & {}u) != 0u) && ({set}))",
                        1u32 << index
                    ));
                }
                self.set_r(dst, &value);
            }
            Op::R2p {
                src,
                mask,
                byte,
                flags,
            } => {
                let bits = self.r(src);
                let shift = u32::from(byte) * 8;
                let bits = self.bind(&format!("{bits} >> {shift}u"));
                let mask = self.operand(mask);
                let mask = self.bind(&mask);
                if flags {
                    self.uses_carry = true;
                    self.uses_flags = true;
                }
                for index in 0..if flags { 4 } else { 7u8 } {
                    let bit = 1u32 << index;
                    let value = format!("(({bits} & {bit}u) != 0u)");
                    self.line(&format!("if (({mask} & {bit}u) != 0u) {{"));
                    self.indent += 1;
                    if flags {
                        let flag = ["ccZ", "ccS", "carry", "ccO"][usize::from(index)];
                        self.line(&format!("{flag} = {value};"));
                    } else {
                        self.set_p(index, &value);
                    }
                    self.indent -= 1;
                    self.line("}");
                }
            }
            _ => unreachable!("emit_integer sorts ops by family"),
        }
    }

    fn emit_logic(&mut self, op: Op) {
        match op {
            Op::Lop {
                dst,
                a,
                ainv,
                b,
                binv,
                op: logic,
                pred,
                cc,
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
                    self.set_p(p, &lop_test(test, &value));
                }
                if cc {
                    self.set_flags(&value);
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
                self.need("lop3");
                let x = self.r(a);
                let y = self.operand(b);
                let z = self.operand(c);
                let value = self.bind(&format!("lop3({x}, {y}, {z}, {lut}u)"));
                self.set_r(dst, &value);
                if let Some((p, test)) = pred {
                    self.set_p(p, &lop_test(test, &value));
                }
            }
            _ => unreachable!("emit_integer sorts ops by family"),
        }
    }

    /// Shifts, bit fields and bit counts.
    fn emit_bits(&mut self, op: Op) {
        match op {
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
                brev,
            } => {
                let x = self.r(a);
                let x = if brev { format!("reverseBits({x})") } else { x };
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
                max,
            } => {
                self.need("shf");
                let low = self.r(lo);
                let high = self.r(hi);
                let count = self.operand(shift);
                let limit = max.limit();
                let count = if wrap {
                    format!("(({count}) & {}u)", limit - 1)
                } else {
                    format!("min({count}, {limit}u)")
                };
                let arithmetic = max == ShfMax::S64;
                self.set_r(
                    dst,
                    &format!("shf({low}, {high}, {count}, {left}, {arithmetic})"),
                );
            }
            Op::Bfe {
                dst,
                a,
                b,
                signed,
                brev,
                cc,
            } => {
                self.need("bfe");
                let x = self.r(a);
                let x = if brev { format!("reverseBits({x})") } else { x };
                let desc = self.operand(b);
                let desc = self.bind(&desc);
                let value = self.bind(&format!(
                    "bfe({x}, {desc} & 0xffu, ({desc} >> 8u) & 0xffu, {signed})"
                ));
                self.set_r(dst, &value);
                if cc {
                    self.set_flags(&value);
                }
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
            _ => unreachable!("emit_integer sorts ops by family"),
        }
    }
}

fn lop_test(test: LopTest, value: &str) -> String {
    match test {
        LopTest::False => "false".to_string(),
        LopTest::True => "true".to_string(),
        LopTest::Zero => format!("({value} == 0u)"),
        LopTest::NonZero => format!("({value} != 0u)"),
    }
}

/// A register-form `iadd3` source's half, zero-extended.
fn half16(value: &str, half: Half16) -> String {
    match half {
        Half16::All => value.to_string(),
        Half16::Lower => format!("(({value}) & 0xffffu)"),
        Half16::Upper => format!("(({value}) >> 16u)"),
    }
}

impl Emitter<'_> {
    /// [`VideoOperand::extract`] as WGSL.
    fn video(&mut self, v: VideoOperand) -> String {
        let raw = self.operand(v.src);
        if v.bits >= 32 {
            return raw;
        }
        let at = u32::from(v.sel) * u32::from(v.bits);
        if v.signed {
            format!(
                "bitcast<u32>(extractBits(bitcast<i32>({raw}), {at}u, {}u))",
                v.bits
            )
        } else {
            format!("extractBits({raw}, {at}u, {}u)", v.bits)
        }
    }
}
