//! Conversions, moves and predicate logic.

use super::emitter::Emitter;
use super::Unsupported;
use crate::gpu::shader::isa::Op;

impl Emitter<'_> {
    pub(super) fn emit_conversion(&mut self, at: usize, op: Op) -> Result<(), Unsupported> {
        match op {
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
            _ => unreachable!("emit_alu sorts ops by group"),
        }
        Ok(())
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
