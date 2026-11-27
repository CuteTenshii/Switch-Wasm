//! Executing ALU and data-movement ops.

mod bits;
mod convert;
mod float;
mod int;
mod load_store;
mod preds;

use super::*;

impl Invocation {
    pub(super) fn run_alu(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
            Op::Ld { .. } | Op::St { .. } | Op::Ipa { .. } => self.run_attr(op),
            Op::Rro { .. }
            | Op::Fadd { .. }
            | Op::Fmul { .. }
            | Op::Ffma { .. }
            | Op::Fmnmx { .. }
            | Op::Mufu { .. }
            | Op::Fswzadd { .. }
            | Op::Fsetp { .. }
            | Op::Fset { .. } => self.run_float(op, env)?,
            Op::Hadd2 { .. }
            | Op::Hmul2 { .. }
            | Op::Hfma2 { .. }
            | Op::Hset2 { .. }
            | Op::Hsetp2 { .. } => self.run_half(op, env)?,
            Op::Iadd { .. }
            | Op::Iadd3 { .. }
            | Op::Iscadd { .. }
            | Op::Lea { .. }
            | Op::Vmad { .. }
            | Op::Imul { .. }
            | Op::Xmad { .. } => self.run_int_add(op, env)?,
            Op::Vsetp { .. }
            | Op::Vmnmx { .. }
            | Op::Imnmx { .. }
            | Op::Isetp { .. }
            | Op::Iset { .. }
            | Op::Icmp { .. }
            | Op::Fcmp { .. }
            | Op::Sel { .. } => self.run_int_compare(op, env)?,
            Op::Bfi { .. }
            | Op::Lop { .. }
            | Op::Lop3 { .. }
            | Op::Bfe { .. }
            | Op::Popc { .. }
            | Op::Flo { .. } => self.run_logic(op, env)?,
            Op::Shl { .. } | Op::Shr { .. } | Op::Shf { .. } => self.run_shift(op, env)?,
            Op::R2p { .. }
            | Op::P2r { .. }
            | Op::Psetp { .. }
            | Op::Pset { .. }
            | Op::Cset { .. }
            | Op::Csetp { .. } => self.run_predicate(op, env)?,
            Op::I2d { .. }
            | Op::D2f { .. }
            | Op::I2f { .. }
            | Op::F2i { .. }
            | Op::F2f { .. }
            | Op::I2i { .. } => self.run_convert(op, env)?,
            Op::Ldc { .. }
            | Op::Ldg { .. }
            | Op::Stg { .. }
            | Op::Ldl { .. }
            | Op::Stl { .. }
            | Op::Lds { .. }
            | Op::Sts { .. }
            | Op::Atom { .. } => self.run_memory(op, env)?,

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
            | Op::Tld { .. }
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
