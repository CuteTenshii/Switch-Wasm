//! A decoded [`Program`] lowered for execution: branch targets resolved to
//! indices, `texs` writes indexed, and bound constant-bank operands folded
//! into immediates.

use super::interp::{texs_writes_for, ConstantSource};
use super::isa::{Op, Operand, Pred};
use super::{Program, TexsWrites};

/// No resolved target: not a branch, or the target was never decoded (an
/// error raised when the branch is taken).
pub const NO_TARGET: u32 = u32::MAX;

pub struct Compiled {
    /// The operations, with foldable constants resolved. Kept apart from the
    /// predicates so each `Op` stays 32 bytes, two per cache line.
    ops: Vec<Op>,
    preds: Vec<Pred>,
    /// Branch targets as indices into `insns`, or [`NO_TARGET`].
    targets: Vec<u32>,
    /// Source byte offsets, for errors and `brx`.
    offsets: Vec<u32>,
    texs_writes: Vec<TexsWrites>,
    interpolated_slots: Vec<usize>,
    /// Each `brx`'s possible targets, keyed by the `brx` index.
    indirect: std::collections::HashMap<usize, Vec<u32>>,
    header: Option<super::ProgramHeader>,
}

impl Compiled {
    /// Lower `program` without folding constants.
    pub fn new(program: &Program) -> Compiled {
        Compiled::lower(program, None)
    }

    /// Lower `program`, folding every constant the draw's bound banks supply.
    /// Constant buffers cannot change during a draw; unbound banks and
    /// out-of-range offsets stay unfolded so the error surfaces at runtime.
    pub fn with_constants(program: &Program, consts: &dyn ConstantSource) -> Compiled {
        Compiled::lower(program, Some(consts))
    }

    fn lower(program: &Program, consts: Option<&dyn ConstantSource>) -> Compiled {
        let ops: Vec<Op> = program
            .insns
            .iter()
            .map(|insn| match consts {
                Some(consts) => fold(insn.op, consts),
                None => insn.op,
            })
            .collect();
        let preds: Vec<Pred> = program.insns.iter().map(|insn| insn.pred).collect();
        let targets = program
            .insns
            .iter()
            .map(|insn| match branch_target(insn.op) {
                Some(target) => program
                    .index_of(target)
                    .map(|i| i as u32)
                    .unwrap_or(NO_TARGET),
                None => NO_TARGET,
            })
            .collect();
        let mut compiled = Compiled {
            ops,
            preds,
            targets,
            offsets: program.offsets.clone(),
            texs_writes: Vec::new(),
            interpolated_slots: Vec::new(),
            indirect: std::collections::HashMap::new(),
            header: program.header,
        };
        compiled.texs_writes = texs_writes_for(&compiled.ops);
        compiled.interpolated_slots = super::interpolated_slots(&compiled.ops);
        compiled.indirect = program
            .indirect
            .iter()
            .filter_map(|(at, targets)| {
                let at = program.index_of(*at)?;
                let targets = targets
                    .iter()
                    .filter_map(|t| program.index_of(*t).map(|i| i as u32))
                    .collect();
                Some((at, targets))
            })
            .collect();
        compiled
    }

    pub fn header(&self) -> Option<super::ProgramHeader> {
        self.header
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    #[inline]
    pub fn op(&self, index: usize) -> Op {
        self.ops[index]
    }

    #[inline]
    pub fn pred(&self, index: usize) -> Pred {
        self.preds[index]
    }

    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Always as long as [`Compiled::ops`].
    pub fn preds(&self) -> &[Pred] {
        &self.preds
    }

    /// The resolved target of the branch at `index`, or [`NO_TARGET`].
    #[inline]
    pub fn target(&self, index: usize) -> u32 {
        self.targets.get(index).copied().unwrap_or(NO_TARGET)
    }

    /// The shader-binary byte offset of instruction `index`, for errors.
    pub fn offset(&self, index: usize) -> u32 {
        self.offsets.get(index).copied().unwrap_or(0)
    }

    /// The index of the instruction at `byte_offset`; only `brx` needs it.
    pub fn index_of(&self, byte_offset: u32) -> Option<usize> {
        self.offsets.binary_search(&byte_offset).ok()
    }

    /// The targets of the `brx` at `index`, as indices.
    pub fn indirect_targets(&self, index: usize) -> Option<&[u32]> {
        self.indirect.get(&index).map(|t| t.as_slice())
    }

    /// See [`super::interpolated_slots`].
    pub fn interpolated_slots(&self) -> &[usize] {
        &self.interpolated_slots
    }

    /// See [`Program::texs_writes`].
    pub fn texs_writes(&self, index: usize) -> &[(u8, super::isa::TexsStore, usize)] {
        self.texs_writes
            .iter()
            .find(|t| t.at == index)
            .map(|t| t.writes.as_slice())
            .unwrap_or(&[])
    }
}

/// The static byte target of a branch; `brx` targets are registers, so it is absent.
fn branch_target(op: Op) -> Option<u32> {
    match op {
        Op::Bra { target } | Op::Ssy { target } | Op::Pbk { target } | Op::Pcnt { target } => {
            Some(target)
        }
        _ => None,
    }
}

/// Replace constant-bank operands this draw already knows with their values.
/// A missed variant only loses a fold.
fn fold(op: Op, consts: &dyn ConstantSource) -> Op {
    fn value(operand: Operand, consts: &dyn ConstantSource) -> Operand {
        match operand {
            Operand::Const { bank, offset } => match consts.read_const(bank, offset) {
                Ok(value) => Operand::Imm(value),
                Err(_) => operand,
            },
            other => other,
        }
    }
    let f = |operand| value(operand, consts);
    match op {
        // ---- float ----
        Op::Fadd {
            dst,
            a,
            am,
            b,
            bm,
            ftz,
            sat,
        } => Op::Fadd {
            dst,
            a,
            am,
            b: f(b),
            bm,
            ftz,
            sat,
        },
        Op::Fmul {
            dst,
            a,
            b,
            bm,
            ftz,
            sat,
            scale,
        } => Op::Fmul {
            dst,
            a,
            b: f(b),
            bm,
            ftz,
            sat,
            scale,
        },
        Op::Ffma {
            dst,
            a,
            b,
            bneg,
            c,
            cneg,
            ftz,
            sat,
        } => Op::Ffma {
            dst,
            a,
            b: f(b),
            bneg,
            c: f(c),
            cneg,
            ftz,
            sat,
        },
        Op::Fmnmx {
            dst,
            a,
            am,
            b,
            bm,
            pred,
            ftz,
        } => Op::Fmnmx {
            dst,
            a,
            am,
            b: f(b),
            bm,
            pred,
            ftz,
        },
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
        } => Op::Fsetp {
            p0,
            p1,
            a,
            am,
            b: f(b),
            bm,
            cmp,
            bop,
            src,
        },
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
        } => Op::Fset {
            dst,
            a,
            am,
            b: f(b),
            bm,
            cmp,
            bop,
            src,
            bf,
        },

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
        } => Op::Hadd2 {
            dst,
            a,
            am,
            asw,
            b: f(b),
            bm,
            bsw,
            merge,
            ftz,
            sat,
        },
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
        } => Op::Hmul2 {
            dst,
            a,
            am,
            asw,
            b: f(b),
            bm,
            bsw,
            merge,
            prec,
            sat,
        },
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
        } => Op::Hfma2 {
            dst,
            a,
            asw,
            b: f(b),
            bneg,
            bsw,
            c: f(c),
            cneg,
            csw,
            merge,
            prec,
            sat,
        },
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
        } => Op::Hset2 {
            dst,
            a,
            am,
            asw,
            b: f(b),
            bm,
            bsw,
            cmp,
            bop,
            src,
            bf,
            ftz,
        },
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
        } => Op::Hsetp2 {
            p0,
            p1,
            a,
            am,
            asw,
            b: f(b),
            bm,
            bsw,
            cmp,
            bop,
            src,
            and,
            ftz,
        },

        // ---- integer ----
        Op::Iadd {
            dst,
            a,
            aneg,
            b,
            bneg,
            cin,
            cout,
        } => Op::Iadd {
            dst,
            a,
            aneg,
            b: f(b),
            bneg,
            cin,
            cout,
        },
        Op::Iadd3 {
            dst,
            a,
            aneg,
            b,
            bneg,
            c,
            cneg,
        } => Op::Iadd3 {
            dst,
            a,
            aneg,
            b: f(b),
            bneg,
            c: f(c),
            cneg,
        },
        Op::Imnmx {
            dst,
            a,
            b,
            pred,
            signed,
        } => Op::Imnmx {
            dst,
            a,
            b: f(b),
            pred,
            signed,
        },
        Op::Iscadd {
            dst,
            a,
            aneg,
            b,
            bneg,
            shift,
        } => Op::Iscadd {
            dst,
            a,
            aneg,
            b: f(b),
            bneg,
            shift,
        },
        Op::Isetp {
            p0,
            p1,
            a,
            b,
            cmp,
            signed,
            bop,
            src,
        } => Op::Isetp {
            p0,
            p1,
            a,
            b: f(b),
            cmp,
            signed,
            bop,
            src,
        },
        Op::Iset {
            dst,
            a,
            b,
            cmp,
            signed,
            bop,
            src,
            bf,
        } => Op::Iset {
            dst,
            a,
            b: f(b),
            cmp,
            signed,
            bop,
            src,
            bf,
        },
        Op::Icmp {
            dst,
            a,
            b,
            c,
            cmp,
            signed,
        } => Op::Icmp {
            dst,
            a,
            b: f(b),
            c,
            cmp,
            signed,
        },
        Op::Imul {
            dst,
            a,
            b,
            signed,
            hi,
        } => Op::Imul {
            dst,
            a,
            b: f(b),
            signed,
            hi,
        },

        // ---- bit manipulation ----
        Op::Bfi {
            dst,
            insert,
            src,
            base,
        } => Op::Bfi {
            dst,
            insert,
            src: f(src),
            base: f(base),
        },
        Op::R2p { src, mask, byte } => Op::R2p {
            src,
            mask: f(mask),
            byte,
        },
        Op::Lop {
            dst,
            a,
            ainv,
            b,
            binv,
            op,
            pred,
        } => Op::Lop {
            dst,
            a,
            ainv,
            b: f(b),
            binv,
            op,
            pred,
        },
        Op::Lop3 { dst, a, b, c, lut } => Op::Lop3 {
            dst,
            a,
            b: f(b),
            c: f(c),
            lut,
        },
        Op::Shl { dst, a, b, wrap } => Op::Shl {
            dst,
            a,
            b: f(b),
            wrap,
        },
        Op::Shr {
            dst,
            a,
            b,
            signed,
            wrap,
        } => Op::Shr {
            dst,
            a,
            b: f(b),
            signed,
            wrap,
        },
        Op::Shf {
            dst,
            lo,
            shift,
            hi,
            left,
            wrap,
            hi_out,
        } => Op::Shf {
            dst,
            lo,
            shift: f(shift),
            hi,
            left,
            wrap,
            hi_out,
        },
        Op::Bfe { dst, a, b, signed } => Op::Bfe {
            dst,
            a,
            b: f(b),
            signed,
        },
        Op::Popc { dst, b, inv } => Op::Popc { dst, b: f(b), inv },
        Op::Flo {
            dst,
            b,
            signed,
            shift,
            inv,
        } => Op::Flo {
            dst,
            b: f(b),
            signed,
            shift,
            inv,
        },
        Op::Sel { dst, a, b, pred } => Op::Sel {
            dst,
            a,
            b: f(b),
            pred,
        },

        // ---- moves and conversions ----
        Op::Mov { dst, src } => Op::Mov { dst, src: f(src) },
        Op::I2f {
            dst,
            src,
            sm,
            src_bytes,
            src_signed,
            sel,
        } => Op::I2f {
            dst,
            src: f(src),
            sm,
            src_bytes,
            src_signed,
            sel,
        },
        Op::F2i {
            dst,
            src,
            sm,
            dst_bytes,
            dst_signed,
            round,
            ftz,
        } => Op::F2i {
            dst,
            src: f(src),
            sm,
            dst_bytes,
            dst_signed,
            round,
            ftz,
        },
        Op::F2f {
            dst,
            src,
            sm,
            round,
            ftz,
            sat,
            src_bits,
            dst_bits,
            hi,
        } => Op::F2f {
            dst,
            src: f(src),
            sm,
            round,
            ftz,
            sat,
            src_bits,
            dst_bits,
            hi,
        },

        // `ldc`'s bank is register-indexed, so it cannot be folded.
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shader::interp::{Env, Invocation, NoTextures, ShaderResult};
    use crate::gpu::shader::isa::{FMod, Instruction};
    use crate::Error;
    use std::collections::HashMap;

    /// A straight-line program at 32-byte-block byte offsets.
    fn program(ops: &[Op]) -> Program {
        let mut p = Program::default();
        let mut offset = crate::gpu::shader::ENTRY_OFFSET;
        for &op in ops {
            p.insns.push(Instruction::always(op));
            p.offsets.push(offset);
            offset = crate::gpu::shader::next_slot(offset);
        }
        p
    }

    /// A constant source with nothing bound.
    struct Unbound;
    impl ConstantSource for Unbound {
        fn read_const(&self, bank: u8, _offset: u16) -> ShaderResult<u32> {
            Err(Box::new(Error::Gpu(format!("no bank {bank}"))))
        }
    }

    #[test]
    fn folding_replaces_a_constant_with_the_value_the_draw_will_read() {
        let consts: HashMap<(u8, u16), f32> = [((3, 16), 2.5f32)].into_iter().collect();
        let p = program(&[Op::Fadd {
            dst: 0,
            a: 1,
            am: FMod::NONE,
            b: Operand::Const {
                bank: 3,
                offset: 16,
            },
            bm: FMod::NONE,
            ftz: false,
            sat: false,
        }]);
        let compiled = Compiled::with_constants(&p, &consts);
        match compiled.op(0) {
            Op::Fadd {
                b: Operand::Imm(bits),
                ..
            } => assert_eq!(f32::from_bits(bits), 2.5),
            other => panic!("not folded: {other:?}"),
        }
    }

    #[test]
    fn a_constant_that_cannot_be_read_is_left_for_the_instruction_to_fail_on() {
        // An unbound bank must still be reported from the reading instruction.
        let b = Operand::Const { bank: 5, offset: 0 };
        let p = program(&[Op::Mov { dst: 0, src: b }]);
        let compiled = Compiled::with_constants(&p, &Unbound);
        assert_eq!(compiled.op(0), Op::Mov { dst: 0, src: b });
    }

    #[test]
    fn folding_cannot_change_what_a_program_computes() {
        // Folding must be invisible: both ways leave the same registers.
        let consts: HashMap<(u8, u16), f32> =
            [((0, 0), 3.0f32), ((0, 4), 0.5f32)].into_iter().collect();
        let ops = [
            Op::Mov {
                dst: 1,
                src: Operand::Imm(2.0f32.to_bits()),
            },
            Op::Fmul {
                dst: 2,
                a: 1,
                b: Operand::Const { bank: 0, offset: 0 },
                bm: FMod::NONE,
                ftz: false,
                sat: false,
                scale: super::super::isa::FmulScale::None,
            },
            Op::Fadd {
                dst: 3,
                a: 2,
                am: FMod::NONE,
                b: Operand::Const { bank: 0, offset: 4 },
                bm: FMod::NONE,
                ftz: false,
                sat: false,
            },
            Op::Exit,
        ];
        let p = program(&ops);
        let env = Env::new(&consts, &NoTextures);

        let mut plain = Invocation::new();
        plain.execute(&Compiled::new(&p), &env).unwrap();
        let mut folded = Invocation::new();
        folded
            .execute(&Compiled::with_constants(&p, &consts), &env)
            .unwrap();

        for reg in 0..4u8 {
            assert_eq!(plain.reg(reg), folded.reg(reg), "r{reg}");
        }
        assert_eq!(folded.reg_f32(3), 2.0 * 3.0 + 0.5);
    }

    #[test]
    fn a_branch_target_becomes_an_index() {
        let p = program(&[
            Op::Nop,
            Op::Bra {
                target: crate::gpu::shader::ENTRY_OFFSET,
            },
            Op::Exit,
        ]);
        let compiled = Compiled::new(&p);
        assert_eq!(compiled.target(1), 0, "the bra resolves to instruction 0");
        assert_eq!(compiled.target(0), NO_TARGET, "a nop branches nowhere");
        assert_eq!(compiled.target(2), NO_TARGET, "nor does an exit");
    }

    #[test]
    fn a_branch_to_an_offset_that_was_never_decoded_is_reported_when_taken() {
        // Unreachable bad branches must not stop lowering.
        let p = program(&[Op::Bra { target: 0x1234 }]);
        let compiled = Compiled::new(&p);
        assert_eq!(compiled.target(0), NO_TARGET);

        let consts = HashMap::new();
        let env = Env::new(&consts, &NoTextures);
        let err = Invocation::new().execute(&compiled, &env).unwrap_err();
        assert!(format!("{err:?}").contains("never decoded"), "got {err:?}");
    }
}
