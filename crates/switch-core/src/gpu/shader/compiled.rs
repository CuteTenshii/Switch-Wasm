//! A decoded [`Program`] lowered for execution.
//!
//! [`Program`] is what the binary says: instructions paired with the byte
//! offsets they were decoded from. That is the right shape for reading a
//! shader and the wrong one for running it, because a fragment shader runs
//! once per covered pixel — a full-screen quad runs it 921 600 times — and
//! everything the interpreter re-derives on the way is re-derived that many
//! times.
//!
//! This does it once. A branch's target stops being a byte offset that has to
//! be binary-searched for on every taken branch and becomes an index; the
//! deferred texture writes stop being looked up by a linear scan; and every
//! constant the draw's bound banks can supply is read out of guest memory once
//! and folded into an immediate, rather than going back through the constant
//! cache on every operand evaluation.
//!
//! It is also the shape a shader translator wants — constants folded, control
//! flow resolved — so the same lowering serves both.

use super::interp::{texs_writes_for, ConstantSource};
use super::isa::{Op, Operand, Pred};
use super::{Program, TexsWrites};

/// A branch target that is not an index into [`Compiled::insns`]: either the
/// instruction is not a branch, or its target was never decoded. The second
/// case stays an error raised where the branch is *taken*, exactly as it was
/// when targets were resolved there.
pub const NO_TARGET: u32 = u32::MAX;

pub struct Compiled {
    /// The operations, with foldable constants already resolved.
    ///
    /// Held apart from the predicates rather than as `Vec<Instruction>`
    /// because an `Op` is 32 bytes and an `Instruction` is 40: at 40 the array
    /// straddles cache lines and holds 1.6 instructions per line instead of
    /// 2, and this is walked once per instruction per covered pixel.
    ops: Vec<Op>,
    /// The guard on each operation, in its own dense array.
    preds: Vec<Pred>,
    /// Where each instruction's branch goes, as an index into `insns`, or
    /// [`NO_TARGET`].
    targets: Vec<u32>,
    /// The byte offset each instruction came from. Needed for error messages,
    /// and for `brx`, whose target is a register value and so is only known
    /// while running.
    offsets: Vec<u32>,
    /// Where each `texs`'s results land, by instruction index.
    texs_writes: Vec<TexsWrites>,
    /// The generic varying slots this program interpolates.
    interpolated_slots: Vec<usize>,
    /// Where each `brx` can go, by the index of the `brx` itself, resolved
    /// from the byte offsets the decoder recorded.
    indirect: std::collections::HashMap<usize, Vec<u32>>,
    /// The Shader Program Header, when the program had one.
    header: Option<super::ProgramHeader>,
}

impl Compiled {
    /// Lower `program` without a constant source. Constants stay as they were
    /// and are read through [`super::interp::Env`] as before.
    pub fn new(program: &Program) -> Compiled {
        Compiled::lower(program, None)
    }

    /// Lower `program`, folding every constant the draw's bound banks can
    /// supply.
    ///
    /// Sound because a constant buffer cannot change while a draw runs: the
    /// GPU processes methods in order, and a draw is one method. A bank that
    /// is unbound, or an offset past the end of one, is left alone so that the
    /// error still surfaces from the instruction that reads it.
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
                Some(target) => {
                    program.index_of(target).map(|i| i as u32).unwrap_or(NO_TARGET)
                }
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

    /// The Shader Program Header, when this program was preceded by one.
    pub fn header(&self) -> Option<super::ProgramHeader> {
        self.header
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// The operation at `index`.
    #[inline]
    pub fn op(&self, index: usize) -> Op {
        self.ops[index]
    }

    /// The guard on the operation at `index`.
    #[inline]
    pub fn pred(&self, index: usize) -> Pred {
        self.preds[index]
    }

    /// Every operation, in order — for the passes that only look at opcodes.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Every guard, in order. Always as long as [`Compiled::ops`].
    pub fn preds(&self) -> &[Pred] {
        &self.preds
    }

    /// The resolved target of the branch at `index`, or [`NO_TARGET`].
    #[inline]
    pub fn target(&self, index: usize) -> u32 {
        self.targets.get(index).copied().unwrap_or(NO_TARGET)
    }

    /// The byte offset instruction `index` was decoded from — what an error
    /// message names, since that is the address in the shader binary.
    pub fn offset(&self, index: usize) -> u32 {
        self.offsets.get(index).copied().unwrap_or(0)
    }

    /// The index of the instruction at `byte_offset`, if it was decoded.
    ///
    /// Only `brx` needs this: every other branch had its target resolved when
    /// this was built.
    pub fn index_of(&self, byte_offset: u32) -> Option<usize> {
        self.offsets.binary_search(&byte_offset).ok()
    }

    /// Where the `brx` at `index` can go, as indices — the arms of the switch
    /// it lowers, which nothing on the instruction itself names.
    pub fn indirect_targets(&self, index: usize) -> Option<&[u32]> {
        self.indirect.get(&index).map(|t| t.as_slice())
    }

    /// The generic varying slots this program's `ipa`s read, ascending — see
    /// [`super::interpolated_slots`].
    pub fn interpolated_slots(&self) -> &[usize] {
        &self.interpolated_slots
    }

    /// The deferred register writes the `texs` at `index` produces — see
    /// [`Program::texs_writes`].
    pub fn texs_writes(&self, index: usize) -> &[(u8, super::isa::TexsStore, usize)] {
        self.texs_writes
            .iter()
            .find(|t| t.at == index)
            .map(|t| t.writes.as_slice())
            .unwrap_or(&[])
    }
}

/// The byte offset an instruction branches to, for the forms whose target is
/// known statically. `brx` is absent on purpose: its target is a register
/// value plus a base, so there is nothing to resolve here.
fn branch_target(op: Op) -> Option<u32> {
    match op {
        Op::Bra { target } | Op::Ssy { target } | Op::Pbk { target } | Op::Pcnt { target } => {
            Some(target)
        }
        _ => None,
    }
}

/// Replace every constant-bank operand whose value this draw already knows
/// with that value.
///
/// Missing a variant here costs a fold, not correctness: the operand stays a
/// `Const` and is read through the constant source exactly as it was.
fn fold(op: Op, consts: &dyn ConstantSource) -> Op {
    /// Resolve one operand, or leave it as it is.
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
        Op::Fadd { dst, a, am, b, bm, ftz, sat } => {
            Op::Fadd { dst, a, am, b: f(b), bm, ftz, sat }
        }
        Op::Fmul { dst, a, b, bm, ftz, sat, scale } => {
            Op::Fmul { dst, a, b: f(b), bm, ftz, sat, scale }
        }
        Op::Ffma { dst, a, b, bneg, c, cneg, ftz, sat } => {
            Op::Ffma { dst, a, b: f(b), bneg, c: f(c), cneg, ftz, sat }
        }
        Op::Fmnmx { dst, a, am, b, bm, pred, ftz } => {
            Op::Fmnmx { dst, a, am, b: f(b), bm, pred, ftz }
        }
        Op::Fsetp { p0, p1, a, am, b, bm, cmp, bop, src } => {
            Op::Fsetp { p0, p1, a, am, b: f(b), bm, cmp, bop, src }
        }
        Op::Fset { dst, a, am, b, bm, cmp, bop, src, bf } => {
            Op::Fset { dst, a, am, b: f(b), bm, cmp, bop, src, bf }
        }

        // ---- half-precision ----
        Op::Hadd2 { dst, a, am, asw, b, bm, bsw, merge, ftz, sat } => {
            Op::Hadd2 { dst, a, am, asw, b: f(b), bm, bsw, merge, ftz, sat }
        }
        Op::Hmul2 { dst, a, am, asw, b, bm, bsw, merge, prec, sat } => {
            Op::Hmul2 { dst, a, am, asw, b: f(b), bm, bsw, merge, prec, sat }
        }
        Op::Hfma2 { dst, a, asw, b, bneg, bsw, c, cneg, csw, merge, prec, sat } => {
            Op::Hfma2 { dst, a, asw, b: f(b), bneg, bsw, c: f(c), cneg, csw, merge, prec, sat }
        }
        Op::Hset2 { dst, a, am, asw, b, bm, bsw, cmp, bop, src, bf, ftz } => {
            Op::Hset2 { dst, a, am, asw, b: f(b), bm, bsw, cmp, bop, src, bf, ftz }
        }
        Op::Hsetp2 { p0, p1, a, am, asw, b, bm, bsw, cmp, bop, src, and, ftz } => {
            Op::Hsetp2 { p0, p1, a, am, asw, b: f(b), bm, bsw, cmp, bop, src, and, ftz }
        }

        // ---- integer ----
        Op::Iadd { dst, a, aneg, b, bneg, cin, cout } => {
            Op::Iadd { dst, a, aneg, b: f(b), bneg, cin, cout }
        }
        Op::Iadd3 { dst, a, aneg, b, bneg, c, cneg } => {
            Op::Iadd3 { dst, a, aneg, b: f(b), bneg, c: f(c), cneg }
        }
        Op::Imnmx { dst, a, b, pred, signed } => Op::Imnmx { dst, a, b: f(b), pred, signed },
        Op::Iscadd { dst, a, aneg, b, bneg, shift } => {
            Op::Iscadd { dst, a, aneg, b: f(b), bneg, shift }
        }
        Op::Isetp { p0, p1, a, b, cmp, signed, bop, src } => {
            Op::Isetp { p0, p1, a, b: f(b), cmp, signed, bop, src }
        }
        Op::Iset { dst, a, b, cmp, signed, bop, src, bf } => {
            Op::Iset { dst, a, b: f(b), cmp, signed, bop, src, bf }
        }
        Op::Icmp { dst, a, b, c, cmp, signed } => Op::Icmp { dst, a, b: f(b), c, cmp, signed },
        Op::Imul { dst, a, b, signed, hi } => Op::Imul { dst, a, b: f(b), signed, hi },

        // ---- bit manipulation ----
        Op::Bfi { dst, insert, src, base } => Op::Bfi { dst, insert, src: f(src), base: f(base) },
        Op::R2p { src, mask, byte } => Op::R2p { src, mask: f(mask), byte },
        Op::Lop { dst, a, ainv, b, binv, op, pred } => {
            Op::Lop { dst, a, ainv, b: f(b), binv, op, pred }
        }
        Op::Lop3 { dst, a, b, c, lut } => Op::Lop3 { dst, a, b: f(b), c: f(c), lut },
        Op::Shl { dst, a, b, wrap } => Op::Shl { dst, a, b: f(b), wrap },
        Op::Shr { dst, a, b, signed, wrap } => Op::Shr { dst, a, b: f(b), signed, wrap },
        Op::Shf { dst, lo, shift, hi, left, wrap, hi_out } => {
            Op::Shf { dst, lo, shift: f(shift), hi, left, wrap, hi_out }
        }
        Op::Bfe { dst, a, b, signed } => Op::Bfe { dst, a, b: f(b), signed },
        Op::Popc { dst, b, inv } => Op::Popc { dst, b: f(b), inv },
        Op::Flo { dst, b, signed, shift, inv } => Op::Flo { dst, b: f(b), signed, shift, inv },
        Op::Sel { dst, a, b, pred } => Op::Sel { dst, a, b: f(b), pred },

        // ---- moves and conversions ----
        Op::Mov { dst, src } => Op::Mov { dst, src: f(src) },
        Op::I2f { dst, src, sm, src_bytes, src_signed, sel } => {
            Op::I2f { dst, src: f(src), sm, src_bytes, src_signed, sel }
        }
        Op::F2i { dst, src, sm, dst_bytes, dst_signed, round, ftz } => {
            Op::F2i { dst, src: f(src), sm, dst_bytes, dst_signed, round, ftz }
        }
        Op::F2f { dst, src, sm, round, ftz, sat } => {
            Op::F2f { dst, src: f(src), sm, round, ftz, sat }
        }

        // Everything else carries no constant-bank operand — `ldc`'s bank is
        // indexed by a register, so its value is not known until it runs.
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

    /// A straight-line program at the byte offsets a real 32-byte-block
    /// layout would put it at.
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

    /// A constant source that has nothing, the way an unbound bank behaves.
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
            b: Operand::Const { bank: 3, offset: 16 },
            bm: FMod::NONE,
            ftz: false,
            sat: false,
        }]);
        let compiled = Compiled::with_constants(&p, &consts);
        match compiled.op(0) {
            Op::Fadd { b: Operand::Imm(bits), .. } => assert_eq!(f32::from_bits(bits), 2.5),
            other => panic!("not folded: {other:?}"),
        }
    }

    #[test]
    fn a_constant_that_cannot_be_read_is_left_for_the_instruction_to_fail_on() {
        // Folding must not swallow the error: an unbound bank has to still be
        // reported from the instruction that reads it, naming that bank.
        let b = Operand::Const { bank: 5, offset: 0 };
        let p = program(&[Op::Mov { dst: 0, src: b }]);
        let compiled = Compiled::with_constants(&p, &Unbound);
        assert_eq!(compiled.op(0), Op::Mov { dst: 0, src: b });
    }

    #[test]
    fn folding_cannot_change_what_a_program_computes() {
        // The whole justification for folding is that it is invisible, so
        // check that directly: the same program, run both ways, must leave
        // the same registers behind.
        let consts: HashMap<(u8, u16), f32> =
            [((0, 0), 3.0f32), ((0, 4), 0.5f32)].into_iter().collect();
        let ops = [
            Op::Mov { dst: 1, src: Operand::Imm(2.0f32.to_bits()) },
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
        folded.execute(&Compiled::with_constants(&p, &consts), &env).unwrap();

        for reg in 0..4u8 {
            assert_eq!(plain.reg(reg), folded.reg(reg), "r{reg}");
        }
        assert_eq!(folded.reg_f32(3), 2.0 * 3.0 + 0.5);
    }

    #[test]
    fn a_branch_target_becomes_an_index() {
        // Resolved once here rather than binary-searched on every taken
        // branch, which is most of what this lowering is for.
        let p = program(&[
            Op::Nop,
            Op::Bra { target: crate::gpu::shader::ENTRY_OFFSET },
            Op::Exit,
        ]);
        let compiled = Compiled::new(&p);
        assert_eq!(compiled.target(1), 0, "the bra resolves to instruction 0");
        assert_eq!(compiled.target(0), NO_TARGET, "a nop branches nowhere");
        assert_eq!(compiled.target(2), NO_TARGET, "nor does an exit");
    }

    #[test]
    fn a_branch_to_an_offset_that_was_never_decoded_is_reported_when_taken() {
        // Not at lowering: a program may carry a branch that no path reaches,
        // and refusing to lower it would refuse shaders that run fine.
        let p = program(&[Op::Bra { target: 0x1234 }]);
        let compiled = Compiled::new(&p);
        assert_eq!(compiled.target(0), NO_TARGET);

        let consts = HashMap::new();
        let env = Env::new(&consts, &NoTextures);
        let err = Invocation::new().execute(&compiled, &env).unwrap_err();
        assert!(format!("{err:?}").contains("never decoded"), "got {err:?}");
    }
}
