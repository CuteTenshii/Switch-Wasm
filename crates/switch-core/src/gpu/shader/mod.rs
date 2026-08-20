//! Decoding a complete Maxwell shader binary, as opposed to a single
//! instruction (that's [`isa`]).
//!
//! Real binaries pack instructions in 32-byte blocks: an 8-byte `sched`
//! control word (which register bank/latency hints the scheduler needs — not
//! a real instruction) followed by three real 8-byte instructions.
//!
//! A shader binary carries no length, so the end has to be found rather than
//! read. The first version of this stopped at the first `exit`, which is
//! right only for straight-line programs — a shader with any control flow has
//! code *after* its first `exit` that a branch reaches. This walks the
//! control-flow graph instead: decode from the entry point, follow every
//! branch target, and stop each path at whatever ends it. Anything no path
//! reaches is padding and never decoded.

pub mod interp;
pub mod isa;

pub use isa::{Instruction, Op};

use crate::{Error, Result};
use std::collections::{BTreeMap, HashSet};

/// Hard cap on decoded instructions per program, so a binary that is missing
/// its `exit` — a corrupt upload, or a control-flow form this decoder cannot
/// follow — can't walk off into unmapped memory.
const MAX_INSTRUCTIONS: usize = 4096;

/// The first real instruction sits in slot 1 of the first 32-byte block,
/// right after that block's `sched` word.
pub const ENTRY_OFFSET: u32 = 8;

/// A decoded program: instructions in ascending address order, each paired
/// with the byte offset it was decoded from so a branch target can be
/// resolved back to an index.
#[derive(Debug, Clone, Default)]
pub struct Program {
    pub insns: Vec<Instruction>,
    pub offsets: Vec<u32>,
}

impl Program {
    pub fn len(&self) -> usize {
        self.insns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.insns.is_empty()
    }

    /// The index of the instruction at `byte_offset`, if it was decoded.
    pub fn index_of(&self, byte_offset: u32) -> Option<usize> {
        self.offsets.binary_search(&byte_offset).ok()
    }
}

/// Whether `offset` names a real instruction rather than a `sched` control
/// word. Slot 0 of every 32-byte block is the control word.
fn is_instruction_slot(offset: u32) -> bool {
    (offset / 8) % 4 != 0
}

/// The next instruction slot after `offset`, skipping the `sched` word that
/// starts each block.
pub fn next_slot(offset: u32) -> u32 {
    let next = offset + 8;
    if is_instruction_slot(next) {
        next
    } else {
        next + 8
    }
}

/// Decode a program by walking its control-flow graph from `ENTRY_OFFSET`.
/// `read` fetches the 8-byte word at a byte offset; it is fallible because a
/// real one reads guest memory, and a program that runs off the end of what
/// is mapped is a decode error rather than a panic.
pub fn decode_program_with(read: &mut dyn FnMut(u32) -> Result<u64>) -> Result<Program> {
    let mut decoded: BTreeMap<u32, Instruction> = BTreeMap::new();
    let mut queued: HashSet<u32> = HashSet::new();
    let mut worklist = vec![ENTRY_OFFSET];
    queued.insert(ENTRY_OFFSET);

    while let Some(mut offset) = worklist.pop() {
        loop {
            if decoded.contains_key(&offset) {
                break; // already walked from here
            }
            if decoded.len() >= MAX_INSTRUCTIONS {
                return Err(Error::Gpu(format!(
                    "shader: program exceeded {MAX_INSTRUCTIONS} instructions"
                )));
            }
            let insn = isa::decode_at(read(offset)?, offset);
            decoded.insert(offset, insn);

            let push = |target: u32, worklist: &mut Vec<u32>, queued: &mut HashSet<u32>| {
                if is_instruction_slot(target) && queued.insert(target) {
                    worklist.push(target);
                }
            };
            // Whether this instruction can fall through to the next one, and
            // what else it can reach.
            let falls_through = match insn.op {
                Op::Exit | Op::Kil => !insn.pred.is_always(),
                Op::Bra { target } => {
                    push(target, &mut worklist, &mut queued);
                    !insn.pred.is_always()
                }
                // `sync`/`brk`/`cont` jump to a point pushed earlier by the
                // matching `ssy`/`pbk`/`pcnt`, which is already queued.
                Op::Sync | Op::Brk | Op::Cont => !insn.pred.is_always(),
                Op::Ssy { target } | Op::Pbk { target } | Op::Pcnt { target } => {
                    push(target, &mut worklist, &mut queued);
                    true
                }
                _ => true,
            };
            if !falls_through {
                break;
            }
            offset = next_slot(offset);
        }
    }

    if decoded.is_empty() {
        return Err(Error::Gpu("shader: empty program".into()));
    }
    let mut program = Program::default();
    for (offset, insn) in decoded {
        program.offsets.push(offset);
        program.insns.push(insn);
    }
    Ok(program)
}

/// [`decode_program_with`] over a byte slice.
pub fn decode_program(bytes: &[u8]) -> Result<Program> {
    if !bytes.len().is_multiple_of(8) {
        return Err(Error::Gpu(format!(
            "shader: program length {} is not a multiple of 8 bytes",
            bytes.len()
        )));
    }
    decode_program_with(&mut |offset: u32| {
        let start = offset as usize;
        bytes
            .get(start..start + 8)
            .map(|w| u64::from_le_bytes(w.try_into().expect("8 bytes")))
            .ok_or_else(|| {
                Error::Gpu(format!("shader: program read at {offset:#x} is past its end"))
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use isa::{FMod, MemSize, MufuOp, Operand, TexDim, RZ};

    /// Just the opcodes, for comparing a decode against an expected list.
    fn ops(program: &Program) -> Vec<Op> {
        program.insns.iter().map(|i| i.op).collect()
    }

    fn word(low: u32, high: u32) -> [u8; 8] {
        let v = ((high as u64) << 32) | low as u64;
        v.to_le_bytes()
    }

    fn block(sched: (u32, u32), a: (u32, u32), b: (u32, u32), c: (u32, u32)) -> Vec<u8> {
        let mut out = Vec::with_capacity(32);
        out.extend_from_slice(&word(sched.0, sched.1));
        out.extend_from_slice(&word(a.0, a.1));
        out.extend_from_slice(&word(b.0, b.1));
        out.extend_from_slice(&word(c.0, c.1));
        out
    }

    #[test]
    fn strips_sched_words_and_stops_at_exit() {
        // solid.frag's first two 32-byte blocks, transcribed from the
        // envydis capture cited in `isa`'s module docs: sched, ipa-pass,
        // mufu-rcp, ipa; sched, ipa, ipa, ipa. The dump's third block (sched,
        // exit, padding...) is truncated after `exit` here on purpose, to
        // prove decode_program stops there rather than reading the trailing
        // `bra`/`nop` padding.
        let mut bytes = block(
            (0xe1a0070f, 0x00240401),
            (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
            (0x00470003, 0x50800000), // mufu rcp $r3 $r0
            (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
        );
        bytes.extend(block(
            (0xb0400341, 0x055c8400),
            (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
            (0x8037ff02, 0xe043ff88), // ipa $r2 a[0x88] $r3 0x0 0x1
            (0xc037ff03, 0xe043ff88), // ipa $r3 a[0x8c] $r3 0x0 0x1
        ));
        bytes.extend(block(
            (0xffe1ffef, 0x001f8000),
            (0x0007000f, 0xe3000000), // exit
            (0xff87000f, 0xe2400fff), // bra 0x50 (padding, never reached)
            (0x00070f00, 0x50b00000), // nop (padding, never reached)
        ));

        let program = decode_program(&bytes).unwrap();
        assert_eq!(
            ops(&program),
            vec![
                Op::Ipa { dst: 0, offset: 0x7c, mul: None, perspective: false, sat: false },
                Op::Mufu { dst: 3, src: 0, sm: FMod::NONE, op: MufuOp::Rcp, sat: false },
                Op::Ipa { dst: 0, offset: 0x80, mul: Some(3), perspective: true, sat: false },
                Op::Ipa { dst: 1, offset: 0x84, mul: Some(3), perspective: true, sat: false },
                Op::Ipa { dst: 2, offset: 0x88, mul: Some(3), perspective: true, sat: false },
                Op::Ipa { dst: 3, offset: 0x8c, mul: Some(3), perspective: true, sat: false },
                Op::Exit,
            ]
        );
    }

    #[test]
    fn mvp_vertex_shader_fixture_decodes_instruction_for_instruction() {
        // mvp.vert in full, transcribed from the envydis capture cited in
        // `isa`'s module docs (a matrix-vector multiply computed via
        // register rotation, then two attribute stores).
        let mut bytes = block(
            (0xfc20070f, 0x081f8441),
            (0x0807ff00, 0xefd9ff80), // ld b128 $r0 a[0x80] 0x0
            (0x00070004, 0x4c681008), // fmul ftz $r4 $r0 c2[0x0]
            (0x00170005, 0x4c681008), // fmul ftz $r5 $r0 c2[0x4]
        );
        bytes.extend(block(
            (0xfc6207e1, 0x081f8400),
            (0x00270006, 0x4c681008), // fmul ftz $r6 $r0 c2[0x8]
            (0x00370000, 0x4c681008), // fmul ftz $r0 $r0 c2[0xc]
            (0x00470104, 0x49a00208), // ffma ftz $r4 $r1 c2[0x10] $r4
        ));
        bytes.extend(block(
            (0xfc2207e1, 0x001f8c40),
            (0x00570105, 0x49a00288), // ffma ftz $r5 $r1 c2[0x14] $r5
            (0x00670106, 0x49a00308), // ffma ftz $r6 $r1 c2[0x18] $r6
            (0x00770100, 0x49a00008), // ffma ftz $r0 $r1 c2[0x1c] $r0
        ));
        bytes.extend(block(
            (0xfc2207e1, 0x081f8440),
            (0x00870201, 0x49a00208), // ffma ftz $r1 $r2 c2[0x20] $r4
            (0x00970204, 0x49a00288), // ffma ftz $r4 $r2 c2[0x24] $r5
            (0x00a70205, 0x49a00308), // ffma ftz $r5 $r2 c2[0x28] $r6
        ));
        bytes.extend(block(
            (0xfc2007e3, 0x081f8440),
            (0x00b70206, 0x49a00008), // ffma ftz $r6 $r2 c2[0x2c] $r0
            (0x00c70300, 0x49a00088), // ffma ftz $r0 $r3 c2[0x30] $r1
            (0x00d70301, 0x49a00208), // ffma ftz $r1 $r3 c2[0x34] $r4
        ));
        bytes.extend(block(
            (0xfcc207e1, 0x00038800),
            (0x00e70302, 0x49a00288), // ffma ftz $r2 $r3 c2[0x38] $r5
            (0x00f70303, 0x49a00308), // ffma ftz $r3 $r3 c2[0x3c] $r6
            (0x0707ff00, 0xeff1ff80), // st b128 a[0x70] $r0 0x0
        ));
        bytes.extend(block(
            (0x1c200f0f, 0x07ffbc01),
            (0x0907ff00, 0xefd9ff80), // ld b128 $r0 a[0x90] 0x0
            (0x0807ff00, 0xeff1ff80), // st b128 a[0x80] $r0 0x0
            (0x0007000f, 0xe3000000), // exit
        ));

        let program = decode_program(&bytes).unwrap();
        assert_eq!(program.len(), 21);
        assert_eq!(program.insns[0].op, Op::Ld { dst: 0, offset: 0x80, idx: RZ, size: MemSize::B128 });
        assert_eq!(
            program.insns[1].op,
            Op::Fmul {
                dst: 4,
                a: 0,
                b: Operand::Const { bank: 2, offset: 0x0 },
                bm: FMod::NONE,
                ftz: true,
                sat: false,
            }
        );
        assert_eq!(
            program.insns[5].op,
            Op::Ffma { dst: 4, a: 1, b: Operand::Const { bank: 2, offset: 0x10 }, bneg: false, c: Operand::Reg(4), cneg: false, ftz: true, sat: false }
        );
        assert_eq!(program.insns[17].op, Op::St { offset: 0x70, idx: RZ, src: 0, size: MemSize::B128 });
        assert_eq!(program.insns[18].op, Op::Ld { dst: 0, offset: 0x90, idx: RZ, size: MemSize::B128 });
        assert_eq!(program.insns[19].op, Op::St { offset: 0x80, idx: RZ, src: 0, size: MemSize::B128 });
        assert_eq!(program.insns[20].op, Op::Exit);
    }

    #[test]
    fn tex_fragment_shader_fixture_decodes_the_texs() {
        // tex.frag in full: ipa pass + mufu rcp for perspective correction,
        // a texs sample, then the vertex-color modulation.
        let mut bytes = block(
            (0xe1a0070f, 0x003c0401),
            (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
            (0x00470004, 0x50800000), // mufu rcp $r4 $r0
            (0x0047ff00, 0xe043ff89), // ipa $r0 a[0x90] $r4 0x0 0x1
        );
        bytes.extend(block(
            (0xe020072f, 0x001cbc03),
            (0x4047ff01, 0xe043ff89), // ipa $r1 a[0x94] $r4 0x0 0x1
            (0x20170000, 0xd8301a40), // texs $r2 $r0 $r0 $r1 0x1a4 t2d rgba
            (0x0047ff05, 0xe043ff88), // ipa $r5 a[0x80] $r4 0x0 0x1
        ));
        bytes.extend(block(
            (0xe1e01ff0, 0x003fc000),
            (0x00570000, 0x5c681000), // fmul ftz $r0 $r0 $r5
            (0x4047ff05, 0xe043ff88), // ipa $r5 a[0x84] $r4 0x0 0x1
            (0x00570101, 0x5c681000), // fmul ftz $r1 $r1 $r5
        ));
        bytes.extend(block(
            (0xfe00070f, 0x001c3c01),
            (0x8047ff05, 0xe043ff88), // ipa $r5 a[0x88] $r4 0x0 0x1
            (0x00570202, 0x5c681000), // fmul ftz $r2 $r2 $r5
            (0xc047ff04, 0xe043ff88), // ipa $r4 a[0x8c] $r4 0x0 0x1
        ));
        bytes.extend(block(
            (0xfde00ff0, 0x001ffc3f),
            (0x00470303, 0x5c681000), // fmul ftz $r3 $r3 $r4
            (0x0007000f, 0xe3000000), // exit
            (0xff87000f, 0xe2400fff), // bra (padding, never reached)
        ));

        let program = decode_program(&bytes).unwrap();
        assert_eq!(
            program.insns[4].op,
            Op::Texs { dst: 0, dst2: 2, coords: [2, 0, 1], handle: 0x1a4, dim: TexDim::T2d, mask: [true, true, true, true] }
        );
        assert_eq!(program.insns.last().unwrap().op, Op::Exit);
    }

    #[test]
    fn a_program_that_never_ends_is_an_error_not_a_hang() {
        // All-zero words decode as unimplemented instructions, which fall
        // through to the next slot; with nothing ending the path the walk
        // runs off the end of the buffer, and that is an error.
        let bytes = block((0, 0), (0, 0), (0, 0), (0, 0));
        assert!(decode_program(&bytes).is_err());
    }

    #[test]
    fn a_misaligned_program_is_an_error() {
        assert!(decode_program(&[0u8; 7]).is_err());
    }
}
