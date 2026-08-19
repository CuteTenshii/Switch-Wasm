//! Decoding a complete Maxwell shader binary, as opposed to a single
//! instruction (that's [`isa`]).
//!
//! Real binaries pack instructions in 32-byte blocks: an 8-byte `sched`
//! control word (which register bank/latency hints the scheduler needs — not
//! a real instruction) followed by three real 8-byte instructions. This
//! module strips those control words and decodes the rest, stopping at the
//! first `exit` — everything after it is branch-target padding a program
//! with no branches never reaches.

pub mod interp;
pub mod isa;

pub use isa::Instruction;

use crate::{Error, Result};

/// Hard cap on decoded instructions per program, so a binary that's missing
/// its `exit` (corrupt upload, or a real feature — loops/branches — this
/// decoder doesn't support yet) can't hang the emulator.
const MAX_INSTRUCTIONS: usize = 4096;

/// Decode `bytes` into its real instructions, stripping the `sched` word
/// that precedes every group of three and stopping at the first `exit`.
pub fn decode_program(bytes: &[u8]) -> Result<Vec<Instruction>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(Error::Gpu(format!(
            "shader: program length {} is not a multiple of 8 bytes",
            bytes.len()
        )));
    }

    let mut out = Vec::new();
    for (slot, chunk) in bytes.chunks_exact(8).enumerate() {
        if slot % 4 == 0 {
            continue; // sched control word
        }
        let word = u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8)"));
        let insn = isa::decode(word);
        let is_exit = insn == Instruction::Exit;
        out.push(insn);
        if is_exit {
            return Ok(out);
        }
        if out.len() >= MAX_INSTRUCTIONS {
            return Err(Error::Gpu(format!(
                "shader: program exceeded {} instructions without hitting exit",
                MAX_INSTRUCTIONS
            )));
        }
    }
    Err(Error::Gpu("shader: program ended without an exit".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use isa::{MemSize, Operand, TexDim};

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
            program,
            vec![
                Instruction::Ipa { dst: 0, offset: 0x7c, mul: None, perspective: false },
                Instruction::MufuRcp { dst: 3, src: 0 },
                Instruction::Ipa { dst: 0, offset: 0x80, mul: Some(3), perspective: true },
                Instruction::Ipa { dst: 1, offset: 0x84, mul: Some(3), perspective: true },
                Instruction::Ipa { dst: 2, offset: 0x88, mul: Some(3), perspective: true },
                Instruction::Ipa { dst: 3, offset: 0x8c, mul: Some(3), perspective: true },
                Instruction::Exit,
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
        assert_eq!(program[0], Instruction::Ld { dst: 0, offset: 0x80, size: MemSize::B128 });
        assert_eq!(
            program[1],
            Instruction::Fmul { dst: 4, a: 0, b: Operand::Const { bank: 2, offset: 0x0 }, ftz: true }
        );
        assert_eq!(
            program[5],
            Instruction::Ffma {
                dst: 4,
                a: 1,
                b: Operand::Const { bank: 2, offset: 0x10 },
                c: 4,
                ftz: true,
            }
        );
        assert_eq!(program[17], Instruction::St { offset: 0x70, src: 0, size: MemSize::B128 });
        assert_eq!(program[18], Instruction::Ld { dst: 0, offset: 0x90, size: MemSize::B128 });
        assert_eq!(program[19], Instruction::St { offset: 0x80, src: 0, size: MemSize::B128 });
        assert_eq!(program[20], Instruction::Exit);
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
            program[4],
            Instruction::Texs {
                dst: 2,
                coords: [0, 0, 1],
                handle: 0x1a4,
                dim: TexDim::T2d,
                mask: [true, true, true, true],
            }
        );
        assert_eq!(*program.last().unwrap(), Instruction::Exit);
    }

    #[test]
    fn a_program_without_exit_is_an_error_not_a_hang() {
        let bytes = block((0, 0), (0, 0), (0, 0), (0, 0));
        assert!(decode_program(&bytes).is_err());
    }

    #[test]
    fn a_misaligned_program_is_an_error() {
        assert!(decode_program(&[0u8; 7]).is_err());
    }
}
