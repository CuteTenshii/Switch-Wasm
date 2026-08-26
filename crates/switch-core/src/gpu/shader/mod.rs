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

pub mod cfg;
pub mod compiled;
pub mod interp;
pub mod isa;
pub mod wgsl;

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

/// The generic varying slots `insns`' `ipa`s read, ascending.
///
/// Interpolating a varying costs three multiply-adds per component and happens
/// once per covered pixel, so a full-screen quad pays for each slot 921 600
/// times. Maxwell's generic attribute space holds 32 of them and a real UI
/// shader reads a handful, so the rasterizer interpolates what the fragment
/// shader asks for rather than the whole space.
///
/// Offsets outside the generic range (`gl_Position`, `1/w`, point-sprite
/// coordinates) are not varyings and are handled on their own.
pub fn interpolated_slots(ops: &[Op]) -> Vec<usize> {
    let mut slots: Vec<usize> = ops
        .iter()
        .filter_map(|op| match op {
            Op::Ipa { offset, .. } => Some(*offset),
            _ => None,
        })
        .filter(|&offset| (GENERIC_ATTR_BASE..GENERIC_ATTR_END).contains(&offset))
        .map(|offset| usize::from(offset - GENERIC_ATTR_BASE) / GENERIC_ATTR_STRIDE)
        .collect();
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// Maxwell's generic attribute space: 32 four-component slots, the wires a
/// vertex shader's outputs reach a fragment shader's inputs on.
const GENERIC_ATTR_BASE: u16 = 0x80;
const GENERIC_ATTR_END: u16 = 0x280;
const GENERIC_ATTR_STRIDE: usize = 0x10;

/// A decoded program: instructions in ascending address order, each paired
/// with the byte offset it was decoded from so a branch target can be
/// resolved back to an index.
#[derive(Debug, Clone, Default)]
pub struct Program {
    pub insns: Vec<Instruction>,
    pub offsets: Vec<u32>,
    /// Where each `brx` can go, by the byte offset of the `brx` itself.
    ///
    /// Kept because the decoder had to read the jump table to find the
    /// switch's arms at all (see `brx_targets`), and throwing that away would
    /// leave anything analysing this program unable to follow the one branch
    /// whose target is not on the instruction.
    pub indirect: BTreeMap<u32, Vec<u32>>,
}

/// Where one `texs` instruction's results land.
#[derive(Debug, Clone)]
pub struct TexsWrites {
    /// Index of the `texs` itself.
    pub at: usize,
    /// One entry per enabled colour channel: `(channel, destination
    /// register, the instruction index the write must land before)`.
    pub writes: Vec<(usize, u8, usize)>,
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

/// Words of shader binary to scan looking for `exit` before giving up — not
/// unbounded (see `shader::MAX_INSTRUCTIONS`'s doc comment for the same
/// reasoning). Reading stops as soon as `exit` is found, so a short real
/// program never touches memory past its own end.
///
/// 1024 was "generous for 2D UI" and was not: the Home Menu's own UI shaders
/// run past it, five of them by a single word.
pub const MAX_PROGRAM_WORDS: u64 = 8192;

/// Decode a shader program straight out of GPU memory, stripping `sched`
/// words and stopping at the first `exit` (mirrors
/// `shader::decode_program`, which needs the whole binary as a slice
/// up front — reading incrementally here means a short real program never
/// touches memory past its own end).
/// Nouveau/Mesa's shader upload convention prepends a fixed-size header
/// (driver bookkeeping, not part of the Maxwell ISA) before the real `sched`/
/// instruction stream — confirmed empirically against a live JKSV capture
/// (its vertex and fragment programs both have a recognisable `sched` word,
/// followed by a real `ld`, starting exactly 0x50 bytes in;
/// `/tmp/dump_vs.bin`/`dump_fs.bin` via a temporary dump added and removed
/// for this investigation). `uam`/deko3d-compiled binaries (hbmenu, this
/// module's own test fixtures) have no such header. The header's own first
/// bytes aren't reliably zero (they carry real driver metadata), so this
/// can't be detected by peeking the first word — instead, decode
/// speculatively assuming no header, and if the very first real instruction
/// (slot 1, right after the first `sched` word) doesn't decode, that's the
/// header showing through: retry assuming one.
const MESA_SHADER_HEADER_BYTES: u64 = 0x50;

/// Decode a bound shader program out of guest memory.
///
/// Here rather than beside one of its callers because the rasterizer, the
/// wgpu backend and compute all need the same answer, and the Mesa header
/// skip below is exactly the sort of thing three copies would disagree about.
pub fn decode_program_from_memory(
    ctx: &crate::gpu::exec::ExecCtx,
    addr: u64,
    bindings: &dyn Fn(u8) -> Option<(u64, u32)>,
) -> Result<Program> {
    let first_real_word = ctx.read_u64(addr + 8)?;
    let addr = if matches!(isa::decode(first_real_word).op, Op::Unimplemented { .. }) {
        addr + MESA_SHADER_HEADER_BYTES
    } else {
        addr
    };
    let limit = MAX_PROGRAM_WORDS * 8;
    decode_program_with_consts(
        &mut |offset: u32| {
            if u64::from(offset) >= limit {
                return Err(Error::Gpu(format!(
                    "shader: program read at {offset:#x} is past the {limit:#x}-byte cap"
                )));
            }
            ctx.read_u64(addr + u64::from(offset))
        },
        &mut |bank: u8, offset: u32| {
            let (base, size) = bindings(bank)
                .ok_or_else(|| Error::Gpu(format!("shader: constant bank {bank} is unbound")))?;
            if offset + 4 > size {
                return Err(Error::Gpu(format!(
                    "shader: read of c{bank}[{offset:#x}] is past its {size:#x}-byte end"
                )));
            }
            ctx.read_u32(base + u64::from(offset))
        },
    )
    .inspect(|program| {
        // `TRACE_SHADER=1` prints every program decoded, in control-flow-walk
        // order. A shader that fails to run says only which instruction it
        // stopped on; this is how you see what came before it.
        if crate::env_flag!("TRACE_SHADER") {
            eprintln!("[shader] program at {addr:#x}, {} instructions", program.offsets.len());
            for (i, &off) in program.offsets.iter().enumerate() {
                eprintln!("  {off:#06x}: {:?}", program.insns[i]);
            }
        }
    })
}

/// Whether `offset` names a real instruction rather than a `sched` control
/// word. Slot 0 of every 32-byte block is the control word.
fn is_instruction_slot(offset: u32) -> bool {
    (offset / 8) % 4 != 0
}

/// A branch target rounded onto the instruction slot it means. A target is a
/// raw byte offset and can land on the `sched` word that starts a 32-byte
/// block, which is not an instruction — hardware takes the next slot, so an
/// unaligned target is the block's first real instruction rather than an
/// error. Every computed branch goes through this: `bra`, `ssy`, `pbk`,
/// `pcnt` and `brx` alike.
pub fn align_slot(offset: u32) -> u32 {
    if offset % 32 == 0 {
        offset + 8
    } else {
        offset
    }
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

/// How many instructions back a jump-table walk will look.
///
/// The walk follows a chain of definitions rather than a run of adjacent
/// instructions, so this is not the size of the idiom — it is a bound on a
/// binary where the shape is simply not there. The Home Menu's fragment
/// shaders put 36 instruction slots between the clamp and the branch, which
/// is exactly why this is not the 32 it used to be.
const BRX_WALK_LIMIT: usize = 1024;

/// The widest jump table [`brx_targets`] will believe in.
const MAX_BRX_ARMS: usize = 256;

/// The targets a `brx` can reach, read out of the jump table its shader
/// compiler put in a constant bank.
///
/// A `switch` lowers to four things: clamp the selector to the last arm,
/// scale it to a word index, load that entry of the table, and branch to it.
/// The clamp is the only thing in the binary that records how many arms the
/// `switch` had, so the walk has to reach it — and it is the one part the
/// scheduler is free to hoist far away from the rest, because it depends on
/// nothing but the selector. In the Home Menu's fragment shaders it sits 36
/// instruction slots ahead of the branch, with the other three packed
/// together just behind it.
///
/// So this walks the selector's *definitions* rather than a window of
/// adjacent instructions: find what wrote the register the `brx` reads, then
/// what wrote that instruction's input, and so on until the clamp. Each step
/// takes the nearest preceding write, which is a use-def chain only if the
/// preceding code is the code that runs — an assumption this makes and cannot
/// check, since the control-flow graph is what is being decoded. What keeps
/// it honest is that the chain must be exactly this idiom: anything else
/// writing a register the chain is following gives up, and so does a write
/// that is predicated, because then the register's value depends on which
/// path reached it. Giving up returns `None` and leaves the caller to fall
/// through — a guess at a table's length reads whatever follows it as code.
fn brx_targets(
    decoded: &BTreeMap<u32, Instruction>,
    at: u32,
    base: u32,
    reg: u8,
    consts: &mut dyn FnMut(u8, u32) -> Result<u32>,
) -> Option<Vec<u32>> {
    // The register whose definition the walk is looking for, rewritten at
    // each step to the input of the instruction that defined it.
    let mut selector = reg;
    let mut table: Option<(u8, i32)> = None;
    let mut arms: Option<usize> = None;

    for insn in decoded.range(..at).rev().take(BRX_WALK_LIMIT).map(|(_, insn)| insn) {
        if !interp::writes(&insn.op).contains(&selector) {
            continue;
        }
        if !insn.pred.is_always() {
            return None;
        }
        match insn.op {
            Op::Ldc { bank, offset, idx, size: isa::MemSize::B32, .. } if table.is_none() => {
                table = Some((bank, offset));
                selector = idx;
            }
            // Scaling the arm number to a byte offset into the table.
            Op::Shl { a, b: isa::Operand::Imm(2), .. } if table.is_some() => {
                selector = a;
            }
            Op::Mov { src: isa::Operand::Reg(src), .. } if table.is_some() => {
                selector = src;
            }
            // The clamp bounds the table. `imnmx` on `PT` is `min`; on
            // anything else it is a per-lane pick between min and max, which
            // bounds nothing. The table has one more entry than the immediate.
            Op::Imnmx { b: isa::Operand::Imm(n), pred, .. }
                if table.is_some() && pred.is_always() =>
            {
                arms = Some(n as usize + 1);
                break;
            }
            _ => return None,
        }
    }

    let (table_bank, table_offset) = table?;
    // A `switch` this wide is not something a shader compiler emits; a match
    // that produces one has recognised the wrong `imnmx`.
    let arms = arms.filter(|&n| n <= MAX_BRX_ARMS)?;
    // Stop at the first entry that cannot be read rather than discarding the
    // ones that could: a table that runs past the end of its bank means the
    // arm count is wrong, and the targets already resolved are still real.
    Some(
        (0..arms)
            .map_while(|i| {
                let offset = table_offset.wrapping_add(i as i32 * 4);
                let entry = consts(table_bank, u32::try_from(offset).ok()?).ok()?;
                Some(align_slot(base.wrapping_add(entry)))
            })
            .collect(),
    )
}

/// Decode a program by walking its control-flow graph from `ENTRY_OFFSET`.
/// `read` fetches the 8-byte word at a byte offset; it is fallible because a
/// real one reads guest memory, and a program that runs off the end of what
/// is mapped is a decode error rather than a panic.
pub fn decode_program_with(read: &mut dyn FnMut(u32) -> Result<u64>) -> Result<Program> {
    decode_program_with_consts(read, &mut |_, _| {
        Err(Error::Gpu("shader: no constant banks bound for this decode".into()))
    })
}

/// [`decode_program_with`], plus the constant-bank reader that resolves a
/// `brx`'s jump table. `consts(bank, byte_offset)` reads one word of a bound
/// constant buffer.
pub fn decode_program_with_consts(
    read: &mut dyn FnMut(u32) -> Result<u64>,
    consts: &mut dyn FnMut(u8, u32) -> Result<u32>,
) -> Result<Program> {
    let mut decoded: BTreeMap<u32, Instruction> = BTreeMap::new();
    let mut indirect: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
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
                // `brx` reaches its arms only through a jump table in a
                // constant bank. The linear walk finds an arm only when the
                // arm before it falls through, and a `switch` whose arms all
                // end in `brk` or `bra` has none that do — the Home Menu's
                // instanced-quad vertex shader is one, and every one of its
                // 222 draws stopped on an arm no path had decoded.
                Op::Brx { base, reg } => {
                    if let Some(targets) = brx_targets(&decoded, offset, base, reg, consts) {
                        for &target in &targets {
                            push(target, &mut worklist, &mut queued);
                        }
                        indirect.insert(offset, targets);
                    }
                    true
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
    let mut program = Program { indirect, ..Program::default() };
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
    use isa::{FMod, FmulScale, MemSize, MufuOp, Operand, TexDim, RZ};
    use crate::Error;

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
                scale: FmulScale::None,
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
            Op::Texs { dst: 0, dst2: 2, coords: [0, 1, RZ], handle: 0x1a4, dim: TexDim::T2d, mask: [true, true, true, true] }
        );
        assert_eq!(program.insns.last().unwrap().op, Op::Exit);
    }

    /// The Home Menu's instanced-quad vertex shader, from the `brx` that
    /// picks a corner's texture coordinate through the three arms it selects
    /// between — transcribed word for word out of a live qlaunch run, along
    /// with the jump table its `c1` held.
    fn brx_switch_fixture() -> (Vec<u8>, [u32; 3]) {
        let mut bytes = vec![0u8; 0x300];
        // Entry: jump straight to the run that sets up the switch, so the
        // fixture keeps the real program's offsets (and so its real
        // displacements) without carrying the 190 instructions before it.
        bytes[8..16].copy_from_slice(&word(0x2f80000f, 0xe2400000)); // bra 0x308
        bytes.extend(block(
            (0xfec007f6, 0x001fd000),
            (0xfff70c0c, 0x1c0fffff), // iadd r12, r12, -1
            (0x00270c0c, 0x38200380), // imnmx r12, r12, 2
            (0x00270c0c, 0x38480000), // shl r12, r12, 2
        ));
        bytes.extend(block(
            (0xffa0073f, 0x001fc002),
            (0x0c070c0c, 0xef940010), // ld r12, c1[0xc0 + r12]
            (0xcc870c0f, 0xe2500fff), // brx r12, -0x338
            (0x0017000a, 0x5c980780), // mov r10, r1      <- arm 0
        ));
        bytes.extend(block(
            (0xfe0007fd, 0x001ff400),
            (0x0007000f, 0xe3400000), // brk
            (0x0027000a, 0x5c980780), // mov r10, r2      <- arm 1
            (0x0007000f, 0xe3400000), // brk
        ));
        bytes.extend(block(
            (0xffa007f0, 0x003fc000),
            (0x0037000a, 0x5c980780), // mov r10, r3      <- arm 2
            (0x0007000f, 0xe3400000), // brk
            (0x00070f00, 0x50b00000), // nop (padding)
        ));
        (bytes, [0x338, 0x350, 0x360])
    }

    fn decode_with_table(bytes: &[u8], table: [u32; 3]) -> Result<Program> {
        decode_program_with_consts(
            &mut |offset: u32| {
                let start = offset as usize;
                bytes
                    .get(start..start + 8)
                    .map(|w| u64::from_le_bytes(w.try_into().expect("8 bytes")))
                    .ok_or_else(|| Error::Gpu(format!("past the end at {offset:#x}")))
            },
            &mut |bank: u8, offset: u32| {
                let index = (offset as usize).checked_sub(192).map(|d| d / 4);
                match (bank, index.and_then(|i| table.get(i))) {
                    (1, Some(&entry)) => Ok(entry),
                    _ => Err(Error::Gpu(format!("no c{bank}[{offset:#x}]"))),
                }
            },
        )
    }

    #[test]
    fn a_brx_reaches_the_arms_its_jump_table_names() {
        // Every arm of this switch ends in `brk`, so nothing falls through
        // into the next one: the linear walk finds arm 0 and stops. Arms 1
        // and 2 exist only in the table, and each of the Home Menu's 222
        // textured draws stopped on one of them.
        let (bytes, table) = brx_switch_fixture();
        let program = decode_with_table(&bytes, table).unwrap();

        for (arm, offset) in [(1u8, 0x338u32), (2, 0x350), (3, 0x368)] {
            let index = program
                .index_of(offset)
                .unwrap_or_else(|| panic!("arm at {offset:#x} was never decoded"));
            assert_eq!(program.insns[index].op, Op::Mov { dst: 10, src: Operand::Reg(arm) });
        }
    }

    #[test]
    fn a_brx_base_is_not_rounded_onto_an_instruction_slot() {
        // Only the *sum* of the base and a table entry is a target. This
        // base is zero, which is a multiple of 32 — rounding it up to the
        // first real instruction slot would add 8 to every arm and land two
        // of the three on the `brk` after the arm instead of the arm itself.
        let (bytes, table) = brx_switch_fixture();
        let program = decode_with_table(&bytes, table).unwrap();
        let brx = program.index_of(0x330).expect("the brx itself");
        assert_eq!(program.insns[brx].op, Op::Brx { base: 0, reg: 12 });
    }

    #[test]
    fn a_brx_whose_table_cannot_be_read_still_decodes_what_falls_through() {
        // No constant banks bound — the decode must not fail, it must just
        // stop knowing where the arms are.
        let (bytes, _) = brx_switch_fixture();
        let program = decode_program(&bytes).unwrap();
        assert!(program.index_of(0x338).is_some(), "arm 0 falls through");
        assert!(program.index_of(0x350).is_none(), "arm 1 is only in the table");
    }

    /// `nop`, which writes nothing and falls through — filler for putting a
    /// measured distance between two instructions.
    const NOP: (u32, u32) = (0x00070f00, 0x50b00000);
    /// `brk`, which ends an arm.
    const BRK: (u32, u32) = (0x0007000f, 0xe3400000);

    /// `brx r12` at `pc`, encoded so its base is zero and every arm comes out
    /// of the jump table. The displacement is pc-relative and 24 bits wide at
    /// bit 20, so it has to be rebuilt for each position rather than reused.
    fn brx_at(pc: u32) -> (u32, u32) {
        let field = 0u32.wrapping_sub(pc + 8) & 0xff_ffff;
        (
            (0xcc870c0f & !(0xfff << 20)) | ((field & 0xfff) << 20),
            (0xe2500fff & !0xfffu32) | (field >> 12),
        )
    }

    /// The same `switch` as [`brx_switch_fixture`], with `gap` blocks of
    /// filler between the clamp and the rest of the idiom, and `between`
    /// spliced in just before the scale. Returns the bytes and the jump table
    /// its `c1` holds.
    fn brx_switch_spread(gap: u32, between: Option<(u32, u32)>) -> (Vec<u8>, [u32; 3]) {
        let mut bytes = block(
            (0, 0),
            (0xfff70c0c, 0x1c0fffff), // iadd r12, r12, -1
            (0x00270c0c, 0x38200380), // imnmx r12, r12, 2
            NOP,
        );
        for _ in 0..gap {
            bytes.extend(block((0, 0), NOP, NOP, NOP));
        }
        if let Some(insn) = between {
            bytes.extend(block((0, 0), insn, NOP, NOP));
        }
        let idiom = bytes.len() as u32;
        bytes.extend(block(
            (0, 0),
            (0x00270c0c, 0x38480000), // shl r12, r12, 2
            (0x0c070c0c, 0xef940010), // ld r12, c1[0xc0 + r12]
            brx_at(idiom + 24),
        ));
        let arms = bytes.len() as u32;
        bytes.extend(block(
            (0, 0),
            (0x0017000a, 0x5c980780), // mov r10, r1   <- arm 0, falls through
            BRK,
            (0x0027000a, 0x5c980780), // mov r10, r2   <- arm 1
        ));
        bytes.extend(block(
            (0, 0),
            BRK,
            (0x0037000a, 0x5c980780), // mov r10, r3   <- arm 2
            BRK,
        ));
        (bytes, [arms + 8, arms + 24, arms + 48])
    }

    #[test]
    fn a_clamp_hoisted_far_from_its_brx_is_still_found() {
        // The scheduler is free to move the clamp, because it depends on
        // nothing but the selector: in the Home Menu's fragment shaders it
        // ends up 36 instruction slots ahead of the branch, and a walk that
        // looked at a fixed window of the 32 instructions before the `brx`
        // read every one of them without ever seeing it. 12 blocks of filler
        // puts 40 instructions in the way, which is more than that window.
        let (bytes, table) = brx_switch_spread(12, None);
        let program = decode_with_table(&bytes, table).unwrap();

        for (arm, offset) in [(1u8, table[0]), (2, table[1]), (3, table[2])] {
            let index = program
                .index_of(offset)
                .unwrap_or_else(|| panic!("arm at {offset:#x} was never decoded"));
            assert_eq!(program.insns[index].op, Op::Mov { dst: 10, src: Operand::Reg(arm) });
        }
    }

    #[test]
    fn a_predicated_write_to_the_selector_abandons_the_table() {
        // `@p0 shl r12, r12, 2` — an instruction the walk would otherwise
        // step straight through, except that only some lanes take it. After
        // it the selector holds two different values at once, and the clamp
        // behind it bounds only one of them, so the arm count read from it
        // would be a guess.
        let (bytes, table) = brx_switch_spread(1, Some((0x00200c0c, 0x38480000)));
        let program = decode_with_table(&bytes, table).unwrap();
        assert!(program.index_of(table[0]).is_some(), "arm 0 falls through");
        assert!(program.index_of(table[1]).is_none(), "arm 1 is only in the table");
    }

    #[test]
    fn an_unrecognised_write_to_the_selector_abandons_the_table() {
        // `iadd r12, r12, -1` is a real part of this switch's lowering — but
        // *behind* the clamp, where it changes nothing. In front of it the
        // walk cannot tell whether the clamp still bounds what the branch
        // reads, so it stops rather than assuming it does.
        let (bytes, table) = brx_switch_spread(1, Some((0xfff70c0c, 0x1c0fffff)));
        let program = decode_with_table(&bytes, table).unwrap();
        assert!(program.index_of(table[1]).is_none(), "arm 1 is only in the table");
    }

    #[test]
    fn only_the_varyings_a_program_interpolates_are_listed() {
        let mut program = Program::default();
        for (offset, at) in [(0x7cu16, 8u32), (0xc4, 16), (0x80, 24), (0xc0, 40), (0x80, 48)] {
            program.offsets.push(at);
            program.insns.push(Instruction {
                pred: isa::Pred::ALWAYS,
                op: Op::Ipa { dst: 0, offset, mul: None, perspective: true, sat: false },
            });
        }
        // 0x7c is `1/w`, not a varying; 0x80 is slot 0 twice; 0xc0/0xc4 are
        // both slot 4.
        let ops: Vec<Op> = program.insns.iter().map(|i| i.op).collect();
        assert_eq!(interpolated_slots(&ops), &[0, 4]);
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
