//! Decoding a complete Maxwell shader binary (single instructions are in [`isa`]).
//!
//! Instructions come in 32-byte blocks: one `sched` control word, then three
//! instructions. The program end is found by walking the control-flow graph.

pub mod cfg;
pub mod compiled;
pub mod interp;
pub mod isa;
pub mod wgsl;

pub use isa::{Instruction, Op};

use crate::{Error, Result};
use std::collections::{BTreeMap, HashSet};

/// Programs bound during a run, recorded on demand for diagnostics.
pub mod uses {
    use super::wgsl::Stage;
    use super::Program;
    use std::cell::RefCell;

    thread_local! {
        static BOUND: RefCell<Option<Vec<(Stage, u64, Program)>>> = const { RefCell::new(None) };
    }

    pub fn record() {
        BOUND.with(|bound| *bound.borrow_mut() = Some(Vec::new()));
    }

    /// Note the program bound as `stage` for the draw about to run.
    pub fn note(stage: Stage, addr: u64, program: &Program) {
        BOUND.with(|bound| {
            if let Some(list) = bound.borrow_mut().as_mut() {
                list.push((stage, addr, program.clone()));
            }
        });
    }

    /// Everything noted since [`record`], in draw order; stops recording.
    pub fn take() -> Vec<(Stage, u64, Program)> {
        BOUND
            .with(|bound| bound.borrow_mut().take())
            .unwrap_or_default()
    }
}

/// Cap on decoded instructions per program.
const MAX_INSTRUCTIONS: usize = 4096;

/// First real instruction: slot 1 of the first block, after its `sched` word.
pub const ENTRY_OFFSET: u32 = 8;

/// The generic varying slots `insns`' `ipa`s read, ascending.
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

/// Generic attribute slots (four components each).
const GENERIC_ATTR_BASE: u16 = 0x80;
const GENERIC_ATTR_END: u16 = 0x280;
const GENERIC_ATTR_STRIDE: usize = 0x10;

/// Instructions in ascending address order, each with its byte offset.
#[derive(Debug, Clone, Default)]
pub struct Program {
    pub insns: Vec<Instruction>,
    pub offsets: Vec<u32>,
    /// Targets of each `brx`, keyed by the `brx`'s byte offset.
    pub indirect: BTreeMap<u32, Vec<u32>>,
    /// The Shader Program Header this program was preceded by, if any.
    pub header: Option<ProgramHeader>,
}

/// The 0x50-byte Shader Program Header. Only the fragment output map is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProgramHeader {
    /// `omap.target`: a nibble per render target, one bit per component.
    pub omap_target: u32,
    /// The program writes a coverage mask after its colours.
    pub omap_sample_mask: bool,
    /// The program writes a depth value after that.
    pub omap_depth: bool,
}

/// Word index of `omap.target`; its flags follow.
const OMAP_TARGET_WORD: u64 = 18;

impl ProgramHeader {
    /// Register holding `component` of render target `rt`, or `None` if unwritten.
    ///
    /// Targets with no writes are skipped; disabled components within a written
    /// target still take a register.
    pub fn fragment_output_reg(&self, rt: u32, component: u32) -> Option<u8> {
        let mut reg = 0u32;
        for target in 0..8 {
            let bits = (self.omap_target >> (target * 4)) & 0xF;
            if bits == 0 {
                continue;
            }
            for c in 0..4 {
                let enabled = bits >> c & 1 != 0;
                if target == rt && c == component {
                    return enabled.then_some(reg as u8);
                }
                reg += 1;
            }
        }
        None
    }

    pub fn writes_any_color(&self) -> bool {
        self.omap_target != 0
    }
}

/// Where one `texs` instruction's results land.
#[derive(Debug, Clone)]
pub struct TexsWrites {
    pub at: usize,
    /// Per destination register: `(register, what lands in it, instruction index
    /// the write must land before)`.
    pub writes: Vec<(u8, isa::TexsStore, usize)>,
}

impl Program {
    pub fn len(&self) -> usize {
        self.insns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.insns.is_empty()
    }

    pub fn index_of(&self, byte_offset: u32) -> Option<usize> {
        self.offsets.binary_search(&byte_offset).ok()
    }
}

/// Words to scan for `exit` before giving up.
pub const MAX_PROGRAM_WORDS: u64 = 8192;

/// Decode a shader program from GPU memory, stripping `sched` words.
///
/// Mesa-compiled binaries carry a 0x50-byte header; if slot 1 does not decode,
/// the decode is retried past it.
const MESA_SHADER_HEADER_BYTES: u64 = 0x50;

/// Decode a bound shader program out of guest memory.
pub fn decode_program_from_memory(
    ctx: &crate::gpu::exec::ExecCtx,
    addr: u64,
    bindings: &dyn Fn(u8) -> Option<(u64, u32)>,
) -> Result<Program> {
    decode_program_from_memory_recording(ctx, addr, bindings).map(|(program, _)| program)
}

/// Guest memory a decode read, so a cached result can be validated.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DecodeReads {
    /// Pages of the program's words, ascending and deduplicated, as GPU virtual
    /// addresses (translate through `ExecCtx::vmm` before watching).
    pub pages: Vec<u64>,
    /// Constant-buffer words read (only for `brx` jump tables), with their values.
    pub consts: Vec<(u64, u32)>,
}

impl DecodeReads {
    /// Whether guest memory still holds what this decode was built from. Only
    /// constant words are re-read; program pages are expected to be watched.
    pub fn constants_unchanged(&self, ctx: &crate::gpu::exec::ExecCtx) -> bool {
        self.consts
            .iter()
            .all(|&(at, value)| ctx.read_u32(at).is_ok_and(|now| now == value))
    }
}

/// [`decode_program_from_memory`], also returning [`DecodeReads`].
pub fn decode_program_from_memory_recording(
    ctx: &crate::gpu::exec::ExecCtx,
    addr: u64,
    bindings: &dyn Fn(u8) -> Option<(u64, u32)>,
) -> Result<(Program, DecodeReads)> {
    let mut pages: Vec<u64> = Vec::new();
    let mut consts: Vec<(u64, u32)> = Vec::new();
    let note = |pages: &mut Vec<u64>, at: u64| {
        let page = at >> crate::mem::PAGE_BITS;
        if !pages.contains(&page) {
            pages.push(page);
        }
    };
    let first_real_word = ctx.read_u64(addr + 8)?;
    note(&mut pages, addr + 8);
    let header_at =
        matches!(isa::decode(first_real_word).op, Op::Unimplemented { .. }).then_some(addr);
    let header = match header_at {
        Some(at) => {
            let target = ctx.read_u32(at + OMAP_TARGET_WORD * 4)?;
            let flags = ctx.read_u32(at + (OMAP_TARGET_WORD + 1) * 4)?;
            note(&mut pages, at + OMAP_TARGET_WORD * 4);
            note(&mut pages, at + (OMAP_TARGET_WORD + 1) * 4);
            Some(ProgramHeader {
                omap_target: target,
                omap_sample_mask: flags & 1 != 0,
                omap_depth: flags >> 1 & 1 != 0,
            })
        }
        None => None,
    };
    let addr = match header_at {
        Some(at) => at + MESA_SHADER_HEADER_BYTES,
        None => addr,
    };
    let limit = MAX_PROGRAM_WORDS * 8;
    let mut program = decode_program_with_consts(
        &mut |offset: u32| {
            if u64::from(offset) >= limit {
                return Err(Error::Gpu(format!(
                    "shader: program read at {offset:#x} is past the {limit:#x}-byte cap"
                )));
            }
            note(&mut pages, addr + u64::from(offset));
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
            let at = base + u64::from(offset);
            let value = ctx.read_u32(at)?;
            consts.push((at, value));
            Ok(value)
        },
    )
    .inspect(|program| {
        // `TRACE_SHADER=1` prints every decoded program in walk order.
        if crate::trace::enabled(crate::trace::Trace::Shader) {
            crate::traceln!(
                "[shader] program at {addr:#x}, {} instructions",
                program.offsets.len()
            );
            for (i, &off) in program.offsets.iter().enumerate() {
                let raw = ctx.read_u64(addr + u64::from(off)).unwrap_or(0);
                crate::traceln!("  {off:#06x}: {raw:016x} {:?}", program.insns[i]);
            }
        }
    })?;
    program.header = header;
    if crate::trace::enabled(crate::trace::Trace::Sph) {
        crate::traceln!("[sph] program at {addr:#x} {header:?}");
    }
    pages.sort_unstable();
    Ok((program, DecodeReads { pages, consts }))
}

/// Whether `offset` is an instruction rather than a `sched` word.
fn is_instruction_slot(offset: u32) -> bool {
    !(offset / 8).is_multiple_of(4)
}

/// A branch target rounded forward off a `sched` word onto the next instruction slot.
pub fn align_slot(offset: u32) -> u32 {
    if offset.is_multiple_of(32) {
        offset + 8
    } else {
        offset
    }
}

/// The next instruction slot after `offset`.
pub fn next_slot(offset: u32) -> u32 {
    let next = offset + 8;
    if is_instruction_slot(next) {
        next
    } else {
        next + 8
    }
}

/// How many definitions back a jump-table walk will look.
const BRX_WALK_LIMIT: usize = 1024;

const MAX_BRX_ARMS: usize = 256;

/// The targets a `brx` can reach, read from its jump table in a constant bank.
///
/// Walks the selector's use-def chain back to the `imnmx` clamp that gives the
/// table length. Any unexpected or predicated write gives up with `None`.
fn brx_targets(
    decoded: &BTreeMap<u32, Instruction>,
    at: u32,
    base: u32,
    reg: u8,
    consts: &mut dyn FnMut(u8, u32) -> Result<u32>,
) -> Option<Vec<u32>> {
    // Register whose definition the walk is looking for.
    let mut selector = reg;
    let mut table: Option<(u8, i32)> = None;
    let mut arms: Option<usize> = None;

    for insn in decoded
        .range(..at)
        .rev()
        .take(BRX_WALK_LIMIT)
        .map(|(_, insn)| insn)
    {
        if !interp::writes(&insn.op).contains(&selector) {
            continue;
        }
        if !insn.pred.is_always() {
            return None;
        }
        match insn.op {
            Op::Ldc {
                bank,
                offset,
                idx,
                size: isa::MemSize::B32,
                ..
            } if table.is_none() => {
                table = Some((bank, offset));
                selector = idx;
            }
            Op::Shl {
                a,
                b: isa::Operand::Imm(2),
                ..
            } if table.is_some() => {
                selector = a;
            }
            Op::Mov {
                src: isa::Operand::Reg(src),
                ..
            } if table.is_some() => {
                selector = src;
            }
            // `imnmx` on `PT` is `min`; the table has one more entry than the immediate.
            Op::Imnmx {
                b: isa::Operand::Imm(n),
                pred,
                ..
            } if table.is_some() && pred.is_always() => {
                arms = Some(n as usize + 1);
                break;
            }
            _ => return None,
        }
    }

    let (table_bank, table_offset) = table?;
    // Wider than any real `switch`: the wrong `imnmx` was matched.
    let arms = arms.filter(|&n| n <= MAX_BRX_ARMS)?;
    // Keep the entries read before the first unreadable one.
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
/// `read` fetches the 8-byte word at a byte offset.
pub fn decode_program_with(read: &mut dyn FnMut(u32) -> Result<u64>) -> Result<Program> {
    decode_program_with_consts(read, &mut |_, _| {
        Err(Error::Gpu(
            "shader: no constant banks bound for this decode".into(),
        ))
    })
}

/// [`decode_program_with`], plus `consts(bank, byte_offset)` for `brx` jump tables.
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
            let falls_through = match insn.op {
                Op::Exit | Op::Kil => !insn.pred.is_always(),
                Op::Bra { target } => {
                    push(target, &mut worklist, &mut queued);
                    !insn.pred.is_always()
                }
                // `brx` arms are only reachable through the jump table.
                Op::Brx { base, reg } => {
                    if let Some(targets) = brx_targets(&decoded, offset, base, reg, consts) {
                        for &target in &targets {
                            push(target, &mut worklist, &mut queued);
                        }
                        indirect.insert(offset, targets);
                    }
                    true
                }
                // `sync`/`brk`/`cont` targets were queued by `ssy`/`pbk`/`pcnt`.
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
    let mut program = Program {
        indirect,
        ..Program::default()
    };
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
                Error::Gpu(format!(
                    "shader: program read at {offset:#x} is past its end"
                ))
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::testing::{block, solid_fragment_shader, word};
    use crate::Error;
    use isa::{FMod, FmulScale, MemSize, MufuOp, Operand, RZ};

    fn decode_recording() -> (u64, DecodeReads) {
        use crate::gpu::exec::{ExecCtx, GpuStats};
        use crate::gpu::syncpt::Host1x;
        use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
        use crate::mem::Memory;
        let mut mem = Memory::new();
        mem.map_zero(0x7000_0000, 0x4000).unwrap();
        let mut vmm = AddressSpace::new();
        let at = vmm
            .map(0x7000_0000, 0x4000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        let mut host1x = Host1x::new();
        let mut stats = GpuStats::default();
        let mut ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        for (i, chunk) in solid_fragment_shader()
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
        {
            let word = u32::from_le_bytes(*chunk);
            ctx.write_u32(at + i as u64 * 4, word).unwrap();
        }
        let (program, reads) = decode_program_from_memory_recording(&ctx, at, &|_| None).unwrap();
        assert!(!program.insns.is_empty(), "decoded something");
        (at, reads)
    }

    #[test]
    fn a_decode_reports_the_pages_it_read_so_a_cache_can_watch_them() {
        // The pages read are GPU virtual pages.
        let (at, reads) = decode_recording();
        assert_eq!(
            reads.pages,
            vec![at >> crate::mem::PAGE_BITS],
            "the one page this program lives on"
        );
        // No `brx`, so no constant words to re-check.
        assert!(reads.consts.is_empty(), "no constant words consumed");
    }

    fn ops(program: &Program) -> Vec<Op> {
        program.insns.iter().map(|i| i.op).collect()
    }

    #[test]
    fn strips_sched_words_and_stops_at_exit() {
        // solid.frag from the envydis capture, trailing padding included.
        let program = decode_program(&solid_fragment_shader()).unwrap();
        assert_eq!(
            ops(&program),
            vec![
                Op::Ipa {
                    dst: 0,
                    offset: 0x7c,
                    mul: None,
                    perspective: false,
                    sat: false,
                    centroid: false
                },
                Op::Mufu {
                    dst: 3,
                    src: 0,
                    sm: FMod::NONE,
                    op: MufuOp::Rcp,
                    sat: false
                },
                Op::Ipa {
                    dst: 0,
                    offset: 0x80,
                    mul: Some(3),
                    perspective: true,
                    sat: false,
                    centroid: false
                },
                Op::Ipa {
                    dst: 1,
                    offset: 0x84,
                    mul: Some(3),
                    perspective: true,
                    sat: false,
                    centroid: false
                },
                Op::Ipa {
                    dst: 2,
                    offset: 0x88,
                    mul: Some(3),
                    perspective: true,
                    sat: false,
                    centroid: false
                },
                Op::Ipa {
                    dst: 3,
                    offset: 0x8c,
                    mul: Some(3),
                    perspective: true,
                    sat: false,
                    centroid: false
                },
                Op::Exit,
            ]
        );
    }

    #[test]
    fn mvp_vertex_shader_fixture_decodes_instruction_for_instruction() {
        // mvp.vert in full, from the envydis capture.
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
        assert_eq!(
            program.insns[0].op,
            Op::Ld {
                dst: 0,
                offset: 0x80,
                idx: RZ,
                size: MemSize::B128
            }
        );
        assert_eq!(
            program.insns[1].op,
            Op::Fmul {
                dst: 4,
                a: 0,
                b: Operand::Const {
                    bank: 2,
                    offset: 0x0
                },
                bm: FMod::NONE,
                ftz: true,
                sat: false,
                scale: FmulScale::None,
            }
        );
        assert_eq!(
            program.insns[5].op,
            Op::Ffma {
                dst: 4,
                a: 1,
                b: Operand::Const {
                    bank: 2,
                    offset: 0x10
                },
                bneg: false,
                c: Operand::Reg(4),
                cneg: false,
                ftz: true,
                sat: false
            }
        );
        assert_eq!(
            program.insns[17].op,
            Op::St {
                offset: 0x70,
                idx: RZ,
                src: 0,
                size: MemSize::B128
            }
        );
        assert_eq!(
            program.insns[18].op,
            Op::Ld {
                dst: 0,
                offset: 0x90,
                idx: RZ,
                size: MemSize::B128
            }
        );
        assert_eq!(
            program.insns[19].op,
            Op::St {
                offset: 0x80,
                idx: RZ,
                src: 0,
                size: MemSize::B128
            }
        );
        assert_eq!(program.insns[20].op, Op::Exit);
    }

    #[test]
    fn the_output_map_hands_out_a_register_per_component_of_a_written_target() {
        let all = ProgramHeader {
            omap_target: 0xF,
            ..ProgramHeader::default()
        };
        for c in 0..4 {
            assert_eq!(all.fragment_output_reg(0, c), Some(c as u8));
        }

        // A disabled component still takes its register.
        let no_red = ProgramHeader {
            omap_target: 0xE,
            ..ProgramHeader::default()
        };
        assert_eq!(no_red.fragment_output_reg(0, 0), None);
        assert_eq!(no_red.fragment_output_reg(0, 1), Some(1));
        assert_eq!(no_red.fragment_output_reg(0, 3), Some(3));

        // An unwritten target is skipped whole.
        let second_only = ProgramHeader {
            omap_target: 0xF0,
            ..ProgramHeader::default()
        };
        assert_eq!(second_only.fragment_output_reg(0, 0), None);
        assert_eq!(second_only.fragment_output_reg(1, 0), Some(0));
        assert_eq!(second_only.fragment_output_reg(1, 3), Some(3));

        assert!(!ProgramHeader::default().writes_any_color());
        assert!(all.writes_any_color());
    }

    /// The Home Menu's instanced-quad vertex shader around its `brx`, with the jump table `c1` held.
    fn brx_switch_fixture() -> (Vec<u8>, [u32; 3]) {
        let mut bytes = vec![0u8; 0x300];
        // Entry jumps to the switch setup, keeping the real program's offsets.
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
        // Every arm ends in `brk`; arms 1 and 2 are only reachable through the table.
        let (bytes, table) = brx_switch_fixture();
        let program = decode_with_table(&bytes, table).unwrap();

        for (arm, offset) in [(1u8, 0x338u32), (2, 0x350), (3, 0x368)] {
            let index = program
                .index_of(offset)
                .unwrap_or_else(|| panic!("arm at {offset:#x} was never decoded"));
            assert_eq!(
                program.insns[index].op,
                Op::Mov {
                    dst: 10,
                    src: Operand::Reg(arm)
                }
            );
        }
    }

    #[test]
    fn a_brx_base_is_not_rounded_onto_an_instruction_slot() {
        // Targets are base + entry; the zero base must not be rounded up on its own.
        let (bytes, table) = brx_switch_fixture();
        let program = decode_with_table(&bytes, table).unwrap();
        let brx = program.index_of(0x330).expect("the brx itself");
        assert_eq!(program.insns[brx].op, Op::Brx { base: 0, reg: 12 });
    }

    #[test]
    fn a_brx_whose_table_cannot_be_read_still_decodes_what_falls_through() {
        // No constant banks bound: decode must still succeed.
        let (bytes, _) = brx_switch_fixture();
        let program = decode_program(&bytes).unwrap();
        assert!(program.index_of(0x338).is_some(), "arm 0 falls through");
        assert!(
            program.index_of(0x350).is_none(),
            "arm 1 is only in the table"
        );
    }

    /// `nop`, used as filler.
    const NOP: (u32, u32) = (0x00070f00, 0x50b00000);
    const BRK: (u32, u32) = (0x0007000f, 0xe3400000);

    /// `brx r12` at `pc`, base zero; the displacement is pc-relative so it is rebuilt per position.
    fn brx_at(pc: u32) -> (u32, u32) {
        let field = 0u32.wrapping_sub(pc + 8) & 0xff_ffff;
        (
            (0xcc870c0f & !(0xfff << 20)) | ((field & 0xfff) << 20),
            (0xe2500fff & !0xfffu32) | (field >> 12),
        )
    }

    /// [`brx_switch_fixture`]'s switch with `gap` filler blocks after the clamp and
    /// `between` spliced in before the scale. Returns the bytes and the jump table.
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
        // 12 filler blocks put 40 instructions between the clamp and the `brx`.
        let (bytes, table) = brx_switch_spread(12, None);
        let program = decode_with_table(&bytes, table).unwrap();

        for (arm, offset) in [(1u8, table[0]), (2, table[1]), (3, table[2])] {
            let index = program
                .index_of(offset)
                .unwrap_or_else(|| panic!("arm at {offset:#x} was never decoded"));
            assert_eq!(
                program.insns[index].op,
                Op::Mov {
                    dst: 10,
                    src: Operand::Reg(arm)
                }
            );
        }
    }

    #[test]
    fn a_predicated_write_to_the_selector_abandons_the_table() {
        // A predicated write to the selector makes the arm count unknowable.
        let (bytes, table) = brx_switch_spread(1, Some((0x00200c0c, 0x38480000)));
        let program = decode_with_table(&bytes, table).unwrap();
        assert!(program.index_of(table[0]).is_some(), "arm 0 falls through");
        assert!(
            program.index_of(table[1]).is_none(),
            "arm 1 is only in the table"
        );
    }

    #[test]
    fn an_unrecognised_write_to_the_selector_abandons_the_table() {
        // A write between the clamp and the branch stops the walk.
        let (bytes, table) = brx_switch_spread(1, Some((0xfff70c0c, 0x1c0fffff)));
        let program = decode_with_table(&bytes, table).unwrap();
        assert!(
            program.index_of(table[1]).is_none(),
            "arm 1 is only in the table"
        );
    }

    #[test]
    fn only_the_varyings_a_program_interpolates_are_listed() {
        let mut program = Program::default();
        for (offset, at) in [
            (0x7cu16, 8u32),
            (0xc4, 16),
            (0x80, 24),
            (0xc0, 40),
            (0x80, 48),
        ] {
            program.offsets.push(at);
            program.insns.push(Instruction {
                pred: isa::Pred::ALWAYS,
                op: Op::Ipa {
                    dst: 0,
                    offset,
                    mul: None,
                    perspective: true,
                    sat: false,
                    centroid: false,
                },
            });
        }
        // 0x7c is `1/w`, not a varying; 0x80 is slot 0 twice; 0xc0/0xc4 are
        // both slot 4.
        let ops: Vec<Op> = program.insns.iter().map(|i| i.op).collect();
        assert_eq!(interpolated_slots(&ops), &[0, 4]);
    }

    #[test]
    fn a_program_that_never_ends_is_an_error_not_a_hang() {
        // Zero words fall through with nothing ending the path, so the walk runs off the buffer.
        let bytes = block((0, 0), (0, 0), (0, 0), (0, 0));
        assert!(decode_program(&bytes).is_err());
    }

    #[test]
    fn a_misaligned_program_is_an_error() {
        assert!(decode_program(&[0u8; 7]).is_err());
    }
}
