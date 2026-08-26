//! A block-translating JIT for the A64 core.
//!
//! The interpreter re-derives everything about an instruction every time it
//! runs it: which of the eight top-level groups it belongs to, which of that
//! group's forms it is, where its operand fields sit, what its immediate
//! decodes to. In a loop body that runs a million times, that work is done a
//! million times and produces the same answer every time.
//!
//! This module does it once. The first time control reaches an address, the
//! translator walks forward from it decoding instructions into [`Op`]s — the
//! operation with its operands already extracted, its immediates already
//! decoded, its PC-relative addresses already resolved — until it reaches
//! something that changes the PC. That run of ops plus its terminator is a
//! [`Block`], cached by entry address, and every later visit executes it with
//! no decoding at all.
//!
//! What this removes is decode, not dispatch — three quarters of where the
//! interpreter's time went (see [`super::Cpu::execute`]'s note on the group
//! dispatcher). It does not generate code.
//!
//! # Why not generate wasm
//!
//! Emitting a wasm module per translation unit and handing it to
//! `WebAssembly.Module` is a real JIT, and it is the next step for speed
//! rather than something the browser forbids. What stops it being *this*
//! step is the memory model, not the platform.
//!
//! A generated module can only address its own linear memory directly. Guest
//! memory cannot be that: this emulator is itself a wasm32 module, whose
//! linear memory caps at 4 GiB, and the guest address space is 4 GiB reaching
//! up to [`super::GUEST_SPACE_END`] — so an identity map does not fit, and
//! would give up the lazy soft-mapping that makes a multi-gigabyte heap cost
//! nothing until it is touched. [`crate::mem::Memory`] is a page table of
//! boxed 4 KiB pages with soft regions, read-only ranges and watchpoints over
//! it. Generated code would have to reach all of that through imported host
//! calls, one per guest load and store — which is most of what a block does,
//! and exactly what the codegen was supposed to make cheap.
//!
//! Two smaller things point the same way: compiling a module goes out through
//! JS, so a basic block is far too small a unit to pay for one, and generated
//! code would not exist in host builds — leaving the test suite and
//! `examples/jit_bench.rs` covering only one of the two engines. Flattening
//! the guest address space behind a base-plus-bounds check is the change that
//! has to come first.
//!
//! # Fidelity
//!
//! Every op executes the same helper the interpreter's decoder would have
//! called with the same arguments, so translated and interpreted execution are
//! the same computation. Anything the translator does not have an op for —
//! SIMD, floating point, the exclusive accessors, the encodings the
//! interpreter rejects as unallocated — becomes [`Op::Interpret`], which hands
//! the raw instruction word straight back to [`super::Cpu::execute`]. That
//! makes the translator's coverage a performance question rather than a
//! correctness one: a form it does not know is slower, never wrong.
//!
//! An op may only be a mid-block [`Op::Interpret`] if it cannot move the PC
//! anywhere but to the following instruction. Every A64 instruction that can
//! is in the branch/exception/system group (bits 28:25 = `101x`), so that
//! group is decoded as a terminator — except the `D503xxxx` hints and
//! barriers, which are no-ops, and the `MRS`/`MSR`/cache-maintenance forms,
//! which [`super::Cpu::system`] always retires to `next_pc`.
//!
//! # Staleness
//!
//! A block is only valid while the instructions it was built from are still
//! there. [`crate::mem::Memory`] records which pages have been translated out
//! of and reports the ones a store has landed on; [`Cpu::jit_block_at`] drains
//! that list before every lookup and drops the blocks translated from those
//! pages. A block never spans a page, so one page's worth of invalidation is
//! exact.

use super::bits::*;
use super::{Cpu, Result, RunReport, RECENT_LEN, SELF_RETURN_TRAMPOLINE, TIME_SLICE};
use crate::mem::{Memory, PAGE_SIZE};
use std::collections::HashMap;
use std::rc::Rc;

/// Longest run of instructions one block may cover. Longer blocks amortize the
/// per-block bookkeeping over more work, but they also make invalidation and
/// the step budget coarser, and past a basic block's typical length there is
/// nothing left to gain — straight-line runs of more than this between two
/// branches are rare in compiled code.
const MAX_BLOCK_OPS: usize = 64;

/// How many blocks the cache holds before it is dropped wholesale. A retail
/// title's hot code is a few thousand blocks; the cap is only there so a
/// program that walks endlessly over fresh code cannot grow the cache without
/// bound.
const MAX_BLOCKS: usize = 64 * 1024;

/// What a load/store does to its base register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Wb {
    /// No writeback: the access is at `base + offset`.
    None,
    /// Pre-index: the access is at `base + offset`, which is also written back.
    Pre,
    /// Post-index: the access is at `base`, and `base + offset` is written back.
    Post,
}

/// One translated instruction: what it does, with its operands already pulled
/// out of the encoding.
#[derive(Debug, Clone, Copy)]
pub(super) enum Op {
    /// A hint, barrier or PSTATE-immediate write the interpreter also retires
    /// with no effect.
    Nop,
    /// Not translated: run the original instruction through the interpreter.
    Interpret { insn: u32 },

    /// A value the translator already computed: `MOVZ`/`MOVN`, and the
    /// PC-relative `ADR`/`ADRP` whose result depends only on where the
    /// instruction is.
    MovConst { rd: u8, val: u64 },
    /// `MOVK`: replace the 16-bit field `mask` selects with `val`.
    MovK { rd: u8, mask: u64, val: u64 },

    AddSubImm { rd: u8, rn: u8, imm: u64, set_flags: bool, sub: bool, sf: bool },
    AddSubShifted { rd: u8, rn: u8, rm: u8, st: u8, sa: u8, set_flags: bool, sub: bool, sf: bool },
    AddSubExtended { rd: u8, rn: u8, rm: u8, option: u8, shift: u8, set_flags: bool, sub: bool, sf: bool },

    /// `AND`/`ORR`/`EOR`/`ANDS` with the bitmask immediate already decoded.
    LogicalImm { rd: u8, rn: u8, imm: u64, opc: u8, sf: bool },
    LogicalShifted { rd: u8, rn: u8, rm: u8, st: u8, sa: u8, opc: u8, invert: bool, sf: bool },

    /// `SBFM`/`BFM`/`UBFM` and the aliases built on them.
    Bitfield { rd: u8, rn: u8, opc: u8, immr: u8, imms: u8, sf: bool },
    Extr { rd: u8, rn: u8, rm: u8, imm: u8, sf: bool },

    /// `CSEL`/`CSINC`/`CSINV`/`CSNEG`.
    CondSel { rd: u8, rn: u8, rm: u8, cond: u8, else_inv: bool, else_inc: bool, sf: bool },
    /// `CCMP`/`CCMN`, register and immediate forms.
    CondCmp { rn: u8, rm: u8, imm: u8, cond: u8, nzcv: u8, sub: bool, is_imm: bool, sf: bool },

    /// `MADD`/`MSUB`.
    Madd { rd: u8, rn: u8, rm: u8, ra: u8, sub: bool, sf: bool },
    /// `SMADDL`/`SMSUBL`/`UMADDL`/`UMSUBL`: the 32x32 widening multiplies.
    MaddLong { rd: u8, rn: u8, rm: u8, ra: u8, sub: bool, signed: bool },
    /// `SMULH`/`UMULH`.
    Mulh { rd: u8, rn: u8, rm: u8, signed: bool },

    LoadStoreImm { rt: u8, rn: u8, sz: u8, opc: u8, wb: Wb, offset: i64 },
    LoadStoreReg { rt: u8, rn: u8, rm: u8, opt: u8, s: u8, sz: u8, opc: u8 },
    /// `LDP`/`STP`/`LDPSW`. `wide` is the 64-bit form, `sext` the `LDPSW` one.
    Pair { rt: u8, rt2: u8, rn: u8, offset: i64, wide: bool, load: bool, sext: bool, wb: Wb },
    /// `LDR <t>, label`, with the literal's address already resolved.
    LoadLiteral { rt: u8, addr: u32, opc: u8 },
}

/// The instruction a block ends on: the one that decides where control goes
/// next. A block with no terminator ran into the block-length or page limit
/// and simply falls through to [`Block::end`].
#[derive(Debug, Clone, Copy)]
pub(super) enum Term {
    /// `B #imm`.
    B { target: u32 },
    /// `BL #imm`.
    Bl { target: u32, ret_pc: u32 },
    /// `B.cond #imm`.
    BCond { cond: u8, target: u32, next: u32 },
    /// `CBZ`/`CBNZ`.
    Cbz { rt: u8, sf: bool, nz: bool, target: u32, next: u32 },
    /// `TBZ`/`TBNZ`.
    Tbz { rt: u8, bit: u8, nz: bool, target: u32, next: u32 },
    /// `BR Xn`.
    Br { rn: u8 },
    /// `BLR Xn`.
    Blr { rn: u8, ret_pc: u32 },
    /// `RET Xn`.
    Ret { rn: u8 },
    /// `SVC #imm`.
    Svc { imm: u16, next: u32 },
    /// A control instruction with no op of its own: the interpreter decodes it
    /// and sets the PC itself.
    Interpret { insn: u32, next: u32 },
    /// The instruction could not be read when the block was translated. Try
    /// again at run time, so the fault is raised against the state the guest
    /// is actually in.
    Fetch,
}

/// A run of instructions with a single entry point, translated once.
#[derive(Debug)]
pub(super) struct Block {
    /// Guest address of the first instruction.
    start: u32,
    /// The straight-line body, `ops[i]` at `start + 4 * i`.
    ops: Vec<Op>,
    /// The original instruction words, body then terminator, kept so a fault
    /// inside a block leaves the same run-up trail an interpreted one does.
    words: Vec<u32>,
    term: Option<Term>,
}

/// How many entries the direct-mapped lookup in front of the block map holds.
/// A hash of the entry address is a large share of what entering a short
/// block costs, and guest code is dense enough that indexing by the address
/// itself nearly always hits.
const LOOKUP_SLOTS: usize = 4096;

/// The translation cache.
#[derive(Debug)]
pub(super) struct Jit {
    /// The most recent block to land in each slot, or `None`. Only a hint:
    /// the entry address is checked against the block's own, and `blocks` is
    /// what actually owns the cache.
    lookup: Vec<Option<Rc<Block>>>,
    blocks: HashMap<u32, Rc<Block>>,
    /// Entry addresses translated out of each page, so a store to that page
    /// drops exactly the blocks that read it.
    by_page: HashMap<u32, Vec<u32>>,
    translated: u64,
    executed: u64,
    invalidated: u64,
}

/// What the translator has been doing, for host-side diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitStats {
    /// Blocks currently held in the cache.
    pub blocks: usize,
    /// Blocks translated since the cache was created.
    pub translated: u64,
    /// Blocks entered.
    pub executed: u64,
    /// Blocks dropped because the memory they came from was written.
    pub invalidated: u64,
}

impl Default for Jit {
    fn default() -> Jit {
        Jit {
            lookup: vec![None; LOOKUP_SLOTS],
            blocks: HashMap::new(),
            by_page: HashMap::new(),
            translated: 0,
            executed: 0,
            invalidated: 0,
        }
    }
}

impl Jit {
    #[inline(always)]
    fn slot(pc: u32) -> usize {
        (pc >> 2) as usize & (LOOKUP_SLOTS - 1)
    }

    /// The block entered at `pc`, if it is already translated.
    #[inline(always)]
    fn get(&mut self, pc: u32) -> Option<Rc<Block>> {
        let slot = Self::slot(pc);
        if let Some(block) = &self.lookup[slot] {
            if block.start == pc {
                return Some(block.clone());
            }
        }
        let block = self.blocks.get(&pc)?.clone();
        self.lookup[slot] = Some(block.clone());
        Some(block)
    }

    fn insert(&mut self, block: Rc<Block>) {
        if self.blocks.len() >= MAX_BLOCKS {
            self.blocks.clear();
            self.by_page.clear();
            self.drop_lookup();
        }
        let page = block.start >> 12;
        self.by_page.entry(page).or_default().push(block.start);
        self.lookup[Self::slot(block.start)] = Some(block.clone());
        self.blocks.insert(block.start, block);
    }

    /// Forget every lookup hint. Called whenever a block is dropped: a hint
    /// outliving the block it points at would keep running stale code.
    fn drop_lookup(&mut self) {
        for slot in &mut self.lookup {
            *slot = None;
        }
    }

    fn invalidate(&mut self, pages: &[u32]) {
        let mut dropped = false;
        for &page in pages {
            if let Some(starts) = self.by_page.remove(&page) {
                for start in starts {
                    if self.blocks.remove(&start).is_some() {
                        self.invalidated += 1;
                        dropped = true;
                    }
                }
            }
        }
        if dropped {
            self.drop_lookup();
        }
    }

    fn clear(&mut self) {
        self.invalidated += self.blocks.len() as u64;
        self.blocks.clear();
        self.by_page.clear();
        self.drop_lookup();
    }

    fn stats(&self) -> JitStats {
        JitStats {
            blocks: self.blocks.len(),
            translated: self.translated,
            executed: self.executed,
            invalidated: self.invalidated,
        }
    }
}

/// One decoded instruction: either part of a block's body, or the terminator
/// that ends it.
enum Decoded {
    Op(Op),
    Term(Term),
}

/// Translate the block starting at `start`.
///
/// Stops at the first instruction that can move the PC, at [`MAX_BLOCK_OPS`],
/// or at the end of the page — never past it, so one page's invalidation
/// covers a block completely.
fn translate(mem: &Memory, start: u32) -> Block {
    let page_room = (PAGE_SIZE - (start as usize & (PAGE_SIZE - 1))) / 4;
    let limit = MAX_BLOCK_OPS.min(page_room.max(1));
    let mut ops = Vec::with_capacity(limit);
    let mut words = Vec::with_capacity(limit);
    for i in 0..limit {
        let pc = start.wrapping_add(4 * i as u32);
        let insn = match mem.fetch(pc) {
            Ok(insn) => insn,
            Err(_) => {
                return Block { start, ops, words, term: Some(Term::Fetch) };
            }
        };
        match decode(insn, pc) {
            Decoded::Term(term) => {
                words.push(insn);
                return Block { start, ops, words, term: Some(term) };
            }
            Decoded::Op(op) => {
                ops.push(op);
                words.push(insn);
            }
        }
    }
    Block { start, ops, words, term: None }
}

/// Classify one instruction the way [`super::Cpu::execute`] does — by bits
/// 28:25, the architecture's own first decode table — and translate it.
fn decode(insn: u32, pc: u32) -> Decoded {
    match (insn >> 25) & 0xF {
        0x8 | 0x9 => Decoded::Op(decode_data_proc_imm(insn, pc)),
        0x5 | 0xD => Decoded::Op(decode_data_proc_reg(insn)),
        0x4 | 0x6 | 0xC | 0xE => Decoded::Op(decode_load_store(insn, pc)),
        // Advanced SIMD and scalar floating point: no ops of their own yet.
        0x7 | 0xF => Decoded::Op(Op::Interpret { insn }),
        0xA | 0xB => decode_branch_or_system(insn, pc),
        // Reserved and SVE. The interpreter rejects them; let it say so.
        _ => Decoded::Op(Op::Interpret { insn }),
    }
}

/// The branch, exception-generation and system group. Everything here either
/// ends the block or is one of the two system forms that provably retire to
/// the following instruction.
fn decode_branch_or_system(insn: u32, pc: u32) -> Decoded {
    let next = pc.wrapping_add(4);
    match (insn >> 24) & 0xFF {
        // B.cond
        0x54 => Decoded::Term(Term::BCond {
            cond: (insn & 0xF) as u8,
            target: branch_target(pc, sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2),
            next,
        }),
        // B #imm
        0x14..=0x17 => Decoded::Term(Term::B {
            target: branch_target(pc, sext_u64(insn & 0x3FF_FFFF, 26) << 2),
        }),
        // TBZ / TBNZ
        0x36 | 0x37 | 0xB6 | 0xB7 => Decoded::Term(Term::Tbz {
            rt: (insn & 0x1F) as u8,
            bit: ((((insn >> 31) & 1) << 5) | ((insn >> 19) & 0x1F)) as u8,
            nz: ((insn >> 24) & 1) == 1,
            target: branch_target(pc, sext_u64((insn >> 5) & 0x3FFF, 14) << 2),
            next,
        }),
        // CBZ / CBNZ
        0x34 | 0x35 | 0xB4 | 0xB5 => Decoded::Term(Term::Cbz {
            rt: (insn & 0x1F) as u8,
            sf: ((insn >> 31) & 1) == 1,
            nz: ((insn >> 24) & 1) == 1,
            target: branch_target(pc, sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2),
            next,
        }),
        // BL #imm
        0x94..=0x97 => Decoded::Term(Term::Bl {
            target: branch_target(pc, sext_u64(insn & 0x3FF_FFFF, 26) << 2),
            ret_pc: next,
        }),
        // BR / BLR / RET
        0xD6 | 0xD7 => {
            let rn = ((insn >> 5) & 0x1F) as u8;
            if ((insn >> 16) & 0x1F) != 0x1F || ((insn >> 10) & 0x3F) != 0 {
                return Decoded::Term(Term::Interpret { insn, next });
            }
            match (insn >> 21) & 0xF {
                0b0000 => Decoded::Term(Term::Br { rn }),
                0b0001 => Decoded::Term(Term::Blr { rn, ret_pc: next }),
                0b0010 => Decoded::Term(Term::Ret { rn }),
                _ => Decoded::Term(Term::Interpret { insn, next }),
            }
        }
        // SVC, and the other exception-generating forms the interpreter faults on.
        0xD4 => {
            if ((insn >> 21) & 0b111) == 0 && (insn & 0x1F) == 0b00001 {
                Decoded::Term(Term::Svc { imm: ((insn >> 5) & 0xFFFF) as u16, next })
            } else {
                Decoded::Term(Term::Interpret { insn, next })
            }
        }
        // MSR/MRS, barriers and hints. `system` retires all of them to the
        // next instruction, so they stay inside the block.
        0xD5 => {
            if (insn >> 16) & 0xFFFF == 0xD503 {
                Decoded::Op(Op::Nop)
            } else {
                Decoded::Op(Op::Interpret { insn })
            }
        }
        _ => Decoded::Term(Term::Interpret { insn, next }),
    }
}

/// The target of a PC-relative branch. `imm` is already the sign-extended
/// byte displacement.
#[inline]
fn branch_target(pc: u32, imm: u64) -> u32 {
    (pc as i64).wrapping_add(imm as i64) as u32
}

fn decode_data_proc_imm(insn: u32, pc: u32) -> Op {
    let sf = (insn >> 31) & 1 == 1;
    match (insn >> 24) & 0x1F {
        // ADR / ADRP: the result depends only on where the instruction is, so
        // the translator computes it and the block just moves a constant.
        0b10000 => {
            let rd = (insn & 0x1F) as u8;
            let immhi = u64::from((insn >> 5) & 0x7_FFFF);
            let immlo = u64::from((insn >> 29) & 0b11);
            let imm = sext_u64((immhi << 2) | immlo, 21);
            let val = if (insn >> 31) & 1 == 1 {
                (u64::from(pc & !0xFFF)).wrapping_add(imm.wrapping_shl(12))
            } else {
                u64::from(pc).wrapping_add(imm)
            };
            Op::MovConst { rd, val }
        }
        // ADD/SUB immediate. Bit 23 is the ADDG/SUBG tag arithmetic the
        // interpreter rejects.
        0b10001 => {
            if ((insn >> 23) & 1) == 1 {
                return Op::Interpret { insn };
            }
            let op = (insn >> 29) & 0b11;
            let imm12 = u64::from((insn >> 10) & 0xFFF);
            Op::AddSubImm {
                rd: (insn & 0x1F) as u8,
                rn: ((insn >> 5) & 0x1F) as u8,
                imm: if ((insn >> 22) & 1) == 1 { imm12 << 12 } else { imm12 },
                set_flags: (op & 1) == 1,
                sub: (op >> 1) == 1,
                sf,
            }
        }
        0b10010 => {
            if ((insn >> 23) & 1) == 1 {
                // MOVN / MOVZ / MOVK
                let rd = (insn & 0x1F) as u8;
                let imm16 = u64::from((insn >> 5) & 0xFFFF);
                let hw = if sf { (insn >> 21) & 0b11 } else { (insn >> 21) & 1 };
                let shift = hw * 16;
                match (insn >> 29) & 0b11 {
                    0b00 => Op::MovConst { rd, val: !(imm16 << shift) & Cpu::mask(sf) },
                    0b10 => Op::MovConst { rd, val: (imm16 << shift) & Cpu::mask(sf) },
                    0b11 => {
                        let mask = (0xFFFFu64 << shift) & Cpu::mask(sf);
                        Op::MovK { rd, mask, val: (imm16 << shift) & mask }
                    }
                    _ => Op::Interpret { insn },
                }
            } else {
                // Logical immediate: the bitmask decodes to a constant.
                let immr = (insn >> 16) & 0x3F;
                let imms = (insn >> 10) & 0x3F;
                match decode_bit_mask(sf, (insn >> 22) & 1, immr, imms) {
                    Some(imm) => Op::LogicalImm {
                        rd: (insn & 0x1F) as u8,
                        rn: ((insn >> 5) & 0x1F) as u8,
                        imm,
                        opc: ((insn >> 29) & 0b11) as u8,
                        sf,
                    },
                    None => Op::Interpret { insn },
                }
            }
        }
        0b10011 => {
            let rd = (insn & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            if ((insn >> 23) & 1) == 0 {
                // Bitfield move. The unallocated encodings go to the
                // interpreter so it raises the same error.
                let (immr, imms) = if sf {
                    if ((insn >> 22) & 1) != 1 {
                        return Op::Interpret { insn };
                    }
                    ((insn >> 16) & 0x3F, (insn >> 10) & 0x3F)
                } else {
                    if ((insn >> 21) & 1) == 1 || ((insn >> 15) & 1) == 1 {
                        return Op::Interpret { insn };
                    }
                    ((insn >> 16) & 0x1F, (insn >> 10) & 0x1F)
                };
                Op::Bitfield {
                    rd,
                    rn,
                    opc: ((insn >> 29) & 0b11) as u8,
                    immr: immr as u8,
                    imms: imms as u8,
                    sf,
                }
            } else {
                // EXTR
                let imm = if sf {
                    if ((insn >> 22) & 1) != 1 || ((insn >> 21) & 1) == 1 {
                        return Op::Interpret { insn };
                    }
                    (insn >> 10) & 0x3F
                } else {
                    if ((insn >> 22) & 1) == 1
                        || ((insn >> 21) & 1) == 1
                        || ((insn >> 15) & 1) == 1
                    {
                        return Op::Interpret { insn };
                    }
                    (insn >> 10) & 0x1F
                };
                Op::Extr { rd, rn, rm: ((insn >> 16) & 0x1F) as u8, imm: imm as u8, sf }
            }
        }
        _ => Op::Interpret { insn },
    }
}

fn decode_data_proc_reg(insn: u32) -> Op {
    let sf = (insn >> 31) & 1 == 1;
    let rd = (insn & 0x1F) as u8;
    let rn = ((insn >> 5) & 0x1F) as u8;
    let rm = ((insn >> 16) & 0x1F) as u8;
    match (insn >> 24) & 0x1F {
        // Logical shifted register.
        0b01010 => Op::LogicalShifted {
            rd,
            rn,
            rm,
            st: ((insn >> 22) & 0b11) as u8,
            sa: ((insn >> 10) & 0x3F) as u8,
            opc: ((insn >> 29) & 0b11) as u8,
            invert: ((insn >> 21) & 1) == 1,
            sf,
        },
        // ADD/SUB, shifted or extended register.
        0b01011 => {
            let op = (insn >> 29) & 0b11;
            let set_flags = (op & 1) == 1;
            let sub = (op >> 1) == 1;
            if ((insn >> 21) & 0b111) == 0b001 {
                Op::AddSubExtended {
                    rd,
                    rn,
                    rm,
                    option: ((insn >> 13) & 0b111) as u8,
                    shift: ((insn >> 10) & 0b111) as u8,
                    set_flags,
                    sub,
                    sf,
                }
            } else {
                Op::AddSubShifted {
                    rd,
                    rn,
                    rm,
                    st: ((insn >> 22) & 0b11) as u8,
                    sa: ((insn >> 10) & 0x3F) as u8,
                    set_flags,
                    sub,
                    sf,
                }
            }
        }
        0b11010 => match (((insn >> 23) & 1), ((insn >> 22) & 1)) {
            // Conditional compare.
            (0, 1) => Op::CondCmp {
                rn,
                rm,
                imm: ((insn >> 16) & 0x1F) as u8,
                cond: ((insn >> 12) & 0xF) as u8,
                nzcv: (insn & 0xF) as u8,
                sub: ((insn >> 30) & 1) == 1,
                is_imm: ((insn >> 11) & 1) == 1,
                sf,
            },
            // Conditional select.
            (1, 0) => Op::CondSel {
                rd,
                rn,
                rm,
                cond: ((insn >> 12) & 0xF) as u8,
                else_inv: ((insn >> 30) & 1) == 1,
                else_inc: ((insn >> 10) & 1) == 1,
                sf,
            },
            // The one- and two-source group (divides, variable shifts, CRC32,
            // bit counts) and ADC/SBC: left to the interpreter.
            _ => Op::Interpret { insn },
        },
        // Three-source: the multiplies.
        0b11011 => {
            let ra = ((insn >> 10) & 0x1F) as u8;
            let sub = ((insn >> 15) & 1) == 1;
            match (insn >> 21) & 0xFF {
                0b11011000 => Op::Madd { rd, rn, rm, ra, sub, sf },
                0b11011001 => Op::MaddLong { rd, rn, rm, ra, sub, signed: true },
                0b11011101 => Op::MaddLong { rd, rn, rm, ra, sub, signed: false },
                0b11011010 => Op::Mulh { rd, rn, rm, signed: true },
                0b11011110 => Op::Mulh { rd, rn, rm, signed: false },
                _ => Op::Interpret { insn },
            }
        }
        _ => Op::Interpret { insn },
    }
}

/// The load/store group, in the order [`super::Cpu::execute`] tries it: the
/// literal forms first, then everything [`super::Cpu::try_load_store`] claims.
fn decode_load_store(insn: u32, pc: u32) -> Op {
    // LDR <t>, label — the address is fixed by where the instruction is.
    if ((insn >> 27) & 0b111) == 0b011 && ((insn >> 26) & 1) == 0 && ((insn >> 24) & 0b11) == 0b00 {
        let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
        return Op::LoadLiteral {
            rt: (insn & 0x1F) as u8,
            addr: (pc as i64).wrapping_add(imm as i64) as u32,
            opc: ((insn >> 30) & 0b11) as u8,
        };
    }

    // The exclusive accessors touch the local monitor, and the V=1 forms are
    // SIMD: both stay with the interpreter.
    let grp_excl = (insn >> 21) & 0x1FF;
    if (0b001000000..=0b001000011).contains(&grp_excl)
        || grp_excl == 0b001000100
        || grp_excl == 0b001000110
    {
        return Op::Interpret { insn };
    }
    if ((insn >> 26) & 1) == 1 {
        return Op::Interpret { insn };
    }

    let sz = ((insn >> 30) & 0b11) as u8;
    let opc = ((insn >> 22) & 0b11) as u8;
    let rt = (insn & 0x1F) as u8;
    let rn = ((insn >> 5) & 0x1F) as u8;

    // Register offset.
    if ((insn >> 27) & 0b111) == 0b111 && ((insn >> 24) & 0b11) == 0b00 && ((insn >> 21) & 1) == 1 {
        let opt = ((insn >> 13) & 0b111) as u8;
        // Only four of the eight extend encodings are defined here; the rest
        // are undefined, and the interpreter faults on them.
        if !matches!(opt, 0b010 | 0b011 | 0b110 | 0b111) {
            return Op::Interpret { insn };
        }
        return Op::LoadStoreReg {
            rt,
            rn,
            rm: ((insn >> 16) & 0x1F) as u8,
            opt,
            s: ((insn >> 12) & 1) as u8,
            sz,
            opc,
        };
    }

    // Immediate offset forms.
    if ((insn >> 27) & 0b111) == 0b111 {
        let mode = (insn >> 24) & 0b11;
        if mode == 0b01 {
            // Unsigned offset, scaled by the access size.
            let scale = 1i64 << sz;
            return Op::LoadStoreImm {
                rt,
                rn,
                sz,
                opc,
                wb: Wb::None,
                offset: i64::from((insn >> 10) & 0xFFF) * scale,
            };
        }
        if mode == 0b00 && ((insn >> 21) & 1) == 0 {
            let offset = sext_u64((insn >> 12) & 0x1FF, 9) as i64;
            let wb = match (insn >> 10) & 0b11 {
                0b01 => Wb::Post,
                0b11 => Wb::Pre,
                // Unscaled (`LDUR`/`STUR`) and the unprivileged forms.
                _ => Wb::None,
            };
            return Op::LoadStoreImm { rt, rn, sz, opc, wb, offset };
        }
    }

    // Load/store pair.
    if ((insn >> 27) & 0b111) == 0b101 && ((insn >> 25) & 1) == 0 {
        let pair_opc = (insn >> 30) & 0b11;
        let load = ((insn >> 22) & 1) == 1;
        // The tagged store-pair and the reserved addressing mode are errors
        // the interpreter reports.
        if (pair_opc == 0b01 && !load) || pair_opc == 0b11 {
            return Op::Interpret { insn };
        }
        let wide = pair_opc == 0b10;
        let scale: i64 = if wide { 8 } else { 4 };
        let wb = match (insn >> 23) & 0b11 {
            0b01 => Wb::Post,
            0b11 => Wb::Pre,
            _ => Wb::None,
        };
        return Op::Pair {
            rt,
            rt2: ((insn >> 10) & 0x1F) as u8,
            rn,
            offset: (sext_u64((insn >> 15) & 0x7F, 7) as i64).wrapping_mul(scale),
            wide,
            load,
            sext: pair_opc == 0b01,
            wb,
        };
    }

    Op::Interpret { insn }
}

impl Cpu {
    /// Whether [`Cpu::run`] goes through the translator.
    pub fn jit_enabled(&self) -> bool {
        self.jit_enabled
    }

    /// Turn the translator on or off. Turning it off drops the cache, so a
    /// caller that switches back gets a translator with no stale state.
    pub fn set_jit_enabled(&mut self, on: bool) {
        if !on {
            self.jit.clear();
        }
        self.jit_enabled = on;
    }

    /// What the translator has been doing since the CPU was created.
    pub fn jit_stats(&self) -> JitStats {
        self.jit.stats()
    }

    /// Drop every translated block. Callers that rewrite guest code behind
    /// [`crate::mem::Memory`]'s back need this; ordinary guest stores are
    /// noticed on their own.
    pub fn jit_flush(&mut self) {
        self.jit.clear();
    }

    /// [`Cpu::run`] over translated blocks.
    ///
    /// The step budget is honoured exactly. A block whose remaining
    /// instructions do not fit is entered anyway and left part-way through,
    /// with `pc` on the instruction that would have come next — the block is
    /// a cache, not a unit of execution, so stopping inside one is no
    /// different from stopping between two interpreted instructions.
    pub(super) fn run_jit(&mut self, max_steps: u64) -> Result<RunReport> {
        let mut steps = 0u64;
        while steps < max_steps && !self.halted {
            // The scheduler's preemption point. The interpreter takes it
            // between any two instructions; here it is between blocks, which
            // is the same thing at the scale of a 20,000-instruction slice.
            if self.slice_used >= TIME_SLICE {
                self.slice_used = 0;
                self.yield_thread();
            }
            self.sweep_timed_waits();
            let block = self.jit_block_at(self.pc);
            let ran = self.exec_block(&block, max_steps - steps)?;
            self.slice_used += ran;
            steps += ran;
        }
        Ok(RunReport { steps, halted: self.halted })
    }

    /// The block entered at `pc`, translating it if this is the first visit.
    ///
    /// Drains the pages guest stores have landed on first, so a block is never
    /// handed out after the instructions behind it have changed.
    fn jit_block_at(&mut self, pc: u32) -> Rc<Block> {
        if self.mem.has_dirty_code() {
            let dirty = self.mem.dirty_code_pages();
            self.jit.invalidate(&dirty);
        }
        self.jit.executed += 1;
        if let Some(block) = self.jit.get(pc) {
            return block;
        }
        let block = Rc::new(translate(&self.mem, pc));
        self.mem.mark_code_page(block.start);
        self.jit.translated += 1;
        self.jit.insert(block.clone());
        block
    }

    /// Run at most `budget` of a block's instructions, reporting how many
    /// retired.
    ///
    /// `self.pc` tracks the instruction being executed throughout, exactly as
    /// it does in the interpreter, so a fault inside a block reports the same
    /// address and the same register state an interpreted one would.
    fn exec_block(&mut self, block: &Block, budget: u64) -> Result<u64> {
        let body = (block.ops.len() as u64).min(budget) as usize;
        let mut pc = block.start;
        for i in 0..body {
            let insn = block.words[i];
            self.recent[self.recent_len % RECENT_LEN] = (pc, insn);
            self.recent_len = self.recent_len.saturating_add(1);
            self.retire();
            self.pc = pc;
            if let Err(e) = self.exec_op(block.ops[i], pc) {
                self.record_fault(&e, pc, insn);
                return Err(e);
            }
            pc = pc.wrapping_add(4);
        }
        let mut ran = body as u64;
        match block.term {
            Some(term) if body == block.ops.len() && ran < budget => {
                let insn = block.words[body];
                self.recent[self.recent_len % RECENT_LEN] = (pc, insn);
                self.recent_len = self.recent_len.saturating_add(1);
                self.retire();
                self.pc = pc;
                if let Err(e) = self.exec_term(term, pc) {
                    self.record_fault(&e, pc, insn);
                    return Err(e);
                }
                ran += 1;
            }
            // Either the budget ran out inside the block, or it covered the
            // body of a block that has no terminator. Both leave `pc` on the
            // next instruction to run.
            _ => self.pc = pc,
        }
        Ok(ran)
    }

    /// Execute one body op. Every arm does what the interpreter's decoder
    /// would have done once it finished decoding.
    #[inline(always)]
    fn exec_op(&mut self, op: Op, pc: u32) -> Result<()> {
        match op {
            Op::Nop => {}
            Op::Interpret { insn } => self.execute(insn, pc.wrapping_add(4))?,

            Op::MovConst { rd, val } => self.write_zr(rd, val),
            Op::MovK { rd, mask, val } => {
                let cur = self.read_zr(rd) & !mask;
                self.write_zr(rd, cur | val);
            }

            Op::AddSubImm { rd, rn, imm, set_flags, sub, sf } => {
                self.add_sub(rd, rn, imm, set_flags, sub, sf, true);
            }
            Op::AddSubShifted { rd, rn, rm, st, sa, set_flags, sub, sf } => {
                let v = shift_reg(self.read_zr(rm) & Cpu::mask(sf), u32::from(st), u32::from(sa), sf);
                self.add_sub(rd, rn, v, set_flags, sub, sf, false);
            }
            Op::AddSubExtended { rd, rn, rm, option, shift, set_flags, sub, sf } => {
                let v = extend_reg(self.read_zr(rm), option, sf) & Cpu::mask(sf);
                let v = v.wrapping_shl(u32::from(shift)) & Cpu::mask(sf);
                self.add_sub(rd, rn, v, set_flags, sub, sf, true);
            }

            Op::LogicalImm { rd, rn, imm, opc, sf } => {
                let a = self.read_zr(rn) & Cpu::mask(sf);
                let r = self.logical(a, imm, opc, sf);
                // Rd == 31 is SP for AND/ORR/EOR and the zero register only
                // for ANDS — one of the few places the two differ.
                if opc == 0b11 {
                    self.write_zr(rd, r);
                } else {
                    self.write_x(rd, r);
                }
            }
            Op::LogicalShifted { rd, rn, rm, st, sa, opc, invert, sf } => {
                let a = self.read_zr(rn) & Cpu::mask(sf);
                let mut b = self.read_zr(rm) & Cpu::mask(sf);
                if invert {
                    b = !b & Cpu::mask(sf);
                }
                let b = shift_reg(b, u32::from(st), u32::from(sa), sf);
                let r = self.logical(a, b, opc, sf);
                self.write_zr(rd, r);
            }

            Op::Bitfield { rd, rn, opc, immr, imms, sf } => {
                let val = self.read_zr(rn) & Cpu::mask(sf);
                let cur = self.read_zr(rd) & Cpu::mask(sf);
                let r = bitfield_apply(u32::from(opc), val, cur, u32::from(immr), u32::from(imms), sf);
                self.write_zr(rd, r);
            }
            Op::Extr { rd, rn, rm, imm, sf } => {
                let size = if sf { 64u32 } else { 32 };
                let a = self.read_zr(rn) & Cpu::mask(sf);
                let b = self.read_zr(rm) & Cpu::mask(sf);
                let imm = u32::from(imm);
                // Rn is the high half of the Rn:Rm pair the field is taken from.
                let r = if imm == 0 {
                    b
                } else {
                    ((b >> imm) | a.wrapping_shl(size.wrapping_sub(imm))) & Cpu::mask(sf)
                };
                self.write_zr(rd, r);
            }

            Op::CondSel { rd, rn, rm, cond, else_inv, else_inc, sf } => {
                let a = self.read_zr(rn) & Cpu::mask(sf);
                let b = self.read_zr(rm) & Cpu::mask(sf);
                let take_a = self.condition_holds(cond);
                let mut else_val = b;
                if else_inv {
                    else_val = !else_val;
                }
                if else_inc {
                    else_val = else_val.wrapping_add(1);
                }
                let r = if take_a { a } else { else_val };
                self.write_zr(rd, r & Cpu::mask(sf));
            }
            Op::CondCmp { rn, rm, imm, cond, nzcv, sub, is_imm, sf } => {
                if self.condition_holds(cond) {
                    let a = self.read_zr(rn) & Cpu::mask(sf);
                    let b = if is_imm { u64::from(imm) } else { self.read_zr(rm) };
                    self.set_nzcv_from_compare(a, b, sub, u64::from(sub), sf);
                } else {
                    self.nzcv = u32::from(nzcv) << 28;
                }
            }

            Op::Madd { rd, rn, rm, ra, sub, sf } => {
                let mask = Cpu::mask(sf);
                let product = (self.read_zr(rn) & mask).wrapping_mul(self.read_zr(rm) & mask);
                let c = self.read_zr(ra) & mask;
                let r = if sub { c.wrapping_sub(product) } else { c.wrapping_add(product) };
                self.write_zr(rd, r & mask);
            }
            Op::MaddLong { rd, rn, rm, ra, sub, signed } => {
                let a = self.read_zr(rn);
                let b = self.read_zr(rm);
                // The multiplicands are the low 32 bits of Rn/Rm, not the
                // whole register.
                let product = if signed {
                    ((i128::from(a as u32 as i32)) * (i128::from(b as u32 as i32))) as u64
                } else {
                    (u128::from(a as u32) * u128::from(b as u32)) as u64
                };
                let c = self.read_zr(ra);
                let r = if sub { c.wrapping_sub(product) } else { c.wrapping_add(product) };
                self.write_zr(rd, r);
            }
            Op::Mulh { rd, rn, rm, signed } => {
                let a = self.read_zr(rn);
                let b = self.read_zr(rm);
                let r = if signed {
                    (((a as i64 as i128) * (b as i64 as i128)) >> 64) as u64
                } else {
                    ((u128::from(a) * u128::from(b)) >> 64) as u64
                };
                self.write_zr(rd, r);
            }

            Op::LoadStoreImm { rt, rn, sz, opc, wb, offset } => {
                let base = self.read_x(rn);
                let (addr, wb_val) = Self::indexed(base, offset, wb);
                self.ld_st_opc(addr as u32, rt, u32::from(sz), u32::from(opc))?;
                if let Some(v) = wb_val {
                    self.write_x(rn, v);
                }
            }
            Op::LoadStoreReg { rt, rn, rm, opt, s, sz, opc } => {
                let offset = self.offset_from_reg(rm, opt, u32::from(s), sz)?;
                let addr = (self.read_x(rn) as i64).wrapping_add(offset) as u32;
                self.ld_st_opc(addr, rt, u32::from(sz), u32::from(opc))?;
            }
            Op::Pair { rt, rt2, rn, offset, wide, load, sext, wb } => {
                let base = self.read_x(rn);
                let (addr, wb_val) = Self::indexed(base, offset, wb);
                let addr = addr as u32;
                let stride = if wide { 8u32 } else { 4 };
                if load {
                    let v0 = self.load_pair_half(addr, wide, sext)?;
                    let v1 = self.load_pair_half(addr.wrapping_add(stride), wide, sext)?;
                    self.write_zr(rt, v0);
                    self.write_zr(rt2, v1);
                } else if wide {
                    self.mem.write_u64(addr, self.read_zr(rt))?;
                    self.mem.write_u64(addr.wrapping_add(8), self.read_zr(rt2))?;
                } else {
                    self.mem.write_u32(addr, self.read_zr(rt) as u32)?;
                    self.mem.write_u32(addr.wrapping_add(4), self.read_zr(rt2) as u32)?;
                }
                if let Some(v) = wb_val {
                    self.write_x(rn, v);
                }
            }
            Op::LoadLiteral { rt, addr, opc } => match opc {
                0b00 => {
                    let val = u64::from(self.mem.read_u32(addr)?);
                    self.write_zr(rt, val);
                }
                0b01 => {
                    let val = self.mem.read_u64(addr)?;
                    self.write_zr(rt, val);
                }
                0b10 => {
                    let val = u64::from(self.mem.read_u32(addr)?);
                    self.write_zr(rt, sext_u64(val, 32));
                }
                // PRFM: a prefetch hint, so nothing to do.
                _ => {}
            },
        }
        Ok(())
    }

    /// The address a load/store touches and the value (if any) to write back
    /// into its base register.
    #[inline(always)]
    fn indexed(base: u64, offset: i64, wb: Wb) -> (u64, Option<u64>) {
        match wb {
            Wb::None => (base.wrapping_add(offset as u64), None),
            Wb::Pre => {
                let addr = base.wrapping_add(offset as u64);
                (addr, Some(addr))
            }
            Wb::Post => (base, Some(base.wrapping_add(offset as u64))),
        }
    }

    /// One register's worth of an `LDP`/`LDPSW`.
    #[inline(always)]
    fn load_pair_half(&self, addr: u32, wide: bool, sext: bool) -> Result<u64> {
        if wide {
            return self.mem.read_u64(addr);
        }
        let w = u64::from(self.mem.read_u32(addr)?);
        Ok(if sext { sext_u64(w, 32) } else { w })
    }

    /// The `AND`/`ORR`/`EOR`/`ANDS` core, shared by the immediate and
    /// shifted-register forms. `ANDS` is the only one that writes flags, and
    /// it leaves C and V alone.
    #[inline(always)]
    fn logical(&mut self, a: u64, b: u64, opc: u8, sf: bool) -> u64 {
        match opc {
            0b00 => a & b,
            0b01 => a | b,
            0b10 => a ^ b,
            _ => {
                let r = a & b;
                let n = (r >> (if sf { 63 } else { 31 })) & 1;
                let z = u64::from(r == 0);
                let c = (self.nzcv >> 29) & 1;
                let v = (self.nzcv >> 28) & 1;
                self.nzcv = ((n as u32) << 31) | ((z as u32) << 30) | (c << 29) | (v << 28);
                r
            }
        }
    }

    /// Execute the instruction a block ends on, leaving `self.pc` wherever
    /// control goes next.
    #[inline(always)]
    fn exec_term(&mut self, term: Term, pc: u32) -> Result<()> {
        match term {
            Term::B { target } => self.pc = target,
            Term::Bl { target, ret_pc } => {
                self.write_zr(30, u64::from(ret_pc));
                self.pc = target;
            }
            Term::BCond { cond, target, next } => {
                self.pc = if self.condition_holds(cond) { target } else { next };
            }
            Term::Cbz { rt, sf, nz, target, next } => {
                let val = self.read_zr(rt);
                let is_zero = if sf { val == 0 } else { (val as u32) == 0 };
                self.pc = if is_zero == !nz { target } else { next };
            }
            Term::Tbz { rt, bit, nz, target, next } => {
                let set = (self.read_zr(rt) >> bit) & 1 == 1;
                self.pc = if set == nz { target } else { next };
            }
            Term::Br { rn } => self.pc = self.read_zr(rn) as u32,
            Term::Blr { rn, ret_pc } => {
                // Read the target before linking: `blr x30` is a
                // return-and-relink, and writing x30 first makes it jump to
                // itself.
                let target = self.read_zr(rn) as u32;
                self.write_zr(30, u64::from(ret_pc));
                self.pc = target;
            }
            Term::Ret { rn } => {
                // A return to address 0 is homebrew's exit path; the boot
                // model routes it through the exit trampoline instead of
                // fetching from NULL.
                let target = self.read_zr(rn) as u32;
                self.pc = if target == 0 { SELF_RETURN_TRAMPOLINE } else { target };
            }
            Term::Svc { imm, next } => {
                // Retire the SVC before dispatching it: a syscall that
                // switches threads installs the incoming thread's PC, and the
                // outgoing one has to resume after its own SVC.
                self.pc = next;
                self.syscall(imm)?;
            }
            Term::Interpret { insn, next } => self.execute(insn, next)?,
            Term::Fetch => {
                let insn = self.mem.fetch(pc)?;
                self.execute(insn, pc.wrapping_add(4))?;
            }
        }
        Ok(())
    }
}
