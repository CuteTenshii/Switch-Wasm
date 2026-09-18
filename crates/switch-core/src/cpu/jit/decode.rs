//! Translation: walking forward from an address, turning each instruction
//! into the one thing it does, and deciding where the block ends.

use super::ir::{Block, Branch, Exit, Op, Term};
use crate::cpu::bits::*;
use crate::cpu::loadstore::{pair_slot, rt_slot, Acc, Ext, PairKind, Wb};
use crate::cpu::system::SysOp;
use crate::cpu::{Cpu, ZR_DISCARD};
use crate::mem::{Memory, PAGE_BITS};

/// Longest run of instructions one block may cover, the branches it follows
/// included. Longer blocks amortize the per-block bookkeeping over more work,
/// but they also make the step budget coarser. Not the binding limit in
/// practice: raising it to 160 moved hbmenu's block entries by 0.2%, because
/// what actually ends a block is an indirect branch or a return.
const MAX_BLOCK_OPS: usize = 64;

/// Whether the block translator has a real op for `insn`, or hands it back to
/// the interpreter to decode again on every execution.
///
/// Exact, and the same answer on every target, which is why it is worth
/// asking. The alternative is to time two engines against each other and call
/// a class translated when they come out within noise of one another, and
/// noise on a desktop is not evidence about the browser.
///
/// PC-relative forms decode against an address; which one makes no difference
/// to whether the encoding is translated, so any aligned address answers.
pub fn translates(insn: u32) -> bool {
    const REPRESENTATIVE_PC: u32 = 0x0800_0000;
    !matches!(
        decode(insn, REPRESENTATIVE_PC),
        Decoded::Op(Op::Interpret { .. }) | Decoded::Term(Term::Interpret { .. })
    )
}

/// One decoded instruction: part of a block's body, a conditional branch the
/// block can run through, or the terminator that ends it.
enum Decoded {
    Op(Op),
    Exit(Exit),
    Term(Term),
}

/// Translate the block starting at `start`.
///
/// Stops at the first instruction that can move the PC somewhere it cannot
/// follow, or at [`MAX_BLOCK_OPS`]. A direct `B` whose target is not already
/// part of the block is followed rather than ending it: the target is fixed,
/// so translation carries on there and the branch becomes an [`Exit::Jump`]
/// that is always taken. On a Just Dance 2019 frame that took block entries
/// from 2.13M to 2.00M and 1% off the frame in the wasm build.
///
/// `BL` is not followed, although its target is as fixed. Doing so copies the
/// callee into a block at every call site, and on the same frame that cost 4%
/// more than it saved in transitions.
///
/// A block may therefore read code from more than one page, and every page it
/// read is listed in [`Block::pages`], so a store to any of them drops it.
pub(super) fn translate(mem: &Memory, start: u32) -> Block {
    let mut ops = Vec::with_capacity(MAX_BLOCK_OPS);
    let mut words = Vec::with_capacity(MAX_BLOCK_OPS);
    let mut exits = Vec::new();
    let mut pages = Vec::with_capacity(2);
    // The straight-line runs translated so far, as `(first, end)` addresses,
    // so a branch back into the block is left to end it rather than unrolled.
    let mut runs: Vec<(u32, u32)> = Vec::with_capacity(4);
    let mut run_start = start;
    let mut pc = start;
    for i in 0..MAX_BLOCK_OPS {
        let page = pc >> PAGE_BITS;
        if !pages.contains(&page) {
            pages.push(page);
        }
        let insn = match mem.fetch(pc) {
            Ok(insn) => insn,
            Err(_) => {
                fuse_compares(&mut ops, &mut exits);
                return Block::new(start, ops, words, exits, Some(Term::Fetch), pages);
            }
        };
        let decoded = match decode(insn, pc) {
            Decoded::Term(Term::B { target }) => Decoded::Exit(Exit::Jump { target }),
            other => other,
        };
        let follow = match decoded {
            Decoded::Exit(Exit::Jump { target }) => {
                let end = pc.wrapping_add(4);
                let inside = |t: u32| {
                    t >= run_start && t < end
                        || runs.iter().any(|&(first, last)| t >= first && t < last)
                };
                // A jump on the last slot the block has room for would leave
                // it with nothing after the jump, which is a block that only
                // moves the PC. Not worth a slot of its own.
                // Nor is a jump to a PLT stub, which the terminator runs
                // better folded than the block would as four more ops.
                let followable = target & 3 == 0
                    && !inside(target)
                    && i + 1 < MAX_BLOCK_OPS
                    && plt_slot(mem, target).is_none();
                followable.then_some(target)
            }
            _ => None,
        };
        match (decoded, follow) {
            (Decoded::Exit(exit), Some(target)) => {
                exits.push(Branch::new(i as u32, exit));
                ops.push(Op::Nop);
                words.push(insn);
                runs.push((run_start, pc.wrapping_add(4)));
                pc = target;
                run_start = target;
                continue;
            }
            // A jump that is not followed ends the block, as it always did.
            (Decoded::Exit(Exit::Jump { target }), None) => {
                words.push(insn);
                fuse_compares(&mut ops, &mut exits);
                let term = fold_plt(mem, Term::B { target }, &mut pages);
                return Block::new(start, ops, words, exits, Some(term), pages);
            }
            (Decoded::Term(term), _) => {
                words.push(insn);
                fuse_compares(&mut ops, &mut exits);
                let term = fold_plt(mem, term, &mut pages);
                return Block::new(start, ops, words, exits, Some(term), pages);
            }
            // A conditional branch does not end the block: its not-taken path
            // is the next instruction, so translation carries on there and the
            // branch becomes an early exit. This is the whole reason blocks
            // are longer than the six or seven instructions a basic block runs
            // to: `b.cond` alone is 12% of hbmenu's frame.
            (Decoded::Exit(exit), None) => {
                exits.push(Branch::new(i as u32, exit));
                ops.push(Op::Nop);
                words.push(insn);
            }
            (Decoded::Op(op), _) => {
                ops.push(op);
                words.push(insn);
            }
        }
        pc = pc.wrapping_add(4);
    }
    fuse_compares(&mut ops, &mut exits);
    Block::new(start, ops, words, exits, None, pages)
}

/// A direct `BL` or `B` whose target is a PLT stub, turned into the terminator
/// that runs the stub as well, with the stub's page added to the ones the
/// block reads.
fn fold_plt(mem: &Memory, term: Term, pages: &mut Vec<u32>) -> Term {
    let (target, folded) = match term {
        Term::Bl { target, ret_pc } => match plt_slot(mem, target) {
            Some(got) => (
                target,
                Term::BlPlt {
                    got,
                    stub: target,
                    ret_pc,
                },
            ),
            None => return term,
        },
        Term::B { target } => match plt_slot(mem, target) {
            Some(got) => (target, Term::BPlt { got, stub: target }),
            None => return term,
        },
        _ => return term,
    };
    let page = target >> PAGE_BITS;
    if !pages.contains(&page) {
        pages.push(page);
    }
    folded
}

/// The GOT slot a PLT stub at `at` jumps through, if `at` is one.
///
/// Every call from one module into another lands on one of these first:
///
/// ```text
/// adrp x16, <slot page>
/// ldr  x17, [x16, #<slot offset>]
/// add  x16, x16, #<slot offset>
/// br   x17
/// ```
///
/// Run as they stand, they are a block of their own: the caller's block ends
/// on its `BL`, the stub is entered, and its `BR` ends it again, so a call
/// across modules was two block transitions and four dispatches more than a
/// call inside one. On a Just Dance 2019 frame that was 500K of the 1.78M
/// blocks entered, since `nn::` sits in the sdk module and the game calls it
/// constantly. Recognised here, the stub is folded into the terminator that
/// reaches it, which loads the slot itself when it runs.
///
/// Only the stub's code is taken as fixed. The slot is loaded on every call,
/// so a module that rebinds an import is followed like it is by the stub.
fn plt_slot(mem: &Memory, at: u32) -> Option<u32> {
    const X16: u32 = 16;
    const X17: u32 = 17;
    // The four words have to be on one page, the one the block registers.
    if at & 3 != 0 || (at & 0xFFF) > 0xFF0 {
        return None;
    }
    let word = |i: u32| mem.fetch(at.wrapping_add(4 * i)).ok();
    let (adrp, ldr, add, br) = (word(0)?, word(1)?, word(2)?, word(3)?);
    let is_adrp = adrp & 0x9F00_0000 == 0x9000_0000 && adrp & 0x1F == X16;
    let is_ldr = ldr & 0xFFC0_0000 == 0xF940_0000 && (ldr >> 5) & 0x1F == X16 && ldr & 0x1F == X17;
    let is_add = add & 0xFFC0_0000 == 0x9100_0000 && (add >> 5) & 0x1F == X16 && add & 0x1F == X16;
    if !is_adrp || !is_ldr || !is_add || br != 0xD61F_0220 {
        return None;
    }
    let offset = ((ldr >> 10) & 0xFFF) * 8;
    if (add >> 10) & 0xFFF != offset {
        return None;
    }
    let immhi = u64::from((adrp >> 5) & 0x7_FFFF);
    let immlo = u64::from((adrp >> 29) & 0b11);
    let page = u64::from(at & !0xFFF).wrapping_add(sext_u64((immhi << 2) | immlo, 21) << 12);
    Some((page as u32).wrapping_add(offset))
}

/// Fold every `CMP`/`CMN` that feeds the conditional branch immediately after
/// it into that branch.
///
/// The pair is the shape of every bounds check and loop condition compiled
/// code emits, and until blocks ran through conditional branches the two were
/// never in the same block to fold. Only a compare whose destination is the
/// zero register qualifies, so the fused op writes nothing but NZCV and the
/// rewrite cannot lose a result.
fn fuse_compares(ops: &mut [Op], exits: &mut [Branch]) {
    for branch in exits.iter_mut() {
        let Exit::Cond { cond, target } = branch.exit else {
            continue;
        };
        if branch.at == 0 {
            continue;
        }
        let prev = (branch.at - 1) as usize;
        let fused = match ops[prev] {
            Op::AddSubImm {
                rd,
                rn,
                rhs,
                carry,
                set_flags: true,
                sf,
            } if rd == ZR_DISCARD as u8 => Exit::CmpImm {
                rn,
                rhs,
                carry,
                sf,
                cond,
                target,
            },
            Op::AddSubReg {
                rd,
                rn,
                rm,
                carry,
                set_flags: true,
                sf,
            } if rd == ZR_DISCARD as u8 => Exit::CmpReg {
                rn,
                rm,
                carry,
                sf,
                cond,
                target,
            },
            _ => continue,
        };
        // The compare's slot becomes filler and the exit moves onto it, so the
        // pair is one dispatch covering two instructions.
        ops[prev] = Op::Nop;
        *branch = Branch::new(branch.at - 1, fused);
    }
}

/// Classify one instruction the way [`crate::cpu::Cpu::execute`] does, by bits
/// 28:25, the architecture's own first decode table, and translate it.
fn decode(insn: u32, pc: u32) -> Decoded {
    match (insn >> 25) & 0xF {
        0x8 | 0x9 => Decoded::Op(decode_data_proc_imm(insn, pc)),
        0x5 | 0xD => Decoded::Op(decode_data_proc_reg(insn)),
        0x4 | 0x6 | 0xC | 0xE => Decoded::Op(decode_load_store(insn, pc)),
        // Advanced SIMD and scalar floating point. No ops of their own, but
        // which of the two decoders owns the encoding is decided here rather
        // than on every execution.
        0x7 | 0xF => Decoded::Op(Op::Fp {
            insn,
            scalar: matches!((insn >> 24) & 0xFF, 0x1E | 0x1F | 0x9E | 0x9F),
            form: Cpu::fp_form(insn),
        }),
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
        0x54 => Decoded::Exit(Exit::Cond {
            cond: (insn & 0xF) as u8,
            target: branch_target(pc, sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2),
        }),
        // B #imm
        0x14..=0x17 => Decoded::Term(Term::B {
            target: branch_target(pc, sext_u64(insn & 0x3FF_FFFF, 26) << 2),
        }),
        // TBZ / TBNZ
        0x36 | 0x37 | 0xB6 | 0xB7 => Decoded::Exit(Exit::Tbz {
            rt: (insn & 0x1F) as u8,
            bit: ((((insn >> 31) & 1) << 5) | ((insn >> 19) & 0x1F)) as u8,
            nz: ((insn >> 24) & 1) == 1,
            target: branch_target(pc, sext_u64((insn >> 5) & 0x3FFF, 14) << 2),
        }),
        // CBZ / CBNZ
        0x34 | 0x35 | 0xB4 | 0xB5 => Decoded::Exit(Exit::Cbz {
            rt: (insn & 0x1F) as u8,
            sf: ((insn >> 31) & 1) == 1,
            nz: ((insn >> 24) & 1) == 1,
            target: branch_target(pc, sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2),
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
                Decoded::Term(Term::Svc {
                    imm: ((insn >> 5) & 0xFFFF) as u16,
                    next,
                })
            } else {
                Decoded::Term(Term::Interpret { insn, next })
            }
        }
        // MSR/MRS, barriers and hints. `system` retires all of them to the
        // next instruction, so they stay inside the block.
        0xD5 => {
            // The hint and barrier group used to shortcut to `Op::Nop` here
            // without asking the classifier, which is right for all of it but
            // `CLREX`, that one clears the local monitor, and skipping the
            // classifier made it a hint in this engine while the interpreter
            // honoured it. `decode_system` answers `Op::Nop` for the rest.
            if ((insn >> 22) & 0x3FF) == 0b1101010100 {
                // The same guard `try_branch_or_system` applies before handing
                // an encoding to `system`. Anything else falls through to the
                // whole-space chain, so it stays an `Interpret`.
                Decoded::Op(decode_system(insn))
            } else {
                Decoded::Op(Op::Interpret { insn })
            }
        }
        _ => Decoded::Term(Term::Interpret { insn, next }),
    }
}

/// The `MRS`/`MSR`/cache-maintenance group, as the one op it is.
///
/// [`SysOp::of`] is the classification the interpreter also runs, it walks
/// six fields and a table to reach the same answer, and this keeps its
/// result in the block instead of re-deriving it on every execution.
fn decode_system(insn: u32) -> Op {
    match SysOp::of(insn) {
        SysOp::Nop => Op::Nop,
        SysOp::Unhandled => Op::System { insn },
        op => Op::Sys { op },
    }
}

/// The register-file slot a five-bit field names when 31 means `XZR` and the
/// field is *written*. Reads need no mapping: slot 31 holds zero and nothing
/// writes it, so only the destinations are resolved here.
///
/// The mapping itself is [`Cpu::zr_write_slot`], which is what the interpreter
/// resolves through per execution. This takes the instruction word the field
/// sits in, which is what a decoder has in hand.
#[inline]
fn zr_write(n: u32) -> u8 {
    Cpu::zr_write_slot(n as u8)
}

/// The slot a five-bit field names when 31 means `SP`, read or written.
#[inline]
fn sp_form(n: u32) -> u8 {
    Cpu::x_slot(n as u8)
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
            let rd = zr_write(insn);
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
            let imm = if ((insn >> 22) & 1) == 1 {
                imm12 << 12
            } else {
                imm12
            };
            let set_flags = (op & 1) == 1;
            let sub = (op >> 1) == 1;
            Op::AddSubImm {
                // Both operands are the SP form here, and the destination is
                // only SP when the instruction does not set flags.
                rd: if set_flags {
                    zr_write(insn)
                } else {
                    sp_form(insn)
                },
                rn: sp_form(insn >> 5),
                // Subtraction is addition of the inverted operand with a carry
                // in, and both are known now.
                rhs: if sub { !imm } else { imm },
                carry: u8::from(sub),
                set_flags,
                sf,
            }
        }
        0b10010 => {
            if ((insn >> 23) & 1) == 1 {
                // MOVN / MOVZ / MOVK
                let rd = zr_write(insn);
                let imm16 = u64::from((insn >> 5) & 0xFFFF);
                let hw = if sf {
                    (insn >> 21) & 0b11
                } else {
                    (insn >> 21) & 1
                };
                let shift = hw * 16;
                match (insn >> 29) & 0b11 {
                    0b00 => Op::MovConst {
                        rd,
                        val: !(imm16 << shift) & Cpu::mask(sf),
                    },
                    0b10 => Op::MovConst {
                        rd,
                        val: (imm16 << shift) & Cpu::mask(sf),
                    },
                    0b11 => Op::MovK {
                        rd,
                        shift: shift as u8,
                        val: imm16 as u16,
                        sf,
                    },
                    _ => Op::Interpret { insn },
                }
            } else {
                // Logical immediate: the bitmask decodes to a constant.
                let immr = (insn >> 16) & 0x3F;
                let imms = (insn >> 10) & 0x3F;
                match decode_bit_mask(sf, (insn >> 22) & 1, immr, imms) {
                    Some(imm) => {
                        let opc = ((insn >> 29) & 0b11) as u8;
                        Op::LogicalImm {
                            // `ANDS` is the only one of the four whose Rd is
                            // the zero register rather than SP.
                            rd: if opc == 0b11 {
                                zr_write(insn)
                            } else {
                                sp_form(insn)
                            },
                            rn: ((insn >> 5) & 0x1F) as u8,
                            imm,
                            opc,
                            sf,
                        }
                    }
                    None => Op::Interpret { insn },
                }
            }
        }
        0b10011 => {
            let rd = zr_write(insn);
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
                let opc = (insn >> 29) & 0b11;
                match Extract::of(opc, immr, imms, sf) {
                    Some(extract) => Op::Extract {
                        rd,
                        rn,
                        extract,
                        sf,
                    },
                    None => Op::Bitfield {
                        rd,
                        rn,
                        opc: opc as u8,
                        immr: immr as u8,
                        imms: imms as u8,
                        sf,
                    },
                }
            } else {
                // EXTR
                let imm = if sf {
                    if ((insn >> 22) & 1) != 1 || ((insn >> 21) & 1) == 1 {
                        return Op::Interpret { insn };
                    }
                    (insn >> 10) & 0x3F
                } else {
                    if ((insn >> 22) & 1) == 1 || ((insn >> 21) & 1) == 1 || ((insn >> 15) & 1) == 1
                    {
                        return Op::Interpret { insn };
                    }
                    (insn >> 10) & 0x1F
                };
                Op::Extr {
                    rd,
                    rn,
                    rm: ((insn >> 16) & 0x1F) as u8,
                    imm: imm as u8,
                    sf,
                }
            }
        }
        _ => Op::Interpret { insn },
    }
}

fn decode_data_proc_reg(insn: u32) -> Op {
    let sf = (insn >> 31) & 1 == 1;
    // Every form in this group reads Rn and Rm as the zero register, and all
    // but the extended ADD/SUB write Rd as one too.
    let rd = zr_write(insn);
    let rn = ((insn >> 5) & 0x1F) as u8;
    let rm = ((insn >> 16) & 0x1F) as u8;
    match (insn >> 24) & 0x1F {
        // Logical shifted register.
        0b01010 => {
            let st = ((insn >> 22) & 0b11) as u8;
            let sa = ((insn >> 10) & 0x3F) as u8;
            let opc = ((insn >> 29) & 0b11) as u8;
            let invert = ((insn >> 21) & 1) == 1;
            if sa == 0 {
                // `mov xd, xm` and `mvn xd, xm` are both this.
                Op::LogicalReg {
                    rd,
                    rn,
                    rm,
                    opc,
                    invert,
                    sf,
                }
            } else {
                Op::LogicalShifted {
                    rd,
                    rn,
                    rm,
                    st,
                    sa,
                    opc,
                    invert,
                    sf,
                }
            }
        }
        // ADD/SUB, shifted or extended register.
        0b01011 => {
            let op = (insn >> 29) & 0b11;
            let set_flags = (op & 1) == 1;
            let sub = (op >> 1) == 1;
            let carry = u8::from(sub);
            if ((insn >> 21) & 0b111) == 0b001 {
                Op::AddSubExtended {
                    // The extended form is the other place register 31 is SP.
                    rd: if set_flags { rd } else { sp_form(insn) },
                    rn: sp_form(insn >> 5),
                    rm,
                    option: ((insn >> 13) & 0b111) as u8,
                    shift: ((insn >> 10) & 0b111) as u8,
                    carry,
                    set_flags,
                    sf,
                }
            } else {
                let st = ((insn >> 22) & 0b11) as u8;
                let sa = ((insn >> 10) & 0x3F) as u8;
                if sa == 0 {
                    // Every shift kind is the identity at zero, and this is
                    // the common form.
                    Op::AddSubReg {
                        rd,
                        rn,
                        rm,
                        carry,
                        set_flags,
                        sf,
                    }
                } else {
                    Op::AddSubShifted {
                        rd,
                        rn,
                        rm,
                        st,
                        sa,
                        carry,
                        set_flags,
                        sf,
                    }
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
            // The two-source group, under the same test the interpreter
            // makes. CRC32 stays with the interpreter, which rejects its
            // malformed operand sizes.
            (1, 1) if ((insn >> 29) & 0b11) == 0b00 => match (insn >> 10) & 0x3F {
                opcode2 @ (0b000010 | 0b000011) => Op::Divide {
                    rd,
                    rn,
                    rm,
                    signed: opcode2 & 1 == 1,
                    sf,
                },
                opcode2 @ 0b001000..=0b001011 => Op::ShiftVar {
                    rd,
                    rn,
                    rm,
                    kind: (opcode2 & 0b11) as u8,
                    sf,
                },
                _ => Op::Interpret { insn },
            },
            // The one-source group (bit counts, byte reversal) and ADC/SBC:
            // left to the interpreter.
            _ => Op::Interpret { insn },
        },
        // Three-source: the multiplies.
        0b11011 => {
            let ra = ((insn >> 10) & 0x1F) as u8;
            let sub = ((insn >> 15) & 1) == 1;
            // Ra is read, Rd written; `rd` above already resolved.

            match (insn >> 21) & 0xFF {
                0b11011000 => Op::Madd {
                    rd,
                    rn,
                    rm,
                    ra,
                    sub,
                    sf,
                },
                0b11011001 => Op::MaddLong {
                    rd,
                    rn,
                    rm,
                    ra,
                    sub,
                    signed: true,
                },
                0b11011101 => Op::MaddLong {
                    rd,
                    rn,
                    rm,
                    ra,
                    sub,
                    signed: false,
                },
                0b11011010 => Op::Mulh {
                    rd,
                    rn,
                    rm,
                    signed: true,
                },
                0b11011110 => Op::Mulh {
                    rd,
                    rn,
                    rm,
                    signed: false,
                },
                _ => Op::Interpret { insn },
            }
        }
        _ => Op::Interpret { insn },
    }
}

/// The load/store group, in the order [`crate::cpu::Cpu::execute`] tries it: the
/// literal forms first, then everything [`crate::cpu::Cpu::try_load_store`] claims.
fn decode_load_store(insn: u32, pc: u32) -> Op {
    // LDR <t>, label: the address is fixed by where the instruction is.
    if ((insn >> 27) & 0b111) == 0b011 && ((insn >> 26) & 1) == 0 && ((insn >> 24) & 0b11) == 0b00 {
        let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
        // The literal forms encode the width in `opc` alone; `Acc::of` reads
        // it as a size:opc pair, so pass the size the width implies.
        let acc = match (insn >> 30) & 0b11 {
            0b00 => Acc::Load32,
            0b01 => Acc::Load64,
            0b10 => Acc::LoadS32,
            _ => Acc::Prefetch,
        };
        return Op::LoadLiteral {
            rt: rt_slot(insn, acc),
            addr: (pc as i64).wrapping_add(imm as i64) as u32,
            acc,
        };
    }

    // The exclusive and acquire/release group, classified by the same field
    // the interpreter tests. The pairs stay with it.
    let grp_excl = (insn >> 21) & 0x1FF;
    let sz = ((insn >> 30) & 0b11) as u8;
    let rn = sp_form(insn >> 5);
    match grp_excl {
        0b001000000 => {
            return Op::StoreExclusive {
                rs: zr_write(insn >> 16),
                rt: (insn & 0x1F) as u8,
                rn,
                sz,
            }
        }
        0b001000010 => {
            return Op::LoadExclusive {
                rt: zr_write(insn),
                rn,
                sz,
            }
        }
        // `STLR`/`LDAR`: ordering is all that sets them apart from a plain
        // store or load, and a single core has nothing to order against.
        0b001000100 | 0b001000110 => {
            let acc = Acc::of(sz, u8::from(grp_excl == 0b001000110));
            return Op::load_store_imm(rt_slot(insn, acc), rn, acc, Wb::None, 0);
        }
        0b001000001 | 0b001000011 => return Op::Interpret { insn },
        _ => {}
    }
    if ((insn >> 26) & 1) == 1 {
        return Op::Interpret { insn };
    }

    let opc = ((insn >> 22) & 0b11) as u8;
    let acc = Acc::of(sz, opc);
    // Rt is read by a store and written by a load, and Rn is always the SP
    // form: the base of an addressing mode is never the zero register.
    let rt = rt_slot(insn, acc);

    // Register offset.
    if ((insn >> 27) & 0b111) == 0b111 && ((insn >> 24) & 0b11) == 0b00 && ((insn >> 21) & 1) == 1 {
        // The undefined extend encodings have no op; the interpreter faults
        // on them.
        let Some(ext) = Ext::of(((insn >> 13) & 0b111) as u8) else {
            return Op::Interpret { insn };
        };
        // `S` scales by log2 of the access size, not by the byte count.
        let shift = if ((insn >> 12) & 1) == 1 { sz } else { 0 };
        return Op::LoadStoreReg {
            rt,
            rn,
            rm: ((insn >> 16) & 0x1F) as u8,
            ext,
            shift,
            acc,
        };
    }

    // Immediate offset forms.
    if ((insn >> 27) & 0b111) == 0b111 {
        let mode = (insn >> 24) & 0b11;
        if mode == 0b01 {
            // Unsigned offset, scaled by the access size.
            let scale = 1i64 << sz;
            let offset = i64::from((insn >> 10) & 0xFFF) * scale;
            return Op::load_store_imm(rt, rn, acc, Wb::None, offset);
        }
        if mode == 0b00 && ((insn >> 21) & 1) == 0 {
            let offset = sext_u64((insn >> 12) & 0x1FF, 9) as i64;
            let wb = match (insn >> 10) & 0b11 {
                0b01 => Wb::Post,
                0b11 => Wb::Pre,
                // Unscaled (`LDUR`/`STUR`) and the unprivileged forms.
                _ => Wb::None,
            };
            return Op::load_store_imm(rt, rn, acc, wb, offset);
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
        let kind = match (load, wide, pair_opc == 0b01) {
            (true, _, true) => PairKind::Load32Sext,
            (true, true, _) => PairKind::Load64,
            (true, false, _) => PairKind::Load32,
            (false, true, _) => PairKind::Store64,
            (false, false, _) => PairKind::Store32,
        };
        // The pair forms have their own Rt/Rt2 slots: `kind`, not `acc`,
        // says whether they are read or written.
        return Op::pair(
            pair_slot(insn, kind),
            pair_slot(insn >> 10, kind),
            rn,
            (sext_u64((insn >> 15) & 0x7F, 7) as i64).wrapping_mul(scale),
            kind,
            wb,
        );
    }

    Op::Interpret { insn }
}
