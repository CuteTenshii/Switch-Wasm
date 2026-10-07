//! Translation: walking forward from an address, turning each instruction
//! into the one thing it does, and deciding where the block ends.

use super::ir::{Block, Branch, Exit, Op, PackedImm, Term};
use crate::cpu::bits::*;
use crate::cpu::loadstore::{pair_slot, rt_slot, Acc, Ext, PairKind, Wb};
use crate::cpu::system::SysOp;
use crate::cpu::{Cpu, ZR_DISCARD};
use crate::mem::{Memory, PAGE_BITS};

/// Longest run of instructions one block may cover, followed branches included.
const MAX_BLOCK_OPS: usize = 64;

/// Whether the block translator has a real op for `insn`, or hands it back to
/// the interpreter to decode again on every execution.
pub fn translates(insn: u32) -> bool {
    const REPRESENTATIVE_PC: u32 = 0x0800_0000;
    !matches!(
        decode(insn, REPRESENTATIVE_PC),
        Decoded::Op(Op::Interpret { .. }) | Decoded::Term(Term::Interpret { .. })
    )
}

/// One decoded instruction: part of a block's body, a conditional branch the
/// block can run through, or the terminator that ends it.
pub(super) enum Decoded {
    Op(Op),
    Exit(Exit),
    Term(Term),
}

/// Translate the block starting at `start`.
///
/// Stops at the first instruction that moves the PC somewhere it cannot
/// follow, or at [`MAX_BLOCK_OPS`]. Direct `B`s to targets outside the block
/// are followed (as an always-taken [`Exit::Jump`]); `BL`s are not. Every
/// page read is listed in [`Block::pages`] so a store to any of them drops it.
pub(super) fn translate(mem: &Memory, start: u32) -> Block {
    let mut ops = Vec::with_capacity(MAX_BLOCK_OPS);
    let mut words = Vec::with_capacity(MAX_BLOCK_OPS);
    let mut exits = Vec::new();
    let mut pages = Vec::with_capacity(2);
    // Straight-line runs so far as `(first, end)`; a branch back into one ends the block.
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
                // Not on the last slot, and not to a PLT stub (the terminator folds those).
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
            // A conditional branch becomes an early exit; translation continues
            // on the not-taken path.
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
/// that runs the stub too, with the stub's page added to the block's.
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

/// The GOT slot a PLT stub at `at` jumps through, if `at` is one:
///
/// ```text
/// adrp x16, <slot page>
/// ldr  x17, [x16, #<slot offset>]
/// add  x16, x16, #<slot offset>
/// br   x17
/// ```
///
/// Only the stub's code is taken as fixed; the slot is loaded on every call.
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

/// Fold every `CMP`/`CMN` to `XZR` that feeds the conditional branch
/// immediately after it into that branch.
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
                // Back to the encoded constant, which fits in 24 bits.
                imm: (rhs ^ 0u64.wrapping_sub(u64::from(carry))) as u32,
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
        // The compare's slot becomes filler and the exit moves onto it.
        ops[prev] = Op::Nop;
        *branch = Branch::new(branch.at - 1, fused);
        if let Some(update) = fuse_update(ops, prev, fused) {
            ops[prev - 1] = Op::Nop;
            *branch = Branch::new(branch.at - 1, update);
        }
    }
}

/// A flagless immediate `ADD`/`SUB` just ahead of a fused [`Exit::CmpImm`]
/// at `at`, folded into it as well.
fn fuse_update(ops: &[Op], at: usize, fused: Exit) -> Option<Exit> {
    let Exit::CmpImm {
        rn,
        imm,
        carry,
        sf,
        cond,
        target,
    } = fused
    else {
        return None;
    };
    let Op::AddSubImm {
        rd,
        rn: source,
        rhs,
        carry: step_carry,
        set_flags: false,
        sf: step_sf,
    } = ops[at.checked_sub(1)?]
    else {
        return None;
    };
    let step = (rhs ^ 0u64.wrapping_sub(u64::from(step_carry))) as u32;
    Some(Exit::UpdateCmpImm {
        rd,
        source,
        step: PackedImm::new(step, step_carry == 1, step_sf)?,
        rn,
        imm: PackedImm::new(imm, carry == 1, sf)?,
        cond,
        target,
    })
}

/// Classify one instruction by bits 28:25, as [`crate::cpu::Cpu::execute`]
/// does, and translate it.
pub(super) fn decode(insn: u32, pc: u32) -> Decoded {
    match (insn >> 25) & 0xF {
        0x8 | 0x9 => Decoded::Op(decode_data_proc_imm(insn, pc)),
        0x5 | 0xD => Decoded::Op(decode_data_proc_reg(insn)),
        0x4 | 0x6 | 0xC | 0xE => Decoded::Op(decode_load_store(insn, pc)),
        // Advanced SIMD and scalar floating point: pick the decoder once here.
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

/// The branch, exception-generation and system group.
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
        // MSR/MRS, barriers and hints retire to the next instruction.
        0xD5 => {
            // `decode_system`, not a `Nop` shortcut: `CLREX` clears the monitor.
            if ((insn >> 22) & 0x3FF) == 0b1101010100 {
                // The same guard `try_branch_or_system` applies.
                Decoded::Op(decode_system(insn))
            } else {
                Decoded::Op(Op::Interpret { insn })
            }
        }
        _ => Decoded::Term(Term::Interpret { insn, next }),
    }
}

/// The `MRS`/`MSR`/cache-maintenance group, classified once by [`SysOp::of`].
fn decode_system(insn: u32) -> Op {
    match SysOp::of(insn) {
        SysOp::Nop => Op::Nop,
        SysOp::Unhandled => Op::System { insn },
        op => Op::Sys { op },
    }
}

/// The register-file slot a five-bit destination field names when 31 means
/// `XZR` (see [`Cpu::zr_write_slot`]). Reads need no mapping: slot 31 is zero.
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
        // ADR / ADRP: the block just moves a constant.
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
        // ADD/SUB immediate. Bit 23 is ADDG/SUBG, which the interpreter rejects.
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
                // Rn is SP form; Rd is SP only without flags.
                rd: if set_flags {
                    zr_write(insn)
                } else {
                    sp_form(insn)
                },
                rn: sp_form(insn >> 5),
                // Subtraction is addition of the inverted operand plus one.
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
                            // `ANDS` alone writes XZR rather than SP.
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
                // Bitfield move; unallocated encodings go to the interpreter.
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
    // Rn and Rm read 31 as XZR; all but extended ADD/SUB write Rd as XZR too.
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
            if sa == 0 && opc == 1 && !invert && rn == 31 {
                return if sf {
                    Op::Mov64 { rd, rn: rm }
                } else {
                    Op::Mov32 { rd, rn: rm }
                };
            }
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
                    // Every shift kind is the identity at zero.
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
            // ADC/ADCS/SBC/SBCS.
            (0, 0) => Op::Adc {
                rd,
                rn,
                rm,
                sub: ((insn >> 30) & 1) == 1,
                set_flags: ((insn >> 29) & 1) == 1,
                sf,
            },
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
            // The two-source group. A CRC32 whose size disagrees with sf stays
            // with the interpreter.
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
                opcode2 @ 0b010000..=0b010111 if ((opcode2 & 0b11) == 0b11) == sf => Op::Crc {
                    rd,
                    rn,
                    rm,
                    sz: (opcode2 & 0b11) as u8,
                    castagnoli: ((opcode2 >> 2) & 1) == 1,
                },
                _ => Op::Interpret { insn },
            },
            // The one-source group.
            (1, 1) if ((insn >> 29) & 0b11) == 0b10 && ((insn >> 10) & 0x3F) <= 0b000110 => {
                Op::OneSource {
                    rd,
                    rn,
                    opcode: ((insn >> 10) & 0x3F) as u8,
                    sf,
                }
            }
            _ => Op::Interpret { insn },
        },
        // Three-source: the multiplies.
        0b11011 => {
            let ra = ((insn >> 10) & 0x1F) as u8;
            let sub = ((insn >> 15) & 1) == 1;
            // Ra is read, Rd written.

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
    // LDR <t>, label.
    if ((insn >> 27) & 0b111) == 0b011 && ((insn >> 26) & 1) == 0 && ((insn >> 24) & 0b11) == 0b00 {
        let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
        // Literal forms encode width in `opc` alone; `Acc::of` wants size:opc.
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

    // Exclusive and acquire/release; the pairs stay with the interpreter.
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
        // `STLR`/`LDAR`: a single core has nothing to order against.
        0b001000100 | 0b001000110 => {
            let acc = Acc::of(sz, u8::from(grp_excl == 0b001000110));
            return Op::load_store_imm(rt_slot(insn, acc), rn, acc, Wb::None, 0);
        }
        0b001000001 | 0b001000011 => return Op::Interpret { insn },
        _ => {}
    }
    if ((insn >> 26) & 1) == 1 {
        // `LDP`/`STP`/`LDNP`/`STNP` of Q registers.
        if ((insn >> 27) & 0b111) == 0b101 && ((insn >> 25) & 1) == 0 && sz == 0b10 {
            let kind = match (insn >> 22) & 1 {
                1 => PairKind::LoadQ,
                _ => PairKind::StoreQ,
            };
            let wb = match (insn >> 23) & 0b11 {
                0b01 => Wb::Post,
                0b11 => Wb::Pre,
                _ => Wb::None,
            };
            return Op::pair(
                pair_slot(insn, kind),
                pair_slot(insn >> 10, kind),
                rn,
                (sext_u64((insn >> 15) & 0x7F, 7) as i64).wrapping_mul(16),
                kind,
                wb,
            );
        }
        return Op::SimdLoadStore { insn };
    }

    let opc = ((insn >> 22) & 0b11) as u8;
    let acc = Acc::of(sz, opc);
    // Rt is read by a store and written by a load; Rn is always the SP form.
    let rt = rt_slot(insn, acc);

    // Register offset.
    if ((insn >> 27) & 0b111) == 0b111 && ((insn >> 24) & 0b11) == 0b00 && ((insn >> 21) & 1) == 1 {
        // Undefined extend encodings fault in the interpreter.
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
        // Tagged store-pair and the reserved mode are interpreter errors.
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
        // Pair forms: `kind`, not `acc`, says whether Rt/Rt2 are read or written.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_update_compare_and_branch_is_one_exit() {
        let mut mem = Memory::new();
        // add x5, x5, #4; cmp w3, #0x2d0; b.ne -8; ret
        let words = [0x910010a5u32, 0x710b407f, 0x54ffffc1, 0xd65f03c0];
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        mem.map(0x1000, &bytes).unwrap();
        let block = translate(&mem, 0x1000);
        assert_eq!(block.exits.len(), 1);
        assert_eq!(block.exits[0].at, 0);
        assert_eq!(block.exits[0].span, 3);
        assert!(block.ops.iter().all(|op| matches!(op, Op::Nop)));
    }
}
