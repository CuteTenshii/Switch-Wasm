//! Translation: walking forward from an address, one op per instruction.

use super::{Block, Kind, Offset, Op, Operand, Term, TermKind, ALWAYS};
use crate::cpu::a32::loadstore::Access;
use crate::cpu::a32::shift::{decode_imm_shift, expand_imm_c};
use crate::mem::{Memory, PAGE_BITS};

/// Longest run of instructions one block may cover.
const MAX_BLOCK_OPS: usize = 64;

enum Decoded {
    Op(Op),
    Term(Term),
}

/// Translate the block starting at `start`, ending at the first instruction
/// that can write the PC. Every page read is listed in [`Block::pages`].
pub(super) fn translate(mem: &Memory, start: u32) -> Block {
    let mut ops = Vec::with_capacity(MAX_BLOCK_OPS);
    let mut words = Vec::with_capacity(MAX_BLOCK_OPS);
    let mut pages = Vec::with_capacity(2);
    let mut pc = start;
    let mut term = None;
    for _ in 0..MAX_BLOCK_OPS {
        let page = pc >> PAGE_BITS;
        if !pages.contains(&page) {
            pages.push(page);
        }
        let Ok(insn) = mem.fetch(pc) else {
            term = Some(Term {
                cond: ALWAYS,
                kind: TermKind::Fetch,
            });
            break;
        };
        words.push(insn);
        match decode(insn, pc) {
            Decoded::Op(op) => ops.push(op),
            Decoded::Term(ended) => {
                term = Some(ended);
                break;
            }
        }
        pc = pc.wrapping_add(4);
    }
    ops.shrink_to_fit();
    words.shrink_to_fit();
    Block {
        start,
        ops,
        words,
        term,
        pages,
    }
}

fn op(cond: u8, kind: Kind) -> Decoded {
    Decoded::Op(Op { cond, kind })
}

fn term(cond: u8, kind: TermKind) -> Decoded {
    Decoded::Term(Term { cond, kind })
}

/// Hand `insn` to the interpreter. The interpreter tests the condition itself.
fn interpret(insn: u32, writes_pc: bool) -> Decoded {
    if writes_pc {
        term(ALWAYS, TermKind::Interpret { insn })
    } else {
        op(ALWAYS, Kind::Interpret { insn })
    }
}

fn decode(insn: u32, pc: u32) -> Decoded {
    let cond = (insn >> 28) as u8;
    if cond == 0xF {
        // The unconditional space: BLX immediate is the only branch.
        return interpret(insn, insn & 0xFE00_0000 == 0xFA00_0000);
    }
    match (insn >> 25) & 0x7 {
        0b000 | 0b001 => data_processing(insn, cond),
        0b010 => single(insn, cond, false),
        0b011 if insn & 0x10 != 0 => interpret(insn, false),
        0b011 => single(insn, cond, true),
        0b100 => block(insn, cond),
        0b101 => {
            let imm = ((insn & 0x00FF_FFFF) << 8) as i32 >> 6;
            let target = pc.wrapping_add(8).wrapping_add(imm as u32);
            if (insn >> 24) & 1 != 0 {
                term(cond, TermKind::Bl { target })
            } else {
                term(cond, TermKind::B { target })
            }
        }
        0b110 => interpret(insn, false),
        _ if (insn >> 24) & 1 != 0 => term(
            cond,
            TermKind::Svc {
                imm: (insn & 0x00FF_FFFF) as u16,
            },
        ),
        _ => interpret(insn, false),
    }
}

/// Anything in the data-processing group the translator has no op for. The
/// destination field is the PC for `BX`, `BLX` and a `MOV` to it.
fn data_processing_fallback(insn: u32) -> Decoded {
    interpret(insn, (insn >> 12) & 0xF == 15)
}

fn data_processing(insn: u32, cond: u8) -> Decoded {
    let immediate = (insn >> 25) & 1 != 0;
    // TST/TEQ/CMP/CMN without S encode the miscellaneous group.
    let misc_slot = (insn >> 23) & 0b11 == 0b10 && (insn >> 20) & 1 == 0;
    if !immediate {
        if insn & 0x90 == 0x90 {
            return if insn & 0x60 != 0 {
                extra(insn, cond)
            } else {
                data_processing_fallback(insn)
            };
        }
        if misc_slot {
            if insn & 0x0FFF_FFF0 == 0x012F_FF10 {
                return term(
                    cond,
                    TermKind::Bx {
                        rm: (insn & 0xF) as u8,
                    },
                );
            }
            return data_processing_fallback(insn);
        }
    } else if misc_slot {
        // MOVW and MOVT have bit 21 clear; a hint has an empty field mask.
        return if (insn >> 21) & 1 != 0 && (insn >> 16) & 0xF == 0 {
            op(cond, Kind::Nop)
        } else {
            data_processing_fallback(insn)
        };
    }

    let opcode = ((insn >> 21) & 0xF) as u8;
    let rd = ((insn >> 12) & 0xF) as u8;
    let compare = (0x8..=0xB).contains(&opcode);
    if !compare && rd == 15 {
        return interpret(insn, true);
    }
    let operand = if immediate {
        let imm12 = insn & 0xFFF;
        let (value, _) = expand_imm_c(imm12, false);
        if (imm12 >> 8) & 0xF == 0 {
            Operand::ImmKeep(value)
        } else {
            Operand::Imm(value)
        }
    } else if (insn >> 4) & 1 == 0 {
        let (ty, amount) = decode_imm_shift(((insn >> 5) & 0b11) as u8, ((insn >> 7) & 0x1F) as u8);
        Operand::RegImm {
            rm: (insn & 0xF) as u8,
            ty,
            amount: amount as u8,
        }
    } else {
        Operand::RegReg {
            rm: (insn & 0xF) as u8,
            ty: ((insn >> 5) & 0b11) as u8,
            rs: ((insn >> 8) & 0xF) as u8,
        }
    };
    op(
        cond,
        Kind::Dp {
            opcode,
            set_flags: (insn >> 20) & 1 != 0,
            rd,
            rn: ((insn >> 16) & 0xF) as u8,
            operand,
        },
    )
}

/// Whether a load or store ends the block: it loads the PC or moves the PC as its base.
fn transfer_writes_pc(access: Access) -> bool {
    (access.load && access.rt == 15) || (access.writes_base() && access.rn == 15)
}

fn single(insn: u32, cond: u8, register: bool) -> Decoded {
    let access = Access::of(insn);
    if transfer_writes_pc(access) {
        return interpret(insn, true);
    }
    let offset = if register {
        let (ty, amount) = decode_imm_shift(((insn >> 5) & 0b11) as u8, ((insn >> 7) & 0x1F) as u8);
        Offset::Reg {
            rm: (insn & 0xF) as u8,
            ty,
            amount: amount as u8,
        }
    } else {
        Offset::Imm((insn & 0xFFF) as u16)
    };
    op(
        cond,
        Kind::Single {
            access,
            byte: (insn >> 22) & 1 != 0,
            offset,
        },
    )
}

fn extra(insn: u32, cond: u8) -> Decoded {
    let access = Access::of(insn);
    if transfer_writes_pc(access) {
        return interpret(insn, true);
    }
    let offset = if (insn >> 22) & 1 != 0 {
        Offset::Imm((((insn >> 4) & 0xF0) | (insn & 0xF)) as u16)
    } else {
        Offset::Reg {
            rm: (insn & 0xF) as u8,
            ty: 0,
            amount: 0,
        }
    };
    op(
        cond,
        Kind::Extra {
            access,
            sh: ((insn >> 5) & 0b11) as u8,
            offset,
        },
    )
}

fn block(insn: u32, cond: u8) -> Decoded {
    let access = Access::of(insn);
    let loads_pc = access.load && insn & 0x8000 != 0;
    // The banked and PSR forms are privileged; the interpreter reports them.
    if loads_pc || (access.wb && access.rn == 15) || (insn >> 22) & 1 != 0 {
        return interpret(insn, true);
    }
    op(
        cond,
        Kind::Block {
            access,
            list: insn as u16,
        },
    )
}
