//! Translation: walking forward from an address, one op per instruction.

use super::super::ir::Code;
use super::{Block, Kind, Offset, Op, Operand, Term, TermKind, ALWAYS};
use crate::cpu::a32::loadstore::Access;
use crate::cpu::a32::shift::{decode_imm_shift, expand_imm_c};
use crate::mem::{Memory, PAGE_BITS};
use std::cell::Cell;

/// Longest run of instructions one block may cover.
const MAX_BLOCK_OPS: usize = 64;

pub(in crate::cpu::jit) enum Decoded {
    Op(Op),
    Term(Term),
}

/// Whether the A32 translator has a real op for `insn`, or hands it back to
/// the interpreter to decode again on every execution.
pub fn a32_translates(insn: u32) -> bool {
    const REPRESENTATIVE_PC: u32 = 0x0800_0000;
    !matches!(
        decode(insn, REPRESENTATIVE_PC),
        Decoded::Op(Op {
            kind: Kind::Interpret { .. },
            ..
        }) | Decoded::Term(Term {
            kind: TermKind::Interpret { .. },
            ..
        })
    )
}

/// Translate the block starting at `start`, ending at the first instruction
/// that can write the PC. Every page read is listed in [`Block::pages`].
pub(in crate::cpu::jit) fn translate(mem: &Memory, start: u32) -> Block {
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
        code: Cell::new(Code::Cold(0)),
        emitted: Cell::new(0),
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

pub(in crate::cpu::jit) fn decode(insn: u32, pc: u32) -> Decoded {
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
        // MRC p15, 0, rt, c13, c0, 2 or 3.
        _ if insn & 0x0FFF_0FDF == 0x0E1D_0F50 && (insn >> 12) & 0xF != 15 => op(
            cond,
            Kind::ReadThreadPointer {
                rt: ((insn >> 12) & 0xF) as u8,
                rw: (insn >> 5) & 1 == 0,
            },
        ),
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
                exclusive(insn, cond)
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
            if insn & 0x0FFF_FFF0 == 0x012F_FF30 && insn & 0xF != 15 {
                return term(
                    cond,
                    TermKind::Blx {
                        rm: (insn & 0xF) as u8,
                    },
                );
            }
            return data_processing_fallback(insn);
        }
    } else if misc_slot {
        // MOVW and MOVT have bit 21 clear; a hint has an empty field mask.
        let rd = ((insn >> 12) & 0xF) as u8;
        return if (insn >> 21) & 1 != 0 && (insn >> 16) & 0xF == 0 {
            op(cond, Kind::Nop)
        } else if (insn >> 21) & 1 == 0 && rd != 15 {
            op(
                cond,
                Kind::MoveWide {
                    rd,
                    imm: (((insn >> 4) & 0xF000) | (insn & 0xFFF)) as u16,
                    top: (insn >> 22) & 1 != 0,
                },
            )
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

/// `LDREX` and `STREX` of a byte, halfword or word; the rest of the group is the interpreter's.
fn exclusive(insn: u32, cond: u8) -> Decoded {
    let size = match (insn >> 21) & 0b11 {
        0b00 => 2,
        0b10 => 0,
        0b11 => 1,
        _ => return data_processing_fallback(insn),
    };
    let load = (insn >> 20) & 1 != 0;
    let (rn, rd, rt) = (
        ((insn >> 16) & 0xF) as u8,
        ((insn >> 12) & 0xF) as u8,
        (insn & 0xF) as u8,
    );
    // Bits 9:8 clear is a plain acquire or release, which leaves the monitor alone.
    let operands = [rn, rd, if load { rn } else { rt }];
    if (insn >> 23) & 0x1F != 0b00011 || (insn >> 8) & 0b11 == 0 || operands.contains(&15) {
        return data_processing_fallback(insn);
    }
    op(
        cond,
        Kind::Exclusive {
            load,
            size,
            rd,
            rn,
            rt,
        },
    )
}

fn single(insn: u32, cond: u8, register: bool) -> Decoded {
    let access = Access::of(insn);
    let byte = (insn >> 22) & 1 != 0;
    let loads_pc =
        access.load && access.rt == 15 && !byte && !(access.writes_base() && access.rn == 15);
    if transfer_writes_pc(access) && !loads_pc {
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
    if loads_pc {
        return term(cond, TermKind::LdrPc { access, offset });
    }
    op(
        cond,
        Kind::Single {
            access,
            byte,
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
    if (access.wb && access.rn == 15) || (insn >> 22) & 1 != 0 {
        return interpret(insn, true);
    }
    if loads_pc {
        return term(
            cond,
            TermKind::LdmPc {
                access,
                list: insn as u16,
            },
        );
    }
    op(
        cond,
        Kind::Block {
            access,
            list: insn as u16,
        },
    )
}
