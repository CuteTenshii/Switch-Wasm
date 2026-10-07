//! Maxwell (GM20B) shader instruction decoding.
//!
//! Bit layouts are ported from envytools' `envydis` `gm107.c` tables. An
//! encoding (or modifier) that isn't modelled decodes to
//! [`Op::Unimplemented`] with its raw bits.

mod alu;
mod alu_wide;
mod fields;
mod half;
mod op;
#[cfg(test)]
mod tests;
mod texture;
mod types;

use alu::*;
use alu_wide::*;
pub use fields::*;
use half::*;
pub use op::*;
pub use texture::*;
pub use types::*;

/// Where a pc-relative branch lands.
fn branch_base(insn: u64, pc: u32) -> u32 {
    (pc as i64 + 8 + sfield(insn, 20, 24)) as u32
}

fn branch_target(insn: u64, pc: u32) -> u32 {
    super::align_slot(branch_base(insn, pc))
}

pub fn decode_at(insn: u64, pc: u32) -> Instruction {
    let op = decode_op(insn, pc);
    // `ssy`/`pbk`/`pcnt` have no predicate.
    let pred = match op {
        Op::Ssy { .. } | Op::Pbk { .. } | Op::Pcnt { .. } => Pred::ALWAYS,
        _ => guard(insn),
    };
    Instruction { pred, op }
}

/// [`decode_at`] for a program with no branches, where the pc doesn't matter.
pub fn decode(insn: u64) -> Instruction {
    decode_at(insn, 0)
}

fn decode_op(insn: u64, pc: u32) -> Op {
    let un = Op::Unimplemented { raw: insn };
    let top = |bits: u32| insn >> (64 - bits);

    match top(16) & 0xfff8 {
        // ---- attribute space ----
        // ld a[], gm107.c 0xefd8/0xfff8
        0xefd8 => {
            if field(insn, 32, 1) != 0 || field(insn, 31, 1) != 0 {
                return un;
            }
            Op::Ld {
                dst: reg(insn, 0, 8),
                offset: field(insn, 20, 10) as u16,
                idx: reg(insn, 8, 8),
                size: attr_size(field(insn, 47, 2)),
            }
        }
        // st a[], 0xeff0/0xfff8
        0xeff0 => {
            if field(insn, 31, 1) != 0 {
                return un;
            }
            Op::St {
                offset: field(insn, 20, 10) as u16,
                idx: reg(insn, 8, 8),
                src: reg(insn, 0, 8),
                size: attr_size(field(insn, 47, 2)),
            }
        }
        // ld c[], 0xef90/0xfff8, bank at [36,41), signed 16-bit offset.
        0xef90 => Op::Ldc {
            dst: reg(insn, 0, 8),
            bank: reg(insn, 36, 5),
            offset: sfield(insn, 20, 16) as i32,
            idx: reg(insn, 8, 8),
            size: mem_size(field(insn, 48, 3)),
        },
        // ldg/stg, 0xeed0/0xeed8, signed 24-bit offset off REG_08.
        0xeed0 => Op::Ldg {
            dst: reg(insn, 0, 8),
            addr: reg(insn, 8, 8),
            offset: sfield(insn, 20, 24) as i32,
            size: mem_size(field(insn, 48, 3)),
        },
        0xeed8 => Op::Stg {
            addr: reg(insn, 8, 8),
            offset: sfield(insn, 20, 24) as i32,
            src: reg(insn, 0, 8),
            size: mem_size(field(insn, 48, 3)),
        },
        0xef48 => Op::Lds {
            dst: reg(insn, 0, 8),
            addr: reg(insn, 8, 8),
            offset: sfield(insn, 20, 24) as i32,
            size: mem_size(field(insn, 48, 3)),
        },
        0xef58 => Op::Sts {
            addr: reg(insn, 8, 8),
            offset: sfield(insn, 20, 24) as i32,
            src: reg(insn, 0, 8),
            size: mem_size(field(insn, 48, 3)),
        },
        // ld/st l[], 0xef40/0xef50.
        0xef40 => Op::Ldl {
            dst: reg(insn, 0, 8),
            addr: reg(insn, 8, 8),
            offset: sfield(insn, 20, 24) as i32,
            size: mem_size(field(insn, 48, 3)),
        },
        0xef50 => Op::Stl {
            addr: reg(insn, 8, 8),
            offset: sfield(insn, 20, 24) as i32,
            src: reg(insn, 0, 8),
            size: mem_size(field(insn, 48, 3)),
        },
        // mov dst, sN, 0xf0c8/0xfff8.
        0xf0c8 => Op::S2r {
            dst: reg(insn, 0, 8),
            sr: reg(insn, 20, 8),
        },
        // depbar/membar.
        0xf0f0 | 0xef98 => Op::Inert,
        // bar: 0xf0a8/0xfff8. The mode's bits are not contiguous.
        0xf0a8 => match (field(insn, 39, 1) << 4)
            | (field(insn, 36, 1) << 3)
            | (field(insn, 35, 1) << 2)
            | field(insn, 32, 2)
        {
            0b00010 => Op::Bar {
                mode: BarMode::RedPopc,
            },
            0b00011 => Op::Bar {
                mode: BarMode::Scan,
            },
            0b00110 => Op::Bar {
                mode: BarMode::RedAnd,
            },
            0b01010 => Op::Bar {
                mode: BarMode::RedOr,
            },
            0b10000 => Op::Bar {
                mode: BarMode::Sync,
            },
            0b10001 => Op::Bar {
                mode: BarMode::Arrive,
            },
            _ => un,
        },
        // suld 0xeb00/0xffe0 and sust 0xeb20/0xffe0.
        0xeb00 | 0xeb08 | 0xeb10 | 0xeb18 | 0xeb20 | 0xeb28 | 0xeb30 | 0xeb38 => {
            decode_surface(insn).unwrap_or(un)
        }
        // red.
        0xebf8 => {
            let (Some(op), Some(ty)) = (atom_op(field(insn, 23, 3)), atom_type(field(insn, 20, 3)))
            else {
                return un;
            };
            Op::Atom {
                dst: RZ,
                addr: reg(insn, 8, 8),
                offset: sfield(insn, 28, 20) as i32,
                src: reg(insn, 0, 8),
                op,
                ty,
                space: AtomSpace::Global,
            }
        }
        // shfl, 0xef10/0xfff8 (Eden's `maxwell.inc`, "1110 1111 0001 0---").
        0xef10 => {
            let index = if field(insn, 28, 1) != 0 {
                Operand::Imm(field(insn, 20, 5) as u32)
            } else {
                Operand::Reg(reg(insn, 20, 8))
            };
            let mask = if field(insn, 29, 1) != 0 {
                Operand::Imm(field(insn, 34, 13) as u32)
            } else {
                Operand::Reg(reg(insn, 39, 8))
            };
            Op::Shfl {
                dst: reg(insn, 0, 8),
                pred: reg(insn, 48, 3),
                src: reg(insn, 8, 8),
                index,
                mask,
                mode: match field(insn, 30, 2) {
                    0 => ShflMode::Idx,
                    1 => ShflMode::Up,
                    2 => ShflMode::Down,
                    _ => ShflMode::Bfly,
                },
            }
        }
        // sync, 0xf0f8/0xfff8.
        0xf0f8 => Op::Sync,
        _ => match top(12) {
            // ---- control flow (0xfff0 masks) ----
            0xe35 => Op::Cont,
            0xe34 => Op::Brk,
            0xe33 if flow_test_can_hold(insn) => Op::Kil,
            0xe33 => Op::Nop,
            0xe30 if flow_test_can_hold(insn) => Op::Exit,
            0xe30 => Op::Nop,
            0xe2b if field(insn, 5, 1) == 0 => Op::Pcnt {
                target: branch_target(insn, pc),
            },
            0xe2a if field(insn, 5, 1) == 0 => Op::Pbk {
                target: branch_target(insn, pc),
            },
            0xe29 if field(insn, 5, 1) == 0 => Op::Ssy {
                target: branch_target(insn, pc),
            },
            0xe24 if field(insn, 5, 1) == 0 && flow_test_can_hold(insn) => Op::Bra {
                target: branch_target(insn, pc),
            },
            // A branch the flow test can never satisfy, like the `exit` below.
            0xe24 if field(insn, 5, 1) == 0 => Op::Nop,
            0xe25 if field(insn, 5, 1) == 0 => {
                // Base plus table entry is the target, so the base is not aligned.
                Op::Brx {
                    base: branch_base(insn, pc),
                    reg: reg(insn, 8, 8),
                }
            }
            // atom.cas.
            0xeef => Op::Atom {
                dst: reg(insn, 0, 8),
                addr: reg(insn, 8, 8),
                offset: sfield(insn, 28, 20) as i32,
                src: reg(insn, 20, 8),
                op: AtomOp::Cas,
                ty: if field(insn, 49, 1) == 0 {
                    AtomType::U32
                } else {
                    AtomType::U64
                },
                space: AtomSpace::Global,
            },
            _ => decode_memory_atomic(insn).unwrap_or_else(|| decode_alu(insn)),
        },
    }
}

/// The three atomics whose opcode masks are wider than a nibble.
fn decode_memory_atomic(insn: u64) -> Option<Op> {
    match insn >> 56 {
        // atom: the op is a full nibble and the type three bits below it.
        0xed => Some(Op::Atom {
            dst: reg(insn, 0, 8),
            addr: reg(insn, 8, 8),
            offset: sfield(insn, 28, 20) as i32,
            src: reg(insn, 20, 8),
            op: atom_op(field(insn, 52, 4))?,
            ty: atom_type(field(insn, 49, 3))?,
            space: AtomSpace::Global,
        }),
        // atoms, a 22-bit offset stored in dwords, and a two-bit type.
        0xec => Some(Op::Atom {
            dst: reg(insn, 0, 8),
            addr: reg(insn, 8, 8),
            offset: (sfield(insn, 30, 22) * 4) as i32,
            src: reg(insn, 20, 8),
            op: atom_op(field(insn, 52, 4))?,
            ty: match field(insn, 28, 2) {
                0 => AtomType::U32,
                1 => AtomType::S32,
                2 => AtomType::U64,
                _ => AtomType::S64,
            },
            space: AtomSpace::Shared,
        }),
        // atoms.cast/.cas.
        _ if insn >> 55 == 0x1dc => {
            if field(insn, 53, 2) != 2 {
                return None;
            }
            Some(Op::Atom {
                dst: reg(insn, 0, 8),
                addr: reg(insn, 8, 8),
                offset: (sfield(insn, 30, 22) * 4) as i32,
                src: reg(insn, 20, 8),
                op: AtomOp::Cas,
                ty: if field(insn, 52, 1) == 0 {
                    AtomType::U32
                } else {
                    AtomType::U64
                },
                space: AtomSpace::Shared,
            })
        }
        _ => None,
    }
}

fn atom_op(bits: u64) -> Option<AtomOp> {
    Some(match bits {
        0 => AtomOp::Add,
        1 => AtomOp::Min,
        2 => AtomOp::Max,
        3 => AtomOp::Inc,
        4 => AtomOp::Dec,
        5 => AtomOp::And,
        6 => AtomOp::Or,
        7 => AtomOp::Xor,
        8 => AtomOp::Exch,
        0xa => AtomOp::SafeAdd,
        _ => return None,
    })
}

/// `tabed00sz`/`tabebf8sz`.
fn atom_type(bits: u64) -> Option<AtomType> {
    Some(match bits {
        0 => AtomType::U32,
        1 => AtomType::S32,
        2 => AtomType::U64,
        3 => AtomType::F32,
        4 => AtomType::U128,
        5 => AtomType::S64,
        _ => return None,
    })
}
