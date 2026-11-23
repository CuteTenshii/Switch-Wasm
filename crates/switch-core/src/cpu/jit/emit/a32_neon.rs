//! Advanced SIMD in A32 blocks, as wasm SIMD128. A D register is the low half of a vector
//! loaded with `load64_zero`, a Q register the whole vector; a D result stores only its low
//! half. Forms whose lanes wasm cannot reproduce bit for bit (halving, rounding,
//! most saturation, `VMAX.F32`, polynomial multiplies) stay on the interpreter.

use super::*;
use crate::cpu::a32::neon::modified_immediate;

mod op {
    pub const V128_LOAD: u32 = 0x00;
    pub const LOAD8_SPLAT: u32 = 0x07;
    pub const V128_STORE: u32 = 0x0B;
    pub const I64X2_EXTRACT: u32 = 0x1D;
    pub const NOT: u32 = 0x4D;
    pub const AND: u32 = 0x4E;
    pub const ANDNOT: u32 = 0x4F;
    pub const OR: u32 = 0x50;
    pub const XOR: u32 = 0x51;
    pub const BITSELECT: u32 = 0x52;
    pub const LOAD64_ZERO: u32 = 0x5D;
    pub const I8X16_POPCNT: u32 = 0x62;
    pub const F32X4_ABS: u32 = 0xE0;
    pub const F32X4_NEG: u32 = 0xE1;
    pub const F32X4_ADD: u32 = 0xE4;
    pub const F32X4_SUB: u32 = 0xE5;
    pub const F32X4_MUL: u32 = 0xE6;
    pub const F32X4_DIV: u32 = 0xE7;
    pub const F32X4_EQ: u32 = 0x41;
    pub const F32X4_GT: u32 = 0x44;
    pub const F32X4_GE: u32 = 0x46;
    pub const TRUNC_SAT_S: u32 = 0xF8;
    pub const TRUNC_SAT_U: u32 = 0xF9;
    pub const CONVERT_S: u32 = 0xFA;
    pub const CONVERT_U: u32 = 0xFB;
}

const SWIZZLE: u32 = 0x0E;

/// Opcode families by lane size: 8, 16, 32, then 64 bits where wasm has them.
const ADD: [u32; 4] = [0x6E, 0x8E, 0xAE, 0xCE];
const SUB: [u32; 4] = [0x71, 0x91, 0xB1, 0xD1];
const ABS: [u32; 4] = [0x60, 0x80, 0xA0, 0xC0];
const NEG: [u32; 4] = [0x61, 0x81, 0xA1, 0xC1];
const MUL: [u32; 3] = [0, 0x95, 0xB5];
const SHL: [u32; 4] = [0x6B, 0x8B, 0xAB, 0xCB];
const SHR_S: [u32; 4] = [0x6C, 0x8C, 0xAC, 0xCC];
const SHR_U: [u32; 4] = [0x6D, 0x8D, 0xAD, 0xCD];
const EQ: [u32; 3] = [0x23, 0x2D, 0x37];
const NE: [u32; 3] = [0x24, 0x2E, 0x38];
const LT_S: [u32; 3] = [0x25, 0x2F, 0x39];
const GT_S: [u32; 3] = [0x27, 0x31, 0x3B];
const GT_U: [u32; 3] = [0x28, 0x32, 0x3C];
const LE_S: [u32; 3] = [0x29, 0x33, 0x3D];
const GE_S: [u32; 3] = [0x2B, 0x35, 0x3F];
const GE_U: [u32; 3] = [0x2C, 0x36, 0x40];
const MIN_S: [u32; 3] = [0x76, 0x96, 0xB6];
const MIN_U: [u32; 3] = [0x77, 0x97, 0xB7];
const MAX_S: [u32; 3] = [0x78, 0x98, 0xB8];
const MAX_U: [u32; 3] = [0x79, 0x99, 0xB9];
const AVGR_U: [u32; 2] = [0x7B, 0x9B];
const ADD_SAT_S: [u32; 2] = [0x6F, 0x8F];
const ADD_SAT_U: [u32; 2] = [0x70, 0x90];
const SUB_SAT_S: [u32; 2] = [0x72, 0x92];
const SUB_SAT_U: [u32; 2] = [0x73, 0x93];
/// By narrow lane size: `extend_low` and `extmul_low` into lanes twice as wide.
const EXTEND_LOW_S: [u32; 3] = [0x87, 0xA7, 0xC7];
const EXTEND_LOW_U: [u32; 3] = [0x89, 0xA9, 0xC9];
const EXTMUL_LOW_S: [u32; 3] = [0x9C, 0xBC, 0xDC];
const EXTMUL_LOW_U: [u32; 3] = [0x9E, 0xBE, 0xDE];

#[derive(Clone, Copy)]
enum Three {
    And,
    Bic,
    Orr,
    Orn,
    Eor,
    Bsl,
    Bit,
    Bif,
    /// One wasm instruction on the two operands.
    Binary(u32),
    Tst(usize),
    Abd {
        size: usize,
        signed: bool,
        accumulate: bool,
    },
    Mla {
        size: usize,
        subtract: bool,
    },
    MlaF {
        subtract: bool,
    },
    AbdF,
    /// `VRECPS` (`2 - ab`) and `VRSQRTS` (`(3 - ab) / 2`).
    Step {
        sqrt: bool,
    },
}

#[derive(Clone, Copy)]
enum Imm {
    Orr,
    Bic,
    Move,
    Not,
}

#[derive(Clone, Copy)]
enum Misc {
    Rev {
        esize: u32,
        container: u32,
    },
    Not,
    Count,
    Abs {
        size: usize,
    },
    Neg {
        size: usize,
    },
    AbsF,
    NegF,
    /// Against zero: 0 greater, 1 at least, 2 equal, 3 at most, 4 less.
    CompareZero {
        size: usize,
        test: u8,
    },
    /// `VCVT` between `f32` and a 32-bit integer: 0 from signed, 1 from unsigned, 2 to signed, 3 to unsigned.
    Convert {
        kind: u8,
    },
    /// `VMOVN`: the low half of each wide lane of a Q into a D.
    Narrow {
        size: usize,
    },
}

#[derive(Clone, Copy)]
enum Shift {
    Right {
        signed: bool,
        accumulate: bool,
    },
    Left,
    /// `VSHRN` of a Q into a D.
    NarrowRight,
    /// `VSHLL` and `VMOVL` of a D into a Q.
    LongLeft {
        signed: bool,
    },
}

#[derive(Clone, Copy)]
enum Long {
    /// `VADDL` and `VSUBL`: both narrow operands widened.
    Both { subtract: bool, signed: bool },
    /// `VADDW` and `VSUBW`: the first operand is already wide.
    Wide { subtract: bool, signed: bool },
    /// `VMULL`, `VMLAL` and `VMLSL`.
    Multiply {
        accumulate: Option<bool>,
        signed: bool,
    },
}

#[derive(Clone, Copy)]
enum Neon {
    /// `VLD1` and `VST1` of consecutive D registers.
    Registers {
        load: bool,
        d: u8,
        rows: u8,
        rn: u8,
        rm: u8,
    },
    /// `VLD2`..`VLD4` and `VST2`..`VST4` of whole registers: element `e` of member `k` of row `r`.
    Structure {
        load: bool,
        d: u8,
        members: u8,
        inc: u8,
        rows: u8,
        esize: u32,
        rn: u8,
        rm: u8,
    },
    Lane {
        load: bool,
        d: u8,
        esize: u32,
        index: u32,
        rn: u8,
        rm: u8,
    },
    LoadAllLanes {
        d: u8,
        esize: u32,
        registers: u8,
        rn: u8,
        rm: u8,
    },
    Three {
        op: Three,
        quad: bool,
        d: u8,
        n: u8,
        m: u8,
    },
    Immediate {
        kind: Imm,
        quad: bool,
        d: u8,
        pattern: u64,
    },
    DupLane {
        quad: bool,
        d: u8,
        m: u8,
        esize: u32,
        index: u32,
    },
    Ext {
        quad: bool,
        d: u8,
        n: u8,
        m: u8,
        bytes: u32,
    },
    Misc {
        op: Misc,
        quad: bool,
        d: u8,
        m: u8,
    },
    Shift {
        op: Shift,
        quad: bool,
        d: u8,
        m: u8,
        esize: u32,
        amount: u32,
    },
    ByScalar {
        multiply: u8,
        float: bool,
        quad: bool,
        size: usize,
        d: u8,
        n: u8,
        m: u8,
        index: u32,
    },
    Long {
        op: Long,
        size: usize,
        d: u8,
        n: u8,
        m: u8,
    },
}

/// Whether `insn` is one the emitter writes as Advanced SIMD.
pub(super) fn neon_writable(insn: u32) -> bool {
    plan(insn).is_some()
}

fn plan(insn: u32) -> Option<Neon> {
    if insn >> 28 != 0xF {
        return None;
    }
    if (insn >> 25) & 7 == 0b001 {
        return data(insn);
    }
    if (insn >> 24) & 0xFF == 0xF4 {
        return load_store(insn);
    }
    None
}

/// `D:Vd`, `N:Vn` and `M:Vm`.
fn registers(insn: u32) -> (u8, u8, u8) {
    (
        (((insn >> 22) & 1) as u8) << 4 | ((insn >> 12) & 0xF) as u8,
        (((insn >> 7) & 1) as u8) << 4 | ((insn >> 16) & 0xF) as u8,
        (((insn >> 5) & 1) as u8) << 4 | (insn & 0xF) as u8,
    )
}

/// Quad operands name even D registers.
fn even(quad: bool, regs: &[u8]) -> bool {
    !quad || regs.iter().all(|r| r & 1 == 0)
}

fn data(insn: u32) -> Option<Neon> {
    if (insn >> 23) & 1 == 0 {
        three_same(insn)
    } else if (insn >> 4) & 1 != 0 {
        if (insn >> 7) & 1 == 0 && (insn >> 19) & 0b111 == 0 {
            immediate(insn)
        } else {
            shift_immediate(insn)
        }
    } else if (insn >> 20) & 0b11 == 0b11 {
        if (insn >> 24) & 1 == 0 {
            ext(insn)
        } else {
            two_reg(insn)
        }
    } else if (insn >> 6) & 1 == 0 {
        three_different(insn)
    } else {
        by_scalar(insn)
    }
}

fn three_same(insn: u32) -> Option<Neon> {
    let unsigned = (insn >> 24) & 1 != 0;
    let size = ((insn >> 20) & 0b11) as usize;
    let opc = (insn >> 8) & 0xF;
    let op = (insn >> 4) & 1 != 0;
    let quad = (insn >> 6) & 1 != 0;
    let (d, n, m) = registers(insn);
    if !even(quad, &[d, n, m]) {
        return None;
    }
    let three = match (opc, op) {
        (0x1, true) => match (unsigned, size) {
            (false, 0) => Three::And,
            (false, 1) => Three::Bic,
            (false, 2) => Three::Orr,
            (false, _) => Three::Orn,
            (true, 0) => Three::Eor,
            (true, 1) => Three::Bsl,
            (true, 2) => Three::Bit,
            (true, _) => Three::Bif,
        },
        (0x8, false) => Three::Binary(if unsigned { SUB[size] } else { ADD[size] }),
        (0x8, true) if size < 3 => {
            if unsigned {
                Three::Binary(EQ[size])
            } else {
                Three::Tst(size)
            }
        }
        (0x6, _) if size < 3 => Three::Binary(match (op, unsigned) {
            (false, false) => MAX_S[size],
            (false, true) => MAX_U[size],
            (true, false) => MIN_S[size],
            (true, true) => MIN_U[size],
        }),
        (0x3, _) if size < 3 => Three::Binary(match (op, unsigned) {
            (false, false) => GT_S[size],
            (false, true) => GT_U[size],
            (true, false) => GE_S[size],
            (true, true) => GE_U[size],
        }),
        (0x0, true) if size < 2 => Three::Binary(if unsigned {
            ADD_SAT_U[size]
        } else {
            ADD_SAT_S[size]
        }),
        (0x2, true) if size < 2 => Three::Binary(if unsigned {
            SUB_SAT_U[size]
        } else {
            SUB_SAT_S[size]
        }),
        (0x1, false) if unsigned && size < 2 => Three::Binary(AVGR_U[size]),
        (0x7, _) if size < 3 => Three::Abd {
            size,
            signed: !unsigned,
            accumulate: op,
        },
        (0x9, false) if (1..3).contains(&size) => Three::Mla {
            size,
            subtract: unsigned,
        },
        (0x9, true) if !unsigned && (1..3).contains(&size) => Three::Binary(MUL[size]),
        (0xD, false) if size & 1 == 0 => match (unsigned, size & 0b10 != 0) {
            (false, false) => Three::Binary(op::F32X4_ADD),
            (false, true) => Three::Binary(op::F32X4_SUB),
            (true, true) => Three::AbdF,
            (true, false) => return None,
        },
        (0xD, true) if size & 1 == 0 => {
            if unsigned {
                Three::Binary(op::F32X4_MUL)
            } else {
                Three::MlaF {
                    subtract: size & 0b10 != 0,
                }
            }
        }
        (0xE, false) if size & 1 == 0 => Three::Binary(match (unsigned, size & 0b10 != 0) {
            (false, _) => op::F32X4_EQ,
            (true, true) => op::F32X4_GT,
            (true, false) => op::F32X4_GE,
        }),
        (0xF, true) if size & 1 == 0 => Three::Step {
            sqrt: size & 0b10 != 0,
        },
        _ => return None,
    };
    Some(Neon::Three {
        op: three,
        quad,
        d,
        n,
        m,
    })
}

fn immediate(insn: u32) -> Option<Neon> {
    let quad = (insn >> 6) & 1 != 0;
    let (d, _, _) = registers(insn);
    let cmode = (insn >> 8) & 0xF;
    let op = (insn >> 5) & 1;
    let imm8 = (((insn >> 24) & 1) << 7) | (((insn >> 16) & 0x7) << 4) | (insn & 0xF);
    if !even(quad, &[d]) {
        return None;
    }
    let pattern = modified_immediate(cmode, op, imm8)?;
    let kind = match (op, cmode) {
        (0, _) if cmode & 1 != 0 && cmode < 0b1100 => Imm::Orr,
        (_, _) if cmode & 1 != 0 && cmode < 0b1100 => Imm::Bic,
        (1, 0b1110) | (0, _) => Imm::Move,
        _ => Imm::Not,
    };
    Some(Neon::Immediate {
        kind,
        quad,
        d,
        pattern,
    })
}

fn ext(insn: u32) -> Option<Neon> {
    let quad = (insn >> 6) & 1 != 0;
    let (d, n, m) = registers(insn);
    let bytes = (insn >> 8) & 0xF;
    if !even(quad, &[d, n, m]) || (!quad && bytes >= 8) {
        return None;
    }
    Some(Neon::Ext {
        quad,
        d,
        n,
        m,
        bytes,
    })
}

fn two_reg(insn: u32) -> Option<Neon> {
    match (insn >> 8) & 0xF {
        0b1000..=0b1011 => None,
        0b1100 => dup_lane(insn),
        _ => misc(insn),
    }
}

fn dup_lane(insn: u32) -> Option<Neon> {
    let quad = (insn >> 6) & 1 != 0;
    let (d, _, m) = registers(insn);
    let imm4 = (insn >> 16) & 0xF;
    let (esize, index) = if imm4 & 1 != 0 {
        (8, imm4 >> 1)
    } else if imm4 & 0b10 != 0 {
        (16, imm4 >> 2)
    } else if imm4 & 0b100 != 0 {
        (32, imm4 >> 3)
    } else {
        return None;
    };
    even(quad, &[d]).then_some(Neon::DupLane {
        quad,
        d,
        m,
        esize,
        index,
    })
}

fn misc(insn: u32) -> Option<Neon> {
    let quad = (insn >> 6) & 1 != 0;
    let (d, _, m) = registers(insn);
    let size = ((insn >> 18) & 0b11) as usize;
    let table = (insn >> 16) & 0b11;
    let op = (insn >> 7) & 0b1111;
    let narrowing = table == 0b10 && op == 0b0100 && (insn >> 6) & 1 == 0;
    if !narrowing && !even(quad, &[d, m]) {
        return None;
    }
    if narrowing && m & 1 != 0 {
        return None;
    }
    let misc = match (table, op) {
        (0b00, 0b0000..=0b0010) => {
            let esize = 8 << size;
            let container = 64 >> op;
            if container <= esize {
                return None;
            }
            Misc::Rev { esize, container }
        }
        (0b00, 0b1010) if size == 0 => Misc::Count,
        (0b00, 0b1011) => Misc::Not,
        (0b01, 0b0000..=0b0100) if size < 3 => Misc::CompareZero {
            size,
            test: op as u8,
        },
        (0b01, 0b0110) => Misc::Abs { size },
        (0b01, 0b0111) => Misc::Neg { size },
        (0b01, 0b1110) if size == 2 => Misc::AbsF,
        (0b01, 0b1111) if size == 2 => Misc::NegF,
        (0b11, 0b1100..=0b1111) if size == 2 => Misc::Convert { kind: op as u8 & 3 },
        (0b10, 0b0100) if narrowing && size < 3 => Misc::Narrow { size },
        _ => return None,
    };
    Some(Neon::Misc {
        op: misc,
        quad,
        d,
        m,
    })
}

/// The element size and shift field of an immediate shift: the highest set bit of `L:imm6`.
fn shift_fields(insn: u32) -> (u32, u32) {
    let imm6 = (insn >> 16) & 0x3F;
    let esize = if (insn >> 7) & 1 != 0 {
        64
    } else if imm6 & 0b10_0000 != 0 {
        32
    } else if imm6 & 0b01_0000 != 0 {
        16
    } else {
        8
    };
    let field = if esize == 64 {
        imm6
    } else {
        imm6 & (esize - 1)
    };
    (esize, field)
}

fn shift_immediate(insn: u32) -> Option<Neon> {
    let unsigned = (insn >> 24) & 1 != 0;
    let quad = (insn >> 6) & 1 != 0;
    let (d, _, m) = registers(insn);
    let (esize, field) = shift_fields(insn);
    let (op, regs_ok) = match (insn >> 8) & 0xF {
        opc @ (0x0 | 0x1) => (
            Shift::Right {
                signed: !unsigned,
                accumulate: opc == 1,
            },
            even(quad, &[d, m]),
        ),
        0x5 if !unsigned => (Shift::Left, even(quad, &[d, m])),
        0x8 if !unsigned && !quad && esize < 64 => (Shift::NarrowRight, m & 1 == 0),
        0xA if !quad && esize < 64 => (Shift::LongLeft { signed: !unsigned }, d & 1 == 0),
        _ => return None,
    };
    regs_ok.then_some(Neon::Shift {
        op,
        quad,
        d,
        m,
        esize,
        amount: match op {
            Shift::Left | Shift::LongLeft { .. } => field,
            _ => esize - field,
        },
    })
}

fn by_scalar(insn: u32) -> Option<Neon> {
    let quad = (insn >> 24) & 1 != 0;
    let size = ((insn >> 20) & 0b11) as usize;
    let opc = (insn >> 8) & 0xF;
    let (d, n, _) = registers(insn);
    let (m, index) = if size == 0b01 {
        (
            (insn & 0x7) as u8,
            (((insn >> 5) & 1) << 1) | ((insn >> 3) & 1),
        )
    } else {
        ((insn & 0xF) as u8, (insn >> 5) & 1)
    };
    if !even(quad, &[d, n]) || !(1..3).contains(&size) {
        return None;
    }
    let (float, multiply) = match opc {
        0x9 => (true, 0),
        0x1 => (true, 1),
        0x5 => (true, 2),
        0x8 => (false, 0),
        0x0 => (false, 1),
        0x4 => (false, 2),
        _ => return None,
    };
    if float && size != 2 {
        return None;
    }
    Some(Neon::ByScalar {
        multiply,
        float,
        quad,
        size,
        d,
        n,
        m,
        index,
    })
}

fn three_different(insn: u32) -> Option<Neon> {
    let unsigned = (insn >> 24) & 1 != 0;
    let size = ((insn >> 20) & 0b11) as usize;
    let opc = (insn >> 8) & 0xF;
    let (d, n, m) = registers(insn);
    if size > 2 || d & 1 != 0 {
        return None;
    }
    let signed = !unsigned;
    let op = match opc {
        0x0 | 0x2 => Long::Both {
            subtract: opc == 2,
            signed,
        },
        0x1 | 0x3 if n & 1 == 0 => Long::Wide {
            subtract: opc == 3,
            signed,
        },
        0x8 | 0xA => Long::Multiply {
            accumulate: Some(opc == 0xA),
            signed,
        },
        0xC => Long::Multiply {
            accumulate: None,
            signed,
        },
        _ => return None,
    };
    Some(Neon::Long { op, size, d, n, m })
}

fn load_store(insn: u32) -> Option<Neon> {
    let single = (insn >> 23) & 1 != 0;
    let load = (insn >> 21) & 1 != 0;
    let rn = ((insn >> 16) & 0xF) as u8;
    let (d, _, _) = registers(insn);
    let rm = (insn & 0xF) as u8;
    if rn == 15 && rm != 15 {
        return None;
    }
    if !single {
        let (members, inc, rows) = match (insn >> 8) & 0xF {
            0b0111 => (1, 1, 1),
            0b1010 => (1, 1, 2),
            0b0110 => (1, 1, 3),
            0b0010 => (1, 1, 4),
            0b1000 => (2, 1, 1),
            0b1001 => (2, 2, 1),
            0b0011 => (2, 2, 2),
            0b0100 => (3, 1, 1),
            0b0101 => (3, 2, 1),
            0b0000 => (4, 1, 1),
            0b0001 => (4, 2, 1),
            _ => return None,
        };
        if members == 1 {
            return Some(Neon::Registers {
                load,
                d,
                rows,
                rn,
                rm,
            });
        }
        return Some(Neon::Structure {
            load,
            d,
            members,
            inc,
            rows,
            esize: 8 << ((insn >> 6) & 0b11),
            rn,
            rm,
        });
    }
    if (insn >> 8) & 0b11 != 0 {
        return None;
    }
    if (insn >> 10) & 0b11 == 0b11 {
        let size = (insn >> 6) & 0b11;
        if !load || size == 3 {
            return None;
        }
        return Some(Neon::LoadAllLanes {
            d,
            esize: 8 << size,
            registers: 1 + ((insn >> 5) & 1) as u8,
            rn,
            rm,
        });
    }
    let index_align = (insn >> 4) & 0xF;
    let (esize, index) = match (insn >> 10) & 0b11 {
        0b00 => (8, index_align >> 1),
        0b01 => (16, index_align >> 2),
        _ => (32, index_align >> 3),
    };
    Some(Neon::Lane {
        load,
        d,
        esize,
        index,
        rn,
        rm,
    })
}

impl Emitter<'_> {
    /// Push a D register as a vector with the high half zero, or a Q register.
    fn neon_push(&mut self, quad: bool, d: u8) {
        self.f.local_get(STATE);
        let at = self.d_at(d);
        if quad {
            self.f.simd_mem(op::V128_LOAD, ALIGN_8, at);
        } else {
            self.f.simd_mem(op::LOAD64_ZERO, ALIGN_8, at);
        }
    }

    /// Pop a vector into a Q register, or its low half into a D register.
    fn neon_pop(&mut self, quad: bool, d: u8) {
        let at = self.d_at(d);
        self.f.local_set(L_V0);
        self.f.local_get(STATE);
        self.f.local_get(L_V0);
        if quad {
            self.f.simd_mem(op::V128_STORE, ALIGN_8, at);
        } else {
            self.f.simd_lane(op::I64X2_EXTRACT, 0);
            self.f.i64_store(ALIGN_8, at);
        }
    }

    /// Push the `esize`-bit lane at byte offset `at` from the state, splatted across the vector.
    fn neon_splat_from(&mut self, esize: u32, at: u32) {
        self.f.local_get(STATE);
        self.f
            .simd_mem(op::LOAD8_SPLAT + (esize / 8).trailing_zeros(), 0, at);
    }

    fn neon_f32_const(&mut self, value: f32) {
        let lane = value.to_bits().to_le_bytes();
        self.f.v128_const([lane; 4].concat().try_into().unwrap());
    }

    /// Write Advanced SIMD instruction `insn`. The caller checked [`neon_writable`].
    pub(super) fn a32_neon(&mut self, insn: u32, at: usize, pc: u32) {
        match plan(insn).expect("`neon_writable` was checked") {
            Neon::Registers {
                load,
                d,
                rows,
                rn,
                rm,
            } => {
                let regs: Vec<u8> = (0..rows).map(|i| (d + i) & 0x1F).collect();
                self.neon_address(rn, pc);
                self.vfp_copy(&regs, true, 8, load, at);
                self.neon_writeback(rn, rm, u32::from(rows) * 8, pc);
            }
            Neon::Structure {
                load,
                d,
                members,
                inc,
                rows,
                esize,
                rn,
                rm,
            } => {
                let total = u32::from(members) * u32::from(rows) * 8;
                self.neon_address(rn, pc);
                if load {
                    self.page_for_read(total, at);
                } else {
                    self.page_for_write(total, at);
                }
                let ebytes = esize / 8;
                let mut offset = 0;
                for row in 0..rows {
                    for e in 0..64 / esize {
                        for k in 0..members {
                            let reg = (d + row + k * inc) & 0x1F;
                            let slot = self.d_at(reg) + e * ebytes;
                            self.neon_element(load, ebytes, slot, offset);
                            offset += ebytes;
                        }
                    }
                }
                self.neon_writeback(rn, rm, total, pc);
            }
            Neon::Lane {
                load,
                d,
                esize,
                index,
                rn,
                rm,
            } => {
                let ebytes = esize / 8;
                self.neon_address(rn, pc);
                if load {
                    self.page_for_read(ebytes, at);
                } else {
                    self.page_for_write(ebytes, at);
                }
                let slot = self.d_at(d) + index * ebytes;
                self.neon_element(load, ebytes, slot, 0);
                self.neon_writeback(rn, rm, ebytes, pc);
            }
            Neon::LoadAllLanes {
                d,
                esize,
                registers,
                rn,
                rm,
            } => {
                let ebytes = esize / 8;
                self.neon_address(rn, pc);
                self.page_for_read(ebytes, at);
                self.in_page();
                match ebytes {
                    1 => self.f.i64_load8_u(0),
                    2 => self.f.i64_load16_u(UNALIGNED, 0),
                    _ => self.f.i64_load32_u(UNALIGNED, 0),
                }
                self.f.i64_const(match esize {
                    8 => 0x0101_0101_0101_0101,
                    16 => 0x0001_0001_0001_0001,
                    _ => 0x0000_0001_0000_0001,
                });
                self.f.i64_mul();
                self.f.local_set(L_T);
                for i in 0..registers {
                    self.f.local_get(STATE);
                    self.f.local_get(L_T);
                    self.f.i64_store(ALIGN_8, self.d_at((d + i) & 0x1F));
                }
                self.neon_writeback(rn, rm, ebytes, pc);
            }
            Neon::Three { op, quad, d, n, m } => self.neon_three(op, quad, d, n, m),
            Neon::Immediate {
                kind,
                quad,
                d,
                pattern,
            } => {
                let wide = |p: u64| {
                    let mut bytes = [0u8; 16];
                    bytes[..8].copy_from_slice(&p.to_le_bytes());
                    bytes[8..].copy_from_slice(&p.to_le_bytes());
                    bytes
                };
                match kind {
                    Imm::Orr | Imm::Bic => {
                        self.neon_push(quad, d);
                        self.f.v128_const(wide(pattern));
                        self.f.simd(if matches!(kind, Imm::Orr) {
                            op::OR
                        } else {
                            op::ANDNOT
                        });
                    }
                    Imm::Move => self.f.v128_const(wide(pattern)),
                    Imm::Not => self.f.v128_const(wide(!pattern)),
                }
                self.neon_pop(quad, d);
            }
            Neon::DupLane {
                quad,
                d,
                m,
                esize,
                index,
            } => {
                let slot = self.d_at(m) + index * (esize / 8);
                self.neon_splat_from(esize, slot);
                self.neon_pop(quad, d);
            }
            Neon::Ext {
                quad,
                d,
                n,
                m,
                bytes,
            } => {
                self.neon_push(quad, n);
                self.neon_push(quad, m);
                let width = if quad { 16 } else { 8 };
                let mut lanes = [0u8; 16];
                for (i, lane) in lanes.iter_mut().enumerate() {
                    let at = i as u32 + bytes;
                    *lane = if at < width {
                        at as u8
                    } else {
                        (16 + at - width) as u8
                    };
                }
                self.f.i8x16_shuffle(lanes);
                self.neon_pop(quad, d);
            }
            Neon::Misc { op, quad, d, m } => self.neon_misc(op, quad, d, m),
            Neon::Shift {
                op,
                quad,
                d,
                m,
                esize,
                amount,
            } => self.neon_shift(op, quad, d, m, esize, amount),
            Neon::ByScalar {
                multiply,
                float,
                quad,
                size,
                d,
                n,
                m,
                index,
            } => {
                let esize = 8u32 << size;
                if multiply != 0 {
                    self.neon_push(quad, d);
                }
                self.neon_push(quad, n);
                self.neon_splat_from(esize, self.d_at(m) + index * (esize / 8));
                self.f.simd(if float { op::F32X4_MUL } else { MUL[size] });
                match multiply {
                    0 => {}
                    1 => self.f.simd(if float { op::F32X4_ADD } else { ADD[size] }),
                    _ => self.f.simd(if float { op::F32X4_SUB } else { SUB[size] }),
                }
                self.neon_pop(quad, d);
            }
            Neon::Long { op, size, d, n, m } => self.neon_long(op, size, d, n, m),
        }
    }

    /// The base register's address into `L_ADDR` and, for the writeback, into `L_A`.
    fn neon_address(&mut self, rn: u8, pc: u32) {
        self.a32_read(rn, pc);
        self.f.local_tee(L_A);
        self.f.i32_wrap_i64();
        self.set_addr32();
    }

    /// `Rm` 15 leaves the base alone; 13 advances it by the transfer size; any other adds it.
    fn neon_writeback(&mut self, rn: u8, rm: u8, bytes: u32, pc: u32) {
        match rm {
            15 => return,
            13 => self.f.i64_const(i64::from(bytes)),
            _ => self.a32_read(rm, pc),
        }
        self.f.local_get(L_A);
        self.f.i64_add();
        self.mask_to(false);
        self.store_reg(rn);
    }

    /// One element between the guest memory `offset` bytes past `L_ADDR` and the state at `slot`.
    fn neon_element(&mut self, load: bool, ebytes: u32, slot: u32, offset: u32) {
        if load {
            self.f.local_get(STATE);
            self.in_page();
        } else {
            self.in_page();
            self.f.local_get(STATE);
        }
        match (load, ebytes) {
            (true, 1) => {
                self.f.i64_load8_u(offset);
                self.f.i64_store8(slot);
            }
            (true, 2) => {
                self.f.i64_load16_u(UNALIGNED, offset);
                self.f.i64_store16(UNALIGNED, slot);
            }
            (true, 4) => {
                self.f.i64_load32_u(UNALIGNED, offset);
                self.f.i64_store32(UNALIGNED, slot);
            }
            (true, _) => {
                self.f.i64_load(UNALIGNED, offset);
                self.f.i64_store(UNALIGNED, slot);
            }
            (false, 1) => {
                self.f.i64_load8_u(slot);
                self.f.i64_store8(offset);
            }
            (false, 2) => {
                self.f.i64_load16_u(UNALIGNED, slot);
                self.f.i64_store16(UNALIGNED, offset);
            }
            (false, 4) => {
                self.f.i64_load32_u(UNALIGNED, slot);
                self.f.i64_store32(UNALIGNED, offset);
            }
            (false, _) => {
                self.f.i64_load(UNALIGNED, slot);
                self.f.i64_store(UNALIGNED, offset);
            }
        }
    }

    fn neon_three(&mut self, op: Three, quad: bool, d: u8, n: u8, m: u8) {
        match op {
            Three::Bsl | Three::Bit | Three::Bif => {
                // `bitselect(x, y, mask)` is `x & mask | y & !mask`.
                let (x, y, mask) = match op {
                    Three::Bsl => (n, m, d),
                    Three::Bit => (n, d, m),
                    _ => (d, n, m),
                };
                self.neon_push(quad, x);
                self.neon_push(quad, y);
                self.neon_push(quad, mask);
                self.f.simd(op::BITSELECT);
            }
            Three::And | Three::Bic | Three::Orr | Three::Orn | Three::Eor => {
                self.neon_push(quad, n);
                self.neon_push(quad, m);
                match op {
                    Three::And => self.f.simd(op::AND),
                    Three::Bic => self.f.simd(op::ANDNOT),
                    Three::Orr => self.f.simd(op::OR),
                    Three::Orn => {
                        self.f.simd(op::NOT);
                        self.f.simd(op::OR);
                    }
                    _ => self.f.simd(op::XOR),
                }
            }
            Three::Binary(opcode) => {
                self.neon_push(quad, n);
                self.neon_push(quad, m);
                self.f.simd(opcode);
            }
            Three::Tst(size) => {
                self.neon_push(quad, n);
                self.neon_push(quad, m);
                self.f.simd(op::AND);
                self.f.v128_const([0; 16]);
                self.f.simd(NE[size]);
            }
            Three::Abd {
                size,
                signed,
                accumulate,
            } => {
                if accumulate {
                    self.neon_push(quad, d);
                }
                self.neon_push(quad, n);
                self.f.local_set(L_V1);
                self.neon_push(quad, m);
                self.f.local_set(L_V2);
                let (max, min) = if signed {
                    (MAX_S[size], MIN_S[size])
                } else {
                    (MAX_U[size], MIN_U[size])
                };
                self.f.local_get(L_V1);
                self.f.local_get(L_V2);
                self.f.simd(max);
                self.f.local_get(L_V1);
                self.f.local_get(L_V2);
                self.f.simd(min);
                self.f.simd(SUB[size]);
                if accumulate {
                    self.f.simd(ADD[size]);
                }
            }
            Three::Mla { size, subtract } => {
                self.neon_push(quad, d);
                self.neon_push(quad, n);
                self.neon_push(quad, m);
                self.f.simd(MUL[size]);
                self.f.simd(if subtract { SUB[size] } else { ADD[size] });
            }
            Three::MlaF { subtract } => {
                self.neon_push(quad, d);
                self.neon_push(quad, n);
                self.neon_push(quad, m);
                self.f.simd(op::F32X4_MUL);
                self.f.simd(if subtract {
                    op::F32X4_SUB
                } else {
                    op::F32X4_ADD
                });
            }
            Three::AbdF => {
                self.neon_push(quad, n);
                self.neon_push(quad, m);
                self.f.simd(op::F32X4_SUB);
                self.f.simd(op::F32X4_ABS);
            }
            Three::Step { sqrt } => {
                self.neon_f32_const(if sqrt { 3.0 } else { 2.0 });
                self.neon_push(quad, n);
                self.neon_push(quad, m);
                self.f.simd(op::F32X4_MUL);
                self.f.simd(op::F32X4_SUB);
                if sqrt {
                    self.neon_f32_const(2.0);
                    self.f.simd(op::F32X4_DIV);
                }
            }
        }
        self.neon_pop(quad, d);
    }

    fn neon_misc(&mut self, op: Misc, quad: bool, d: u8, m: u8) {
        if let Misc::Narrow { size } = op {
            self.neon_push(true, m);
            let narrow = 1usize << size;
            let mut lanes = [0u8; 16];
            for (i, lane) in lanes.iter_mut().take(8).enumerate() {
                *lane = ((i / narrow) * 2 * narrow + i % narrow) as u8;
            }
            self.f.v128_const(lanes);
            self.f.simd(SWIZZLE);
            self.neon_pop(false, d);
            return;
        }
        self.neon_push(quad, m);
        match op {
            Misc::Rev { esize, container } => {
                let (eb, cb) = ((esize / 8) as usize, (container / 8) as usize);
                let mut lanes = [0u8; 16];
                for (i, lane) in lanes.iter_mut().enumerate() {
                    let (group, within) = (i / cb, i % cb);
                    let (elem, byte) = (within / eb, within % eb);
                    *lane = (group * cb + (cb / eb - 1 - elem) * eb + byte) as u8;
                }
                self.f.v128_const(lanes);
                self.f.simd(SWIZZLE);
            }
            Misc::Not => self.f.simd(op::NOT),
            Misc::Count => self.f.simd(op::I8X16_POPCNT),
            Misc::Abs { size } => self.f.simd(ABS[size]),
            Misc::Neg { size } => self.f.simd(NEG[size]),
            Misc::AbsF => self.f.simd(op::F32X4_ABS),
            Misc::NegF => self.f.simd(op::F32X4_NEG),
            Misc::CompareZero { size, test } => {
                self.f.v128_const([0; 16]);
                self.f.simd(match test {
                    0 => GT_S[size],
                    1 => GE_S[size],
                    2 => EQ[size],
                    3 => LE_S[size],
                    _ => LT_S[size],
                });
            }
            Misc::Convert { kind } => self.f.simd(match kind {
                0 => op::CONVERT_S,
                1 => op::CONVERT_U,
                2 => op::TRUNC_SAT_S,
                _ => op::TRUNC_SAT_U,
            }),
            Misc::Narrow { .. } => unreachable!("handled above"),
        }
        self.neon_pop(quad, d);
    }

    fn neon_shift(&mut self, op: Shift, quad: bool, d: u8, m: u8, esize: u32, amount: u32) {
        let size = (esize / 8).trailing_zeros() as usize;
        match op {
            Shift::Right { signed, accumulate } => {
                if accumulate {
                    self.neon_push(quad, d);
                }
                if amount == esize && !signed {
                    self.f.v128_const([0; 16]);
                } else {
                    self.neon_push(quad, m);
                    self.f.i32_const(amount.min(esize - 1) as i32);
                    self.f.simd(if signed { SHR_S[size] } else { SHR_U[size] });
                }
                if accumulate {
                    self.f.simd(ADD[size]);
                }
                self.neon_pop(quad, d);
            }
            Shift::Left => {
                self.neon_push(quad, m);
                self.f.i32_const(amount as i32);
                self.f.simd(SHL[size]);
                self.neon_pop(quad, d);
            }
            Shift::NarrowRight => {
                self.neon_push(true, m);
                self.f.i32_const(amount as i32);
                self.f.simd(SHR_U[size + 1]);
                let narrow = 1usize << size;
                let mut lanes = [0u8; 16];
                for (i, lane) in lanes.iter_mut().take(8).enumerate() {
                    *lane = ((i / narrow) * 2 * narrow + i % narrow) as u8;
                }
                self.f.v128_const(lanes);
                self.f.simd(SWIZZLE);
                self.neon_pop(false, d);
            }
            Shift::LongLeft { signed } => {
                self.neon_push(false, m);
                self.f.simd(if signed {
                    EXTEND_LOW_S[size]
                } else {
                    EXTEND_LOW_U[size]
                });
                if amount != 0 {
                    self.f.i32_const(amount as i32);
                    self.f.simd(SHL[size + 1]);
                }
                self.neon_pop(true, d);
            }
        }
    }

    fn neon_long(&mut self, op: Long, size: usize, d: u8, n: u8, m: u8) {
        let extend = |signed: bool| {
            if signed {
                EXTEND_LOW_S[size]
            } else {
                EXTEND_LOW_U[size]
            }
        };
        match op {
            Long::Both { subtract, signed } => {
                self.neon_push(false, n);
                self.f.simd(extend(signed));
                self.neon_push(false, m);
                self.f.simd(extend(signed));
                self.f.simd(if subtract {
                    SUB[size + 1]
                } else {
                    ADD[size + 1]
                });
            }
            Long::Wide { subtract, signed } => {
                self.neon_push(true, n);
                self.neon_push(false, m);
                self.f.simd(extend(signed));
                self.f.simd(if subtract {
                    SUB[size + 1]
                } else {
                    ADD[size + 1]
                });
            }
            Long::Multiply { accumulate, signed } => {
                if accumulate.is_some() {
                    self.neon_push(true, d);
                }
                self.neon_push(false, n);
                self.neon_push(false, m);
                self.f.simd(if signed {
                    EXTMUL_LOW_S[size]
                } else {
                    EXTMUL_LOW_U[size]
                });
                match accumulate {
                    Some(false) => self.f.simd(ADD[size + 1]),
                    Some(true) => self.f.simd(SUB[size + 1]),
                    None => {}
                }
            }
        }
        self.neon_pop(true, d);
    }
}
