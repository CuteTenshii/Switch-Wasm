//! Maxwell (GM20B) shader instruction decoding.
//!
//! Bit layouts are ported from `envydis`'s `gm107.c` tables (envytools,
//! github.com/envytools/envytools) — opcode values and masks, operand bit
//! positions, and the modifier sub-tables, transcribed row by row. The
//! original subset was additionally verified against `uam`-compiled GLSL
//! fixtures disassembled with `envydis -m gm107`: a solid-color fragment
//! shader, an MVP-transform vertex shader, and a textured + vertex-color
//! fragment shader, and those captures are still the tests below.
//!
//! Three facts shape this decoder:
//!
//! - The rasterizer's fixed-function interpolator hands the fragment shader
//!   a linearly-interpolated `1/w` at a fixed attribute slot (`a[0x7c]` in
//!   every fixture). `ipa pass` reads it raw. The shader then computes `w =
//!   mufu rcp(1/w)` once and feeds it back into `ipa` (non-`pass`, the
//!   "perspective" mode) as the multiplier for every other varying, which the
//!   interpolator has already linearly interpolated pre-divided by `w`:
//!   `ipa.perspective(attr/w) * w == attr`. That's the whole perspective-
//!   correction idiom; there's no other division of labour to model.
//! - Every instruction carries a guard predicate: a 3-bit register at
//!   `[16, 19)` plus a negate flag at bit 19. Register 7 is `PT`, hardware's
//!   always-true placeholder, so `0b0111` with bit 19 clear is "unpredicated".
//!   Unlike the first version of this decoder, a real predicate is now
//!   decoded and carried rather than making the whole instruction
//!   unsupported — shaders with any control flow at all predicate constantly.
//! - Maxwell has no integer-multiply instruction in the usual sense. 32-bit
//!   multiplies come out as chains of `xmad`, which multiplies two 16-bit
//!   halves and accumulates.
//!
//! An encoding this decoder doesn't recognise — or recognises with a modifier
//! whose behaviour isn't modelled — becomes [`Op::Unimplemented`], which
//! carries the raw bits so a real capture stays inspectable rather than
//! silently mis-executing.

/// `ld`/`st`'s transfer size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemSize {
    U8,
    S8,
    U16,
    S16,
    B32,
    B64,
    B96,
    B128,
}

impl MemSize {
    /// How many 32-bit registers a transfer of this size covers.
    pub fn regs(self) -> u8 {
        match self {
            MemSize::B64 => 2,
            MemSize::B96 => 3,
            MemSize::B128 => 4,
            _ => 1,
        }
    }

    /// How many bytes it moves.
    pub fn bytes(self) -> u32 {
        match self {
            MemSize::U8 | MemSize::S8 => 1,
            MemSize::U16 | MemSize::S16 => 2,
            MemSize::B32 => 4,
            MemSize::B64 => 8,
            MemSize::B96 => 12,
            MemSize::B128 => 16,
        }
    }
}

/// The right-hand operand of an ALU op: a register, a slot in a bound
/// constant buffer (`cN[offset]`), or an inline immediate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(u8),
    Const { bank: u8, offset: u16 },
    Imm(u32),
}

/// A guard or source predicate. `reg` 7 is `PT`, hardware's always-true
/// register, so [`Pred::ALWAYS`] is the unpredicated case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pred {
    pub reg: u8,
    pub negate: bool,
}

impl Pred {
    pub const PT: u8 = 7;
    pub const ALWAYS: Pred = Pred { reg: Pred::PT, negate: false };
    pub const NEVER: Pred = Pred { reg: Pred::PT, negate: true };

    pub fn is_always(self) -> bool {
        self.reg == Pred::PT && !self.negate
    }
}

/// A float source's sign/magnitude modifiers, applied in that order:
/// `abs` first, then `neg`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FMod {
    pub neg: bool,
    pub abs: bool,
}

impl FMod {
    pub const NONE: FMod = FMod { neg: false, abs: false };

    pub fn apply(self, v: f32) -> f32 {
        let v = if self.abs { v.abs() } else { v };
        if self.neg {
            -v
        } else {
            v
        }
    }
}

/// A float comparison (`tab5bb0_0`). The `u` suffixes are the unordered
/// variants, which also compare true when either side is NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FCmp {
    Never,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    Num,
    Nan,
    LtU,
    EqU,
    LeU,
    GtU,
    NeU,
    GeU,
    Always,
}

/// An integer comparison (`tab5b60_0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ICmp {
    Never,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    Always,
}

/// How a `set`/`setp` combines its comparison with its source predicate
/// (`tab5bb0_1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
    Xor,
}

/// `lop`'s bitwise operation (`tab5c40_0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
    Xor,
    PassB,
}

/// `mufu`'s sub-operation (`tab5080_0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MufuOp {
    Cos,
    Sin,
    Ex2,
    Lg2,
    Rcp,
    Rsq,
    Sqrt,
}

/// A float rounding mode (`tab5cb0_1`/`tab5ca8_0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FRound {
    Nearest,
    Floor,
    Ceil,
    Trunc,
}

/// `texs`'s sample dimensionality (envydis's `d000_1`/`d200_1` tables).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexDim {
    T1d,
    T2d,
    T3d,
    TCube,
}

/// A decoded instruction: its guard predicate plus what it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction {
    pub pred: Pred,
    pub op: Op,
}

impl Instruction {
    /// An unpredicated instruction, which is what most of them are.
    pub fn always(op: Op) -> Instruction {
        Instruction { pred: Pred::ALWAYS, op }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    // ---- attribute space ----
    /// `ld.<size> dst, a[offset]` — attribute-space load.
    Ld { dst: u8, offset: u16, idx: u8, size: MemSize },
    /// `st.<size> a[offset], src` — attribute-space store.
    St { offset: u16, idx: u8, src: u8, size: MemSize },
    /// `ipa[.pass] dst, a[offset], mul` — fixed-function interpolation.
    /// `perspective = false` is `ipa pass`; `perspective = true` multiplies
    /// the fetched value by `mul` (`RZ` decodes to `None`).
    Ipa { dst: u8, offset: u16, mul: Option<u8>, perspective: bool, sat: bool },

    // ---- float ALU ----
    Fadd { dst: u8, a: u8, am: FMod, b: Operand, bm: FMod, ftz: bool, sat: bool },
    Fmul { dst: u8, a: u8, b: Operand, bm: FMod, ftz: bool, sat: bool },
    Ffma { dst: u8, a: u8, b: Operand, bneg: bool, c: Operand, cneg: bool, ftz: bool, sat: bool },
    Fmnmx { dst: u8, a: u8, am: FMod, b: Operand, bm: FMod, pred: Pred, ftz: bool },
    Fsetp { p0: u8, p1: u8, a: u8, am: FMod, b: Operand, bm: FMod, cmp: FCmp, bop: BoolOp, src: Pred },
    Fset { dst: u8, a: u8, am: FMod, b: Operand, bm: FMod, cmp: FCmp, bop: BoolOp, src: Pred, bf: bool },
    Mufu { dst: u8, src: u8, sm: FMod, op: MufuOp, sat: bool },

    // ---- integer ALU ----
    Iadd { dst: u8, a: u8, aneg: bool, b: Operand, bneg: bool },
    Iadd3 { dst: u8, a: u8, aneg: bool, b: Operand, bneg: bool, c: Operand, cneg: bool },
    Imnmx { dst: u8, a: u8, b: Operand, pred: Pred, signed: bool },
    Iscadd { dst: u8, a: u8, aneg: bool, b: Operand, bneg: bool, shift: u8 },
    Isetp { p0: u8, p1: u8, a: u8, b: Operand, cmp: ICmp, signed: bool, bop: BoolOp, src: Pred },
    Iset { dst: u8, a: u8, b: Operand, cmp: ICmp, signed: bool, bop: BoolOp, src: Pred, bf: bool },
    Icmp { dst: u8, a: u8, b: Operand, c: u8, cmp: ICmp, signed: bool },
    Imul { dst: u8, a: u8, b: Operand, signed: bool, hi: bool },
    /// `xmad dst, a.h[ah], b.h[bh], c` — the 16x16+32 multiply-accumulate
    /// Maxwell builds every wider integer multiply out of.
    Xmad {
        dst: u8,
        a: u8,
        ah: bool,
        asigned: bool,
        b: Operand,
        bh: bool,
        bsigned: bool,
        c: Operand,
        psl: bool,
        mrg: bool,
    },
    Lop { dst: u8, a: u8, ainv: bool, b: Operand, binv: bool, op: LogicOp },
    Lop3 { dst: u8, a: u8, b: Operand, c: Operand, lut: u8 },
    Shl { dst: u8, a: u8, b: Operand, wrap: bool },
    Shr { dst: u8, a: u8, b: Operand, signed: bool, wrap: bool },
    Shf { dst: u8, lo: u8, shift: Operand, hi: u8, left: bool, wrap: bool, hi_out: bool },
    Bfe { dst: u8, a: u8, b: Operand, signed: bool },
    Popc { dst: u8, b: Operand, inv: bool },
    Flo { dst: u8, b: Operand, signed: bool, shift: bool, inv: bool },
    Sel { dst: u8, a: u8, b: Operand, pred: Pred },

    // ---- conversions ----
    /// Integer -> float. `src_bytes`/`src_signed` describe the source's
    /// integer width; the destination is always f32 here.
    I2f { dst: u8, src: Operand, sm: FMod, src_bytes: u8, src_signed: bool, sel: u8 },
    /// Float -> integer, with an explicit rounding mode.
    F2i { dst: u8, src: Operand, sm: FMod, dst_bytes: u8, dst_signed: bool, round: FRound, ftz: bool },
    /// Float -> float: on f32 this is only ever a round/saturate.
    F2f { dst: u8, src: Operand, sm: FMod, round: FRound, sat: bool, ftz: bool },
    /// Integer -> integer: a width conversion, optionally saturating.
    I2i { dst: u8, src: Operand, sm: FMod, src_bytes: u8, src_signed: bool, dst_signed: bool, sat: bool, sel: u8 },

    // ---- moves ----
    Mov { dst: u8, src: Operand },
    Mov32i { dst: u8, imm: u32 },
    /// `mov dst, sN` — a special register (`tid`, `laneid`, ...).
    S2r { dst: u8, sr: u8 },
    Psetp { p0: u8, p1: u8, a: Pred, b: Pred, c: Pred, op1: BoolOp, op2: BoolOp },

    // ---- memory ----
    /// `ld cN[idx + offset]` — a constant-buffer load into registers.
    Ldc { dst: u8, bank: u8, offset: i32, idx: u8, size: MemSize },
    /// `ldg dst, [addr + offset]` — a global load.
    Ldg { dst: u8, addr: u8, offset: i32, size: MemSize },
    /// `stg [addr + offset], src` — a global store.
    Stg { addr: u8, offset: i32, src: u8, size: MemSize },
    /// `ld dst, l[addr + offset]` — a local (per-thread scratch) load.
    Ldl { dst: u8, addr: u8, offset: i32, size: MemSize },
    /// `st l[addr + offset], src`.
    Stl { addr: u8, offset: i32, src: u8, size: MemSize },

    // ---- texture ----
    /// `texs dst, coords.., handle, dim, mask` — texture sample with an
    /// immediate handle.
    Texs { dst: u8, dst2: u8, coords: [u8; 3], handle: u16, dim: TexDim, mask: [bool; 4] },

    // ---- control ----
    /// `bra target` — `target` is an instruction's byte offset within the
    /// program, already resolved from the pc-relative encoding.
    Bra { target: u32 },
    /// `ssy target` — push a reconvergence point.
    Ssy { target: u32 },
    /// `sync` — pop one and jump there.
    Sync,
    /// `pbk target` — push a loop-break point.
    Pbk { target: u32 },
    Brk,
    /// `pcnt target` — push a loop-continue point.
    Pcnt { target: u32 },
    Cont,
    Exit,
    /// `kil` — discard this fragment.
    Kil,
    Nop,
    /// A barrier/fence with no effect on a scalar interpreter, kept as a
    /// distinct op so it doesn't read as unsupported.
    Inert,

    /// A bit pattern this decoder doesn't recognise, or recognises but with
    /// an unhandled modifier. Carries the raw bits.
    Unimplemented { raw: u64 },
}

fn field(insn: u64, pos: u32, len: u32) -> u64 {
    (insn >> pos) & ((1u64 << len) - 1)
}

fn sfield(insn: u64, pos: u32, len: u32) -> i64 {
    let v = field(insn, pos, len);
    let sign = 1u64 << (len - 1);
    if v & sign != 0 {
        (v | !((1u64 << len) - 1)) as i64
    } else {
        v as i64
    }
}

fn reg(insn: u64, pos: u32, len: u32) -> u8 {
    field(insn, pos, len) as u8
}

/// `RZ`, the register that reads as zero and discards writes.
pub const RZ: u8 = 0xff;

fn opt_reg(r: u8) -> Option<u8> {
    if r == RZ {
        None
    } else {
        Some(r)
    }
}

/// The guard predicate every instruction carries: `PRED16` at `[16, 19)`
/// with its negate flag at bit 19.
fn guard(insn: u64) -> Pred {
    Pred { reg: reg(insn, 16, 3), negate: field(insn, 19, 1) != 0 }
}

/// A source predicate at `[pos, pos+3)` with its negate flag at `not`.
fn src_pred(insn: u64, pos: u32, not: u32) -> Pred {
    Pred { reg: reg(insn, pos, 3), negate: field(insn, not, 1) != 0 }
}

/// `C34_RZ_O14_20`: bank at `[34, 39)`, offset a signed 14-bit word index at
/// `[20, 34)` scaled by 4.
fn const_operand(insn: u64) -> Operand {
    Operand::Const {
        bank: reg(insn, 34, 5),
        offset: (sfield(insn, 20, 14) << 2) as u16,
    }
}

/// `S20_20`: 19 bits at 20 plus a sign bit at 56.
fn imm20(insn: u64) -> u32 {
    let v = field(insn, 20, 19) | (field(insn, 56, 1) << 19);
    // Sign-extend the 20-bit field to 32 bits.
    if v & (1 << 19) != 0 {
        (v | !0xf_ffff) as u32
    } else {
        v as u32
    }
}

/// `F20_20`: the same 20 bits, but they are the *top* 20 bits of an f32.
fn imm20f(insn: u64) -> u32 {
    ((field(insn, 20, 19) | (field(insn, 56, 1) << 19)) << 12) as u32
}

fn fcmp(bits: u64) -> FCmp {
    match bits {
        0 => FCmp::Never,
        1 => FCmp::Lt,
        2 => FCmp::Eq,
        3 => FCmp::Le,
        4 => FCmp::Gt,
        5 => FCmp::Ne,
        6 => FCmp::Ge,
        7 => FCmp::Num,
        8 => FCmp::Nan,
        9 => FCmp::LtU,
        10 => FCmp::EqU,
        11 => FCmp::LeU,
        12 => FCmp::GtU,
        13 => FCmp::NeU,
        14 => FCmp::GeU,
        _ => FCmp::Always,
    }
}

fn icmp(bits: u64) -> ICmp {
    match bits {
        0 => ICmp::Never,
        1 => ICmp::Lt,
        2 => ICmp::Eq,
        3 => ICmp::Le,
        4 => ICmp::Gt,
        5 => ICmp::Ne,
        6 => ICmp::Ge,
        _ => ICmp::Always,
    }
}

fn bool_op(bits: u64) -> Option<BoolOp> {
    match bits {
        0 => Some(BoolOp::And),
        1 => Some(BoolOp::Or),
        2 => Some(BoolOp::Xor),
        _ => None,
    }
}

/// `tab5cb8_1`/`tab5ce0_1`-style integer type: `(bytes, signed)`.
fn int_type(bits: u64) -> Option<(u8, bool)> {
    match bits {
        0 => Some((1, false)),
        1 => Some((2, false)),
        2 => Some((4, false)),
        4 => Some((1, true)),
        5 => Some((2, true)),
        6 => Some((4, true)),
        _ => None,
    }
}

fn fround(bits: u64) -> FRound {
    match bits {
        0 => FRound::Nearest,
        1 => FRound::Floor,
        2 => FRound::Ceil,
        _ => FRound::Trunc,
    }
}

/// `tab8000_0`/`tabeed0sz`-style transfer size.
fn mem_size(bits: u64) -> MemSize {
    match bits {
        0 => MemSize::U8,
        1 => MemSize::S8,
        2 => MemSize::U16,
        3 => MemSize::S16,
        4 => MemSize::B32,
        5 => MemSize::B64,
        6 => MemSize::B128,
        _ => MemSize::B32,
    }
}

/// `tabeff0_0`: `ld`/`st` in attribute space only carry the wide sizes.
fn attr_size(bits: u64) -> MemSize {
    match bits {
        0 => MemSize::B32,
        1 => MemSize::B64,
        2 => MemSize::B96,
        _ => MemSize::B128,
    }
}

/// Where a pc-relative branch lands: the 24-bit signed word at `[20, 44)`,
/// relative to the instruction *after* this one (envydis's `.addend = 8`).
fn branch_target(insn: u64, pc: u32) -> u32 {
    (pc as i64 + 8 + sfield(insn, 20, 24)) as u32
}

/// Decode a single 8-byte Maxwell instruction word sitting at byte offset
/// `pc` within its program (needed to resolve pc-relative branches). Never
/// panics: an unrecognised or unsupported bit pattern decodes to
/// [`Op::Unimplemented`].
pub fn decode_at(insn: u64, pc: u32) -> Instruction {
    Instruction { pred: guard(insn), op: decode_op(insn, pc) }
}

/// [`decode_at`] for a program with no branches, where the pc doesn't
/// matter.
pub fn decode(insn: u64) -> Instruction {
    decode_at(insn, 0)
}

fn decode_op(insn: u64, pc: u32) -> Op {
    let un = Op::Unimplemented { raw: insn };
    let top = |bits: u32| insn >> (64 - bits);

    // Matching runs longest-mask-first, the same order envydis's table is
    // written in, so a narrow opcode can't shadow a wide one.
    match top(16) & 0xfff8 {
        // ---- attribute space ----
        // ld a[] — gm107.c 0xefd8/0xfff8
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
        // st a[] — 0xeff0/0xfff8
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
        // ld c[] — 0xef90/0xfff8, bank at [36,41), signed 16-bit offset.
        0xef90 => Op::Ldc {
            dst: reg(insn, 0, 8),
            bank: reg(insn, 36, 5),
            offset: sfield(insn, 20, 16) as i32,
            idx: reg(insn, 8, 8),
            size: mem_size(field(insn, 48, 3)),
        },
        // ldg/stg — 0xeed0/0xeed8, signed 24-bit offset off REG_08.
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
        // ld/st l[] — 0xef40/0xef50.
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
        // mov dst, sN — 0xf0c8/0xfff8.
        0xf0c8 => Op::S2r { dst: reg(insn, 0, 8), sr: reg(insn, 20, 8) },
        // depbar/membar/bar: scheduling and memory ordering, both no-ops for
        // a scalar interpreter that runs one invocation to completion.
        0xf0f0 | 0xef98 | 0xf0a8 => Op::Inert,
        // sync — 0xf0f8/0xfff8.
        0xf0f8 => Op::Sync,
        _ => match top(12) {
            // ---- control flow (0xfff0 masks) ----
            0xe35 => Op::Cont,
            0xe34 => Op::Brk,
            0xe33 => Op::Kil,
            0xe30 => Op::Exit,
            0xe2b if field(insn, 5, 1) == 0 => Op::Pcnt { target: branch_target(insn, pc) },
            0xe2a if field(insn, 5, 1) == 0 => Op::Pbk { target: branch_target(insn, pc) },
            0xe29 if field(insn, 5, 1) == 0 => Op::Ssy { target: branch_target(insn, pc) },
            0xe24 if field(insn, 5, 1) == 0 => Op::Bra { target: branch_target(insn, pc) },
            _ => decode_alu(insn),
        },
    }
}

fn decode_alu(insn: u64) -> Op {
    let un = Op::Unimplemented { raw: insn };

    // The three operand forms of a "normal" ALU op share a sub-opcode: the
    // top byte selects register (0x5c..), constant (0x4c..) or immediate
    // (0x38.., masked 0xfef8 because bit 56 belongs to the immediate) and
    // the rest of the opcode is identical. Resolving the form once lets each
    // op below be written once — but the form byte has to be checked too,
    // because a *different* opcode group reuses the same low byte (0x49a0 is
    // `ffma`, 0x4ca0 is `sel`), so anything outside this group goes to
    // [`decode_alu_wide`].
    let form = insn >> 48;
    let (rhs_int, rhs_float) = match form >> 8 {
        0x5c => (Operand::Reg(reg(insn, 20, 8)), Operand::Reg(reg(insn, 20, 8))),
        0x4c => (const_operand(insn), const_operand(insn)),
        0x38 | 0x39 => (Operand::Imm(imm20(insn)), Operand::Imm(imm20f(insn))),
        _ => return decode_alu_wide(insn),
    };
    let rhs_int = Some(rhs_int);
    let rhs_float = Some(rhs_float);
    // The low three bits of the opcode field are modifier bits (the group's
    // mask is 0xfff8), so they are masked off before dispatching.
    let sub = form & 0x00f8;

    match sub {
        // ---- float ----
        // fadd — ftz 44, sat 50, a: neg 48/abs 46, b: neg 45/abs 49.
        0x58 => {
            let Some(b) = rhs_float else { return un };
            Op::Fadd {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                am: FMod { neg: field(insn, 48, 1) != 0, abs: field(insn, 46, 1) != 0 },
                b,
                bm: FMod { neg: field(insn, 45, 1) != 0, abs: field(insn, 49, 1) != 0 },
                ftz: field(insn, 44, 1) != 0,
                sat: field(insn, 50, 1) != 0,
            }
        }
        // fmul — ftz/fmz at 44..46, scale at 41..44, sat 50, b: neg 48.
        0x68 => {
            let Some(b) = rhs_float else { return un };
            if field(insn, 41, 3) != 0 {
                return un; // the d2/d4/d8/m2/m4/m8 pre-scales
            }
            Op::Fmul {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                bm: FMod { neg: field(insn, 48, 1) != 0, abs: false },
                ftz: field(insn, 44, 2) == 1,
                sat: field(insn, 50, 1) != 0,
            }
        }
        // fmnmx — ftz 44, a: neg 48/abs 46, b: neg 45/abs 49, pred at 39.
        0x60 => {
            let Some(b) = rhs_float else { return un };
            Op::Fmnmx {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                am: FMod { neg: field(insn, 48, 1) != 0, abs: field(insn, 46, 1) != 0 },
                b,
                bm: FMod { neg: field(insn, 45, 1) != 0, abs: field(insn, 49, 1) != 0 },
                pred: src_pred(insn, 39, 42),
                ftz: field(insn, 44, 1) != 0,
            }
        }
        // ---- integer ----
        // iadd — sat 50, x 43, a: neg 49, b: neg 48.
        0x10 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 50, 1) != 0 || field(insn, 43, 1) != 0 {
                return un; // saturating / extended-carry add
            }
            Op::Iadd {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                aneg: field(insn, 49, 1) != 0,
                b,
                bneg: field(insn, 48, 1) != 0,
            }
        }
        // iscadd — shift at 39..44.
        0x18 => {
            let Some(b) = rhs_int else { return un };
            Op::Iscadd {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                aneg: field(insn, 49, 1) != 0,
                b,
                bneg: field(insn, 48, 1) != 0,
                shift: field(insn, 39, 5) as u8,
            }
        }
        // imnmx — signed 48, pred at 39.
        0x20 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 43, 2) != 0 {
                return un; // the xlo/xmed/xhi extended forms
            }
            Op::Imnmx {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                pred: src_pred(insn, 39, 42),
                signed: field(insn, 48, 1) != 0,
            }
        }
        // shr — signed 48, wrap 39, brev 40, x 44.
        0x28 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 40, 1) != 0 || field(insn, 44, 1) != 0 {
                return un; // bit-reverse / extended
            }
            Op::Shr {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                signed: field(insn, 48, 1) != 0,
                wrap: field(insn, 39, 1) != 0,
            }
        }
        // flo — signed 48, shift 41, inv 40.
        0x30 => {
            let Some(b) = rhs_int else { return un };
            Op::Flo {
                dst: reg(insn, 0, 8),
                b,
                signed: field(insn, 48, 1) != 0,
                shift: field(insn, 41, 1) != 0,
                inv: field(insn, 40, 1) != 0,
            }
        }
        // imul — hi 39, signedness at 41 (a) and 40 (b) in tab5c38_0/1.
        0x38 => {
            let Some(b) = rhs_int else { return un };
            Op::Imul {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                signed: field(insn, 41, 1) != 0,
                hi: field(insn, 39, 1) != 0,
            }
        }
        // lop — op at 41..43, inv 39 (a) / 40 (b), x 43.
        0x40 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 43, 1) != 0 || field(insn, 44, 2) != 0 {
                return un; // extended-carry / predicate-writing forms
            }
            let op = match field(insn, 41, 2) {
                0 => LogicOp::And,
                1 => LogicOp::Or,
                2 => LogicOp::Xor,
                _ => LogicOp::PassB,
            };
            Op::Lop {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                ainv: field(insn, 39, 1) != 0,
                b,
                binv: field(insn, 40, 1) != 0,
                op,
            }
        }
        // shl — wrap 39, x 43.
        0x48 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 43, 1) != 0 {
                return un;
            }
            Op::Shl {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                wrap: field(insn, 39, 1) != 0,
            }
        }
        // bfe — signed 48, brev 40.
        0x00 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 40, 1) != 0 {
                return un;
            }
            Op::Bfe {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                signed: field(insn, 48, 1) != 0,
            }
        }
        // popc — inv 40.
        0x08 => {
            let Some(b) = rhs_int else { return un };
            Op::Popc { dst: reg(insn, 0, 8), b, inv: field(insn, 40, 1) != 0 }
        }
        // ---- moves and selects ----
        // mov — the 4-bit byte-enable mask at 39..43 must be "all".
        0x98 => {
            let Some(src) = rhs_int else { return un };
            if field(insn, 39, 4) != 0xf {
                return un;
            }
            Op::Mov { dst: reg(insn, 0, 8), src }
        }
        // sel — pred at 39.
        0xa0 => {
            let Some(b) = rhs_int else { return un };
            Op::Sel {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                pred: src_pred(insn, 39, 42),
            }
        }
        // ---- conversions ----
        // i2f — dst type 8..10 (must be f32), src type in tab5cb8_1, byte
        // select at 41..43, src: neg 45/abs 49.
        0xb8 => {
            let Some(src) = rhs_int else { return un };
            if field(insn, 8, 2) != 2 {
                return un; // only f32 destinations
            }
            let bits = field(insn, 10, 2) | (field(insn, 13, 1) << 2);
            let Some((src_bytes, src_signed)) = int_type(bits) else { return un };
            Op::I2f {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod { neg: field(insn, 45, 1) != 0, abs: field(insn, 49, 1) != 0 },
                src_bytes,
                src_signed,
                sel: field(insn, 41, 2) as u8,
            }
        }
        // f2i — dst type in tab5cb0_2, src type 10..12 (must be f32),
        // rounding at 39..41.
        0xb0 => {
            let Some(src) = rhs_float else { return un };
            if field(insn, 10, 2) != 2 {
                return un; // only f32 sources
            }
            let bits = field(insn, 8, 2) | (field(insn, 12, 1) << 2);
            let Some((dst_bytes, dst_signed)) = int_type(bits) else { return un };
            Op::F2i {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod { neg: field(insn, 45, 1) != 0, abs: field(insn, 49, 1) != 0 },
                dst_bytes,
                dst_signed,
                round: fround(field(insn, 39, 2)),
                ftz: field(insn, 44, 1) != 0,
            }
        }
        // f2f — both types must be f32; rounding at 39..41 plus bit 42.
        0xa8 => {
            let Some(src) = rhs_float else { return un };
            if field(insn, 8, 2) != 2 || field(insn, 10, 2) != 2 {
                return un;
            }
            if field(insn, 42, 1) == 0 && field(insn, 39, 2) != 0 {
                return un; // "pass" without an explicit rounding mode
            }
            Op::F2f {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod { neg: field(insn, 45, 1) != 0, abs: field(insn, 49, 1) != 0 },
                round: fround(field(insn, 39, 2)),
                sat: field(insn, 50, 1) != 0,
                ftz: field(insn, 44, 1) != 0,
            }
        }
        // i2i — src type in tab5ce0_1, dst type in tab5ce0_0.
        0xe0 => {
            let Some(src) = rhs_int else { return un };
            let sbits = field(insn, 10, 2) | (field(insn, 13, 1) << 2);
            let dbits = field(insn, 8, 2) | (field(insn, 12, 1) << 2);
            let (Some((src_bytes, src_signed)), Some((_, dst_signed))) =
                (int_type(sbits), int_type(dbits))
            else {
                return un;
            };
            Op::I2i {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod { neg: field(insn, 45, 1) != 0, abs: field(insn, 49, 1) != 0 },
                src_bytes,
                src_signed,
                dst_signed,
                sat: field(insn, 50, 1) != 0,
                sel: field(insn, 41, 2) as u8,
            }
        }
        _ => decode_alu_wide(insn),
    }
}

/// The ops whose opcode field is wider or narrower than the 0xfff8 group
/// [`decode_alu`] handles.
fn decode_alu_wide(insn: u64) -> Op {
    let un = Op::Unimplemented { raw: insn };
    let form = insn >> 48;

    // ---- 0xfff0-masked: fsetp/isetp/iset/icmp/prmt/lop3/bfi ----
    match form >> 4 {
        // fsetp — cmp 48..52, ftz 47, bop 45..47.
        0x5bb | 0x4bb | 0x36b => {
            let b = match form >> 12 {
                0x5 => Operand::Reg(reg(insn, 20, 8)),
                0x4 => const_operand(insn),
                _ => Operand::Imm(imm20f(insn)),
            };
            let Some(bop) = bool_op(field(insn, 45, 2)) else { return un };
            return Op::Fsetp {
                p0: reg(insn, 3, 3),
                p1: reg(insn, 0, 3),
                a: reg(insn, 8, 8),
                am: FMod { neg: field(insn, 43, 1) != 0, abs: field(insn, 7, 1) != 0 },
                b,
                bm: FMod { neg: field(insn, 6, 1) != 0, abs: field(insn, 44, 1) != 0 },
                cmp: fcmp(field(insn, 48, 4)),
                bop,
                src: src_pred(insn, 39, 42),
            };
        }
        // isetp — cmp 49..52, signed 48, bop 45..47, x 43.
        0x5b6 | 0x4b6 | 0x366 => {
            let b = match form >> 12 {
                0x5 => Operand::Reg(reg(insn, 20, 8)),
                0x4 => const_operand(insn),
                _ => Operand::Imm(imm20(insn)),
            };
            let Some(bop) = bool_op(field(insn, 45, 2)) else { return un };
            if field(insn, 43, 1) != 0 {
                return un; // extended-carry compare
            }
            return Op::Isetp {
                p0: reg(insn, 3, 3),
                p1: reg(insn, 0, 3),
                a: reg(insn, 8, 8),
                b,
                cmp: icmp(field(insn, 49, 3)),
                signed: field(insn, 48, 1) != 0,
                bop,
                src: src_pred(insn, 39, 42),
            };
        }
        // iset — the register-writing form of isetp.
        0x5b5 | 0x4b5 | 0x365 => {
            let b = match form >> 12 {
                0x5 => Operand::Reg(reg(insn, 20, 8)),
                0x4 => const_operand(insn),
                _ => Operand::Imm(imm20(insn)),
            };
            let Some(bop) = bool_op(field(insn, 45, 2)) else { return un };
            return Op::Iset {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                cmp: icmp(field(insn, 49, 3)),
                signed: field(insn, 48, 1) != 0,
                bop,
                src: src_pred(insn, 39, 42),
                bf: field(insn, 44, 1) != 0,
            };
        }
        // icmp — c is the third source; the operand order differs between
        // the register and constant forms.
        0x5b4 | 0x4b4 | 0x534 => {
            let (b, c) = match form >> 12 {
                0x5 if form >> 4 == 0x5b4 => (Operand::Reg(reg(insn, 20, 8)), reg(insn, 39, 8)),
                0x5 => (const_operand(insn), reg(insn, 39, 8)),
                _ => (const_operand(insn), reg(insn, 39, 8)),
            };
            return Op::Icmp {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                c,
                cmp: icmp(field(insn, 49, 3)),
                signed: field(insn, 48, 1) != 0,
            };
        }
        // iadd3 — three-way add, negation per source.
        0x5cc | 0x4cc | 0x38c => {
            let (b, c) = match form >> 12 {
                0x5 => (Operand::Reg(reg(insn, 20, 8)), Operand::Reg(reg(insn, 39, 8))),
                0x4 => (const_operand(insn), Operand::Reg(reg(insn, 39, 8))),
                _ => (Operand::Imm(imm20(insn)), Operand::Reg(reg(insn, 39, 8))),
            };
            if field(insn, 48, 1) != 0 {
                return un; // extended-carry
            }
            return Op::Iadd3 {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                aneg: field(insn, 51, 1) != 0,
                b,
                bneg: field(insn, 50, 1) != 0,
                c,
                cneg: field(insn, 49, 1) != 0,
            };
        }
        // psetp — a pure predicate op.
        _ => {}
    }

    if insn & 0xfff8_0000_0000_0000 == 0x5090_0000_0000_0000 {
        let (Some(op1), Some(op2)) = (bool_op(field(insn, 24, 2)), bool_op(field(insn, 45, 2)))
        else {
            return un;
        };
        return Op::Psetp {
            p0: reg(insn, 3, 3),
            p1: reg(insn, 0, 3),
            a: src_pred(insn, 12, 15),
            b: src_pred(insn, 29, 32),
            c: src_pred(insn, 39, 42),
            op1,
            op2,
        };
    }

    // nop — 0x50b0/0xfff8.
    if insn & 0xfff8_0000_0000_0000 == 0x50b0_0000_0000_0000 {
        return Op::Nop;
    }

    // mufu — subop at 20..24, sat 50, src: neg 48 / abs 46.
    if insn & 0xfff8_0000_0000_0000 == 0x5080_0000_0000_0000 {
        let mufu = match field(insn, 20, 4) {
            0 => MufuOp::Cos,
            1 => MufuOp::Sin,
            2 => MufuOp::Ex2,
            3 => MufuOp::Lg2,
            4 => MufuOp::Rcp,
            5 => MufuOp::Rsq,
            8 => MufuOp::Sqrt,
            _ => return un,
        };
        return Op::Mufu {
            dst: reg(insn, 0, 8),
            src: reg(insn, 8, 8),
            sm: FMod { neg: field(insn, 48, 1) != 0, abs: field(insn, 46, 1) != 0 },
            op: mufu,
            sat: field(insn, 50, 1) != 0,
        };
    }

    // lop3 — the LUT byte sits in a different field in each form.
    if insn & 0xfff8_0000_0000_0000 == 0x5be0_0000_0000_0000 {
        if field(insn, 38, 1) != 0 || field(insn, 36, 2) != 0 {
            return un;
        }
        return Op::Lop3 {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b: Operand::Reg(reg(insn, 20, 8)),
            c: Operand::Reg(reg(insn, 39, 8)),
            lut: field(insn, 28, 8) as u8,
        };
    }
    if insn & 0xfc00_0000_0000_0000 == 0x3c00_0000_0000_0000 {
        return Op::Lop3 {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b: Operand::Imm(imm20(insn)),
            c: Operand::Reg(RZ),
            lut: field(insn, 48, 8) as u8,
        };
    }

    // ffma — three operand orders across four opcodes.
    if insn & 0xff80_0000_0000_0000 == 0x5980_0000_0000_0000 {
        return decode_ffma(insn, Operand::Reg(reg(insn, 20, 8)), Operand::Reg(reg(insn, 39, 8)));
    }
    if insn & 0xff80_0000_0000_0000 == 0x4980_0000_0000_0000 {
        return decode_ffma(insn, const_operand(insn), Operand::Reg(reg(insn, 39, 8)));
    }
    if insn & 0xff80_0000_0000_0000 == 0x5180_0000_0000_0000 {
        // The register/constant operands are the other way round here.
        return decode_ffma(insn, Operand::Reg(reg(insn, 39, 8)), const_operand(insn));
    }
    if insn & 0xfe80_0000_0000_0000 == 0x3280_0000_0000_0000 {
        return decode_ffma(insn, Operand::Imm(imm20f(insn)), Operand::Reg(reg(insn, 39, 8)));
    }

    // xmad — 16x16 multiply-accumulate. Only the plain modes are decoded:
    // the `.x` extended-carry and the psl/mrg merge forms change what the
    // accumulate does, so they stay unimplemented rather than approximated.
    if insn & 0xffc0_0000_0000_0000 == 0x5b00_0000_0000_0000 {
        return decode_xmad(
            insn,
            Operand::Reg(reg(insn, 20, 8)),
            Operand::Reg(reg(insn, 39, 8)),
            field(insn, 35, 1) != 0,
            field(insn, 50, 3),
            field(insn, 38, 1) != 0,
            field(insn, 36, 1) != 0,
            field(insn, 37, 1) != 0,
        );
    }
    if insn & 0xff80_0000_0000_0000 == 0x5100_0000_0000_0000 {
        return decode_xmad(
            insn,
            const_operand(insn),
            Operand::Reg(reg(insn, 39, 8)),
            field(insn, 52, 1) != 0,
            field(insn, 50, 2),
            field(insn, 54, 1) != 0,
            false,
            false,
        );
    }
    // xmad, immediate form: the register form with a **15-bit** immediate at
    // 20..35 in place of the b register, and every modifier left exactly
    // where the register form keeps it.
    //
    // The width is what makes this form readable. A 32-bit multiply by a
    // constant lowers to a pair — `xmad d, a, K, RZ` then
    // `xmad.psl d, a.h1, K, c` — and both halves multiply by the *same* K.
    // Reading the immediate as the usual 20 bits makes the second one
    // multiply by `K | 0x10000` instead, because bit 36 is `psl` and not part
    // of the immediate at all. Fifteen bits is the width at which the pair
    // agrees, and it leaves `bh` at 35 where the register form has it.
    //
    // Without this arm 59 of "A Short Hike"'s 297 draws were dropped by the
    // rasterizer and its frames came out black.
    if insn & 0xffc0_0000_0000_0000 == 0x3600_0000_0000_0000 {
        return decode_xmad(
            insn,
            Operand::Imm(field(insn, 20, 15) as u32),
            Operand::Reg(reg(insn, 39, 8)),
            field(insn, 35, 1) != 0,
            field(insn, 50, 3),
            field(insn, 38, 1) != 0,
            field(insn, 36, 1) != 0,
            field(insn, 37, 1) != 0,
        );
    }

    // fset — the register-writing form of fsetp.
    if insn & 0xff00_0000_0000_0000 == 0x5800_0000_0000_0000
        || insn & 0xfe00_0000_0000_0000 == 0x4800_0000_0000_0000
    {
        let b = if insn >> 56 == 0x58 {
            Operand::Reg(reg(insn, 20, 8))
        } else {
            const_operand(insn)
        };
        let Some(bop) = bool_op(field(insn, 45, 2)) else {
            return un;
        };
        return Op::Fset {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            am: FMod { neg: field(insn, 43, 1) != 0, abs: field(insn, 54, 1) != 0 },
            b,
            bm: FMod { neg: field(insn, 53, 1) != 0, abs: field(insn, 44, 1) != 0 },
            cmp: fcmp(field(insn, 48, 4)),
            bop,
            src: src_pred(insn, 39, 42),
            bf: field(insn, 52, 1) != 0,
        };
    }

    // ipa — a[]-relative, non-indexed.
    if insn & 0xff00_0040_0000_ff00 == 0xe000_0000_0000_ff00 {
        let mode = field(insn, 54, 2);
        if field(insn, 52, 2) != 0 || mode > 1 {
            return un; // centroid/offset sampling, and the sc/constant modes
        }
        return Op::Ipa {
            dst: reg(insn, 0, 8),
            offset: field(insn, 28, 10) as u16,
            mul: opt_reg(reg(insn, 20, 8)),
            perspective: mode == 1,
            sat: field(insn, 51, 1) != 0,
        };
    }

    // texs — the immediate-handle sample.
    if insn & 0xf600_0000_0000_0000 == 0xd000_0000_0000_0000 {
        if field(insn, 49, 1) == 0 {
            if let (Some(dim), Some(mask)) =
                (decode_tex_dim(field(insn, 53, 4)), decode_tex_mask(field(insn, 50, 3)))
            {
                return Op::Texs {
                    dst: reg(insn, 0, 8),
                    dst2: reg(insn, 28, 8),
                    coords: [reg(insn, 28, 8), reg(insn, 8, 8), reg(insn, 20, 8)],
                    handle: field(insn, 36, 13) as u16,
                    dim,
                    mask,
                };
            }
        }
        return un;
    }

    // The 32-bit-immediate forms.
    if insn & 0xfff0_0000_0000_0000 == 0x0100_0000_0000_0000 {
        return Op::Mov32i { dst: reg(insn, 0, 8), imm: field(insn, 20, 32) as u32 };
    }
    if insn & 0xfc00_0000_0000_0000 == 0x0800_0000_0000_0000 {
        // fadd32i
        return Op::Fadd {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            am: FMod { neg: field(insn, 56, 1) != 0, abs: field(insn, 54, 1) != 0 },
            b: Operand::Imm(field(insn, 20, 32) as u32),
            bm: FMod::NONE,
            ftz: field(insn, 55, 1) != 0,
            sat: false,
        };
    }
    if insn & 0xff00_0000_0000_0000 == 0x1e00_0000_0000_0000 {
        // fmul32i
        return Op::Fmul {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b: Operand::Imm(field(insn, 20, 32) as u32),
            bm: FMod::NONE,
            ftz: field(insn, 55, 1) != 0,
            sat: field(insn, 54, 1) != 0,
        };
    }
    if insn & 0xfe80_0000_0000_0000 == 0x1c00_0000_0000_0000 {
        // iadd32i
        return Op::Iadd {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            aneg: field(insn, 56, 1) != 0,
            b: Operand::Imm(field(insn, 20, 32) as u32),
            bneg: false,
        };
    }
    if insn & 0xfc00_0000_0000_0000 == 0x0400_0000_0000_0000 {
        // lop32i
        let op = match field(insn, 53, 2) {
            0 => LogicOp::And,
            1 => LogicOp::Or,
            2 => LogicOp::Xor,
            _ => LogicOp::PassB,
        };
        return Op::Lop {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            ainv: field(insn, 55, 1) != 0,
            b: Operand::Imm(field(insn, 20, 32) as u32),
            binv: field(insn, 56, 1) != 0,
            op,
        };
    }

    un
}

fn decode_ffma(insn: u64, b: Operand, c: Operand) -> Op {
    if field(insn, 51, 2) != 0 {
        return Op::Unimplemented { raw: insn }; // explicit rounding modes
    }
    Op::Ffma {
        dst: reg(insn, 0, 8),
        a: reg(insn, 8, 8),
        b,
        bneg: field(insn, 48, 1) != 0,
        c,
        cneg: field(insn, 49, 1) != 0,
        ftz: field(insn, 53, 2) == 1,
        sat: field(insn, 50, 1) != 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_xmad(
    insn: u64,
    b: Operand,
    c: Operand,
    bh: bool,
    mode: u64,
    x: bool,
    psl: bool,
    mrg: bool,
) -> Op {
    if x || mode != 0 {
        return Op::Unimplemented { raw: insn };
    }
    let sign = field(insn, 48, 2);
    Op::Xmad {
        dst: reg(insn, 0, 8),
        a: reg(insn, 8, 8),
        ah: field(insn, 53, 1) != 0,
        asigned: sign == 1 || sign == 3,
        b,
        bh,
        bsigned: sign == 2 || sign == 3,
        c,
        psl,
        mrg,
    }
}

/// `d000_1`/`d200_1`-shared 4-bit field.
fn decode_tex_dim(bits: u64) -> Option<TexDim> {
    match bits {
        0 => Some(TexDim::T1d),
        1 => Some(TexDim::T2d),
        4 => Some(TexDim::T3d),
        6 => Some(TexDim::TCube),
        _ => None,
    }
}

/// `d200_2`'s multi-channel field (the `rgb`/`rga`/`rba`/`gba`/`rgba` rows).
fn decode_tex_mask(bits: u64) -> Option<[bool; 4]> {
    const R: [bool; 4] = [true, false, false, false];
    const G: [bool; 4] = [false, true, false, false];
    const B: [bool; 4] = [false, false, true, false];
    const A: [bool; 4] = [false, false, false, true];
    let or = |a: [bool; 4], b: [bool; 4]| [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]];
    match bits {
        0 => Some(or(or(R, G), B)),        // rgb
        1 => Some(or(or(R, G), A)),        // rga
        2 => Some(or(or(R, B), A)),        // rba
        3 => Some(or(or(G, B), A)),        // gba
        4 => Some(or(or(or(R, G), B), A)), // rgba
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every raw word in the first group below was captured with `envydis -n
    // -i -m gm107` against `uam`-compiled fixtures or a live JKSV run, and
    // cross-checked field by field; see the module docs for provenance. The
    // words in the second group are assembled here from the same envytools
    // table this decoder is written against, so they check the field
    // positions rather than an independent capture.

    fn op(word: u64) -> Op {
        decode(word).op
    }

    #[test]
    fn decodes_ipa_pass_then_mufu_rcp() {
        // solid.frag: "ipa pass $r0 a[0x7c] 0x0 0x0 0x1"
        assert_eq!(
            op(0xe003ff87cff7ff00),
            Op::Ipa { dst: 0, offset: 0x7c, mul: None, perspective: false, sat: false }
        );
        // "mufu rcp $r3 $r0"
        assert_eq!(
            op(0x5080000000470003),
            Op::Mufu { dst: 3, src: 0, sm: FMod::NONE, op: MufuOp::Rcp, sat: false }
        );
        // "ipa $r0 a[0x80] $r3 0x0 0x1"
        assert_eq!(
            op(0xe043ff880037ff00),
            Op::Ipa { dst: 0, offset: 0x80, mul: Some(3), perspective: true, sat: false }
        );
    }

    #[test]
    fn decodes_exit() {
        assert_eq!(op(0xe3000000_0007000f), Op::Exit);
    }

    #[test]
    fn decodes_ld_st_b128_attribute_space() {
        // mvp.vert: "ld b128 $r0 a[0x80] 0x0"
        assert_eq!(
            op(0xefd9ff80_0807ff00),
            Op::Ld { dst: 0, offset: 0x80, idx: RZ, size: MemSize::B128 }
        );
        // "st b128 a[0x70] $r0 0x0"
        assert_eq!(
            op(0xeff1ff80_0707ff00),
            Op::St { offset: 0x70, idx: RZ, src: 0, size: MemSize::B128 }
        );
    }

    #[test]
    fn decodes_fmul_constant_bank_and_register_forms() {
        // mvp.vert: "fmul ftz $r4 $r0 c2[0x0]"
        assert_eq!(
            op(0x4c681008_00070004),
            Op::Fmul {
                dst: 4,
                a: 0,
                b: Operand::Const { bank: 2, offset: 0x0 },
                bm: FMod::NONE,
                ftz: true,
                sat: false,
            }
        );
        // mvp.vert: "fmul ftz $r5 $r0 c2[0x4]"
        assert_eq!(
            op(0x4c681008_00170005),
            Op::Fmul {
                dst: 5,
                a: 0,
                b: Operand::Const { bank: 2, offset: 0x4 },
                bm: FMod::NONE,
                ftz: true,
                sat: false,
            }
        );
        // tex.frag: "fmul ftz $r0 $r0 $r5"
        assert_eq!(
            op(0x5c681000_00570000),
            Op::Fmul {
                dst: 0,
                a: 0,
                b: Operand::Reg(5),
                bm: FMod::NONE,
                ftz: true,
                sat: false,
            }
        );
    }

    #[test]
    fn decodes_fadd_constant_bank_form() {
        // Captured from a live JKSV run (real Mesa/nouveau nvc0-compiled
        // code, not a `uam` fixture): "fadd ftz $r4 $r2 c0[0x30]".
        assert_eq!(
            op(0x4c58100000c70204),
            Op::Fadd {
                dst: 4,
                a: 2,
                am: FMod::NONE,
                b: Operand::Const { bank: 0, offset: 0x30 },
                bm: FMod::NONE,
                ftz: true,
                sat: false,
            }
        );
    }

    #[test]
    fn decodes_mov32i() {
        // Captured from a live JKSV run: "mov32i $r0 0x3f800000" (loads the
        // float bit pattern for 1.0).
        assert_eq!(op(0x0103f8000007f000), Op::Mov32i { dst: 0, imm: 0x3f800000 });
    }

    #[test]
    fn decodes_ffma_constant_bank_chain() {
        // mvp.vert: "ffma ftz $r4 $r1 c2[0x10] $r4"
        assert_eq!(
            op(0x49a00208_00470104),
            Op::Ffma {
                dst: 4,
                a: 1,
                b: Operand::Const { bank: 2, offset: 0x10 },
                bneg: false,
                c: Operand::Reg(4),
                cneg: false,
                ftz: true,
                sat: false,
            }
        );
        // "ffma ftz $r0 $r3 c2[0x30] $r1"
        assert_eq!(
            op(0x49a00088_00c70300),
            Op::Ffma {
                dst: 0,
                a: 3,
                b: Operand::Const { bank: 2, offset: 0x30 },
                bneg: false,
                c: Operand::Reg(1),
                cneg: false,
                ftz: true,
                sat: false,
            }
        );
    }

    #[test]
    fn decodes_texs() {
        // tex.frag: envydis prints "texs $r2 $r0 $r0 $r1 0x1a4 t2d rgba", but
        // envydis's print order doesn't match this ISA's real dst/coord
        // roles — confirmed empirically (see `interp`'s module docs) by
        // running the decoded program against known texture/colour inputs
        // and checking the output against `texture.rgba * vColor.rgba`:
        // the real destination is REG_00 (here 0, i.e. $r0, not the
        // first-printed $r2), and REG_28 (here 2) is an unused coordinate
        // slot for a plain 2D sample.
        assert_eq!(
            op(0xd8301a40_20170000),
            Op::Texs {
                dst: 0,
                dst2: 2,
                coords: [2, 0, 1],
                handle: 0x1a4,
                dim: TexDim::T2d,
                mask: [true, true, true, true],
            }
        );
    }

    #[test]
    fn unrecognised_bits_are_unimplemented_not_a_panic() {
        assert_eq!(op(0), Op::Unimplemented { raw: 0 });
        assert_eq!(op(u64::MAX), Op::Unimplemented { raw: u64::MAX });
    }

    // ---- the wider instruction set ----

    /// Assemble one instruction: an opcode's top 16 bits, the always-true
    /// guard predicate, and whatever operand fields the caller sets.
    /// The 32-bit-multiply-by-a-constant pair, as "A Short Hike" emits it.
    /// Both halves must come out multiplying by the *same* constant: reading
    /// the immediate as the usual 20 bits makes the second one multiply by
    /// `K | 0x10000`, because bit 36 is `psl` and not part of the immediate.
    #[test]
    fn xmad_immediate_keeps_its_modifiers_where_the_register_form_does() {
        // xmad R1, R2, 0x7, RZ
        let lo = asm(0x3600, &[(0, 8, 1), (8, 8, 2), (20, 15, 7), (39, 8, 255)]);
        match op(lo) {
            Op::Xmad { dst, a, ah, b, c, psl, mrg, .. } => {
                assert_eq!((dst, a), (1, 2));
                assert_eq!(b, Operand::Imm(7));
                assert_eq!(c, Operand::Reg(255));
                assert!(!ah && !psl && !mrg);
            }
            other => panic!("expected xmad, got {other:?}"),
        }
        // xmad.psl R1, R2.h1, 0x7, R0 — the same constant, one bit up.
        let hi = asm(
            0x3600,
            &[(0, 8, 1), (8, 8, 2), (20, 15, 7), (36, 1, 1), (39, 8, 0), (53, 1, 1)],
        );
        match op(hi) {
            Op::Xmad { b, c, ah, psl, .. } => {
                assert_eq!(b, Operand::Imm(7), "the immediate absorbed the psl bit");
                assert_eq!(c, Operand::Reg(0));
                assert!(ah, "a.h1 not decoded");
                assert!(psl, "psl not decoded");
            }
            other => panic!("expected xmad.psl, got {other:?}"),
        }
    }

    fn asm(opcode: u16, fields: &[(u32, u32, u64)]) -> u64 {
        let mut w = (opcode as u64) << 48;
        w |= 0x7 << 16; // PT, not negated
        for &(pos, len, value) in fields {
            w |= (value & ((1u64 << len) - 1)) << pos;
        }
        w
    }

    #[test]
    fn a_guard_predicate_is_decoded_rather_than_rejected() {
        // The same `exit`, guarded by `!p1`. The first version of this
        // decoder made any predicated instruction unsupported, which is
        // every instruction in a shader with control flow.
        let raw = 0xe3000000_0007000f & !(0xf << 16) | (1 << 16) | (1 << 19);
        let insn = decode(raw);
        assert_eq!(insn.op, Op::Exit);
        assert_eq!(insn.pred, Pred { reg: 1, negate: true });
        assert!(!insn.pred.is_always());
        assert!(decode(0xe3000000_0007000f).pred.is_always());
    }

    #[test]
    fn decodes_source_modifiers_on_fadd() {
        // fadd $r0, -|$r1|, $r2 — neg 48 / abs 46 on a, both clear on b.
        let raw = asm(0x5c58, &[(0, 8, 0), (8, 8, 1), (20, 8, 2), (48, 1, 1), (46, 1, 1)]);
        assert_eq!(
            op(raw),
            Op::Fadd {
                dst: 0,
                a: 1,
                am: FMod { neg: true, abs: true },
                b: Operand::Reg(2),
                bm: FMod::NONE,
                ftz: false,
                sat: false,
            }
        );
    }

    #[test]
    fn decodes_isetp_and_its_predicate_destinations() {
        // isetp.lt.and p0, pt, r1, r2, pt — cmp at 49, signed at 48,
        // destinations at [3,6) and [0,3), source predicate at [39,42).
        let raw = asm(
            0x5b60,
            &[(0, 3, 7), (3, 3, 0), (8, 8, 1), (20, 8, 2), (39, 3, 7), (48, 1, 1), (49, 3, 1)],
        );
        assert_eq!(
            op(raw),
            Op::Isetp {
                p0: 0,
                p1: 7,
                a: 1,
                b: Operand::Reg(2),
                cmp: ICmp::Lt,
                signed: true,
                bop: BoolOp::And,
                src: Pred::ALWAYS,
            }
        );
    }

    #[test]
    fn decodes_a_relative_branch_to_an_absolute_offset() {
        // `bra` is pc-relative with an addend of 8, so from the instruction
        // at 0x18 an offset of -0x10 lands at 0x10.
        let raw = asm(0xe240, &[(20, 24, (-0x10i64) as u64)]);
        assert_eq!(decode_at(raw, 0x18).op, Op::Bra { target: 0x10 });
    }

    #[test]
    fn decodes_the_reconvergence_ops() {
        assert_eq!(decode_at(asm(0xe290, &[(20, 24, 0x18)]), 0).op, Op::Ssy { target: 0x20 });
        assert_eq!(op(asm(0xf0f8, &[])), Op::Sync);
        assert_eq!(op(asm(0xe340, &[])), Op::Brk);
        assert_eq!(op(asm(0x50b0, &[])), Op::Nop);
    }

    #[test]
    fn decodes_integer_alu() {
        // iadd r0, r1, -r2
        assert_eq!(
            op(asm(0x5c10, &[(0, 8, 0), (8, 8, 1), (20, 8, 2), (48, 1, 1)])),
            Op::Iadd { dst: 0, a: 1, aneg: false, b: Operand::Reg(2), bneg: true }
        );
        // shl r3, r4, 0x2 (immediate form)
        assert_eq!(
            op(asm(0x3848, &[(0, 8, 3), (8, 8, 4), (20, 19, 2)])),
            Op::Shl { dst: 3, a: 4, b: Operand::Imm(2), wrap: false }
        );
        // lop.and r0, r1, r2
        assert_eq!(
            op(asm(0x5c40, &[(0, 8, 0), (8, 8, 1), (20, 8, 2)])),
            Op::Lop {
                dst: 0,
                a: 1,
                ainv: false,
                b: Operand::Reg(2),
                binv: false,
                op: LogicOp::And,
            }
        );
        // mov r5, r6 — the byte-enable mask must be "all four".
        assert_eq!(
            op(asm(0x5c98, &[(0, 8, 5), (20, 8, 6), (39, 4, 0xf)])),
            Op::Mov { dst: 5, src: Operand::Reg(6) }
        );
    }

    #[test]
    fn decodes_conversions() {
        // i2f.f32.s32 r0, r1
        assert_eq!(
            op(asm(0x5cb8, &[(0, 8, 0), (20, 8, 1), (8, 2, 2), (10, 2, 2), (13, 1, 1)])),
            Op::I2f {
                dst: 0,
                src: Operand::Reg(1),
                sm: FMod::NONE,
                src_bytes: 4,
                src_signed: true,
                sel: 0,
            }
        );
        // f2i.s32.f32.trunc r2, r3
        assert_eq!(
            op(asm(
                0x5cb0,
                &[(0, 8, 2), (20, 8, 3), (10, 2, 2), (8, 2, 2), (12, 1, 1), (39, 2, 3)]
            )),
            Op::F2i {
                dst: 2,
                src: Operand::Reg(3),
                sm: FMod::NONE,
                dst_bytes: 4,
                dst_signed: true,
                round: FRound::Trunc,
                ftz: false,
            }
        );
    }

    #[test]
    fn a_constant_offset_is_a_signed_word_index_scaled_by_four() {
        // c0[0x30] from the JKSV capture: 0xc in the 14-bit field, x4.
        assert_eq!(const_operand(0x4c58100000c70204), Operand::Const { bank: 0, offset: 0x30 });
    }
}
