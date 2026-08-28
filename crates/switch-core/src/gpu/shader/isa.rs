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

/// `fmul`'s pre-scale, applied to its **first** operand before the multiply.
/// A shader uses it to fold a constant halving or doubling into a multiply it
/// was doing anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmulScale {
    None,
    D2,
    D4,
    D8,
    M8,
    M4,
    M2,
}

impl FmulScale {
    fn decode(bits: u64) -> Option<FmulScale> {
        Some(match bits {
            0 => FmulScale::None,
            1 => FmulScale::D2,
            2 => FmulScale::D4,
            3 => FmulScale::D8,
            4 => FmulScale::M8,
            5 => FmulScale::M4,
            6 => FmulScale::M2,
            _ => return None,
        })
    }

    pub fn factor(self) -> f32 {
        match self {
            FmulScale::None => 1.0,
            FmulScale::D2 => 0.5,
            FmulScale::D4 => 0.25,
            FmulScale::D8 => 0.125,
            FmulScale::M8 => 8.0,
            FmulScale::M4 => 4.0,
            FmulScale::M2 => 2.0,
        }
    }
}

/// Which halves of a source register feed a half-precision op's two lanes.
///
/// `F32` is the odd one out: the source is a single f32 rather than a pair of
/// halves, and both lanes read it. Maxwell lets one operand of a half
/// instruction be full precision, which is how a shader multiplies a `half2`
/// by a `float` without converting anything first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HSwizzle {
    /// Lane 0 from the low half, lane 1 from the high half.
    H1H0,
    F32,
    /// Both lanes from the low half.
    H0H0,
    /// Both lanes from the high half.
    H1H1,
}

impl HSwizzle {
    fn decode(bits: u64) -> HSwizzle {
        match bits {
            0 => HSwizzle::H1H0,
            1 => HSwizzle::F32,
            2 => HSwizzle::H0H0,
            _ => HSwizzle::H1H1,
        }
    }
}

/// How a half-precision op writes its two lane results back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HMerge {
    /// Pack both lanes into the destination.
    H1H0,
    /// Widen lane 0 to f32 and write the whole register.
    F32,
    /// Replace only the destination's low half, with lane 0.
    MrgH0,
    /// Replace only its high half, with lane 1.
    MrgH1,
}

impl HMerge {
    fn decode(bits: u64) -> HMerge {
        match bits {
            0 => HMerge::H1H0,
            1 => HMerge::F32,
            2 => HMerge::MrgH0,
            _ => HMerge::MrgH1,
        }
    }
}

/// `hmul2`/`hfma2`'s denormal and zero handling (`HalfPrecision`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HPrecision {
    None,
    /// Flush a subnormal operand to zero.
    Ftz,
    /// D3D9's rule: anything multiplied by zero is zero, NaN and infinity
    /// included.
    Fmz,
}

impl HPrecision {
    /// The fourth encoding is hardware's "don't care", which is free to be
    /// the plain mode.
    fn decode(bits: u64) -> HPrecision {
        match bits {
            1 => HPrecision::Ftz,
            2 => HPrecision::Fmz,
            _ => HPrecision::None,
        }
    }

    /// Whether a product one of whose operands is zero answers zero whatever
    /// the other one is.
    ///
    /// Saturation already forces that answer, so hardware does not do both —
    /// and stating it here is what keeps the interpreter and the WGSL backend
    /// from each deciding it separately.
    pub fn zeroes_products(self, sat: bool) -> bool {
        self == HPrecision::Fmz && !sat
    }
}

/// What `lop`'s test form asks of the result before it writes its predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LopTest {
    /// `.T` — the predicate is set unconditionally.
    True,
    /// `.Z` — set when the result is zero.
    Zero,
    /// `.NZ` — set when any bit of the result is set.
    NonZero,
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
    /// A 2D array. The third coordinate slot holds the *layer*, as an integer
    /// in the low half of its register rather than a float — see
    /// [`texs_encoding`].
    T2dArray,
    T3d,
    TCube,
}

/// `bar`'s sub-operation (`tabf0a8_0`). The reduction forms are decoded so
/// that one can be named when it is refused: they combine a value across every
/// lane of a warp, which a scalar interpreter has no way to see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarMode {
    Sync,
    Arrive,
    RedPopc,
    RedAnd,
    RedOr,
    Scan,
}

/// Which lane a `shfl` reads (`ShuffleMode` in Eden's `warp_shuffle.cpp`).
///
/// Every mode names a source lane relative to the one executing it: an
/// absolute index, a fixed distance below or above, or the lane whose id
/// differs in the bits `index` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShflMode {
    Idx,
    Up,
    Down,
    Bfly,
}

/// Which address space an atomic addresses. `atom`/`red` are global,
/// `atoms` is shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomSpace {
    Global,
    Shared,
}

/// An atomic's read-modify-write (`tabed00_0`/`tabec00_0`/`tabebf8_0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomOp {
    Add,
    Min,
    Max,
    /// Increment, wrapping to zero once the value reaches the operand.
    Inc,
    /// Decrement, wrapping to the operand once the value reaches zero.
    Dec,
    And,
    Or,
    Xor,
    Exch,
    /// Compare-and-swap: `src` is the comparand and `src + 1` the new value.
    Cas,
    /// `safeadd` — an add the hardware may drop under contention. Nothing
    /// here is contended, so it is an add.
    SafeAdd,
}

/// How an atomic interprets the memory it operates on
/// (`tabed00sz`/`tabec00sz`/`tabebf8sz`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomType {
    U32,
    S32,
    U64,
    S64,
    F32,
    U128,
}

impl AtomType {
    /// How many 32-bit registers a value of this type covers.
    pub fn regs(self) -> u8 {
        match self {
            AtomType::U64 | AtomType::S64 => 2,
            AtomType::U128 => 4,
            _ => 1,
        }
    }
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
    /// `ipa[.pass][.centroid] dst, a[offset], mul` — fixed-function
    /// interpolation. `perspective = false` is `ipa pass`; `perspective =
    /// true` multiplies the fetched value by `mul` (`RZ` decodes to `None`).
    /// `centroid` samples the varying inside the primitive's covered area
    /// rather than at the pixel centre.
    Ipa { dst: u8, offset: u16, mul: Option<u8>, perspective: bool, sat: bool, centroid: bool },

    // ---- float ALU ----
    Fadd { dst: u8, a: u8, am: FMod, b: Operand, bm: FMod, ftz: bool, sat: bool },
    Fmul { dst: u8, a: u8, b: Operand, bm: FMod, ftz: bool, sat: bool, scale: FmulScale },
    Ffma { dst: u8, a: u8, b: Operand, bneg: bool, c: Operand, cneg: bool, ftz: bool, sat: bool },
    Fmnmx { dst: u8, a: u8, am: FMod, b: Operand, bm: FMod, pred: Pred, ftz: bool },
    Fsetp { p0: u8, p1: u8, a: u8, am: FMod, b: Operand, bm: FMod, cmp: FCmp, bop: BoolOp, src: Pred },
    Fset { dst: u8, a: u8, am: FMod, b: Operand, bm: FMod, cmp: FCmp, bop: BoolOp, src: Pred, bf: bool },
    Mufu { dst: u8, src: u8, sm: FMod, op: MufuOp, sat: bool },

    // ---- half-precision ALU ----
    // A register is a pair of halves and each of these computes both lanes at
    // once, which is why a Unity shader — written in `half` throughout — is
    // most of these and few of the f32 ops above. [`HSwizzle`] says where each
    // source's two lanes come from and [`HMerge`] where the result goes.
    Hadd2 {
        dst: u8,
        a: u8,
        am: FMod,
        asw: HSwizzle,
        b: Operand,
        bm: FMod,
        bsw: HSwizzle,
        merge: HMerge,
        ftz: bool,
        sat: bool,
    },
    Hmul2 {
        dst: u8,
        a: u8,
        am: FMod,
        asw: HSwizzle,
        b: Operand,
        bm: FMod,
        bsw: HSwizzle,
        merge: HMerge,
        prec: HPrecision,
        sat: bool,
    },
    Hfma2 {
        dst: u8,
        a: u8,
        asw: HSwizzle,
        b: Operand,
        bneg: bool,
        bsw: HSwizzle,
        c: Operand,
        cneg: bool,
        csw: HSwizzle,
        merge: HMerge,
        prec: HPrecision,
        sat: bool,
    },
    /// The two lanes' comparisons land in the two halves of `dst`.
    Hset2 {
        dst: u8,
        a: u8,
        am: FMod,
        asw: HSwizzle,
        b: Operand,
        bm: FMod,
        bsw: HSwizzle,
        cmp: FCmp,
        bop: BoolOp,
        src: Pred,
        bf: bool,
        ftz: bool,
    },
    /// Unlike `fsetp`, the two destination predicates are the two *lanes*, not
    /// a result and its inverse — until `and`, which ands them together and
    /// then writes the inverse into `p1` after all.
    Hsetp2 {
        p0: u8,
        p1: u8,
        a: u8,
        am: FMod,
        asw: HSwizzle,
        b: Operand,
        bm: FMod,
        bsw: HSwizzle,
        cmp: FCmp,
        bop: BoolOp,
        src: Pred,
        and: bool,
        ftz: bool,
    },

    // ---- integer ALU ----
    /// `cin` is `IADD.X`, which adds the carry a previous `IADD.CC` left
    /// behind, and `cout` is that `.CC`. Together they are how a shader adds a
    /// 64-bit number in two halves — every global-memory address a Maxwell
    /// program computes is one of these pairs.
    Iadd { dst: u8, a: u8, aneg: bool, b: Operand, bneg: bool, cin: bool, cout: bool },
    Iadd3 { dst: u8, a: u8, aneg: bool, b: Operand, bneg: bool, c: Operand, cneg: bool },
    Imnmx { dst: u8, a: u8, b: Operand, pred: Pred, signed: bool },
    Iscadd { dst: u8, a: u8, aneg: bool, b: Operand, bneg: bool, shift: u8 },
    Isetp { p0: u8, p1: u8, a: u8, b: Operand, cmp: ICmp, signed: bool, bop: BoolOp, src: Pred },
    Iset { dst: u8, a: u8, b: Operand, cmp: ICmp, signed: bool, bop: BoolOp, src: Pred, bf: bool },
    Icmp { dst: u8, a: u8, b: Operand, c: u8, cmp: ICmp, signed: bool },
    /// `bfi dst, insert, src, base`: splice `insert` into `base`. `src` packs
    /// the destination field's offset in its low byte and its width in the
    /// next — one operand carrying two numbers, which is why a shader building
    /// a bitfield does it in one instruction rather than a shift and two masks.
    Bfi { dst: u8, insert: u8, src: Operand, base: Operand },
    /// `r2p pr, src, mask`: move bits of `src` into the predicate registers,
    /// one per set bit of `mask`. `byte` selects which byte of `src` supplies
    /// them.
    R2p { src: u8, mask: Operand, byte: u8 },
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
    /// `pred` is the `.T`/`.Z`/`.NZ` form, which tests the result and writes a
    /// predicate as well as (usually) discarding the value into `RZ`.
    Lop { dst: u8, a: u8, ainv: bool, b: Operand, binv: bool, op: LogicOp, pred: Option<(u8, LopTest)> },
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
    /// `ld dst, s[addr + offset]` — a load from the CTA's shared memory.
    Lds { dst: u8, addr: u8, offset: i32, size: MemSize },
    /// `st s[addr + offset], src`.
    Sts { addr: u8, offset: i32, src: u8, size: MemSize },
    /// `atom`/`atoms`/`red` — a read-modify-write of one location. `red` is
    /// this with `dst` = [`RZ`]: the same operation, its old value discarded.
    Atom {
        dst: u8,
        addr: u8,
        offset: i32,
        src: u8,
        op: AtomOp,
        ty: AtomType,
        space: AtomSpace,
    },

    // ---- texture ----
    /// `texs dst, coords.., handle, dim, mask` — texture sample with an
    /// immediate handle.
    ///
    /// `coords` is in sample order — `u`, `v`, then the third axis — and
    /// holds [`RZ`] in the slots `dim` does not use. `dst`/`dst2` are the
    /// two destination registers the enabled channels are split between;
    /// see [`texs_destinations`].
    Texs {
        dst: u8,
        dst2: u8,
        coords: [u8; 3],
        handle: u16,
        dim: TexDim,
        mask: [bool; 4],
        /// The `.F16` form, which packs two channels into each destination
        /// register as halves instead of giving each one a register.
        f16: bool,
    },

    // ---- warp ----
    /// `shfl.<mode> p, dst, src, index, mask` — read another lane's `src`.
    ///
    /// `index` selects the lane the mode's own way, and `mask` packs two
    /// fields: a clamp in its low five bits and a segment mask at bit 8,
    /// which together bound which lanes this one may reach. `pred` is set to
    /// whether the lane it computed was inside that bound; a lane that was
    /// not keeps its own value.
    Shfl { dst: u8, pred: u8, src: u8, index: Operand, mask: Operand, mode: ShflMode },
    /// `fswzadd dst, a, b, swizzle` — add `a` and `b` with a sign per lane,
    /// the two-bit code for this one selected out of `swizzle` by `laneid`.
    ///
    /// It is the other half of a derivative: `shfl` fetches the neighbour's
    /// value and this subtracts in whichever direction the lane's position in
    /// the quad calls for.
    Fswzadd { dst: u8, a: u8, b: u8, swizzle: u8, ftz: bool },

    // ---- control ----
    /// `bra target` — `target` is an instruction's byte offset within the
    /// program, already resolved from the pc-relative encoding.
    Bra { target: u32 },
    /// `ssy target` — push a reconvergence point.
    /// `brx Ra, imm`: an indexed branch, which is how a `switch` lowers. The
    /// register holds an entry a jump table in a constant bank supplied, and
    /// the target is that entry plus this instruction's own pc-relative base —
    /// so an interpreter, unlike a recompiler, needs no table tracking at all.
    Brx { base: u32, reg: u8 },
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
    /// `bar.<mode>` — a CTA-wide barrier.
    Bar { mode: BarMode },
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
/// A branch's pc-relative displacement resolved to a raw byte offset, before
/// any slot alignment.
fn branch_base(insn: u64, pc: u32) -> u32 {
    (pc as i64 + 8 + sfield(insn, 20, 24)) as u32
}

fn branch_target(insn: u64, pc: u32) -> u32 {
    super::align_slot(branch_base(insn, pc))
}

/// Decode a single 8-byte Maxwell instruction word sitting at byte offset
/// `pc` within its program (needed to resolve pc-relative branches). Never
/// panics: an unrecognised or unsupported bit pattern decodes to
/// [`Op::Unimplemented`].
pub fn decode_at(insn: u64, pc: u32) -> Instruction {
    let op = decode_op(insn, pc);
    // `ssy`/`pbk`/`pcnt` have no predicate: the bits every other instruction
    // keeps its guard in belong to their branch target, and they read as zero
    // — which is `@p0`, a predicate that is false until something sets it. So
    // the push was skipped and the `sync`/`brk` that matched it found an empty
    // reconvergence stack, which is where every one of the Home Menu's 222
    // textured draws stopped.
    let pred = match op {
        Op::Ssy { .. } | Op::Pbk { .. } | Op::Pcnt { .. } => Pred::ALWAYS,
        _ => guard(insn),
    };
    Instruction { pred, op }
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
        // ld/st s[] — 0xef48/0xef58, the shared-memory pair of the local
        // ones below and encoded identically.
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
        // depbar/membar: scheduling and memory ordering, no-ops for a scalar
        // interpreter that runs one invocation at a time.
        0xf0f0 | 0xef98 => Op::Inert,
        // bar — 0xf0a8/0xfff8. The mode's bits are not contiguous.
        0xf0a8 => match (field(insn, 39, 1) << 4)
            | (field(insn, 36, 1) << 3)
            | (field(insn, 35, 1) << 2)
            | field(insn, 32, 2)
        {
            0b00010 => Op::Bar { mode: BarMode::RedPopc },
            0b00011 => Op::Bar { mode: BarMode::Scan },
            0b00110 => Op::Bar { mode: BarMode::RedAnd },
            0b01010 => Op::Bar { mode: BarMode::RedOr },
            0b10000 => Op::Bar { mode: BarMode::Sync },
            0b10001 => Op::Bar { mode: BarMode::Arrive },
            _ => un,
        },
        // red — 0xebf8/0xfff8. A global atomic whose old value is discarded,
        // so it decodes to the same op with RZ as its destination.
        0xebf8 => {
            let (Some(op), Some(ty)) =
                (atom_op(field(insn, 23, 3)), atom_type(field(insn, 20, 3)))
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
        // shfl — 0xef10/0xfff8 (Eden's `maxwell.inc`,
        // "1110 1111 0001 0---"). Both operands can be a register or an
        // immediate, and each has its own flag saying which — the lane index
        // five bits at 20, the clamp/segment pair thirteen at 34.
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
            0xe25 if field(insn, 5, 1) == 0 => {
                // The *sum* of the base and the table entry is the target, so
                // this is where alignment must not happen: a base that is a
                // multiple of 32 is a real displacement, not a `sched` word to
                // step over, and rounding it up shifts every arm of the switch
                // one instruction along.
                Op::Brx { base: branch_base(insn, pc), reg: reg(insn, 8, 8) }
            }
            // atom.cas — 0xeef0/0xfff0, whose size field is one bit because
            // the operation is fixed.
            0xeef => Op::Atom {
                dst: reg(insn, 0, 8),
                addr: reg(insn, 8, 8),
                offset: sfield(insn, 28, 20) as i32,
                src: reg(insn, 20, 8),
                op: AtomOp::Cas,
                ty: if field(insn, 49, 1) == 0 { AtomType::U32 } else { AtomType::U64 },
                space: AtomSpace::Global,
            },
            _ => decode_memory_atomic(insn).unwrap_or_else(|| decode_alu(insn)),
        },
    }
}

/// The three atomics whose opcode masks are wider than a nibble: `atom`
/// (0xed00/0xff00), `atoms` (0xec00/0xff00) and `atoms.cas` (0xee00/0xff80).
fn decode_memory_atomic(insn: u64) -> Option<Op> {
    match insn >> 56 {
        // atom — the op is a full nibble and the type three bits below it.
        0xed => Some(Op::Atom {
            dst: reg(insn, 0, 8),
            addr: reg(insn, 8, 8),
            offset: sfield(insn, 28, 20) as i32,
            src: reg(insn, 20, 8),
            op: atom_op(field(insn, 52, 4))?,
            ty: atom_type(field(insn, 49, 3))?,
            space: AtomSpace::Global,
        }),
        // atoms — a 22-bit offset stored in dwords, and a two-bit type.
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
        // atoms.cast/.cas — `cast` is a lock-and-load form nothing here
        // models, so only the compare-and-swap arm decodes.
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
                ty: if field(insn, 52, 1) == 0 { AtomType::U32 } else { AtomType::U64 },
                space: AtomSpace::Shared,
            })
        }
        _ => None,
    }
}

/// `tabed00_0`/`tabec00_0` — the same eight operations in the same order for
/// `atom` and `atoms`, with two more that only `atom` reaches.
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
            let Some(scale) = FmulScale::decode(field(insn, 41, 3)) else { return un };
            Op::Fmul {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                bm: FMod { neg: field(insn, 48, 1) != 0, abs: false },
                ftz: field(insn, 44, 2) == 1,
                sat: field(insn, 50, 1) != 0,
                scale,
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
        // r2p — move register bits into the predicate registers. Bit 40
        // picks the destination file: `PR` (0) is the predicates, `CC` (1) the
        // condition-code flags, which nothing here models.
        0xf0 => {
            let Some(mask) = rhs_int else { return un };
            if field(insn, 40, 1) != 0 {
                return un; // the CC form
            }
            Op::R2p { src: reg(insn, 8, 8), mask, byte: field(insn, 41, 2) as u8 }
        }
        // ---- integer ----
        // iadd — sat 50, x 43, a: neg 49, b: neg 48.
        0x10 => {
            let Some(b) = rhs_int else { return un };
            if field(insn, 50, 1) != 0 {
                return un; // saturating add
            }
            Op::Iadd {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                aneg: field(insn, 49, 1) != 0,
                b,
                bneg: field(insn, 48, 1) != 0,
                cin: field(insn, 43, 1) != 0,
                cout: field(insn, 47, 1) != 0,
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
            if field(insn, 43, 1) != 0 {
                return un; // extended-carry form
            }
            let op = match field(insn, 41, 2) {
                0 => LogicOp::And,
                1 => LogicOp::Or,
                2 => LogicOp::Xor,
                _ => LogicOp::PassB,
            };
            // The test form writes a predicate from the result as well as the
            // register, and the register is usually `RZ`: this is how a shader
            // asks "are any of these bits set" in one instruction.
            let pred = match field(insn, 44, 2) {
                0 => None,
                1 => Some((reg(insn, 48, 3), LopTest::True)),
                2 => Some((reg(insn, 48, 3), LopTest::Zero)),
                _ => Some((reg(insn, 48, 3), LopTest::NonZero)),
            };
            Op::Lop {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                ainv: field(insn, 39, 1) != 0,
                b,
                binv: field(insn, 40, 1) != 0,
                op,
                pred,
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
        // rro — the range-reduction operator that precedes `mufu`.
        //
        // On hardware `mufu sin`/`cos`/`ex2` take an argument already folded
        // into the range their tables cover, and `rro` is what folds it. The
        // `mufu` here is not a table: it calls the host's `sin`, `cos` and
        // `exp2`, which take the argument as it comes. So the fold is the
        // identity, and modelling it as one is what makes the pair compute
        // the function rather than something adjacent to it.
        //
        // The modifiers are refused rather than ignored: a negate or an
        // absolute value dropped on the floor is a wrong answer that looks
        // like a right one.
        0x90 => {
            let Some(src) = rhs_float else { return un };
            if field(insn, 45, 1) != 0 || field(insn, 49, 1) != 0 || field(insn, 50, 1) != 0 {
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

    // The half-precision group first: its opcodes are spread across five
    // different masks, none of which any arm below claims.
    if let Some(op) = decode_half(insn) {
        return op;
    }

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
        0x5b4 | 0x4b4 | 0x534 | 0x364 | 0x374 => {
            let (b, c) = match form >> 4 {
                0x5b4 => (Operand::Reg(reg(insn, 20, 8)), reg(insn, 39, 8)),
                0x364 | 0x374 => (Operand::Imm(imm20(insn)), reg(insn, 39, 8)),
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
        // bfi — `src` carries the field's offset and width packed into one
        // operand; `base` is what the insert lands in. The four forms differ
        // only in where those two come from.
        0x5bf | 0x4bf | 0x53f | 0x36f | 0x37f => {
            let (src, base) = match form >> 4 {
                0x5bf => (Operand::Reg(reg(insn, 20, 8)), Operand::Reg(reg(insn, 39, 8))),
                0x4bf => (const_operand(insn), Operand::Reg(reg(insn, 39, 8))),
                0x53f => (Operand::Reg(reg(insn, 39, 8)), const_operand(insn)),
                _ => (Operand::Imm(imm20(insn)), Operand::Reg(reg(insn, 39, 8))),
            };
            return Op::Bfi { dst: reg(insn, 0, 8), insert: reg(insn, 8, 8), src, base };
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

    // vote.vtg — 0x50e0/0xfff8, and nothing to do. The vertex-stage vote
    // writes neither a register nor a predicate: it tells the hardware's
    // tessellation and stream-out fixed function about the warp, and there is
    // no such fixed function here. Eden stubs it the same way
    // (`shader_recompiler/frontend/maxwell/translate/impl/vote.cpp`, where
    // `VOTE_vtg` logs and emits no IR) while implementing plain `VOTE` fully.
    //
    // Refusing it is not free: a refused instruction fails the whole draw, and
    // this one sits two instructions before `exit` in Just Dance 2023's
    // loading-screen vertex shader — every draw the title made, all 52 of
    // them, and the frame it presented was the clear colour and nothing else.
    if insn & 0xfff8_0000_0000_0000 == 0x50e0_0000_0000_0000 {
        return Op::Nop;
    }

    // fswzadd — 0x50f8/0xfff8. `ndv` at 38 is a scheduling hint about
    // divergence and has no effect here; the condition-code write at 47 is
    // refused rather than dropped, since a shader that reads the flag would
    // read whatever was left there. Only round-to-nearest is decoded: this
    // is the tail of a derivative, and a rounding mode silently ignored
    // biases every one of them.
    if insn & 0xfff8_0000_0000_0000 == 0x50f8_0000_0000_0000 {
        if field(insn, 47, 1) != 0 || field(insn, 39, 2) != 0 {
            return un;
        }
        return Op::Fswzadd {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b: reg(insn, 20, 8),
            swizzle: field(insn, 28, 8) as u8,
            ftz: field(insn, 44, 1) != 0,
        };
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
        // The sample mode (`SampleMode` in Eden's decode): 0 default, 1
        // centroid, 2 offset. Offset samples wherever a register points,
        // which is a per-invocation position neither renderer has; the
        // sc/constant interpolation modes are not interpolation at all.
        let sample = field(insn, 52, 2);
        if sample > 1 || mode > 1 {
            return un;
        }
        return Op::Ipa {
            dst: reg(insn, 0, 8),
            offset: field(insn, 28, 10) as u16,
            mul: opt_reg(reg(insn, 20, 8)),
            perspective: mode == 1,
            sat: field(insn, 51, 1) != 0,
            centroid: sample == 1,
        };
    }

    // texs — the immediate-handle sample. Bit 49 is `nodep`, a scheduling
    // hint with no effect on what the instruction computes, so it is ignored
    // rather than decoded: refusing it cost the Home Menu 156 of its textured
    // draws, every one of them an ordinary 2D sample.
    if insn & 0xf600_0000_0000_0000 == 0xd000_0000_0000_0000 {
        let dst = reg(insn, 0, 8);
        let dst2 = reg(insn, 28, 8);
        let (a, b) = (reg(insn, 8, 8), reg(insn, 20, 8));
        if let (Some((dim, coords)), Some(mask)) = (
            texs_encoding(field(insn, 53, 4), a, b),
            decode_tex_mask(field(insn, 50, 3), dst, dst2),
        ) {
            return Op::Texs {
                dst,
                dst2,
                coords,
                handle: field(insn, 36, 13) as u16,
                dim,
                mask,
                // `Precision` counts F16 as 0 and F32 as 1, so the bit
                // being *clear* is the packed form.
                f16: field(insn, 59, 1) == 0,
            };
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
            // `fmul32i` spends the bits a pre-scale would need on its
            // 32-bit immediate, so it has none.
            scale: FmulScale::None,
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
            cin: false,
            // `iadd32i` writes the carry from bit 52.
            cout: field(insn, 52, 1) != 0,
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
            pred: None,
        };
    }

    un
}

/// The half pair an immediate form of a half-precision op carries: two halves
/// whose low six mantissa bits the encoding has no room for, each with a sign
/// bit of its own well away from the rest of it.
fn half_imm(insn: u64) -> u32 {
    let low = (field(insn, 20, 9) << 6) | (field(insn, 29, 1) << 15);
    let high = (field(insn, 30, 9) << 22) | (field(insn, 56, 1) << 31);
    (low | high) as u32
}

/// The second operand of a half op's constant-or-immediate pair, and where
/// its two lanes come from. Bit 55 of the opcode picks which form it is, and
/// the two carry their lanes differently: a constant bank holds one f32 that
/// both lanes read, an immediate holds a packed pair.
fn half_cbuf_or_imm(insn: u64, cbuf: bool) -> (Operand, HSwizzle) {
    if cbuf {
        (const_operand(insn), HSwizzle::F32)
    } else {
        (Operand::Imm(half_imm(insn)), HSwizzle::H1H0)
    }
}

/// The half-precision group: `hadd2`, `hmul2`, `hfma2`, `hset2` and `hsetp2`,
/// in every operand form.
///
/// Opcodes and field positions come from Eden's `maxwell.inc` and its
/// `half_floating_point_*.cpp` translators. The forms differ in more than
/// where the second operand comes from — a register form keeps `b`'s
/// modifiers below bit 32 and a constant or immediate one puts them up at
/// 52..57 — so each is written out rather than shared.
///
/// Without these, "A Short Hike" lost 145 of its 295 draws, each to the first
/// `hadd2` in its shader, and with them the two full-screen quads it
/// composites its frame out of. A Unity shader is written in `half`
/// throughout, so this is most of its arithmetic rather than a corner of it.
fn decode_half(insn: u64) -> Option<Op> {
    let top = insn >> 48;
    let un = || Some(Op::Unimplemented { raw: insn });
    let dst = reg(insn, 0, 8);
    let a = reg(insn, 8, 8);
    let merge = HMerge::decode(field(insn, 49, 2));
    let asw = HSwizzle::decode(field(insn, 47, 2));
    let reg20 = Operand::Reg(reg(insn, 20, 8));
    let reg39 = Operand::Reg(reg(insn, 39, 8));
    // Every immediate form carries the same packed pair, and every `32I` form
    // the same full 32 bits in place of it.
    let imm = Operand::Imm(half_imm(insn));
    let imm32 = Operand::Imm(field(insn, 20, 32) as u32);
    let bsw_reg = HSwizzle::decode(field(insn, 28, 2));
    let no_mod = FMod::NONE;

    // ---- hadd2 ----
    if top & 0xfff8 == 0x5d10 {
        return Some(Op::Hadd2 {
            dst,
            a,
            am: FMod { neg: field(insn, 43, 1) != 0, abs: field(insn, 44, 1) != 0 },
            asw,
            b: reg20,
            bm: FMod { neg: field(insn, 31, 1) != 0, abs: field(insn, 30, 1) != 0 },
            bsw: bsw_reg,
            merge,
            ftz: field(insn, 39, 1) != 0,
            sat: field(insn, 32, 1) != 0,
        });
    }
    if top & 0xfe80 == 0x7a80 || top & 0xfe80 == 0x7a00 {
        let cbuf = top & 0x0080 != 0;
        let (b, bsw) = half_cbuf_or_imm(insn, cbuf);
        return Some(Op::Hadd2 {
            dst,
            a,
            am: FMod { neg: field(insn, 43, 1) != 0, abs: field(insn, 44, 1) != 0 },
            asw,
            b,
            // An immediate form spends the bits a modifier would need on the
            // pair's own two signs.
            bm: if cbuf {
                FMod { neg: field(insn, 56, 1) != 0, abs: field(insn, 54, 1) != 0 }
            } else {
                no_mod
            },
            bsw,
            merge,
            ftz: field(insn, 39, 1) != 0,
            sat: field(insn, 52, 1) != 0,
        });
    }
    // hadd2_32i — its own field positions, and the merge is fixed.
    if top & 0xfe00 == 0x2c00 {
        return Some(Op::Hadd2 {
            dst,
            a,
            am: FMod { neg: field(insn, 56, 1) != 0, abs: false },
            asw: HSwizzle::decode(field(insn, 53, 2)),
            b: imm32,
            bm: no_mod,
            bsw: HSwizzle::H1H0,
            merge: HMerge::H1H0,
            ftz: field(insn, 55, 1) != 0,
            sat: field(insn, 52, 1) != 0,
        });
    }

    // ---- hmul2 ----
    if top & 0xfff8 == 0x5d08 {
        return Some(Op::Hmul2 {
            dst,
            a,
            am: FMod { neg: false, abs: field(insn, 44, 1) != 0 },
            asw,
            b: reg20,
            bm: FMod { neg: field(insn, 31, 1) != 0, abs: field(insn, 30, 1) != 0 },
            bsw: bsw_reg,
            merge,
            prec: HPrecision::decode(field(insn, 39, 2)),
            sat: field(insn, 32, 1) != 0,
        });
    }
    if top & 0xfe80 == 0x7880 || top & 0xfe80 == 0x7800 {
        let cbuf = top & 0x0080 != 0;
        let (b, bsw) = half_cbuf_or_imm(insn, cbuf);
        return Some(Op::Hmul2 {
            dst,
            a,
            am: FMod { neg: field(insn, 43, 1) != 0, abs: field(insn, 44, 1) != 0 },
            asw,
            b,
            bm: if cbuf {
                FMod { neg: false, abs: field(insn, 54, 1) != 0 }
            } else {
                no_mod
            },
            bsw,
            merge,
            prec: HPrecision::decode(field(insn, 39, 2)),
            sat: field(insn, 52, 1) != 0,
        });
    }
    if top & 0xfe00 == 0x2a00 {
        return Some(Op::Hmul2 {
            dst,
            a,
            am: no_mod,
            asw: HSwizzle::decode(field(insn, 53, 2)),
            b: imm32,
            bm: no_mod,
            bsw: HSwizzle::H1H0,
            merge: HMerge::H1H0,
            prec: HPrecision::decode(field(insn, 55, 2)),
            sat: field(insn, 52, 1) != 0,
        });
    }

    // ---- hfma2 ----
    if top & 0xfff8 == 0x5d00 {
        return Some(Op::Hfma2 {
            dst,
            a,
            asw,
            b: reg20,
            bneg: field(insn, 31, 1) != 0,
            bsw: bsw_reg,
            c: reg39,
            cneg: field(insn, 30, 1) != 0,
            csw: HSwizzle::decode(field(insn, 35, 2)),
            merge,
            prec: HPrecision::decode(field(insn, 37, 2)),
            sat: field(insn, 32, 1) != 0,
        });
    }
    // The `rc`, `cr` and `imm` forms share every modifier position and differ
    // only in which of `b` and `c` is the constant bank.
    if top & 0xf880 == 0x6080 || top & 0xf880 == 0x7080 || top & 0xf880 == 0x7000 {
        let (b, bsw, c, csw) = if top & 0xf880 == 0x6080 {
            (reg39, HSwizzle::decode(field(insn, 53, 2)), const_operand(insn), HSwizzle::F32)
        } else if top & 0x0080 != 0 {
            (const_operand(insn), HSwizzle::F32, reg39, HSwizzle::decode(field(insn, 53, 2)))
        } else {
            (imm, HSwizzle::H1H0, reg39, HSwizzle::decode(field(insn, 53, 2)))
        };
        return Some(Op::Hfma2 {
            dst,
            a,
            asw,
            b,
            // The immediate form spends bit 56 on the high half's sign, so it
            // is the one form with no negate for `b`.
            bneg: top & 0xf880 != 0x7000 && field(insn, 56, 1) != 0,
            bsw,
            c,
            cneg: field(insn, 51, 1) != 0,
            csw,
            merge,
            prec: HPrecision::decode(field(insn, 57, 2)),
            sat: field(insn, 52, 1) != 0,
        });
    }
    // hfma2_32i — the addend is the destination register, which is the only
    // place the encoding has left to name it.
    if top & 0xfe00 == 0x2800 {
        return Some(Op::Hfma2 {
            dst,
            a,
            asw: HSwizzle::decode(field(insn, 53, 2)),
            b: imm32,
            bneg: false,
            bsw: HSwizzle::H1H0,
            c: Operand::Reg(dst),
            cneg: field(insn, 52, 1) != 0,
            csw: HSwizzle::H1H0,
            merge: HMerge::H1H0,
            prec: HPrecision::decode(field(insn, 55, 2)),
            sat: false,
        });
    }

    // ---- hset2 / hsetp2 ----
    // Both read `a`'s modifiers, their source predicate and their boolean
    // combiner from the same places, and `hset2`'s `bf` is `hsetp2`'s `and`.
    let set_am = FMod { neg: field(insn, 43, 1) != 0, abs: field(insn, 44, 1) != 0 };
    let src = src_pred(insn, 39, 42);
    let is_set2 = top & 0xfff8 == 0x5d18 || top & 0xfe00 == 0x7c00;
    let is_setp2 = top & 0xfff8 == 0x5d20 || top & 0xfe00 == 0x7e00;
    if is_set2 || is_setp2 {
        let Some(bop) = bool_op(field(insn, 45, 2)) else { return un() };
        let register_form = top & 0xf000 == 0x5000;
        let cbuf = !register_form && top & 0x0080 != 0;
        let (b, bm, bsw) = if register_form {
            (
                reg20,
                FMod { neg: field(insn, 31, 1) != 0, abs: field(insn, 30, 1) != 0 },
                bsw_reg,
            )
        } else if cbuf {
            (
                const_operand(insn),
                FMod { neg: field(insn, 56, 1) != 0, abs: field(insn, 54, 1) != 0 },
                HSwizzle::F32,
            )
        } else {
            (imm, no_mod, HSwizzle::H1H0)
        };
        // The comparison is four bits either just above the swizzle or up
        // among the constant form's modifiers.
        let cmp = fcmp(if register_form { field(insn, 35, 4) } else { field(insn, 49, 4) });
        let flag = field(insn, if register_form { 49 } else { 53 }, 1) != 0;
        if is_set2 {
            return Some(Op::Hset2 {
                dst,
                a,
                am: set_am,
                asw,
                b,
                bm,
                bsw,
                cmp,
                bop,
                src,
                bf: flag,
                ftz: field(insn, if register_form { 50 } else { 54 }, 1) != 0,
            });
        }
        return Some(Op::Hsetp2 {
            p0: reg(insn, 3, 3),
            p1: reg(insn, 0, 3),
            a,
            am: set_am,
            asw,
            b,
            bm,
            bsw,
            cmp,
            bop,
            src,
            and: flag,
            // `hsetp2` spends its destination predicates on the bits every
            // other form keeps `ftz` in, so its own sits down at bit 6.
            ftz: field(insn, 6, 1) != 0,
        });
    }

    None
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
/// A `texs`'s encoding field: which texture shape it samples, and where its
/// coordinates come from. The two operand registers are `REG_08` (`a`) and
/// `REG_20` (`b`), but which axes they feed differs per encoding — a 2D
/// sample takes `(a, b)`, while one that also carries an explicit LOD takes
/// `(a, a + 1)` and leaves `b` for the level.
///
/// The encodings left out are the ones this rasteriser has no model for:
/// depth-compare (4, 5, 6, 9) and 2D arrays (7, 8). They decode to
/// [`Op::Unimplemented`] and their draw is skipped with a reason, which is
/// better than sampling the wrong thing silently.
fn texs_encoding(bits: u64, a: u8, b: u8) -> Option<(TexDim, [u8; 3])> {
    let next = a.wrapping_add(1);
    match bits {
        // 1D.LZ
        0 => Some((TexDim::T1d, [a, RZ, RZ])),
        // 2D, 2D.LZ
        1 | 2 => Some((TexDim::T2d, [a, b, RZ])),
        // 2D.LL — `b` is the level, not a coordinate.
        3 => Some((TexDim::T2d, [a, next, RZ])),
        // ARRAY_2D, ARRAY_2D.LZ — the layer comes from `a`, and the
        // coordinates from `a + 1` and `b`.
        7 | 8 => Some((TexDim::T2dArray, [next, b, a])),
        // 3D, 3D.LZ
        10 | 11 => Some((TexDim::T3d, [a, next, b])),
        // CUBE, CUBE.LL
        12 | 13 => Some((TexDim::TCube, [a, next, b])),
        _ => None,
    }
}

/// Which colour channels a `texs` writes.
///
/// The three-bit selector does not name the channels on its own: it indexes
/// a different row depending on how many destination registers the
/// instruction has, since each one takes at most two channels. With both
/// present the rows are the three- and four-channel sets
/// (`rgb`/`rga`/`rba`/`gba`/`rgba`); with only one they are the one- and
/// two-channel sets (`r`/`g`/`b`/`a`/`rg`/`ra`/`ga`/`ba`). Reading the
/// four-destination row for a single-destination sample turns a one-channel
/// fetch — which is what a glyph out of an alpha atlas is — into a
/// three-channel one landing on registers the shader is still using.
///
/// The last three selectors of the two-destination row are encodings this
/// decoder does not know; they come back `None`, which makes the whole
/// instruction [`Op::Unimplemented`] rather than a guess.
fn decode_tex_mask(selector: u64, dst: u8, dst2: u8) -> Option<[bool; 4]> {
    const ONE_DEST: [u8; 8] = [0x1, 0x2, 0x4, 0x8, 0x3, 0x9, 0xa, 0xc];
    const TWO_DEST: [u8; 8] = [0x7, 0xb, 0xd, 0xe, 0xf, 0x0, 0x0, 0x0];
    let row = match (dst != RZ, dst2 != RZ) {
        (false, false) => return None, // a sample with nowhere to land
        (true, true) => TWO_DEST,
        _ => ONE_DEST,
    };
    let bits = row[selector as usize & 7];
    if bits == 0 {
        return None;
    }
    Some([bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0])
}

/// What one of a `texs`'s destination registers ends up holding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexsStore {
    /// The whole register is one channel, as an `f32`.
    Float(usize),
    /// Two channels packed as halves, low first. The second is `None` when an
    /// odd number of channels is enabled, which hardware pads with zero.
    Halves(usize, Option<usize>),
}

/// Where a `texs`'s enabled colour channels land, as `(channel, register)`.
///
/// A `texs` has *two* destination registers, and each holds at most two
/// channels: the first two enabled channels go to `dst` and `dst + 1`, the
/// rest to `dst2` and `dst2 + 1`. The pair is not one run of four.
///
/// The distinction is invisible whenever `dst2 == dst + 2`, which is what
/// the `tex.frag` fixture this decoder was first checked against does — so
/// the run-of-four reading survived until a shader with `dst = $r4,
/// dst2 = $r2` ran under it. There channels 2 and 3 landed on `$r6`/`$r7`,
/// and `$r6` was holding the `1/w` every later `ipa` multiplies by.
pub fn texs_destinations(
    dst: u8,
    dst2: u8,
    mask: [bool; 4],
    f16: bool,
) -> Vec<(u8, TexsStore)> {
    let enabled: Vec<usize> =
        mask.iter().enumerate().filter(|(_, &on)| on).map(|(channel, _)| channel).collect();
    if !f16 {
        return enabled
            .into_iter()
            .enumerate()
            .map(|(n, channel)| {
                let reg = if n < 2 {
                    dst.wrapping_add(n as u8)
                } else {
                    dst2.wrapping_add(n as u8 - 2)
                };
                (reg, TexsStore::Float(channel))
            })
            .collect();
    }
    // Two channels to a register, `dst` then `dst2` — so four channels need
    // two registers rather than four, and the shader reads them back with the
    // half swizzles the `h*2` ops carry.
    enabled
        .chunks(2)
        .enumerate()
        .map(|(n, pair)| {
            let reg = if n == 0 { dst } else { dst2 };
            (reg, TexsStore::Halves(pair[0], pair.get(1).copied()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guard-predicate field holding `PT`.
    const PT: u64 = 7 << 16;

    #[test]
    fn decodes_the_shared_memory_pair() {
        // ld/st s[] sit one nibble above their local counterparts and are
        // encoded identically: a signed 24-bit byte offset off r8.
        assert_eq!(
            decode((0xef48u64 | 4) << 48 | PT | 0x20 << 20 | 5 << 8 | 3).op,
            Op::Lds { dst: 3, addr: 5, offset: 0x20, size: MemSize::B32 }
        );
        assert_eq!(
            decode((0xef58u64 | 5) << 48 | PT | 6 << 8 | 2).op,
            Op::Sts { addr: 6, offset: 0, src: 2, size: MemSize::B64 }
        );
        // Still the local pair, not the shared one.
        assert_eq!(
            decode((0xef40u64 | 4) << 48 | PT | 5 << 8 | 3).op,
            Op::Ldl { dst: 3, addr: 5, offset: 0, size: MemSize::B32 }
        );
    }

    #[test]
    fn a_negative_shared_offset_stays_negative() {
        let offset = (-8i64 as u64) & 0xFF_FFFF;
        assert_eq!(
            decode((0xef48u64 | 4) << 48 | PT | offset << 20 | 5 << 8 | 3).op,
            Op::Lds { dst: 3, addr: 5, offset: -8, size: MemSize::B32 }
        );
    }

    #[test]
    fn decodes_each_barrier_form() {
        // The mode's bits are not contiguous: 0x9b at bit 32, which is what
        // makes `sync` (0x80) and `arrive` (0x81) one bit apart and the
        // reduction forms scattered below them.
        let bar = |mode: u64| decode(0xf0a8u64 << 48 | mode << 32 | PT).op;
        assert_eq!(bar(0x80), Op::Bar { mode: BarMode::Sync });
        assert_eq!(bar(0x81), Op::Bar { mode: BarMode::Arrive });
        assert_eq!(bar(0x02), Op::Bar { mode: BarMode::RedPopc });
        assert_eq!(bar(0x03), Op::Bar { mode: BarMode::Scan });
        assert_eq!(bar(0x0a), Op::Bar { mode: BarMode::RedAnd });
        assert_eq!(bar(0x12), Op::Bar { mode: BarMode::RedOr });
        // membar and depbar are still the no-ops they were.
        assert_eq!(decode(0xef98u64 << 48 | PT).op, Op::Inert);
        assert_eq!(decode(0xf0f0u64 << 48 | PT).op, Op::Inert);
    }

    /// `shfl` carries two operands that are each either a register or an
    /// immediate, with a flag apiece saying which — and the immediates sit in
    /// different fields from the registers, so reading the wrong one gives a
    /// plausible lane number rather than an error.
    #[test]
    fn decodes_a_warp_shuffle_in_each_mode_and_operand_form() {
        // shfl.<mode> p0, r3, r4, 0x1, 0x1c
        let immediate = |mode: u64| {
            decode(
                0xef10u64 << 48
                    | 0x1c << 34
                    | mode << 30
                    | 1 << 29
                    | 1 << 28
                    | 1 << 20
                    | PT
                    | 4 << 8
                    | 3,
            )
            .op
        };
        for (bits, mode) in
            [(0, ShflMode::Idx), (1, ShflMode::Up), (2, ShflMode::Down), (3, ShflMode::Bfly)]
        {
            assert_eq!(
                immediate(bits),
                Op::Shfl {
                    dst: 3,
                    pred: 0,
                    src: 4,
                    index: Operand::Imm(1),
                    mask: Operand::Imm(0x1c),
                    mode,
                }
            );
        }

        // The same instruction with both operands in registers: the lane
        // index at 20 and the clamp/segment pair at 39.
        assert_eq!(
            decode(0xef10u64 << 48 | 3 << 30 | 6 << 39 | 5 << 20 | PT | 4 << 8 | 3 | 2 << 48).op,
            Op::Shfl {
                dst: 3,
                pred: 2,
                src: 4,
                index: Operand::Reg(5),
                mask: Operand::Reg(6),
                mode: ShflMode::Bfly,
            }
        );
    }

    #[test]
    fn decodes_the_per_lane_add_a_derivative_ends_with() {
        // fswzadd r3, r1, r2, 0xe4
        let fswzadd = |extra: u64| decode(0x50f8u64 << 48 | extra | 0xe4 << 28 | 2 << 20 | PT | 1 << 8 | 3).op;
        assert_eq!(
            fswzadd(0),
            Op::Fswzadd { dst: 3, a: 1, b: 2, swizzle: 0xe4, ftz: false }
        );
        assert_eq!(
            fswzadd(1 << 44),
            Op::Fswzadd { dst: 3, a: 1, b: 2, swizzle: 0xe4, ftz: true }
        );
        // A condition-code write and a rounding mode other than nearest are
        // refused rather than dropped: both change what a later instruction
        // reads, and this one is the tail of every derivative in the shader.
        for extra in [1u64 << 47, 1 << 39, 2 << 39] {
            assert!(matches!(fswzadd(extra), Op::Unimplemented { .. }), "{extra:#x}");
        }
    }

    #[test]
    fn decodes_a_global_atomic_with_its_operation_and_type() {
        // atom.max.s32 r3, [r5 + -8], r7
        let offset = (-8i64 as u64) & 0xF_FFFF;
        assert_eq!(
            decode(0xed00u64 << 48 | 2 << 52 | 1 << 49 | offset << 28 | 7 << 20 | PT | 5 << 8 | 3)
                .op,
            Op::Atom {
                dst: 3,
                addr: 5,
                offset: -8,
                src: 7,
                op: AtomOp::Max,
                ty: AtomType::S32,
                space: AtomSpace::Global,
            }
        );
    }

    #[test]
    fn a_shared_atomic_counts_its_offset_in_dwords() {
        // The one place the two atomic encodings genuinely differ: `atoms`
        // stores a dword index where `atom` stores a byte offset, so reading
        // it as bytes divides every address by four.
        assert_eq!(
            decode(0xec00u64 << 48 | 8 << 52 | 3 << 30 | 7 << 20 | PT | 5 << 8 | 3).op,
            Op::Atom {
                dst: 3,
                addr: 5,
                offset: 12,
                src: 7,
                op: AtomOp::Exch,
                ty: AtomType::U32,
                space: AtomSpace::Shared,
            }
        );
    }

    #[test]
    fn red_is_an_atomic_that_discards_its_old_value() {
        // Which is exactly RZ as the destination, so the interpreter needs no
        // second path for it.
        assert_eq!(
            decode(0xebf8u64 << 48 | 2 << 20 | 4 << 28 | PT | 5 << 8 | 3).op,
            Op::Atom {
                dst: RZ,
                addr: 5,
                offset: 4,
                src: 3,
                op: AtomOp::Add,
                ty: AtomType::U64,
                space: AtomSpace::Global,
            }
        );
    }

    #[test]
    fn decodes_compare_and_swap_in_both_address_spaces() {
        assert_eq!(
            decode(0xeef0u64 << 48 | 7 << 20 | PT | 5 << 8 | 3).op,
            Op::Atom {
                dst: 3,
                addr: 5,
                offset: 0,
                src: 7,
                op: AtomOp::Cas,
                ty: AtomType::U32,
                space: AtomSpace::Global,
            }
        );
        assert_eq!(
            decode(0xee00u64 << 48 | 2 << 53 | 1 << 52 | 7 << 20 | PT | 5 << 8 | 3).op,
            Op::Atom {
                dst: 3,
                addr: 5,
                offset: 0,
                src: 7,
                op: AtomOp::Cas,
                ty: AtomType::U64,
                space: AtomSpace::Shared,
            }
        );
    }

    #[test]
    fn an_atomic_operation_this_decoder_has_no_name_for_is_not_invented() {
        // Slot 9 is unassigned in envydis's table; decoding it as one of its
        // neighbours would silently run the wrong reduction.
        assert!(matches!(
            decode(0xed00u64 << 48 | 9 << 52 | PT).op,
            Op::Unimplemented { .. }
        ));
    }

    /// `compiled::Compiled` holds these in a dense array on the strength of
    /// an `Op` being 32 bytes; a wider one halves what a cache line carries
    /// through the loop that runs once per covered pixel.
    #[test]
    fn an_op_still_fits_in_thirty_two_bytes() {
        assert_eq!(std::mem::size_of::<Op>(), 32);
    }

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
            Op::Ipa { dst: 0, offset: 0x7c, mul: None, perspective: false, sat: false, centroid: false }
        );
        // "mufu rcp $r3 $r0"
        assert_eq!(
            op(0x5080000000470003),
            Op::Mufu { dst: 3, src: 0, sm: FMod::NONE, op: MufuOp::Rcp, sat: false }
        );
        // "ipa $r0 a[0x80] $r3 0x0 0x1"
        assert_eq!(
            op(0xe043ff880037ff00),
            Op::Ipa { dst: 0, offset: 0x80, mul: Some(3), perspective: true, sat: false, centroid: false }
        );
    }

    /// Bits 52..54 are the sample mode, and Minecraft's fragment shaders
    /// open with `ipa.centroid` — refusing it dropped every draw of the
    /// title, first from the backend and then from the rasterizer it fell
    /// back to. `Offset` stays refused: it samples wherever a register
    /// points, which is a position neither renderer has.
    #[test]
    fn decodes_the_ipa_sample_modes() {
        assert_eq!(
            op(0xe013ff87cff7ff06),
            Op::Ipa { dst: 6, offset: 0x7c, mul: None, perspective: false, sat: false, centroid: true }
        );
        // The same instruction with sample mode 2 (offset) and 3.
        assert!(matches!(op(0xe023ff87cff7ff06), Op::Unimplemented { .. }));
        assert!(matches!(op(0xe033ff87cff7ff06), Op::Unimplemented { .. }));
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
                scale: FmulScale::None,
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
                scale: FmulScale::None,
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
                scale: FmulScale::None,
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
        // and checking the output against `texture.rgba * vColor.rgba`.
        // The destinations are REG_00 ($r0) and REG_28 ($r2), which take two
        // channels each; the coordinates are REG_08 ($r0) and REG_20 ($r1).
        assert_eq!(
            op(0xd8301a40_20170000),
            Op::Texs {
                dst: 0,
                dst2: 2,
                coords: [0, 1, RZ],
                handle: 0x1a4,
                dim: TexDim::T2d,
                mask: [true, true, true, true],
                f16: false,
            }
        );
        // The same word with the destinations four apart, which is where the
        // two readings of the pair diverge: rgba lands on $r4/$r5 then
        // $r2/$r3, never on $r6/$r7.
        assert_eq!(
            texs_destinations(4, 2, [true, true, true, true], false),
            vec![
                (4, TexsStore::Float(0)),
                (5, TexsStore::Float(1)),
                (2, TexsStore::Float(2)),
                (3, TexsStore::Float(3)),
            ]
        );
    }

    #[test]
    fn an_f16_texs_packs_two_channels_into_each_destination() {
        // Bit 59 halves the register count: rgba lands as two packed pairs
        // rather than four floats, which is what the `h*2` ops that read the
        // result back are expecting. Reading it as four floats is what drew
        // Asphalt 9's red car green.
        assert_eq!(
            texs_destinations(1, 0, [true, true, true, true], true),
            vec![(1, TexsStore::Halves(0, Some(1))), (0, TexsStore::Halves(2, Some(3)))]
        );
        // An odd count pads the unused half with zero rather than spilling
        // into another register.
        assert_eq!(
            texs_destinations(4, 6, [true, true, true, false], true),
            vec![(4, TexsStore::Halves(0, Some(1))), (6, TexsStore::Halves(2, None))]
        );
        assert_eq!(
            texs_destinations(4, RZ, [false, false, false, true], true),
            vec![(4, TexsStore::Halves(3, None))]
        );
    }

    #[test]
    fn the_precision_bit_is_decoded_and_its_polarity_is_backwards() {
        // `Precision` numbers F16 as 0 and F32 as 1, so a set bit is the
        // *unpacked* form. The captured fixture above has it set, which is
        // why it was right to read as four floats.
        assert!(matches!(op(0xd8301a40_20170000), Op::Texs { f16: false, .. }));
        assert!(matches!(op(0xd8301a40_20170000 & !(1 << 59)), Op::Texs { f16: true, .. }));
    }

    #[test]
    fn a_one_destination_texs_reads_the_single_and_double_channel_masks() {
        // Selector 0 is `rgb` when both destinations are present and plain
        // `r` when only one is — the case an alpha-atlas glyph fetch hits.
        assert_eq!(decode_tex_mask(0, 0, 2), Some([true, true, true, false]));
        assert_eq!(decode_tex_mask(0, 0, RZ), Some([true, false, false, false]));
        assert_eq!(decode_tex_mask(3, 0, RZ), Some([false, false, false, true]));
        assert_eq!(decode_tex_mask(7, 0, RZ), Some([false, false, true, true]));
        // Both destinations, but a selector past the four this decoder knows.
        assert_eq!(decode_tex_mask(5, 0, 2), None);
        // Nowhere to put the result at all.
        assert_eq!(decode_tex_mask(0, RZ, RZ), None);
    }

    #[test]
    fn a_two_channel_texs_fills_only_the_first_destination() {
        // `ga` into $r4: two channels, so $r2 is never touched.
        assert_eq!(
            texs_destinations(4, RZ, [false, true, false, true], false),
            vec![(4, TexsStore::Float(1)), (5, TexsStore::Float(3))]
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
        // 0 + 8 + 0x18 is 0x20, which is a `sched` word rather than an
        // instruction — so the target is the slot after it. See
        // [`super::super::align_slot`].
        assert_eq!(decode_at(asm(0xe290, &[(20, 24, 0x18)]), 0).op, Op::Ssy { target: 0x28 });
        // And one that lands on a real slot is left alone.
        assert_eq!(decode_at(asm(0xe290, &[(20, 24, 0x20)]), 0).op, Op::Ssy { target: 0x28 });
        assert_eq!(op(asm(0xf0f8, &[])), Op::Sync);
        assert_eq!(op(asm(0xe340, &[])), Op::Brk);
        assert_eq!(op(asm(0x50b0, &[])), Op::Nop);
    }

    /// The exact word Just Dance 2023's loading-screen vertex shader carries,
    /// two instructions before its `exit`. It has to decode to something, or
    /// the draw it belongs to never happens.
    #[test]
    fn a_vertex_stage_vote_is_a_nop() {
        assert_eq!(op(0x50e2_4321_1117_0000), Op::Nop);
        // The whole 0x50e0 group, not just that encoding: the three bits the
        // mask leaves free are the vote's operands, which nothing here reads.
        for low in 0..8u64 {
            assert_eq!(op(0x50e0_0000_0000_0000 | (low << 48)), Op::Nop);
        }
    }

    #[test]
    fn a_reconvergence_push_is_never_predicated() {
        // The bits every other instruction keeps its guard in belong to these
        // three's branch target, and read as zero — which is `@p0`, false
        // until something sets it. Decoded that way the push is skipped and
        // the `sync` that matched it finds an empty stack.
        for opcode in [0xe290u16, 0xe2a0, 0xe2b0] {
            let raw = asm(opcode, &[(20, 24, 0x20)]);
            assert!(decode_at(raw, 0).pred.is_always(), "opcode {opcode:#x}");
        }
        // `bra` in the same group *is* predicated, and keeps its guard.
        // `asm` writes PT into those bits, so clear them before setting p3.
        let bra = (asm(0xe240, &[(20, 24, 0x20)]) & !(0x7 << 16)) | (3 << 16);
        assert_eq!(decode_at(bra, 0).pred, Pred { reg: 3, negate: false });
    }

    #[test]
    fn decodes_integer_alu() {
        // iadd r0, r1, -r2
        assert_eq!(
            op(asm(0x5c10, &[(0, 8, 0), (8, 8, 1), (20, 8, 2), (48, 1, 1)])),
            Op::Iadd { dst: 0, a: 1, aneg: false, b: Operand::Reg(2), bneg: true, cin: false, cout: false }
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
                pred: None,
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

    /// The six half-precision encodings "A Short Hike" actually issues, taken
    /// from the words its own draws were skipped on. Between them they cover
    /// every operand form the group has except `hfma2`'s and the `32I`s.
    #[test]
    fn the_half_instructions_a_unity_shader_issues() {
        // hadd2.f32 $r12 $r12 $r17 — the swizzles and the merge are all F32,
        // which is a plain float add issued on the half unit. 110 of the 145
        // skipped draws stopped on one of these.
        assert_eq!(
            op(0x5d12800011170c0c),
            Op::Hadd2 {
                dst: 12,
                a: 12,
                am: FMod::NONE,
                asw: HSwizzle::F32,
                b: Operand::Reg(17),
                bm: FMod::NONE,
                bsw: HSwizzle::F32,
                merge: HMerge::F32,
                ftz: false,
                sat: false,
            }
        );
        // hmul2.f32 $r0 $r9 $r4
        assert_eq!(
            op(0x5d0a800010470900),
            Op::Hmul2 {
                dst: 0,
                a: 9,
                am: FMod::NONE,
                asw: HSwizzle::F32,
                b: Operand::Reg(4),
                bm: FMod::NONE,
                bsw: HSwizzle::F32,
                merge: HMerge::F32,
                prec: HPrecision::None,
                sat: false,
            }
        );
        // hmul2.f32 $r4 $r5 c1[0xc] — the constant form keeps `b`'s
        // modifiers up at 52..57 rather than below 32.
        assert_eq!(
            op(0x7882800400370504),
            Op::Hmul2 {
                dst: 4,
                a: 5,
                am: FMod::NONE,
                asw: HSwizzle::F32,
                b: Operand::Const { bank: 1, offset: 0xc },
                bm: FMod::NONE,
                bsw: HSwizzle::F32,
                merge: HMerge::F32,
                prec: HPrecision::None,
                sat: false,
            }
        );
        // hadd2 $r0 -$r9 (1.0, 1.0) — a "one minus" through the immediate
        // form, whose pair is two halves missing their low six mantissa bits.
        assert_eq!(
            op(0x7a02883c0f070900),
            Op::Hadd2 {
                dst: 0,
                a: 9,
                am: FMod { neg: true, abs: false },
                asw: HSwizzle::F32,
                b: Operand::Imm(0x3C00_3C00),
                bm: FMod::NONE,
                bsw: HSwizzle::H1H0,
                merge: HMerge::F32,
                ftz: false,
                sat: false,
            }
        );
        // hadd2 $r8.h0 -$rZ.h0_h0 c1[0x0] — the merging form, which keeps
        // the half of $r8 it does not write.
        assert_eq!(
            op(0x7a8508040007ff08),
            Op::Hadd2 {
                dst: 8,
                a: RZ,
                am: FMod { neg: true, abs: false },
                asw: HSwizzle::H0H0,
                b: Operand::Const { bank: 1, offset: 0 },
                bm: FMod::NONE,
                bsw: HSwizzle::F32,
                merge: HMerge::MrgH0,
                ftz: false,
                sat: false,
            }
        );
        // hsetp2.eq.and $p0 $pT $rZ.h0_h0 c3[0x140] $pT — the two
        // destinations are the two lanes, and the second is `PT`, which is
        // not writable.
        assert_eq!(
            op(0x7e85038c0507ff07),
            Op::Hsetp2 {
                p0: 0,
                p1: Pred::PT,
                a: RZ,
                am: FMod::NONE,
                asw: HSwizzle::H0H0,
                b: Operand::Const { bank: 3, offset: 0x140 },
                bm: FMod::NONE,
                bsw: HSwizzle::F32,
                cmp: FCmp::Eq,
                bop: BoolOp::And,
                src: Pred::ALWAYS,
                and: false,
                ftz: false,
            }
        );
    }

    /// The forms no capture covers yet, assembled from the field positions
    /// rather than observed — so a transcription slip in one shows up here
    /// rather than in a frame.
    #[test]
    fn every_half_operand_form_reaches_its_op() {
        // hfma2 $r1 $r2 $r3 $r4, register form.
        let hfma_reg = asm(0x5d00, &[(0, 8, 1), (8, 8, 2), (20, 8, 3), (39, 8, 4)]);
        assert!(matches!(
            op(hfma_reg),
            Op::Hfma2 { dst: 1, a: 2, b: Operand::Reg(3), c: Operand::Reg(4), .. }
        ));
        // hfma2 with the constant bank as `b` (`cr`) and as `c` (`rc`).
        let hfma_cr = asm(0x7080, &[(0, 8, 1), (8, 8, 2), (39, 8, 4), (20, 14, 3), (34, 5, 2)]);
        assert!(matches!(
            op(hfma_cr),
            Op::Hfma2 { b: Operand::Const { bank: 2, offset: 0xc }, c: Operand::Reg(4), .. }
        ));
        let hfma_rc = asm(0x6080, &[(0, 8, 1), (8, 8, 2), (39, 8, 4), (20, 14, 3), (34, 5, 2)]);
        assert!(matches!(
            op(hfma_rc),
            Op::Hfma2 { b: Operand::Reg(4), c: Operand::Const { bank: 2, offset: 0xc }, .. }
        ));
        // The `32I` forms take a whole 32-bit pair, and `hfma2`'s addend is
        // its own destination because the encoding has nowhere else to put it.
        assert!(matches!(
            op(asm(0x2c00, &[(0, 8, 1), (8, 8, 2), (20, 32, 0x3c00_3c00)])),
            Op::Hadd2 { dst: 1, a: 2, b: Operand::Imm(0x3c00_3c00), merge: HMerge::H1H0, .. }
        ));
        assert!(matches!(
            op(asm(0x2a00, &[(0, 8, 1), (8, 8, 2), (20, 32, 0x3c00_3c00)])),
            Op::Hmul2 { dst: 1, b: Operand::Imm(0x3c00_3c00), merge: HMerge::H1H0, .. }
        ));
        assert!(matches!(
            op(asm(0x2800, &[(0, 8, 1), (8, 8, 2), (20, 32, 0x3c00_3c00)])),
            Op::Hfma2 { dst: 1, c: Operand::Reg(1), merge: HMerge::H1H0, .. }
        ));
        // hset2, whose register form puts its comparison at 35 and its `.bf`
        // where every other form's merge sits.
        assert!(matches!(
            op(asm(0x5d18, &[(0, 8, 1), (8, 8, 2), (20, 8, 3), (35, 4, 4), (49, 1, 1)])),
            Op::Hset2 { dst: 1, a: 2, b: Operand::Reg(3), cmp: FCmp::Gt, bf: true, .. }
        ));
        assert!(matches!(
            op(asm(0x7c80, &[(0, 8, 1), (8, 8, 2), (49, 4, 4), (20, 14, 3), (34, 5, 2)])),
            Op::Hset2 { cmp: FCmp::Gt, b: Operand::Const { bank: 2, offset: 0xc }, .. }
        ));
        // hsetp2's `.h_and` collapses both lanes into one predicate.
        assert!(matches!(
            op(asm(0x5d20, &[(3, 3, 1), (0, 3, 2), (8, 8, 2), (20, 8, 3), (35, 4, 1), (49, 1, 1)])),
            Op::Hsetp2 { p0: 1, p1: 2, cmp: FCmp::Lt, and: true, .. }
        ));
    }

    /// An immediate half pair is two nine-bit fields with their signs stored
    /// a long way from the rest of them.
    #[test]
    fn an_immediate_half_pair_reassembles_both_signs() {
        // -1.0 in the low half (0xbc00) and +2.0 in the high (0x4000).
        let insn = asm(0x7a00, &[(20, 9, 0xf0), (29, 1, 1), (30, 9, 0x100), (56, 1, 0)]);
        assert_eq!(half_imm(insn), 0x4000_bc00);
    }
}

