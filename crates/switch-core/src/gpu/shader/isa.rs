//! Maxwell (GM20B) shader instruction decoding.
//!
//! Bit layouts are ported from envytools' `envydis` `gm107.c` tables. An
//! encoding (or modifier) that isn't modelled decodes to
//! [`Op::Unimplemented`] with its raw bits.

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

/// The right-hand operand of an ALU op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(u8),
    Const { bank: u8, offset: u16 },
    Imm(u32),
}

/// A guard or source predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pred {
    pub reg: u8,
    pub negate: bool,
}

impl Pred {
    pub const PT: u8 = 7;
    pub const ALWAYS: Pred = Pred {
        reg: Pred::PT,
        negate: false,
    };
    pub const NEVER: Pred = Pred {
        reg: Pred::PT,
        negate: true,
    };

    pub fn is_always(self) -> bool {
        self.reg == Pred::PT && !self.negate
    }
}

/// A float source's sign/magnitude modifiers, applied in that order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FMod {
    pub neg: bool,
    pub abs: bool,
}

impl FMod {
    pub const NONE: FMod = FMod {
        neg: false,
        abs: false,
    };

    pub fn apply(self, v: f32) -> f32 {
        let v = if self.abs { v.abs() } else { v };
        if self.neg {
            -v
        } else {
            v
        }
    }
}

/// A float comparison (`tab5bb0_0`).
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

/// How a `set`/`setp` combines its comparison with its source predicate (`tab5bb0_1`).
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
    /// D3D9's rule.
    Fmz,
}

impl HPrecision {
    /// The fourth encoding is hardware's "don't care", which is free to be the plain mode.
    fn decode(bits: u64) -> HPrecision {
        match bits {
            1 => HPrecision::Ftz,
            2 => HPrecision::Fmz,
            _ => HPrecision::None,
        }
    }

    /// Whether a product one of whose operands is zero answers zero whatever the other one is.
    pub fn zeroes_products(self, sat: bool) -> bool {
        self == HPrecision::Fmz && !sat
    }
}

/// What `lop`'s test form asks of the result before it writes its predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LopTest {
    /// `.T`: the predicate is set unconditionally.
    True,
    /// `.Z`: set when the result is zero.
    Zero,
    /// `.NZ`: set when any bit of the result is set.
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

/// What an `xmad` adds its product to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XmadC {
    Full,
    Lo,
    Hi,
    Bcc,
}

/// `texs`'s sample dimensionality (envydis's `d000_1`/`d200_1` tables).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexDim {
    T1d,
    T2d,
    /// A 2D array.
    T2dArray,
    T3d,
    TCube,
    /// An array of cubemaps.
    TCubeArray,
}

/// `bar`'s sub-operation (`tabf0a8_0`).
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShflMode {
    Idx,
    Up,
    Down,
    Bfly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceDim {
    D1,
    Buffer1d,
    Array1d,
    D2,
    Array2d,
    D3,
}

/// How much a raw (`.D`) surface access moves, and whether a narrow load is sign-extended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceSize {
    U8,
    S8,
    U16,
    S16,
    B32,
    B64,
    B128,
}

impl SurfaceSize {
    pub fn bytes(self) -> u32 {
        match self {
            SurfaceSize::U8 | SurfaceSize::S8 => 1,
            SurfaceSize::U16 | SurfaceSize::S16 => 2,
            SurfaceSize::B32 => 4,
            SurfaceSize::B64 => 8,
            SurfaceSize::B128 => 16,
        }
    }

    /// Registers the access fills or drains, one per 32 bits and at least one.
    pub fn words(self) -> usize {
        (self.bytes() as usize).div_ceil(4)
    }
}

/// What a surface instruction moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceData {
    /// `.P`.
    Formatted([bool; 4]),
    /// `.D`: the texel's bytes, uninterpreted.
    Raw(SurfaceSize),
}

impl SurfaceData {
    /// The channels a load writes, in the order its registers take them.
    pub fn channels(self) -> [bool; 4] {
        match self {
            SurfaceData::Formatted(mask) => mask,
            SurfaceData::Raw(size) => std::array::from_fn(|i| i < size.words()),
        }
    }
}

/// What a `vote` asks of its warp's predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteMode {
    All,
    Any,
    Eq,
}

/// Which address space an atomic addresses.
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
    /// `safeadd`.
    SafeAdd,
}

/// How an atomic interprets the memory it operates on (`tabed00sz`/`tabec00sz`/`tabebf8sz`).
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
        Instruction {
            pred: Pred::ALWAYS,
            op,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    // ---- attribute space ----
    /// `ld.<size> dst, a[offset]`, attribute-space load.
    Ld {
        dst: u8,
        offset: u16,
        idx: u8,
        size: MemSize,
    },
    /// `st.<size> a[offset], src`, attribute-space store.
    St {
        offset: u16,
        idx: u8,
        src: u8,
        size: MemSize,
    },
    /// `ipa[.pass][.centroid] dst, a[offset], mul`, fixed-function interpolation.
    Ipa {
        dst: u8,
        offset: u16,
        mul: Option<u8>,
        perspective: bool,
        sat: bool,
        centroid: bool,
    },

    // ---- float ALU ----
    /// `rro dst, src`.
    Rro {
        dst: u8,
        src: Operand,
        sm: FMod,
    },
    Fadd {
        dst: u8,
        a: u8,
        am: FMod,
        b: Operand,
        bm: FMod,
        ftz: bool,
        sat: bool,
    },
    Fmul {
        dst: u8,
        a: u8,
        b: Operand,
        bm: FMod,
        ftz: bool,
        sat: bool,
        scale: FmulScale,
    },
    Ffma {
        dst: u8,
        a: u8,
        b: Operand,
        bneg: bool,
        c: Operand,
        cneg: bool,
        ftz: bool,
        sat: bool,
    },
    Fmnmx {
        dst: u8,
        a: u8,
        am: FMod,
        b: Operand,
        bm: FMod,
        pred: Pred,
        ftz: bool,
    },
    Fsetp {
        p0: u8,
        p1: u8,
        a: u8,
        am: FMod,
        b: Operand,
        bm: FMod,
        cmp: FCmp,
        bop: BoolOp,
        src: Pred,
    },
    Fset {
        dst: u8,
        a: u8,
        am: FMod,
        b: Operand,
        bm: FMod,
        cmp: FCmp,
        bop: BoolOp,
        src: Pred,
        bf: bool,
    },
    Mufu {
        dst: u8,
        src: u8,
        sm: FMod,
        op: MufuOp,
        sat: bool,
    },

    // ---- half-precision ALU ----
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
    Iadd {
        dst: u8,
        a: u8,
        aneg: bool,
        b: Operand,
        bneg: bool,
        cin: bool,
        cout: bool,
    },
    Iadd3 {
        dst: u8,
        a: u8,
        aneg: bool,
        b: Operand,
        bneg: bool,
        c: Operand,
        cneg: bool,
    },
    Imnmx {
        dst: u8,
        a: u8,
        b: Operand,
        pred: Pred,
        signed: bool,
    },
    /// `vmnmx dst, a, b, c`.
    Vmnmx {
        dst: u8,
        a: u8,
        b: u8,
        c: u8,
        /// Whether the first operation is a maximum (`.MX`).
        max: bool,
        /// Whether the second is a maximum rather than a minimum.
        then_max: bool,
        /// Whether the first operation compares signed, which both operands agree on.
        signed: bool,
        /// Whether the second operation compares signed.
        then_signed: bool,
    },
    Iscadd {
        dst: u8,
        a: u8,
        aneg: bool,
        b: Operand,
        bneg: bool,
        shift: u8,
    },
    Isetp {
        p0: u8,
        p1: u8,
        a: u8,
        b: Operand,
        cmp: ICmp,
        signed: bool,
        bop: BoolOp,
        src: Pred,
    },
    Iset {
        dst: u8,
        a: u8,
        b: Operand,
        cmp: ICmp,
        signed: bool,
        bop: BoolOp,
        src: Pred,
        bf: bool,
    },
    Icmp {
        dst: u8,
        a: u8,
        b: Operand,
        c: u8,
        cmp: ICmp,
        signed: bool,
    },
    /// `bfi dst, insert, src, base`.
    Bfi {
        dst: u8,
        insert: u8,
        src: Operand,
        base: Operand,
    },
    /// `r2p pr, src, mask`.
    R2p {
        src: u8,
        mask: Operand,
        byte: u8,
    },
    Imul {
        dst: u8,
        a: u8,
        b: Operand,
        signed: bool,
        hi: bool,
    },
    Xmad {
        dst: u8,
        a: u8,
        ah: bool,
        asigned: bool,
        b: Operand,
        bh: bool,
        bsigned: bool,
        c: Operand,
        cmode: XmadC,
        psl: bool,
        mrg: bool,
    },
    Lop {
        dst: u8,
        a: u8,
        ainv: bool,
        b: Operand,
        binv: bool,
        op: LogicOp,
        pred: Option<(u8, LopTest)>,
    },
    Lop3 {
        dst: u8,
        a: u8,
        b: Operand,
        c: Operand,
        lut: u8,
    },
    Shl {
        dst: u8,
        a: u8,
        b: Operand,
        wrap: bool,
    },
    Shr {
        dst: u8,
        a: u8,
        b: Operand,
        signed: bool,
        wrap: bool,
    },
    Shf {
        dst: u8,
        lo: u8,
        shift: Operand,
        hi: u8,
        left: bool,
        wrap: bool,
        hi_out: bool,
    },
    Bfe {
        dst: u8,
        a: u8,
        b: Operand,
        signed: bool,
    },
    Popc {
        dst: u8,
        b: Operand,
        inv: bool,
    },
    Flo {
        dst: u8,
        b: Operand,
        signed: bool,
        shift: bool,
        inv: bool,
    },
    Sel {
        dst: u8,
        a: u8,
        b: Operand,
        pred: Pred,
    },

    // ---- conversions ----
    /// Integer -> float.
    I2f {
        dst: u8,
        src: Operand,
        sm: FMod,
        src_bytes: u8,
        src_signed: bool,
        sel: u8,
    },
    /// Float -> integer, with an explicit rounding mode.
    F2i {
        dst: u8,
        src: Operand,
        sm: FMod,
        dst_bytes: u8,
        dst_signed: bool,
        round: FRound,
        ftz: bool,
    },
    /// Float -> float.
    F2f {
        dst: u8,
        src: Operand,
        sm: FMod,
        round: Option<FRound>,
        sat: bool,
        ftz: bool,
        /// Source and destination widths, 16 or 32.
        src_bits: u8,
        dst_bits: u8,
        /// Which half a 16-bit source is read from. Meaningless at 32 bits.
        hi: bool,
    },
    /// Integer -> integer: a width conversion, optionally saturating.
    I2i {
        dst: u8,
        src: Operand,
        sm: FMod,
        src_bytes: u8,
        src_signed: bool,
        dst_signed: bool,
        sat: bool,
        sel: u8,
        /// `.CC`.
        cc: bool,
    },

    // ---- moves ----
    Mov {
        dst: u8,
        src: Operand,
    },
    Mov32i {
        dst: u8,
        imm: u32,
    },
    /// `mov dst, sN`, a special register (`tid`, `laneid`, ...).
    S2r {
        dst: u8,
        sr: u8,
    },
    Psetp {
        p0: u8,
        p1: u8,
        a: Pred,
        b: Pred,
        c: Pred,
        op1: BoolOp,
        op2: BoolOp,
    },
    /// `csetp.<test>.<op> p0, p1, cc, src`.
    Csetp {
        p0: u8,
        p1: u8,
        test: u8,
        src: Pred,
        op: BoolOp,
    },

    // ---- memory ----
    /// `ld cN[idx + offset]`, a constant-buffer load into registers.
    Ldc {
        dst: u8,
        bank: u8,
        offset: i32,
        idx: u8,
        size: MemSize,
    },
    /// `ldg dst, [addr + offset]`, a global load.
    Ldg {
        dst: u8,
        addr: u8,
        offset: i32,
        size: MemSize,
    },
    /// `stg [addr + offset], src`, a global store.
    Stg {
        addr: u8,
        offset: i32,
        src: u8,
        size: MemSize,
    },
    /// `ld dst, l[addr + offset]`, a local (per-thread scratch) load.
    Ldl {
        dst: u8,
        addr: u8,
        offset: i32,
        size: MemSize,
    },
    /// `st l[addr + offset], src`.
    Stl {
        addr: u8,
        offset: i32,
        src: u8,
        size: MemSize,
    },
    /// `ld dst, s[addr + offset]`, a load from the CTA's shared memory.
    Lds {
        dst: u8,
        addr: u8,
        offset: i32,
        size: MemSize,
    },
    /// `st s[addr + offset], src`.
    Sts {
        addr: u8,
        offset: i32,
        src: u8,
        size: MemSize,
    },
    /// `atom`/`atoms`/`red`.
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
    /// `texs dst, coords.., handle, dim, mask`, texture sample with an immediate handle.
    Texs {
        dst: u8,
        dst2: u8,
        coords: [u8; 3],
        /// A shadow sample's reference register.
        dref: Option<u8>,
        handle: u16,
        dim: TexDim,
        mask: [bool; 4],
        f16: bool,
    },

    /// `tex dst, coords.., handle, dim, mask`, the general texture sample.
    Tex {
        dst: u8,
        /// The coordinate registers, `dim` of them.
        coords: [u8; 3],
        layer: Option<u8>,
        /// A shadow sample's reference (`.DC`).
        dref: Option<u8>,
        /// `.AOFFI`'s register.
        offset: Option<u8>,
        /// The register the `.LL`/`.LB` modes take their level or bias from.
        lod: Option<u8>,
        handle: u16,
        /// A bindless sample's (`tex.b`) handle register.
        handle_reg: Option<u8>,
        dim: TexDim,
        mask: [bool; 4],
    },
    /// `txq dst, lod, dimension, handle, mask`.
    Txq {
        dst: u8,
        lod: u8,
        handle: u16,
        mask: [bool; 4],
    },
    /// `tld4.<component> dst, coords.., handle, dim, mask`.
    Tld4 {
        dst: u8,
        coords: [u8; 3],
        layer: Option<u8>,
        /// `.AOFFI`'s register, as in [`Op::Tex`].
        offset: Option<u8>,
        handle: u16,
        dim: TexDim,
        /// Which channel, after the descriptor's swizzle.
        component: u8,
        mask: [bool; 4],
    },

    // ---- warp ----
    /// `shfl.<mode> p, dst, src, index, mask`.
    Shfl {
        dst: u8,
        pred: u8,
        src: u8,
        index: Operand,
        mask: Operand,
        mode: ShflMode,
    },
    /// `suld dst, [coords], handle`.
    Suld {
        dst: u8,
        /// The first coordinate register; [`SurfaceDim`] says how many follow.
        coords: u8,
        /// A dword index into the texture bank, as `tex`'s is, when `handle_reg` is `None`.
        handle: u16,
        handle_reg: Option<u8>,
        dim: SurfaceDim,
        data: SurfaceData,
    },
    /// `sust [coords], src, handle`.
    Sust {
        src: u8,
        coords: u8,
        handle: u16,
        handle_reg: Option<u8>,
        dim: SurfaceDim,
        data: SurfaceData,
    },
    /// `vote.mode dst, pred, src`.
    Vote {
        dst: u8,
        pred: u8,
        src: Pred,
        mode: VoteMode,
    },
    /// `fswzadd dst, a, b, swizzle`.
    Fswzadd {
        dst: u8,
        a: u8,
        b: u8,
        swizzle: u8,
        ftz: bool,
    },

    // ---- control ----
    /// `bra target`.
    Bra {
        target: u32,
    },
    /// `ssy target`.
    Brx {
        base: u32,
        reg: u8,
    },
    Ssy {
        target: u32,
    },
    /// `sync`, pop one and jump there.
    Sync,
    /// `pbk target`: push a loop-break point.
    Pbk {
        target: u32,
    },
    Brk,
    /// `pcnt target`: push a loop-continue point.
    Pcnt {
        target: u32,
    },
    Cont,
    Exit,
    /// `kil`, discard this fragment.
    Kil,
    /// `bar.<mode>`, a CTA-wide barrier.
    Bar {
        mode: BarMode,
    },
    Nop,
    Inert,

    /// A bit pattern this decoder doesn't recognise, or recognises but with an unhandled modifier.
    Unimplemented {
        raw: u64,
    },
}

/// `suld`/`sust`, fields as Eden's `surface_load_store.cpp` reads them.
fn decode_surface(insn: u64) -> Option<Op> {
    const IGN: u64 = 0;
    if field(insn, 49, 2) != IGN {
        return None;
    }
    let raw = field(insn, 52, 1) != 0;
    if raw && field(insn, 23, 1) != 0 {
        return None;
    }
    let dim = match field(insn, 33, 3) {
        0 => SurfaceDim::D1,
        1 => SurfaceDim::Buffer1d,
        2 => SurfaceDim::Array1d,
        3 => SurfaceDim::D2,
        4 => SurfaceDim::Array2d,
        5 => SurfaceDim::D3,
        _ => return None,
    };
    let data = if raw {
        SurfaceData::Raw(match field(insn, 20, 3) {
            0 => SurfaceSize::U8,
            1 => SurfaceSize::S8,
            2 => SurfaceSize::U16,
            3 => SurfaceSize::S16,
            4 => SurfaceSize::B32,
            5 => SurfaceSize::B64,
            6 => SurfaceSize::B128,
            _ => return None,
        })
    } else {
        let swizzle = field(insn, 20, 4);
        SurfaceData::Formatted(std::array::from_fn(|i| swizzle >> i & 1 != 0))
    };
    let store = field(insn, 53, 1) != 0;
    match data {
        SurfaceData::Formatted(mask) if store && mask != [true; 4] => return None,
        SurfaceData::Formatted([false, false, false, false]) => return None,
        _ => {}
    }
    let bound = field(insn, 51, 1) != 0;
    let (handle, handle_reg) = if bound {
        (field(insn, 36, 13) as u16, None)
    } else {
        (0, Some(reg(insn, 39, 8)))
    };
    let coords = reg(insn, 8, 8);
    Some(if store {
        Op::Sust {
            src: reg(insn, 0, 8),
            coords,
            handle,
            handle_reg,
            dim,
            data,
        }
    } else {
        Op::Suld {
            dst: reg(insn, 0, 8),
            coords,
            handle,
            handle_reg,
            dim,
            data,
        }
    })
}

fn field(insn: u64, pos: u32, len: u32) -> u64 {
    (insn >> pos) & ((1u64 << len) - 1)
}

/// Whether a control-flow instruction's condition-code test (bits 0..5) can ever be true.
fn flow_test_can_hold(insn: u64) -> bool {
    const NEVER: u64 = 0;
    const FCSM_TR: u64 = 28;
    !matches!(field(insn, 0, 5), NEVER | FCSM_TR)
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

/// The guard predicate every instruction carries.
fn guard(insn: u64) -> Pred {
    Pred {
        reg: reg(insn, 16, 3),
        negate: field(insn, 19, 1) != 0,
    }
}

/// A source predicate at `[pos, pos+3)` with its negate flag at `not`.
fn src_pred(insn: u64, pos: u32, not: u32) -> Pred {
    Pred {
        reg: reg(insn, pos, 3),
        negate: field(insn, not, 1) != 0,
    }
}

/// `C34_RZ_O14_20`.
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

pub trait FlowLogic<T> {
    fn constant(&self, value: bool) -> T;
    fn not(&self, a: T) -> T;
    fn and(&self, a: T, b: T) -> T;
    fn or(&self, a: T, b: T) -> T;
    fn xor(&self, a: T, b: T) -> T;
}

pub fn flow_test<T: Clone>(test: u8, [z, s, c, o]: [T; 4], l: &impl FlowLogic<T>) -> Option<T> {
    Some(match test {
        0 => l.constant(false),                    // F
        1 => l.xor(l.and(s.clone(), l.not(z)), o), // LT
        2 => l.and(l.not(s), z),                   // EQ
        3 => l.xor(s, l.or(z, o)),                 // LE
        4 => l.and(l.xor(l.not(s), o), l.not(z)),  // GT
        5 => l.not(z),                             // NE
        6 => l.not(l.xor(s, o)),                   // GE
        7 => l.or(l.not(s), l.not(z)),             // NUM
        8 => l.and(s, z),                          // NaN
        9 => l.xor(s, o),                          // LTU
        10 => z,                                   // EQU
        11 => l.or(l.xor(s, o), z),                // LEU
        12 => l.xor(l.not(s), l.or(z, o)),         // GTU
        13 => l.or(s, l.not(z)),                   // NEU
        14 => l.xor(l.or(l.not(s), z), o),         // GEU
        15 => l.constant(true),                    // T
        16 => l.not(o),                            // OFF
        17 => l.not(c),                            // LO
        18 => l.not(s),                            // SFF
        19 => l.or(z, l.not(c)),                   // LS
        20 => l.and(c, l.not(z)),                  // HI
        21 => s,                                   // SFT
        22 => c,                                   // HS
        23 => o,                                   // OFT
        30 => l.or(s, z),                          // RLE
        31 => l.and(l.not(s), l.not(z)),           // RGT
        _ => return None,
    })
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

/// The `FloatFormat` a conversion names, in bits.
fn float_width(bits: u64) -> Option<u8> {
    match bits {
        1 => Some(16),
        2 => Some(32),
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

fn decode_alu(insn: u64) -> Op {
    let un = Op::Unimplemented { raw: insn };

    // The three operand forms of a "normal" ALU op share a sub-opcode.
    let form = insn >> 48;
    let (rhs_int, rhs_float) = match form >> 8 {
        0x5c => (
            Operand::Reg(reg(insn, 20, 8)),
            Operand::Reg(reg(insn, 20, 8)),
        ),
        0x4c => (const_operand(insn), const_operand(insn)),
        0x38 | 0x39 => (Operand::Imm(imm20(insn)), Operand::Imm(imm20f(insn))),
        _ => return decode_alu_wide(insn),
    };
    let rhs_int = Some(rhs_int);
    let rhs_float = Some(rhs_float);
    let sub = form & 0x00f8;

    match sub {
        // ---- float ----
        // fadd: ftz 44, sat 50, a: neg 48/abs 46, b: neg 45/abs 49.
        0x58 => {
            let Some(b) = rhs_float else { return un };
            Op::Fadd {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                am: FMod {
                    neg: field(insn, 48, 1) != 0,
                    abs: field(insn, 46, 1) != 0,
                },
                b,
                bm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                ftz: field(insn, 44, 1) != 0,
                sat: field(insn, 50, 1) != 0,
            }
        }
        // fmul: ftz/fmz at 44..46, scale at 41..44, sat 50, b: neg 48.
        0x68 => {
            let Some(b) = rhs_float else { return un };
            let Some(scale) = FmulScale::decode(field(insn, 41, 3)) else {
                return un;
            };
            Op::Fmul {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b,
                bm: FMod {
                    neg: field(insn, 48, 1) != 0,
                    abs: false,
                },
                ftz: field(insn, 44, 2) == 1,
                sat: field(insn, 50, 1) != 0,
                scale,
            }
        }
        // fmnmx: ftz 44, a: neg 48/abs 46, b: neg 45/abs 49, pred at 39.
        0x60 => {
            let Some(b) = rhs_float else { return un };
            Op::Fmnmx {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                am: FMod {
                    neg: field(insn, 48, 1) != 0,
                    abs: field(insn, 46, 1) != 0,
                },
                b,
                bm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                pred: src_pred(insn, 39, 42),
                ftz: field(insn, 44, 1) != 0,
            }
        }
        // r2p.
        0xf0 => {
            let Some(mask) = rhs_int else { return un };
            if field(insn, 40, 1) != 0 {
                return un; // the CC form
            }
            Op::R2p {
                src: reg(insn, 8, 8),
                mask,
                byte: field(insn, 41, 2) as u8,
            }
        }
        // ---- integer ----
        // iadd: sat 50, x 43, a: neg 49, b: neg 48.
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
        // iscadd, shift at 39..44.
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
        // imnmx, signed 48, pred at 39.
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
        // shr, signed 48, wrap 39, brev 40, x 44.
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
        // flo, signed 48, shift 41, inv 40.
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
        // imul, hi 39, signedness at 41 (a) and 40 (b) in tab5c38_0/1.
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
        // lop, op at 41..43, inv 39 (a) / 40 (b), x 43.
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
        // shl, wrap 39, x 43.
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
        // bfe, signed 48, brev 40.
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
        // popc, inv 40.
        0x08 => {
            let Some(b) = rhs_int else { return un };
            Op::Popc {
                dst: reg(insn, 0, 8),
                b,
                inv: field(insn, 40, 1) != 0,
            }
        }
        // ---- moves and selects ----
        // mov: the 4-bit byte-enable mask at 39..43 must be "all".
        0x98 => {
            let Some(src) = rhs_int else { return un };
            if field(insn, 39, 4) != 0xf {
                return un;
            }
            Op::Mov {
                dst: reg(insn, 0, 8),
                src,
            }
        }
        // rro, the range-reduction operator that precedes `mufu`.
        0x90 => {
            let Some(src) = rhs_float else { return un };
            if field(insn, 50, 1) != 0 {
                return un;
            }
            Op::Rro {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
            }
        }
        // sel, pred at 39.
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
        // i2f.
        0xb8 => {
            let Some(src) = rhs_int else { return un };
            if field(insn, 8, 2) != 2 {
                return un; // only f32 destinations
            }
            let bits = field(insn, 10, 2) | (field(insn, 13, 1) << 2);
            let Some((src_bytes, src_signed)) = int_type(bits) else {
                return un;
            };
            Op::I2f {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                src_bytes,
                src_signed,
                sel: field(insn, 41, 2) as u8,
            }
        }
        // f2i.
        0xb0 => {
            let Some(src) = rhs_float else { return un };
            if field(insn, 10, 2) != 2 {
                return un; // only f32 sources
            }
            let bits = field(insn, 8, 2) | (field(insn, 12, 1) << 2);
            let Some((dst_bytes, dst_signed)) = int_type(bits) else {
                return un;
            };
            Op::F2i {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                dst_bytes,
                dst_signed,
                round: fround(field(insn, 39, 2)),
                ftz: field(insn, 44, 1) != 0,
            }
        }
        // f2f, f16 and f32 in either direction, and the rounding a same-width conversion names.
        0xa8 => {
            let (Some(dst_bits), Some(src_bits)) = (
                float_width(field(insn, 8, 2)),
                float_width(field(insn, 10, 2)),
            ) else {
                return un; // f64, which nothing here models
            };
            let hi = field(insn, 41, 1) != 0;
            let src = if src_bits == 16 {
                match rhs_int {
                    Some(Operand::Imm(v)) => Operand::Imm((v & 0xffff) | (v << 16)),
                    Some(other) => other,
                    None => return un,
                }
            } else {
                let Some(src) = rhs_float else { return un };
                src
            };
            let round = if src_bits == dst_bits {
                // `RoundingOp` is bits 39, 40 and 42.
                match field(insn, 39, 2) | (field(insn, 42, 1) << 3) {
                    0 | 3 => None,
                    8..=11 => Some(fround(field(insn, 39, 2))),
                    _ => return un,
                }
            } else {
                if field(insn, 39, 2) != 0 {
                    return un;
                }
                None
            };
            Op::F2f {
                dst: reg(insn, 0, 8),
                src,
                sm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                round,
                sat: field(insn, 50, 1) != 0,
                ftz: field(insn, 44, 1) != 0,
                src_bits,
                dst_bits,
                hi,
            }
        }
        // i2i, src type in tab5ce0_1, dst type in tab5ce0_0.
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
                sm: FMod {
                    neg: field(insn, 45, 1) != 0,
                    abs: field(insn, 49, 1) != 0,
                },
                src_bytes,
                src_signed,
                dst_signed,
                sat: field(insn, 50, 1) != 0,
                sel: field(insn, 41, 2) as u8,
                cc: field(insn, 47, 1) != 0,
            }
        }
        _ => decode_alu_wide(insn),
    }
}

/// The ops whose opcode field is wider or narrower than the 0xfff8 group [`decode_alu`] handles.
fn decode_alu_wide(insn: u64) -> Op {
    let un = Op::Unimplemented { raw: insn };
    let form = insn >> 48;

    // The half-precision group first.
    if let Some(op) = decode_half(insn) {
        return op;
    }

    // ---- 0xfff0-masked: fsetp/isetp/iset/icmp/prmt/lop3/bfi ----
    // Each immediate form is listed twice, as `0x36…`/`0x38…` and one above.
    match form >> 4 {
        // fsetp, cmp 48..52, ftz 47, bop 45..47.
        0x5bb | 0x4bb | 0x36b | 0x37b => {
            let b = match form >> 12 {
                0x5 => Operand::Reg(reg(insn, 20, 8)),
                0x4 => const_operand(insn),
                _ => Operand::Imm(imm20f(insn)),
            };
            let Some(bop) = bool_op(field(insn, 45, 2)) else {
                return un;
            };
            return Op::Fsetp {
                p0: reg(insn, 3, 3),
                p1: reg(insn, 0, 3),
                a: reg(insn, 8, 8),
                am: FMod {
                    neg: field(insn, 43, 1) != 0,
                    abs: field(insn, 7, 1) != 0,
                },
                b,
                bm: FMod {
                    neg: field(insn, 6, 1) != 0,
                    abs: field(insn, 44, 1) != 0,
                },
                cmp: fcmp(field(insn, 48, 4)),
                bop,
                src: src_pred(insn, 39, 42),
            };
        }
        // isetp, cmp 49..52, signed 48, bop 45..47, x 43.
        0x5b6 | 0x4b6 | 0x366 | 0x376 => {
            let b = match form >> 12 {
                0x5 => Operand::Reg(reg(insn, 20, 8)),
                0x4 => const_operand(insn),
                _ => Operand::Imm(imm20(insn)),
            };
            let Some(bop) = bool_op(field(insn, 45, 2)) else {
                return un;
            };
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
        // iset, the register-writing form of isetp.
        0x5b5 | 0x4b5 | 0x365 | 0x375 => {
            let b = match form >> 12 {
                0x5 => Operand::Reg(reg(insn, 20, 8)),
                0x4 => const_operand(insn),
                _ => Operand::Imm(imm20(insn)),
            };
            let Some(bop) = bool_op(field(insn, 45, 2)) else {
                return un;
            };
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
        // icmp.
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
        // bfi.
        0x5bf | 0x4bf | 0x53f | 0x36f | 0x37f => {
            let (src, base) = match form >> 4 {
                0x5bf => (
                    Operand::Reg(reg(insn, 20, 8)),
                    Operand::Reg(reg(insn, 39, 8)),
                ),
                0x4bf => (const_operand(insn), Operand::Reg(reg(insn, 39, 8))),
                0x53f => (Operand::Reg(reg(insn, 39, 8)), const_operand(insn)),
                _ => (Operand::Imm(imm20(insn)), Operand::Reg(reg(insn, 39, 8))),
            };
            return Op::Bfi {
                dst: reg(insn, 0, 8),
                insert: reg(insn, 8, 8),
                src,
                base,
            };
        }
        // iadd3: three-way add, negation per source.
        0x5cc | 0x4cc | 0x38c | 0x39c => {
            let (b, c) = match form >> 12 {
                0x5 => (
                    Operand::Reg(reg(insn, 20, 8)),
                    Operand::Reg(reg(insn, 39, 8)),
                ),
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
        // psetp, a pure predicate op.
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

    // csetp, 0x50a0/0xfff8: Eden's `CSETP`.
    if insn & 0xfff8_0000_0000_0000 == 0x50a0_0000_0000_0000 {
        let Some(op) = bool_op(field(insn, 45, 2)) else {
            return un;
        };
        return Op::Csetp {
            p0: reg(insn, 3, 3),
            p1: reg(insn, 0, 3),
            test: field(insn, 8, 5) as u8,
            src: src_pred(insn, 39, 42),
            op,
        };
    }

    // nop, 0x50b0/0xfff8.
    if insn & 0xfff8_0000_0000_0000 == 0x50b0_0000_0000_0000 {
        return Op::Nop;
    }

    // vote.vtg.
    if insn & 0xfff8_0000_0000_0000 == 0x50e0_0000_0000_0000 {
        return Op::Nop;
    }

    // vote.
    if insn & 0xfff8_0000_0000_0000 == 0x50d8_0000_0000_0000 {
        let mode = match field(insn, 48, 2) {
            0 => VoteMode::All,
            1 => VoteMode::Any,
            2 => VoteMode::Eq,
            _ => return Op::Unimplemented { raw: insn },
        };
        return Op::Vote {
            dst: reg(insn, 0, 8),
            pred: reg(insn, 45, 3),
            src: src_pred(insn, 39, 42),
            mode,
        };
    }

    // fswzadd.
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

    // mufu: subop at 20..24, sat 50, src: neg 48 / abs 46.
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
            sm: FMod {
                neg: field(insn, 48, 1) != 0,
                abs: field(insn, 46, 1) != 0,
            },
            op: mufu,
            sat: field(insn, 50, 1) != 0,
        };
    }

    // lop3: the LUT byte sits in a different field in each form.
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
    // vmnmx, 0x3a00/0xfe00.
    if insn & 0xfe00_0000_0000_0000 == 0x3a00_0000_0000_0000 {
        const WORD: u64 = 3;
        const MIN: u64 = 5;
        const MAX: u64 = 6;
        let then = field(insn, 51, 3);
        let whole_words = field(insn, 37, 2) == WORD && field(insn, 29, 2) == WORD;
        let signed = field(insn, 48, 1) != 0;
        if !whole_words
            || field(insn, 50, 1) == 0
            || signed != (field(insn, 49, 1) != 0)
            || field(insn, 47, 1) != 0
            || field(insn, 55, 1) != 0
            || !matches!(then, MIN | MAX)
        {
            return Op::Unimplemented { raw: insn };
        }
        return Op::Vmnmx {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b: reg(insn, 20, 8),
            c: reg(insn, 39, 8),
            max: field(insn, 56, 1) != 0,
            then_max: then == MAX,
            signed,
            then_signed: field(insn, 54, 1) != 0,
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

    // ffma, three operand orders across four opcodes.
    if insn & 0xff80_0000_0000_0000 == 0x5980_0000_0000_0000 {
        return decode_ffma(
            insn,
            Operand::Reg(reg(insn, 20, 8)),
            Operand::Reg(reg(insn, 39, 8)),
        );
    }
    if insn & 0xff80_0000_0000_0000 == 0x4980_0000_0000_0000 {
        return decode_ffma(insn, const_operand(insn), Operand::Reg(reg(insn, 39, 8)));
    }
    if insn & 0xff80_0000_0000_0000 == 0x5180_0000_0000_0000 {
        // The register/constant operands are the other way round here.
        return decode_ffma(insn, Operand::Reg(reg(insn, 39, 8)), const_operand(insn));
    }
    if insn & 0xfe80_0000_0000_0000 == 0x3280_0000_0000_0000 {
        return decode_ffma(
            insn,
            Operand::Imm(imm20f(insn)),
            Operand::Reg(reg(insn, 39, 8)),
        );
    }

    // xmad, 16x16 multiply-accumulate, in each of its four operand forms.
    if insn & 0xffc0_0000_0000_0000 == 0x5b00_0000_0000_0000 {
        return decode_xmad(
            insn,
            XmadForm {
                b: Operand::Reg(reg(insn, 20, 8)),
                c: Operand::Reg(reg(insn, 39, 8)),
                bh: field(insn, 35, 1) != 0,
                mode: field(insn, 50, 3),
                x: field(insn, 38, 1) != 0,
                psl: field(insn, 36, 1) != 0,
                mrg: field(insn, 37, 1) != 0,
            },
        );
    }
    // The `rc` form multiplies by the register and adds the bank; `cr` is the other way round.
    if insn & 0xff80_0000_0000_0000 == 0x5100_0000_0000_0000 {
        return decode_xmad(
            insn,
            XmadForm {
                b: Operand::Reg(reg(insn, 39, 8)),
                c: const_operand(insn),
                bh: field(insn, 52, 1) != 0,
                mode: field(insn, 50, 2),
                x: field(insn, 54, 1) != 0,
                psl: false,
                mrg: false,
            },
        );
    }
    if insn & 0xfe00_0000_0000_0000 == 0x4e00_0000_0000_0000 {
        return decode_xmad(
            insn,
            XmadForm {
                b: const_operand(insn),
                c: Operand::Reg(reg(insn, 39, 8)),
                bh: field(insn, 52, 1) != 0,
                mode: field(insn, 50, 2),
                x: field(insn, 54, 1) != 0,
                psl: field(insn, 55, 1) != 0,
                mrg: field(insn, 56, 1) != 0,
            },
        );
    }
    // The immediate form carries a **16-bit** `b` at 20..36 and multiplies by its low half always.
    if insn & 0xfec0_0000_0000_0000 == 0x3600_0000_0000_0000 {
        return decode_xmad(
            insn,
            XmadForm {
                b: Operand::Imm(field(insn, 20, 16) as u32),
                c: Operand::Reg(reg(insn, 39, 8)),
                bh: false,
                mode: field(insn, 50, 3),
                x: field(insn, 38, 1) != 0,
                psl: field(insn, 36, 1) != 0,
                mrg: field(insn, 37, 1) != 0,
            },
        );
    }

    // fset, the register-writing form of fsetp.
    if insn & 0xff00_0000_0000_0000 == 0x5800_0000_0000_0000
        || insn & 0xfe00_0000_0000_0000 == 0x4800_0000_0000_0000
        || insn & 0xfe00_0000_0000_0000 == 0x3000_0000_0000_0000
    {
        let b = match insn >> 57 {
            0x2c => Operand::Reg(reg(insn, 20, 8)),
            0x24 => const_operand(insn),
            _ => Operand::Imm(imm20f(insn)),
        };
        let Some(bop) = bool_op(field(insn, 45, 2)) else {
            return un;
        };
        return Op::Fset {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            am: FMod {
                neg: field(insn, 43, 1) != 0,
                abs: field(insn, 54, 1) != 0,
            },
            b,
            bm: FMod {
                neg: field(insn, 53, 1) != 0,
                abs: field(insn, 44, 1) != 0,
            },
            cmp: fcmp(field(insn, 48, 4)),
            bop,
            src: src_pred(insn, 39, 42),
            bf: field(insn, 52, 1) != 0,
        };
    }

    // ipa, a[]-relative, non-indexed.
    if insn & 0xff00_0040_0000_ff00 == 0xe000_0000_0000_ff00 {
        // The interpolation mode (bits 54..56).
        let mode = field(insn, 54, 2);
        // The sample mode (`SampleMode` in Eden's decode).
        let sample = field(insn, 52, 2);
        if sample > 1 {
            return un;
        }
        let multiply = mode == 1;
        return Op::Ipa {
            dst: reg(insn, 0, 8),
            offset: field(insn, 28, 10) as u16,
            mul: opt_reg(reg(insn, 20, 8)).filter(|_| multiply),
            perspective: multiply,
            sat: field(insn, 51, 1) != 0,
            centroid: sample == 1,
        };
    }

    // texs.
    if insn & 0xf600_0000_0000_0000 == 0xd000_0000_0000_0000 {
        let dst = reg(insn, 0, 8);
        let dst2 = reg(insn, 28, 8);
        let (a, b) = (reg(insn, 8, 8), reg(insn, 20, 8));
        if let (Some((dim, coords, dref)), Some(mask)) = (
            texs_encoding(field(insn, 53, 4), a, b),
            decode_tex_mask(field(insn, 50, 3), dst, dst2),
        ) {
            return Op::Texs {
                dst,
                dst2,
                coords,
                dref,
                handle: field(insn, 36, 13) as u16,
                dim,
                mask,
                f16: field(insn, 59, 1) == 0,
            };
        }
        return un;
    }

    // tex.b: the bindless sample, 0xdeb8/0xfff8.
    if insn & 0xfff8_0000_0000_0000 == 0xdeb8_0000_0000_0000 {
        return decode_tex(insn, true);
    }
    // txq: a bound texture's size, 0xdf48/0xfff8.
    if insn & 0xfff8_0000_0000_0000 == 0xdf48_0000_0000_0000 {
        return decode_txq(insn);
    }
    // tld4: the gather, `110010` at the top and `111` at [51, 54).
    if insn >> 58 == 0b11_0010 && field(insn, 51, 3) == 0b111 {
        return decode_tld4(insn);
    }
    // tex.
    if insn & 0xf800_0000_0000_0000 == 0xc000_0000_0000_0000 {
        return decode_tex(insn, false);
    }

    // The 32-bit-immediate forms.
    if insn & 0xfff0_0000_0000_0000 == 0x0100_0000_0000_0000 {
        return Op::Mov32i {
            dst: reg(insn, 0, 8),
            imm: field(insn, 20, 32) as u32,
        };
    }
    if insn & 0xfc00_0000_0000_0000 == 0x0800_0000_0000_0000 {
        // fadd32i
        return Op::Fadd {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            am: FMod {
                neg: field(insn, 56, 1) != 0,
                abs: field(insn, 54, 1) != 0,
            },
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

/// The half pair an immediate form of a half-precision op carries.
fn half_imm(insn: u64) -> u32 {
    let low = (field(insn, 20, 9) << 6) | (field(insn, 29, 1) << 15);
    let high = (field(insn, 30, 9) << 22) | (field(insn, 56, 1) << 31);
    (low | high) as u32
}

/// The second operand of a half op's constant-or-immediate pair, and where its two lanes come from.
fn half_cbuf_or_imm(insn: u64, cbuf: bool) -> (Operand, HSwizzle) {
    if cbuf {
        (const_operand(insn), HSwizzle::F32)
    } else {
        (Operand::Imm(half_imm(insn)), HSwizzle::H1H0)
    }
}

/// The half-precision group.
fn decode_half(insn: u64) -> Option<Op> {
    let top = insn >> 48;
    let un = || Some(Op::Unimplemented { raw: insn });
    let dst = reg(insn, 0, 8);
    let a = reg(insn, 8, 8);
    let merge = HMerge::decode(field(insn, 49, 2));
    let asw = HSwizzle::decode(field(insn, 47, 2));
    let reg20 = Operand::Reg(reg(insn, 20, 8));
    let reg39 = Operand::Reg(reg(insn, 39, 8));
    let imm = Operand::Imm(half_imm(insn));
    let imm32 = Operand::Imm(field(insn, 20, 32) as u32);
    let bsw_reg = HSwizzle::decode(field(insn, 28, 2));
    let no_mod = FMod::NONE;

    // ---- hadd2 ----
    if top & 0xfff8 == 0x5d10 {
        return Some(Op::Hadd2 {
            dst,
            a,
            am: FMod {
                neg: field(insn, 43, 1) != 0,
                abs: field(insn, 44, 1) != 0,
            },
            asw,
            b: reg20,
            bm: FMod {
                neg: field(insn, 31, 1) != 0,
                abs: field(insn, 30, 1) != 0,
            },
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
            am: FMod {
                neg: field(insn, 43, 1) != 0,
                abs: field(insn, 44, 1) != 0,
            },
            asw,
            b,
            // An immediate form spends the bits a modifier would need on the pair's own two signs.
            bm: if cbuf {
                FMod {
                    neg: field(insn, 56, 1) != 0,
                    abs: field(insn, 54, 1) != 0,
                }
            } else {
                no_mod
            },
            bsw,
            merge,
            ftz: field(insn, 39, 1) != 0,
            sat: field(insn, 52, 1) != 0,
        });
    }
    // hadd2_32i: its own field positions, and the merge is fixed.
    if top & 0xfe00 == 0x2c00 {
        return Some(Op::Hadd2 {
            dst,
            a,
            am: FMod {
                neg: field(insn, 56, 1) != 0,
                abs: false,
            },
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
            am: FMod {
                neg: false,
                abs: field(insn, 44, 1) != 0,
            },
            asw,
            b: reg20,
            bm: FMod {
                neg: field(insn, 31, 1) != 0,
                abs: field(insn, 30, 1) != 0,
            },
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
            am: FMod {
                neg: field(insn, 43, 1) != 0,
                abs: field(insn, 44, 1) != 0,
            },
            asw,
            b,
            bm: if cbuf {
                FMod {
                    neg: false,
                    abs: field(insn, 54, 1) != 0,
                }
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
    if top & 0xf880 == 0x6080 || top & 0xf880 == 0x7080 || top & 0xf880 == 0x7000 {
        let (b, bsw, c, csw) = if top & 0xf880 == 0x6080 {
            (
                reg39,
                HSwizzle::decode(field(insn, 53, 2)),
                const_operand(insn),
                HSwizzle::F32,
            )
        } else if top & 0x0080 != 0 {
            (
                const_operand(insn),
                HSwizzle::F32,
                reg39,
                HSwizzle::decode(field(insn, 53, 2)),
            )
        } else {
            (
                imm,
                HSwizzle::H1H0,
                reg39,
                HSwizzle::decode(field(insn, 53, 2)),
            )
        };
        return Some(Op::Hfma2 {
            dst,
            a,
            asw,
            b,
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
    // hfma2_32i.
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
    let set_am = FMod {
        neg: field(insn, 43, 1) != 0,
        abs: field(insn, 44, 1) != 0,
    };
    let src = src_pred(insn, 39, 42);
    let is_set2 = top & 0xfff8 == 0x5d18 || top & 0xfe00 == 0x7c00;
    let is_setp2 = top & 0xfff8 == 0x5d20 || top & 0xfe00 == 0x7e00;
    if is_set2 || is_setp2 {
        let Some(bop) = bool_op(field(insn, 45, 2)) else {
            return un();
        };
        let register_form = top & 0xf000 == 0x5000;
        let cbuf = !register_form && top & 0x0080 != 0;
        let (b, bm, bsw) = if register_form {
            (
                reg20,
                FMod {
                    neg: field(insn, 31, 1) != 0,
                    abs: field(insn, 30, 1) != 0,
                },
                bsw_reg,
            )
        } else if cbuf {
            (
                const_operand(insn),
                FMod {
                    neg: field(insn, 56, 1) != 0,
                    abs: field(insn, 54, 1) != 0,
                },
                HSwizzle::F32,
            )
        } else {
            (imm, no_mod, HSwizzle::H1H0)
        };
        let cmp = fcmp(if register_form {
            field(insn, 35, 4)
        } else {
            field(insn, 49, 4)
        });
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

/// What one `xmad` form supplies.
struct XmadForm {
    b: Operand,
    c: Operand,
    bh: bool,
    mode: u64,
    x: bool,
    psl: bool,
    mrg: bool,
}

fn decode_xmad(insn: u64, form: XmadForm) -> Op {
    let un = Op::Unimplemented { raw: insn };
    if form.x {
        return un; // the extended-carry form
    }
    let cmode = match form.mode {
        0 => XmadC::Full,
        1 => XmadC::Lo,
        2 => XmadC::Hi,
        4 => XmadC::Bcc,
        // `csfu` folds a sign into an unsigned product.
        _ => return un,
    };
    let sign = field(insn, 48, 2);
    Op::Xmad {
        dst: reg(insn, 0, 8),
        a: reg(insn, 8, 8),
        ah: field(insn, 53, 1) != 0,
        asigned: sign == 1 || sign == 3,
        b: form.b,
        bh: form.bh,
        bsigned: sign == 2 || sign == 3,
        c: form.c,
        cmode,
        psl: form.psl,
        mrg: form.mrg,
    }
}

/// `d000_1`/`d200_1`-shared 4-bit field.
fn texs_encoding(bits: u64, a: u8, b: u8) -> Option<(TexDim, [u8; 3], Option<u8>)> {
    let next = a.wrapping_add(1);
    let after_b = b.wrapping_add(1);
    Some(match bits {
        // 1D.LZ
        0 => (TexDim::T1d, [a, RZ, RZ], None),
        // 2D, 2D.LZ
        1 | 2 => (TexDim::T2d, [a, b, RZ], None),
        // 2D.LL: `b` is the level, not a coordinate.
        3 => (TexDim::T2d, [a, next, RZ], None),
        // 2D.DC, 2D.LZ.DC: the reference is `b`.
        4 | 6 => (TexDim::T2d, [a, next, RZ], Some(b)),
        // 2D.LL.DC.
        5 => (TexDim::T2d, [a, next, RZ], Some(after_b)),
        // ARRAY_2D, ARRAY_2D.LZ.
        7 | 8 => (TexDim::T2dArray, [next, b, a], None),
        // ARRAY_2D.LZ.DC
        9 => (TexDim::T2dArray, [next, b, a], Some(after_b)),
        // 3D, 3D.LZ
        10 | 11 => (TexDim::T3d, [a, next, b], None),
        // CUBE, CUBE.LL
        12 | 13 => (TexDim::TCube, [a, next, b], None),
        _ => return None,
    })
}

/// Which colour channels a `texs` writes.
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

fn decode_tex(insn: u64, bindless: bool) -> Op {
    let un = Op::Unimplemented { raw: insn };
    let (aoffi_at, blod_at, lc_at) = if bindless { (36, 37, 40) } else { (54, 55, 58) };
    if field(insn, lc_at, 1) != 0 {
        return un;
    }
    // The dimensionalities `TexDim` names.
    let dim = match field(insn, 28, 3) {
        0 => TexDim::T1d,
        2 => TexDim::T2d,
        3 => TexDim::T2dArray,
        4 => TexDim::T3d,
        6 => TexDim::TCube,
        7 => TexDim::TCubeArray,
        _ => return un,
    };
    let dst = reg(insn, 0, 8);
    let bits = field(insn, 31, 4);
    if bits == 0 || dst == RZ {
        return un; // a sample with nowhere to land
    }
    let coord = reg(insn, 8, 8);
    let (layer, first) = match dim {
        TexDim::T2dArray | TexDim::TCubeArray => (Some(coord), coord.wrapping_add(1)),
        _ => (None, coord),
    };
    let mut meta = reg(insn, 20, 8);
    let mut take = || {
        let r = meta;
        meta = meta.wrapping_add(1);
        r
    };
    // The handle comes first, ahead of everything else the meta register chain carries.
    let handle_reg = bindless.then(&mut take);
    // `blod`.
    let lod = match field(insn, blod_at, 3) {
        0 | 1 => None,
        2 | 3 | 6 | 7 => Some(take()),
        _ => return un,
    };
    let offset = (field(insn, aoffi_at, 1) != 0).then(&mut take);
    let dref = (field(insn, 50, 1) != 0).then(&mut take);
    Op::Tex {
        dst,
        coords: [first, first.wrapping_add(1), first.wrapping_add(2)],
        layer,
        dref,
        offset,
        lod,
        handle: if bindless {
            0
        } else {
            field(insn, 36, 13) as u16
        },
        handle_reg,
        dim,
        mask: [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0],
    }
}

/// The four-bit channel mask a texture instruction keeps at `[31, 35)`.
fn texture_mask(insn: u64) -> [bool; 4] {
    let bits = field(insn, 31, 4);
    [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0]
}

/// `txq`.
fn decode_txq(insn: u64) -> Op {
    const DIMENSION: u64 = 1;
    let dst = reg(insn, 0, 8);
    let mask = texture_mask(insn);
    if dst == RZ || field(insn, 22, 3) != DIMENSION || mask == [false; 4] {
        return Op::Unimplemented { raw: insn };
    }
    Op::Txq {
        dst,
        lod: reg(insn, 8, 8),
        handle: field(insn, 36, 13) as u16,
        mask,
    }
}

/// `tld4`, bound, of a 2D image or array.
fn decode_tld4(insn: u64) -> Op {
    let un = Op::Unimplemented { raw: insn };
    // A cube's gather picks its face out of a direction first, which the gather here does not do.
    let dim = match field(insn, 28, 3) {
        2 => TexDim::T2d,
        3 => TexDim::T2dArray,
        _ => return un,
    };
    let dst = reg(insn, 0, 8);
    let mask = texture_mask(insn);
    if dst == RZ || mask == [false; 4] || field(insn, 50, 1) != 0 {
        return un;
    }
    let coord = reg(insn, 8, 8);
    let (layer, first) = match dim {
        TexDim::T2dArray | TexDim::TCubeArray => (Some(coord), coord.wrapping_add(1)),
        _ => (None, coord),
    };
    let offset = match field(insn, 54, 2) {
        0 => None,
        1 => Some(reg(insn, 20, 8)),
        _ => return un,
    };
    Op::Tld4 {
        dst,
        coords: [first, first.wrapping_add(1), first.wrapping_add(2)],
        layer,
        offset,
        handle: field(insn, 36, 13) as u16,
        dim,
        component: field(insn, 56, 2) as u8,
        mask,
    }
}

/// What one of a `texs`'s destination registers ends up holding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexsStore {
    /// The whole register is one channel, as an `f32`.
    Float(usize),
    /// Two channels packed as halves, low first.
    Halves(usize, Option<usize>),
}

/// Where a `texs`'s enabled colour channels land, as `(channel, register)`.
pub fn texs_destinations(dst: u8, dst2: u8, mask: [bool; 4], f16: bool) -> Vec<(u8, TexsStore)> {
    let enabled: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, &on)| on)
        .map(|(channel, _)| channel)
        .collect();
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
        // ld/st s[] sit one nibble above their local counterparts and are encoded identically.
        assert_eq!(
            decode((0xef48u64 | 4) << 48 | PT | 0x20 << 20 | 5 << 8 | 3).op,
            Op::Lds {
                dst: 3,
                addr: 5,
                offset: 0x20,
                size: MemSize::B32
            }
        );
        assert_eq!(
            decode((0xef58u64 | 5) << 48 | PT | 6 << 8 | 2).op,
            Op::Sts {
                addr: 6,
                offset: 0,
                src: 2,
                size: MemSize::B64
            }
        );
        // Still the local pair, not the shared one.
        assert_eq!(
            decode((0xef40u64 | 4) << 48 | PT | 5 << 8 | 3).op,
            Op::Ldl {
                dst: 3,
                addr: 5,
                offset: 0,
                size: MemSize::B32
            }
        );
    }

    #[test]
    fn a_negative_shared_offset_stays_negative() {
        let offset = (-8i64 as u64) & 0xFF_FFFF;
        assert_eq!(
            decode((0xef48u64 | 4) << 48 | PT | offset << 20 | 5 << 8 | 3).op,
            Op::Lds {
                dst: 3,
                addr: 5,
                offset: -8,
                size: MemSize::B32
            }
        );
    }

    #[test]
    fn decodes_each_barrier_form() {
        // The mode's bits are not contiguous.
        let bar = |mode: u64| decode(0xf0a8u64 << 48 | mode << 32 | PT).op;
        assert_eq!(
            bar(0x80),
            Op::Bar {
                mode: BarMode::Sync
            }
        );
        assert_eq!(
            bar(0x81),
            Op::Bar {
                mode: BarMode::Arrive
            }
        );
        assert_eq!(
            bar(0x02),
            Op::Bar {
                mode: BarMode::RedPopc
            }
        );
        assert_eq!(
            bar(0x03),
            Op::Bar {
                mode: BarMode::Scan
            }
        );
        assert_eq!(
            bar(0x0a),
            Op::Bar {
                mode: BarMode::RedAnd
            }
        );
        assert_eq!(
            bar(0x12),
            Op::Bar {
                mode: BarMode::RedOr
            }
        );
        // membar and depbar are still the no-ops they were.
        assert_eq!(decode(0xef98u64 << 48 | PT).op, Op::Inert);
        assert_eq!(decode(0xf0f0u64 << 48 | PT).op, Op::Inert);
    }

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
        for (bits, mode) in [
            (0, ShflMode::Idx),
            (1, ShflMode::Up),
            (2, ShflMode::Down),
            (3, ShflMode::Bfly),
        ] {
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

        // The same instruction with both operands in registers.
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
        let fswzadd = |extra: u64| {
            decode(0x50f8u64 << 48 | extra | 0xe4 << 28 | 2 << 20 | PT | 1 << 8 | 3).op
        };
        assert_eq!(
            fswzadd(0),
            Op::Fswzadd {
                dst: 3,
                a: 1,
                b: 2,
                swizzle: 0xe4,
                ftz: false
            }
        );
        assert_eq!(
            fswzadd(1 << 44),
            Op::Fswzadd {
                dst: 3,
                a: 1,
                b: 2,
                swizzle: 0xe4,
                ftz: true
            }
        );
        for extra in [1u64 << 47, 1 << 39, 2 << 39] {
            assert!(
                matches!(fswzadd(extra), Op::Unimplemented { .. }),
                "{extra:#x}"
            );
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
        // The one place the two atomic encodings genuinely differ.
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
        // Which is exactly RZ as the destination, so the interpreter needs no second path for it.
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
        assert!(matches!(
            decode(0xed00u64 << 48 | 9 << 52 | PT).op,
            Op::Unimplemented { .. }
        ));
    }

    #[test]
    fn an_op_still_fits_in_thirty_two_bytes() {
        assert_eq!(std::mem::size_of::<Op>(), 32);
    }

    fn op(word: u64) -> Op {
        decode(word).op
    }

    #[test]
    fn decodes_ipa_pass_then_mufu_rcp() {
        // solid.frag: "ipa pass $r0 a[0x7c] 0x0 0x0 0x1"
        assert_eq!(
            op(0xe003ff87cff7ff00),
            Op::Ipa {
                dst: 0,
                offset: 0x7c,
                mul: None,
                perspective: false,
                sat: false,
                centroid: false
            }
        );
        // "mufu rcp $r3 $r0"
        assert_eq!(
            op(0x5080000000470003),
            Op::Mufu {
                dst: 3,
                src: 0,
                sm: FMod::NONE,
                op: MufuOp::Rcp,
                sat: false
            }
        );
        // "ipa $r0 a[0x80] $r3 0x0 0x1"
        assert_eq!(
            op(0xe043ff880037ff00),
            Op::Ipa {
                dst: 0,
                offset: 0x80,
                mul: Some(3),
                perspective: true,
                sat: false,
                centroid: false
            }
        );
    }

    #[test]
    fn decodes_the_ipa_sample_modes() {
        assert_eq!(
            op(0xe013ff87cff7ff06),
            Op::Ipa {
                dst: 6,
                offset: 0x7c,
                mul: None,
                perspective: false,
                sat: false,
                centroid: true
            }
        );
        // The same instruction with sample mode 2 (offset) and 3.
        assert!(matches!(op(0xe023ff87cff7ff06), Op::Unimplemented { .. }));
        assert!(matches!(op(0xe033ff87cff7ff06), Op::Unimplemented { .. }));
    }

    /// Bits 54..56 are the interpolation mode.
    #[test]
    fn decodes_the_ipa_interpolation_modes() {
        let ipa = |raw| match op(raw) {
            Op::Ipa {
                mul, perspective, ..
            } => (mul, perspective),
            other => panic!("not an ipa: {other:?}"),
        };
        assert_eq!(ipa(0xe003ff8800_37ff00), (None, false));
        assert_eq!(ipa(0xe043ff8800_37ff00), (Some(3), true));
        assert_eq!(ipa(0xe083ff8800_37ff00), (None, false));
        assert_eq!(ipa(0xe0c3ff8800_37ff00), (None, false));
        // One of A Short Hike's own, whose multiplier field is already RZ.
        assert_eq!(
            op(0xe083ff890ff7ff00),
            Op::Ipa {
                dst: 0,
                offset: 0x90,
                mul: None,
                perspective: false,
                sat: false,
                centroid: false,
            }
        );
    }

    /// `f2f` between the two float widths.
    #[test]
    fn decodes_f2f_between_half_and_single() {
        // Two of the title's own: `f2f.f16.f16.floor` off H0 and off H1.
        assert_eq!(
            op(0x5ca8048000370503),
            Op::F2f {
                dst: 3,
                src: Operand::Reg(3),
                sm: FMod::NONE,
                round: Some(FRound::Floor),
                sat: false,
                ftz: false,
                src_bits: 16,
                dst_bits: 16,
                hi: false,
            }
        );
        assert!(matches!(
            op(0x5ca8068000370500),
            Op::F2f {
                round: Some(FRound::Floor),
                src_bits: 16,
                dst_bits: 16,
                hi: true,
                ..
            }
        ));
        let widths = |dst: u64, src: u64| {
            let raw = 0x5ca8_0000_0000_0000 | (dst << 8) | (src << 10);
            match op(raw) {
                Op::F2f {
                    src_bits,
                    dst_bits,
                    round,
                    ..
                } => (src_bits, dst_bits, round),
                other => panic!("not an f2f: {other:?}"),
            }
        };
        assert_eq!(widths(2, 1), (16, 32, None));
        assert_eq!(widths(1, 2), (32, 16, None));
        assert_eq!(widths(2, 2), (32, 32, None));
        // f64 is not modelled, and neither is a cast that rounds any way but to nearest.
        assert!(matches!(
            op(0x5ca8_0000_0000_0300),
            Op::Unimplemented { .. }
        ));
        assert!(matches!(
            op(0x5ca8_0080_0000_0200),
            Op::Unimplemented { .. }
        ));
    }

    /// `fset` against a 20-bit float immediate.
    #[test]
    fn decodes_fset_against_an_immediate() {
        assert_eq!(
            op(0x309303bf00070301),
            Op::Fset {
                dst: 1,
                a: 3,
                am: FMod::NONE,
                b: Operand::Imm(0.5f32.to_bits()),
                bm: FMod::NONE,
                cmp: FCmp::Le,
                bop: BoolOp::And,
                src: Pred {
                    reg: 7,
                    negate: false
                },
                bf: true,
            }
        );
        // The same instruction in its register and constant-bank forms.
        assert!(matches!(
            op(0x5800_0000_0000_0000),
            Op::Fset {
                b: Operand::Reg(_),
                ..
            }
        ));
        assert!(matches!(
            op(0x4800_0000_0000_0000),
            Op::Fset {
                b: Operand::Const { .. },
                ..
            }
        ));
    }

    /// The depth-compare encodings, which name a reference register beside their coordinates.
    #[test]
    fn decodes_the_texs_depth_compare_forms() {
        let texs = |enc: u64| {
            let raw = 0xd000_0000_0000_0000u64
                | (1 << 59)
                | (enc << 53)
                | (0x20 << 36)
                | (8 << 8)
                | (20 << 20)
                | 2;
            match op(raw) {
                Op::Texs {
                    coords, dref, dim, ..
                } => (dim, coords, dref),
                other => panic!("expected texs, got {other:?}"),
            }
        };
        // The plain 2D sample it is otherwise identical to.
        assert_eq!(texs(1), (TexDim::T2d, [8, 20, RZ], None));
        // 2d.dc, 2d.lz.dc: the reference is `b`.
        assert_eq!(texs(4), (TexDim::T2d, [8, 9, RZ], Some(20)));
        assert_eq!(texs(6), (TexDim::T2d, [8, 9, RZ], Some(20)));
        // 2d.ll.dc: `b` is the level, so the reference follows it.
        assert_eq!(texs(5), (TexDim::T2d, [8, 9, RZ], Some(21)));
        // array_2d.lz.dc: the layer still comes from `a`.
        assert_eq!(texs(9), (TexDim::T2dArray, [9, 20, 8], Some(21)));

        // A Short Hike's own three, all `2d.lz.dc`.
        for raw in [
            0xd0c200aff0670404u64,
            0xd8c200aff1270001,
            0xd8c200aff1270004,
        ] {
            assert!(
                matches!(op(raw), Op::Texs { dref: Some(_), .. }),
                "{raw:#018x} is a shadow sample"
            );
        }
    }

    #[test]
    fn an_exit_its_flow_test_can_never_satisfy_is_not_an_exit() {
        // `exit` carries a condition-code test beside its predicate, and both have to hold.
        assert_eq!(op(0xe3000000_0007001c), Op::Nop); // FCSM_TR
        assert_eq!(op(0xe3000000_00070000), Op::Nop); // F
        assert_eq!(op(0xe3000000_0007000f), Op::Exit); // T

        // `kil` carries the same field.
        assert_eq!(op(0xe3300000_00070000), Op::Nop);
        assert_eq!(op(0xe3300000_0007000f), Op::Kil);
        assert_eq!(op(0xe2400fffff870000), Op::Nop);
        assert!(matches!(op(0xe2400fffff87000f), Op::Bra { .. }));
    }

    #[test]
    fn decodes_ld_st_b128_attribute_space() {
        // mvp.vert: "ld b128 $r0 a[0x80] 0x0"
        assert_eq!(
            op(0xefd9ff80_0807ff00),
            Op::Ld {
                dst: 0,
                offset: 0x80,
                idx: RZ,
                size: MemSize::B128
            }
        );
        // "st b128 a[0x70] $r0 0x0"
        assert_eq!(
            op(0xeff1ff80_0707ff00),
            Op::St {
                offset: 0x70,
                idx: RZ,
                src: 0,
                size: MemSize::B128
            }
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
                b: Operand::Const {
                    bank: 2,
                    offset: 0x0
                },
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
                b: Operand::Const {
                    bank: 2,
                    offset: 0x4
                },
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
        assert_eq!(
            op(0x4c58100000c70204),
            Op::Fadd {
                dst: 4,
                a: 2,
                am: FMod::NONE,
                b: Operand::Const {
                    bank: 0,
                    offset: 0x30
                },
                bm: FMod::NONE,
                ftz: true,
                sat: false,
            }
        );
    }

    #[test]
    fn decodes_mov32i() {
        // Captured from a live JKSV run.
        assert_eq!(
            op(0x0103f8000007f000),
            Op::Mov32i {
                dst: 0,
                imm: 0x3f800000
            }
        );
    }

    #[test]
    fn decodes_ffma_constant_bank_chain() {
        // mvp.vert: "ffma ftz $r4 $r1 c2[0x10] $r4"
        assert_eq!(
            op(0x49a00208_00470104),
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
                sat: false,
            }
        );
        // "ffma ftz $r0 $r3 c2[0x30] $r1"
        assert_eq!(
            op(0x49a00088_00c70300),
            Op::Ffma {
                dst: 0,
                a: 3,
                b: Operand::Const {
                    bank: 2,
                    offset: 0x30
                },
                bneg: false,
                c: Operand::Reg(1),
                cneg: false,
                ftz: true,
                sat: false,
            }
        );
    }

    #[test]
    fn decodes_tex() {
        assert_eq!(
            op(0xc07a0080a0770401),
            Op::Tex {
                dst: 1,
                coords: [4, 5, 6],
                layer: None,
                dref: None,
                offset: Some(7),
                lod: None,
                handle: 8,
                handle_reg: None,
                dim: TexDim::T2d,
                mask: [true, false, false, false],
            }
        );
        assert_eq!(
            op(0xc0f80083a0b70e08),
            Op::Tex {
                dst: 8,
                coords: [14, 15, 16],
                layer: None,
                dref: None,
                offset: Some(11),
                lod: None,
                handle: 8,
                handle_reg: None,
                dim: TexDim::T2d,
                mask: [true, true, true, false],
            }
        );
        // `.LL` takes a level out of the meta register, so everything after it moves along one.
        let ll = 0xc07a0080a0770401 | 3 << 55;
        assert!(matches!(
            op(ll),
            Op::Tex {
                lod: Some(7),
                offset: Some(8),
                ..
            }
        ));
        assert!(matches!(op(ll | 1 << 58), Op::Unimplemented { .. }));
        assert!(matches!(
            op(0xc07a0080a0770401 & !(0xf << 31)),
            Op::Unimplemented { .. }
        ));
        assert!(matches!(
            op(0xc07a0080a0770401 | u64::from(RZ)),
            Op::Unimplemented { .. }
        ));
    }

    #[test]
    fn decodes_an_i2i_cc_and_a_csetp() {
        assert!(matches!(
            decode(0x5ce0800000170aff).op,
            Op::I2i {
                dst: RZ,
                cc: true,
                ..
            }
        ));
        assert_eq!(
            decode(0x50a0038000070d07).op,
            Op::Csetp {
                p0: 0,
                p1: 7,
                test: 13,
                src: Pred::ALWAYS,
                op: BoolOp::And,
            }
        );
    }

    #[test]
    fn decodes_a_txq_and_a_tld4() {
        assert_eq!(
            decode(0xdf48008180470800).op,
            Op::Txq {
                dst: 0,
                lod: 8,
                handle: 8,
                mask: [true, true, false, false],
            }
        );
        assert_eq!(
            decode(0xc83a0086aff70208).op,
            Op::Tld4 {
                dst: 8,
                coords: [2, 3, 4],
                layer: None,
                offset: None,
                handle: 8,
                dim: TexDim::T2d,
                component: 0,
                mask: [true, false, true, true],
            }
        );
        // The green channel, and a shadow gather, which is not decoded.
        let green = 0xc83a0086aff70208u64 | 1 << 56;
        assert!(matches!(decode(green).op, Op::Tld4 { component: 1, .. }));
        let shadow = 0xc83a0086aff70208u64 | 1 << 50;
        assert!(matches!(decode(shadow).op, Op::Unimplemented { .. }));
    }

    #[test]
    fn decodes_a_bindless_tex() {
        // Tomodachi Life's `tex.b`.
        assert_eq!(
            op(0xdeba0007a0270000),
            Op::Tex {
                dst: 0,
                coords: [0, 1, 2],
                layer: None,
                dref: None,
                offset: None,
                lod: None,
                handle: 0,
                handle_reg: Some(2),
                dim: TexDim::T2d,
                mask: [true; 4],
            }
        );
        assert!(matches!(
            op(0xdeba0007a0270000 | 3 << 37),
            Op::Tex {
                handle_reg: Some(2),
                lod: Some(3),
                ..
            }
        ));
        // `.LC` is at bit 40 in this form.
        assert!(matches!(
            op(0xdeba0007a0270000 | 1 << 40),
            Op::Unimplemented { .. }
        ));
    }

    #[test]
    fn decodes_a_cube_array_tex() {
        // Tomodachi Life's, the two a draw fell back on before cube arrays decoded.
        assert_eq!(
            op(0xc03a0087fff70400),
            Op::Tex {
                dst: 0,
                coords: [5, 6, 7],
                layer: Some(4),
                dref: None,
                offset: None,
                lod: None,
                handle: 8,
                handle_reg: None,
                dim: TexDim::TCubeArray,
                mask: [true; 4],
            }
        );
        assert!(matches!(
            op(0xc1ba0087f0970400),
            Op::Tex {
                layer: Some(4),
                lod: Some(9),
                dim: TexDim::TCubeArray,
                ..
            }
        ));
    }

    #[test]
    fn decodes_vmnmx_only_where_every_field_is_one_it_models() {
        // Tomodachi Life's `vmnmx r7, r9, r12, r7`.
        const WORD: u64 = 0x3a2c03e060c70907;
        assert_eq!(
            op(WORD),
            Op::Vmnmx {
                dst: 7,
                a: 9,
                b: 12,
                c: 7,
                max: false,
                then_max: false,
                signed: false,
                then_signed: false,
            }
        );
        assert!(matches!(
            op(WORD & !(7 << 51) | 6 << 51 | 1 << 56),
            Op::Vmnmx {
                max: true,
                then_max: true,
                ..
            }
        ));
        for (why, word) in [
            ("an immediate b", WORD & !(1 << 50)),
            ("a byte of a", WORD & !(1 << 38)),
            ("operands of different signedness", WORD | 1 << 48),
            ("saturation", WORD | 1 << 55),
            ("condition codes", WORD | 1 << 47),
            ("an accumulate", WORD & !(7 << 51) | 4 << 51),
        ] {
            assert!(
                matches!(op(word), Op::Unimplemented { .. }),
                "{why} is not modelled"
            );
        }
    }

    #[test]
    fn decodes_texs() {
        // tex.frag.
        assert_eq!(
            op(0xd8301a40_20170000),
            Op::Texs {
                dst: 0,
                dst2: 2,
                coords: [0, 1, RZ],
                dref: None,
                handle: 0x1a4,
                dim: TexDim::T2d,
                mask: [true, true, true, true],
                f16: false,
            }
        );
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
        // Bit 59 halves the register count.
        assert_eq!(
            texs_destinations(1, 0, [true, true, true, true], true),
            vec![
                (1, TexsStore::Halves(0, Some(1))),
                (0, TexsStore::Halves(2, Some(3)))
            ]
        );
        // An odd count pads the unused half with zero rather than spilling into another register.
        assert_eq!(
            texs_destinations(4, 6, [true, true, true, false], true),
            vec![
                (4, TexsStore::Halves(0, Some(1))),
                (6, TexsStore::Halves(2, None))
            ]
        );
        assert_eq!(
            texs_destinations(4, RZ, [false, false, false, true], true),
            vec![(4, TexsStore::Halves(3, None))]
        );
    }

    #[test]
    fn the_precision_bit_is_decoded_and_its_polarity_is_backwards() {
        // `Precision` numbers F16 as 0 and F32 as 1, so a set bit is the *unpacked* form.
        assert!(matches!(
            op(0xd8301a40_20170000),
            Op::Texs { f16: false, .. }
        ));
        assert!(matches!(
            op(0xd8301a40_20170000 & !(1 << 59)),
            Op::Texs { f16: true, .. }
        ));
    }

    #[test]
    fn a_one_destination_texs_reads_the_single_and_double_channel_masks() {
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

    /// Assemble one instruction.
    #[test]
    fn xmad_immediate_keeps_its_modifiers_where_the_register_form_does() {
        // xmad R1, R2, 0x7, RZ
        let lo = asm(0x3600, &[(0, 8, 1), (8, 8, 2), (20, 15, 7), (39, 8, 255)]);
        match op(lo) {
            Op::Xmad {
                dst,
                a,
                ah,
                b,
                c,
                psl,
                mrg,
                ..
            } => {
                assert_eq!((dst, a), (1, 2));
                assert_eq!(b, Operand::Imm(7));
                assert_eq!(c, Operand::Reg(255));
                assert!(!ah && !psl && !mrg);
            }
            other => panic!("expected xmad, got {other:?}"),
        }
        // xmad.psl R1, R2.h1, 0x7, R0, the same constant, one bit up.
        let hi = asm(
            0x3600,
            &[
                (0, 8, 1),
                (8, 8, 2),
                (20, 15, 7),
                (36, 1, 1),
                (39, 8, 0),
                (53, 1, 1),
            ],
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

    #[test]
    fn decodes_the_xmad_select_modes_and_operand_forms() {
        let cmode = |raw| match op(raw) {
            Op::Xmad { cmode, .. } => Some(cmode),
            Op::Unimplemented { .. } => None,
            other => panic!("expected xmad, got {other:?}"),
        };
        for (mode, want) in [
            (0, Some(XmadC::Full)),
            (1, Some(XmadC::Lo)),
            (2, Some(XmadC::Hi)),
            // `csfu`, which Eden does not implement either.
            (3, None),
            (4, Some(XmadC::Bcc)),
        ] {
            assert_eq!(cmode(asm(0x5b00, &[(50, 3, mode)])), want, "mode {mode}");
        }
        // A Short Hike's own.
        assert!(matches!(
            op(0x36247f9000180303),
            Op::Xmad {
                cmode: XmadC::Lo,
                ..
            }
        ));
        assert!(matches!(
            op(0x5b30041800970704),
            Op::Xmad {
                cmode: XmadC::Bcc,
                ..
            }
        ));

        // `rc` multiplies by the register and adds the bank; `cr` is the other way round.
        let operands = |raw| match op(raw) {
            Op::Xmad { b, c, .. } => (b, c),
            other => panic!("expected xmad, got {other:?}"),
        };
        let (b, c) = operands(asm(0x5100, &[(39, 8, 5)]));
        assert_eq!(b, Operand::Reg(5));
        assert!(matches!(c, Operand::Const { .. }));
        let (b, c) = operands(asm(0x4e00, &[(39, 8, 5)]));
        assert!(matches!(b, Operand::Const { .. }));
        assert_eq!(c, Operand::Reg(5));
    }

    const FLOW_TEST_T: u64 = 0xF;

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
        // The same `exit`, guarded by `!p1`.
        let raw = 0xe3000000_0007000f & !(0xf << 16) | (1 << 16) | (1 << 19);
        let insn = decode(raw);
        assert_eq!(insn.op, Op::Exit);
        assert_eq!(
            insn.pred,
            Pred {
                reg: 1,
                negate: true
            }
        );
        assert!(!insn.pred.is_always());
        assert!(decode(0xe3000000_0007000f).pred.is_always());
    }

    #[test]
    fn decodes_source_modifiers_on_fadd() {
        // fadd $r0, -|$r1|, $r2: neg 48 / abs 46 on a, both clear on b.
        let raw = asm(
            0x5c58,
            &[(0, 8, 0), (8, 8, 1), (20, 8, 2), (48, 1, 1), (46, 1, 1)],
        );
        assert_eq!(
            op(raw),
            Op::Fadd {
                dst: 0,
                a: 1,
                am: FMod {
                    neg: true,
                    abs: true
                },
                b: Operand::Reg(2),
                bm: FMod::NONE,
                ftz: false,
                sat: false,
            }
        );
    }

    #[test]
    fn decodes_isetp_and_its_predicate_destinations() {
        let raw = asm(
            0x5b60,
            &[
                (0, 3, 7),
                (3, 3, 0),
                (8, 8, 1),
                (20, 8, 2),
                (39, 3, 7),
                (48, 1, 1),
                (49, 3, 1),
            ],
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
        let raw = asm(0xe240, &[(20, 24, (-0x10i64) as u64), (0, 5, FLOW_TEST_T)]);
        assert_eq!(decode_at(raw, 0x18).op, Op::Bra { target: 0x10 });
    }

    #[test]
    fn decodes_the_reconvergence_ops() {
        assert_eq!(
            decode_at(asm(0xe290, &[(20, 24, 0x18)]), 0).op,
            Op::Ssy { target: 0x28 }
        );
        // And one that lands on a real slot is left alone.
        assert_eq!(
            decode_at(asm(0xe290, &[(20, 24, 0x20)]), 0).op,
            Op::Ssy { target: 0x28 }
        );
        assert_eq!(op(asm(0xf0f8, &[])), Op::Sync);
        assert_eq!(op(asm(0xe340, &[])), Op::Brk);
        assert_eq!(op(asm(0x50b0, &[])), Op::Nop);
    }

    #[test]
    fn a_vertex_stage_vote_is_a_nop() {
        assert_eq!(op(0x50e2_4321_1117_0000), Op::Nop);
        // The whole 0x50e0 group, not just that encoding.
        for low in 0..8u64 {
            assert_eq!(op(0x50e0_0000_0000_0000 | (low << 48)), Op::Nop);
        }
    }

    #[test]
    fn a_range_reduction_keeps_its_modifiers() {
        let neg = FMod {
            neg: true,
            abs: false,
        };
        assert_eq!(
            op(0x5c90_2000_0077_001b),
            Op::Rro {
                dst: 27,
                src: Operand::Reg(7),
                sm: neg,
            }
        );
        assert_eq!(
            op(0x5c90_2000_01f7_0018),
            Op::Rro {
                dst: 24,
                src: Operand::Reg(31),
                sm: neg,
            }
        );
    }

    /// A negative immediate sets bit 56, which moves the opcode up one.
    #[test]
    fn a_negative_immediate_does_not_change_the_opcode() {
        assert!(matches!(
            op(0x3764_03ff_fff7_0207),
            Op::Isetp {
                a: 2,
                b: Operand::Imm(u32::MAX),
                cmp: ICmp::Eq,
                signed: false,
                ..
            }
        ));
        assert!(matches!(
            op(0x37b3_03bf_8007_0407),
            Op::Fsetp {
                a: 4,
                b: Operand::Imm(0xbf80_0000),
                ..
            }
        ));
        assert!(matches!(
            op(0x3754_03ff_fff7_0403),
            Op::Iset {
                dst: 3,
                b: Operand::Imm(u32::MAX),
                ..
            }
        ));
        assert!(matches!(
            op(0x39c0_0300_0057_0402),
            Op::Iadd3 {
                dst: 2,
                b: Operand::Imm(0xfff8_0005),
                ..
            }
        ));
    }

    #[test]
    fn surface_accesses_decode_as_envydis_reads_them() {
        assert_eq!(
            op(0xeb20_1386_00f7_0018),
            Op::Sust {
                src: 24,
                coords: 0,
                handle: 0,
                handle_reg: Some(39),
                dim: SurfaceDim::D2,
                data: SurfaceData::Formatted([true; 4]),
            }
        );
        assert_eq!(
            op(0xeb18_0406_0047_0804),
            Op::Suld {
                dst: 4,
                coords: 8,
                handle: 0x40,
                handle_reg: None,
                dim: SurfaceDim::D2,
                data: SurfaceData::Raw(SurfaceSize::B32),
            }
        );
        assert_eq!(
            op(0xeb00_1388_0097_0804),
            Op::Suld {
                dst: 4,
                coords: 8,
                handle: 0,
                handle_reg: Some(39),
                dim: SurfaceDim::Array2d,
                data: SurfaceData::Formatted([true, false, false, true]),
            }
        );
        assert_eq!(
            op(0xeb18_0030_0007_0100),
            Op::Suld {
                dst: 0,
                coords: 1,
                handle: 3,
                handle_reg: None,
                dim: SurfaceDim::D1,
                data: SurfaceData::Raw(SurfaceSize::U8),
            }
        );
        assert!(matches!(
            op(0xeb20_1386_0077_0018),
            Op::Unimplemented { .. }
        ));
        assert!(matches!(
            op(0xeb20_1386_00f7_0018 | 2 << 49),
            Op::Unimplemented { .. }
        ));
    }

    #[test]
    fn a_vote_decodes_its_mode_and_both_predicates() {
        assert_eq!(
            op(0x50d8_e380_0007_0002),
            Op::Vote {
                dst: 2,
                pred: Pred::PT,
                src: Pred::ALWAYS,
                mode: VoteMode::All,
            }
        );
        assert_eq!(
            op(0x50d9_a200_0007_0005),
            Op::Vote {
                dst: 5,
                pred: 5,
                src: Pred {
                    reg: 4,
                    negate: false
                },
                mode: VoteMode::Any,
            }
        );
        assert_eq!(
            op(0x50da_4400_0007_0003),
            Op::Vote {
                dst: 3,
                pred: 2,
                src: Pred {
                    reg: 0,
                    negate: true
                },
                mode: VoteMode::Eq,
            }
        );
    }

    #[test]
    fn a_reconvergence_push_is_never_predicated() {
        for opcode in [0xe290u16, 0xe2a0, 0xe2b0] {
            let raw = asm(opcode, &[(20, 24, 0x20)]);
            assert!(decode_at(raw, 0).pred.is_always(), "opcode {opcode:#x}");
        }
        // `bra` in the same group *is* predicated, and keeps its guard.
        let bra = (asm(0xe240, &[(20, 24, 0x20), (0, 5, FLOW_TEST_T)]) & !(0x7 << 16)) | (3 << 16);
        assert_eq!(
            decode_at(bra, 0).pred,
            Pred {
                reg: 3,
                negate: false
            }
        );
    }

    #[test]
    fn decodes_integer_alu() {
        // iadd r0, r1, -r2
        assert_eq!(
            op(asm(0x5c10, &[(0, 8, 0), (8, 8, 1), (20, 8, 2), (48, 1, 1)])),
            Op::Iadd {
                dst: 0,
                a: 1,
                aneg: false,
                b: Operand::Reg(2),
                bneg: true,
                cin: false,
                cout: false
            }
        );
        // shl r3, r4, 0x2 (immediate form)
        assert_eq!(
            op(asm(0x3848, &[(0, 8, 3), (8, 8, 4), (20, 19, 2)])),
            Op::Shl {
                dst: 3,
                a: 4,
                b: Operand::Imm(2),
                wrap: false
            }
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
        // mov r5, r6: the byte-enable mask must be "all four".
        assert_eq!(
            op(asm(0x5c98, &[(0, 8, 5), (20, 8, 6), (39, 4, 0xf)])),
            Op::Mov {
                dst: 5,
                src: Operand::Reg(6)
            }
        );
    }

    #[test]
    fn decodes_conversions() {
        // i2f.f32.s32 r0, r1
        assert_eq!(
            op(asm(
                0x5cb8,
                &[(0, 8, 0), (20, 8, 1), (8, 2, 2), (10, 2, 2), (13, 1, 1)]
            )),
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
                &[
                    (0, 8, 2),
                    (20, 8, 3),
                    (10, 2, 2),
                    (8, 2, 2),
                    (12, 1, 1),
                    (39, 2, 3)
                ]
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
    fn the_half_instructions_a_unity_shader_issues() {
        // hadd2.f32 $r12 $r12 $r17.
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
        // hmul2.f32 $r4 $r5 c1[0xc].
        assert_eq!(
            op(0x7882800400370504),
            Op::Hmul2 {
                dst: 4,
                a: 5,
                am: FMod::NONE,
                asw: HSwizzle::F32,
                b: Operand::Const {
                    bank: 1,
                    offset: 0xc
                },
                bm: FMod::NONE,
                bsw: HSwizzle::F32,
                merge: HMerge::F32,
                prec: HPrecision::None,
                sat: false,
            }
        );
        assert_eq!(
            op(0x7a02883c0f070900),
            Op::Hadd2 {
                dst: 0,
                a: 9,
                am: FMod {
                    neg: true,
                    abs: false
                },
                asw: HSwizzle::F32,
                b: Operand::Imm(0x3C00_3C00),
                bm: FMod::NONE,
                bsw: HSwizzle::H1H0,
                merge: HMerge::F32,
                ftz: false,
                sat: false,
            }
        );
        // hadd2 $r8.h0 -$rZ.h0_h0 c1[0x0].
        assert_eq!(
            op(0x7a8508040007ff08),
            Op::Hadd2 {
                dst: 8,
                a: RZ,
                am: FMod {
                    neg: true,
                    abs: false
                },
                asw: HSwizzle::H0H0,
                b: Operand::Const { bank: 1, offset: 0 },
                bm: FMod::NONE,
                bsw: HSwizzle::F32,
                merge: HMerge::MrgH0,
                ftz: false,
                sat: false,
            }
        );
        assert_eq!(
            op(0x7e85038c0507ff07),
            Op::Hsetp2 {
                p0: 0,
                p1: Pred::PT,
                a: RZ,
                am: FMod::NONE,
                asw: HSwizzle::H0H0,
                b: Operand::Const {
                    bank: 3,
                    offset: 0x140
                },
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

    #[test]
    fn every_half_operand_form_reaches_its_op() {
        // hfma2 $r1 $r2 $r3 $r4, register form.
        let hfma_reg = asm(0x5d00, &[(0, 8, 1), (8, 8, 2), (20, 8, 3), (39, 8, 4)]);
        assert!(matches!(
            op(hfma_reg),
            Op::Hfma2 {
                dst: 1,
                a: 2,
                b: Operand::Reg(3),
                c: Operand::Reg(4),
                ..
            }
        ));
        // hfma2 with the constant bank as `b` (`cr`) and as `c` (`rc`).
        let hfma_cr = asm(
            0x7080,
            &[(0, 8, 1), (8, 8, 2), (39, 8, 4), (20, 14, 3), (34, 5, 2)],
        );
        assert!(matches!(
            op(hfma_cr),
            Op::Hfma2 {
                b: Operand::Const {
                    bank: 2,
                    offset: 0xc
                },
                c: Operand::Reg(4),
                ..
            }
        ));
        let hfma_rc = asm(
            0x6080,
            &[(0, 8, 1), (8, 8, 2), (39, 8, 4), (20, 14, 3), (34, 5, 2)],
        );
        assert!(matches!(
            op(hfma_rc),
            Op::Hfma2 {
                b: Operand::Reg(4),
                c: Operand::Const {
                    bank: 2,
                    offset: 0xc
                },
                ..
            }
        ));
        assert!(matches!(
            op(asm(0x2c00, &[(0, 8, 1), (8, 8, 2), (20, 32, 0x3c00_3c00)])),
            Op::Hadd2 {
                dst: 1,
                a: 2,
                b: Operand::Imm(0x3c00_3c00),
                merge: HMerge::H1H0,
                ..
            }
        ));
        assert!(matches!(
            op(asm(0x2a00, &[(0, 8, 1), (8, 8, 2), (20, 32, 0x3c00_3c00)])),
            Op::Hmul2 {
                dst: 1,
                b: Operand::Imm(0x3c00_3c00),
                merge: HMerge::H1H0,
                ..
            }
        ));
        assert!(matches!(
            op(asm(0x2800, &[(0, 8, 1), (8, 8, 2), (20, 32, 0x3c00_3c00)])),
            Op::Hfma2 {
                dst: 1,
                c: Operand::Reg(1),
                merge: HMerge::H1H0,
                ..
            }
        ));
        assert!(matches!(
            op(asm(
                0x5d18,
                &[(0, 8, 1), (8, 8, 2), (20, 8, 3), (35, 4, 4), (49, 1, 1)]
            )),
            Op::Hset2 {
                dst: 1,
                a: 2,
                b: Operand::Reg(3),
                cmp: FCmp::Gt,
                bf: true,
                ..
            }
        ));
        assert!(matches!(
            op(asm(
                0x7c80,
                &[(0, 8, 1), (8, 8, 2), (49, 4, 4), (20, 14, 3), (34, 5, 2)]
            )),
            Op::Hset2 {
                cmp: FCmp::Gt,
                b: Operand::Const {
                    bank: 2,
                    offset: 0xc
                },
                ..
            }
        ));
        // hsetp2's `.h_and` collapses both lanes into one predicate.
        assert!(matches!(
            op(asm(
                0x5d20,
                &[
                    (3, 3, 1),
                    (0, 3, 2),
                    (8, 8, 2),
                    (20, 8, 3),
                    (35, 4, 1),
                    (49, 1, 1)
                ]
            )),
            Op::Hsetp2 {
                p0: 1,
                p1: 2,
                cmp: FCmp::Lt,
                and: true,
                ..
            }
        ));
    }

    #[test]
    fn an_immediate_half_pair_reassembles_both_signs() {
        // -1.0 in the low half (0xbc00) and +2.0 in the high (0x4000).
        let insn = asm(
            0x7a00,
            &[(20, 9, 0xf0), (29, 1, 1), (30, 9, 0x100), (56, 1, 0)],
        );
        assert_eq!(half_imm(insn), 0x4000_bc00);
    }
}
