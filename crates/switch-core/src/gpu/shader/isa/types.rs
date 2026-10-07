//! Operand, modifier and mode types the decoded ops carry.

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
    pub(super) fn decode(bits: u64) -> Option<FmulScale> {
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
    pub(super) fn decode(bits: u64) -> HSwizzle {
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
    pub(super) fn decode(bits: u64) -> HMerge {
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
    pub(super) fn decode(bits: u64) -> HPrecision {
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
