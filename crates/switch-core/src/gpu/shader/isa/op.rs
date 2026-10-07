//! The decoded instruction and its op.

use super::*;

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
