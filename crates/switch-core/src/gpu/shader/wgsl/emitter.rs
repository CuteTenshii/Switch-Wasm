//! The emitter state and the operand and modifier builders it shares.

use crate::gpu::shader::compiled::Compiled;
use crate::gpu::shader::isa::{
    BoolOp, FCmp, FMod, FRound, HMerge, HSwizzle, ICmp, MemSize, Op, Operand, Pred, TexDim, RZ,
};
use crate::gpu::texture::TextureSlot;
use std::collections::BTreeSet;

pub(super) struct Emitter<'a> {
    pub(super) program: &'a Compiled,
    pub(super) body: String,
    pub(super) indent: usize,
    /// Collected while emitting, not by a separate pass.
    pub(super) regs: BTreeSet<u8>,
    pub(super) preds: BTreeSet<u8>,
    pub(super) helpers: BTreeSet<&'static str>,
    pub(super) uses_carry: bool,
    /// Whether the zero, sign or overflow flags are used.
    pub(super) uses_flags: bool,
    pub(super) uses_stack: bool,
    pub(super) quad: Option<usize>,
    pub(super) quad_swap: Option<usize>,
    /// The interface the emitted text reaches through.
    pub(super) loads: BTreeSet<usize>,
    pub(super) stores: BTreeSet<usize>,
    pub(super) centroid_loads: BTreeSet<usize>,
    pub(super) banks: BTreeSet<u8>,
    pub(super) textures: Vec<(TextureSlot, TexDim, bool)>,
    pub(super) globals: Vec<(u8, u16)>,
    /// Counter naming `let` bindings.
    temps: usize,
    pub(super) block: usize,
    /// Textures only `txq`'d, bound as 2D.
    pub(super) queried: Vec<TextureSlot>,
    pub(super) texture_offsets: Vec<(i32, i32)>,
}

impl<'a> Emitter<'a> {
    pub(super) fn new(program: &'a Compiled) -> Emitter<'a> {
        Emitter {
            program,
            body: String::new(),
            indent: 4,
            regs: BTreeSet::new(),
            preds: BTreeSet::new(),
            helpers: BTreeSet::new(),
            uses_carry: false,
            uses_flags: false,
            uses_stack: false,
            quad: None,
            quad_swap: None,
            loads: BTreeSet::new(),
            stores: BTreeSet::new(),
            centroid_loads: BTreeSet::new(),
            banks: BTreeSet::new(),
            textures: Vec::new(),
            globals: Vec::new(),
            temps: 0,
            block: 0,
            queried: Vec::new(),
            texture_offsets: Vec::new(),
        }
    }

    /// A `ldg` base descriptor from a constant bank plus a register offset:
    ///
    /// ```text
    /// iadd.cout r10, r7, c0[0x110]      // low half, carrying out
    /// iadd.cin  r11, RZ, c0[0x114]      // high half, carrying in
    /// ldg       r10, [r10]
    /// ```
    pub(super) fn global_base(&self, at: usize, addr: u8) -> Option<(u8, u16, u8)> {
        self.indexed_global_base(at, addr)
            .or_else(|| self.direct_global_base(at, addr))
    }

    /// A descriptor loaded whole into the address pair, read with no index:
    ///
    /// ```text
    /// ldc.64 r0, c1[0x40]
    /// ldg.64 r2, [r0]
    /// ```
    fn direct_global_base(&self, at: usize, addr: u8) -> Option<(u8, u16, u8)> {
        let lo = self.sole_writer(at, addr)?;
        let hi = self.sole_writer(at, addr.wrapping_add(1))?;
        match (self.program.op(lo), self.program.op(hi)) {
            (
                Op::Ldc {
                    dst,
                    bank,
                    offset,
                    idx: RZ,
                    size: MemSize::B64,
                },
                _,
            ) if lo == hi && dst == addr => Some((bank, (offset as u32 & 0xffff) as u16, RZ)),
            (
                Op::Mov {
                    src: Operand::Const { bank, offset },
                    ..
                },
                Op::Mov {
                    src:
                        Operand::Const {
                            bank: hi_bank,
                            offset: hi_offset,
                        },
                    ..
                },
            ) if hi_bank == bank && hi_offset == offset.wrapping_add(4) => Some((bank, offset, RZ)),
            _ => None,
        }
    }

    /// The indexed form of [`Emitter::global_base`].
    fn indexed_global_base(&self, at: usize, addr: u8) -> Option<(u8, u16, u8)> {
        let (mut lo, mut hi) = (None, None);
        for i in (0..at).rev() {
            match self.program.op(i) {
                Op::Iadd {
                    dst,
                    a,
                    b: Operand::Const { bank, offset },
                    aneg: false,
                    bneg: false,
                    cout: true,
                    ..
                } if dst == addr && lo.is_none() => lo = Some((bank, offset, a)),
                // The high half adds only the carry.
                Op::Iadd {
                    dst,
                    a: RZ,
                    b: Operand::Const { bank, offset },
                    aneg: false,
                    bneg: false,
                    cin: true,
                    ..
                } if dst == addr.wrapping_add(1) && hi.is_none() => hi = Some((bank, offset)),
                // Any other write breaks the pattern.
                other => {
                    let writes = super::interp::writes(&other);
                    if writes.contains(&addr) && lo.is_none() {
                        return None;
                    }
                    if writes.contains(&addr.wrapping_add(1)) && hi.is_none() {
                        return None;
                    }
                }
            }
            if lo.is_some() && hi.is_some() {
                break;
            }
        }
        let ((bank, offset, index), (hi_bank, hi_offset)) = (lo?, hi?);
        if hi_bank != bank || hi_offset != offset.wrapping_add(4) {
            return None;
        }
        Some((bank, offset, index))
    }

    /// The constant word a bindless `tex.b` reads its handle from:
    ///
    /// ```text
    /// ldc   r2, c3[0x10]
    /// tex.b r0, r4, r2, 0x2, 2D, 0xf
    /// ```
    pub(super) fn bindless_slot(&self, at: usize, reg: u8) -> Option<TextureSlot> {
        let writer = self.sole_writer(at, reg)?;
        match self.program.op(writer) {
            Op::Mov {
                src: Operand::Const { bank, offset },
                ..
            } => Some(TextureSlot::Bindless { bank, offset }),
            // Wide loads fill consecutive registers from consecutive words.
            Op::Ldc {
                dst,
                bank,
                offset,
                idx: RZ,
                size,
            } if size.bytes() >= 4 => {
                let word = u32::from(reg.wrapping_sub(dst)) * 4;
                let offset = (offset as u32).wrapping_add(word) & 0xffff;
                Some(TextureSlot::Bindless {
                    bank,
                    offset: offset as u16,
                })
            }
            _ => None,
        }
    }

    /// The unguarded instruction whose value `reg` holds at `at`: the nearest
    /// earlier write in the block, else the program's only write.
    fn sole_writer(&self, at: usize, reg: u8) -> Option<usize> {
        let writes_reg = |i: usize| super::interp::writes(&self.program.op(i)).contains(&reg);
        let writer = match (self.block..at).rev().find(|&i| writes_reg(i)) {
            Some(writer) => writer,
            None => {
                let mut writers = (0..self.program.len()).filter(|&i| i != at && writes_reg(i));
                let only = writers.next()?;
                if writers.next().is_some() {
                    return None;
                }
                only
            }
        };
        (self.program.pred(writer) == Pred::ALWAYS).then_some(writer)
    }

    /// The immediate `reg` holds at `at`. See [`Emitter::sole_writer`].
    pub(super) fn constant_in(&self, at: usize, reg: u8) -> Option<u32> {
        match self.program.op(self.sole_writer(at, reg)?) {
            Op::Mov32i { imm, .. } => Some(imm),
            Op::Mov {
                src: Operand::Imm(imm),
                ..
            } => Some(imm),
            // A copy of RZ is a zero.
            Op::Mov {
                src: Operand::Reg(RZ),
                ..
            } => Some(0),
            _ => None,
        }
    }

    pub(super) fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.body.push_str("  ");
        }
        self.body.push_str(text);
        self.body.push('\n');
    }

    pub(super) fn need(&mut self, helper: &'static str) {
        self.helpers.insert(helper);
        if helper == "mulhi_s" {
            self.helpers.insert("mulhi_u");
        }
        if helper == "shf" {
            self.helpers.insert("shl32");
            self.helpers.insert("shr32");
        }
    }

    /// A `let` holding `value`, for when it is read more than once.
    pub(super) fn bind(&mut self, value: &str) -> String {
        self.temps += 1;
        let name = format!("t{}", self.temps);
        self.line(&format!("let {name} = {value};"));
        name
    }

    // ---- operands ----

    pub(super) fn r(&mut self, reg: u8) -> String {
        if reg == RZ {
            return "0u".to_string();
        }
        self.regs.insert(reg);
        format!("r{reg}")
    }

    pub(super) fn f(&mut self, reg: u8) -> String {
        let value = self.r(reg);
        format!("bitcast<f32>({value})")
    }

    pub(super) fn operand(&mut self, operand: Operand) -> String {
        match operand {
            Operand::Reg(reg) => self.r(reg),
            Operand::Imm(value) => format!("{value}u"),
            Operand::Const { bank, offset } => {
                self.banks.insert(bank);
                format!("cbRead({bank}u, {offset}u)")
            }
        }
    }

    pub(super) fn operand_f(&mut self, operand: Operand) -> String {
        let value = self.operand(operand);
        let value = self.runtime_if_non_finite(value, |bits| f32::from_bits(bits).is_finite());
        format!("bitcast<f32>({value})")
    }

    /// Bind non-finite literals to a `let`: WGSL rejects them as constant expressions.
    fn runtime_if_non_finite(&mut self, bits: String, finite: fn(u32) -> bool) -> String {
        match bits.strip_suffix('u').and_then(|n| n.parse::<u32>().ok()) {
            Some(value) if !finite(value) => self.bind(&bits),
            _ => bits,
        }
    }

    fn p(&mut self, pred: u8) -> String {
        if pred >= 7 {
            return "true".to_string();
        }
        self.preds.insert(pred);
        format!("p{pred}")
    }

    /// Whether a guard or source predicate holds.
    pub(super) fn holds(&mut self, pred: Pred) -> String {
        if pred.reg >= 7 {
            return if pred.negate { "false" } else { "true" }.to_string();
        }
        let name = self.p(pred.reg);
        if pred.negate {
            format!("!{name}")
        } else {
            name
        }
    }

    // ---- destinations ----

    /// Write a register; `RZ` discards.
    pub(super) fn set_r(&mut self, dst: u8, value: &str) {
        if dst == RZ {
            return;
        }
        self.regs.insert(dst);
        self.line(&format!("r{dst} = {value};"));
    }

    pub(super) fn set_f(&mut self, dst: u8, value: &str) {
        self.set_r(dst, &format!("bitcast<u32>({value})"));
    }

    /// Write a predicate; `PT` and above are not writable.
    pub(super) fn set_p(&mut self, dst: u8, value: &str) {
        if dst >= 7 {
            return;
        }
        self.preds.insert(dst);
        self.line(&format!("p{dst} = {value};"));
    }

    // ---- expression builders ----

    pub(super) fn fmod(&mut self, modifier: FMod, value: String) -> String {
        let value = if modifier.abs {
            format!("abs({value})")
        } else {
            value
        };
        if modifier.neg {
            format!("-({value})")
        } else {
            value
        }
    }

    pub(super) fn flush(&mut self, ftz: bool, value: String) -> String {
        if ftz {
            self.need("ftz");
            format!("ftz({value})")
        } else {
            value
        }
    }

    pub(super) fn saturate(&mut self, sat: bool, value: String) -> String {
        if sat {
            self.need("fsat");
            format!("fsat({value})")
        } else {
            value
        }
    }

    // ---- half-precision ----

    /// One source's two lanes, flushed and modified.
    pub(super) fn half_source(&mut self, bits: String, m: FMod, sw: HSwizzle, ftz: bool) -> String {
        let bits = match sw {
            HSwizzle::F32 => {
                self.runtime_if_non_finite(bits, |bits| f32::from_bits(bits).is_finite())
            }
            _ => self.runtime_if_non_finite(bits, |bits| {
                [bits, bits >> 16]
                    .iter()
                    .all(|half| (half >> 10) & 0x1f != 0x1f)
            }),
        };
        let lanes = match sw {
            HSwizzle::H1H0 => format!("unpack2x16float({bits})"),
            HSwizzle::H0H0 => format!("unpack2x16float({bits}).xx"),
            HSwizzle::H1H1 => format!("unpack2x16float({bits}).yy"),
            // Not a pair at all: one f32 that both lanes read.
            HSwizzle::F32 => format!("vec2<f32>(bitcast<f32>({bits}))"),
        };
        let lanes = if !ftz {
            lanes
        } else if sw == HSwizzle::F32 {
            self.need("ftz2");
            format!("ftz2({lanes})")
        } else {
            self.need("hftz");
            format!("hftz({lanes})")
        };
        self.fmod(m, lanes)
    }

    pub(super) fn half_saturate(&mut self, sat: bool, value: String) -> String {
        if sat {
            self.need("fsat2");
            format!("fsat2({value})")
        } else {
            value
        }
    }

    pub(super) fn half_merge(&mut self, dst: u8, lanes: &str, merge: HMerge) -> String {
        match merge {
            HMerge::H1H0 => format!("pack2x16float({lanes})"),
            HMerge::F32 => format!("bitcast<u32>(({lanes}).x)"),
            HMerge::MrgH0 => {
                let kept = self.r(dst);
                format!("(({kept} & 0xffff0000u) | (pack2x16float({lanes}) & 0x0000ffffu))")
            }
            HMerge::MrgH1 => {
                let kept = self.r(dst);
                format!("(({kept} & 0x0000ffffu) | (pack2x16float({lanes}) & 0xffff0000u))")
            }
        }
    }

    /// `.fmz` zeroing as a lane-wise condition.
    pub(super) fn half_zeroed(&mut self, a: &str, b: &str) -> String {
        format!("(({a} == vec2<f32>(0.0)) | ({b} == vec2<f32>(0.0)))")
    }

    /// Each lane's comparison combined with the source predicate.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn half_compare(
        &mut self,
        a: u8,
        am: FMod,
        asw: HSwizzle,
        b: Operand,
        bm: FMod,
        bsw: HSwizzle,
        cmp: FCmp,
        bop: BoolOp,
        src: Pred,
        ftz: bool,
    ) -> (String, String) {
        let x = self.r(a);
        let x = self.half_source(x, am, asw, ftz);
        let x = self.bind(&x);
        let y = self.operand(b);
        let y = self.half_source(y, bm, bsw, ftz);
        let y = self.bind(&y);
        let guard = self.holds(src);
        let guard = self.bind(&guard);
        let low = self.float_compare(cmp, &format!("{x}.x"), &format!("{y}.x"));
        let low = self.combine(bop, &low, &guard);
        let low = self.bind(&low);
        let high = self.float_compare(cmp, &format!("{x}.y"), &format!("{y}.y"));
        let high = self.combine(bop, &high, &guard);
        let high = self.bind(&high);
        (low, high)
    }

    pub(super) fn ineg(&mut self, neg: bool, value: String) -> String {
        if neg {
            format!("(0u - ({value}))")
        } else {
            value
        }
    }

    pub(super) fn inv(&mut self, invert: bool, value: String) -> String {
        if invert {
            format!("(~({value}))")
        } else {
            value
        }
    }

    pub(super) fn float_compare(&mut self, cmp: FCmp, a: &str, b: &str) -> String {
        // WGSL has no `isNan`.
        let unordered = format!("(({a}) != ({a}) || ({b}) != ({b}))");
        match cmp {
            FCmp::Never => "false".to_string(),
            FCmp::Lt => format!("(({a}) < ({b}))"),
            FCmp::Eq => format!("(({a}) == ({b}))"),
            FCmp::Le => format!("(({a}) <= ({b}))"),
            FCmp::Gt => format!("(({a}) > ({b}))"),
            FCmp::Ge => format!("(({a}) >= ({b}))"),
            FCmp::Ne => format!("(!{unordered} && ({a}) != ({b}))"),
            FCmp::Num => format!("(!{unordered})"),
            FCmp::Nan => unordered,
            FCmp::LtU => format!("({unordered} || ({a}) < ({b}))"),
            FCmp::EqU => format!("({unordered} || ({a}) == ({b}))"),
            FCmp::LeU => format!("({unordered} || ({a}) <= ({b}))"),
            FCmp::GtU => format!("({unordered} || ({a}) > ({b}))"),
            FCmp::GeU => format!("({unordered} || ({a}) >= ({b}))"),
            FCmp::NeU => format!("({unordered} || ({a}) != ({b}))"),
            FCmp::Always => "true".to_string(),
        }
    }

    pub(super) fn int_compare(&mut self, cmp: ICmp, a: &str, b: &str, signed: bool) -> String {
        let (a, b) = if signed {
            (format!("bitcast<i32>({a})"), format!("bitcast<i32>({b})"))
        } else {
            (a.to_string(), b.to_string())
        };
        match cmp {
            ICmp::Never => "false".to_string(),
            ICmp::Lt => format!("(({a}) < ({b}))"),
            ICmp::Eq => format!("(({a}) == ({b}))"),
            ICmp::Le => format!("(({a}) <= ({b}))"),
            ICmp::Gt => format!("(({a}) > ({b}))"),
            ICmp::Ne => format!("(({a}) != ({b}))"),
            ICmp::Ge => format!("(({a}) >= ({b}))"),
            ICmp::Always => "true".to_string(),
        }
    }

    pub(super) fn combine(&mut self, op: BoolOp, a: &str, b: &str) -> String {
        match op {
            BoolOp::And => format!("({a} && {b})"),
            BoolOp::Or => format!("({a} || {b})"),
            BoolOp::Xor => format!("({a} != {b})"),
        }
    }

    pub(super) fn set_result(&mut self, taken: &str, bf: bool) -> String {
        let one = if bf { "0x3f800000u" } else { "0xffffffffu" };
        format!("select(0u, {one}, {taken})")
    }

    pub(super) fn round(&mut self, mode: FRound, value: String) -> String {
        // WGSL's `round` breaks ties to even, matching `.rn`.
        match mode {
            FRound::Nearest => format!("round({value})"),
            FRound::Floor => format!("floor({value})"),
            FRound::Ceil => format!("ceil({value})"),
            FRound::Trunc => format!("trunc({value})"),
        }
    }
}
