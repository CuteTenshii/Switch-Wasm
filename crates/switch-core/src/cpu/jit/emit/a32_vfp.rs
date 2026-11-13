//! VFP in A32 blocks, as `cpu/a32/vfp.rs` runs it: arithmetic happens in `f64` and a
//! single is narrowed on the way out. Forms the interpreter refuses, fuses or leaves to
//! FPSCR's rounding mode stay on the interpreter.

use super::*;
use crate::cpu::a32::vfp::{expand_imm_f32, expand_imm_f64};

const F64_ADD: u8 = 0xA0;
const F64_SUB: u8 = 0xA1;
const F64_MUL: u8 = 0xA2;
const F64_DIV: u8 = 0xA3;
const F64_NEG: u8 = 0x9A;
const F64_SQRT: u8 = 0x9F;
const F64_EQ: u8 = 0x61;
const F64_NE: u8 = 0x62;
const F64_LT: u8 = 0x63;
const F64_PROMOTE_F32: u8 = 0xBB;
const F32_DEMOTE_F64: u8 = 0xB6;
const F64_CONVERT_I32_S: u8 = 0xB7;
const F64_CONVERT_I32_U: u8 = 0xB8;
const I32_REINTERPRET_F32: u8 = 0xBC;
const F32_REINTERPRET_I32: u8 = 0xBE;
const I64_REINTERPRET_F64: u8 = 0xBD;
const F64_REINTERPRET_I64: u8 = 0xBF;
/// `i32.trunc_sat_f64_s` and `_u`, after the `0xFC` prefix.
const TRUNC_SAT_S: u32 = 2;
const TRUNC_SAT_U: u32 = 3;

const SIGN32: i64 = 0x8000_0000;
const SIGN64: i64 = i64::MIN;

#[derive(Clone, Copy)]
enum Binary {
    Add,
    Sub,
    Mul,
    /// `VNMUL`: the product negated.
    NegMul,
    Div,
}

#[derive(Clone, Copy)]
enum Sign {
    Copy,
    Abs,
    Neg,
}

/// A VFP instruction the emitter writes, its fields settled. Register numbers are in the
/// numbering of their width: `Sn` or `Dn`.
#[derive(Clone, Copy)]
enum Vfp {
    /// `VLDR` and `VSTR`.
    Transfer {
        load: bool,
        double: bool,
        add: bool,
        rn: u8,
        reg: u8,
        offset: u32,
    },
    /// `VLDM`, `VSTM`, `VPUSH` and `VPOP`: `count` registers over `bytes`.
    Multiple {
        load: bool,
        double: bool,
        add: bool,
        wb: bool,
        rn: u8,
        reg: u8,
        count: u8,
        bytes: u32,
    },
    /// `VMOV` between two core registers and a D register or two S registers.
    CorePair {
        to_core: bool,
        double: bool,
        rt: u8,
        rt2: u8,
        reg: u8,
    },
    Binary {
        op: Binary,
        double: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `VMLA`, `VMLS`, `VNMLA` and `VNMLS`: unfused.
    Accumulate {
        negate_product: bool,
        negate_acc: bool,
        double: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Immediate {
        double: bool,
        rd: u8,
        bits: u64,
    },
    Sign {
        kind: Sign,
        double: bool,
        rd: u8,
        rm: u8,
    },
    Sqrt {
        double: bool,
        rd: u8,
        rm: u8,
    },
    /// `VCMP` and `VCMPE`, against `rm` or against zero.
    Compare {
        double: bool,
        rd: u8,
        rm: Option<u8>,
    },
    /// `VCVT` between single and double: `double` is the source's width.
    Precision {
        double: bool,
        rm: u8,
        dest: u8,
    },
    FromInt {
        double: bool,
        signed: bool,
        rd: u8,
        sm: u8,
    },
    ToInt {
        double: bool,
        signed: bool,
        rm: u8,
        sd: u8,
    },
    /// `VMOV Sn, Rt` and `VMOV Rt, Sn`.
    Single {
        to_core: bool,
        rt: u8,
        s: u8,
    },
    /// `VMRS` and `VMSR` of FPSCR; `VMRS APSR_nzcv` when `rt` is 15.
    Fpscr {
        to_core: bool,
        rt: u8,
    },
    /// `VMOV` between a core register and one lane of a D register.
    Lane {
        to_core: bool,
        signed: bool,
        rt: u8,
        d: u8,
        esize: u8,
        index: u8,
    },
    /// `VDUP` of a core register into one or two D registers.
    Dup {
        rt: u8,
        d: u8,
        esize: u8,
        registers: u8,
    },
}

/// Whether `insn` is one the emitter writes as VFP.
pub(super) fn vfp_writable(insn: u32) -> bool {
    plan(insn).is_some()
}

fn plan(insn: u32) -> Option<Vfp> {
    if insn >> 28 == 0xF || !matches!((insn >> 8) & 0xF, 10 | 11) {
        return None;
    }
    let double = (insn >> 8) & 1 != 0;
    match (insn >> 24) & 0xF {
        0b1100 | 0b1101 => load_store(insn, double),
        0b1110 if insn & 0x10 != 0 => transfer(insn, double),
        0b1110 => data(insn, double),
        _ => None,
    }
}

fn load_store(insn: u32, double: bool) -> Option<Vfp> {
    let d_bit = ((insn >> 22) & 1) as u8;
    let rn = ((insn >> 16) & 0xF) as u8;
    let vd = ((insn >> 12) & 0xF) as u8;
    let imm8 = insn & 0xFF;
    let load = (insn >> 20) & 1 != 0;
    let add = (insn >> 23) & 1 != 0;
    let reg = if double {
        (d_bit << 4) | vd
    } else {
        (vd << 1) | d_bit
    };
    match (insn >> 21) & 0xF {
        0b0010 => {
            let (rt, rt2) = (((insn >> 12) & 0xF) as u8, rn);
            if rt == 15 || rt2 == 15 {
                return None;
            }
            let vm = (insn & 0xF) as u8;
            let m_bit = ((insn >> 5) & 1) as u8;
            Some(Vfp::CorePair {
                to_core: load,
                double,
                rt,
                rt2,
                reg: if double {
                    (m_bit << 4) | vm
                } else {
                    (vm << 1) | m_bit
                },
            })
        }
        0b0000 | 0b0001 | 0b0011 | 0b1101 | 0b1111 => None,
        _ if (insn >> 24) & 1 != 0 && (insn >> 21) & 1 == 0 => Some(Vfp::Transfer {
            load,
            double,
            add,
            rn,
            reg,
            offset: imm8 * 4,
        }),
        _ => {
            let count = if double { imm8 / 2 } else { imm8 };
            let wb = (insn >> 21) & 1 != 0;
            if rn == 15 || count == 0 || count > 32 {
                return None;
            }
            Some(Vfp::Multiple {
                load,
                double,
                add,
                wb,
                rn,
                reg,
                count: count as u8,
                bytes: imm8 * 4,
            })
        }
    }
}

/// The register numbers of a data-processing instruction: destination, first and second source.
fn operands(insn: u32, double: bool) -> (u8, u8, u8) {
    let d_bit = ((insn >> 22) & 1) as u8;
    let n_bit = ((insn >> 7) & 1) as u8;
    let m_bit = ((insn >> 5) & 1) as u8;
    let vd = ((insn >> 12) & 0xF) as u8;
    let vn = ((insn >> 16) & 0xF) as u8;
    let vm = (insn & 0xF) as u8;
    if double {
        ((d_bit << 4) | vd, (n_bit << 4) | vn, (m_bit << 4) | vm)
    } else {
        ((vd << 1) | d_bit, (vn << 1) | n_bit, (vm << 1) | m_bit)
    }
}

fn data(insn: u32, double: bool) -> Option<Vfp> {
    let (rd, rn, rm) = operands(insn, double);
    let negate = (insn >> 6) & 1 != 0;
    let binary = |op| {
        Some(Vfp::Binary {
            op,
            double,
            rd,
            rn,
            rm,
        })
    };
    match ((insn >> 23) & 1, (insn >> 20) & 0b11) {
        (0, opc @ (0b00 | 0b01)) => Some(Vfp::Accumulate {
            negate_product: negate,
            negate_acc: opc == 0b01,
            double,
            rd,
            rn,
            rm,
        }),
        (0, 0b10) => binary(if negate { Binary::NegMul } else { Binary::Mul }),
        (0, _) => binary(if negate { Binary::Sub } else { Binary::Add }),
        (1, 0b00) => binary(Binary::Div),
        (1, 0b11) => extended(insn, double, rd, rm),
        _ => None,
    }
}

fn extended(insn: u32, double: bool, rd: u8, rm: u8) -> Option<Vfp> {
    let opc2 = (insn >> 16) & 0xF;
    let opc3 = (insn >> 6) & 0b11;
    if opc3 & 1 == 0 {
        let imm8 = ((insn >> 12) & 0xF0) | (insn & 0xF);
        return Some(Vfp::Immediate {
            double,
            rd,
            bits: if double {
                expand_imm_f64(imm8)
            } else {
                u64::from(expand_imm_f32(imm8))
            },
        });
    }
    let d_bit = ((insn >> 22) & 1) as u8;
    let vd = ((insn >> 12) & 0xF) as u8;
    let sign = |kind| {
        Some(Vfp::Sign {
            kind,
            double,
            rd,
            rm,
        })
    };
    match (opc2, opc3) {
        (0b0000, 0b01) => sign(Sign::Copy),
        (0b0000, _) => sign(Sign::Abs),
        (0b0001, 0b01) => sign(Sign::Neg),
        (0b0001, _) => Some(Vfp::Sqrt { double, rd, rm }),
        (0b0100 | 0b0101, _) => Some(Vfp::Compare {
            double,
            rd,
            rm: (opc2 == 0b0100).then_some(rm),
        }),
        (0b0111, _) => Some(Vfp::Precision {
            double,
            rm,
            dest: if double {
                (vd << 1) | d_bit
            } else {
                (d_bit << 4) | vd
            },
        }),
        (0b1000, _) => Some(Vfp::FromInt {
            double,
            signed: (insn >> 7) & 1 != 0,
            rd,
            sm: (((insn & 0xF) as u8) << 1) | ((insn >> 5) & 1) as u8,
        }),
        (0b1100 | 0b1101, _) => Some(Vfp::ToInt {
            double,
            signed: opc2 & 1 != 0,
            rm,
            sd: (vd << 1) | d_bit,
        }),
        _ => None,
    }
}

fn transfer(insn: u32, double: bool) -> Option<Vfp> {
    let to_core = (insn >> 20) & 1 != 0;
    let rt = ((insn >> 12) & 0xF) as u8;
    let vn = ((insn >> 16) & 0xF) as u8;
    if double {
        return lane_transfer(insn, to_core, rt);
    }
    if (insn >> 21) & 0b111 == 0b111 {
        if vn != 1 || (!to_core && rt == 15) {
            return None;
        }
        return Some(Vfp::Fpscr { to_core, rt });
    }
    if (insn >> 21) & 0b111 != 0 || rt == 15 {
        return None;
    }
    Some(Vfp::Single {
        to_core,
        rt,
        s: (vn << 1) | ((insn >> 7) & 1) as u8,
    })
}

fn lane_transfer(insn: u32, to_core: bool, rt: u8) -> Option<Vfp> {
    if rt == 15 {
        return None;
    }
    let d = (((insn >> 7) & 1) as u8) << 4 | ((insn >> 16) & 0xF) as u8;
    if !to_core && (insn >> 23) & 1 != 0 {
        let esize = match ((insn >> 22) & 1, (insn >> 5) & 1) {
            (1, 0) => 8,
            (0, 1) => 16,
            (0, 0) => 32,
            _ => return None,
        };
        return Some(Vfp::Dup {
            rt,
            d,
            esize,
            registers: 1 + ((insn >> 21) & 1) as u8,
        });
    }
    let opc = ((insn >> 21) & 0b11) << 2 | (insn >> 5) & 0b11;
    let (esize, index) = if opc & 0b1000 != 0 {
        (8, opc & 0b111)
    } else if opc & 0b0001 != 0 {
        (16, opc >> 1 & 0b11)
    } else if opc & 0b0010 == 0 {
        (32, opc >> 2 & 1)
    } else {
        return None;
    };
    Some(Vfp::Lane {
        to_core,
        signed: (insn >> 23) & 1 == 0 && esize != 32,
        rt,
        d,
        esize,
        index: index as u8,
    })
}

impl Emitter<'_> {
    /// The offset of `S<n>` from the state.
    pub(super) fn s_at(&self, n: u8) -> u32 {
        self.layout.vregs + 4 * u32::from(n)
    }

    /// The offset of `D<n>` from the state.
    pub(super) fn d_at(&self, n: u8) -> u32 {
        self.layout.vregs + 8 * u32::from(n)
    }

    /// Push `S<n>` or `D<n>` as an `f64`, a single widened.
    fn vfp_get(&mut self, n: u8, double: bool) {
        self.f.local_get(STATE);
        if double {
            self.f.i64_load(ALIGN_8, self.d_at(n));
            self.f.op_raw(F64_REINTERPRET_I64);
        } else {
            self.f.i32_load(ALIGN_4, self.s_at(n));
            self.f.op_raw(F32_REINTERPRET_I32);
            self.f.op_raw(F64_PROMOTE_F32);
        }
    }

    /// Pop an `f64` into `S<n>`, narrowed, or `D<n>`.
    fn vfp_put(&mut self, n: u8, double: bool) {
        if double {
            self.f.local_set(L_F64);
            self.f.local_get(STATE);
            self.f.local_get(L_F64);
            self.f.op_raw(I64_REINTERPRET_F64);
            self.f.i64_store(ALIGN_8, self.d_at(n));
        } else {
            self.f.op_raw(F32_DEMOTE_F64);
            self.f.local_set(L_F32);
            self.f.local_get(STATE);
            self.f.local_get(L_F32);
            self.f.op_raw(I32_REINTERPRET_F32);
            self.f.i32_store(ALIGN_4, self.s_at(n));
        }
    }

    /// Write VFP instruction `insn`, the `at`th of its block, at `pc`. The caller checked
    /// [`vfp_writable`].
    pub(super) fn a32_vfp(&mut self, insn: u32, at: usize, pc: u32) {
        let plan = plan(insn).expect("`vfp_writable` was checked");
        match plan {
            Vfp::Transfer {
                load,
                double,
                add,
                rn,
                reg,
                offset,
            } => {
                self.a32_read(rn, pc);
                self.f.i64_const(if add {
                    i64::from(offset)
                } else {
                    -i64::from(offset)
                });
                self.f.i64_add();
                self.f.i32_wrap_i64();
                self.f.local_set(L_ADDR);
                let n = if double { 8 } else { 4 };
                self.vfp_copy(&[reg], double, n, load, at);
            }
            Vfp::Multiple {
                load,
                double,
                add,
                wb,
                rn,
                reg,
                count,
                bytes,
            } => {
                let bytes = i64::from(bytes);
                self.a32_read(rn, pc);
                self.f.local_set(L_A);
                self.f.local_get(L_A);
                if !add {
                    self.f.i64_const(-bytes);
                    self.f.i64_add();
                }
                self.f.i32_wrap_i64();
                self.f.local_set(L_ADDR);
                let size = if double { 8 } else { 4 };
                let regs: Vec<u8> = (0..count).map(|i| reg.wrapping_add(i) & 0x1F).collect();
                self.vfp_copy(&regs, double, size, load, at);
                if wb {
                    self.f.local_get(L_A);
                    self.f.i64_const(if add { bytes } else { -bytes });
                    self.f.i64_add();
                    self.mask_to(false);
                    self.store_reg(rn);
                }
            }
            Vfp::CorePair {
                to_core,
                double,
                rt,
                rt2,
                reg,
            } => match (to_core, double) {
                (true, true) => {
                    self.f.local_get(STATE);
                    self.f.i64_load(ALIGN_8, self.d_at(reg));
                    self.f.local_set(L_T);
                    self.f.local_get(L_T);
                    self.mask_to(false);
                    self.store_reg(rt);
                    self.f.local_get(L_T);
                    self.f.i64_const(32);
                    self.f.i64_shr_u();
                    self.store_reg(rt2);
                }
                (false, true) => {
                    self.f.local_get(STATE);
                    self.a32_read(rt2, pc);
                    self.f.i64_const(32);
                    self.f.i64_shl();
                    self.a32_read(rt, pc);
                    self.f.i64_or();
                    self.f.i64_store(ALIGN_8, self.d_at(reg));
                }
                (true, false) => {
                    for (r, s) in [(rt, reg), (rt2, reg.wrapping_add(1) & 0x1F)] {
                        self.f.local_get(STATE);
                        self.f.i64_load32_u(ALIGN_4, self.s_at(s));
                        self.store_reg(r);
                    }
                }
                (false, false) => {
                    for (r, s) in [(rt, reg), (rt2, reg.wrapping_add(1) & 0x1F)] {
                        self.f.local_get(STATE);
                        self.a32_read(r, pc);
                        self.f.i64_store32(ALIGN_4, self.s_at(s));
                    }
                }
            },
            Vfp::Binary {
                op,
                double,
                rd,
                rn,
                rm,
            } => {
                self.vfp_get(rn, double);
                self.vfp_get(rm, double);
                self.f.op_raw(match op {
                    Binary::Add => F64_ADD,
                    Binary::Sub => F64_SUB,
                    Binary::Mul | Binary::NegMul => F64_MUL,
                    Binary::Div => F64_DIV,
                });
                if matches!(op, Binary::NegMul) {
                    self.f.op_raw(F64_NEG);
                }
                self.vfp_put(rd, double);
            }
            Vfp::Accumulate {
                negate_product,
                negate_acc,
                double,
                rd,
                rn,
                rm,
            } => {
                self.vfp_get(rn, double);
                self.vfp_get(rm, double);
                self.f.op_raw(F64_MUL);
                if negate_product {
                    self.f.op_raw(F64_NEG);
                }
                self.vfp_get(rd, double);
                if negate_acc {
                    self.f.op_raw(F64_NEG);
                }
                self.f.op_raw(F64_ADD);
                self.vfp_put(rd, double);
            }
            Vfp::Immediate { double, rd, bits } => {
                self.f.local_get(STATE);
                if double {
                    self.f.i64_const(bits as i64);
                    self.f.i64_store(ALIGN_8, self.d_at(rd));
                } else {
                    self.f.i32_const(bits as u32 as i32);
                    self.f.i32_store(ALIGN_4, self.s_at(rd));
                }
            }
            Vfp::Sign {
                kind,
                double,
                rd,
                rm,
            } => {
                self.f.local_get(STATE);
                self.f.local_get(STATE);
                if double {
                    self.f.i64_load(ALIGN_8, self.d_at(rm));
                    match kind {
                        Sign::Copy => {}
                        Sign::Abs => {
                            self.f.i64_const(!SIGN64);
                            self.f.i64_and();
                        }
                        Sign::Neg => {
                            self.f.i64_const(SIGN64);
                            self.f.i64_xor();
                        }
                    }
                    self.f.i64_store(ALIGN_8, self.d_at(rd));
                } else {
                    self.f.i32_load(ALIGN_4, self.s_at(rm));
                    match kind {
                        Sign::Copy => {}
                        Sign::Abs => {
                            self.f.i32_const(!SIGN32 as i32);
                            self.f.i32_and();
                        }
                        Sign::Neg => {
                            self.f.i32_const(SIGN32 as i32);
                            self.f.i32_xor();
                        }
                    }
                    self.f.i32_store(ALIGN_4, self.s_at(rd));
                }
            }
            Vfp::Sqrt { double, rd, rm } => {
                self.vfp_get(rm, double);
                self.f.op_raw(F64_SQRT);
                self.vfp_put(rd, double);
            }
            Vfp::Compare { double, rd, rm } => self.vfp_compare(double, rd, rm),
            Vfp::Precision { double, rm, dest } => {
                self.vfp_get(rm, double);
                self.vfp_put(dest, !double);
            }
            Vfp::FromInt {
                double,
                signed,
                rd,
                sm,
            } => {
                self.f.local_get(STATE);
                self.f.i32_load(ALIGN_4, self.s_at(sm));
                self.f.op_raw(if signed {
                    F64_CONVERT_I32_S
                } else {
                    F64_CONVERT_I32_U
                });
                self.vfp_put(rd, double);
            }
            Vfp::ToInt {
                double,
                signed,
                rm,
                sd,
            } => {
                self.f.local_get(STATE);
                self.vfp_get(rm, double);
                self.f
                    .trunc_sat(if signed { TRUNC_SAT_S } else { TRUNC_SAT_U });
                self.f.i32_store(ALIGN_4, self.s_at(sd));
            }
            Vfp::Single { to_core, rt, s } => {
                if to_core {
                    self.f.local_get(STATE);
                    self.f.i64_load32_u(ALIGN_4, self.s_at(s));
                    self.store_reg(rt);
                } else {
                    self.f.local_get(STATE);
                    self.a32_read(rt, pc);
                    self.f.i64_store32(ALIGN_4, self.s_at(s));
                }
            }
            Vfp::Fpscr { to_core, rt } => self.vfp_fpscr(to_core, rt, pc),
            Vfp::Lane {
                to_core,
                signed,
                rt,
                d,
                esize,
                index,
            } => {
                let at = self.d_at(d) + u32::from(index) * u32::from(esize / 8);
                self.f.local_get(STATE);
                if to_core {
                    match (esize, signed) {
                        (8, false) => self.f.i64_load8_u(at),
                        (8, true) => self.f.i64_load8_s(at),
                        (16, false) => self.f.i64_load16_u(UNALIGNED, at),
                        (16, true) => self.f.i64_load16_s(UNALIGNED, at),
                        _ => self.f.i64_load32_u(UNALIGNED, at),
                    }
                    self.mask_to(false);
                    self.store_reg(rt);
                } else {
                    self.a32_read(rt, pc);
                    match esize {
                        8 => self.f.i64_store8(at),
                        16 => self.f.i64_store16(UNALIGNED, at),
                        _ => self.f.i64_store32(UNALIGNED, at),
                    }
                }
            }
            Vfp::Dup {
                rt,
                d,
                esize,
                registers,
            } => {
                let (mask, ones) = match esize {
                    8 => (0xFF, 0x0101_0101_0101_0101),
                    16 => (0xFFFF, 0x0001_0001_0001_0001),
                    _ => (0xFFFF_FFFF, 0x0000_0001_0000_0001),
                };
                self.a32_read(rt, pc);
                self.f.i64_const(mask);
                self.f.i64_and();
                self.f.i64_const(ones);
                self.f.i64_mul();
                self.f.local_set(L_T);
                for i in 0..registers {
                    self.f.local_get(STATE);
                    self.f.local_get(L_T);
                    self.f
                        .i64_store(ALIGN_8, self.d_at(d.wrapping_add(i) & 0x1F));
                }
            }
        }
    }

    /// Move `regs` between the guest memory at `L_ADDR` and the register file, checking the
    /// whole range first so a hand-back leaves nothing half moved.
    pub(super) fn vfp_copy(&mut self, regs: &[u8], double: bool, size: u32, load: bool, at: usize) {
        let total = size * regs.len() as u32;
        if load {
            self.page_for_read(total, at);
        } else {
            self.page_for_write(total, at);
        }
        for (i, &reg) in regs.iter().enumerate() {
            let offset = size * i as u32;
            if load {
                self.f.local_get(STATE);
                self.in_page();
                if double {
                    self.f.i64_load(UNALIGNED, offset);
                    self.f.i64_store(ALIGN_8, self.d_at(reg));
                } else {
                    self.f.i32_load(UNALIGNED, offset);
                    self.f.i32_store(ALIGN_4, self.s_at(reg));
                }
            } else {
                self.in_page();
                self.f.local_get(STATE);
                if double {
                    self.f.i64_load(ALIGN_8, self.d_at(reg));
                    self.f.i64_store(UNALIGNED, offset);
                } else {
                    self.f.i32_load(ALIGN_4, self.s_at(reg));
                    self.f.i32_store(UNALIGNED, offset);
                }
            }
        }
    }

    /// `VCMP`: NZCV of the comparison into FPSCR's flags.
    fn vfp_compare(&mut self, double: bool, rd: u8, rm: Option<u8>) {
        self.vfp_get(rd, double);
        self.f.local_set(L_F64);
        match rm {
            Some(rm) => self.vfp_get(rm, double),
            None => self.f.f64_const(0.0),
        }
        self.f.local_set(L_F64B);
        // `select` keeps its first operand when the condition holds.
        self.f.i32_const(0x6000_0000);
        self.f.i32_const(0x2000_0000);
        self.f.local_get(L_F64);
        self.f.local_get(L_F64B);
        self.f.op_raw(F64_EQ);
        self.f.select();
        self.f.local_set(L_C);
        self.f.i32_const(0x8000_0000u32 as i32);
        self.f.local_get(L_C);
        self.f.local_get(L_F64);
        self.f.local_get(L_F64B);
        self.f.op_raw(F64_LT);
        self.f.select();
        self.f.local_set(L_C);
        self.f.i32_const(0x3000_0000);
        self.f.local_get(L_C);
        self.f.local_get(L_F64);
        self.f.local_get(L_F64);
        self.f.op_raw(F64_NE);
        self.f.local_get(L_F64B);
        self.f.local_get(L_F64B);
        self.f.op_raw(F64_NE);
        self.f.i32_or();
        self.f.select();
        self.f.local_set(L_C);
        self.f.local_get(STATE);
        self.f.local_get(L_C);
        self.f.i32_store(ALIGN_4, self.layout.fpscr_nzcv);
    }

    /// `VMRS` and `VMSR` of FPSCR, held as its flags, FPCR's control bits and FPSR's status bits.
    fn vfp_fpscr(&mut self, to_core: bool, rt: u8, pc: u32) {
        const CONTROL: i32 = 0x07C0_0000;
        const STATUS: i32 = 0x9F;
        const FLAGS: i32 = 0xF000_0000u32 as i32;
        if to_core && rt == 15 {
            self.f.local_get(STATE);
            self.f.i32_load(ALIGN_4, self.layout.fpscr_nzcv);
            self.f.i32_const(FLAGS);
            self.f.i32_and();
            self.store_nzcv();
        } else if to_core {
            self.f.local_get(STATE);
            self.f.i32_load(ALIGN_4, self.layout.fpscr_nzcv);
            for (at, mask) in [(self.layout.fpcr, CONTROL), (self.layout.fpsr, STATUS)] {
                self.f.local_get(STATE);
                self.f.i32_load(ALIGN_4, at);
                self.f.i32_const(mask);
                self.f.i32_and();
                self.f.i32_or();
            }
            self.f.i64_extend_i32_u();
            self.store_reg(rt);
        } else {
            self.a32_read(rt, pc);
            self.f.i32_wrap_i64();
            self.f.local_set(L_C);
            for (at, mask) in [
                (self.layout.fpscr_nzcv, FLAGS),
                (self.layout.fpcr, CONTROL),
                (self.layout.fpsr, STATUS),
            ] {
                self.f.local_get(STATE);
                self.f.local_get(L_C);
                self.f.i32_const(mask);
                self.f.i32_and();
                self.f.i32_store(ALIGN_4, at);
            }
        }
    }
}
