//! A small disassembler for the A64 subset the interpreter implements, for debug traces.

use std::fmt::Write;

fn sext(v: u32, bits: u32) -> i64 {
    let sign = 1u64 << (bits - 1);
    let mask = (1u64 << bits) - 1;
    let v = (v as u64) & mask;
    (if v & sign != 0 { v | !mask } else { v }) as i64
}

/// Hex with an explicit sign, e.g. `#-0x20`.
fn simm(v: i64) -> String {
    if v < 0 {
        format!("-0x{:x}", v.unsigned_abs())
    } else {
        format!("0x{:x}", v)
    }
}

fn reg64(i: u32) -> String {
    if i == 31 {
        "sp".into()
    } else {
        format!("x{}", i)
    }
}
fn reg32(i: u32) -> String {
    if i == 31 {
        "wsp".into()
    } else {
        format!("w{}", i)
    }
}
fn zr64(i: u32) -> String {
    if i == 31 {
        "xzr".into()
    } else {
        format!("x{}", i)
    }
}
fn zr32(i: u32) -> String {
    if i == 31 {
        "wzr".into()
    } else {
        format!("w{}", i)
    }
}

fn cond(c: u32) -> &'static str {
    match c & 0xF {
        0x0 => "eq",
        0x1 => "ne",
        0x2 => "cs",
        0x3 => "cc",
        0x4 => "mi",
        0x5 => "pl",
        0x6 => "vs",
        0x7 => "vc",
        0x8 => "hi",
        0x9 => "ls",
        0xA => "ge",
        0xB => "lt",
        0xC => "gt",
        0xD => "le",
        _ => "al",
    }
}

fn shift_name(st: u32) -> &'static str {
    match st {
        0 => "lsl",
        1 => "lsr",
        2 => "asr",
        _ => "ror",
    }
}

fn ext_name(opt: u32) -> &'static str {
    match opt {
        0b000 => "uxtb",
        0b001 => "uxth",
        0b010 => "uxtw",
        0b011 => "uxtx",
        0b100 => "sxtb",
        0b101 => "sxth",
        0b110 => "sxtw",
        _ => "sxtx",
    }
}

/// Register name for a load/store width: sz 3 => 64-bit, otherwise 32-bit.
fn mem_reg(sz: u32, i: u32) -> String {
    if sz == 3 {
        zr64(i)
    } else {
        zr32(i)
    }
}

/// Disassemble one A64 instruction into a readable mnemonic line.
pub fn disassemble(insn: u32) -> String {
    let mut s = String::with_capacity(48);
    if disasm_into(insn, &mut s).is_err() {
        s = format!(".word {:#010x}", insn);
    }
    s
}

fn disasm_into(insn: u32, s: &mut String) -> std::fmt::Result {
    let op26 = (insn >> 26) & 0x3F;
    if op26 == 0b000101 {
        let imm = sext(insn & 0x3FF_FFFF, 26) << 2;
        return write!(s, "b #{}", simm(imm));
    }
    if op26 == 0b100101 {
        let imm = sext(insn & 0x3FF_FFFF, 26) << 2;
        return write!(s, "bl #{}", simm(imm));
    }

    // load literal
    if ((insn >> 27) & 0b111) == 0b011 && ((insn >> 26) & 1) == 0 && ((insn >> 24) & 0b11) == 0b00 {
        let rt = insn & 0x1F;
        let imm = sext((insn >> 5) & 0x7_FFFF, 19) << 2;
        let (name, sz) = match (insn >> 30) & 0b11 {
            0b00 => ("ldr", 2),
            0b01 => ("ldr", 3),
            0b10 => ("ldrsw", 2),
            _ => return write!(s, "prfm #{}", simm(imm)), // prefetch literal
        };
        return write!(s, "{} {}, #{}", name, mem_reg(sz, rt), simm(imm));
    }

    // branch register
    if ((insn >> 25) & 0x7F) == 0b1101011 {
        let opc = (insn >> 21) & 0xF;
        if ((insn >> 16) & 0x1F) == 0x1F && ((insn >> 10) & 0x3F) == 0 {
            let rn = (insn >> 5) & 0x1F;
            return match opc {
                0b0000 => write!(s, "br {}", zr64(rn)),
                0b0001 => write!(s, "blr {}", zr64(rn)),
                0b0010 => write!(s, "ret {}", zr64(rn)),
                _ => write!(s, "branch.opc{:b}", opc),
            };
        }
    }

    // exceptions
    if ((insn >> 24) & 0xFF) == 0b11010100 {
        let kind = (insn >> 21) & 0b111;
        let imm = (insn >> 5) & 0xFFFF;
        return match kind {
            0b000 if (insn & 0x1F) == 0b00001 => write!(s, "svc #{:#x}", imm),
            0b001 => write!(s, "brk #{:#x}", imm),
            _ => write!(s, "exc.{:b}", kind),
        };
    }

    // system
    if ((insn >> 22) & 0x3FF) == 0b1101010100 {
        return disasm_system(insn, s);
    }

    // conditional branch
    if ((insn >> 24) & 0xFF) == 0b01010100 {
        let imm = sext((insn >> 5) & 0x7_FFFF, 19) << 2;
        let c = insn & 0xF;
        return write!(s, "b.{} #{}", cond(c), simm(imm));
    }

    // compare & branch
    if ((insn >> 25) & 0x3F) == 0b011010 {
        let rt = insn & 0x1F;
        let sf = (insn >> 31) & 1;
        let nz = (insn >> 24) & 1;
        let imm = sext((insn >> 5) & 0x7_FFFF, 19) << 2;
        let r = if sf == 1 { zr64(rt) } else { zr32(rt) };
        let op = if nz == 1 { "cbnz" } else { "cbz" };
        return write!(s, "{} {}, #{}", op, r, simm(imm));
    }

    // test bit & branch
    if ((insn >> 25) & 0x3F) == 0b011011 {
        let rt = insn & 0x1F;
        let sf = (insn >> 31) & 1;
        let nz = (insn >> 24) & 1;
        let bit = ((insn >> 31) & 1) << 5 | ((insn >> 19) & 0x1F);
        let imm = sext((insn >> 5) & 0x3FFF, 14) << 2;
        let r = if sf == 1 { zr64(rt) } else { zr32(rt) };
        let op = if nz == 1 { "tbnz" } else { "tbz" };
        return write!(s, "{} {}, #{}, #{}", op, r, bit, simm(imm));
    }

    if disasm_ld_st(insn, s)? {
        return Ok(());
    }

    // ADR/ADRP: fixed bits[28:24] == 10000 (bits[30:29] carry immlo).
    if ((insn >> 24) & 0x1F) == 0b10000 {
        let rd = insn & 0x1F;
        let immhi = (insn >> 5) & 0x7_FFFF;
        let immlo = (insn >> 29) & 0b11;
        let imm = sext((immhi << 2) | immlo, 21);
        let page = (insn >> 31) & 1;
        let name = if page == 1 { "adrp" } else { "adr" };
        return write!(s, "{} {}, #{}", name, zr64(rd), simm(imm));
    }

    if disasm_dp_imm(insn, s)? {
        return Ok(());
    }
    if disasm_dp_reg(insn, s)? {
        return Ok(());
    }

    write!(s, ".word {:#010x}", insn)
}

fn disasm_system(insn: u32, s: &mut String) -> std::fmt::Result {
    let l = (insn >> 21) & 1;
    let op0 = (insn >> 19) & 0b11;
    let op1 = (insn >> 16) & 0b111;
    let crn = (insn >> 12) & 0xF;
    let crm = (insn >> 8) & 0xF;
    let op2 = (insn >> 5) & 0b111;
    let rt = insn & 0x1F;

    if (insn >> 16) & 0xFFFF == 0xD503 {
        if insn == 0xD503_201F {
            return write!(s, "nop");
        }
        // CRn == 3 is the barrier space rather than the hint space.
        if crn == 3 {
            let name = match op2 {
                0b010 => "clrex",
                0b100 => "dsb",
                0b101 => "dmb",
                0b110 => "isb",
                _ => "hint",
            };
            return if name == "hint" {
                write!(s, "hint")
            } else if name == "clrex" {
                write!(s, "clrex")
            } else {
                write!(s, "{name} #{crm:#x}")
            };
        }
        return write!(s, "hint");
    }
    if l == 1 {
        let sys = format!("s{}{}{}{}{}", op0, op1, crn, crm, op2);
        return match (op1, crn, crm, op2) {
            (0b010, 0b0100, 0b0010, 0b000) | (0b011, 0b0100, 0b0010, 0b000) => {
                write!(s, "mrs {}, nzcv", zr64(rt))
            }
            _ => write!(s, "mrs {}, {}", zr64(rt), sys),
        };
    }
    match (op0, op1, crn, crm, op2) {
        (0, 0b010, 0b0100, 0b0010, 0b000) | (0, 0b011, 0b0100, 0b0010, 0b000) => {
            write!(s, "msr nzcv, #{:#x}", (insn >> 8) & 0xF)
        }
        _ => write!(s, "msr s{}{}{}{}{}", op0, op1, crn, crm, op2),
    }
}

/// `PRFM`'s Rt field encodes the prefetch hint as type:target:policy.
fn prfetch_hint(rt: u32) -> String {
    let ty = match (rt >> 3) & 0b11 {
        0b00 => "pld",
        0b01 => "pli",
        0b10 => "pst",
        _ => return format!("#{rt:#x}"),
    };
    let target = match (rt >> 1) & 0b11 {
        0b00 => "l1",
        0b01 => "l2",
        0b10 => "l3",
        _ => return format!("#{rt:#x}"),
    };
    let policy = if rt & 1 == 1 { "strm" } else { "keep" };
    format!("{ty}{target}{policy}")
}

fn ldst_reg(v: bool, width: u32, opc: u32, i: u32) -> String {
    if !v {
        // size == 11 with opc 1x is PRFM, whose Rt is a hint.
        if width == 3 && opc >= 0b10 {
            return prfetch_hint(i);
        }
        // Sign-extending loads take the destination width from `opc`: 10 is 64-bit, 11 is 32-bit.
        let is64 = if opc >= 0b10 { opc == 0b10 } else { width == 3 };
        return if is64 { zr64(i) } else { zr32(i) };
    }
    let c = match width {
        0 => 'b',
        1 => 'h',
        2 => 's',
        3 => 'd',
        _ => 'q',
    };
    format!("{c}{i}")
}

/// A SIMD&FP load/store's access width: `size` with `opc<1>` as an extra high bit.
fn simd_ldst_width(sz: u32, opc: u32) -> u32 {
    if opc & 0b10 != 0 {
        4
    } else {
        sz
    }
}

/// Name a load/store. `infix` is "" for scaled/indexed, "u" for unscaled, "t" for
/// unprivileged forms.
fn ldst_name(v: bool, sz: u32, opc: u32, infix: &str) -> String {
    if v {
        return if opc & 1 == 1 {
            format!("ld{infix}r")
        } else {
            format!("st{infix}r")
        };
    }
    // size == 11 with opc 1x is PRFM, not a sign-extending load.
    if sz == 3 && opc >= 0b10 {
        return if infix.is_empty() {
            "prfm".to_string()
        } else {
            format!("prf{infix}m")
        };
    }
    let suffix = match sz {
        0 => "b",
        1 => "h",
        _ => "",
    };
    match opc {
        0b00 => format!("st{infix}r{suffix}"),
        0b01 => format!("ld{infix}r{suffix}"),
        // Sign-extending loads: the suffix names the source width (size 10 is `ldrsw`).
        _ => {
            let s = match sz {
                0 => "sb",
                1 => "sh",
                _ => "sw",
            };
            format!("ld{infix}r{s}")
        }
    }
}

fn disasm_ld_st(insn: u32, s: &mut String) -> Result<bool, std::fmt::Error> {
    let grp_excl = (insn >> 21) & 0x1FF;
    if (0b001000000..=0b001000011).contains(&grp_excl)
        || grp_excl == 0b001000100
        || grp_excl == 0b001000110
    {
        let sz = (insn >> 30) & 0b11;
        let rn = (insn >> 5) & 0x1F;
        let rt = insn & 0x1F;
        let rt2 = (insn >> 10) & 0x1F;
        let rs = (insn >> 16) & 0x1F;
        // `o0` (bit 15) selects the acquire/release forms (`ldaxr`/`stlxr`).
        let w = match sz {
            0 => "b",
            1 => "h",
            _ => "",
        };
        let o0 = (insn >> 15) & 1 == 1;
        return Ok(match grp_excl {
            0b001000000 => {
                let n = if o0 { "stlxr" } else { "stxr" };
                write!(s, "{n}{w} w{}, {}, [{}]", rs, mem_reg(sz, rt), reg64(rn)).is_ok()
            }
            0b001000010 => {
                let n = if o0 { "ldaxr" } else { "ldxr" };
                write!(s, "{n}{w} {}, [{}]", mem_reg(sz, rt), reg64(rn)).is_ok()
            }
            0b001000001 => {
                let n = if o0 { "stlxp" } else { "stxp" };
                write!(
                    s,
                    "{n} w{}, {}, {}, [{}]",
                    rs,
                    zr64(rt),
                    zr64(rt2),
                    reg64(rn)
                )
                .is_ok()
            }
            0b001000011 => {
                let n = if o0 { "ldaxp" } else { "ldxp" };
                write!(s, "{n} {}, {}, [{}]", zr64(rt), zr64(rt2), reg64(rn)).is_ok()
            }
            0b001000100 => write!(s, "stlr{w} {}, [{}]", mem_reg(sz, rt), reg64(rn)).is_ok(),
            _ => write!(s, "ldar{w} {}, [{}]", mem_reg(sz, rt), reg64(rn)).is_ok(),
        });
    }

    // register-offset form; bits[31:30] are the access size.
    if ((insn >> 27) & 0b111) == 0b111 && ((insn >> 24) & 0b11) == 0b00 && ((insn >> 21) & 1) == 1 {
        let sz = (insn >> 30) & 0b11;
        let opc = (insn >> 22) & 0b11;
        let v = (insn >> 26) & 1 == 1;
        let rn = (insn >> 5) & 0x1F;
        let rt = insn & 0x1F;
        let rm = (insn >> 16) & 0x1F;
        let opt = (insn >> 13) & 0b111;
        let sbit = (insn >> 12) & 1;
        let width = if v { simd_ldst_width(sz, opc) } else { sz };
        let shift = if sbit == 1 { width } else { 0 };
        let name = ldst_name(v, sz, opc, "");
        // option<0>: 32-bit index for `uxtw`/`sxtw`, 64-bit for `lsl`/`sxtx`.
        let rm_s = if opt & 1 == 0 { zr32(rm) } else { zr64(rm) };
        write!(
            s,
            "{} {}, [{}, {}",
            name,
            ldst_reg(v, width, opc, rt),
            reg64(rn),
            rm_s
        )?;
        if opt == 0b011 {
            if shift > 0 {
                write!(s, ", lsl #{}", shift)?;
            }
        } else {
            write!(s, ", {}", ext_name(opt))?;
            if shift > 0 {
                write!(s, " #{}", shift)?;
            }
        }
        write!(s, "]")?;
        return Ok(true);
    }

    // immediate offset forms
    if ((insn >> 27) & 0b111) == 0b111 {
        let mode = (insn >> 24) & 0b11;
        let sz = (insn >> 30) & 0b11;
        let opc = (insn >> 22) & 0b11;
        let v = (insn >> 26) & 1 == 1;
        let rn = (insn >> 5) & 0x1F;
        let rt = insn & 0x1F;
        let width = if v { simd_ldst_width(sz, opc) } else { sz };
        let rt_s = ldst_reg(v, width, opc, rt);
        if mode == 0b01 {
            let imm = ((insn >> 10) & 0xFFF) as u64;
            let scaled = imm.wrapping_mul(1u64 << width);
            write!(s, "{} {}, [{}", ldst_name(v, sz, opc, ""), rt_s, reg64(rn))?;
            if scaled > 0 {
                write!(s, ", #{:#x}", scaled)?;
            }
            return Ok(write!(s, "]").is_ok());
        }
        if mode == 0b00 && ((insn >> 21) & 1) == 0 {
            // bits[11:10]: 00 unscaled (`stur`/`ldur`), 10 unprivileged (`sttr`/`ldtr`).
            let idx = (insn >> 10) & 0b11;
            let imm = sext((insn >> 12) & 0x1FF, 9);
            let base = reg64(rn);
            let infix = match idx {
                0b00 => "u",
                0b10 => "t",
                _ => "",
            };
            let name = ldst_name(v, sz, opc, infix);
            return Ok(match idx {
                0b01 => write!(s, "{name} {rt_s}, [{base}], #{imm}").is_ok(),
                0b11 => write!(s, "{name} {rt_s}, [{base}, #{imm}]!").is_ok(),
                _ => write!(s, "{name} {rt_s}, [{base}, #{imm}]").is_ok(),
            });
        }
    }

    // paired
    if ((insn >> 27) & 0b111) == 0b101 && ((insn >> 25) & 1) == 0 {
        let opc = (insn >> 30) & 0b11;
        let l = (insn >> 22) & 1;
        let v = (insn >> 26) & 1 == 1;
        let mode = (insn >> 23) & 0b11;
        let imm = sext((insn >> 15) & 0x7F, 7);
        let rn = (insn >> 5) & 0x1F;
        let rt = insn & 0x1F;
        let rt2 = (insn >> 10) & 0x1F;
        // SIMD&FP pairs: opc 00 is `s`, 01 `d`, 10 `q`. Integer opc 01 with L is LDPSW.
        let ldpsw = !v && opc == 0b01 && l == 1;
        let width = if v {
            opc + 2
        } else if opc == 0b10 {
            3
        } else {
            2
        };
        let scale = 1i64 << width;
        let name = if ldpsw {
            "ldpsw"
        } else if l == 1 {
            "ldp"
        } else {
            "stp"
        };
        let (rn_l, rt_l, rt2_l) = if v {
            (
                reg64(rn),
                ldst_reg(true, width, 0, rt),
                ldst_reg(true, width, 0, rt2),
            )
        } else if width == 3 || ldpsw {
            (reg64(rn), zr64(rt), zr64(rt2))
        } else {
            (reg64(rn), zr32(rt), zr32(rt2))
        };
        return Ok(match mode {
            0b00 => write!(
                s,
                "{} {}, {}, [{}, #{}]",
                name,
                rt_l,
                rt2_l,
                rn_l,
                simm(imm * scale)
            )
            .is_ok(),
            0b01 => write!(
                s,
                "{} {}, {}, [{}], #{}",
                name,
                rt_l,
                rt2_l,
                rn_l,
                simm(imm * scale)
            )
            .is_ok(),
            0b10 => write!(
                s,
                "{} {}, {}, [{}, #{}]",
                name,
                rt_l,
                rt2_l,
                rn_l,
                simm(imm * scale)
            )
            .is_ok(),
            _ => write!(
                s,
                "{} {}, {}, [{}, #{}]!",
                name,
                rt_l,
                rt2_l,
                rn_l,
                simm(imm * scale)
            )
            .is_ok(),
        });
    }
    Ok(false)
}

type RegFn = fn(u32) -> String;

fn disasm_dp_imm(insn: u32, s: &mut String) -> Result<bool, std::fmt::Error> {
    let grp = (insn >> 24) & 0x1F;
    let sf = (insn >> 31) & 1;
    let (zr, sp): (RegFn, RegFn) = if sf == 1 {
        (zr64, reg64)
    } else {
        (zr32, reg32)
    };
    match grp {
        0b10001 => {
            if ((insn >> 23) & 1) == 1 {
                return Ok(false);
            }
            let op = (insn >> 29) & 0b11;
            let sh = (insn >> 22) & 1;
            let imm12 = ((insn >> 10) & 0xFFF) as i64;
            let rn = (insn >> 5) & 0x1F;
            let rd = insn & 0x1F;
            let imm = if sh == 1 { imm12 << 12 } else { imm12 };
            if rd == 31 && (op == 0b01 || op == 0b11) {
                let name = if op == 0b01 { "cmn" } else { "cmp" };
                return Ok(write!(s, "{} {}, #{:#x}", name, sp(rn), imm).is_ok());
            }
            let name = match op {
                0b00 => "add",
                0b01 => "adds",
                0b10 => "sub",
                _ => "subs",
            };
            Ok(write!(s, "{} {}, {}, #{:#x}", name, sp(rd), sp(rn), imm).is_ok())
        }
        0b10010 => {
            if ((insn >> 23) & 1) == 1 {
                let opc = (insn >> 29) & 0b11;
                let imm16 = (insn >> 5) & 0xFFFF;
                let rd = insn & 0x1F;
                // hw is bits[22:21] in both widths.
                let hw = (insn >> 21) & 0b11;
                let name = match opc {
                    0b00 => "movn",
                    0b10 => "movz",
                    _ => "movk",
                };
                let imm = (imm16 as u64) << (hw * 16);
                return Ok(write!(s, "{} {}, #{:#x}", name, zr(rd), imm).is_ok());
            }
            let opc = (insn >> 29) & 0b11;
            let n = (insn >> 22) & 1;
            let immr = (insn >> 16) & 0x3F;
            let imms = (insn >> 10) & 0x3F;
            let rn = (insn >> 5) & 0x1F;
            let rd = insn & 0x1F;
            let name = match opc {
                0b00 => "and",
                0b01 => "orr",
                0b10 => "eor",
                _ => "ands",
            };
            let mask = crate::cpu::decode_bit_mask(sf == 1, n, immr, imms).unwrap_or(0);
            // Rd 31 is SP for and/orr/eor, XZR for ands.
            let d = if opc == 0b11 { zr(rd) } else { sp(rd) };
            Ok(write!(s, "{} {}, {}, #{:#x}", name, d, zr(rn), mask).is_ok())
        }
        0b10011 => {
            let rn = (insn >> 5) & 0x1F;
            let rd = insn & 0x1F;
            if ((insn >> 23) & 1) == 0 {
                let opc = (insn >> 29) & 0b11;
                let (immr, imms) = if sf == 1 {
                    ((insn >> 16) & 0x3F, (insn >> 10) & 0x3F)
                } else {
                    ((insn >> 16) & 0x1F, (insn >> 10) & 0x1F)
                };
                let (lsb, msb) = (immr as i64, imms as i64);
                let datasize = if sf == 1 { 64 } else { 32 };
                let (r_l, z_l) = (sp, zr);
                let name = match opc {
                    0b00 => {
                        if msb >= lsb {
                            format!(
                                "sbfx {}, {}, #{}, #{}",
                                z_l(rd),
                                z_l(rn),
                                lsb,
                                msb - lsb + 1
                            )
                        } else {
                            format!(
                                "sbfiz {}, {}, #{}, #{}",
                                z_l(rd),
                                z_l(rn),
                                datasize - lsb,
                                msb + 1
                            )
                        }
                    }
                    0b01 => {
                        if msb >= lsb {
                            format!(
                                "bfxil {}, {}, #{}, #{}",
                                z_l(rd),
                                z_l(rn),
                                lsb,
                                msb - lsb + 1
                            )
                        } else {
                            format!(
                                "bfi {}, {}, #{}, #{}",
                                z_l(rd),
                                z_l(rn),
                                datasize - lsb,
                                msb + 1
                            )
                        }
                    }
                    _ => {
                        if msb >= lsb {
                            format!(
                                "ubfx {}, {}, #{}, #{}",
                                z_l(rd),
                                z_l(rn),
                                lsb,
                                msb - lsb + 1
                            )
                        } else {
                            format!(
                                "ubfiz {}, {}, #{}, #{}",
                                z_l(rd),
                                z_l(rn),
                                datasize - lsb,
                                msb + 1
                            )
                        }
                    }
                };
                let _ = r_l;
                return Ok(write!(s, "{}", name).is_ok());
            }
            let rm = (insn >> 16) & 0x1F;
            let imm = if sf == 1 {
                (insn >> 10) & 0x3F
            } else {
                (insn >> 10) & 0x1F
            };
            Ok(write!(s, "extr {}, {}, {}, #{:#x}", zr(rd), zr(rn), zr(rm), imm).is_ok())
        }
        _ => Ok(false),
    }
}

fn disasm_dp_reg(insn: u32, s: &mut String) -> Result<bool, std::fmt::Error> {
    let grp = (insn >> 24) & 0x1F;
    let sf = (insn >> 31) & 1;
    let zr = if sf == 1 { zr64 } else { zr32 };
    let sp = if sf == 1 { reg64 } else { reg32 };
    match grp {
        0b01010 => {
            let opc = (insn >> 29) & 0b11;
            let st = (insn >> 22) & 0b11;
            let invert = (insn >> 21) & 1;
            let rm = (insn >> 16) & 0x1F;
            let sa = (insn >> 10) & 0x3F;
            let rn = (insn >> 5) & 0x1F;
            let rd = insn & 0x1F;
            // MOV Xd, Xm == ORR Xd, XZR, Xm (LSL #0)
            if opc == 0b01 && invert == 0 && rn == 31 && st == 0 && sa == 0 {
                return Ok(write!(s, "mov {}, {}", zr(rd), zr(rm)).is_ok());
            }
            // TST == ANDS XZR, Xn, Xm
            if opc == 0b11 && rd == 31 && invert == 0 {
                write!(s, "tst {}, {}", zr(rn), zr(rm))?;
                if sa > 0 || st != 0 {
                    write!(s, ", {} #{}", shift_name(st), sa)?;
                }
                return Ok(true);
            }
            let base = match (opc, invert) {
                (0b00, 0) => "and",
                (0b00, 1) => "bic",
                (0b01, 0) => "orr",
                (0b01, 1) => "orn",
                (0b10, 0) => "eor",
                (0b10, 1) => "eon",
                (0b11, 0) => "ands",
                _ => "bics",
            };
            write!(s, "{} {}, {}, {}", base, zr(rd), zr(rn), zr(rm))?;
            if sa > 0 || st != 0 {
                write!(s, ", {} #{}", shift_name(st), sa)?;
            }
            Ok(true)
        }
        0b01011 => {
            let op = (insn >> 29) & 0b11;
            let rn = (insn >> 5) & 0x1F;
            let rd = insn & 0x1F;
            let rm = (insn >> 16) & 0x1F;
            let compare = rd == 31 && (op == 0b01 || op == 0b11);
            let name = if compare {
                if op == 0b01 {
                    "cmn"
                } else {
                    "cmp"
                }
            } else {
                match op {
                    0b00 => "add",
                    0b01 => "adds",
                    0b10 => "sub",
                    _ => "subs",
                }
            };
            if ((insn >> 21) & 0b111) == 0b001 {
                let option = (insn >> 13) & 0b111;
                let shift = (insn >> 10) & 0b111;
                if compare {
                    write!(s, "{} {}, {}, {}", name, sp(rn), zr(rm), ext_name(option))?;
                } else {
                    write!(
                        s,
                        "{} {}, {}, {}, {}",
                        name,
                        sp(rd),
                        sp(rn),
                        zr(rm),
                        ext_name(option)
                    )?;
                }
                if shift > 0 {
                    write!(s, " #{}", shift)?;
                }
            } else {
                // Shifted register: register 31 is XZR, not SP.
                let st = (insn >> 22) & 0b11;
                let sa = (insn >> 10) & 0x3F;
                if compare {
                    write!(s, "{} {}, {}", name, zr(rn), zr(rm))?;
                } else {
                    write!(s, "{} {}, {}, {}", name, zr(rd), zr(rn), zr(rm))?;
                }
                if sa > 0 || st != 0 {
                    write!(s, ", {} #{}", shift_name(st), sa)?;
                }
            }
            Ok(true)
        }
        0b11010 => {
            if ((insn >> 22) & 1) == 1 {
                if ((insn >> 23) & 1) == 1 {
                    let opcode2 = (insn >> 10) & 0x3F;
                    let rn = (insn >> 5) & 0x1F;
                    let rd = insn & 0x1F;
                    let rm = (insn >> 16) & 0x1F;
                    if ((insn >> 29) & 0b11) == 0b00 {
                        if (opcode2 & 0b111000) == 0b010000 {
                            // CRC32/CRC32C: W accumulator; only the doubleword form takes an X data operand.
                            let name = match opcode2 & 0b111 {
                                0b000 => "crc32b",
                                0b001 => "crc32h",
                                0b010 => "crc32w",
                                0b011 => "crc32x",
                                0b100 => "crc32cb",
                                0b101 => "crc32ch",
                                0b110 => "crc32cw",
                                _ => "crc32cx",
                            };
                            let data = if (opcode2 & 0b11) == 0b11 {
                                zr64(rm)
                            } else {
                                zr32(rm)
                            };
                            return Ok(
                                write!(s, "{} {}, {}, {}", name, zr32(rd), zr32(rn), data).is_ok()
                            );
                        }
                        let name = match opcode2 {
                            0b000010 => "udiv",
                            0b000011 => "sdiv",
                            0b001000 => "lslv",
                            0b001001 => "lsrv",
                            0b001010 => "asrv",
                            0b001011 => "rorv",
                            _ => {
                                return Ok(write!(
                                    s,
                                    "dp.2src.{:06b} {}, {}, {}",
                                    opcode2,
                                    zr(rd),
                                    zr(rn),
                                    zr(rm)
                                )
                                .is_ok())
                            }
                        };
                        Ok(write!(s, "{} {}, {}, {}", name, zr(rd), zr(rn), zr(rm)).is_ok())
                    } else if ((insn >> 29) & 0b11) == 0b10 {
                        let sf = (insn >> 31) & 1 == 1;
                        let name = match opcode2 {
                            0b000000 => "rbit",
                            0b000001 => "rev16",
                            // The 32-bit form is REV; only the 64-bit form has REV32.
                            0b000010 => {
                                if sf {
                                    "rev32"
                                } else {
                                    "rev"
                                }
                            }
                            0b000011 => "rev",
                            0b000100 => "clz",
                            0b000101 => "cls",
                            _ => "ctz",
                        };
                        Ok(write!(s, "{} {}, {}", name, zr(rd), zr(rn)).is_ok())
                    } else {
                        Ok(write!(s, "dp.2src {:b}", opcode2).is_ok())
                    }
                } else {
                    let op = (insn >> 30) & 1;
                    let imm_flag = (insn >> 11) & 1;
                    let c = (insn >> 12) & 0xF;
                    let nzcv = insn & 0xF;
                    let rn = (insn >> 5) & 0x1F;
                    // op (bit 30): 1 is CCMP, 0 is CCMN.
                    let name = if op == 1 { "ccmp" } else { "ccmn" };
                    if imm_flag == 1 {
                        Ok(write!(
                            s,
                            "{} {}, #{:#x}, #{:#x}, {}",
                            name,
                            zr(rn),
                            (insn >> 16) & 0x1F,
                            nzcv,
                            cond(c)
                        )
                        .is_ok())
                    } else {
                        Ok(write!(
                            s,
                            "{} {}, {}, #{:#x}, {}",
                            name,
                            zr(rn),
                            zr((insn >> 16) & 0x1F),
                            nzcv,
                            cond(c)
                        )
                        .is_ok())
                    }
                }
            } else if ((insn >> 23) & 1) == 1 {
                let else_inv = (insn >> 30) & 1;
                let else_inc = (insn >> 10) & 1;
                let c = (insn >> 12) & 0xF;
                let rn = (insn >> 5) & 0x1F;
                let rd = insn & 0x1F;
                let rm = (insn >> 16) & 0x1F;
                let name = match (else_inv, else_inc) {
                    (0, 0) => "csel",
                    (0, 1) => "csinc",
                    (1, 0) => "csinv",
                    _ => "csneg",
                };
                Ok(write!(
                    s,
                    "{} {}, {}, {}, {}",
                    name,
                    zr(rd),
                    zr(rn),
                    zr(rm),
                    cond(c)
                )
                .is_ok())
            } else {
                let op = (insn >> 29) & 0b11;
                let rn = (insn >> 5) & 0x1F;
                let rd = insn & 0x1F;
                let rm = (insn >> 16) & 0x1F;
                let name = match op {
                    0b00 => "adc",
                    0b01 => "adcs",
                    0b10 => "sbc",
                    _ => "sbcs",
                };
                Ok(write!(s, "{} {}, {}, {}", name, zr(rd), zr(rn), zr(rm)).is_ok())
            }
        }
        0b11011 => {
            if ((insn >> 21) & 0xFF) == 0b11011000 {
                let o0 = (insn >> 15) & 1;
                let rn = (insn >> 5) & 0x1F;
                let rd = insn & 0x1F;
                let rm = (insn >> 16) & 0x1F;
                let ra = (insn >> 10) & 0x1F;
                let name = if o0 == 1 { "msub" } else { "madd" };
                Ok(write!(s, "{} {}, {}, {}, {}", name, zr(rd), zr(rn), zr(rm), zr(ra)).is_ok())
            } else {
                let rn = (insn >> 5) & 0x1F;
                let rd = insn & 0x1F;
                let rm = (insn >> 16) & 0x1F;
                let ra = (insn >> 10) & 0x1F;
                let o0 = (insn >> 15) & 1;
                let (name, operands) = match (insn >> 21) & 0xFF {
                    0b11011001 => ("smaddl", true), // / SMSUBL
                    0b11011101 => ("umaddl", true), // / UMSUBL
                    0b11011010 => ("smulh", false), // / UMULH: no Ra
                    0b11011110 => ("umulh", false),
                    _ => return Ok(write!(s, "mul.long").is_ok()),
                };
                let name = if o0 == 1 {
                    match name {
                        "smaddl" => "smsubl",
                        "umaddl" => "umsubl",
                        _ => name,
                    }
                } else {
                    name
                };
                if operands {
                    Ok(
                        write!(s, "{} {}, {}, {}, {}", name, zr(rd), zr(rn), zr(rm), zr(ra))
                            .is_ok(),
                    )
                } else {
                    Ok(write!(s, "{} {}, {}, {}", name, zr(rd), zr(rn), zr(rm)).is_ok())
                }
            }
        }
        _ => Ok(false),
    }
}
