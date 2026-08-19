//! Maxwell (GM20B) shader instruction decoding.
//!
//! Bit layouts are ported from `envydis`'s `gm107.c` tables (envytools,
//! github.com/envytools/envytools) and verified against `uam`-compiled GLSL
//! fixtures disassembled with `envydis -m gm107`: a solid-color fragment
//! shader, an MVP-transform vertex shader, and a textured + vertex-color
//! fragment shader. Every field extracted below was cross-checked against
//! that disassembly, instruction for instruction.
//!
//! Two facts fell out of that exercise that shape this decoder:
//!
//! - The rasterizer's fixed-function interpolator hands the fragment shader
//!   a linearly-interpolated `1/w` at a fixed attribute slot (`a[0x7c]` in
//!   every fixture). `ipa pass` reads it raw. The shader then computes `w =
//!   mufu rcp(1/w)` once and feeds it back into `ipa` (non-`pass`, the
//!   "perspective" mode) as the multiplier for every other varying, which the
//!   interpolator has already linearly interpolated pre-divided by `w`:
//!   `ipa.perspective(attr/w) * w == attr`. That's the whole perspective-
//!   correction idiom; there's no other division of labour to model.
//! - Every instruction carries a 4-bit predicate field (`T(pred)` in envydis)
//!   at bits `[16, 20)`; `0b0111` is hardware's "always true" placeholder.
//!   This ISA subset has no branches, so anything predicated decodes to
//!   [`Instruction::Unimplemented`].

/// `ld`/`st`'s transfer size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemSize {
    B32,
    B64,
    B96,
    B128,
}

/// The right-hand operand of an ALU op: either a register or a slot in a
/// bound constant buffer (`cN[offset]`, `N` a constant-buffer bank index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(u8),
    Const { bank: u8, offset: u16 },
}

/// `texs`'s sample dimensionality (envydis's `d000_1`/`d200_1` tables).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexDim {
    T1d,
    T2d,
    T3d,
    TCube,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    /// `ld.<size> dst, a[offset]` — attribute-space load, no index register.
    Ld { dst: u8, offset: u16, size: MemSize },
    /// `st.<size> a[offset], src` — attribute-space store.
    St { offset: u16, src: u8, size: MemSize },
    /// `ipa[.idx] dst, a[offset], mul` — fixed-function interpolation.
    /// `perspective = false` is `ipa pass`; `perspective = true` multiplies
    /// the fetched value by `mul` (`RZ` decodes to `None`).
    Ipa {
        dst: u8,
        offset: u16,
        mul: Option<u8>,
        perspective: bool,
    },
    /// `mufu rcp dst, src` — reciprocal. The only `mufu` subop this ISA
    /// subset needs.
    MufuRcp { dst: u8, src: u8 },
    /// `fmul[.ftz] dst, a, b`.
    Fmul { dst: u8, a: u8, b: Operand, ftz: bool },
    /// `fadd[.ftz] dst, a, b`.
    Fadd { dst: u8, a: u8, b: Operand, ftz: bool },
    /// `mov32i dst, imm` — load a raw 32-bit immediate (the register file is
    /// untyped, so this is however the caller wants to read it: a float bit
    /// pattern, an integer, whatever).
    Mov32i { dst: u8, imm: u32 },
    /// `ffma[.ftz] dst, a, b, c` — `dst = a*b + c`.
    Ffma {
        dst: u8,
        a: u8,
        b: Operand,
        c: u8,
        ftz: bool,
    },
    /// `texs dst, coords.., handle, dim, mask` — texture sample. Only the
    /// operand shape is decoded here; TIC/TSC handle resolution and actual
    /// sampling are Stage 7's job.
    Texs {
        dst: u8,
        coords: [u8; 3],
        handle: u16,
        dim: TexDim,
        mask: [bool; 4],
    },
    Exit,
    /// A bit pattern this decoder doesn't recognise, or recognises but with
    /// an unhandled modifier (negation, saturate, predication, an
    /// addressing mode we don't support, ...). Carries the raw bits so a
    /// real capture's exact encoding stays inspectable rather than silently
    /// mis-executing.
    Unimplemented { raw: u64 },
}

fn field(insn: u64, pos: u32, len: u32) -> u64 {
    (insn >> pos) & ((1u64 << len) - 1)
}

fn reg(insn: u64, pos: u32, len: u32) -> u8 {
    field(insn, pos, len) as u8
}

const RZ: u8 = 0xff;

fn opt_reg(r: u8) -> Option<u8> {
    if r == RZ {
        None
    } else {
        Some(r)
    }
}

fn mem_size(bits: u64) -> MemSize {
    match bits {
        0 => MemSize::B32,
        1 => MemSize::B64,
        2 => MemSize::B96,
        _ => MemSize::B128,
    }
}

fn const_operand(insn: u64) -> Operand {
    Operand::Const {
        bank: reg(insn, 34, 5),
        offset: (field(insn, 20, 14) << 2) as u16,
    }
}

/// Decode a single 8-byte Maxwell instruction word. Never panics: an
/// unrecognised or unsupported bit pattern decodes to
/// [`Instruction::Unimplemented`].
pub fn decode(insn: u64) -> Instruction {
    // T(pred) — require the "always true" placeholder; see module docs.
    if field(insn, 16, 4) != 0x7 {
        return Instruction::Unimplemented { raw: insn };
    }

    // ld — attribute space, no index register. gm107.c: 0xefd8/0xfff8...
    if insn & 0xfff8_0000_0000_0000 == 0xefd8_0000_0000_0000 {
        let o = field(insn, 32, 1);
        let p = field(insn, 31, 1);
        let idx = reg(insn, 8, 8);
        let extra = reg(insn, 39, 8);
        if o == 0 && p == 0 && idx == RZ && extra == RZ {
            return Instruction::Ld {
                dst: reg(insn, 0, 8),
                offset: field(insn, 20, 10) as u16,
                size: mem_size(field(insn, 47, 2)),
            };
        }
        return Instruction::Unimplemented { raw: insn };
    }

    // st — attribute space, no index register. gm107.c: 0xeff0/0xfff8...
    if insn & 0xfff8_0000_0000_0000 == 0xeff0_0000_0000_0000 {
        let p = field(insn, 31, 1);
        let idx = reg(insn, 8, 8);
        let extra = reg(insn, 39, 8);
        if p == 0 && idx == RZ && extra == RZ {
            return Instruction::St {
                offset: field(insn, 20, 10) as u16,
                src: reg(insn, 0, 8),
                size: mem_size(field(insn, 47, 2)),
            };
        }
        return Instruction::Unimplemented { raw: insn };
    }

    // ipa — a[]-relative, non-indexed. gm107.c: 0xe0.....ff00 / mask
    // 0xff0000400000ff00 (the low byte pins the unused index register to RZ;
    // the idx-addressed variant lives at a different mask and isn't decoded
    // here since none of our fixtures emit it).
    if insn & 0xff00_0040_0000_ff00 == 0xe000_0000_0000_ff00 {
        let mode = field(insn, 54, 2);
        let centroid_offset = field(insn, 52, 2);
        let sat = field(insn, 51, 1);
        let pred47 = field(insn, 47, 3);
        let extra = reg(insn, 39, 8);
        if centroid_offset == 0 && sat == 0 && pred47 == 0x7 && extra == RZ && mode <= 1 {
            return Instruction::Ipa {
                dst: reg(insn, 0, 8),
                offset: field(insn, 28, 10) as u16,
                mul: opt_reg(reg(insn, 20, 8)),
                perspective: mode == 1,
            };
        }
        return Instruction::Unimplemented { raw: insn };
    }

    // mufu — gm107.c: 0x5080/0xfff8...
    if insn & 0xfff8_0000_0000_0000 == 0x5080_0000_0000_0000 {
        let sat = field(insn, 50, 1);
        let neg = field(insn, 48, 1);
        let abs = field(insn, 46, 1);
        let subop = field(insn, 20, 4);
        if sat == 0 && neg == 0 && abs == 0 && subop == 0x4 {
            return Instruction::MufuRcp {
                dst: reg(insn, 0, 8),
                src: reg(insn, 8, 8),
            };
        }
        return Instruction::Unimplemented { raw: insn };
    }

    // fmul, constant-bank form: dst = a * cN[offset]. gm107.c: 0x4c68/0xfff8...
    if insn & 0xfff8_0000_0000_0000 == 0x4c68_0000_0000_0000 {
        return decode_fmul(insn, const_operand(insn));
    }
    // fmul, register-register form: dst = a * b. gm107.c: 0x5c68/0xfff8...
    if insn & 0xfff8_0000_0000_0000 == 0x5c68_0000_0000_0000 {
        return decode_fmul(insn, Operand::Reg(reg(insn, 20, 8)));
    }

    // fadd, constant-bank form: dst = a + cN[offset]. gm107.c: 0x4c58/0xfff8...
    if insn & 0xfff8_0000_0000_0000 == 0x4c58_0000_0000_0000 {
        return decode_fadd(insn, const_operand(insn));
    }
    // fadd, register-register form: dst = a + b. gm107.c: 0x5c58/0xfff8...
    if insn & 0xfff8_0000_0000_0000 == 0x5c58_0000_0000_0000 {
        return decode_fadd(insn, Operand::Reg(reg(insn, 20, 8)));
    }

    // ffma, constant-bank form: dst = a * cN[offset] + c. gm107.c: 0x4980/0xff80...
    if insn & 0xff80_0000_0000_0000 == 0x4980_0000_0000_0000 {
        let ftz = match decode_ftz(field(insn, 53, 2)) {
            Some(ftz) => ftz,
            None => return Instruction::Unimplemented { raw: insn },
        };
        let round = field(insn, 51, 2);
        let sat = field(insn, 50, 1);
        let cc = field(insn, 47, 1);
        let neg_b = field(insn, 48, 1);
        let neg_c = field(insn, 49, 1);
        if round == 0 && sat == 0 && cc == 0 && neg_b == 0 && neg_c == 0 {
            return Instruction::Ffma {
                dst: reg(insn, 0, 8),
                a: reg(insn, 8, 8),
                b: const_operand(insn),
                c: reg(insn, 39, 8),
                ftz,
            };
        }
        return Instruction::Unimplemented { raw: insn };
    }

    // texs — gm107.c: 0xd000/0xf600... Bit 59 (`ZNV(59, f16, F_SM60)` in
    // envydis) only means anything on the later SM60 variant; on gm107 it's
    // outside the table's mask and uam sets it without it changing the
    // instruction's meaning, so it's not gated here.
    if insn & 0xf600_0000_0000_0000 == 0xd000_0000_0000_0000 {
        let nodep = field(insn, 49, 1);
        let dim_bits = field(insn, 53, 4);
        let mask_bits = field(insn, 50, 3);
        if nodep == 0 {
            if let (Some(dim), Some(mask)) = (decode_tex_dim(dim_bits), decode_tex_mask(mask_bits))
            {
                return Instruction::Texs {
                    dst: reg(insn, 0, 8),
                    coords: [reg(insn, 28, 8), reg(insn, 8, 8), reg(insn, 20, 8)],
                    handle: field(insn, 36, 13) as u16,
                    dim,
                    mask,
                };
            }
        }
        return Instruction::Unimplemented { raw: insn };
    }

    // mov32i — gm107.c: 0x0100/0xfff0...
    if insn & 0xfff0_0000_0000_0000 == 0x0100_0000_0000_0000 {
        return Instruction::Mov32i {
            dst: reg(insn, 0, 8),
            imm: field(insn, 20, 32) as u32,
        };
    }

    // exit — gm107.c: 0xe300/0xfff0...
    if insn & 0xfff0_0000_0000_0000 == 0xe300_0000_0000_0000 {
        return Instruction::Exit;
    }

    Instruction::Unimplemented { raw: insn }
}

fn decode_fmul(insn: u64, b: Operand) -> Instruction {
    let ftz = match decode_ftz(field(insn, 44, 2)) {
        Some(ftz) => ftz,
        None => return Instruction::Unimplemented { raw: insn },
    };
    let round = field(insn, 41, 3);
    let sat = field(insn, 50, 1);
    let cc = field(insn, 47, 1);
    let neg = field(insn, 48, 1);
    if round == 0 && sat == 0 && cc == 0 && neg == 0 {
        Instruction::Fmul {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b,
            ftz,
        }
    } else {
        Instruction::Unimplemented { raw: insn }
    }
}

/// `fadd`'s `ftz` is a plain flag (bit 44), unlike `fmul`'s two-value
/// `ftz`/`fmz` sub-table at the same position — real gm107.c table entries,
/// not a simplification.
fn decode_fadd(insn: u64, b: Operand) -> Instruction {
    let ftz = field(insn, 44, 1) != 0;
    let sat = field(insn, 50, 1);
    let cc = field(insn, 47, 1);
    let neg_a = field(insn, 48, 1);
    let abs_a = field(insn, 46, 1);
    let neg_b = field(insn, 45, 1);
    let abs_b = field(insn, 49, 1);
    if sat == 0 && cc == 0 && neg_a == 0 && abs_a == 0 && neg_b == 0 && abs_b == 0 {
        Instruction::Fadd {
            dst: reg(insn, 0, 8),
            a: reg(insn, 8, 8),
            b,
            ftz,
        }
    } else {
        Instruction::Unimplemented { raw: insn }
    }
}

/// `5c68_0`/`5980_0`-style ftz/fmz field: 0 = neither, 1 = ftz, 2 = fmz
/// (unsupported — not needed by any fixture).
fn decode_ftz(bits: u64) -> Option<bool> {
    match bits {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

/// `d000_1`/`d200_1`-shared 4-bit field. Only the one combination our
/// fixtures exercise (`t2d`, no lod-clamp modifier) is supported; every
/// other real encoding decodes to `None` (-> `Unimplemented`) until Stage 7
/// needs it.
fn decode_tex_dim(bits: u64) -> Option<TexDim> {
    match bits {
        1 => Some(TexDim::T2d),
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
        0 => Some(or(or(R, G), B)),          // rgb
        1 => Some(or(or(R, G), A)),          // rga
        2 => Some(or(or(R, B), A)),          // rba
        3 => Some(or(or(G, B), A)),          // gba
        4 => Some(or(or(or(R, G), B), A)),   // rgba
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every raw word below was captured with `envydis -n -i -m gm107`
    // against `uam`-compiled fixtures and cross-checked field by field; see
    // the module docs for provenance.

    #[test]
    fn decodes_ipa_pass_then_mufu_rcp() {
        // solid.frag: "ipa pass $r0 a[0x7c] 0x0 0x0 0x1"
        assert_eq!(
            decode(0xe003ff87cff7ff00),
            Instruction::Ipa {
                dst: 0,
                offset: 0x7c,
                mul: None,
                perspective: false,
            }
        );
        // "mufu rcp $r3 $r0"
        assert_eq!(
            decode(0x5080000000470003),
            Instruction::MufuRcp { dst: 3, src: 0 }
        );
        // "ipa $r0 a[0x80] $r3 0x0 0x1"
        assert_eq!(
            decode(0xe043ff880037ff00),
            Instruction::Ipa {
                dst: 0,
                offset: 0x80,
                mul: Some(3),
                perspective: true,
            }
        );
    }

    #[test]
    fn decodes_exit() {
        assert_eq!(decode(0xe3000000_0007000f), Instruction::Exit);
    }

    #[test]
    fn decodes_ld_st_b128_attribute_space() {
        // mvp.vert: "ld b128 $r0 a[0x80] 0x0"
        assert_eq!(
            decode(0xefd9ff80_0807ff00),
            Instruction::Ld { dst: 0, offset: 0x80, size: MemSize::B128 }
        );
        // "st b128 a[0x70] $r0 0x0"
        assert_eq!(
            decode(0xeff1ff80_0707ff00),
            Instruction::St { offset: 0x70, src: 0, size: MemSize::B128 }
        );
    }

    #[test]
    fn decodes_fmul_constant_bank_and_register_forms() {
        // mvp.vert: "fmul ftz $r4 $r0 c2[0x0]"
        assert_eq!(
            decode(0x4c681008_00070004),
            Instruction::Fmul {
                dst: 4,
                a: 0,
                b: Operand::Const { bank: 2, offset: 0x0 },
                ftz: true,
            }
        );
        // mvp.vert: "fmul ftz $r5 $r0 c2[0x4]"
        assert_eq!(
            decode(0x4c681008_00170005),
            Instruction::Fmul {
                dst: 5,
                a: 0,
                b: Operand::Const { bank: 2, offset: 0x4 },
                ftz: true,
            }
        );
        // tex.frag: "fmul ftz $r0 $r0 $r5"
        assert_eq!(
            decode(0x5c681000_00570000),
            Instruction::Fmul {
                dst: 0,
                a: 0,
                b: Operand::Reg(5),
                ftz: true,
            }
        );
    }

    #[test]
    fn decodes_fadd_constant_bank_form() {
        // Captured from a live JKSV run (real Mesa/nouveau nvc0-compiled
        // code, not a `uam` fixture): "fadd ftz $r4 $r2 c0[0x30]".
        assert_eq!(
            decode(0x4c58100000c70204),
            Instruction::Fadd {
                dst: 4,
                a: 2,
                b: Operand::Const { bank: 0, offset: 0x30 },
                ftz: true,
            }
        );
    }

    #[test]
    fn decodes_mov32i() {
        // Captured from a live JKSV run: "mov32i $r0 0x3f800000" (loads the
        // float bit pattern for 1.0).
        assert_eq!(
            decode(0x0103f8000007f000),
            Instruction::Mov32i { dst: 0, imm: 0x3f800000 }
        );
    }

    #[test]
    fn decodes_ffma_constant_bank_chain() {
        // mvp.vert: "ffma ftz $r4 $r1 c2[0x10] $r4"
        assert_eq!(
            decode(0x49a00208_00470104),
            Instruction::Ffma {
                dst: 4,
                a: 1,
                b: Operand::Const { bank: 2, offset: 0x10 },
                c: 4,
                ftz: true,
            }
        );
        // "ffma ftz $r0 $r3 c2[0x30] $r1"
        assert_eq!(
            decode(0x49a00088_00c70300),
            Instruction::Ffma {
                dst: 0,
                a: 3,
                b: Operand::Const { bank: 2, offset: 0x30 },
                c: 1,
                ftz: true,
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
            decode(0xd8301a40_20170000),
            Instruction::Texs {
                dst: 0,
                coords: [2, 0, 1],
                handle: 0x1a4,
                dim: TexDim::T2d,
                mask: [true, true, true, true],
            }
        );
    }

    #[test]
    fn unrecognised_bits_are_unimplemented_not_a_panic() {
        assert_eq!(decode(0), Instruction::Unimplemented { raw: 0 });
        assert_eq!(decode(u64::MAX), Instruction::Unimplemented { raw: u64::MAX });
    }
}
