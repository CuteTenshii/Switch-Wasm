//! Loads and stores: the integer, pair, exclusive and SIMD&FP addressing
//! modes, including the structure load/store forms.
//!
//! The types below are what a load or store *does*, separated from the
//! encoding that asked for it. The interpreter derives one per execution and
//! the block translator bakes one into an [`super::jit::ir::Op`], and both then
//! run the same [`Cpu::access`], [`Cpu::indexed`] and [`Cpu::pair`].

use super::bits::*;
use super::Cpu;
use crate::{Error, Result};

/// What a load/store does to its base register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Wb {
    /// No writeback: the access is at `base + offset`.
    None,
    /// Pre-index: the access is at `base + offset`, which is also written back.
    Pre,
    /// Post-index: the access is at `base`, and `base + offset` is written back.
    Post,
}

/// What a load or store actually does: its width, its direction and its
/// sign-extension, as the `size` and `opc` fields together select them.
///
/// Deriving this is the `PRFM` test, the load/store test, the sign-extend
/// test, a match to pick the width and a second to pick the sign-extension
/// width. All of it is constant per instruction, which is why the translator
/// resolves it once — and why there is one classifier rather than two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Acc {
    Store8,
    Store16,
    Store32,
    Store64,
    Load8,
    Load16,
    Load32,
    Load64,
    LoadS8,
    LoadS16,
    LoadS32,
    /// `PRFM`, a hint with no architectural effect. Still an access, because
    /// the addressing mode's writeback happens whether or not it does.
    Prefetch,
}

impl Acc {
    /// Whether the access writes `Rt`. Decides which of register 31's two
    /// meanings the `Rt` field names, and so which slot it resolves to.
    pub(super) fn writes_rt(self) -> bool {
        !matches!(
            self,
            Acc::Store8 | Acc::Store16 | Acc::Store32 | Acc::Store64 | Acc::Prefetch
        )
    }

    /// The access a `size`:`opc` pair selects.
    ///
    /// `opc` selects the access: 00 = STR, 01 = LDR, 10/11 = sign-extending
    /// loads (LDRSB/LDRSH/LDRSW). The load bit is NOT `opc & 1` — treating
    /// opc=10 as a store silently corrupted the target (observed as a bogus
    /// `ldrsw` index in NX-Shell's tokenizer). size=11 with opc=10/11 is
    /// `PRFM`, which must not run as a sign-extending load or it clobbers a
    /// register: libtransistor's memcpy starts with `prfm pldl1keep, [x1]`,
    /// and running that as `ldrsw x0, [x1]` made memcpy write to the source
    /// magic value as an address.
    pub(super) fn of(sz: u8, opc: u8) -> Acc {
        if sz == 0b11 && opc >= 0b10 {
            return Acc::Prefetch;
        }
        match (opc, sz) {
            (0b00, 0b00) => Acc::Store8,
            (0b00, 0b01) => Acc::Store16,
            (0b00, 0b10) => Acc::Store32,
            (0b00, _) => Acc::Store64,
            (0b01, 0b00) => Acc::Load8,
            (0b01, 0b01) => Acc::Load16,
            (0b01, 0b10) => Acc::Load32,
            (0b01, _) => Acc::Load64,
            (_, 0b00) => Acc::LoadS8,
            (_, 0b01) => Acc::LoadS16,
            (_, _) => Acc::LoadS32,
        }
    }
}

/// How a register-offset load/store extends its index register. The four
/// encodings A64 defines collapse to three behaviours; the other four are
/// undefined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Ext {
    /// `UXTW`: the low 32 bits, zero-extended.
    Uxtw,
    /// `SXTW`: the low 32 bits, sign-extended.
    Sxtw,
    /// `LSL`, `UXTX` and `SXTX`, all of which take the register as it stands.
    None,
}

impl Ext {
    /// The extension an `option` field selects, or `None` where the encoding
    /// is undefined. The signed pair is 110/111 — not 111/110 as in some
    /// tables — and a byte or halfword extend here is UNDEFINED, so it faults
    /// rather than guessing.
    pub(super) fn of(option: u8) -> Option<Ext> {
        match option {
            0b010 => Some(Ext::Uxtw),
            0b110 => Some(Ext::Sxtw),
            0b011 | 0b111 => Some(Ext::None),
            _ => None,
        }
    }
}

/// What a load/store pair moves, and which way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PairKind {
    Load32,
    /// `LDPSW`: two words, each sign-extended to 64 bits.
    Load32Sext,
    Load64,
    Store32,
    Store64,
}

impl PairKind {
    /// Whether the pair writes its two registers.
    pub(super) fn loads(self) -> bool {
        matches!(
            self,
            PairKind::Load32 | PairKind::Load32Sext | PairKind::Load64
        )
    }

    /// The distance between the two registers' addresses.
    #[inline(always)]
    pub(super) fn stride(self) -> u32 {
        match self {
            PairKind::Load64 | PairKind::Store64 => 8,
            _ => 4,
        }
    }
}

/// The slot a load or store's `Rt` field names, which depends on whether the
/// access reads it or writes it.
#[inline]
pub(super) fn rt_slot(rt: u32, acc: Acc) -> u8 {
    if acc.writes_rt() {
        Cpu::zr_write_slot(rt as u8)
    } else {
        (rt & 0x1F) as u8
    }
}

/// The slot a pair's `Rt`/`Rt2` field names, on the same rule.
#[inline]
pub(super) fn pair_slot(rt: u32, kind: PairKind) -> u8 {
    if kind.loads() {
        Cpu::zr_write_slot(rt as u8)
    } else {
        (rt & 0x1F) as u8
    }
}

impl Cpu {
    /// SIMD (V=1) memory ops: the Q-register (128-bit) subset libnx's
    /// `memset`/`memcpy` uses. Handles unsigned-immediate and unscaled
    /// STR/LDR Q, plus STP/LDP Q (signed-offset / pre-index). Everything else
    /// that sets V=1 is left unimplemented.
    pub(super) fn try_simd_load_store(&mut self, insn: u32) -> Result<bool> {
        let grp = (insn >> 27) & 0b111;
        // Scalar SIMD LDR/STR (V=1): bits[29:27] = 111, bit26 = 1. The size/opc
        // pairs select the width — opc 00/01 are STR/LDR of B/H/S/D (size
        // 00/01/10/11, byte offset scaled 1/2/4/8), opc 10/11 are STR/LDR Q
        // (128-bit, size must be 00, offset scaled 16). mode=01 is the
        // unsigned-offset form (imm12), mode=00 the unscaled STUR/LDUR (imm9).
        // `ldr b29, [x0, #0x280]` = 0x3D4A001D, `stur q17, [x0, #0x8]` = 0x3C808011.
        if grp == 0b111 {
            let sz = (insn >> 30) & 0b11;
            let opc = (insn >> 22) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let (bytes, load) = match (sz, opc) {
                (0b00, 0b00) => (1u64, false),
                (0b00, 0b01) => (1, true),
                (0b01, 0b00) => (2, false),
                (0b01, 0b01) => (2, true),
                (0b10, 0b00) => (4, false),
                (0b10, 0b01) => (4, true),
                (0b11, 0b00) => (8, false),
                (0b11, 0b01) => (8, true),
                (0b00, 0b10) => (16, false),
                (0b00, 0b11) => (16, true),
                _ => return Ok(false),
            };
            let mode = (insn >> 24) & 0b11;
            let (addr, writeback) = match mode {
                // Unsigned offset (immediate). imm12 occupies bits[21:10], so
                // bit 21 must NOT be treated as a register-offset flag here —
                // `ldr b29, [x0, #0xc80]` was being misread as a register load
                // using a garbage Rm.
                0b01 => {
                    let scale = if bytes == 16 { 16u64 } else { bytes };
                    let imm = ((insn >> 10) & 0xFFF) as u64;
                    ((self.read_x(rn).wrapping_add(imm * scale)) as u32, None)
                }
                // Register offset (mode 0b00, bit 21 set): addr = Xn + (Rm << LSL).
                0b00 if ((insn >> 21) & 1) == 1 => {
                    let rm = ((insn >> 16) & 0x1F) as u8;
                    let s = (insn >> 12) & 1;
                    let shift = if s == 1 { bytes.trailing_zeros() } else { 0 };
                    (
                        (self
                            .read_x(rn)
                            .wrapping_add(self.read_x(rm).wrapping_shl(shift)))
                            as u32,
                        None,
                    )
                }
                // Unscaled (STUR/LDUR), post-index and pre-index all share
                // mode 0b00 with a 9-bit signed byte offset; bits[11:10] pick
                // which. Missing the indexed forms leaves the base register
                // unchanged, so `str q0, [x2], #16` loops forever.
                0b00 => {
                    let imm = sext_u64((insn >> 12) & 0x1FF, 9) as i64;
                    let base = self.read_x(rn) as i64;
                    let updated = base.wrapping_add(imm) as u64;
                    match (insn >> 10) & 0b11 {
                        0b01 => (base as u32, Some(updated)),    // post-index
                        0b11 => (updated as u32, Some(updated)), // pre-index
                        _ => (updated as u32, None),             // unscaled
                    }
                }
                _ => return Ok(false),
            };
            if load {
                self.vregs[rt as usize] = match bytes {
                    1 => self.mem.read_u8(addr)? as u128,
                    2 => self.mem.read_u16(addr)? as u128,
                    4 => self.mem.read_u32(addr)? as u128,
                    8 => self.mem.read_u64(addr)? as u128,
                    _ => self.load_q(addr)?,
                };
            } else {
                match bytes {
                    1 => self.mem.write_u8(addr, self.vregs[rt as usize] as u8)?,
                    2 => self.mem.write_u16(addr, self.vregs[rt as usize] as u16)?,
                    4 => self.mem.write_u32(addr, self.vregs[rt as usize] as u32)?,
                    8 => self.mem.write_u64(addr, self.vregs[rt as usize] as u64)?,
                    _ => self.store_q(addr, self.vregs[rt as usize])?,
                }
            }
            if let Some(updated) = writeback {
                self.write_x(rn, updated);
            }
            return Ok(true);
        }
        // AdvSIMD load/store single structure: bit31=0, Q=bit30,
        // bits[29:24]=001101, wback=bit23 (the post-index forms), L=bit22,
        // R=bit21, Rm=bits[20:16], opcode=bits[15:13], S=bit12,
        // size=bits[11:10]. `scale` is opcode[2:1] and picks the element
        // width; scale 0b11 is the load-and-replicate group (LD1R/LD2R/LD3R/
        // LD4R), where `size` carries the width instead and every lane gets
        // the same element.
        if ((insn >> 31) & 1) == 0 && ((insn >> 24) & 0x3F) == 0b001101 {
            let q = (insn >> 30) & 1;
            let wback = (insn >> 23) & 1 == 1;
            let load = (insn >> 22) & 1 == 1;
            let r = (insn >> 21) & 1;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let opcode = (insn >> 13) & 0b111;
            let s = (insn >> 12) & 1;
            let size = (insn >> 10) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let selem = ((opcode & 1) << 1 | r) + 1;
            let mut scale = opcode >> 1;
            let mut replicate = false;
            // The lane index is spread across Q, S and `size`, and which bits
            // belong to it depends on the element width.
            let index;
            match scale {
                0b11 => {
                    if !load || s == 1 {
                        return Ok(false);
                    }
                    scale = size;
                    replicate = true;
                    index = 0;
                }
                0b00 => index = (q << 3) | (s << 2) | size,
                0b01 => {
                    if size & 1 != 0 {
                        return Ok(false);
                    }
                    index = (q << 2) | (s << 1) | (size >> 1);
                }
                _ => {
                    if size & 0b10 != 0 {
                        return Ok(false);
                    }
                    if size & 1 == 0 {
                        index = (q << 1) | s;
                    } else {
                        if s == 1 {
                            return Ok(false);
                        }
                        index = q;
                        scale = 0b11;
                    }
                }
            }
            let esize = 8u32 << scale;
            let ebytes = u64::from(esize / 8);
            let lanes = if q == 1 { 128 / esize } else { 64 / esize };
            let base = self.read_x(rn);
            let mut offs = 0u64;
            for sel in 0..selem {
                let reg = ((u32::from(rt) + sel) % 32) as u8;
                let addr = base.wrapping_add(offs) as u32;
                if replicate {
                    let elem = self.load_by_size(addr, scale, false)?;
                    let mask = elem_mask(esize);
                    let mut val = 0u128;
                    for lane in 0..lanes {
                        val |= (u128::from(elem) & mask) << (esize * lane);
                    }
                    // A 64-bit destination zeroes the register's top half.
                    self.vregs[reg as usize] = val;
                } else if load {
                    let elem = self.load_by_size(addr, scale, false)?;
                    self.write_vreg_elem(reg, index, esize, elem);
                } else {
                    let elem = self.read_vreg_elem(reg, index, esize);
                    self.store_by_size(addr, scale, elem)?;
                }
                offs += ebytes;
            }
            if wback {
                // Rm == 31 selects the immediate form, whose increment is the
                // number of bytes transferred; anything else is a register
                // increment. Without this the base of `ld1 {v1.16b, v2.16b},
                // [x2], #32` never advanced, so newlib's strrchr returned a
                // pointer 32 bytes below the string.
                let step = if rm == 31 { offs } else { self.read_x(rm) };
                self.write_x(rn, base.wrapping_add(step));
            }
            return Ok(true);
        }
        // AdvSIMD load/store multiple structures: bit31=0, Q=bit30,
        // bits[29:24]=001100, wback=bit23 (the post-index forms), L=bit22,
        // bit21=0, Rm=bits[20:16], opcode=bits[15:12] selects (rpt, selem),
        // size=bits[11:10]. `selem` > 1 is the interleaving LD2/LD3/LD4 and
        // ST2/ST3/ST4 group.
        if ((insn >> 31) & 1) == 0 && ((insn >> 24) & 0x3F) == 0b001100 && ((insn >> 21) & 1) == 0 {
            let q = (insn >> 30) & 1;
            let wback = (insn >> 23) & 1 == 1;
            let load = (insn >> 22) & 1 == 1;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let opcode = (insn >> 12) & 0b1111;
            let size = (insn >> 10) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let (rpt, selem) = match opcode {
                0b0000 => (1u32, 4u32),
                0b0010 => (4, 1),
                0b0100 => (1, 3),
                0b0110 => (3, 1),
                0b0111 => (1, 1),
                0b1000 => (1, 2),
                0b1010 => (2, 1),
                _ => return Ok(false),
            };
            // A structure of 64-bit elements can't be interleaved into 64-bit
            // registers: there is only one lane to interleave.
            if size == 0b11 && q == 0 && selem != 1 {
                return Ok(false);
            }
            let esize = 8u32 << size;
            let ebytes = u64::from(esize / 8);
            let vec_bytes = if q == 1 { 16u64 } else { 8 };
            let lanes = if q == 1 { 128 / esize } else { 64 / esize };
            let base = self.read_x(rn);
            let mut offs = 0u64;
            if selem == 1 {
                // Contiguous: each register is a plain 64/128-bit chunk of
                // memory whatever the element size, so move it in one go.
                for i in 0..rpt {
                    let addr = base.wrapping_add(offs) as u32;
                    let reg = ((u32::from(rt) + i) % 32) as usize;
                    if load {
                        self.vregs[reg] = if q == 1 {
                            self.load_q(addr)?
                        } else {
                            u128::from(self.mem.read_u64(addr)?)
                        };
                    } else if q == 1 {
                        self.store_q(addr, self.vregs[reg])?;
                    } else {
                        self.mem.write_u64(addr, self.vregs[reg] as u64)?;
                    }
                    offs += vec_bytes;
                }
            } else {
                if load && q == 0 {
                    // Loading a 64-bit register zeroes its top half, and the
                    // lane writes below only touch the bottom one.
                    for i in 0..rpt * selem {
                        self.vregs[((u32::from(rt) + i) % 32) as usize] &= elem_mask(64);
                    }
                }
                for r in 0..rpt {
                    for lane in 0..lanes {
                        let mut reg = u32::from(rt) + r;
                        for _ in 0..selem {
                            let addr = base.wrapping_add(offs) as u32;
                            if load {
                                let elem = self.load_by_size(addr, size, false)?;
                                self.write_vreg_elem((reg % 32) as u8, lane, esize, elem);
                            } else {
                                let elem = self.read_vreg_elem((reg % 32) as u8, lane, esize);
                                self.store_by_size(addr, size, elem)?;
                            }
                            offs += ebytes;
                            reg += 1;
                        }
                    }
                }
            }
            if wback {
                let step = if rm == 31 { offs } else { self.read_x(rm) };
                self.write_x(rn, base.wrapping_add(step));
            }
            return Ok(true);
        }
        // SIMD&FP register-offset form (V=1): bits[29:27]==111, bits[25:24]==00,
        // bit21==1. Same B/H/S/D/Q size mapping as the immediate forms.
        if grp == 0b111
            && ((insn >> 25) & 1) == 0
            && ((insn >> 24) & 1) == 0
            && ((insn >> 21) & 1) == 1
        {
            let size = (insn >> 30) & 0b11;
            let opc = (insn >> 22) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let opt = ((insn >> 13) & 0b111) as u8;
            let s = (insn >> 12) & 1;
            let is_q = size == 0 && (opc == 0b10 || opc == 0b11);
            let is_b = size == 0 && (opc == 0b00 || opc == 0b01);
            let is_h = size == 1 && (opc == 0b00 || opc == 0b01);
            let is_s = size == 2 && (opc == 0b00 || opc == 0b01);
            let is_d = size == 3 && (opc == 0b00 || opc == 0b01);
            if !is_q && !is_b && !is_h && !is_s && !is_d {
                return Ok(false);
            }
            let off_sz = if is_q { 4 } else { size as u8 };
            let offset = self.offset_from_reg(rm, opt, s, off_sz)?;
            let addr = (self.read_x(rn) as i64).wrapping_add(offset) as u32;
            let elem_bytes: u32 = if is_q {
                16
            } else if is_d {
                8
            } else if is_s {
                4
            } else if is_h {
                2
            } else {
                1
            };
            let load = if is_q { opc == 0b11 } else { opc == 0b01 };
            if load {
                self.vregs[rt as usize] = match elem_bytes {
                    16 => self.load_q(addr)?,
                    8 => self.mem.read_u64(addr)? as u128,
                    4 => self.mem.read_u32(addr)? as u128,
                    2 => self.mem.read_u16(addr)? as u128,
                    _ => self.mem.read_u8(addr)? as u128,
                };
            } else {
                match elem_bytes {
                    16 => self.store_q(addr, self.vregs[rt as usize])?,
                    8 => self.mem.write_u64(addr, self.vregs[rt as usize] as u64)?,
                    4 => self.mem.write_u32(addr, self.vregs[rt as usize] as u32)?,
                    2 => self.mem.write_u16(addr, self.vregs[rt as usize] as u16)?,
                    _ => self.mem.write_u8(addr, self.vregs[rt as usize] as u8)?,
                }
            }
            return Ok(true);
        }
        if grp == 0b111 {
            // Unsigned immediate (mode 01) and unscaled (mode 00) forms for
            // SIMD&FP registers. The 128-bit Q form reuses size=00 with
            // opc=10 (STR) / 11 (LDR); size=00 opc=00/01 is S (32-bit) and
            // size=01 opc=00/01 is D (64-bit).
            let mode = (insn >> 24) & 0b11;
            let opc = (insn >> 22) & 0b11;
            let size = (insn >> 30) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let is_q = size == 0 && (opc == 0b10 || opc == 0b11);
            let is_b = size == 0 && (opc == 0b00 || opc == 0b01);
            let is_h = size == 1 && (opc == 0b00 || opc == 0b01);
            let is_s = size == 2 && (opc == 0b00 || opc == 0b01);
            let is_d = size == 3 && (opc == 0b00 || opc == 0b01);
            if !is_q && !is_b && !is_h && !is_s && !is_d {
                return Ok(false);
            }
            // B/H/S/D/Q use size 00/01/10/11/00(opc 10/11); the imm is scaled
            // by the element byte count.
            let elem_bytes: u32 = if is_q {
                16
            } else if is_d {
                8
            } else if is_s {
                4
            } else if is_h {
                2
            } else {
                1
            };
            let shift = elem_bytes.trailing_zeros();
            let base = self.read_x(rn);
            let (addr, writeback, wb_val) = if mode == 0b01 {
                // Unsigned immediate, no writeback.
                let imm = (((insn >> 10) & 0xFFF) as u64) << shift;
                (base.wrapping_add(imm) as u32, false, 0)
            } else if mode == 0b00 && ((insn >> 21) & 1) == 0 {
                // Unscaled/pre/post-indexed immediate: imm9 is a byte offset,
                // and bits[11:10] select the addressing mode.
                let imm = sext_u64((insn >> 12) & 0x1FF, 9) as i64;
                let idx = (insn >> 10) & 0b11;
                match idx {
                    0b00 => (base.wrapping_add(imm as u64) as u32, false, 0),
                    0b01 => (base as u32, true, base.wrapping_add(imm as u64)),
                    0b11 => {
                        let addr = base.wrapping_add(imm as u64);
                        (addr as u32, true, addr)
                    }
                    _ => return Ok(false),
                }
            } else {
                return Ok(false);
            };
            let load = if is_q { opc == 0b11 } else { opc == 0b01 };
            if load {
                // Loads zero the destination register above the element.
                self.vregs[rt as usize] = match elem_bytes {
                    16 => self.load_q(addr)?,
                    8 => self.mem.read_u64(addr)? as u128,
                    4 => self.mem.read_u32(addr)? as u128,
                    2 => self.mem.read_u16(addr)? as u128,
                    _ => self.mem.read_u8(addr)? as u128,
                };
            } else {
                match elem_bytes {
                    16 => self.store_q(addr, self.vregs[rt as usize])?,
                    8 => self.mem.write_u64(addr, self.vregs[rt as usize] as u64)?,
                    4 => self.mem.write_u32(addr, self.vregs[rt as usize] as u32)?,
                    2 => self.mem.write_u16(addr, self.vregs[rt as usize] as u16)?,
                    _ => self.mem.write_u8(addr, self.vregs[rt as usize] as u8)?,
                }
            }
            if writeback {
                self.write_x(rn, wb_val);
            }
            return Ok(true);
        }
        if grp == 0b101 && ((insn >> 25) & 1) == 0 {
            // STP/LDP SIMD&FP: size 00/01/10 → S/D/Q pairs, imm scaled by
            // 4<<size (4/8/16 bytes). size=11 is unallocated.
            let size = (insn >> 30) & 0b11;
            if size == 0b11 {
                return Ok(false);
            }
            let bytes: u32 = 4 << size;
            let l = (insn >> 22) & 1;
            let mode = (insn >> 23) & 0b11;
            let imm = sext_u64((insn >> 15) & 0x7F, 7) as i64;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let rt2 = ((insn >> 10) & 0x1F) as u8;
            let base = self.read_x(rn);
            let scaled = (imm as u64).wrapping_mul(bytes as u64);
            let (addr, writeback, wb) = match mode {
                0b00 => (base.wrapping_add(scaled), false, 0),
                0b01 => (base, true, base.wrapping_add(scaled)),
                0b10 => (base.wrapping_add(scaled), false, 0),
                _ => (base.wrapping_add(scaled), true, base.wrapping_add(scaled)),
            };
            let addr = addr as u32;
            if l == 1 {
                let (v0, v1) = match size {
                    0 => (
                        self.mem.read_u32(addr)? as u128,
                        self.mem.read_u32(addr.wrapping_add(bytes))? as u128,
                    ),
                    1 => (
                        self.mem.read_u64(addr)? as u128,
                        self.mem.read_u64(addr.wrapping_add(bytes))? as u128,
                    ),
                    _ => (self.load_q(addr)?, self.load_q(addr.wrapping_add(bytes))?),
                };
                self.vregs[rt as usize] = v0;
                self.vregs[rt2 as usize] = v1;
            } else {
                match size {
                    0 => {
                        self.mem.write_u32(addr, self.vregs[rt as usize] as u32)?;
                        self.mem
                            .write_u32(addr.wrapping_add(bytes), self.vregs[rt2 as usize] as u32)?;
                    }
                    1 => {
                        self.mem.write_u64(addr, self.vregs[rt as usize] as u64)?;
                        self.mem
                            .write_u64(addr.wrapping_add(bytes), self.vregs[rt2 as usize] as u64)?;
                    }
                    _ => {
                        self.store_q(addr, self.vregs[rt as usize])?;
                        self.store_q(addr.wrapping_add(bytes), self.vregs[rt2 as usize])?;
                    }
                }
            }
            if writeback {
                self.write_x(rn, wb);
            }
            return Ok(true);
        }
        Ok(false)
    }

    #[inline]
    pub(super) fn load_q(&self, addr: u32) -> Result<u128> {
        Ok((self.mem.read_u64(addr)? as u128)
            | ((self.mem.read_u64(addr.wrapping_add(8))? as u128) << 64))
    }

    #[inline]
    pub(super) fn store_q(&mut self, addr: u32, v: u128) -> Result<()> {
        self.mem.write_u64(addr, v as u64)?;
        self.mem.write_u64(addr.wrapping_add(8), (v >> 64) as u64)
    }

    /// Read lane `index` of Vn, `esize` bits wide, in little-endian lane order.
    #[inline(always)]
    pub(super) fn read_vreg_elem(&self, reg: u8, index: u32, esize: u32) -> u64 {
        lane(self.vregs[reg as usize], esize, index)
    }

    /// Write lane `index` of Vn, leaving every other lane alone.
    #[inline(always)]
    pub(super) fn write_vreg_elem(&mut self, reg: u8, index: u32, esize: u32, val: u64) {
        self.vregs[reg as usize] = set_lane(self.vregs[reg as usize], esize, index, val);
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn try_load_store(&mut self, insn: u32, _next_pc: &mut u32) -> Result<bool> {
        // Exclusive accessors.
        let grp_excl = (insn >> 21) & 0x1FF;
        if (0b001000000..=0b001000011).contains(&grp_excl)
            || grp_excl == 0b001000100
            || grp_excl == 0b001000110
        {
            let sz = (insn >> 30) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let rt2 = ((insn >> 10) & 0x1F) as u8;
            let base = self.read_x(rn);
            match grp_excl {
                0b001000000 => {
                    // STXR Ws, Xt, [Xn]: succeeds only against a monitor this
                    // thread's own LDXR set at the same address. A failed one
                    // stores **nothing** and reports 1, which every guest
                    // answers by looping back to the LDXR.
                    //
                    // This used to succeed unconditionally, which was safe
                    // only while threads could lose the CPU at a blocking
                    // syscall and nowhere else: no guest puts one between the
                    // two halves of a read-modify-write, so every pair was
                    // atomic by construction. Preemption ended that, and "A
                    // Short Hike" started losing a doubly-linked-list update
                    // and calling through the null it left behind.
                    if self.exclusive.take() == Some(base as u32) {
                        let val = self.read_zr(rt);
                        self.store_by_size(base as u32, sz, val)?;
                        self.write_zr(((insn >> 16) & 0x1F) as u8, 0);
                    } else {
                        self.write_zr(((insn >> 16) & 0x1F) as u8, 1);
                    }
                }
                0b001000010 => {
                    // LDXR Xt, [Xn]
                    let val = self.load_by_size(base as u32, sz, false)?;
                    self.exclusive = Some(base as u32);
                    self.write_zr(rt, val);
                }
                0b001000001 => {
                    // STXP: 64-bit pair store, on the same monitor.
                    if self.exclusive.take() == Some(base as u32) {
                        let v0 = self.read_zr(rt);
                        let v1 = self.read_zr(rt2);
                        self.mem.write_u64(base as u32, v0)?;
                        self.mem.write_u64(base.wrapping_add(8) as u32, v1)?;
                        self.write_zr(((insn >> 16) & 0x1F) as u8, 0);
                    } else {
                        self.write_zr(((insn >> 16) & 0x1F) as u8, 1);
                    }
                }
                0b001000011 => {
                    // LDXP: 64-bit pair load
                    let v0 = self.mem.read_u64(base as u32)?;
                    let v1 = self.mem.read_u64(base.wrapping_add(8) as u32)?;
                    self.exclusive = Some(base as u32);
                    self.write_zr(rt, v0);
                    self.write_zr(rt2, v1);
                }
                0b001000100 => {
                    // STLR: store-release
                    self.store_by_size(base as u32, sz, self.read_zr(rt))?;
                }
                0b001000110 => {
                    // LDAR: load-acquire
                    let val = self.load_by_size(base as u32, sz, false)?;
                    self.write_zr(rt, val);
                }
                _ => unreachable!(),
            }
            return Ok(true);
        }

        // SIMD (V=1) memory ops: minimal Q-register subset for libnx memset.
        if ((insn >> 26) & 1) == 1 {
            return self.try_simd_load_store(insn);
        }

        // Register-offset form: bit21 == 1 (any size — the previous
        // bits[31:27]==11111 test only matched the 64-bit forms, so 8/16/32-bit
        // register-offset loads/stores fell through as "unimplemented").
        if ((insn >> 27) & 0b111) == 0b111
            && ((insn >> 26) & 1) == 0
            && ((insn >> 24) & 0b11) == 0b00
            && ((insn >> 21) & 1) == 1
        {
            let sz = (insn >> 30) & 0b11;
            let opc = (insn >> 22) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let opt = ((insn >> 13) & 0b111) as u8;
            let s = (insn >> 12) & 1;
            let offset = self.offset_from_reg(rm, opt, s, sz as u8)?;
            let addr = (self.read_x(rn) as i64).wrapping_add(offset) as u32;
            self.ld_st_opc(addr, rt, sz, opc)?;
            return Ok(true);
        }

        // Immediate offset forms: bits[29:27] == 111, V=0
        if ((insn >> 27) & 0b111) == 0b111 && ((insn >> 26) & 1) == 0 {
            let mode = (insn >> 24) & 0b11;
            let sz = (insn >> 30) & 0b11;
            let opc = (insn >> 22) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            if mode == 0b01 {
                // Unsigned offset
                let imm = ((insn >> 10) & 0xFFF) as u64;
                let scale = match sz {
                    0b00 => 1,
                    0b01 => 2,
                    0b10 => 4,
                    _ => 8,
                };
                let addr = self.read_x(rn).wrapping_add(imm.wrapping_mul(scale)) as u32;
                self.ld_st_opc(addr, rt, sz, opc)?;
                return Ok(true);
            }
            if mode == 0b00 && ((insn >> 21) & 1) == 0 {
                // Unscaled / pre / post index
                let idx = (insn >> 10) & 0b11;
                let imm = sext_u64((insn >> 12) & 0x1FF, 9) as i64;
                let base = self.read_x(rn);
                let (addr, writeback) = match idx {
                    0b00 | 0b10 => (base.wrapping_add(imm as u64), false),
                    0b01 => (base, true),                       // post-index
                    _ => (base.wrapping_add(imm as u64), true), // pre-index
                };
                self.ld_st_opc(addr as u32, rt, sz, opc)?;
                if writeback {
                    let new_base = if idx == 0b01 {
                        base.wrapping_add(imm as u64)
                    } else {
                        addr
                    };
                    self.write_x(rn, new_base);
                }
                return Ok(true);
            }
        }

        // Paired load/store: bits[29:27] == 101, V=0. The bit25==0 check
        // distinguishes pairs from the SUBS-shifted-register space (which has
        // bits[29:27]=101 too but bit25=1).
        if ((insn >> 27) & 0b111) == 0b101 && ((insn >> 26) & 1) == 0 && ((insn >> 25) & 1) == 0 {
            return self.try_pair(insn);
        }

        Ok(false)
    }

    /// One load or store named by its raw `size`:`opc` fields — the form the
    /// interpreter has after decoding. [`Acc::of`] settles what that means and
    /// [`Cpu::access`] performs it, which is the same pair the translator uses.
    #[inline(always)]
    pub(super) fn ld_st_opc(&mut self, addr: u32, rt: u8, sz: u32, opc: u32) -> Result<()> {
        let acc = Acc::of(sz as u8, opc as u8);
        self.access(addr, rt_slot(u32::from(rt), acc), acc)
    }

    /// Perform one load or store whose width, direction and sign-extension are
    /// already settled, against a `Rt` already resolved to a slot.
    #[inline(always)]
    pub(super) fn access(&mut self, addr: u32, rt: u8, acc: Acc) -> Result<()> {
        match acc {
            Acc::Load8 => {
                let v = u64::from(self.mem.read_u8(addr)?);
                self.set_reg_at(rt, v);
            }
            Acc::Load16 => {
                let v = u64::from(self.mem.read_u16(addr)?);
                self.set_reg_at(rt, v);
            }
            Acc::Load32 => {
                let v = u64::from(self.mem.read_u32(addr)?);
                self.set_reg_at(rt, v);
            }
            Acc::Load64 => {
                let v = self.mem.read_u64(addr)?;
                self.set_reg_at(rt, v);
            }
            Acc::LoadS8 => {
                let v = u64::from(self.mem.read_u8(addr)?);
                self.set_reg_at(rt, sext_u64(v, 8));
            }
            Acc::LoadS16 => {
                let v = u64::from(self.mem.read_u16(addr)?);
                self.set_reg_at(rt, sext_u64(v, 16));
            }
            Acc::LoadS32 => {
                let v = u64::from(self.mem.read_u32(addr)?);
                self.set_reg_at(rt, sext_u64(v, 32));
            }
            Acc::Store8 => self.mem.write_u8(addr, self.reg_at(rt) as u8)?,
            Acc::Store16 => self.mem.write_u16(addr, self.reg_at(rt) as u16)?,
            Acc::Store32 => self.mem.write_u32(addr, self.reg_at(rt) as u32)?,
            Acc::Store64 => self.mem.write_u64(addr, self.reg_at(rt))?,
            // PRFM: a hint. The addressing mode's writeback still happens.
            Acc::Prefetch => {}
        }
        Ok(())
    }

    /// The address a load/store touches and the value (if any) to write back
    /// into its base register.
    #[inline(always)]
    pub(super) fn indexed(base: u64, offset: i64, wb: Wb) -> (u64, Option<u64>) {
        match wb {
            Wb::None => (base.wrapping_add(offset as u64), None),
            Wb::Pre => {
                let addr = base.wrapping_add(offset as u64);
                (addr, Some(addr))
            }
            Wb::Post => (base, Some(base.wrapping_add(offset as u64))),
        }
    }

    /// The byte offset a register-offset addressing mode adds, with the
    /// extension and the scale already settled.
    ///
    /// `S` scales by log2 of the access size, not by the byte count — the byte
    /// count over-shifted table indices (`ldrsw x8,[x9,x8,lsl#2]` read entry
    /// 4x too far, loading 0 and jumping into the table itself).
    #[inline(always)]
    pub(super) fn reg_offset(&self, rm: u8, ext: Ext, shift: u8) -> i64 {
        let index = match ext {
            Ext::Uxtw => u64::from(self.reg_at(rm) as u32),
            Ext::Sxtw => sext_u64(self.reg_at(rm), 32),
            Ext::None => self.reg_at(rm),
        };
        index.wrapping_shl(u32::from(shift)) as i64
    }

    /// `LDP`/`STP`/`LDPSW`, against slots already resolved and an addressing
    /// mode already classified.
    #[inline(always)]
    pub(super) fn pair(
        &mut self,
        rt: u8,
        rt2: u8,
        rn: u8,
        offset: i64,
        kind: PairKind,
        wb: Wb,
    ) -> Result<()> {
        let base = self.reg_at(rn);
        let (addr, wb_val) = Self::indexed(base, offset, wb);
        let addr = addr as u32;
        let second = addr.wrapping_add(kind.stride());
        match kind {
            // Both halves are read before either register is written, so
            // `ldp x0, x1, [x0]` sees the memory it was pointed at.
            PairKind::Load64 => {
                let v0 = self.mem.read_u64(addr)?;
                let v1 = self.mem.read_u64(second)?;
                self.set_reg_at(rt, v0);
                self.set_reg_at(rt2, v1);
            }
            PairKind::Load32 => {
                let v0 = u64::from(self.mem.read_u32(addr)?);
                let v1 = u64::from(self.mem.read_u32(second)?);
                self.set_reg_at(rt, v0);
                self.set_reg_at(rt2, v1);
            }
            PairKind::Load32Sext => {
                let v0 = u64::from(self.mem.read_u32(addr)?);
                let v1 = u64::from(self.mem.read_u32(second)?);
                self.set_reg_at(rt, sext_u64(v0, 32));
                self.set_reg_at(rt2, sext_u64(v1, 32));
            }
            PairKind::Store64 => {
                self.mem.write_u64(addr, self.reg_at(rt))?;
                self.mem.write_u64(second, self.reg_at(rt2))?;
            }
            PairKind::Store32 => {
                self.mem.write_u32(addr, self.reg_at(rt) as u32)?;
                self.mem.write_u32(second, self.reg_at(rt2) as u32)?;
            }
        }
        if let Some(v) = wb_val {
            self.set_reg_at(rn, v);
        }
        Ok(())
    }

    #[inline(always)]
    pub(super) fn load_by_size(&self, addr: u32, sz: u32, sign: bool) -> Result<u64> {
        let raw = match sz {
            0b00 => self.mem.read_u8(addr)? as u64,
            0b01 => self.mem.read_u16(addr)? as u64,
            0b10 => self.mem.read_u32(addr)? as u64,
            _ => self.mem.read_u64(addr)?,
        };
        Ok(if sign {
            let width = match sz {
                0b00 => 8,
                0b01 => 16,
                0b10 => 32,
                _ => 64,
            };
            sext_u64(raw, width)
        } else {
            raw
        })
    }

    #[inline(always)]
    pub(super) fn store_by_size(&mut self, addr: u32, sz: u32, val: u64) -> Result<()> {
        match sz {
            0b00 => self.mem.write_u8(addr, val as u8),
            0b01 => self.mem.write_u16(addr, val as u16),
            0b10 => self.mem.write_u32(addr, val as u32),
            _ => self.mem.write_u64(addr, val),
        }
    }

    pub(super) fn offset_from_reg(&self, rm: u8, opt: u8, s: u32, sz: u8) -> Result<i64> {
        let Some(ext) = Ext::of(opt) else {
            return Err(Error::Cpu(format!("bad register offset option {}", opt)));
        };
        let shift = if s == 1 { sz } else { 0 };
        Ok(self.reg_offset(rm & 0x1F, ext, shift))
    }

    pub(super) fn try_pair(&mut self, insn: u32) -> Result<bool> {
        let opc = (insn >> 30) & 0b11;
        let l = (insn >> 22) & 1;
        // opc=01 is the LDP-signed / STGP space; only loads make sense for us.
        if opc == 0b01 && l == 0 {
            return Err(Error::Cpu(format!(
                "unimplemented tagged store-pair at {:#x}",
                self.pc
            )));
        }
        if opc == 0b11 {
            return Err(Error::Cpu(format!(
                "unimplemented pair addressing mode at {:#x}",
                self.pc
            )));
        }
        let wide = opc == 0b10;
        let kind = match (l == 1, wide, opc == 0b01) {
            (true, _, true) => PairKind::Load32Sext,
            (true, true, _) => PairKind::Load64,
            (true, false, _) => PairKind::Load32,
            (false, true, _) => PairKind::Store64,
            (false, false, _) => PairKind::Store32,
        };
        let wb = match (insn >> 23) & 0b11 {
            0b01 => Wb::Post,
            0b11 => Wb::Pre,
            _ => Wb::None,
        };
        let scale: i64 = if wide { 8 } else { 4 };
        let offset = (sext_u64((insn >> 15) & 0x7F, 7) as i64).wrapping_mul(scale);
        self.pair(
            pair_slot(insn, kind),
            pair_slot(insn >> 10, kind),
            Cpu::x_slot((insn >> 5) as u8),
            offset,
            kind,
            wb,
        )?;
        Ok(true)
    }

    // ---------- data processing: immediate ----------
}
