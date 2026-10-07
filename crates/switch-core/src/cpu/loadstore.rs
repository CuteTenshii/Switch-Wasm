//! Loads and stores: integer, pair, exclusive and SIMD&FP addressing modes,
//! including the structure forms. The access types here are shared
//! with the JIT.

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

/// A load or store's width, direction and sign-extension, from `size`:`opc`.
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
    /// Sign-extending loads into a 64-bit register (`opc == 10`).
    LoadS8,
    LoadS16,
    LoadS32,
    /// Sign-extending loads into a W register (`opc == 11`); the top half is zeroed.
    LoadS8To32,
    LoadS16To32,
    /// `PRFM`: no effect, but the addressing mode's writeback still happens.
    Prefetch,
}

impl Acc {
    pub(super) fn writes_rt(self) -> bool {
        !matches!(
            self,
            Acc::Store8 | Acc::Store16 | Acc::Store32 | Acc::Store64 | Acc::Prefetch
        )
    }

    /// `size == 11` with `opc == 1x` is `PRFM`, not a sign-extending load.
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
            (0b10, 0b00) => Acc::LoadS8,
            (0b10, 0b01) => Acc::LoadS16,
            (0b11, 0b00) => Acc::LoadS8To32,
            (0b11, 0b01) => Acc::LoadS16To32,
            (_, _) => Acc::LoadS32,
        }
    }
}

/// How a register-offset load/store extends its index register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Ext {
    /// `UXTW`: the low 32 bits, zero-extended.
    Uxtw,
    /// `SXTW`: the low 32 bits, sign-extended.
    Sxtw,
    /// `LSL`, `UXTX` and `SXTX`.
    None,
}

impl Ext {
    /// `None` for the undefined `option` encodings.
    pub(super) fn of(option: u8) -> Option<Ext> {
        match option {
            0b010 => Some(Ext::Uxtw),
            0b110 => Some(Ext::Sxtw),
            0b011 | 0b111 => Some(Ext::None),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PairKind {
    Load32,
    /// `LDPSW`: two words, each sign-extended to 64 bits.
    Load32Sext,
    Load64,
    Store32,
    Store64,
    /// SIMD&FP `LDP`/`STP` of Q registers; Rt and Rt2 index `vregs`.
    LoadQ,
    StoreQ,
}

impl PairKind {
    pub(super) fn loads(self) -> bool {
        matches!(
            self,
            PairKind::Load32 | PairKind::Load32Sext | PairKind::Load64 | PairKind::LoadQ
        )
    }

    pub(super) fn vector(self) -> bool {
        matches!(self, PairKind::LoadQ | PairKind::StoreQ)
    }
}

/// The slot `Rt` names, which depends on whether the access writes it.
#[inline]
pub(super) fn rt_slot(rt: u32, acc: Acc) -> u8 {
    if acc.writes_rt() {
        Cpu::zr_write_slot(rt as u8)
    } else {
        (rt & 0x1F) as u8
    }
}

#[inline]
pub(super) fn pair_slot(rt: u32, kind: PairKind) -> u8 {
    if kind.loads() && !kind.vector() {
        Cpu::zr_write_slot(rt as u8)
    } else {
        (rt & 0x1F) as u8
    }
}

impl Cpu {
    /// SIMD&FP loads and stores (V=1).
    pub(super) fn try_simd_load_store(&mut self, insn: u32) -> Result<bool> {
        let grp = (insn >> 27) & 0b111;
        // Scalar SIMD&FP LDR/STR: unsigned-offset (mode 01) and unscaled (mode 00).
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
                // Unsigned offset: bit 21 is part of imm12 here, not a register flag.
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
                // Unscaled, post-index and pre-index: bits[11:10] pick which.
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
        // AdvSIMD load/store single structure; scale 0b11 is LD1R..LD4R.
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
            // The lane index is spread across Q, S and `size`.
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
                // Rm == 31 is the immediate form: increment by bytes transferred.
                let step = if rm == 31 { offs } else { self.read_x(rm) };
                self.write_x(rn, base.wrapping_add(step));
            }
            return Ok(true);
        }
        // AdvSIMD load/store multiple structures.
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
                // Contiguous: each register is one plain chunk of memory.
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
                    // Loading a 64-bit register zeroes its top half.
                    for i in 0..rpt * selem {
                        self.vregs[((u32::from(rt) + i) % 32) as usize] &= elem_mask(64);
                    }
                }
                for r in 0..rpt {
                    for lane in 0..lanes {
                        for reg in (u32::from(rt) + r..).take(selem as usize) {
                            let addr = base.wrapping_add(offs) as u32;
                            if load {
                                let elem = self.load_by_size(addr, size, false)?;
                                self.write_vreg_elem((reg % 32) as u8, lane, esize, elem);
                            } else {
                                let elem = self.read_vreg_elem((reg % 32) as u8, lane, esize);
                                self.store_by_size(addr, size, elem)?;
                            }
                            offs += ebytes;
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
        // SIMD&FP register-offset form (bit 21 set).
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
            // SIMD&FP unsigned-immediate (mode 01) and unscaled (mode 00) forms.
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
            // B/H/S/D/Q use size 00/01/10/11/00 (opc 10/11); imm is scaled by element size.
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
                let imm = (((insn >> 10) & 0xFFF) as u64) << shift;
                (base.wrapping_add(imm) as u32, false, 0)
            } else if mode == 0b00 && ((insn >> 21) & 1) == 0 {
                // Unscaled/pre/post-indexed: bits[11:10] select the mode.
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
            // STP/LDP SIMD&FP: size 00/01/10 is S/D/Q; 11 is unallocated.
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

    /// Write lane `index` of Vn, leaving the other lanes alone.
    #[inline(always)]
    pub(super) fn write_vreg_elem(&mut self, reg: u8, index: u32, esize: u32, val: u64) {
        self.vregs[reg as usize] = set_lane(self.vregs[reg as usize], esize, index, val);
    }

    /// `LDXR`/`LDAXR`: load and arm this thread's monitor.
    #[inline(always)]
    pub(super) fn load_exclusive(&mut self, rt: u8, rn: u8, sz: u8) -> Result<()> {
        let addr = self.reg_at(rn) as u32;
        let val = self.load_by_size(addr, u32::from(sz), false)?;
        self.exclusive = Some(addr);
        self.set_reg_at(rt, val);
        Ok(())
    }

    /// `STXR`/`STLXR`: stores only against this thread's monitor at the same
    /// address; otherwise stores nothing and writes 1 to `rs`.
    #[inline(always)]
    pub(super) fn store_exclusive(&mut self, rs: u8, rt: u8, rn: u8, sz: u8) -> Result<()> {
        let addr = self.reg_at(rn) as u32;
        if self.exclusive.take() == Some(addr) {
            let val = self.reg_at(rt);
            self.store_by_size(addr, u32::from(sz), val)?;
            self.set_reg_at(rs, 0);
        } else {
            self.set_reg_at(rs, 1);
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn try_load_store(&mut self, insn: u32, _next_pc: &mut u32) -> Result<bool> {
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
                0b001000000 => self.store_exclusive(
                    Self::zr_write_slot(((insn >> 16) & 0x1F) as u8),
                    rt,
                    Self::x_slot(rn),
                    sz as u8,
                )?,
                0b001000010 => {
                    self.load_exclusive(Self::zr_write_slot(rt), Self::x_slot(rn), sz as u8)?
                }
                0b001000001 => {
                    // STXP: `sz` 10 is two words, 11 two doublewords.
                    if self.exclusive.take() == Some(base as u32) {
                        let v0 = self.read_zr(rt);
                        let v1 = self.read_zr(rt2);
                        if sz == 0b10 {
                            self.mem.write_u32(base as u32, v0 as u32)?;
                            self.mem.write_u32(base.wrapping_add(4) as u32, v1 as u32)?;
                        } else {
                            self.mem.write_u64(base as u32, v0)?;
                            self.mem.write_u64(base.wrapping_add(8) as u32, v1)?;
                        }
                        self.write_zr(((insn >> 16) & 0x1F) as u8, 0);
                    } else {
                        self.write_zr(((insn >> 16) & 0x1F) as u8, 1);
                    }
                }
                0b001000011 => {
                    // LDXP: the 32-bit form zero-extends each half.
                    let (v0, v1) = if sz == 0b10 {
                        (
                            u64::from(self.mem.read_u32(base as u32)?),
                            u64::from(self.mem.read_u32(base.wrapping_add(4) as u32)?),
                        )
                    } else {
                        (
                            self.mem.read_u64(base as u32)?,
                            self.mem.read_u64(base.wrapping_add(8) as u32)?,
                        )
                    };
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

        if ((insn >> 26) & 1) == 1 {
            return self.try_simd_load_store(insn);
        }

        // Register-offset form: bit 21 set.
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

        // Immediate offset forms: bits[29:27] == 111, V=0.
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

        // Pairs: bits[29:27] == 101, V=0, bit 25 clear (set is SUBS shifted register).
        if ((insn >> 27) & 0b111) == 0b101 && ((insn >> 26) & 1) == 0 && ((insn >> 25) & 1) == 0 {
            return self.try_pair(insn);
        }

        Ok(false)
    }

    /// One load or store named by its raw `size`:`opc` fields.
    #[inline(always)]
    pub(super) fn ld_st_opc(&mut self, addr: u32, rt: u8, sz: u32, opc: u32) -> Result<()> {
        let acc = Acc::of(sz as u8, opc as u8);
        self.access(addr, rt_slot(u32::from(rt), acc), acc)
    }

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
            Acc::LoadS8To32 => {
                let v = u64::from(self.mem.read_u8(addr)?);
                self.set_reg_at(rt, u64::from(sext_u64(v, 8) as u32));
            }
            Acc::LoadS16To32 => {
                let v = u64::from(self.mem.read_u16(addr)?);
                self.set_reg_at(rt, u64::from(sext_u64(v, 16) as u32));
            }
            Acc::LoadS32 => {
                let v = u64::from(self.mem.read_u32(addr)?);
                self.set_reg_at(rt, sext_u64(v, 32));
            }
            Acc::Store8 => self.mem.write_u8(addr, self.reg_at(rt) as u8)?,
            Acc::Store16 => self.mem.write_u16(addr, self.reg_at(rt) as u16)?,
            Acc::Store32 => self.mem.write_u32(addr, self.reg_at(rt) as u32)?,
            Acc::Store64 => self.mem.write_u64(addr, self.reg_at(rt))?,
            // PRFM: the writeback still happens.
            Acc::Prefetch => {}
        }
        Ok(())
    }

    /// The accessed address and the base writeback value, if any.
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

    /// The byte offset a register-offset mode adds; `S` scales by log2 of the size.
    #[inline(always)]
    pub(super) fn reg_offset(&self, rm: u8, ext: Ext, shift: u8) -> i64 {
        let index = match ext {
            Ext::Uxtw => u64::from(self.reg_at(rm) as u32),
            Ext::Sxtw => sext_u64(self.reg_at(rm), 32),
            Ext::None => self.reg_at(rm),
        };
        index.wrapping_shl(u32::from(shift)) as i64
    }

    /// `LDP`/`STP`/`LDPSW`.
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
        match kind {
            // Read both halves before writing, so `ldp x0, x1, [x0]` works.
            PairKind::Load64 => {
                let (v0, v1) = self.mem.read_u64_pair(addr)?;
                self.set_reg_at(rt, v0);
                self.set_reg_at(rt2, v1);
            }
            PairKind::Load32 => {
                let (v0, v1) = self.mem.read_u32_pair(addr)?;
                self.set_reg_at(rt, u64::from(v0));
                self.set_reg_at(rt2, u64::from(v1));
            }
            PairKind::Load32Sext => {
                let (v0, v1) = self.mem.read_u32_pair(addr)?;
                self.set_reg_at(rt, sext_u64(u64::from(v0), 32));
                self.set_reg_at(rt2, sext_u64(u64::from(v1), 32));
            }
            PairKind::Store64 => {
                self.mem
                    .write_u64_pair(addr, self.reg_at(rt), self.reg_at(rt2))?;
            }
            PairKind::Store32 => {
                self.mem
                    .write_u32_pair(addr, self.reg_at(rt) as u32, self.reg_at(rt2) as u32)?;
            }
            // As `try_simd_load_store` does it, so a fault leaves the same partial state.
            PairKind::LoadQ => {
                let (v0, v1) = (self.load_q(addr)?, self.load_q(addr.wrapping_add(16))?);
                self.vregs[rt as usize] = v0;
                self.vregs[rt2 as usize] = v1;
            }
            PairKind::StoreQ => {
                self.store_q(addr, self.vregs[rt as usize])?;
                self.store_q(addr.wrapping_add(16), self.vregs[rt2 as usize])?;
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
        // opc=01 is LDPSW for loads; the store form is STGP.
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
}
