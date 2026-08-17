//! Loads and stores: the integer, pair, exclusive and SIMD&FP addressing
//! modes, including the structure load/store forms.

use super::bits::*;
use super::Cpu;
use crate::{Error, Result};

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
                        (self.read_x(rn).wrapping_add(self.read_x(rm).wrapping_shl(shift))) as u32,
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
                        0b01 => (base as u32, Some(updated)), // post-index
                        0b11 => (updated as u32, Some(updated)), // pre-index
                        _ => (updated as u32, None),          // unscaled
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
        // Single structure (LD1/ST1 into one lane): bit31=0, q=bit30,
        // bits[29:24]=001101, p=bit23 (post-index), bit22=1 (load) / 0
        // (store), scale in bits[15:14], selem in bits[13]/[21], lane index in
        // bits[12:10]/[12]/[13] depending on scale.
        if ((insn >> 31) & 1) == 0 && ((insn >> 24) & 0x3F) == 0b001101 {
            let q = (insn >> 30) & 1 == 1;
            let p = (insn >> 23) & 1 == 1;
            let load = (insn >> 22) & 1 == 1;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let scale = (insn >> 14) & 0b11;
            let esize = 1u32 << scale;
            let selem = (((insn >> 21) & 1) << 1 | (insn >> 13) & 1) as u32 + 1;
            let qbit = if q { 1u32 } else { 0 };
            let index = match scale {
                0 => ((insn >> 10) & 0b111) | (qbit << 3),
                1 => ((insn >> 10) & 0b11) | (qbit << 2),
                2 => ((insn >> 12) & 1) | (qbit << 1),
                _ => qbit,
            };
            let mut addr = self.read_x(rn);
            let esize_u = esize as u64;
            for xs in 0..selem {
                let reg = (rt as u32 + xs) % 32;
                let shift = (index as u32) * esize;
                let mask = (1u128 << esize) - 1;
                if load {
                    let val = self.load_by_size(addr as u32, scale, false)?;
                    self.vregs[reg as usize] =
                        (self.vregs[reg as usize] & !(mask << shift)) | ((val as u128) << shift);
                } else {
                    let val = ((self.vregs[reg as usize] >> shift) & mask) as u64;
                    self.store_by_size(addr as u32, scale, val)?;
                }
                addr = addr.wrapping_add(esize_u);
            }
            if p {
                if rm == 31 {
                    self.write_x(rn, addr);
                } else {
                    self.write_x(rn, self.read_x(rn).wrapping_add(self.read_x(rm)));
                }
            }
            return Ok(true);
        }
        // Multiple structures (LD1/LD2/LD3/LD4, ST1/ST2/ST3/ST4): bit31=0,
        // bits[29:24]=001100, bit23=p (post-index), bit22=1 (load) / 0
        // (store), L=bit21, opcode=bits[15:12] selects (rpt, selem), sz in
        // bits[11:10]. Only the `selem==1` forms are contiguous (each
        // register is a plain 64/128-bit chunk); selem>1 interleaves, which
        // isn't needed yet.
        if ((insn >> 31) & 1) == 0
            && ((insn >> 28) & 0b11) == 0b00
            && ((insn >> 24) & 0x0F) == 0b1100
        {
            let q = (insn >> 30) & 1 == 1;
            let load = (insn >> 22) & 1 == 1;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let opcode = (insn >> 12) & 0b1111;
            let (rpt, selem) = match opcode {
                0b0000 => (1, 4),
                0b0010 => (4, 1),
                0b0100 => (1, 3),
                0b0110 => (3, 1),
                0b0111 => (1, 1),
                0b1000 => (1, 2),
                0b1010 => (2, 1),
                _ => return Ok(false),
            };
            if selem != 1 {
                return Ok(false);
            }
            let vec_bytes: u32 = if q { 16 } else { 8 };
            let addr = self.read_x(rn);
            for i in 0..rpt {
                let a = addr.wrapping_add((i as u64) * vec_bytes as u64) as u32;
                if load {
                    self.vregs[(rt + i) as usize % 32] = self.load_q(a)?;
                } else {
                    self.store_q(a, self.vregs[(rt + i) as usize % 32])?;
                }
            }
            if rm != 31 {
                let total = (rpt as u64) * (vec_bytes as u64);
                self.write_x(rn, addr.wrapping_add(total));
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
                    _ => (
                        self.load_q(addr)?,
                        self.load_q(addr.wrapping_add(bytes))?,
                    ),
                };
                self.vregs[rt as usize] = v0;
                self.vregs[rt2 as usize] = v1;
            } else {
                match size {
                    0 => {
                        self.mem.write_u32(addr, self.vregs[rt as usize] as u32)?;
                        self.mem.write_u32(addr.wrapping_add(bytes), self.vregs[rt2 as usize] as u32)?;
                    }
                    1 => {
                        self.mem.write_u64(addr, self.vregs[rt as usize] as u64)?;
                        self.mem.write_u64(addr.wrapping_add(bytes), self.vregs[rt2 as usize] as u64)?;
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
        Ok((self.mem.read_u64(addr)? as u128) | ((self.mem.read_u64(addr.wrapping_add(8))? as u128) << 64))
    }

    #[inline]
    pub(super) fn store_q(&mut self, addr: u32, v: u128) -> Result<()> {
        self.mem.write_u64(addr, v as u64)?;
        self.mem.write_u64(addr.wrapping_add(8), (v >> 64) as u64)
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn try_load_store(&mut self, insn: u32, _next_pc: &mut u32) -> Result<bool> {
        // Exclusive accessors.
        let grp_excl = (insn >> 21) & 0x1FF;
        if (0b001000000..=0b001000011).contains(&grp_excl) || grp_excl == 0b001000100 || grp_excl == 0b001000110 {
            let sz = (insn >> 30) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let rt2 = ((insn >> 10) & 0x1F) as u8;
            let base = self.read_x(rn);
            match grp_excl {
                0b001000000 => {
                    // STXR Ws, Xt, [Xn]
                    let val = self.read_zr(rt);
                    self.store_by_size(base as u32, sz, val)?;
                    self.write_zr(((insn >> 16) & 0x1F) as u8, 0); // success
                }
                0b001000010 => {
                    // LDXR Xt, [Xn]
                    let val = self.load_by_size(base as u32, sz, false)?;
                    self.write_zr(rt, val);
                }
                0b001000001 => {
                    // STXP: 64-bit pair store
                    let v0 = self.read_zr(rt);
                    let v1 = self.read_zr(rt2);
                    self.mem.write_u64(base as u32, v0)?;
                    self.mem.write_u64(base.wrapping_add(8) as u32, v1)?;
                    self.write_zr(((insn >> 16) & 0x1F) as u8, 0);
                }
                0b001000011 => {
                    // LDXP: 64-bit pair load
                    let v0 = self.mem.read_u64(base as u32)?;
                    let v1 = self.mem.read_u64(base.wrapping_add(8) as u32)?;
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
                let addr = self
                    .read_x(rn)
                    .wrapping_add(imm.wrapping_mul(scale))
                    as u32;
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
                    0b01 => (base, true),                                  // post-index
                    _ => (base.wrapping_add(imm as u64), true),            // pre-index
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
        if ((insn >> 27) & 0b111) == 0b101
            && ((insn >> 26) & 1) == 0
            && ((insn >> 25) & 1) == 0
        {
            return self.try_pair(insn);
        }

        Ok(false)
    }

    pub(super) fn ld_st_opc(&mut self, addr: u32, rt: u8, sz: u32, opc: u32) -> Result<()> {
        // opc selects the access: 00 = STR, 01 = LDR, 10/11 = sign-extending
        // loads (LDRSB/LDRSH/LDRSW). The load bit is NOT opc&1 — treating
        // opc=10 as a store silently corrupted the target (observed as a
        // bogus `ldrsw` index in NX-Shell's tokenizer).
        //
        // size=11 with opc=10/11 is PRFM (prefetch hint) — it must NOT be
        // executed as a sign-extending load, or the load clobbers a register
        // (libtransistor's memcpy starts with `prfm pldl1keep, [x1]`, and
        // treating it as `ldrsw x0, [x1]` made memcpy write to the source
        // magic value as an address, zero-filling the real destination).
        if sz == 0b11 && opc >= 0b10 {
            return Ok(());
        }
        let load = opc != 0b00;
        let sign = (opc >> 1) == 1;
        if load {
            let val = self.load_by_size(addr, sz, sign)?;
            self.write_zr(rt, val);
        } else {
            self.store_by_size(addr, sz, self.read_zr(rt))?;
        }
        Ok(())
    }

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

    pub(super) fn store_by_size(&mut self, addr: u32, sz: u32, val: u64) -> Result<()> {
        match sz {
            0b00 => self.mem.write_u8(addr, val as u8),
            0b01 => self.mem.write_u16(addr, val as u16),
            0b10 => self.mem.write_u32(addr, val as u32),
            _ => self.mem.write_u64(addr, val),
        }
    }

    pub(super) fn offset_from_reg(&self, rm: u8, opt: u8, s: u32, sz: u8) -> Result<i64> {
        // Register-offset loads/stores shift Rm by `LSL #scale` where scale is
        // log2(size) (2 for word, 3 for doubleword), NOT the byte count — the
        // byte count over-shifted table indices (e.g. `ldrsw x8,[x9,x8,lsl#2]`
        // read entry 4x too far, loading 0 and jumping into the table itself).
        let shift = if s == 1 { sz as u32 } else { 0 };
        let v = self.read_zr(rm);
        let ext = match opt {
            0b011 => v,                                       // LSL / UXTX
            0b010 => (v as u32) as i64 as u64,                // UXTW
            0b111 => sext_u64(v, 32),                         // SXTW
            0b000 => (v as u8) as u64,                        // UXTB
            0b001 => (v as u16) as u64,                       // UXTH
            0b100 => sext_u64(v, 8),                          // SXTB
            0b101 => sext_u64(v, 16),                         // SXTH
            0b110 => v,                                       // SXTX
            _ => return Err(Error::Cpu(format!("bad register offset option {}", opt))),
        };
        Ok((ext.wrapping_shl(shift)) as i64)
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
        let sz = if opc == 0b10 { 0b11 } else { 0b10 };
        let scale = if sz == 0b11 { 8u64 } else { 4 };
        let mode = (insn >> 23) & 0b11;
        let imm = sext_u64((insn >> 15) & 0x7F, 7) as i64;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rt = (insn & 0x1F) as u8;
        let rt2 = ((insn >> 10) & 0x1F) as u8;
        let scaled = (imm as u64).wrapping_mul(scale);

        let base = self.read_x(rn);
        let (addr, writeback, wb_val) = match mode {
            0b00 => (base.wrapping_add(scaled), false, 0),        // signed offset
            0b01 => (base, true, base.wrapping_add(scaled)),      // post-index
            0b10 => (base.wrapping_add(scaled), false, 0),        // offset
            _ => (base.wrapping_add(scaled), true, base.wrapping_add(scaled)), // pre-index
        };
        let addr = addr as u32;

        if l == 1 {
            // LDP: load rt, rt2
            let v0 = if sz == 0b11 {
                self.mem.read_u64(addr)?
            } else {
                let w = self.mem.read_u32(addr)?;
                if opc == 0b01 {
                    sext_u64(w as u64, 32)
                } else {
                    w as u64
                }
            };
            let v1 = if sz == 0b11 {
                self.mem.read_u64(addr.wrapping_add(scale as u32))?
            } else {
                let w = self.mem.read_u32(addr.wrapping_add(scale as u32))?;
                if opc == 0b01 {
                    sext_u64(w as u64, 32)
                } else {
                    w as u64
                }
            };
            self.write_zr(rt, v0);
            self.write_zr(rt2, v1);
        } else {
            // STP: store rt, rt2
            if sz == 0b11 {
                self.mem.write_u64(addr, self.read_zr(rt))?;
                self.mem.write_u64(addr.wrapping_add(8), self.read_zr(rt2))?;
            } else {
                self.mem.write_u32(addr, self.read_zr(rt) as u32)?;
                self.mem.write_u32(addr.wrapping_add(4), self.read_zr(rt2) as u32)?;
            }
        }
        if writeback {
            self.write_x(rn, wb_val);
        }
        Ok(true)
    }

    // ---------- data processing: immediate ----------
}
