//! Register file layout, register access and condition flags.

use super::*;

/// Meaningful register slots: X0..=X30, then the zero register's discard slot and SP,
/// so register 31's meaning is an index chosen at decode time.
pub(crate) const REG_SLOTS: usize = 34;

/// Slots allocated: the full `u8` range, so indexing by a slot byte needs no bounds check.
pub(crate) const REG_FILE: usize = 256;
const _: () = assert!(REG_FILE == u8::MAX as usize + 1 && REG_FILE >= REG_SLOTS);

/// A slot as an index into the register file.
#[inline(always)]
fn reg_slot(slot: u8) -> usize {
    debug_assert!(
        (slot as usize) < REG_SLOTS,
        "register slot {slot} is not one"
    );
    slot as usize
}
/// Reads of `XZR`; never written.
const ZR_SLOT: usize = 31;
/// The read path indexes with the encoding's 5-bit field, so this must be 31.
const _: () = assert!(ZR_SLOT == 31);
/// Writes to `XZR` land here.
pub(crate) const ZR_DISCARD: usize = 32;

/// Exposed for difftest harnesses, which must skip it: its contents are not guest state.
pub const DISCARD_SLOT: usize = ZR_DISCARD;
pub(crate) const SP_SLOT: usize = 33;

/// Per condition code, a 16-bit mask with bit `nzcv` set when the code holds.
/// Branchless evaluation of `B.cond`.
pub(crate) const CONDITION_MASKS: [u16; 16] = condition_masks();

const fn condition_masks() -> [u16; 16] {
    let mut table = [0u16; 16];
    let mut cond = 0usize;
    while cond < 16 {
        let mut nzcv = 0usize;
        while nzcv < 16 {
            let n = (nzcv >> 3) & 1;
            let z = (nzcv >> 2) & 1;
            let c = (nzcv >> 1) & 1;
            let v = nzcv & 1;
            let holds = match cond {
                0x0 => z == 1,           // EQ
                0x1 => z == 0,           // NE
                0x2 => c == 1,           // CS
                0x3 => c == 0,           // CC
                0x4 => n == 1,           // MI
                0x5 => n == 0,           // PL
                0x6 => v == 1,           // VS
                0x7 => v == 0,           // VC
                0x8 => c == 1 && z == 0, // HI
                0x9 => c == 0 || z == 1, // LS
                0xA => n == v,           // GE
                0xB => n != v,           // LT
                0xC => z == 0 && n == v, // GT
                0xD => z == 1 || n != v, // LE
                _ => true,               // AL / NV
            };
            if holds {
                table[cond] |= 1 << nzcv;
            }
            nzcv += 1;
        }
        cond += 1;
    }
    table
}

impl Cpu {
    // ---- register access ----

    #[inline]
    pub fn get_pc(&self) -> u32 {
        self.pc
    }

    /// SP from [`SP_SLOT`] in A64 or `r13` in AArch32.
    #[inline]
    pub fn sp(&self) -> u64 {
        match self.mode {
            ExecMode::A64 => self.regs[SP_SLOT],
            ExecMode::A32 => self.regs[13],
        }
    }

    pub fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    /// Read a register where 31 is SP.
    #[inline(always)]
    pub fn read_x(&self, idx: u8) -> u64 {
        self.regs[reg_slot(Self::x_slot(idx))]
    }

    /// The slot a register number names when 31 means SP.
    #[inline(always)]
    pub(crate) fn x_slot(idx: u8) -> u8 {
        let idx = idx & 0x1F;
        idx + (SP_SLOT as u8 - 31) * u8::from(idx == 31)
    }

    /// The slot a register number names when 31 means `XZR` and is written.
    #[inline(always)]
    pub(crate) fn zr_write_slot(idx: u8) -> u8 {
        let idx = idx & 0x1F;
        idx + (ZR_DISCARD as u8 - 31) * u8::from(idx == 31)
    }

    pub fn read_reg(&self, idx: u8) -> u64 {
        self.read_x(idx)
    }

    pub fn read_vreg(&self, idx: u8) -> u128 {
        self.vregs.0[idx as usize]
    }

    pub fn tls_base(&self) -> u32 {
        self.tpidr as u32
    }

    pub fn set_vreg(&mut self, idx: u8, val: u128) {
        self.vregs.0[idx as usize] = val;
    }

    /// Read a register where 31 is `XZR`.
    #[inline(always)]
    pub(crate) fn read_zr(&self, idx: u8) -> u64 {
        self.regs[(idx & 0x1F) as usize]
    }

    /// Write a register where 31 is `XZR`.
    #[inline(always)]
    pub(crate) fn write_zr(&mut self, idx: u8, val: u64) {
        self.regs[reg_slot(Self::zr_write_slot(idx))] = val;
    }

    /// Write a register where 31 is SP.
    #[inline(always)]
    pub(crate) fn write_x(&mut self, idx: u8, val: u64) {
        self.regs[reg_slot(Self::x_slot(idx))] = val;
    }

    /// Read the register file by slot; see [`REG_SLOTS`].
    #[inline(always)]
    pub(crate) fn reg_at(&self, slot: u8) -> u64 {
        self.regs[reg_slot(slot)]
    }

    #[inline(always)]
    pub(crate) fn set_reg_at(&mut self, slot: u8, val: u64) {
        self.regs[reg_slot(slot)] = val;
    }

    pub fn set_reg(&mut self, idx: u8, val: u64) {
        self.write_zr(idx, val);
    }

    pub fn read_u32_reg(&self, idx: u8) -> u32 {
        self.read_zr(idx) as u32
    }

    pub fn set_pc_and_sp(&mut self, pc: u32, sp: u64) {
        self.pc = pc;
        match self.mode {
            ExecMode::A64 => self.regs[SP_SLOT] = sp,
            ExecMode::A32 => self.regs[13] = sp,
        }
    }

    #[inline]
    pub fn nzcv(&self) -> u32 {
        self.nzcv
    }

    #[inline(always)]
    pub(crate) fn condition_holds(&self, cond: u8) -> bool {
        (CONDITION_MASKS[(cond & 0xF) as usize] >> (self.nzcv >> 28)) & 1 == 1
    }

    #[inline(always)]
    pub(crate) fn mask(sf: bool) -> u64 {
        if sf {
            u64::MAX
        } else {
            u32::MAX as u64
        }
    }

    /// `a + b + carry_in` as (result, carry-out, overflow), with operands masked to the operation size.
    #[inline(always)]
    pub(crate) fn add_carry_overflow(a: u64, b: u64, carry_in: u64, sf: bool) -> (u64, u32, u32) {
        let mask = Self::mask(sf);
        let a = a & mask;
        let b = b & mask;
        // Two exclusive `overflowing_add`s form the carry chain without u128 or a width branch.
        let (sum, c1) = a.overflowing_add(b);
        let (sum, c2) = sum.overflowing_add(carry_in);
        let carry = if sf {
            u32::from(c1 | c2)
        } else {
            ((sum >> 32) & 1) as u32
        };
        let result = sum & mask;
        // Both operands the same sign and the result a different one.
        let sign = 1u64 << (if sf { 63 } else { 31 });
        let overflow = u32::from((!(a ^ b) & (a ^ result) & sign) != 0);
        (result, carry, overflow)
    }

    pub(crate) fn set_nzcv_from_alu(&mut self, result: u64, sf: bool, carry: u32, overflow: u32) {
        let n = ((result >> (if sf { 63 } else { 31 })) & 1) as u32;
        let z = (result == 0) as u32;
        self.nzcv = (n << 31) | (z << 30) | (carry << 29) | (overflow << 28);
    }

    pub(crate) fn set_nzcv_from_compare(
        &mut self,
        a: u64,
        b: u64,
        sub: bool,
        carry_in: u64,
        sf: bool,
    ) {
        let (result, carry, overflow) = if sub {
            Self::add_carry_overflow(a, !b, carry_in, sf)
        } else {
            Self::add_carry_overflow(a, b, carry_in, sf)
        };
        self.set_nzcv_from_alu(result, sf, carry, overflow);
    }

    /// The ADD/SUB core. `sp_form`: register 31 is SP (immediate and extended forms)
    /// rather than XZR (shifted-register form).
    #[inline(always)]
    pub(crate) fn add_sub(
        &mut self,
        rd: u8,
        rn: u8,
        rhs: u64,
        set_flags: bool,
        sub: bool,
        sf: bool,
        sp_form: bool,
    ) {
        // Rd=31 is SP only for the non-flag-setting immediate and extended forms.
        let rd = if set_flags || !sp_form {
            Self::zr_write_slot(rd)
        } else {
            Self::x_slot(rd)
        };
        let rn = if sp_form { Self::x_slot(rn) } else { rn & 0x1F };
        // Subtraction is addition of the inverted operand with a carry in.
        self.add_sub_pre(
            rd,
            rn,
            if sub { !rhs } else { rhs },
            u8::from(sub),
            set_flags,
            sf,
        );
    }

    /// ADD/SUB with direction folded into `rhs`/`carry` and register 31 resolved; shared
    /// by [`Cpu::add_sub`] and the block translator.
    #[inline(always)]
    pub(crate) fn add_sub_pre(
        &mut self,
        rd: u8,
        rn: u8,
        rhs: u64,
        carry: u8,
        set_flags: bool,
        sf: bool,
    ) {
        let a = self.reg_at(rn) & Self::mask(sf);
        let (result, c, v) = Self::add_carry_overflow(a, rhs, u64::from(carry), sf);
        if set_flags {
            self.set_nzcv_from_alu(result, sf, c, v);
        }
        self.set_reg_at(rd, result);
    }
}
