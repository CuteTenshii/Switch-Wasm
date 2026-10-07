//! Turning a translated block into wasm.
//!
//! A block function takes the address of the [`crate::cpu::Cpu`] in linear
//! memory and reaches guest state through baked [`Layout`] offsets. Guest
//! accesses walk the page table inline and make exactly the checks of
//! [`crate::mem::Memory::peek`]/[`crate::mem::Memory::poke`], handing the
//! instruction back to the interpreter when they decline. `run` returns how
//! many instructions retired, with [`LEFT`] set when it left at a taken branch.
//! A block with anything the emitter cannot write is [`Refused`].

use super::decode::{decode, translate, Decoded};
use super::ir::{Block, Exit, Op};
use super::wasm::{Func, Module, I32, I64};
use crate::cpu::bits::mask_of_width;
use crate::cpu::loadstore::{Acc, Ext, PairKind, Wb};
use crate::cpu::{Cpu, CONDITION_MASKS};
use crate::mem::{PAGE_BITS, PAGE_SIZE};

/// Whether the emitter has a way to write `insn` out as wasm. Block-ending and
/// branching instructions answer `false`: those are exits, not ops.
pub fn emits(insn: u32) -> bool {
    // Any aligned address works for PC-relative forms.
    const REPRESENTATIVE_PC: u32 = 0x0800_0000;
    let Decoded::Op(op) = decode(insn, REPRESENTATIVE_PC) else {
        return false;
    };
    let mut f = scratch_func();
    Emitter {
        f: &mut f,
        layout: Layout {
            regs: 0,
            nzcv: 0,
            pc: 0,
            pages: 0,
            read_watch_lo: 0,
            read_watch_hi: 0,
            watch_lo: 0,
            watch_hi: 0,
            readonly_lo: 0,
            readonly_hi: 0,
            watched: 0,
        },
    }
    .op(&op, 0)
}

/// Whether the emitted form of `insn` can decline and hand the instruction
/// back to the interpreter (every guest access can). Derived by emitting the
/// instruction under two retired counts and comparing the bodies.
pub fn defers(insn: u32) -> bool {
    const REPRESENTATIVE_PC: u32 = 0x0800_0000;
    let Decoded::Op(op) = decode(insn, REPRESENTATIVE_PC) else {
        return false;
    };
    let body = |retired: usize| {
        let mut f = scratch_func();
        let layout = Layout {
            regs: 0,
            nzcv: 0,
            pc: 0,
            pages: 0,
            read_watch_lo: 0,
            read_watch_hi: 0,
            watch_lo: 0,
            watch_hi: 0,
            readonly_lo: 0,
            readonly_hi: 0,
            watched: 0,
        };
        Emitter { f: &mut f, layout }.op(&op, retired);
        f.code().to_vec()
    };
    body(0) != body(1)
}

/// A function body with the locals every emitted block declares.
fn scratch_func() -> Func {
    let mut f = Func::new();
    f.locals(1, 6, I64);
    f.locals(1, 5, I32);
    f
}

/// Byte offsets of the guest state an emitted block touches, from the pointer
/// it is handed. Taken from `offset_of!`, since `Cpu` is not `#[repr(C)]`.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    /// Start of the `[u64; REG_FILE]` register file.
    pub regs: u32,
    /// The packed NZCV word, in its architectural bit positions.
    pub nzcv: u32,
    /// The guest program counter, written only by a taken branch leaving the block.
    pub pc: u32,
    /// Where the page table's pointer is kept (the table itself is boxed).
    pub pages: u32,
    /// The read watchpoint, as the two `u32`s of its `[start, end)`.
    pub read_watch_lo: u32,
    pub read_watch_hi: u32,
    /// The write watchpoint, likewise.
    pub watch_lo: u32,
    pub watch_hi: u32,
    /// The envelope of the write-protected ranges, likewise.
    pub readonly_lo: u32,
    pub readonly_hi: u32,
    /// Where the pointer to the cached-pages bitmap is kept; null until
    /// something watches a page.
    pub watched: u32,
}

impl Layout {
    /// Where that state sits inside a real [`Cpu`].
    pub const fn of_cpu() -> Layout {
        let mem = std::mem::offset_of!(Cpu, mem) as u32;
        let m = crate::mem::Offsets::OF_MEMORY;
        Layout {
            regs: std::mem::offset_of!(Cpu, regs) as u32,
            nzcv: std::mem::offset_of!(Cpu, nzcv) as u32,
            pc: std::mem::offset_of!(Cpu, pc) as u32,
            pages: mem + m.pages,
            read_watch_lo: mem + m.read_watch_lo,
            read_watch_hi: mem + m.read_watch_hi,
            watch_lo: mem + m.watch_lo,
            watch_hi: mem + m.watch_hi,
            readonly_lo: mem + m.readonly_lo,
            readonly_hi: mem + m.readonly_hi,
            watched: mem + m.watched,
        }
    }
}

/// Log2 bytes per page-table entry. An entry is an `Option<Box<_>>`, a bare
/// pointer with `None` as null, and only `wasm32` runs emitted code.
const PAGE_ENTRY_SHIFT: i32 = 2;
const _: () = assert!(
    std::mem::size_of::<Option<Box<[u8; PAGE_SIZE]>>>() == std::mem::size_of::<*const u8>(),
    "an unmapped page is emitted as a null entry, which needs Option<Box<_>> to be one pointer"
);
#[cfg(target_arch = "wasm32")]
const _: () = assert!(
    std::mem::size_of::<Option<Box<[u8; PAGE_SIZE]>>>() == 1 << PAGE_ENTRY_SHIFT,
    "emitted code indexes the page table by this shift, so it has to be the entry's size"
);

/// Alignment hints, as log2. Guest addresses promise no alignment.
const ALIGN_4: u8 = 2;
const ALIGN_8: u8 = 3;
const UNALIGNED: u8 = 0;

/// Why a block was not written out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// An exit the walk over the body never reached.
    ControlFlow,
    /// An op with no emitter, as the instruction word it was decoded from.
    Op(u32),
    /// The body grew past [`MAX_BODY_BYTES`].
    TooLong,
}

/// Set in `run`'s result when the block left at a taken branch; the guest `pc`
/// holds the target. A block never spans a page, so counts stay below 1,024.
pub const LEFT: u32 = 1 << 31;

/// The parameter every block function takes.
const STATE: u32 = 0;

/// Scratch locals.
const L_A: u32 = 1;
const L_B: u32 = 2;
const L_T: u32 = 3;
const L_R: u32 = 4;
/// The duplicate operand of a 32-bit rotate.
const L_S: u32 = 5;
/// The base register's writeback value, computed before the access.
const L_W: u32 = 6;
const L_C: u32 = 7;
/// The guest address of an access.
const L_ADDR: u32 = 8;
/// The host address of the guest page that address is on.
const L_PAGE: u32 = 9;
/// The guest page index, for the page table and the watched-pages bitmap.
const L_IDX: u32 = 10;
/// The watched-pages bitmap.
const L_WP: u32 = 11;

/// A block bigger than this is not emitted, to bound module compile time.
const MAX_BODY_BYTES: usize = 64 * 1024;

/// The immediate of a fused `CMP` inverted at emit time, as
/// [`Emitter::invert_if`] does at run time.
fn invert_if_const(v: u64, carry: u8) -> u64 {
    v ^ 0u64.wrapping_sub(u64::from(carry))
}

/// The mask an operation of this width applies to its operands and result.
fn width_mask(sf: bool) -> i64 {
    if sf {
        -1
    } else {
        0xFFFF_FFFF
    }
}

/// The bit an operation of this width calls the sign.
fn sign_bit(sf: bool) -> i64 {
    if sf {
        i64::MIN
    } else {
        0x8000_0000u32 as i64
    }
}

/// How wide an operation of this width is.
fn size_of(sf: bool) -> u32 {
    if sf {
        64
    } else {
        32
    }
}

struct Emitter<'a> {
    f: &'a mut Func,
    layout: Layout,
}

impl Emitter<'_> {
    /// Push `regs[slot]` unnarrowed, for ops that shift its high bits out.
    fn read_reg_raw(&mut self, slot: u8) {
        self.f.local_get(STATE);
        self.f
            .i64_load(ALIGN_8, self.layout.regs + 8 * u32::from(slot));
    }

    /// Narrow the value on the stack to the operation's width.
    fn mask_to(&mut self, sf: bool) {
        if !sf {
            self.f.i64_const(width_mask(false));
            self.f.i64_and();
        }
    }

    /// Sign-extend the value on the stack from the operation's width.
    fn sext_to(&mut self, sf: bool) {
        if !sf {
            self.f.i64_extend32_s();
        }
    }

    /// Push `regs[slot]`, already narrowed to the operation's width.
    fn read_reg(&mut self, slot: u8, sf: bool) {
        self.read_reg_raw(slot);
        self.mask_to(sf);
    }

    /// Push the address a register write needs; it goes under the value.
    fn addr_regs(&mut self) {
        self.f.local_get(STATE);
    }

    fn store_reg(&mut self, slot: u8) {
        self.f
            .i64_store(ALIGN_8, self.layout.regs + 8 * u32::from(slot));
    }

    /// Write the local `L_R` into `regs[slot]`.
    fn write_reg_from_r(&mut self, slot: u8) {
        self.addr_regs();
        self.f.local_get(L_R);
        self.store_reg(slot);
    }

    /// Invert the value on the stack when the operation subtracts, as
    /// [`super::exec`]'s `invert_if` does, and narrow it again.
    fn invert_if(&mut self, carry: u8, sf: bool) {
        if carry != 0 {
            self.f.i64_const(-1);
            self.f.i64_xor();
            self.mask_to(sf);
        }
    }

    /// Push 1 if `cond` holds under the current NZCV and 0 if it does not:
    /// its [`CONDITION_MASKS`] row shifted by the flags.
    fn cond_holds(&mut self, cond: u8) {
        self.f
            .i32_const(i32::from(CONDITION_MASKS[(cond & 0xF) as usize]));
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, self.layout.nzcv);
        self.f.i32_const(28);
        self.f.i32_shr_u();
        self.f.i32_shr_u();
        self.f.i32_const(1);
        self.f.i32_and();
    }

    /// [`crate::cpu::bits::shift_reg`] applied to the value on the stack. An
    /// out-of-range distance answers as the interpreter does.
    fn shift_const(&mut self, st: u8, sa: u8, sf: bool) {
        let size = size_of(sf);
        let sa = u32::from(sa);
        match st {
            0 => {
                if sa >= size {
                    self.f.drop_value();
                    self.f.i64_const(0);
                } else if sa != 0 {
                    self.f.i64_const(i64::from(sa));
                    self.f.i64_shl();
                    self.mask_to(sf);
                }
            }
            1 => {
                if sa >= size {
                    self.f.drop_value();
                    self.f.i64_const(0);
                } else if sa != 0 {
                    self.f.i64_const(i64::from(sa));
                    self.f.i64_shr_u();
                }
            }
            2 => {
                if sa != 0 {
                    self.sext_to(sf);
                    self.f.i64_const(i64::from(sa.min(size - 1)));
                    self.f.i64_shr_s();
                    self.mask_to(sf);
                }
            }
            _ => {
                let sa = sa % size;
                if sa != 0 {
                    self.rotate_right_const(sa, sf);
                }
            }
        }
    }

    /// Rotate the value on the stack right by `sa`, nonzero and in range. The
    /// 32-bit form is written as two shifts, using `L_S`.
    fn rotate_right_const(&mut self, sa: u32, sf: bool) {
        if sf {
            self.f.i64_const(i64::from(sa));
            self.f.i64_rotr();
            return;
        }
        self.f.local_tee(L_S);
        self.f.i64_const(i64::from(sa));
        self.f.i64_shr_u();
        self.f.local_get(L_S);
        self.f.i64_const(i64::from(32 - sa));
        self.f.i64_shl();
        self.f.i64_or();
        self.mask_to(false);
    }

    /// [`crate::cpu::bits::extend_reg`] applied to the value on the stack.
    fn extend_const(&mut self, option: u8, sf: bool) {
        match option & 0b111 {
            0b000 => {
                self.f.i64_const(0xFF);
                self.f.i64_and();
            }
            0b001 => {
                self.f.i64_const(0xFFFF);
                self.f.i64_and();
            }
            0b010 => {
                self.f.i64_const(width_mask(false));
                self.f.i64_and();
            }
            0b100 => self.f.i64_extend8_s(),
            0b101 => self.f.i64_extend16_s(),
            0b110 => self.f.i64_extend32_s(),
            // UXTX and SXTX are the whole register.
            _ => {}
        }
        self.mask_to(sf);
    }

    /// `a + b + carry` from `L_A` and `L_B` into `L_R`, narrowed, with the
    /// carry out in `L_C` when flags are wanted. Only 64-bit operations need
    /// the explicit carry chain.
    fn add_carry(&mut self, carry: u8, set_flags: bool, sf: bool) {
        let chain = set_flags && sf;

        self.f.local_get(L_A);
        self.f.local_get(L_B);
        self.f.i64_add();
        self.f.local_set(L_T);

        if chain {
            // c1 = t <u a, the carry out of the first add
            self.f.local_get(L_T);
            self.f.local_get(L_A);
            self.f.i64_lt_u();
            self.f.local_set(L_C);
        }

        if carry != 0 {
            if chain {
                // c = c1 | (t + 1 <u t), and the two can never both be set
                self.f.local_get(L_T);
                self.f.i64_const(1);
                self.f.i64_add();
                self.f.local_get(L_T);
                self.f.i64_lt_u();
                self.f.local_get(L_C);
                self.f.i32_or();
                self.f.local_set(L_C);
            }
            self.f.local_get(L_T);
            self.f.i64_const(1);
            self.f.i64_add();
            self.f.local_set(L_T);
        }

        self.f.local_get(L_T);
        self.mask_to(sf);
        self.f.local_set(L_R);

        if set_flags && !sf {
            self.f.local_get(L_T);
            self.f.i64_const(32);
            self.f.i64_shr_u();
            self.f.i32_wrap_i64();
            self.f.i32_const(1);
            self.f.i32_and();
            self.f.local_set(L_C);
        }
    }

    /// `ADD`/`SUB`/`ADDS`/`SUBS` with operands in `L_A` and `L_B`, as
    /// [`crate::cpu::Cpu::add_sub_pre`] receives them.
    fn add_sub(&mut self, rd: u8, carry: u8, set_flags: bool, sf: bool) {
        self.add_carry(carry, set_flags, sf);
        if set_flags {
            self.pack_nzcv(sf, true);
            self.store_nzcv();
        }
        self.write_reg_from_r(rd);
    }

    /// Push the packed NZCV word from `L_R` and, when `arithmetic`, C and V
    /// from `L_A`, `L_B` and `L_C`; otherwise C and V are kept (`ANDS`).
    fn pack_nzcv(&mut self, sf: bool, arithmetic: bool) {
        let shift = if sf { 63 } else { 31 };

        // N
        self.f.local_get(L_R);
        self.f.i64_const(shift);
        self.f.i64_shr_u();
        self.f.i32_wrap_i64();
        self.f.i32_const(1);
        self.f.i32_and();
        self.f.i32_const(31);
        self.f.i32_shl();

        // Z
        self.f.local_get(L_R);
        self.f.i64_eqz();
        self.f.i32_const(30);
        self.f.i32_shl();
        self.f.i32_or();

        if arithmetic {
            // C
            self.f.local_get(L_C);
            self.f.i32_const(29);
            self.f.i32_shl();
            self.f.i32_or();

            // V: both operands the same sign and the result a different one.
            self.f.local_get(L_A);
            self.f.local_get(L_B);
            self.f.i64_xor();
            self.f.i64_const(-1);
            self.f.i64_xor();
            self.f.local_get(L_A);
            self.f.local_get(L_R);
            self.f.i64_xor();
            self.f.i64_and();
            self.f.i64_const(sign_bit(sf));
            self.f.i64_and();
            self.f.i64_const(0);
            self.f.i64_ne();
            self.f.i32_const(28);
            self.f.i32_shl();
            self.f.i32_or();
        } else {
            // `ANDS` leaves C and V exactly as they were.
            self.f.local_get(STATE);
            self.f.i32_load(ALIGN_4, self.layout.nzcv);
            self.f.i32_const(0x3000_0000);
            self.f.i32_and();
            self.f.i32_or();
        }
    }

    /// Store the packed word on the stack into NZCV.
    fn store_nzcv(&mut self) {
        self.f.local_set(L_C);
        self.f.local_get(STATE);
        self.f.local_get(L_C);
        self.f.i32_store(ALIGN_4, self.layout.nzcv);
    }

    /// `AND`/`ORR`/`EOR`/`ANDS` with the second operand already in `L_B`.
    fn logical(&mut self, rd: u8, rn: u8, opc: u8, sf: bool) {
        self.read_reg(rn, sf);
        self.f.local_get(L_B);
        match opc {
            0b00 | 0b11 => self.f.i64_and(),
            0b01 => self.f.i64_or(),
            _ => self.f.i64_xor(),
        }
        self.f.local_set(L_R);
        if opc == 0b11 {
            self.pack_nzcv(sf, false);
            self.store_nzcv();
        }
        self.write_reg_from_r(rd);
    }

    /// `MADD`/`MSUB` and the widening `SMADDL`/`UMADDL` family. A 32x32
    /// product fits in 64 bits.
    fn multiply_accumulate(
        &mut self,
        rd: u8,
        ra: u8,
        sub: bool,
        sf: bool,
        operands: impl Fn(&mut Self),
    ) {
        self.addr_regs();
        self.read_reg(ra, sf);
        operands(self);
        self.f.i64_mul();
        if sub {
            self.f.i64_sub();
        } else {
            self.f.i64_add();
        }
        self.mask_to(sf);
        self.store_reg(rd);
    }

    /// `UDIV`/`SDIV`, `L_A` by `L_B`. Division by zero answers zero and
    /// `i64::MIN / -1` wraps, guarded with real branches since wasm traps.
    fn divide(&mut self, signed: bool, sf: bool) {
        self.f.local_get(L_B);
        self.f.i64_eqz();
        self.f.if_result(I64);
        self.f.i64_const(0);
        self.f.else_();
        if signed && sf {
            self.f.local_get(L_B);
            self.f.i64_const(-1);
            self.f.i64_eq();
            self.f.if_result(I64);
            // `x.wrapping_div(-1)` is `x.wrapping_neg()`.
            self.f.i64_const(0);
            self.f.local_get(L_A);
            self.f.i64_sub();
            self.f.else_();
            self.f.local_get(L_A);
            self.f.local_get(L_B);
            self.f.i64_div_s();
            self.f.end();
        } else {
            self.f.local_get(L_A);
            self.f.local_get(L_B);
            if signed {
                self.f.i64_div_s();
            } else {
                self.f.i64_div_u();
            }
        }
        self.f.end();
    }

    /// Leave the block if the value on the stack is nonzero, reporting
    /// `retired` instructions done; the interpreter resumes there.
    fn deopt_if(&mut self, retired: usize) {
        self.f.if_void();
        self.f.i32_const(retired as i32);
        self.f.return_();
        self.f.end();
    }

    /// Leave the block at a taken branch, if the value on the stack is
    /// nonzero, reporting `retired` (the branch included) and `target`.
    fn leave_if(&mut self, retired: usize, target: u32) {
        self.f.if_void();
        self.f.local_get(STATE);
        self.f.i32_const(target as i32);
        self.f.i32_store(ALIGN_4, self.layout.pc);
        self.f.i32_const((retired as u32 | LEFT) as i32);
        self.f.return_();
        self.f.end();
    }

    /// The flag-setting half of a fused compare-and-branch, operands in `L_A`
    /// and `L_B`.
    fn compare_flags(&mut self, carry: u8, sf: bool) {
        self.add_carry(carry, true, sf);
        self.pack_nzcv(sf, true);
        self.store_nzcv();
    }

    /// Emit the conditional branch `retired` instructions into the block, or
    /// answer `false`. The not-taken path is whatever is emitted next.
    fn exit(&mut self, exit: &Exit, retired: usize) -> bool {
        let target = match *exit {
            Exit::Cond { cond, target } => {
                self.cond_holds(cond);
                target
            }
            // The compare runs whether or not the branch is taken.
            Exit::CmpImm {
                rn,
                imm,
                carry,
                sf,
                cond,
                target,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                let rhs = invert_if_const(u64::from(imm), carry) & width_mask(sf) as u64;
                self.f.i64_const(rhs as i64);
                self.f.local_set(L_B);
                self.compare_flags(carry, sf);
                self.cond_holds(cond);
                target
            }
            // So does the update.
            Exit::UpdateCmpImm {
                rd,
                source,
                step,
                rn,
                imm,
                cond,
                target,
            } => {
                let update = Op::AddSubImm {
                    rd,
                    rn: source,
                    rhs: invert_if_const(step.imm(), step.carry()),
                    carry: step.carry(),
                    set_flags: false,
                    sf: step.sf(),
                };
                let compare = Exit::CmpImm {
                    rn,
                    imm: imm.imm() as u32,
                    carry: imm.carry(),
                    sf: imm.sf(),
                    cond,
                    target,
                };
                return self.op(&update, retired - 3) && self.exit(&compare, retired);
            }
            Exit::CmpReg {
                rn,
                rm,
                carry,
                sf,
                cond,
                target,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.read_reg(rm, sf);
                self.invert_if(carry, sf);
                self.f.local_set(L_B);
                self.compare_flags(carry, sf);
                self.cond_holds(cond);
                target
            }
            // Register 31 reads as the zero register, so the field is the slot.
            Exit::Cbz { rt, sf, nz, target } => {
                self.read_reg(rt & 0x1F, sf);
                self.is_zero(!nz);
                target
            }
            Exit::Tbz {
                rt,
                bit,
                nz,
                target,
            } => {
                self.read_reg_raw(rt & 0x1F);
                self.f.i64_const(i64::from(bit));
                self.f.i64_shr_u();
                self.f.i64_const(1);
                self.f.i64_and();
                self.is_zero(!nz);
                target
            }
            // A followed `B`: the following ops are already its target's.
            Exit::Jump { .. } => return true,
        };
        self.leave_if(retired, target);
        true
    }

    /// Replace the value on the stack with whether it is zero (or nonzero
    /// when `want_zero` is false).
    fn is_zero(&mut self, want_zero: bool) {
        if want_zero {
            self.f.i64_eqz();
        } else {
            self.f.i64_const(0);
            self.f.i64_ne();
        }
    }

    /// Leave an access's guest address in `L_ADDR` and any writeback value in
    /// `L_W`, per [`crate::cpu::Cpu::indexed`].
    fn address(&mut self, rn: u8, offset: i64, wb: Wb) {
        if matches!(wb, Wb::None) {
            self.read_reg_raw(rn);
            if offset != 0 {
                self.f.i64_const(offset);
                self.f.i64_add();
            }
        } else {
            self.read_reg_raw(rn);
            self.f.local_tee(L_A);
            self.f.i64_const(offset);
            self.f.i64_add();
            self.f.local_set(L_W);
            match wb {
                Wb::Post => self.f.local_get(L_A),
                _ => self.f.local_get(L_W),
            }
        }
        self.f.i32_wrap_i64();
        self.f.local_set(L_ADDR);
    }

    /// The same for the register-offset form, per
    /// [`crate::cpu::Cpu::reg_offset`].
    fn address_reg(&mut self, rn: u8, rm: u8, ext: Ext, shift: u8) {
        self.read_reg_raw(rn);
        self.read_reg_raw(rm);
        match ext {
            Ext::Uxtw => self.mask_to(false),
            Ext::Sxtw => self.f.i64_extend32_s(),
            Ext::None => {}
        }
        if shift != 0 {
            self.f.i64_const(i64::from(shift));
            self.f.i64_shl();
        }
        self.f.i64_add();
        self.f.i32_wrap_i64();
        self.f.local_set(L_ADDR);
    }

    /// Write `L_W` back to the base register, for the modes that have one.
    fn write_back(&mut self, rn: u8, wb: Wb) {
        if !matches!(wb, Wb::None) {
            self.addr_regs();
            self.f.local_get(L_W);
            self.store_reg(rn);
        }
    }

    /// Leave the page `L_ADDR` is on in `L_PAGE`, or leave the block if
    /// reading `n` bytes there needs more than the page table
    /// ([`crate::mem::Memory::peek`]'s checks, in its order).
    fn page_for_read(&mut self, n: u32, retired: usize) {
        self.page_number();
        let crosses = n > 1;
        if crosses {
            self.crosses_page(n);
        }
        self.covers(n, self.layout.read_watch_lo, self.layout.read_watch_hi);
        if crosses {
            self.f.i32_or();
        }
        self.deopt_if(retired);
        self.page_or_defer(retired);
    }

    /// The same for a store, with [`crate::mem::Memory::poke`]'s checks.
    fn page_for_write(&mut self, n: u32, retired: usize) {
        self.page_for_write_halves(n, None, retired);
    }

    /// [`Emitter::page_for_write`] for a pair, also testing write protection
    /// `second` bytes in, as [`crate::mem::Memory::poke_pair`] does.
    fn page_for_write_halves(&mut self, n: u32, second: Option<u32>, retired: usize) {
        self.page_number();
        let crosses = n > 1;
        if crosses {
            self.crosses_page(n);
        }
        self.within(self.layout.readonly_lo, self.layout.readonly_hi, 0);
        if crosses {
            self.f.i32_or();
        }
        if let Some(at) = second {
            self.within(self.layout.readonly_lo, self.layout.readonly_hi, at);
            self.f.i32_or();
        }
        self.covers(n, self.layout.watch_lo, self.layout.watch_hi);
        self.f.i32_or();
        self.deopt_if(retired);
        self.owes_a_report(retired);
        self.page_or_defer(retired);
    }

    /// Leave the guest page `L_ADDR` is on in `L_IDX`.
    fn page_number(&mut self) {
        self.f.local_get(L_ADDR);
        self.f.i32_const(PAGE_BITS as i32);
        self.f.i32_shr_u();
        self.f.local_set(L_IDX);
    }

    /// Leave that page's storage in `L_PAGE`, or leave the block if it is null.
    fn page_or_defer(&mut self, retired: usize) {
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, self.layout.pages);
        self.f.local_get(L_IDX);
        self.f.i32_const(PAGE_ENTRY_SHIFT);
        self.f.i32_shl();
        self.f.i32_add();
        self.f.i32_load(ALIGN_4, 0);
        self.f.local_tee(L_PAGE);
        self.f.i32_eqz();
        self.deopt_if(retired);
    }

    /// Push whether `L_ADDR` is inside the write-protected envelope `[lo, hi)`;
    /// the full path decides the exact ranges.
    fn within(&mut self, lo: u32, hi: u32, at: u32) {
        let addr = |e: &mut Self| {
            e.f.local_get(L_ADDR);
            if at != 0 {
                e.f.i32_const(at as i32);
                e.f.i32_add();
            }
        };
        addr(self);
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, lo);
        self.f.i32_ge_u();
        addr(self);
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, hi);
        self.f.i32_lt_u();
        self.f.i32_and();
    }

    /// Leave the block if anything has cached page `L_IDX`'s contents, when
    /// the bitmap exists.
    fn owes_a_report(&mut self, retired: usize) {
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, self.layout.watched);
        self.f.local_tee(L_WP);
        self.f.if_void();
        self.f.local_get(L_WP);
        self.f.local_get(L_IDX);
        self.f.i32_const(6);
        self.f.i32_shr_u();
        self.f.i32_const(3);
        self.f.i32_shl();
        self.f.i32_add();
        self.f.i64_load(ALIGN_8, 0);
        // The shift is taken modulo 64, masking the bit index.
        self.f.i64_const(1);
        self.f.local_get(L_IDX);
        self.f.i64_extend_i32_u();
        self.f.i64_shl();
        self.f.i64_and();
        self.f.i64_const(0);
        self.f.i64_ne();
        self.deopt_if(retired);
        self.f.end();
    }

    /// Push whether the `n` bytes at `L_ADDR` run past the end of their page.
    fn crosses_page(&mut self, n: u32) {
        self.f.local_get(L_ADDR);
        self.f.i32_const((PAGE_SIZE - 1) as i32);
        self.f.i32_and();
        self.f.i32_const((PAGE_SIZE as u32 - n) as i32);
        self.f.i32_gt_u();
    }

    /// Push whether the watchpoint at `lo`/`hi` overlaps the `n` bytes at
    /// `L_ADDR`, wrapping included.
    fn covers(&mut self, n: u32, lo: u32, hi: u32) {
        self.f.local_get(L_ADDR);
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, hi);
        self.f.i32_lt_u();
        self.f.local_get(L_ADDR);
        self.f.i32_const(n as i32);
        self.f.i32_add();
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, lo);
        self.f.i32_gt_u();
        self.f.i32_and();
    }

    /// Push the host address of the access: page plus offset.
    fn in_page(&mut self) {
        self.f.local_get(L_PAGE);
        self.f.local_get(L_ADDR);
        self.f.i32_const((PAGE_SIZE - 1) as i32);
        self.f.i32_and();
        self.f.i32_add();
    }

    /// A load into `rt` from `L_ADDR`. Only the sign-extend-to-32 forms need
    /// anything after the wasm load.
    fn load(&mut self, rt: u8, acc: Acc, retired: usize) {
        self.page_for_read(access_bytes(acc), retired);
        self.addr_regs();
        self.in_page();
        match acc {
            Acc::Load8 => self.f.i64_load8_u(0),
            Acc::Load16 => self.f.i64_load16_u(UNALIGNED, 0),
            Acc::Load32 => self.f.i64_load32_u(UNALIGNED, 0),
            Acc::Load64 => self.f.i64_load(UNALIGNED, 0),
            Acc::LoadS8 => self.f.i64_load8_s(0),
            Acc::LoadS16 => self.f.i64_load16_s(UNALIGNED, 0),
            Acc::LoadS32 => self.f.i64_load32_s(UNALIGNED, 0),
            Acc::LoadS8To32 => {
                self.f.i64_load8_s(0);
                self.mask_to(false);
            }
            Acc::LoadS16To32 => {
                self.f.i64_load16_s(UNALIGNED, 0);
                self.mask_to(false);
            }
            _ => unreachable!("only the loads reach here; `writes_rt` is what sorts them"),
        }
        self.store_reg(rt);
    }

    /// A store of `rt` to `L_ADDR`.
    fn store(&mut self, rt: u8, acc: Acc, retired: usize) {
        self.page_for_write(access_bytes(acc), retired);
        self.in_page();
        self.read_reg_raw(rt);
        match acc {
            Acc::Store8 => self.f.i64_store8(0),
            Acc::Store16 => self.f.i64_store16(UNALIGNED, 0),
            Acc::Store32 => self.f.i64_store32(UNALIGNED, 0),
            Acc::Store64 => self.f.i64_store(UNALIGNED, 0),
            _ => unreachable!("only the stores reach here; `writes_rt` is what sorts them"),
        }
    }

    /// A load of two `n`-byte registers from `L_ADDR`, checked as one access.
    fn load_pair(&mut self, rt: u8, rt2: u8, n: u32, signed: bool, retired: usize) {
        self.page_for_read(2 * n, retired);
        for (slot, at) in [(rt, 0), (rt2, n)] {
            self.addr_regs();
            self.in_page();
            match (n, signed) {
                (8, _) => self.f.i64_load(UNALIGNED, at),
                (4, false) => self.f.i64_load32_u(UNALIGNED, at),
                _ => self.f.i64_load32_s(UNALIGNED, at),
            }
            self.store_reg(slot);
        }
    }

    /// A store of two `n`-byte registers to `L_ADDR`, with
    /// [`crate::mem::Memory::poke_pair`]'s checks.
    fn store_pair(&mut self, rt: u8, rt2: u8, n: u32, retired: usize) {
        self.page_for_write_halves(2 * n, Some(n), retired);
        for (slot, at) in [(rt, 0), (rt2, n)] {
            self.in_page();
            self.read_reg_raw(slot);
            match n {
                8 => self.f.i64_store(UNALIGNED, at),
                _ => self.f.i64_store32(UNALIGNED, at),
            }
        }
    }

    /// A single-register access. `PRFM` has no guard: it touches no memory.
    fn access(&mut self, rt: u8, acc: Acc, retired: usize) {
        match acc {
            Acc::Prefetch => {}
            _ if acc.writes_rt() => self.load(rt, acc, retired),
            _ => self.store(rt, acc, retired),
        }
    }

    /// One op, or `false` if there is no way to write it yet.
    fn op(&mut self, op: &Op, retired: usize) -> bool {
        match *op {
            Op::Nop => true,

            Op::MovConst { rd, val } => {
                self.addr_regs();
                self.f.i64_const(val as i64);
                self.store_reg(rd);
                true
            }

            Op::Mov32 { rd, rn } => {
                self.addr_regs();
                self.read_reg(rn, false);
                self.store_reg(rd);
                true
            }

            Op::Mov64 { rd, rn } => {
                self.addr_regs();
                self.read_reg(rn, true);
                self.store_reg(rd);
                true
            }

            Op::MovK { rd, shift, val, sf } => {
                // Replace the field, then narrow: a 32-bit MOVK zeroes the top half.
                let keep = !(0xFFFFu64 << shift) & (width_mask(sf) as u64);
                self.addr_regs();
                self.read_reg_raw(rd);
                self.f.i64_const(keep as i64);
                self.f.i64_and();
                self.f.i64_const((u64::from(val) << shift) as i64);
                self.f.i64_or();
                self.store_reg(rd);
                true
            }

            Op::AddSubImm {
                rd,
                rn,
                rhs,
                carry,
                set_flags,
                sf,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.f.i64_const((rhs & width_mask(sf) as u64) as i64);
                self.f.local_set(L_B);
                self.add_sub(rd, carry, set_flags, sf);
                true
            }

            Op::AddSubReg {
                rd,
                rn,
                rm,
                carry,
                set_flags,
                sf,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.read_reg(rm, sf);
                self.invert_if(carry, sf);
                self.f.local_set(L_B);
                self.add_sub(rd, carry, set_flags, sf);
                true
            }

            Op::AddSubShifted {
                rd,
                rn,
                rm,
                st,
                sa,
                carry,
                set_flags,
                sf,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.read_reg(rm, sf);
                self.shift_const(st, sa, sf);
                self.invert_if(carry, sf);
                self.f.local_set(L_B);
                self.add_sub(rd, carry, set_flags, sf);
                true
            }

            Op::AddSubExtended {
                rd,
                rn,
                rm,
                option,
                shift,
                carry,
                set_flags,
                sf,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.read_reg_raw(rm);
                self.extend_const(option, sf);
                if shift != 0 {
                    self.f.i64_const(i64::from(shift));
                    self.f.i64_shl();
                    self.mask_to(sf);
                }
                self.invert_if(carry, sf);
                self.f.local_set(L_B);
                self.add_sub(rd, carry, set_flags, sf);
                true
            }

            Op::LogicalImm {
                rd,
                rn,
                imm,
                opc,
                sf,
            } => {
                self.f.i64_const((imm & width_mask(sf) as u64) as i64);
                self.f.local_set(L_B);
                self.logical(rd, rn, opc, sf);
                true
            }

            Op::LogicalReg {
                rd,
                rn,
                rm,
                opc,
                invert,
                sf,
            } => {
                self.read_reg(rm, sf);
                self.invert_if(u8::from(invert), sf);
                self.f.local_set(L_B);
                self.logical(rd, rn, opc, sf);
                true
            }

            Op::LogicalShifted {
                rd,
                rn,
                rm,
                st,
                sa,
                opc,
                invert,
                sf,
            } => {
                self.read_reg(rm, sf);
                self.shift_const(st, sa, sf);
                // `BIC`/`ORN`/`EON` invert the shifted operand.
                self.invert_if(u8::from(invert), sf);
                self.f.local_set(L_B);
                self.logical(rd, rn, opc, sf);
                true
            }

            Op::Extract {
                rd,
                rn,
                extract,
                sf,
            } => {
                let (left, right, up, signed) = extract.parts();
                self.addr_regs();
                // `left` shifts the high bits out, so read the register whole.
                self.read_reg_raw(rn);
                if left != 0 {
                    self.f.i64_const(i64::from(left));
                    self.f.i64_shl();
                }
                if right != 0 {
                    self.f.i64_const(i64::from(right));
                    if signed {
                        self.f.i64_shr_s();
                    } else {
                        self.f.i64_shr_u();
                    }
                }
                if up != 0 {
                    self.f.i64_const(i64::from(up));
                    self.f.i64_shl();
                }
                self.mask_to(sf);
                self.store_reg(rd);
                true
            }

            Op::Bitfield {
                rd,
                rn,
                opc,
                immr,
                imms,
                sf,
            } => {
                // Only `BFM` reaches here with anything to do.
                if opc != 0b01 {
                    self.addr_regs();
                    self.f.i64_const(0);
                    self.store_reg(rd);
                    return true;
                }
                let (lsb, msb) = (u32::from(immr), u32::from(imms));
                let (field, right, left) = if msb >= lsb {
                    (mask_of_width(msb - lsb + 1, sf), lsb, 0)
                } else {
                    let up = size_of(sf) - lsb;
                    (mask_of_width(msb + 1, sf).wrapping_shl(up), 0, up)
                };
                self.addr_regs();
                self.read_reg(rd, sf);
                self.f.i64_const(!field as i64);
                self.f.i64_and();
                self.read_reg(rn, sf);
                if right != 0 {
                    self.f.i64_const(i64::from(right));
                    self.f.i64_shr_u();
                }
                if left != 0 {
                    self.f.i64_const(i64::from(left));
                    self.f.i64_shl();
                }
                self.f.i64_const(field as i64);
                self.f.i64_and();
                self.f.i64_or();
                self.mask_to(sf);
                self.store_reg(rd);
                true
            }

            Op::Extr {
                rd,
                rn,
                rm,
                imm,
                sf,
            } => {
                let size = size_of(sf);
                let imm = u32::from(imm);
                if imm >= size {
                    return false;
                }
                self.addr_regs();
                self.read_reg(rm, sf);
                if imm != 0 {
                    // Rn is the *high* half of the pair being shifted down.
                    self.f.i64_const(i64::from(imm));
                    self.f.i64_shr_u();
                    self.read_reg(rn, sf);
                    self.f.i64_const(i64::from(size - imm));
                    self.f.i64_shl();
                    self.f.i64_or();
                    self.mask_to(sf);
                }
                self.store_reg(rd);
                true
            }

            Op::ShiftVar {
                rd,
                rn,
                rm,
                kind,
                sf,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                self.read_reg(rm, sf);
                if !sf {
                    // Modulo 32 for a 32-bit shift, not wasm's 64.
                    self.f.i64_const(31);
                    self.f.i64_and();
                }
                self.f.local_set(L_B);
                self.addr_regs();
                match kind {
                    0 => {
                        self.f.local_get(L_A);
                        self.f.local_get(L_B);
                        self.f.i64_shl();
                        self.mask_to(sf);
                    }
                    1 => {
                        self.f.local_get(L_A);
                        self.f.local_get(L_B);
                        self.f.i64_shr_u();
                    }
                    2 => {
                        self.f.local_get(L_A);
                        self.sext_to(sf);
                        self.f.local_get(L_B);
                        self.f.i64_shr_s();
                        self.mask_to(sf);
                    }
                    _ if sf => {
                        self.f.local_get(L_A);
                        self.f.local_get(L_B);
                        self.f.i64_rotr();
                    }
                    // A 32-bit variable rotate as two shifts; at zero the left
                    // shift by 32 is discarded by the final narrowing.
                    _ => {
                        self.f.local_get(L_A);
                        self.f.local_get(L_B);
                        self.f.i64_shr_u();
                        self.f.local_get(L_A);
                        self.f.i64_const(32);
                        self.f.local_get(L_B);
                        self.f.i64_sub();
                        self.f.i64_shl();
                        self.f.i64_or();
                        self.mask_to(false);
                    }
                }
                self.store_reg(rd);
                true
            }

            Op::Divide {
                rd,
                rn,
                rm,
                signed,
                sf,
            } => {
                self.read_reg(rn, sf);
                if signed {
                    self.sext_to(sf);
                }
                self.f.local_set(L_A);
                self.read_reg(rm, sf);
                if signed {
                    self.sext_to(sf);
                }
                self.f.local_set(L_B);
                self.addr_regs();
                self.divide(signed, sf);
                self.mask_to(sf);
                self.store_reg(rd);
                true
            }

            Op::Madd {
                rd,
                rn,
                rm,
                ra,
                sub,
                sf,
            } => {
                self.multiply_accumulate(rd, ra, sub, sf, |e| {
                    e.read_reg(rn, sf);
                    e.read_reg(rm, sf);
                });
                true
            }

            Op::MaddLong {
                rd,
                rn,
                rm,
                ra,
                sub,
                signed,
            } => {
                self.multiply_accumulate(rd, ra, sub, true, |e| {
                    for r in [rn, rm] {
                        e.read_reg_raw(r);
                        if signed {
                            // Fills the top with the sign, so it narrows too.
                            e.f.i64_extend32_s();
                        } else {
                            e.mask_to(false);
                        }
                    }
                });
                true
            }

            Op::CondSel {
                rd,
                rn,
                rm,
                cond,
                else_inv,
                else_inc,
                sf,
            } => {
                self.addr_regs();
                self.read_reg(rn, sf);
                self.read_reg(rm, sf);
                // The invert and increment apply to the else value only.
                if else_inv {
                    self.f.i64_const(-1);
                    self.f.i64_xor();
                }
                if else_inc {
                    self.f.i64_const(1);
                    self.f.i64_add();
                }
                self.cond_holds(cond);
                self.f.select();
                self.mask_to(sf);
                self.store_reg(rd);
                true
            }

            Op::CondCmp {
                rn,
                rm,
                imm,
                cond,
                nzcv,
                sub,
                is_imm,
                sf,
            } => {
                self.read_reg(rn, sf);
                self.f.local_set(L_A);
                if is_imm {
                    self.f.i64_const(i64::from(imm));
                } else {
                    // Narrowed, as [`crate::cpu::Cpu::add_carry_overflow`] does.
                    self.read_reg(rm, sf);
                }
                self.invert_if(u8::from(sub), sf);
                self.f.local_set(L_B);
                self.add_carry(u8::from(sub), true, sf);
                // Select between the compare's flags and the immediate ones.
                self.f.local_get(STATE);
                self.pack_nzcv(sf, true);
                self.f.i32_const((u32::from(nzcv) << 28) as i32);
                self.cond_holds(cond);
                self.f.select();
                self.f.i32_store(ALIGN_4, self.layout.nzcv);
                true
            }

            Op::Load64 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.load(rt, Acc::Load64, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Load32 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.load(rt, Acc::Load32, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Load8 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.load(rt, Acc::Load8, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Store64 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.store(rt, Acc::Store64, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Store32 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.store(rt, Acc::Store32, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Store8 { rt, rn, wb, offset } => {
                self.address(rn, offset, wb);
                self.store(rt, Acc::Store8, retired);
                self.write_back(rn, wb);
                true
            }
            Op::LoadStoreImm {
                rt,
                rn,
                acc,
                wb,
                offset,
            } => {
                self.address(rn, offset, wb);
                self.access(rt, acc, retired);
                self.write_back(rn, wb);
                true
            }
            Op::LoadStoreReg {
                rt,
                rn,
                rm,
                ext,
                shift,
                acc,
            } => {
                self.address_reg(rn, rm, ext, shift);
                self.access(rt, acc, retired);
                true
            }
            Op::PairLoad64 {
                rt,
                rt2,
                rn,
                offset,
                wb,
            } => {
                self.address(rn, offset, wb);
                self.load_pair(rt, rt2, 8, false, retired);
                self.write_back(rn, wb);
                true
            }
            Op::PairStore64 {
                rt,
                rt2,
                rn,
                offset,
                wb,
            } => {
                self.address(rn, offset, wb);
                self.store_pair(rt, rt2, 8, retired);
                self.write_back(rn, wb);
                true
            }
            Op::Pair {
                rt,
                rt2,
                rn,
                offset,
                kind,
                wb,
            } => {
                self.address(rn, offset, wb);
                match kind {
                    PairKind::Load64 => self.load_pair(rt, rt2, 8, false, retired),
                    PairKind::Load32 => self.load_pair(rt, rt2, 4, false, retired),
                    PairKind::Load32Sext => self.load_pair(rt, rt2, 4, true, retired),
                    PairKind::Store64 => self.store_pair(rt, rt2, 8, retired),
                    PairKind::Store32 => self.store_pair(rt, rt2, 4, retired),
                }
                self.write_back(rn, wb);
                true
            }
            Op::LoadLiteral { rt, addr, acc } => {
                self.f.i32_const(addr as i32);
                self.f.local_set(L_ADDR);
                self.access(rt, acc, retired);
                true
            }

            _ => false,
        }
    }
}

/// How many bytes an access touches.
fn access_bytes(acc: Acc) -> u32 {
    match acc {
        Acc::Load8 | Acc::LoadS8 | Acc::LoadS8To32 | Acc::Store8 => 1,
        Acc::Load16 | Acc::LoadS16 | Acc::LoadS16To32 | Acc::Store16 => 2,
        Acc::Load32 | Acc::LoadS32 | Acc::Store32 => 4,
        Acc::Load64 | Acc::Store64 => 8,
        // `PRFM` reads nothing, so nothing is ever asked of its address.
        Acc::Prefetch => 0,
    }
}

/// Emit `block`'s body as a module exporting `run`.
pub(super) fn emit_block(block: &Block, layout: Layout) -> Result<Vec<u8>, Refused> {
    let mut f = scratch_func();
    let mut e = Emitter { f: &mut f, layout };
    let mut next_exit = 0usize;
    let mut i = 0usize;
    while i < block.ops.len() {
        // The same walk [`super::exec`] makes over ops and branch spans.
        match block.exits.get(next_exit) {
            Some(branch) if branch.at as usize == i => {
                let retired = i + branch.span as usize;
                if !e.exit(&branch.exit, retired) {
                    return Err(Refused::ControlFlow);
                }
                i = retired;
                next_exit += 1;
            }
            _ => {
                if !e.op(&block.ops[i], i) {
                    return Err(Refused::Op(block.words[i]));
                }
                i += 1;
            }
        }
        if e.f.len() > MAX_BODY_BYTES {
            return Err(Refused::TooLong);
        }
    }
    // An exit the walk did not reach would be silently dropped; refuse.
    if next_exit != block.exits.len() {
        return Err(Refused::ControlFlow);
    }
    f.i32_const(block.ops.len() as i32);
    f.end();

    let mut m = Module::new();
    let ty = m.add_type(vec![I32], vec![I32]);
    let idx = m.add_func(ty, f);
    m.export("run", idx);
    Ok(m.finish())
}

impl Cpu {
    /// Translate the block at `pc` and emit it against `layout`, for
    /// `examples/emit_difftest.rs`. Also reports the address of each covered
    /// instruction and the one after.
    pub fn emit_block_at(&self, pc: u32, layout: Layout) -> Result<(Vec<u8>, Vec<u32>), Refused> {
        let block = translate(&self.mem, pc);
        let bytes = emit_block(&block, layout)?;
        let mut jumps = block.exits.iter().filter_map(|branch| match branch.exit {
            Exit::Jump { target } => Some((branch.at as usize + branch.span as usize, target)),
            _ => None,
        });
        let mut next_jump = jumps.next();
        let mut path = Vec::with_capacity(block.ops.len() + 1);
        let mut at = pc;
        for i in 0..=block.ops.len() {
            if let Some((_, target)) = next_jump.filter(|&(after, _)| after == i) {
                at = target;
                next_jump = jumps.next();
            }
            path.push(at);
            at = at.wrapping_add(4);
        }
        Ok((bytes, path))
    }

    /// The register file by slot, including the three register 31 resolves to.
    pub fn reg_slots(&self) -> [u64; crate::cpu::REG_SLOTS] {
        let mut out = [0u64; crate::cpu::REG_SLOTS];
        out.copy_from_slice(&self.regs[..crate::cpu::REG_SLOTS]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::jit::ir::Block;

    fn block_of(ops: Vec<Op>) -> Block {
        let words = vec![0u32; ops.len()];
        Block::new(0x1000, ops, words, Vec::new(), None, vec![1])
    }

    const LAYOUT: Layout = Layout {
        regs: 0,
        nzcv: 2048,
        pc: 2084,
        pages: 2052,
        read_watch_lo: 2056,
        read_watch_hi: 2060,
        watch_lo: 2064,
        watch_hi: 2068,
        readonly_lo: 2072,
        readonly_hi: 2076,
        watched: 2080,
    };

    #[test]
    fn an_unwritable_op_refuses_the_whole_block() {
        let good = block_of(vec![Op::MovConst { rd: 0, val: 7 }]);
        assert!(emit_block(&good, LAYOUT).is_ok());

        let ops = vec![
            Op::MovConst { rd: 0, val: 7 },
            Op::Interpret { insn: 0xD503201F },
        ];
        let words = vec![0, 0xD503201F];
        let bad = Block::new(0x1000, ops, words, Vec::new(), None, vec![1]);
        assert_eq!(emit_block(&bad, LAYOUT), Err(Refused::Op(0xD503201F)));
    }

    /// Walks guest state from the `Cpu`'s address by [`Layout::of_cpu`]
    /// offsets alone, as emitted code does.
    #[test]
    fn emitted_code_finds_guest_state_through_the_layout() {
        /// One page-table entry.
        const ENTRY: usize = std::mem::size_of::<Option<Box<[u8; PAGE_SIZE]>>>();

        const SLOT: u8 = 5;
        const VALUE: u64 = 0x0123_4567_89AB_CDEF;
        const ADDR: u32 = 0x0800_1234;
        const BYTE: u8 = 0xA7;

        let mut cpu = Cpu::new();
        cpu.set_reg_at(SLOT, VALUE);
        cpu.nzcv = 0xC000_0000;
        cpu.mem.write_u8(ADDR, BYTE).expect("the page is writable");
        cpu.mem.watch_reads(0x1000, 0x40);
        cpu.mem.watch_writes(0x2000, 0x80);
        cpu.mem.mark_readonly(0x3000, 0x4000);
        cpu.mem.mark_code_page(ADDR);

        let layout = Layout::of_cpu();
        let base = &cpu as *const Cpu as usize;
        // SAFETY: every read is at an in-bounds offset from a live `Cpu`, of
        // the type the field there holds.
        unsafe {
            let u32_at = |off: u32| *((base + off as usize) as *const u32);

            assert_eq!(
                *((base + layout.regs as usize + 8 * SLOT as usize) as *const u64),
                VALUE,
                "the register file is not where an emitted block reads it"
            );
            assert_eq!(u32_at(layout.nzcv), 0xC000_0000, "NZCV moved");

            assert_eq!(
                (u32_at(layout.read_watch_lo), u32_at(layout.read_watch_hi)),
                (0x1000, 0x1040),
                "the read watchpoint moved"
            );
            assert_eq!(
                (u32_at(layout.watch_lo), u32_at(layout.watch_hi)),
                (0x2000, 0x2080),
                "the write watchpoint moved"
            );
            assert_eq!(
                (u32_at(layout.readonly_lo), u32_at(layout.readonly_hi)),
                (0x3000, 0x4000),
                "the write-protected envelope moved"
            );

            // Marking a page allocates the bitmap and sets the bit stores test.
            let watched = *((base + layout.watched as usize) as *const *const u64);
            assert!(!watched.is_null(), "the watched-page bitmap is not there");
            let page = (ADDR >> PAGE_BITS) as usize;
            assert_ne!(
                *watched.add(page >> 6) & (1u64 << (page & 63)),
                0,
                "the marked page's bit is not where an emitted store looks"
            );

            // The page-table walk: table pointer, entry, byte at the offset.
            let table = *((base + layout.pages as usize) as *const *const u8);
            let entry = *(table.add(page * ENTRY) as *const *const u8);
            assert!(!entry.is_null(), "a written page has no storage");
            assert_eq!(
                *entry.add((ADDR as usize) & (PAGE_SIZE - 1)),
                BYTE,
                "the page-table walk does not reach the byte that was written"
            );
            const NOWHERE: u32 = 0xF000_0000;
            let blank = *(table.add((NOWHERE >> PAGE_BITS) as usize * ENTRY) as *const *const u8);
            assert!(blank.is_null(), "an unmapped page is not a null entry");
        }
    }

    #[test]
    fn a_followed_branch_is_emitted() {
        use crate::cpu::jit::ir::{Branch, Exit};

        let ops = vec![Op::Nop, Op::MovConst { rd: 0, val: 7 }];
        let exits = vec![Branch::new(0, Exit::Jump { target: 0x2000 })];
        let b = Block::new(0x1000, ops, vec![0, 0], exits, None, vec![1]);
        assert!(emit_block(&b, LAYOUT).is_ok());
    }

    #[test]
    fn an_exit_past_the_body_refuses_rather_than_being_dropped() {
        use crate::cpu::jit::ir::{Branch, Exit};

        let ops = vec![Op::MovConst { rd: 0, val: 7 }];
        let exits = vec![Branch::new(4, Exit::Cond { cond: 0, target: 8 })];
        let b = Block::new(0x1000, ops, vec![0], exits, None, vec![1]);
        assert_eq!(emit_block(&b, LAYOUT), Err(Refused::ControlFlow));
    }

    #[test]
    fn every_conditional_branch_is_written() {
        use crate::cpu::jit::ir::{Branch, Exit};

        let branches = [
            Exit::Cond {
                cond: 0,
                target: 0x2000,
            },
            Exit::Cbz {
                rt: 3,
                sf: true,
                nz: false,
                target: 0x2000,
            },
            Exit::Tbz {
                rt: 3,
                bit: 40,
                nz: true,
                target: 0x2000,
            },
            Exit::CmpImm {
                rn: 3,
                imm: 0x20,
                carry: 1,
                sf: false,
                cond: 11,
                target: 0x2000,
            },
            Exit::CmpReg {
                rn: 3,
                rm: 4,
                carry: 1,
                sf: true,
                cond: 11,
                target: 0x2000,
            },
        ];
        for exit in branches {
            // Ops before and after the branch.
            let ops = vec![Op::MovConst { rd: 0, val: 7 }, Op::Nop, Op::Nop];
            let exits = vec![Branch::new(1, exit)];
            let b = Block::new(0x1000, ops, vec![0; 3], exits, None, vec![1]);
            assert!(
                emit_block(&b, LAYOUT).is_ok(),
                "{exit:?} was not written out"
            );
        }
    }

    /// A fused compare retires two instructions with the branch.
    #[test]
    fn a_fused_compare_retires_the_pair() {
        use crate::cpu::jit::ir::{Branch, Exit};

        let pair = Branch::new(
            0,
            Exit::CmpImm {
                rn: 3,
                imm: 1,
                carry: 1,
                sf: true,
                cond: 0,
                target: 0x2000,
            },
        );
        assert_eq!(pair.span, 2);
        let b = Block::new(
            0x1000,
            vec![Op::Nop, Op::Nop],
            vec![0; 2],
            vec![pair],
            None,
            vec![1],
        );
        let bytes = emit_block(&b, LAYOUT).expect("a fused compare is written");
        // `2 | LEFT` as the signed LEB128 `i32.const` operand.
        let mut want = Vec::new();
        super::super::wasm::sleb(&mut want, i64::from((2u32 | LEFT) as i32));
        assert!(
            bytes.windows(want.len()).any(|w| w == want),
            "the leave does not report the pair"
        );
    }
}
