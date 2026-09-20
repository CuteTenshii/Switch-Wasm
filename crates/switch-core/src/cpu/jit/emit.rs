//! Turning a translated block into wasm.
//!
//! [`super::decode`] already resolved what each instruction *does*; this
//! writes that out as wasm instead of interpreting it, so the per-instruction
//! dispatch that [`super::exec`] pays disappears into straight-line code the
//! browser compiles once.
//!
//! # What a block is handed
//!
//! One parameter: the address of the [`crate::cpu::Cpu`] in the emulator's own
//! linear memory. Guest state is reached from it by baked-in field offsets
//! ([`Layout`]), so an emitted `add x0, x1, x2` is two `i64.load`s, an add and
//! an `i64.store` against the register file where it already lives. Nothing is
//! copied in or out, and nothing has to move: the module imports the host's
//! memory rather than defining one of its own.
//!
//! Taking the address as a *parameter* rather than baking it is what makes an
//! emitted block independent of which `Cpu` runs it. A block belongs to a
//! guest address, and guest threads share one `Cpu`, but the test suite builds
//! many, and a baked pointer would silently address a freed one.
//!
//! # Everything the translator settled is a constant here
//!
//! The interpreter's helpers take the operand width, the shift type, the
//! shift distance, the extension and the condition as arguments and branch on
//! them; an op holds all five as fields the translator has already filled in.
//! So a shifted-register `ADD` emits the one shift it is, an `ASR` emits its
//! own distance, `extend_reg`'s eight-way match becomes the single mask or
//! sign-extension that option selects, and a condition becomes its row of
//! [`CONDITION_MASKS`] shifted by NZCV: two instructions where the interpreter
//! runs a table lookup and a branch. None of those matches survives into the
//! emitted code.
//!
//! # Coverage is a performance question, not a correctness one
//!
//! [`emit_block`] gives back a [`Refused`] for a block containing anything it
//! cannot write, and that block keeps running on the interpreter. This is the
//! same rule [`super::decode`]'s `Op::Interpret` follows: a form the emitter
//! does not know is slower, never wrong. So the supported set can grow one op
//! at a time, each addition backed by `emit_difftest`, rather than needing to
//! be complete before any of it can run. [`Refused::Op`] carries the
//! instruction word, which is what `examples/emit_difftest.rs` ranks to say
//! which op is worth writing next.
//!
//! # Guest memory
//!
//! An access is written out as the page-table walk it is.
//! [`crate::mem::Memory`] keeps one pointer per 4 KiB of the guest's address
//! space and nulls the ones with no storage behind them, so an emitted access
//! shifts the address down twelve, loads that pointer, and reads or writes at
//! the offset into it. Four instructions and no call.
//!
//! What made this look impossible was the rest of `Memory`: soft regions,
//! write-protected ranges, watchpoints, the reports a store owes anything
//! caching guest memory. None of it is on this path, because the interpreter
//! already splits the same way. [`crate::mem::Memory::peek`] and
//! [`crate::mem::Memory::poke`] are the parts of an access the page table can
//! answer on its own; they decline the rest, whereupon [`super::exec`] runs the
//! whole instruction again out of line. An emitted access makes exactly their
//! checks, in their order, and declines what they decline.
//!
//! A store has three more of those checks than a load, and one is a bitmap of
//! the pages something has cached the contents of. That is the one thing here
//! that guest memory had to be changed for: the bitmap was a `Vec`, and nothing
//! promises where in one the pointer and the length sit, so there was nowhere
//! for an emitted store to look. It is a fixed-size bitmap behind a single
//! pointer now, which is what it always was.
//!
//! What is not written is the pair forms and the exclusives. A pair is two
//! accesses in one page with its own boundary test, and the exclusives carry a
//! reservation that is not in this model at all.
//!
//! # What a block reports
//!
//! `run` gives back the number of leading instructions it retired, and guest
//! state is exactly what those instructions left: an access that declines does
//! so before it has written a register, so the instruction it stopped at has
//! not half happened. The interpreter picks up from there, which is the
//! handover [`super::exec`] already makes when a step budget runs out inside a
//! block, and it resumes by translating the block at the address it stopped on
//! rather than by re-entering this one part-way.
//!
//! [`defers`] is which instructions can do that, and it matters outside this
//! module because it is what the difftests need in order to know which answer
//! from `run` is the right one.
//!
//! # Where wasm and A64 disagree
//!
//! Three places, each of which costs emitted instructions that the operation
//! itself does not suggest.
//!
//! A shift takes its distance modulo the operand width in both, but wasm's
//! width is the one it is operating on. A 32-bit A64 shift held in an `i64`
//! wants its distance modulo 32 and would get it modulo 64, so the variable
//! shifts mask the amount themselves.
//!
//! A division by zero traps in wasm and answers zero in A64, and `i64.div_s`
//! traps again on `i64::MIN / -1` where A64 wraps. Both are guarded with real
//! branches rather than [`Func::select`], which evaluates the arm it does not
//! pick.
//!
//! There is no 128-bit integer, so `SMULH`/`UMULH` are not written at all and
//! stay with the interpreter.

use super::decode::{decode, translate, Decoded};
use super::ir::{Block, Op};
use super::wasm::{Func, Module, I32, I64};
use crate::cpu::bits::mask_of_width;
use crate::cpu::loadstore::{Acc, Ext, Wb};
use crate::cpu::{Cpu, CONDITION_MASKS};
use crate::mem::{PAGE_BITS, PAGE_SIZE};

/// Whether the emitter has a way to write `insn` out as wasm, which is what
/// [`super::translates`] asks of the translator.
///
/// Worth asking from outside because an emitted block is all or nothing: one
/// instruction with no arm here takes its whole block back to the
/// interpreter, so this is the predicate that says which blocks can be
/// emitted at all. `examples/emit_selftest.rs` sweeps the encoding space with
/// it to find instructions to exercise, rather than assembling them by hand
/// and testing whatever it actually encoded.
///
/// An instruction that ends a block or branches out of one answers `false`:
/// those are the translator's terminators and exits, and no emitter writes
/// them yet.
pub fn emits(insn: u32) -> bool {
    // PC-relative forms decode against an address; which one makes no
    // difference to whether there is an arm for the result, so any aligned
    // address answers, exactly as in [`super::translates`].
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

/// Whether the emitted form of `insn` can decline the instruction and hand it
/// back to the interpreter, rather than always doing it itself.
///
/// Every guest access can: the page it names may have no storage, may not hold
/// all of the access, or may be watched, and each of those is a case the
/// emitted code leaves to the full path. Nothing else can, so a block of
/// instructions that all answer `false` retires every one of them.
///
/// That distinction is what the difftests need in order to know which answer
/// from `run` is the right one, so it is derived from the emitter rather than
/// listed beside it: a declining op writes the count it would report into its
/// body, so emitting the same instruction under two different counts and
/// comparing the bodies asks the arms themselves, and cannot fall out of step
/// with them the way a second list of ops would.
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
/// it is handed.
///
/// Taken from `core::mem::offset_of!` at the call site rather than being
/// spelled out here: `Cpu` is not `#[repr(C)]`, so the only offsets that are
/// right are the ones the compiler actually chose for this build, and the
/// emitter runs in that same build.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    /// Start of the `[u64; REG_FILE]` register file.
    pub regs: u32,
    /// The packed NZCV word, in its architectural bit positions.
    pub nzcv: u32,
    /// Where the page table's own pointer is kept, not where the table is: an
    /// emitted block loads it and indexes it, because
    /// [`crate::mem::Memory`] holds the table behind a `Box` and the
    /// allocation it points at is not known when the block is written.
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
    /// Where the pointer to the bitmap of pages something has cached the
    /// contents of is kept; null until something watches one. A store landing
    /// on a watched page owes a report that an emitted one cannot make.
    pub watched: u32,
}

/// Bytes per page-table entry, and so the shift that turns a page number into
/// an offset into the table.
///
/// One entry is one `Option<Box<[u8; PAGE_SIZE]>>`, which the language
/// guarantees is laid out as the pointer alone with `None` written as null:
/// that null *is* how [`crate::mem::Memory`] says a page has no storage, so an
/// emitted access tests for it directly. Four bytes because the only target
/// that ever runs emitted code is `wasm32`, which the second assertion is
/// against; the first is the part that holds on the host too, where the
/// pointer is wider but the niche is the same.
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

/// Alignment hints, as the log2 the format wants. Guest state is naturally
/// aligned because this emulator laid it out; a guest address is aligned to
/// nothing the emitter can promise, and A64 permits unaligned accesses.
const ALIGN_4: u8 = 2;
const ALIGN_8: u8 = 3;
const UNALIGNED: u8 = 0;

/// Why a block was not written out.
///
/// The interesting one is [`Refused::Op`]: it names an instruction real code
/// runs that the emitter has no way to write, and a count of those across a
/// title's blocks is the list of what to write next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// The block runs through a conditional branch. Nothing about control
    /// flow is written yet, so the whole block stays with the interpreter.
    ControlFlow,
    /// An op with no emitter, as the instruction word it was decoded from.
    Op(u32),
    /// The body grew past [`MAX_BODY_BYTES`].
    TooLong,
}

/// The parameter every block function takes.
const STATE: u32 = 0;

/// Scratch locals. Named rather than numbered at the use site because the
/// stack discipline makes an off-by-one here validate and compute nonsense.
const L_A: u32 = 1;
const L_B: u32 = 2;
const L_T: u32 = 3;
const L_R: u32 = 4;
/// Held by the one operation that needs its operand twice and has it once: a
/// 32-bit rotate, which wasm has no instruction for.
const L_S: u32 = 5;
/// The value an addressing mode with writeback leaves in its base register,
/// computed before the access and written after it.
const L_W: u32 = 6;
const L_C: u32 = 7;
/// The guest address an access is at, narrowed to the 32 bits the guest
/// address space has.
const L_ADDR: u32 = 8;
/// The host address of the guest page that address is on.
const L_PAGE: u32 = 9;
/// Which guest page that is, which both the page table and the bitmap of
/// watched pages are indexed by.
const L_IDX: u32 = 10;
/// The bitmap of watched pages, held because the test against it is inside a
/// check that there is one at all.
const L_WP: u32 = 11;

/// A block bigger than this is not emitted. A translated block is bounded
/// already, but the guard keeps one pathological block from dominating a
/// module's compile time.
const MAX_BODY_BYTES: usize = 64 * 1024;

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
    /// Push `regs[slot]` as it is stored, all 64 bits of it.
    ///
    /// The right read for anything that shifts its operand's own bits out of
    /// the way before using it, which is what `SBFM`/`UBFM` do.
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

    /// Sign-extend the value on the stack from the operation's width, which a
    /// 64-bit operand already is.
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

    /// Push the address a write to the register file needs. A store takes its
    /// address *under* its value, so this comes first and the slot is named
    /// again by [`Emitter::store_reg`], which carries it as the instruction's
    /// static offset.
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

    /// Invert the value on the stack when the operation subtracts, exactly as
    /// [`super::exec`]'s `invert_if` does, and narrow it again because
    /// inverting a 32-bit value sets the top half.
    fn invert_if(&mut self, carry: u8, sf: bool) {
        if carry != 0 {
            self.f.i64_const(-1);
            self.f.i64_xor();
            self.mask_to(sf);
        }
    }

    /// Push 1 if `cond` holds under the current NZCV and 0 if it does not.
    ///
    /// [`crate::cpu::Cpu::condition_holds`] indexes [`CONDITION_MASKS`] with
    /// the condition and shifts the row it finds by the flags. The row is a
    /// constant here, so what is left is that shift.
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

    /// [`crate::cpu::bits::shift_reg`] applied to the value on the stack, with
    /// the shift type and distance already known.
    ///
    /// An out-of-range distance is not reachable from an allocated encoding,
    /// but the interpreter answers for one, so this does too: the logical
    /// shifts give zero and the arithmetic one gives the sign, which is what
    /// clamping its distance to the top bit produces.
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

    /// Rotate the value on the stack right by `sa`, which is in range and not
    /// zero.
    ///
    /// wasm rotates 64-bit values and 32-bit ones, and a 32-bit guest value
    /// here is held in an `i64`, so the narrow form is written out as the two
    /// shifts it is. The operand is needed twice and arrives once, which is
    /// what `L_S` is for.
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

    /// [`crate::cpu::bits::extend_reg`] applied to the value on the stack,
    /// with the option already known: one mask or one sign-extension.
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

    /// `a + b + carry` from `L_A` and `L_B`, leaving the result narrowed to
    /// the operation's width in `L_R` and, when the flags are wanted, the
    /// carry out in `L_C`.
    ///
    /// The carry-in is a constant here, so the second half of the two-add
    /// carry chain is only written when the operation is a subtraction, and
    /// the chain itself only when the operation is 64 bits wide: a 32-bit
    /// operation carries out of bit 31, and both operands were narrowed, so
    /// the sum cannot have wrapped and the carry is simply there in the
    /// untruncated word.
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

    /// `ADD`/`SUB`/`ADDS`/`SUBS` once both operands are in `L_A` and `L_B`,
    /// with the direction already folded into `L_B` and `carry` exactly as
    /// [`crate::cpu::Cpu::add_sub_pre`] receives them.
    fn add_sub(&mut self, rd: u8, carry: u8, set_flags: bool, sf: bool) {
        self.add_carry(carry, set_flags, sf);
        if set_flags {
            self.pack_nzcv(sf, true);
            self.store_nzcv();
        }
        self.write_reg_from_r(rd);
    }

    /// Push NZCV as the packed word it is stored as, built from `L_R` and,
    /// for an arithmetic operation, `L_A`, `L_B` and `L_C`.
    ///
    /// `arithmetic` says whether C and V are computed from the operands (an
    /// add or a subtract) or carried over from what NZCV already held, which
    /// is what `ANDS` does.
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

    /// Store the packed word on the stack into NZCV, putting the address back
    /// under it.
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

    /// `MADD`/`MSUB` and the widening `SMADDL`/`UMADDL` family, which differ
    /// only in how the two multiplicands are read.
    ///
    /// The widening forms take the low 32 bits of each operand whatever the
    /// destination width, and a 32x32 product fits in 64 bits, so neither
    /// needs the 128-bit arithmetic wasm does not have.
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

    /// `UDIV`/`SDIV`, with `L_A` the dividend and `L_B` the divisor, both
    /// already sign-extended for the signed forms.
    ///
    /// A64 answers zero for a division by zero and wraps `INT_MIN / -1`;
    /// wasm traps on both. Neither guard can be a [`Func::select`], which
    /// would run the division it did not pick, so both are real branches.
    ///
    /// Only the 64-bit signed form needs the second guard. A 32-bit one has
    /// its operands sign-extended from 32 bits, so `INT_MIN / -1` is
    /// `0x8000_0000` in an `i64` and nothing overflows.
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
            // `x.wrapping_div(-1)` is `x.wrapping_neg()`, which is the answer
            // for `i64::MIN` as well as for everything else.
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

    /// Leave the block, reporting that `retired` of its instructions are done.
    ///
    /// The value on the stack decides: non-zero leaves. What is left behind is
    /// guest state as it stood before the instruction being emitted, so the
    /// interpreter picks that instruction up whole, which is the same handover
    /// [`super::exec`] already makes when a step budget runs out inside a
    /// block.
    fn deopt_if(&mut self, retired: usize) {
        self.f.if_void();
        self.f.i32_const(retired as i32);
        self.f.return_();
        self.f.end();
    }

    /// Leave the guest address of an access in `L_ADDR`, and the value its
    /// addressing mode writes back, if it writes one, in `L_W`.
    ///
    /// The three modes are [`crate::cpu::Cpu::indexed`]: the access is at
    /// `base + offset` except under post-indexing, where it is at `base` and
    /// `base + offset` is only what the base register ends up holding.
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

    /// The same for the register-offset form, which has no writeback:
    /// [`crate::cpu::Cpu::reg_offset`]'s extension and scale, both already
    /// resolved.
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
    /// reading `n` bytes there takes more than the page table.
    ///
    /// The three conditions are [`crate::mem::Memory::peek`]'s, in its order:
    /// the access crosses a page boundary, the read watchpoint covers part of
    /// it, or the page has no storage. Each is something the full path has
    /// more to do about, and each is rare, so the emitted access is the
    /// straight-line case and the interpreter keeps the rest.
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

    /// The same for a store, whose conditions are
    /// [`crate::mem::Memory::poke`]'s: the boundary again, a write-protected
    /// address, the write watchpoint, and a page whose contents something has
    /// cached, which a store owes a report to.
    fn page_for_write(&mut self, n: u32, retired: usize) {
        self.page_number();
        let crosses = n > 1;
        if crosses {
            self.crosses_page(n);
        }
        self.within(self.layout.readonly_lo, self.layout.readonly_hi);
        if crosses {
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

    /// Leave that page's storage in `L_PAGE`, or leave the block if it has
    /// none. An unmapped page is a null entry, so this is the entry itself.
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

    /// Push whether `L_ADDR` is inside `[lo, hi)`, which is the envelope test
    /// [`crate::mem::Memory::is_readonly`] makes before it walks its list.
    ///
    /// The list is not walked here: a store inside the envelope hands back and
    /// the full path decides whether that address really is protected. Every
    /// protected range is a module's `.text`, so hardly a store a title makes
    /// is in the envelope at all.
    fn within(&mut self, lo: u32, hi: u32) {
        self.f.local_get(L_ADDR);
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, lo);
        self.f.i32_ge_u();
        self.f.local_get(L_ADDR);
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, hi);
        self.f.i32_lt_u();
        self.f.i32_and();
    }

    /// Leave the block if anything has cached page `L_IDX`'s contents, which
    /// is [`crate::mem::Memory`]'s `watches_code`.
    ///
    /// Inside a check that the bitmap is there at all, because it is not
    /// allocated until something watches a page and a run with no JIT and no
    /// GPU backend never has one.
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
        // The shift is taken modulo 64, which is the masking the bit index
        // would otherwise need.
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

    /// Push whether the watchpoint held at `lo`/`hi` covers any of the `n`
    /// bytes at `L_ADDR`: `addr < end && addr + n > start`, which is the
    /// overlap test [`crate::mem::Memory::peek`] makes, wrapping included.
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

    /// Push the host address the access is at: the page, plus the offset into
    /// it, which the boundary check above has already proved leaves room.
    fn in_page(&mut self) {
        self.f.local_get(L_PAGE);
        self.f.local_get(L_ADDR);
        self.f.i32_const((PAGE_SIZE - 1) as i32);
        self.f.i32_and();
        self.f.i32_add();
    }

    /// A load into `rt`, with the address already in `L_ADDR`.
    ///
    /// The widening is the load instruction's own: A64's zero- and
    /// sign-extending loads are exactly wasm's `_u` and `_s` forms, so only
    /// the two that sign-extend to 32 bits and then zero the top half need
    /// anything after the access.
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

    /// A store of `rt`, with the address already in `L_ADDR`.
    ///
    /// The narrowing is the store instruction's own: A64 stores the low bytes
    /// of the register whatever the width, which is what wasm's narrow stores
    /// do, so nothing is masked first.
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

    /// A single-register access, whatever addressed it.
    ///
    /// `PRFM` is an access with no memory in it, and its addressing mode still
    /// writes back, so it is the one arm that emits no guard at all: there is
    /// nothing for the page table to answer.
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

            Op::MovK { rd, shift, val, sf } => {
                // The field is replaced, not merged, and the whole result is
                // narrowed after: a 32-bit MOVK zeroes the top half.
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
                // `BIC`/`ORN`/`EON` invert the *shifted* operand, not the
                // register, which is the order the ARM ARM's pseudocode has.
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
                // The operand's own high bits are shifted out by `left`, so
                // this reads the register whole rather than narrowing first.
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
                // Only `BFM` reaches here with anything to do: the forms that
                // discard the destination are `Op::Extract`, and the
                // unallocated `opc` writes zero.
                if opc != 0b01 {
                    self.addr_regs();
                    self.f.i64_const(0);
                    self.store_reg(rd);
                    return true;
                }
                let (lsb, msb) = (u32::from(immr), u32::from(imms));
                // Both branches of `bitfield_insert` are the same merge of a
                // placed field into the destination; they differ in where the
                // field comes from and which bits it lands on.
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
                    // wasm would take this modulo 64; a 32-bit shift wants it
                    // modulo 32.
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
                    // A 32-bit rotate by a variable amount, as the two shifts
                    // it is. At an amount of zero the left shift is by 32 and
                    // the narrowing at the end discards it, which is what
                    // makes the no-rotation case come out right without a
                    // branch of its own.
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
                            // Takes the low half and fills the rest with its
                            // sign, so the narrowing is part of it.
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
                // The invert and the increment belong to the *else* value,
                // not to whichever value the condition picks.
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
                    // Narrowed, as [`crate::cpu::Cpu::add_carry_overflow`]
                    // narrows both its operands. Read raw, a 32-bit `CCMN`
                    // carried the top half of Rm into a sum whose carry is
                    // taken from bit 32, and reported C from bits the
                    // operation does not have. `CCMP` hid it: inverting the
                    // operand masks it again on the way past.
                    self.read_reg(rm, sf);
                }
                self.invert_if(u8::from(sub), sf);
                self.f.local_set(L_B);
                self.add_carry(u8::from(sub), true, sf);
                // Both answers are values, so the condition picks between the
                // flags the compare produced and the ones the instruction
                // carries rather than branching around the compare.
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

/// How many bytes an access touches, which is what its page and its
/// watchpoint are tested against.
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
///
/// `run` takes the address of the guest state and returns how many
/// instructions it retired, which for now is always the whole body: a block
/// with an op the emitter cannot write is not emitted at all.
pub(super) fn emit_block(block: &Block, layout: Layout) -> Result<Vec<u8>, Refused> {
    let mut f = scratch_func();
    let mut e = Emitter { f: &mut f, layout };
    for (i, op) in block.ops.iter().enumerate() {
        if !e.op(op, i) {
            return Err(Refused::Op(block.words[i]));
        }
        if e.f.len() > MAX_BODY_BYTES {
            return Err(Refused::TooLong);
        }
    }
    // Control flow is not written yet. A block with a conditional exit would
    // need its branch and the not-taken path, and one with a terminator has to
    // say where control went; until both exist, only straight-line blocks are
    // emitted and everything else stays with the interpreter.
    //
    // Asked after the body and not before it, which wastes the body of a block
    // that is refused anyway. Nothing calls this on a hot path, and the count
    // of blocks refused here is then the count of blocks that *only* control
    // flow is holding back, which is the number that says whether writing
    // branches would pay. Asked first, it hides every block that would still
    // need a load written before it could be emitted.
    if !block.exits.is_empty() {
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
    /// Translate the block at `pc` and emit it, for `examples/emit_difftest.rs`.
    ///
    /// `layout` is where the harness has put guest state in the memory it
    /// hands the module, which for a test is a bare buffer rather than a
    /// `Cpu`. Reports how many instructions the block covers, so
    /// the harness can step the interpreter over exactly the same ones.
    pub fn emit_block_at(&self, pc: u32, layout: Layout) -> Result<(Vec<u8>, usize), Refused> {
        let block = translate(&self.mem, pc);
        let bytes = emit_block(&block, layout)?;
        Ok((bytes, block.ops.len()))
    }

    /// The register file by *slot*, which is what an emitted block addresses.
    ///
    /// [`Cpu::read_reg`] takes an encoding's five-bit field and so cannot
    /// reach the three slots register 31 resolves to; a difference in `SP` or
    /// in the discard slot is exactly the kind an emitter gets wrong, so the
    /// harness compares all of them.
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
        pages: 2052,
        read_watch_lo: 2056,
        read_watch_hi: 2060,
        watch_lo: 2064,
        watch_hi: 2068,
        readonly_lo: 2072,
        readonly_hi: 2076,
        watched: 2080,
    };

    /// An op with no emitter has to take the whole block out of the emitted
    /// path rather than being skipped, or the block would run with an
    /// instruction missing. The refusal names the instruction, which is what
    /// the coverage report ranks.
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

    /// The module has to carry the magic and version a browser checks first,
    /// so a malformed header is caught here rather than as a `CompileError`
    /// with no offset.
    #[test]
    fn an_emitted_block_is_a_wasm_module() {
        let b = block_of(vec![
            Op::MovConst { rd: 1, val: 0x1234 },
            Op::AddSubImm {
                rd: 2,
                rn: 1,
                rhs: 1,
                carry: 0,
                set_flags: true,
                sf: true,
            },
        ]);
        let bytes = emit_block(&b, LAYOUT).expect("both ops are writable");
        assert_eq!(&bytes[..4], &[0x00, 0x61, 0x73, 0x6D]);
        assert_eq!(&bytes[4..8], &[0x01, 0x00, 0x00, 0x00]);
    }

    /// A block that runs through a conditional branch says so, rather than
    /// being counted against the ops it contains: the two refusals call for
    /// completely different work.
    #[test]
    fn a_conditional_branch_refuses_as_control_flow() {
        use crate::cpu::jit::ir::{Branch, Exit};

        let ops = vec![Op::MovConst { rd: 0, val: 7 }];
        let exits = vec![Branch::new(0, Exit::Cond { cond: 0, target: 8 })];
        let b = Block::new(0x1000, ops, vec![0], exits, None, vec![1]);
        assert_eq!(emit_block(&b, LAYOUT), Err(Refused::ControlFlow));
    }
}
