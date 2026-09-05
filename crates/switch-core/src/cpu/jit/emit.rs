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
//! # Coverage is a performance question, not a correctness one
//!
//! [`emit_block`] returns `None` for a block containing anything it cannot
//! write, and that block keeps running on the interpreter. This is the same
//! rule [`super::decode`]'s `Op::Interpret` follows: a form the emitter does
//! not know is slower, never wrong. So the supported set can grow one op at a
//! time, each addition backed by `emit_difftest`, rather than needing to be
//! complete before any of it can run.

use super::decode::translate;
use super::ir::{Block, Op};
use super::wasm::{Func, Module, I32, I64};
use crate::cpu::Cpu;

/// Byte offsets of the guest state an emitted block touches, from the pointer
/// it is handed.
///
/// Taken from `core::mem::offset_of!` at the call site rather than being
/// spelled out here: `Cpu` is not `#[repr(C)]`, so the only offsets that are
/// right are the ones the compiler actually chose for this build, and the
/// emitter runs in that same build.
#[derive(Debug, Clone, Copy)]
pub(super) struct Layout {
    /// Start of the `[u64; REG_FILE]` register file.
    pub(super) regs: u32,
    /// The packed NZCV word, in its architectural bit positions.
    pub(super) nzcv: u32,
}

/// The parameter every block function takes.
const STATE: u32 = 0;

/// Scratch locals. Named rather than numbered at the use site because the
/// stack discipline makes an off-by-one here validate and compute nonsense.
const L_A: u32 = 1;
const L_B: u32 = 2;
const L_T: u32 = 3;
const L_R: u32 = 4;
const L_C: u32 = 5;

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

struct Emitter<'a> {
    f: &'a mut Func,
    layout: Layout,
}

impl Emitter<'_> {
    /// Push `regs[slot]`, already narrowed to the operation's width.
    fn read_reg(&mut self, slot: u8, sf: bool) {
        self.f.local_get(STATE);
        self.f.i64_load(self.layout.regs + 8 * u32::from(slot));
        if !sf {
            self.f.i64_const(width_mask(false));
            self.f.i64_and();
        }
    }

    /// Push the address a write to the register file needs. A store takes its
    /// address *under* its value, so this comes first and the slot is named
    /// again by [`Emitter::store_reg`], which carries it as the instruction's
    /// static offset.
    fn addr_regs(&mut self) {
        self.f.local_get(STATE);
    }

    fn store_reg(&mut self, slot: u8) {
        self.f.i64_store(self.layout.regs + 8 * u32::from(slot));
    }

    /// Write the local `L_R` into `regs[slot]`.
    fn write_reg_from_r(&mut self, slot: u8) {
        self.addr_regs();
        self.f.local_get(L_R);
        self.store_reg(slot);
    }

    /// `ADD`/`SUB`/`ADDS`/`SUBS` once both operands are in `L_A` and `L_B`,
    /// with the direction already folded into `L_B` and `carry` exactly as
    /// [`crate::cpu::Cpu::add_sub_pre`] receives them.
    ///
    /// The carry-in is a constant here, so the second half of the two-add
    /// carry chain is only written when the operation is a subtraction.
    fn add_sub(&mut self, rd: u8, carry: u8, set_flags: bool, sf: bool) {
        // t = a + b
        self.f.local_get(L_A);
        self.f.local_get(L_B);
        self.f.i64_add();
        self.f.local_set(L_T);

        if set_flags {
            // c1 = t <u a, the carry out of the first add
            self.f.local_get(L_T);
            self.f.local_get(L_A);
            self.f.i64_lt_u();
            self.f.local_set(L_C);
        }

        if carry != 0 {
            if set_flags {
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

        // r = t & mask
        self.f.local_get(L_T);
        if !sf {
            self.f.i64_const(width_mask(false));
            self.f.i64_and();
        }
        self.f.local_set(L_R);

        if set_flags {
            // A 32-bit operation carries out of bit 31, and both operands were
            // narrowed, so the sum cannot have wrapped and the carry is simply
            // there in the untruncated word.
            if !sf {
                self.f.local_get(L_T);
                self.f.i64_const(32);
                self.f.i64_shr_u();
                self.f.i32_wrap_i64();
                self.f.i32_const(1);
                self.f.i32_and();
                self.f.local_set(L_C);
            }
            self.emit_nzcv(sf, true);
        }
        self.write_reg_from_r(rd);
    }

    /// Pack NZCV from `L_A`, `L_B`, `L_R` and `L_C` and store it.
    ///
    /// `arithmetic` says whether V is computed from the operands (an add or a
    /// subtract) or carried over from what NZCV already held, which is what
    /// `ANDS` does.
    fn emit_nzcv(&mut self, sf: bool, arithmetic: bool) {
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
            self.f.i32_load(self.layout.nzcv);
            self.f.i32_const(0x3000_0000);
            self.f.i32_and();
            self.f.i32_or();
        }

        // The packed word is on the stack; put the address under it.
        self.f.local_set(L_C);
        self.f.local_get(STATE);
        self.f.local_get(L_C);
        self.f.i32_store(self.layout.nzcv);
    }

    /// `AND`/`ORR`/`EOR`/`ANDS` with the second operand already in `L_B`.
    fn logical(&mut self, rd: u8, rn: u8, opc: u8, sf: bool) {
        self.read_reg(rn, sf);
        self.f.local_set(L_A);
        self.f.local_get(L_A);
        self.f.local_get(L_B);
        match opc {
            0b00 | 0b11 => self.f.i64_and(),
            0b01 => self.f.i64_or(),
            _ => self.f.i64_xor(),
        }
        self.f.local_set(L_R);
        if opc == 0b11 {
            self.emit_nzcv(sf, false);
        }
        self.write_reg_from_r(rd);
    }

    /// One op, or `false` if there is no way to write it yet.
    fn op(&mut self, op: &Op) -> bool {
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
                self.f.local_get(STATE);
                self.f.i64_load(self.layout.regs + 8 * u32::from(rd));
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
                if carry != 0 {
                    // A subtraction is the addition of the inverted operand,
                    // narrowed again because inverting a 32-bit value sets the
                    // top half.
                    self.f.i64_const(-1);
                    self.f.i64_xor();
                    if !sf {
                        self.f.i64_const(width_mask(false));
                        self.f.i64_and();
                    }
                }
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
                if invert {
                    self.f.i64_const(-1);
                    self.f.i64_xor();
                    if !sf {
                        self.f.i64_const(width_mask(false));
                        self.f.i64_and();
                    }
                }
                self.f.local_set(L_B);
                self.logical(rd, rn, opc, sf);
                true
            }

            _ => false,
        }
    }
}

/// Emit `block`'s body as a module exporting `run`.
///
/// `run` takes the address of the guest state and returns how many
/// instructions it retired, which for now is always the whole body: a block
/// with an op the emitter cannot write is not emitted at all.
///
/// Returns `None` when the block cannot be written, which the caller reads as
/// "keep interpreting this one".
pub(super) fn emit_block(block: &Block, layout: Layout) -> Option<Vec<u8>> {
    // Control flow is not written yet. A block with a conditional exit would
    // need its branch and the not-taken path, and one with a terminator has to
    // say where control went; until both exist, only straight-line blocks are
    // emitted and everything else stays with the interpreter.
    if !block.exits.is_empty() {
        return None;
    }
    let mut f = Func::new();
    f.locals(1, 4, I64);
    f.locals(1, 1, I32);

    let mut e = Emitter { f: &mut f, layout };
    for op in &block.ops {
        if !e.op(op) {
            return None;
        }
        if e.f.len() > MAX_BODY_BYTES {
            return None;
        }
    }
    f.i32_const(block.ops.len() as i32);
    f.end();

    let mut m = Module::new();
    let ty = m.add_type(vec![I32], vec![I32]);
    let idx = m.add_func(ty, f);
    m.export("run", idx);
    Some(m.finish())
}

impl Cpu {
    /// Translate the block at `pc` and emit it, for `examples/emit_difftest.rs`.
    ///
    /// The offsets are where the harness has put the register file and NZCV in
    /// the memory it hands the module, which for a test is a bare buffer
    /// rather than a `Cpu`. Reports how many instructions the block covers, so
    /// the harness can step the interpreter over exactly the same ones.
    pub fn emit_block_at(&self, pc: u32, regs: u32, nzcv: u32) -> Option<(Vec<u8>, usize)> {
        let block = translate(&self.mem, pc);
        let bytes = emit_block(&block, Layout { regs, nzcv })?;
        Some((bytes, block.ops.len()))
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
        Block::new(0x1000, ops, words, Vec::new(), None)
    }

    const LAYOUT: Layout = Layout {
        regs: 0,
        nzcv: 2048,
    };

    /// An op with no emitter yet has to take the whole block out of the
    /// emitted path rather than being skipped, or the block would run with an
    /// instruction missing.
    #[test]
    fn an_unwritable_op_refuses_the_whole_block() {
        let good = block_of(vec![Op::MovConst { rd: 0, val: 7 }]);
        assert!(emit_block(&good, LAYOUT).is_some());

        let bad = block_of(vec![
            Op::MovConst { rd: 0, val: 7 },
            Op::Interpret { insn: 0xD503201F },
        ]);
        assert!(emit_block(&bad, LAYOUT).is_none());
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
}
