//! A block-translating JIT for the A64 core.
//!
//! The interpreter re-derives everything about an instruction every time it
//! runs it: which of the eight top-level groups it belongs to, which of that
//! group's forms it is, where its operand fields sit, what its immediate
//! decodes to. In a loop body that runs a million times, that work is done a
//! million times and produces the same answer every time.
//!
//! This module does it once. The first time control reaches an address, the
//! translator walks forward from it decoding instructions into [`ir::Op`]s
//! — the operation with its operands already extracted, its immediates already
//! decoded, its PC-relative addresses already resolved, and every field the
//! interpreter re-reads per execution (a load's width and direction, a
//! register offset's extension, whether an add is really a subtract, which
//! system register an `MRS` names, which floating-point form an encoding is)
//! already resolved to the one thing the instruction does. That run of ops
//! plus its terminator is a [`ir::Block`], cached by entry address, and every
//! later visit executes it with no decoding at all.
//!
//! A block does not end at the first branch. The three conditional branches —
//! `B.cond`, `CBZ`/`CBNZ`, `TBZ`/`TBNZ` — have the following instruction as
//! their not-taken path, so translation carries on through them and each
//! becomes an [`ir::Exit`] the body is checked against on the way past. Only an
//! instruction that *always* leaves ends a block. `b.cond` alone is 12% of an
//! hbmenu frame, so this is the difference between blocks that average seven
//! instructions and blocks that average thirteen.
//!
//! What this removes is decode, not dispatch — three quarters of where the
//! interpreter's time went (see [`super::Cpu::execute`]'s note on the group
//! dispatcher). It does not generate code.
//!
//! # Why not generate wasm
//!
//! Emitting a wasm module per translation unit and handing it to
//! `WebAssembly.Module` is a real JIT, and it is the next step for speed
//! rather than something the browser forbids. What stops it being *this*
//! step is the memory model, not the platform.
//!
//! A generated module can only address its own linear memory directly. Guest
//! memory cannot be that: this emulator is itself a wasm32 module, whose
//! linear memory caps at 4 GiB, and the guest address space is 4 GiB reaching
//! up to [`super::GUEST_SPACE_END`] — so an identity map does not fit, and
//! would give up the lazy soft-mapping that makes a multi-gigabyte heap cost
//! nothing until it is touched. [`crate::mem::Memory`] is a page table of
//! boxed 4 KiB pages with soft regions, read-only ranges and watchpoints over
//! it. Generated code would have to reach all of that through imported host
//! calls, one per guest load and store — which is most of what a block does,
//! and exactly what the codegen was supposed to make cheap.
//!
//! Two smaller things point the same way: compiling a module goes out through
//! JS, so a basic block is far too small a unit to pay for one, and generated
//! code would not exist in host builds — leaving the test suite and
//! `examples/jit_difftest.rs` covering only one of the two engines. Flattening
//! the guest address space behind a base-plus-bounds check is the change that
//! has to come first.
//!
//! # Fidelity
//!
//! Every op executes the same helper the interpreter's decoder would have
//! called with the same arguments, so translated and interpreted execution are
//! the same computation. Anything the translator does not have an op for — the
//! exclusive accessors, the divides and variable shifts, the encodings the
//! interpreter rejects as unallocated — becomes [`ir::Op::Interpret`], which hands
//! the raw instruction word straight back to [`super::Cpu::execute`]. That
//! makes the translator's coverage a performance question rather than a
//! correctness one: a form it does not know is slower, never wrong.
//!
//! SIMD and floating point are half-way between. They have no ops of their
//! own, but which decoder owns an encoding — and, for the scalar forms, which
//! of [`super::fp::FpForm`]'s eight groups it belongs to — is settled at
//! translation time, so [`ir::Op::Fp`] enters the right handler directly instead
//! of walking a guard chain. The classification lives in
//! [`super::Cpu::fp_form`] and the interpreter asks the same function, so the
//! two cannot drift.
//!
//! An op may only be a mid-block [`ir::Op::Interpret`] if it cannot move the PC
//! anywhere but to the following instruction. Every A64 instruction that can
//! is in the branch/exception/system group (bits 28:25 = `101x`), so that
//! group is decoded as a terminator — except the `D503xxxx` hints and
//! barriers, which are no-ops, and the `MRS`/`MSR`/cache-maintenance forms,
//! which [`super::Cpu::system`] always retires to `next_pc`.
//!
//! # Staleness
//!
//! A block is only valid while the instructions it was built from are still
//! there. [`crate::mem::Memory`] records which pages have been translated out
//! of and reports the ones a store has landed on;
//! [`crate::cpu::Cpu::jit_block_at`] drains that list before every lookup and
//! drops the blocks translated from those pages. A block never spans a page,
//! so one page's worth of invalidation is exact.
//!
//! # Where things are
//!
//! [`ir`] is what a block is made of, [`cache`] is which blocks exist and when
//! a guest store takes one away, [`decode`] builds them and [`exec`] runs
//! them. What an instruction *means* is in none of them: those bodies are
//! shared with the interpreter and live with the semantics, in
//! [`crate::cpu::alu`], [`crate::cpu::loadstore`] and [`crate::cpu::system`].

mod cache;
mod decode;
mod exec;
pub(in crate::cpu) mod ir;

pub use cache::JitStats;
pub use decode::translates;

pub(in crate::cpu) use cache::Jit;
