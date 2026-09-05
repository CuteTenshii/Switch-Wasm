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
//!, the operation with its operands already extracted, its immediates already
//! decoded, its PC-relative addresses already resolved, and every field the
//! interpreter re-reads per execution (a load's width and direction, a
//! register offset's extension, whether an add is really a subtract, which
//! system register an `MRS` names, which floating-point form an encoding is)
//! already resolved to the one thing the instruction does. That run of ops
//! plus its terminator is a [`ir::Block`], cached by entry address, and every
//! later visit executes it with no decoding at all.
//!
//! A block does not end at the first branch. The three conditional branches,
//! `B.cond`, `CBZ`/`CBNZ`, `TBZ`/`TBNZ`: have the following instruction as
//! their not-taken path, so translation carries on through them and each
//! becomes an [`ir::Exit`] the body is checked against on the way past. Only an
//! instruction that *always* leaves ends a block. `b.cond` alone is 12% of an
//! hbmenu frame, so this is the difference between blocks that average seven
//! instructions and blocks that average thirteen.
//!
//! What this removes is decode, not dispatch, three quarters of where the
//! interpreter's time went (see [`super::Cpu::execute`]'s note on the group
//! dispatcher). It does not generate code.
//!
//! # Generating wasm
//!
//! [`emit`] writes a block out as wasm rather than interpreting it, using the
//! encoder in [`wasm`]. It is started but not wired in: only straight-line
//! data-processing blocks are written, and [`emit::emit_block`] returns `None`
//! for everything else, so nothing executes emitted code yet.
//!
//! What made this look impossible was the memory model, and two thirds of that
//! turned out to be wrong. A generated module can only address *its own*
//! linear memory, but it can **import** the host's, and this emulator is
//! itself a wasm module: the guest register file and NZCV are at fixed offsets
//! inside its `Cpu`, so an emitted block reaches them with a plain `i64.load`
//! and nothing is copied or called. Compiling was supposed to be too dear for
//! so small a unit, and measured it is not: 7,062 blocks compile in 1.8 ms, so
//! the answer is to batch a module rather than to emit one per block.
//!
//! What is genuinely still open is guest *memory*. [`crate::mem::Memory`] is a
//! page table of boxed 4 KiB pages with soft regions, read-only ranges and
//! watchpoints, and emitted code cannot walk that without a host call per
//! access, which is most of what a block does and exactly what codegen was
//! meant to make cheap. Flattening the address space behind one bounds check
//! has to come before loads and stores are emitted.
//!
//! Generated code also does not exist in host builds, which would have left it
//! untested; `examples/emit_difftest.rs` and `tools/emit_difftest.mjs` are the
//! answer to that, running the emitted module under V8 against the interpreter
//! on blocks a real title executes.
//!
//! # Fidelity
//!
//! Every op executes the same helper the interpreter's decoder would have
//! called with the same arguments, so translated and interpreted execution are
//! the same computation. Anything the translator does not have an op for, the
//! exclusive accessors, the divides and variable shifts, the encodings the
//! interpreter rejects as unallocated: becomes [`ir::Op::Interpret`], which hands
//! the raw instruction word straight back to [`super::Cpu::execute`]. That
//! makes the translator's coverage a performance question rather than a
//! correctness one: a form it does not know is slower, never wrong.
//!
//! SIMD and floating point are half-way between. They have no ops of their
//! own, but which decoder owns an encoding, and, for the scalar forms, which
//! of [`super::fp::FpForm`]'s eight groups it belongs to, is settled at
//! translation time, so [`ir::Op::Fp`] enters the right handler directly instead
//! of walking a guard chain. The classification lives in
//! [`super::Cpu::fp_form`] and the interpreter asks the same function, so the
//! two cannot drift.
//!
//! An op may only be a mid-block [`ir::Op::Interpret`] if it cannot move the PC
//! anywhere but to the following instruction. Every A64 instruction that can
//! is in the branch/exception/system group (bits 28:25 = `101x`), so that
//! group is decoded as a terminator, except the `D503xxxx` hints and
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
//! a guest store takes one away, [`decode`] builds them, [`exec`] runs them,
//! and [`emit`] writes them out as wasm through [`wasm`]. What an instruction *means* is in none of them: those bodies are
//! shared with the interpreter and live with the semantics, in
//! [`crate::cpu::alu`], [`crate::cpu::loadstore`] and [`crate::cpu::system`].

mod cache;
mod decode;
mod emit;
mod exec;
pub(in crate::cpu) mod ir;
mod wasm;

pub use cache::JitStats;
pub use decode::translates;

pub(in crate::cpu) use cache::Jit;
