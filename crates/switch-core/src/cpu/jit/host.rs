//! Where emitted wasm goes to become something that can be called.
//!
//! [`super::emit`] produces the bytes of a module; compiling one and making it
//! callable needs `WebAssembly.Module`, which is a browser API. This crate has
//! no dependencies and cannot reach it (see AGENTS.md), so the embedder
//! supplies the two operations that do, and everything that decides *when* to
//! use them stays here.
//!
//! # An entry point is an address, whatever the target calls one
//!
//! [`JitHost::install`] answers the thing this build calls a code address: on
//! `wasm32` an index into the module's function table, because that is what a
//! function pointer there *is*, and on the host the address of a real
//! function. [`enter`] calls it either way, which is why nothing in this file
//! needs to know which target it is on, and why the whole of the machinery
//! around it can be driven by a host test with a Rust function standing in for
//! a compiled block.
//!
//! The obvious alternative, an import the emitted code is called through,
//! would be a host call per block entry. A retail frame enters a block every
//! 6.1 instructions, so at a million-odd entries a frame the glue alone would
//! cost more than the dispatch the emitter exists to remove. The boundary is
//! crossed once, when a block is installed, and never again while it runs.
//!
//! # What the embedder has to do
//!
//! Compile the module with the emulator's own linear memory imported as
//! `e`.`m`, put its `run` export somewhere an indirect call can reach, and
//! answer where. `switch-wasm` does this with `wasm_bindgen::memory` and
//! `wasm_bindgen::function_table`, which are that same memory and the very
//! table this module's own indirect calls go through.

use crate::cpu::Cpu;
use std::sync::OnceLock;

/// What an emitted block is once it can be called: an address in whatever
/// sense this target has one. Zero means there is none.
pub type Entry = usize;

/// The signature [`super::emit::emit_block`] gives a block's `run` export: the
/// address of the [`Cpu`] in, and out the number of the block's leading
/// instructions it retired.
type Run = unsafe extern "C" fn(state: usize) -> u32;

/// The two operations an embedder supplies so emitted blocks can run.
///
/// Plain function pointers rather than a trait object: there is one
/// implementation per target, it is installed once at start-up, and this way
/// the type needs no allocation, no lifetime and no `dyn`.
#[derive(Debug, Clone, Copy)]
pub struct JitHost {
    /// Compile `code`, an emitted module, and answer the [`Entry`] its `run`
    /// export can be called at.
    ///
    /// Zero means it could not, whatever the reason, and the block it was for
    /// goes back to the interpreter for good. A failure here is never an
    /// error: an emitted block is an optimisation, and not having one is how
    /// the emulator ran before any of this existed.
    pub install: fn(code: &[u8]) -> Entry,
    /// Give an [`Entry`] back, because the block that held it has been
    /// dropped. It may be handed out again.
    pub release: fn(entry: Entry),
}

static HOST: OnceLock<JitHost> = OnceLock::new();

/// Name the [`JitHost`] this build's emitted code runs through.
///
/// The first call wins and answers `true`; a later one changes nothing and
/// answers `false`, because an entry point handed out by one host means
/// nothing to another. Without a call, no block is ever emitted and the
/// translator runs exactly as it did before.
pub fn set_jit_host(host: JitHost) -> bool {
    HOST.set(host).is_ok()
}

/// Whether there is anywhere to put emitted code, which is what decides
/// whether emitting it is worth the work.
#[inline]
pub(super) fn available() -> bool {
    HOST.get().is_some()
}

/// Compile `code` and answer where it can be called, or zero.
pub(super) fn install(code: &[u8]) -> Entry {
    match HOST.get() {
        Some(host) => (host.install)(code),
        None => 0,
    }
}

/// Give an installed entry point back.
pub(super) fn release(entry: Entry) {
    if let Some(host) = HOST.get() {
        (host.release)(entry);
    }
}

/// Run the block installed at `entry` against `state`, and report how many of
/// its leading instructions it retired.
///
/// # Safety
///
/// `entry` has to be one [`install`] answered and has not been [`release`]d,
/// and `state` the address of a live [`Cpu`]. What runs reads and writes that
/// `Cpu` and the guest pages its page table points at, and nothing else: it is
/// written by [`super::emit`], out of this same build, against
/// [`super::emit::Layout`]'s offsets into that very type.
#[inline(always)]
pub(super) unsafe fn enter(entry: Entry, state: *mut Cpu) -> u32 {
    // A function pointer is an address on the host and a table index on
    // wasm32, and `Entry` is whichever of those this build means, so the
    // transmute is between two spellings of one value rather than a
    // reinterpretation of anything.
    let run = std::mem::transmute::<Entry, Run>(entry);
    run(state as usize)
}
