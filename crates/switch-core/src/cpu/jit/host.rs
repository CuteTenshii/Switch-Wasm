//! The embedder hooks that compile emitted wasm and make it callable.
//!
//! [`JitHost::install`] returns an [`Entry`]: a function-table index on wasm32,
//! a real function address on the host. The embedder compiles the module with
//! the emulator's memory imported as `e`.`m` and puts `run` in the function table.

use crate::cpu::Cpu;
use std::sync::OnceLock;

/// A callable block address; zero means none.
pub type Entry = usize;

/// A block's `run` export: takes the [`Cpu`] address, returns instructions retired.
type Run = unsafe extern "C" fn(state: usize) -> u32;

/// The two operations an embedder supplies so emitted blocks can run.
#[derive(Debug, Clone, Copy)]
pub struct JitHost {
    /// Compile an emitted module and return its `run` [`Entry`], or zero on any
    /// failure, leaving the block to the interpreter.
    pub install: fn(code: &[u8]) -> Entry,
    /// Free an [`Entry`] for reuse.
    pub release: fn(entry: Entry),
    /// Whether the engine has tail calls, so compiled blocks may jump into each
    /// other through the function table, imported as `e`.`t`.
    pub tail_calls: bool,
}

static HOST: OnceLock<JitHost> = OnceLock::new();

/// The first call wins and returns `true`; without one, nothing is emitted.
pub fn set_jit_host(host: JitHost) -> bool {
    HOST.set(host).is_ok()
}

#[inline]
pub(super) fn available() -> bool {
    HOST.get().is_some()
}

/// Whether blocks are emitted to jump straight into each other.
#[inline]
pub(super) fn chains() -> bool {
    HOST.get().is_some_and(|host| host.tail_calls)
}

pub(super) fn install(code: &[u8]) -> Entry {
    match HOST.get() {
        Some(host) => (host.install)(code),
        None => 0,
    }
}

pub(super) fn release(entry: Entry) {
    if let Some(host) = HOST.get() {
        (host.release)(entry);
    }
}

/// Run the block at `entry` and return how many leading instructions it retired.
///
/// # Safety
///
/// `entry` must come from [`install`] and not be [`release`]d, and `state` must
/// be the address of a live [`Cpu`] matching [`super::emit::Layout`].
#[inline(always)]
pub(super) unsafe fn enter(entry: Entry, state: *mut Cpu) -> u32 {
    // `Entry` is this target's function-pointer representation.
    let run = std::mem::transmute::<Entry, Run>(entry);
    run(state as usize)
}
