//! Debugging levers and traces.

use wasm_bindgen::prelude::wasm_bindgen;

use crate::log::collect;
use crate::session;

/// Enable/disable the block translator; the browser has no `SWITCH_NO_JIT`.
#[wasm_bindgen]
pub fn switch_set_jit(enabled: bool) {
    session().cpu.set_jit_enabled(enabled);
}

/// Enable/disable the per-instruction disassembly trace.
#[wasm_bindgen]
pub fn switch_set_trace(enabled: bool) {
    let s = session();
    s.cpu.trace_enabled = enabled;
    if !enabled {
        // What was traced before it was turned off still reaches the log.
        collect(&mut s.cpu);
    }
}

/// A full register snapshot. Traces queued so far reach the log first.
#[wasm_bindgen]
pub fn switch_reg_dump() -> String {
    let s = session();
    collect(&mut s.cpu);
    s.cpu.reg_dump()
}

/// One line per guest thread: state, what it is blocked on, and where it stopped. Traces
/// queued so far reach the log first.
#[wasm_bindgen]
pub fn switch_thread_dump() -> String {
    let s = session();
    collect(&mut s.cpu);
    s.cpu.thread_dump()
}

/// Turn every diagnostic channel on or off.
#[wasm_bindgen]
pub fn switch_set_trace_channels(on: bool) {
    switch_core::trace::set_all(on);
}
