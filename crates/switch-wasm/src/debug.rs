//! Debugging levers and traces.

use crate::log::{collect, push, PageLevel};
use crate::session;

/// Enable/disable the block translator; the browser has no `SWITCH_NO_JIT`.
#[no_mangle]
pub extern "C" fn switch_set_jit(handle: u32, enabled: u32) {
    session(handle).cpu.set_jit_enabled(enabled != 0);
}

/// Enable/disable the per-instruction disassembly trace.
#[no_mangle]
pub extern "C" fn switch_set_trace(handle: u32, enabled: u32) {
    let s = session(handle);
    s.cpu.trace_enabled = enabled != 0;
    if enabled == 0 {
        // What was traced before it was turned off still reaches the log.
        collect(&mut s.cpu);
    }
}

/// Log a full register snapshot.
#[no_mangle]
pub extern "C" fn switch_log_regs(handle: u32) {
    let s = session(handle);
    collect(&mut s.cpu);
    push(PageLevel::Log, &s.cpu.reg_dump());
}

/// Log one line per guest thread: state, what it is blocked on, and where it stopped.
#[no_mangle]
pub extern "C" fn switch_log_threads(handle: u32) {
    let s = session(handle);
    collect(&mut s.cpu);
    push(PageLevel::Log, &s.cpu.thread_dump());
}

/// Turn every diagnostic channel on (nonzero) or off.
#[no_mangle]
pub extern "C" fn switch_set_trace_channels(on: u32) {
    switch_core::trace::set_all(on != 0);
}
