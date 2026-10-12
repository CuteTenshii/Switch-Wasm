//! Debugging levers and traces.

use crate::{session, write_into};

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
        s.cpu.trace.clear();
    }
}

/// Copy the debug trace (disassembly and fault context) into `buf` and clear it.
#[no_mangle]
pub extern "C" fn switch_drain_trace(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    // Fold in traces from components with no `Cpu`.
    s.cpu.absorb_traces();
    let n = s.cpu.trace.len().min(maxlen as usize);
    if n > 0 && !buf.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(s.cpu.trace.as_ptr(), buf, n);
        }
        s.cpu.trace.drain(..n);
    }
    n as u32
}

/// Write a full register snapshot as text into `buf`. Returns bytes written.
#[no_mangle]
pub extern "C" fn switch_dump_regs(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let dump = s.cpu.reg_dump();
    write_into(buf, maxlen, dump.as_bytes())
}

/// One line per guest thread: state, what it is blocked on, and where it stopped.
#[no_mangle]
pub extern "C" fn switch_thread_dump(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    write_into(buf, maxlen, s.cpu.thread_dump().as_bytes())
}

/// Turn every diagnostic channel on (nonzero) or off.
#[no_mangle]
pub extern "C" fn switch_set_trace_channels(on: u32) {
    switch_core::trace::set_all(on != 0);
}
