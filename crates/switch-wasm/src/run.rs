//! Running the guest and reading its state.

use crate::{session, write_into};

/// Run up to `max_steps` instructions. Returns steps executed or -1 on error.
#[no_mangle]
pub extern "C" fn switch_run(handle: u32, max_steps: u64) -> i64 {
    let s = session(handle);
    match s.cpu.run(max_steps) {
        Ok(report) => report.steps as i64,
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// True if the machine halted via SVC #0.
#[no_mangle]
pub extern "C" fn switch_halted(handle: u32) -> i32 {
    session(handle).cpu.halted as i32
}

/// Copy the last `fatal:u` report for this program, if it made one.
#[no_mangle]
pub extern "C" fn switch_guest_fatal(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let fatal = session(handle).cpu.guest_fatal().unwrap_or("");
    write_into(buf, maxlen, fatal.as_bytes())
}

/// Copy accumulated console output into `buf` and clear it. Returns bytes copied.
#[no_mangle]
pub extern "C" fn switch_drain_output(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let n = s.cpu.out.len().min(maxlen as usize);
    if n > 0 && !buf.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(s.cpu.out.as_ptr(), buf, n);
        }
        s.cpu.out.drain(..n);
    }
    n as u32
}

/// Read register `idx` (0..=31; 31 = SP).
#[no_mangle]
pub extern "C" fn switch_get_reg(handle: u32, idx: u32) -> u64 {
    session(handle).cpu.read_x(idx as u8)
}

/// Current PC.
#[no_mangle]
pub extern "C" fn switch_get_pc(handle: u32) -> u32 {
    session(handle).cpu.get_pc()
}

/// The guest clock in CPU cycles. It idles forward when every thread is blocked;
/// [`switch_get_steps`] is the instruction count.
#[no_mangle]
pub extern "C" fn switch_get_cycles(handle: u32) -> u64 {
    session(handle).cpu.cycles
}

/// Instructions actually retired.
#[no_mangle]
pub extern "C" fn switch_get_steps(handle: u32) -> u64 {
    session(handle).cpu.steps
}

/// Guest RAM backed by host storage, in bytes.
#[no_mangle]
pub extern "C" fn switch_guest_ram(handle: u32) -> u64 {
    session(handle).cpu.mem.mapped_bytes()
}
