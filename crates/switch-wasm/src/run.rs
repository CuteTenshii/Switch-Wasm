//! Running the guest and reading its state.

use crate::{session, write_into};

/// Run up to `max_steps` instructions. Returns steps executed or -1 on error.
#[no_mangle]
pub extern "C" fn switch_run(max_steps: u64) -> i64 {
    let s = session();
    // The page's clock and time zone, read here rather than handed in before every slice.
    #[cfg(all(target_arch = "wasm32", feature = "jit"))]
    {
        let now = js_sys::Date::new_0();
        s.cpu
            .set_utc_offset(-(now.get_timezone_offset() as i32) * 60);
        s.cpu
            .set_unix_time((now.get_time() / 1000.0).floor() as i64);
    }
    #[cfg(all(target_arch = "wasm32", feature = "gpu"))]
    crate::idb::usage::refresh_if_due(s.cpu.take_storage_usage_wanted());
    let steps = match s.cpu.run(max_steps) {
        Ok(report) => report.steps as i64,
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    };
    crate::audio::feed();
    steps
}

/// Tie the guest clock to real time: `elapsed` cycles have passed since the last call.
/// Returns how many cycles the clock is ahead, to wait out before the next slice.
#[no_mangle]
pub extern "C" fn switch_pace(elapsed: u64) -> u64 {
    session().cpu.pace_to_real_time(elapsed)
}

/// True if the machine halted via SVC #0.
#[no_mangle]
pub extern "C" fn switch_halted() -> i32 {
    session().cpu.halted as i32
}

/// Copy the last `fatal:u` report for this program, if it made one.
#[no_mangle]
pub extern "C" fn switch_guest_fatal(buf: *mut u8, maxlen: u32) -> u32 {
    let fatal = session().cpu.guest_fatal().unwrap_or("");
    write_into(buf, maxlen, fatal.as_bytes())
}

/// Read register `idx` (0..=31; 31 = SP).
#[no_mangle]
pub extern "C" fn switch_get_reg(idx: u32) -> u64 {
    session().cpu.read_x(idx as u8)
}

/// Current PC.
#[no_mangle]
pub extern "C" fn switch_get_pc() -> u32 {
    session().cpu.get_pc()
}

/// The guest clock in CPU cycles. It idles forward when every thread is blocked;
/// [`switch_get_steps`] is the instruction count.
#[no_mangle]
pub extern "C" fn switch_get_cycles() -> u64 {
    session().cpu.cycles
}

/// Instructions actually retired.
#[no_mangle]
pub extern "C" fn switch_get_steps() -> u64 {
    session().cpu.steps
}

/// The title's guest RAM backed by host storage, in bytes, even while an applet runs.
#[no_mangle]
pub extern "C" fn switch_guest_ram() -> u64 {
    session().cpu.title_backed()
}
