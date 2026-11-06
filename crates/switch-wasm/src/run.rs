//! Running the guest and reading its state.

use wasm_bindgen::prelude::wasm_bindgen;

use crate::session;

/// Run up to `max_steps` instructions. Returns steps executed or -1 on error.
#[wasm_bindgen]
pub fn switch_run(max_steps: f64) -> f64 {
    let s = session();
    crate::input::exchange(&mut s.cpu);
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
    let steps = match s.cpu.run(max_steps as u64) {
        Ok(report) => report.steps as f64,
        Err(e) => {
            s.last_error = e.to_string();
            -1.0
        }
    };
    crate::input::release_if_seen(&mut s.cpu);
    #[cfg(all(target_arch = "wasm32", feature = "gpu"))]
    {
        crate::keyboard::announce(&mut s.cpu);
        crate::idb::persist::tick();
    }
    crate::audio::feed();
    steps
}

/// Tie the guest clock to real time: `elapsed` cycles have passed since the last call.
/// Returns how many cycles the clock is ahead, to wait out before the next slice.
#[wasm_bindgen]
pub fn switch_pace(elapsed: f64) -> f64 {
    session().cpu.pace_to_real_time(elapsed as u64) as f64
}

/// True if the machine halted via SVC #0.
#[wasm_bindgen]
pub fn switch_halted() -> bool {
    session().cpu.halted
}

/// The last `fatal:u` report for this program, or empty if it made none.
#[wasm_bindgen]
pub fn switch_guest_fatal() -> String {
    session().cpu.guest_fatal().unwrap_or("").to_owned()
}

/// Read register `idx` (0..=31; 31 = SP).
#[wasm_bindgen]
pub fn switch_get_reg(idx: u32) -> u64 {
    session().cpu.read_x(idx as u8)
}

/// Instructions actually retired.
#[wasm_bindgen]
pub fn switch_get_steps() -> f64 {
    session().cpu.steps as f64
}

/// The title's guest RAM backed by host storage, in bytes, even while an applet runs.
#[wasm_bindgen]
pub fn switch_guest_ram() -> f64 {
    session().cpu.title_backed() as f64
}
