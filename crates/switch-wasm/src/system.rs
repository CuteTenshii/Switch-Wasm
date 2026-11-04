//! System settings: font, operation mode, clock and battery.

use crate::session;

/// Set the shared system font `pl:u` serves (TTF/OTF bytes), before `plInitialize`.
/// Returns the bytes taken.
#[no_mangle]
pub extern "C" fn switch_load_font(ptr: *const u8, len: u32) -> u32 {
    let s = session();
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    s.cpu.set_shared_font(data.to_vec());
    s.cpu.shared_font_len() as u32
}

/// Dock or undock the console (0 handheld). Queues the AM messages titles react to.
#[no_mangle]
pub extern "C" fn switch_set_operation_mode(docked: u32) {
    let mode = if docked == 0 {
        switch_core::cpu::OperationMode::Handheld
    } else {
        switch_core::cpu::OperationMode::Docked
    };
    session().cpu.set_operation_mode(mode);
}

/// Set the battery level `psm` reports.
#[no_mangle]
pub extern "C" fn switch_set_battery(percent: u32, charging: u32) {
    session()
        .cpu
        .set_battery(percent.min(100) as u8, charging != 0);
}
