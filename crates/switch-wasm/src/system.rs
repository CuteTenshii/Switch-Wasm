//! System settings: operation mode, clock and battery.

use crate::session;

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
