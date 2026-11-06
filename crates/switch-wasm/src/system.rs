//! System settings: operation mode, clock and battery.

use wasm_bindgen::prelude::wasm_bindgen;

use crate::session;

/// Dock or undock the console. Queues the AM messages titles react to.
#[wasm_bindgen]
pub fn switch_set_operation_mode(docked: bool) {
    let mode = if docked {
        switch_core::cpu::OperationMode::Docked
    } else {
        switch_core::cpu::OperationMode::Handheld
    };
    session().cpu.set_operation_mode(mode);
}

/// Set the battery level `psm` reports.
#[wasm_bindgen]
pub fn switch_set_battery(percent: u32, charging: bool) {
    session().cpu.set_battery(percent.min(100) as u8, charging);
}
