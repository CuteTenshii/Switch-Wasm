//! System settings: operation mode, clock and battery.

use wasm_bindgen::prelude::wasm_bindgen;

use switch_core::cpu::OperationMode;
use switch_core::gpu::renderer::DeviceTiming;

use crate::log::{push, PageLevel};
use crate::session;

/// Dock or undock the console, and say what that changes. Queues the AM messages titles react to.
#[wasm_bindgen]
pub fn switch_set_operation_mode(docked: bool) {
    let mode = if docked {
        OperationMode::Docked
    } else {
        OperationMode::Handheld
    };
    session().cpu.set_operation_mode(mode);
    let (_, height) = mode.display_size();
    let (name, rest) = match mode {
        OperationMode::Docked => ("Docked", "boost clocks, no touchscreen"),
        OperationMode::Handheld => ("Handheld", "normal clocks, touchscreen"),
    };
    push(PageLevel::Log, &format!("{name} - {height}p, {rest}."));
}

/// Draw every frame on the CPU rasterizer (or give the frames back to the device), so the two
/// can be compared on the same scene.
#[wasm_bindgen]
pub fn switch_force_software(on: bool) {
    session().cpu.nv.gpu.force_software(on);
}

/// Measure how long the device takes over each pass (`"passes"`), each draw in a pass of its
/// own (`"draws"`), or nothing (anything else), for the GPU report.
#[wasm_bindgen]
pub fn switch_set_device_timing(mode: &str) {
    let timing = match mode {
        "passes" => DeviceTiming::Passes,
        "draws" => DeviceTiming::Draws,
        _ => DeviceTiming::Off,
    };
    session().cpu.nv.gpu.set_device_timing(timing);
}

/// Set the battery level `psm` reports.
#[wasm_bindgen]
pub fn switch_set_battery(percent: u32, charging: bool) {
    session().cpu.set_battery(percent.min(100) as u8, charging);
}
