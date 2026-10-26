//! Translator and GPU reports.

use crate::{session, write_into};

/// What the translator has been doing, as JSON.
#[no_mangle]
pub extern "C" fn switch_jit_stats_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let stats = s.cpu.jit_stats();
    let json = format!(
        "{{\"enabled\":{},\"blocks\":{},\"translated\":{},\"executed\":{},\"linked\":{},\"invalidated\":{},\"interpreted\":{},\"interpretedGroups\":[{}],\"emitted\":{},\"enteredEmitted\":{},\"chained\":{}}}",
        s.cpu.jit_enabled(),
        stats.blocks,
        stats.translated,
        stats.executed,
        stats.linked,
        stats.invalidated,
        stats.interpreted,
        stats.interpreted_groups
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(","),
        stats.emitted,
        stats.entered_emitted,
        stats.chained
    );
    write_into(buf, maxlen, json.as_bytes())
}

/// The GPU backend's report as JSON, or `{}` for the software rasterizer.
#[no_mangle]
pub extern "C" fn switch_gpu_report_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    // The frame count comes from the core; the backend only sees clears and draws.
    let frames = s.cpu.nv.gpu.frames;
    let json = s.cpu.nv.gpu.renderer_report();
    let json = match json.strip_suffix('}') {
        Some(body) if body.len() > 1 => format!("{body},\"frames\":{frames}}}"),
        _ => format!("{{\"frames\":{frames}}}"),
    };
    write_into(buf, maxlen, json.as_bytes())
}

/// Whether the GPU backend has lost its device. Cheap; polled every slice.
#[no_mangle]
pub extern "C" fn switch_gpu_lost(handle: u32) -> u32 {
    let s = session(handle);
    u32::from(s.cpu.nv.gpu.renderer_lost())
}
