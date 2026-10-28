//! Translator and GPU reports.

use crate::{session, write_into};
use switch_core::json;

/// What the translator has been doing, as JSON.
#[no_mangle]
pub extern "C" fn switch_jit_stats_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let stats = s.cpu.jit_stats();
    let json = json::object(|o| {
        o.field("enabled", s.cpu.jit_enabled())
            .field("blocks", stats.blocks)
            .field("translated", stats.translated)
            .field("executed", stats.executed)
            .field("linked", stats.linked)
            .field("invalidated", stats.invalidated)
            .field("interpreted", stats.interpreted);
        o.array("interpretedGroups", |a| {
            a.extend(stats.interpreted_groups);
        });
        o.field("emitted", stats.emitted)
            .field("enteredEmitted", stats.entered_emitted)
            .field("chained", stats.chained);
    });
    write_into(buf, maxlen, json.as_bytes())
}

/// The GPU backend's report as JSON, or `{}` for the software rasterizer.
#[no_mangle]
pub extern "C" fn switch_gpu_report_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    // The frame count comes from the core; the backend only sees clears and draws.
    let gpu = &s.cpu.nv.gpu;
    let json = json::object(|o| {
        gpu.write_renderer_report(o);
        o.field("frames", gpu.frames);
    });
    write_into(buf, maxlen, json.as_bytes())
}

/// Whether the GPU backend has lost its device. Cheap; polled every slice.
#[no_mangle]
pub extern "C" fn switch_gpu_lost(handle: u32) -> u32 {
    let s = session(handle);
    u32::from(s.cpu.nv.gpu.renderer_lost())
}
