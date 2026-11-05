//! Translator, GPU and memory reports.

use switch_core::json;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::session;

/// What the translator has been doing, as JSON.
#[wasm_bindgen]
pub fn switch_jit_stats_json() -> String {
    let s = session();
    let stats = s.cpu.jit_stats();
    json::object(|o| {
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
    })
}

/// The GPU backend's report as JSON, or `{}` for the software rasterizer.
#[wasm_bindgen]
pub fn switch_gpu_report_json() -> String {
    let s = session();
    // The frame count comes from the core; the backend only sees clears and draws.
    let gpu = &s.cpu.nv.gpu;
    json::object(|o| {
        gpu.write_renderer_report(o);
        o.field("frames", gpu.frames);
    })
}

/// Whether the GPU backend has lost its device. Cheap; polled every slice.
#[no_mangle]
pub extern "C" fn switch_gpu_lost() -> u32 {
    let s = session();
    u32::from(s.cpu.nv.gpu.renderer_lost())
}

/// The title's backed guest memory by region, as JSON. The page adds the heap size itself.
#[wasm_bindgen]
pub fn switch_memory_json() -> String {
    let memory = session().cpu.title_memory();
    json::object(|o| {
        o.field("backed", memory.backed).field("cap", memory.cap);
        o.array("regions", |a| {
            for &(region, (start, end), bytes) in &memory.regions {
                a.object(|r| {
                    r.field("key", region.key())
                        .field("start", start)
                        .field("end", end)
                        .field("bytes", bytes);
                });
            }
        });
        o.field("elsewhere", memory.elsewhere)
            .field("allProcesses", memory.all_processes);
    })
}
