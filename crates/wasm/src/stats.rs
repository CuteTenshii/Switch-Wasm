//! Translator, thread, GPU, memory and audio reports.

use std::sync::atomic::{AtomicBool, Ordering};

use switch_core::cpu::{AudioActivity, Pacing};
use switch_core::json;
use switch_core::trace::Level;
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
            .field("chained", stats.chained)
            .field("retired", stats.retired)
            .field("retiredEmitted", stats.retired_emitted)
            .field("dropped", stats.dropped)
            .field("retiredCold", stats.retired_cold)
            .field("retiredNever", stats.retired_never);
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

/// Whether the installed backend's loss has been logged; a newly installed one clears it.
pub(crate) static GPU_LOSS_LOGGED: AtomicBool = AtomicBool::new(false);

/// Whether the GPU backend has lost its device, logging why the first time. Cheap; polled
/// every slice.
#[wasm_bindgen]
pub fn switch_gpu_lost() -> bool {
    let s = session();
    let Some(why) = s.cpu.nv.gpu.renderer_lost().map(str::to_owned) else {
        return false;
    };
    if !GPU_LOSS_LOGGED.swap(true, Ordering::Relaxed) {
        s.cpu.log(Level::Error, &format!("[gpu] {why}"));
    }
    true
}

/// Each thread's share of the last second's instructions, as JSON.
#[wasm_bindgen]
pub fn switch_threads_json() -> String {
    crate::activity::threads_json(&session().activity)
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

/// The guest's mixers, the host ring and the clock's catch-ups, as JSON for the Audio
/// debug section. Counts are totals since they started; the page turns them into rates.
#[wasm_bindgen]
pub fn switch_audio_json() -> String {
    let cpu = &session().cpu;
    audio_json(&cpu.audio_activity(), cpu.pacing())
}

pub(crate) fn audio_json(audio: &AudioActivity, pacing: Pacing) -> String {
    json::object(|o| {
        o.object("clock", |c| {
            c.field("hz", pacing.clock_hz)
                .field("jumps", pacing.jumps)
                .field("skipped", pacing.skipped_cycles)
                .field("ahead", pacing.ahead_cycles);
        });
        o.object("ring", |r| {
            r.field("sampleRate", audio.sample_rate)
                .field("channels", audio.channels)
                .field("produced", audio.produced)
                .field("taken", audio.taken)
                .field("dropped", audio.dropped)
                .field("backlog", audio.backlog);
        });
        o.array("renderers", |a| {
            for r in &audio.renderers {
                a.object(|o| {
                    o.field("handle", r.handle)
                        .field("sampleRate", r.sample_rate)
                        .field("started", r.started)
                        .field("updates", r.updates)
                        .field("renderedFrames", r.rendered_frames)
                        .field("frameWakes", r.frame_wakes)
                        .field("frameWakesDropped", r.frame_wakes_dropped)
                        .field("voices", r.voices)
                        .field("voicesPlaying", r.voices_playing)
                        .field("sinkChannels", r.sink_channels)
                        .field("underruns", r.underruns);
                    o.array("voiceList", |v| {
                        for voice in &r.voice_list {
                            v.object(|o| {
                                o.field("id", voice.id)
                                    .field("format", voice.format)
                                    .field("sampleRate", voice.sample_rate)
                                    .field("pitch", voice.pitch)
                                    .field("channels", voice.channels)
                                    .field("volume", voice.volume)
                                    .field("playing", voice.playing)
                                    .field("queuedBuffers", voice.queued_buffers)
                                    .field("underruns", voice.underruns);
                            });
                        }
                    });
                });
            }
        });
        o.array("outputs", |a| {
            for out in &audio.outputs {
                a.object(|o| {
                    o.field("handle", out.handle)
                        .field("sampleRate", out.sample_rate)
                        .field("channels", out.channels)
                        .field("started", out.started)
                        .field("appendedFrames", out.appended_frames)
                        .field("pendingBuffers", out.pending_buffers)
                        .field("discardedFrames", out.discarded_frames);
                });
            }
        });
    })
}
