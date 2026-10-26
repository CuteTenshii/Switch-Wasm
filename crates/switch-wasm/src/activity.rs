//! The once-a-second activity report in the page's log: frames, surfaces, threads,
//! files, path operations, audio, and problems (unanswered requests, refused draws,
//! failed ioctls, idle threads, controller styles, memory growth).

use std::collections::HashMap;

use switch_core::cpu::{AudioActivity, Cpu, GapKind, ThreadReport};
use switch_core::gpu::activity::{Kind, Tally};

use crate::log::{push, PageLevel};
use crate::session;

/// Counters that run from boot, as the last report saw them.
#[derive(Default)]
struct Counters {
    frames: u64,
    draws: u64,
    draws_skipped: u64,
    clears: u64,
    clears_elided: u64,
    copies: u64,
    dispatches: u64,
    fs_failures: u64,
    audio: AudioActivity,
    /// `HidNpadStyleTag` bits: what the title accepts (0 before it says) and the pad's style.
    input: (u32, u32),
}

impl Counters {
    fn read(cpu: &Cpu) -> Counters {
        let gpu = &cpu.nv.gpu;
        Counters {
            frames: gpu.frames,
            draws: gpu.stats.draws,
            draws_skipped: gpu.stats.draws_skipped,
            clears: gpu.stats.clears,
            clears_elided: gpu.stats.clears_elided,
            copies: gpu.stats.copies,
            dispatches: gpu.stats.dispatches,
            fs_failures: cpu.fs_activity.failures,
            audio: cpu.audio_activity(),
            input: cpu.npad_styles(),
        }
    }
}

/// What the last report said, so the next one only says what changed.
#[derive(Default)]
pub(crate) struct Reporter {
    previous: Counters,
    /// Each thread's last reported state, so unchanged threads aren't repeated.
    thread_states: HashMap<usize, String>,
    /// Idle seconds at each thread's last warning.
    idle_warned: HashMap<usize, u64>,
    /// Guest RAM and WebAssembly heap at the last memory line.
    memory: (u64, u64),
}

/// Write what changed since the last report into the log. `seconds` is the time since
/// it, or 0 for the first, which then gives no frame rate.
#[no_mangle]
pub extern "C" fn switch_log_activity(handle: u32, seconds: f64) {
    let s = session(handle);
    report(&mut s.cpu, &mut s.activity, seconds);
}

fn report(cpu: &mut Cpu, reporter: &mut Reporter, seconds: f64) {
    let now = Counters::read(cpu);
    let before = std::mem::take(&mut reporter.previous);
    let mut activity = cpu.nv.gpu.take_activity();
    gpu(&before, &now, seconds, activity.take());
    let threads = threads(cpu, reporter);
    fs(cpu, &before, &now);
    audio(&before.audio, &now.audio);
    problems(cpu, activity.take_refusals());
    idle_threads(reporter, &threads);
    input(before.input, now.input);
    memory(cpu, reporter);
    reporter.previous = now;
}

fn count(n: u64, noun: &str) -> String {
    count_as(n, noun, &format!("{noun}s"))
}

fn count_as(n: u64, noun: &str, plural: &str) -> String {
    format!("{n} {}", if n == 1 { noun } else { plural })
}

/// Only the parts present, so a quiet second prints nothing.
fn joined(parts: &[Option<String>]) -> String {
    parts
        .iter()
        .flatten()
        .cloned()
        .collect::<Vec<_>>()
        .join(", ")
}

/// A byte size to two decimals.
fn size(n: u64) -> String {
    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    if n >= GIB {
        format!("{:.2} GiB", n as f64 / GIB as f64)
    } else if n >= MIB {
        format!("{:.2} MiB", n as f64 / MIB as f64)
    } else if n >= KIB {
        format!("{:.1} KiB", n as f64 / KIB as f64)
    } else {
        format!("{n} B")
    }
}

/// A large count as 12.3M; M/B/T rather than SI prefixes, so it doesn't read as bytes.
fn large(n: u64) -> String {
    let f = n as f64;
    if f >= 1e12 {
        format!("{:.2}T", f / 1e12)
    } else if f >= 1e9 {
        format!("{:.2}B", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.1}M", f / 1e6)
    } else {
        let digits = n.to_string();
        let mut out = String::with_capacity(digits.len() + digits.len() / 3);
        for (i, c) in digits.chars().enumerate() {
            if i > 0 && (digits.len() - i).is_multiple_of(3) {
                out.push(',');
            }
            out.push(c);
        }
        out
    }
}

fn hex(n: u32) -> String {
    format!("{n:#x}")
}

fn gpu(before: &Counters, now: &Counters, seconds: f64, surfaces: Vec<(Kind, Tally)>) {
    let delta = |now: u64, before: u64| now.saturating_sub(before);
    let frames = delta(now.frames, before.frames);
    let draws = delta(now.draws, before.draws);
    let skipped = delta(now.draws_skipped, before.draws_skipped);
    let clears = delta(now.clears, before.clears);
    let elided = delta(now.clears_elided, before.clears_elided);
    let copies = delta(now.copies, before.copies);
    let dispatches = delta(now.dispatches, before.dispatches);
    let rate = if seconds > 0.0 {
        format!(" ({:.1}/s)", frames as f64 / seconds)
    } else {
        String::new()
    };
    let elided = if elided > 0 {
        format!(" ({elided} elided)")
    } else {
        String::new()
    };
    let line = joined(&[
        (frames > 0).then(|| count(frames, "frame") + &rate),
        (draws > 0).then(|| count(draws, "draw")),
        // A skipped draw is a hole in the frame.
        (skipped > 0).then(|| count(skipped, "draw") + " skipped"),
        (clears > 0).then(|| count(clears, "clear") + &elided),
        (copies > 0).then(|| count_as(copies, "copy", "copies")),
        (dispatches > 0).then(|| count_as(dispatches, "compute dispatch", "compute dispatches")),
    ]);
    if !line.is_empty() {
        let level = if skipped > 0 {
            PageLevel::Warn
        } else {
            PageLevel::Debug
        };
        push(level, &format!("[gpu] {line}"));
    }
    for (kind, tally) in surfaces {
        let (n, amount, label) = (tally.count, tally.amount, &tally.label);
        let line = match kind {
            Kind::Draw => {
                let skipped = if tally.failed > 0 {
                    format!(", {} skipped", tally.failed)
                } else {
                    String::new()
                };
                format!(
                    "draws into {label}: {}, {}{skipped}",
                    count(n, "draw"),
                    count_as(amount, "vertex", "vertices")
                )
            }
            Kind::Clear => format!("clears of {label}: {n}"),
            Kind::Copy => format!(
                "copy {label}: {} ({})",
                count_as(n, "copy", "copies"),
                size(amount)
            ),
            Kind::Upload => format!("inline uploads to {label}: {n} ({})", size(amount)),
            Kind::Blit => format!(
                "2D blit {label}: {} ({})",
                count(n, "blit"),
                count(amount, "pixel")
            ),
            Kind::Present => format!("presented {label}: {}", count(n, "frame")),
            // Only ever a refusal; see `problems`.
            Kind::Dispatch => continue,
        };
        let level = if tally.failed > 0 {
            PageLevel::Warn
        } else {
            PageLevel::Debug
        };
        push(level, &format!("[gpu] {line}"));
    }
}

/// Thread lifecycle events, then each thread that ran or changed state.
fn threads(cpu: &mut Cpu, reporter: &mut Reporter) -> Vec<ThreadReport> {
    let (threads, events, dropped) = cpu.take_thread_report();
    for line in events {
        push(PageLevel::Debug, &format!("[thread] {line}"));
    }
    if dropped > 0 {
        push(
            PageLevel::Debug,
            &format!("[thread] ...and {dropped} more thread events than could be listed"),
        );
    }
    let total: u64 = threads.iter().map(|t| t.ran).sum();
    for thread in &threads {
        let changed = reporter.thread_states.get(&thread.index) != Some(&thread.state);
        reporter
            .thread_states
            .insert(thread.index, thread.state.clone());
        if thread.ran == 0 && !changed {
            continue;
        }
        let ran = if thread.ran > 0 {
            let share = if total > 0 {
                format!(
                    " ({}%)",
                    (thread.ran as f64 / total as f64 * 100.0).round() as u64
                )
            } else {
                String::new()
            };
            format!(
                "{} instructions{share}, {}",
                large(thread.ran),
                count_as(thread.switches, "switch", "switches")
            )
        } else {
            "did not run".to_owned()
        };
        let who = match &thread.name {
            Some(name) => format!("{name}, via {}", thread.entry),
            None => thread.entry.clone(),
        };
        push(
            PageLevel::Debug,
            &format!(
                "[thread] {}{} ({who}, handle {:x}, priority {}): {ran}; {}; {}",
                thread.index,
                if thread.running { "*" } else { "" },
                thread.handle,
                thread.priority,
                thread.state,
                thread.at
            ),
        );
    }
    threads
}

fn fs(cpu: &mut Cpu, before: &Counters, now: &Counters) {
    for (name, io) in cpu.fs_activity.take_files() {
        let line = joined(&[
            (io.reads > 0)
                .then(|| format!("{} ({})", count(io.reads, "read"), size(io.read_bytes))),
            (io.writes > 0)
                .then(|| format!("{} ({})", count(io.writes, "write"), size(io.write_bytes))),
        ]);
        if !line.is_empty() {
            push(PageLevel::Debug, &format!("[fs] {name}: {line}"));
        }
    }
    let (journal, dropped) = cpu.fs_activity.take_journal();
    for entry in journal {
        push(PageLevel::Debug, &format!("[fs] {entry}"));
    }
    if dropped > 0 {
        push(
            PageLevel::Debug,
            &format!("[fs] ...and {dropped} more path operations than could be listed"),
        );
    }
    let failures = now.fs_failures.saturating_sub(before.fs_failures);
    if failures > 0 {
        push(
            PageLevel::Debug,
            &format!("[fs] {failures} requests failed"),
        );
    }
}

/// A frame count as time at `rate`.
fn duration(frames: f64, rate: u32) -> String {
    if rate == 0 {
        return format!(
            "{frames} {}",
            if frames == 1.0 { "frame" } else { "frames" }
        );
    }
    let ms = frames / f64::from(rate) * 1000.0;
    if ms >= 1000.0 {
        format!("{:.1} s", ms / 1000.0)
    } else {
        format!("{} ms", ms.round())
    }
}

fn layout(channels: u32) -> String {
    match channels {
        1 => "mono".to_owned(),
        2 => "stereo".to_owned(),
        n => format!("{n} channels"),
    }
}

/// Audio devices and renderers that changed, and samples between guest and page.
fn audio(before: &AudioActivity, now: &AudioActivity) {
    let outputs_before: HashMap<u64, _> = before.outputs.iter().map(|o| (o.handle, o)).collect();
    for o in &now.outputs {
        let was = outputs_before.get(&o.handle).copied();
        let name = format!("audout {:x}", o.handle);
        if was.is_none() {
            push(
                PageLevel::Log,
                &format!(
                    "[audio] {name} opened: {} Hz {}",
                    o.sample_rate,
                    layout(o.channels)
                ),
            );
        }
        let since = |now: u64, then: Option<u64>| now.saturating_sub(then.unwrap_or(0));
        let buffers = since(o.appended_buffers, was.map(|w| w.appended_buffers));
        let frames = since(o.appended_frames, was.map(|w| w.appended_frames));
        let released = since(o.released_buffers, was.map(|w| w.released_buffers));
        let discarded = since(o.discarded_frames, was.map(|w| w.discarded_frames));
        let unplayable = since(o.unplayable_buffers, was.map(|w| w.unplayable_buffers));
        let volume = if o.volume.is_finite() { o.volume } else { 0.0 };
        let state_changed = was.is_none_or(|w| w.started != o.started || w.volume != o.volume);
        let line = joined(&[
            (buffers > 0).then(|| {
                format!(
                    "{} ({}) appended",
                    count(buffers, "buffer"),
                    duration(frames as f64, o.sample_rate)
                )
            }),
            (released > 0).then(|| format!("{released} released")),
            (o.pending_buffers > 0 && buffers > 0)
                .then(|| format!("{} pending", o.pending_buffers)),
        ]);
        if !line.is_empty() || state_changed {
            let mut state = if o.started { "started" } else { "stopped" }.to_owned();
            if volume != 1.0 {
                state += &format!(", volume {}%", (f64::from(volume) * 100.0).round());
            }
            let line = if line.is_empty() {
                String::new()
            } else {
                format!(": {line}")
            };
            push(PageLevel::Debug, &format!("[audio] {name} {state}{line}"));
        }
        if discarded > 0 {
            push(
                PageLevel::Warn,
                &format!(
                    "[audio] {name}: {} appended while stopped, never played",
                    duration(discarded as f64, o.sample_rate)
                ),
            );
        }
        if unplayable > 0 {
            push(
                PageLevel::Warn,
                &format!(
                    "[audio] {name}: {} described outside itself, not played",
                    count(unplayable, "buffer")
                ),
            );
        }
    }
    for gone in &before.outputs {
        if !now.outputs.iter().any(|o| o.handle == gone.handle) {
            push(
                PageLevel::Log,
                &format!("[audio] audout {:x} closed", gone.handle),
            );
        }
    }

    let renderers_before: HashMap<u64, _> =
        before.renderers.iter().map(|r| (r.handle, r)).collect();
    for r in &now.renderers {
        let was = renderers_before.get(&r.handle).copied();
        let name = format!("renderer {:x}", r.handle);
        if was.is_none() {
            push(
                PageLevel::Log,
                &format!(
                    "[audio] {name} opened: {} Hz, {}",
                    r.sample_rate,
                    count(u64::from(r.voices), "voice")
                ),
            );
        }
        let frames = r
            .rendered_frames
            .saturating_sub(was.map_or(0, |w| w.rendered_frames));
        let updates = r.updates.saturating_sub(was.map_or(0, |w| w.updates));
        let state_changed = was.is_none_or(|w| {
            w.started != r.started
                || w.sink_channels != r.sink_channels
                || w.voices_playing != r.voices_playing
        });
        if frames > 0 || updates > 0 || state_changed {
            let sink = if r.sink_channels > 0 {
                layout(r.sink_channels)
            } else {
                "no playable sink".to_owned()
            };
            let line = joined(&[
                (frames > 0).then(|| format!("{} rendered", count(frames, "frame"))),
                (updates > 0).then(|| count(updates, "update")),
                Some(format!(
                    "{} of {} playing",
                    r.voices_playing,
                    count(u64::from(r.voices), "voice")
                )),
                Some(sink),
            ]);
            // Frames mixed into nothing.
            let silent = r.started && frames > 0 && r.sink_channels == 0;
            let level = if silent {
                PageLevel::Warn
            } else {
                PageLevel::Debug
            };
            let state = if r.started { "started" } else { "stopped" };
            push(level, &format!("[audio] {name} {state}: {line}"));
        }
    }
    for gone in &before.renderers {
        if !now.renderers.iter().any(|r| r.handle == gone.handle) {
            push(
                PageLevel::Log,
                &format!("[audio] renderer {:x} closed", gone.handle),
            );
        }
    }

    let produced = now.produced.saturating_sub(before.produced);
    let taken = now.taken.saturating_sub(before.taken);
    let dropped = now.dropped.saturating_sub(before.dropped);
    let per_frame = f64::from(now.channels.max(1));
    if produced > 0 || taken > 0 || dropped > 0 {
        let line = joined(&[
            Some(format!(
                "{} produced",
                duration(produced as f64 / per_frame, now.sample_rate)
            )),
            Some(format!(
                "{} taken by the page",
                duration(taken as f64 / per_frame, now.sample_rate)
            )),
            (now.backlog > 0).then(|| {
                format!(
                    "{} queued",
                    duration(now.backlog as f64 / per_frame, now.sample_rate)
                )
            }),
        ]);
        push(
            PageLevel::Debug,
            &format!(
                "[audio] {} Hz {}: {line}",
                now.sample_rate,
                layout(now.channels)
            ),
        );
    }
    if dropped > 0 {
        push(
            PageLevel::Warn,
            &format!(
                "[audio] {} dropped: the page fell a second behind",
                duration(dropped as f64 / per_frame, now.sample_rate)
            ),
        );
    }
}

/// nvdrv's error codes, as `nvdrv.rs` names them.
fn nv_error(error: u32) -> String {
    match error {
        1 => "not implemented".to_owned(),
        2 => "not supported".to_owned(),
        4 => "bad parameter".to_owned(),
        6 => "insufficient memory".to_owned(),
        8 => "invalid state".to_owned(),
        0x30006 => "config variable not found".to_owned(),
        _ => format!("error {}", hex(error)),
    }
}

fn problems(cpu: &mut Cpu, refusals: Vec<(Kind, String, u64)>) {
    let gaps = cpu.take_service_gaps();
    // Stubs answering every frame are ordinary; refusals are what stall a title.
    let asked = |kinds: &[GapKind]| -> Vec<String> {
        gaps.iter()
            .filter(|gap| kinds.contains(&gap.kind))
            .map(|gap| {
                let command = gap.command.map_or(String::new(), |c| format!(" cmd {c}"));
                let kind = match gap.kind {
                    GapKind::Refused => "refused",
                    GapKind::Missing => "no such service",
                    GapKind::Stub => "stubbed",
                    GapKind::Ioctl => "no handler",
                };
                format!("{}{command} ({kind}) x{}", gap.name, gap.calls)
            })
            .collect()
    };
    let failing = asked(&[GapKind::Refused, GapKind::Missing, GapKind::Ioctl]);
    if !failing.is_empty() {
        push(
            PageLevel::Warn,
            &format!("[ipc] unanswered: {}", failing.join(", ")),
        );
    }
    let stubbed = asked(&[GapKind::Stub]);
    if !stubbed.is_empty() {
        push(
            PageLevel::Debug,
            &format!("[ipc] answered by stubs: {}", stubbed.join(", ")),
        );
    }
    for (kind, reason, times) in refusals {
        let what = match kind {
            Kind::Dispatch => count_as(times, "compute dispatch", "compute dispatches"),
            _ => count(times, "draw"),
        };
        push(PageLevel::Warn, &format!("[gpu] {what} refused: {reason}"));
    }
    for (node, request, error, calls) in cpu.take_nv_errors() {
        push(
            PageLevel::Warn,
            &format!(
                "[nv] {node} ioctl {} (nr {}) failed: {} x{calls}",
                hex(request),
                hex(request & 0xff),
                nv_error(error)
            ),
        );
    }
}

/// The idle seconds a thread is named at after `after`: 5, 30, then every minute.
fn next_idle_warning(after: u64) -> u64 {
    [5, 30]
        .into_iter()
        .find(|&s| s > after)
        .unwrap_or((after / 60 + 1) * 60)
}

/// Name threads waiting a long time without work, and what they wait on.
fn idle_threads(reporter: &mut Reporter, threads: &[ThreadReport]) {
    for thread in threads {
        let idle = thread.idle_ms / 1000;
        if idle
            < reporter
                .idle_warned
                .get(&thread.index)
                .copied()
                .unwrap_or(0)
        {
            reporter.idle_warned.remove(&thread.index);
        }
        let warned = reporter
            .idle_warned
            .get(&thread.index)
            .copied()
            .unwrap_or(0);
        if !thread.state.starts_with("waiting") || idle < next_idle_warning(warned) {
            continue;
        }
        reporter.idle_warned.insert(thread.index, idle);
        let who = match &thread.name {
            Some(name) => format!("{} ({name})", thread.index),
            None => format!("{} ({})", thread.index, thread.entry),
        };
        push(
            PageLevel::Debug,
            &format!(
                "[thread] {who} has done no work for {idle} s of guest time; {}",
                thread.state
            ),
        );
    }
}

/// `HidNpadStyleTag` bits, as a player would name the controller.
const NPAD_STYLES: [&str; 12] = [
    "Pro Controller",
    "handheld",
    "Joy-Con pair",
    "left Joy-Con",
    "right Joy-Con",
    "GameCube controller",
    "Poke Ball Plus",
    "NES controller",
    "handheld NES controllers",
    "SNES controller",
    "N64 controller",
    "Sega Genesis controller",
];

fn style_names(bits: u32) -> String {
    let names: Vec<&str> = NPAD_STYLES
        .iter()
        .enumerate()
        .filter(|&(bit, _)| bits & (1 << bit) != 0)
        .map(|(_, &name)| name)
        .collect();
    if names.is_empty() {
        "nothing this console can present".to_owned()
    } else {
        names.join(", ")
    }
}

fn input(before: (u32, u32), now: (u32, u32)) {
    let (supported, presented) = now;
    if now == before || supported == 0 {
        return;
    }
    // The handheld slot carries every button, so accepting it means full input.
    let handheld = if supported & (1 << 1) != 0 {
        ", and the handheld slot has every button"
    } else {
        ""
    };
    push(
        PageLevel::Log,
        &format!(
            "[input] the title accepts {}; player 1 is presented as {}{handheld}",
            style_names(supported),
            style_names(presented)
        ),
    );
}

/// The WebAssembly heap in bytes, or 0 off wasm32.
fn heap_bytes() -> u64 {
    #[cfg(target_arch = "wasm32")]
    return core::arch::wasm32::memory_size(0) as u64 * 65536;
    #[cfg(not(target_arch = "wasm32"))]
    0
}

/// Memory growth worth a line, and a warning near wasm32's 4 GiB.
fn memory(cpu: &Cpu, reporter: &mut Reporter) {
    const STEP: u64 = 64 << 20;
    const WARN_AT: u64 = 3584 << 20;
    let guest = cpu.mem.mapped_bytes();
    let heap = heap_bytes();
    let (guest_then, heap_then) = reporter.memory;
    if guest.abs_diff(guest_then) < STEP && heap.abs_diff(heap_then) < STEP {
        return;
    }
    reporter.memory = (guest, heap);
    let near = heap >= WARN_AT;
    let (level, warning) = if near {
        (
            PageLevel::Warn,
            ": close to the limit, the next large allocation may fail",
        )
    } else {
        (PageLevel::Debug, "")
    };
    push(
        level,
        &format!(
            "[memory] guest RAM {}, WebAssembly heap {} of 4 GiB{warning}",
            size(guest),
            size(heap)
        ),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_counts_read_as_the_page_printed_them() {
        assert_eq!(size(512), "512 B");
        assert_eq!(size(4096 + 512), "4.5 KiB");
        assert_eq!(size(46 << 20), "46.00 MiB");
        assert_eq!(size(5_809_000_000), "5.41 GiB");
        assert_eq!(large(17), "17");
        assert_eq!(large(950_653), "950,653");
        assert_eq!(large(1_000), "1,000");
        assert_eq!(large(53_300_000), "53.3M");
        assert_eq!(duration(7200.0, 48000), "150 ms");
        assert_eq!(duration(96000.0, 48000), "2.0 s");
        assert_eq!(duration(3.0, 0), "3 frames");
    }

    #[test]
    fn idle_threads_are_named_at_5_and_30_seconds_then_every_minute() {
        assert_eq!(next_idle_warning(0), 5);
        assert_eq!(next_idle_warning(5), 30);
        assert_eq!(next_idle_warning(30), 60);
        assert_eq!(next_idle_warning(1440), 1500);
    }

    #[test]
    fn styles_are_named_by_their_bits() {
        assert_eq!(style_names(0b111), "Pro Controller, handheld, Joy-Con pair");
        assert_eq!(style_names(0), "nothing this console can present");
    }
}
