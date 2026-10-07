//! Translator, GPU and activity reports.

use crate::{json_escape, session, write_into};

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

/// The `"audio"` member of `switch_activity_json`, with its leading comma.
fn audio_activity_json(audio: &switch_core::cpu::AudioActivity) -> String {
    let outputs: Vec<String> = audio
        .outputs
        .iter()
        .map(|o| {
            format!(
                "{{\"handle\":{},\"sampleRate\":{},\"channels\":{},\"started\":{},\
                 \"volume\":{},\"appendedBuffers\":{},\"appendedFrames\":{},\
                 \"releasedBuffers\":{},\"pendingBuffers\":{},\"discardedFrames\":{},\
                 \"unplayableBuffers\":{}}}",
                o.handle,
                o.sample_rate,
                o.channels,
                o.started,
                if o.volume.is_finite() { o.volume } else { 0.0 },
                o.appended_buffers,
                o.appended_frames,
                o.released_buffers,
                o.pending_buffers,
                o.discarded_frames,
                o.unplayable_buffers
            )
        })
        .collect();
    let renderers: Vec<String> = audio
        .renderers
        .iter()
        .map(|r| {
            format!(
                "{{\"handle\":{},\"sampleRate\":{},\"started\":{},\"updates\":{},\
                 \"renderedFrames\":{},\"voices\":{},\"voicesPlaying\":{},\"sinkChannels\":{}}}",
                r.handle,
                r.sample_rate,
                r.started,
                r.updates,
                r.rendered_frames,
                r.voices,
                r.voices_playing,
                r.sink_channels
            )
        })
        .collect();
    format!(
        ",\"audio\":{{\"sampleRate\":{},\"channels\":{},\"samplesProduced\":{},\
         \"samplesTaken\":{},\"samplesDropped\":{},\"backlog\":{},\"outputs\":[{}],\
         \"renderers\":[{}]}}",
        audio.sample_rate,
        audio.channels,
        audio.produced,
        audio.taken,
        audio.dropped,
        audio.backlog,
        outputs.join(","),
        renderers.join(",")
    )
}

/// The session's activity counters as JSON. Counters run from boot (the worker diffs
/// them); the lists are taken, fitted to `maxlen`, with overflow counted in
/// `gpuDropped`, `filesDropped` and `dropped`.
#[no_mangle]
pub extern "C" fn switch_activity_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let cpu = &mut session(handle).cpu;
    let gpu = &cpu.nv.gpu;
    let stats = gpu.stats;
    let mut out = format!(
        "{{\"frames\":{},\"submissions\":{},\"draws\":{},\"drawsSkipped\":{},\
         \"clears\":{},\"clearsElided\":{},\"copies\":{},\"dispatches\":{},\"failures\":{}",
        gpu.frames,
        stats.submissions,
        stats.draws,
        stats.draws_skipped,
        stats.clears,
        stats.clears_elided,
        stats.copies,
        stats.dispatches,
        cpu.fs_activity.failures,
    )
    .into_bytes();
    out.extend_from_slice(audio_activity_json(&cpu.audio_activity()).as_bytes());
    let (supported, presented) = cpu.npad_styles();
    out.extend_from_slice(
        format!(",\"input\":{{\"supported\":{supported},\"presented\":{presented}}}").as_bytes(),
    );
    // Room for the closing fields.
    let budget = (maxlen as usize).saturating_sub(200);
    // Entries of the problem lists below that did not fit, summed.
    let mut problems_dropped = 0u64;
    let mut push_list = |out: &mut Vec<u8>, name: &str, entries: Vec<Vec<u8>>| {
        out.extend_from_slice(format!(",\"{name}\":[").as_bytes());
        let mut first = true;
        for entry in entries {
            if out.len() + entry.len() + 1 > budget {
                problems_dropped += 1;
                continue;
            }
            if !first {
                out.push(b',');
            }
            out.extend_from_slice(&entry);
            first = false;
        }
        out.push(b']');
    };

    let files = cpu.fs_activity.take_files();
    let mut files_dropped = 0u64;
    out.extend_from_slice(b",\"files\":[");
    let mut first = true;
    for (name, io) in &files {
        let mut entry = Vec::with_capacity(name.len() + 96);
        if !first {
            entry.push(b',');
        }
        entry.extend_from_slice(b"{\"name\":\"");
        json_escape(name, &mut entry);
        entry.extend_from_slice(
            format!(
                "\",\"reads\":{},\"readBytes\":{},\"writes\":{},\"writeBytes\":{}}}",
                io.reads, io.read_bytes, io.writes, io.write_bytes
            )
            .as_bytes(),
        );
        if out.len() + entry.len() > budget {
            files_dropped += 1;
            continue;
        }
        out.extend_from_slice(&entry);
        first = false;
    }
    out.push(b']');

    let mut gpu_activity = cpu.nv.gpu.take_activity();
    let surfaces = gpu_activity.take();
    let refusals = gpu_activity
        .take_refusals()
        .into_iter()
        .map(|(kind, reason, count)| {
            let mut entry = format!("{{\"kind\":\"{}\",\"reason\":\"", kind.name()).into_bytes();
            json_escape(&reason, &mut entry);
            entry.extend_from_slice(format!("\",\"count\":{count}}}").as_bytes());
            entry
        })
        .collect();
    push_list(&mut out, "refusals", refusals);
    let gaps = cpu
        .take_service_gaps()
        .into_iter()
        .map(|gap| {
            let mut entry = format!("{{\"kind\":\"{}\",\"name\":\"", gap.kind.name()).into_bytes();
            json_escape(&gap.name, &mut entry);
            let command = gap.command.map_or("null".to_owned(), |c| c.to_string());
            entry.extend_from_slice(
                format!("\",\"command\":{command},\"calls\":{}}}", gap.calls).as_bytes(),
            );
            entry
        })
        .collect();
    push_list(&mut out, "gaps", gaps);
    let nv_errors = cpu
        .take_nv_errors()
        .into_iter()
        .map(|(node, request, error, calls)| {
            let mut entry = Vec::from("{\"node\":\"");
            json_escape(&node, &mut entry);
            entry.extend_from_slice(
                format!("\",\"request\":{request},\"error\":{error},\"calls\":{calls}}}")
                    .as_bytes(),
            );
            entry
        })
        .collect();
    push_list(&mut out, "nvErrors", nv_errors);
    let mut gpu_dropped = 0u64;
    out.extend_from_slice(b",\"gpu\":[");
    let mut first = true;
    for (kind, tally) in &surfaces {
        let mut entry = Vec::with_capacity(tally.label.len() + 96);
        if !first {
            entry.push(b',');
        }
        entry.extend_from_slice(format!("{{\"kind\":\"{}\",\"label\":\"", kind.name()).as_bytes());
        json_escape(&tally.label, &mut entry);
        entry.extend_from_slice(
            format!(
                "\",\"count\":{},\"amount\":{},\"failed\":{}}}",
                tally.count, tally.amount, tally.failed
            )
            .as_bytes(),
        );
        if out.len() + entry.len() > budget {
            gpu_dropped += 1;
            continue;
        }
        out.extend_from_slice(&entry);
        first = false;
    }
    out.push(b']');

    let (threads, thread_log, mut thread_log_dropped) = cpu.take_thread_report();
    out.extend_from_slice(b",\"threads\":[");
    for (i, thread) in threads.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(
            format!(
                "{{\"index\":{},\"handle\":{},\"priority\":{},\"running\":{},\"ran\":{},\"switches\":{},\"idleMs\":{},\"entry\":\"",
                thread.index, thread.handle, thread.priority, thread.running, thread.ran, thread.switches, thread.idle_ms
            )
            .as_bytes(),
        );
        json_escape(&thread.entry, &mut out);
        out.extend_from_slice(b"\",\"at\":\"");
        json_escape(&thread.at, &mut out);
        out.extend_from_slice(b"\",\"state\":\"");
        json_escape(&thread.state, &mut out);
        out.extend_from_slice(b"\",\"name\":");
        match &thread.name {
            Some(name) => {
                out.push(b'"');
                json_escape(name, &mut out);
                out.push(b'"');
            }
            None => out.extend_from_slice(b"null"),
        }
        out.push(b'}');
    }
    out.extend_from_slice(b"],\"threadLog\":[");
    let mut first = true;
    for line in &thread_log {
        let mut entry = Vec::with_capacity(line.len() + 3);
        if !first {
            entry.push(b',');
        }
        entry.push(b'"');
        json_escape(line, &mut entry);
        entry.push(b'"');
        if out.len() + entry.len() > budget {
            thread_log_dropped += 1;
            continue;
        }
        out.extend_from_slice(&entry);
        first = false;
    }
    out.push(b']');

    let (journal, mut dropped) = cpu.fs_activity.take_journal();
    out.extend_from_slice(b",\"journal\":[");
    let mut first = true;
    for line in &journal {
        let mut entry = Vec::with_capacity(line.len() + 3);
        if !first {
            entry.push(b',');
        }
        entry.push(b'"');
        json_escape(line, &mut entry);
        entry.push(b'"');
        if out.len() + entry.len() > budget {
            dropped += 1;
            continue;
        }
        out.extend_from_slice(&entry);
        first = false;
    }
    out.extend_from_slice(
        format!(
            "],\"dropped\":{dropped},\"filesDropped\":{files_dropped},\"gpuDropped\":{gpu_dropped},\
             \"threadLogDropped\":{thread_log_dropped},\"problemsDropped\":{problems_dropped}}}"
        )
        .as_bytes(),
    );
    write_into(buf, maxlen, &out)
}

/// Whether the GPU backend has lost its device. Cheap; polled every slice.
#[no_mangle]
pub extern "C" fn switch_gpu_lost(handle: u32) -> u32 {
    let s = session(handle);
    u32::from(s.cpu.nv.gpu.renderer_lost())
}
