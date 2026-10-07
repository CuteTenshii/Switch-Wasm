//! Debugging levers, traces and crash reports.

use switch_core::cpu::Cpu;

use crate::{json_escape, session, session_opt, write_into, PANICKED};

/// Enable/disable the block translator; the browser has no `SWITCH_NO_JIT`.
#[no_mangle]
pub extern "C" fn switch_set_jit(handle: u32, enabled: u32) {
    session(handle).cpu.set_jit_enabled(enabled != 0);
}

/// Enable/disable the per-instruction disassembly trace.
#[no_mangle]
pub extern "C" fn switch_set_trace(handle: u32, enabled: u32) {
    let s = session(handle);
    s.cpu.trace_enabled = enabled != 0;
    if enabled == 0 {
        s.cpu.trace.clear();
    }
}

/// Copy the debug trace (disassembly and fault context) into `buf` and clear it.
#[no_mangle]
pub extern "C" fn switch_drain_trace(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    // Fold in traces from components with no `Cpu`.
    s.cpu.absorb_traces();
    let n = s.cpu.trace.len().min(maxlen as usize);
    if n > 0 && !buf.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(s.cpu.trace.as_ptr(), buf, n);
        }
        s.cpu.trace.drain(..n);
    }
    n as u32
}

/// Write a full register snapshot as text into `buf`. Returns bytes written.
#[no_mangle]
pub extern "C" fn switch_dump_regs(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let dump = s.cpu.reg_dump();
    write_into(buf, maxlen, dump.as_bytes())
}

/// One line per guest thread: state, what it is blocked on, and where it stopped.
#[no_mangle]
pub extern "C" fn switch_thread_dump(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    write_into(buf, maxlen, s.cpu.thread_dump().as_bytes())
}

/// The guest's call stack, innermost first, as a JSON array of addresses.
#[no_mangle]
pub extern "C" fn switch_backtrace_json(handle: u32, depth: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    write_into(
        buf,
        maxlen,
        backtrace_json(&s.cpu, depth as usize).as_bytes(),
    )
}

fn backtrace_json(cpu: &Cpu, depth: usize) -> String {
    let frames: Vec<String> = cpu
        .backtrace(depth)
        .iter()
        .map(|pc| format!("{pc}"))
        .collect();
    format!("[{}]", frames.join(","))
}

/// Make every blocked thread runnable and return the count. A debugging lever.
#[no_mangle]
pub extern "C" fn switch_wake_blocked(handle: u32) -> u32 {
    session(handle).cpu.wake_all_blocked() as u32
}

/// Make every created-but-never-started thread runnable and return the count.
#[no_mangle]
pub extern "C" fn switch_start_created_threads(handle: u32) -> u32 {
    session(handle).cpu.start_created_threads() as u32
}

/// Turn every diagnostic channel on (nonzero) or off.
#[no_mangle]
pub extern "C" fn switch_set_trace_channels(on: u32) {
    switch_core::trace::set_all(on != 0);
}

#[no_mangle]
pub extern "C" fn switch_version(buf: *mut u8, maxlen: u32) -> u32 {
    write_into(buf, maxlen, build_version().as_bytes())
}

/// `<crate version>+<commit>`, or just the crate version.
fn build_version() -> String {
    let commit = env!("SWITCH_BUILD_COMMIT");
    if commit.is_empty() {
        env!("CARGO_PKG_VERSION").to_string()
    } else {
        format!("{}+{commit}", env!("CARGO_PKG_VERSION"))
    }
}

/// Service commands this run asked for and did not get, as JSON: `unimplemented`
/// (refused) and `stubbed` (answered with nothing behind it).
#[no_mangle]
pub extern "C" fn switch_unimplemented_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let mut out = Vec::with_capacity(4096);
    out.extend_from_slice(b"{\"unimplemented\":");
    ipc_list_json(&s.cpu.unimplemented_ipc(), &mut out);
    out.extend_from_slice(b",\"stubbed\":");
    ipc_list_json(&s.cpu.stubbed_ipc(), &mut out);
    out.push(b'}');
    write_into(buf, maxlen, &out)
}

fn ipc_list_json(pairs: &[(String, Option<u32>)], out: &mut Vec<u8>) {
    out.push(b'[');
    for (i, (iface, cmd)) in pairs.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(b"{\"iface\":\"");
        json_escape(iface, out);
        match cmd {
            Some(id) => out.extend_from_slice(format!("\",\"cmd\":{id}}}").as_bytes()),
            None => out.extend_from_slice(b"\",\"cmd\":null}"),
        }
    }
    out.push(b']');
}

/// Everything worth putting in a bug report about this run, as JSON. `panicked`
/// distinguishes an emulator bug from a guest fault.
#[no_mangle]
pub extern "C" fn switch_crash_report_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let mut out = Vec::with_capacity(16 * 1024);
    out.extend_from_slice(b"{\"version\":\"");
    json_escape(&build_version(), &mut out);
    out.extend_from_slice(
        format!(
            "\",\"panicked\":{}",
            PANICKED.load(std::sync::atomic::Ordering::Relaxed)
        )
        .as_bytes(),
    );
    out.extend_from_slice(b",\"traceMask\":");
    out.extend_from_slice(switch_core::trace::mask().to_string().as_bytes());

    // Not `session`, which panics on a dead handle.
    let Some(s) = session_opt(handle) else {
        out.extend_from_slice(b",\"session\":null}");
        return write_into(buf, maxlen, &out);
    };

    out.extend_from_slice(b",\"lastError\":\"");
    json_escape(&s.last_error, &mut out);
    out.extend_from_slice(b"\",\"guestFatal\":");
    match s.cpu.guest_fatal() {
        Some(fatal) => {
            out.extend_from_slice(b"\"");
            json_escape(fatal, &mut out);
            out.extend_from_slice(b"\"");
        }
        None => out.extend_from_slice(b"null"),
    }
    out.extend_from_slice(b",\"title\":");
    match &s.control {
        Some(control) => {
            out.extend_from_slice(b"{\"id\":\"");
            out.extend_from_slice(format!("{:016x}", control.title_id).as_bytes());
            out.extend_from_slice(b"\",\"name\":\"");
            json_escape(&control.name, &mut out);
            out.extend_from_slice(b"\",\"version\":\"");
            json_escape(&control.nacp.display_version, &mut out);
            out.extend_from_slice(b"\"}");
        }
        // A homebrew NRO has no Control NCA; name it by program id.
        None => {
            out.extend_from_slice(format!("{{\"id\":\"{:016x}\"}}", s.cpu.program_id()).as_bytes());
        }
    }

    let stats = s.cpu.jit_stats();
    out.extend_from_slice(
        format!(
            ",\"cpu\":{{\"pc\":{},\"mode\":\"{:?}\",\"steps\":{},\"cycles\":{},\"halted\":{},\
             \"thread\":{},\"guestRam\":{},\"docked\":{}}}",
            s.cpu.get_pc(),
            s.cpu.mode(),
            s.cpu.steps,
            s.cpu.cycles,
            s.cpu.halted,
            s.cpu.current_thread_index(),
            s.cpu.mem.mapped_bytes(),
            s.cpu.operation_mode() != switch_core::cpu::OperationMode::Handheld,
        )
        .as_bytes(),
    );
    out.extend_from_slice(
        format!(
            ",\"jit\":{{\"enabled\":{},\"blocks\":{},\"translated\":{},\"executed\":{},\
             \"linked\":{},\"invalidated\":{},\"interpreted\":{},\"interpretedGroups\":[{}]}}",
            s.cpu.jit_enabled(),
            stats.blocks,
            stats.translated,
            stats.executed,
            stats.linked,
            stats.invalidated,
            stats.interpreted,
            stats
                .interpreted_groups
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
        .as_bytes(),
    );

    out.extend_from_slice(b",\"gpu\":");
    let report = s.cpu.nv.gpu.renderer_report();
    let frames = s.cpu.nv.gpu.frames;
    match report.strip_suffix('}') {
        Some(body) if body.len() > 1 => {
            out.extend_from_slice(format!("{body},\"frames\":{frames}}}").as_bytes())
        }
        _ => out.extend_from_slice(format!("{{\"frames\":{frames}}}").as_bytes()),
    }

    out.extend_from_slice(b",\"backtrace\":");
    out.extend_from_slice(backtrace_json(&s.cpu, 16).as_bytes());
    out.extend_from_slice(b",\"registers\":\"");
    json_escape(&s.cpu.reg_dump(), &mut out);
    out.extend_from_slice(b"\",\"threads\":\"");
    json_escape(&s.cpu.thread_dump(), &mut out);

    out.extend_from_slice(b"\",\"unimplemented\":");
    ipc_list_json(&s.cpu.unimplemented_ipc(), &mut out);
    out.extend_from_slice(b",\"stubbed\":");
    ipc_list_json(&s.cpu.stubbed_ipc(), &mut out);

    // Last, so truncation loses the trace first.
    s.cpu.absorb_traces();
    out.extend_from_slice(b",\"trace\":\"");
    json_escape(&String::from_utf8_lossy(&s.cpu.trace), &mut out);
    out.extend_from_slice(b"\"}");
    write_into(buf, maxlen, &out)
}
