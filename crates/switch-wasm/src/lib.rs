//! WASM bindings for switch-core.
//!
//! Compiled as a `cdylib` for `wasm32-unknown-unknown` with no external
//! dependencies. The browser JS calls these via a raw `extern "C"` ABI:
//! buffers flow in and out through wasm linear memory, which JS writes to and
//! reads with a `DataView` over the exported `memory`.
//!
//! Design notes:
//! * A handle is an index into a global session table, so JS never deals with
//!   raw pointers.
//! * Structured results (file listings, NCA info) are returned as a tiny JSON
//!   string written into a JS-provided buffer — no JSON crate required.
//! * Errors are captured in the session and read back via `switch_last_error`.
//! * A memory-mapped framebuffer (like the Switch GPU's) is rendered by the
//!   host: homebrew writes pixels to [`FB_BASE`], JS snapshots it each frame.

use std::alloc::{alloc, dealloc, Layout};
use std::sync::Mutex;

use switch_core::cpu::{Cpu, SyscallMode};
use switch_core::elf::load_elf;
use switch_core::nca::Nca;
use switch_core::nro::load_nro;
use switch_core::nsp::Pfs0;

/// Framebuffer base address, width, height and stride (RGBA, little-endian).
pub use switch_core::{FB_BASE, FB_HEIGHT, FB_STRIDE, FB_WIDTH};
/// Memory-mapped input register: JS writes an ASCII key here, homebrew polls
/// and acknowledges (writes 0) when consumed.
pub const INPUT_ADDR: u32 = switch_core::INPUT_ADDR;

struct Session {
    /// Cached NSP image (kept so file payloads can be extracted on demand).
    nsp_data: Vec<u8>,
    /// Parsed file table of the last NSP.
    nsp_files: Vec<switch_core::nsp::Pfs0File>,
    cpu: Cpu,
    last_error: String,
}

static SESSIONS: Mutex<Vec<Option<Session>>> = Mutex::new(Vec::new());

fn session(handle: u32) -> &'static mut Session {
    let mut guard = SESSIONS.lock().unwrap();
    let slot = guard
        .get_mut(handle as usize)
        .and_then(|s| s.as_mut())
        .expect("invalid session handle");
    // SAFETY: wasm is single-threaded; the Mutex guarantees exclusive access
    // for the duration of the call.
    unsafe { std::mem::transmute::<&mut Session, &'static mut Session>(slot) }
}

fn new_handle(session: Session) -> u32 {
    let mut guard = SESSIONS.lock().unwrap();
    for (i, slot) in guard.iter_mut().enumerate() {
        if slot.is_none() {
            *slot = Some(session);
            return i as u32;
        }
    }
    guard.push(Some(session));
    (guard.len() - 1) as u32
}

/// Allocate `len` bytes of wasm linear memory for passing buffers in from JS.
#[no_mangle]
pub extern "C" fn switch_alloc(len: u32) -> *mut u8 {
    let layout = Layout::from_size_align(len as usize, 1).unwrap();
    unsafe { alloc(layout) }
}

/// Free a buffer previously returned by `switch_alloc`.
#[no_mangle]
pub extern "C" fn switch_free(ptr: *mut u8, len: u32) {
    let layout = Layout::from_size_align(len as usize, 1).unwrap();
    unsafe { dealloc(ptr, layout) }
}

/// Create a fresh machine, return its handle.
#[no_mangle]
pub extern "C" fn switch_new() -> u32 {
    // The framebuffer and input pages are pre-mapped by Cpu::new, and the
    // stack + low-memory shim are provided by bootstrap so libnx-style
    // homebrew gets the runtime environment the real loader sets up.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    new_handle(Session {
        nsp_data: Vec::new(),
        nsp_files: Vec::new(),
        cpu,
        last_error: String::new(),
    })
}

/// Drop a machine.
#[no_mangle]
pub extern "C" fn switch_free_session(handle: u32) {
    let mut guard = SESSIONS.lock().unwrap();
    if let Some(slot) = guard.get_mut(handle as usize) {
        *slot = None;
    }
}

/// Copy the last error message into `buf` (NUL-terminated). Returns length.
#[no_mangle]
pub extern "C" fn switch_last_error(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let bytes = s.last_error.as_bytes();
    let n = bytes.len().min(maxlen as usize).saturating_sub(1);
    if n > 0 && !buf.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n);
            *buf.add(n) = 0;
        }
    }
    n as u32
}

/// Load an NSP image into the session. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn switch_load_nsp(handle: u32, ptr: *const u8, len: u32) -> i32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    match Pfs0::parse(data) {
        Ok(pfs0) => {
            s.nsp_data = data.to_vec();
            s.nsp_files = pfs0.files;
            s.last_error.clear();
            0
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Parse the file table of the current NSP and return it as JSON.
/// Writes up to `maxlen` bytes into `buf`; returns bytes written.
#[no_mangle]
pub extern "C" fn switch_nsp_files_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let mut out = Vec::new();
    out.extend_from_slice(b"[");
    for (i, f) in s.nsp_files.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b",");
        }
        out.extend_from_slice(b"{\"name\":\"");
        json_escape(&f.name, &mut out);
        out.extend_from_slice(b"\",\"offset\":");
        out.extend_from_slice(f.offset.to_string().as_bytes());
        out.extend_from_slice(b",\"size\":");
        out.extend_from_slice(f.size.to_string().as_bytes());
        out.extend_from_slice(b"}");
    }
    out.extend_from_slice(b"]");
    write_into(buf, maxlen, &out)
}

/// Copy the payload of NSP file `index` into `buf`. Returns bytes copied or -1.
#[no_mangle]
pub extern "C" fn switch_extract_file(
    handle: u32,
    index: u32,
    buf: *mut u8,
    maxlen: u32,
) -> i64 {
    let s = session(handle);
    if let Some(f) = s.nsp_files.get(index as usize) {
        let start = f.offset as usize;
        let end = (start + f.size as usize).min(s.nsp_data.len());
        let n = (end - start).min(maxlen as usize);
        if n > 0 && !buf.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(s.nsp_data.as_ptr().add(start), buf, n);
            }
        }
        n as i64
    } else {
        -1
    }
}

/// Parse an NCA from `ptr`/`len` and return a JSON summary.
#[no_mangle]
pub extern "C" fn switch_parse_nca(ptr: *const u8, len: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    let mut out = Vec::new();
    match Nca::parse(data) {
        Ok(nca) => {
            out.extend_from_slice(b"{\"title_id\":\"");
            out.extend_from_slice(format!("{:016x}", nca.title_id).as_bytes());
            out.extend_from_slice(b"\",\"content_type\":\"");
            out.extend_from_slice(nca.content_type.name().as_bytes());
            out.extend_from_slice(b"\",\"sdk_version\":\"");
            out.extend_from_slice(format!("{:08x}", nca.sdk_version).as_bytes());
            out.extend_from_slice(b"\",\"crypto_type\":");
            out.extend_from_slice(nca.crypto_type.to_string().as_bytes());
            out.extend_from_slice(b",\"encrypted\":");
            out.extend_from_slice(if nca.is_encrypted() { b"true" } else { b"false" });
            out.extend_from_slice(b",\"file_size\":");
            out.extend_from_slice(nca.file_size.to_string().as_bytes());
            out.extend_from_slice(b",\"sections\":[");
            for (i, sec) in nca.sections.iter().enumerate() {
                if i > 0 {
                    out.extend_from_slice(b",");
                }
                out.extend_from_slice(b"{\"offset\":");
                out.extend_from_slice(sec.media_offset.to_string().as_bytes());
                out.extend_from_slice(b",\"size\":");
                out.extend_from_slice(sec.media_size.to_string().as_bytes());
                out.extend_from_slice(b",\"fs_type\":");
                out.extend_from_slice(if sec.fs_type == 0 { b"\"PFS0\"" } else { b"\"ROMFS\"" });
                out.extend_from_slice(b"}");
            }
            out.extend_from_slice(b"]}");
        }
        Err(e) => {
            out.extend_from_slice(b"{\"error\":\"");
            json_escape(&e.to_string(), &mut out);
            out.extend_from_slice(b"\"}");
        }
    }
    write_into(buf, maxlen, &out)
}

/// Load an NRO homebrew image into the CPU. Returns entry address or -1.
#[no_mangle]
pub extern "C" fn switch_load_nro(handle: u32, ptr: *const u8, len: u32) -> i64 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    match load_nro(&mut s.cpu.mem, data) {
        Ok(loaded) => {
            s.cpu.set_pc(loaded.entry);
            boot_entry_regs(&mut s.cpu);
            s.cpu.out.clear();
            s.cpu.trace.clear();
            s.cpu.halted = false;
            s.last_error.clear();
            loaded.entry as i64
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Load an AArch64 ELF into the CPU. Returns entry address or -1.
#[no_mangle]
pub extern "C" fn switch_load_elf(handle: u32, ptr: *const u8, len: u32) -> i64 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    match load_elf(&mut s.cpu.mem, data) {
        Ok(elf) => {
            s.cpu.set_pc(elf.entry as u32);
            boot_entry_regs(&mut s.cpu);
            s.cpu.out.clear();
            s.cpu.trace.clear();
            s.cpu.halted = false;
            s.last_error.clear();
            elf.entry as i64
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Reset the integer registers for a clean boot and pass the loader's entry
/// convention: `x0 = 0`, `x1 = 1`. Real homebrew crt0s (e.g. hbmenu) store
/// these into runtime globals and expect `x1` to be non-zero.
fn boot_entry_regs(cpu: &mut Cpu) {
    for i in 0..=30u8 {
        cpu.set_reg(i, 0);
    }
    cpu.set_reg(1, 1);
}

/// Configure the syscall ABI (0 = None, 1 = Uart demo ABI, 2 = Horizon
/// stubs for real libnx homebrew).
#[no_mangle]
pub extern "C" fn switch_set_syscall_mode(handle: u32, mode: u32) {
    let s = session(handle);
    s.cpu.syscall_mode = match mode {
        1 => SyscallMode::Uart,
        2 => SyscallMode::Horizon,
        _ => SyscallMode::None,
    };
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

/// Copy accumulated debug trace (disassembly + fault context) into `buf` and
/// clear it. Fault context is always recorded, even with tracing disabled.
#[no_mangle]
pub extern "C" fn switch_drain_trace(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
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

/// Framebuffer geometry accessors.
#[no_mangle]
pub extern "C" fn switch_fb_width() -> u32 {
    FB_WIDTH
}
#[no_mangle]
pub extern "C" fn switch_fb_height() -> u32 {
    FB_HEIGHT
}

/// Copy the current framebuffer (RGBA) into `buf`. Returns bytes copied.
#[no_mangle]
pub extern "C" fn switch_fb_snapshot(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let n = ((FB_WIDTH * FB_HEIGHT * 4) as usize).min(maxlen as usize);
    let out = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    match s.cpu.mem.read_into(FB_BASE, out) {
        Ok(()) => n as u32,
        Err(_) => 0,
    }
}

/// Write `len` bytes from `ptr` into emulated memory at `addr` (used for the
/// memory-mapped input register and similar). Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn switch_write_mem(handle: u32, addr: u32, ptr: *const u8, len: u32) -> i32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    match s.cpu.mem.map(addr, data) {
        Ok(()) => 0,
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Run up to `max_steps` instructions. Returns steps executed or -1 on error.
#[no_mangle]
pub extern "C" fn switch_run(handle: u32, max_steps: u64) -> i64 {
    let s = session(handle);
    match s.cpu.run(max_steps) {
        Ok(report) => {
            if report.halted {
                // push a marker the frontend can detect
            }
            report.steps as i64
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// True if the machine halted via SVC #0.
#[no_mangle]
pub extern "C" fn switch_halted(handle: u32) -> i32 {
    session(handle).cpu.halted as i32
}

/// Copy accumulated console output into `buf` and clear it. Returns bytes copied.
#[no_mangle]
pub extern "C" fn switch_drain_output(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let n = s.cpu.out.len().min(maxlen as usize);
    if n > 0 && !buf.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(s.cpu.out.as_ptr(), buf, n);
        }
        s.cpu.out.drain(..n);
    }
    n as u32
}

/// Read register `idx` (0..=31; 31 = SP).
#[no_mangle]
pub extern "C" fn switch_get_reg(handle: u32, idx: u32) -> u64 {
    session(handle).cpu.read_x(idx as u8)
}

/// Current PC.
#[no_mangle]
pub extern "C" fn switch_get_pc(handle: u32) -> u32 {
    session(handle).cpu.get_pc()
}

/// Total instructions executed.
#[no_mangle]
pub extern "C" fn switch_get_cycles(handle: u32) -> u64 {
    session(handle).cpu.cycles
}

// ---- small JSON helpers ----

fn json_escape(s: &str, out: &mut Vec<u8>) {
    for b in s.bytes() {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            0x20..=0x7E => out.push(b),
            _ => {
                out.extend_from_slice(format!("\\u{:04x}", b).as_bytes());
            }
        }
    }
}

fn write_into(buf: *mut u8, maxlen: u32, data: &[u8]) -> u32 {
    let n = data.len().min(maxlen as usize);
    if n > 0 && !buf.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf, n);
        }
    }
    n as u32
}
