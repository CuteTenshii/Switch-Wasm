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
use std::sync::atomic::{AtomicU32, Ordering};

/// A `Sync` wrapper for single-threaded interior mutability (wasm). Safer
/// than `static mut` and not gated like `std::cell::SyncUnsafeCell`.
#[repr(transparent)]
struct SyncCell<T>(std::cell::UnsafeCell<T>);
unsafe impl<T> Sync for SyncCell<T> {}
impl<T> SyncCell<T> {
    const fn new(v: T) -> Self {
        Self(std::cell::UnsafeCell::new(v))
    }
    fn get(&self) -> *mut T {
        self.0.get()
    }
}

use switch_core::cpu::{Cpu, SyscallMode};
use switch_core::elf::load_elf;
use switch_core::nca::Nca;
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
    /// Keys loaded from prod.keys / title.keys, used to decrypt NCA headers.
    keys: switch_core::keys::KeySet,
    cpu: Cpu,
    last_error: String,
}

// Single-threaded wasm: a std `Mutex` would abort on any reentrant `lock()`
// (the wasm `no_threads` backend asserts), and a panic while one is held
// leaves it locked forever. Use plain `SyncUnsafeCell`s — there is exactly one
// "thread" (the JS event loop) and every export runs to completion before the
// next is called, so unsynchronized access is safe.
static SESSIONS: SyncCell<Vec<Option<Session>>> = SyncCell::new(Vec::new());

/// Last Rust panic message captured by the panic hook (fixed buffer, so the
/// hook itself never allocates and can't recurse).
static PANIC_MSG: SyncCell<[u8; 512]> = SyncCell::new([0u8; 512]);

/// Next session-handle counter (independent of slot reuse so stale handles
/// never alias a recycled slot).
static HANDLE_COUNTER: AtomicU32 = AtomicU32::new(0);

fn session(handle: u32) -> &'static mut Session {
    // SAFETY: single-threaded wasm; see the `SESSIONS` comment.
    let slots = unsafe { &mut *SESSIONS.get() };
    let len = slots.len();
    let slot = slots
        .get_mut(handle as usize)
        .and_then(|s| s.as_mut())
        .unwrap_or_else(|| panic!("invalid session handle {handle} (slots len {len})"));
    unsafe { std::mem::transmute::<&mut Session, &'static mut Session>(slot) }
}

fn new_handle(session: Session) -> u32 {
    let id = HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed);
    // SAFETY: single-threaded wasm; see the `SESSIONS` comment.
    let slots = unsafe { &mut *SESSIONS.get() };
    if id as usize >= slots.len() {
        slots.push(Some(session));
    } else {
        slots[id as usize] = Some(session);
    }
    id
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
    // Surface Rust panics to the frontend (they otherwise trap silently as
    // "unreachable" in wasm). The hook appends to a static buffer read back
    // through `switch_last_error`.
    std::panic::set_hook(Box::new(|info| {
        let msg = format!("PANIC: {info}");
        // SAFETY: single-threaded wasm; the hook runs once per panic.
        let guard = unsafe { &mut *PANIC_MSG.get() };
        let n = msg.len().min(guard.len() - 1);
        guard[..n].copy_from_slice(&msg.as_bytes()[..n]);
        guard[n] = 0;
    }));
    // The framebuffer and input pages are pre-mapped by Cpu::new, and the
    // stack + low-memory shim are provided by bootstrap so libnx-style
    // homebrew gets the runtime environment the real loader sets up.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    new_handle(Session {
        nsp_data: Vec::new(),
        nsp_files: Vec::new(),
        keys: switch_core::keys::KeySet::default(),
        cpu,
        last_error: String::new(),
    })
}

/// Drop a machine.
#[no_mangle]
pub extern "C" fn switch_free_session(handle: u32) {
    // SAFETY: single-threaded wasm; see the `SESSIONS` comment.
    let slots = unsafe { &mut *SESSIONS.get() };
    if let Some(slot) = slots.get_mut(handle as usize) {
        *slot = None;
    }
}

/// Copy the last error message into `buf` (NUL-terminated). Returns length.
/// Also surfaces any Rust panic captured by the panic hook.
#[no_mangle]
pub extern "C" fn switch_last_error(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    // A captured panic takes priority (and doesn't need a valid handle).
    // SAFETY: single-threaded wasm; see the `SESSIONS` comment.
    let panicked = unsafe { &mut *PANIC_MSG.get() };
    if panicked[0] != 0 {
        let len = panicked.iter().position(|&b| b == 0).unwrap_or(panicked.len());
        let n = len.min(maxlen as usize).saturating_sub(1);
        if n > 0 && !buf.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(panicked.as_ptr(), buf, n);
                *buf.add(n) = 0;
            }
        }
        panicked.fill(0);
        return n as u32;
    }
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

/// Load an NSP image into the session. Takes ownership of the buffer at `ptr`
/// (do not free it afterwards) — for a multi-GB NSP that halves the wasm
/// memory footprint versus copying. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn switch_load_nsp(handle: u32, ptr: *const u8, len: u32) -> i32 {
    let s = session(handle);
    if ptr.is_null() {
        s.last_error = "null NSP buffer".into();
        return -1;
    }
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    match Pfs0::parse(data) {
        Ok(pfs0) => {
            // Move ownership of the staged wasm buffer into the session.
            // SAFETY: `ptr` came from `switch_alloc(len)` (same global
            // allocator, same Layout), and the caller no longer frees it.
            let owned = unsafe { Vec::from_raw_parts(ptr as *mut u8, len as usize, len as usize) };
            s.nsp_data = owned;
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

/// Read a slice of NSP file `index` starting at `file_offset` into `buf`
/// (clamped to the file). Used to grab just an NCA header without allocating
/// the whole (potentially huge) payload in wasm memory. Returns bytes copied
/// or -1.
#[no_mangle]
pub extern "C" fn switch_read_file(
    handle: u32,
    index: u32,
    file_offset: u64,
    buf: *mut u8,
    maxlen: u32,
) -> i64 {
    let s = session(handle);
    if let Some(f) = s.nsp_files.get(index as usize) {
        let start = f.offset as usize + file_offset as usize;
        let end = (f.offset as usize + f.size as usize).min(s.nsp_data.len());
        if start >= end {
            return 0;
        }
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

/// Parse an NCA from `ptr`/`len` and return a JSON summary. If the session has
/// keys loaded and the header is encrypted, it is decrypted transparently.
#[no_mangle]
pub extern "C" fn switch_parse_nca(handle: u32, ptr: *const u8, len: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    let mut out = Vec::new();
    match Nca::parse_with_keys(data, Some(&s.keys)) {
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
            // Return the raw error; the frontend adds friendly context.
            out.extend_from_slice(b"{\"error\":\"");
            json_escape(&e.to_string(), &mut out);
            out.extend_from_slice(b"\"}");
        }
    }
    write_into(buf, maxlen, &out)
}

/// Load `prod.keys` / `title.keys` text files into the session. Either pointer
/// may be NULL with length 0. Returns 0 on success, -1 on parse failure.
#[no_mangle]
pub extern "C" fn switch_load_keys(
    handle: u32,
    prod_ptr: *const u8,
    prod_len: u32,
    title_ptr: *const u8,
    title_len: u32,
) -> i32 {
    let s = session(handle);
    let prod = if !prod_ptr.is_null() && prod_len > 0 {
        unsafe { std::slice::from_raw_parts(prod_ptr, prod_len as usize) }
    } else {
        &[]
    };
    let title = if !title_ptr.is_null() && title_len > 0 {
        unsafe { std::slice::from_raw_parts(title_ptr, title_len as usize) }
    } else {
        &[]
    };
    let prod_text = String::from_utf8_lossy(prod);
    let title_text = String::from_utf8_lossy(title);
    let prod_entries = switch_core::keys::parse_keys_file(&prod_text);
    let title_entries = switch_core::keys::parse_keys_file(&title_text);
    let mut ks = switch_core::keys::keyset_from_prod(&prod_entries);
    ks.title_keys = switch_core::keys::keyset_from_title(&title_entries);
    s.keys = ks;
    s.last_error.clear();
    0
}

/// Load an NRO homebrew image into the CPU. Returns entry address or -1.
#[no_mangle]
pub extern "C" fn switch_load_nro(handle: u32, ptr: *const u8, len: u32) -> i64 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    match s.cpu.boot_homebrew(data) {
        Ok(loaded) => {
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
            boot_entry_regs(&mut s.cpu, 0);
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
/// convention. The libnx "HOME BREW" crt0 detects the entry kind from its
/// arguments: `x0 = env block ptr` with `x1 = UINT64_MAX` selects the NRO /
/// homebrew-ABI path (which parses the env and runs `main`), while `x0 = 0`
/// selects the NSO path (plain boot). Non-self-relocating NROs (e.g. sdl)
/// keep the plain `x0 = 0, x1 = 1` convention.
fn boot_entry_regs(cpu: &mut Cpu, env_addr: u32) {
    for i in 0..=30u8 {
        cpu.set_reg(i, 0);
    }
    cpu.set_reg(0, env_addr as u64);
    cpu.set_reg(1, if env_addr != 0 { u64::MAX } else { 1 });
    // Point LR at the exit trampoline so a direct-entered `main` that returns
    // cleanly exits instead of branching to NULL (pc=0).
    cpu.set_reg(30, switch_core::cpu::SELF_RETURN_TRAMPOLINE as u64);
}

/// Configure the syscall ABI (0 = None, 2 = Horizon stubs for real homebrew).
#[no_mangle]
pub extern "C" fn switch_set_syscall_mode(handle: u32, mode: u32) {
    let s = session(handle);
    s.cpu.syscall_mode = match mode {
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

/// Framebuffer geometry. Once the guest has presented a frame through the
/// display's buffer queue, this is the real console resolution (usually
/// 1280x720); before that it is the memory-mapped demo framebuffer.
#[no_mangle]
pub extern "C" fn switch_fb_width(handle: u32) -> u32 {
    let s = session(handle);
    if s.cpu.nv.gpu.frames > 0 { s.cpu.nv.gpu.framebuffer.width } else { FB_WIDTH }
}

#[no_mangle]
pub extern "C" fn switch_fb_height(handle: u32) -> u32 {
    let s = session(handle);
    if s.cpu.nv.gpu.frames > 0 { s.cpu.nv.gpu.framebuffer.height } else { FB_HEIGHT }
}

/// Number of frames the guest has presented. JS polls this to know when the
/// screen changed and what resolution to size the canvas to.
#[no_mangle]
pub extern "C" fn switch_frame_count(handle: u32) -> u32 {
    session(handle).cpu.nv.gpu.frames as u32
}

/// Copy the current screen (RGBA8888) into `buf`. Returns bytes copied.
///
/// This is the scanned-out frame the guest last handed to the display, or the
/// memory-mapped demo framebuffer when nothing has been presented yet.
#[no_mangle]
pub extern "C" fn switch_fb_snapshot(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    if s.cpu.nv.gpu.frames > 0 {
        let fb = &s.cpu.nv.gpu.framebuffer;
        let n = (fb.pixels.len() * 4).min(maxlen as usize);
        let out = unsafe { std::slice::from_raw_parts_mut(buf, n) };
        for (chunk, pixel) in out.chunks_exact_mut(4).zip(fb.pixels.iter()) {
            chunk.copy_from_slice(&pixel.to_le_bytes());
        }
        return n as u32;
    }
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

/// Feed the host gamepad state to the guest. `buttons` is a `HidNpadButton`
/// bitfield (A=0x1, B=0x2, X=0x4, Y=0x8, L=0x10, R=0x20, ZL=0x40, ZR=0x80,
/// Plus=0x100, Minus=0x200, DpadLeft=0x400, DpadUp=0x800, DpadRight=0x1000,
/// DpadDown=0x2000, StickL=0x4000, StickR=0x8000); sticks are -32768..32767.
/// Written to the memory-mapped input register and, once libnx maps its hid
/// shared memory, mirrored into the `HidSharedMemory` layout so `padUpdate`
/// sees it.
#[no_mangle]
pub extern "C" fn switch_set_input(
    handle: u32,
    buttons: u64,
    stick_lx: i32,
    stick_ly: i32,
    stick_rx: i32,
    stick_ry: i32,
) {
    session(handle)
        .cpu
        .set_gamepad_state(buttons, stick_lx, stick_ly, stick_rx, stick_ry);
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

/// Guest RAM currently backed by host storage, in bytes — the emulated
/// console's memory use, not the wasm heap's.
#[no_mangle]
pub extern "C" fn switch_guest_ram(handle: u32) -> u64 {
    session(handle).cpu.mem.mapped_bytes()
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
