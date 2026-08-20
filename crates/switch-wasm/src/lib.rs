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

use switch_core::cpu::Cpu;
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
    // Horizon is the only ABI a session ever runs: the frontend loads NROs,
    // NCAs and NSPs, all of which are real Switch programs.
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
                out.extend_from_slice(b",\"fs_type\":\"");
                // The section entry itself carries no filesystem-type byte;
                // that only lives in the (separately encrypted) FS header,
                // which needs the full NCA_FULL_HEADER_SIZE-byte header to
                // decrypt. The lightweight "inspect this NCA" path may only
                // have the base header, in which case this is unknown.
                let fs_type = match nca.fs_headers.get(i).and_then(|o| o.as_ref()) {
                    Some(fs) if sec.media_size > 0 => {
                        if fs.fs_type == 1 { "PFS0" } else { "ROMFS" }
                    }
                    _ => "?",
                };
                out.extend_from_slice(fs_type.as_bytes());
                out.extend_from_slice(b"\"}");
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

/// Collect a title's ExeFS modules in Nintendo's required load order —
/// `rtld`, `main`, `subsdk0..subsdk9`, `sdk` — skipping whichever of those a
/// title doesn't have. `main` is required by every real title; the rest are
/// common but not guaranteed (a title with no imports from `sdk` might omit
/// it, for instance).
fn collect_modules<'a>(pfs0: &Pfs0, exefs: &'a [u8]) -> Vec<(&'static str, &'a [u8])> {
    const MODULE_ORDER: &[&str] = &[
        "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5",
        "subsdk6", "subsdk7", "subsdk8", "subsdk9", "sdk",
    ];
    MODULE_ORDER
        .iter()
        .filter_map(|&name| {
            let f = pfs0.find(name)?;
            let start = f.offset as usize;
            let end = start + f.size as usize; // Pfs0::parse already bounds-checked every entry
            Some((name, &exefs[start..end]))
        })
        .collect()
}

/// Shared by `switch_load_nca` and `switch_load_nca_from_nsp`: decrypt a
/// Program NCA's ExeFS from `raw` (the full, still-encrypted NCA bytes),
/// extract `main` and boot it. Returns entry address or -1.
///
/// Takes `keys`/`cpu`/`last_error` as separate borrows (rather than `&mut
/// Session`) so a caller can pass `raw` borrowed from another field of the
/// same `Session` (e.g. a slice of `nsp_data`) without that overlapping with
/// the `&mut Cpu` borrow — which is what lets `switch_load_nca_from_nsp` boot
/// straight out of the already-staged NSP buffer instead of copying a
/// possibly hundreds-of-MB NCA first.
fn load_and_boot_nca(
    keys: &switch_core::keys::KeySet,
    cpu: &mut Cpu,
    last_error: &mut String,
    raw: &[u8],
) -> i64 {
    let nca = match Nca::parse_with_keys(raw, Some(keys)) {
        Ok(nca) => nca,
        Err(e) => {
            *last_error = e.to_string();
            return -1;
        }
    };
    let exefs_index = match nca.exefs_section_index() {
        Some(i) => i,
        None => {
            *last_error = "no ExeFS (PFS0) section in this NCA".into();
            return -1;
        }
    };
    let exefs = match nca.decrypt_pfs0_section(raw, keys, exefs_index) {
        Ok(v) => v,
        Err(e) => {
            *last_error = e.to_string();
            return -1;
        }
    };
    let pfs0 = match Pfs0::parse(&exefs) {
        Ok(p) => p,
        Err(e) => {
            *last_error = e.to_string();
            return -1;
        }
    };
    let modules = collect_modules(&pfs0, &exefs);
    if !modules.iter().any(|(name, _)| *name == "main") {
        *last_error = "no 'main' executable in this NCA's ExeFS".into();
        return -1;
    }

    // RomFS is optional (Meta/Control-only content, or a title with no
    // assets of its own, has none) and a failure to decrypt it shouldn't
    // block booting — the title just won't have its asset storage mounted.
    if let Some(romfs_index) = nca.romfs_section_index() {
        if let Ok(romfs) = nca.decrypt_romfs_section(raw, keys, romfs_index) {
            cpu.set_romfs(romfs);
        }
    }

    match cpu.boot_retail_program(&modules) {
        Ok(loaded) => {
            last_error.clear();
            loaded[0].entry as i64
        }
        Err(e) => {
            *last_error = e.to_string();
            -1
        }
    }
}

/// Decrypt a standalone Program NCA file (using whatever keys are loaded),
/// extract its ExeFS `main` executable and boot it. Takes ownership of the
/// buffer at `ptr` (do not free it afterwards) — an NCA can be hundreds of MB,
/// so this avoids a second copy the way `switch_load_nsp` does. Returns entry
/// address or -1 — check `switch_last_error` either way, since the entry can
/// legitimately be 0 for some NSO layouts.
///
/// This gets a real title as far as its own crt0; there is no Horizon service
/// surface for a full retail SDK program yet, so expect it to run until the
/// first missing service rather than to a menu.
#[no_mangle]
pub extern "C" fn switch_load_nca(handle: u32, ptr: *const u8, len: u32) -> i64 {
    let s = session(handle);
    if ptr.is_null() {
        s.last_error = "null NCA buffer".into();
        return -1;
    }
    // SAFETY: `ptr` came from `switch_alloc(len)` (same global allocator,
    // same Layout), and the caller no longer frees it.
    let owned = unsafe { Vec::from_raw_parts(ptr as *mut u8, len as usize, len as usize) };
    load_and_boot_nca(&s.keys, &mut s.cpu, &mut s.last_error, &owned)
}

/// Decrypt NSP file `index` as a Program NCA (using whatever keys are
/// loaded), extract its ExeFS `main` executable and boot it. Returns entry
/// address or -1 — check `switch_last_error` either way, since the entry can
/// legitimately be 0 for some NSO layouts.
///
/// This gets a real title as far as its own crt0; there is no Horizon service
/// surface for a full retail SDK program yet, so expect it to run until the
/// first missing service rather than to a menu.
#[no_mangle]
pub extern "C" fn switch_load_nca_from_nsp(handle: u32, index: u32) -> i64 {
    let s = session(handle);
    let (start, end) = match s.nsp_files.get(index as usize) {
        Some(f) => match (f.offset as usize).checked_add(f.size as usize) {
            Some(end) if end <= s.nsp_data.len() => (f.offset as usize, end),
            _ => {
                s.last_error = "NCA file entry exceeds the loaded NSP".into();
                return -1;
            }
        },
        None => {
            s.last_error = "no such NSP file index".into();
            return -1;
        }
    };

    // Title-key crypto: the key doesn't live in the NCA header's own key
    // area, and scene NSP releases almost always bundle the ticket that
    // unlocks it right next to the content — try that before falling back to
    // whatever an external title.keys provided.
    if let Ok(nca) = switch_core::nca::Nca::parse_with_keys(&s.nsp_data[start..end], Some(&s.keys)) {
        if nca.has_rights_id() && s.keys.title_key(&nca.rights_id).is_none() {
            if let Ok(title_key) =
                switch_core::ticket::find_and_decrypt_title_key(&nca.rights_id, &s.nsp_files, &s.nsp_data, &s.keys)
            {
                s.keys.title_keys.push((nca.rights_id, title_key));
            }
        }
    }

    let raw = &s.nsp_data[start..end];
    load_and_boot_nca(&s.keys, &mut s.cpu, &mut s.last_error, raw)
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

/// What the guest last asked the rumble motors to do, packed as
/// `(weak << 16) | strong` with each field 0..=1000.
///
/// Switch rumble drives two linear resonant actuators independently, which is
/// the same shape the browser's Gamepad API exposes as `dual-rumble`: the low
/// band maps onto `strongMagnitude` and the high band onto `weakMagnitude`.
/// Packed into one word so the page can poll it alongside input in a single
/// call.
#[no_mangle]
pub extern "C" fn switch_vibration(handle: u32) -> u32 {
    let s = session(handle);
    let (low, high) = s.cpu.vibration();
    let scale = |v: f32| (v * 1000.0).round().clamp(0.0, 1000.0) as u32;
    (scale(high) << 16) | scale(low)
}

/// The format of the PCM `switch_audio_pull` returns, packed as
/// `(channels << 24) | sample_rate`. Zero until the guest opens an audio
/// device — before that there is nothing to play and no rate to play it at.
#[no_mangle]
pub extern "C" fn switch_audio_format(handle: u32) -> u32 {
    let (rate, channels) = session(handle).cpu.audio_format();
    if rate == 0 {
        return 0;
    }
    (channels << 24) | (rate & 0x00ff_ffff)
}

/// Move up to `max_samples` interleaved 16-bit samples into `buf`, returning
/// how many were written. What is pulled is gone from the queue.
#[no_mangle]
pub extern "C" fn switch_audio_pull(handle: u32, buf: *mut u8, max_samples: u32) -> u32 {
    let s = session(handle);
    let mut samples = vec![0i16; max_samples as usize];
    let n = s.cpu.take_audio(&mut samples);
    let out = unsafe { std::slice::from_raw_parts_mut(buf, n * 2) };
    for (chunk, sample) in out.chunks_exact_mut(2).zip(samples.iter()) {
        chunk.copy_from_slice(&sample.to_le_bytes());
    }
    n as u32
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

// ---------------------------------------------------------------------------
// The emulated SD card.
//
// `Vfs` lives in memory for the life of a session, so on its own nothing the
// guest writes survives a reload. These are the two directions a host needs to
// back it with a real store: put files on the card before booting, and find
// out what the guest changed so only that has to be written back.
//
// Paths are guest paths — a `sdmc:` prefix and any number of slashes are
// normalized away, so "sdmc:/switch/x" and "/switch/x" are the same file.
// ---------------------------------------------------------------------------

/// Read a UTF-8 path out of guest-supplied wasm memory.
fn sd_path(ptr: *const u8, len: u32) -> String {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// Put a file on the SD card, replacing whatever is at that path. This is the
/// host's own load path — restoring the card from a store, or a file the user
/// dropped in — so it is deliberately **not** reported by
/// `switch_sd_take_changes_json`: a restored file has not changed.
#[no_mangle]
pub extern "C" fn switch_sd_write_file(
    handle: u32,
    path_ptr: *const u8,
    path_len: u32,
    data_ptr: *const u8,
    data_len: u32,
) -> i32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len as usize) };
    s.cpu.fs.write_file(&sd_path(path_ptr, path_len), data.to_vec());
    0
}

/// Create a directory on the SD card and any missing parents. Same "host load
/// path" reasoning as `switch_sd_write_file`: not reported as a change.
#[no_mangle]
pub extern "C" fn switch_sd_create_dir(handle: u32, path_ptr: *const u8, path_len: u32) -> i32 {
    session(handle).cpu.fs.create_dir(&sd_path(path_ptr, path_len));
    0
}

/// Delete a path from the SD card. Returns 1 if something was there, 0 if not.
#[no_mangle]
pub extern "C" fn switch_sd_remove(handle: u32, path_ptr: *const u8, path_len: u32) -> i32 {
    i32::from(session(handle).cpu.fs.remove(&sd_path(path_ptr, path_len)))
}

/// Size of a file on the SD card, or -1 when the path is not one (missing, or
/// a directory).
#[no_mangle]
pub extern "C" fn switch_sd_file_size(handle: u32, path_ptr: *const u8, path_len: u32) -> i64 {
    match session(handle).cpu.fs.size(&sd_path(path_ptr, path_len)) {
        Some(size) => size as i64,
        None => -1,
    }
}

/// Copy a file off the SD card into `buf`, starting at `offset`. Returns the
/// number of bytes copied, or -1 when the path is not a file. Call
/// `switch_sd_file_size` first to size the buffer; a file larger than `maxlen`
/// can be pulled in slices.
#[no_mangle]
pub extern "C" fn switch_sd_read_file(
    handle: u32,
    path_ptr: *const u8,
    path_len: u32,
    offset: u64,
    buf: *mut u8,
    maxlen: u32,
) -> i64 {
    let s = session(handle);
    let out = unsafe { std::slice::from_raw_parts_mut(buf, maxlen as usize) };
    match s.cpu.fs.read(&sd_path(path_ptr, path_len), offset, out) {
        Some(n) => n as i64,
        None => -1,
    }
}

/// How many paths the guest has changed and not yet had drained. Lets a host
/// skip the JSON round trip on the overwhelmingly common "nothing changed"
/// tick.
#[no_mangle]
pub extern "C" fn switch_sd_pending_changes(handle: u32) -> u32 {
    session(handle).cpu.fs.pending_changes() as u32
}

/// Drain what the guest has changed on the SD card since the last call, as
/// JSON: `[{"path":"/switch/a.json","kind":"file","size":12},
/// {"path":"/switch/d","kind":"dir","size":0},
/// {"path":"/switch/gone","kind":"deleted","size":0}]`.
///
/// Each entry says what is at the path *now*, so a host can store it or drop
/// it from its store without asking again. **The drain happens even if the
/// result does not fit in `buf`** — call `switch_sd_pending_changes` first and
/// size the buffer, or changes will be lost.
#[no_mangle]
pub extern "C" fn switch_sd_take_changes_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let mut out = Vec::from("[");
    for (i, change) in s.cpu.fs.take_changes().into_iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        let kind = match change.kind {
            Some(switch_core::vfs::ENTRY_TYPE_DIR) => "dir",
            Some(_) => "file",
            None => "deleted",
        };
        out.extend_from_slice(b"{\"path\":\"");
        // Guest paths can hold anything a filename can; escape what JSON
        // cannot carry raw rather than emitting a broken document.
        for &byte in change.path.as_bytes() {
            match byte {
                b'"' | b'\\' => {
                    out.push(b'\\');
                    out.push(byte);
                }
                0x00..=0x1F => out.extend_from_slice(format!("\\u{:04x}", byte).as_bytes()),
                _ => out.push(byte),
            }
        }
        out.extend_from_slice(b"\",\"kind\":\"");
        out.extend_from_slice(kind.as_bytes());
        out.extend_from_slice(b"\",\"size\":");
        out.extend_from_slice(change.size.to_string().as_bytes());
        out.push(b'}');
    }
    out.push(b']');
    let n = out.len().min(maxlen as usize);
    let dst = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    dst.copy_from_slice(&out[..n]);
    n as u32
}

/// Give the session the font `pl:u` serves as the shared system font, as the
/// contents of a TrueType/OpenType file. Homebrew that draws text reads it out
/// of pl's shared memory and hands it to FreeType, so this has to be set before
/// the guest calls `plInitialize` for any text to appear. Returns the number of
/// bytes taken.
#[no_mangle]
pub extern "C" fn switch_load_font(handle: u32, ptr: *const u8, len: u32) -> u32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    s.cpu.set_shared_font(data.to_vec());
    s.cpu.shared_font_len() as u32
}

/// Feed the host gamepad state to the guest. `buttons` is a `HidNpadButton`
/// bitfield in Horizon's order (A=1<<0, B=1<<1, X=1<<2, Y=1<<3, StickL=1<<4,
/// StickR=1<<5, L=1<<6, R=1<<7, ZL=1<<8, ZR=1<<9, Plus=1<<10, Minus=1<<11,
/// DpadLeft=1<<12, DpadUp=1<<13, DpadRight=1<<14, DpadDown=1<<15); sticks are
/// -32768..32767 with positive being right and up. Written to the memory-mapped
/// input register and, once libnx maps its hid shared memory, mirrored into the
/// `HidSharedMemory` layout so `padUpdate` sees it.
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

/// Set the wall-clock time `time:u`/`time:s` report, as POSIX seconds (UTC).
/// `wasm32-unknown-unknown` has no OS clock, so without the host calling this
/// (with `Date.now() / 1000`) the emulated RTC reads the Unix epoch.
#[no_mangle]
pub extern "C" fn switch_set_time(handle: u32, unix_seconds: i64) {
    session(handle).cpu.set_unix_time(unix_seconds);
}

/// Set the battery level `psm` reports. `wasm32-unknown-unknown` has no
/// battery API of its own; the host pushes this from the browser's Battery
/// Status API, where available.
#[no_mangle]
pub extern "C" fn switch_set_battery(handle: u32, percent: u32, charging: u32) {
    session(handle).cpu.set_battery(percent.min(100) as u8, charging != 0);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The session table is a [`SyncCell`]: sound in wasm, which is
    /// single-threaded, and *not* sound under `cargo test`, which is not. Two
    /// tests each calling `switch_new` mutate the same `Vec` at once, and the
    /// panic that eventually falls out crosses an `extern "C"` boundary and
    /// aborts the whole harness rather than failing one test. Every test that
    /// touches a session holds this for its duration.
    static HOST: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A session, and the lock that makes owning one exclusive.
    ///
    /// `switch_new` also installs a panic hook that captures the message for
    /// `switch_last_error`, which in a test swallows the assertion. Put the
    /// default back so failures are readable.
    fn new_session() -> (std::sync::MutexGuard<'static, ()>, u32) {
        // A test that fails while holding the lock poisons it; the next one
        // still needs it, and the failure has already been reported.
        let guard = HOST.lock().unwrap_or_else(|e| e.into_inner());
        let handle = switch_new();
        let _ = std::panic::take_hook();
        (guard, handle)
    }

    fn take_changes(handle: u32) -> String {
        let cap = 64 * 1024;
        let mut buf = vec![0u8; cap];
        let n = switch_sd_take_changes_json(handle, buf.as_mut_ptr(), cap as u32);
        String::from_utf8(buf[..n as usize].to_vec()).unwrap()
    }

    fn put(handle: u32, path: &str, data: &[u8]) {
        switch_sd_write_file(
            handle,
            path.as_ptr(),
            path.len() as u32,
            data.as_ptr(),
            data.len() as u32,
        );
    }

    #[test]
    fn the_sd_card_round_trips_through_the_host_entry_points() {
        let (_host, handle) = new_session();

        // Restoring the card is the host's own load path, so it must not come
        // back as a change — otherwise every restored file is written straight
        // back to the store it was just read from, on the first flush.
        put(handle, "sdmc:/switch/restored.txt", b"hello");
        assert_eq!(switch_sd_pending_changes(handle), 0);
        assert_eq!(take_changes(handle), "[]");

        // A guest write is. `IFile::Write` at offset 0 of a file the guest
        // created, which is what a config save looks like.
        {
            let cpu = &mut session(handle).cpu;
            assert!(cpu.fs.create_file("/switch/cfg.json", 0));
            cpu.fs.write("/switch/cfg.json", 0, br#"{"v":5}"#).unwrap();
            cpu.fs.guest_create_dir("/switch/saves");
            cpu.fs.remove("/switch/restored.txt");
        }
        assert_eq!(switch_sd_pending_changes(handle), 3);
        assert_eq!(
            take_changes(handle),
            r#"[{"path":"/switch/cfg.json","kind":"file","size":7},"#.to_owned()
                + r#"{"path":"/switch/restored.txt","kind":"deleted","size":0},"#
                + r#"{"path":"/switch/saves","kind":"dir","size":0}]"#
        );
        // Draining clears them: the page only ever writes back what is new.
        assert_eq!(switch_sd_pending_changes(handle), 0);
        assert_eq!(take_changes(handle), "[]");

        // Reading a file back out is how the page gets the bytes to store.
        let path = "/switch/cfg.json";
        assert_eq!(switch_sd_file_size(handle, path.as_ptr(), path.len() as u32), 7);
        let mut out = [0u8; 16];
        let n = switch_sd_read_file(
            handle,
            path.as_ptr(),
            path.len() as u32,
            0,
            out.as_mut_ptr(),
            out.len() as u32,
        );
        assert_eq!(n, 7);
        assert_eq!(&out[..7], br#"{"v":5}"#);

        // Offsets let a large save be pulled in slices.
        let n = switch_sd_read_file(
            handle,
            path.as_ptr(),
            path.len() as u32,
            4,
            out.as_mut_ptr(),
            out.len() as u32,
        );
        assert_eq!(n, 3);
        assert_eq!(&out[..3], b":5}");

        // A directory is not a file, and neither is a path with nothing at it.
        let dir = "/switch";
        assert_eq!(switch_sd_file_size(handle, dir.as_ptr(), dir.len() as u32), -1);
        let missing = "/switch/nope";
        assert_eq!(
            switch_sd_read_file(
                handle,
                missing.as_ptr(),
                missing.len() as u32,
                0,
                out.as_mut_ptr(),
                out.len() as u32
            ),
            -1
        );
        switch_free_session(handle);
    }

    #[test]
    fn a_path_json_cannot_carry_raw_is_escaped() {
        // Guest paths hold whatever a filename can, and the page parses this
        // with JSON.parse — one unescaped quote and the whole batch is lost.
        let (_host, handle) = new_session();
        session(handle).cpu.fs.create_file(r#"/switch/a"b\c"#, 0);
        let json = take_changes(handle);
        assert_eq!(json, r#"[{"path":"/switch/a\"b\\c","kind":"file","size":0}]"#);
        switch_free_session(handle);
    }
}
