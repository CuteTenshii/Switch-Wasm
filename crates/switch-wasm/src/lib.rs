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

use switch_core::cpu::{Cpu, TouchPoint};
use switch_core::elf::load_elf;
use switch_core::nca::Nca;
use switch_core::nsp::Pfs0;
use switch_core::source::{ByteSource, Window};

/// Framebuffer base address, width, height and stride (RGBA, little-endian).
pub use switch_core::{FB_BASE, FB_HEIGHT, FB_STRIDE, FB_WIDTH};
/// Memory-mapped input register: JS writes an ASCII key here, homebrew polls
/// and acknowledges (writes 0) when consumed.
pub const INPUT_ADDR: u32 = switch_core::INPUT_ADDR;

struct Session {
    /// The container the frontend has open, read from the host a range at a
    /// time. `None` until `switch_open_nsp`/`switch_open_nca`.
    container: Option<HostSource>,
    /// Parsed file table of the last NSP.
    nsp_files: Vec<switch_core::nsp::Pfs0File>,
    /// Keys loaded from prod.keys / title.keys, used to decrypt NCA headers.
    keys: switch_core::keys::KeySet,
    /// The title's name, developer and icon, from the last Control NCA read.
    /// Cached because the icon is fetched separately from the text: JS needs
    /// its size before it can hand over a buffer to copy it into.
    control: Option<switch_core::control::Control>,
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

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" {
    /// Read `len` bytes at `offset` of host file `file`, into wasm memory at
    /// `ptr`, returning how many were actually read. File 0 is the open
    /// container; the rest are system data archives the host has added.
    ///
    /// This is the only import the module declares. It exists because a
    /// retail container cannot be handed over as a buffer — it is larger than
    /// the whole wasm32 address space — so the browser keeps the file where it
    /// is and serves ranges out of it synchronously (`FileReaderSync`, in the
    /// worker that owns this module).
    fn host_read(file: u32, offset: u64, ptr: *mut u8, len: u32) -> u32;
}

/// The same read, for host builds (`cargo test -p switch-wasm`), which have
/// no JS behind them: it serves whatever [`set_host_container`] installed.
///
/// # Safety
/// `ptr` must be valid for writes of `len` bytes.
#[cfg(not(target_arch = "wasm32"))]
unsafe fn host_read(file: u32, offset: u64, ptr: *mut u8, len: u32) -> u32 {
    if file != 0 {
        return 0; // host builds serve only the container
    }
    // SAFETY: single-threaded wasm; see the `SESSIONS` comment. Host builds
    // hold the test lock while they use this.
    let data = unsafe { &*HOST_CONTAINER.get() };
    if offset >= data.len() as u64 {
        return 0;
    }
    let start = offset as usize;
    let n = (len as usize).min(data.len() - start);
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().add(start), ptr, n) };
    n as u32
}

#[cfg(not(target_arch = "wasm32"))]
static HOST_CONTAINER: SyncCell<Vec<u8>> = SyncCell::new(Vec::new());

/// Install the bytes host builds serve as "the container the host has open",
/// so the streaming loader can be exercised without a browser. The browser
/// build has a real file behind `host_read` and never calls this.
#[cfg(not(target_arch = "wasm32"))]
pub fn set_host_container(data: Vec<u8>) {
    // SAFETY: single-threaded wasm; see the `SESSIONS` comment.
    unsafe { *HOST_CONTAINER.get() = data };
}

/// A [`ByteSource`] over the container the host has open.
///
/// Stateless and `Copy` — the size is all there is to it, since every read
/// goes straight back out to the host — which is what lets a session hand
/// copies of it to the file table, the ticket lookup and the RomFS chain
/// without any of them borrowing the session.
#[derive(Debug, Clone, Copy)]
struct HostSource {
    /// Which host file this reads: 0 is the open container, and each system
    /// data archive the host added has its own.
    file: u32,
    len: u64,
}

impl ByteSource for HostSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, switch_core::Error> {
        if offset >= self.len {
            return Ok(0);
        }
        let want = ((out.len() as u64).min(self.len - offset)) as usize;
        let mut done = 0;
        while done < want {
            // The host reports a length in a `u32`; nothing this side asks
            // for that much at once, but the loop is also what absorbs a
            // short read at the host's own cache-chunk boundary.
            let ask = (want - done).min(u32::MAX as usize);
            let got = unsafe {
                host_read(self.file, offset + done as u64, out[done..].as_mut_ptr(), ask as u32)
            } as usize;
            if got == 0 {
                break;
            }
            done += got;
        }
        if done != want {
            return Err(switch_core::Error::Io(format!(
                "host read of {} bytes at {:#x} returned {}",
                want, offset, done
            )));
        }
        Ok(done)
    }
}

/// Allocate `len` bytes of wasm linear memory for passing buffers in from JS.
///
/// Returns null for a request this target cannot serve, rather than trapping
/// on the way there: `Layout` rejects any size above `isize::MAX` — 2 GiB on
/// wasm32 — and the `.unwrap()` that used to follow lowered to `unreachable`,
/// which took the module down with `RuntimeError: unreachable executed` and
/// nothing to say what had asked for what. Callers must check.
#[no_mangle]
pub extern "C" fn switch_alloc(len: u32) -> *mut u8 {
    match Layout::from_size_align(len as usize, 1) {
        Ok(layout) => unsafe { alloc(layout) },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a buffer previously returned by `switch_alloc`. A null pointer (an
/// allocation that was refused) frees nothing.
#[no_mangle]
pub extern "C" fn switch_free(ptr: *mut u8, len: u32) {
    if ptr.is_null() {
        return;
    }
    let Ok(layout) = Layout::from_size_align(len as usize, 1) else {
        return;
    };
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
        container: None,
        nsp_files: Vec::new(),
        keys: switch_core::keys::KeySet::default(),
        control: None,
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

/// How many message bytes fit in a `maxlen`-byte buffer that also has to hold
/// a terminating NUL.
///
/// The NUL comes out of the **buffer**, not out of the message. Writing this
/// as `len.min(maxlen).saturating_sub(1)` instead took the byte off the
/// message every time, however much room was left: with 512 bytes free,
/// "no container is open" reached the console as "no container is ope".
fn nul_reserved(maxlen: u32) -> usize {
    (maxlen as usize).saturating_sub(1)
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
        let n = len.min(nul_reserved(maxlen));
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
    let n = bytes.len().min(nul_reserved(maxlen));
    if n > 0 && !buf.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n);
            *buf.add(n) = 0;
        }
    }
    n as u32
}

/// Open the `size`-byte container the host has ready as an NSP: read its
/// PFS0 file table and keep it. Returns 0 on success, -1 on error.
///
/// Nothing is copied into wasm memory — the file stays with the host and is
/// read through `host_read` from here on. It has to be: a retail container
/// runs to several gigabytes, which is more than this target can address at
/// all, let alone allocate in one buffer.
#[no_mangle]
pub extern "C" fn switch_open_nsp(handle: u32, size: u64) -> i32 {
    let s = session(handle);
    let container = HostSource { file: 0, len: size };
    s.container = Some(container);
    s.nsp_files = Vec::new();
    s.control = None;
    match Pfs0::read_from(&container) {
        Ok(pfs0) => {
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

/// Open the `size`-byte container the host has ready as a single standalone
/// `.nca` — no file table, the whole container is the NCA. Returns 0.
#[no_mangle]
pub extern "C" fn switch_open_nca(handle: u32, size: u64) -> i32 {
    let s = session(handle);
    s.container = Some(HostSource { file: 0, len: size });
    s.nsp_files = Vec::new();
    s.control = None;
    s.last_error.clear();
    0
}

/// Register host file `file` as a system data archive: parse it as an NCA,
/// take its RomFS, and file it under its title id for
/// `OpenDataStorageByDataId` to serve.
///
/// This is the content a title mounts that is not its own — the system's Mii
/// and amiibo models, the shared bad-word lists. Each lives in its own NCA on
/// a console's NAND, so the frontend hands them over one file at a time and
/// nothing is read until a title actually asks for one.
///
/// Returns 0 if it was registered, -1 if the file is not a data archive this
/// build can read.
#[no_mangle]
pub extern "C" fn switch_add_archive(handle: u32, file: u32, size: u64) -> i32 {
    let s = session(handle);
    let src = HostSource { file, len: size };
    let nca = match Nca::parse_source(&src, Some(&s.keys)) {
        Ok(nca) => nca,
        Err(e) => {
            s.last_error = e.to_string();
            return -1;
        }
    };
    use switch_core::nca::ContentType;
    if !matches!(nca.content_type, ContentType::Data | ContentType::PublicData) {
        s.last_error = format!("not a data archive (content type {})", nca.content_type.name());
        return -1;
    }
    let Some(index) = nca.romfs_section_index() else {
        s.last_error = "data archive has no RomFS section".into();
        return -1;
    };
    match nca.romfs_source(src, &s.keys, index) {
        Ok(romfs) => {
            s.cpu.add_data_archive(nca.title_id, Box::new(romfs));
            s.last_error.clear();
            0
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Register a system data archive from bytes the host is holding, rather than
/// from a file it can ask for again later.
///
/// The difference is the whole point of a NAND. A browser will not hand a page
/// a file it was not asked for, so an archive registered through
/// [`switch_add_archive`] — which keeps only a reference to a `File` — is gone
/// the moment the page reloads, and a firmware dump has to be re-picked every
/// session. Bytes can be kept. This is what a console's NAND is here: content
/// the host stores on the emulator's behalf and hands back unprompted.
///
/// Returns the archive's title id, which is what a title asks for it by, or 0
/// if the bytes are not a data archive this build can read.
#[no_mangle]
pub extern "C" fn switch_nand_add_archive(handle: u32, ptr: *const u8, len: u32) -> u64 {
    let s = session(handle);
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec();
    let nca = match Nca::parse_with_keys(&bytes, Some(&s.keys)) {
        Ok(nca) => nca,
        Err(e) => {
            s.last_error = e.to_string();
            return 0;
        }
    };
    use switch_core::nca::ContentType;
    if !matches!(nca.content_type, ContentType::Data | ContentType::PublicData) {
        s.last_error = format!("not a data archive (content type {})", nca.content_type.name());
        return 0;
    }
    let Some(index) = nca.romfs_section_index() else {
        s.last_error = "data archive has no RomFS section".into();
        return 0;
    };
    let title_id = nca.title_id;
    match nca.romfs_source(switch_core::source::MemSource(bytes), &s.keys, index) {
        Ok(romfs) => {
            s.cpu.add_data_archive(title_id, Box::new(romfs));
            s.last_error.clear();
            title_id
        }
        Err(e) => {
            s.last_error = e.to_string();
            0
        }
    }
}

/// What a firmware NCA is, without reading the whole thing.
///
/// The host has to sort a firmware dump into what it should keep and what it
/// should not, and a dump is mostly metadata: reading every file to find out
/// would mean pulling gigabytes through the page. This reads the header out of
/// the `File` the host still holds and answers from that.
///
/// Writes the content type to `kind_out` — 0 program, 1 data archive, 2
/// anything else — and returns the title id, or 0 if the file is not an NCA
/// this build can read.
#[no_mangle]
pub extern "C" fn switch_nand_identify(
    handle: u32,
    file: u32,
    size: u64,
    kind_out: *mut u32,
) -> u64 {
    let s = session(handle);
    let src = HostSource { file, len: size };
    let nca = match Nca::parse_source(&src, Some(&s.keys)) {
        Ok(nca) => nca,
        Err(e) => {
            s.last_error = e.to_string();
            return 0;
        }
    };
    use switch_core::nca::ContentType;
    let kind = match nca.content_type {
        ContentType::Program => 0,
        ContentType::Data | ContentType::PublicData => 1,
        _ => 2,
    };
    if !kind_out.is_null() {
        unsafe { *kind_out = kind };
    }
    s.last_error.clear();
    nca.title_id
}

/// Boot a Program NCA the host is holding the bytes of — a title installed on
/// the NAND rather than one opened out of a container the user just picked.
///
/// This is what makes an applet launchable: the Home Menu and the Mii editor
/// ship as bare NCAs inside firmware, so there is no NSP to open them from,
/// and until the NAND kept them there was nothing to launch.
///
/// Returns the entry address, or -1 with the reason in `switch_last_error`.
#[no_mangle]
pub extern "C" fn switch_nand_launch(handle: u32, ptr: *const u8, len: u32) -> i64 {
    let s = session(handle);
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec();
    load_and_boot_nca(
        &s.keys,
        &mut s.cpu,
        &mut s.last_error,
        switch_core::source::MemSource(bytes),
    )
}

/// The open container as a source, or an error recorded in the session.
fn container(s: &mut Session) -> Option<HostSource> {
    match s.container {
        Some(c) => Some(c),
        None => {
            s.last_error = "no container is open".into();
            None
        }
    }
}

/// A source over NSP file `index` — the window an NCA is read through.
fn nsp_file_source(s: &mut Session, index: u32) -> Option<Window<HostSource>> {
    let container = container(s)?;
    let f = match s.nsp_files.get(index as usize) {
        Some(f) => f,
        None => {
            s.last_error = "no such NSP file index".into();
            return None;
        }
    };
    match Window::new(container, f.offset, f.size, &f.name) {
        Ok(w) => Some(w),
        Err(e) => {
            s.last_error = e.to_string();
            None
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

/// Read a slice of NSP file `index` starting at `file_offset` into `buf`
/// (clamped to the file). Used to grab just an NCA header rather than the
/// whole (potentially many-gigabyte) payload. Returns bytes copied or -1.
#[no_mangle]
pub extern "C" fn switch_read_file(
    handle: u32,
    index: u32,
    file_offset: u64,
    buf: *mut u8,
    maxlen: u32,
) -> i64 {
    let s = session(handle);
    let Some(file) = nsp_file_source(s, index) else {
        return -1;
    };
    if buf.is_null() || maxlen == 0 || file_offset >= file.len() {
        return 0;
    }
    let n = (maxlen as u64).min(file.len() - file_offset) as usize;
    // SAFETY: JS allocated `maxlen` bytes at `buf` with `switch_alloc`, and
    // `n` is no larger.
    let out = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    match file.read_at(file_offset, out) {
        Ok(got) => got as i64,
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
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

/// Read the title's control data — name, publisher, version and icon — from
/// the Control NCA in the open container and cache it in the session, for
/// `switch_control_json` and `switch_control_icon` to read back.
///
/// Returns 0 on success, -1 when the container has no readable Control NCA.
/// That includes having no `prod.keys` loaded: an NCA's content type is
/// inside its encrypted header, so without the header key none of the
/// container's NCAs can even be identified.
#[no_mangle]
pub extern "C" fn switch_load_control_from_nsp(handle: u32) -> i32 {
    let s = session(handle);
    s.control = None;
    let Some(container) = container(s) else {
        return -1;
    };
    let found = switch_core::control::find_control_nca(&s.nsp_files, &container, &s.keys);
    let Some((index, nca)) = found else {
        s.last_error =
            "no Control NCA in this container (or its header couldn't be decrypted — load prod.keys)"
                .into();
        return -1;
    };
    // Title-key crypto: as when booting the Program NCA, the section key
    // isn't in the header's own key area and the ticket that unlocks it
    // ships next to the content.
    if nca.has_rights_id() && s.keys.resolved_title_key(&nca.rights_id).is_none() {
        if let Ok(title_key) = switch_core::ticket::find_and_decrypt_title_key_from(
            &nca.rights_id,
            &s.nsp_files,
            &container,
            &s.keys,
        ) {
            s.keys.add_resolved_title_key(nca.rights_id, title_key);
        }
    }
    let Some(file) = nsp_file_source(s, index as u32) else {
        return -1;
    };
    match switch_core::control::Control::from_source(file, &s.keys) {
        Ok(control) => {
            s.control = Some(control);
            s.last_error.clear();
            0
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Same, for a container opened as a single standalone Control NCA
/// (`switch_open_nca`) rather than as a container holding one.
#[no_mangle]
pub extern "C" fn switch_load_control_from_nca(handle: u32) -> i32 {
    let s = session(handle);
    s.control = None;
    let Some(container) = container(s) else {
        return -1;
    };
    match switch_core::control::Control::from_source(container, &s.keys) {
        Ok(control) => {
            s.control = Some(control);
            s.last_error.clear();
            0
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// The cached control data as JSON, or `{}` when none has been read. The
/// icon itself comes from `switch_control_icon`; `icon_size` here is the
/// buffer that needs.
#[no_mangle]
pub extern "C" fn switch_control_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let mut out = Vec::new();
    let Some(control) = &s.control else {
        out.extend_from_slice(b"{}");
        return write_into(buf, maxlen, &out);
    };
    let nacp = &control.nacp;
    out.extend_from_slice(b"{\"title_id\":\"");
    out.extend_from_slice(format!("{:016x}", control.title_id).as_bytes());
    out.extend_from_slice(b"\",\"name\":\"");
    json_escape(&control.name, &mut out);
    out.extend_from_slice(b"\",\"publisher\":\"");
    json_escape(&control.publisher, &mut out);
    out.extend_from_slice(b"\",\"language\":\"");
    json_escape(control.language, &mut out);
    out.extend_from_slice(b"\",\"version\":\"");
    json_escape(&nacp.display_version, &mut out);
    out.extend_from_slice(b"\",\"isbn\":\"");
    json_escape(&nacp.isbn, &mut out);
    out.extend_from_slice(b"\",\"error_code_category\":\"");
    json_escape(&nacp.application_error_code_category, &mut out);
    out.extend_from_slice(b"\",\"startup_user_account\":\"");
    out.extend_from_slice(nacp.startup_user_account.name().as_bytes());
    out.extend_from_slice(b"\",\"screenshot\":\"");
    out.extend_from_slice(nacp.screenshot.name().as_bytes());
    out.extend_from_slice(b"\",\"video_capture\":\"");
    out.extend_from_slice(nacp.video_capture.name().as_bytes());
    out.extend_from_slice(b"\",\"demo\":");
    out.extend_from_slice(if nacp.is_demo { b"true" } else { b"false" });
    out.extend_from_slice(b",\"languages\":[");
    for (i, title) in nacp.titles.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b",");
        }
        out.extend_from_slice(b"\"");
        json_escape(title.language, &mut out);
        out.extend_from_slice(b"\"");
    }
    out.extend_from_slice(b"],\"ratings\":[");
    for (i, rating) in nacp.ratings.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b",");
        }
        out.extend_from_slice(b"{\"organisation\":\"");
        json_escape(rating.organisation, &mut out);
        out.extend_from_slice(b"\",\"age\":");
        out.extend_from_slice(rating.age.to_string().as_bytes());
        out.extend_from_slice(b"}");
    }
    out.extend_from_slice(b"],\"add_on_content_base_id\":\"");
    out.extend_from_slice(format!("{:016x}", nacp.add_on_content_base_id).as_bytes());
    out.extend_from_slice(b"\",\"save_data_owner_id\":\"");
    out.extend_from_slice(format!("{:016x}", nacp.save_data_owner_id).as_bytes());
    out.extend_from_slice(b"\",\"user_save_size\":");
    out.extend_from_slice(nacp.user_account_save_data_size.to_string().as_bytes());
    out.extend_from_slice(b",\"user_save_journal_size\":");
    out.extend_from_slice(nacp.user_account_save_data_journal_size.to_string().as_bytes());
    out.extend_from_slice(b",\"device_save_size\":");
    out.extend_from_slice(nacp.device_save_data_size.to_string().as_bytes());
    out.extend_from_slice(b",\"device_save_journal_size\":");
    out.extend_from_slice(nacp.device_save_data_journal_size.to_string().as_bytes());
    out.extend_from_slice(b",\"bcat_storage_size\":");
    out.extend_from_slice(nacp.bcat_delivery_cache_storage_size.to_string().as_bytes());
    out.extend_from_slice(b",\"icon_mime\":\"");
    out.extend_from_slice(control.icon_mime().as_bytes());
    out.extend_from_slice(b"\",\"icon_size\":");
    out.extend_from_slice(control.icon.len().to_string().as_bytes());
    out.extend_from_slice(b"}");
    write_into(buf, maxlen, &out)
}

/// Copy the cached title icon into `buf`. Returns bytes copied, or -1 when
/// no control data has been read.
#[no_mangle]
pub extern "C" fn switch_control_icon(handle: u32, buf: *mut u8, maxlen: u32) -> i64 {
    let s = session(handle);
    let Some(control) = &s.control else {
        return -1;
    };
    let n = control.icon.len().min(maxlen as usize);
    if n > 0 && !buf.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(control.icon.as_ptr(), buf, n);
        }
    }
    n as i64
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
/// Program NCA's ExeFS from `nca_src` (the whole, still-encrypted NCA),
/// extract `main` and boot it. Returns entry address or -1.
///
/// `nca_src` is a source rather than a buffer, and is consumed: the title's
/// RomFS outlives this call as a decrypting view of it, so the container has
/// to stay readable for as long as the title runs. Nothing but the ExeFS is
/// ever held in memory.
///
/// Takes `keys`/`cpu`/`last_error` as separate borrows rather than `&mut
/// Session` so the caller can keep reading the session's file table while
/// this holds `&mut Cpu`.
fn load_and_boot_nca<S: ByteSource + 'static>(
    keys: &switch_core::keys::KeySet,
    cpu: &mut Cpu,
    last_error: &mut String,
    nca_src: S,
) -> i64 {
    let nca = match Nca::parse_source(&nca_src, Some(keys)) {
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
    let exefs = match nca.read_pfs0_section(&nca_src, keys, exefs_index) {
        Ok(v) => v,
        Err(e) => {
            *last_error = e.to_string();
            return -1;
        }
    };
    // Say whether the bytes about to be executed were checked, not just that
    // they decrypted: the master hash only vouches for the section's hash
    // table, and a fault inside a title's own crt0 is worth chasing in the
    // title only once the image it ran on is known to be intact.
    match nca.pfs0_hash_coverage(exefs_index) {
        Some((block, blocks)) => cpu.diagnostic(&format!(
            "[exefs] {:#x} bytes, {} blocks of {:#x} verified against the section hash table",
            exefs.len(),
            blocks,
            block
        )),
        None => cpu.diagnostic(&format!(
            "[exefs] {:#x} bytes — hash table geometry unrecognised, contents NOT verified",
            exefs.len()
        )),
    }

    let pfs0 = match Pfs0::parse(&exefs) {
        Ok(p) => p,
        Err(e) => {
            *last_error = e.to_string();
            return -1;
        }
    };
    // `pm` reports this, and a system applet derives its own `AppletId` from
    // it — without it every title looks like the same default program.
    cpu.set_program_id(nca.program_id);

    let modules = collect_modules(&pfs0, &exefs);
    // Everything the ExeFS holds, next to what was actually loaded from it. A
    // title whose `sdk` or `subsdk0` were silently left behind aborts in its
    // own init looking exactly like a title that hit a missing service.
    cpu.diagnostic(&format!(
        "[exefs] entries: {} — loading: {}",
        pfs0.files
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        modules
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(", ")
    ));
    if !modules.iter().any(|(name, _)| *name == "main") {
        *last_error = "no 'main' executable in this NCA's ExeFS".into();
        return -1;
    }

    // RomFS is optional (Meta/Control-only content, or a title with no
    // assets of its own, has none) and a failure to decrypt it shouldn't
    // block booting — the title just won't have its asset storage mounted.
    // The source is handed to the CPU rather than decrypted up front: a
    // retail RomFS is the entire game, and the guest reads it a range at a
    // time through `IStorage` anyway.
    if let Some(romfs_index) = nca.romfs_section_index() {
        match nca.romfs_source(nca_src, keys, romfs_index) {
            Ok(romfs) => cpu.set_romfs_source(Box::new(romfs)),
            Err(e) => cpu.diagnostic(&format!("romfs unavailable: {}", e)),
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

/// Decrypt the open container as a standalone Program NCA (using whatever
/// keys are loaded), extract its ExeFS `main` executable and boot it. Returns
/// entry address or -1 — check `switch_last_error` either way, since the
/// entry can legitimately be 0 for some NSO layouts.
///
/// This gets a real title as far as its own crt0; there is no Horizon service
/// surface for a full retail SDK program yet, so expect it to run until the
/// first missing service rather than to a menu.
#[no_mangle]
pub extern "C" fn switch_load_nca(handle: u32) -> i64 {
    let s = session(handle);
    let Some(container) = container(s) else {
        return -1;
    };
    load_and_boot_nca(&s.keys, &mut s.cpu, &mut s.last_error, container)
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
    let Some(container) = container(s) else {
        return -1;
    };
    let Some(nca_src) = nsp_file_source(s, index) else {
        return -1;
    };

    // Title-key crypto: the key doesn't live in the NCA header's own key
    // area, and scene NSP releases almost always bundle the ticket that
    // unlocks it right next to the content — try that before falling back to
    // whatever an external title.keys provided.
    if let Ok(nca) = Nca::parse_source(&nca_src, Some(&s.keys)) {
        if nca.has_rights_id() && s.keys.resolved_title_key(&nca.rights_id).is_none() {
            if let Ok(title_key) = switch_core::ticket::find_and_decrypt_title_key_from(
                &nca.rights_id,
                &s.nsp_files,
                &container,
                &s.keys,
            ) {
                s.keys.add_resolved_title_key(nca.rights_id, title_key);
            }
        }
    }

    load_and_boot_nca(&s.keys, &mut s.cpu, &mut s.last_error, nca_src)
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

/// Enable/disable the block translator (see `switch_core::cpu::jit`).
///
/// On by default. The host-side switch reads `SWITCH_NO_JIT` from the
/// environment, which a browser does not have, so this is the only way a page
/// can fall back to the plain interpreter — worth having when a title
/// misbehaves and the question is whether translation is why.
#[no_mangle]
pub extern "C" fn switch_set_jit(handle: u32, enabled: u32) {
    session(handle).cpu.set_jit_enabled(enabled != 0);
}

/// What the translator has been doing, as JSON.
#[no_mangle]
pub extern "C" fn switch_jit_stats_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let stats = s.cpu.jit_stats();
    let json = format!(
        "{{\"enabled\":{},\"blocks\":{},\"translated\":{},\"executed\":{},\"invalidated\":{}}}",
        s.cpu.jit_enabled(),
        stats.blocks,
        stats.translated,
        stats.executed,
        stats.invalidated
    );
    write_into(buf, maxlen, json.as_bytes())
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
    let changes = session(handle).cpu.fs.take_changes();
    write_changes_json(&changes, buf, maxlen)
}

/// Serialize drained [`Change`](switch_core::vfs::Change)s into `buf`.
///
/// Shared by the SD card and by save data, because they are the same
/// question asked of different storage: what is at this path now, so a host
/// can store it or drop it from its store without asking again.
fn write_changes_json(changes: &[switch_core::vfs::Change], buf: *mut u8, maxlen: u32) -> u32 {
    let mut out = Vec::from("[");
    for (i, change) in changes.iter().enumerate() {
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

// ---------- save data ----------
//
// The same shape as the SD card above, with a save id in front of every call.
// A console keeps saves on its NAND rather than its card, and they are the
// only writable storage a title has that another title cannot see — so they
// are stored separately, and a path means nothing without the id it belongs
// to.

/// Every save the running session has opened, as JSON: `["0100000000001000"]`.
///
/// A save is created on first open, so this is also the list of what there is
/// to persist — a host drains and stores each of these in turn.
#[no_mangle]
pub extern "C" fn switch_save_ids_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let mut ids = session(handle).cpu.save_ids();
    ids.sort_unstable();
    let mut out = Vec::from("[");
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(format!("\"{id:016x}\"").as_bytes());
    }
    out.push(b']');
    let n = out.len().min(maxlen as usize);
    let dst = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    dst.copy_from_slice(&out[..n]);
    n as u32
}

/// How many paths the guest has changed in this save and not yet had drained.
#[no_mangle]
pub extern "C" fn switch_save_pending_changes(handle: u32, save_id: u64) -> u32 {
    session(handle).cpu.save_data_mut(save_id).pending_changes() as u32
}

/// Drain what the guest has changed in this save, in the same JSON as
/// `switch_sd_take_changes_json`. **The drain happens even if the result does
/// not fit in `buf`** — size it from `switch_save_pending_changes` first.
#[no_mangle]
pub extern "C" fn switch_save_take_changes_json(
    handle: u32,
    save_id: u64,
    buf: *mut u8,
    maxlen: u32,
) -> u32 {
    let changes = session(handle).cpu.save_data_mut(save_id).take_changes();
    write_changes_json(&changes, buf, maxlen)
}

/// Put a file into a save, creating the save if this is the first thing in it.
/// The host's own load path, so — like `switch_sd_write_file` — it is
/// deliberately not reported as a change: a restored file has not changed.
#[no_mangle]
pub extern "C" fn switch_save_write_file(
    handle: u32,
    save_id: u64,
    path_ptr: *const u8,
    path_len: u32,
    data_ptr: *const u8,
    data_len: u32,
) -> i32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len as usize) };
    let path = sd_path(path_ptr, path_len);
    s.cpu.save_data_mut(save_id).write_file(&path, data.to_vec());
    0
}

/// Create a directory in a save and any missing parents. Not reported as a
/// change, for the same reason as `switch_save_write_file`.
#[no_mangle]
pub extern "C" fn switch_save_create_dir(
    handle: u32,
    save_id: u64,
    path_ptr: *const u8,
    path_len: u32,
) -> i32 {
    let s = session(handle);
    let path = sd_path(path_ptr, path_len);
    s.cpu.save_data_mut(save_id).create_dir(&path);
    0
}

/// Size of a file in a save, or -1 when the path is not one.
#[no_mangle]
pub extern "C" fn switch_save_file_size(
    handle: u32,
    save_id: u64,
    path_ptr: *const u8,
    path_len: u32,
) -> i64 {
    let s = session(handle);
    let path = sd_path(path_ptr, path_len);
    match s.cpu.save_data_mut(save_id).size(&path) {
        Some(size) => size as i64,
        None => -1,
    }
}

/// Copy a file out of a save into `buf`, starting at `offset`. Returns the
/// bytes copied, or -1 when the path is not a file.
#[no_mangle]
pub extern "C" fn switch_save_read_file(
    handle: u32,
    save_id: u64,
    path_ptr: *const u8,
    path_len: u32,
    offset: u64,
    buf: *mut u8,
    maxlen: u32,
) -> i64 {
    let s = session(handle);
    let path = sd_path(path_ptr, path_len);
    let out = unsafe { std::slice::from_raw_parts_mut(buf, maxlen as usize) };
    match s.cpu.save_data_mut(save_id).read(&path, offset, out) {
        Some(n) => n as i64,
        None => -1,
    }
}

/// Give a fresh session a save it had in an earlier one, so a host can restore
/// before the guest asks. Returns 0.
#[no_mangle]
pub extern "C" fn switch_save_create(handle: u32, save_id: u64) -> i32 {
    session(handle).cpu.save_data_mut(save_id);
    0
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

/// Feed the host's touchscreen contacts to the guest. `ptr` points at `count`
/// packed `u32` triples - `finger_id`, `x`, `y` - in the console's 1280x720
/// digitizer space (`TOUCH_SCREEN_WIDTH`/`HEIGHT` in `cpu/mod.rs`), *not* in
/// whatever resolution the guest happens to be presenting at. `count` above
/// `TOUCH_MAX` (16) is truncated.
///
/// A lift is `count` = 0: the state is republished with no contacts. The guest
/// only sees any of this once it has mapped hid's shared memory, and nothing is
/// buffered until then, so the host has to keep calling while a finger is down.
#[no_mangle]
pub extern "C" fn switch_set_touch(handle: u32, ptr: *const u32, count: u32) {
    let n = (count as usize).min(switch_core::cpu::TOUCH_MAX);
    let mut points = [TouchPoint::default(); switch_core::cpu::TOUCH_MAX];
    if n > 0 && !ptr.is_null() {
        let raw = unsafe { std::slice::from_raw_parts(ptr, n * 3) };
        for (i, point) in points[..n].iter_mut().enumerate() {
            point.finger_id = raw[i * 3];
            point.x = raw[i * 3 + 1];
            point.y = raw[i * 3 + 2];
        }
    }
    session(handle).cpu.set_touch_state(&points[..n]);
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
    fn save_data_round_trips_and_stays_out_of_the_sd_card() {
        const SAVE: u64 = 0x0100_0000_0000_1000;
        let (_host, handle) = new_session();

        // Restoring is the host's own load path, so it must not come back as a
        // change — otherwise every restored file is written straight back to
        // the store it was just read from, on the first flush.
        let path = "/settings.dat";
        let body = b"saved";
        assert_eq!(
            switch_save_write_file(
                handle,
                SAVE,
                path.as_ptr(),
                path.len() as u32,
                body.as_ptr(),
                body.len() as u32,
            ),
            0
        );
        assert_eq!(switch_save_pending_changes(handle, SAVE), 0);

        // Opening the save is enough to have one to persist.
        let mut ids = [0u8; 64];
        let n = switch_save_ids_json(handle, ids.as_mut_ptr(), ids.len() as u32) as usize;
        assert_eq!(std::str::from_utf8(&ids[..n]).unwrap(), r#"["0100000000001000"]"#);

        // A guest write is a change, and it lands in the save rather than on
        // the card — the two are different storage, and a title's save is not
        // something the next title to mount the card should find.
        session(handle).cpu.save_data_mut(SAVE).write("/settings.dat", 0, b"12345").unwrap();
        assert_eq!(switch_save_pending_changes(handle, SAVE), 1);
        let mut buf = [0u8; 256];
        let n = switch_save_take_changes_json(handle, SAVE, buf.as_mut_ptr(), buf.len() as u32);
        assert_eq!(
            std::str::from_utf8(&buf[..n as usize]).unwrap(),
            r#"[{"path":"/settings.dat","kind":"file","size":5}]"#
        );
        assert_eq!(switch_save_pending_changes(handle, SAVE), 0);
        assert_eq!(session(handle).cpu.fs.entry_type("/settings.dat"), None);

        // And reading it back is how the host gets the bytes to store.
        assert_eq!(
            switch_save_file_size(handle, SAVE, path.as_ptr(), path.len() as u32),
            5
        );
        let mut out = [0u8; 16];
        let read = switch_save_read_file(
            handle,
            SAVE,
            path.as_ptr(),
            path.len() as u32,
            0,
            out.as_mut_ptr(),
            out.len() as u32,
        );
        assert_eq!(read, 5);
        assert_eq!(&out[..5], b"12345");
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

    /// Build a PFS0 container holding `files`, laid out the way a real `.nsp`
    /// is: header, entry table, string table, then the payloads.
    fn build_nsp(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut names = Vec::new();
        let mut name_offsets = Vec::new();
        for (name, _) in files {
            name_offsets.push(names.len() as u32);
            names.extend_from_slice(name.as_bytes());
            names.push(0);
        }
        let entries_end = 0x10 + files.len() * 24;
        let payload_base = entries_end + names.len();

        let mut image = Vec::new();
        image.extend_from_slice(&0x3053_4650u32.to_le_bytes()); // "PFS0"
        image.extend_from_slice(&(files.len() as u32).to_le_bytes());
        image.extend_from_slice(&(names.len() as u32).to_le_bytes());
        image.extend_from_slice(&0u32.to_le_bytes());
        let mut at = payload_base as u64;
        for (i, (_, payload)) in files.iter().enumerate() {
            image.extend_from_slice(&at.to_le_bytes());
            image.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            image.extend_from_slice(&name_offsets[i].to_le_bytes());
            image.extend_from_slice(&0u32.to_le_bytes());
            at += payload.len() as u64;
        }
        image.extend_from_slice(&names);
        for (_, payload) in files {
            image.extend_from_slice(payload);
        }
        image
    }

    /// The container path end to end, through the same `host_read` the
    /// browser serves from a file on disk: nothing but the header and the
    /// ranges actually asked for ever crosses into this side.
    #[test]
    fn a_container_is_read_through_the_host_without_being_staged() {
        let (_host, handle) = new_session();
        let payload: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let image = build_nsp(&[("main.nca", &payload), ("notes.txt", b"hello")]);
        let size = image.len() as u64;
        set_host_container(image);

        assert_eq!(switch_open_nsp(handle, size), 0);

        let mut buf = vec![0u8; 4096];
        let n = switch_nsp_files_json(handle, buf.as_mut_ptr(), buf.len() as u32);
        let json = String::from_utf8(buf[..n as usize].to_vec()).unwrap();
        assert!(json.contains(r#"{"name":"main.nca","offset":"#), "{json}");
        assert!(json.contains(r#""size":4096}"#), "{json}");
        assert!(json.contains(r#"{"name":"notes.txt""#), "{json}");

        // A read is relative to the file inside the container, and stops at
        // its end rather than running on into the next one.
        let mut out = vec![0u8; 32];
        let got = switch_read_file(handle, 0, 0x1000 - 8, out.as_mut_ptr(), out.len() as u32);
        assert_eq!(got, 8);
        assert_eq!(&out[..8], &payload[0x1000 - 8..]);

        let got = switch_read_file(handle, 1, 0, out.as_mut_ptr(), out.len() as u32);
        assert_eq!(got, 5);
        assert_eq!(&out[..5], b"hello");

        // Past the end of a file is nothing, and past the end of the table is
        // an error — neither is a read of whatever happens to be next.
        assert_eq!(switch_read_file(handle, 1, 5, out.as_mut_ptr(), 32), 0);
        assert_eq!(switch_read_file(handle, 7, 0, out.as_mut_ptr(), 32), -1);

        // The payload is not an NCA, so booting it has to come back as a
        // readable error rather than a trap — the failure this whole path
        // replaced was a `RuntimeError: unreachable` with nothing behind it.
        assert_eq!(switch_load_nca_from_nsp(handle, 0), -1);
        let mut err = vec![0u8; 512];
        let n = switch_last_error(handle, err.as_mut_ptr(), err.len() as u32);
        let text = String::from_utf8(err[..n as usize].to_vec()).unwrap();
        assert!(text.contains("bad magic"), "{text}");

        switch_free_session(handle);
    }

    /// The allocator entry point JS calls before every buffer it passes in.
    /// A request past `isize::MAX` is refused rather than reaching `Layout`,
    /// whose error path is an `unreachable` that takes down the module.
    #[test]
    fn an_impossible_allocation_is_refused_not_fatal() {
        assert!(!switch_alloc(64).is_null());
        // Only wasm32 has a `usize` small enough for `switch_alloc`'s `u32`
        // to reach the limit; on a 64-bit host every `u32` is allocatable, so
        // the refusal itself is what is checked there.
        if (u32::MAX as u64) > isize::MAX as u64 {
            assert!(switch_alloc(u32::MAX).is_null());
        }
        // Freeing what was never allocated is a no-op, not a fault.
        switch_free(std::ptr::null_mut(), 64);
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

    #[test]
    fn an_error_message_keeps_its_last_character() {
        // The NUL a C string needs comes out of the buffer, not out of the
        // message. Taking it off the message instead cost every error its last
        // byte however much room was left, so the console showed "Launch
        // failed: no container is ope" and read as a truncated console rather
        // than as a bug in the copy.
        let (_host, handle) = new_session();
        const MSG: &str = "no container is open";
        session(handle).last_error = MSG.to_string();

        let mut buf = [0xAAu8; 64];
        let n = switch_last_error(handle, buf.as_mut_ptr(), buf.len() as u32);
        assert_eq!(n as usize, MSG.len());
        assert_eq!(&buf[..n as usize], MSG.as_bytes());
        assert_eq!(buf[n as usize], 0, "the copy has to stay NUL-terminated");

        // A message that genuinely does not fit loses the bytes the buffer
        // cannot hold -- and exactly those, with the NUL inside the buffer.
        session(handle).last_error = MSG.to_string();
        let mut small = [0xAAu8; 8];
        let n = switch_last_error(handle, small.as_mut_ptr(), small.len() as u32);
        assert_eq!(n as usize, small.len() - 1);
        assert_eq!(&small[..n as usize], &MSG.as_bytes()[..small.len() - 1]);
        assert_eq!(small[small.len() - 1], 0);

        switch_free_session(handle);
    }
}
