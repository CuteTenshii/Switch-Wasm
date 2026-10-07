//! WASM bindings for switch-core over a raw `extern "C"` ABI. Buffers cross through
//! linear memory, handles index a global session table, structured results are
//! hand-written JSON, and errors are read back with `switch_last_error`.

// Exports take pointers into linear memory that the JS caller owns for the call.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicU32, Ordering};

/// A `Sync` wrapper for single-threaded interior mutability (wasm).
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

#[cfg(feature = "gpu")]
mod gpu;
#[cfg(target_arch = "wasm32")]
mod heap;
#[cfg(all(feature = "jit", target_arch = "wasm32"))]
mod jit;

use switch_core::cpu::{Cpu, SaveKey, TouchPoint};
use switch_core::elf::load_elf;
use switch_core::nca::Nca;
use switch_core::nsp::Pfs0;
use switch_core::source::{ByteSource, Window};
use switch_core::trace::Level;

/// Framebuffer base address, width, height and stride (RGBA, little-endian).
pub use switch_core::{FB_BASE, FB_HEIGHT, FB_STRIDE, FB_WIDTH};
/// Memory-mapped input register: JS writes an ASCII key, homebrew acknowledges with 0.
pub const INPUT_ADDR: u32 = switch_core::INPUT_ADDR;

struct Session {
    /// The open container, read from the host by range.
    container: Option<HostSource>,
    nsp_files: Vec<switch_core::nsp::Pfs0File>,
    /// The update added for the open title, held as a host file until boot.
    update: Option<Update>,
    /// Add-on content, one entry per DLC archive, held as host files until boot.
    dlc: Vec<Dlc>,
    keys: switch_core::keys::KeySet,
    /// Control data from the last Control NCA read, cached so the icon can be fetched separately.
    control: Option<switch_core::control::Control>,
    /// Users staged by `switch_user_stage`, installed whole by `switch_users_commit`.
    staged_users: Vec<switch_core::cpu::UserAccount>,
    cpu: Cpu,
    last_error: String,
}

// Single-threaded wasm: every export runs to completion, so unsynchronized access is safe
// (a `Mutex` would abort on reentry).
static SESSIONS: SyncCell<Vec<Option<Session>>> = SyncCell::new(Vec::new());

/// Last panic message, in a fixed buffer so the hook never allocates.
static PANIC_MSG: SyncCell<[u8; 2048]> = SyncCell::new([0u8; 2048]);

/// Whether the hook has fired since the host last looked; outlives the taken message.
static PANICKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The largest `n <= limit` at which `s` may be cut on a char boundary.
fn floor_char_boundary(s: &str, limit: usize) -> usize {
    if s.len() <= limit {
        return s.len();
    }
    let mut n = limit;
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    n
}

/// Session-handle counter, independent of slot reuse so stale handles never alias.
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

/// The session `handle` names, or `None`. See [`session`] for the panicking form.
fn session_opt(handle: u32) -> Option<&'static mut Session> {
    // SAFETY: single-threaded wasm; see the `SESSIONS` comment.
    let slots = unsafe { &mut *SESSIONS.get() };
    let slot = slots.get_mut(handle as usize)?.as_mut()?;
    Some(unsafe { std::mem::transmute::<&mut Session, &'static mut Session>(slot) })
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

/// The host read as a `wasm-bindgen` import, since wasm-bindgen owns the import object.
/// `@host/files` is resolved by the bundler (see `vite.config.ts`).
#[cfg(all(target_arch = "wasm32", feature = "gpu"))]
#[wasm_bindgen::prelude::wasm_bindgen(raw_module = "@host/files")]
extern "C" {
    #[wasm_bindgen(js_name = hostRead)]
    fn host_read_js(file: u32, offset: u64, ptr: u32, len: u32) -> u32;
}

/// # Safety
/// `ptr` must be valid for writes of `len` bytes.
#[cfg(all(target_arch = "wasm32", feature = "gpu"))]
unsafe fn host_read(file: u32, offset: u64, ptr: *mut u8, len: u32) -> u32 {
    host_read_js(file, offset, ptr as u32, len)
}

#[cfg(all(target_arch = "wasm32", not(feature = "gpu")))]
#[link(wasm_import_module = "env")]
extern "C" {
    /// Read `len` bytes at `offset` of host file `file` into wasm memory at `ptr`, returning
    /// the count read. File 0 is the open container; the rest are system data archives.
    fn host_read(file: u32, offset: u64, ptr: *mut u8, len: u32) -> u32;
}

/// Host-build stand-in serving [`set_host_container`].
///
/// # Safety
/// `ptr` must be valid for writes of `len` bytes.
#[cfg(not(target_arch = "wasm32"))]
unsafe fn host_read(file: u32, offset: u64, ptr: *mut u8, len: u32) -> u32 {
    if file != 0 {
        return 0; // host builds serve only the container
    }
    // SAFETY: single-threaded; host builds hold the test lock.
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

/// Install the bytes host builds serve as the open container.
#[cfg(not(target_arch = "wasm32"))]
pub fn set_host_container(data: Vec<u8>) {
    // SAFETY: single-threaded wasm; see the `SESSIONS` comment.
    unsafe { *HOST_CONTAINER.get() = data };
}

/// A [`ByteSource`] over a host file. `Copy`, since every read goes to the host.
#[derive(Debug, Clone, Copy)]
struct HostSource {
    /// 0 is the open container; each system data archive has its own.
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
            // Also absorbs short reads at the host's cache-chunk boundary.
            let ask = (want - done).min(u32::MAX as usize);
            let got = unsafe {
                host_read(
                    self.file,
                    offset + done as u64,
                    out[done..].as_mut_ptr(),
                    ask as u32,
                )
            } as usize;
            if got == 0 {
                break;
            }
            done += got;
        }
        switch_core::trace!(
            switch_core::trace::Trace::Io,
            "[io] host file {} read {want:#x} bytes at {offset:#x} -> {done:#x}",
            self.file
        );
        if done != want {
            return Err(switch_core::Error::Io(format!(
                "host read of {} bytes at {:#x} returned {}",
                want, offset, done
            )));
        }
        Ok(done)
    }
}

/// Allocate `len` bytes of linear memory for JS. Returns null for sizes above
/// `isize::MAX`; callers must check.
#[no_mangle]
pub extern "C" fn switch_alloc(len: u32) -> *mut u8 {
    // The earliest call any host makes.
    install_panic_hook();
    match Layout::from_size_align(len as usize, 1) {
        Ok(layout) => unsafe { alloc(layout) },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a buffer from `switch_alloc`. Null frees nothing.
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

/// Install the panic hook, once. Call right after instantiating.
#[no_mangle]
pub extern "C" fn switch_init() {
    install_panic_hook();
    attach_jit();
}

/// Let the core run the blocks it emits, where the build can compile them.
fn attach_jit() {
    #[cfg(all(feature = "jit", target_arch = "wasm32"))]
    jit::attach();
}

/// Capture Rust panics into a static buffer for [`switch_last_error`], since they
/// otherwise trap silently.
fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let msg = format!("PANIC: {info}");
            // SAFETY: single-threaded wasm; the hook runs once per panic.
            let guard = unsafe { &mut *PANIC_MSG.get() };
            // Truncate by character so the page's UTF-8 decoder gets a whole string.
            let n = floor_char_boundary(&msg, guard.len() - 1);
            guard[..n].copy_from_slice(&msg.as_bytes()[..n]);
            guard[n] = 0;
            PANICKED.store(true, Ordering::Relaxed);
        }));
    });
}

/// Clear module-level state (panic flag, panic message, pending traces) left by
/// the previous session.
fn forget_the_last_session() {
    PANICKED.store(false, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: single-threaded wasm; see the `SESSIONS` comment.
    unsafe { &mut *PANIC_MSG.get() }.fill(0);
    // Drop traces the previous session left.
    let _ = switch_core::trace::take_pending();
}

/// Create a fresh machine, return its handle.
#[no_mangle]
pub extern "C" fn switch_new() -> u32 {
    install_panic_hook();
    // Also here for hosts that never call `switch_init`.
    attach_jit();
    forget_the_last_session();
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    new_handle(Session {
        container: None,
        nsp_files: Vec::new(),
        update: None,
        dlc: Vec::new(),
        keys: switch_core::keys::KeySet::default(),
        control: None,
        staged_users: Vec::new(),
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

/// Message bytes that fit in `maxlen` alongside a terminating NUL.
fn nul_reserved(maxlen: u32) -> usize {
    (maxlen as usize).saturating_sub(1)
}

/// Copy the last error message into `buf` (NUL-terminated). Returns length.
/// Also surfaces any Rust panic captured by the panic hook.
#[no_mangle]
pub extern "C" fn switch_last_error(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    // A captured panic takes priority and needs no valid handle.
    // SAFETY: single-threaded wasm; see the `SESSIONS` comment.
    let panicked = unsafe { &mut *PANIC_MSG.get() };
    if panicked[0] != 0 {
        let len = panicked
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(panicked.len());
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

/// Open the `size`-byte container (NSP or XCI) the host has ready, read through
/// `host_read`. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn switch_open_nsp(handle: u32, size: u64) -> i32 {
    let s = session(handle);
    let container = HostSource { file: 0, len: size };
    s.container = Some(container);
    s.nsp_files = Vec::new();
    s.control = None;
    match switch_core::xci::read_container(&container) {
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

/// Open the host's container as a single standalone `.nca`. Returns 0.
#[no_mangle]
pub extern "C" fn switch_open_nca(handle: u32, size: u64) -> i32 {
    let s = session(handle);
    s.container = Some(HostSource { file: 0, len: size });
    s.nsp_files = Vec::new();
    s.control = None;
    s.last_error.clear();
    0
}

/// Register host file `file` as a system data archive for `OpenDataStorageByDataId`.
/// Returns 0, or -1 if it is not a readable data archive.
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
    if !matches!(
        nca.content_type,
        ContentType::Data | ContentType::PublicData
    ) {
        s.last_error = format!(
            "not a data archive (content type {})",
            nca.content_type.name()
        );
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

/// Register the host's update container for the title about to run. Returns the
/// base program id it patches, or 0 with the reason in `switch_last_error`.
/// Pairing is checked at boot.
#[no_mangle]
pub extern "C" fn switch_add_update(handle: u32, file: u32, size: u64) -> u64 {
    let s = session(handle);
    let src = HostSource { file, len: size };
    let files = match Pfs0::read_from(&src) {
        Ok(pfs0) => pfs0.files,
        Err(e) => {
            s.last_error = format!("an update has to be an NSP: {e}");
            return 0;
        }
    };
    let Some((index, nca)) = switch_core::nca::find_nca_by_type(
        &files,
        &src,
        &s.keys,
        switch_core::nca::ContentType::Program,
    ) else {
        s.last_error =
            "no Program NCA in this container (or its header couldn't be decrypted — load prod.keys)"
                .into();
        return 0;
    };
    // An update carries its own ticket.
    let _ = switch_core::ticket::load_bundled_title_key(&mut s.keys, &nca, &files, &src);
    if !nca.is_update() {
        s.last_error =
            "this container is a title in its own right, not an update: its RomFS is its own"
                .into();
        return 0;
    }
    let f = &files[index];
    let program = (f.offset, f.size);
    let program_id = nca.program_id;
    s.update = Some(Update {
        nca,
        src,
        program,
        files,
    });
    s.last_error.clear();
    program_id
}

/// The update's NACP display version ("1.0.1") into `buf`, or empty.
#[no_mangle]
pub extern "C" fn switch_update_version(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let version = s
        .update
        .as_ref()
        .and_then(|update| {
            let (index, _) =
                switch_core::control::find_control_nca(&update.files, &update.src, &s.keys)?;
            let f = update.files.get(index)?;
            let window = Window::new(update.src, f.offset, f.size, &f.name).ok()?;
            let control = switch_core::control::Control::from_source(window, &s.keys).ok()?;
            Some(control.nacp.display_version)
        })
        .unwrap_or_default();
    write_into(buf, maxlen, version.as_bytes())
}

/// Register a container of add-on content. Returns how many pieces it holds, or 0
/// with the reason in `switch_last_error`. Pairing is checked at boot.
#[no_mangle]
pub extern "C" fn switch_add_dlc(handle: u32, file: u32, size: u64) -> u32 {
    let s = session(handle);
    let src = HostSource { file, len: size };
    let files = match Pfs0::read_from(&src) {
        Ok(pfs0) => pfs0.files,
        Err(e) => {
            s.last_error = format!("add-on content has to be an NSP: {e}");
            return 0;
        }
    };
    // A container with a Program NCA is a game or an update, not DLC.
    if switch_core::nca::find_nca_by_type(
        &files,
        &src,
        &s.keys,
        switch_core::nca::ContentType::Program,
    )
    .is_some()
    {
        s.last_error = "this container holds a program — add-on content is data only".into();
        return 0;
    }

    let mut found = 0;
    for f in &files {
        if !f.name.to_ascii_lowercase().ends_with(".nca") {
            continue;
        }
        let Ok(window) = Window::new(src, f.offset, f.size, &f.name) else {
            continue;
        };
        let Ok(nca) = Nca::parse_source(&window, Some(&s.keys)) else {
            continue;
        };
        use switch_core::nca::ContentType;
        if !matches!(
            nca.content_type,
            ContentType::Data | ContentType::PublicData
        ) || !is_add_on_content_id(nca.title_id)
        {
            continue;
        }
        // Each piece carries its own ticket.
        let _ = switch_core::ticket::load_bundled_title_key(&mut s.keys, &nca, &files, &src);
        if nca.romfs_section_index().is_none() {
            continue;
        }
        s.dlc.retain(|held| held.content_id != nca.title_id);
        s.dlc.push(Dlc {
            content_id: nca.title_id,
            src,
            nca: (f.offset, f.size),
        });
        found += 1;
    }
    if found == 0 {
        s.last_error = "no add-on content in this container".into();
    } else {
        s.last_error.clear();
    }
    found
}

/// The session's add-on content as JSON: content id, index and base title.
#[no_mangle]
pub extern "C" fn switch_dlc_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let mut out = Vec::new();
    out.push(b'[');
    for (i, dlc) in s.dlc.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(
            format!(
                "{{\"id\":\"{:016x}\",\"title_id\":\"{:016x}\",\"index\":{}}}",
                dlc.content_id,
                dlc.content_id & !0x1FFF,
                dlc.content_id & 0x7FF
            )
            .as_bytes(),
        );
    }
    out.push(b']');
    write_into(buf, maxlen, &out)
}

#[no_mangle]
pub extern "C" fn switch_clear_dlc(handle: u32) {
    session(handle).dlc.clear();
}

#[no_mangle]
pub extern "C" fn switch_clear_update(handle: u32) {
    session(handle).update = None;
}

/// Identify a firmware NCA from its header: writes the content type to `kind_out`
/// (0 program, 1 data archive, 2 other) and returns the title id, or 0.
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

/// Boot a Program NCA from the NAND (an applet). Returns the entry address, or -1
/// with the reason in `switch_last_error`.
#[no_mangle]
pub extern "C" fn switch_nand_launch(handle: u32, ptr: *const u8, len: u32) -> i64 {
    let s = session(handle);
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec();
    load_and_boot_nca(
        &s.keys,
        &mut s.cpu,
        &mut s.last_error,
        switch_core::source::MemSource(bytes),
        Added::default(),
    )
}

/// An update container added for the title about to run. Its Program NCA holds
/// full replacement modules and a RomFS patch over the base's.
struct Update {
    /// Its program id is the base title's, which is what pairing checks.
    nca: Nca,
    src: HostSource,
    /// Where the Program NCA sits inside the update container.
    program: (u64, u64),
    /// Kept for the update's Control NCA.
    files: Vec<switch_core::nsp::Pfs0File>,
}

impl Update {
    /// A fresh window over the update's Program NCA.
    fn program_window(&self) -> Result<Window<HostSource>, switch_core::Error> {
        Window::new(
            self.src,
            self.program.0,
            self.program.1,
            "update program nca",
        )
    }
}

/// One add-on content Data NCA, mounted by its title id like a system data archive.
#[derive(Debug, Clone, Copy)]
struct Dlc {
    content_id: u64,
    src: HostSource,
    /// Where the Data NCA sits inside the container.
    nca: (u64, u64),
}

impl Dlc {
    fn window(&self) -> Result<Window<HostSource>, switch_core::Error> {
        Window::new(self.src, self.nca.0, self.nca.1, "add-on content nca")
    }
}

#[derive(Default, Clone, Copy)]
struct Added<'a> {
    update: Option<&'a Update>,
    dlc: &'a [Dlc],
}

/// Whether a title id is add-on content: the base plus 0x1000 and an 11-bit index.
fn is_add_on_content_id(title_id: u64) -> bool {
    (0x1000..0x1800).contains(&(title_id & 0x1FFF))
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

/// A source over NSP file `index`: the window an NCA is read through.
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

/// Read a slice of NSP file `index` from `file_offset` into `buf`. Returns bytes
/// copied or -1.
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
    // SAFETY: JS allocated `maxlen` bytes at `buf`, and `n` is no larger.
    let out = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    match file.read_at(file_offset, out) {
        Ok(got) => got as i64,
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Parse an NCA from `ptr`/`len` and return a JSON summary, decrypting with loaded keys.
#[no_mangle]
pub extern "C" fn switch_parse_nca(
    handle: u32,
    ptr: *const u8,
    len: u32,
    buf: *mut u8,
    maxlen: u32,
) -> u32 {
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
            out.extend_from_slice(if nca.is_encrypted() {
                b"true"
            } else {
                b"false"
            });
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
                // The filesystem type lives in the FS header, which needs the full header to decrypt.
                let fs_type = match nca.fs_headers.get(i).and_then(|o| o.as_ref()) {
                    Some(fs) if sec.media_size > 0 => {
                        if fs.fs_type == 1 {
                            "PFS0"
                        } else {
                            "ROMFS"
                        }
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

/// Cache control data and pass its NACP figures (save data, add-on base id) to the CPU.
fn cache_control(s: &mut Session, control: switch_core::control::Control) {
    s.cpu
        .set_save_data_quota(switch_core::cpu::SaveDataQuota::from(&control.nacp));
    s.cpu
        .set_add_on_content_base_id(control.nacp.add_on_content_base_id);
    s.control = Some(control);
}

/// Read and cache the open container's Control NCA. Returns 0, or -1 when it has
/// none readable (including when no `prod.keys` are loaded).
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
    let _ =
        switch_core::ticket::load_bundled_title_key(&mut s.keys, &nca, &s.nsp_files, &container);
    let Some(file) = nsp_file_source(s, index as u32) else {
        return -1;
    };
    match switch_core::control::Control::from_source(file, &s.keys) {
        Ok(control) => {
            cache_control(s, control);
            s.last_error.clear();
            0
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Same, for a container opened as a standalone Control NCA.
#[no_mangle]
pub extern "C" fn switch_load_control_from_nca(handle: u32) -> i32 {
    let s = session(handle);
    s.control = None;
    let Some(container) = container(s) else {
        return -1;
    };
    match switch_core::control::Control::from_source(container, &s.keys) {
        Ok(control) => {
            cache_control(s, control);
            s.last_error.clear();
            0
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// The cached control data as JSON, or `{}`. `icon_size` sizes `switch_control_icon`'s buffer.
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
    out.extend_from_slice(
        nacp.user_account_save_data_journal_size
            .to_string()
            .as_bytes(),
    );
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

/// Copy the cached icon into `buf`. Returns bytes copied, or -1.
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
    s.control = None;
    match s.cpu.boot_homebrew(data) {
        Ok(loaded) => {
            // Cached for display only: homebrew runs in another title's process.
            s.control = switch_core::control::Control::from_nro(data);
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

/// ExeFS modules in load order (`rtld`, `main`, `subsdk0..9`, `sdk`), skipping absent ones.
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

/// Decrypt a Program NCA's ExeFS from `nca_src`, load it and boot. Returns entry or -1.
/// `nca_src` stays alive as the title's RomFS source. `added` holds the update and DLC.
fn load_and_boot_nca<S: ByteSource + 'static>(
    keys: &switch_core::keys::KeySet,
    cpu: &mut Cpu,
    last_error: &mut String,
    nca_src: S,
    added: Added<'_>,
) -> i64 {
    let nca = match Nca::parse_source(&nca_src, Some(keys)) {
        Ok(nca) => nca,
        Err(e) => {
            *last_error = e.to_string();
            return -1;
        }
    };
    // Refuse an update for a different title.
    let update = match added.update {
        Some(u) if u.nca.program_id == nca.program_id => Some(u),
        Some(u) => {
            *last_error = format!(
                "the update added to this session is for title {:016x}, but this container is {:016x}",
                u.nca.program_id, nca.program_id
            );
            return -1;
        }
        None => None,
    };
    // An update's ExeFS is a complete replacement set of modules.
    let program = update.map_or(&nca, |u| &u.nca);
    let exefs_index = match program.exefs_section_index() {
        Some(i) => i,
        None => {
            *last_error = "no ExeFS (PFS0) section in this NCA".into();
            return -1;
        }
    };
    let exefs = match update {
        Some(u) => u
            .program_window()
            .and_then(|window| program.read_pfs0_section(window, keys, exefs_index)),
        None => program.read_pfs0_section(&nca_src, keys, exefs_index),
    };
    let exefs = match exefs {
        Ok(v) => v,
        Err(e) => {
            *last_error = e.to_string();
            return -1;
        }
    };
    if update.is_some() {
        cpu.diagnostic(
            Level::Info,
            &format!(
                "[update] booting the update's modules for {:016x}, over this container's RomFS",
                nca.program_id
            ),
        );
    }
    // Report whether the ExeFS hash coverage was checked.
    match program.pfs0_hash_coverage(exefs_index) {
        Some((block, blocks)) => cpu.diagnostic(
            Level::Info,
            &format!(
                "[exefs] {:#x} bytes, {} blocks of {:#x} verified against the section hash table",
                exefs.len(),
                blocks,
                block
            ),
        ),
        None => cpu.diagnostic(
            Level::Warn,
            &format!(
                "[exefs] {:#x} bytes — hash table geometry unrecognised, contents NOT verified",
                exefs.len()
            ),
        ),
    }

    let pfs0 = match Pfs0::parse(&exefs) {
        Ok(p) => p,
        Err(e) => {
            *last_error = e.to_string();
            return -1;
        }
    };
    // Reported by `pm`; applets derive their `AppletId` from it.
    cpu.set_program_id(program.program_id);

    let modules = collect_modules(&pfs0, &exefs);
    // Report what the ExeFS holds next to what was loaded.
    cpu.diagnostic(
        Level::Info,
        &format!(
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
        ),
    );
    if !modules.iter().any(|(name, _)| *name == "main") {
        *last_error = "no 'main' executable in this NCA's ExeFS".into();
        return -1;
    }

    // RomFS is optional and failures do not block booting; it is read by range.
    match update {
        // An update's RomFS is a patch, readable only over the base's.
        Some(u) => {
            let patched = u.program_window().and_then(|window| {
                switch_core::bktr::patched_romfs_source(&u.nca, window, &nca, nca_src, keys)
            });
            match patched {
                Ok(romfs) => cpu.set_romfs_source(Box::new(romfs)),
                Err(e) => cpu.diagnostic(
                    Level::Error,
                    &format!("the update's RomFS is unreadable: {}", e),
                ),
            }
        }
        None => {
            if let Some(romfs_index) = nca.romfs_section_index() {
                match nca.romfs_source(nca_src, keys, romfs_index) {
                    Ok(romfs) => cpu.set_romfs_source(Box::new(romfs)),
                    Err(e) => cpu.diagnostic(Level::Error, &format!("romfs unavailable: {}", e)),
                }
            }
        }
    }

    // The address space layout from the NPDM system resource size; must precede the boot.
    let system_resource = switch_core::npdm::Npdm::system_resource_size_of(&pfs0, &exefs);
    cpu.diagnostic(
        Level::Info,
        &format!(
            "[npdm] system resource {system_resource:#x} — {}",
            if system_resource == 0 {
                "plain heap"
            } else {
                "virtual address memory"
            }
        ),
    );
    cpu.set_system_resource_size(system_resource);

    // Main thread priority.
    if let Some(priority) = switch_core::npdm::Npdm::main_thread_priority_of(&pfs0, &exefs) {
        cpu.set_main_thread_priority(priority);
    }
    // Main thread core.
    if let Some(core) = switch_core::npdm::Npdm::main_thread_core_of(&pfs0, &exefs) {
        cpu.set_main_thread_core(core);
    }
    // Allowed cores.
    if let Some(mask) = switch_core::npdm::Npdm::core_mask_of(&pfs0, &exefs) {
        cpu.set_process_core_mask(mask);
    }

    // 32- or 64-bit, from the NPDM flags; the entry ABI differs.
    if !switch_core::npdm::Npdm::is_64_bit_of(&pfs0, &exefs) {
        cpu.diagnostic(
            Level::Info,
            "[npdm] AArch32 title — running the A32 interpreter",
        );
        cpu.set_mode(switch_core::cpu::ExecMode::A32);
    }

    match cpu.boot_retail_program(&modules) {
        Ok(loaded) => {
            // After the modules: booting clears the diagnostic buffer.
            mount_add_on_content(cpu, keys, added.dlc);
            last_error.clear();
            loaded[0].entry as i64
        }
        Err(e) => {
            *last_error = e.to_string();
            -1
        }
    }
}

/// Mount the added DLC whose ids belong to this title; report and skip the rest.
fn mount_add_on_content(cpu: &mut Cpu, keys: &switch_core::keys::KeySet, dlc: &[Dlc]) {
    for entry in dlc {
        let romfs = entry.window().and_then(|window| {
            let nca = Nca::parse_source(&window, Some(keys))?;
            let index = nca
                .romfs_section_index()
                .ok_or_else(|| switch_core::Error::Nca("no RomFS in this archive".into()))?;
            nca.romfs_source(window, keys, index)
        });
        match romfs {
            Ok(romfs) => {
                let size = romfs.len();
                match cpu.add_add_on_content(entry.content_id, Box::new(romfs)) {
                    Some(index) => cpu.diagnostic(
                        Level::Info,
                        &format!(
                            "[aoc] {:016x} mounted as add-on content {index}, {size:#x} bytes",
                            entry.content_id
                        ),
                    ),
                    None => cpu.diagnostic(
                        Level::Warn,
                        &format!(
                            "[aoc] {:016x} is not this title's add-on content — not mounted",
                            entry.content_id
                        ),
                    ),
                }
            }
            Err(e) => cpu.diagnostic(
                Level::Error,
                &format!("[aoc] {:016x} could not be read: {e}", entry.content_id),
            ),
        }
    }
}

/// Boot the open container as a standalone Program NCA. Returns entry or -1;
/// check `switch_last_error`, since 0 can be a valid entry.
#[no_mangle]
pub extern "C" fn switch_load_nca(handle: u32) -> i64 {
    let s = session(handle);
    let Some(container) = container(s) else {
        return -1;
    };
    let added = Added {
        update: s.update.as_ref(),
        dlc: &s.dlc,
    };
    load_and_boot_nca(&s.keys, &mut s.cpu, &mut s.last_error, container, added)
}

/// The index of the Program NCA in the open container, or -1 (including when no
/// `prod.keys` are loaded). Pass it to `switch_load_nca_from_nsp`.
#[no_mangle]
pub extern "C" fn switch_program_nca_index(handle: u32) -> i32 {
    let s = session(handle);
    let Some(container) = container(s) else {
        return -1;
    };
    let found = switch_core::nca::find_nca_by_type(
        &s.nsp_files,
        &container,
        &s.keys,
        switch_core::nca::ContentType::Program,
    );
    match found {
        Some((index, _)) => {
            s.last_error.clear();
            index as i32
        }
        None => {
            s.last_error =
                "no Program NCA in this container (or its header couldn't be decrypted — load prod.keys)"
                    .into();
            -1
        }
    }
}

/// Boot NSP file `index` as a Program NCA. Returns entry or -1; check
/// `switch_last_error`, since 0 can be a valid entry.
#[no_mangle]
pub extern "C" fn switch_load_nca_from_nsp(handle: u32, index: u32) -> i64 {
    let s = session(handle);
    let Some(container) = container(s) else {
        return -1;
    };
    let Some(nca_src) = nsp_file_source(s, index) else {
        return -1;
    };

    // Try the bundled ticket before external title.keys.
    if let Ok(nca) = Nca::parse_source(&nca_src, Some(&s.keys)) {
        let _ = switch_core::ticket::load_bundled_title_key(
            &mut s.keys,
            &nca,
            &s.nsp_files,
            &container,
        );
    }

    let added = Added {
        update: s.update.as_ref(),
        dlc: &s.dlc,
    };
    load_and_boot_nca(&s.keys, &mut s.cpu, &mut s.last_error, nca_src, added)
}

/// Load an AArch64 ELF into the CPU. Returns entry address or -1.
#[no_mangle]
pub extern "C" fn switch_load_elf(handle: u32, ptr: *const u8, len: u32) -> i64 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    // An ELF carries no control data.
    s.control = None;
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

/// Reset the integer registers and set the entry convention: `x0 = env`,
/// `x1 = UINT64_MAX` for the homebrew ABI, `x0 = 0` for NSO, `x0 = 0, x1 = 1` otherwise.
fn boot_entry_regs(cpu: &mut Cpu, env_addr: u32) {
    for i in 0..=30u8 {
        cpu.set_reg(i, 0);
    }
    cpu.set_reg(0, env_addr as u64);
    cpu.set_reg(1, if env_addr != 0 { u64::MAX } else { 1 });
    // LR to the exit trampoline, so a returning `main` exits.
    cpu.set_reg(30, switch_core::cpu::SELF_RETURN_TRAMPOLINE as u64);
}

/// Enable/disable the block translator; the browser has no `SWITCH_NO_JIT`.
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
        "{{\"enabled\":{},\"blocks\":{},\"translated\":{},\"executed\":{},\"linked\":{},\"invalidated\":{},\"interpreted\":{},\"interpretedGroups\":[{}],\"emitted\":{},\"enteredEmitted\":{}}}",
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
        stats.entered_emitted
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

/// The last rumble request, packed as `(weak << 16) | strong`, each 0..=1000.
#[no_mangle]
pub extern "C" fn switch_vibration(handle: u32) -> u32 {
    let s = session(handle);
    let (low, high) = s.cpu.vibration();
    let scale = |v: f32| (v * 1000.0).round().clamp(0.0, 1000.0) as u32;
    (scale(high) << 16) | scale(low)
}

/// The PCM format `switch_audio_pull` returns, packed as `(channels << 24) | sample_rate`.
/// Zero until the guest opens an audio device.
#[no_mangle]
pub extern "C" fn switch_audio_format(handle: u32) -> u32 {
    let (rate, channels) = session(handle).cpu.audio_format();
    if rate == 0 {
        return 0;
    }
    (channels << 24) | (rate & 0x00ff_ffff)
}

/// Move up to `max_samples` interleaved 16-bit samples into `buf`, returning the count.
#[no_mangle]
pub extern "C" fn switch_audio_pull(handle: u32, buf: *mut u8, max_samples: u32) -> u32 {
    let s = session(handle);
    let mut samples = vec![0i16; max_samples as usize];
    let n = s.cpu.take_audio(&mut samples);
    let out = unsafe { std::slice::from_raw_parts_mut(buf, n * 2) };
    for (chunk, sample) in out.as_chunks_mut::<2>().0.iter_mut().zip(samples.iter()) {
        *chunk = sample.to_le_bytes();
    }
    n as u32
}

/// Framebuffer geometry: the presented resolution, or the demo framebuffer's before
/// the first present.
#[no_mangle]
pub extern "C" fn switch_fb_width(handle: u32) -> u32 {
    let s = session(handle);
    if s.cpu.nv.gpu.frames > 0 {
        s.cpu.nv.gpu.framebuffer.width
    } else {
        FB_WIDTH
    }
}

#[no_mangle]
pub extern "C" fn switch_fb_height(handle: u32) -> u32 {
    let s = session(handle);
    if s.cpu.nv.gpu.frames > 0 {
        s.cpu.nv.gpu.framebuffer.height
    } else {
        FB_HEIGHT
    }
}

/// Number of frames the guest has presented.
#[no_mangle]
pub extern "C" fn switch_frame_count(handle: u32) -> u32 {
    session(handle).cpu.nv.gpu.frames as u32
}

/// Copy the current screen (RGBA8888) into `buf`. Returns bytes copied. Alpha is
/// forced opaque, as scan-out ignores it.
#[no_mangle]
pub extern "C" fn switch_fb_snapshot(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    if s.cpu.nv.gpu.frames > 0 {
        let fb = &s.cpu.nv.gpu.framebuffer;
        let n = (fb.pixels.len() * 4).min(maxlen as usize);
        let out = unsafe { std::slice::from_raw_parts_mut(buf, n) };
        for (chunk, pixel) in out.as_chunks_mut::<4>().0.iter_mut().zip(fb.pixels.iter()) {
            *chunk = (pixel | 0xFF00_0000).to_le_bytes();
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

// The emulated SD card. Paths are guest paths; `sdmc:` and extra slashes are
// normalized away.

/// Read a UTF-8 path out of guest-supplied wasm memory.
fn sd_path(ptr: *const u8, len: u32) -> String {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// Put a file on the SD card. Not reported as a change: this is the host's restore path.
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
    s.cpu
        .fs
        .write_file(&sd_path(path_ptr, path_len), data.to_vec());
    0
}

/// Create a directory and any missing parents. Not reported as a change.
#[no_mangle]
pub extern "C" fn switch_sd_create_dir(handle: u32, path_ptr: *const u8, path_len: u32) -> i32 {
    session(handle)
        .cpu
        .fs
        .create_dir(&sd_path(path_ptr, path_len));
    0
}

/// Delete a path from the SD card. Returns 1 if something was there, 0 if not.
#[no_mangle]
pub extern "C" fn switch_sd_remove(handle: u32, path_ptr: *const u8, path_len: u32) -> i32 {
    i32::from(session(handle).cpu.fs.remove(&sd_path(path_ptr, path_len)))
}

/// Size of a file on the SD card, or -1 when the path is not a file.
#[no_mangle]
pub extern "C" fn switch_sd_file_size(handle: u32, path_ptr: *const u8, path_len: u32) -> i64 {
    match session(handle).cpu.fs.size(&sd_path(path_ptr, path_len)) {
        Some(size) => size as i64,
        None => -1,
    }
}

/// Copy a file off the SD card into `buf` from `offset`. Returns bytes copied, or -1.
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

/// How many paths the guest has changed and not yet drained.
#[no_mangle]
pub extern "C" fn switch_sd_pending_changes(handle: u32) -> u32 {
    session(handle).cpu.fs.pending_changes() as u32
}

/// Drain the guest's SD card changes as JSON, e.g.
/// `[{"path":"/switch/a.json","kind":"file","size":12}]`. Drains even if the
/// result does not fit; size `buf` from `switch_sd_pending_changes`.
#[no_mangle]
pub extern "C" fn switch_sd_take_changes_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let changes = session(handle).cpu.fs.take_changes();
    write_changes_json(&changes, buf, maxlen)
}

/// Serialize drained [`Change`](switch_core::vfs::Change)s into `buf`.
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

// save data
//
// A save is `save_id` plus the uid as `user_lo` (first eight bytes) and `user_hi`
// (last eight), little-endian, both zero for a save no user owns. See `SaveKey`.

/// Every save the session has opened, as JSON:
/// `["8000000000000050","0100000000001000@<32 hex digits of uid>"]`.
#[no_mangle]
pub extern "C" fn switch_save_ids_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let mut keys = session(handle).cpu.save_keys();
    keys.sort_unstable();
    let mut out = Vec::from("[");
    for (i, key) in keys.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(format!("\"{key}\"").as_bytes());
    }
    out.push(b']');
    let n = out.len().min(maxlen as usize);
    let dst = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    dst.copy_from_slice(&out[..n]);
    n as u32
}

/// How many paths the guest has changed in this save and not yet had drained.
#[no_mangle]
pub extern "C" fn switch_save_pending_changes(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
) -> u32 {
    let key = SaveKey::from_halves(save_id, user_lo, user_hi);
    session(handle).cpu.save_data_mut(key).pending_changes() as u32
}

/// Drain a save's changes, as `switch_sd_take_changes_json` does. Drains even if the
/// result does not fit.
#[no_mangle]
pub extern "C" fn switch_save_take_changes_json(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
    buf: *mut u8,
    maxlen: u32,
) -> u32 {
    let key = SaveKey::from_halves(save_id, user_lo, user_hi);
    let changes = session(handle).cpu.save_data_mut(key).take_changes();
    write_changes_json(&changes, buf, maxlen)
}

/// Put a file into a save, creating the save. Not reported as a change.
#[no_mangle]
pub extern "C" fn switch_save_write_file(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
    path_ptr: *const u8,
    path_len: u32,
    data_ptr: *const u8,
    data_len: u32,
) -> i32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len as usize) };
    let path = sd_path(path_ptr, path_len);
    s.cpu
        .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
        .write_file(&path, data.to_vec());
    0
}

/// Create a directory in a save and any missing parents. Not reported as a change.
#[no_mangle]
pub extern "C" fn switch_save_create_dir(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
    path_ptr: *const u8,
    path_len: u32,
) -> i32 {
    let s = session(handle);
    let path = sd_path(path_ptr, path_len);
    s.cpu
        .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
        .create_dir(&path);
    0
}

/// Size of a file in a save, or -1 when the path is not one.
#[no_mangle]
pub extern "C" fn switch_save_file_size(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
    path_ptr: *const u8,
    path_len: u32,
) -> i64 {
    let s = session(handle);
    let path = sd_path(path_ptr, path_len);
    match s
        .cpu
        .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
        .size(&path)
    {
        Some(size) => size as i64,
        None => -1,
    }
}

/// Copy a file out of a save into `buf` from `offset`. Returns bytes copied, or -1.
#[no_mangle]
pub extern "C" fn switch_save_read_file(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
    path_ptr: *const u8,
    path_len: u32,
    offset: u64,
    buf: *mut u8,
    maxlen: u32,
) -> i64 {
    let s = session(handle);
    let path = sd_path(path_ptr, path_len);
    let out = unsafe { std::slice::from_raw_parts_mut(buf, maxlen as usize) };
    match s
        .cpu
        .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
        .read(&path, offset, out)
    {
        Some(n) => n as i64,
        None => -1,
    }
}

/// Create a save in a fresh session so the host can restore it. Returns 0.
#[no_mangle]
pub extern "C" fn switch_save_create(handle: u32, save_id: u64, user_lo: u64, user_hi: u64) -> i32 {
    let key = SaveKey::from_halves(save_id, user_lo, user_hi);
    session(handle).cpu.save_data_mut(key);
    0
}

/// Set the shared system font `pl:u` serves (TTF/OTF bytes), before `plInitialize`.
/// Returns the bytes taken.
#[no_mangle]
pub extern "C" fn switch_load_font(handle: u32, ptr: *const u8, len: u32) -> u32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    s.cpu.set_shared_font(data.to_vec());
    s.cpu.shared_font_len() as u32
}

/// Feed gamepad state. `buttons` is a `HidNpadButton` bitfield (A=1<<0, B=1<<1,
/// X=1<<2, Y=1<<3, StickL=1<<4, StickR=1<<5, L=1<<6, R=1<<7, ZL=1<<8, ZR=1<<9,
/// Plus=1<<10, Minus=1<<11, DpadLeft=1<<12, DpadUp=1<<13, DpadRight=1<<14,
/// DpadDown=1<<15); sticks are -32768..32767, positive right and up.
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

/// Feed touch contacts: `count` packed `u32` triples (`finger_id`, `x`, `y`) in
/// 1280x720 digitizer space, truncated to 16. `count` 0 is a lift.
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

/// Dock or undock the console (0 handheld). Queues the AM messages titles react to.
#[no_mangle]
pub extern "C" fn switch_set_operation_mode(handle: u32, docked: u32) {
    let mode = if docked == 0 {
        switch_core::cpu::OperationMode::Handheld
    } else {
        switch_core::cpu::OperationMode::Docked
    };
    session(handle).cpu.set_operation_mode(mode);
}

// The console's users: staged one at a time, committed whole before the title starts.
// A uid travels as two little-endian halves.

fn uid_from_halves(lo: u64, hi: u64) -> [u8; 16] {
    let mut uid = [0u8; 16];
    uid[..8].copy_from_slice(&lo.to_le_bytes());
    uid[8..].copy_from_slice(&hi.to_le_bytes());
    uid
}

/// Stage one user. `edited_at` is POSIX seconds; `picture_len` 0 means a generated
/// picture. Pictures are baseline JPEG.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn switch_user_stage(
    handle: u32,
    uid_lo: u64,
    uid_hi: u64,
    name_ptr: *const u8,
    name_len: u32,
    edited_at: i64,
    picture_ptr: *const u8,
    picture_len: u32,
) {
    let name = sd_path(name_ptr, name_len);
    let picture = (picture_len > 0)
        .then(|| unsafe { std::slice::from_raw_parts(picture_ptr, picture_len as usize) }.to_vec());
    let mut user =
        switch_core::cpu::UserAccount::new(uid_from_halves(uid_lo, uid_hi), &name, picture);
    user.edited_at = edited_at;
    session(handle).staged_users.push(user);
}

/// Install the staged users with the given one playing. Returns 0, or (changing
/// nothing) 1 for no users or more than eight, 2 for a zero uid, 3 for a duplicate
/// uid, 4 for a playing user not in the list. The staged list is emptied either way.
#[no_mangle]
pub extern "C" fn switch_users_commit(handle: u32, current_lo: u64, current_hi: u64) -> u32 {
    use switch_core::cpu::UsersRefused;
    let s = session(handle);
    let users = std::mem::take(&mut s.staged_users);
    match s
        .cpu
        .set_users(users, uid_from_halves(current_lo, current_hi))
    {
        Ok(()) => 0,
        Err(UsersRefused::Count) => 1,
        Err(UsersRefused::ZeroUid) => 2,
        Err(UsersRefused::DuplicateUid) => 3,
        Err(UsersRefused::UnknownCurrent) => 4,
    }
}

/// Whether the guest has edited a profile since the last call.
#[no_mangle]
pub extern "C" fn switch_take_profile_edits(handle: u32) -> u32 {
    u32::from(session(handle).cpu.take_profile_edits())
}

/// The users as JSON: `[{"uid":"<32 hex digits>","nickname":"Player","editedAt":0,
/// "pictureLen":0}]`. Pictures come from `switch_user_picture`.
#[no_mangle]
pub extern "C" fn switch_users_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let mut out = Vec::from("[");
    for (i, user) in session(handle).cpu.users().iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(b"{\"uid\":\"");
        for byte in user.uid {
            out.extend_from_slice(format!("{byte:02x}").as_bytes());
        }
        out.extend_from_slice(b"\",\"nickname\":\"");
        json_escape(&user.nickname, &mut out);
        out.extend_from_slice(
            format!(
                "\",\"editedAt\":{},\"pictureLen\":{}}}",
                user.edited_at,
                user.picture.as_ref().map_or(0, Vec::len)
            )
            .as_bytes(),
        );
    }
    out.push(b']');
    let n = out.len().min(maxlen as usize);
    let dst = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    dst.copy_from_slice(&out[..n]);
    n as u32
}

/// Copy a user's picture into `buf`. Returns bytes copied, or 0.
#[no_mangle]
pub extern "C" fn switch_user_picture(
    handle: u32,
    uid_lo: u64,
    uid_hi: u64,
    buf: *mut u8,
    maxlen: u32,
) -> u32 {
    let uid = uid_from_halves(uid_lo, uid_hi);
    let users = session(handle).cpu.users();
    let Some(picture) = users
        .iter()
        .find(|user| user.uid == uid)
        .and_then(|user| user.picture.as_ref())
    else {
        return 0;
    };
    let n = picture.len().min(maxlen as usize);
    let dst = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    dst.copy_from_slice(&picture[..n]);
    n as u32
}

/// Set the wall-clock time `time:u`/`time:s` report, as POSIX seconds (UTC).
#[no_mangle]
pub extern "C" fn switch_set_time(handle: u32, unix_seconds: i64) {
    session(handle).cpu.set_unix_time(unix_seconds);
}

/// Set the battery level `psm` reports.
#[no_mangle]
pub extern "C" fn switch_set_battery(handle: u32, percent: u32, charging: u32) {
    session(handle)
        .cpu
        .set_battery(percent.min(100) as u8, charging != 0);
}

/// Run up to `max_steps` instructions. Returns steps executed or -1 on error.
#[no_mangle]
pub extern "C" fn switch_run(handle: u32, max_steps: u64) -> i64 {
    let s = session(handle);
    match s.cpu.run(max_steps) {
        Ok(report) => report.steps as i64,
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

/// Copy the last `fatal:u` report for this program, if it made one.
#[no_mangle]
pub extern "C" fn switch_guest_fatal(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let fatal = session(handle).cpu.guest_fatal().unwrap_or("");
    write_into(buf, maxlen, fatal.as_bytes())
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

/// The guest clock in CPU cycles. It idles forward when every thread is blocked;
/// [`switch_get_steps`] is the instruction count.
#[no_mangle]
pub extern "C" fn switch_get_cycles(handle: u32) -> u64 {
    session(handle).cpu.cycles
}

/// Instructions actually retired.
#[no_mangle]
pub extern "C" fn switch_get_steps(handle: u32) -> u64 {
    session(handle).cpu.steps
}

/// Guest RAM backed by host storage, in bytes.
#[no_mangle]
pub extern "C" fn switch_guest_ram(handle: u32) -> u64 {
    session(handle).cpu.mem.mapped_bytes()
}

// ---- small JSON helpers ----

/// Escape a string into a JSON string body, per character. Non-ASCII is emitted as UTF-8.
fn json_escape(s: &str, out: &mut Vec<u8>) {
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            c if (c as u32) < 0x20 => {
                out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes());
            }
            c => out.extend_from_slice(c.encode_utf8(&mut [0u8; 4]).as_bytes()),
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

// Host builds only: these use `set_host_container` and a `Mutex`.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// Serializes tests: the session table is not thread-safe under `cargo test`.
    static HOST: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A session and the lock. Restores the default panic hook so failures are readable.
    fn new_session() -> (std::sync::MutexGuard<'static, ()>, u32) {
        let guard = HOST.lock().unwrap_or_else(|e| e.into_inner());
        let handle = switch_new();
        let _ = std::panic::take_hook();
        (guard, handle)
    }

    fn json_from(fill: impl Fn(*mut u8, u32) -> u32) -> String {
        let cap = 1024 * 1024;
        let mut buf = vec![0u8; cap];
        let n = fill(buf.as_mut_ptr(), cap as u32);
        String::from_utf8(buf[..n as usize].to_vec()).unwrap()
    }

    /// The raw JSON text of a `"key":` field.
    fn field<'a>(json: &'a str, key: &str) -> &'a str {
        let at = json
            .find(&format!("\"{key}\":"))
            .unwrap_or_else(|| panic!("no {key} in {json:.400}"));
        let rest = &json[at + key.len() + 3..];
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        for (i, c) in rest.char_indices() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    in_string = false;
                    if depth == 0 {
                        return &rest[..i + 1];
                    }
                }
                continue;
            }
            match c {
                '"' => in_string = true,
                '{' | '[' => depth += 1,
                '}' | ']' if depth > 0 => {
                    depth -= 1;
                    if depth == 0 {
                        return &rest[..i + 1];
                    }
                }
                ',' | '}' | ']' if depth == 0 => return &rest[..i],
                _ => {}
            }
        }
        rest
    }

    #[test]
    fn a_panic_message_is_cut_between_characters_not_through_one() {
        // Truncation must not split a character.
        let msg = "PANIC: ⚠⚠⚠⚠";
        for limit in 0..msg.len() {
            let n = floor_char_boundary(msg, limit);
            assert!(n <= limit);
            assert!(
                std::str::from_utf8(&msg.as_bytes()[..n]).is_ok(),
                "cut at {limit} split a character"
            );
        }
        assert_eq!(floor_char_boundary(msg, msg.len() + 10), msg.len());
    }

    #[test]
    fn the_page_turns_every_trace_channel_on_and_off() {
        let _host = HOST.lock().unwrap_or_else(|e| e.into_inner());
        switch_set_trace_channels(1);
        let all_on = switch_core::trace::ALL
            .iter()
            .all(|&channel| switch_core::trace::enabled(channel));
        switch_set_trace_channels(0);
        assert!(all_on);
        assert_eq!(switch_core::trace::mask(), 0);
    }

    #[test]
    fn the_activity_report_names_each_file_and_hands_it_over_once() {
        let (_host, handle) = new_session();
        let json = json_from(|buf, cap| switch_activity_json(handle, buf, cap));
        assert_eq!(field(&json, "draws"), "0");
        assert_eq!(field(&json, "files"), "[]");
        assert_eq!(field(&json, "gpu"), "[]");
        assert_eq!(field(&json, "threadLog"), "[]");
        assert!(field(&json, "threads").contains("\"index\":0"), "{json}");
        assert!(field(&json, "threads").contains("\"idleMs\":"), "{json}");
        assert_eq!(field(&json, "journal"), "[]");
        for list in ["refusals", "gaps", "nvErrors"] {
            assert_eq!(field(&json, list), "[]", "{list}");
        }
        assert_eq!(field(&json, "input"), r#"{"supported":0,"presented":1}"#);
        assert_eq!(field(&json, "problemsDropped"), "0");

        let activity = &mut session(handle).cpu.fs_activity;
        activity.read("romfs", 0x100);
        activity.read("romfs", 0x20);
        activity.wrote("sdmc:/cfg.json", 7);
        activity.record("fs OpenFile \"/b\" on sdmc -> 2002-0001 (0x202)".to_owned());
        let json = json_from(|buf, cap| switch_activity_json(handle, buf, cap));
        assert_eq!(
            field(&json, "files"),
            r#"[{"name":"romfs","reads":2,"readBytes":288,"writes":0,"writeBytes":0},{"name":"sdmc:/cfg.json","reads":0,"readBytes":0,"writes":1,"writeBytes":7}]"#
        );
        assert_eq!(
            field(&json, "journal"),
            r#"["fs OpenFile \"/b\" on sdmc -> 2002-0001 (0x202)"]"#
        );
        let again = json_from(|buf, cap| switch_activity_json(handle, buf, cap));
        assert_eq!(field(&again, "files"), "[]", "taken, not read");
        assert_eq!(field(&again, "journal"), "[]", "taken, not read");

        // An answer too small stays parseable and counts what it dropped.
        let activity = &mut session(handle).cpu.fs_activity;
        activity.read(&"x".repeat(1000), 1);
        activity.record("y".repeat(1000));
        let mut buf = vec![0u8; 700];
        let n = switch_activity_json(handle, buf.as_mut_ptr(), buf.len() as u32);
        let small = String::from_utf8(buf[..n as usize].to_vec()).unwrap();
        assert!(small.ends_with('}'), "{small}");
        assert_eq!(field(&small, "files"), "[]");
        assert_eq!(field(&small, "filesDropped"), "1");
        assert_eq!(field(&small, "dropped"), "1");
    }

    #[test]
    fn a_crash_report_names_the_build_even_with_no_session_behind_it() {
        // A report needs no live session.
        let _host = HOST.lock().unwrap_or_else(|e| e.into_inner());
        let json = json_from(|buf, cap| switch_crash_report_json(u32::MAX, buf, cap));
        assert_eq!(field(&json, "session"), "null");
        assert!(field(&json, "version").len() > 2, "{json}");
        assert!(json.starts_with('{') && json.ends_with('}'), "{json}");
    }

    #[test]
    fn a_crash_report_carries_the_run_it_is_about() {
        let (_host, handle) = new_session();
        let cpu = &mut session(handle).cpu;
        cpu.diagnostic(Level::Error, "[test] the thing that went wrong");
        session(handle).last_error = "a fault worth reporting".to_string();

        let json = json_from(|buf, cap| switch_crash_report_json(handle, buf, cap));
        assert_eq!(field(&json, "lastError"), "\"a fault worth reporting\"");
        for key in [
            "version",
            "panicked",
            "title",
            "cpu",
            "jit",
            "gpu",
            "backtrace",
            "registers",
            "threads",
            "unimplemented",
            "stubbed",
            "trace",
        ] {
            assert!(
                !field(&json, key).is_empty(),
                "{key} is empty in {json:.400}"
            );
        }
        assert!(
            field(&json, "trace").contains("the thing that went wrong"),
            "the trace has to carry what was said: {json:.400}"
        );
        assert!(field(&json, "backtrace").starts_with('['));
        assert!(field(&json, "registers").contains("pc="));
    }

    #[test]
    fn the_threads_a_guest_parked_can_be_released_from_the_browser() {
        let (_host, handle) = new_session();
        assert!(!json_from(|buf, cap| { switch_thread_dump(handle, buf, cap) }).is_empty());
        assert_eq!(switch_wake_blocked(handle), 0);
        assert_eq!(switch_start_created_threads(handle), 0);

        // Two threads, one started; main then waits on a zero word, yielding to it.
        let cpu = &mut session(handle).cpu;
        let mut create = || {
            guest_svc(cpu, 0x08, &[0, 0x0800_1000, 0, 0x2880_0000, 44, 0]);
            assert_eq!(cpu.reg(0), 0, "CreateThread failed");
            cpu.reg(1)
        };
        let started = create();
        create();
        guest_svc(cpu, 0x09, &[started]);
        guest_svc(cpu, 0x34, &[0x0800_2000, 2, 0, u64::MAX]);
        assert_eq!(switch_wake_blocked(handle), 1);
        assert_eq!(switch_start_created_threads(handle), 1);
        assert_eq!(switch_start_created_threads(handle), 0);
    }

    /// Run one `svc #imm` with `args` in X0 upward.
    fn guest_svc(cpu: &mut Cpu, imm: u32, args: &[u64]) {
        let pc = 0x0800_0000 + imm * 4;
        cpu.mem.write_u32(pc, 0xD400_0001 | imm << 5).unwrap();
        for (i, &arg) in args.iter().enumerate() {
            cpu.set_reg(i as u8, arg);
        }
        cpu.set_pc(pc);
        cpu.step().unwrap();
    }

    #[test]
    fn what_a_title_asked_for_and_did_not_get_is_a_list_not_a_scrollback() {
        let (_host, handle) = new_session();
        let json = json_from(|buf, cap| switch_unimplemented_json(handle, buf, cap));
        assert_eq!(field(&json, "unimplemented"), "[]");
        assert_eq!(field(&json, "stubbed"), "[]");

        // A request (type 4) for command 5 on an unopened handle.
        let cpu = &mut session(handle).cpu;
        let tls = cpu.tls_base();
        for (i, word) in [4, 8, 0, 0, 0x4943_4653, 0, 5, 0].into_iter().enumerate() {
            cpu.mem.write_u32(tls + i as u32 * 4, word).unwrap();
        }
        guest_svc(cpu, 0x21, &[0x1234]);
        let json = json_from(|buf, cap| switch_unimplemented_json(handle, buf, cap));
        assert_eq!(
            field(&json, "unimplemented"),
            r#"[{"iface":"<untracked session>","cmd":5}]"#
        );
        assert_eq!(field(&json, "stubbed"), "[]");
    }

    /// Reset clears module-level panic and trace state.
    #[test]
    fn a_new_session_inherits_nothing_from_the_one_before_it() {
        let (_host, first) = new_session();

        PANICKED.store(true, Ordering::Relaxed);
        // SAFETY: single-threaded under the `HOST` lock.
        let planted = b"PANIC: the old session died here";
        let guard = unsafe { &mut *PANIC_MSG.get() };
        guard[..planted.len()].copy_from_slice(planted);
        guard[planted.len()] = 0;
        switch_core::trace::emit("[test] the old session traced this");

        switch_free_session(first);
        let second = switch_new();

        let json = json_from(|buf, cap| switch_crash_report_json(second, buf, cap));
        assert_eq!(
            field(&json, "panicked"),
            "false",
            "a reset console has not panicked"
        );
        assert!(
            !field(&json, "trace").contains("the old session traced this"),
            "the new session opened with the old one's trace: {json:.400}"
        );

        let mut buf = [0u8; 256];
        let n = switch_last_error(second, buf.as_mut_ptr(), buf.len() as u32);
        assert_eq!(
            &buf[..n as usize],
            b"",
            "the dead session's panic is not this session's last error"
        );
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
        const USER_LO: u64 = 0x0706_0504_0302_0100;
        const USER_HI: u64 = 0x0f0e_0d0c_0b0a_0908;
        let key = SaveKey::from_halves(SAVE, USER_LO, USER_HI);
        let (_host, handle) = new_session();

        // Restores are not reported as changes.
        let path = "/settings.dat";
        let body = b"saved";
        assert_eq!(
            switch_save_write_file(
                handle,
                SAVE,
                USER_LO,
                USER_HI,
                path.as_ptr(),
                path.len() as u32,
                body.as_ptr(),
                body.len() as u32,
            ),
            0
        );
        assert_eq!(
            switch_save_pending_changes(handle, SAVE, USER_LO, USER_HI),
            0
        );

        // Opening a save is enough to list it; the shared save is a separate one.
        switch_save_create(handle, SAVE, 0, 0);
        let mut ids = [0u8; 128];
        let n = switch_save_ids_json(handle, ids.as_mut_ptr(), ids.len() as u32) as usize;
        assert_eq!(
            std::str::from_utf8(&ids[..n]).unwrap(),
            r#"["0100000000001000","0100000000001000@000102030405060708090a0b0c0d0e0f"]"#
        );

        // A guest write is a change, in the save rather than on the card.
        session(handle)
            .cpu
            .save_data_mut(key)
            .write("/settings.dat", 0, b"12345")
            .unwrap();
        assert_eq!(
            switch_save_pending_changes(handle, SAVE, USER_LO, USER_HI),
            1
        );
        assert_eq!(switch_save_pending_changes(handle, SAVE, 0, 0), 0);
        let mut buf = [0u8; 256];
        let n = switch_save_take_changes_json(
            handle,
            SAVE,
            USER_LO,
            USER_HI,
            buf.as_mut_ptr(),
            buf.len() as u32,
        );
        assert_eq!(
            std::str::from_utf8(&buf[..n as usize]).unwrap(),
            r#"[{"path":"/settings.dat","kind":"file","size":5}]"#
        );
        assert_eq!(
            switch_save_pending_changes(handle, SAVE, USER_LO, USER_HI),
            0
        );
        assert_eq!(session(handle).cpu.fs.entry_type("/settings.dat"), None);

        assert_eq!(
            switch_save_file_size(
                handle,
                SAVE,
                USER_LO,
                USER_HI,
                path.as_ptr(),
                path.len() as u32
            ),
            5
        );
        assert_eq!(
            switch_save_file_size(handle, SAVE, 0, 0, path.as_ptr(), path.len() as u32),
            -1,
            "another user's save, or the shared one, does not have it"
        );
        let mut out = [0u8; 16];
        let read = switch_save_read_file(
            handle,
            SAVE,
            USER_LO,
            USER_HI,
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
    fn users_are_staged_committed_and_read_back() {
        let (_host, handle) = new_session();
        let (ann_lo, ann_hi) = (0x0706_0504_0302_0100u64, 0x0f0e_0d0c_0b0a_0908u64);
        let (ben_lo, ben_hi) = (0x1111u64, 0x2222u64);
        let picture = [0xFFu8, 0xD8, 0xFF, 0xD9];
        let stage = |lo, hi, name: &str, picture: &[u8]| {
            switch_user_stage(
                handle,
                lo,
                hi,
                name.as_ptr(),
                name.len() as u32,
                1_700_000_000,
                picture.as_ptr(),
                picture.len() as u32,
            )
        };
        stage(ann_lo, ann_hi, "Ann \"A\"", &picture);
        stage(ben_lo, ben_hi, "Ben", &[]);
        assert_eq!(switch_users_commit(handle, ben_lo, ben_hi), 0);
        assert_eq!(session(handle).cpu.user_nickname(), "Ben");

        let mut buf = [0u8; 512];
        let n = switch_users_json(handle, buf.as_mut_ptr(), buf.len() as u32) as usize;
        assert_eq!(
            std::str::from_utf8(&buf[..n]).unwrap(),
            concat!(
                r#"[{"uid":"000102030405060708090a0b0c0d0e0f","nickname":"Ann \"A\"","#,
                r#""editedAt":1700000000,"pictureLen":4},"#,
                r#"{"uid":"11110000000000002222000000000000","nickname":"Ben","#,
                r#""editedAt":1700000000,"pictureLen":0}]"#
            )
        );
        let mut out = [0u8; 8];
        assert_eq!(
            switch_user_picture(handle, ann_lo, ann_hi, out.as_mut_ptr(), out.len() as u32),
            4
        );
        assert_eq!(&out[..4], &picture);
        assert_eq!(
            switch_user_picture(handle, ben_lo, ben_hi, out.as_mut_ptr(), out.len() as u32),
            0
        );

        // A list naming a player not in it is refused and changes nothing.
        stage(ann_lo, ann_hi, "Ann", &[]);
        assert_eq!(switch_users_commit(handle, 9, 9), 4);
        assert_eq!(session(handle).cpu.users().len(), 2);
        assert_eq!(switch_take_profile_edits(handle), 0);
    }

    #[test]
    fn the_sd_card_round_trips_through_the_host_entry_points() {
        let (_host, handle) = new_session();

        // Restores are not reported as changes.
        put(handle, "sdmc:/switch/restored.txt", b"hello");
        assert_eq!(switch_sd_pending_changes(handle), 0);
        assert_eq!(take_changes(handle), "[]");

        // A guest write is: `IFile::Write` at offset 0 of a new file.
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
        // Draining clears them.
        assert_eq!(switch_sd_pending_changes(handle), 0);
        assert_eq!(take_changes(handle), "[]");

        let path = "/switch/cfg.json";
        assert_eq!(
            switch_sd_file_size(handle, path.as_ptr(), path.len() as u32),
            7
        );
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

        let dir = "/switch";
        assert_eq!(
            switch_sd_file_size(handle, dir.as_ptr(), dir.len() as u32),
            -1
        );
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

    /// Build a PFS0: header, entry table, string table, payloads.
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

    /// The container is read through `host_read` without being staged.
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

        // Reads are relative to the inner file and stop at its end.
        let mut out = vec![0u8; 32];
        let got = switch_read_file(handle, 0, 0x1000 - 8, out.as_mut_ptr(), out.len() as u32);
        assert_eq!(got, 8);
        assert_eq!(&out[..8], &payload[0x1000 - 8..]);

        let got = switch_read_file(handle, 1, 0, out.as_mut_ptr(), out.len() as u32);
        assert_eq!(got, 5);
        assert_eq!(&out[..5], b"hello");

        // Past a file's end is empty; past the table is an error.
        assert_eq!(switch_read_file(handle, 1, 5, out.as_mut_ptr(), 32), 0);
        assert_eq!(switch_read_file(handle, 7, 0, out.as_mut_ptr(), 32), -1);

        // A non-NCA payload fails with a readable error.
        assert_eq!(switch_load_nca_from_nsp(handle, 0), -1);
        let mut err = vec![0u8; 512];
        let n = switch_last_error(handle, err.as_mut_ptr(), err.len() as u32);
        let text = String::from_utf8(err[..n as usize].to_vec()).unwrap();
        assert!(text.contains("bad magic"), "{text}");

        switch_free_session(handle);
    }

    /// A cartridge image opens through the same entry point as an `.nsp`.
    #[test]
    fn a_cartridge_image_opens_as_a_container() {
        use switch_core::nsp::testing::partition_fs;
        use switch_core::nsp::PartitionKind;

        let (_host, handle) = new_session();
        let payload: Vec<u8> = (0..=255u8).cycle().take(2048).collect();
        let secure = partition_fs(
            PartitionKind::Hfs0,
            &[("program.nca", &payload), ("meta.cnmt.nca", b"cnmt")],
        );
        // The firmware partition's NCAs are not listed as the game's.
        let update = partition_fs(PartitionKind::Hfs0, &[("system.nca", b"firmware")]);
        let image =
            switch_core::xci::testing::cartridge(&[("update", &update), ("secure", &secure)]);
        let size = image.len() as u64;
        set_host_container(image);

        assert_eq!(switch_open_nsp(handle, size), 0);

        let mut buf = vec![0u8; 4096];
        let n = switch_nsp_files_json(handle, buf.as_mut_ptr(), buf.len() as u32);
        let json = String::from_utf8(buf[..n as usize].to_vec()).unwrap();
        assert!(json.contains(r#"{"name":"program.nca""#), "{json}");
        assert!(json.contains(r#"{"name":"meta.cnmt.nca""#), "{json}");
        assert!(!json.contains("system.nca"), "{json}");

        // Offsets are the image's own.
        let mut out = vec![0u8; 16];
        let got = switch_read_file(handle, 0, 0x700, out.as_mut_ptr(), out.len() as u32);
        assert_eq!(got, 16);
        assert_eq!(out[..], payload[0x700..0x710]);

        switch_free_session(handle);
    }

    #[test]
    fn a_path_json_cannot_carry_raw_is_escaped() {
        let (_host, handle) = new_session();
        session(handle).cpu.fs.create_file(r#"/switch/a"b\c"#, 0);
        let json = take_changes(handle);
        assert_eq!(
            json,
            r#"[{"path":"/switch/a\"b\\c","kind":"file","size":0}]"#
        );
        switch_free_session(handle);
    }

    #[test]
    fn a_non_ascii_title_name_survives_the_json() {
        // A `\uXXXX` escape names a code point, so multi-byte characters go out raw.
        let mut out = Vec::new();
        json_escape("JUST DANCE® 2017 — 日本語", &mut out);
        assert_eq!(String::from_utf8(out).unwrap(), "JUST DANCE® 2017 — 日本語");

        let mut out = Vec::new();
        json_escape("a\"b\\c\nd\u{7}e", &mut out);
        assert_eq!(String::from_utf8(out).unwrap(), r#"a\"b\\c\nd\u0007e"#);
    }

    #[test]
    fn an_error_message_keeps_its_last_character() {
        // The NUL comes out of the buffer, not the message.
        let (_host, handle) = new_session();
        const MSG: &str = "no container is open";
        session(handle).last_error = MSG.to_string();

        let mut buf = [0xAAu8; 64];
        let n = switch_last_error(handle, buf.as_mut_ptr(), buf.len() as u32);
        assert_eq!(n as usize, MSG.len());
        assert_eq!(&buf[..n as usize], MSG.as_bytes());
        assert_eq!(buf[n as usize], 0, "the copy has to stay NUL-terminated");

        // A message that does not fit loses exactly what the buffer cannot hold.
        session(handle).last_error = MSG.to_string();
        let mut small = [0xAAu8; 8];
        let n = switch_last_error(handle, small.as_mut_ptr(), small.len() as u32);
        assert_eq!(n as usize, small.len() - 1);
        assert_eq!(&small[..n as usize], &MSG.as_bytes()[..small.len() - 1]);
        assert_eq!(small[small.len() - 1], 0);

        switch_free_session(handle);
    }
}

/// What [`crate::gpu::switch_gpu_open`] answers before a channel exists; the worker retries on it.
#[cfg(feature = "gpu")]
pub(crate) const NO_CHANNEL_YET: &str = "the title has not opened a channel yet";

/// Whether the guest has opened a 3D channel, so a device is not built before it can be used.
#[cfg(feature = "gpu")]
pub(crate) fn gpu_channel_open(handle: u32) -> bool {
    session(handle)
        .cpu
        .nv
        .gpu
        .channels
        .values()
        .next()
        .is_some()
}

/// Install a GPU backend on the session's `Gpu` (not a channel).
#[cfg(feature = "gpu")]
fn install_gpu(handle: u32, gpu: switch_gpu::Gpu) {
    session(handle).cpu.nv.gpu.set_renderer(Box::new(gpu));
}
