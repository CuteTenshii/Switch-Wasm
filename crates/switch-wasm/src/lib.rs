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

mod audio;
mod boot;
mod container;
mod debug;
mod display;
mod input;
mod keyboard;
mod nand;
mod run;
mod stats;
mod storage;
mod system;
mod users;

use switch_core::cpu::Cpu;
use switch_core::nca::Nca;
use switch_core::source::{ByteSource, Window};

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
mod tests;

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
