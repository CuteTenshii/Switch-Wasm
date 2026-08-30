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

#[cfg(feature = "gpu")]
mod gpu;

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
    /// The update the page has added for the title in the open container, if
    /// it has added one. Held as another host file, read only when the title
    /// boots: an update container is as large as any other.
    update: Option<Update>,
    /// The add-on content the page has added, one entry per DLC archive. Also
    /// held as host files and read at boot, when the title id they are
    /// numbered against is finally known.
    dlc: Vec<Dlc>,
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

/// The host read, as a `wasm-bindgen` import.
///
/// With the `gpu` feature the module is a wasm-bindgen module, and
/// wasm-bindgen builds the whole import object itself — there is no seam to
/// hand it an extra `env.host_read` through. So the import moves into its
/// world too, and the module then declares no `env` at all.
///
/// `@host/files` is a bare specifier the bundler resolves (see
/// `vite.config.ts`), because the generated glue sits in cargo's target
/// directory and a relative path from there to `web/worker/` is not a thing
/// worth writing down twice. `ptr` is a `u32` rather than a pointer because
/// wasm-bindgen has no pointer type; it is the same integer either way.
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
        update: None,
        dlc: Vec::new(),
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

/// Open the `size`-byte container the host has ready: read its file table
/// and keep it. Returns 0 on success, -1 on error.
///
/// Either kind of container — an `.nsp`, or a cartridge image whose
/// partitions flatten into the same table — so the page hands both here and
/// everything downstream (the Program NCA scan, the Control NCA, the file
/// list) is one path.
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
/// A host file is not necessarily a file the user picked. The frontend's NAND
/// keeps its content in the browser's own storage and hands it back as a
/// handle the host reads ranges out of, which is what lets a whole firmware
/// dump be re-registered every session for the cost of its headers.
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

/// Register the update container the page is holding as the update for the
/// title it is about to run.
///
/// `file` is a host file index, the same way [`switch_add_archive`] takes
/// one: an update NSP is as large as any other container and is never read
/// through. Only its header, its file table and its ticket are read here.
///
/// Returns the program id the update patches — which is the *base* title's,
/// so the page can pair the two containers by it — or 0 if the file is not an
/// update this build can read, with the reason in `switch_last_error`.
///
/// The pairing is only checked when the title boots: the page may add the
/// update before opening the base container or after, and neither order is
/// the wrong one.
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
    // An update is ticketed separately from the game it patches, so its own
    // title key has to come out of its own container.
    let _ = switch_core::ticket::load_bundled_title_key(&mut s.keys, &nca, &files, &src);
    // A game is not an update, and saying so here is what keeps the page from
    // offering to apply one container to another at random.
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

/// The update's own version, the way its NACP spells it for a reader
/// ("1.0.1"), written into `buf` as UTF-8.
///
/// An update ships its own Control NCA, and the version in it is what a
/// console shows beside the title once the update is installed — so it is what
/// the page shows too, rather than the raw `v65536` in the container's name.
///
/// Empty when the update carries no Control NCA (legal: an update that changes
/// only data need not restate the title's metadata) or when the session has no
/// update at all.
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

/// Register a container of add-on content for the title the page has open.
///
/// `file` is a host file index, as [`switch_add_archive`] takes one: nothing
/// is read but the container's header, its file table and its tickets, and
/// each archive stays where it is until the title mounts it.
///
/// Returns how many pieces of add-on content the container holds — a DLC
/// package usually carries one, but nothing says it must — or 0 if it holds
/// none, with the reason in `switch_last_error`.
///
/// Which title they belong to is not settled here. An add-on content id is a
/// base title's plus an index, and a page may add one before opening the game
/// it goes with; the pairing is checked at boot, against the id the title
/// actually declares.
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
    // A container with a Program NCA is a game or an update, whatever else it
    // also holds. Saying so here is what keeps the page from offering to
    // "add" a title to itself.
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
        // Each piece is ticketed on its own — a DLC is bought separately from
        // the game and from every other piece of it.
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

/// The add-on content this session holds, as JSON: the content id, the index
/// it is numbered with, and the base title it belongs to.
///
/// The page pairs on the base title so it can say which game a piece of
/// content is waiting for; the title itself settles it at boot.
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

/// Forget the add-on content this session holds, so the next boot is the
/// title on its own.
#[no_mangle]
pub extern "C" fn switch_clear_dlc(handle: u32) {
    session(handle).dlc.clear();
}

/// Forget the update this session had, so the next boot is the plain title.
#[no_mangle]
pub extern "C" fn switch_clear_update(handle: u32) {
    session(handle).update = None;
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
    // No update: what the NAND holds is system content, and the page pairs
    // an update with a container the user picked, not with an applet.
    load_and_boot_nca(
        &s.keys,
        &mut s.cpu,
        &mut s.last_error,
        switch_core::source::MemSource(bytes),
        Added::default(),
    )
}

/// An update container the page has added for the title it is about to run.
///
/// An update NSP holds no game: its Program NCA carries the patched modules
/// in full — so an update runs by booting *its* ExeFS — and a RomFS section
/// holding only the ranges the update changed, which reads over the base
/// container's RomFS and nowhere else. Both files stay with the browser; this
/// is a handle on the second one, exactly as `container` is on the first.
struct Update {
    /// The update's Program NCA, parsed. Its program id is the *base* title's
    /// — the `...800` update id lives only on the container's Meta NCA — so
    /// this is what a base container is paired against.
    nca: Nca,
    src: HostSource,
    /// Where the Program NCA sits inside the update container.
    program: (u64, u64),
    /// The update container's file table, kept for its Control NCA — an
    /// update states its own version, and that is what the page shows.
    files: Vec<switch_core::nsp::Pfs0File>,
}

impl Update {
    /// A fresh window over the update's Program NCA. Each reader wants its
    /// own, and a [`HostSource`] is a handle rather than a buffer, so this
    /// costs nothing.
    fn program_window(&self) -> Result<Window<HostSource>, switch_core::Error> {
        Window::new(
            self.src,
            self.program.0,
            self.program.1,
            "update program nca",
        )
    }
}

/// One piece of add-on content the page has added for the title it is about
/// to run.
///
/// A DLC container is nothing like an update: no Program NCA, no patch, no
/// base to read over. It is one Data NCA with an ordinary RomFS, whose title
/// id is the title's add-on base plus an index — and a title mounts it by
/// that id exactly as it mounts a system data archive. All `aoc:u` adds is
/// telling the title the index exists.
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

/// What the page has added beside the container being launched.
#[derive(Default, Clone, Copy)]
struct Added<'a> {
    update: Option<&'a Update>,
    dlc: &'a [Dlc],
}

/// Whether a title id is an add-on content id: a base title's, plus an index.
/// The add-on offset is 0x1000 and the index is 11 bits, so `...1001` is a
/// title's DLC #1 and `...0800` — an update — is not add-on content at all.
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
                // The section entry itself carries no filesystem-type byte;
                // that only lives in the (separately encrypted) FS header,
                // which needs the full NCA_FULL_HEADER_SIZE-byte header to
                // decrypt. The lightweight "inspect this NCA" path may only
                // have the base header, in which case this is unknown.
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

/// Cache a title's control data on the session, and tell the CPU what the
/// NACP inside it says — the save-data figures the `IApplicationFunctions`
/// commands report back, and the id its add-on content is numbered from.
///
/// The NACP is the only place those exist, and it is in the Control NCA
/// rather than the Program one — so a title launched without the control
/// having been read gets the CPU's defaults instead. Reading it is what the
/// page already does to show the title's name and icon, which is why this is
/// the point they arrive.
fn cache_control(s: &mut Session, control: switch_core::control::Control) {
    s.cpu
        .set_save_data_quota(switch_core::cpu::SaveDataQuota::from(&control.nacp));
    s.cpu
        .set_add_on_content_base_id(control.nacp.add_on_content_base_id);
    s.control = Some(control);
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
    s.control = None;
    match s.cpu.boot_homebrew(data) {
        Ok(loaded) => {
            // The NRO's own icon and name, for `switch_control_json` and
            // `switch_control_icon` to answer with. Cached for display only,
            // not through `cache_control`: HBL runs homebrew inside another
            // title's process, so its NACP never governs the save data — and
            // the figures in one are `nacptool`'s boilerplate anyway.
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
/// `added` is what the page put beside the container: an update, whose modules
/// and patched RomFS replace this one's, and add-on content, which mounts
/// alongside it once the title id it is numbered against is known.
///
/// Takes `keys`/`cpu`/`last_error` as separate borrows rather than `&mut
/// Session` so the caller can keep reading the session's file table while
/// this holds `&mut Cpu`.
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
    // An update whose title id is not this one is a mistake worth stopping
    // for: applying it would compose two unrelated RomFS images, and ignoring
    // it would silently launch a version the page said it was not launching.
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
    // What boots is the update's Program NCA when there is one: an update's
    // ExeFS is a complete replacement set of modules, not a delta.
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
        cpu.diagnostic(&format!(
            "[update] booting the update's modules for {:016x}, over this container's RomFS",
            nca.program_id
        ));
    }
    // Say whether the bytes about to be executed were checked, not just that
    // they decrypted: the master hash only vouches for the section's hash
    // table, and a fault inside a title's own crt0 is worth chasing in the
    // title only once the image it ran on is known to be intact.
    match program.pfs0_hash_coverage(exefs_index) {
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
    cpu.set_program_id(program.program_id);

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
    match update {
        // An update's own RomFS holds only what it changed, indexed against
        // the base game's — the two are only readable together.
        Some(u) => {
            let patched = u.program_window().and_then(|window| {
                switch_core::bktr::patched_romfs_source(&u.nca, window, &nca, nca_src, keys)
            });
            match patched {
                Ok(romfs) => cpu.set_romfs_source(Box::new(romfs)),
                Err(e) => cpu.diagnostic(&format!("the update's RomFS is unreadable: {}", e)),
            }
        }
        None => {
            if let Some(romfs_index) = nca.romfs_section_index() {
                match nca.romfs_source(nca_src, keys, romfs_index) {
                    Ok(romfs) => cpu.set_romfs_source(Box::new(romfs)),
                    Err(e) => cpu.diagnostic(&format!("romfs unavailable: {}", e)),
                }
            }
        }
    }

    // The address space this title gets, chosen by its own manifest: a title
    // declaring no system resource keeps the plain heap and the larger total
    // memory, one declaring a system resource gets virtual address memory and
    // the layout that pays for it. Must precede `boot_retail_program` —
    // `nn::init` reads the resulting figures as soon as it runs.
    let system_resource = switch_core::npdm::Npdm::system_resource_size_of(&pfs0, &exefs);
    cpu.diagnostic(&format!(
        "[npdm] system resource {system_resource:#x} — {}",
        if system_resource == 0 {
            "plain heap"
        } else {
            "virtual address memory"
        }
    ));
    cpu.set_system_resource_size(system_resource);

    // And which instruction set it runs, from bit 0 of the same manifest's
    // flags. Also before the boot: the entry ABI puts the return trampoline in
    // a different register in each state.
    if !switch_core::npdm::Npdm::is_64_bit_of(&pfs0, &exefs) {
        cpu.diagnostic("[npdm] AArch32 title — running the A32 interpreter");
        cpu.set_mode(switch_core::cpu::ExecMode::A32);
    }

    match cpu.boot_retail_program(&modules) {
        Ok(loaded) => {
            // After the modules, not before: booting clears the diagnostic
            // buffer, and what mounted is worth saying where it can be read.
            // Nothing has run yet, so nothing can have asked for content that
            // is not there.
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

/// Give the title the add-on content the page added for it.
///
/// Called once the program id is set, because that — with the NACP's own
/// override, which arrives with the Control NCA — is what an add-on content
/// id is numbered against. Content belonging to another title is reported and
/// skipped rather than mounted under an id nothing will ask for.
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
                    Some(index) => cpu.diagnostic(&format!(
                        "[aoc] {:016x} mounted as add-on content {index}, {size:#x} bytes",
                        entry.content_id
                    )),
                    None => cpu.diagnostic(&format!(
                        "[aoc] {:016x} is not this title's add-on content — not mounted",
                        entry.content_id
                    )),
                }
            }
            Err(e) => cpu.diagnostic(&format!(
                "[aoc] {:016x} could not be read: {e}",
                entry.content_id
            )),
        }
    }
}

/// Decrypt the open container as a standalone Program NCA (using whatever
/// keys are loaded), extract its ExeFS `main` executable and boot it. Returns
/// entry address or -1 — check `switch_last_error` either way, since the
/// entry can legitimately be 0 for some NSO layouts.
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

/// The index of the Program NCA in the open container, or -1 if it has none.
///
/// This is what lets a container be booted without being read through first:
/// every file in an NSP is named after its own hash, so which one holds the
/// title's executable is visible only in each NCA's (encrypted) header. -1
/// therefore also covers having no `prod.keys` loaded, and `switch_last_error`
/// says which of the two it was.
///
/// Hand the answer straight back to `switch_load_nca_from_nsp`.
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

/// Decrypt NSP file `index` as a Program NCA (using whatever keys are
/// loaded), extract its ExeFS `main` executable and boot it. Returns entry
/// address or -1 — check `switch_last_error` either way, since the entry can
/// legitimately be 0 for some NSO layouts.
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
    // An ELF carries no control data, and what is left of the last title's
    // would be reported as this one's.
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
        "{{\"enabled\":{},\"blocks\":{},\"translated\":{},\"executed\":{},\"invalidated\":{},\"interpreted\":{}}}",
        s.cpu.jit_enabled(),
        stats.blocks,
        stats.translated,
        stats.executed,
        stats.invalidated,
        stats.interpreted
    );
    write_into(buf, maxlen, json.as_bytes())
}

/// What the installed GPU backend has been doing, as JSON.
///
/// `{}` while the software rasterizer has the frame — it never declines a
/// draw and has nothing to report. A device backend answers its draw and
/// fallback counts, every distinct reason a draw fell back, whether the
/// software-frame latch has tripped, and where its time went.
///
/// Asked for rather than printed because a browser is where these matter and
/// the browser is exactly where they could not be had: `eprintln!` goes
/// nowhere, and the env vars that gate them natively are always empty on
/// wasm32.
#[no_mangle]
pub extern "C" fn switch_gpu_report_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    // The frame count comes from here rather than the backend, which has no
    // idea what a frame is — it sees clears and draws. Without it every
    // reading has to be normalised by draws, and a flush costs what a frame
    // costs however many draws went into it: a frame that only clears the
    // screen reads the whole target back exactly like one that draws it.
    let frames = s.cpu.nv.gpu.frames;
    let json = s.cpu.nv.gpu.renderer_report();
    let json = match json.strip_suffix('}') {
        Some(body) if body.len() > 1 => format!("{body},\"frames\":{frames}}}"),
        _ => format!("{{\"frames\":{frames}}}"),
    };
    write_into(buf, maxlen, json.as_bytes())
}

/// Whether the installed GPU backend has lost its device and wants replacing.
///
/// Cheap by design — the worker asks after every slice, and a JSON report
/// parsed that often would cost more than the answer is worth.
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
    s.cpu
        .fs
        .write_file(&sd_path(path_ptr, path_len), data.to_vec());
    0
}

/// Create a directory on the SD card and any missing parents. Same "host load
/// path" reasoning as `switch_sd_write_file`: not reported as a change.
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
    s.cpu
        .save_data_mut(save_id)
        .write_file(&path, data.to_vec());
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

/// Dock or undock the console: 0 handheld, anything else docked.
///
/// Safe to call while a title is running, which is the point of it — a real
/// console is docked mid-game and the title is expected to cope. What makes
/// it cope is the pair of AM messages this queues, not the number itself: a
/// title reads the operation mode once and lays out for it, and only goes
/// back to ask when it is told the mode changed.
#[no_mangle]
pub extern "C" fn switch_set_operation_mode(handle: u32, docked: u32) {
    let mode = if docked == 0 {
        switch_core::cpu::OperationMode::Handheld
    } else {
        switch_core::cpu::OperationMode::Docked
    };
    session(handle).cpu.set_operation_mode(mode);
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
    session(handle)
        .cpu
        .set_battery(percent.min(100) as u8, charging != 0);
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

/// The guest clock, in cycles of the CPU `svcGetSystemTick` is scaled from.
///
/// Not an instruction count: it idles forward to the earliest sleeper when
/// every thread is blocked. [`switch_get_steps`] is the count.
#[no_mangle]
pub extern "C" fn switch_get_cycles(handle: u32) -> u64 {
    session(handle).cpu.cycles
}

/// Instructions actually retired, which the guest's idle never advances.
#[no_mangle]
pub extern "C" fn switch_get_steps(handle: u32) -> u64 {
    session(handle).cpu.steps
}

/// Guest RAM currently backed by host storage, in bytes — the emulated
/// console's memory use, not the wasm heap's.
#[no_mangle]
pub extern "C" fn switch_guest_ram(handle: u32) -> u64 {
    session(handle).cpu.mem.mapped_bytes()
}

// ---- small JSON helpers ----

/// Escape a string into a JSON string body.
///
/// Per **character**, not per byte. A `\uXXXX` escape in JSON names a code
/// point, so escaping the bytes of a multi-byte character one at a time spells
/// a different string: ® (U+00AE, UTF-8 `C2 AE`) came out as
/// `\u00c2\u00ae`, which a parser reads back as Â®. A title's name
/// comes straight out of its NACP and is full of characters like it, so
/// *JUST DANCE® 2017* reached the page as *JUST DANCEÂ® 2017*.
///
/// Everything above ASCII is emitted as itself: a JSON document is UTF-8, and
/// the page decodes this buffer with a UTF-8 `TextDecoder`.
fn json_escape(s: &str, out: &mut Vec<u8>) {
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            // The rest of C0 has no shorthand and cannot appear raw.
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
        assert_eq!(
            std::str::from_utf8(&ids[..n]).unwrap(),
            r#"["0100000000001000"]"#
        );

        // A guest write is a change, and it lands in the save rather than on
        // the card — the two are different storage, and a title's save is not
        // something the next title to mount the card should find.
        session(handle)
            .cpu
            .save_data_mut(SAVE)
            .write("/settings.dat", 0, b"12345")
            .unwrap();
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

    /// A cartridge image is opened through the same entry point as an
    /// `.nsp`, and presents the page the same file list — the whole of what
    /// the browser needed to boot one.
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
        // A cartridge carries a firmware bundle beside the title, and the
        // page must not be shown its NCAs as if they were the game's.
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

        // The offsets in it are the image's own, so a read through them lands
        // on the file and not on a partition header.
        let mut out = vec![0u8; 16];
        let got = switch_read_file(handle, 0, 0x700, out.as_mut_ptr(), out.len() as u32);
        assert_eq!(got, 16);
        assert_eq!(out[..], payload[0x700..0x710]);

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
        assert_eq!(
            json,
            r#"[{"path":"/switch/a\"b\\c","kind":"file","size":0}]"#
        );
        switch_free_session(handle);
    }

    #[test]
    fn a_non_ascii_title_name_survives_the_json() {
        // A `\uXXXX` escape names a code point, so escaping a multi-byte
        // character one byte at a time spells something else entirely — and
        // every retail title's name is full of them. This went out as
        // `\u00c2\u00ae` and came back as "JUST DANCEÂ® 2017".
        let mut out = Vec::new();
        json_escape("JUST DANCE® 2017 — 日本語", &mut out);
        assert_eq!(String::from_utf8(out).unwrap(), "JUST DANCE® 2017 — 日本語");

        // What JSON genuinely cannot carry raw still goes out escaped.
        let mut out = Vec::new();
        json_escape("a\"b\\c\nd\u{7}e", &mut out);
        assert_eq!(String::from_utf8(out).unwrap(), r#"a\"b\\c\nd\u0007e"#);
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

/// What [`crate::gpu::switch_gpu_open`] answers before the guest has opened a
/// channel. Named because the worker matches on it to know the attempt is
/// worth repeating rather than abandoning.
#[cfg(feature = "gpu")]
pub(crate) const NO_CHANNEL_YET: &str = "the title has not opened a channel yet";

/// Whether the guest has opened a 3D channel yet.
///
/// The backend no longer goes on a channel, so this is not about where it
/// lands — it is about not building one before the title can use it.
/// `requestDevice` builds a real device in the GPU process whether or not
/// anything will draw, and wgpu's web backend frees nothing when one is
/// dropped. The Home Menu opens its channel 11.6M steps in, which against a
/// 1M-step run slice is eleven attempts before the twelfth lands.
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

/// Install a GPU backend on a session.
///
/// It goes on the session's one `Gpu`, not on a channel. A title may have
/// several channels — Asphalt 9 opens four — and picking one of them left the
/// device on a channel the title never drew through, which reads from outside
/// as a device that opened and a rasterizer that kept the frame.
#[cfg(feature = "gpu")]
fn install_gpu(handle: u32, gpu: switch_gpu::Gpu) {
    session(handle).cpu.nv.gpu.set_renderer(Box::new(gpu));
}
