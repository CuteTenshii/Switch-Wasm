//! WASM bindings for switch-core. Every export goes through `wasm-bindgen`. The one session is
//! a global, structured results are hand-written JSON, and a failure's reason is thrown to
//! the caller as a `Result`'s error (a trap's is the panic hook's message,
//! `switch_panic_message`).

use wasm_bindgen::prelude::wasm_bindgen;

/// A `Sync` wrapper for interior mutability touched only by the worker that calls exports.
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

#[cfg(target_arch = "wasm32")]
mod blob;
mod gpu;
#[cfg(target_arch = "wasm32")]
mod heap;
mod hostfiles;
#[cfg(target_arch = "wasm32")]
mod jit;
#[cfg(target_arch = "wasm32")]
mod screen;
#[cfg(target_arch = "wasm32")]
mod video;

mod activity;
#[cfg(target_arch = "wasm32")]
mod applets;
mod audio;
mod boot;
mod container;
mod cores;
mod debug;
mod display;
#[cfg(target_arch = "wasm32")]
mod idb;
mod input;
mod keyboard;
mod log;
mod motion;
mod nand;
mod net;
#[cfg(target_arch = "wasm32")]
mod opfs;
#[cfg(target_arch = "wasm32")]
mod post;
mod run;
mod stats;
mod storage;
mod system;
mod users;

#[cfg(target_arch = "wasm32")]
use hostfiles::read as host_read;
use switch_core::cpu::Cpu;
use switch_core::nca::Nca;
use switch_core::source::{ByteSource, Window};

/// Framebuffer base address, width, height and stride (RGBA, little-endian).
pub use switch_core::{FB_BASE, FB_HEIGHT, FB_STRIDE, FB_WIDTH};
/// Memory-mapped input register: JS writes an ASCII key, homebrew acknowledges with 0.
pub const INPUT_ADDR: u64 = switch_core::INPUT_ADDR;

struct Session {
    /// The open container, read from the host by range.
    container: Option<HostSource>,
    nsp_files: Vec<switch_core::nsp::Pfs0File>,
    /// The open container's NCAs whose headers parse, with their index in `nsp_files`.
    ncas: Vec<(usize, switch_core::nca::Nca)>,
    /// The update added for the open title, held as a host file until boot.
    update: Option<Update>,
    /// Add-on content, one entry per DLC archive, held as host files until boot.
    dlc: Vec<Dlc>,
    keys: switch_core::keys::KeySet,
    /// Control data from the last Control NCA read, cached so the icon can be fetched separately.
    control: Option<switch_core::control::Control>,
    /// The icon of the last file `switch_peek_title` read, until `switch_peek_icon` takes it.
    peeked_icon: Vec<u8>,
    /// The open container's NACP, given to the process each launch of it boots in.
    container_nacp: Option<switch_core::control::Nacp>,
    /// Users staged by `switch_user_stage`, installed whole by `switch_users_commit`.
    staged_users: Vec<switch_core::cpu::UserAccount>,
    cpu: Cpu,
    last_error: String,
    activity: activity::Reporter,
}

// Only the system worker calls exports, each to completion; a `Mutex` would deadlock on reentry.
static SESSION: SyncCell<Option<Session>> = SyncCell::new(None);

/// Last panic message, in a fixed buffer readable without a live session.
static PANIC_MSG: SyncCell<[u8; 2048]> = SyncCell::new([0u8; 2048]);

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

fn session() -> &'static mut Session {
    session_if_live().expect("there is no session")
}

/// The session, or `None` before `switch_init`.
fn session_if_live() -> Option<&'static mut Session> {
    // SAFETY: only the system worker calls exports; see the `SESSION` comment.
    unsafe { &mut *SESSION.get() }.as_mut()
}

/// Host-build stand-in serving [`set_host_container`].
#[cfg(not(target_arch = "wasm32"))]
fn host_read(file: u32, offset: u64, out: &mut [u8]) -> usize {
    if file != 0 {
        return 0; // host builds serve only the container
    }
    // SAFETY: only the system worker calls exports; host builds hold the test lock.
    let data = unsafe { &*HOST_CONTAINER.get() };
    if offset >= data.len() as u64 {
        return 0;
    }
    let start = offset as usize;
    let n = out.len().min(data.len() - start);
    out[..n].copy_from_slice(&data[start..start + n]);
    n
}

#[cfg(not(target_arch = "wasm32"))]
static HOST_CONTAINER: SyncCell<Vec<u8>> = SyncCell::new(Vec::new());

/// Install the bytes host builds serve as the open container.
#[cfg(not(target_arch = "wasm32"))]
pub fn set_host_container(data: Vec<u8>) {
    // SAFETY: only the system worker calls exports; see the `SESSION` comment.
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
        let done = host_read(self.file, offset, &mut out[..want]);
        if done != want {
            return Err(switch_core::Error::Io(format!(
                "host read of {} bytes at {:#x} returned {}",
                want, offset, done
            )));
        }
        Ok(done)
    }
}

/// Set up the module and build its machine, replacing any earlier one (which keeps its
/// logged lines). Call right after instantiating.
#[wasm_bindgen]
pub fn switch_init() {
    install_panic_hook();
    attach_jit();
    forget_the_last_session();
    input::reset();
    if let Some(old) = session_if_live() {
        old.cpu.stop_cores();
        log::collect(&mut old.cpu);
    }
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.set_shared_font(include_bytes!("../../../web/font.ttf").to_vec());
    // SAFETY: only the system worker calls exports; see the `SESSION` comment.
    unsafe {
        *SESSION.get() = Some(Session {
            container: None,
            nsp_files: Vec::new(),
            ncas: Vec::new(),
            update: None,
            dlc: Vec::new(),
            keys: switch_core::keys::KeySet::default(),
            control: None,
            peeked_icon: Vec::new(),
            container_nacp: None,
            staged_users: Vec::new(),
            cpu,
            last_error: String::new(),
            activity: activity::Reporter::default(),
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        let engines = &mut session().cpu.nv.video;
        engines.set_h264(Box::new(video::WebDecoder::new(video::H264)));
        engines.set_vp9(video::vp9_decoder);
        video::probe_vp9();
        audio::announce_ring();
        input::announce_mailbox();
        storage::start_claim();
    }
}

/// Let the core run the blocks it emits, where the build can compile them.
fn attach_jit() {
    #[cfg(target_arch = "wasm32")]
    jit::attach();
}

/// Capture Rust panics into a static buffer for [`switch_panic_message`], since they
/// otherwise trap silently.
fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let msg = format!("PANIC: {info}");
            // SAFETY: only the system worker runs guest code, so panics do not overlap.
            let guard = unsafe { &mut *PANIC_MSG.get() };
            // Truncate by character so the page's UTF-8 decoder gets a whole string.
            let n = floor_char_boundary(&msg, guard.len() - 1);
            guard[..n].copy_from_slice(&msg.as_bytes()[..n]);
            guard[n] = 0;
        }));
    });
}

/// Clear the panic message the previous session left; its log lines stay queued.
fn forget_the_last_session() {
    // SAFETY: only the system worker calls exports; see the `SESSION` comment.
    unsafe { &mut *PANIC_MSG.get() }.fill(0);
}

/// The Rust panic that trapped the last call, if there was one; reading it forgets it.
#[wasm_bindgen]
pub fn switch_panic_message() -> String {
    // SAFETY: only the system worker calls exports; see the `SESSION` comment.
    let panicked = unsafe { &mut *PANIC_MSG.get() };
    let len = panicked
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(panicked.len());
    let message = String::from_utf8_lossy(&panicked[..len]).into_owned();
    panicked.fill(0);
    message
}

/// Why the last `run` or `nand_identify` failed.
#[wasm_bindgen]
pub fn switch_last_error() -> String {
    session_if_live().map_or_else(String::new, |s| s.last_error.clone())
}

/// An update container held for the title it patches, applied only when that title launches.
/// Its Program NCA holds full replacement modules and a RomFS patch over the base's.
struct Update {
    /// Its program id is the base title's, which is what pairing checks.
    nca: Nca,
    src: HostSource,
    /// Where the Program NCA sits inside the update container.
    program: (u64, u64),
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

/// The open container as a source.
fn container(s: &Session) -> Result<HostSource, String> {
    s.container.ok_or_else(|| "no container is open".to_owned())
}

/// A source over NSP file `index`: the window an NCA is read through.
fn nsp_file_source(s: &Session, index: u32) -> Result<Window<HostSource>, String> {
    let container = container(s)?;
    let f = s
        .nsp_files
        .get(index as usize)
        .ok_or_else(|| "no such NSP file index".to_owned())?;
    Window::new(container, f.offset, f.size, &f.name).map_err(|e| e.to_string())
}

// Host builds only: these use `set_host_container` and a `Mutex`.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

/// Install a GPU backend on the session's `Gpu` (not a channel).
fn install_gpu(gpu: switch_gpu::Gpu) {
    session().cpu.nv.gpu.set_renderer(Box::new(gpu));
    stats::GPU_LOSS_LOGGED.store(false, std::sync::atomic::Ordering::Relaxed);
}
