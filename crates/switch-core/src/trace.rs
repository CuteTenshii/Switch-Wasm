//! Which diagnostics are on, what severity they carry, and where they go.
//!
//! Every trace in this emulator used to be gated by an environment variable
//! read through [`crate::env_flag`], and every one of them wrote to stderr.
//! Neither exists in a browser: `wasm32-unknown-unknown` has no WASI, so
//! `std::env::var` always fails and `eprintln!` goes nowhere. That left the
//! twenty-odd `TRACE_*` switches — the most detailed account this emulator can
//! give of itself — reachable only from the command line, on a project whose
//! target is the browser.
//!
//! So the switches live in a mask that can be set at run time, and what they
//! print goes to a sink the host drains as well as to stderr. The environment
//! still seeds the mask, so a CLI run behaves exactly as it did.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

/// One diagnostic channel. The name is the environment variable that seeds it
/// and the string the host enables it by, so there is one spelling of each
/// switch rather than two that can drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trace {
    /// Guest syscalls, minus the three that fire every scheduling round.
    Svc,
    /// Service requests: the unhandled results, the buffers, the domains.
    Ipc,
    /// Blocking and waking: which thread parked on what, and who released it.
    Wait,
    /// Guest memory maps and unmaps.
    Map,
    /// `nvdrv` ioctls and their results.
    Nv,
    /// Per-method GPU command traces.
    Gpu,
    /// GPU texture binds and the descriptors behind them.
    GpuTex,
    /// Texture decode: formats, swizzles and the surfaces they produce.
    Tex,
    /// Per-draw tallies from the software rasterizer.
    Draw,
    /// The graphics pipeline state a draw was issued with.
    Pipeline,
    /// Vertex, index and constant uploads.
    Upload,
    /// Shader control flow as the translator recovered it.
    Cfg,
    /// The WGSL a shader translated to.
    Wgsl,
    /// Decoded Maxwell shader programs.
    Shader,
    /// Shader program headers.
    Sph,
    /// Raw 3D-engine register writes.
    Regs,
    /// Audio: the renderer's commands and the output stream.
    Audio,
    /// The error-report journal: contexts submitted, reports filed.
    Erpt,
    /// Shared-font requests.
    Font,
}

/// Every channel, in the order the host is offered them.
pub const ALL: [Trace; 19] = [
    Trace::Svc,
    Trace::Ipc,
    Trace::Wait,
    Trace::Map,
    Trace::Nv,
    Trace::Gpu,
    Trace::GpuTex,
    Trace::Tex,
    Trace::Draw,
    Trace::Pipeline,
    Trace::Upload,
    Trace::Cfg,
    Trace::Wgsl,
    Trace::Shader,
    Trace::Sph,
    Trace::Regs,
    Trace::Audio,
    Trace::Erpt,
    Trace::Font,
];

impl Trace {
    /// The channel's bit in the mask. Its position is [`ALL`]'s order, so a
    /// mask is only meaningful against the build that produced it.
    #[inline]
    pub const fn bit(self) -> u32 {
        1 << self as u32
    }

    /// The environment variable that seeds this channel, which is also the
    /// name the host turns it on by.
    pub const fn name(self) -> &'static str {
        match self {
            Trace::Svc => "TRACE_SVC",
            Trace::Ipc => "TRACE_IPC",
            Trace::Wait => "TRACE_WAIT",
            Trace::Map => "TRACE_MAP",
            Trace::Nv => "TRACE_NV",
            Trace::Gpu => "TRACE_GPU",
            Trace::GpuTex => "TRACE_GPU_TEX",
            Trace::Tex => "TRACE_TEX",
            Trace::Draw => "TRACE_DRAW",
            Trace::Pipeline => "TRACE_PIPELINE",
            Trace::Upload => "TRACE_UPLOAD",
            Trace::Cfg => "TRACE_CFG",
            Trace::Wgsl => "TRACE_WGSL",
            Trace::Shader => "TRACE_SHADER",
            Trace::Sph => "TRACE_SPH",
            Trace::Regs => "TRACE_REGS",
            Trace::Audio => "TRACE_AUDIO",
            Trace::Erpt => "TRACE_ERPT",
            Trace::Font => "TRACE_FONT",
        }
    }

    /// The channel a name spells, if any.
    pub fn from_name(name: &str) -> Option<Trace> {
        ALL.iter().copied().find(|t| t.name() == name)
    }
}

/// How much a diagnostic matters. The host colours by this and can filter on
/// it; without it a title's fatal abort and a stubbed-out command arrive
/// looking identical, which is how the interesting one gets scrolled past.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// The emulator or the guest has failed at something it was asked to do.
    Error,
    /// Something was answered wrongly or not at all, and a later failure is
    /// likely to be this one's fault.
    Warn,
    /// A milestone worth seeing on every run.
    Info,
    /// A mask-gated trace: on only because someone asked for it.
    Debug,
}

impl Level {
    /// The byte that carries this level through the trace buffer.
    ///
    /// The buffer is a stream of text the host splits into lines, so the
    /// level has to travel *in* the text. A leading control byte is the one
    /// marker that cannot be confused with a line's content: `[fatal]` and
    /// `0x01` are both things a disassembly line can start with, and a
    /// register dump contains every printable character there is.
    ///
    /// Lines with no marker inherit the level of the line before them, so a
    /// fault's register dump and instruction trail stay with the fault rather
    /// than reverting to the default.
    #[inline]
    pub const fn marker(self) -> u8 {
        match self {
            Level::Error => 0x01,
            Level::Warn => 0x02,
            Level::Info => 0x03,
            Level::Debug => 0x04,
        }
    }
}

/// The bit pattern a mask has before the environment has been read. All ones
/// is safe to reserve: [`ALL`] is nineteen channels, so the top bits are not
/// reachable by any real mask.
const UNSEEDED: u32 = u32::MAX;

static MASK: AtomicU32 = AtomicU32::new(UNSEEDED);

/// Read the environment once and record what it asked for.
///
/// Cold because it runs at most once per process: the mask it stores is never
/// [`UNSEEDED`] again, even when the environment named nothing.
#[cold]
fn seed() -> u32 {
    let mut mask = 0;
    for channel in ALL {
        if std::env::var(channel.name()).is_ok() {
            mask |= channel.bit();
        }
    }
    MASK.store(mask, Ordering::Relaxed);
    mask
}

/// The channels currently on.
#[inline]
pub fn mask() -> u32 {
    let mask = MASK.load(Ordering::Relaxed);
    if mask == UNSEEDED {
        return seed();
    }
    mask
}

/// Turn exactly the channels in `mask` on, and every other channel off.
///
/// This is what the browser has instead of an environment: the page offers
/// the [`ALL`] list and hands back what was ticked.
pub fn set_mask(new: u32) {
    // Never store the sentinel: a host asking for every channel at once would
    // otherwise leave the mask looking unread and get the environment's
    // answer on the next call.
    MASK.store(new & !UNSEEDED_GUARD, Ordering::Relaxed);
}

/// The bits no channel uses, cleared out of any mask a host hands in so that
/// [`UNSEEDED`] stays unreachable.
const UNSEEDED_GUARD: u32 = !((1 << ALL.len()) - 1);

/// Whether `what` is on.
#[inline]
pub fn enabled(what: Trace) -> bool {
    mask() & what.bit() != 0
}

/// Text traced by code that has no [`crate::cpu::Cpu`] in reach — the
/// rasterizer, the shader translator, the texture decoder — waiting to be
/// folded into the trace buffer the host drains.
///
/// The alternative was threading a sink through every free function in
/// `gpu/`, which is a signature change to code whose whole job is to be
/// called from a hot loop.
static PENDING: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// `PENDING`'s length, so the common case — nothing traced, because nothing is
/// on — costs a relaxed load rather than a lock.
static PENDING_LEN: AtomicUsize = AtomicUsize::new(0);

/// How much untaken trace text the sink holds before it starts dropping.
///
/// It has to have a cap: a native run never takes from it at all, since stderr
/// is where its traces go. Dropping is from the front, for the same reason
/// the trace buffer drops from the front — what happened most recently is what
/// is being asked about.
const PENDING_CAP: usize = 256 * 1024;

/// Emit one line, to stderr and to the sink the host drains.
///
/// Callers gate on [`enabled`] first: this does no checking of its own, so
/// that a channel that is off costs one load and a branch at the call site.
pub fn emit(line: &str) {
    // Natively this is the channel — a CLI run traces to stderr exactly as it
    // did when these were environment switches.
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{line}");
    let Ok(mut pending) = PENDING.lock() else {
        return;
    };
    pending.push(Level::Debug.marker());
    pending.extend_from_slice(line.as_bytes());
    pending.push(b'\n');
    if pending.len() > PENDING_CAP {
        let drop_to = pending.len() - PENDING_CAP;
        pending.drain(..drop_to);
    }
    PENDING_LEN.store(pending.len(), Ordering::Relaxed);
}

/// Take everything the sink holds, leaving it empty.
pub fn take_pending() -> Vec<u8> {
    if PENDING_LEN.load(Ordering::Relaxed) == 0 {
        return Vec::new();
    }
    let Ok(mut pending) = PENDING.lock() else {
        return Vec::new();
    };
    PENDING_LEN.store(0, Ordering::Relaxed);
    std::mem::take(&mut pending)
}

/// Write one already-decided-on line to both channels.
///
/// Spelled like `eprintln!` because it replaces one at roughly a hundred
/// sites that were already gated by a switch of their own — `ctx.trace`, a
/// tally's `enabled`, an inverted early return — and rewriting each of those
/// guards into a [`trace!`] would have changed what they mean.
#[macro_export]
macro_rules! traceln {
    ($($arg:tt)*) => {
        $crate::trace::emit(&format!($($arg)*))
    };
}

/// Trace one formatted line on a channel, evaluating the format arguments
/// only when the channel is on.
///
/// The guard is the point: `trace!(Trace::Ipc, "{}", expensive())` costs a
/// load and a branch when `TRACE_IPC` is off, which is what lets these sit in
/// per-syscall and per-draw paths.
#[macro_export]
macro_rules! trace {
    ($channel:expr, $($arg:tt)*) => {{
        if $crate::trace::enabled($channel) {
            $crate::trace::emit(&format!($($arg)*));
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_channel_has_its_own_bit() {
        let mut seen = 0u32;
        for channel in ALL {
            assert_eq!(seen & channel.bit(), 0, "{} repeats a bit", channel.name());
            seen |= channel.bit();
        }
        assert_ne!(seen, UNSEEDED, "the sentinel has to stay unreachable");
    }

    #[test]
    fn names_round_trip() {
        for channel in ALL {
            assert_eq!(Trace::from_name(channel.name()), Some(channel));
        }
        assert_eq!(Trace::from_name("TRACE_NOTHING"), None);
    }

    #[test]
    fn a_full_mask_does_not_read_as_unseeded() {
        // Every bit set is exactly what a page with all the boxes ticked
        // sends, and storing it verbatim would make the next read seed itself
        // from the environment instead.
        //
        // Checked against the constants rather than by setting the mask: the
        // mask is the whole process's, and a test that turned every channel on
        // would trace whatever else was running beside it.
        assert_ne!(u32::MAX & !UNSEEDED_GUARD, UNSEEDED);
        for channel in ALL {
            assert_eq!(channel.bit() & UNSEEDED_GUARD, 0, "{}", channel.name());
        }
    }

    #[test]
    fn levels_are_distinct_and_not_text() {
        let markers = [
            Level::Error.marker(),
            Level::Warn.marker(),
            Level::Info.marker(),
            Level::Debug.marker(),
        ];
        for (i, a) in markers.iter().enumerate() {
            assert!(*a < b' ', "a marker must not be a printable character");
            for b in &markers[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
