//! Diagnostic channels: a runtime mask seeded from the environment, severity
//! levels, and a sink the host drains alongside stderr.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

/// One diagnostic channel, named by the environment variable that seeds it.
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
    /// Draws and clears: render targets, refused draws, rasterizer coverage.
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
    /// Guest filesystem traffic through `fsp-srv`.
    Fs,
    /// Range reads out of the host's files.
    Io,
    /// Copy-engine transfers, inline uploads and 2D-engine blits.
    Copy,
    /// Frames handed to the display: which surface was scanned out, and how.
    Present,
    /// The video engines: nvdec and VIC channels and their methods.
    Video,
}

pub const ALL: [Trace; 24] = [
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
    Trace::Fs,
    Trace::Io,
    Trace::Copy,
    Trace::Present,
    Trace::Video,
];

impl Trace {
    /// The channel's bit in the mask, in [`ALL`] order.
    #[inline]
    pub const fn bit(self) -> u32 {
        1 << self as u32
    }

    /// The environment variable that seeds this channel, also its host-facing name.
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
            Trace::Fs => "TRACE_FS",
            Trace::Io => "TRACE_IO",
            Trace::Copy => "TRACE_COPY",
            Trace::Present => "TRACE_PRESENT",
            Trace::Video => "TRACE_VIDEO",
        }
    }

    pub fn from_name(name: &str) -> Option<Trace> {
        ALL.iter().copied().find(|t| t.name() == name)
    }
}

/// How much a diagnostic matters; the host colours and filters by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// The emulator or the guest has failed at something it was asked to do.
    Error,
    /// Something was answered wrongly or not at all; likely a later failure's cause.
    Warn,
    /// A milestone worth seeing on every run.
    Info,
    /// A mask-gated trace: on only because someone asked for it.
    Debug,
}

impl Level {
    /// Control byte that marks this level in the trace text.
    /// Unmarked lines inherit the previous line's level.
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

/// Mask value before the environment has been read; unreachable by real masks.
const UNSEEDED: u32 = u32::MAX;

static MASK: AtomicU32 = AtomicU32::new(UNSEEDED);

/// Read the environment once and record what it asked for.
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

#[inline]
pub fn mask() -> u32 {
    let mask = MASK.load(Ordering::Relaxed);
    if mask == UNSEEDED {
        return seed();
    }
    mask
}

/// Turn every channel on or off.
pub fn set_all(on: bool) {
    let all = ALL.iter().fold(0, |mask, channel| mask | channel.bit());
    MASK.store(if on { all } else { 0 }, Ordering::Relaxed);
}

#[inline]
pub fn enabled(what: Trace) -> bool {
    mask() & what.bit() != 0
}

/// Trace text from code with no [`crate::cpu::Cpu`] in reach, awaiting the host.
static PENDING: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// `PENDING`'s length, so an empty sink costs a relaxed load, not a lock.
static PENDING_LEN: AtomicUsize = AtomicUsize::new(0);

/// Cap on untaken sink text; the oldest is dropped first.
const PENDING_CAP: usize = 256 * 1024;

/// Emit one line to stderr and the host sink. Callers gate on [`enabled`].
pub fn emit(line: &str) {
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

/// Write one already-gated line to both channels.
#[macro_export]
macro_rules! traceln {
    ($($arg:tt)*) => {
        $crate::trace::emit(&format!($($arg)*))
    };
}

/// Trace one line on a channel, formatting only when it is on.
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
    fn levels_are_the_marker_bytes_the_page_maps() {
        assert_eq!(Level::Error.marker(), 0x01);
        assert_eq!(Level::Warn.marker(), 0x02);
        assert_eq!(Level::Info.marker(), 0x03);
        assert_eq!(Level::Debug.marker(), 0x04);
    }
}
