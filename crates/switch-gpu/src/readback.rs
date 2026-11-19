//! In-flight readback state for surfaces copied off the device. Copy, map and read
//! cannot happen in one call on the web, and the map callback runs from the event
//! loop, so its state is an atomic polled by the next slice.

use switch_core::gpu::surface::SampleGrid;
use switch_core::gpu::upload::Target;

use crate::Shape;

/// A readback that has been asked for and not yet copied out.
#[derive(Debug)]
pub(crate) struct Pending {
    pub(crate) staging: wgpu::Buffer,
    pub(crate) what: Readback,
    /// The ask it was made in; see `Space::asked`.
    pub(crate) generation: u64,
    /// [`MAP_WAITING`] until the map callback runs, then [`MAP_READY`] or [`MAP_FAILED`].
    pub(crate) state: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

/// What a readback brings back.
#[derive(Debug)]
pub(crate) enum Readback {
    Surface {
        target: Target,
        /// Bytes per row on the device; for depth this is the device format's, not the guest's.
        row_bytes: u32,
        /// `row_bytes` rounded up to the 256 bytes `copyTextureToBuffer` wants.
        padded: u32,
        /// Held as 32-bit floats for its 16-bit unorm channels, narrowed on landing.
        widened: bool,
        /// CPU pages the guest wrote over, which keep what the guest wrote.
        kept: Vec<u32>,
    },
    /// A buffer shaders wrote through, put back as the words that changed since `original`.
    /// `source` is what was written, kept until it lands for draws that read it meanwhile.
    Global {
        addr: u64,
        original: Vec<u8>,
        source: wgpu::Buffer,
    },
}

impl Pending {
    /// The surface this brings back, if it is one.
    pub(crate) fn target(&self) -> Option<&Target> {
        match &self.what {
            Readback::Surface { target, .. } => Some(target),
            Readback::Global { .. } => None,
        }
    }

    /// Destroy what the readback holds on the device; see [`Gpu::scratch`](crate::Gpu).
    pub(crate) fn destroy(&self) {
        self.staging.destroy();
        if let Readback::Global { source, .. } = &self.what {
            source.destroy();
        }
    }

    /// The guest memory a buffer readback writes, if it is one.
    pub(crate) fn global_span(&self) -> Option<std::ops::Range<u64>> {
        match &self.what {
            Readback::Surface { .. } => None,
            Readback::Global { addr, original, .. } => Some(*addr..*addr + original.len() as u64),
        }
    }
}

pub(crate) const MAP_WAITING: u8 = 0;

pub(crate) const MAP_READY: u8 = 1;

pub(crate) const MAP_FAILED: u8 = 2;

/// One resource a single draw made, held only until that draw is submitted.
#[derive(Debug)]
pub(crate) enum Scratch {
    Buffer(wgpu::Buffer),
    Texture(wgpu::Texture),
}

/// Guest memory shaders write through atomics, held on the device until the next flush.
#[derive(Debug)]
pub(crate) struct HeldGlobal {
    pub(crate) buffer: wgpu::Buffer,
    /// The guest's bytes when it was uploaded; only what differs goes back.
    pub(crate) original: Vec<u8>,
}

/// A render target held on the device, and its guest memory origin.
#[derive(Debug)]
pub(crate) struct Held {
    pub(crate) texture: wgpu::Texture,
    pub(crate) target: Target,
    /// Drawn into since upload; clean surfaces need no write-back.
    pub(crate) dirty: bool,
    /// Bumped by every write, so a copy taken from the texture can tell it is still current.
    pub(crate) writes: u64,
    /// What draws render into when coverage is not measured in the surface's texels.
    /// See [`Shape`].
    pub(crate) companion: Option<Companion>,
    /// The CPU pages being watched for guest writes; empty until they can be.
    /// See [`Gpu::watch_held`](crate::Gpu).
    pub(crate) watched: Vec<u32>,
    /// The same depth surface at its other size. See [`Gpu::switch_crop`](crate::Gpu).
    pub(crate) other: Option<Box<Held>>,
    /// Pages wholly inside the surface that the guest wrote since watching began, by GPU range
    /// and CPU page: newer than the device's texels there, so they are read from guest memory
    /// and a readback skips them.
    pub(crate) overwritten: Vec<(std::ops::Range<u64>, u32)>,
}

/// A surface's stand-in, and the grid it stands in for (kept until the flush).
#[derive(Debug)]
pub(crate) struct Companion {
    pub(crate) shape: Shape,
    pub(crate) texture: wgpu::Texture,
    pub(crate) grid: SampleGrid,
}
