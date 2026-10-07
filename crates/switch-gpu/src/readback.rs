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
    pub(crate) target: Target,
    /// Bytes per row on the device; for depth this is the device format's, not the guest's.
    pub(crate) row_bytes: u32,
    /// `row_bytes` rounded up to the 256 bytes `copyTextureToBuffer` wants.
    pub(crate) padded: u32,
    /// [`MAP_WAITING`] until the map callback runs, then [`MAP_READY`] or [`MAP_FAILED`].
    pub(crate) state: std::sync::Arc<std::sync::atomic::AtomicU8>,
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

/// A render target held on the device, and its guest memory origin.
#[derive(Debug)]
pub(crate) struct Held {
    pub(crate) texture: wgpu::Texture,
    pub(crate) target: Target,
    /// Drawn into since upload; clean surfaces need no write-back.
    pub(crate) dirty: bool,
    /// What draws render into when coverage is not measured in the surface's texels.
    /// See [`Shape`].
    pub(crate) companion: Option<Companion>,
}

/// A surface's stand-in, and the grid it stands in for (kept until the flush).
#[derive(Debug)]
pub(crate) struct Companion {
    pub(crate) shape: Shape,
    pub(crate) texture: wgpu::Texture,
    pub(crate) grid: SampleGrid,
}
