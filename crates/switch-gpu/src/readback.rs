//! What a surface being read back off the device is, while it is in flight.
//!
//! A draw renders into a device texture; guest memory is where the surface
//! actually lives. Getting it back is three steps that cannot happen in one
//! call on the web — copy into a staging buffer, map it, read it — and these
//! are what a backend holds between them.
//!
//! The map state is an atomic rather than a wait because in a browser the
//! callback runs from the event loop: nothing inside a time slice can make it
//! happen, so the slice ends and the next one reads the flag.

use switch_core::gpu::surface::SampleGrid;
use switch_core::gpu::upload::Target;

use crate::Shape;

/// A readback that has been asked for and not yet copied out.
///
/// Kept as a type because asking and collecting are the two halves a browser
/// has to put an `await` between — see [`Gpu::write_back`], which today does
/// both with a wait in the middle.
#[derive(Debug)]
pub(crate) struct Pending {
    pub(crate) staging: wgpu::Buffer,
    pub(crate) target: Target,
    /// Bytes in one row of what the device holds. The surface's own for a
    /// colour target; for a depth one it is the device format's, which is
    /// not the guest's — a `Z24S8` texel is four bytes in memory and four
    /// bytes of `f32` on the device, and a `ZF32_X24S8` texel is eight and
    /// four.
    pub(crate) row_bytes: u32,
    /// That stride rounded up to the 256 bytes `copyTextureToBuffer` wants.
    pub(crate) padded: u32,
    /// What the map callback reported: [`MAP_WAITING`] until it runs, then
    /// [`MAP_READY`] or [`MAP_FAILED`]. Read rather than waited on, because
    /// on the web the callback runs from the event loop and nothing inside a
    /// slice can make that happen.
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

/// A render target held on the device, and where in guest memory it came
/// from.
#[derive(Debug)]
pub(crate) struct Held {
    pub(crate) texture: wgpu::Texture,
    pub(crate) target: Target,
    /// Whether anything has been drawn into it since it was uploaded. A
    /// surface nothing touched need not be written back.
    pub(crate) dirty: bool,
    /// What draws render into, when the expanded surface's own texels are
    /// not what a draw's coverage is measured in — see [`Shape`]. Gathered
    /// from the surface when it is made and scattered back into it before
    /// the surface is read, so that guest memory only ever sees the expanded
    /// form.
    pub(crate) companion: Option<Companion>,
}

/// A surface's stand-in, and the grid it stands in for.
///
/// The grid is kept rather than read again at the end: what puts a companion
/// back is a flush, and by then the register file has moved on to whatever
/// the next frame is doing.
#[derive(Debug)]
pub(crate) struct Companion {
    pub(crate) shape: Shape,
    pub(crate) texture: wgpu::Texture,
    pub(crate) grid: SampleGrid,
}
