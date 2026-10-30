//! The draw/clear backend interface. [`Software`] is the reference
//! implementation; a GPU backend implements all of it, clears included, since
//! a render target lives in guest memory and both must agree on it.

use crate::gpu::engine::threed::Engine3D;
use crate::gpu::exec::ExecCtx;
use crate::gpu::surface::Surface;
use crate::Result;
use std::ops::Range;

/// A 2D engine source copy, as its registers describe it.
#[derive(Debug, Clone, Copy)]
pub struct Blit {
    pub src: Surface,
    pub dst: Surface,
    /// In destination texels: x, y, width and height.
    pub dst_rect: (u32, u32, u32, u32),
    /// The source position the first destination texel reads.
    pub src_origin: (f64, f64),
    /// Source texels per destination texel.
    pub step: (f64, f64),
    /// Four taps blended, rather than one texel read.
    pub filtered: bool,
}

/// What [`Renderer::blit`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlitRoute {
    /// The backend ran it.
    Done,
    /// It touches nothing the backend holds, so the engine can run it on guest memory as it is.
    GuestMemory,
    /// The engine has to flush the backend, then run it on guest memory.
    Flush,
}

/// A backend that turns the 3D engine's draws and clears into pixels.
pub trait Renderer: std::fmt::Debug {
    /// Draw [`Engine3D::last_draw`]. An error skips the draw and leaves the
    /// render target alone.
    fn draw(&mut self, engine: &Engine3D, ctx: &mut ExecCtx) -> Result<()>;

    /// Clear the enabled `channels` of colour target `target`, layer `layer`,
    /// to the engine's clear colour and within its clear rectangle.
    fn clear_color(
        &mut self,
        engine: &Engine3D,
        ctx: &mut ExecCtx,
        target: u32,
        layer: u32,
        channels: [bool; 4],
    ) -> Result<()>;

    /// Clear depth and/or stencil to the engine's clear values.
    fn clear_depth_stencil(
        &mut self,
        engine: &Engine3D,
        ctx: &mut ExecCtx,
        depth: bool,
        stencil: bool,
    ) -> Result<()>;

    /// Run a 2D blit where its surfaces are, or say what the engine must do first.
    fn blit(&mut self, blit: &Blit, ctx: &mut ExecCtx) -> Result<BlitRoute> {
        let _ = (blit, ctx);
        Ok(BlitRoute::Flush)
    }

    /// Whether guest memory is current for these GPU ranges, and stays so if an engine
    /// writes them, so that the engine need not flush the backend first.
    fn guest_memory_is_current(&mut self, ranges: &[Range<u64>], ctx: &mut ExecCtx) -> bool {
        let _ = (ranges, ctx);
        false
    }

    /// Make guest memory agree with whatever the backend is holding. Draws
    /// must not block on readback; this is where waiting is allowed.
    fn flush(&mut self, ctx: &mut ExecCtx) -> Result<Flush> {
        let _ = ctx;
        Ok(Flush::Done)
    }

    /// Start getting a presented frame back without waiting for the ones already on their
    /// way; see [`Submit`]. `scanout` is the CPU address the display reads, when known; a
    /// backend may keep every other surface for later.
    fn submit_frame(&mut self, ctx: &mut ExecCtx, scanout: Option<u32>) -> Result<Submit> {
        let _ = (ctx, scanout);
        Ok(Submit::Done)
    }

    /// The newest [`Submit::Queued`] generation whose surfaces are all in guest memory.
    fn frames_landed(&mut self, ctx: &mut ExecCtx) -> Result<u64> {
        let _ = ctx;
        Ok(u64::MAX)
    }

    /// What this backend has been doing, as the fields of a JSON object.
    fn report(&self, out: &mut crate::json::Object) {
        let _ = out;
    }

    /// Forget what is held for an address space that was closed.
    fn address_space_closed(&mut self, as_id: u32) {
        let _ = as_id;
    }

    /// Whether the device was lost and the backend wants replacing.
    fn lost(&self) -> bool {
        false
    }
}

/// Where a presented frame stands after [`Renderer::submit_frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Submit {
    /// Guest memory has the frame.
    Done,
    /// The frame lands with this generation; see [`Renderer::frames_landed`].
    Queued(u64),
    /// Readbacks already fill the queue, and this frame was not asked for.
    Full,
}

/// Whether a [`Renderer::flush`] finished, or wants asking again after the
/// host's event loop has run. Guest memory must not be read while `Pending`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flush {
    /// Guest memory agrees with the backend.
    Done,
    /// A readback is in flight. Ask again once the host has run.
    Pending,
}

/// The CPU rasterizer: [`crate::gpu::raster`] for draws, and the 3D
/// engine's own clear paths. Stateless.
#[derive(Debug, Default, Clone, Copy)]
pub struct Software;

impl Renderer for Software {
    fn draw(&mut self, engine: &Engine3D, ctx: &mut ExecCtx) -> Result<()> {
        crate::gpu::raster::draw(engine, ctx)
    }

    fn clear_color(
        &mut self,
        engine: &Engine3D,
        ctx: &mut ExecCtx,
        target: u32,
        layer: u32,
        channels: [bool; 4],
    ) -> Result<()> {
        engine.clear_color(target, layer, channels, ctx)
    }

    fn clear_depth_stencil(
        &mut self,
        engine: &Engine3D,
        ctx: &mut ExecCtx,
        depth: bool,
        stencil: bool,
    ) -> Result<()> {
        engine.clear_depth_stencil(depth, stencil, ctx)
    }
}
