//! The draw/clear backend interface. [`Software`] is the reference
//! implementation; a GPU backend implements all of it, clears included, since
//! a render target lives in guest memory and both must agree on it.

use crate::gpu::engine::threed::Engine3D;
use crate::gpu::exec::ExecCtx;
use crate::Result;

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

    /// Make guest memory agree with whatever the backend is holding. Draws
    /// must not block on readback; this is where waiting is allowed.
    fn flush(&mut self, ctx: &mut ExecCtx) -> Result<Flush> {
        let _ = ctx;
        Ok(Flush::Done)
    }

    /// What this backend has been doing, as a JSON object.
    fn report_json(&self) -> String {
        "{}".to_string()
    }

    /// Whether the device was lost and the backend wants replacing.
    fn lost(&self) -> bool {
        false
    }
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
