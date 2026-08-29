//! What turns a draw stream into pixels.
//!
//! There is one implementation today — [`Software`], the rasterizer in
//! [`crate::gpu::raster`] — and the trait exists so that a second one can be
//! added beside it rather than in place of it. That matters for two reasons
//! beyond tidiness.
//!
//! The first is that a GPU backend is the only way this emulator gets fast:
//! about 85% of a Home Menu frame is rasterization, and no amount of
//! sharpening the software path changes that. The second is that a GPU
//! backend needs something to be checked *against*, and a bit-exact software
//! rasterizer with a frame that is byte-comparable to the previous build is
//! exactly that — the same way `bcn_difftest` checks the block codecs against
//! a second implementation rather than against a rendered frame.
//!
//! # What a second backend has to answer
//!
//! Both methods are handed the [`Engine3D`] the draw was issued on, so a
//! backend reads whatever state it needs through that engine's accessors —
//! which are already typed, and are the list of what a pipeline has to be
//! built from: [`Engine3D::render_target`], [`Engine3D::depth_target`],
//! [`Engine3D::viewport_transform`], [`Engine3D::apply_scissor`],
//! [`Engine3D::depth_state`], [`Engine3D::blend_target`],
//! [`Engine3D::cull_state`], [`Engine3D::sample_grid`],
//! [`Engine3D::vertex_attrib`], [`Engine3D::vertex_array`],
//! [`Engine3D::bound_constbuf`], [`Engine3D::program`],
//! [`Engine3D::tex_header_pool`] and [`Engine3D::instance_id`].
//!
//! The hard part is not that list. It is that a render target lives in *guest
//! memory* — `present` deswizzles block-linear pixels straight out of it — so
//! a backend that keeps its surfaces GPU-side owns the question of when to
//! write them back. Nothing here answers that, because nothing here needs to
//! yet.
//!
//! # Clears are part of it
//!
//! A clear is not a separate concern that can stay behind: on a GPU it is a
//! render pass's load operation, and a backend whose draws live in a GPU
//! surface while its clears wrote guest memory would disagree with itself
//! about what is on screen. So both are behind the same trait, and a backend
//! implements all of it or none of it.

use crate::gpu::engine::threed::Engine3D;
use crate::gpu::exec::ExecCtx;
use crate::Result;

/// A backend that turns the 3D engine's draws and clears into pixels.
///
/// `Debug` because [`Engine3D`] derives it and holds one of these.
pub trait Renderer: std::fmt::Debug {
    /// Draw [`Engine3D::last_draw`].
    ///
    /// An error means the draw was not carried out; the caller counts it in
    /// `GpuStats::draws_skipped` and leaves the render target alone, which is
    /// how a shader using an instruction the interpreter does not decode
    /// costs one draw rather than the frame.
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

    /// Make guest memory agree with whatever the backend is holding.
    ///
    /// A backend that keeps its surfaces on a device has to be told when
    /// something outside it is about to look at them, because a render
    /// target is guest memory and the guest is entitled to read it. `present`
    /// is the one that always matters: the display deswizzles the surface
    /// straight out of memory, so a frame drawn on a device and never handed
    /// back is a frame the host never sees.
    ///
    /// It exists so that [`Renderer::draw`] does not have to hand anything
    /// back *itself*. A draw that read its target back would be a draw that
    /// blocked, and in a browser a blocking readback is not slow — it is a
    /// deadlock, since the promise it waits on can only resolve once the
    /// event loop runs. Draws encode and return; this is where the waiting
    /// is allowed to be.
    ///
    /// [`Software`] has nothing to hand back: it writes guest memory as it
    /// goes, so the default is what it wants.
    fn flush(&mut self, ctx: &mut ExecCtx) -> Result<Flush> {
        let _ = ctx;
        Ok(Flush::Done)
    }

    /// What this backend has been doing, as a JSON object.
    ///
    /// A backend that can decline a draw is the only thing that knows why it
    /// declined, and in a browser that answer reaches nobody: `eprintln!`
    /// goes nowhere and an env var cannot be set. So it is asked for instead,
    /// and the page shows it beside the translator's own counters.
    ///
    /// [`Software`] never declines anything and has nothing to report.
    fn report_json(&self) -> String {
        "{}".to_string()
    }

    /// Whether this backend has stopped being able to render and wants
    /// replacing.
    ///
    /// A device is not a thing a browser promises to keep: a driver reset, a
    /// GPU process restart or memory pressure takes it away, and the backend
    /// hands every frame to the rasterizer from then on. That is the right
    /// answer for the frame it happens in and the wrong one for the rest of
    /// the session — 30x wrong, measured — because a lost device can simply
    /// be asked for again. Cheap enough to ask once a slice, which is what
    /// makes a fresh one possible without a reload.
    fn lost(&self) -> bool {
        false
    }
}

/// Whether a [`Renderer::flush`] finished, or wants asking again.
///
/// A backend that keeps its surfaces on a device gets them back by mapping a
/// buffer, and a map completes when the host's event loop runs — which it
/// cannot do underneath a call that is waiting for it. So a flush is allowed
/// to say "not yet" instead of blocking, and the caller has to let the host
/// make progress before reading the surface.
///
/// The caller may **not** simply carry on and read guest memory anyway. That
/// is the reading that produced a black frame every time this was tried by
/// landing a readback one flush late: a double-buffered title presents the
/// surface whose readback was just asked for, and the copy in guest memory is
/// whatever was there before the GPU drew.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flush {
    /// Guest memory agrees with the backend.
    Done,
    /// A readback is in flight. Ask again once the host has run.
    Pending,
}

/// The rasterizer that runs on the CPU: [`crate::gpu::raster`] for draws, and
/// the 3D engine's own register-decoding clear paths.
///
/// Stateless. Everything a draw needs is either in the engine's registers or
/// in guest memory, and what the rasterizer caches — decoded programs, parsed
/// texture descriptors, decoded compressed blocks — it caches for the length
/// of one draw and no longer.
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
