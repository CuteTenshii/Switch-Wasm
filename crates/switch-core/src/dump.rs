//! Diagnostic artefacts for a native harness to keep. The core never writes
//! files: with no [`Sink`] installed, as in the browser, nothing is produced.

use std::sync::OnceLock;

/// What a harness can ask to be handed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Artefact {
    /// Every texture the rasterizer decodes, as a PPM.
    Texture,
    /// Each shader pair the rasterizer translates, as WGSL (with [`crate::trace::Trace::Wgsl`] on).
    RasterWgsl,
    /// Each WGSL module a device backend compiles.
    DeviceWgsl,
}

pub trait Sink: Send + Sync {
    fn wants(&self, what: Artefact) -> bool;
    /// Keep `bytes` under `name`, a bare file name.
    fn keep(&self, what: Artefact, name: &str, bytes: &[u8]);
}

static SINK: OnceLock<Box<dyn Sink>> = OnceLock::new();

/// Install the process's sink; false if one already was.
pub fn install(sink: Box<dyn Sink>) -> bool {
    SINK.set(sink).is_ok()
}

pub fn wants(what: Artefact) -> bool {
    SINK.get().is_some_and(|sink| sink.wants(what))
}

pub fn keep(what: Artefact, name: &str, bytes: &[u8]) {
    if let Some(sink) = SINK.get() {
        sink.keep(what, name, bytes);
    }
}
