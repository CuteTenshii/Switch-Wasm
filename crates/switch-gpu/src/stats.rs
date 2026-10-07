//! Backend counters and device error tracking.

use switch_core::gpu::upload::Uploads;

pub(crate) const MAX_DEVICE_ERRORS: usize = 16;

/// Everything the device has rejected since it was opened.
#[derive(Debug, Default)]
pub(crate) struct DeviceErrors {
    /// The oldest rejection not yet taken by [`Gpu::device_error`].
    pub(crate) fresh: Option<String>,
    /// Each distinct message once, in the order first seen.
    pub(crate) distinct: Vec<String>,
    /// Every rejection, including repeats.
    pub(crate) count: u64,
}

impl DeviceErrors {
    pub(crate) fn record(&mut self, message: String) {
        self.count += 1;
        if !self.distinct.contains(&message) {
            eprintln!("[gpu] the device rejected something: {message}");
            if self.distinct.len() < MAX_DEVICE_ERRORS {
                self.distinct.push(message.clone());
            }
        }
        self.fresh.get_or_insert(message);
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct UploadBytes {
    pub(crate) vertex: u64,
    pub(crate) index: u64,
    pub(crate) constants: u64,
    pub(crate) textures: u64,
}

impl UploadBytes {
    /// Textures are counted by [`UploadBytes::add_texture`], only when actually read.
    pub(crate) fn add_but_textures(&mut self, uploads: &Uploads) {
        self.vertex += uploads
            .vertex
            .iter()
            .map(|v| v.bytes.len() as u64)
            .sum::<u64>();
        self.index += uploads.index.as_ref().map_or(0, |i| i.bytes.len() as u64);
        self.constants += uploads
            .constants
            .iter()
            .map(|c| c.bytes.len() as u64)
            .sum::<u64>();
    }

    pub(crate) fn add_texture(&mut self, bytes: usize) {
        self.textures += bytes as u64;
    }
}

/// Where a draw's time goes, in microseconds, over a whole run.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Times {
    /// Decoding and translating both shaders.
    pub(crate) translate: u128,
    pub(crate) upload: u128,
    /// Generating the WGSL and creating the modules.
    pub(crate) modules: u128,
    pub(crate) pipeline: u128,
    pub(crate) encode: u128,
    /// Writing surfaces back to guest memory.
    pub(crate) flush: u128,
    /// Encoding and submitting copies of held surfaces.
    pub(crate) flush_ask: u128,
    /// Waiting for the maps; ~0 in a browser, where the wait happens at the slice boundary.
    pub(crate) flush_wait: u128,
    /// Writing the mapped data into guest memory.
    pub(crate) flush_land: u128,
}

pub(crate) fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
