//! What the backend counts about itself, and what it does with a refusal.
//!
//! None of this is needed to draw a frame. It is here because "where did the
//! frame go" and "why did the device refuse that" are answered by numbers
//! taken while the frame was drawn, and a backend that cannot answer them is
//! one that gets optimised by guesswork.

use switch_core::gpu::upload::Uploads;

/// How many distinct device errors are worth keeping. A rejected draw repeats
/// its rejection every frame, so the list stops growing almost immediately and
/// the count carries the rest.
pub(crate) const MAX_DEVICE_ERRORS: usize = 16;

/// Everything the device has rejected since it was opened.
///
/// Keeping only the first — which is what this was — reported nothing at all
/// once the pipelines were built: the only production reader runs before a
/// pipeline is created, and a title that builds its four in the first frames
/// never creates another. Every rejection after that sat unread.
#[derive(Debug, Default)]
pub(crate) struct DeviceErrors {
    /// The oldest rejection nothing has asked about yet, taken by
    /// [`Gpu::device_error`]. Oldest rather than newest because the first
    /// rejection is the one with a cause; the rest are usually its echo.
    pub(crate) fresh: Option<String>,
    /// Each distinct message once, in the order first seen — the same shape
    /// as `reasons`, and for the same reason.
    pub(crate) distinct: Vec<String>,
    /// Every rejection, including the repeats and anything past
    /// [`MAX_DEVICE_ERRORS`].
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

/// What `Uploads::of` lifted out of guest memory over a whole run.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct UploadBytes {
    pub(crate) vertex: u64,
    pub(crate) index: u64,
    pub(crate) constants: u64,
    pub(crate) textures: u64,
}

impl UploadBytes {
    /// What one draw lifted out of guest memory, by what it was.
    ///
    /// Bytes rather than milliseconds: a byte read out of guest memory is the
    /// same byte under V8, and counting them is what showed textures to be
    /// 96.5% of this, and so the only one of the four worth caching.
    ///
    /// Textures are counted by [`UploadBytes::add_texture`] instead, and only
    /// when one was really read. A draw served from the cache never touched
    /// guest memory for it, and counting it here would report reads that did
    /// not happen — which is exactly what this said before the cache existed
    /// to make the two differ.
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
    /// Decoding both shaders out of guest memory and translating them.
    pub(crate) translate: u128,
    /// Reading the vertices, indices, constants and textures.
    pub(crate) upload: u128,
    /// Generating the WGSL and handing it to the device.
    pub(crate) modules: u128,
    /// Building the pipeline and its bind groups.
    pub(crate) pipeline: u128,
    /// Encoding and submitting the pass.
    pub(crate) encode: u128,
    /// Handing surfaces back to guest memory.
    pub(crate) flush: u128,
    /// The three phases `flush` is made of, which answer different questions
    /// and have different fixes. They sum to roughly `flush`; the remainder is
    /// the early-outs, which is worth seeing as a gap rather than hiding in
    /// one of them.
    ///
    /// Encoding the copies out of every held surface and submitting them.
    /// Device work asked for, and the cost of asking.
    pub(crate) flush_ask: u128,
    /// Waiting for the maps. Natively this blocks and is the copy actually
    /// happening; in a browser `poll` cannot do anything at all, so this being
    /// ~0 there is correct rather than suspicious — the wait moved to the
    /// slice boundary, where `Flush::Pending` puts it.
    pub(crate) flush_wait: u128,
    /// Reading the mapping and writing it through the page table into guest
    /// memory. A whole surface a frame, and the phase a device-to-canvas
    /// present would delete outright.
    pub(crate) flush_land: u128,
}

/// One JSON string literal. A fallback reason is an error message and is free
/// to contain a quote or a backslash; the page parses this with `JSON.parse`,
/// which is entitled to reject the whole object over one of them.
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
