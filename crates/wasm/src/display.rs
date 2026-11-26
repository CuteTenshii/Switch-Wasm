//! The presented framebuffer.

use wasm_bindgen::prelude::wasm_bindgen;

use crate::{session, FB_HEIGHT, FB_WIDTH};

/// Framebuffer size as `[width, height]`: the presented resolution, or the demo
/// framebuffer's before the first present.
#[wasm_bindgen]
pub fn switch_fb_size() -> Vec<u32> {
    let s = session();
    if s.cpu.nv.gpu.frames > 0 {
        let fb = &s.cpu.nv.gpu.framebuffer;
        vec![fb.width, fb.height]
    } else {
        vec![FB_WIDTH, FB_HEIGHT]
    }
}

/// Number of frames the guest has presented.
#[wasm_bindgen]
pub fn switch_frame_count() -> u32 {
    session().cpu.nv.gpu.frames as u32
}
