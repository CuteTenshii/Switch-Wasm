//! The presented framebuffer.

use crate::{session, FB_HEIGHT, FB_WIDTH};

/// Framebuffer geometry: the presented resolution, or the demo framebuffer's before
/// the first present.
#[no_mangle]
pub extern "C" fn switch_fb_width() -> u32 {
    let s = session();
    if s.cpu.nv.gpu.frames > 0 {
        s.cpu.nv.gpu.framebuffer.width
    } else {
        FB_WIDTH
    }
}

#[no_mangle]
pub extern "C" fn switch_fb_height() -> u32 {
    let s = session();
    if s.cpu.nv.gpu.frames > 0 {
        s.cpu.nv.gpu.framebuffer.height
    } else {
        FB_HEIGHT
    }
}

/// Number of frames the guest has presented.
#[no_mangle]
pub extern "C" fn switch_frame_count() -> u32 {
    session().cpu.nv.gpu.frames as u32
}
