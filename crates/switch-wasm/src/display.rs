//! The presented framebuffer.

use crate::{session, FB_BASE, FB_HEIGHT, FB_WIDTH};

/// Framebuffer geometry: the presented resolution, or the demo framebuffer's before
/// the first present.
#[no_mangle]
pub extern "C" fn switch_fb_width(handle: u32) -> u32 {
    let s = session(handle);
    if s.cpu.nv.gpu.frames > 0 {
        s.cpu.nv.gpu.framebuffer.width
    } else {
        FB_WIDTH
    }
}

#[no_mangle]
pub extern "C" fn switch_fb_height(handle: u32) -> u32 {
    let s = session(handle);
    if s.cpu.nv.gpu.frames > 0 {
        s.cpu.nv.gpu.framebuffer.height
    } else {
        FB_HEIGHT
    }
}

/// Number of frames the guest has presented.
#[no_mangle]
pub extern "C" fn switch_frame_count(handle: u32) -> u32 {
    session(handle).cpu.nv.gpu.frames as u32
}

/// Copy the current screen (RGBA8888) into `buf`. Returns bytes copied. Alpha is
/// forced opaque, as scan-out ignores it.
#[no_mangle]
pub extern "C" fn switch_fb_snapshot(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    if s.cpu.nv.gpu.frames > 0 {
        let fb = &s.cpu.nv.gpu.framebuffer;
        let n = (fb.pixels.len() * 4).min(maxlen as usize);
        let out = unsafe { std::slice::from_raw_parts_mut(buf, n) };
        for (chunk, pixel) in out.as_chunks_mut::<4>().0.iter_mut().zip(fb.pixels.iter()) {
            *chunk = (pixel | 0xFF00_0000).to_le_bytes();
        }
        return n as u32;
    }
    let n = ((FB_WIDTH * FB_HEIGHT * 4) as usize).min(maxlen as usize);
    let out = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    match s.cpu.mem.read_into(FB_BASE, out) {
        Ok(()) => n as u32,
        Err(_) => 0,
    }
}
