//! How long scan-out takes: `present_bench [frames] [width] [height]`.
//!
//! [`switch_core::gpu::Gpu::present`] is the one piece of per-frame work that
//! runs whichever renderer produced the surface — the software rasterizer and
//! the wgpu backend both hand it a block-linear image in guest memory, and it
//! walks that image a pixel at a time into the `Vec<u32>` the canvas wants. So
//! it is the floor under the frame rate, and enabling a GPU cannot move it.
//!
//! Just Dance 2019 is the case that makes the point: it issues **no draws at
//! all**, and still costs a full 1280x720 scan-out every frame. Measuring that
//! against a real title means a three-minute boot per attempt; this builds the
//! same surface directly and reports ms per frame, so the loop can be worked
//! on in seconds instead.
//!
//! The pixels are a gradient rather than a constant: a constant surface is
//! exactly the input that would let a wrong fast path look right.
mod common;

use std::time::Instant;
use switch_core::gpu::surface::Layout;
use switch_core::gpu::{DisplayBuffer, Gpu, NV_LAYOUT_BLOCK_LINEAR};
use switch_core::mem::Memory;

/// Where the surface is mapped. Clear of every region a process is given, as
/// the demo framebuffer is.
const BASE: u32 = 0xF400_0000;
/// `NvColorFormat` for A8B8G8R8, which is what a title's swapchain uses and
/// what `present` decodes to `RGBA8Unorm` (surface format `0xD5` — the one the
/// `[gpu]` traces show Just Dance presenting).
const COLOR_FORMAT: u64 = 0x01_0053_2120;
/// Gobs per block in the vertical direction, as `block_height_log2`.
const BLOCK_HEIGHT_LOG2: u32 = 4;

fn main() {
    let frames = common::opt_num(1).unwrap_or(60) as u32;
    let width = common::opt_num(2).unwrap_or(1280) as u32;
    let height = common::opt_num(3).unwrap_or(720) as u32;

    let mut mem = Memory::new();
    let mut gpu = Gpu::new();

    // A block-linear surface the size the display gets. `Layout::offset` is
    // what decides where a texel lives, so filling through it puts every byte
    // exactly where `present` will look for it.
    let layout = Layout::BlockLinear {
        block_height_gobs: 1 << BLOCK_HEIGHT_LOG2,
    };
    let width_bytes = width * 4;
    let size = layout.layer_stride(width_bytes, height);
    mem.map_zero(BASE, size as usize).expect("map the surface");
    for y in 0..height {
        for x in 0..width {
            let texel =
                (x * 4 & 0xFF) | ((y * 4 & 0xFF) << 8) | (((x ^ y) & 0xFF) << 16) | 0xFF00_0000;
            let at = BASE.wrapping_add(layout.offset(x * 4, y, width_bytes));
            mem.write_u32(at, texel).expect("fill the surface");
        }
    }

    let handle = gpu.nvmap.create(size);
    gpu.nvmap
        .alloc(handle, 0, 0, 0x1000, 0, BASE)
        .expect("alloc the surface");
    let id = gpu.nvmap.get(handle).expect("the handle we just made").id;
    let buffer = DisplayBuffer {
        nvmap_id: id,
        offset: 0,
        width,
        height,
        pitch: width_bytes,
        layout: NV_LAYOUT_BLOCK_LINEAR,
        block_height_log2: BLOCK_HEIGHT_LOG2,
        color_format: COLOR_FORMAT,
        transform: 0,
    };

    // One outside the timing, so a cold page table is not counted as scan-out.
    gpu.present(&mem, &buffer).expect("present");
    let checksum: u64 = gpu.framebuffer.pixels.iter().map(|&p| u64::from(p)).sum();

    let start = Instant::now();
    for _ in 0..frames {
        gpu.present(&mem, &buffer).expect("present");
    }
    let elapsed = start.elapsed();

    let per_frame = elapsed.as_secs_f64() / f64::from(frames);
    println!(
        "{width}x{height} block-linear: {:.2} ms/frame over {frames} frames ({:.1} fps ceiling)",
        per_frame * 1000.0,
        1.0 / per_frame
    );
    println!(
        "{:.1} ns/pixel, checksum {checksum:#x}",
        per_frame * 1e9 / f64::from(width * height)
    );
}
