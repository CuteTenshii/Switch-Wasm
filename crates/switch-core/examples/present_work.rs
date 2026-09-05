//! What scan-out costs, in work rather than in milliseconds:
//! `present_work [width] [height]`.
//!
//! [`switch_core::gpu::Gpu::present`] is the one piece of per-frame work that
//! runs whichever renderer produced the surface, the software rasterizer and
//! the wgpu backend both hand it a block-linear image in guest memory, and it
//! walks that image into the `Vec<u32>` the canvas wants. So it is the floor
//! under the frame rate, and enabling a GPU cannot move it. Just Dance 2019 is
//! the case that makes the point: it issues **no draws at all**, and still
//! costs a full 1280x720 scan-out every frame.
//!
//! This used to report ms/frame and an "fps ceiling", measured on the host.
//! Neither is a fact about this project: the browser runs the same loop
//! through its own compiler, over a bounds-checked 32-bit linear memory, and
//! the ratio between the two is not a constant you can divide out. What *is*
//! the same on both is how much work the loop is asked to do, bytes lifted
//! out of guest memory, deswizzle lookups, pixels converted, and every
//! optimisation worth making moves one of those numbers. A change that leaves
//! all three alone did not make scan-out cheaper; it made this host faster at
//! it.
//!
//! The counts are derived here rather than measured inside `present`, because
//! the only place to count a pixel is the per-pixel loop and a counter there
//! is a cost paid 921,600 times a frame to learn something arithmetic already
//! knows. They mirror the loop in `Gpu::present`, a `run_at` per contiguous
//! run, `count` pixels from each, so a change to that loop's shape belongs
//! here too.
//!
//! Every case also presents for real and prints a checksum. The pixels are a
//! gradient rather than a constant: a constant surface is exactly the input
//! that would let a wrong fast path look right.
mod common;

use switch_core::gpu::surface::Layout;
use switch_core::gpu::{Crop, DisplayBuffer, Gpu, NV_LAYOUT_BLOCK_LINEAR, NV_LAYOUT_PITCH};
use switch_core::mem::Memory;

/// Where the surface is mapped. Clear of every region a process is given, as
/// the demo framebuffer is.
const BASE: u32 = 0xF400_0000;
/// `NvColorFormat` for A8B8G8R8, which is what a title's swapchain uses and
/// what `present` decodes to `RGBA8Unorm` (surface format `0xD5`, the one the
/// `[gpu]` traces show Just Dance presenting).
const COLOR_FORMAT: u64 = 0x01_0053_2120;
/// Bytes per pixel of that format, which is what `present` walks the surface in.
const BPP: u32 = 4;
/// Gobs per block in the vertical direction, as `block_height_log2`.
const BLOCK_HEIGHT_LOG2: u32 = 4;

/// The work one `present` of this configuration is asked to do.
struct Work {
    /// Bytes lifted out of guest memory, which is the whole surface however
    /// small the crop is.
    surface_bytes: u32,
    /// Bytes of that surface no output pixel ever reads.
    unsampled_bytes: u32,
    /// Calls to [`Layout::run_at`], one per contiguous run of pixels.
    lookups: u64,
    /// Pixels written to the framebuffer.
    pixels: u64,
    /// Sum of the presented pixels, so a cheaper path that changes the image
    /// cannot pass as an improvement.
    checksum: u64,
}

/// Count what `present` will do to this buffer, then do it.
fn measure(layout: Layout, buffer: &DisplayBuffer, mem: &Memory, gpu: &mut Gpu) -> Work {
    let width_bytes = match layout {
        Layout::Pitch { pitch } => pitch,
        Layout::BlockLinear { .. } => buffer.width * BPP,
    };
    let surface_bytes = layout.layer_stride(width_bytes, buffer.height);
    let (crop_x, crop_y, out_width, out_height) = buffer.crop.window(buffer.width, buffer.height);

    let mut lookups = 0u64;
    let mut sampled = vec![false; surface_bytes as usize];
    for row in 0..out_height {
        let y = crop_y + row;
        let mut x = 0;
        while x < out_width {
            let (offset, run) = layout.run_at((crop_x + x) * BPP, y, width_bytes);
            lookups += 1;
            let count = (run / BPP).clamp(1, out_width - x);
            for byte in offset..(offset + count * BPP).min(surface_bytes) {
                sampled[byte as usize] = true;
            }
            x += count;
        }
    }

    gpu.present(mem, buffer).expect("present");
    Work {
        surface_bytes,
        unsampled_bytes: sampled.iter().filter(|seen| !**seen).count() as u32,
        lookups,
        pixels: u64::from(out_width) * u64::from(out_height),
        checksum: gpu.framebuffer.pixels.iter().map(|&p| u64::from(p)).sum(),
    }
}

/// Build the surface, present it, and report what that took.
///
/// A fresh [`Memory`] and [`Gpu`] per case: `present` keeps its scan-out
/// buffer between frames, and a case that inherited the last one's would be
/// reporting the previous surface's allocation rather than its own.
fn case(name: &str, width: u32, height: u32, pitch_linear: bool, crop: Crop) {
    let mut mem = Memory::new();
    let mut gpu = Gpu::new();
    let layout = if pitch_linear {
        Layout::Pitch { pitch: width * BPP }
    } else {
        Layout::BlockLinear {
            block_height_gobs: 1 << BLOCK_HEIGHT_LOG2,
        }
    };
    let width_bytes = width * BPP;
    let size = layout.layer_stride(width_bytes, height);
    mem.map_zero(BASE, size as usize).expect("map the surface");
    // Filled through `Layout::offset`, which is what decides where a texel
    // lives, so every byte lands exactly where `present` will look for it.
    for y in 0..height {
        for x in 0..width {
            let texel =
                ((x * 4) & 0xFF) | (((y * 4) & 0xFF) << 8) | (((x ^ y) & 0xFF) << 16) | 0xFF00_0000;
            let at = BASE.wrapping_add(layout.offset(x * BPP, y, width_bytes));
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
        layout: if pitch_linear {
            NV_LAYOUT_PITCH
        } else {
            NV_LAYOUT_BLOCK_LINEAR
        },
        block_height_log2: BLOCK_HEIGHT_LOG2,
        color_format: COLOR_FORMAT,
        transform: 0,
        crop,
    };

    let work = measure(layout, &buffer, &mem, &mut gpu);
    let mib = |bytes: u32| f64::from(bytes) / (1024.0 * 1024.0);
    println!("{name}");
    println!(
        "  {:>10} pixels   {:>10} lookups ({:.1} pixels each)",
        work.pixels,
        work.lookups,
        work.pixels as f64 / work.lookups as f64,
    );
    println!(
        "  {:>10.2} MiB read out of guest memory, {:.2} MiB of it never sampled ({:.0}%)",
        mib(work.surface_bytes),
        mib(work.unsampled_bytes),
        f64::from(work.unsampled_bytes) * 100.0 / f64::from(work.surface_bytes),
    );
    println!("  checksum {:#x}", work.checksum);
}

fn main() {
    let width = common::opt_num(1).unwrap_or(1280) as u32;
    let height = common::opt_num(2).unwrap_or(720) as u32;

    case(
        "block-linear, whole surface",
        width,
        height,
        false,
        Crop::ALL,
    );
    case(
        "pitch-linear, whole surface",
        width,
        height,
        true,
        Crop::ALL,
    );
    // The shape a title that allocates 1080p and queues 720p presents every
    // frame: the crop bounds the pixel loop, and nothing bounds the read.
    case(
        "block-linear 1920x1080, cropped to 1280x720",
        1920,
        1080,
        false,
        Crop {
            left: 0,
            top: 0,
            right: 1280,
            bottom: 720,
        },
    );
}
