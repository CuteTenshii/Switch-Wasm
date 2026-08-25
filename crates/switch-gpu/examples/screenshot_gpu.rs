//! `screenshot_nca` with the GPU backend installed, so the two can be
//! compared frame for frame:
//! `screenshot_gpu <path.nca> <prod.keys> <title.keys> <out.ppm> [frame]`.
//!
//! The point is the comparison. Run `screenshot_nca` and this over the same
//! frame and `cmp` the two PPMs — a byte-identical pair is the only evidence
//! that a GPU backend renders what the reference does, and the software
//! rasterizer exists to be that reference.
//!
//! The example scaffolding is `switch-core`'s own, included by path rather
//! than copied. A second copy of "how an example boots a title" is how the
//! two drifted apart the last time.
#[path = "../../switch-core/examples/common/mod.rs"]
mod common;

use common::{Flow, Pace};
use switch_core::cpu::Cpu;

const USAGE: &str = "screenshot_gpu <path.nca> <prod.keys> <title.keys> <out.ppm> [frame]";

fn main() {
    let title = common::Title::open_nca(
        common::arg(1, USAGE),
        common::arg(2, USAGE),
        Some(common::arg(3, USAGE)),
    );
    let out = common::arg(4, USAGE);
    let want = common::opt_num(5).unwrap_or(1);

    let mut gpu = match switch_gpu::Gpu::open() {
        Ok(gpu) => Some(gpu),
        Err(why) => {
            eprintln!("no GPU backend ({why}): this is `screenshot_nca` with extra steps");
            None
        }
    };

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    // `DOCKED=1` docks before boot; `DOCK_AT=<frame>` docks once that frame
    // has been presented, which is what a real dock does — a running title is
    // told by the AM messages `set_operation_mode` queues, and one that was
    // never running had nothing to tell.
    if std::env::var("DOCKED").is_ok() {
        cpu.set_operation_mode(switch_core::cpu::OperationMode::Docked);
    }
    let dock_at = common::env_u64("DOCK_AT", u64::MAX);
    title.mount_romfs(&mut cpu);
    common::load_fallback_font(&mut cpu);
    common::register_firmware(&mut cpu, &title.keys);
    title.boot(&mut cpu);

    // The channel a guest draws through does not exist until the guest opens
    // it, so the backend cannot be installed before boot. This installs it
    // into the first channel that appears, which is the one every title this
    // runs has drawn through.
    let run = common::drive(
        &mut cpu,
        Pace::Blocks,
        common::env_u64("STEPS", u64::MAX),
        |cpu, _| {
            if gpu.is_some() {
                if let Some(channel) = cpu.nv.gpu.channels.values_mut().next() {
                    if let Some(gpu) = gpu.take() {
                        println!("[gpu] installed on channel {}", channel.id);
                        channel.three_d.set_renderer(Box::new(gpu));
                    }
                }
            }
            if cpu.nv.gpu.frames >= dock_at {
                cpu.set_operation_mode(switch_core::cpu::OperationMode::Docked);
            }
            if cpu.nv.gpu.frames >= want {
                Flow::Stop
            } else {
                Flow::Continue
            }
        },
    );
    common::report(&cpu, &run);
    if cpu.nv.gpu.framebuffer.is_empty() {
        println!("no frame");
        return;
    }
    common::write_ppm(&out, &cpu.nv.gpu.framebuffer);
}
