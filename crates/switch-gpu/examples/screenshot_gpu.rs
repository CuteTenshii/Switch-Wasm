//! `screenshot_title` with the GPU backend installed, so the two can be
//! compared frame for frame:
//! `screenshot_gpu <container> <prod.keys> [title.keys] <out.ppm> [frame]`.
//!
//! Either kind of container, decided by its header: the backend this exists to
//! measure is only worth measuring against a title that draws, and those ship
//! in an NSP.
//!
//! The point is the comparison. Run `screenshot_title` and this over the same
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

const USAGE: &str = "screenshot_gpu <container> <prod.keys> [title.keys] <out.ppm> [frame]";

fn main() {
    let args = common::container_args(USAGE);
    let title = args.open();
    let out = args.need(0).to_string();
    let want = args.rest_num(1).unwrap_or(1);

    let mut gpu = match switch_gpu::Gpu::open() {
        Ok(gpu) => Some(gpu),
        Err(why) => {
            eprintln!("no GPU backend ({why}): this is `screenshot_title` with extra steps");
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

    // The backend goes on the session, not on a channel, so it can be
    // installed before the guest has opened one — and it is reached whichever
    // channel the title turns out to draw through.
    if let Some(gpu) = gpu.take() {
        println!("[gpu] installed");
        cpu.nv.gpu.set_renderer(Box::new(gpu));
    }
    let run = common::drive(
        &mut cpu,
        Pace::Blocks,
        common::env_u64("STEPS", u64::MAX),
        |cpu, _| {
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
    // What the browser's Rendering panel shows, for a run that has no browser:
    // how many draws the device took, how many fell back and why.
    println!("rendering: {}", cpu.nv.gpu.renderer_report());
    if cpu.nv.gpu.framebuffer.is_empty() {
        println!("no frame");
        return;
    }
    common::write_ppm(&out, &cpu.nv.gpu.framebuffer);
}
