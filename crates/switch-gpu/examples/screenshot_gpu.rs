//! `screenshot_title` with the GPU backend installed; `cmp` the two PPMs to
//! check the backend against the software reference:
//! `screenshot_gpu <container> <prod.keys> [title.keys] <out.ppm> [frame]`.
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
    // `DOCKED=1` docks before boot; `DOCK_AT=<frame>` docks after that frame.
    if std::env::var("DOCKED").is_ok() {
        cpu.set_operation_mode(switch_core::cpu::OperationMode::Docked);
    }
    let dock_at = common::env_u64("DOCK_AT", u64::MAX);
    title.mount_romfs(&mut cpu);
    common::load_fallback_font(&mut cpu);
    common::register_firmware(&mut cpu, &title.keys);
    title.boot(&mut cpu);

    // Installed on the session so it precedes and covers every channel.
    if let Some(gpu) = gpu.take() {
        println!("[gpu] installed");
        cpu.nv.gpu.set_renderer(Box::new(gpu));
    }
    let mut debug = common::Debug::from_env();
    debug.arm(&mut cpu);
    let pace = if debug.stepwise() {
        Pace::Instructions
    } else {
        Pace::Blocks
    };
    let run = common::drive(
        &mut cpu,
        pace,
        common::env_u64("STEPS", u64::MAX),
        |cpu, done| {
            debug.tick(cpu, done);
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
    debug.report();
    debug.stop_state(&cpu);
    // The browser's Rendering panel: device draws and fallbacks.
    println!("rendering: {}", cpu.nv.gpu.renderer_report());
    if cpu.nv.gpu.framebuffer.is_empty() {
        println!("no frame");
        return;
    }
    common::write_ppm(&out, &cpu.nv.gpu.framebuffer);
}
