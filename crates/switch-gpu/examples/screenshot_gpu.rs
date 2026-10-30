//! `screenshot_title` with the GPU backend installed; `cmp` the two PPMs to
//! check the backend against the software reference:
//! `screenshot_gpu <target> [prod.keys] [title.keys] <out.ppm> [frame] [font.ttf]`.
//! `PRESS=<buttons>@<interval>` taps buttons on the way, as in `boot_nsp`.
#[path = "../../switch-core/examples/common/mod.rs"]
mod common;

use common::Flow;

const USAGE: &str = "screenshot_gpu <target> [prod.keys] [title.keys] <out.ppm> [frame] [font.ttf]";

fn main() {
    let args = common::program_args(USAGE);
    let program = args.open_program();
    let out = args.need(0).to_string();
    let want = args.rest_num(1).unwrap_or(1);

    let mut gpu = match switch_gpu::Gpu::open() {
        Ok(mut gpu) => {
            configure(&mut gpu);
            Some(gpu)
        }
        Err(why) => {
            eprintln!("no GPU backend ({why}): this is `screenshot_title` with extra steps");
            None
        }
    };

    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    // Booting docks at once for `DOCKED=1`; `DOCK_AT=<frame>` docks after that frame.
    let dock_at = common::env_u64("DOCK_AT", u64::MAX);
    program.boot(&mut cpu);

    // Installed on the session so it precedes and covers every channel.
    if let Some(gpu) = gpu.take() {
        println!("[gpu] installed");
        cpu.nv.gpu.set_renderer(Box::new(gpu));
    }
    let mut debug = common::Debug::from_env();
    debug.arm(&mut cpu);
    let mut press = common::Press::from_env();
    let pace = debug.pace(false);
    let run = common::drive(
        &mut cpu,
        pace,
        common::env_u64("STEPS", u64::MAX),
        |cpu, done| {
            debug.tick(cpu, done);
            if let Some(press) = press.as_mut() {
                press.tick(cpu, done);
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

/// The device's diagnostic switches, from the environment.
fn configure(gpu: &mut switch_gpu::Gpu) {
    let on = |name: &str| std::env::var(name).is_ok();
    gpu.set_defer_readbacks(on("GPU_DEFER_READBACKS"));
    gpu.set_interleave(on("GPU_INTERLEAVE"));
    gpu.set_web_limits(on("GPU_WEB_LIMITS"));
    gpu.set_device_msaa(on("GPU_DEVICE_MSAA"));
    gpu.set_times(on("GPU_TIMES"));
    gpu.set_only(
        std::env::var("GPU_ONLY")
            .ok()
            .as_deref()
            .and_then(switch_gpu::draw_range),
    );
}
