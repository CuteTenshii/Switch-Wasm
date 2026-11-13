//! `screenshot_title` with the GPU backend installed; `cmp` the two PPMs to
//! check the backend against the software reference:
//! `screenshot_gpu <target> --out <out.ppm> [--keys <prod.keys>] [--frame <n>]`.
#[path = "../../switch-core/examples/common/mod.rs"]
mod common;

use clap::Parser;
use common::Flow;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    target: common::Target,
    #[command(flatten)]
    shared: common::Shared,
    #[command(flatten)]
    boot: common::Boot,
    #[command(flatten)]
    report: common::Report,
    #[command(flatten)]
    play: common::Play,
    #[command(flatten)]
    shots: common::Shots,
    #[command(flatten)]
    probe: common::Probe,
    /// Where the frame is written.
    #[arg(long, value_name = "OUT.PPM")]
    out: String,
    /// The presented frame to write.
    #[arg(long, default_value_t = 1)]
    frame: u64,
    /// Instructions to run.
    #[arg(long, default_value_t = u64::MAX)]
    steps: u64,
    /// Dock after this presented frame, like a real dock mid-run (`--docked` docks at boot).
    #[arg(long, default_value_t = u64::MAX, value_name = "FRAME")]
    dock_at: u64,
    #[command(flatten)]
    device: Device,
}

/// The device's diagnostic switches.
#[derive(clap::Args)]
struct Device {
    #[arg(long)]
    defer_readbacks: bool,
    #[arg(long)]
    sync_every_present: bool,
    #[arg(long)]
    interleave: bool,
    #[arg(long)]
    web_limits: bool,
    #[arg(long)]
    device_msaa: bool,
    #[arg(long)]
    times: bool,
    /// Draw only this draw index, or the half-open range `a..b`.
    #[arg(long, value_parser = draw_range, value_name = "N|A..B")]
    only: Option<std::ops::Range<u32>>,
}

fn draw_range(spec: &str) -> Result<std::ops::Range<u32>, String> {
    switch_gpu::draw_range(spec).ok_or_else(|| format!("{spec:?} is not N or A..B"))
}

fn main() {
    let cli = common::parse::<Cli>(|cli| {
        common::Flags::default()
            .with(&cli.shared)
            .with(&cli.boot)
            .with(&cli.report)
            .with(&cli.play)
            .with(&cli.shots)
            .with(&cli.probe)
    });
    let program = cli.target.open_program();
    let (out, want) = (&cli.out, cli.frame);

    let mut gpu = match switch_gpu::Gpu::open() {
        Ok(mut gpu) => {
            configure(&mut gpu, &cli.device);
            Some(gpu)
        }
        Err(why) => {
            eprintln!("no GPU backend ({why}): this is `screenshot_title` with extra steps");
            None
        }
    };

    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    let dock_at = cli.dock_at;
    program.boot(&mut cpu);

    // Installed on the session so it precedes and covers every channel.
    let mut last_shot = 0;
    let real: Box<dyn switch_core::gpu::renderer::Renderer> = match gpu.take() {
        Some(gpu) => {
            println!("[gpu] installed");
            Box::new(gpu)
        }
        None => Box::new(switch_core::gpu::renderer::Software),
    };
    let mut draw_gate = common::DrawGate::install(&mut cpu, real);
    let mut debug = common::Debug::from_flags();
    debug.arm(&mut cpu);
    let mut press = common::Press::from_flags();
    let pace = debug.pace(false);
    let cores = common::start_cores(&mut cpu);
    let run = common::drive(&mut cpu, pace, cli.steps, |cpu, done| {
        debug.tick(cpu, done);
        if let Some(gate) = draw_gate.as_mut() {
            gate.tick(cpu);
        }
        common::shoot(cpu, out, &mut last_shot);
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
    });
    common::stop_cores(&mut cpu, cores);
    common::report(&cpu, &run);
    debug.report();
    debug.stop_state(&cpu);
    // The browser's Rendering panel: device draws and fallbacks.
    println!("rendering: {}", cpu.nv.gpu.renderer_report());
    if cpu.nv.gpu.framebuffer.is_empty() {
        println!("no frame");
        return;
    }
    common::write_ppm(out, &cpu.nv.gpu.framebuffer);
}

fn configure(gpu: &mut switch_gpu::Gpu, device: &Device) {
    gpu.set_defer_readbacks(device.defer_readbacks);
    gpu.set_sync_every_present(device.sync_every_present);
    gpu.set_interleave(device.interleave);
    gpu.set_web_limits(device.web_limits);
    gpu.set_device_msaa(device.device_msaa);
    gpu.set_times(device.times);
    gpu.set_only(device.only.clone());
}
