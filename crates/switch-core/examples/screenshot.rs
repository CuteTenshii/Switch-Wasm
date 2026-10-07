//! Boot an NRO and write the Nth presented frame to a PPM:
//! `screenshot <path.nro> <out.ppm> [frame] [font.ttf]`. Takes the [`common::Debug`] knobs.
mod common;

use common::{Flow, Pace};
use switch_core::cpu::Cpu;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        common::usage("screenshot <path.nro> <out.ppm> [frame] [font.ttf]")
    };
    let Some(out) = args.next() else {
        common::usage("screenshot <path.nro> <out.ppm> [frame] [font.ttf]")
    };
    let want = args.next().and_then(|a| a.parse::<u64>().ok()).unwrap_or(1);
    let font = args.next();

    let data = common::read(&path);
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    match font {
        Some(font) => cpu.set_shared_font(common::read(&font)),
        None => common::load_fallback_font(&mut cpu),
    }
    cpu.boot_homebrew(&data).expect("boot");

    let mut debug = common::Debug::from_env();
    debug.arm(&mut cpu);
    let pace = if debug.stepwise() {
        Pace::Instructions
    } else {
        Pace::Blocks
    };
    let budget = common::env_u64("STEPS", 200_000_000);
    let run = common::drive(&mut cpu, pace, budget, |cpu, done| {
        if cpu.nv.gpu.frames >= want {
            return Flow::Stop;
        }
        debug.tick(cpu, done);
        Flow::Continue
    });
    common::report(&cpu, &run);
    debug.report();
    debug.stop_state(&cpu);

    if cpu.nv.gpu.framebuffer.is_empty() {
        println!("no frame was presented");
        return;
    }
    common::write_ppm(&out, &cpu.nv.gpu.framebuffer);
}
