//! Boot an NRO and write a presented frame to a PPM:
//! `screenshot <nro> <out.ppm> [frame-index] [font.ttf]`.
use std::fs;
use switch_core::cpu::{Cpu, SyscallMode};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: screenshot <nro> <out.ppm> [frame] [font]");
    let out = args.next().expect("output path");
    let want = args.next().and_then(|a| a.parse::<u64>().ok()).unwrap_or(1);
    // The shared system font `pl:u` hands to the guest, which is what homebrew
    // renders its text with. The frontend fetches the same file.
    let font = args
        .next()
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/assets/font.ttf").into());
    let data = fs::read(&path).expect("read nro");

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.syscall_mode = SyscallMode::Horizon;
    match fs::read(&font) {
        Ok(bytes) => cpu.set_shared_font(bytes),
        Err(e) => println!("no font at {font} ({e}): text will not render"),
    }
    cpu.boot_homebrew(&data).expect("boot");

    let mut steps = 0u64;
    while !cpu.halted && cpu.nv.gpu.frames < want && steps < 200_000_000 {
        if let Err(e) = cpu.step() {
            println!("FAULT at {steps}: {e}");
            break;
        }
        steps += 1;
    }
    let fb = &cpu.nv.gpu.framebuffer;
    println!(
        "steps={steps} frames={} {}x{} stats={:?}",
        cpu.nv.gpu.frames, fb.width, fb.height, cpu.nv.gpu.stats
    );
    if fb.is_empty() {
        println!("no frame was presented");
        return;
    }
    let mut ppm = format!("P6\n{} {}\n255\n", fb.width, fb.height).into_bytes();
    for px in &fb.pixels {
        ppm.extend_from_slice(&[*px as u8, (*px >> 8) as u8, (*px >> 16) as u8]);
    }
    fs::write(&out, ppm).expect("write ppm");
    let non_black = fb.pixels.iter().filter(|p| **p & 0x00FF_FFFF != 0).count();
    println!("wrote {out}: {non_black}/{} non-black pixels", fb.pixels.len());
}
