//! Where a frame's emulated instructions go: `hotspots <nro> [font.ttf]`.
//!
//! Skips startup, then counts every instruction of one steady-state frame by
//! address and by top-level encoding byte. The address histogram says which guest
//! function to blame (hbmenu spends most of a frame in its own software gradient
//! fill, not in the emulator); the encoding histogram says which decoder paths
//! are worth optimising.
use std::fs;
use switch_core::cpu::Cpu;

/// Where NROs are loaded, and how much of the space to histogram.
const IMAGE_BASE: u32 = 0x0800_0000;
const IMAGE_SIZE: u32 = 0x0100_0000;
/// Instructions per bucket in the address histogram (a 4 KiB page).
const BUCKET: usize = 1024;

fn main() {
    let mut args = std::env::args().skip(1);
    let nro = args.next().expect("usage: hotspots <nro> [font.ttf]");
    let font = args
        .next()
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/assets/font.ttf").into());

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    if let Ok(bytes) = fs::read(&font) {
        cpu.set_shared_font(bytes);
    }
    cpu.boot_homebrew(&fs::read(&nro).expect("read nro")).expect("boot");

    // Two frames of startup, so what follows is a steady-state frame.
    while !cpu.halted && cpu.nv.gpu.frames < 2 {
        cpu.step().expect("startup");
    }

    let mut by_addr = vec![0u32; (IMAGE_SIZE / 4) as usize];
    let mut by_top = [0u64; 256];
    let start = cpu.nv.gpu.frames;
    let mut total = 0u64;
    while !cpu.halted && cpu.nv.gpu.frames == start {
        let pc = cpu.get_pc();
        if (IMAGE_BASE..IMAGE_BASE + IMAGE_SIZE).contains(&pc) {
            by_addr[((pc - IMAGE_BASE) / 4) as usize] += 1;
            let insn = cpu.mem.read_u32(pc).unwrap_or(0);
            by_top[((insn >> 24) & 0xFF) as usize] += 1;
        }
        cpu.step().expect("frame");
        total += 1;
    }
    println!("one frame = {total} instructions");

    let mut buckets: Vec<(u64, u32)> = by_addr
        .chunks(BUCKET)
        .enumerate()
        .map(|(i, c)| {
            let count = c.iter().map(|&x| u64::from(x)).sum();
            (count, IMAGE_BASE + (i * BUCKET * 4) as u32)
        })
        .filter(|(count, _)| *count > 0)
        .collect();
    buckets.sort_by_key(|(count, _)| std::cmp::Reverse(*count));
    println!("--- hottest guest code (4 KiB buckets) ---");
    for (count, addr) in buckets.iter().take(10) {
        println!("{addr:#010x}  {count:>12}  {:5.2}%", pct(*count, total));
    }

    let mut tops: Vec<(u64, usize)> = by_top.iter().copied().zip(0..256).filter(|(n, _)| *n > 0).collect();
    tops.sort_by_key(|(n, _)| std::cmp::Reverse(*n));
    println!("--- instruction mix (bits 31:24) ---");
    for (count, top) in tops.iter().take(16) {
        println!("{top:#04x}  {count:>12}  {:5.2}%", pct(*count, total));
    }
}

fn pct(count: u64, total: u64) -> f64 {
    count as f64 * 100.0 / total as f64
}
