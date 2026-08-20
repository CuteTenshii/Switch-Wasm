//! Break on guest PCs and dump registers, or profile where a run spends its
//! time: `trace <nro> [pc...]`. With no PCs, prints the hottest PCs at the end.
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use switch_core::cpu::Cpu;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: trace <nro> [pc...]");
    let watch: HashSet<u32> = args
        .map(|a| u32::from_str_radix(a.trim_start_matches("0x"), 16).expect("pc"))
        .collect();
    let data = fs::read(&path).expect("read nro");
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.boot_homebrew(&data).expect("boot");

    let mut steps = 0u64;
    let mut hot: HashMap<u32, u64> = HashMap::new();
    let cap: u64 = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(60_000_000);
    while !cpu.halted && steps < cap {
        let pc = cpu.get_pc();
        if watch.contains(&pc) {
            print!("{steps} pc={pc:#x}");
            for r in [0u8, 1, 2, 3, 19, 20, 21, 22, 23] {
                print!(" x{r}={:#x}", cpu.read_x(r));
            }
            println!(" lr={:#x}", cpu.read_x(30));
        }
        if watch.is_empty() && steps % 64 == 0 {
            *hot.entry(pc).or_default() += 1;
        }
        if let Err(e) = cpu.step() {
            println!("FAULT at {steps}: {e} pc={pc:#x}");
            break;
        }
        steps += 1;
    }
    println!("stopped after {steps} steps, halted={} pc={:#x}", cpu.halted, cpu.get_pc());
    if !hot.is_empty() {
        let mut top: Vec<_> = hot.into_iter().collect();
        top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        println!("--- hottest PCs ---");
        for (pc, n) in top.into_iter().take(20) {
            println!("  {pc:#010x}  {n}");
        }
    }
    println!("--- console ---\n{}", String::from_utf8_lossy(&cpu.out));
}
