//! Replays the T32 snippets `tools/thumb_difftest.py` generated and compares what the
//! interpreter leaves with what `qemu-system-arm` left.
//!
//! ```text
//! python3 tools/thumb_difftest.py --out target/thumb-difftest
//! cargo run --release -p switch-core --example thumb_difftest -- target/thumb-difftest
//! ```

use switch_core::cpu::{Cpu, ExecMode};

const BASE: u32 = 0x4020_0000;
const BUF: u32 = 0x4100_0000;
const RESULTS: u32 = 0x4200_0000;
const CASE: usize = 64;
/// What is compared of a result word: N, Z, C, V, Q and GE of the status word, the rest whole.
const STATUS_MASK: u32 = 0xF80F_0000;
const SHOWN: usize = 20;

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/thumb-difftest".into());
    let image = std::fs::read(format!("{dir}/emu.bin")).expect("run tools/thumb_difftest.py first");
    let cases = std::fs::read_to_string(format!("{dir}/cases.txt")).expect("no case list");
    let count = cases.lines().count();
    let expected = std::fs::read(format!("{dir}/expected.bin")).unwrap_or_default();

    let mut cpu = Cpu::new();
    cpu.mem
        .map(u64::from(BASE), &image)
        .expect("cannot map the image");
    // Where the qemu image builds its page table, which the replay fills in too.
    cpu.mem
        .map_zero(0x4010_0000, 0x4000)
        .expect("cannot map the table");
    cpu.mem
        .map_zero(u64::from(BUF), 0x1000)
        .expect("cannot map the buffer");
    cpu.mem
        .map_zero(u64::from(RESULTS), (count * CASE) as u64)
        .expect("cannot map the results");
    cpu.set_mode(ExecMode::A32);
    cpu.set_pc(BASE);
    // Four hundred instructions a case with room to spare.
    let report = cpu.run(count as u64 * 2000 + 10_000);
    if let Err(e) = &report {
        println!("the run faulted: {e}");
        let snippets = image.len() - count * CASE;
        let case = (cpu.get_pc().wrapping_sub(BASE) as usize).wrapping_sub(snippets) / CASE;
        let regs: Vec<String> = (0..15).map(|i| format!("{:x}", cpu.read_x(i))).collect();
        println!("registers: {}", regs.join(" "));
        println!(
            "stopped at pc={:#x}, in case {case}: {}",
            cpu.get_pc(),
            cases.lines().nth(case).unwrap_or("?")
        );
    }
    if expected.len() != count * CASE {
        println!("no qemu results for {count} cases; the run above is all there is");
        return;
    }
    let mut got = vec![0u8; expected.len()];
    cpu.mem
        .read_into(u64::from(RESULTS), &mut got)
        .expect("results are mapped");

    if let Some(case) = std::env::var("THUMB_CASE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        println!("{}", cases.lines().nth(case).unwrap_or("?"));
        for w in 0..15 {
            println!(
                "  word {w:2}: interpreter {:#010x}, qemu {:#010x}",
                word(&got, case * CASE + 4 * w),
                word(&expected, case * CASE + 4 * w)
            );
        }
        return;
    }
    let (mut bad, mut shown) = (0usize, 0usize);
    for (i, text) in cases.lines().enumerate().take(count) {
        let differs: Vec<usize> = (0..15)
            .filter(|&w| {
                let (a, b) = (
                    word(&got, i * CASE + 4 * w),
                    word(&expected, i * CASE + 4 * w),
                );
                if w == 11 {
                    a & STATUS_MASK != b & STATUS_MASK
                } else {
                    a != b
                }
            })
            .collect();
        if differs.is_empty() {
            continue;
        }
        bad += 1;
        if shown < SHOWN {
            shown += 1;
            println!("case {i}: {text}");
            for w in differs {
                let name = match w {
                    0..=10 => format!("r{w}"),
                    11 => "status".into(),
                    12 => "sp".into(),
                    13 => "buffer sum".into(),
                    _ => "buffer mix".into(),
                };
                println!(
                    "    {name}: interpreter {:#010x}, qemu {:#010x}",
                    word(&got, i * CASE + 4 * w),
                    word(&expected, i * CASE + 4 * w)
                );
            }
        }
    }
    println!("{count} cases, {} agree, {bad} differ", count - bad);
    std::process::exit(i32::from(bad != 0 || report.is_err()));
}
