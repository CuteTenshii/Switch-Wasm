//! Throughput per instruction class: `cargo run --release -p switch-core
//! --example bench`.
//!
//! Each case is a loop of sixteen copies of one instruction plus a counter
//! decrement and a branch, so the reported figure is close to the per-instruction
//! cost of that class. `b .` is the floor: one instruction, decoded by the first
//! check there is, so it measures what a step costs before any real decode work.
//! Compare a change against these numbers *and* against a real frame
//! (`examples/jit_bench.rs`, `examples/hotspots.rs`, `tools/wasm_bench.mjs`) — a
//! decode reordering that doubles a microbenchmark can be worth nothing on a
//! real instruction mix.
//!
//! This measures whichever engine `Cpu::run` is using, so it reports the block
//! translator by default and the plain interpreter under `SWITCH_NO_JIT=1`.
//! A class the translator has no op for shows the two within noise of each
//! other, which is what says it fell back.
use std::time::Instant;
use switch_core::cpu::Cpu;

const CODE: u32 = 0x1000;
const DATA: u32 = 0x8000;
/// Copies of the instruction under test per loop iteration.
const BODY: u64 = 16;

fn run(name: &str, code: &[u32], steps: u64, setup: impl Fn(&mut Cpu)) {
    let mut cpu = Cpu::new();
    cpu.mem.map_zero(CODE, 0x1000).unwrap();
    cpu.mem.map_zero(DATA, 0x1000).unwrap();
    for (i, insn) in code.iter().enumerate() {
        cpu.mem.map(CODE + 4 * i as u32, &insn.to_le_bytes()).unwrap();
    }
    setup(&mut cpu);
    cpu.set_pc(CODE);
    let t = Instant::now();
    let report = cpu.run(steps).unwrap();
    let secs = t.elapsed().as_secs_f64();
    assert_eq!(report.steps, steps, "{name}: stopped early");
    println!("{name:<12} {:>7.1} M steps/s", steps as f64 / secs / 1e6);
}

/// A loop body of `BODY` copies of `insn`, then `subs x0, x0, #1` and `b.ne`.
fn loop_body(insn: u32) -> Vec<u32> {
    let mut code = vec![insn; BODY as usize];
    code.push(0xf100_0400); // subs x0, x0, #1
    let back = ((-(BODY as i64 + 1)) & 0x7FFFF) as u32;
    code.push(0x5400_0001 | (back << 5)); // b.ne <top>
    code
}

fn bench(name: &str, insn: u32, iters: u64) {
    run(name, &loop_body(insn), iters * (BODY + 2), |cpu| {
        cpu.set_reg(0, iters); // loop counter
        cpu.set_reg(2, u64::from(DATA) + 0x10); // load/store base
        cpu.set_reg(3, 3);
    });
}

fn main() {
    let iters = 500_000;
    run("b .", &[0x1400_0000], 8_000_000, |_| {});
    for (name, insn) in [
        ("nop", 0xd503_201f),
        ("add imm", 0x9100_0421),
        ("add reg", 0x8b03_0821),
        ("orr imm", 0xb240_0421),
        ("madd", 0x9b03_7c21),
        ("ldr w", 0xb940_0041),
        ("ldr x", 0xf940_0041),
        ("str w", 0xb900_0041),
        ("ldrb w", 0x3940_0041),
        ("fmul s", 0x1e22_0821),
        ("fmadd s", 0x1f02_0821),
        ("scvtf s", 0x1e22_0061),
        ("fcvtzu s", 0x7ea1_b821),
        ("umov b", 0x0e01_3c01),
    ] {
        bench(name, insn, iters);
    }
}
