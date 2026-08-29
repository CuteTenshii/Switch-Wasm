//! Which of a real frame's shader programs the WGSL translator cannot take:
//! `shader_coverage <container> <prod.keys> [title.keys] [frame]`.
//!
//! The fragment shader interpreter is about half of a frame in any title the
//! software rasterizer draws — 49.9% of a Just Dance 2017 frame under `perf`,
//! against 8.0% for the whole emulated CPU — and every one of those 921,600
//! invocations re-runs a program the decoder already turned into a `Compiled`
//! once. Running it on the device instead means `gpu/shader/wgsl.rs` being
//! able to translate it, and the useful question is not how many opcodes that
//! module has arms for but how many of the ones a frame *executes* it is
//! still missing.
//!
//! This is `jit_coverage` for shaders, with one difference the two
//! translators' shapes force. `cpu::translates` answers per instruction, so
//! that tool can weigh every encoding a frame ran. WGSL translation is per
//! program and stops at the first thing it cannot emit, so what comes back
//! here is the *first* blocker in each program: fixing the top row can reveal
//! another behind it, and the loop is to fix and re-run. What the headline
//! counts do not depend on is that ordering — a program either translates
//! whole or it does not.
//!
//! Reported twice, because a warp shuffle is a question about the device
//! rather than about the translator: once for a device with no optional
//! features, and once for one with WGSL's quad operations.
//!
//! `SWITCH_FIRMWARE=<dir>` as everywhere else. A system applet is the subject
//! to prefer here: qlaunch reaches a frame in 35 million instructions and is
//! almost pure rasterizer, where Just Dance spends billions before it draws.
mod common;

use std::collections::BTreeMap;
use switch_core::cpu::Cpu;
use switch_core::gpu::shader::compiled::Compiled;
use switch_core::gpu::shader::wgsl::{self, Caps, Stage, Unsupported};
use switch_core::gpu::shader::{uses, Program};

const USAGE: &str = "shader_coverage <container> <prod.keys> [title.keys] [frame]";

/// How much of the boot to allow before giving up on reaching the frame.
const BOOT_BUDGET: u64 = 20_000_000_000;
/// How long to allow for the one frame being recorded.
const FRAME_BUDGET: u64 = 2_000_000_000;
/// How many distinct blockers to name.
const ROWS: usize = 20;

/// One program a frame bound, and how many draws used it.
struct Used {
    stage: Stage,
    addr: u64,
    program: Program,
    draws: u64,
}

/// A blocker's identity with the instruction index dropped, so that the same
/// missing opcode in two programs is one row rather than two.
fn blocker(why: Unsupported) -> String {
    match why {
        // `Op`'s `Debug` opens with the variant name and then its fields; the
        // name alone is what names the gap.
        Unsupported::Op { op, .. } => {
            let text = format!("{op:?}");
            let name = text
                .split(|c: char| !c.is_alphanumeric())
                .next()
                .unwrap_or("?");
            format!("op {name}")
        }
        Unsupported::Subgroups { .. } => "warp shuffle (quad operations)".into(),
        Unsupported::DepthCompare { .. } => "shadow sample (depth texture + comparison)".into(),
        Unsupported::TextureDimension { dim } => format!("texture dimension {dim:?}"),
        Unsupported::UndecodedTarget { .. } => "branch to an undecoded target".into(),
        Unsupported::IndirectBranch { .. } => "brx with an unread jump table".into(),
    }
}

/// Take one program all the way to a module, which is the whole test: a
/// texture dimensionality is rejected when the bindings are laid out rather
/// than when the instructions are emitted, so stopping at `translate` would
/// call a program translatable that nothing can bind.
fn compiles(program: &Program, stage: Stage, caps: Caps) -> Result<(), Unsupported> {
    let translated = wgsl::translate_for(&Compiled::new(program), caps)?;
    let layout = wgsl::Layout::of(&translated, stage);
    wgsl::module(&translated, stage, &layout)?;
    Ok(())
}

/// Translate everything `used` holds under `caps` and print what came back.
fn report(label: &str, used: &[Used], caps: Caps) {
    let mut blocked: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut ok_programs = 0u64;
    let mut ok_draws = 0u64;
    let mut ok_fragment = 0u64;
    let mut fragment = 0u64;

    for entry in used {
        if entry.stage == Stage::Fragment {
            fragment += 1;
        }
        match compiles(&entry.program, entry.stage, caps) {
            Ok(()) => {
                ok_programs += 1;
                ok_draws += entry.draws;
                if entry.stage == Stage::Fragment {
                    ok_fragment += 1;
                }
            }
            Err(why) => {
                let row = blocked.entry(blocker(why)).or_default();
                row.0 += entry.draws;
                row.1 += 1;
            }
        }
    }

    let programs = used.len() as u64;
    let draws: u64 = used.iter().map(|u| u.draws).sum();
    let pct = |part: u64, whole: u64| {
        if whole == 0 {
            0.0
        } else {
            part as f64 * 100.0 / whole as f64
        }
    };
    println!("--- {label} ---");
    println!(
        "  programs: {ok_programs} of {programs} translate ({:.0}%)",
        pct(ok_programs, programs)
    );
    println!(
        "  fragment: {ok_fragment} of {fragment} translate ({:.0}%) \
         — the stage the interpreter spends the frame in",
        pct(ok_fragment, fragment)
    );
    println!(
        "  draws:    {ok_draws} of {draws} covered ({:.0}%)",
        pct(ok_draws, draws)
    );
    if blocked.is_empty() {
        println!("  nothing blocked");
        return;
    }
    let mut rows: Vec<(u64, u64, String)> = blocked
        .into_iter()
        .map(|(why, (draws, programs))| (draws, programs, why))
        .collect();
    rows.sort_by_key(|(draws, programs, why)| {
        (
            std::cmp::Reverse(*draws),
            std::cmp::Reverse(*programs),
            why.clone(),
        )
    });
    println!("  first blocker in each program, by draws blocked:");
    for (blocked_draws, blocked_programs, why) in rows.iter().take(ROWS) {
        println!(
            "    {blocked_draws:>6} draws  {blocked_programs:>4} program(s)  {:5.1}% of the frame  {why}",
            pct(*blocked_draws, draws),
        );
    }
}

fn main() {
    let args = common::container_args(USAGE);
    let title = args.open();
    let want_frame = args.rest_num(0).unwrap_or(4);

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    title.mount_romfs(&mut cpu);
    common::load_fallback_font(&mut cpu);
    common::register_firmware(&mut cpu, &title.keys);
    title.boot(&mut cpu);

    // Nothing is sampled during the boot, so it runs through the block
    // translator rather than the interpreter.
    let boot = common::run_to(&mut cpu, BOOT_BUDGET, |cpu| cpu.nv.gpu.frames >= want_frame);
    if cpu.nv.gpu.frames < want_frame {
        println!(
            "never reached frame {want_frame}: stopped at {} after {} steps",
            cpu.nv.gpu.frames, boot.steps
        );
        common::report(&cpu, &boot);
        return;
    }

    // One frame, recorded. Every draw notes the programs it was about to run,
    // decoded — reading them back afterwards would need the GPU address space
    // the draw was using, and that has moved on by the time this returns.
    uses::record();
    let target = cpu.nv.gpu.frames + 1;
    let frame = common::run_to(&mut cpu, FRAME_BUDGET, |cpu| cpu.nv.gpu.frames >= target);
    let bound = uses::take();

    // Two draws binding one address are one program. The decode folds the
    // constant banks in, so the same address can decode differently under
    // different bindings; the first is kept, which is the program that draw
    // ran.
    let mut used: Vec<Used> = Vec::new();
    let mut index: BTreeMap<(u64, bool), usize> = BTreeMap::new();
    for (stage, addr, program) in bound {
        let key = (addr, stage == Stage::Fragment);
        match index.get(&key) {
            Some(&at) => used[at].draws += 1,
            None => {
                index.insert(key, used.len());
                used.push(Used {
                    stage,
                    addr,
                    program,
                    draws: 1,
                });
            }
        }
    }

    let draws: u64 = used.iter().map(|u| u.draws).sum();
    let fragment = used.iter().filter(|u| u.stage == Stage::Fragment).count();
    println!(
        "frame {want_frame} at step {}, recorded over the next {} steps",
        boot.steps, frame.steps
    );
    println!(
        "{draws} draw(s), {} distinct program(s): {} vertex, {fragment} fragment",
        used.len(),
        used.len() - fragment,
    );
    if used.is_empty() {
        println!("no draw ran in that frame — nothing to translate");
        return;
    }
    println!();
    report("no optional device features", &used, Caps::NONE);
    println!();
    report(
        "with quad operations (subgroups)",
        &used,
        Caps {
            subgroups: true,
            ..Caps::NONE
        },
    );

    // Named so a blocker can be disassembled and read against the source: the
    // addresses are the guest's, and `--example disasm_flat` takes them.
    println!();
    println!("--- programs, by draws ---");
    let mut ranked: Vec<&Used> = used.iter().collect();
    ranked.sort_by_key(|u| (std::cmp::Reverse(u.draws), u.addr));
    for entry in ranked.iter().take(ROWS) {
        let verdict = match compiles(&entry.program, entry.stage, Caps::NONE) {
            Ok(()) => "compiles".to_string(),
            Err(why) => blocker(why),
        };
        println!(
            "  {:#012x}  {:?}  {:>5} draw(s)  {:>4} insn  {verdict}",
            entry.addr,
            entry.stage,
            entry.draws,
            entry.program.insns.len(),
        );
    }
}
