//! List which of a frame's shader programs the WGSL translator cannot take:
//! `shader_coverage <container> <prod.keys> [title.keys] [frame]`.
//!
//! Reports the first blocker per program. `BEFORE=<n>` prints the `n` instructions
//! before each blocker; `FRAMES=<n>` records `n` frames instead of one.
mod common;

use std::collections::BTreeMap;
use switch_core::cpu::Cpu;
use switch_core::gpu::shader::compiled::Compiled;
use switch_core::gpu::shader::wgsl::{self, Caps, Stage, Unsupported};
use switch_core::gpu::shader::{uses, Program};

const USAGE: &str = "shader_coverage <container> <prod.keys> [title.keys] [frame]";

const BOOT_BUDGET: u64 = 20_000_000_000;
const FRAME_BUDGET: u64 = 2_000_000_000;
const ROWS: usize = 20;

struct Used {
    stage: Stage,
    addr: u64,
    program: Program,
    draws: u64,
}

/// A blocker with the instruction index dropped, so one opcode is one row.
fn blocker(why: Unsupported) -> String {
    match why {
        Unsupported::Op { op, .. } => {
            let text = format!("{op:?}");
            let name = text
                .split(|c: char| !c.is_alphanumeric())
                .next()
                .unwrap_or("?");
            format!("op {name}")
        }
        Unsupported::Quad { .. } => "quad operation outside a fragment shader".into(),
        Unsupported::DepthCompare { .. } => "shadow sample (depth texture + comparison)".into(),
        Unsupported::TextureDimension { dim } => format!("texture dimension {dim:?}"),
        Unsupported::UndecodedTarget { .. } => "branch to an undecoded target".into(),
        Unsupported::IndirectBranch { .. } => "brx with an unread jump table".into(),
        Unsupported::UntracedHandle { .. } => "bindless handle not from a constant bank".into(),
    }
}

/// Build the full module: some refusals only happen when bindings are laid out.
fn compiles(program: &Program, stage: Stage, caps: Caps) -> Result<(), Unsupported> {
    let translated = wgsl::translate_for(&Compiled::new(program), caps)?;
    let layout = wgsl::Layout::of(&translated, stage);
    wgsl::module(&translated, stage, &layout)?;
    Ok(())
}

fn show_before() -> Option<usize> {
    std::env::var("BEFORE").ok()?.parse().ok()
}

fn blocked_at(why: &Unsupported) -> Option<usize> {
    match *why {
        Unsupported::Op { at, .. }
        | Unsupported::UndecodedTarget { at }
        | Unsupported::IndirectBranch { at }
        | Unsupported::Quad { at }
        | Unsupported::DepthCompare { at }
        | Unsupported::UntracedHandle { at } => Some(at),
        Unsupported::TextureDimension { .. } => None,
    }
}

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
                if let (Some(at), Some(before)) = (blocked_at(&why), show_before()) {
                    println!(
                        "  {:?} program at {:#x} blocked at instruction {at}:",
                        entry.stage, entry.addr
                    );
                    for i in at.saturating_sub(before)..=at {
                        if let Some(insn) = entry.program.insns.get(i) {
                            println!("    {i:>5}  {:?}  {:?}", insn.pred, insn.op);
                        }
                    }
                }
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

    let boot = common::run_to(&mut cpu, BOOT_BUDGET, |cpu| cpu.nv.gpu.frames >= want_frame);
    if cpu.nv.gpu.frames < want_frame {
        println!(
            "never reached frame {want_frame}: stopped at {} after {} steps",
            cpu.nv.gpu.frames, boot.steps
        );
        common::report(&cpu, &boot);
        return;
    }

    let frames: u64 = std::env::var("FRAMES")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(1);
    uses::record();
    let target = cpu.nv.gpu.frames + frames;
    let frame = common::run_to(&mut cpu, FRAME_BUDGET * frames, |cpu| {
        cpu.nv.gpu.frames >= target
    });
    let bound = uses::take();

    // Keyed by address; the first decode is kept.
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
        "frame {want_frame} at step {}, {frames} frame(s) recorded over the next {} steps",
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
    report("no optional device features (a browser)", &used, Caps::NONE);
    println!();
    report(
        "with quad operations (subgroups)",
        &used,
        Caps {
            subgroups: true,
            ..Caps::NONE
        },
    );

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
