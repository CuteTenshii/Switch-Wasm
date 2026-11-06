//! Boot an NRO and print its console output: `boot_nx <path.nro>`.
mod common;

use std::path::Path;

use clap::Parser;
use common::Flow;
use switch_core::cpu::Cpu;

#[derive(Parser)]
struct Cli {
    nro: String,
    #[command(flatten)]
    shared: common::Shared,
    /// Instructions to run.
    #[arg(long, default_value_t = u64::MAX)]
    steps: u64,
    /// Copy this host directory onto the SD card; an NRO inside it is launched from its SD path,
    /// as the page's Files dialog does, so it finds the files beside it.
    #[arg(long, value_name = "DIR")]
    sd: Option<String>,
    /// Print this SD card file when the run ends (a homebrew's own log).
    #[arg(long, value_name = "SD PATH")]
    show: Option<String>,
}

fn main() {
    let cli = common::parse::<Cli>(|cli| &cli.shared);
    let data = common::read(&cli.nro);
    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    let sd_path = cli
        .sd
        .as_ref()
        .and_then(|dir| copy_to_sd(&mut cpu, Path::new(dir), Path::new(&cli.nro)));
    let loaded = cpu
        .boot_homebrew(&data, sd_path.as_deref())
        .expect("boot nro");
    println!("entry = {:#010x}", loaded.entry);

    let mut debug = common::Debug::from_flags();
    debug.arm(&mut cpu);
    let pace = debug.pace(false);
    let run = common::drive(&mut cpu, pace, cli.steps, |cpu, done| {
        debug.tick(cpu, done);
        Flow::Continue
    });
    debug.report();
    debug.stop_state(&cpu);
    if let Some(fault) = &run.fault {
        println!("FAULT at step {}: {fault}", run.steps);
    }
    if run.halted {
        println!(
            "HALTED at step {} pc={:#x} x0={:#x}",
            run.steps,
            cpu.get_pc(),
            cpu.read_x(0)
        );
    }

    common::print_console(&cpu, 60);
    if let Some(show) = &cli.show {
        match cpu.fs.file(show) {
            Some(bytes) => {
                println!("--- {show} ({} bytes) ---", bytes.len());
                print!("{}", String::from_utf8_lossy(bytes));
            }
            None => println!("--- {show}: not on the SD card ---"),
        }
    }
}

/// Copy `dir` onto the SD card root. Returns the NRO's SD path when it lies inside `dir`.
fn copy_to_sd(cpu: &mut Cpu, dir: &Path, nro: &Path) -> Option<String> {
    let mut pending = vec![dir.to_path_buf()];
    while let Some(at) = pending.pop() {
        let entries = std::fs::read_dir(&at)
            .unwrap_or_else(|e| panic!("SD: cannot list {}: {e}", at.display()));
        for entry in entries {
            let entry = entry.unwrap_or_else(|e| panic!("SD: {}: {e}", at.display()));
            let host = entry.path();
            let sd = sd_name(dir, &host);
            if host.is_dir() {
                cpu.fs.create_dir(&sd);
                pending.push(host);
            } else {
                cpu.fs.write_file(&sd, common::read(&host));
            }
        }
    }
    // Absolute without resolving links, so a directory linked into `dir` still counts as inside.
    let (root, nro) = (
        std::path::absolute(dir).ok()?,
        std::path::absolute(nro).ok()?,
    );
    nro.starts_with(&root).then(|| sd_name(&root, &nro))
}

fn sd_name(root: &Path, host: &Path) -> String {
    let relative = host
        .strip_prefix(root)
        .expect("a path under the SD directory");
    format!("/{}", relative.to_string_lossy().replace('\\', "/"))
}
