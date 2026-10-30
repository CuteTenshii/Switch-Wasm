//! Boot an NRO and print its console output: `boot_nx <path.nro>`.
//!
//! `SD=<dir>` copies a host directory onto the SD card; an NRO inside it is launched from
//! its SD path, as the page's Files dialog does, so it finds the files beside it.
//! `SHOW=<sd path>` prints that SD card file when the run ends (a homebrew's own log).
//! `DUMP=`, `TRAP_WRITE=`, `TRAP_READ=` and `WATCH_PC=` are described in [`common::Debug`].
mod common;

use std::path::Path;

use common::Flow;
use switch_core::cpu::Cpu;

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        common::usage("boot_nx <path.nro>")
    };
    let data = common::read(&path);
    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    let sd_path = std::env::var("SD")
        .ok()
        .and_then(|dir| copy_to_sd(&mut cpu, Path::new(&dir), Path::new(&path)));
    let loaded = match &sd_path {
        Some(at) => cpu.boot_homebrew_at(&data, at),
        None => cpu.boot_homebrew(&data),
    }
    .expect("boot nro");
    println!("entry = {:#010x}", loaded.entry);

    let mut debug = common::Debug::from_env();
    debug.arm(&mut cpu);
    let pace = debug.pace(false);
    let run = common::drive(
        &mut cpu,
        pace,
        common::env_u64("STEPS", u64::MAX),
        |cpu, done| {
            debug.tick(cpu, done);
            Flow::Continue
        },
    );
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
    if let Ok(show) = std::env::var("SHOW") {
        match cpu.fs.file(&show) {
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
