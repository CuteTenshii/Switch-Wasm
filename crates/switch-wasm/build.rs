//! Bakes the build's git commit into `SWITCH_BUILD_COMMIT` (empty without git).

use std::process::Command;

fn main() {
    println!("cargo:rustc-env=SWITCH_BUILD_COMMIT={}", commit());
    for path in ["../../.git/HEAD", "../../.git/refs/heads"] {
        println!("cargo:rerun-if-changed={path}");
    }
}

/// Short commit hash, suffixed `-dirty` when the tree had uncommitted changes.
fn commit() -> String {
    let Some(hash) = git(&["rev-parse", "--short", "HEAD"]) else {
        return String::new();
    };
    match git(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(changes) if !changes.is_empty() => format!("{hash}-dirty"),
        _ => hash,
    }
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_string())
}
