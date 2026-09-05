//! Bakes the commit this module was built from into the module.
//!
//! Nothing used to identify a build. A crash report pasted into an issue named
//! a pc, a title and a fault and gave no way at all to tell which code
//! produced them, so reading one meant guessing at how old it was.
//!
//! Best-effort by design: a build from a tarball, or on a machine with no
//! `git`, has no commit to name and says so by leaving the string empty
//! rather than by failing. A version with no commit is worth more than no
//! build at all.

use std::process::Command;

fn main() {
    println!("cargo:rustc-env=SWITCH_BUILD_COMMIT={}", commit());
    // Without this the value is baked once and then cached across every later
    // build, so every report after the first names the wrong commit.
    for path in ["../../.git/HEAD", "../../.git/refs/heads"] {
        println!("cargo:rerun-if-changed={path}");
    }
}

/// The short commit, with a `-dirty` suffix when the tree it was built from
/// had uncommitted changes, which is the case a report most needs flagged,
/// since that commit alone does not describe the build.
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
