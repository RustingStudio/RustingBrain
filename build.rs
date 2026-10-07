//! Records which commit the library was built from, for [`rusting_brain::BUILD`].

use std::process::Command;

fn main() {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
    };
    // A crates.io build has no repository, so the hash is unknown there.
    let hash = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=RUSTING_BRAIN_GIT_HASH={hash}");

    // Rebuild when HEAD moves: a checkout changes HEAD, a commit changes the
    // branch ref it points at.
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
        println!("cargo:rerun-if-changed={git_dir}/refs/heads");
        println!("cargo:rerun-if-changed={git_dir}/packed-refs");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
