//! The binary runs its command on a thread of its own, so a process whose
//! main thread has a small stack still starts, parses and prints.

#![cfg(all(unix, feature = "cli"))]

use std::process::Command;

/// The built binary.
const BLAZR: &str = env!("CARGO_BIN_EXE_blazr");

/// A main-thread stack far below what the command's frames need.
const MAIN_STACK_KIB: u32 = 128;

#[test]
fn a_small_main_stack_still_runs_the_command() {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("ulimit -s {MAIN_STACK_KIB} && exec \"$0\" \"$@\""))
        .arg(BLAZR)
        .arg("--help")
        .output()
        .expect("sh runs the binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exit {:?}\nstdout:\n{stdout}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("Usage"), "{stdout}");
}
