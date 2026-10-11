//! Build `examples/capi/smoke.c` against the `staticlib` and run it.
//!
//! The test runs `cargo build --lib --features capi`, links the C program
//! with the system C compiler, and runs it on a Llama-family model. The
//! program loads, generates 16 tokens, stops twice, and unloads twice.
//!
//! It prints one skip line and passes when the model, the C compiler, or a
//! Linux host is absent. `CC` picks the compiler, else `cc`, `gcc`, `clang`.
//!
//! ```bash
//! BOOSTR_MODELS_DIR=/path/to/models cargo test --features capi --test capi_smoke_c -- --nocapture
//! ```

#![cfg(feature = "capi")]

use std::path::{Path, PathBuf};
use std::process::Command;

mod common;
use common::{model_fixture, skip_notice};

const ENV_MODEL: &str = "BLAZR_TEST_LLAMA_MODEL";
const MODELS_DIR_ENTRY: &str = "llama-3.2-1b";

/// The first C compiler that answers `--version`.
fn c_compiler() -> Option<String> {
    let candidates = std::env::var("CC")
        .ok()
        .into_iter()
        .chain(["cc", "gcc", "clang"].map(String::from));
    candidates.into_iter().find(|cc| {
        Command::new(cc)
            .arg("--version")
            .output()
            .is_ok_and(|out| out.status.success())
    })
}

/// `target/<profile>`: the test binary lives in its `deps` directory.
fn profile_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("test binary path");
    exe.parent()
        .and_then(Path::parent)
        .expect("test binary sits in target/<profile>/deps")
        .to_path_buf()
}

#[test]
fn c_program_loads_generates_cancels_and_unloads() {
    if !cfg!(target_os = "linux") {
        println!("skip: the C smoke test links with Linux system libraries only");
        return;
    }
    let Some(model) = model_fixture(ENV_MODEL, MODELS_DIR_ENTRY) else {
        skip_notice("a Llama-family checkpoint", ENV_MODEL);
        return;
    };
    let Some(cc) = c_compiler() else {
        println!("skip: no C compiler answers; set CC or install cc, gcc or clang");
        return;
    };

    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let profile = profile_dir();
    let mut build = Command::new(env!("CARGO"));
    build
        .current_dir(manifest)
        .args(["build", "--lib", "--features", "capi"]);
    if profile.file_name().is_some_and(|name| name == "release") {
        build.arg("--release");
    }
    let status = build.status().expect("run cargo build");
    assert!(status.success(), "cargo build --lib --features capi failed");

    let staticlib = profile.join("libblazr.a");
    assert!(staticlib.is_file(), "{} is missing", staticlib.display());
    let out_dir = profile.join("capi-smoke");
    std::fs::create_dir_all(&out_dir).expect("create the smoke output directory");
    let program = out_dir.join("smoke");
    let compile = Command::new(&cc)
        .arg(manifest.join("examples/capi/smoke.c"))
        .arg("-I")
        .arg(manifest.join("include"))
        .arg("-o")
        .arg(&program)
        .arg(&staticlib)
        .args([
            "-lgcc_s",
            "-lutil",
            "-lrt",
            "-lpthread",
            "-lm",
            "-ldl",
            "-lc",
        ])
        .output()
        .expect("run the C compiler");
    assert!(
        compile.status.success(),
        "{cc} failed:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&program)
        .arg(&model)
        .output()
        .expect("run the smoke program");
    let stdout = String::from_utf8_lossy(&run.stdout);
    println!("{stdout}");
    assert!(
        run.status.success(),
        "smoke exited with {}:\n{stdout}\n{}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(stdout.contains("tokens: 16"), "{stdout}");
    assert!(stdout.contains("callback stop after 3 tokens"), "{stdout}");
    assert!(stdout.contains("cancelled after 3 tokens"), "{stdout}");
    assert!(stdout.contains("already unloaded"), "{stdout}");
    assert!(stdout.trim_end().ends_with("ok"), "{stdout}");
}
