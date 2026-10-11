//! `include/blazr.h` must match the `extern "C"` functions in `src/capi`.
//!
//! The test reads both as text. Every `blazr_*` function in one must be in
//! the other, with the same parameter count. With `cbindgen` on `PATH` or
//! in `~/.cargo/bin`, the test also regenerates the header and requires it
//! to match the committed one byte for byte.
//!
//! Regenerate: `cbindgen --config cbindgen.toml --output include/blazr.h src/capi/mod.rs`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const REGENERATE: &str =
    "regenerate: cbindgen --config cbindgen.toml --output include/blazr.h src/capi/mod.rs";

fn manifest() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Parameter count of a comma-separated list. `void`, empty and a trailing
/// comma count 0.
fn arity(params: &str) -> usize {
    params
        .split(',')
        .map(str::trim)
        .filter(|param| !param.is_empty() && *param != "void")
        .count()
}

/// `name -> parameter count` for each `blazr_` name in `text` that opens a
/// parameter list and whose line passes `is_declaration`. Comment lines
/// are skipped.
fn functions(text: &str, is_declaration: impl Fn(&str) -> bool) -> BTreeMap<String, usize> {
    let code: String = text
        .lines()
        .filter(|line| {
            let line = line.trim_start();
            !line.starts_with("//") && !line.starts_with("///") && !line.starts_with('*')
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut found = BTreeMap::new();
    let mut rest = code.as_str();
    while let Some(at) = rest.find("blazr_") {
        let tail = &rest[at..];
        let name_len = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(tail.len());
        let name = &tail[..name_len];
        let after = &tail[name_len..];
        let line_start = rest[..at].rfind('\n').map_or(0, |i| i + 1);
        let prefix = &rest[line_start..at];
        if after.starts_with('(') && is_declaration(prefix) {
            let close = after.find(')').expect("a parameter list closes");
            found.insert(name.to_string(), arity(&after[1..close]));
        }
        rest = &tail[name_len..];
    }
    found
}

fn header_functions() -> BTreeMap<String, usize> {
    let header =
        std::fs::read_to_string(manifest().join("include/blazr.h")).expect("read include/blazr.h");
    // A prototype line starts with its return type, never with a call site.
    functions(&header, |prefix| {
        let prefix = prefix.trim();
        !prefix.is_empty() && !prefix.contains('(') && !prefix.contains('`')
    })
}

fn rust_functions() -> BTreeMap<String, usize> {
    let mut all = BTreeMap::new();
    let dir = manifest().join("src/capi");
    for entry in std::fs::read_dir(&dir).expect("read src/capi") {
        let path = entry.expect("directory entry").path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            let source = std::fs::read_to_string(&path).expect("read a capi source file");
            all.extend(functions(&source, |prefix| {
                prefix.contains("extern \"C\" fn ")
            }));
        }
    }
    all
}

#[test]
fn header_and_rust_export_the_same_functions() {
    let header = header_functions();
    let rust = rust_functions();
    assert!(rust.len() >= 13, "found only {rust:?} in src/capi");
    assert_eq!(
        header, rust,
        "include/blazr.h and src/capi disagree. {REGENERATE}"
    );
}

fn cbindgen() -> Option<PathBuf> {
    let on_path = Command::new("cbindgen")
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success());
    if on_path {
        return Some(PathBuf::from("cbindgen"));
    }
    let home = std::env::var_os("HOME")?;
    let local = PathBuf::from(home).join(".cargo/bin/cbindgen");
    local.is_file().then_some(local)
}

#[test]
fn header_matches_a_fresh_cbindgen_run() {
    let Some(cbindgen) = cbindgen() else {
        println!("skip: cbindgen is not installed; the arity check still ran");
        return;
    };
    let out_dir = std::env::temp_dir().join(format!("blazr-capi-header-{}", std::process::id()));
    std::fs::create_dir_all(&out_dir).expect("create a scratch directory");
    let fresh = out_dir.join("blazr.h");
    let run = Command::new(cbindgen)
        .current_dir(manifest())
        .args(["--config", "cbindgen.toml", "--quiet", "--output"])
        .arg(&fresh)
        .arg("src/capi/mod.rs")
        .output()
        .expect("run cbindgen");
    assert!(
        run.status.success(),
        "cbindgen failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let fresh_text = std::fs::read_to_string(&fresh).expect("read the fresh header");
    let committed =
        std::fs::read_to_string(manifest().join("include/blazr.h")).expect("read include/blazr.h");
    let _ = std::fs::remove_dir_all(&out_dir);
    assert!(
        fresh_text == committed,
        "include/blazr.h is stale. {REGENERATE}"
    );
}
