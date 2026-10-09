//! Convert model format (delegates to compressr)

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Result};
use colored::Colorize;

/// Convert model between formats using compressr
pub fn convert(
    input: String,
    output: Option<String>,
    format: String,
    quantization: Option<String>,
    verbose: bool,
) -> Result<()> {
    // Find compressr binary
    let compressr_bin = find_compressr()?;

    eprintln!(
        "{} {} → {}",
        "Converting".green().bold(),
        input.bold(),
        format.bold()
    );

    let mut cmd = Command::new(&compressr_bin);
    cmd.arg("convert").arg(&input).arg("--format").arg(&format);

    if let Some(ref out) = output {
        cmd.arg("--output").arg(out);
    }

    if let Some(ref quant) = quantization {
        cmd.arg("--quantization").arg(quant);
    }

    // compressr's `convert` has no verbose flag: it logs through
    // `RUST_LOG`. A level the caller set stays.
    if verbose && std::env::var_os("RUST_LOG").is_none() {
        cmd.env("RUST_LOG", "debug");
    }

    let status = cmd.status().map_err(|e| {
        anyhow!(
            "Failed to run compressr at '{}': {}\n{}",
            compressr_bin.display(),
            e,
            INSTALL_HINT
        )
    })?;

    if !status.success() {
        return Err(anyhow!(
            "compressr exited with status {}",
            status.code().unwrap_or(-1)
        ));
    }

    eprintln!("{} Conversion complete", "✓".green());
    Ok(())
}

/// How to get compressr when it is not found.
const INSTALL_HINT: &str =
    "Install compressr with: cargo install --git https://github.com/ml-rust/compressr --locked";

/// The compressr binary: on `PATH`, next to this executable, or in a
/// sibling `compressr` checkout's build directory.
fn find_compressr() -> Result<PathBuf> {
    let name = format!("compressr{}", std::env::consts::EXE_SUFFIX);
    if let Some(found) = std::env::var_os("PATH").and_then(|path| search_path(&path, "compressr")) {
        return Ok(found);
    }

    let beside_exe = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(&name)));
    let dev_paths = [
        Path::new("../compressr/target/release").join(&name),
        Path::new("../compressr/target/debug").join(&name),
    ];
    if let Some(found) = beside_exe
        .into_iter()
        .chain(dev_paths)
        .find(|p| p.is_file())
    {
        return Ok(found);
    }

    Err(anyhow!(
        "compressr binary not found on PATH.\n{INSTALL_HINT}"
    ))
}

/// The first file named `program` in the directories of `path`, a
/// `PATH`-style list. On Windows each `PATHEXT` extension is tried too.
fn search_path(path: &OsString, program: &str) -> Option<PathBuf> {
    let names = candidate_names(program);
    std::env::split_paths(path)
        .filter(|dir| !dir.as_os_str().is_empty())
        .flat_map(|dir| names.iter().map(move |name| dir.join(name)))
        .find(|candidate| candidate.is_file())
}

/// The file names `program` runs as on this platform.
fn candidate_names(program: &str) -> Vec<String> {
    if cfg!(windows) {
        let exts = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
        std::iter::once(program.to_string())
            .chain(
                exts.split(';')
                    .filter(|ext| !ext.is_empty())
                    .map(|ext| format!("{program}{}", ext.to_ascii_lowercase())),
            )
            .collect()
    } else {
        vec![program.to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_program_in_a_later_path_directory() {
        let empty = tempfile::tempdir().expect("temp dir");
        let holder = tempfile::tempdir().expect("temp dir");
        let file = holder
            .path()
            .join(format!("tool{}", std::env::consts::EXE_SUFFIX));
        std::fs::write(&file, b"").expect("write the program");
        let path = std::env::join_paths([empty.path(), holder.path()]).expect("join PATH");
        assert_eq!(search_path(&path, "tool"), Some(file));
    }

    #[test]
    fn a_missing_program_is_not_found() {
        let empty = tempfile::tempdir().expect("temp dir");
        let path = std::env::join_paths([empty.path()]).expect("join PATH");
        assert_eq!(search_path(&path, "tool"), None);
    }
}
