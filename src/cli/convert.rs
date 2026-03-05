//! Convert model format (delegates to compressr)

use std::path::PathBuf;
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
        cmd.arg("--out").arg(out);
    }

    if let Some(ref quant) = quantization {
        cmd.arg("--quantization").arg(quant);
    }

    if verbose {
        cmd.arg("--verbose");
    }

    let status = cmd.status().map_err(|e| {
        anyhow!(
            "Failed to run compressr at '{}': {}\n\
             Install compressr: cd compressr && cargo install --path .",
            compressr_bin.display(),
            e
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

/// Find the compressr binary in PATH or sibling directory
fn find_compressr() -> Result<PathBuf> {
    // Check PATH first
    if let Ok(output) = Command::new("which").arg("compressr").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            return Ok(PathBuf::from(path));
        }
    }

    // Check sibling target directory (development setup)
    let dev_paths = [
        "../compressr/target/release/compressr",
        "../compressr/target/debug/compressr",
    ];
    for p in &dev_paths {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }

    Err(anyhow!(
        "compressr binary not found.\n\
         Install it with: cd compressr && cargo install --path ."
    ))
}
