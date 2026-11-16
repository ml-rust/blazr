//! Pull model from HuggingFace Hub

use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use colored::Colorize;
use hf_hub::repository::RepoTypeModel;
use hf_hub::HFRepositorySync;
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};

use crate::engine::model_dir::stays_in_dir;
use crate::loader::hub_client::pull_plan::{plan, shards_of, SAFETENSORS_INDEX};
use crate::loader::hub_client::{sync_client, sync_files, sync_get, sync_model};

/// Threshold above which we show a byte-level copy progress bar (10 MB)
const PROGRESS_BAR_THRESHOLD: u64 = 10 * 1024 * 1024;

/// How a `pull` error tells the caller to name one file.
const PICK: &str = "Name the file to fetch with --file <name>";

/// Pull model from HuggingFace Hub into `output_dir`: `--output`, then
/// `BLAZR_MODEL_DIR`, then `model_dir`, then `./models`.
///
/// Without `file`, the repo's file list picks the files: its configs and
/// tokenizer files, then `model.safetensors`, the shards its index names,
/// or its one GGUF file.
///
/// # Errors
///
/// An error when the file list or any download fails, when the repo holds
/// no weights blazr loads or several GGUF files, and when offline mode
/// (`HF_HUB_OFFLINE`) is on.
pub async fn pull(repo: String, file: Option<String>, output_dir: PathBuf) -> Result<()> {
    std::fs::create_dir_all(&output_dir)?;

    eprintln!("{} from {}", "Downloading".green().bold(), repo.bold());

    let client = sync_client(None)?;
    let repo_api = sync_model(&client, &repo);

    if let Some(ref filename) = file {
        if !stays_in_dir(filename) {
            bail!(
                "--file '{filename}' leaves {}. Pass a path in the repo",
                output_dir.display()
            );
        }
        let size = download_file(&repo_api, &repo, filename, &output_dir)?;
        eprintln!("\n{} 1 file downloaded ({})", "✓".green(), HumanBytes(size),);
        return Ok(());
    }

    let model_name = repo.split('/').next_back().unwrap_or(&repo);
    if !stays_in_dir(model_name) {
        bail!(
            "repo id '{repo}' leaves {}. Pass an id such as org/name",
            output_dir.display()
        );
    }
    let files =
        sync_files(&repo_api).map_err(|e| anyhow!("listing the files of {repo} failed: {e}"))?;
    let plan = plan(&files).map_err(|e| anyhow!(e.message(&repo, PICK)))?;
    let model_dir = output_dir.join(model_name);
    std::fs::create_dir_all(&model_dir)?;

    eprintln!(
        "  Destination: {}\n",
        model_dir.display().to_string().dimmed()
    );

    let fetch = |name: &str| download_file(&repo_api, &repo, name, &model_dir);
    let (mut total_bytes, mut file_count) = fetch_all(&plan.files, fetch)?;
    if plan.sharded {
        let index_path = model_dir.join(SAFETENSORS_INDEX);
        let index = std::fs::read_to_string(&index_path)
            .with_context(|| format!("reading {}", index_path.display()))?;
        let shards = shards_of(&index).map_err(|e| anyhow!("{repo}: {e}"))?;
        let (bytes, count) = fetch_all(&shards, fetch)?;
        total_bytes += bytes;
        file_count += count;
    }

    eprintln!(
        "\n{} {} files downloaded ({}) to: {}",
        "✓".green(),
        file_count,
        HumanBytes(total_bytes),
        model_dir.display().to_string().bold()
    );

    Ok(())
}

/// Fetches every file of `names` with `fetch`, in order, and returns the
/// bytes and the file count. The first error stops the pull.
fn fetch_all(names: &[String], mut fetch: impl FnMut(&str) -> Result<u64>) -> Result<(u64, usize)> {
    let mut bytes = 0u64;
    for name in names {
        bytes += fetch(name)?;
    }
    Ok((bytes, names.len()))
}

/// Download a single file with progress indication.
/// Returns the file size in bytes on success.
fn download_file(
    repo_api: &HFRepositorySync<RepoTypeModel>,
    repo: &str,
    filename: &str,
    dest_dir: &std::path::Path,
) -> Result<u64> {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("  {spinner:.cyan} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    spinner.set_message(format!("Fetching {}", filename));
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));

    let cached_path = sync_get(repo_api, filename, true);
    spinner.finish_and_clear();
    let cached_path =
        cached_path.map_err(|e| anyhow!("downloading {filename} from {repo} failed: {e}"))?;

    let size = std::fs::metadata(&cached_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let dest = dest_dir.join(filename);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // For large files, show a byte-level progress bar during copy
    if size >= PROGRESS_BAR_THRESHOLD {
        copy_with_progress(&cached_path, &dest, size)?;
    } else {
        std::fs::copy(&cached_path, &dest)?;
    }

    eprintln!(
        "  {} {} {}",
        "✓".green(),
        filename,
        format!("({})", HumanBytes(size)).dimmed()
    );

    Ok(size)
}

/// Copy a file with a byte-level progress bar
fn copy_with_progress(
    src: &std::path::Path,
    dest: &std::path::Path,
    total_size: u64,
) -> Result<()> {
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  {bar:40.cyan/dim} {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("━╸─"),
    );

    let mut reader = std::io::BufReader::new(std::fs::File::open(src)?);
    let mut writer = std::io::BufWriter::new(std::fs::File::create(dest)?);
    let mut buf = vec![0u8; 256 * 1024]; // 256 KB buffer

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        pb.inc(n as u64);
    }

    writer.flush()?;
    pb.finish_and_clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A failed download used to be skipped, and `pull` exited 0.
    #[test]
    fn a_failed_download_stops_the_pull() {
        let names: Vec<String> = ["config.json", "model.safetensors", "tokenizer.json"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let mut tried = Vec::new();
        let result = fetch_all(&names, |name| {
            tried.push(name.to_string());
            match name {
                "model.safetensors" => Err(anyhow!("downloading {name} failed: timed out")),
                _ => Ok(10),
            }
        });
        let err = result.expect_err("a failed download is an error");
        assert!(err.to_string().contains("model.safetensors"), "{err}");
        assert_eq!(tried, ["config.json", "model.safetensors"]);
    }

    #[test]
    fn every_fetched_file_counts() {
        let names = vec!["a".to_string(), "b".to_string()];
        assert_eq!(fetch_all(&names, |_| Ok(5)).expect("pull"), (10, 2));
    }
}
