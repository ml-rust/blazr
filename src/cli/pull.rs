//! Pull model from HuggingFace Hub

use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::Result;
use colored::Colorize;
use hf_hub::api::sync::Api;
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};

/// Threshold above which we show a byte-level copy progress bar (10 MB)
const PROGRESS_BAR_THRESHOLD: u64 = 10 * 1024 * 1024;

/// Pull model from HuggingFace Hub
pub async fn pull(repo: String, file: Option<String>, output: Option<PathBuf>) -> Result<()> {
    let output_dir = output.unwrap_or_else(|| {
        std::env::var("BLAZR_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./models"))
    });

    std::fs::create_dir_all(&output_dir)?;

    eprintln!("{} from {}", "Downloading".green().bold(), repo.bold());

    let api = Api::new()?;
    let repo_api = api.model(repo.clone());
    let mut total_bytes = 0u64;
    let mut file_count = 0usize;

    if let Some(ref filename) = file {
        let size = download_file(&repo_api, filename, &output_dir)?;
        eprintln!("\n{} 1 file downloaded ({})", "✓".green(), HumanBytes(size),);
    } else {
        let model_name = repo.split('/').next_back().unwrap_or(&repo);
        let model_dir = output_dir.join(model_name);
        std::fs::create_dir_all(&model_dir)?;

        eprintln!(
            "  Destination: {}\n",
            model_dir.display().to_string().dimmed()
        );

        let files_to_try = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
        ];

        for filename in files_to_try {
            if let Ok(size) = download_file(&repo_api, filename, &model_dir) {
                total_bytes += size;
                file_count += 1;
            }
        }

        // Check for sharded safetensors
        let (s_bytes, s_count) = download_sharded_safetensors(&repo_api, &model_dir);
        total_bytes += s_bytes;
        file_count += s_count;

        // Check for GGUF files
        let (g_bytes, g_count) = download_gguf_files(&repo_api, &model_dir, &repo);
        total_bytes += g_bytes;
        file_count += g_count;

        eprintln!(
            "\n{} {} files downloaded ({}) to: {}",
            "✓".green(),
            file_count,
            HumanBytes(total_bytes),
            model_dir.display().to_string().bold()
        );
    }

    Ok(())
}

/// Download a single file with progress indication.
/// Returns the file size in bytes on success.
fn download_file(
    repo_api: &hf_hub::api::sync::ApiRepo,
    filename: &str,
    dest_dir: &std::path::Path,
) -> Result<u64> {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("  {spinner:.cyan} {msg}")
            .unwrap(),
    );
    spinner.set_message(format!("Fetching {}", filename));
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));

    let cached_path = repo_api.get(filename)?;
    spinner.finish_and_clear();

    let size = std::fs::metadata(&cached_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let dest = dest_dir.join(filename);

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
            .unwrap()
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

/// Try to download sharded safetensors files. Returns (total_bytes, file_count).
fn download_sharded_safetensors(
    repo_api: &hf_hub::api::sync::ApiRepo,
    model_dir: &std::path::Path,
) -> (u64, usize) {
    let mut total = 0u64;
    let mut count = 0usize;

    if let Ok(cached) = repo_api.get("model.safetensors.index.json") {
        let dest = model_dir.join("model.safetensors.index.json");
        let _ = std::fs::copy(&cached, &dest);

        if let Ok(content) = std::fs::read_to_string(&dest) {
            if let Ok(index) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                    let mut shard_files: Vec<String> = weight_map
                        .values()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    shard_files.sort();
                    shard_files.dedup();

                    for shard in &shard_files {
                        match download_file(repo_api, shard, model_dir) {
                            Ok(size) => {
                                total += size;
                                count += 1;
                            }
                            Err(e) => {
                                eprintln!("  {} Failed to download {}: {}", "✗".red(), shard, e);
                            }
                        }
                    }
                }
            }
        }
    }

    (total, count)
}

/// Try to download GGUF files from the repo. Returns (total_bytes, file_count).
fn download_gguf_files(
    repo_api: &hf_hub::api::sync::ApiRepo,
    model_dir: &std::path::Path,
    repo: &str,
) -> (u64, usize) {
    let model_name = repo.split('/').next_back().unwrap_or(repo).to_lowercase();

    let quant_suffixes = [
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "Q6_K", "Q3_K_M", "Q2_K",
    ];

    for suffix in &quant_suffixes {
        let filename = format!("{}.{}.gguf", model_name, suffix);
        if let Ok(size) = download_file(repo_api, &filename, model_dir) {
            return (size, 1);
        }
    }

    let unquant = format!("{}.gguf", model_name);
    if let Ok(size) = download_file(repo_api, &unquant, model_dir) {
        return (size, 1);
    }

    (0, 0)
}
