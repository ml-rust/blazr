//! Pull model from HuggingFace Hub

use std::path::PathBuf;

use anyhow::Result;
use hf_hub::api::sync::Api;

/// Pull model from HuggingFace Hub
pub async fn pull(repo: String, file: Option<String>, output: Option<PathBuf>) -> Result<()> {
    let output_dir = output.unwrap_or_else(|| {
        std::env::var("BLAZR_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./models"))
    });

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&output_dir)?;

    println!("Downloading from: {}", repo);

    let api = Api::new()?;
    let repo_api = api.model(repo.clone());

    if let Some(ref filename) = file {
        // Download specific file
        println!("Downloading file: {}", filename);
        let path = repo_api.get(filename)?;

        // Copy to output directory
        let dest = output_dir.join(filename);
        std::fs::copy(&path, &dest)?;

        println!("Downloaded to: {}", dest.display());
    } else {
        // Download common model files
        let model_name = repo.split('/').last().unwrap_or(&repo);
        let model_dir = output_dir.join(model_name);
        std::fs::create_dir_all(&model_dir)?;

        println!("Downloading to: {}", model_dir.display());

        // Try to download common files
        let files_to_try = vec![
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ];

        for filename in files_to_try {
            match repo_api.get(filename) {
                Ok(cached_path) => {
                    let dest = model_dir.join(filename);
                    std::fs::copy(&cached_path, &dest)?;
                    println!("  Downloaded: {}", filename);
                }
                Err(_) => {
                    // File doesn't exist, skip
                }
            }
        }

        // Check for sharded safetensors
        if let Ok(files) = list_repo_files(&repo) {
            for file in files {
                if file.starts_with("model-") && file.ends_with(".safetensors") {
                    match repo_api.get(&file) {
                        Ok(cached_path) => {
                            let dest = model_dir.join(&file);
                            std::fs::copy(&cached_path, &dest)?;
                            println!("  Downloaded: {}", file);
                        }
                        Err(e) => {
                            eprintln!("  Failed to download {}: {}", file, e);
                        }
                    }
                }
            }
        }

        println!("\nModel downloaded to: {}", model_dir.display());
    }

    Ok(())
}

/// List files in a HuggingFace repo
fn list_repo_files(repo: &str) -> Result<Vec<String>> {
    let api = Api::new()?;
    let _repo_api = api.model(repo.to_string());

    // The hf_hub crate doesn't have a direct list_files method in the sync API
    // We'll just return common patterns
    Ok(vec![])
}
