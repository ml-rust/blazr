//! List models command

use std::path::PathBuf;

use anyhow::Result;

/// List available models
pub async fn list(verbose: bool) -> Result<()> {
    let model_dir = std::env::var("BLAZR_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./models"));

    if !model_dir.exists() {
        println!("No models directory found at: {}", model_dir.display());
        println!("\nSet BLAZR_MODEL_DIR environment variable or create a ./models directory.");
        return Ok(());
    }

    println!("Models in {}:\n", model_dir.display());

    let mut found_any = false;

    for entry in std::fs::read_dir(&model_dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        let is_model = if path.is_dir() {
            path.join("model.safetensors").exists()
                || path.join("config.json").exists()
                || has_gguf_files(&path)
        } else {
            let ext = path.extension().and_then(|e| e.to_str());
            matches!(ext, Some("safetensors") | Some("gguf"))
        };

        if is_model {
            found_any = true;

            if verbose {
                print_model_details(&path, &name)?;
            } else {
                let format = detect_format(&path);
                println!("  {} ({})", name, format);
            }
        }
    }

    if !found_any {
        println!("  No models found.");
        println!("\nTo add models:");
        println!(
            "  - Place SafeTensors or GGUF files in {}",
            model_dir.display()
        );
        println!("  - Use 'blazr pull <repo>' to download from HuggingFace");
    }

    Ok(())
}

fn has_gguf_files(dir: &std::path::Path) -> bool {
    glob::glob(&dir.join("*.gguf").to_string_lossy())
        .map(|g| g.count() > 0)
        .unwrap_or(false)
}

fn detect_format(path: &std::path::Path) -> &'static str {
    if path.is_file() {
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => "SafeTensors",
            Some("gguf") => "GGUF",
            _ => "Unknown",
        }
    } else if path.join("model.safetensors").exists() {
        "SafeTensors"
    } else if has_gguf_files(path) {
        "GGUF"
    } else {
        "Unknown"
    }
}

fn print_model_details(path: &std::path::Path, name: &str) -> Result<()> {
    println!("  {}", name);
    println!("    Path: {}", path.display());
    println!("    Format: {}", detect_format(path));

    // Try to read config for more details
    let config_path = if path.is_dir() {
        path.join("config.json")
    } else {
        path.parent()
            .map(|p| p.join("config.json"))
            .unwrap_or_default()
    };

    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(hidden_size) = config.get("hidden_size").and_then(|v| v.as_u64()) {
                    println!("    Hidden size: {}", hidden_size);
                }
                if let Some(num_layers) = config.get("num_hidden_layers").and_then(|v| v.as_u64()) {
                    println!("    Layers: {}", num_layers);
                }
                if let Some(vocab_size) = config.get("vocab_size").and_then(|v| v.as_u64()) {
                    println!("    Vocab size: {}", vocab_size);
                }
            }
        }
    }

    // File size
    if let Ok(metadata) = std::fs::metadata(path) {
        if metadata.is_file() {
            let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
            if size_mb > 1024.0 {
                println!("    Size: {:.2} GB", size_mb / 1024.0);
            } else {
                println!("    Size: {:.2} MB", size_mb);
            }
        }
    }

    println!();
    Ok(())
}
