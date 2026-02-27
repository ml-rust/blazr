//! Model info command

use std::path::PathBuf;

use anyhow::{anyhow, Result};

use crate::loader::{detect_model_source, get_gguf_info, ModelFormat};

/// Show model information
pub async fn info(model: String) -> Result<()> {
    let model_path = find_model_path(&model)?;

    println!("Model: {}\n", model);
    println!("Path: {}", model_path.display());

    let source = detect_model_source(&model_path)?;

    match source.format {
        ModelFormat::SafeTensors => {
            print_safetensors_info(&model_path)?;
        }
        ModelFormat::Gguf => {
            print_gguf_info(&model_path)?;
        }
    }

    Ok(())
}

fn find_model_path(model: &str) -> Result<PathBuf> {
    // Try direct path
    let direct = PathBuf::from(model);
    if direct.exists() {
        return Ok(direct);
    }

    // Try in model directory
    let model_dir = std::env::var("BLAZR_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./models"));

    let in_dir = model_dir.join(model);
    if in_dir.exists() {
        return Ok(in_dir);
    }

    Err(anyhow!("Model not found: {}", model))
}

fn print_safetensors_info(path: &std::path::Path) -> Result<()> {
    println!("Format: SafeTensors\n");

    // Try to read config.json
    let config_path = if path.is_dir() {
        path.join("config.json")
    } else {
        path.parent()
            .map(|p| p.join("config.json"))
            .unwrap_or_default()
    };

    if config_path.exists() {
        println!("Configuration:");

        let content = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;

        if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
            println!("  Architecture: {}", model_type);
        }
        if let Some(hidden_size) = config.get("hidden_size").and_then(|v| v.as_u64()) {
            println!("  Hidden size: {}", hidden_size);
        }
        if let Some(num_layers) = config.get("num_hidden_layers").and_then(|v| v.as_u64()) {
            println!("  Layers: {}", num_layers);
        }
        if let Some(num_heads) = config.get("num_attention_heads").and_then(|v| v.as_u64()) {
            println!("  Attention heads: {}", num_heads);
        }
        if let Some(num_kv_heads) = config.get("num_key_value_heads").and_then(|v| v.as_u64()) {
            println!("  KV heads: {}", num_kv_heads);
        }
        if let Some(vocab_size) = config.get("vocab_size").and_then(|v| v.as_u64()) {
            println!("  Vocab size: {}", vocab_size);
        }
        if let Some(max_pos) = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
        {
            println!("  Max context: {}", max_pos);
        }
        if let Some(intermediate) = config.get("intermediate_size").and_then(|v| v.as_u64()) {
            println!("  FFN dim: {}", intermediate);
        }
    }

    // File size
    let weights_path = if path.is_dir() {
        path.join("model.safetensors")
    } else {
        path.to_path_buf()
    };

    if weights_path.exists() {
        if let Ok(metadata) = std::fs::metadata(&weights_path) {
            let size_gb = metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0);
            println!("\nWeights size: {:.2} GB", size_gb);
        }
    }

    Ok(())
}

fn print_gguf_info(path: &std::path::Path) -> Result<()> {
    println!("Format: GGUF\n");

    let info = get_gguf_info(path)?;

    println!("Configuration:");
    println!("  Architecture: {}", info.architecture);

    if let Some(vocab_size) = info.vocab_size {
        println!("  Vocab size: {}", vocab_size);
    }
    if let Some(hidden_size) = info.hidden_size {
        println!("  Hidden size: {}", hidden_size);
    }
    if let Some(num_layers) = info.num_layers {
        println!("  Layers: {}", num_layers);
    }
    if let Some(num_heads) = info.num_heads {
        println!("  Attention heads: {}", num_heads);
    }
    if let Some(num_kv_heads) = info.num_kv_heads {
        println!("  KV heads: {}", num_kv_heads);
    }
    if let Some(context_length) = info.context_length {
        println!("  Max context: {}", context_length);
    }

    println!("\nQuantization: {}", info.quantization_type);

    if let Some(size) = info.file_size_bytes {
        let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
        if size_gb >= 1.0 {
            println!("File size: {:.2} GB", size_gb);
        } else {
            let size_mb = size as f64 / (1024.0 * 1024.0);
            println!("File size: {:.2} MB", size_mb);
        }
    }

    Ok(())
}
