//! Model info command

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use colored::Colorize;

use crate::loader::{detect_model_source, get_gguf_info, ModelFormat};

/// Show model information
pub async fn info(model: String) -> Result<()> {
    let model_path = find_model_path(&model)?;

    eprintln!("{}: {}", "Model".bold(), model);
    eprintln!("{}: {}", "Path".bold(), model_path.display());

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

/// Estimate parameter count from config.json fields
fn estimate_params(config: &serde_json::Value) -> Option<u64> {
    let hidden = config.get("hidden_size")?.as_u64()?;
    let layers = config.get("num_hidden_layers")?.as_u64()?;
    let vocab = config.get("vocab_size")?.as_u64()?;
    let intermediate = config
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(hidden * 4);
    let num_heads = config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(32);
    let head_dim = hidden / num_heads;
    let num_kv_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_heads);

    // Embedding + LM head
    let embed = vocab * hidden * 2;
    // Per-layer: attention (Q, K, V, O) + MLP (gate, up, down) + norms
    let attn = hidden * (num_heads * head_dim) // Q
        + hidden * (num_kv_heads * head_dim)   // K
        + hidden * (num_kv_heads * head_dim)   // V
        + (num_heads * head_dim) * hidden; // O
    let mlp = hidden * intermediate * 3; // gate + up + down (SwiGLU)
    let norms = hidden * 2; // RMSNorm x2 per layer
    let per_layer = attn + mlp + norms;

    Some(embed + layers * per_layer + hidden) // +hidden for final norm
}

fn format_params(count: u64) -> String {
    if count >= 1_000_000_000_000 {
        format!("{:.1}T", count as f64 / 1e12)
    } else if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1e6)
    } else {
        format!("{}", count)
    }
}

fn print_safetensors_info(path: &std::path::Path) -> Result<()> {
    eprintln!("{}: {}", "Format".bold(), "SafeTensors".green());

    // Try to read config.json
    let config_path = if path.is_dir() {
        path.join("config.json")
    } else {
        path.parent()
            .map(|p| p.join("config.json"))
            .unwrap_or_default()
    };

    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;

        eprintln!();
        if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
            eprintln!("  {}: {}", "Architecture".dimmed(), model_type);
        }
        if let Some(params) = estimate_params(&config) {
            eprintln!("  {}: ~{}", "Parameters".dimmed(), format_params(params));
        }
        if let Some(hidden_size) = config.get("hidden_size").and_then(|v| v.as_u64()) {
            eprintln!("  {}: {}", "Hidden size".dimmed(), hidden_size);
        }
        if let Some(num_layers) = config.get("num_hidden_layers").and_then(|v| v.as_u64()) {
            eprintln!("  {}: {}", "Layers".dimmed(), num_layers);
        }
        if let Some(num_heads) = config.get("num_attention_heads").and_then(|v| v.as_u64()) {
            eprintln!("  {}: {}", "Attention heads".dimmed(), num_heads);
        }
        if let Some(num_kv_heads) = config.get("num_key_value_heads").and_then(|v| v.as_u64()) {
            eprintln!("  {}: {}", "KV heads".dimmed(), num_kv_heads);
        }
        if let Some(vocab_size) = config.get("vocab_size").and_then(|v| v.as_u64()) {
            eprintln!("  {}: {}", "Vocab size".dimmed(), vocab_size);
        }
        if let Some(max_pos) = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
        {
            eprintln!("  {}: {}", "Max context".dimmed(), max_pos);
        }
        if let Some(intermediate) = config.get("intermediate_size").and_then(|v| v.as_u64()) {
            eprintln!("  {}: {}", "FFN dim".dimmed(), intermediate);
        }

        // Detect quantization from config
        if let Some(quant_config) = config.get("quantization_config") {
            if let Some(method) = quant_config.get("quant_method").and_then(|v| v.as_str()) {
                let bits = quant_config
                    .get("bits")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4);
                eprintln!(
                    "  {}: {} ({}b)",
                    "Quantization".dimmed(),
                    method.to_uppercase().yellow(),
                    bits
                );
            }
        }
    }

    // Check for license
    let base_dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(path)
    };
    for name in ["LICENSE", "LICENSE.md", "LICENSE.txt"] {
        let license_path = base_dir.join(name);
        if license_path.exists() {
            let content = std::fs::read_to_string(&license_path).unwrap_or_default();
            let first_line = content.lines().next().unwrap_or("(see file)");
            eprintln!("  {}: {}", "License".dimmed(), first_line);
            break;
        }
    }

    // File size (sum all safetensors files)
    let weights_dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(path)
    };
    let total_bytes: u64 = std::fs::read_dir(weights_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .filter_map(|e| e.metadata().ok().map(|m| m.len()))
        .sum();

    if total_bytes > 0 {
        let size_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        if size_gb >= 1.0 {
            eprintln!("\n  {}: {:.2} GB", "Weights size".dimmed(), size_gb);
        } else {
            let size_mb = total_bytes as f64 / (1024.0 * 1024.0);
            eprintln!("\n  {}: {:.1} MB", "Weights size".dimmed(), size_mb);
        }
    }

    Ok(())
}

fn print_gguf_info(path: &std::path::Path) -> Result<()> {
    eprintln!("{}: {}", "Format".bold(), "GGUF".green());

    let info = get_gguf_info(path)?;

    eprintln!();
    eprintln!("  {}: {}", "Architecture".dimmed(), info.architecture);

    if let Some(vocab_size) = info.vocab_size {
        eprintln!("  {}: {}", "Vocab size".dimmed(), vocab_size);
    }
    if let Some(hidden_size) = info.hidden_size {
        eprintln!("  {}: {}", "Hidden size".dimmed(), hidden_size);
    }
    if let Some(num_layers) = info.num_layers {
        eprintln!("  {}: {}", "Layers".dimmed(), num_layers);
    }
    if let Some(num_heads) = info.num_heads {
        eprintln!("  {}: {}", "Attention heads".dimmed(), num_heads);
    }
    if let Some(num_kv_heads) = info.num_kv_heads {
        eprintln!("  {}: {}", "KV heads".dimmed(), num_kv_heads);
    }
    if let Some(context_length) = info.context_length {
        eprintln!("  {}: {}", "Max context".dimmed(), context_length);
    }

    eprintln!(
        "\n  {}: {}",
        "Quantization".dimmed(),
        info.quantization_type.yellow()
    );

    if let Some(size) = info.file_size_bytes {
        let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
        if size_gb >= 1.0 {
            eprintln!("  {}: {:.2} GB", "File size".dimmed(), size_gb);
        } else {
            let size_mb = size as f64 / (1024.0 * 1024.0);
            eprintln!("  {}: {:.1} MB", "File size".dimmed(), size_mb);
        }
    }

    Ok(())
}
