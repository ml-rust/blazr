//! List models command

use std::path::PathBuf;

use anyhow::Result;
use colored::Colorize;

use crate::loader::{detect_model_source, get_gguf_info, ModelFormat};

/// List available models
pub async fn list(verbose: bool) -> Result<()> {
    let model_dir = std::env::var("BLAZR_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./models"));

    if !model_dir.exists() {
        eprintln!(
            "{} No models directory found at: {}",
            "Warning:".yellow(),
            model_dir.display()
        );
        eprintln!(
            "\nSet {} or create a ./models directory.",
            "BLAZR_MODEL_DIR".bold()
        );
        return Ok(());
    }

    eprintln!(
        "{} {}\n",
        "Models in".dimmed(),
        model_dir.display().to_string().bold()
    );

    let mut entries: Vec<_> = std::fs::read_dir(&model_dir)?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut found_any = false;

    for entry in entries {
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
            print_model_entry(&path, &name, verbose);
        }
    }

    if !found_any {
        eprintln!("  No models found.\n");
        eprintln!(
            "  Place SafeTensors or GGUF files in {}",
            model_dir.display()
        );
        eprintln!(
            "  Or use {} to download from HuggingFace",
            "blazr pull <repo>".bold()
        );
    }

    Ok(())
}

fn has_gguf_files(dir: &std::path::Path) -> bool {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .any(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
}

fn print_model_entry(path: &std::path::Path, name: &str, verbose: bool) {
    // Detect format
    let format = detect_format(path);
    let format_str = match format {
        ModelFormat::SafeTensors => "SafeTensors".green(),
        ModelFormat::Gguf => "GGUF".cyan(),
    };

    // Gather metadata
    let size = total_model_size(path);
    let size_str = format_size(size);

    match format {
        ModelFormat::Gguf => print_gguf_entry(path, name, &format_str, &size_str, verbose),
        ModelFormat::SafeTensors => {
            print_safetensors_entry(path, name, &format_str, &size_str, verbose)
        }
    }
}

fn print_gguf_entry(
    path: &std::path::Path,
    name: &str,
    format_str: &colored::ColoredString,
    size_str: &str,
    verbose: bool,
) {
    // Try to get GGUF metadata
    let gguf_path = if path.is_dir() {
        // Find first .gguf file in directory
        std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .find(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
            .map(|e| e.path())
    } else {
        Some(path.to_path_buf())
    };

    let info = gguf_path.and_then(|p| get_gguf_info(&p).ok());

    if let Some(ref info) = info {
        eprintln!(
            "  {}  {}  {}  {}  {}",
            name.bold(),
            format_str,
            info.quantization_type.yellow(),
            info.architecture.dimmed(),
            size_str.dimmed(),
        );
        if verbose {
            print_gguf_verbose(info);
        }
    } else {
        eprintln!("  {}  {}  {}", name.bold(), format_str, size_str.dimmed(),);
    }
}

fn print_gguf_verbose(info: &crate::loader::GgufInfo) {
    if let Some(layers) = info.num_layers {
        eprint!("         {}: {}", "layers".dimmed(), layers);
    }
    if let Some(hidden) = info.hidden_size {
        eprint!("  {}: {}", "hidden".dimmed(), hidden);
    }
    if let Some(heads) = info.num_heads {
        eprint!("  {}: {}", "heads".dimmed(), heads);
    }
    if let Some(ctx) = info.context_length {
        eprint!("  {}: {}", "ctx".dimmed(), ctx);
    }
    eprintln!();
}

fn print_safetensors_entry(
    path: &std::path::Path,
    name: &str,
    format_str: &colored::ColoredString,
    size_str: &str,
    verbose: bool,
) {
    let config_path = if path.is_dir() {
        path.join("config.json")
    } else {
        path.parent()
            .map(|p| p.join("config.json"))
            .unwrap_or_default()
    };

    let config = config_path
        .exists()
        .then(|| std::fs::read_to_string(&config_path).ok())
        .flatten()
        .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok());

    let family = config
        .as_ref()
        .and_then(|c| c.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let kind = detect_model_kind(config.as_ref(), path);
    let kind_tag = match kind {
        ModelKind::Llm => "LLM".blue(),
        ModelKind::Vision => "VISION".magenta(),
        ModelKind::Asr => "ASR".cyan(),
        ModelKind::Tts => "TTS".yellow(),
        ModelKind::Encoder => "ENCODER".bright_blue(),
    };

    // Detect quantization
    let quant = config
        .as_ref()
        .and_then(|c| c.get("quantization_config"))
        .and_then(|qc| {
            let method = qc.get("quant_method")?.as_str()?;
            let bits = qc.get("bits").and_then(|v| v.as_u64()).unwrap_or(4);
            Some(format!("{} {}b", method.to_uppercase(), bits))
        });

    if let Some(ref q) = quant {
        eprintln!(
            "  {}  {}  {}  {}  {}  {}",
            name.bold(),
            format_str,
            kind_tag,
            q.yellow(),
            family.dimmed(),
            size_str.dimmed(),
        );
    } else {
        eprintln!(
            "  {}  {}  {}  {}  {}",
            name.bold(),
            format_str,
            kind_tag,
            family.dimmed(),
            size_str.dimmed(),
        );
    }

    if verbose {
        if let Some(ref c) = config {
            print_safetensors_verbose(c);
        }
    }
}

fn print_safetensors_verbose(config: &serde_json::Value) {
    let mut parts = Vec::new();
    if let Some(v) = config.get("num_hidden_layers").and_then(|v| v.as_u64()) {
        parts.push(format!("{}: {}", "layers".dimmed(), v));
    }
    if let Some(v) = config.get("hidden_size").and_then(|v| v.as_u64()) {
        parts.push(format!("{}: {}", "hidden".dimmed(), v));
    }
    if let Some(v) = config.get("num_attention_heads").and_then(|v| v.as_u64()) {
        parts.push(format!("{}: {}", "heads".dimmed(), v));
    }
    if let Some(v) = config.get("vocab_size").and_then(|v| v.as_u64()) {
        parts.push(format!("{}: {}", "vocab".dimmed(), v));
    }
    if !parts.is_empty() {
        eprintln!("         {}", parts.join("  "));
    }
}

/// Classify a safetensors checkpoint into a high-level model kind so
/// `blazr list` can distinguish LLMs from vision embedders / ASR / TTS.
#[derive(Debug, Clone, Copy)]
enum ModelKind {
    Llm,
    Vision,
    Asr,
    Tts,
    Encoder,
}

fn detect_model_kind(config: Option<&serde_json::Value>, path: &std::path::Path) -> ModelKind {
    if let Some(cfg) = config {
        let model_type = cfg.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
        let archs: Vec<&str> = cfg
            .get("architectures")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let mt = model_type.to_ascii_lowercase();
        let archs_lower: Vec<String> = archs.iter().map(|s| s.to_ascii_lowercase()).collect();

        if mt.contains("siglip") || mt.contains("clip_vision") || mt.contains("vit") {
            return ModelKind::Vision;
        }
        if mt.contains("whisper") || archs_lower.iter().any(|a| a.contains("whisper")) {
            return ModelKind::Asr;
        }
        // StyleTTS2 / Kokoro / Piper / VITS-style
        if mt.contains("tts")
            || archs_lower
                .iter()
                .any(|a| a.contains("styletts") || a.contains("tts"))
        {
            return ModelKind::Tts;
        }
        if mt == "bert" || archs_lower.iter().any(|a| a.contains("bertmodel")) {
            return ModelKind::Encoder;
        }
    }
    // Heuristic: if the directory name contains a known tag and no explicit
    // config hints exist, fall back to that.
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if name.contains("siglip") || name.contains("clip-vit") {
        return ModelKind::Vision;
    }
    if name.contains("whisper") {
        return ModelKind::Asr;
    }
    if name.contains("kokoro") || name.contains("styletts") || name.contains("piper") {
        return ModelKind::Tts;
    }
    ModelKind::Llm
}

fn detect_format(path: &std::path::Path) -> ModelFormat {
    if let Ok(source) = detect_model_source(path) {
        return source.format;
    }
    if path.is_file() {
        match path.extension().and_then(|e| e.to_str()) {
            Some("gguf") => ModelFormat::Gguf,
            _ => ModelFormat::SafeTensors,
        }
    } else if has_gguf_files(path) {
        ModelFormat::Gguf
    } else {
        ModelFormat::SafeTensors
    }
}

fn total_model_size(path: &std::path::Path) -> u64 {
    if path.is_file() {
        return std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    // Sum all weight files in directory
    std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "safetensors" || ext == "gguf" || ext == "bin")
        })
        .filter_map(|e| e.metadata().ok().map(|m| m.len()))
        .sum()
}

fn format_size(bytes: u64) -> String {
    if bytes == 0 {
        return String::new();
    }
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if gb >= 1.0 {
        format!("{:.1} GB", gb)
    } else {
        let mb = bytes as f64 / (1024.0 * 1024.0);
        format!("{:.0} MB", mb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(json: &str) -> Option<serde_json::Value> {
        serde_json::from_str(json).ok()
    }

    #[test]
    fn kind_detects_siglip() {
        let c = cfg(r#"{"model_type": "siglip_vision_model"}"#);
        let k = detect_model_kind(c.as_ref(), std::path::Path::new("x"));
        assert!(matches!(k, ModelKind::Vision));
    }

    #[test]
    fn kind_detects_clip_vit() {
        let c = cfg(r#"{"model_type": "clip_vision_model"}"#);
        let k = detect_model_kind(c.as_ref(), std::path::Path::new("x"));
        assert!(matches!(k, ModelKind::Vision));
    }

    #[test]
    fn kind_detects_whisper_from_arch() {
        let c = cfg(r#"{"architectures": ["WhisperForConditionalGeneration"]}"#);
        let k = detect_model_kind(c.as_ref(), std::path::Path::new("x"));
        assert!(matches!(k, ModelKind::Asr));
    }

    #[test]
    fn kind_detects_bert_encoder() {
        let c = cfg(r#"{"model_type": "bert"}"#);
        let k = detect_model_kind(c.as_ref(), std::path::Path::new("x"));
        assert!(matches!(k, ModelKind::Encoder));
    }

    #[test]
    fn kind_falls_back_to_dir_name() {
        let k = detect_model_kind(None, std::path::Path::new("/models/siglip-base-patch16"));
        assert!(matches!(k, ModelKind::Vision));
        let k = detect_model_kind(None, std::path::Path::new("/models/whisper-base"));
        assert!(matches!(k, ModelKind::Asr));
        let k = detect_model_kind(None, std::path::Path::new("/models/kokoro-82m"));
        assert!(matches!(k, ModelKind::Tts));
    }

    #[test]
    fn kind_defaults_to_llm() {
        let c = cfg(r#"{"model_type": "llama"}"#);
        let k = detect_model_kind(c.as_ref(), std::path::Path::new("/models/llama-3"));
        assert!(matches!(k, ModelKind::Llm));
    }
}
