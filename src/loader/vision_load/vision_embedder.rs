//! Standalone vision embedder loading (SigLIP / CLIP).
//!
//! Reads an HF-style checkpoint directory: `config.json` + `model.safetensors`
//! (sharded `model.safetensors.index.json` not yet supported here — use the
//! merged single-file variant common for sub-1B vision encoders).

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use boostr::model::config::VisionConfig;
use boostr::model::vision::ImageEmbedder;
use boostr::{DType, Runtime};

/// Minimal subset of HF vision `config.json` fields we care about.
///
/// Supports both top-level layouts (`google/siglip-*`) and nested
/// `vision_config` blocks (`openai/clip-vit-*`, `google/siglip2-*`).
#[derive(Debug, Deserialize)]
struct HfVisionConfig {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    image_size: Option<usize>,
    #[serde(default)]
    patch_size: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    vision_config: Option<Box<HfVisionConfig>>,
}

impl HfVisionConfig {
    /// Flatten: if `vision_config` is nested, pull those fields up.
    fn effective(self) -> Self {
        if let Some(inner) = self.vision_config {
            *inner
        } else {
            self
        }
    }
}

/// Auto-detect `encoder_type` from the model_type string or directory name hints.
fn detect_encoder_type(model_type: Option<&str>, dir_hint: &str) -> String {
    let hint = dir_hint.to_ascii_lowercase();
    if let Some(mt) = model_type {
        let mt = mt.to_ascii_lowercase();
        if mt.contains("siglip") {
            return "siglip".into();
        }
        if mt.contains("clip") {
            return "clip".into();
        }
    }
    if hint.contains("siglip") {
        "siglip".into()
    } else {
        "clip".into()
    }
}

/// Load a standalone SigLIP/CLIP image embedder from an HF-style directory.
///
/// Expected layout:
/// ```text
/// <dir>/
///   config.json
///   model.safetensors
/// ```
pub fn load_vision_embedder_from_dir<R: Runtime<DType = DType>, P: AsRef<Path>>(
    dir: P,
    device: &R::Device,
) -> Result<ImageEmbedder<R>> {
    let dir = dir.as_ref();
    let config_path = dir.join("config.json");
    let config_bytes = std::fs::read(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let hf: HfVisionConfig = serde_json::from_slice(&config_bytes)
        .with_context(|| format!("parsing {}", config_path.display()))?;
    let hf = hf.effective();

    let dir_hint = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    let encoder_type = detect_encoder_type(hf.model_type.as_deref(), &dir_hint);

    let vision_config = VisionConfig {
        encoder_type,
        hidden_size: hf
            .hidden_size
            .ok_or_else(|| anyhow!("config.json missing hidden_size"))?,
        num_layers: hf
            .num_hidden_layers
            .ok_or_else(|| anyhow!("config.json missing num_hidden_layers"))?,
        num_heads: hf
            .num_attention_heads
            .ok_or_else(|| anyhow!("config.json missing num_attention_heads"))?,
        patch_size: hf
            .patch_size
            .ok_or_else(|| anyhow!("config.json missing patch_size"))?,
        image_size: hf
            .image_size
            .ok_or_else(|| anyhow!("config.json missing image_size"))?,
        intermediate_size: hf
            .intermediate_size
            .ok_or_else(|| anyhow!("config.json missing intermediate_size"))?,
        projector_type: "linear".into(),
        projector_depth: 2,
        select_layer: None,
    };

    let safetensors_path = find_safetensors(dir)?;
    let embedder = ImageEmbedder::<R>::from_safetensors(&safetensors_path, &vision_config, device)
        .map_err(|e| {
            anyhow!(
                "loading vision weights from {}: {e}",
                safetensors_path.display()
            )
        })?;
    Ok(embedder)
}

fn find_safetensors(dir: &Path) -> Result<PathBuf> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(single);
    }
    // Fall back to any *.safetensors file in the directory.
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        if entry.path().extension().and_then(|s| s.to_str()) == Some("safetensors") {
            return Ok(entry.path());
        }
    }
    Err(anyhow!(
        "no safetensors file found in {} (expected model.safetensors)",
        dir.display()
    ))
}

/// Parse a `--vision-model NAME=PATH` CLI argument.
pub fn parse_vision_model_arg(arg: &str) -> Result<(String, PathBuf)> {
    let (name, path) = arg
        .split_once('=')
        .ok_or_else(|| anyhow!("--vision-model expects NAME=PATH, got: {arg}"))?;
    let name = name.trim();
    let path = path.trim();
    if name.is_empty() || path.is_empty() {
        return Err(anyhow!(
            "--vision-model NAME and PATH must both be non-empty"
        ));
    }
    Ok((name.to_string(), PathBuf::from(path)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_name_path() {
        let (n, p) = parse_vision_model_arg("siglip=/models/siglip-base").unwrap();
        assert_eq!(n, "siglip");
        assert_eq!(p, PathBuf::from("/models/siglip-base"));
    }

    #[test]
    fn rejects_missing_eq() {
        assert!(parse_vision_model_arg("/models/siglip").is_err());
    }

    #[test]
    fn rejects_empty_parts() {
        assert!(parse_vision_model_arg("=/models/x").is_err());
        assert!(parse_vision_model_arg("name=").is_err());
    }

    #[test]
    fn detects_siglip_from_model_type() {
        assert_eq!(detect_encoder_type(Some("SiglipVisionModel"), ""), "siglip");
        assert_eq!(detect_encoder_type(Some("clip_vision_model"), ""), "clip");
        assert_eq!(detect_encoder_type(None, "siglip-base-patch16"), "siglip");
        assert_eq!(detect_encoder_type(None, "clip-vit-large"), "clip");
    }
}
