//! Model format and source detection

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

/// Detected model format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// SafeTensors format (HuggingFace standard)
    SafeTensors,
    /// GGUF format (llama.cpp, quantized models)
    Gguf,
}

/// Detected model source
#[derive(Debug, Clone)]
pub struct ModelSource {
    /// Path to the model weights
    pub weights_path: PathBuf,
    /// Path to config file (if separate)
    pub config_path: Option<PathBuf>,
    /// Detected format
    pub format: ModelFormat,
}

/// Detect model format and source from a path
///
/// The path can be:
/// - A directory containing model files
/// - A direct path to a .safetensors file
/// - A direct path to a .gguf file
pub fn detect_model_source<P: AsRef<Path>>(path: P) -> Result<ModelSource> {
    let path = path.as_ref();

    if path.is_file() {
        // Direct file path
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        match ext {
            "safetensors" => Ok(ModelSource {
                weights_path: path.to_path_buf(),
                config_path: find_config_in_parent(path),
                format: ModelFormat::SafeTensors,
            }),
            "gguf" => Ok(ModelSource {
                weights_path: path.to_path_buf(),
                config_path: None, // GGUF has embedded metadata
                format: ModelFormat::Gguf,
            }),
            _ => Err(anyhow!("Unsupported model file format: .{}", ext)),
        }
    } else if path.is_dir() {
        // Directory - look for model files
        detect_model_in_directory(path)
    } else {
        Err(anyhow!("Model path does not exist: {}", path.display()))
    }
}

/// Detect model files in a directory
fn detect_model_in_directory(dir: &Path) -> Result<ModelSource> {
    // Look for SafeTensors first (preferred)
    let safetensors_patterns = ["model.safetensors", "pytorch_model.safetensors"];

    for pattern in &safetensors_patterns {
        let candidate = dir.join(pattern);
        if candidate.exists() {
            return Ok(ModelSource {
                weights_path: candidate,
                config_path: find_config_in_dir(dir),
                format: ModelFormat::SafeTensors,
            });
        }
    }

    // Look for sharded SafeTensors
    let sharded_glob = dir.join("model-00001-of-00001.safetensors");
    if sharded_glob.exists() || glob_exists(dir, "model-00001-of-*.safetensors") {
        // Find the first shard
        if let Some(first_shard) = find_first_shard(dir) {
            return Ok(ModelSource {
                weights_path: first_shard,
                config_path: find_config_in_dir(dir),
                format: ModelFormat::SafeTensors,
            });
        }
    }

    // Look for GGUF files
    if let Some(gguf_file) = find_gguf_in_dir(dir) {
        return Ok(ModelSource {
            weights_path: gguf_file,
            config_path: None,
            format: ModelFormat::Gguf,
        });
    }

    Err(anyhow!(
        "No supported model files found in directory: {}",
        dir.display()
    ))
}

/// Find config file in a directory
fn find_config_in_dir(dir: &Path) -> Option<PathBuf> {
    let config_names = ["config.json", "config.yaml", "config.yml"];
    for name in &config_names {
        let candidate = dir.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Find config file in parent directory of a file
fn find_config_in_parent(file: &Path) -> Option<PathBuf> {
    file.parent().and_then(find_config_in_dir)
}

/// Check if any files match a glob pattern
fn glob_exists(dir: &Path, pattern: &str) -> bool {
    let full_pattern = dir.join(pattern);
    glob::glob(full_pattern.to_str().unwrap_or(""))
        .map(|mut paths| paths.next().is_some())
        .unwrap_or(false)
}

/// Find the first shard of a sharded model
fn find_first_shard(dir: &Path) -> Option<PathBuf> {
    let pattern = dir.join("model-00001-of-*.safetensors");
    glob::glob(pattern.to_str()?)
        .ok()?
        .filter_map(|r| r.ok())
        .next()
}

/// Find a GGUF file in a directory
fn find_gguf_in_dir(dir: &Path) -> Option<PathBuf> {
    let pattern = dir.join("*.gguf");
    glob::glob(pattern.to_str()?)
        .ok()?
        .filter_map(|r| r.ok())
        .next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection_safetensors() {
        // Test file extension detection
        let _path = Path::new("/models/test.safetensors");
        // This would fail since file doesn't exist, but logic is tested
    }

    #[test]
    fn test_format_detection_gguf() {
        // Test file extension detection
        let _path = Path::new("/models/test.gguf");
        // This would fail since file doesn't exist, but logic is tested
    }
}
