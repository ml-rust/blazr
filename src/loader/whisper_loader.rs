//! Standalone Whisper loader and CLI argument parsing.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use boostr::model::audio::WhisperBundle;
use boostr::{DType, Runtime};

/// Load a Whisper bundle from an HF-style directory (config.json + tokenizer.json + model.safetensors).
pub fn load_whisper_from_dir<R: Runtime<DType = DType>, P: AsRef<Path>>(
    dir: P,
    device: &R::Device,
) -> Result<WhisperBundle<R>> {
    let dir = dir.as_ref();
    WhisperBundle::from_dir(dir, device)
        .map_err(|e| anyhow!("loading whisper model from {}: {e}", dir.display()))
        .context("whisper bundle load")
}

/// Parse `--asr-model NAME=PATH` into `(name, path)`.
pub fn parse_asr_model_arg(arg: &str) -> Result<(String, PathBuf)> {
    let (name, path) = arg
        .split_once('=')
        .ok_or_else(|| anyhow!("--asr-model expects NAME=PATH, got: {arg}"))?;
    let name = name.trim();
    let path = path.trim();
    if name.is_empty() || path.is_empty() {
        return Err(anyhow!("--asr-model NAME and PATH must both be non-empty"));
    }
    Ok((name.to_string(), PathBuf::from(path)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_name_path() {
        let (n, p) = parse_asr_model_arg("whisper=/models/whisper-base").unwrap();
        assert_eq!(n, "whisper");
        assert_eq!(p, PathBuf::from("/models/whisper-base"));
    }

    #[test]
    fn rejects_missing_eq() {
        assert!(parse_asr_model_arg("/models/whisper").is_err());
    }

    #[test]
    fn rejects_empty_parts() {
        assert!(parse_asr_model_arg("=/p").is_err());
        assert!(parse_asr_model_arg("n=").is_err());
    }
}
