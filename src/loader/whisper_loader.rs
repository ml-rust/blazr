//! Standalone Whisper loader and CLI argument parsing.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use boostr::{DType, Runtime, TypeConversionOps};
use voxr::WhisperBundle;

/// Load a Whisper bundle from an HF-style directory (config.json +
/// tokenizer.json + model.safetensors), casting every weight to F32.
///
/// The mel front end produces F32 and numr's ops require input and weight to
/// share a dtype, so an F16 checkpoint (`whisper-large-v3` ships one) is cast
/// on the way in. This is the same load every other Whisper consumer in the
/// workspace performs.
pub fn load_whisper_from_dir<R, P>(dir: P, device: &R::Device) -> Result<WhisperBundle<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
    P: AsRef<Path>,
{
    let dir = dir.as_ref();
    let client = R::default_client(device);
    WhisperBundle::from_dir_with_dtype(dir, device, &client, DType::F32)
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
