//! Parses `--tts-model NAME=PATH`.

use std::path::PathBuf;

use anyhow::{Result, anyhow};

/// Parse `--tts-model NAME=PATH` into `(name, path)`.
pub fn parse_tts_model_arg(arg: &str) -> Result<(String, PathBuf)> {
    let (name, path) = arg
        .split_once('=')
        .ok_or_else(|| anyhow!("--tts-model expects NAME=PATH, got: {arg}"))?;
    let name = name.trim();
    let path = path.trim();
    if name.is_empty() || path.is_empty() {
        return Err(anyhow!("--tts-model NAME and PATH must both be non-empty"));
    }
    Ok((name.to_string(), PathBuf::from(path)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_name_path() {
        let (n, p) = parse_tts_model_arg("kokoro=/models/kokoro").unwrap();
        assert_eq!(n, "kokoro");
        assert_eq!(p, PathBuf::from("/models/kokoro"));
    }

    #[test]
    fn rejects_missing_eq() {
        assert!(parse_tts_model_arg("/p").is_err());
    }

    #[test]
    fn rejects_empty_parts() {
        assert!(parse_tts_model_arg("=/p").is_err());
        assert!(parse_tts_model_arg("n=").is_err());
    }
}
