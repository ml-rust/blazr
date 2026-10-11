//! The user config file `~/.blazr/config.yaml`.
//!
//! One file holds the user keys and the `server`, `inference` and
//! `generation` sections. `resolve.rs` applies flags and environment
//! variables over it.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{DeviceChoice, GenerationConfig, InferenceConfig, InferenceOverlay, ServerConfig};

/// The file as written. A section key the file omits holds its default.
#[derive(Debug, Default, Serialize, Deserialize)]
struct RawFile {
    #[serde(default)]
    default_model: Option<String>,
    #[serde(default)]
    device: Option<DeviceChoice>,
    #[serde(default)]
    num_ctx: Option<usize>,
    #[serde(default)]
    model_dir: Option<PathBuf>,
    /// Older name of `server.port`.
    #[serde(default)]
    port: Option<u16>,
    /// Older name of `generation.temperature`.
    #[serde(default)]
    temperature: Option<f32>,
    /// Older name of `generation.max_tokens`.
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    server: ServerConfig,
    #[serde(default)]
    inference: InferenceConfig,
    #[serde(default)]
    generation: GenerationConfig,
}

/// The parsed user config file.
///
/// `server` and `generation` hold the built-in default for each key the
/// file omits. `inference` holds only the keys the file sets.
#[derive(Debug, Clone, Default)]
pub struct FileConfig {
    /// Model to load when a command names none.
    pub default_model: Option<String>,
    /// Device to run on.
    pub device: Option<DeviceChoice>,
    /// Initial KV cache size in tokens.
    pub num_ctx: Option<usize>,
    /// Model search directory.
    pub model_dir: Option<PathBuf>,
    /// HTTP server settings for `blazr serve`.
    pub server: ServerConfig,
    /// Inference keys written over every loaded model.
    pub inference: InferenceOverlay,
    /// Sampling defaults for commands and for requests that omit a value.
    pub generation: GenerationConfig,
}

/// A parsed file plus the keys it holds that no struct reads.
#[derive(Debug)]
pub struct ParsedFile {
    pub config: FileConfig,
    /// Dotted paths of unknown keys, such as `server.prot`.
    pub unknown_keys: Vec<String>,
}

impl FileConfig {
    /// The file path: `~/.blazr/config.yaml`.
    pub fn path() -> Option<PathBuf> {
        dirs::home_dir().map(|home| home.join(".blazr").join("config.yaml"))
    }

    /// Load the file at [`FileConfig::path`]. A missing file gives defaults.
    ///
    /// # Errors
    ///
    /// An error when the file exists and does not read or parse.
    pub fn load() -> Result<Self> {
        match Self::path() {
            Some(path) => Self::load_from(&path),
            None => Ok(Self::default()),
        }
    }

    /// Load `path` and log one warning per unknown key.
    ///
    /// # Errors
    ///
    /// An error when the file exists and does not read or parse.
    pub fn load_from(path: &Path) -> Result<Self> {
        let parsed = Self::read(path)?;
        for key in &parsed.unknown_keys {
            tracing::warn!(
                "{}: unknown key '{}' is ignored. Check its spelling in docs/configuration.md",
                path.display(),
                key
            );
        }
        Ok(parsed.config)
    }

    /// Read and parse `path`. A missing file gives defaults.
    ///
    /// # Errors
    ///
    /// An error when the file exists and does not read or parse.
    pub fn read(path: &Path) -> Result<ParsedFile> {
        if !path.exists() {
            return Ok(ParsedFile {
                config: Self::default(),
                unknown_keys: Vec::new(),
            });
        }
        let text = std::fs::read_to_string(path).map_err(|e| {
            anyhow!(
                "{}: the file did not read: {e}. Check its permissions",
                path.display()
            )
        })?;
        Self::parse(&text, path)
    }

    /// Parse the text of the file. `path` names the file in errors.
    ///
    /// # Errors
    ///
    /// An error naming the file and line for bad YAML or a bad value.
    pub fn parse(text: &str, path: &Path) -> Result<ParsedFile> {
        let tree: Value = serde_saphyr::from_str(text).map_err(|e| yaml_error(path, &e))?;
        let tree = match tree {
            Value::Null => {
                return Ok(ParsedFile {
                    config: Self::default(),
                    unknown_keys: Vec::new(),
                })
            }
            Value::Object(map) => map,
            _ => {
                return Err(anyhow!(
                    "{} line 1: the file is not a map of keys. Write one `key: value` per line",
                    path.display()
                ))
            }
        };
        let raw: RawFile = serde_saphyr::from_str(text).map_err(|e| yaml_error(path, &e))?;

        let mut unknown_keys = Vec::new();
        if let Value::Object(known) = serde_json::to_value(&raw)? {
            collect_unknown(&tree, &known, "", &mut unknown_keys);
        }

        let RawFile {
            default_model,
            device,
            num_ctx,
            model_dir,
            port,
            temperature,
            max_tokens,
            mut server,
            inference,
            mut generation,
        } = raw;
        if let Some(port) = legacy(&tree, "port", "server", port, path)? {
            server.port = port;
        }
        if let Some(t) = legacy(&tree, "temperature", "generation", temperature, path)? {
            generation.temperature = t;
        }
        if let Some(m) = legacy(&tree, "max_tokens", "generation", max_tokens, path)? {
            generation.max_tokens = m;
        }

        let inference_keys: BTreeSet<String> = match tree.get("inference") {
            Some(Value::Object(section)) => section.keys().cloned().collect(),
            _ => BTreeSet::new(),
        };
        let inference = InferenceOverlay::from_section(&inference, &inference_keys)?;

        Ok(ParsedFile {
            config: Self {
                default_model,
                device,
                num_ctx,
                model_dir,
                server,
                inference,
                generation,
            },
            unknown_keys,
        })
    }
}

/// A top-level key older files used in place of `section.key`.
///
/// Both set is an error. The section key alone needs nothing.
fn legacy<T>(
    tree: &Map<String, Value>,
    key: &str,
    section: &str,
    value: Option<T>,
    path: &Path,
) -> Result<Option<T>> {
    let in_section = tree
        .get(section)
        .and_then(Value::as_object)
        .is_some_and(|map| map.contains_key(key));
    if value.is_some() && in_section {
        return Err(anyhow!(
            "{}: both {key} and {section}.{key} are set. Remove the top-level {key}",
            path.display()
        ));
    }
    Ok(value)
}

/// Push the dotted path of each key in `tree` that `known` lacks.
fn collect_unknown(
    tree: &Map<String, Value>,
    known: &Map<String, Value>,
    prefix: &str,
    out: &mut Vec<String>,
) {
    for (key, value) in tree {
        let dotted = format!("{prefix}{key}");
        match (known.get(key), value) {
            (None, _) => out.push(dotted),
            (Some(Value::Object(inner_known)), Value::Object(inner)) => {
                collect_unknown(inner, inner_known, &format!("{dotted}."), out);
            }
            _ => {}
        }
    }
}

/// A YAML error that names the file, the line and the next action.
fn yaml_error(path: &Path, error: &serde_saphyr::Error) -> anyhow::Error {
    let line = error
        .location()
        .map_or_else(|| "?".to_string(), |at| at.line().to_string());
    anyhow!(
        "{} line {line}: {error}\nFix that line, or move the file away to run on built-in defaults",
        path.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReasoningFormat;

    fn parse(text: &str) -> Result<ParsedFile> {
        FileConfig::parse(text, Path::new("/home/u/.blazr/config.yaml"))
    }

    #[test]
    fn empty_and_comment_only_files_give_defaults() {
        for text in ["", "\n", "# nothing set\n"] {
            let parsed = parse(text).unwrap();
            assert!(parsed.config.default_model.is_none());
            assert_eq!(parsed.config.server.port, 8080);
            assert!(parsed.unknown_keys.is_empty());
        }
    }

    #[test]
    fn reads_user_keys_and_every_section() {
        let text = "\
default_model: mistral-7b
device: cuda:1
num_ctx: 4096
model_dir: /srv/models
server:
  host: 127.0.0.1
  max_concurrent_requests: 4
  reasoning_format: none
  slo:
    ttft_p95_ms: 500
inference:
  max_batch_size: 8
  tensor_parallel_size: 2
generation:
  top_p: 0.9
";
        let parsed = parse(text).unwrap();
        let config = parsed.config;
        assert!(parsed.unknown_keys.is_empty(), "{:?}", parsed.unknown_keys);
        assert_eq!(config.default_model.as_deref(), Some("mistral-7b"));
        assert_eq!(config.device, Some(DeviceChoice::Cuda(1)));
        assert_eq!(config.num_ctx, Some(4096));
        assert_eq!(config.model_dir, Some(PathBuf::from("/srv/models")));
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.max_concurrent_requests, 4);
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.server.reasoning_format, ReasoningFormat::None);
        assert_eq!(
            config.server.slo.as_ref().and_then(|s| s.ttft_p95_ms),
            Some(500)
        );
        assert_eq!(config.inference.tensor_parallel_size(), Some(2));
        assert_eq!(
            config.inference.keys().collect::<Vec<_>>(),
            ["max_batch_size", "tensor_parallel_size"]
        );
        assert_eq!(config.generation.top_p, 0.9);
        assert_eq!(config.generation.max_tokens, 2048);
    }

    #[test]
    fn old_top_level_keys_still_parse() {
        let text = "\
default_model: mistral-7b
device: cuda
num_ctx: 4096
model_dir: /srv/models
port: 9090
temperature: 0.8
max_tokens: 1024
";
        let parsed = parse(text).unwrap();
        assert!(parsed.unknown_keys.is_empty());
        assert_eq!(parsed.config.server.port, 9090);
        assert_eq!(parsed.config.generation.temperature, 0.8);
        assert_eq!(parsed.config.generation.max_tokens, 1024);
        assert_eq!(parsed.config.device, Some(DeviceChoice::Cuda(0)));
    }

    #[test]
    fn old_key_and_section_key_together_is_an_error() {
        let err = parse("port: 9090\nserver:\n  port: 9091\n").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("config.yaml"), "{message}");
        assert!(message.contains("server.port"), "{message}");
    }

    #[test]
    fn unknown_keys_are_named_with_their_path() {
        let text = "\
prot: 1
server:
  hots: x
  slo:
    ttft_p51_ms: 3
inference:
  flash_attention: true
  dtype: f32
generation:
  logit_bias:
    \"42\": -1.5
";
        let parsed = parse(text).unwrap();
        assert_eq!(
            parsed.unknown_keys,
            [
                "prot",
                "server.hots",
                "server.slo.ttft_p51_ms",
                "inference.flash_attention",
                "inference.dtype",
            ]
        );
        assert_eq!(parsed.config.generation.logit_bias.get(&42), Some(&-1.5));
        assert_eq!(parsed.config.inference.keys().count(), 0);
    }

    #[test]
    fn a_bad_value_names_the_file_and_line() {
        let err = parse("server:\n  port: not-a-port\n").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("config.yaml line 2"), "{message}");
        assert!(message.contains("Fix that line"), "{message}");
    }

    #[test]
    fn a_bad_device_names_the_file_and_line() {
        let err = parse("num_ctx: 1\ndevice: gpu\n").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("config.yaml line 2"), "{message}");
        assert!(message.contains("cuda:N"), "{message}");
    }

    #[test]
    fn broken_yaml_names_the_file_and_line() {
        let err = parse("server:\n  port: [1\n").unwrap_err();
        assert!(err.to_string().contains("config.yaml line"), "{err}");
    }

    #[test]
    fn a_list_at_the_top_is_an_error() {
        let err = parse("- a\n- b\n").unwrap_err();
        assert!(err.to_string().contains("not a map"), "{err}");
    }
}
