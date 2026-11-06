//! What a model on disk is, and whether the text server can load it.
//!
//! The checks read `config.json`, file names, and the first keys of a GGUF or
//! SafeTensors header. They read no tensor, so a caller runs them before it
//! frees device memory.

use std::fmt;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{anyhow, Result};
use boostr::format::SafeTensorsLoader;
use boostr::model::registry::unsupported_model_refusal;
use serde_json::Value;

use super::detect::{detect_model_source, ModelFormat};
use super::formats::gguf_config::SUPPORTED_GGUF_ARCHS;
use super::formats::safetensors::detect_architecture_from_loader;
use crate::tokenizer::vocab_name_for_size;

/// The longest GGUF key or architecture name this reader takes.
const MAX_GGUF_KEY: usize = 4096;

/// Weight files the speech and vision bundles ship, beyond the text formats.
const WEIGHT_EXTENSIONS: &[&str] = &["safetensors", "gguf", "pth", "pt", "bin", "jit", "onnx"];

/// `model_type` values boostr routes to a load that has no tensor name map.
const NO_NAME_MAP: &[&str] = &["internlm2"];

/// A high-level classification of a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// A text model the executor runs.
    Llm,
    /// A vision embedder or a vision tower.
    Vision,
    /// A speech recognition model.
    Asr,
    /// A text to speech model.
    Tts,
    /// A text encoder with no text generation.
    Encoder,
    /// An audio codec.
    Codec,
    /// A voice activity detector.
    Vad,
}

impl ModelKind {
    /// The tag `blazr list` prints.
    pub fn label(self) -> &'static str {
        match self {
            Self::Llm => "LLM",
            Self::Vision => "VISION",
            Self::Asr => "ASR",
            Self::Tts => "TTS",
            Self::Encoder => "ENCODER",
            Self::Codec => "CODEC",
            Self::Vad => "VAD",
        }
    }

    /// What a caller does with a model of this kind instead of a text request.
    fn next_action(self) -> &'static str {
        match self {
            Self::Llm => "Request a text model",
            Self::Vision => {
                "Start the server with --vision-model, or name a vision tower with --mmproj"
            }
            Self::Asr => {
                "Start the server with --asr-model and send audio to /v1/audio/transcriptions"
            }
            Self::Tts => "Start the server with --tts-model and send text to /v1/audio/speech",
            Self::Encoder => "Request a GGUF embedding model from /v1/models",
            Self::Codec | Self::Vad => "Request a text model from /v1/models",
        }
    }
}

/// Classify a checkpoint from its `config.json`. The name of `path` decides
/// only when the config states neither `model_type` nor `architectures`.
pub fn detect_model_kind(config: Option<&Value>, path: &Path) -> ModelKind {
    let model_type = config
        .and_then(|c| c.get("model_type"))
        .and_then(Value::as_str);
    let archs: Vec<String> = config
        .and_then(|c| c.get("architectures"))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_ascii_lowercase)
                .collect()
        })
        .unwrap_or_default();
    if model_type.is_some() || !archs.is_empty() {
        return kind_of_config(&model_type.unwrap_or("").to_ascii_lowercase(), &archs);
    }
    kind_of_name(path)
}

/// The kind a `model_type` and lower-case `architectures` state.
fn kind_of_config(model_type: &str, archs: &[String]) -> ModelKind {
    let arch_has = |tag: &str| archs.iter().any(|a| a.contains(tag));
    // `vits` holds `vit`, so the speech checks come before the vision check.
    if model_type.contains("whisper") || arch_has("whisper") {
        return ModelKind::Asr;
    }
    if model_type.contains("tts")
        || model_type == "vits"
        || model_type.contains("voxcpm")
        || arch_has("styletts")
        || arch_has("tts")
        || arch_has("vits")
    {
        return ModelKind::Tts;
    }
    if model_type.contains("siglip") || model_type.contains("clip") || model_type.contains("vit") {
        return ModelKind::Vision;
    }
    if model_type == "neucodec" || arch_has("neucodec") {
        return ModelKind::Codec;
    }
    if model_type == "bert" || arch_has("bertmodel") {
        return ModelKind::Encoder;
    }
    ModelKind::Llm
}

/// The kind the file or directory name of `path` states.
fn kind_of_name(path: &Path) -> ModelKind {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if name.contains("siglip") || name.contains("clip-vit") || name.starts_with("mmproj") {
        return ModelKind::Vision;
    }
    if name.contains("whisper") {
        return ModelKind::Asr;
    }
    if name.contains("kokoro") || name.contains("styletts") || name.contains("piper") {
        return ModelKind::Tts;
    }
    if name.contains("neucodec") {
        return ModelKind::Codec;
    }
    if name.contains("silero") {
        return ModelKind::Vad;
    }
    ModelKind::Llm
}

/// The parsed `config.json` of a directory, or of the directory holding a
/// SafeTensors file. `None` for a GGUF file, which carries its own metadata.
fn config_of(path: &Path) -> Option<Value> {
    let dir = if path.is_dir() {
        path
    } else if path.extension().is_some_and(|e| e == "safetensors") {
        path.parent()?
    } else {
        return None;
    };
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// The kind of the model at `path`. A GGUF file is a text model unless its
/// name marks it as a vision tower.
pub fn model_kind_at(path: &Path) -> ModelKind {
    if path.is_file() && path.extension().is_some_and(|e| e == "gguf") {
        return if kind_of_name(path) == ModelKind::Vision {
            ModelKind::Vision
        } else {
            ModelKind::Llm
        };
    }
    detect_model_kind(config_of(path).as_ref(), path)
}

/// Why the text server cannot load a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotLoadable {
    /// The path holds no weights the loader reads. The text is the loader's.
    NoWeights(String),
    /// The path holds a model of another kind.
    WrongKind(ModelKind),
    /// The checkpoint states an architecture the loader refuses.
    Architecture(String),
    /// The checkpoint holds no tokenizer and matches no bundled vocabulary.
    Tokenizer(String),
}

impl fmt::Display for NotLoadable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoWeights(reason) => f.write_str(reason),
            Self::WrongKind(kind) => write!(
                f,
                "its kind is {}, which the text routes do not run. {}",
                kind.label(),
                kind.next_action()
            ),
            Self::Architecture(reason) | Self::Tokenizer(reason) => {
                write!(f, "{reason}. Request another model from /v1/models")
            }
        }
    }
}

impl std::error::Error for NotLoadable {}

/// Why boostr or the name map refuses `model_type`, or `None`.
fn refused_architecture(model_type: &str) -> Option<String> {
    unsupported_model_refusal(model_type).or_else(|| {
        NO_NAME_MAP
            .contains(&model_type)
            .then(|| format!("model type '{model_type}' has no tensor name map"))
    })
}

/// Why the config of a checkpoint with no `tokenizer.json` names no tokenizer:
/// its `vocab_size` matches no bundled vocabulary.
fn unmatched_vocabulary(path: &Path, config: &Value) -> Option<String> {
    let dir = if path.is_dir() { path } else { path.parent()? };
    if dir.join("tokenizer.json").exists() {
        return None;
    }
    let vocab = usize::try_from(config.get("vocab_size")?.as_u64()?).ok()?;
    vocab_name_for_size(vocab).err().map(|e| e.to_string())
}

/// What the start of a GGUF file states, read with no tensor and no token.
#[derive(Debug, PartialEq, Eq)]
enum GgufHead {
    /// The file opens with another magic number.
    NotGguf,
    /// The file ends early, or uses a layout this reader skips.
    Unread,
    /// The keys the loader needs first.
    Read {
        architecture: Option<String>,
        has_vocabulary: bool,
    },
}

/// One little-endian integer of `N` bytes, as a `u64`.
fn read_le<const N: usize>(file: &mut impl Read) -> std::io::Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes[..N])?;
    Ok(u64::from_le_bytes(bytes))
}

/// The bytes of one GGUF scalar of type `kind`. `None` for a string, an array,
/// or an unknown type.
fn gguf_scalar_width(kind: u64) -> Option<i64> {
    match kind {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// Move past one value of GGUF type `kind`, or fail on an unknown type. A
/// scalar array moves in one seek, so a false count in a bad file cannot spin.
fn skip_gguf_value(file: &mut (impl Read + Seek), kind: u64) -> std::io::Result<()> {
    let offset = |n: u64| i64::try_from(n).map_err(std::io::Error::other);
    let width = match kind {
        8 => offset(read_le::<8>(file)?)?,
        9 => {
            let element = read_le::<4>(file)?;
            let count = read_le::<8>(file)?;
            let Some(width) = gguf_scalar_width(element) else {
                for _ in 0..count {
                    skip_gguf_value(file, element)?;
                }
                return Ok(());
            };
            offset(count)?
                .checked_mul(width)
                .ok_or_else(|| std::io::Error::other("GGUF array size overflows"))?
        }
        _ => gguf_scalar_width(kind)
            .ok_or_else(|| std::io::Error::other("unknown GGUF value type"))?,
    };
    file.seek(std::io::SeekFrom::Current(width))?;
    Ok(())
}

/// A UTF-8 name of at most [`MAX_GGUF_KEY`] bytes. `None` for a longer one.
fn read_gguf_name(file: &mut impl Read) -> std::io::Result<Option<Vec<u8>>> {
    let length = usize::try_from(read_le::<8>(file)?).map_err(std::io::Error::other)?;
    if length > MAX_GGUF_KEY {
        return Ok(None);
    }
    let mut name = vec![0u8; length];
    file.read_exact(&mut name)?;
    Ok(Some(name))
}

/// Read keys until `tokenizer.ggml.tokens` is found or the keys end. The token
/// array stays unread. Every writer places `general.architecture` before it.
fn scan_gguf_keys(file: &mut (impl Read + Seek)) -> std::io::Result<GgufHead> {
    if read_le::<4>(file)? < 2 {
        return Ok(GgufHead::Unread);
    }
    read_le::<8>(file)?;
    let keys = read_le::<8>(file)?;
    let mut architecture = None;
    for _ in 0..keys {
        let Some(key) = read_gguf_name(file)? else {
            return Ok(GgufHead::Unread);
        };
        let kind = read_le::<4>(file)?;
        if key == b"tokenizer.ggml.tokens" {
            return Ok(GgufHead::Read {
                architecture,
                has_vocabulary: true,
            });
        }
        if key == b"general.architecture" && kind == 8 {
            let Some(value) = read_gguf_name(file)? else {
                return Ok(GgufHead::Unread);
            };
            architecture = Some(String::from_utf8_lossy(&value).into_owned());
        } else {
            skip_gguf_value(file, kind)?;
        }
    }
    Ok(GgufHead::Read {
        architecture,
        has_vocabulary: false,
    })
}

/// [`GgufHead`] of the file at `path`.
fn read_gguf_head(path: &Path) -> GgufHead {
    let Ok(mut file) = std::fs::File::open(path).map(BufReader::new) else {
        return GgufHead::Unread;
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return GgufHead::Unread;
    }
    if &magic != b"GGUF" {
        return GgufHead::NotGguf;
    }
    scan_gguf_keys(&mut file).unwrap_or(GgufHead::Unread)
}

/// Why the header of a GGUF file refuses a load. `None` when the header reads
/// well or does not read here, which leaves the file to the loader.
fn gguf_refusal(path: &Path) -> Option<NotLoadable> {
    let GgufHead::Read {
        architecture,
        has_vocabulary,
    } = read_gguf_head(path)
    else {
        return None;
    };
    if let Some(architecture) = architecture {
        if !SUPPORTED_GGUF_ARCHS.contains(&architecture.as_str()) {
            return Some(NotLoadable::Architecture(format!(
                "GGUF architecture '{architecture}' is not supported"
            )));
        }
    }
    (!has_vocabulary)
        .then(|| NotLoadable::Tokenizer("the GGUF file holds no tokenizer.ggml.tokens".to_string()))
}

/// Whether `blazr list` and `/v1/models` offer the GGUF file at `path`. A file
/// with another magic number, a refused architecture or no vocabulary is left
/// out. An empty or truncated file stays, and its load reports the error.
pub(crate) fn gguf_file_is_model(path: &Path) -> bool {
    model_kind_at(path) == ModelKind::Llm
        && read_gguf_head(path) != GgufHead::NotGguf
        && gguf_refusal(path).is_none()
}

/// Check that the text server can load `path`, reading no tensor.
///
/// # Errors
///
/// [`NotLoadable`] when `path` holds a model of another kind, states an
/// architecture the loader refuses, names no tokenizer, or holds no weights the
/// loader reads.
pub fn check_loadable(path: &Path) -> Result<(), NotLoadable> {
    let kind = model_kind_at(path);
    if kind != ModelKind::Llm {
        return Err(NotLoadable::WrongKind(kind));
    }
    if let Some(config) = config_of(path) {
        let refused = config
            .get("model_type")
            .and_then(Value::as_str)
            .and_then(refused_architecture);
        if let Some(reason) = refused {
            return Err(NotLoadable::Architecture(reason));
        }
        if let Some(reason) = unmatched_vocabulary(path, &config) {
            return Err(NotLoadable::Tokenizer(reason));
        }
    }
    let source = detect_model_source(path).map_err(|e| NotLoadable::NoWeights(e.to_string()))?;
    match source.format {
        ModelFormat::Gguf => gguf_refusal(&source.weights_path).map_or(Ok(()), Err),
        ModelFormat::SafeTensors => Ok(()),
    }
}

/// Whether the SafeTensors file at `path` holds tensors the architecture
/// detector reads. A file of mel filters or test vectors does not.
pub(crate) fn safetensors_file_is_model(path: &Path) -> Result<()> {
    let loader = SafeTensorsLoader::open(path)
        .map_err(|e| anyhow!("Failed to open SafeTensors {}: {e}", path.display()))?;
    detect_architecture_from_loader(&loader).map(|_| ())
}

/// Whether the directory `dir` holds a weight file of any format.
pub(crate) fn holds_weight_file(dir: &Path) -> bool {
    std::fs::read_dir(dir).into_iter().flatten().any(|entry| {
        entry.ok().is_some_and(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| WEIGHT_EXTENSIONS.contains(&x))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(json: &str) -> Option<Value> {
        serde_json::from_str(json).ok()
    }

    fn kind(json: &str, path: &str) -> ModelKind {
        detect_model_kind(cfg(json).as_ref(), Path::new(path))
    }

    #[test]
    fn kinds_come_from_the_config() {
        assert_eq!(
            kind(r#"{"model_type": "siglip_vision_model"}"#, "x"),
            ModelKind::Vision
        );
        assert_eq!(
            kind(r#"{"model_type": "clip_vision_model"}"#, "x"),
            ModelKind::Vision
        );
        assert_eq!(
            kind(
                r#"{"architectures": ["WhisperForConditionalGeneration"]}"#,
                "x"
            ),
            ModelKind::Asr
        );
        assert_eq!(kind(r#"{"model_type": "bert"}"#, "x"), ModelKind::Encoder);
        assert_eq!(kind(r#"{"model_type": "vits"}"#, "x"), ModelKind::Tts);
        assert_eq!(kind(r#"{"model_type": "neucodec"}"#, "x"), ModelKind::Codec);
    }

    #[test]
    fn a_stated_text_model_wins_over_its_directory_name() {
        assert_eq!(
            kind(r#"{"model_type": "llama"}"#, "/models/llama-3"),
            ModelKind::Llm
        );
        assert_eq!(
            kind(
                r#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"#,
                "/models/whisper-prompt-lm"
            ),
            ModelKind::Llm
        );
    }

    #[test]
    fn with_no_stated_type_the_name_decides() {
        let by_name = |p: &str| detect_model_kind(None, Path::new(p));
        assert_eq!(by_name("/models/siglip-base-patch16"), ModelKind::Vision);
        assert_eq!(by_name("/models/whisper-base"), ModelKind::Asr);
        assert_eq!(by_name("/models/kokoro-82m"), ModelKind::Tts);
        assert_eq!(by_name("/models/silero-vad"), ModelKind::Vad);
        assert_eq!(by_name("/models/llama-3"), ModelKind::Llm);
        // Kokoro's config.json states no type.
        assert_eq!(kind("{}", "/models/Kokoro-82M"), ModelKind::Tts);
    }

    #[test]
    fn check_loadable_names_each_refusal() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path();
        let whisper = root.join("whisper-tiny");
        std::fs::create_dir(&whisper).expect("mkdir");
        std::fs::write(whisper.join("config.json"), r#"{"model_type": "whisper"}"#).expect("write");
        std::fs::write(whisper.join("model.safetensors"), b"").expect("write");
        assert_eq!(
            check_loadable(&whisper),
            Err(NotLoadable::WrongKind(ModelKind::Asr))
        );

        let empty = root.join("config-only");
        std::fs::create_dir(&empty).expect("mkdir");
        std::fs::write(empty.join("config.json"), "{}").expect("write");
        assert!(matches!(
            check_loadable(&empty),
            Err(NotLoadable::NoWeights(_))
        ));

        let mmproj = root.join("mmproj-f16.gguf");
        std::fs::write(&mmproj, b"").expect("write");
        assert_eq!(
            check_loadable(&mmproj),
            Err(NotLoadable::WrongKind(ModelKind::Vision))
        );

        for model_type in ["deepseek_v3", "internlm2"] {
            let refused = root.join(model_type);
            std::fs::create_dir(&refused).expect("mkdir");
            std::fs::write(
                refused.join("config.json"),
                format!(r#"{{"model_type": "{model_type}"}}"#),
            )
            .expect("write");
            std::fs::write(refused.join("model.safetensors"), b"").expect("write");
            assert!(
                matches!(check_loadable(&refused), Err(NotLoadable::Architecture(_))),
                "{model_type}"
            );
        }

        let gguf = root.join("q.gguf");
        std::fs::write(&gguf, b"").expect("write");
        assert_eq!(check_loadable(&gguf), Ok(()));

        // A vocabulary of 111 with no tokenizer.json matches nothing.
        let small = root.join("small-vocab");
        std::fs::create_dir(&small).expect("mkdir");
        std::fs::write(
            small.join("config.json"),
            r#"{"model_type": "llama", "vocab_size": 111}"#,
        )
        .expect("write");
        std::fs::write(small.join("model.safetensors"), b"").expect("write");
        assert!(matches!(
            check_loadable(&small),
            Err(NotLoadable::Tokenizer(_))
        ));
    }

    /// One GGUF header: `keys` in order, each a string value, or a one-entry
    /// string array under `tokenizer.ggml.tokens`.
    fn gguf_with(keys: &[(&str, &str)]) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(keys.len() as u64).to_le_bytes());
        let string = |bytes: &mut Vec<u8>, text: &str| {
            bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        };
        for (key, value) in keys {
            string(&mut bytes, key);
            if *key == "tokenizer.ggml.tokens" {
                bytes.extend_from_slice(&9u32.to_le_bytes());
                bytes.extend_from_slice(&8u32.to_le_bytes());
                bytes.extend_from_slice(&1u64.to_le_bytes());
            } else {
                bytes.extend_from_slice(&8u32.to_le_bytes());
            }
            string(&mut bytes, value);
        }
        bytes
    }

    #[test]
    fn the_gguf_head_offers_only_what_the_loader_reads() {
        let dir = tempfile::tempdir().expect("temp dir");
        let write = |name: &str, bytes: &[u8]| {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).expect("write");
            path
        };
        let good = write(
            "good.gguf",
            &gguf_with(&[
                ("general.architecture", "llama"),
                ("general.name", "x"),
                ("tokenizer.ggml.tokens", "a"),
            ]),
        );
        assert!(gguf_file_is_model(&good));

        let bert = write(
            "bert.gguf",
            &gguf_with(&[
                ("general.architecture", "bert"),
                ("tokenizer.ggml.tokens", "a"),
            ]),
        );
        assert!(!gguf_file_is_model(&bert));
        assert!(matches!(
            check_loadable(&bert),
            Err(NotLoadable::Architecture(_))
        ));

        let no_vocabulary = write(
            "no-vocabulary.gguf",
            &gguf_with(&[("general.architecture", "llama")]),
        );
        assert!(!gguf_file_is_model(&no_vocabulary));
        assert!(matches!(
            check_loadable(&no_vocabulary),
            Err(NotLoadable::Tokenizer(_))
        ));

        assert!(!gguf_file_is_model(&write("html.gguf", b"Invalid path")));
        assert!(gguf_file_is_model(&write("empty.gguf", b"")));
    }
}
