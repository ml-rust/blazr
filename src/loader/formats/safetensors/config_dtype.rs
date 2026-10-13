//! The dtype a Hugging Face `config.json` names for its weights.

use std::path::Path;

use boostr::DType;
use serde_json::Value;

use crate::config::torch_dtype;

/// Keys that name the weight dtype, in the order they are read.
const DTYPE_KEYS: [&str; 2] = ["torch_dtype", "dtype"];

/// The dtype a model's `config.json` names. `None` when the file, every key
/// or a known value is missing. See [`torch_dtype_of`].
pub fn model_torch_dtype(dir: &Path) -> Option<DType> {
    let content = std::fs::read_to_string(dir.join("config.json")).ok()?;
    torch_dtype_of(&content)
}

/// The dtype the text of a `config.json` names.
///
/// Read in order: `torch_dtype`, `dtype`, then the same two keys under
/// `text_config`. transformers v5 writes `dtype`. A
/// `Qwen3_5ForConditionalGeneration` config can name it only under
/// `text_config`. The first key present with a known value wins.
pub fn torch_dtype_of(config_json: &str) -> Option<DType> {
    let json = serde_json::from_str::<Value>(config_json).ok()?;
    dtype_in(&json).or_else(|| json.get("text_config").and_then(dtype_in))
}

/// The first of [`DTYPE_KEYS`] in `object` that names a known dtype.
fn dtype_in(object: &Value) -> Option<DType> {
    DTYPE_KEYS.iter().find_map(|key| {
        object
            .get(key)
            .and_then(Value::as_str)
            .and_then(torch_dtype)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torch_dtype_is_read_first() {
        let json = r#"{"torch_dtype": "float32", "dtype": "bfloat16"}"#;
        assert_eq!(torch_dtype_of(json), Some(DType::F32));
    }

    #[test]
    fn dtype_is_read_when_torch_dtype_is_absent() {
        assert_eq!(torch_dtype_of(r#"{"dtype": "float32"}"#), Some(DType::F32));
    }

    #[test]
    fn text_config_is_read_when_the_top_level_names_none() {
        let json = r#"{"text_config": {"dtype": "float32"}}"#;
        assert_eq!(torch_dtype_of(json), Some(DType::F32));
        let json = r#"{"text_config": {"torch_dtype": "float32", "dtype": "bfloat16"}}"#;
        assert_eq!(torch_dtype_of(json), Some(DType::F32));
    }

    #[test]
    fn top_level_beats_text_config() {
        let json = r#"{"dtype": "float32", "text_config": {"dtype": "float16"}}"#;
        assert_eq!(torch_dtype_of(json), Some(DType::F32));
    }

    #[test]
    fn unknown_value_falls_through_to_the_next_key() {
        let json = r#"{"torch_dtype": "auto", "dtype": "float32"}"#;
        assert_eq!(torch_dtype_of(json), Some(DType::F32));
    }

    #[test]
    fn no_key_or_bad_json_is_none() {
        assert_eq!(torch_dtype_of(r#"{"text_config": {}}"#), None);
        assert_eq!(torch_dtype_of("not json"), None);
    }

    #[test]
    fn real_qwen35_configs_name_bf16() {
        use super::super::hf_qwen35::tests::{DECIDER_CONFIG, JEBADIAH_CONFIG};
        assert_eq!(torch_dtype_of(DECIDER_CONFIG), Some(DType::BF16));
        assert_eq!(torch_dtype_of(JEBADIAH_CONFIG), Some(DType::BF16));
    }

    #[test]
    fn model_dir_reads_its_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"dtype": "float32"}"#).unwrap();
        assert_eq!(model_torch_dtype(dir.path()), Some(DType::F32));
        assert_eq!(model_torch_dtype(&dir.path().join("missing")), None);
    }
}
