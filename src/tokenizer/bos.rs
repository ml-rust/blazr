//! The BOS rule a SafeTensors checkpoint states in its own files.
//!
//! A SafeTensors checkpoint runs with its own `tokenizer.json`. Without one,
//! it gets a bundled vocabulary matched by size. That
//! vocabulary knows nothing of the model's BOS rule. HF `transformers` and
//! llama.cpp prepend BOS for the Llama and Mistral families, and so the GGUF
//! path does from `tokenizer.ggml.add_bos_token`. This module reads the same
//! rule from the checkpoint directory:
//!
//! - `tokenizer_config.json` `add_bos_token`, when it is a bool
//! - else the HF default of a Llama-family `tokenizer_class`, which is true
//! - else the leading special token of the `tokenizer.json` post-processor
//!
//! The BOS id comes from the post-processor, then `added_tokens`, then the
//! bundled vocabulary, then `config.json` `bos_token_id`.

use std::path::Path;

use anyhow::{anyhow, Result};
use serde_json::Value;
use splintr::AnyTokenizer;

use super::splintr_tokenizer::from_vocab_size;

/// `tokenizer_class` values whose HF tokenizer defaults `add_bos_token` to
/// true.
const BOS_BY_DEFAULT: &[&str] = &[
    "LlamaTokenizer",
    "LlamaTokenizerFast",
    "CodeLlamaTokenizer",
    "CodeLlamaTokenizerFast",
    "GemmaTokenizer",
    "GemmaTokenizerFast",
];

/// The tokenizer files of one checkpoint directory, each parsed when present.
#[derive(Debug, Default)]
pub struct CheckpointTokenizerFiles {
    /// `tokenizer_config.json`.
    pub tokenizer_config: Option<Value>,
    /// `tokenizer.json`.
    pub tokenizer_json: Option<Value>,
    /// `config.json`.
    pub config: Option<Value>,
}

impl CheckpointTokenizerFiles {
    /// Read the files beside `model`, a checkpoint directory or a weights
    /// file in one.
    ///
    /// # Errors
    ///
    /// An error when a file exists but does not parse as JSON.
    pub fn read(model: &Path) -> Result<Self> {
        let dir = if model.is_dir() {
            model
        } else {
            model.parent().unwrap_or(Path::new("."))
        };
        Ok(Self {
            tokenizer_config: read_json(&dir.join("tokenizer_config.json"))?,
            tokenizer_json: read_json(&dir.join("tokenizer.json"))?,
            config: read_json(&dir.join("config.json"))?,
        })
    }
}

fn read_json(path: &Path) -> Result<Option<Value>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(anyhow!("{} does not read: {e}", path.display())),
    };
    serde_json::from_str(&text).map(Some).map_err(|e| {
        anyhow!(
            "{} is not valid JSON: {e}. Fix or remove the file",
            path.display()
        )
    })
}

/// What the files say about BOS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BosDecision {
    /// No tokenizer file states a rule. The vocabulary's own policy stays.
    Unstated,
    /// The files state that no BOS is prepended.
    Off,
    /// The files state that this BOS id is prepended.
    Add(u32),
}

/// Decide the BOS rule from the files. `vocab_id` looks a token name up in
/// the vocabulary the model runs with.
///
/// # Errors
///
/// An error when the files ask for BOS but name no id, or when their id
/// differs from the vocabulary's id for the same token.
pub fn decide_bos(
    files: &CheckpointTokenizerFiles,
    vocab_id: impl Fn(&str) -> Option<u32>,
) -> Result<BosDecision> {
    let tc = files.tokenizer_config.as_ref();
    let template = files
        .tokenizer_json
        .as_ref()
        .and_then(template_leading_special);
    if tc.is_none() && files.tokenizer_json.is_none() {
        return Ok(BosDecision::Unstated);
    }
    let add = match tc
        .and_then(|c| c.get("add_bos_token"))
        .and_then(Value::as_bool)
    {
        Some(flag) => flag,
        None => match tc
            .and_then(|c| c.get("tokenizer_class"))
            .and_then(Value::as_str)
        {
            Some(class) if BOS_BY_DEFAULT.contains(&class) => true,
            _ => template.is_some(),
        },
    };
    if !add {
        return Ok(BosDecision::Off);
    }

    let name = tc
        .and_then(|c| token_content(c.get("bos_token")))
        .or_else(|| template.as_ref().map(|(name, _)| name.clone()));
    let from_files = name.as_deref().and_then(|name| {
        template
            .as_ref()
            .filter(|(t, _)| t == name)
            .and_then(|(_, id)| *id)
            .or_else(|| {
                files
                    .tokenizer_json
                    .as_ref()
                    .and_then(|j| added_token_id(j, name))
            })
    });
    let from_vocab = name.as_deref().and_then(&vocab_id);
    let from_config = files
        .config
        .as_ref()
        .and_then(|c| c.get("bos_token_id"))
        .and_then(Value::as_u64)
        .and_then(|id| u32::try_from(id).ok());

    let shown = name.as_deref().unwrap_or("(unnamed)");
    if let (Some(file), Some(vocab)) = (from_files, from_vocab) {
        if file != vocab {
            return Err(anyhow!(
                "the checkpoint's BOS token {shown} is id {file}, but the vocabulary blazr \
                 matched gives it id {vocab}. The checkpoint needs its own vocabulary"
            ));
        }
    }
    from_files
        .or(from_vocab)
        .or(from_config)
        .map(BosDecision::Add)
        .ok_or_else(|| {
            anyhow!(
                "the checkpoint asks for a BOS token ({shown}) but no file gives its id. \
                 Set bos_token_id in config.json"
            )
        })
}

/// The special token before `$A` in the `tokenizer.json` post-processor's
/// single template, with its id when the processor lists one.
fn template_leading_special(json: &Value) -> Option<(String, Option<u32>)> {
    let processor = json.get("post_processor")?;
    let template = match processor.get("type").and_then(Value::as_str)? {
        "TemplateProcessing" => processor,
        "Sequence" => processor
            .get("processors")?
            .as_array()?
            .iter()
            .find(|p| p.get("type").and_then(Value::as_str) == Some("TemplateProcessing"))?,
        _ => return None,
    };
    let first = template.get("single")?.as_array()?.first()?;
    let name = first.get("SpecialToken")?.get("id")?.as_str()?.to_string();
    let id = template
        .get("special_tokens")
        .and_then(|t| t.get(&name))
        .and_then(|t| t.get("ids"))
        .and_then(Value::as_array)
        .and_then(|ids| ids.first())
        .and_then(Value::as_u64)
        .and_then(|id| u32::try_from(id).ok());
    Some((name, id))
}

/// The id of the `added_tokens` entry whose content is `name`.
fn added_token_id(json: &Value, name: &str) -> Option<u32> {
    json.get("added_tokens")?
        .as_array()?
        .iter()
        .find(|t| t.get("content").and_then(Value::as_str) == Some(name))?
        .get("id")?
        .as_u64()
        .and_then(|id| u32::try_from(id).ok())
}

/// A token name stored as a string or as an `AddedToken` object.
fn token_content(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content")?.as_str().map(str::to_string),
        _ => None,
    }
}

/// `tokenizer` with the BOS rule of the checkpoint at `model` applied.
///
/// # Errors
///
/// As [`CheckpointTokenizerFiles::read`] and [`decide_bos`].
pub fn apply_checkpoint_bos(tokenizer: AnyTokenizer, model: &Path) -> Result<AnyTokenizer> {
    apply_bos_files(tokenizer, &CheckpointTokenizerFiles::read(model)?)
}

/// `tokenizer` with the BOS rule that `files` state applied.
///
/// # Errors
///
/// As [`decide_bos`].
pub fn apply_bos_files(
    tokenizer: AnyTokenizer,
    files: &CheckpointTokenizerFiles,
) -> Result<AnyTokenizer> {
    let decision = decide_bos(files, |name| tokenizer.special_token_id(name))?;
    Ok(match decision {
        BosDecision::Unstated => tokenizer,
        BosDecision::Off => tokenizer.with_leading_bos(None),
        BosDecision::Add(id) => tokenizer.with_leading_bos(Some(id)),
    })
}

/// The tokenizer for a SafeTensors checkpoint, with the BOS rule of its own
/// files.
///
/// - A `tokenizer.json` beside the weights is the model's own tokenizer, and
///   it wins.
/// - Without one, the bundled vocabulary that `vocab_size` matches serves.
///
/// Embedding row counts are often padded past the tokenizer's size, so
/// Qwen2, Phi-3, Gemma and StarCoder2 match no bundled vocabulary by size.
///
/// # Errors
///
/// - A `tokenizer.json` that does not load. Blazr never falls back to a
///   guessed vocabulary then.
/// - As [`from_vocab_size`] and [`apply_checkpoint_bos`].
pub fn checkpoint_tokenizer(model: &Path, vocab_size: usize) -> Result<AnyTokenizer> {
    let dir = if model.is_dir() {
        model
    } else {
        model.parent().unwrap_or(Path::new("."))
    };
    let json = dir.join("tokenizer.json");
    let base = if json.is_file() {
        splintr::from_json_path(&json).map_err(|e| {
            anyhow!(
                "{} does not load as a tokenizer: {e}. Fix or remove the file",
                json.display()
            )
        })?
    } else {
        from_vocab_size(vocab_size)?
    };
    apply_checkpoint_bos(base, model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn llama3_json() -> Value {
        json!({
            "added_tokens": [{"id": 128000, "content": "<|begin_of_text|>"}],
            "post_processor": {"type": "Sequence", "processors": [
                {"type": "ByteLevel"},
                {"type": "TemplateProcessing",
                 "single": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
                            {"Sequence": {"id": "A", "type_id": 0}}],
                 "special_tokens": {"<|begin_of_text|>": {"ids": [128000]}}}
            ]}
        })
    }

    fn files(
        tc: Option<Value>,
        tj: Option<Value>,
        config: Option<Value>,
    ) -> CheckpointTokenizerFiles {
        CheckpointTokenizerFiles {
            tokenizer_config: tc,
            tokenizer_json: tj,
            config,
        }
    }

    fn no_vocab(_: &str) -> Option<u32> {
        None
    }

    #[test]
    fn add_bos_true_uses_the_named_token() {
        let f = files(
            Some(json!({"add_bos_token": true, "bos_token": "<s>"})),
            Some(json!({"added_tokens": [{"id": 1, "content": "<s>"}]})),
            None,
        );
        assert_eq!(decide_bos(&f, no_vocab).unwrap(), BosDecision::Add(1));
    }

    #[test]
    fn add_bos_false_overrides_the_template() {
        let f = files(
            Some(json!({"add_bos_token": false})),
            Some(llama3_json()),
            None,
        );
        assert_eq!(decide_bos(&f, no_vocab).unwrap(), BosDecision::Off);
    }

    /// Llama 3 states no flag. Its post-processor template prepends BOS.
    #[test]
    fn absent_flag_follows_the_template() {
        let f = files(
            Some(
                json!({"bos_token": "<|begin_of_text|>", "tokenizer_class": "PreTrainedTokenizerFast"}),
            ),
            Some(llama3_json()),
            None,
        );
        assert_eq!(decide_bos(&f, no_vocab).unwrap(), BosDecision::Add(128000));
    }

    /// A Llama tokenizer class defaults the flag to true. The id falls back to
    /// the vocabulary, then to `config.json`.
    #[test]
    fn absent_flag_follows_the_llama_class_default() {
        let tc = json!({"bos_token": {"content": "<s>"}, "tokenizer_class": "LlamaTokenizer"});
        let f = files(Some(tc.clone()), None, None);
        let vocab = |name: &str| (name == "<s>").then_some(1);
        assert_eq!(decide_bos(&f, vocab).unwrap(), BosDecision::Add(1));
        let f = files(Some(tc), None, Some(json!({"bos_token_id": 1})));
        assert_eq!(decide_bos(&f, no_vocab).unwrap(), BosDecision::Add(1));
    }

    #[test]
    fn absent_flag_and_no_template_adds_nothing() {
        let f = files(
            Some(json!({"bos_token": null, "tokenizer_class": "Qwen2Tokenizer"})),
            Some(json!({"post_processor": {"type": "ByteLevel"}})),
            None,
        );
        assert_eq!(decide_bos(&f, no_vocab).unwrap(), BosDecision::Off);
    }

    #[test]
    fn no_tokenizer_files_leave_the_vocabulary_alone() {
        let f = files(None, None, Some(json!({"bos_token_id": 1})));
        assert_eq!(decide_bos(&f, no_vocab).unwrap(), BosDecision::Unstated);
    }

    #[test]
    fn a_vocabulary_id_mismatch_is_an_error() {
        let f = files(None, Some(llama3_json()), None);
        let vocab = |_: &str| Some(5);
        assert!(decide_bos(&f, vocab).is_err());
        let f = files(Some(json!({"add_bos_token": true})), None, None);
        assert!(decide_bos(&f, no_vocab).is_err(), "no id anywhere");
    }

    /// The bundled Llama 3 vocabulary gets BOS from the checkpoint files.
    #[test]
    fn bundled_vocabulary_takes_the_checkpoint_bos() {
        for name in ["llama3", "mistral_v1", "mistral_v2"] {
            let bundled = super::super::from_pretrained(name).unwrap();
            assert_eq!(
                bundled.policy().leading_special(),
                None,
                "{name} adds no BOS"
            );
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            json!({"add_bos_token": true, "bos_token": "<|begin_of_text|>"}).to_string(),
        )
        .unwrap();
        let tok = checkpoint_tokenizer(dir.path(), 128256).unwrap();
        assert_eq!(tok.encode("Hi")[0], 128000);
        assert_eq!(
            tok.encode("Hi").iter().filter(|&&id| id == 128000).count(),
            1
        );
    }

    /// A byte-level BPE `tokenizer.json` with two tokens and a BOS template.
    fn own_tokenizer_json() -> Value {
        json!({
            "added_tokens": [{"id": 2, "content": "<s>", "special": true}],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                           {"Sequence": {"id": "A", "type_id": 0}}],
                "special_tokens": {"<s>": {"ids": [2]}}},
            "model": {"type": "BPE", "vocab": {"a": 0, "Ġa": 1}, "merges": []}
        })
    }

    /// The checkpoint's own `tokenizer.json` serves, whatever the padded
    /// embedding size, and its template BOS is kept once.
    #[test]
    fn own_tokenizer_json_wins_over_the_size_match() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer.json"),
            own_tokenizer_json().to_string(),
        )
        .unwrap();
        // 151936 matches no bundled vocabulary.
        let tok = checkpoint_tokenizer(dir.path(), 151936).unwrap();
        assert_eq!(tok.encode("a a"), vec![2, 0, 1]);
    }

    /// A `tokenizer.json` that does not load is an error, never a guess.
    #[test]
    fn broken_tokenizer_json_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), "{\"model\": 3}").unwrap();
        let Err(err) = checkpoint_tokenizer(dir.path(), 128256) else {
            panic!("a broken tokenizer.json must not load");
        };
        assert!(err.to_string().contains("tokenizer.json"), "{err}");
    }

    #[test]
    fn bundled_vocabulary_without_files_keeps_its_policy() {
        let dir = tempfile::tempdir().unwrap();
        let tok = checkpoint_tokenizer(dir.path(), 128256).unwrap();
        assert!(!tok.encode("Hi").contains(&128000));
    }
}
