//! Chat template from GGUF metadata. Reads the header only.

use std::path::Path;

use anyhow::{Result, anyhow};

use boostr::format::{Gguf, GgufMetadata};

use crate::model::chat_template::{ModelTemplate, TemplateOrigin};

/// `tokenizer.chat_template` from the GGUF at `path`, or `None` when the file
/// carries no template.
pub fn gguf_chat_template(path: &Path) -> Result<Option<ModelTemplate>> {
    let gguf = Gguf::open(path)
        .map_err(|e| anyhow!("open GGUF {} for its chat template: {e}", path.display()))?;
    Ok(chat_template_from_metadata(gguf.metadata()))
}

/// The template the metadata names, with its BOS and EOS spellings.
///
/// `tokenizer.chat_template` wins. Otherwise `tokenizer.chat_templates` lists
/// names, each stored under `tokenizer.chat_template.<name>`; `default` wins
/// there, then the first listed.
pub fn chat_template_from_metadata(metadata: &GgufMetadata) -> Option<ModelTemplate> {
    let source = select_source(metadata)?;
    Some(ModelTemplate {
        source,
        origin: TemplateOrigin::Gguf,
        bos_token: token_string(metadata, "tokenizer.ggml.bos_token_id"),
        eos_token: token_string(metadata, "tokenizer.ggml.eos_token_id"),
    })
}

fn select_source(metadata: &GgufMetadata) -> Option<String> {
    if let Some(source) = metadata.get_string("tokenizer.chat_template") {
        return Some(source.to_string());
    }
    let names = metadata.get_string_array("tokenizer.chat_templates")?;
    let chosen = names
        .iter()
        .find(|n| n.as_str() == "default")
        .or_else(|| names.first())?;
    metadata
        .get_string(&format!("tokenizer.chat_template.{chosen}"))
        .map(str::to_string)
}

/// The vocabulary entry the id under `key` points at, or `""`.
fn token_string(metadata: &GgufMetadata, key: &str) -> String {
    let Some(id) = metadata.get_u32(key) else {
        return String::new();
    };
    metadata
        .get_array("tokenizer.ggml.tokens")
        .and_then(|tokens| tokens.get(id as usize))
        .and_then(|t| t.as_string())
        .unwrap_or_default()
        .to_string()
}
