//! Where a model's chat template comes from.

use std::path::Path;

use anyhow::{anyhow, Context, Result};

use super::jinja::{JinjaTemplate, TemplateOrigin};
use super::template::{ChatTemplate, BUILTIN_NAMES};

/// A template text a model ships with, plus the token strings it renders with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelTemplate {
    pub source: String,
    pub origin: TemplateOrigin,
    pub bos_token: String,
    pub eos_token: String,
}

impl ModelTemplate {
    /// Compile into [`ChatTemplate::Jinja`].
    pub fn compile(&self) -> Result<ChatTemplate> {
        let template =
            JinjaTemplate::new(&self.source, self.origin, &self.bos_token, &self.eos_token)?;
        Ok(ChatTemplate::Jinja(template))
    }
}

impl ChatTemplate {
    /// Pick the template for the model at `path`.
    ///
    /// Order: `override_` (a built-in name or a template file) > the model's
    /// own Jinja template > the built-in format for `model_type`.
    pub fn detect(path: &Path, model_type: &str, override_: Option<&str>) -> Result<Self> {
        let shipped = Self::from_model(path)?;
        if let Some(value) = override_ {
            return Self::from_override(value, shipped.as_ref());
        }
        match shipped {
            Some(model_template) => model_template.compile(),
            None => Ok(Self::from_model_type(model_type)),
        }
    }

    /// The model's own template, when it ships one.
    ///
    /// A GGUF file carries it in `tokenizer.chat_template`; a checkpoint
    /// directory in `chat_template.jinja` or `tokenizer_config.json`.
    pub fn from_model(path: &Path) -> Result<Option<ModelTemplate>> {
        if path.is_file() {
            let is_gguf = path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
            return if is_gguf {
                crate::loader::gguf_chat_template(path)
            } else {
                Ok(None)
            };
        }
        from_tokenizer_config(path)
    }

    /// Resolve `--chat-template`: a built-in name, or a path to a template
    /// file. A file renders with the model's own `bos_token`/`eos_token` when
    /// the model ships them.
    fn from_override(value: &str, shipped: Option<&ModelTemplate>) -> Result<Self> {
        if let Some(builtin) = Self::builtin(value) {
            return Ok(builtin);
        }
        let file = Path::new(value);
        if file.is_file() {
            let source = std::fs::read_to_string(file)
                .with_context(|| format!("read chat template {}", file.display()))?;
            let (bos, eos) = shipped
                .map(|m| (m.bos_token.as_str(), m.eos_token.as_str()))
                .unwrap_or(("", ""));
            let template = JinjaTemplate::new(&source, TemplateOrigin::File, bos, eos)?;
            return Ok(ChatTemplate::Jinja(template));
        }
        Err(anyhow!(
            "chat template {value:?} is neither a built-in name ({}) nor a file",
            BUILTIN_NAMES.join(", ")
        ))
    }
}

/// The template a checkpoint directory ships.
///
/// `chat_template.jinja` wins over the `chat_template` field of
/// `tokenizer_config.json`, as in Hugging Face `transformers`. The
/// `bos_token` and `eos_token` come from `tokenizer_config.json` either way.
fn from_tokenizer_config(dir: &Path) -> Result<Option<ModelTemplate>> {
    let path = dir.join("tokenizer_config.json");
    let config = if path.is_file() {
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        Some(text)
    } else {
        None
    };
    let file = dir.join(CHAT_TEMPLATE_FILE);
    if file.is_file() {
        let source =
            std::fs::read_to_string(&file).with_context(|| format!("read {}", file.display()))?;
        let (bos_token, eos_token) = match &config {
            Some(text) => {
                let json: serde_json::Value = serde_json::from_str(text)
                    .with_context(|| format!("parse {}", path.display()))?;
                (
                    token_string(json.get("bos_token")),
                    token_string(json.get("eos_token")),
                )
            }
            None => (String::new(), String::new()),
        };
        return Ok(Some(ModelTemplate {
            source,
            origin: TemplateOrigin::ChatTemplateFile,
            bos_token,
            eos_token,
        }));
    }
    match config {
        Some(text) => template_from_tokenizer_config(&text)
            .with_context(|| format!("parse {}", path.display())),
        None => Ok(None),
    }
}

/// The template file `transformers` saves beside `tokenizer_config.json`.
pub const CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";

/// `chat_template` from the text of a `tokenizer_config.json`, when it
/// has one.
///
/// # Errors
///
/// An error when the text is not JSON.
pub fn template_from_tokenizer_config(text: &str) -> Result<Option<ModelTemplate>> {
    let json: serde_json::Value = serde_json::from_str(text)?;
    let Some(source) = select_template(json.get("chat_template")) else {
        return Ok(None);
    };
    Ok(Some(ModelTemplate {
        source,
        origin: TemplateOrigin::TokenizerConfig,
        bos_token: token_string(json.get("bos_token")),
        eos_token: token_string(json.get("eos_token")),
    }))
}

/// `chat_template` is a string, or a list of `{name, template}` entries where
/// `default` wins and the first entry is the fallback.
fn select_template(value: Option<&serde_json::Value>) -> Option<String> {
    match value? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(entries) => {
            let named = |name: &str| {
                entries
                    .iter()
                    .find(|e| e.get("name").and_then(|n| n.as_str()) == Some(name))
            };
            named("default")
                .or_else(|| entries.first())
                .and_then(|e| e.get("template"))
                .and_then(|t| t.as_str())
                .map(str::to_string)
        }
        _ => None,
    }
}

/// A token field is a string, or an added-token object with `content`.
fn token_string(value: Option<&serde_json::Value>) -> String {
    match value {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Object(obj)) => obj
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("blazr-chat-template-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn tokenizer_config_template_is_read_with_token_objects() {
        let dir = temp_dir("config");
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"bos_token": {"content": "<s>"}, "eos_token": "</s>",
                "chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"}"#,
        )
        .expect("write");
        let shipped = ChatTemplate::from_model(&dir)
            .expect("reads")
            .expect("ships a template");
        assert_eq!(shipped.origin, TemplateOrigin::TokenizerConfig);
        assert_eq!(shipped.bos_token, "<s>");
        assert_eq!(shipped.eos_token, "</s>");
        let template = ChatTemplate::detect(&dir, "llama", None).expect("detects");
        let prompt = template
            .apply(&[super::super::ChatMessage::new("user", "Hi")])
            .expect("renders");
        assert_eq!(prompt, "<s>Hi");
        assert_eq!(template.describe(), "tokenizer_config jinja");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// `chat_template.jinja` wins over the config's `chat_template`, and the
    /// token strings still come from the config.
    #[test]
    fn chat_template_file_wins_over_tokenizer_config() {
        let dir = temp_dir("jinja-file");
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"bos_token": "<s>", "eos_token": {"content": "</s>"},
                "chat_template": "CONFIG"}"#,
        )
        .expect("write");
        std::fs::write(
            dir.join(CHAT_TEMPLATE_FILE),
            "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}",
        )
        .expect("write");
        let shipped = ChatTemplate::from_model(&dir)
            .expect("reads")
            .expect("ships a template");
        assert_eq!(shipped.origin, TemplateOrigin::ChatTemplateFile);
        assert_eq!(shipped.bos_token, "<s>");
        assert_eq!(shipped.eos_token, "</s>");
        let prompt = shipped
            .compile()
            .expect("compiles")
            .apply(&[super::super::ChatMessage::new("user", "Hi")])
            .expect("renders");
        assert_eq!(prompt, "<s>Hi");

        std::fs::remove_file(dir.join("tokenizer_config.json")).expect("remove");
        let shipped = ChatTemplate::from_model(&dir)
            .expect("reads")
            .expect("the file alone is a template");
        assert_eq!(shipped.bos_token, "");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn tokenizer_config_list_prefers_default() {
        let value = serde_json::json!([
            {"name": "rag", "template": "R"},
            {"name": "default", "template": "D"}
        ]);
        assert_eq!(select_template(Some(&value)).as_deref(), Some("D"));
        let value = serde_json::json!([{"name": "tool_use", "template": "T"}]);
        assert_eq!(select_template(Some(&value)).as_deref(), Some("T"));
        assert_eq!(select_template(None), None);
    }

    #[test]
    fn override_wins_and_unknown_override_errors() {
        let dir = temp_dir("override");
        let template = ChatTemplate::detect(&dir, "qwen35", Some("llama3")).expect("builtin");
        assert_eq!(template, ChatTemplate::Llama3);
        let err =
            ChatTemplate::detect(&dir, "qwen35", Some("no-such-format")).expect_err("unknown");
        assert!(err.to_string().contains("no-such-format"));
        let file = dir.join("custom.jinja");
        std::fs::write(&file, "{% for m in messages %}[{{ m.role }}]{% endfor %}").expect("write");
        let template = ChatTemplate::detect(&dir, "qwen35", Some(file.to_str().expect("utf-8")))
            .expect("file template");
        assert_eq!(template.describe(), "file jinja");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn no_shipped_template_falls_back_to_model_type() {
        let dir = temp_dir("fallback");
        let template = ChatTemplate::detect(&dir, "qwen35", None).expect("detects");
        assert_eq!(template, ChatTemplate::ChatML);
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
