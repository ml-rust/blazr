//! What a decider model was trained in: `decider_config.json`'s layout
//! and flags, and the fixed ids of the chat layout.
//!
//! Ported from `decider/prompt.py` (`resolve_layout`, `ChatTemplate`) and
//! the config reads of `decider/infer.py` (`Decider.__init__`).

use serde_json::{Map, Value};

use super::encode::{ChatFrame, DecisionEncoder};
use super::family::DecisionError;
use super::labels::DeciderLabels;
use super::py_text::{py_repr, py_repr_str, py_truthy};
use crate::model::chat_template::{JinjaTemplate, RenderOptions};

/// Options up to this many render as `(A) text` lines in one string.
pub const NARROW: usize = 10;
/// The answer word of the chat layout.
pub const ANSWER: &str = "Answer";

/// The prompt layout a model was trained in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// No chat template: every released decider.
    Plain,
    /// The tokenizer's chat template around one user turn, thinking off.
    Chat,
}

/// What `decider_config.json` says about rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeciderConfig {
    /// `layout`, or `chat_template: true` for the chat layout.
    pub layout: Layout,
    /// `neutralize_none` (default true).
    pub neutralize_none: bool,
    /// `isolated_levels` (default false).
    pub isolated_levels: bool,
    /// `schema_first` (default false): the model's default layout is the
    /// schema-first one.
    pub schema_first: bool,
}

impl DeciderConfig {
    /// Read the config, as `resolve_layout` and `Decider.__init__` do.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] for a layout the decider does not know, or
    /// a plain layout that also says `chat_template: true`.
    pub fn from_json(cfg: &Value) -> Result<Self, DecisionError> {
        let empty = Map::new();
        let cfg = cfg.as_object().unwrap_or(&empty);
        let chat_template = cfg.get("chat_template") == Some(&Value::Bool(true));
        let layout = match cfg.get("layout").filter(|v| !v.is_null()) {
            None if chat_template => Layout::Chat,
            None => Layout::Plain,
            Some(Value::String(s)) if s == "plain" => Layout::Plain,
            Some(Value::String(s)) if s == "chat" => Layout::Chat,
            Some(other) => {
                return Err(DecisionError::Model(format!(
                    "decider_config.json names the prompt layout {}; this version of decider-ai \
                     knows 'plain', 'chat'. Upgrade decider-ai, or check the model's config.",
                    py_repr(other)
                )))
            }
        };
        if layout == Layout::Plain && chat_template && cfg.contains_key("layout") {
            return Err(DecisionError::Model(
                "decider_config.json says \"layout\": \"plain\" and \"chat_template\": true; \
                 these contradict each other"
                    .to_string(),
            ));
        }
        let flag = |key: &str, default: bool| cfg.get(key).map_or(default, py_truthy);
        Ok(Self {
            layout,
            neutralize_none: flag("neutralize_none", true),
            isolated_levels: flag("isolated_levels", false),
            schema_first: flag("schema_first", false),
        })
    }
}

/// The chat layout's fixed ids.
#[derive(Debug, Clone)]
pub struct ChatLayout {
    /// The template head: the user turn opened.
    pub head: Vec<u32>,
    /// The template tail: the user turn closed, the assistant turn opened
    /// with thinking closed.
    pub tail: Vec<u32>,
}

/// `ChatTemplate.__init__`: the template around one user turn, thinking
/// switched off, a thinking block the template leaves open closed, and the
/// checks that each narrow label and the answer prefix tokenize apart.
pub fn chat_layout(
    encoder: &DecisionEncoder,
    labels: &DeciderLabels,
    template: Option<&JinjaTemplate>,
) -> Result<ChatLayout, DecisionError> {
    let template = template.ok_or_else(|| {
        DecisionError::Model(
            "the model is configured for the chat layout (\"layout\": \"chat\") but its \
             tokenizer has no chat template"
                .to_string(),
        )
    })?;
    let mut extra = Map::new();
    if template.source().contains("enable_thinking") {
        extra.insert("enable_thinking".into(), Value::Bool(false));
    }
    let options = RenderOptions {
        add_generation_prompt: true,
        tools: None,
        extra,
    };
    let frame = ChatFrame::render(template, None, &options)?;
    let mut tail = frame.tail;
    for (opened, closed) in [
        ("<think>\n", "</think>\n\n"),
        ("<think>", "</think>\n\n"),
        ("<|channel>thought\n", "<channel|>"),
    ] {
        if tail.ends_with(opened) {
            tail.push_str(closed);
        }
    }
    let head_ids = encoder.template(&frame.head)?;
    let tail_ids = encoder.template(&tail)?;
    let pre = encoder.template(&format!("{ANSWER}: ("))?;
    for label in labels.labels.iter().take(NARROW) {
        let mut want = pre.clone();
        want.push(label.id);
        if encoder.template(&format!("{ANSWER}: ({}", label.text))? != want {
            return Err(DecisionError::Model(format!(
                "chat layout: the label {} does not tokenize as one token after {}: (",
                py_repr_str(&label.text),
                py_repr_str(ANSWER)
            )));
        }
    }
    let mut want = tail_ids.clone();
    want.extend_from_slice(&pre);
    if encoder.template(&format!("{tail}{ANSWER}: ("))? != want {
        return Err(DecisionError::Model(
            "chat layout: the chat template tail merges with the answer prefix".to_string(),
        ));
    }
    Ok(ChatLayout {
        head: head_ids,
        tail: tail_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn config_reads_like_resolve_layout() {
        let plain = DeciderConfig::from_json(&json!({"layout": "plain", "neutralize_none": false,
            "isolated_levels": true}))
        .expect("reads");
        assert_eq!(plain.layout, Layout::Plain);
        assert!(!plain.neutralize_none && plain.isolated_levels && !plain.schema_first);
        let defaults = DeciderConfig::from_json(&json!({})).expect("reads");
        assert!(defaults.neutralize_none && !defaults.isolated_levels);
        assert_eq!(
            DeciderConfig::from_json(&json!({"chat_template": true})).map(|c| c.layout),
            Ok(Layout::Chat)
        );
        assert!(
            DeciderConfig::from_json(&json!({"layout": "plain", "chat_template": true})).is_err()
        );
        assert_eq!(
            DeciderConfig::from_json(&json!({"layout": "fancy"})),
            Err(DecisionError::Model(
                "decider_config.json names the prompt layout 'fancy'; this version of decider-ai \
                 knows 'plain', 'chat'. Upgrade decider-ai, or check the model's config."
                    .to_string()
            ))
        );
    }
}
