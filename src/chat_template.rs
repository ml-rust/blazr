//! Chat template support for per-model message formatting
//!
//! Detects and applies the correct chat template based on model type or
//! `tokenizer_config.json` chat_template field.

use std::path::Path;

/// Supported chat template formats
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Llama 3 format: `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,
    /// Mistral/Llama 2 format: `[INST] content [/INST]`
    MistralInstruct,
    /// ChatML format: `<|im_start|>role\ncontent<|im_end|>`
    /// Used by: Qwen, Yi, many fine-tunes
    ChatML,
    /// Phi-3 format: `<|system|>\ncontent<|end|>\n<|user|>\ncontent<|end|>\n<|assistant|>\n`
    Phi3,
    /// Gemma format: `<start_of_turn>role\ncontent<end_of_turn>`
    Gemma,
    /// DeepSeek format (V2/V3/R1)
    DeepSeek,
    /// Raw Jinja2 template string from tokenizer_config.json
    Jinja(String),
    /// Generic fallback: `role: content\n`
    #[default]
    Generic,
}

/// A chat message with role and content
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatTemplate {
    /// Detect the chat template from a model directory.
    ///
    /// 1. Try parsing `tokenizer_config.json` for a `chat_template` field
    /// 2. Fall back to model_type-based detection
    pub fn detect(model_dir: &Path, model_type: &str) -> Self {
        // Try tokenizer_config.json first
        if let Some(template) = Self::from_tokenizer_config(model_dir) {
            return template;
        }

        // Fall back to model_type detection
        Self::from_model_type(model_type)
    }

    /// Detect from tokenizer_config.json chat_template field
    fn from_tokenizer_config(model_dir: &Path) -> Option<Self> {
        let config_path = model_dir.join("tokenizer_config.json");
        let content = std::fs::read_to_string(&config_path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;
        let template_str = json.get("chat_template")?.as_str()?;

        // Detect known patterns from the Jinja2 template string
        if template_str.contains("<|start_header_id|>") {
            Some(ChatTemplate::Llama3)
        } else if template_str.contains("<|im_start|>") {
            Some(ChatTemplate::ChatML)
        } else if template_str.contains("[INST]") {
            Some(ChatTemplate::MistralInstruct)
        } else if template_str.contains("<|system|>") && template_str.contains("<|end|>") {
            Some(ChatTemplate::Phi3)
        } else if template_str.contains("<start_of_turn>") {
            Some(ChatTemplate::Gemma)
        } else if template_str.contains("<|begin▁of▁sentence|>")
            || template_str.contains("<｜begin▁of▁sentence｜>")
        {
            Some(ChatTemplate::DeepSeek)
        } else {
            // Store the raw Jinja2 template for potential future rendering
            Some(ChatTemplate::Jinja(template_str.to_string()))
        }
    }

    /// Detect from model_type string (from config.json)
    pub fn from_model_type(model_type: &str) -> Self {
        match model_type {
            "llama" => ChatTemplate::Llama3, // Default to Llama3 for llama models
            "mistral" => ChatTemplate::MistralInstruct,
            "qwen2" | "qwen2_moe" => ChatTemplate::ChatML,
            "phi3" | "phi" => ChatTemplate::Phi3,
            "gemma" | "gemma2" => ChatTemplate::Gemma,
            "deepseek_v2" | "deepseek_v3" => ChatTemplate::DeepSeek,
            "yi" => ChatTemplate::ChatML,
            "internlm2" => ChatTemplate::ChatML,
            "starcoder2" | "codellama" => ChatTemplate::Generic,
            _ => ChatTemplate::Generic,
        }
    }

    /// Parse a template name string into a ChatTemplate variant.
    /// Used for per-request template overrides.
    pub fn from_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "llama3" | "llama" => ChatTemplate::Llama3,
            "mistral" | "llama2" => ChatTemplate::MistralInstruct,
            "chatml" | "qwen" | "yi" => ChatTemplate::ChatML,
            "phi3" | "phi" => ChatTemplate::Phi3,
            "gemma" | "gemma2" => ChatTemplate::Gemma,
            "deepseek" => ChatTemplate::DeepSeek,
            "generic" | "raw" => ChatTemplate::Generic,
            _ => ChatTemplate::Generic,
        }
    }

    /// Format a list of chat messages into a prompt string
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatTemplate::Llama3 => format_llama3(messages),
            ChatTemplate::MistralInstruct => format_mistral(messages),
            ChatTemplate::ChatML => format_chatml(messages),
            ChatTemplate::Phi3 => format_phi3(messages),
            ChatTemplate::Gemma => format_gemma(messages),
            ChatTemplate::DeepSeek => format_deepseek(messages),
            ChatTemplate::Jinja(_) => {
                // For unrecognized Jinja templates, fall back to ChatML as safest default
                format_chatml(messages)
            }
            ChatTemplate::Generic => format_generic(messages),
        }
    }
}

/// Llama 3 format
fn format_llama3(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");

    for msg in messages {
        prompt.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }

    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

/// Mistral / Llama 2 [INST] format
fn format_mistral(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    let mut system_text = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                system_text = msg.content.clone();
            }
            "user" => {
                prompt.push_str("[INST] ");
                if !system_text.is_empty() {
                    prompt.push_str(&system_text);
                    prompt.push_str("\n\n");
                    system_text.clear();
                }
                prompt.push_str(&msg.content);
                prompt.push_str(" [/INST]");
            }
            "assistant" => {
                prompt.push(' ');
                prompt.push_str(&msg.content);
                prompt.push_str("</s>");
            }
            _ => {}
        }
    }

    prompt
}

/// ChatML format (Qwen, Yi, many fine-tunes)
fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Phi-3 format
fn format_phi3(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        prompt.push_str(&format!("<|{}|>\n{}<|end|>\n", msg.role, msg.content));
    }

    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Gemma format
fn format_gemma(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        let role = match msg.role.as_str() {
            "assistant" => "model",
            other => other,
        };
        prompt.push_str(&format!(
            "<start_of_turn>{}\n{}<end_of_turn>\n",
            role, msg.content
        ));
    }

    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// DeepSeek format
fn format_deepseek(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<|begin▁of▁sentence|>");

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&msg.content);
            }
            "user" => {
                prompt.push_str(&format!("<|User|>{}", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<|Assistant|>{}<|end▁of▁sentence|>", msg.content));
            }
            _ => {}
        }
    }

    prompt.push_str("<|Assistant|>");
    prompt
}

/// Generic fallback: `role: content\n`
fn format_generic(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }

    prompt.push_str("assistant: ");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(pairs: &[(&str, &str)]) -> Vec<ChatMessage> {
        pairs
            .iter()
            .map(|(role, content)| ChatMessage {
                role: role.to_string(),
                content: content.to_string(),
            })
            .collect()
    }

    #[test]
    fn test_llama3_format() {
        let messages = msgs(&[("system", "You are helpful."), ("user", "Hello")]);
        let result = ChatTemplate::Llama3.apply(&messages);
        assert!(result.starts_with("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(result.contains("You are helpful.<|eot_id|>"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_chatml_format() {
        let messages = msgs(&[("user", "Hi")]);
        let result = ChatTemplate::ChatML.apply(&messages);
        assert!(result.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_mistral_format() {
        let messages = msgs(&[("system", "Be concise."), ("user", "Hello")]);
        let result = ChatTemplate::MistralInstruct.apply(&messages);
        assert!(result.contains("[INST] Be concise.\n\nHello [/INST]"));
    }

    #[test]
    fn test_phi3_format() {
        let messages = msgs(&[("user", "Hello")]);
        let result = ChatTemplate::Phi3.apply(&messages);
        assert!(result.contains("<|user|>\nHello<|end|>"));
        assert!(result.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_gemma_format() {
        let messages = msgs(&[("user", "Hello")]);
        let result = ChatTemplate::Gemma.apply(&messages);
        assert!(result.contains("<start_of_turn>user\nHello<end_of_turn>"));
        assert!(result.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_model_type_detection() {
        assert_eq!(ChatTemplate::from_model_type("llama"), ChatTemplate::Llama3);
        assert_eq!(
            ChatTemplate::from_model_type("mistral"),
            ChatTemplate::MistralInstruct
        );
        assert_eq!(ChatTemplate::from_model_type("qwen2"), ChatTemplate::ChatML);
        assert_eq!(ChatTemplate::from_model_type("phi3"), ChatTemplate::Phi3);
        assert_eq!(ChatTemplate::from_model_type("gemma2"), ChatTemplate::Gemma);
        assert_eq!(
            ChatTemplate::from_model_type("unknown"),
            ChatTemplate::Generic
        );
    }
}
