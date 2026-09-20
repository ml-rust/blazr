//! The chat template a model renders prompts with.

use splintr::{AnyTokenizer, FxHashSet};

use super::builtin::{
    format_chatml, format_deepseek, format_gemma, format_generic, format_llama3, format_mistral,
    format_phi3, CHATML_MARKERS, DEEPSEEK_MARKERS, GEMMA_MARKERS, LLAMA3_MARKERS, MISTRAL_MARKERS,
    NO_MARKERS, PHI3_MARKERS,
};
use super::jinja::{JinjaTemplate, RenderOptions, TemplateError};
use super::message::ChatMessage;

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
    /// The model's own Jinja template, rendered as is.
    Jinja(JinjaTemplate),
    /// Generic fallback: `role: content\n`
    #[default]
    Generic,
}

/// Built-in names `--chat-template` and the `template` request field accept.
pub const BUILTIN_NAMES: &[&str] = &[
    "llama3", "llama", "mistral", "llama2", "chatml", "qwen", "yi", "phi3", "phi", "gemma",
    "gemma2", "deepseek", "generic", "raw",
];

impl ChatTemplate {
    /// Detect from model_type string (from config.json)
    pub fn from_model_type(model_type: &str) -> Self {
        match model_type {
            "llama" => ChatTemplate::Llama3, // Default to Llama3 for llama models
            "mistral" => ChatTemplate::MistralInstruct,
            "qwen2" | "qwen2_moe" | "qwen35" => ChatTemplate::ChatML,
            "phi3" | "phi" => ChatTemplate::Phi3,
            "gemma" | "gemma2" => ChatTemplate::Gemma,
            "deepseek_v2" | "deepseek_v3" => ChatTemplate::DeepSeek,
            "yi" => ChatTemplate::ChatML,
            "internlm2" => ChatTemplate::ChatML,
            "starcoder2" | "codellama" => ChatTemplate::Generic,
            _ => ChatTemplate::Generic,
        }
    }

    /// The built-in format `name` denotes, or `None` for an unknown name.
    pub fn builtin(name: &str) -> Option<Self> {
        Some(match name.to_lowercase().as_str() {
            "llama3" | "llama" => ChatTemplate::Llama3,
            "mistral" | "llama2" => ChatTemplate::MistralInstruct,
            "chatml" | "qwen" | "yi" => ChatTemplate::ChatML,
            "phi3" | "phi" => ChatTemplate::Phi3,
            "gemma" | "gemma2" => ChatTemplate::Gemma,
            "deepseek" => ChatTemplate::DeepSeek,
            "generic" | "raw" => ChatTemplate::Generic,
            _ => return None,
        })
    }

    /// Parse a template name string into a ChatTemplate variant.
    /// Used for per-request template overrides. An unknown name is `Generic`.
    pub fn from_name(name: &str) -> Self {
        Self::builtin(name).unwrap_or_default()
    }

    /// Format `messages` into a prompt with the generation turn opened.
    pub fn apply(&self, messages: &[ChatMessage]) -> Result<String, TemplateError> {
        self.apply_with(messages, &RenderOptions::default())
    }

    /// Format `messages` into a prompt under `options`.
    ///
    /// A built-in format reads none of `options`: it always opens the
    /// assistant turn and takes tools as text in the messages.
    pub fn apply_with(
        &self,
        messages: &[ChatMessage],
        options: &RenderOptions,
    ) -> Result<String, TemplateError> {
        Ok(match self {
            ChatTemplate::Llama3 => format_llama3(messages),
            ChatTemplate::MistralInstruct => format_mistral(messages),
            ChatTemplate::ChatML => format_chatml(messages),
            ChatTemplate::Phi3 => format_phi3(messages),
            ChatTemplate::Gemma => format_gemma(messages),
            ChatTemplate::DeepSeek => format_deepseek(messages),
            ChatTemplate::Jinja(template) => template.render(messages, options)?,
            ChatTemplate::Generic => format_generic(messages),
        })
    }

    /// The special tokens this template writes into the prompt itself.
    ///
    /// This is the allow-list the assembled prompt is encoded under: every
    /// marker `apply` emits must be in it, and nothing else may be. A Jinja
    /// template's set comes from [`bind_special_tokens`](Self::bind_special_tokens).
    pub fn allowed_special(&self) -> &FxHashSet<String> {
        match self {
            ChatTemplate::Llama3 => &LLAMA3_MARKERS,
            ChatTemplate::MistralInstruct => &MISTRAL_MARKERS,
            ChatTemplate::ChatML => &CHATML_MARKERS,
            ChatTemplate::Phi3 => &PHI3_MARKERS,
            ChatTemplate::Gemma => &GEMMA_MARKERS,
            ChatTemplate::DeepSeek => &DEEPSEEK_MARKERS,
            ChatTemplate::Jinja(template) => template.allowed_special(),
            // `role: content` spells no special token, so any special token in
            // the assembled prompt came from the caller.
            ChatTemplate::Generic => &NO_MARKERS,
        }
    }

    /// Compute a Jinja template's allow-list against `tokenizer`.
    /// A built-in format's list is fixed, so this is a no-op for it.
    pub fn bind_special_tokens(&mut self, tokenizer: &AnyTokenizer) {
        if let ChatTemplate::Jinja(template) = self {
            template.bind_special_tokens(tokenizer);
        }
    }

    /// Whether the template takes `tools` and structured tool turns itself.
    /// A built-in format needs them flattened into message text.
    pub fn renders_tools_natively(&self) -> bool {
        matches!(self, ChatTemplate::Jinja(_))
    }

    /// One line naming the template and where it came from.
    pub fn describe(&self) -> String {
        match self {
            ChatTemplate::Llama3 => "builtin llama3".to_string(),
            ChatTemplate::MistralInstruct => "builtin mistral".to_string(),
            ChatTemplate::ChatML => "builtin chatml".to_string(),
            ChatTemplate::Phi3 => "builtin phi3".to_string(),
            ChatTemplate::Gemma => "builtin gemma".to_string(),
            ChatTemplate::DeepSeek => "builtin deepseek".to_string(),
            ChatTemplate::Jinja(template) => template.origin().label().to_string(),
            ChatTemplate::Generic => "builtin generic".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_detection() {
        assert_eq!(ChatTemplate::from_model_type("llama"), ChatTemplate::Llama3);
        assert_eq!(
            ChatTemplate::from_model_type("mistral"),
            ChatTemplate::MistralInstruct
        );
        assert_eq!(ChatTemplate::from_model_type("qwen2"), ChatTemplate::ChatML);
        assert_eq!(
            ChatTemplate::from_model_type("qwen35"),
            ChatTemplate::ChatML
        );
        assert_eq!(ChatTemplate::from_model_type("phi3"), ChatTemplate::Phi3);
        assert_eq!(ChatTemplate::from_model_type("gemma2"), ChatTemplate::Gemma);
        assert_eq!(
            ChatTemplate::from_model_type("unknown"),
            ChatTemplate::Generic
        );
    }

    #[test]
    fn test_model_type_aliases() {
        assert_eq!(ChatTemplate::from_model_type("yi"), ChatTemplate::ChatML);
        assert_eq!(
            ChatTemplate::from_model_type("internlm2"),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::from_model_type("qwen2_moe"),
            ChatTemplate::ChatML
        );
        assert_eq!(ChatTemplate::from_model_type("phi"), ChatTemplate::Phi3);
        assert_eq!(
            ChatTemplate::from_model_type("deepseek_v2"),
            ChatTemplate::DeepSeek
        );
        assert_eq!(
            ChatTemplate::from_model_type("starcoder2"),
            ChatTemplate::Generic
        );
    }

    #[test]
    fn test_from_name() {
        assert_eq!(ChatTemplate::from_name("llama3"), ChatTemplate::Llama3);
        assert_eq!(ChatTemplate::from_name("LLAMA"), ChatTemplate::Llama3);
        assert_eq!(
            ChatTemplate::from_name("mistral"),
            ChatTemplate::MistralInstruct
        );
        assert_eq!(ChatTemplate::from_name("chatml"), ChatTemplate::ChatML);
        assert_eq!(ChatTemplate::from_name("qwen"), ChatTemplate::ChatML);
        assert_eq!(ChatTemplate::from_name("phi3"), ChatTemplate::Phi3);
        assert_eq!(ChatTemplate::from_name("gemma"), ChatTemplate::Gemma);
        assert_eq!(ChatTemplate::from_name("deepseek"), ChatTemplate::DeepSeek);
        assert_eq!(ChatTemplate::from_name("raw"), ChatTemplate::Generic);
        assert_eq!(ChatTemplate::from_name("unknown"), ChatTemplate::Generic);
        assert_eq!(ChatTemplate::builtin("unknown"), None);
    }

    #[test]
    fn every_builtin_name_resolves() {
        for name in BUILTIN_NAMES {
            assert!(ChatTemplate::builtin(name).is_some(), "{name}");
        }
    }

    #[test]
    fn apply_opens_the_assistant_turn() {
        let messages = vec![ChatMessage::new("user", "Hi")];
        let result = ChatTemplate::ChatML
            .apply(&messages)
            .expect("builtin renders");
        assert_eq!(
            result,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(ChatTemplate::ChatML.describe(), "builtin chatml");
        assert!(!ChatTemplate::ChatML.renders_tools_natively());
    }

    #[test]
    fn jinja_renders_and_reports_origin() {
        let template = JinjaTemplate::new(
            "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}",
            super::super::jinja::TemplateOrigin::Inline,
            "",
            "",
        )
        .expect("compiles");
        let template = ChatTemplate::Jinja(template);
        let messages = vec![ChatMessage::new("user", "Hi")];
        assert_eq!(
            template.apply(&messages).expect("renders"),
            "<user>Hi<assistant>"
        );
        let options = RenderOptions {
            add_generation_prompt: false,
            ..RenderOptions::default()
        };
        assert_eq!(
            template.apply_with(&messages, &options).expect("renders"),
            "<user>Hi"
        );
        assert_eq!(template.describe(), "inline jinja");
        assert!(template.renders_tools_natively());
    }

    #[test]
    fn jinja_raise_exception_is_an_error_naming_the_template() {
        let template = JinjaTemplate::new(
            "{{ raise_exception('no system turn allowed') }}",
            super::super::jinja::TemplateOrigin::Inline,
            "",
            "",
        )
        .expect("compiles");
        let err = ChatTemplate::Jinja(template)
            .apply(&[])
            .expect_err("raise_exception fails the render");
        let text = err.to_string();
        assert!(text.starts_with("chat template (inline jinja):"), "{text}");
        assert!(text.contains("no system turn allowed"), "{text}");
    }

    #[test]
    fn jinja_syntax_error_is_reported_at_compile() {
        let err = JinjaTemplate::new(
            "{% for m in messages %}",
            super::super::jinja::TemplateOrigin::Inline,
            "",
            "",
        )
        .expect_err("an unclosed block does not compile");
        assert!(err.to_string().contains("inline jinja"));
    }
}
