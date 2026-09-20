//! Built-in chat formats and the special tokens each one writes.

use std::sync::LazyLock;

use splintr::FxHashSet;

use super::message::ChatMessage;

/// Collect marker strings into the set `SpecialMode::Allow` borrows.
fn marker_set(markers: &[&str]) -> FxHashSet<String> {
    markers.iter().map(|m| (*m).to_string()).collect()
}

/// The empty allow-list: no special token may be matched at all.
///
/// Used for caller-supplied message content, which the server never intends
/// to carry control tokens, and for the generic format, which emits none.
pub static NO_MARKERS: LazyLock<FxHashSet<String>> = LazyLock::new(FxHashSet::default);

pub static LLAMA3_MARKERS: LazyLock<FxHashSet<String>> = LazyLock::new(|| {
    marker_set(&[
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ])
});

pub static MISTRAL_MARKERS: LazyLock<FxHashSet<String>> =
    LazyLock::new(|| marker_set(&["[INST]", "[/INST]", "</s>"]));

pub static CHATML_MARKERS: LazyLock<FxHashSet<String>> =
    LazyLock::new(|| marker_set(&["<|im_start|>", "<|im_end|>"]));

pub static PHI3_MARKERS: LazyLock<FxHashSet<String>> =
    LazyLock::new(|| marker_set(&["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]));

pub static GEMMA_MARKERS: LazyLock<FxHashSet<String>> =
    LazyLock::new(|| marker_set(&["<start_of_turn>", "<end_of_turn>"]));

pub static DEEPSEEK_MARKERS: LazyLock<FxHashSet<String>> = LazyLock::new(|| {
    marker_set(&[
        "<|begin▁of▁sentence|>",
        "<|end▁of▁sentence|>",
        "<|User|>",
        "<|Assistant|>",
    ])
});

/// Llama 3 format
pub fn format_llama3(messages: &[ChatMessage]) -> String {
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

/// Mistral / Llama 2 `[INST]` format
pub fn format_mistral(messages: &[ChatMessage]) -> String {
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
pub fn format_chatml(messages: &[ChatMessage]) -> String {
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
pub fn format_phi3(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        prompt.push_str(&format!("<|{}|>\n{}<|end|>\n", msg.role, msg.content));
    }

    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Gemma format
pub fn format_gemma(messages: &[ChatMessage]) -> String {
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
pub fn format_deepseek(messages: &[ChatMessage]) -> String {
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
pub fn format_generic(messages: &[ChatMessage]) -> String {
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
            .map(|(role, content)| ChatMessage::new(*role, *content))
            .collect()
    }

    #[test]
    fn test_llama3_format() {
        let messages = msgs(&[("system", "You are helpful."), ("user", "Hello")]);
        let result = format_llama3(&messages);
        assert!(result.starts_with("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(result.contains("You are helpful.<|eot_id|>"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_chatml_format() {
        let messages = msgs(&[("user", "Hi")]);
        let result = format_chatml(&messages);
        assert!(result.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_mistral_format() {
        let messages = msgs(&[("system", "Be concise."), ("user", "Hello")]);
        let result = format_mistral(&messages);
        assert!(result.contains("[INST] Be concise.\n\nHello [/INST]"));
    }

    #[test]
    fn test_phi3_format() {
        let messages = msgs(&[("user", "Hello")]);
        let result = format_phi3(&messages);
        assert!(result.contains("<|user|>\nHello<|end|>"));
        assert!(result.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_gemma_format() {
        let messages = msgs(&[("user", "Hello")]);
        let result = format_gemma(&messages);
        assert!(result.contains("<start_of_turn>user\nHello<end_of_turn>"));
        assert!(result.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_deepseek_format() {
        let messages = msgs(&[("system", "You are a helpful assistant."), ("user", "Hi")]);
        let result = format_deepseek(&messages);
        assert!(result.starts_with("<|begin▁of▁sentence|>"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("<|User|>Hi"));
        assert!(result.ends_with("<|Assistant|>"));
    }

    #[test]
    fn test_deepseek_multi_turn() {
        let messages = msgs(&[
            ("user", "Hello"),
            ("assistant", "Hi there!"),
            ("user", "How are you?"),
        ]);
        let result = format_deepseek(&messages);
        assert!(result.contains("<|User|>Hello"));
        assert!(result.contains("<|Assistant|>Hi there!<|end▁of▁sentence|>"));
        assert!(result.contains("<|User|>How are you?"));
        assert!(result.ends_with("<|Assistant|>"));
    }

    #[test]
    fn test_generic_format() {
        let messages = msgs(&[("system", "Be brief."), ("user", "Hi")]);
        let result = format_generic(&messages);
        assert_eq!(result, "system: Be brief.\nuser: Hi\nassistant: ");
    }

    #[test]
    fn test_multi_turn_llama3() {
        let messages = msgs(&[
            ("user", "Hello"),
            ("assistant", "Hi! How can I help?"),
            ("user", "What is 2+2?"),
        ]);
        let result = format_llama3(&messages);
        assert!(result.contains("Hello<|eot_id|>"));
        assert!(result.contains("Hi! How can I help?<|eot_id|>"));
        assert!(result.contains("What is 2+2?<|eot_id|>"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_multi_turn_chatml() {
        let messages = msgs(&[
            ("system", "You are helpful."),
            ("user", "Hello"),
            ("assistant", "Hi!"),
            ("user", "Bye"),
        ]);
        let result = format_chatml(&messages);
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\nHi!<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nBye<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_mistral_multi_turn() {
        let messages = msgs(&[
            ("user", "Hello"),
            ("assistant", "Hi!"),
            ("user", "How are you?"),
        ]);
        let result = format_mistral(&messages);
        assert!(result.contains("[INST] Hello [/INST]"));
        assert!(result.contains(" Hi!</s>"));
        assert!(result.contains("[INST] How are you? [/INST]"));
    }

    #[test]
    fn test_gemma_role_mapping() {
        let messages = msgs(&[("user", "Hello"), ("assistant", "Hi!")]);
        let result = format_gemma(&messages);
        assert!(result.contains("<start_of_turn>user\nHello<end_of_turn>"));
        assert!(result.contains("<start_of_turn>model\nHi!<end_of_turn>"));
        assert!(result.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_empty_messages() {
        let messages: Vec<ChatMessage> = vec![];
        assert!(format_llama3(&messages).contains("assistant"));
        assert!(format_chatml(&messages).contains("assistant"));
        assert_eq!(format_generic(&messages), "assistant: ");
    }
}
