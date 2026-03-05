//! Think mode extraction for reasoning models (DeepSeek-R1, QwQ, etc.)
//!
//! Models that emit `<think>...</think>` blocks embed chain-of-thought reasoning
//! in the generated text. This module extracts those blocks so they can be
//! returned separately from the visible assistant content.

/// Extracted thinking content from a generation result
#[derive(Debug, Clone)]
pub struct ThinkResult {
    /// The visible content (with `<think>` blocks removed)
    pub content: String,
    /// The extracted thinking blocks (inner text only, in order)
    pub thinking: Vec<String>,
}

/// Extract `<think>...</think>` blocks from generated text.
///
/// Returns a `ThinkResult` with the thinking blocks separated from the
/// visible content. Handles multiple blocks, nested-looking text inside
/// blocks, and incomplete trailing blocks.
pub fn extract_thinking(text: &str) -> ThinkResult {
    let mut content = String::with_capacity(text.len());
    let mut thinking = Vec::new();
    let mut remaining = text;

    loop {
        match remaining.find("<think>") {
            Some(start) => {
                // Append text before the tag
                content.push_str(&remaining[..start]);

                let after_open = &remaining[start + 7..]; // len("<think>") == 7
                match after_open.find("</think>") {
                    Some(end) => {
                        let block = after_open[..end].trim().to_string();
                        if !block.is_empty() {
                            thinking.push(block);
                        }
                        remaining = &after_open[end + 8..]; // len("</think>") == 8
                    }
                    None => {
                        // Incomplete trailing block — treat as thinking
                        let block = after_open.trim().to_string();
                        if !block.is_empty() {
                            thinking.push(block);
                        }
                        remaining = "";
                    }
                }
            }
            None => {
                content.push_str(remaining);
                break;
            }
        }
    }

    ThinkResult { content, thinking }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_thinking() {
        let result = extract_thinking("Hello, world!");
        assert_eq!(result.content, "Hello, world!");
        assert!(result.thinking.is_empty());
    }

    #[test]
    fn test_single_think_block() {
        let text = "<think>Let me reason about this...</think>The answer is 42.";
        let result = extract_thinking(text);
        assert_eq!(result.content, "The answer is 42.");
        assert_eq!(result.thinking, vec!["Let me reason about this..."]);
    }

    #[test]
    fn test_multiple_think_blocks() {
        let text = "<think>First thought</think>Part 1. <think>Second thought</think>Part 2.";
        let result = extract_thinking(text);
        assert_eq!(result.content, "Part 1. Part 2.");
        assert_eq!(result.thinking.len(), 2);
        assert_eq!(result.thinking[0], "First thought");
        assert_eq!(result.thinking[1], "Second thought");
    }

    #[test]
    fn test_incomplete_trailing_block() {
        let text = "Some text<think>still thinking...";
        let result = extract_thinking(text);
        assert_eq!(result.content, "Some text");
        assert_eq!(result.thinking, vec!["still thinking..."]);
    }

    #[test]
    fn test_empty_think_block() {
        let text = "<think></think>Content";
        let result = extract_thinking(text);
        assert_eq!(result.content, "Content");
        assert!(result.thinking.is_empty());
    }
}
