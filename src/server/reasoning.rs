//! Split of model output into reasoning and answer.
//!
//! A reasoning model brackets its chain of thought in a tag pair. The
//! template can open the block itself at the end of the prompt, or the model
//! can emit the open tag as its first output. The parser handles both, and
//! feeds one piece at a time so a stream stays incremental.

use crate::config::ReasoningFormat;

/// Tag pair that brackets reasoning in model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningTags {
    pub open: &'static str,
    pub close: &'static str,
}

impl ReasoningTags {
    /// The `<think>` pair.
    pub const THINK: ReasoningTags = ReasoningTags {
        open: "<think>",
        close: "</think>",
    };
}

impl Default for ReasoningTags {
    fn default() -> Self {
        Self::THINK
    }
}

/// True when `prompt` ends inside an open reasoning block: its last open tag
/// comes after its last close tag.
pub fn prompt_opens_reasoning(prompt: &str, tags: ReasoningTags) -> bool {
    match (prompt.rfind(tags.open), prompt.rfind(tags.close)) {
        (Some(open), Some(close)) => open > close,
        (Some(_), None) => true,
        (None, _) => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Before the first non-space output: an open tag here starts reasoning.
    Start,
    /// Inside the reasoning block, until the close tag.
    Reasoning,
    /// Right after the close tag: leading whitespace is dropped.
    AfterClose,
    /// The answer.
    Content,
}

/// Incremental splitter. Each fed piece yields `(reasoning, content)`; only a
/// partial tag prefix is held back between calls.
#[derive(Debug, Clone)]
pub struct ReasoningParser {
    tags: ReasoningTags,
    state: State,
    held: String,
}

impl ReasoningParser {
    /// `prompt_open` is [`prompt_opens_reasoning`] for the rendered prompt.
    pub fn new(tags: ReasoningTags, prompt_open: bool) -> Self {
        Self {
            tags,
            state: if prompt_open {
                State::Reasoning
            } else {
                State::Start
            },
            held: String::new(),
        }
    }

    /// A parser for `format`, or `None` when the output stays raw.
    pub fn for_output(format: ReasoningFormat, prompt: &str) -> Option<Self> {
        match format {
            ReasoningFormat::Deepseek => {
                let tags = ReasoningTags::default();
                Some(Self::new(tags, prompt_opens_reasoning(prompt, tags)))
            }
            ReasoningFormat::None => None,
        }
    }

    /// Feed one piece of output.
    pub fn feed(&mut self, piece: &str) -> (String, String) {
        let mut reasoning = String::new();
        let mut content = String::new();
        self.held.push_str(piece);
        loop {
            match self.state {
                State::Start => {
                    let trimmed = self.held.trim_start();
                    if let Some(rest) = trimmed.strip_prefix(self.tags.open) {
                        self.held = rest.to_string();
                        self.state = State::Reasoning;
                    } else if self.tags.open.starts_with(trimmed) {
                        return (reasoning, content);
                    } else {
                        self.state = State::Content;
                    }
                }
                State::Reasoning => {
                    if let Some(pos) = self.held.find(self.tags.close) {
                        reasoning.push_str(&self.held[..pos]);
                        self.held.drain(..pos + self.tags.close.len());
                        self.state = State::AfterClose;
                    } else {
                        let keep = partial_prefix_len(&self.held, self.tags.close);
                        let emit_to = self.held.len() - keep;
                        reasoning.push_str(&self.held[..emit_to]);
                        self.held.drain(..emit_to);
                        return (reasoning, content);
                    }
                }
                State::AfterClose => {
                    let trimmed = self.held.trim_start();
                    if trimmed.is_empty() {
                        self.held.clear();
                        return (reasoning, content);
                    }
                    self.held = trimmed.to_string();
                    self.state = State::Content;
                }
                State::Content => {
                    content.push_str(&self.held);
                    self.held.clear();
                    return (reasoning, content);
                }
            }
        }
    }

    /// Release what is held back once the output ends. An unclosed block
    /// counts as reasoning.
    pub fn finish(&mut self) -> (String, String) {
        let held = std::mem::take(&mut self.held);
        match self.state {
            State::Reasoning => (held, String::new()),
            State::Start | State::Content => (String::new(), held),
            State::AfterClose => (String::new(), String::new()),
        }
    }
}

/// Length of the longest suffix of `text` that is a strict prefix of `tag`.
fn partial_prefix_len(text: &str, tag: &str) -> usize {
    let max = text.len().min(tag.len() - 1);
    (1..=max)
        .rev()
        .find(|&k| text.is_char_boundary(text.len() - k) && text.ends_with(&tag[..k]))
        .unwrap_or(0)
}

/// Split a complete output for `format`, or pass it through unsplit when
/// the format carries no reasoning tags.
pub fn split_reasoning_for_format(
    format: ReasoningFormat,
    prompt: &str,
    text: String,
) -> (Option<String>, String) {
    match format {
        ReasoningFormat::Deepseek => split_reasoning(prompt, &text, ReasoningTags::default()),
        ReasoningFormat::None => (None, text),
    }
}

/// Split a complete output. Reasoning is `None` when empty.
pub fn split_reasoning(prompt: &str, text: &str, tags: ReasoningTags) -> (Option<String>, String) {
    let mut parser = ReasoningParser::new(tags, prompt_opens_reasoning(prompt, tags));
    let (mut reasoning, mut content) = parser.feed(text);
    let (tail_reasoning, tail_content) = parser.finish();
    reasoning.push_str(&tail_reasoning);
    content.push_str(&tail_content);
    let reasoning = if reasoning.is_empty() {
        None
    } else {
        Some(reasoning)
    };
    (reasoning, content)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAGS: ReasoningTags = ReasoningTags::THINK;
    const OPEN_PROMPT: &str = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
    const CLOSED_PROMPT: &str =
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    fn stream(prompt_open: bool, pieces: &[&str]) -> (String, String) {
        let mut parser = ReasoningParser::new(TAGS, prompt_open);
        let (mut reasoning, mut content) = (String::new(), String::new());
        for piece in pieces {
            let (r, c) = parser.feed(piece);
            reasoning.push_str(&r);
            content.push_str(&c);
        }
        let (r, c) = parser.finish();
        reasoning.push_str(&r);
        content.push_str(&c);
        (reasoning, content)
    }

    #[test]
    fn prompt_tail_detection() {
        assert!(prompt_opens_reasoning(OPEN_PROMPT, TAGS));
        assert!(!prompt_opens_reasoning(CLOSED_PROMPT, TAGS));
        assert!(!prompt_opens_reasoning("plain prompt", TAGS));
        let prior_turn = "<think>a</think>answer<|im_end|>\n<|im_start|>assistant\n<think>\n";
        assert!(prompt_opens_reasoning(prior_turn, TAGS));
    }

    #[test]
    fn prompt_opens_block() {
        let (reasoning, content) =
            split_reasoning(OPEN_PROMPT, "let me think\n</think>\n\nThe answer.", TAGS);
        assert_eq!(reasoning.as_deref(), Some("let me think\n"));
        assert_eq!(content, "The answer.");
    }

    #[test]
    fn model_emits_open_tag() {
        let (reasoning, content) = split_reasoning("prompt", "<think>hmm</think>\n\nAnswer", TAGS);
        assert_eq!(reasoning.as_deref(), Some("hmm"));
        assert_eq!(content, "Answer");
        let (reasoning, content) = stream(false, &["\n<th", "ink>", "x", "</think>", "y"]);
        assert_eq!(reasoning, "x");
        assert_eq!(content, "y");
    }

    #[test]
    fn no_reasoning_at_all() {
        let (reasoning, content) = split_reasoning("prompt", "Just an answer.", TAGS);
        assert_eq!(reasoning, None);
        assert_eq!(content, "Just an answer.");
        let (reasoning, content) = stream(false, &["<b>", "bold</b>"]);
        assert_eq!(reasoning, "");
        assert_eq!(content, "<b>bold</b>");
    }

    #[test]
    fn close_tag_split_across_deltas() {
        let mut parser = ReasoningParser::new(TAGS, true);
        assert_eq!(parser.feed("abc"), ("abc".to_string(), String::new()));
        assert_eq!(parser.feed("</th"), (String::new(), String::new()));
        assert_eq!(parser.feed("ink>"), (String::new(), String::new()));
        assert_eq!(parser.feed("\n\n"), (String::new(), String::new()));
        assert_eq!(parser.feed("Hi"), (String::new(), "Hi".to_string()));
        assert_eq!(parser.finish(), (String::new(), String::new()));
    }

    #[test]
    fn false_prefix_is_released_as_reasoning() {
        let (reasoning, content) = stream(true, &["a <", "/b> c", "</think>d"]);
        assert_eq!(reasoning, "a </b> c");
        assert_eq!(content, "d");
    }

    #[test]
    fn whitespace_after_close_is_dropped() {
        let (reasoning, content) = stream(true, &["r</think>", "\n", "\n", "A\n\nB"]);
        assert_eq!(reasoning, "r");
        assert_eq!(content, "A\n\nB");
    }

    #[test]
    fn unclosed_block_is_reasoning() {
        let (reasoning, content) = stream(true, &["still ", "thinking</th"]);
        assert_eq!(reasoning, "still thinking</th");
        assert_eq!(content, "");
    }

    #[test]
    fn thinking_disabled_by_template() {
        let (reasoning, content) = split_reasoning(CLOSED_PROMPT, "Plain answer.", TAGS);
        assert_eq!(reasoning, None);
        assert_eq!(content, "Plain answer.");
    }

    #[test]
    fn format_none_yields_no_parser() {
        assert!(ReasoningParser::for_output(ReasoningFormat::None, OPEN_PROMPT).is_none());
        assert!(ReasoningParser::for_output(ReasoningFormat::Deepseek, OPEN_PROMPT).is_some());
    }
}
