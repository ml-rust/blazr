//! The parser state machine: text scanning and the release of content and calls.

use crate::server::json_scan::ObjectScan;
use crate::server::tools::{Tool, ToolCall};

pub(super) const TAG_OPEN: &str = "<tool_call>";
pub(super) const TAG_CLOSE: &str = "</tool_call>";

/// What one fed piece released.
#[derive(Debug, Default)]
pub struct ToolEvents {
    /// Text that is no call, in output order.
    pub content: String,
    /// Calls whose boundary arrived, in output order.
    pub calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum State {
    /// Before the first call: text is content.
    Text,
    /// After the first call: text between and after calls is dropped.
    AfterCall,
    /// Inside a `<tool_call>` block, until `</tool_call>`.
    InTag,
    /// Inside a bare JSON object, until its braces balance.
    InBare,
}

/// Scan state of the candidate call the parser holds.
#[derive(Debug, Clone, Default)]
pub(super) struct Candidate {
    /// Whitespace before the candidate. A call drops it.
    pub(super) gap: String,
    /// The end scan of the candidate's JSON object.
    pub(super) scan: ObjectScan,
    /// Byte offset where the next `</tool_call>` search starts.
    pub(super) searched: usize,
    /// The bare object's first key is `name` or `arguments`.
    pub(super) key_ok: bool,
}

/// Incremental parser that splits model output into content and tool calls.
///
/// Spellings:
/// - `<tool_call>{"name": "fn", "arguments": {...}}</tool_call>`, anywhere.
/// - `<tool_call><function=fn><parameter=k>v</parameter></function></tool_call>`, anywhere.
/// - `{"name": "fn", "arguments": {...}}` as the first non-whitespace of the
///   output or the first non-whitespace of a line. A bare object mid-line is content.
///
/// Rules:
/// - Whitespace right before a call is dropped.
/// - After the first call, text between and after calls is dropped. Further
///   calls still parse.
/// - A call whose name no tool in `declared` has stays content as raw text.
///   An empty `declared` admits every call.
/// - A bare object whose first key is not `name` or `arguments` is content
///   as soon as that key arrives.
/// - [`ToolCallParser::finish`] releases an unterminated block or object as content.
#[derive(Debug, Clone)]
pub struct ToolCallParser {
    pub(super) declared: Vec<Tool>,
    pub(super) state: State,
    /// Text not yet released. In a candidate state it starts with the trigger.
    pub(super) held: String,
    /// Calls released so far.
    emitted: usize,
    /// The last released or dropped non-space text ends a line, or none came yet.
    pub(super) line_start: bool,
    pub(super) candidate: Candidate,
}

impl ToolCallParser {
    /// A parser that admits calls to `declared`.
    pub fn new(declared: Vec<Tool>) -> Self {
        Self {
            declared,
            state: State::Text,
            held: String::new(),
            emitted: 0,
            line_start: true,
            candidate: Candidate::default(),
        }
    }

    /// Feed one piece of output.
    pub fn feed(&mut self, piece: &str) -> ToolEvents {
        let mut out = ToolEvents::default();
        self.held.push_str(piece);
        self.run(&mut out, false);
        out
    }

    /// Release what is held back once the output ends.
    pub fn finish(&mut self) -> ToolEvents {
        let mut out = ToolEvents::default();
        self.run(&mut out, true);
        out
    }

    /// Step the state machine until it needs more input. With `end` set,
    /// nothing stays held.
    fn run(&mut self, out: &mut ToolEvents, end: bool) {
        loop {
            let progressed = match self.state {
                State::Text | State::AfterCall => self.scan_text(out, end),
                State::InTag => self.step_tag(out, end),
                State::InBare => self.step_bare(out, end),
            };
            if !progressed {
                break;
            }
        }
    }

    /// Release the gap and the first `len` bytes of the candidate as text,
    /// then scan the rest as text.
    pub(super) fn reject(&mut self, len: usize, out: &mut ToolEvents) {
        let mut text = std::mem::take(&mut self.candidate.gap);
        text.push_str(&self.held[..len]);
        self.held.drain(..len);
        self.release(&text, out);
        self.resume();
    }

    /// Send `call`, whose text is the first `len` bytes of the candidate.
    pub(super) fn emit(&mut self, call: ToolCall, len: usize, out: &mut ToolEvents) {
        out.calls.push(call);
        self.emitted += 1;
        self.candidate = Candidate::default();
        self.held.drain(..len);
        self.line_start = true;
        self.resume();
    }

    /// Return to text scanning.
    fn resume(&mut self) {
        self.state = if self.emitted == 0 {
            State::Text
        } else {
            State::AfterCall
        };
    }

    /// Release `text` as content before the first call, or drop it after.
    pub(super) fn release(&mut self, text: &str, out: &mut ToolEvents) {
        if let Some(last) = text
            .chars()
            .rev()
            .find(|c| *c == '\n' || !c.is_whitespace())
        {
            self.line_start = last == '\n';
        }
        if self.emitted == 0 {
            out.content.push_str(text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::tools::ToolFunction;

    fn tools(names: &[&str]) -> Vec<Tool> {
        names
            .iter()
            .map(|name| Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: name.to_string(),
                    description: None,
                    parameters: None,
                },
            })
            .collect()
    }

    /// Content and `(name, arguments)` pairs of `text`, fed in pieces of `step` chars.
    fn parse(text: &str, step: usize, declared: &[&str]) -> (String, Vec<(String, String)>) {
        let mut parser = ToolCallParser::new(tools(declared));
        let chars: Vec<char> = text.chars().collect();
        let mut out = ToolEvents::default();
        for piece in chars.chunks(step) {
            let got = parser.feed(&piece.iter().collect::<String>());
            out.content.push_str(&got.content);
            out.calls.extend(got.calls);
        }
        let tail = parser.finish();
        out.content.push_str(&tail.content);
        out.calls.extend(tail.calls);
        let calls = out
            .calls
            .into_iter()
            .map(|c| (c.function.name, c.function.arguments))
            .collect();
        (out.content, calls)
    }

    /// The result of `text` fed whole, after a check that char-by-char feeding gives the same.
    fn both(text: &str, declared: &[&str]) -> (String, Vec<(String, String)>) {
        let whole = parse(text, text.chars().count().max(1), declared);
        assert_eq!(
            parse(text, 1, declared),
            whole,
            "char-by-char vs whole: {text:?}"
        );
        assert_eq!(
            parse(text, 3, declared),
            whole,
            "3-char pieces vs whole: {text:?}"
        );
        whole
    }

    fn call(name: &str, arguments: &str) -> (String, String) {
        (name.to_string(), arguments.to_string())
    }

    #[test]
    fn a_tagged_json_call_follows_its_content() {
        let text = "Checking.\n<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
        let (content, calls) = both(text, &["weather"]);
        assert_eq!(content, "Checking.");
        assert_eq!(calls, vec![call("weather", r#"{"city":"Paris"}"#)]);
    }

    #[test]
    fn an_xml_call_parses() {
        let text = "<tool_call>\n<function=weather>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = both(text, &["weather"]);
        assert_eq!(content, "");
        assert_eq!(calls, vec![call("weather", r#"{"days":3}"#)]);
    }

    #[test]
    fn a_bare_call_at_the_start_or_a_line_start_parses() {
        let start = "  {\"name\": \"weather\", \"arguments\": {}}";
        assert_eq!(
            both(start, &["weather"]),
            (String::new(), vec![call("weather", "{}")])
        );
        let line = "Sure.\n  {\"name\": \"weather\", \"arguments\": {}}";
        assert_eq!(
            both(line, &["weather"]),
            ("Sure.".to_string(), vec![call("weather", "{}")])
        );
    }

    #[test]
    fn a_bare_object_mid_line_is_content() {
        let text = "Use {\"name\": \"weather\", \"arguments\": {}} later.";
        assert_eq!(both(text, &["weather"]), (text.to_string(), vec![]));
    }

    #[test]
    fn parallel_calls_keep_their_order() {
        let tagged = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\n<tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>";
        let bare = "{\"name\":\"a\",\"arguments\":{}}\n{\"name\":\"b\",\"arguments\":{\"x\":1}}";
        for text in [tagged, bare] {
            let (content, calls) = both(text, &["a", "b"]);
            assert_eq!(content, "", "{text}");
            assert_eq!(
                calls,
                vec![call("a", "{}"), call("b", r#"{"x":1}"#)],
                "{text}"
            );
        }
    }

    #[test]
    fn an_undeclared_call_stays_content_and_a_later_declared_one_parses() {
        let ghost = "<tool_call>{\"name\":\"ghost\",\"arguments\":{}}</tool_call>";
        let text =
            format!("{ghost}\n<tool_call>{{\"name\":\"known\",\"arguments\":{{}}}}</tool_call>");
        let (content, calls) = both(&text, &["known"]);
        assert_eq!(content, ghost);
        assert_eq!(calls, vec![call("known", "{}")]);
        let bare = "{\"name\":\"ghost\",\"arguments\":{}}";
        assert_eq!(both(bare, &["known"]), (bare.to_string(), vec![]));
    }

    #[test]
    fn malformed_and_unterminated_calls_are_content() {
        for text in [
            "<tool_call>not json</tool_call>",
            "<tool_call>{\"name\":\"weather\",\"argu",
            "<tool_call>{\"name\":\"weather\",\"arguments\":{}}",
            "{\"name\":\"weather\",\"arguments\":{\"city\":\"Par",
            "{\"id\": 1, \"name\":\"weather\",\"arguments\":{}}",
            "<tool_c",
        ] {
            assert_eq!(
                both(text, &["weather"]),
                (text.to_string(), vec![]),
                "{text}"
            );
        }
    }

    #[test]
    fn a_false_tag_prefix_is_released() {
        for text in ["a <tool_x> b", "<b>bold</b>", "x <", "1 < 2 and 3 <tool"] {
            assert_eq!(
                both(text, &["weather"]),
                (text.to_string(), vec![]),
                "{text}"
            );
        }
    }

    #[test]
    fn whitespace_before_a_call_is_trimmed_and_kept_otherwise() {
        let text = "Hi \n\n<tool_call>{\"name\":\"w\",\"arguments\":{}}</tool_call>";
        assert_eq!(
            both(text, &["w"]),
            ("Hi".to_string(), vec![call("w", "{}")])
        );
        assert_eq!(both("Hi \n\n", &["w"]), ("Hi \n\n".to_string(), vec![]));
    }

    #[test]
    fn text_after_the_first_call_is_dropped() {
        let text = "A\n<tool_call>{\"name\":\"w\",\"arguments\":{}}</tool_call> done. <tool_x> {\"id\":1} <tool_call>junk";
        assert_eq!(both(text, &["w"]), ("A".to_string(), vec![call("w", "{}")]));
    }

    #[test]
    fn delimiters_inside_json_strings_do_not_end_the_call() {
        let text = "<tool_call>{\"name\":\"w\",\"arguments\":{\"s\":\"} { </tool_call> \\\" }\"}}</tool_call>";
        let (content, calls) = both(text, &["w"]);
        assert_eq!(content, "");
        assert_eq!(calls, vec![call("w", r#"{"s":"} { </tool_call> \" }"}"#)]);
        let bare = "{\"name\":\"w\",\"arguments\":{\"s\":\"}\\n{\"}}";
        assert_eq!(both(bare, &["w"]).1, vec![call("w", r#"{"s":"}\n{"}"#)]);
    }

    #[test]
    fn multibyte_text_and_arguments_survive() {
        let text =
            "Caf\u{e9} \u{1f600}\n{\"name\":\"w\",\"arguments\":{\"q\":\"\u{4e2d}\u{6587}\"}}";
        let (content, calls) = both(text, &["w"]);
        assert_eq!(content, "Caf\u{e9} \u{1f600}");
        assert_eq!(calls, vec![call("w", "{\"q\":\"\u{4e2d}\u{6587}\"}")]);
    }

    #[test]
    fn a_non_call_object_is_released_at_its_first_key() {
        let mut parser = ToolCallParser::new(tools(&["w"]));
        assert_eq!(parser.feed("{\"").content, "");
        assert_eq!(parser.feed("id\": ").content, "{\"id\":");
        assert_eq!(parser.finish().content, " ");
    }

    #[test]
    fn content_streams_before_a_call_arrives() {
        let mut parser = ToolCallParser::new(tools(&["w"]));
        assert_eq!(parser.feed("Hello there").content, "Hello there");
        assert_eq!(parser.feed(" <tool").content, "");
        let got = parser.feed("_call>{\"name\":\"w\",\"arguments\":{}}");
        assert!(got.content.is_empty() && got.calls.is_empty());
        let got = parser.feed("</tool_call>");
        assert_eq!(got.calls.len(), 1);
        assert!(parser.finish().content.is_empty());
    }
}
