//! The chat stream choices one generated token becomes: reasoning split,
//! tool calls, and finish.

use super::chat_stream::{ChatStreamChoice, ChatStreamDelta};
use super::chat_token_text::{split_token_text, TokenText};
use super::reasoning::ReasoningParser;
use super::streaming::StreamToken;
use super::tool_call_delta::call_deltas;
use crate::server::tool_parser::ToolCallParser;

/// Turns each stream token into the choices of its chunks.
pub(super) struct ChatChunker {
    reasoning: Option<ReasoningParser>,
    tools: Option<ToolCallParser>,
    /// Calls sent so far. The next call takes this index.
    calls: usize,
}

impl ChatChunker {
    pub(super) fn new(reasoning: Option<ReasoningParser>, tools: Option<ToolCallParser>) -> Self {
        Self {
            reasoning,
            tools,
            calls: 0,
        }
    }

    /// The choices `token` sends, in order. `token` carries no error.
    ///
    /// - The first choice carries the released text and the token's logprobs.
    ///   It is left out when a parser holds all the text back and the token
    ///   has no logprobs, unless the token is the final one with no call.
    /// - Each call released by this token follows, as [`call_deltas`] entries.
    /// - A final token that releases a call ends with an empty choice that
    ///   carries `finish_reason`. Otherwise the first choice carries it.
    /// - `finish_reason` is `tool_calls` once any call went out.
    pub(super) fn token(&mut self, token: StreamToken) -> Vec<ChatStreamChoice> {
        let is_final = token.finish_reason.is_some();
        let logprobs = token.logprobs();
        let TokenText {
            reasoning,
            content,
            calls,
        } = split_token_text(
            self.reasoning.as_mut(),
            self.tools.as_mut(),
            token.text,
            is_final,
        );
        let any_call = self.calls + calls.len() > 0;
        let finish = token
            .finish_reason
            .map(|r| if any_call { "tool_calls" } else { r.as_str() }.to_string());

        let held = reasoning.is_empty() && content.is_empty();
        let parsing = self.reasoning.is_some() || self.tools.is_some();
        let text = ChatStreamDelta {
            content: (!content.is_empty()).then_some(content),
            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
            ..ChatStreamDelta::default()
        };
        if calls.is_empty() {
            if is_final || !held || !parsing || logprobs.is_some() {
                return vec![ChatStreamChoice::new(text, finish, logprobs)];
            }
            return Vec::new();
        }

        let mut choices = Vec::new();
        if !held || logprobs.is_some() {
            choices.push(ChatStreamChoice::new(text, None, logprobs));
        }
        for call in calls {
            for delta in call_deltas(self.calls, call) {
                let delta = ChatStreamDelta {
                    tool_calls: Some(vec![delta]),
                    ..ChatStreamDelta::default()
                };
                choices.push(ChatStreamChoice::new(delta, None, None));
            }
            self.calls += 1;
        }
        if finish.is_some() {
            choices.push(ChatStreamChoice::new(
                ChatStreamDelta::default(),
                finish,
                None,
            ));
        }
        choices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::FinishReason;
    use crate::server::reasoning::ReasoningTags;
    use crate::server::streaming::sse_test::{entry, expected_entry, token};
    use crate::server::tool_calls::extract_tool_calls;
    use crate::server::tools::{Tool, ToolFunction};
    use serde_json::{json, Value};

    fn declared() -> Vec<Tool> {
        ["weather", "time"]
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

    fn chunker(reasoning: bool) -> ChatChunker {
        let reasoning = reasoning.then(|| ReasoningParser::new(ReasoningTags::THINK, false));
        ChatChunker::new(reasoning, Some(ToolCallParser::new(declared())))
    }

    /// Each choice, as JSON, of `pieces` streamed with the last finishing on `finish`.
    fn run(mut chunker: ChatChunker, pieces: &[&str], finish: FinishReason) -> Vec<Value> {
        pieces
            .iter()
            .enumerate()
            .flat_map(|(i, piece)| {
                let last = (i + 1 == pieces.len()).then_some(finish);
                chunker.token(StreamToken::text(piece.to_string(), last))
            })
            .map(|choice| serde_json::to_value(&choice).unwrap())
            .collect()
    }

    /// All `delta.content` text, in order.
    fn content(choices: &[Value]) -> String {
        choices
            .iter()
            .filter_map(|c| c["delta"]["content"].as_str())
            .collect()
    }

    /// `(index, id, name, joined arguments)` of each streamed call.
    fn calls(choices: &[Value]) -> Vec<(u64, String, String, String)> {
        let mut out: Vec<(u64, String, String, String)> = Vec::new();
        for delta in choices
            .iter()
            .filter_map(|c| c["delta"]["tool_calls"].as_array())
        {
            let delta = &delta[0];
            let index = delta["index"].as_u64().unwrap();
            if let Some(id) = delta["id"].as_str() {
                assert_eq!(delta["type"], "function", "{delta}");
                assert_eq!(delta["function"]["arguments"], "", "{delta}");
                let name = delta["function"]["name"].as_str().unwrap().to_string();
                out.push((index, id.to_string(), name, String::new()));
            } else {
                let call = out.last_mut().unwrap();
                assert_eq!(call.0, index);
                call.3
                    .push_str(delta["function"]["arguments"].as_str().unwrap());
            }
        }
        out
    }

    fn finish(choices: &[Value]) -> &Value {
        &choices.last().unwrap()["finish_reason"]
    }

    const CALL: &str = concat!(
        r#"<tool_call>{"name":"weather","arguments":{"city":"Paris","#,
        r#""note":"a long note that spans more than one fragment"}}</tool_call>"#,
    );

    #[test]
    fn content_streams_live_then_the_call_follows() {
        let pieces = [
            "Let me",
            " check.",
            "\n<tool",
            "_call>{\"name\":\"weather\",",
            "\"arguments\":{\"city\":\"Paris\"}}",
            "</tool_call>",
        ];
        let got = run(chunker(false), &pieces, FinishReason::Eos);
        assert_eq!(got[0]["delta"], json!({"content": "Let me"}));
        assert_eq!(got[1]["delta"], json!({"content": " check."}));
        assert_eq!(
            got.len(),
            2 + 1 + 1 + 1,
            "content, call head, one fragment, finish"
        );
        let streamed = calls(&got);
        assert_eq!(streamed.len(), 1);
        assert!(streamed[0].1.starts_with("call_"));
        assert_eq!(streamed[0].2, "weather");
        let args: Value = serde_json::from_str(&streamed[0].3).unwrap();
        assert_eq!(args, json!({"city": "Paris"}));
        assert_eq!(finish(&got), "tool_calls");
        assert_eq!(got.last().unwrap()["delta"], json!({}));
    }

    #[test]
    fn long_arguments_stream_as_fragments_that_rejoin() {
        let got = run(chunker(false), &[CALL], FinishReason::Stop);
        let fragments = got
            .iter()
            .filter_map(|c| c["delta"]["tool_calls"].get(0))
            .filter(|delta| delta.get("id").is_none())
            .count();
        assert!(fragments > 1, "{got:?}");
        let args: Value = serde_json::from_str(&calls(&got)[0].3).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn two_calls_take_indexes_0_and_1() {
        let text = concat!(
            r#"<tool_call>{"name":"weather","arguments":{}}</tool_call>"#,
            "\n",
            r#"<tool_call>{"name":"time","arguments":{"tz":"UTC"}}</tool_call>"#,
        );
        let got = run(
            chunker(false),
            &[&text[..40], &text[40..]],
            FinishReason::Stop,
        );
        let streamed = calls(&got);
        let summary: Vec<(u64, &str, &str)> = streamed
            .iter()
            .map(|c| (c.0, c.2.as_str(), c.3.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![(0, "weather", "{}"), (1, "time", r#"{"tz":"UTC"}"#)]
        );
        assert_ne!(streamed[0].1, streamed[1].1);
        assert_eq!(finish(&got), "tool_calls");
    }

    #[test]
    fn plain_content_with_tools_is_unchanged() {
        let pieces = ["Hello", " <b>", "world</b>", " { x }\n"];
        let got = run(chunker(false), &pieces, FinishReason::Stop);
        assert_eq!(content(&got), pieces.concat());
        assert!(calls(&got).is_empty());
        assert_eq!(finish(&got), "stop");
    }

    #[test]
    fn reasoning_then_a_call() {
        let pieces = [
            "<think>plan",
            "</think>\n",
            r#"{"name":"weather","arguments":{}}"#,
        ];
        let got = run(chunker(true), &pieces, FinishReason::Eos);
        let reasoning: String = got
            .iter()
            .filter_map(|c| c["delta"]["reasoning_content"].as_str())
            .collect();
        assert_eq!(reasoning, "plan");
        assert_eq!(content(&got), "");
        assert_eq!(calls(&got)[0].2, "weather");
        assert_eq!(finish(&got), "tool_calls");
    }

    #[test]
    fn a_forced_bare_call_streams_as_tool_calls() {
        let pieces = [r#"{"name":"#, r#""time","#, r#""arguments":{"tz":"UTC"}}"#];
        let got = run(chunker(false), &pieces, FinishReason::Stop);
        assert_eq!(content(&got), "");
        assert_eq!(calls(&got)[0].3, r#"{"tz":"UTC"}"#);
        assert_eq!(finish(&got), "tool_calls");
    }

    #[test]
    fn a_call_cut_by_length_or_a_stop_sequence_is_content() {
        for reason in [FinishReason::Length, FinishReason::Stop] {
            let pieces = ["Hi\n", r#"{"name":"weather","#, r#""argu"#];
            let got = run(chunker(false), &pieces, reason);
            assert_eq!(content(&got), pieces.concat());
            assert!(calls(&got).is_empty());
            assert_eq!(finish(&got), reason.as_str());
        }
    }

    #[test]
    fn a_call_stays_tool_calls_when_the_output_runs_on_to_length() {
        let pieces = [r#"{"name":"weather","arguments":{}}"#, " and more"];
        let got = run(chunker(false), &pieces, FinishReason::Length);
        assert_eq!(calls(&got).len(), 1);
        assert_eq!(content(&got), "");
        assert_eq!(finish(&got), "tool_calls");
    }

    #[test]
    fn an_undeclared_call_is_content() {
        let text = r#"<tool_call>{"name":"ghost","arguments":{}}</tool_call>"#;
        let got = run(
            chunker(false),
            &[&text[..20], &text[20..]],
            FinishReason::Stop,
        );
        assert_eq!(content(&got), text);
        assert!(calls(&got).is_empty());
        assert_eq!(finish(&got), "stop");
    }

    #[test]
    fn held_tokens_with_logprobs_send_an_empty_delta() {
        let pieces = [
            "Hi",
            " <tool_call>",
            r#"{"name":"weather","arguments":{}}"#,
            "</tool_call>",
        ];
        let mut chunker = chunker(false);
        let mut got = Vec::new();
        for (i, piece) in pieces.iter().enumerate() {
            let last = (i + 1 == pieces.len()).then_some(FinishReason::Eos);
            let choices: Vec<Value> = chunker
                .token(token(i as u32 + 1, piece, last))
                .iter()
                .map(|c| serde_json::to_value(c).unwrap())
                .collect();
            let chunk = json!({"choices": [choices[0].clone()]});
            assert_eq!(
                entry(&chunk),
                Some(&expected_entry(i as u32 + 1, piece)),
                "{piece}"
            );
            assert!(choices[1..].iter().all(|c| c.get("logprobs").is_none()));
            got.extend(choices);
        }
        assert_eq!(got[1]["delta"], json!({}));
        assert_eq!(got[2]["delta"], json!({}));
        assert_eq!(got[3]["delta"], json!({}));
        assert_eq!(calls(&got).len(), 1);
        assert_eq!(finish(&got), "tool_calls");
    }

    #[test]
    fn a_stream_and_a_complete_response_agree() {
        let texts = [
            concat!(
                "Sure.\n",
                r#"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#,
                " trailing",
            ),
            concat!(
                r#"{"name":"time","arguments":{}}"#,
                "\n",
                r#"{"name":"weather","arguments":{"a":[1,2]}}"#,
            ),
            "No call here, just {braces} and <b>tags</b>.",
            concat!(
                r#"<tool_call>{"name":"ghost","arguments":{}}</tool_call>"#,
                "\n<tool_call><function=time></function></tool_call>",
            ),
            "Cut short\n{\"name\":\"weather\",\"argu",
        ];
        for text in texts {
            let pieces: Vec<String> = text.chars().map(String::from).collect();
            let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
            let got = run(chunker(false), &pieces, FinishReason::Stop);
            let (want_content, want_calls) = extract_tool_calls(text, &declared());
            let want_calls = want_calls.unwrap_or_default();
            let streamed = calls(&got);
            let summary: Vec<(&str, &str)> = streamed
                .iter()
                .map(|c| (c.2.as_str(), c.3.as_str()))
                .collect();
            let want: Vec<(&str, &str)> = want_calls
                .iter()
                .map(|c| (c.function.name.as_str(), c.function.arguments.as_str()))
                .collect();
            assert_eq!(summary, want, "{text}");
            // A complete response trims the content before a call. A stream
            // has sent its leading text by then.
            let streamed_content = content(&got);
            let want_content = want_content.unwrap_or_default();
            if want.is_empty() {
                assert_eq!(streamed_content, want_content, "{text}");
            } else {
                assert_eq!(streamed_content.trim(), want_content, "{text}");
            }
            let want_finish = if want.is_empty() {
                "stop"
            } else {
                "tool_calls"
            };
            assert_eq!(finish(&got), want_finish, "{text}");
        }
    }
}
