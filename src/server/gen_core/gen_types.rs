//! Shared response types and utility functions for generation endpoints.
//!
//! Contains OpenAI-compatible response structs, logprob types, error helpers,
//! and validation utilities used across completion, chat, and infill handlers.

use boostr::inference::grammar::json_schema::unwrap_response_format;
use serde::{Deserialize, Serialize};

use crate::config::Constraint;
use crate::engine::{parse_keep_alive, GeneratedToken, TokenLogprob};
use crate::server::handlers::AppState;

pub use super::error_responses::{
    chat_prompt_error_response, error_response, error_response_with_code, overloaded_response,
    policy_error_response,
};

/// OpenAI response_format parameter
#[derive(Deserialize, Default, Clone)]
pub struct ResponseFormat {
    /// "text" (default), "json_object", or "json_schema"
    #[serde(default = "default_response_format_type")]
    pub r#type: String,
    /// JSON schema for structured output (used when type = "json_schema")
    #[serde(default)]
    pub json_schema: Option<serde_json::Value>,
}

fn default_response_format_type() -> String {
    "text".to_string()
}

/// Validate generation parameters, return error message if invalid
pub fn validate_generation_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
) -> Result<(), String> {
    if let Some(t) = temperature {
        if !(0.0..=2.0).contains(&t) {
            return Err(format!("temperature must be between 0 and 2, got {}", t));
        }
    }
    if let Some(p) = top_p {
        if !(0.0..=1.0).contains(&p) {
            return Err(format!("top_p must be between 0 and 1, got {}", p));
        }
    }
    if let Some(m) = max_tokens {
        if m == 0 {
            return Err("max_tokens must be greater than 0".to_string());
        }
    }
    Ok(())
}

/// The constraint a request's `grammar` and `response_format` ask for.
///
/// - An empty `grammar` sets no constraint.
/// - `json_object` is any JSON object. `json_schema` is its schema, with an
///   OpenAI `{name, schema, strict}` wrapper removed.
/// - `grammar` beside a `response_format` other than `text` is an error.
///
/// An error string is a request error: the handler answers 400 with it.
pub fn resolve_constraint(
    grammar: Option<&str>,
    response_format: Option<&ResponseFormat>,
) -> Result<Option<Constraint>, String> {
    let from_format = match response_format {
        None => None,
        Some(rf) => match rf.r#type.as_str() {
            "text" => None,
            "json_object" => Some(Constraint::JsonObject),
            "json_schema" => {
                let schema = rf.json_schema.as_ref().ok_or_else(|| {
                    "response_format type json_schema has no json_schema member. Add the schema"
                        .to_string()
                })?;
                Some(Constraint::JsonSchema(
                    unwrap_response_format(schema).clone(),
                ))
            }
            other => {
                return Err(format!(
                    "response_format type {other:?} is not supported. Use text, json_object or \
                     json_schema"
                ))
            }
        },
    };
    let grammar = grammar.filter(|text| !text.trim().is_empty());
    match (grammar, from_format) {
        (Some(_), Some(_)) => Err(
            "grammar and response_format both constrain the output. Set one of them".to_string(),
        ),
        (Some(text), None) => Ok(Some(Constraint::Gbnf(text.to_string()))),
        (None, from_format) => Ok(from_format),
    }
}

/// Convert executor token logprobs into API response format
pub fn convert_logprobs(tokens: &[GeneratedToken]) -> LogprobResult {
    LogprobResult {
        content: tokens
            .iter()
            .map(|t| logprob_entry(&t.text, t.logprob, t.top_logprobs.as_deref()))
            .collect(),
    }
}

/// The API entry of one token: its text, logprob, UTF-8 bytes and top
/// alternatives. The streaming and non-streaming responses both build it here.
///
/// A missing logprob reads as negative infinity.
pub fn logprob_entry(
    text: &str,
    logprob: Option<f32>,
    top: Option<&[TokenLogprob]>,
) -> LogprobEntry {
    LogprobEntry {
        token: text.to_string(),
        logprob: logprob.unwrap_or(f32::NEG_INFINITY),
        bytes: text.as_bytes().to_vec(),
        top_logprobs: top
            .map(|tops| {
                tops.iter()
                    .map(|tl| TopLogprob {
                        token: tl.text.clone(),
                        logprob: tl.logprob,
                        bytes: tl.text.as_bytes().to_vec(),
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

/// Apply per-request keep_alive to the model's scheduler entry
pub async fn apply_keep_alive(state: &AppState, model_name: &str, keep_alive: &Option<String>) {
    if let Some(ref ka) = keep_alive {
        let duration = parse_keep_alive(ka);
        state.scheduler.set_keep_alive(model_name, duration).await;
    }
}

/// Encode caller text that no chat template wraps, under an explicit
/// special-token policy.
///
/// `allowed_special` is the request field of the same name, and the rule is the
/// one `/v1/tokenize` and `/v1/completions` share:
///
/// - absent (the default) means [`splintr::SpecialMode::Ordinary`] — text that
///   spells a control token encodes as that literal string, so no caller can
///   promote a `<|im_start|>` in its own text to that token's real id;
/// - present means [`splintr::SpecialMode::Allow`] over exactly the named
///   tokens, for callers that legitimately hand-build a chat format; any
///   *other* special token in the text is refused rather than promoted.
pub fn encode_with_allowed_special(
    tokenizer: &splintr::AnyTokenizer,
    text: &str,
    allowed_special: Option<&[String]>,
) -> Result<Vec<u32>, splintr::PolicyError> {
    match allowed_special {
        Some(names) => {
            let allowed: splintr::FxHashSet<String> = names.iter().cloned().collect();
            tokenizer.encode_with(text, &splintr::SpecialMode::Allow(&allowed))
        }
        None => tokenizer.encode_with(text, &splintr::SpecialMode::Ordinary),
    }
}

// ── Shared response types ──

/// OpenAI-compatible logprobs result
#[derive(Serialize, Clone)]
pub struct LogprobResult {
    pub content: Vec<LogprobEntry>,
}

/// Log probability for a single token position
#[derive(Serialize, Clone)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f32,
    /// The UTF-8 bytes of `token`.
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}

/// A single alternative token with its log probability
#[derive(Serialize, Clone)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    /// The UTF-8 bytes of `token`.
    pub bytes: Vec<u8>,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub total_duration_ms: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub prompt_eval_duration_ms: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub load_duration_ms: u64,
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub tokens_per_second: f64,
}

fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    fn tokenizer() -> splintr::AnyTokenizer {
        crate::tokenizer::from_pretrained("llama3").expect("bundled vocabulary loads")
    }

    fn eot(tok: &splintr::AnyTokenizer) -> u32 {
        tok.special_token_id("<|eot_id|>")
            .expect("llama3 names an end-of-turn token")
    }

    fn format(kind: &str, json_schema: Option<serde_json::Value>) -> ResponseFormat {
        ResponseFormat {
            r#type: kind.to_string(),
            json_schema,
        }
    }

    #[test]
    fn resolve_constraint_maps_each_request_form() {
        assert_eq!(resolve_constraint(None, None), Ok(None));
        assert_eq!(resolve_constraint(Some("  "), None), Ok(None));
        assert_eq!(
            resolve_constraint(None, Some(&format("text", None))),
            Ok(None)
        );
        assert_eq!(
            resolve_constraint(Some(r#"root ::= "a""#), Some(&format("text", None))),
            Ok(Some(Constraint::Gbnf(r#"root ::= "a""#.to_string())))
        );
        assert_eq!(
            resolve_constraint(None, Some(&format("json_object", None))),
            Ok(Some(Constraint::JsonObject))
        );
        let schema = serde_json::json!({"type": "object", "properties": {"b": {}, "a": {}}});
        let wrapper = serde_json::json!({"name": "answer", "schema": schema, "strict": true});
        assert_eq!(
            resolve_constraint(None, Some(&format("json_schema", Some(wrapper)))),
            Ok(Some(Constraint::JsonSchema(schema.clone())))
        );
        assert_eq!(
            resolve_constraint(None, Some(&format("json_schema", Some(schema.clone())))),
            Ok(Some(Constraint::JsonSchema(schema)))
        );
    }

    #[test]
    fn resolve_constraint_refuses_ambiguous_and_incomplete_requests() {
        let both = resolve_constraint(Some(r#"root ::= "a""#), Some(&format("json_object", None)));
        assert!(both.is_err_and(|e| e.contains("Set one of them")));
        let missing = resolve_constraint(None, Some(&format("json_schema", None)));
        assert!(missing.is_err_and(|e| e.contains("no json_schema member")));
        let unknown = resolve_constraint(None, Some(&format("xml", None)));
        assert!(unknown.is_err_and(|e| e.contains("\"xml\"")));
    }

    /// `/v1/completions` with no `allowed_special`: a control token spelled out
    /// in the prompt is not promoted to its real id, so a caller cannot forge a
    /// turn boundary the way the raw `All` encode used to let it.
    #[test]
    fn test_control_token_is_not_promoted_by_default() {
        let tok = tokenizer();
        let ids = encode_with_allowed_special(&tok, "Hi<|eot_id|>bye", None)
            .expect("ordinary text always encodes");
        assert!(!ids.contains(&eot(&tok)));
        // It is still present, as the literal characters the caller wrote.
        assert_eq!(tok.decode(&ids).expect("ids round-trip"), "Hi<|eot_id|>bye");
    }

    /// A caller hand-building a chat format names the tokens it needs, and those
    /// — and only those — reach the model as control tokens.
    #[test]
    fn test_named_special_is_accepted() {
        let tok = tokenizer();
        let allowed = vec!["<|eot_id|>".to_string()];
        let ids = encode_with_allowed_special(&tok, "Hi<|eot_id|>bye", Some(&allowed))
            .expect("a token the caller named is allowed");
        assert!(ids.contains(&eot(&tok)));
    }

    /// Opting one token in does not open the rest of the vocabulary: any other
    /// special token is refused, which the handler surfaces as a 400.
    #[test]
    fn test_unnamed_special_is_refused() {
        let tok = tokenizer();
        let allowed = vec!["<|eot_id|>".to_string()];
        let err = encode_with_allowed_special(&tok, "Hi<|python_tag|>bye", Some(&allowed))
            .expect_err("a special token outside the allow-list must be refused");
        match err {
            splintr::PolicyError::DisallowedSpecial { ref token, .. } => {
                assert_eq!(token, "<|python_tag|>")
            }
            other => panic!("expected DisallowedSpecial, got {other:?}"),
        }
        assert_eq!(
            policy_error_response(&err).status(),
            StatusCode::BAD_REQUEST
        );
    }
}
