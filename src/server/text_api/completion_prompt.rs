//! The `prompt` field of `/v1/completions` and its encoding.

use axum::http::StatusCode;
use axum::response::Response;
use serde::Deserialize;

use super::embeddings::options::check_token_ids;
use super::fim_prompt::{build_fim_prompt, FIM_MARKERS};
use crate::server::generation::{
    encode_with_allowed_special, error_response, policy_error_response,
};

/// The `prompt` field: a string, strings, token ids, or several token id lists.
///
/// Serde tries the variants of an untagged enum in order. `Tokens` and
/// `TokenBatch` come before `Texts`, so an array of integers never reads as
/// strings. An array of strings fails both token variants and reaches `Texts`.
#[derive(Deserialize, Debug, PartialEq)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Text(String),
    Tokens(Vec<u32>),
    TokenBatch(Vec<Vec<u32>>),
    Texts(Vec<String>),
}

/// One prompt of a request, in request order.
#[derive(Debug, PartialEq)]
pub enum PromptItem<'a> {
    Text(&'a str),
    Tokens(&'a [u32]),
}

impl CompletionPrompt {
    /// The prompts of this field, in order.
    pub fn items(&self) -> Vec<PromptItem<'_>> {
        match self {
            Self::Text(s) => vec![PromptItem::Text(s)],
            Self::Texts(v) => v.iter().map(|s| PromptItem::Text(s)).collect(),
            Self::Tokens(t) => vec![PromptItem::Tokens(t)],
            Self::TokenBatch(rows) => rows.iter().map(|r| PromptItem::Tokens(r)).collect(),
        }
    }
}

/// How the server turns a prompt into model input.
pub struct PromptContext<'a> {
    pub tokenizer: &'a splintr::AnyTokenizer,
    pub vocab_size: usize,
    /// Decoded `context` tokens. Applies to text prompts only.
    pub prefix: &'a str,
    /// Text after the completion. Non-empty selects fill-in-the-middle.
    pub suffix: Option<&'a str>,
    pub allowed_special: Option<&'a [String]>,
    /// Fill [`ResolvedPrompt::echo`].
    pub echo: bool,
}

/// A prompt as model input.
pub struct ResolvedPrompt {
    pub tokens: Vec<u32>,
    /// The prompt text a choice repeats under `echo`. Empty without `echo`.
    pub echo: String,
}

/// A prompt that did not resolve.
pub enum PromptError {
    /// A request error. The handler answers 400 with the message.
    Request(String),
    /// The special-token policy refused the text. The handler answers 400.
    Policy(splintr::PolicyError),
}

impl PromptError {
    /// The 400 response for this error.
    pub fn into_response(self) -> Response {
        match self {
            Self::Request(message) => {
                error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error")
            }
            Self::Policy(e) => policy_error_response(&e),
        }
    }
}

/// Encode one prompt.
///
/// - Text: `prefix` + text, then the tokenizer.
/// - Text with a suffix: the fill-in-the-middle prompt, then the tokenizer.
/// - Tokens: used as given, after a vocabulary check. A suffix is an error.
pub fn resolve_prompt(
    item: &PromptItem<'_>,
    ctx: &PromptContext<'_>,
) -> Result<ResolvedPrompt, PromptError> {
    match item {
        PromptItem::Text(text) => resolve_text(text, ctx),
        PromptItem::Tokens(ids) => resolve_tokens(ids, ctx),
    }
}

fn resolve_text(text: &str, ctx: &PromptContext<'_>) -> Result<ResolvedPrompt, PromptError> {
    let body = format!("{}{}", ctx.prefix, text);
    let tokens = match ctx.suffix.filter(|s| !s.is_empty()) {
        Some(suffix) => {
            let prompt =
                build_fim_prompt(ctx.tokenizer, &body, suffix).map_err(PromptError::Request)?;
            // The caller's allowed tokens and the FIM markers are the only
            // control tokens the assembled prompt can spell.
            let mut allowed: splintr::FxHashSet<String> = FIM_MARKERS.clone();
            allowed.extend(ctx.allowed_special.unwrap_or(&[]).iter().cloned());
            ctx.tokenizer
                .encode_with(&prompt, &splintr::SpecialMode::Allow(&allowed))
                .map_err(PromptError::Policy)?
        }
        None => encode_with_allowed_special(ctx.tokenizer, &body, ctx.allowed_special)
            .map_err(PromptError::Policy)?,
    };
    Ok(ResolvedPrompt {
        tokens,
        echo: if ctx.echo { body } else { String::new() },
    })
}

/// Check a token id prompt before it reaches the model.
///
/// # Errors
///
/// The list is empty, an id is at or above `vocab_size`, or a suffix is set.
/// A suffix needs text, because the server wraps text in the FIM markers.
fn check_token_prompt(ids: &[u32], vocab_size: usize, suffix: Option<&str>) -> Result<(), String> {
    if ids.is_empty() {
        return Err("prompt token list must not be empty".to_string());
    }
    check_token_ids(ids, vocab_size)?;
    if suffix.is_some_and(|s| !s.is_empty()) {
        return Err(
            "suffix is not supported with token id prompts. Send the prompt as text".to_string(),
        );
    }
    Ok(())
}

fn resolve_tokens(ids: &[u32], ctx: &PromptContext<'_>) -> Result<ResolvedPrompt, PromptError> {
    check_token_prompt(ids, ctx.vocab_size, ctx.suffix).map_err(PromptError::Request)?;
    let echo = if ctx.echo {
        ctx.tokenizer
            .decode(ids)
            .map_err(|e| PromptError::Request(format!("Failed to decode prompt tokens: {e}")))?
    } else {
        String::new()
    };
    Ok(ResolvedPrompt {
        tokens: ids.to_vec(),
        echo,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt(json: &str) -> CompletionPrompt {
        serde_json::from_str(json).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn prompt_reads_as_a_string() {
        assert_eq!(prompt(r#""hi""#), CompletionPrompt::Text("hi".into()));
    }

    #[test]
    fn prompt_reads_as_strings() {
        assert_eq!(
            prompt(r#"["a","b"]"#),
            CompletionPrompt::Texts(vec!["a".into(), "b".into()])
        );
    }

    #[test]
    fn integer_arrays_do_not_read_as_strings() {
        assert_eq!(prompt("[1,2,3]"), CompletionPrompt::Tokens(vec![1, 2, 3]));
        assert_eq!(
            prompt("[[1,2],[3]]"),
            CompletionPrompt::TokenBatch(vec![vec![1, 2], vec![3]])
        );
    }

    #[test]
    fn items_keep_request_order() {
        let texts = prompt(r#"["a","b"]"#);
        assert_eq!(
            texts.items(),
            vec![PromptItem::Text("a"), PromptItem::Text("b")]
        );
        let batch = prompt("[[1],[2,3]]");
        assert_eq!(
            batch.items(),
            vec![PromptItem::Tokens(&[1]), PromptItem::Tokens(&[2, 3])]
        );
        assert_eq!(prompt("[7]").items(), vec![PromptItem::Tokens(&[7])]);
    }

    #[test]
    fn an_id_at_or_above_the_vocabulary_is_a_400_naming_the_id() {
        assert!(check_token_prompt(&[0, 9], 10, None).is_ok());
        let message = check_token_prompt(&[0, 10], 10, None).unwrap_err();
        assert!(message.contains("10"));
        let response = PromptError::Request(message).into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn an_empty_token_list_is_an_error() {
        assert!(check_token_prompt(&[], 10, None).is_err());
    }

    #[test]
    fn a_suffix_with_token_ids_is_an_error() {
        assert!(check_token_prompt(&[1], 10, Some("x")).is_err());
        assert!(check_token_prompt(&[1], 10, Some("")).is_ok());
    }

    #[test]
    fn a_negative_id_is_not_a_prompt() {
        assert!(serde_json::from_str::<CompletionPrompt>("[-1]").is_err());
    }
}
