//! Wire types of the embeddings request.

use serde::Deserialize;

use crate::server::multimodal::{ContentPart, ImageUrl};

/// Embedding request (OpenAI-compatible).
#[derive(Deserialize)]
pub struct EmbeddingRequest {
    /// Model name.
    pub model: String,
    /// Text, token ids, or multimodal parts to embed.
    pub input: EmbeddingInput,
    /// `float` (default) or `base64`. Any other value is a 400.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Pooling strategy: `mean` (default), `cls`, `last`, or `none`.
    #[serde(default = "default_pooling")]
    pub pooling: String,
    /// L2-normalize each vector. Defaults to true, as OpenAI returns unit vectors.
    #[serde(default = "default_normalize")]
    pub normalize: bool,
    /// Keep only the first `dimensions` values of each vector.
    #[serde(default)]
    pub dimensions: Option<usize>,
    // OpenAI's `user` field has no use here. Serde skips it as an unknown field.
}

fn default_pooling() -> String {
    "mean".to_string()
}

fn default_normalize() -> bool {
    true
}

/// Embedding input.
///
/// Serde tries the variants of an untagged enum in order. `Tokens` and
/// `TokenBatch` come before `Multimodal` and `Batch`, so an array of
/// integers never falls through to the string or part variants.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// One text string.
    Single(String),
    /// One sequence of token ids. Skips tokenization.
    Tokens(Vec<u32>),
    /// Several sequences of token ids. Skips tokenization.
    TokenBatch(Vec<Vec<u32>>),
    /// Content parts: `{"type": "text", ...}` and `{"type": "image_url", ...}`.
    Multimodal(Vec<ContentPart>),
    /// Several text strings.
    Batch(Vec<String>),
}

/// One unit of embedding work, in input order.
pub(super) enum EmbedItem<'a> {
    Text(&'a str),
    Tokens(&'a [u32]),
    Image(&'a ImageUrl),
}

impl EmbeddingInput {
    /// Flatten the input into ordered work items.
    pub(super) fn items(&self) -> Vec<EmbedItem<'_>> {
        match self {
            EmbeddingInput::Single(s) => vec![EmbedItem::Text(s.as_str())],
            EmbeddingInput::Batch(v) => v.iter().map(|s| EmbedItem::Text(s.as_str())).collect(),
            EmbeddingInput::Tokens(ids) => vec![EmbedItem::Tokens(ids.as_slice())],
            EmbeddingInput::TokenBatch(rows) => rows
                .iter()
                .map(|r| EmbedItem::Tokens(r.as_slice()))
                .collect(),
            EmbeddingInput::Multimodal(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(EmbedItem::Text(text.as_str())),
                    ContentPart::ImageUrl { image_url } => Some(EmbedItem::Image(image_url)),
                    ContentPart::InputAudio { .. } => None,
                })
                .collect(),
        }
    }

    /// True when any content part is an image.
    pub(super) fn has_images(&self) -> bool {
        match self {
            EmbeddingInput::Multimodal(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> EmbeddingRequest {
        let json = format!(r#"{{"model": "test", "input": {input}}}"#);
        serde_json::from_str(&json).expect("request parses")
    }

    #[test]
    fn a_user_field_is_accepted_and_ignored() {
        let json = r#"{"model": "test", "input": "hi", "user": "u-1"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).expect("request parses");
        assert_eq!(texts(&req.input), vec!["hi"]);
    }

    fn texts(input: &EmbeddingInput) -> Vec<&str> {
        input
            .items()
            .into_iter()
            .filter_map(|i| match i {
                EmbedItem::Text(t) => Some(t),
                _ => None,
            })
            .collect()
    }

    fn token_rows(input: &EmbeddingInput) -> Vec<Vec<u32>> {
        input
            .items()
            .into_iter()
            .filter_map(|i| match i {
                EmbedItem::Tokens(t) => Some(t.to_vec()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn single_string() {
        let req = parse(r#""hello""#);
        assert!(matches!(req.input, EmbeddingInput::Single(_)));
        assert_eq!(texts(&req.input), vec!["hello"]);
    }

    #[test]
    fn string_batch() {
        let req = parse(r#"["hello", "world"]"#);
        assert!(matches!(req.input, EmbeddingInput::Batch(_)));
        assert_eq!(texts(&req.input), vec!["hello", "world"]);
    }

    #[test]
    fn token_array() {
        let req = parse("[1, 2, 3]");
        assert!(matches!(req.input, EmbeddingInput::Tokens(_)));
        assert_eq!(token_rows(&req.input), vec![vec![1, 2, 3]]);
    }

    #[test]
    fn token_batch() {
        let req = parse("[[1, 2], [3]]");
        assert!(matches!(req.input, EmbeddingInput::TokenBatch(_)));
        assert_eq!(token_rows(&req.input), vec![vec![1, 2], vec![3]]);
    }

    #[test]
    fn multimodal_with_image() {
        let req = parse(
            r#"[{"type": "text", "text": "a cat"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]"#,
        );
        assert!(matches!(req.input, EmbeddingInput::Multimodal(_)));
        assert_eq!(texts(&req.input), vec!["a cat"]);
        assert!(req.input.has_images());
    }

    #[test]
    fn multimodal_text_only() {
        let req =
            parse(r#"[{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]"#);
        assert!(matches!(req.input, EmbeddingInput::Multimodal(_)));
        assert_eq!(texts(&req.input), vec!["hello", "world"]);
        assert!(!req.input.has_images());
    }

    #[test]
    fn empty_arrays_parse_as_empty_token_rows() {
        let flat = parse("[]");
        assert!(matches!(flat.input, EmbeddingInput::Tokens(ref ids) if ids.is_empty()));
        let nested = parse("[[]]");
        assert!(matches!(nested.input, EmbeddingInput::TokenBatch(_)));
        assert!(token_rows(&nested.input).iter().all(|r| r.is_empty()));
    }

    #[test]
    fn negative_token_id_is_rejected() {
        let json = r#"{"model": "test", "input": [-1, 2]}"#;
        assert!(serde_json::from_str::<EmbeddingRequest>(json).is_err());
    }

    #[test]
    fn defaults() {
        let req = parse(r#""hello""#);
        assert_eq!(req.encoding_format, None);
        assert_eq!(req.pooling, "mean");
        assert!(req.normalize);
        assert_eq!(req.dimensions, None);
    }
}
