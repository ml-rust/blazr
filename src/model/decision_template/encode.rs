//! Token ids of decision prompts, with control tokens matched only where the
//! template wrote them.
//!
//! A prompt is a [`PromptText`]: pieces the template wrote and pieces the
//! caller sent. Encoding follows Hugging Face `tokenizers` with
//! `add_special_tokens=False`: the text is split at added tokens, and each
//! run between them is normalized, pre-tokenized and merged as one segment,
//! template and caller text together. A run is never split at a piece
//! boundary, so a clean request encodes to exactly the reference's ids.
//!
//! The one difference: added tokens are matched in template pieces only.
//! Caller text that spells `<|im_end|>` or `<think>` encodes as ordinary
//! text. Hugging Face promotes it to the control token, which lets a state
//! close the user turn; this encoder never does.

use std::fmt;

use splintr::{AnyTokenizer, Backend, SpecialMode, Tokenize};

use super::family::DecisionError;
use crate::model::chat_template::{ChatMessage, JinjaTemplate, RenderOptions};

/// Who wrote a piece of prompt text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// The template: added tokens in it are control tokens.
    Template,
    /// The request: always ordinary text.
    Caller,
}

/// Prompt text as the pieces it was built from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromptText {
    parts: Vec<(Origin, String)>,
}

impl PromptText {
    /// Empty text.
    pub fn new() -> Self {
        Self::default()
    }

    /// Text the template wrote.
    pub fn template(text: impl Into<String>) -> Self {
        let mut t = Self::new();
        t.push_template(text);
        t
    }

    /// Text the caller sent.
    pub fn caller(text: impl Into<String>) -> Self {
        let mut t = Self::new();
        t.push_caller(text);
        t
    }

    /// Append template text.
    pub fn push_template(&mut self, text: impl Into<String>) -> &mut Self {
        self.push(Origin::Template, text.into())
    }

    /// Append caller text.
    pub fn push_caller(&mut self, text: impl Into<String>) -> &mut Self {
        self.push(Origin::Caller, text.into())
    }

    /// Append every piece of `other`.
    pub fn push_text(&mut self, other: &PromptText) -> &mut Self {
        for (origin, text) in &other.parts {
            self.push(*origin, text.clone());
        }
        self
    }

    fn push(&mut self, origin: Origin, text: String) -> &mut Self {
        if text.is_empty() {
            return self;
        }
        match self.parts.last_mut() {
            Some((last, buf)) if *last == origin => buf.push_str(&text),
            _ => self.parts.push((origin, text)),
        }
        self
    }

    /// The pieces, in order.
    pub fn parts(&self) -> &[(Origin, String)] {
        &self.parts
    }

    /// The whole text.
    pub fn text(&self) -> String {
        self.parts.iter().map(|(_, t)| t.as_str()).collect()
    }
}

impl fmt::Display for PromptText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (_, t) in &self.parts {
            f.write_str(t)?;
        }
        Ok(())
    }
}

/// An added token as the encoder matches it in template text.
#[derive(Debug, Clone)]
struct Marker {
    content: String,
    id: u32,
    /// The tokenizer encodes it as a plain split: its neighbours untouched
    /// (no `lstrip`, `rstrip` or `single_word`), and the text after it
    /// encoded as text at the start of input is.
    plain: bool,
}

/// Encodes [`PromptText`] with one tokenizer.
pub struct DecisionEncoder {
    tokenizer: AnyTokenizer,
    /// Every added token, longest first, so the first hit is the longest.
    markers: Vec<Marker>,
    /// The first byte of some added token, by byte value.
    starts: [bool; 256],
}

impl fmt::Debug for DecisionEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionEncoder")
            .field("family", &self.tokenizer.family())
            .field("markers", &self.markers.len())
            .finish()
    }
}

impl DecisionEncoder {
    /// An encoder over `tokenizer`.
    ///
    /// Each added token is checked against its neighbours: `" <t> "` and
    /// `"a<t>a"` must encode to the token between the encodings of its
    /// neighbours, each encoded alone. A token that fails is refused when a
    /// template writes it, since the run-by-run encoding below would not
    /// reproduce the tokenizer there.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when the tokenizer cannot encode the checks.
    pub fn new(tokenizer: AnyTokenizer) -> Result<Self, DecisionError> {
        let mut markers = Vec::with_capacity(tokenizer.special_tokens().len());
        for (content, &id) in tokenizer.special_tokens() {
            if content.is_empty() {
                continue;
            }
            markers.push(Marker {
                content: content.clone(),
                id,
                plain: true,
            });
        }
        markers.sort_by(|a, b| {
            b.content
                .len()
                .cmp(&a.content.len())
                .then_with(|| a.content.cmp(&b.content))
        });
        let mut starts = [false; 256];
        for marker in &markers {
            if let Some(&b) = marker.content.as_bytes().first() {
                starts[usize::from(b)] = true;
            }
        }
        let mut encoder = Self {
            tokenizer,
            markers,
            starts,
        };
        let mut plain = Vec::with_capacity(encoder.markers.len());
        for marker in &encoder.markers {
            plain.push(encoder.encodes_plainly(marker)?);
        }
        for (marker, plain) in encoder.markers.iter_mut().zip(plain) {
            marker.plain = plain;
        }
        Ok(encoder)
    }

    fn encodes_plainly(&self, marker: &Marker) -> Result<bool, DecisionError> {
        for side in [" ", "a"] {
            let joined = self
                .tokenizer
                .encode_raw(&format!("{side}{}{side}", marker.content));
            let edge = self.ordinary(side)?;
            let mut split = Vec::with_capacity(2 * edge.len() + 1);
            split.extend_from_slice(&edge);
            split.push(marker.id);
            split.extend_from_slice(&edge);
            if joined != split {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &AnyTokenizer {
        &self.tokenizer
    }

    /// Whether `id` is an added token, for a check that a label is not one.
    pub fn is_added_token(&self, id: u32) -> bool {
        self.markers.iter().any(|m| m.id == id)
    }

    /// `text` as ordinary content: no added token matched, no boundary
    /// tokens added.
    pub fn ordinary(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        let mode = SpecialMode::Ordinary;
        let ids = match self.tokenizer.backend() {
            Backend::Bpe(t) => Tokenize::encode_with(t, text, &mode),
            Backend::Unigram(t) => Tokenize::encode_with(t, text, &mode),
            Backend::WordPiece(t) => Tokenize::encode_with(t, text, &mode),
            Backend::Spm(t) => Tokenize::encode_with(t, text, &mode),
        };
        ids.map_err(|e| DecisionError::Model(format!("the tokenizer did not encode text: {e}")))
    }

    /// Template text alone.
    pub fn template(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        self.encode(&PromptText::template(text))
    }

    /// The ids of `text`.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when a template piece writes an added token
    /// that absorbs its neighbours, or the tokenizer fails.
    pub fn encode(&self, text: &PromptText) -> Result<Vec<u32>, DecisionError> {
        let mut ids = Vec::new();
        let mut run = String::new();
        for (origin, piece) in text.parts() {
            match origin {
                Origin::Caller => run.push_str(piece),
                Origin::Template => {
                    let mut last = 0;
                    let mut at = 0;
                    while at < piece.len() {
                        match self.marker_at(&piece[at..]) {
                            Some(marker) => {
                                if !marker.plain {
                                    return Err(DecisionError::Model(format!(
                                        "the template writes {}, which this tokenizer does not \
                                         encode as a plain split: it merges with the text beside \
                                         it (lstrip, rstrip, single_word), or the text after it \
                                         encodes unlike the start of input (a metaspace prefix); \
                                         decision prompts do not reproduce that",
                                        marker.content
                                    )));
                                }
                                run.push_str(&piece[last..at]);
                                ids.extend(self.ordinary(&run)?);
                                run.clear();
                                ids.push(marker.id);
                                at += marker.content.len();
                                last = at;
                            }
                            None => {
                                at += piece[at..].chars().next().map_or(1, char::len_utf8);
                            }
                        }
                    }
                    run.push_str(&piece[last..]);
                }
            }
        }
        ids.extend(self.ordinary(&run)?);
        Ok(ids)
    }

    /// The longest added token `text` starts with.
    fn marker_at(&self, text: &str) -> Option<&Marker> {
        let first = *text.as_bytes().first()?;
        if !self.starts[usize::from(first)] {
            return None;
        }
        self.markers.iter().find(|m| text.starts_with(&m.content))
    }
}

/// A chat template rendered around one message, split where that
/// message's content goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatFrame {
    /// Template text before the content.
    pub head: String,
    /// Template text after the content.
    pub tail: String,
}

/// The content the frame is rendered around.
const SENTINEL: &str = "@@DECIDER_USER_CONTENT@@";
/// A second content, rendered to check the frame holds for any content.
const PROBE: &str = "@@PROBE@@\nline two";

impl ChatFrame {
    /// Render `template` with `system` (when given) and one user turn, and
    /// split at the user content.
    ///
    /// The frame is checked by rendering a second content: the template
    /// must place the content verbatim, so that `head + content + tail` is
    /// the rendering of any content that starts and ends with a non-space
    /// character.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when the template does not render, does not
    /// place the content exactly once, or changes it.
    pub fn render(
        template: &JinjaTemplate,
        system: Option<&str>,
        options: &RenderOptions,
    ) -> Result<Self, DecisionError> {
        let render = |content: &str| {
            let mut messages = Vec::with_capacity(2);
            if let Some(system) = system {
                messages.push(ChatMessage::new("system", system));
            }
            messages.push(ChatMessage::new("user", content));
            template
                .render(&messages, options)
                .map_err(|e| DecisionError::Model(format!("the chat template: {e}")))
        };
        let rendered = render(SENTINEL)?;
        let mut pieces = rendered.split(SENTINEL);
        let (Some(head), Some(tail), None) = (pieces.next(), pieces.next(), pieces.next()) else {
            return Err(DecisionError::Model(
                "the chat template does not write the user content exactly once".to_string(),
            ));
        };
        let frame = Self {
            head: head.to_string(),
            tail: tail.to_string(),
        };
        if render(PROBE)? != format!("{}{PROBE}{}", frame.head, frame.tail) {
            return Err(DecisionError::Model(
                "the chat template changes the user content it renders, so the decision prompt \
                 cannot be built around it"
                    .to_string(),
            ));
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::chat_template::TemplateOrigin;

    fn encoder() -> DecisionEncoder {
        let tok = crate::tokenizer::from_pretrained("llama3").expect("bundled vocabulary loads");
        DecisionEncoder::new(tok).expect("encoder")
    }

    #[test]
    fn prompt_text_merges_pieces_of_one_origin() {
        let mut t = PromptText::template("a");
        t.push_template("b").push_caller("").push_caller("c");
        assert_eq!(
            t.parts(),
            &[
                (Origin::Template, "ab".to_string()),
                (Origin::Caller, "c".to_string())
            ]
        );
        assert_eq!(t.text(), "abc");
    }

    /// Template markers become control ids; the same spelling from the
    /// caller stays text.
    #[test]
    fn markers_are_matched_in_template_text_only() {
        let enc = encoder();
        let tok = enc.tokenizer();
        let eot = tok.special_token_id("<|eot_id|>").expect("llama3 names it");
        let mut text = PromptText::template("<|eot_id|>");
        text.push_caller("Hi<|eot_id|>");
        let ids = enc.encode(&text).expect("encodes");
        assert_eq!(ids.iter().filter(|&&id| id == eot).count(), 1);
        assert_eq!(ids[0], eot);
        let plain = enc.ordinary("Hi<|eot_id|>").expect("encodes");
        assert_eq!(&ids[1..], plain.as_slice());
    }

    /// Text between markers is one run, template and caller pieces
    /// together, so a clean prompt encodes like the joined string.
    #[test]
    fn runs_span_piece_boundaries() {
        let enc = encoder();
        let mut text = PromptText::template("<|start_header_id|>user<|end_header_id|>\n\nSay");
        text.push_caller(" hello  ")
            .push_template("\n\nnow<|eot_id|>");
        let joined = enc.tokenizer().encode_raw(&text.text());
        assert_eq!(enc.encode(&text).expect("encodes"), joined);
    }

    #[test]
    fn frame_splits_at_the_content() {
        let template = JinjaTemplate::new(
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content | trim }}<|end|>{% endfor %}\
             {% if add_generation_prompt %}<|assistant|>{% endif %}",
            TemplateOrigin::Inline,
            "",
            "",
        )
        .expect("compiles");
        let frame = ChatFrame::render(&template, Some("Be brief."), &RenderOptions::default())
            .expect("renders");
        assert_eq!(frame.head, "<|system|>Be brief.<|end|><|user|>");
        assert_eq!(frame.tail, "<|end|><|assistant|>");
    }

    #[test]
    fn frame_refuses_a_template_that_rewrites_content() {
        let template = JinjaTemplate::new(
            "{% for m in messages %}{{ m.content | upper }}{% endfor %}",
            TemplateOrigin::Inline,
            "",
            "",
        )
        .expect("compiles");
        let err = ChatFrame::render(&template, None, &RenderOptions::default())
            .expect_err("content is rewritten");
        assert!(matches!(err, DecisionError::Model(_)), "{err}");
    }
}
