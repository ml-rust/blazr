//! Per-request text of generated tokens.
//!
//! Decoding one id at a time with `decode(&[id])` loses text. A SentencePiece
//! decode strips the dummy-prefix space from the start of every call. So each
//! `▁city` piece decodes to `city`, and `a city that` streams as
//! `acitythat`. A byte-fallback character split over several ids also decodes
//! to nothing, one id at a time.
//!
//! [`TokenTextStream`] holds one splintr streaming decoder per request. The
//! prompt primes it, so the dummy-prefix strip lands on the prompt and each
//! generated token keeps its leading space. The concatenated output then
//! equals `decode(prompt + generated)` minus `decode(prompt)`.

use splintr::{AnyTokenizer, StreamingDecoder};

/// The text each generated token adds, for one request.
pub struct TokenTextStream {
    /// `None` when the tokenizer declares a decode pipeline that does not
    /// stream. Each id then decodes through `decode_token`.
    decoder: Option<StreamingDecoder>,
}

impl TokenTextStream {
    /// A stream that continues after `context`, the ids already in the
    /// sequence. The context text itself is never emitted.
    pub fn new(tokenizer: &AnyTokenizer, context: &[u32]) -> Self {
        let decoder = tokenizer.streaming_decoder().ok().map(|mut decoder| {
            // The emitted text is the prompt's own, so it is dropped.
            let _ = decoder.add_tokens_lossy(context);
            decoder
        });
        Self { decoder }
    }

    /// The text `id` completes. Empty while a multi-byte character is still
    /// incomplete. Its bytes arrive with a later token or with
    /// [`finish`](Self::finish).
    pub fn push(&mut self, tokenizer: &AnyTokenizer, id: u32) -> String {
        match &mut self.decoder {
            Some(decoder) => decoder.add_token_lossy(id).unwrap_or_default(),
            None => tokenizer
                .decode_token(id)
                .or_else(|_| tokenizer.decode(&[id]))
                .unwrap_or_default(),
        }
    }

    /// The text one generated token carries in a response.
    ///
    /// An end-of-sequence token adds no text of its own. The request's last
    /// token, `is_final`, also carries the bytes still held.
    pub fn next_text(
        &mut self,
        tokenizer: &AnyTokenizer,
        id: u32,
        is_eos: bool,
        is_final: bool,
    ) -> String {
        let mut text = if is_eos {
            String::new()
        } else {
            self.push(tokenizer, id)
        };
        if is_eos || is_final {
            text.push_str(&self.finish());
        }
        text
    }

    /// The bytes still held at the end of a request. An incomplete character
    /// becomes U+FFFD.
    pub fn finish(&mut self) -> String {
        self.decoder
            .as_mut()
            .map(StreamingDecoder::flush)
            .unwrap_or_default()
    }
}

/// The bytes one id contributes to decoded text, without the stream-start
/// space strip.
///
/// A grammar DFA and a top-logprob entry need per-id text with no request
/// state. A byte-fallback id yields its raw byte here, which a `String`
/// cannot hold alone.
pub fn token_piece_bytes(tokenizer: &AnyTokenizer, id: u32) -> Vec<u8> {
    tokenizer
        .decode_token_bytes(id)
        .or_else(|_| tokenizer.decode(&[id]).map(String::into_bytes))
        .unwrap_or_default()
}

/// [`token_piece_bytes`] as text. Bytes that are not valid UTF-8 alone
/// become U+FFFD.
pub fn token_piece(tokenizer: &AnyTokenizer, id: u32) -> String {
    String::from_utf8_lossy(&token_piece_bytes(tokenizer, id)).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use splintr::{GgufVocab, Tokenize};

    /// A small SentencePiece BPE vocabulary in GGUF form. Scores are merge
    /// ranks: a higher score merges first.
    fn spm_vocab() -> AnyTokenizer {
        let mut tokens: Vec<String> = ["<unk>", "<s>", "</s>"]
            .into_iter()
            .map(String::from)
            .collect();
        let mut token_type = vec![2u32, 3, 3];
        // Byte-fallback tokens for every byte.
        for b in 0..=255u8 {
            tokens.push(format!("<0x{b:02X}>"));
            token_type.push(6);
        }
        let pieces = [
            "▁", "a", "c", "i", "t", "y", "h", "e", "s", "▁a", "▁c", "▁t", "it", "ity", "▁city",
            "th", "▁th", "at", "▁that", "is", "▁is", "▁this",
        ];
        for piece in pieces {
            tokens.push(piece.to_string());
            token_type.push(1);
        }
        let n = tokens.len();
        // Longer pieces rank higher, so whole words merge.
        let scores: Vec<f32> = tokens.iter().map(|t| t.chars().count() as f32).collect();
        assert_eq!(scores.len(), n);
        splintr::from_gguf_vocab(GgufVocab {
            model: "llama".to_string(),
            tokens,
            scores: Some(scores),
            token_type: Some(token_type),
            add_space_prefix: Some(true),
            add_bos_token: Some(true),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            unknown_token_id: Some(0),
            ..GgufVocab::default()
        })
        .expect("synthetic SPM vocabulary loads")
    }

    fn id_of(tok: &AnyTokenizer, piece: &str) -> u32 {
        (0..tok.vocab_size() as u32)
            .find(|&id| {
                tok.decode_token_bytes(id).ok() == Some(piece.replace('▁', " ").into_bytes())
            })
            .unwrap_or_else(|| panic!("piece {piece:?} is in the vocabulary"))
    }

    #[test]
    fn whole_sequence_round_trips_with_spaces() {
        let tok = spm_vocab();
        let text = "a city that is\nthis";
        let ids = tok.encode(text);
        assert_eq!(ids[0], 1, "add_bos_token puts <s> first");
        assert_eq!(tok.decode(&ids).unwrap(), text);
    }

    /// The old per-id decode strips every token's space. The stream keeps
    /// them, and a prompt-primed stream keeps the first one too.
    #[test]
    fn stream_keeps_spaces_token_by_token() {
        let tok = spm_vocab();
        let prompt = tok.encode("this is");
        // The dummy prefix makes the first piece `▁a`, a real continuation.
        let generated = tok.encode_raw("a city that is\nthis");

        let per_id: String = generated
            .iter()
            .map(|&id| tok.decode(&[id]).unwrap())
            .collect();
        assert!(
            !per_id.contains(' '),
            "per-id decode drops spaces: {per_id:?}"
        );

        let mut stream = TokenTextStream::new(&tok, &prompt);
        let mut out = String::new();
        for &id in &generated {
            out.push_str(&stream.push(&tok, id));
        }
        out.push_str(&stream.finish());

        let mut all = prompt.clone();
        all.extend(&generated);
        let whole = tok.decode(&all).unwrap();
        let prompt_text = tok.decode(&prompt).unwrap();
        assert_eq!(out, whole[prompt_text.len()..]);
        assert_eq!(out, " a city that is\nthis");
    }

    /// A character split across byte-fallback ids arrives whole, once.
    #[test]
    fn stream_joins_byte_fallback_characters() {
        let tok = spm_vocab();
        let ids = tok.encode_raw("a é");
        assert!(ids.len() >= 3, "é falls back to bytes: {ids:?}");
        let mut stream = TokenTextStream::new(&tok, &[]);
        let pieces: Vec<String> = ids.iter().map(|&id| stream.push(&tok, id)).collect();
        assert_eq!(pieces.concat() + &stream.finish(), "a é");
        assert!(pieces.iter().all(|p| !p.contains('\u{FFFD}')), "{pieces:?}");
        assert_eq!(token_piece(&tok, id_of(&tok, "▁city")), " city");
    }

    /// A real SentencePiece GGUF, read for its vocabulary only. Set
    /// `BLAZR_TEST_SPM_GGUF` to the file. The test skips without it.
    #[test]
    fn real_spm_gguf_streams_spaces() {
        let Ok(path) = std::env::var("BLAZR_TEST_SPM_GGUF") else {
            eprintln!("skip: BLAZR_TEST_SPM_GGUF is unset");
            return;
        };
        let gguf = boostr::format::Gguf::open_with_mmap(&path, false).expect("GGUF opens");
        let vocab = boostr::format::extract_gguf_vocab(gguf.metadata()).expect("vocabulary");
        let tok = splintr::from_gguf_vocab(vocab).expect("tokenizer builds");

        let prompt = tok.encode("Paris is");
        let sentence = "a city that sits on the Seine.\nIt has 2 million people: café, naïve.";
        let generated = tok.encode_raw(sentence);
        let mut stream = TokenTextStream::new(&tok, &prompt);
        let mut out = String::new();
        for &id in &generated {
            out.push_str(&stream.push(&tok, id));
        }
        out.push_str(&stream.finish());
        eprintln!("ids: {generated:?}\nstreamed: {out:?}");
        assert_eq!(tok.decode(&generated).unwrap(), sentence);
        assert_eq!(out, format!(" {sentence}"));
    }
}
