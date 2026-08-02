//! Loaders for splintr's pretrained vocabularies.
//!
//! # Supported Vocabularies
//!
//! Blazr uses splintr for tokenization, which supports these pretrained vocabularies:
//!
//! | Vocab Name     | Description                    | Base vocab size |
//! |----------------|--------------------------------|-----------------|
//! | `cl100k_base`  | GPT-4, GPT-3.5-turbo           | 100,277         |
//! | `o200k_base`   | GPT-4o                         | 200,019         |
//! | `llama3`       | Meta Llama 3 family            | 128,256         |
//! | `deepseek_v3`  | DeepSeek V3/R1                 | 128,815         |
//! | `mistral_v1`   | Mistral 7B v0.1/v0.2, Mixtral  | 32,000          |
//! | `mistral_v2`   | Mistral 7B v0.3, Codestral     | 32,768          |
//! | `mistral_v3`   | Mistral NeMo/Large 2/Pixtral   | 131,072         |
//!
//! Splintr appends its 54 agent tokens above every id the base vocabulary uses, so a
//! checkpoint trained through splintr reports the *extended* size instead; both are
//! recognised by [`vocab_name_for_size`].
//!
//! # Custom Vocabularies
//!
//! Custom vocabularies (`tokenizer_vocab: custom` in config) are not yet supported.
//! Future support will use the `.tiktoken` format (base64-encoded tokens with ranks).
//!
//! If you need a custom vocabulary, you have two options:
//! 1. Train your model with one of the supported vocabularies above
//! 2. Modify blazr's tokenizer module to load your custom `.tiktoken` file
//!
//! The `.tiktoken` format is a simple text format where each line contains
//! `<base64_token> <rank>` — e.g. `SGVsbG8= 0`, where "SGVsbG8=" decodes to "Hello".

use anyhow::{anyhow, Result};

use splintr::{AnyTokenizer, Tokenize};

/// The bundled vocabularies blazr identifies a checkpoint against, in the order they
/// are tried. Whisper is deliberately absent: its vocabulary is selected from the audio
/// model's own variant, never guessed from a text model's embedding shape.
const CANDIDATE_VOCABS: &[&str] = &[
    "cl100k_base",
    "o200k_base",
    "llama3",
    "deepseek_v3",
    "mistral_v1",
    "mistral_v2",
    "mistral_v3",
];

/// Create a tokenizer from a pretrained vocabulary name.
///
/// # Supported vocabularies
///
/// - `cl100k_base` - GPT-4, GPT-3.5-turbo (~100k tokens)
/// - `o200k_base` - GPT-4o (~200k tokens)
/// - `llama3` / `llama3.1` / `llama3.2` / `llama3.3` - Meta Llama 3 family (~128k tokens)
/// - `deepseek_v3` / `deepseek-v3` - DeepSeek V3/R1 (~129k tokens)
/// - `mistral` / `mistral_v1` / `mistral_v2` / `mistral_v3` - Mistral families
///
/// Boundary tokens (BOS/EOS) are owned by the returned tokenizer's own special-token
/// policy, not by this loader: blazr's chat templates already spell the model's BOS
/// marker in the prompt text, and splintr matches it there.
///
/// # Errors
///
/// Returns an error if:
/// - The vocabulary name is "custom" (not yet supported)
/// - The vocabulary name is not recognized by splintr
pub fn from_pretrained(name: &str) -> Result<AnyTokenizer> {
    // Check for custom vocab (case-insensitive)
    if name.eq_ignore_ascii_case("custom") {
        return Err(anyhow!(
            "Custom vocabularies are not yet supported in blazr.\n\
             \n\
             Supported vocabularies: {}\n\
             \n\
             To use a custom vocabulary, you would need to:\n\
             1. Prepare your vocab in .tiktoken format (base64_token rank per line)\n\
             2. Modify blazr's tokenizer module to load from file\n\
             \n\
             For now, please train your model with one of the supported vocabularies.",
            CANDIDATE_VOCABS.join(", ")
        ));
    }

    splintr::from_pretrained(name)
        .map_err(|e| anyhow!("Failed to create tokenizer '{}': {}", name, e))
}

/// Load tokenizer with default vocabulary (llama3)
pub fn load_tokenizer() -> Result<AnyTokenizer> {
    from_pretrained("llama3")
}

/// Load tokenizer with specific vocabulary
pub fn load_tokenizer_with_vocab(vocab: &str) -> Result<AnyTokenizer> {
    from_pretrained(vocab)
}

/// Identify which bundled vocabulary a checkpoint uses from the row count of its
/// token-embedding tensor.
///
/// A checkpoint reports one of exactly two sizes: the vocabulary's *base* size, as the
/// upstream reference tokenizer counts it, or splintr's *extended* size once its agent
/// tokens are appended. Both are asked of splintr directly — the extended size is not
/// `base + 54`, because llama3 and deepseek_v3 leave gaps between the base vocabulary
/// and the agent block, so any arithmetic here would be wrong for them.
///
/// # Errors
///
/// Returns an error when no bundled vocabulary reports that size. Guessing would hand
/// the model a tokenizer it was not trained with, which does not fail loudly — it
/// generates fluent-looking garbage — so an unrecognised size must stop the load.
pub fn vocab_name_for_size(vocab_size: usize) -> Result<&'static str> {
    // Base sizes first: they are constants, so this pass costs nothing.
    for &name in CANDIDATE_VOCABS {
        let base = splintr::base_vocab_size_by_name(name)
            .map_err(|e| anyhow!("Unknown bundled vocabulary '{}': {}", name, e))?;
        if base as usize == vocab_size {
            return Ok(name);
        }
    }

    // Then the extended sizes, which are only knowable by building each vocabulary.
    for &name in CANDIDATE_VOCABS {
        let tokenizer = from_pretrained(name)?;
        if tokenizer.vocab_size() == vocab_size {
            return Ok(name);
        }
    }

    let known = CANDIDATE_VOCABS
        .iter()
        .map(|&name| match splintr::base_vocab_size_by_name(name) {
            Ok(base) => format!("{} ({})", name, base),
            Err(e) => format!("{} (unavailable: {})", name, e),
        })
        .collect::<Vec<_>>()
        .join(", ");

    Err(anyhow!(
        "Cannot identify a tokenizer for a vocabulary of {} tokens. \
         Known vocabularies (base sizes): {}. \
         Serving this model needs its own tokenizer, not a guessed one.",
        vocab_size,
        known
    ))
}

/// Create a tokenizer from a checkpoint's `vocab_size` by identifying the vocabulary.
pub fn from_vocab_size(vocab_size: usize) -> Result<AnyTokenizer> {
    from_pretrained(vocab_name_for_size(vocab_size)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vocab_size_mistral() {
        assert_eq!(vocab_name_for_size(32000).unwrap(), "mistral_v1");
    }

    #[test]
    fn test_from_vocab_size_mistral_v2() {
        assert_eq!(vocab_name_for_size(32768).unwrap(), "mistral_v2");
    }

    #[test]
    fn test_from_vocab_size_mistral_v3() {
        assert_eq!(vocab_name_for_size(131072).unwrap(), "mistral_v3");
    }

    #[test]
    fn test_from_vocab_size_cl100k() {
        assert_eq!(vocab_name_for_size(100277).unwrap(), "cl100k_base");
    }

    #[test]
    fn test_from_vocab_size_cl100k_with_agents() {
        let extended = from_pretrained("cl100k_base").unwrap().vocab_size();
        assert_eq!(vocab_name_for_size(extended).unwrap(), "cl100k_base");
    }

    #[test]
    fn test_from_vocab_size_llama3() {
        assert_eq!(vocab_name_for_size(128256).unwrap(), "llama3");
    }

    #[test]
    fn test_from_vocab_size_llama3_with_agents() {
        let extended = from_pretrained("llama3").unwrap().vocab_size();
        assert_eq!(vocab_name_for_size(extended).unwrap(), "llama3");
    }

    #[test]
    fn test_from_vocab_size_deepseek() {
        assert_eq!(vocab_name_for_size(128815).unwrap(), "deepseek_v3");
    }

    #[test]
    fn test_from_vocab_size_o200k() {
        assert_eq!(vocab_name_for_size(200019).unwrap(), "o200k_base");
    }

    #[test]
    fn test_from_vocab_size_unknown_errors() {
        assert!(vocab_name_for_size(500000).is_err());
        assert!(vocab_name_for_size(32100).is_err());
    }

    #[test]
    fn test_from_pretrained_llama3_variants() {
        for name in &["llama3", "llama3.1", "llama3.2", "llama3.3"] {
            let tok = from_pretrained(name).unwrap();
            assert!(tok.vocab_size() > 100000, "llama3 vocab should be >100k");
        }
    }

    #[test]
    fn test_from_pretrained_custom_errors() {
        assert!(from_pretrained("custom").is_err());
        assert!(from_pretrained("Custom").is_err());
        assert!(from_pretrained("CUSTOM").is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tok = from_pretrained("mistral_v1").unwrap();
        let text = "Hello, world!";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids).unwrap();
        assert!(decoded.contains("Hello"));
    }

    #[test]
    fn test_eos_token() {
        let tok = from_pretrained("llama3").unwrap();
        let eos = tok
            .eos_token_id()
            .expect("llama3 states an end-of-sequence token");
        assert!(tok.is_eos(eos));
        assert!(!tok.is_eos(0));
    }
}
