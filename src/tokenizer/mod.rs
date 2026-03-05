use anyhow::{anyhow, Result};

pub mod gguf_tokenizer;
pub use gguf_tokenizer::GgufTokenizer;

/// Trait for tokenizers used by blazr
///
/// This allows using either splintr-based tokenizers or GGUF-embedded tokenizers.
pub trait TokenizerTrait: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode token IDs to text
    fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Check if a token is the EOS token
    fn is_eos(&self, token_id: u32) -> bool;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get EOS token ID
    fn eos_token_id(&self) -> u32;

    /// Look up a special token by name (e.g. "<|fim_prefix|>").
    /// Returns None if the token is not in this vocabulary.
    fn special_token_id(&self, name: &str) -> Option<u32> {
        let _ = name;
        None
    }
}

/// Boxed tokenizer type for use in executors
pub type BoxedTokenizer = Box<dyn TokenizerTrait>;

impl TokenizerTrait for BoxedTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        (**self).encode(text)
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        (**self).decode(ids)
    }
    fn is_eos(&self, token_id: u32) -> bool {
        (**self).is_eos(token_id)
    }
    fn vocab_size(&self) -> usize {
        (**self).vocab_size()
    }
    fn eos_token_id(&self) -> u32 {
        (**self).eos_token_id()
    }
    fn special_token_id(&self, name: &str) -> Option<u32> {
        (**self).special_token_id(name)
    }
}

/// Tokenizer wrapper using splintr's pretrained vocabularies.
///
/// # Supported Vocabularies
///
/// Blazr uses splintr for tokenization, which supports these pretrained vocabularies:
///
/// | Vocab Name     | Description                    | Vocab Size (with agents) |
/// |----------------|--------------------------------|--------------------------|
/// | `cl100k_base`  | GPT-4, GPT-3.5-turbo          | ~100,331 tokens          |
/// | `o200k_base`   | GPT-4o                        | ~200,073 tokens          |
/// | `llama3`       | Meta Llama 3 family           | ~128,354 tokens          |
/// | `deepseek_v3`  | DeepSeek V3/R1                | ~128,954 tokens          |
/// | `mistral`      | Mistral 7B family             | ~32,000 tokens           |
///
/// # Custom Vocabularies
///
/// Custom vocabularies (`tokenizer_vocab: custom` in config) are not yet supported.
/// Future support will use the `.tiktoken` format (base64-encoded tokens with ranks).
///
/// If you need a custom vocabulary, you have two options:
/// 1. Train your model with one of the supported vocabularies above
/// 2. Modify blazr's tokenizer module to load your custom `.tiktoken` file
///
/// The `.tiktoken` format is a simple text format where each line contains:
/// ```text
/// <base64_token> <rank>
/// ```
/// Example: `SGVsbG8= 0` where "SGVsbG8=" decodes to "Hello" with rank 0.
pub struct Tokenizer {
    inner: splintr::Tokenizer,
    eos_token_id: u32,
    bos_token_id: Option<u32>,
    vocab_name: String,
}

/// Supported vocabulary names (primary names only, used in error messages)
const SUPPORTED_VOCABS: &[&str] = &[
    "cl100k_base",
    "o200k_base",
    "llama3",
    "deepseek_v3",
    "mistral",
];

impl Tokenizer {
    /// Create a tokenizer from a pretrained vocabulary name.
    ///
    /// # Supported vocabularies
    ///
    /// - `cl100k_base` - GPT-4, GPT-3.5-turbo (~100k tokens)
    /// - `o200k_base` - GPT-4o (~200k tokens)
    /// - `llama3` / `llama3.1` / `llama3.2` / `llama3.3` - Meta Llama 3 family (~128k tokens)
    /// - `deepseek_v3` / `deepseek-v3` - DeepSeek V3/R1 (~129k tokens)
    /// - `mistral` - Mistral 7B family (~32k tokens)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The vocabulary name is "custom" (not yet supported)
    /// - The vocabulary name is not recognized by splintr
    pub fn from_pretrained(name: &str) -> Result<Self> {
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
                SUPPORTED_VOCABS.join(", ")
            ));
        }

        let inner = splintr::from_pretrained(name)
            .map_err(|e| anyhow!("Failed to create tokenizer '{}': {}", name, e))?;
        let eos_token_id = splintr::eos_token_id_by_name(name);
        let bos_token_id = splintr::bos_token_id_by_name(name);

        Ok(Self {
            inner,
            eos_token_id,
            bos_token_id,
            vocab_name: name.to_string(),
        })
    }

    /// Create a new tokenizer with Llama3 vocabulary (convenience method)
    pub fn new_llama3() -> Result<Self> {
        Self::from_pretrained("llama3")
    }

    /// Encode text to token IDs
    ///
    /// For models that use a BOS (beginning of sequence) token, this will
    /// automatically prepend it to the token sequence.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();

        // Prepend BOS token if the vocabulary has one
        if let Some(bos_id) = self.bos_token_id {
            tokens.push(bos_id);
        }

        tokens.extend(self.inner.encode_with_special(text));
        Ok(tokens)
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids)
            .map_err(|e| anyhow!("Decode error: {}", e))
    }

    /// Check if token is EOS
    pub fn is_eos(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Get the EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get the vocabulary name
    pub fn vocab_name(&self) -> &str {
        &self.vocab_name
    }

    /// Create tokenizer from vocab_size by auto-detecting the vocabulary
    ///
    /// This matches the logic in the old blazr config.
    pub fn from_vocab_size(vocab_size: usize) -> Result<Self> {
        let vocab_name = match vocab_size {
            // Mistral range: 32000 tokens
            v if v <= 32100 => "mistral",
            // cl100k_base range: typically 100257 (base) to 100331 (with all agent tokens)
            v if v <= 100350 => "cl100k_base",
            // llama3 range: 128000 (base) to 128354 (with agent tokens)
            v if v <= 128400 => "llama3",
            // deepseek_v3 range: 128000 (base) to 128954 (with agent tokens)
            v if v <= 129000 => "deepseek_v3",
            // o200k_base range: 199999 (base) to 200073 (with agent tokens)
            v if v <= 200100 => "o200k_base",
            // Default to llama3 for unknown sizes
            _ => "llama3",
        };

        Self::from_pretrained(vocab_name)
    }
}

/// Load tokenizer with default vocabulary (llama3)
pub fn load_tokenizer() -> Result<Tokenizer> {
    Tokenizer::from_pretrained("llama3")
}

/// Load tokenizer with specific vocabulary
pub fn load_tokenizer_with_vocab(vocab: &str) -> Result<Tokenizer> {
    Tokenizer::from_pretrained(vocab)
}

// Implement TokenizerTrait for splintr-based Tokenizer
impl TokenizerTrait for Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Tokenizer::encode(self, text)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        Tokenizer::decode(self, ids)
    }

    fn is_eos(&self, token_id: u32) -> bool {
        Tokenizer::is_eos(self, token_id)
    }

    fn vocab_size(&self) -> usize {
        Tokenizer::vocab_size(self)
    }

    fn eos_token_id(&self) -> u32 {
        Tokenizer::eos_token_id(self)
    }

    fn special_token_id(&self, name: &str) -> Option<u32> {
        splintr::PretrainedVocab::from_name(&self.vocab_name)
            .map(splintr::special_tokens)
            .and_then(|tokens| tokens.get(name).copied())
    }
}

// Implement TokenizerTrait for GgufTokenizer
impl TokenizerTrait for GgufTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        GgufTokenizer::encode(self, text).map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        GgufTokenizer::decode(self, ids).map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn is_eos(&self, token_id: u32) -> bool {
        GgufTokenizer::is_eos(self, token_id)
    }

    fn vocab_size(&self) -> usize {
        GgufTokenizer::vocab_size(self)
    }

    fn eos_token_id(&self) -> u32 {
        GgufTokenizer::eos_token_id(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vocab_size_mistral() {
        let tok = Tokenizer::from_vocab_size(32000).unwrap();
        assert_eq!(tok.vocab_name(), "mistral");
    }

    #[test]
    fn test_from_vocab_size_mistral_upper_bound() {
        let tok = Tokenizer::from_vocab_size(32100).unwrap();
        assert_eq!(tok.vocab_name(), "mistral");
    }

    #[test]
    fn test_from_vocab_size_cl100k() {
        let tok = Tokenizer::from_vocab_size(100257).unwrap();
        assert_eq!(tok.vocab_name(), "cl100k_base");
    }

    #[test]
    fn test_from_vocab_size_cl100k_with_agents() {
        let tok = Tokenizer::from_vocab_size(100331).unwrap();
        assert_eq!(tok.vocab_name(), "cl100k_base");
    }

    #[test]
    fn test_from_vocab_size_llama3() {
        let tok = Tokenizer::from_vocab_size(128000).unwrap();
        assert_eq!(tok.vocab_name(), "llama3");
    }

    #[test]
    fn test_from_vocab_size_llama3_with_agents() {
        let tok = Tokenizer::from_vocab_size(128354).unwrap();
        assert_eq!(tok.vocab_name(), "llama3");
    }

    #[test]
    fn test_from_vocab_size_deepseek() {
        let tok = Tokenizer::from_vocab_size(128954).unwrap();
        assert_eq!(tok.vocab_name(), "deepseek_v3");
    }

    #[test]
    fn test_from_vocab_size_o200k() {
        let tok = Tokenizer::from_vocab_size(200073).unwrap();
        assert_eq!(tok.vocab_name(), "o200k_base");
    }

    #[test]
    fn test_from_vocab_size_unknown_defaults_llama3() {
        let tok = Tokenizer::from_vocab_size(500000).unwrap();
        assert_eq!(tok.vocab_name(), "llama3");
    }

    #[test]
    fn test_from_pretrained_llama3_variants() {
        for name in &["llama3", "llama3.1", "llama3.2", "llama3.3"] {
            let tok = Tokenizer::from_pretrained(name).unwrap();
            assert!(tok.vocab_size() > 100000, "llama3 vocab should be >100k");
        }
    }

    #[test]
    fn test_from_pretrained_custom_errors() {
        assert!(Tokenizer::from_pretrained("custom").is_err());
        assert!(Tokenizer::from_pretrained("Custom").is_err());
        assert!(Tokenizer::from_pretrained("CUSTOM").is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tok = Tokenizer::from_pretrained("mistral").unwrap();
        let text = "Hello, world!";
        let ids = tok.encode(text).unwrap();
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids).unwrap();
        assert!(decoded.contains("Hello"));
    }

    #[test]
    fn test_eos_token() {
        let tok = Tokenizer::from_pretrained("llama3").unwrap();
        let eos = tok.eos_token_id();
        assert!(tok.is_eos(eos));
        assert!(!tok.is_eos(0));
    }
}
