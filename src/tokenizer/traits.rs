use anyhow::Result;

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
