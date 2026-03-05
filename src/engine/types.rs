//! Engine types shared across executor, sampling, and cuda_graphs modules.

/// Why generation stopped
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit an EOS token
    Eos,
    /// Reached max_tokens limit
    Length,
    /// Matched a stop sequence
    Stop,
}

impl FinishReason {
    /// OpenAI API string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Eos => "stop",
            FinishReason::Length => "length",
            FinishReason::Stop => "stop",
        }
    }
}

/// A single token's log probability entry
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    /// Token ID
    pub token_id: u32,
    /// Decoded text for this token
    pub text: String,
    /// Log probability
    pub logprob: f32,
}

/// A generated token with metadata
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    /// Token ID
    pub token_id: u32,
    /// Decoded text
    pub text: String,
    /// Log probability of the chosen token (if logprobs enabled)
    pub logprob: Option<f32>,
    /// Top-N alternative tokens with their log probabilities (if logprobs enabled)
    pub top_logprobs: Option<Vec<TokenLogprob>>,
    /// Set on the final token to indicate why generation ended
    pub finish_reason: Option<FinishReason>,
}

/// Result of a complete (non-streaming) generation
#[derive(Debug)]
pub struct GenerationResult {
    /// Generated text
    pub text: String,
    /// Number of prompt tokens
    pub prompt_tokens: usize,
    /// Number of completion tokens generated
    pub completion_tokens: usize,
    /// Why generation finished
    pub finish_reason: FinishReason,
    /// Time to first token (prefill duration) in milliseconds
    pub prompt_eval_duration_ms: u64,
    /// Per-token logprobs (when logprobs enabled)
    pub token_logprobs: Option<Vec<GeneratedToken>>,
}

/// Check if a string is valid JSON (object or array)
pub fn is_valid_json(s: &str) -> bool {
    let trimmed = s.trim();
    serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
}
