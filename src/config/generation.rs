//! Generation configuration settings

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Configuration for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Temperature for sampling (higher = more random)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p nucleus sampling threshold
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k sampling (0 = disabled)
    #[serde(default)]
    pub top_k: usize,

    /// Min-p sampling threshold (0.0 = disabled).
    /// Filters tokens with probability < min_p * max_probability.
    #[serde(default = "default_min_p")]
    pub min_p: f32,

    /// Repetition penalty multiplier (1.0 = no penalty).
    /// Divides logits of tokens in the recent window by this value.
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,

    /// Number of recent tokens to consider for repetition penalty
    #[serde(default = "default_repeat_last_n")]
    pub repeat_last_n: usize,

    /// Frequency penalty: penalizes tokens proportional to their occurrence count.
    /// Subtracted from logit: logit -= freq_penalty * count(token)
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Presence penalty: penalizes tokens that have appeared at all.
    /// Subtracted from logit: logit -= pres_penalty * (count(token) > 0 ? 1 : 0)
    #[serde(default)]
    pub presence_penalty: f32,

    /// Stop sequences
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Random seed (None = random)
    #[serde(default)]
    pub seed: Option<u64>,

    /// Print prompt tokens before generation (like llama-cli --verbose-prompt)
    #[serde(default)]
    pub verbose_prompt: bool,

    /// Per-token logit bias: token_id → additive bias applied before sampling.
    /// Positive values increase likelihood, negative decrease.
    #[serde(default)]
    pub logit_bias: HashMap<u32, f32>,

    /// Whether to return log probabilities for each generated token.
    #[serde(default)]
    pub logprobs: bool,

    /// Number of top log probabilities to return per token position (1-20).
    /// Only used when `logprobs` is true.
    #[serde(default = "default_top_logprobs")]
    pub top_logprobs: usize,

    /// Mirostat sampling mode (0 = disabled, 1 = Mirostat v1, 2 = Mirostat v2).
    /// Target-entropy sampling that adaptively adjusts the candidate set.
    #[serde(default)]
    pub mirostat_mode: u8,

    /// Mirostat target entropy (tau). Higher values = more random output.
    /// Default: 5.0 (typical for natural language).
    #[serde(default = "default_mirostat_tau")]
    pub mirostat_tau: f32,

    /// Mirostat learning rate (eta). Controls how fast the algorithm adapts.
    /// Default: 0.1
    #[serde(default = "default_mirostat_eta")]
    pub mirostat_eta: f32,

    /// Dynamic temperature range. When > 0, temperature is adjusted per-token
    /// based on logit entropy. Actual temp = base_temp ± dynatemp_range.
    #[serde(default)]
    pub dynatemp_range: f32,

    /// Dynamic temperature exponent. Controls the entropy→temperature mapping curve.
    /// Default: 1.0 (linear).
    #[serde(default = "default_dynatemp_exponent")]
    pub dynatemp_exponent: f32,

    /// JSON mode: when true, retry generation up to 3 times if output is not valid JSON.
    #[serde(default)]
    pub json_mode: bool,

    /// DRY (Don't Repeat Yourself) penalty multiplier.
    /// Penalizes n-gram repetitions by subtracting `dry_multiplier * match_length`
    /// from logits of tokens that would extend a repeated n-gram.
    /// 0.0 = disabled. Typical values: 0.8-1.5.
    #[serde(default)]
    pub dry_multiplier: f32,

    /// DRY penalty base: minimum n-gram length to penalize.
    /// Default: 2 (penalizes repeated bigrams and longer).
    #[serde(default = "default_dry_base")]
    pub dry_base: usize,

    /// DRY penalty allowed length: how many tokens back to scan for repeated n-grams.
    /// Default: 0 (scan entire history). Set to limit computational cost.
    #[serde(default)]
    pub dry_allowed_length: usize,

    /// DRY penalty sequence breakers: tokens that break n-gram matching.
    /// E.g., newline, period. Default: empty (no breakers).
    #[serde(default)]
    pub dry_sequence_breakers: Vec<String>,

    /// Typical sampling threshold (0.0 = disabled).
    /// Filters tokens to those whose information content is close to the
    /// expected information (entropy). Values like 0.95 keep tokens within
    /// the 95th percentile of typicality. Typical range: 0.9-1.0.
    #[serde(default)]
    pub typical_p: f32,
}

fn default_max_tokens() -> usize {
    2048
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_min_p() -> f32 {
    0.05
}

fn default_repeat_penalty() -> f32 {
    1.1
}

fn default_repeat_last_n() -> usize {
    64
}

fn default_top_logprobs() -> usize {
    5
}

fn default_mirostat_tau() -> f32 {
    5.0
}

fn default_mirostat_eta() -> f32 {
    0.1
}

fn default_dynatemp_exponent() -> f32 {
    1.0
}

fn default_dry_base() -> usize {
    2
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: 0,
            min_p: default_min_p(),
            repeat_penalty: default_repeat_penalty(),
            repeat_last_n: default_repeat_last_n(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_sequences: Vec::new(),
            seed: None,
            verbose_prompt: false,
            logit_bias: HashMap::new(),
            logprobs: false,
            top_logprobs: default_top_logprobs(),
            mirostat_mode: 0,
            mirostat_tau: default_mirostat_tau(),
            mirostat_eta: default_mirostat_eta(),
            dynatemp_range: 0.0,
            dynatemp_exponent: default_dynatemp_exponent(),
            json_mode: false,
            dry_multiplier: 0.0,
            dry_base: default_dry_base(),
            dry_allowed_length: 0,
            dry_sequence_breakers: Vec::new(),
            typical_p: 0.0,
        }
    }
}

impl GenerationConfig {
    /// Create a greedy decoding config (temperature = 0)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            ..Default::default()
        }
    }

    /// Create a creative sampling config
    pub fn creative() -> Self {
        Self {
            temperature: 1.2,
            top_p: 0.95,
            top_k: 50,
            ..Default::default()
        }
    }

    /// Create a balanced sampling config
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            ..Default::default()
        }
    }

    /// Check if greedy decoding should be used.
    ///
    /// Returns true only when temperature is 0. When a seed is set with
    /// temperature > 0, proper seeded stochastic sampling is used instead.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// Check if any token penalties are active
    pub fn has_penalties(&self) -> bool {
        self.repeat_penalty != 1.0 || self.frequency_penalty != 0.0 || self.presence_penalty != 0.0
    }
}
