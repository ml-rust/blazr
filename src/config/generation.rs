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
