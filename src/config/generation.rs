//! Generation configuration settings

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

    /// Top-k sampling (None = disabled)
    #[serde(default)]
    pub top_k: Option<usize>,

    /// Repetition penalty (1.0 = no penalty)
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,

    /// Frequency penalty
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Presence penalty
    #[serde(default)]
    pub presence_penalty: f32,

    /// Stop sequences
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Random seed (None = random)
    #[serde(default)]
    pub seed: Option<u64>,
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

fn default_repeat_penalty() -> f32 {
    1.0
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: None,
            repeat_penalty: default_repeat_penalty(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_sequences: Vec::new(),
            seed: None,
        }
    }
}

impl GenerationConfig {
    /// Create a greedy decoding config (temperature = 0)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            ..Default::default()
        }
    }

    /// Create a creative sampling config
    pub fn creative() -> Self {
        Self {
            temperature: 1.2,
            top_p: 0.95,
            top_k: Some(50),
            ..Default::default()
        }
    }

    /// Create a balanced sampling config
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            ..Default::default()
        }
    }

    /// Check if greedy decoding should be used
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}
