use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::detect::{DetectedConfig, LayerType, ModelFormat};

/// Model configuration - can be loaded from config.json or auto-detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Core dimensions (required)
    pub hidden_size: usize,
    #[serde(alias = "num_hidden_layers")]
    pub num_layers: usize,
    pub vocab_size: usize,

    // Tokenizer vocabulary name (e.g., "cl100k_base", "llama3", "deepseek_v3")
    // If not specified, will be auto-detected from vocab_size
    #[serde(default)]
    pub tokenizer_vocab: Option<String>,

    // Model format (auto-detected)
    #[serde(skip)]
    pub format: ModelFormat,

    // Layer types (auto-detected if not provided)
    #[serde(skip)]
    pub layer_types: Vec<LayerType>,

    // Whether embeddings are tied (lm_head = embed_tokens)
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // Mamba2 parameters (optional - only needed for Mamba2 layers)
    #[serde(default)]
    pub mamba2_num_heads: Option<usize>,
    #[serde(default)]
    pub mamba2_head_dim: Option<usize>,
    #[serde(default)]
    pub mamba2_state_size: Option<usize>,
    #[serde(default = "default_chunk_size")]
    pub mamba2_chunk_size: usize,
    #[serde(default)]
    pub mamba2_expand: Option<usize>,
    #[serde(default)]
    pub mamba2_conv_kernel: Option<usize>,
    #[serde(default = "default_n_groups")]
    pub mamba2_n_groups: usize,

    // Mamba3 parameters (optional - extends Mamba2 with new innovations)
    #[serde(default)]
    pub mamba3_enabled: Option<bool>,
    #[serde(default)]
    pub mamba3_complex_rope: Option<bool>,
    #[serde(default)]
    pub mamba3_mimo_rank: Option<usize>,
    #[serde(default)]
    pub mamba3_use_conv: Option<bool>,

    // MLA parameters (optional - only needed for MLA layers)
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub kv_latent_dim: Option<usize>,
    #[serde(default)]
    pub q_latent_dim: Option<usize>,
    #[serde(default)]
    pub d_rope: Option<usize>,

    // MoE parameters (optional - only needed for MoE layers)
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default = "default_experts_per_tok")]
    pub experts_per_tok: usize,
    #[serde(default)]
    pub shared_expert_enabled: bool,
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    // Standard transformer parameters (optional - only needed for standard layers)
    #[serde(default, alias = "num_key_value_heads")]
    pub num_kv_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,

    // RoPE parameters
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    // Layer configuration (legacy - used if layer_types not set)
    #[serde(default)]
    pub mamba_layers: Vec<usize>,

    // RMSNorm
    #[serde(default = "default_rms_norm_eps", alias = "rms_norm_eps")]
    pub rms_norm_eps: f64,

    // Inference settings
    #[serde(default = "default_max_seq_len", alias = "max_position_embeddings")]
    pub max_seq_len: usize,
}

/// RoPE scaling configuration (for Llama3-style scaling)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(default = "default_rope_factor")]
    pub factor: f64,
    #[serde(default = "default_low_freq_factor")]
    pub low_freq_factor: f64,
    #[serde(default = "default_high_freq_factor")]
    pub high_freq_factor: f64,
    #[serde(default = "default_original_max_pos")]
    pub original_max_position_embeddings: usize,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f64 {
    10000.0
}

fn default_rope_factor() -> f64 {
    1.0
}

fn default_low_freq_factor() -> f64 {
    1.0
}

fn default_high_freq_factor() -> f64 {
    4.0
}

fn default_original_max_pos() -> usize {
    8192
}

fn default_max_seq_len() -> usize {
    4096
}

fn default_chunk_size() -> usize {
    256
}

fn default_experts_per_tok() -> usize {
    2
}

fn default_n_groups() -> usize {
    1
}

impl Config {
    /// Load configuration from checkpoint directory (optional - may not exist)
    pub fn load(checkpoint_path: &Path) -> Result<Self> {
        let config_path = checkpoint_path.join("config.json");
        let contents = std::fs::read_to_string(&config_path)?;
        let mut config: Self = serde_json::from_str(&contents)?;

        // If layer_types not set but mamba_layers is, derive layer_types
        if config.layer_types.is_empty() && !config.mamba_layers.is_empty() {
            config.layer_types = (0..config.num_layers)
                .map(|i| {
                    if config.mamba_layers.contains(&i) {
                        LayerType::Mamba2
                    } else {
                        // Assume MLA+MoE for non-Mamba layers
                        LayerType::MlaWithMoe
                    }
                })
                .collect();
        }

        Ok(config)
    }

    /// Create config from auto-detected architecture
    pub fn from_detected(detected: DetectedConfig) -> Self {
        Self {
            hidden_size: detected.hidden_size,
            num_layers: detected.num_layers,
            vocab_size: detected.vocab_size,
            tokenizer_vocab: None, // Will be auto-detected from vocab_size
            format: detected.format,
            layer_types: detected.layer_types,
            tie_word_embeddings: detected.tie_word_embeddings,
            mamba2_num_heads: detected.mamba2_num_heads,
            mamba2_head_dim: detected.mamba2_head_dim,
            mamba2_state_size: detected.mamba2_state_size,
            mamba2_chunk_size: default_chunk_size(),
            mamba2_expand: detected.mamba2_expand,
            mamba2_conv_kernel: detected.mamba2_conv_kernel,
            mamba2_n_groups: default_n_groups(),
            mamba3_enabled: detected.mamba3_enabled,
            mamba3_complex_rope: detected.mamba3_complex_rope,
            mamba3_mimo_rank: detected.mamba3_mimo_rank,
            mamba3_use_conv: detected.mamba3_use_conv,
            num_attention_heads: detected.num_attention_heads,
            kv_latent_dim: detected.kv_latent_dim,
            q_latent_dim: detected.q_latent_dim,
            d_rope: detected.d_rope,
            num_experts: detected.num_experts,
            experts_per_tok: default_experts_per_tok(),
            shared_expert_enabled: detected.shared_expert_enabled,
            intermediate_size: detected.intermediate_size,
            num_kv_heads: detected.num_kv_heads,
            head_dim: detected.head_dim,
            rope_theta: default_rope_theta(),
            rope_scaling: None,
            mamba_layers: Vec::new(), // Not used when layer_types is set
            rms_norm_eps: default_rms_norm_eps(),
            max_seq_len: default_max_seq_len(),
        }
    }

    /// Merge auto-detected config with loaded config (detected fills in missing values)
    pub fn merge_with_detected(mut self, detected: DetectedConfig) -> Self {
        // Always use detected format and layer_types
        self.format = detected.format;
        if self.layer_types.is_empty() {
            self.layer_types = detected.layer_types;
        }

        // Use detected tie_word_embeddings if not set
        if !self.tie_word_embeddings {
            self.tie_word_embeddings = detected.tie_word_embeddings;
        }

        // Fill in missing Mamba2 params
        if self.mamba2_num_heads.is_none() {
            self.mamba2_num_heads = detected.mamba2_num_heads;
        }
        if self.mamba2_head_dim.is_none() {
            self.mamba2_head_dim = detected.mamba2_head_dim;
        }
        if self.mamba2_state_size.is_none() {
            self.mamba2_state_size = detected.mamba2_state_size;
        }
        if self.mamba2_expand.is_none() {
            self.mamba2_expand = detected.mamba2_expand;
        }
        if self.mamba2_conv_kernel.is_none() {
            self.mamba2_conv_kernel = detected.mamba2_conv_kernel;
        }

        // Fill in missing Mamba3 params
        if self.mamba3_enabled.is_none() {
            self.mamba3_enabled = detected.mamba3_enabled;
        }
        if self.mamba3_complex_rope.is_none() {
            self.mamba3_complex_rope = detected.mamba3_complex_rope;
        }
        if self.mamba3_mimo_rank.is_none() {
            self.mamba3_mimo_rank = detected.mamba3_mimo_rank;
        }
        if self.mamba3_use_conv.is_none() {
            self.mamba3_use_conv = detected.mamba3_use_conv;
        }

        // Fill in missing MLA params
        if self.num_attention_heads.is_none() {
            self.num_attention_heads = detected.num_attention_heads;
        }
        if self.kv_latent_dim.is_none() {
            self.kv_latent_dim = detected.kv_latent_dim;
        }
        if self.q_latent_dim.is_none() {
            self.q_latent_dim = detected.q_latent_dim;
        }
        if self.d_rope.is_none() {
            self.d_rope = detected.d_rope;
        }

        // Fill in missing MoE params
        if self.num_experts.is_none() {
            self.num_experts = detected.num_experts;
        }
        if self.intermediate_size.is_none() {
            self.intermediate_size = detected.intermediate_size;
        }
        if !self.shared_expert_enabled && detected.shared_expert_enabled {
            self.shared_expert_enabled = true;
        }

        // Fill in missing transformer params
        if self.num_kv_heads.is_none() {
            self.num_kv_heads = detected.num_kv_heads;
        }
        if self.head_dim.is_none() {
            self.head_dim = detected.head_dim;
        }

        self
    }

    /// Get layer type for a specific layer
    pub fn layer_type(&self, layer_idx: usize) -> LayerType {
        if layer_idx < self.layer_types.len() {
            self.layer_types[layer_idx]
        } else if self.mamba_layers.contains(&layer_idx) {
            LayerType::Mamba2
        } else {
            LayerType::MlaWithMoe
        }
    }

    /// Check if a layer uses Mamba2 or Mamba3 (vs attention)
    pub fn is_mamba_layer(&self, layer_idx: usize) -> bool {
        matches!(
            self.layer_type(layer_idx),
            LayerType::Mamba2 | LayerType::Mamba3
        )
    }

    /// Check if a layer uses Mamba3 specifically
    pub fn is_mamba3_layer(&self, layer_idx: usize) -> bool {
        matches!(self.layer_type(layer_idx), LayerType::Mamba3)
    }

    /// Check if a layer uses MoE (vs standard MLP)
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        matches!(self.layer_type(layer_idx), LayerType::MlaWithMoe)
    }

    /// Check if a layer uses standard transformer attention
    pub fn is_standard_transformer(&self, layer_idx: usize) -> bool {
        matches!(self.layer_type(layer_idx), LayerType::StandardTransformer)
    }

    /// Mamba2 inner dimension
    pub fn mamba2_d_inner(&self) -> usize {
        self.hidden_size * self.mamba2_expand.unwrap_or(2)
    }

    /// Mamba2 conv dimension (includes B, C projections)
    pub fn mamba2_conv_dim(&self) -> usize {
        let d_inner = self.mamba2_d_inner();
        let n_groups = self.mamba2_n_groups;
        let state_size = self.mamba2_state_size_required();
        d_inner + 2 * n_groups * state_size
    }

    /// Get required Mamba2 params (panics if not set)
    pub fn mamba2_num_heads_required(&self) -> usize {
        self.mamba2_num_heads.expect("mamba2_num_heads not set")
    }

    pub fn mamba2_head_dim_required(&self) -> usize {
        self.mamba2_head_dim.expect("mamba2_head_dim not set")
    }

    pub fn mamba2_state_size_required(&self) -> usize {
        self.mamba2_state_size.expect("mamba2_state_size not set")
    }

    pub fn mamba2_conv_kernel_required(&self) -> usize {
        self.mamba2_conv_kernel.unwrap_or(4)
    }

    pub fn mamba2_n_groups(&self) -> usize {
        self.mamba2_n_groups
    }

    /// Get required attention params
    pub fn num_attention_heads_required(&self) -> usize {
        self.num_attention_heads
            .expect("num_attention_heads not set")
    }

    pub fn kv_latent_dim_required(&self) -> usize {
        self.kv_latent_dim.expect("kv_latent_dim not set")
    }

    pub fn q_latent_dim_required(&self) -> usize {
        self.q_latent_dim.expect("q_latent_dim not set")
    }

    pub fn d_rope_required(&self) -> usize {
        self.d_rope.expect("d_rope not set")
    }

    /// Get required MoE params
    pub fn num_experts_required(&self) -> usize {
        self.num_experts.expect("num_experts not set")
    }

    pub fn intermediate_size_required(&self) -> usize {
        self.intermediate_size.expect("intermediate_size not set")
    }

    /// Check if Mamba3 is enabled
    pub fn mamba3_is_enabled(&self) -> bool {
        self.mamba3_enabled.unwrap_or(false)
    }

    /// Check if Mamba3 complex RoPE is enabled
    pub fn mamba3_complex_rope_enabled(&self) -> bool {
        self.mamba3_complex_rope.unwrap_or(true)
    }

    /// Get Mamba3 MIMO rank (0 = SISO)
    pub fn mamba3_mimo_rank_value(&self) -> usize {
        self.mamba3_mimo_rank.unwrap_or(0)
    }

    /// Check if Mamba3 uses conv (default: false, trapezoidal replaces it)
    pub fn mamba3_use_conv_enabled(&self) -> bool {
        self.mamba3_use_conv.unwrap_or(false)
    }

    /// Get the tokenizer vocabulary name
    /// If not specified in config, auto-detect from vocab_size
    pub fn tokenizer_vocab_name(&self) -> String {
        if let Some(ref vocab) = self.tokenizer_vocab {
            return vocab.clone();
        }

        // Auto-detect from vocab_size
        // Reference sizes (with agent tokens):
        // - cl100k_base: 100331 tokens (100277 + 54 agent tokens)
        // - o200k_base: 200073 tokens (200019 + 54 agent tokens)
        // - llama3: 128354 tokens (128300 + 54 agent tokens)
        // - deepseek_v3: 128954 tokens (128900 + 54 agent tokens)
        match self.vocab_size {
            // cl100k_base range: typically 100257 (base) to 100331 (with all agent tokens)
            v if v <= 100350 => "cl100k_base".to_string(),
            // llama3 range: 128000 (base) to 128354 (with agent tokens)
            v if v <= 128400 => "llama3".to_string(),
            // deepseek_v3 range: 128000 (base) to 128954 (with agent tokens)
            v if v <= 129000 => "deepseek_v3".to_string(),
            // o200k_base range: 199999 (base) to 200073 (with agent tokens)
            v if v <= 200100 => "o200k_base".to_string(),
            // Default to llama3 for unknown sizes
            _ => "llama3".to_string(),
        }
    }
}
