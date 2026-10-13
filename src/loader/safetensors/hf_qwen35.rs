//! Hugging Face Qwen3.5 `config.json` detection, over boostr's
//! `qwen35_config_from_hf`.

use anyhow::{anyhow, Result};
use boostr::model::qwen35::{is_qwen35_hf_config, qwen35_config_from_hf};
use boostr::model::UniversalConfig;
use serde_json::Value;

/// The `qwen35` config of a Hugging Face Qwen3.5 `config.json`, or `None`
/// when the text is not JSON or its top-level `model_type` is not
/// `qwen3_5` or `qwen3_5_text`.
///
/// `Qwen3_5ForConditionalGeneration` (`qwen3_5`) nests the text model
/// under `text_config`. `Qwen3_5ForCausalLM` (`qwen3_5_text`) keeps it at
/// the top level. boostr reads both.
///
/// # Errors
///
/// `Some(Err)` when the config is Qwen3.5 but boostr refuses it: a missing
/// key, or an option boostr does not run.
pub(super) fn qwen35_config_of(config_json: &str) -> Option<Result<UniversalConfig>> {
    let json = serde_json::from_str::<Value>(config_json).ok()?;
    is_qwen35_hf_config(&json)
        .then(|| qwen35_config_from_hf(&json).map_err(|e| anyhow!("config.json: {e}")))
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::config::config_from_text;
    use super::*;
    use boostr::DType;

    /// `decider-4b/config.json` (`Qwen3_5ForCausalLM`, flat), as published.
    pub(in super::super) const DECIDER_CONFIG: &str = r#"{
  "architectures": ["Qwen3_5ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attn_output_gate": true,
  "bos_token_id": null,
  "dtype": "bfloat16",
  "eos_token_id": 248044,
  "full_attention_interval": 4,
  "head_dim": 256,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "intermediate_size": 9216,
  "layer_types": [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention"
  ],
  "linear_conv_kernel_dim": 4,
  "linear_key_head_dim": 128,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 32,
  "linear_value_head_dim": 128,
  "mamba_ssm_dtype": "float32",
  "max_position_embeddings": 262144,
  "mlp_only_layers": [],
  "model_type": "qwen3_5_text",
  "mtp_num_hidden_layers": 1,
  "mtp_use_dedicated_embeddings": false,
  "num_attention_heads": 16,
  "num_hidden_layers": 32,
  "num_key_value_heads": 4,
  "pad_token_id": null,
  "partial_rotary_factor": 0.25,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "default"
  },
  "tie_word_embeddings": true,
  "transformers_version": "5.17.0",
  "use_cache": true,
  "vocab_size": 248320
}"#;

    /// `jebadiah-4b-v2/config.json` (`Qwen3_5ForConditionalGeneration`,
    /// nested), with `vision_config` cut to two keys.
    pub(in super::super) const JEBADIAH_CONFIG: &str = r#"{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "image_token_id": 248056,
  "model_type": "qwen3_5",
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "attn_output_gate": true,
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9216,
    "layer_types": [
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention"
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "mlp_only_layers": [],
    "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": 16,
    "num_hidden_layers": 32,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": true,
    "use_cache": true,
    "vocab_size": 248320,
    "mamba_ssm_dtype": "float32",
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "rope_type": "default",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25
    }
  },
  "tie_word_embeddings": true,
  "transformers_version": "4.57.0.dev0",
  "video_token_id": 248057,
  "vision_config": {"depth": 24, "model_type": "qwen3_5"},
  "vision_end_token_id": 248054,
  "vision_start_token_id": 248053
}"#;

    /// The fields both real configs share.
    fn check_4b(config: &UniversalConfig) {
        assert_eq!(config.model_type, "qwen35");
        assert_eq!(config.vocab_size, 248_320);
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.intermediate_size, Some(9216));
        assert!(config.tie_word_embeddings);
        assert!(config.vision.is_none());
        let layers = config.hybrid_layers.as_ref().unwrap();
        let want: Vec<usize> = (0..32).filter(|i| i % 4 == 3).collect();
        assert_eq!(layers.attention_layers, want);
        let gdn = config.gdn.as_ref().unwrap();
        assert_eq!((gdn.key_heads, gdn.value_heads), (16, 32));
        assert_eq!(gdn.state_size, 128);
        let attn = config.qwen35_attention.as_ref().unwrap();
        assert_eq!((attn.num_heads, attn.num_kv_heads), (16, 4));
        assert_eq!((attn.head_dim, attn.rope_dim), (256, 64));
    }

    fn no_detect() -> Result<crate::model::detect::DetectedConfig> {
        Err(anyhow!("tensor-name detection ran for a Qwen3.5 config"))
    }

    #[test]
    fn flat_text_config_is_qwen35() {
        check_4b(&qwen35_config_of(DECIDER_CONFIG).unwrap().unwrap());
    }

    #[test]
    fn nested_conditional_generation_config_is_qwen35() {
        check_4b(&qwen35_config_of(JEBADIAH_CONFIG).unwrap().unwrap());
    }

    #[test]
    fn other_configs_are_not_qwen35() {
        let llama = r#"{"architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "vocab_size": 32, "hidden_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 2}"#;
        assert!(qwen35_config_of(llama).is_none());
        assert!(qwen35_config_of("not json").is_none());
    }

    /// Both real configs reach the `qwen35` branch before any generic
    /// parse, and tensor-name detection never runs.
    #[test]
    fn config_from_text_takes_the_qwen35_branch() {
        for text in [DECIDER_CONFIG, JEBADIAH_CONFIG] {
            let config = config_from_text(Some(text), no_detect, DType::F32).unwrap();
            check_4b(&config.model);
            assert_eq!(config.dtype(), "f32");
        }
    }

    /// A Qwen3.5 config boostr refuses is an error, never a fall-through
    /// to detection or the generic parse.
    #[test]
    fn refused_qwen35_config_is_an_error() {
        let text = DECIDER_CONFIG.replace(r#""hidden_act": "silu""#, r#""hidden_act": "gelu""#);
        assert!(qwen35_config_of(&text).unwrap().is_err());
        let err = config_from_text(Some(&text), no_detect, DType::F32).unwrap_err();
        assert!(err.to_string().contains("hidden_act"), "{err}");
    }
}
