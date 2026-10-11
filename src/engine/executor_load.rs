//! One-call model loading for embedders: path in, ready [`Executor`] out.
//!
//! The CLI, the C ABI and a host with no file system share these loaders.
//! [`load_executor`] reads a GGUF file or a SafeTensors checkpoint
//! directory. [`load_executor_from_bytes`] takes the bytes of a GGUF file or
//! of a single SafeTensors file plus its `config.json`.

use std::path::Path;

use anyhow::{anyhow, Result};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};

use super::executor::Executor;
use crate::config::{DtypeRequest, DEFAULT_NUM_CTX};
use crate::loader::{self, detect_model_source, ModelFormat, SafeTensorsBytes};
use crate::model::chat_template::source::template_from_tokenizer_config;
use crate::model::chat_template::ChatTemplate;
use crate::tokenizer::checkpoint_tokenizer;

/// How [`load_executor`] loads a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    /// The compute dtype. `None` takes the checkpoint's dtype.
    pub dtype: Option<DtypeRequest>,
    /// The KV cache context length.
    pub num_ctx: usize,
    /// Run one warmup forward pass before returning.
    pub warmup: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            dtype: None,
            num_ctx: DEFAULT_NUM_CTX,
            warmup: true,
        }
    }
}

/// Load the GGUF file or SafeTensors checkpoint at `path` on `device`.
///
/// A GGUF file carries its own tokenizer. A SafeTensors checkpoint takes the
/// tokenizer files in its directory. The chat template is the model's own.
///
/// # Errors
///
/// An error when the path holds no model, the weights do not load, or the
/// warmup pass fails.
pub fn load_executor<R>(
    path: &Path,
    device: R::Device,
    options: &LoadOptions,
) -> Result<Executor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::RoPEPackedOps<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    if !path.exists() {
        return Err(anyhow!(
            "model path {} does not exist. Pass a GGUF file or a SafeTensors directory",
            path.display()
        ));
    }
    let source = detect_model_source(path)?;
    let dtype = options.dtype.as_ref();
    let (model, config, tokenizer) = match source.format {
        ModelFormat::Gguf => loader::load_gguf_with_tokenizer_as::<R, _>(path, &device, dtype)?,
        ModelFormat::SafeTensors => {
            let (model, config) = loader::load_model_as::<R, _>(path, &device, dtype)?;
            let tokenizer = checkpoint_tokenizer(path, config.vocab_size())?;
            (model, config, tokenizer)
        }
    };
    let template = ChatTemplate::detect(path, config.model_type(), None)?;
    let executor = Executor::new(model, config, tokenizer, device, options.num_ctx)?
        .with_chat_template(template);
    if options.warmup {
        executor.warmup()?;
    }
    Ok(executor)
}

/// A whole model held in memory, as [`load_executor_from_bytes`] takes it.
#[derive(Debug, Clone)]
pub enum ModelBytes {
    /// The bytes of one GGUF file. It carries its tokenizer and chat template.
    Gguf(Vec<u8>),
    /// One `.safetensors` file plus its JSON files.
    SafeTensors(SafeTensorsBytes),
}

/// [`load_executor`] for a model in memory, such as one a browser fetched.
///
/// A SafeTensors model takes its chat template from `tokenizer_config_json`
/// when that text has one. Otherwise the template follows the model type.
///
/// # Errors
///
/// An error when the bytes hold no model, the weights do not load, or the
/// warmup pass fails.
pub fn load_executor_from_bytes<R>(
    bytes: ModelBytes,
    device: R::Device,
    options: &LoadOptions,
) -> Result<Executor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::RoPEPackedOps<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    let dtype = options.dtype.as_ref();
    let (model, config, tokenizer, shipped) = match bytes {
        ModelBytes::Gguf(bytes) => {
            let loaded = loader::load_gguf_bytes_with_tokenizer_as::<R>(bytes, &device, dtype)?;
            (
                loaded.model,
                loaded.config,
                loaded.tokenizer,
                loaded.chat_template,
            )
        }
        ModelBytes::SafeTensors(input) => {
            let shipped = input
                .tokenizer_config_json
                .as_deref()
                .map(template_from_tokenizer_config)
                .transpose()
                .map_err(|e| anyhow!("tokenizer_config.json does not parse: {e}"))?
                .flatten();
            let (model, config, tokenizer) =
                loader::load_safetensors_bytes::<R>(input, &device, dtype)?;
            (model, config, tokenizer, shipped)
        }
    };
    let template = match shipped {
        Some(shipped) => shipped.compile()?,
        None => ChatTemplate::from_model_type(config.model_type()),
    };
    let executor = Executor::new(model, config, tokenizer, device, options.num_ctx)?
        .with_chat_template(template);
    if options.warmup {
        executor.warmup()?;
    }
    Ok(executor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::{CpuDevice, CpuRuntime};
    use futures::StreamExt;

    use crate::config::GenerationConfig;

    const HIDDEN: usize = 16;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const FFN: usize = 32;
    const HEAD_DIM: usize = HIDDEN / HEADS;

    /// Deterministic small weights.
    fn weights(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
            })
            .collect()
    }

    /// Llama tensors in Hugging Face names and row-major shapes.
    fn llama_tensors(vocab: usize) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let ones = |n: usize| vec![1.0f32; n];
        let mut seed = 0;
        let mut dense = |rows: usize, cols: usize| {
            seed += 1;
            weights(rows * cols, seed)
        };
        let layer = "model.layers.0";
        vec![
            (
                "model.embed_tokens.weight".into(),
                vec![vocab, HIDDEN],
                dense(vocab, HIDDEN),
            ),
            (
                format!("{layer}.input_layernorm.weight"),
                vec![HIDDEN],
                ones(HIDDEN),
            ),
            (
                format!("{layer}.self_attn.q_proj.weight"),
                vec![HIDDEN, HIDDEN],
                dense(HIDDEN, HIDDEN),
            ),
            (
                format!("{layer}.self_attn.k_proj.weight"),
                vec![KV_HEADS * HEAD_DIM, HIDDEN],
                dense(KV_HEADS * HEAD_DIM, HIDDEN),
            ),
            (
                format!("{layer}.self_attn.v_proj.weight"),
                vec![KV_HEADS * HEAD_DIM, HIDDEN],
                dense(KV_HEADS * HEAD_DIM, HIDDEN),
            ),
            (
                format!("{layer}.self_attn.o_proj.weight"),
                vec![HIDDEN, HIDDEN],
                dense(HIDDEN, HIDDEN),
            ),
            (
                format!("{layer}.post_attention_layernorm.weight"),
                vec![HIDDEN],
                ones(HIDDEN),
            ),
            (
                format!("{layer}.mlp.gate_proj.weight"),
                vec![FFN, HIDDEN],
                dense(FFN, HIDDEN),
            ),
            (
                format!("{layer}.mlp.up_proj.weight"),
                vec![FFN, HIDDEN],
                dense(FFN, HIDDEN),
            ),
            (
                format!("{layer}.mlp.down_proj.weight"),
                vec![HIDDEN, FFN],
                dense(HIDDEN, FFN),
            ),
            ("model.norm.weight".into(), vec![HIDDEN], ones(HIDDEN)),
            (
                "lm_head.weight".into(),
                vec![vocab, HIDDEN],
                dense(vocab, HIDDEN),
            ),
        ]
    }

    /// A SafeTensors image: header length, JSON header, F32 data.
    fn safetensors_bytes(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape, data) in tensors {
            let end = offset + data.len() * 4;
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}),
            );
            offset = end;
        }
        let header = serde_json::Value::Object(header).to_string();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        for (_, _, data) in tensors {
            out.extend(data.iter().flat_map(|v| v.to_le_bytes()));
        }
        out
    }

    fn generate_one<R>(executor: &Executor<R>, prompt: &[u32]) -> u32
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + UnaryOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + SamplingOps<R>
            + boostr::GrammarDfaOps<R>
            + ModelClient<R>
            + boostr::RoPEPackedOps<R>
            + boostr::quant::DequantOps<R>,
    {
        let config = GenerationConfig {
            max_tokens: 1,
            temperature: 0.0,
            repeat_penalty: 1.0,
            ..GenerationConfig::default()
        };
        futures::executor::block_on(async {
            let mut tokens = std::pin::pin!(executor.generate(prompt, &config));
            let first = tokens.next().await.expect("one token").expect("generate");
            assert!(tokens.next().await.is_none(), "max_tokens is 1");
            first.token_id
        })
    }

    #[test]
    fn safetensors_from_bytes_generates_a_token() {
        // 32000 is the bundled Mistral v1 vocabulary.
        let vocab = 32000;
        let config_json = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": HIDDEN,
            "intermediate_size": FFN,
            "num_hidden_layers": 1,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "vocab_size": vocab,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "torch_dtype": "float32",
            "tie_word_embeddings": false,
            "bos_token_id": 1,
            "eos_token_id": 2
        })
        .to_string();
        let bytes = ModelBytes::SafeTensors(SafeTensorsBytes {
            weights: safetensors_bytes(&llama_tensors(vocab)),
            config_json: Some(config_json),
            tokenizer_json: None,
            tokenizer_config_json: None,
        });
        let options = LoadOptions {
            num_ctx: 64,
            ..LoadOptions::default()
        };
        let executor =
            load_executor_from_bytes::<CpuRuntime>(bytes, CpuDevice::new(), &options).unwrap();
        assert_eq!(executor.vocab_size(), vocab);
        let token = generate_one(&executor, &[1, 415, 5565]);
        assert!((token as usize) < vocab);
    }

    /// GGUF metadata values the test writes.
    enum Kv {
        U32(u32),
        F32(f32),
        Str(&'static str),
        Strs(Vec<String>),
        F32s(Vec<f32>),
        I32s(Vec<i32>),
    }

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    /// A GGUF v3 image with F32 tensors. `tensors` use row-major shapes.
    fn gguf_bytes(pairs: &[(&str, Kv)], tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        const ALIGN: usize = 32;
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
        for (key, value) in pairs {
            put_str(&mut out, key);
            match value {
                Kv::U32(v) => {
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Kv::F32(v) => {
                    out.extend_from_slice(&6u32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Kv::Str(v) => {
                    out.extend_from_slice(&8u32.to_le_bytes());
                    put_str(&mut out, v);
                }
                Kv::Strs(items) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&8u32.to_le_bytes());
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for item in items {
                        put_str(&mut out, item);
                    }
                }
                Kv::F32s(items) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&6u32.to_le_bytes());
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    out.extend(items.iter().flat_map(|v| v.to_le_bytes()));
                }
                Kv::I32s(items) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&5u32.to_le_bytes());
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    out.extend(items.iter().flat_map(|v| v.to_le_bytes()));
                }
            }
        }
        let mut offset = 0usize;
        for (name, shape, data) in tensors {
            put_str(&mut out, name);
            out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            // GGML order lists the innermost axis first.
            for dim in shape.iter().rev() {
                out.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            out.extend_from_slice(&0u32.to_le_bytes()); // GGML F32
            out.extend_from_slice(&(offset as u64).to_le_bytes());
            offset += (data.len() * 4).div_ceil(ALIGN) * ALIGN;
        }
        for (_, _, data) in tensors {
            out.resize(out.len().div_ceil(ALIGN) * ALIGN, 0);
            out.extend(data.iter().flat_map(|v| v.to_le_bytes()));
        }
        out
    }

    /// Hugging Face tensor name to its GGUF name.
    fn gguf_name(hf: &str) -> String {
        let table = [
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "blk.0.attn_q.weight",
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "blk.0.attn_k.weight",
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                "blk.0.attn_v.weight",
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "blk.0.attn_output.weight",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "blk.0.ffn_norm.weight",
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "blk.0.ffn_gate.weight",
            ),
            ("model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
            (
                "model.layers.0.mlp.down_proj.weight",
                "blk.0.ffn_down.weight",
            ),
        ];
        table
            .iter()
            .find(|(from, _)| *from == hf)
            .map(|(_, to)| (*to).to_string())
            .unwrap_or_else(|| panic!("no GGUF name for {hf}"))
    }

    #[test]
    fn gguf_from_bytes_generates_a_token() {
        // SentencePiece pieces: three controls, then single characters.
        let mut tokens: Vec<String> = ["<unk>", "<s>", "</s>", "\u{2581}"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        tokens.extend(('a'..='z').map(|c| c.to_string()));
        tokens.extend(('a'..='z').map(|c| format!("\u{2581}{c}")));
        let vocab = tokens.len();
        let scores: Vec<f32> = (0..vocab).map(|i| -(i as f32)).collect();
        let types: Vec<i32> = (0..vocab).map(|i| if i < 3 { 3 } else { 1 }).collect();
        let tensors: Vec<_> = llama_tensors(vocab)
            .into_iter()
            .map(|(name, shape, data)| (gguf_name(&name), shape, data))
            .collect();
        let pairs = [
            ("general.architecture", Kv::Str("llama")),
            ("llama.context_length", Kv::U32(64)),
            ("llama.embedding_length", Kv::U32(HIDDEN as u32)),
            ("llama.block_count", Kv::U32(1)),
            ("llama.feed_forward_length", Kv::U32(FFN as u32)),
            ("llama.attention.head_count", Kv::U32(HEADS as u32)),
            ("llama.attention.head_count_kv", Kv::U32(KV_HEADS as u32)),
            ("llama.attention.layer_norm_rms_epsilon", Kv::F32(1e-5)),
            ("llama.rope.freq_base", Kv::F32(10000.0)),
            ("tokenizer.ggml.model", Kv::Str("llama")),
            ("tokenizer.ggml.tokens", Kv::Strs(tokens)),
            ("tokenizer.ggml.scores", Kv::F32s(scores)),
            ("tokenizer.ggml.token_type", Kv::I32s(types)),
            ("tokenizer.ggml.bos_token_id", Kv::U32(1)),
            ("tokenizer.ggml.eos_token_id", Kv::U32(2)),
            ("tokenizer.ggml.unknown_token_id", Kv::U32(0)),
        ];
        let bytes = ModelBytes::Gguf(gguf_bytes(&pairs, &tensors));
        let options = LoadOptions {
            num_ctx: 64,
            ..LoadOptions::default()
        };
        let executor =
            load_executor_from_bytes::<CpuRuntime>(bytes, CpuDevice::new(), &options).unwrap();
        assert_eq!(executor.vocab_size(), vocab);
        let prompt = splintr::Tokenize::encode(executor.tokenizer(), "a cab");
        assert!(!prompt.is_empty());
        let token = generate_one(&executor, &prompt);
        assert!((token as usize) < vocab);
    }

    #[test]
    fn bytes_that_are_no_model_are_an_error() {
        let options = LoadOptions::default();
        let err = load_executor_from_bytes::<CpuRuntime>(
            ModelBytes::Gguf(b"not a gguf".to_vec()),
            CpuDevice::new(),
            &options,
        )
        .err()
        .expect("garbage is refused");
        assert!(format!("{err:#}").contains("not a GGUF file"), "{err:#}");
    }
}
