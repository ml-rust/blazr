//! `hybrid` arm of [`Executor::generate`]: a KV cache for the attention
//! layers plus `LayeredSsmState` for the Mamba2 layers, through
//! `forward_hybrid`.
//!
//! A hybrid with no SSM layer, such as an MLA model, takes the contiguous
//! KV branch instead. The sampling, stop and streaming steps mirror that
//! branch.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::inference::{LayeredKvCache, LayeredKvCacheConfig, LayeredSsmState};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;

use super::super::executor::Executor;
use super::super::sampler::sampler_state::SamplerState;
use super::super::types::{FinishReason, GeneratedToken, PrefillCancel};

impl<R: Runtime<DType = DType>> Executor<R>
where
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
        + RoPEPackedOps<R>
        + boostr::quant::traits::DequantOps<R>,
{
    /// A KV cache sized for the attention layers only, and zeroed SSM state
    /// for the SSM layers. `num_layers()` counts both.
    ///
    /// # Errors
    ///
    /// A model that is not a hybrid with SSM layers, or a cache that does
    /// not allocate.
    pub(in crate::engine) fn hybrid_caches(
        &self,
        initial_capacity: usize,
        max_seq_len: usize,
    ) -> Result<(LayeredKvCache<R>, LayeredSsmState<R>)> {
        let model = self.model()?;
        let not_hybrid = || anyhow!("model {model:?} is not a hybrid model with SSM layers");
        let num_attention = model.num_attention_layers().ok_or_else(not_hybrid)?;
        let num_ssm = model.num_ssm_layers().ok_or_else(not_hybrid)?;
        let mamba = model.mamba_config().ok_or_else(not_hybrid)?;
        let kv_config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads: model.num_kv_heads().unwrap_or(1),
            initial_capacity,
            max_seq_len,
            head_dim: model.head_dim().unwrap_or(1),
            dtype: self.config.kv_cache_dtype()?,
        };
        let kv_cache = LayeredKvCache::new(num_attention, &kv_config, &self.device)
            .map_err(|e| anyhow!("Failed to create KV cache: {}", e))?;
        let ssm_state = LayeredSsmState::new(
            num_ssm,
            1,
            mamba,
            self.config.compute_dtype()?,
            &self.device,
        )
        .map_err(|e| anyhow!("Failed to create SSM state: {}", e))?;
        Ok((kv_cache, ssm_state))
    }

    /// Prefill then greedy/sampled decode through `forward_hybrid`.
    ///
    /// The prefill runs in chunks and reads `cancel` before each one. A
    /// cancelled prefill ends the stream with no item.
    pub(in crate::engine) fn generate_hybrid<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        gen_config: &'a GenerationConfig,
        input: Tensor<R>,
        max_tokens: usize,
        mut sampler: SamplerState<R>,
        cancel: PrefillCancel<'a>,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            let max_seq_len = self.config.max_seq_len();
            let initial_capacity = (prompt_tokens.len() + max_tokens).min(max_seq_len);
            let (mut kv_cache, mut ssm_state) = self.hybrid_caches(initial_capacity, max_seq_len)?;
            let chunk = self.config.inference.prefill_chunk_size;

            tracing::info!(phase = "prefill_start", backend = "hybrid", prompt_tokens = prompt_tokens.len());
            let prefilled = cancel.run(0..prompt_tokens.len(), chunk, |r| {
                let piece = self.prompt_piece(&input, prompt_tokens, r.clone())?;
                self.model()?
                    .forward_hybrid(&piece, &mut kv_cache, &mut ssm_state, r.start)
                    .map_err(|e| anyhow!("hybrid prefill of positions {}..{} failed: {e}", r.start, r.end))
            })?;
            let Some(mut logits) = prefilled else { return; };
            tracing::info!(phase = "prefill_end", backend = "hybrid");
            tracing::info!(phase = "decode_start", backend = "hybrid", max_tokens = max_tokens);
            for i in 0..max_tokens {
                let cur_logits = &logits;
                let pending = self.sample_begin(cur_logits, &mut sampler, gen_config, None)?;

                let next_input = pending.tensor.reshape(&[1, 1])?;
                let position = kv_cache.seq_len();
                let next_logits = self.model()?
                    .forward_hybrid(&next_input, &mut kv_cache, &mut ssm_state, position)
                    .map_err(|e| anyhow!("Forward pass failed: {}", e))?;

                let next_token = self.sample_finish(&pending, &mut sampler)?;
                let finish = self.decode_finish_reason(next_token, i, max_tokens);

                yield Ok(self.make_token(next_token, &mut sampler.text, cur_logits, gen_config, finish)?);
                if finish == Some(FinishReason::Eos) { break; }
                logits = next_logits;
            }
            tracing::info!(phase = "decode_end", backend = "hybrid");
        }
    }
}

#[cfg(test)]
pub(in crate::engine) mod tests {
    use std::collections::HashMap;
    use std::path::Path;

    use boostr::inference::{LayeredKvCache, LayeredSsmState};
    use boostr::model::{LoadedModel, UniversalConfig};
    use boostr::{CpuDevice, CpuRuntime, Tensor};
    use futures::StreamExt;

    use crate::config::BlazrConfig;
    use crate::engine::requests::batch_test_rig::greedy;
    use crate::engine::Executor;

    pub(in crate::engine) const VOCAB: usize = 24;
    const PROMPT: [u32; 5] = [3, 10, 17, 1, 8];
    const NEW_TOKENS: usize = 6;

    /// The `config.json` oxidizr writes: boostr's `UniversalConfig`.
    /// `ssm[i]` makes layer `i` Mamba2, every other layer MLA.
    pub(in crate::engine) fn config(ssm: &[bool], moe: bool) -> UniversalConfig {
        let ssm_layers: Vec<usize> = (0..ssm.len()).filter(|&i| ssm[i]).collect();
        let attention_layers: Vec<usize> = (0..ssm.len()).filter(|&i| !ssm[i]).collect();
        let mut yaml = format!(
            "model_type: {}\nvocab_size: {VOCAB}\nhidden_size: 16\nnum_layers: {}\n\
             max_seq_len: 32\nrms_norm_eps: 0.00001\n\
             attention:\n  num_heads: 2\n  kv_latent_dim: 8\n  q_latent_dim: 6\n  d_rope: 4\n",
            if ssm_layers.is_empty() {
                "llama"
            } else {
                "hybrid"
            },
            ssm.len()
        );
        yaml.push_str(if moe {
            "moe:\n  num_experts: 3\n  experts_per_tok: 2\n  shared_expert: true\n  \
             intermediate_size: 12\n"
        } else {
            "intermediate_size: 20\n"
        });
        if !ssm_layers.is_empty() {
            yaml.push_str(&format!(
                "ssm:\n  variant: mamba2\n  num_heads: 2\n  head_dim: 16\n  state_size: 4\n  \
                 chunk_size: 4\n  n_groups: 1\n  conv_kernel: 4\n  expand: 2\n\
                 hybrid_layers:\n  ssm_layers: {ssm_layers:?}\n  attention_layers: {attention_layers:?}\n"
            ));
        }
        serde_saphyr::from_str(&yaml).unwrap()
    }

    /// A seeded checkpoint of `config` under the names oxidizr writes, as
    /// `model.safetensors` in `dir`, with `config.json` when `with_config`.
    pub(in crate::engine) fn write_checkpoint(
        dir: &Path,
        config: &UniversalConfig,
        with_config: bool,
    ) {
        let device = CpuDevice::new();
        let mut shapes: Vec<(String, Vec<usize>, bool)> = vec![
            ("model.embed_tokens.weight".into(), vec![VOCAB, 16], false),
            ("model.norm.weight".into(), vec![16], true),
            ("lm_head.weight".into(), vec![VOCAB, 16], false),
        ];
        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            let mut add = |name: &str, shape: &[usize], norm: bool| {
                shapes.push((format!("{p}.{name}"), shape.to_vec(), norm))
            };
            if config.is_ssm_layer(i) {
                add("norm.weight", &[16], true);
                add("mamba2.in_proj.weight", &[74, 16], false);
                add("mamba2.conv1d.weight", &[40, 1, 4], false);
                add("mamba2.out_proj.weight", &[16, 32], false);
                add("mamba2.A_log", &[2], false);
                add("mamba2.dt_bias", &[2], false);
                add("mamba2.D", &[2], false);
                continue;
            }
            add("input_layernorm.weight", &[16], true);
            add("post_attention_layernorm.weight", &[16], true);
            add("self_attn.q_a_proj.weight", &[6, 16], false);
            add("self_attn.q_a_layernorm.weight", &[6], true);
            add("self_attn.q_b_proj.weight", &[24, 6], false);
            add("self_attn.kv_a_proj_with_mqa.weight", &[12, 16], false);
            add("self_attn.kv_a_layernorm.weight", &[8], true);
            add("self_attn.kv_b_proj.weight", &[32, 8], false);
            add("self_attn.o_proj.weight", &[16, 16], false);
            if config.moe.is_some() {
                add("moe.router.gate.weight", &[3, 16], false);
                for e in ["experts.0", "experts.1", "experts.2", "shared_expert"] {
                    add(&format!("moe.{e}.gate_proj.weight"), &[12, 16], false);
                    add(&format!("moe.{e}.up_proj.weight"), &[12, 16], false);
                    add(&format!("moe.{e}.down_proj.weight"), &[16, 12], false);
                }
            } else {
                add("mlp.gate_proj.weight", &[20, 16], false);
                add("mlp.up_proj.weight", &[20, 16], false);
                add("mlp.down_proj.weight", &[16, 20], false);
            }
        }
        let mut state = 0x6f78_6964_u64;
        let mut tensors = HashMap::new();
        for (name, shape, norm) in shapes {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let u = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    if norm {
                        1.0 + 0.4 * u
                    } else {
                        0.6 * u
                    }
                })
                .collect();
            tensors.insert(
                name,
                Tensor::<CpuRuntime>::from_slice(&data, &shape, &device).unwrap(),
            );
        }
        boostr::format::safetensors::save_safetensors(
            dir.join("model.safetensors"),
            &tensors,
            None,
        )
        .unwrap();
        if with_config {
            std::fs::write(
                dir.join("config.json"),
                serde_json::to_string(config).unwrap(),
            )
            .unwrap();
        }
    }

    pub(in crate::engine) fn load(dir: &Path) -> (LoadedModel<CpuRuntime>, BlazrConfig) {
        crate::loader::load_model::<CpuRuntime, _>(dir, &CpuDevice::new()).unwrap()
    }

    /// The argmax of the last logits row.
    fn argmax_last(logits: &Tensor<CpuRuntime>) -> u32 {
        let v: Vec<f32> = logits.contiguous().unwrap().to_vec();
        let row = &v[v.len() - VOCAB..];
        (0..VOCAB)
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap() as u32
    }

    /// Every logit of `PROMPT` and the greedy tokens after it, through the
    /// model's own forwards, with no executor.
    fn reference(model: &LoadedModel<CpuRuntime>) -> (Vec<f32>, Vec<u32>) {
        let device = CpuDevice::new();
        let heads = model.num_kv_heads().unwrap();
        let width = model.head_dim().unwrap();
        let layers = model.num_attention_layers().unwrap();
        let mut kv = LayeredKvCache::new_positional(
            layers,
            1,
            heads,
            4,
            32,
            width,
            boostr::DType::F32,
            &device,
        )
        .unwrap();
        let mut ssm = model.mamba_config().map(|m| {
            LayeredSsmState::new(
                model.num_ssm_layers().unwrap(),
                1,
                m,
                boostr::DType::F32,
                &device,
            )
            .unwrap()
        });
        let mut step = |ids: &[i64], at: usize| {
            let input = Tensor::<CpuRuntime>::from_slice(ids, &[1, ids.len()], &device).unwrap();
            match ssm.as_mut() {
                Some(state) => model.forward_hybrid(&input, &mut kv, state, at),
                None => model.forward_with_kv_cache(&input, &mut kv, at),
            }
            .unwrap()
        };
        let prompt: Vec<i64> = PROMPT.iter().map(|&t| t as i64).collect();
        let logits = step(&prompt, 0);
        let all = logits.contiguous().unwrap().to_vec::<f32>();
        let mut tokens = vec![argmax_last(&logits)];
        for i in 1..NEW_TOKENS {
            let last = *tokens.last().unwrap();
            tokens.push(argmax_last(&step(&[last as i64], PROMPT.len() + i - 1)));
        }
        (all, tokens)
    }

    /// The greedy tokens of `PROMPT` through the executor, after its warmup.
    async fn served(dir: &Path) -> Vec<u32> {
        let (model, config) = load(dir);
        let tokenizer = crate::tokenizer::from_pretrained("llama3").unwrap();
        let exec = Executor::new(model, config, tokenizer, CpuDevice::new(), 32).unwrap();
        exec.warmup().unwrap();
        let items: Vec<_> = exec.generate(&PROMPT, &greedy(NEW_TOKENS)).collect().await;
        items.into_iter().map(|t| t.unwrap().token_id).collect()
    }

    async fn check(ssm: &[bool], moe: bool) {
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path(), &config(ssm, moe), true);
        let (model, _) = load(dir.path());
        assert!(matches!(model, LoadedModel::Hybrid(_)), "{model:?}");
        assert_eq!(
            (model.num_kv_heads(), model.head_dim()),
            (Some(1), Some(12))
        );
        let (_, want) = reference(&model);
        assert_eq!(served(dir.path()).await, want, "ssm {ssm:?} moe {moe}");
    }

    #[tokio::test]
    async fn oxidizr_mla_checkpoint_generates_through_the_loader() {
        check(&[false, false], false).await;
    }

    #[tokio::test]
    async fn oxidizr_mla_moe_checkpoint_generates_through_the_loader() {
        check(&[false, false], true).await;
    }

    #[tokio::test]
    async fn oxidizr_mamba2_mla_moe_hybrid_generates_through_the_loader() {
        check(&[true, true, false], true).await;
    }

    /// With no `config.json`, the tensor names and shapes give the same model.
    #[test]
    fn a_hybrid_checkpoint_without_config_json_detects_the_same_model() {
        let config = config(&[true, false], true);
        let with = tempfile::tempdir().unwrap();
        let without = tempfile::tempdir().unwrap();
        write_checkpoint(with.path(), &config, true);
        write_checkpoint(without.path(), &config, false);
        let (detected, detected_config) = load(without.path());
        let detected_split = detected_config.model.hybrid_layers.clone().unwrap();
        assert_eq!(detected_split.ssm_layers, vec![0]);
        assert_eq!(detected_split.attention_layers, vec![1]);
        let (want, _) = reference(&load(with.path()).0);
        let (got, _) = reference(&detected);
        let diff = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        assert!(diff <= 1e-4, "max logit diff {diff}");
    }

    #[test]
    fn paged_attention_is_refused_for_an_mla_model() {
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path(), &config(&[false], false), true);
        let (model, mut config) = load(dir.path());
        config.inference.paged_attention = true;
        let tokenizer = crate::tokenizer::from_pretrained("llama3").unwrap();
        let err = Executor::new(model, config, tokenizer, CpuDevice::new(), 32)
            .err()
            .expect("paged attention refused")
            .to_string();
        assert!(err.contains("MLA"), "{err}");
    }
}
