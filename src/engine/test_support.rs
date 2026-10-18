//! Test fixtures shared by the engine's unit tests.

use anyhow::Result;

use boostr::model::traits::Model;
use boostr::model::LoadedModel;
use boostr::nn::{VarBuilder, VarMap};
use boostr::{CpuDevice, CpuRuntime, Tensor};

use crate::config::BlazrConfig;
use crate::engine::Executor;

/// Config of a one-layer llama small enough to build in a unit test.
pub(crate) const TINY_LLAMA_YAML: &str =
    "model_type: llama\nvocab_size: 32\nhidden_size: 16\nnum_layers: 1\n\
     max_seq_len: 16\nintermediate_size: 32\nattention:\n  num_heads: 2\n";

/// Hidden width, FFN width and vocab of [`TINY_LLAMA_YAML`].
const HIDDEN: usize = 16;
const INTER: usize = 32;
const VOCAB: usize = 32;

/// A started executor over a tiny llama with all-zero weights.
pub(crate) fn tiny_executor() -> Result<Executor<CpuRuntime>> {
    let device = CpuDevice::new();
    let config: boostr::model::UniversalConfig = serde_saphyr::from_str(TINY_LLAMA_YAML)?;
    let model = boostr::model::llama::Llama::<CpuRuntime>::from_config(&config, &device)?;
    let tokenizer = crate::tokenizer::from_pretrained("llama3")?;
    Executor::new(
        LoadedModel::Llama(Box::new(model)),
        BlazrConfig::from_universal_with_dtype(config, "f32"),
        tokenizer,
        device,
        16,
    )
}

/// The paged KV pool a [`seeded_executor`] serves.
#[derive(Clone, Copy)]
pub(crate) struct PagedPool {
    /// Tokens per KV block.
    pub block_size: usize,
    /// Blocks in the shared pool.
    pub blocks: usize,
}

/// A started executor over the tiny llama with seeded random weights.
///
/// Different prompts produce different logits, so its greedy ids depend on
/// the KV cache. With `paged`, paged attention is on over that pool and the
/// executor admits two sequences per batch.
pub(crate) fn seeded_executor(seed: u64, paged: Option<PagedPool>) -> Result<Executor<CpuRuntime>> {
    let device = CpuDevice::new();
    let config: boostr::model::UniversalConfig = serde_saphyr::from_str(TINY_LLAMA_YAML)?;
    let mut weights = SeededWeights {
        state: seed,
        map: VarMap::new(),
    };
    weights.fill(&device)?;
    let mut map = weights.map;
    let mut vb = VarBuilder::new(&mut map, &device);
    let model = LoadedModel::<CpuRuntime>::load(&config, &mut vb)?;

    let mut blazr_config = BlazrConfig::from_universal_with_dtype(config, "f32");
    if let Some(pool) = paged {
        blazr_config.inference.paged_attention = true;
        blazr_config.inference.block_size = pool.block_size;
        blazr_config.inference.kv_pool_blocks = pool.blocks;
        blazr_config.inference.max_batch_size = 2;
    }
    let tokenizer = crate::tokenizer::from_pretrained("llama3")?;
    Executor::new(model, blazr_config, tokenizer, device, 16)
}

/// Checkpoint tensors for [`TINY_LLAMA_YAML`], drawn from a linear
/// congruential generator.
struct SeededWeights {
    state: u64,
    map: VarMap<CpuRuntime>,
}

impl SeededWeights {
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((self.state >> 40) as f32) / ((1u64 << 24) as f32);
        lo + (hi - lo) * unit
    }

    fn put(
        &mut self,
        device: &CpuDevice,
        name: &str,
        shape: &[usize],
        lo: f32,
        hi: f32,
    ) -> Result<()> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| self.uniform(lo, hi)).collect();
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, shape, device)?;
        self.map.insert(name.to_string(), tensor);
        Ok(())
    }

    fn linear(&mut self, device: &CpuDevice, name: &str, out: usize, inp: usize) -> Result<()> {
        let scale = 1.0 / (inp as f32).sqrt();
        self.put(device, name, &[out, inp], -scale, scale)
    }

    fn fill(&mut self, device: &CpuDevice) -> Result<()> {
        self.put(
            device,
            "model.embed_tokens.weight",
            &[VOCAB, HIDDEN],
            -1.0,
            1.0,
        )?;
        let p = "model.layers.0";
        self.put(
            device,
            &format!("{p}.input_layernorm.weight"),
            &[HIDDEN],
            0.5,
            1.5,
        )?;
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            self.linear(
                device,
                &format!("{p}.self_attn.{proj}.weight"),
                HIDDEN,
                HIDDEN,
            )?;
        }
        self.put(
            device,
            &format!("{p}.post_attention_layernorm.weight"),
            &[HIDDEN],
            0.5,
            1.5,
        )?;
        self.linear(device, &format!("{p}.mlp.gate_proj.weight"), INTER, HIDDEN)?;
        self.linear(device, &format!("{p}.mlp.up_proj.weight"), INTER, HIDDEN)?;
        self.linear(device, &format!("{p}.mlp.down_proj.weight"), HIDDEN, INTER)?;
        self.put(device, "model.norm.weight", &[HIDDEN], 0.5, 1.5)?;
        self.linear(device, "lm_head.weight", VOCAB, HIDDEN)
    }
}
