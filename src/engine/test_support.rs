//! Test fixtures shared by the engine's unit tests.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};

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

/// Context length of a [`full_vocab_executor`]: room for a short JSON object.
pub(crate) const FULL_VOCAB_CTX: usize = 64;

/// The tiny llama with all-zero weights over every llama3 token id.
///
/// Its logits are all zero, so a test steers it with `logit_bias`. The EOS
/// id is inside its vocab, so a grammar can end generation. With `paged`,
/// paged attention is on over that pool and two sequences batch.
pub(crate) fn full_vocab_executor(paged: Option<PagedPool>) -> Result<Executor<CpuRuntime>> {
    let (tokenizer, vocab) = full_vocab_tokenizer()?;
    let yaml = full_vocab_yaml(vocab);
    let device = CpuDevice::new();
    let config: boostr::model::UniversalConfig = serde_saphyr::from_str(&yaml)?;
    let model = boostr::model::llama::Llama::<CpuRuntime>::from_config(&config, &device)?;
    let mut blazr_config = BlazrConfig::from_universal_with_dtype(config, "f32");
    enable_paged(&mut blazr_config, paged);
    Executor::new(
        LoadedModel::Llama(Box::new(model)),
        blazr_config,
        tokenizer,
        device,
        FULL_VOCAB_CTX,
    )
}

/// The llama3 tokenizer and a vocab size that holds every id, EOS included.
fn full_vocab_tokenizer() -> Result<(crate::tokenizer::AnyTokenizer, usize)> {
    let tokenizer = crate::tokenizer::from_pretrained("llama3")?;
    let eos = tokenizer
        .eos_token_id()
        .ok_or_else(|| anyhow!("llama3 states no end-of-sequence token"))?;
    let vocab = splintr::Tokenize::vocab_size(&tokenizer).max(eos as usize + 1);
    Ok((tokenizer, vocab))
}

/// [`TINY_LLAMA_YAML`] with `vocab` rows and a context of [`FULL_VOCAB_CTX`].
fn full_vocab_yaml(vocab: usize) -> String {
    TINY_LLAMA_YAML
        .replace("vocab_size: 32", &format!("vocab_size: {vocab}"))
        .replace("max_seq_len: 16", &format!("max_seq_len: {FULL_VOCAB_CTX}"))
}

/// Turn paged attention on over `paged`, when set.
fn enable_paged(config: &mut BlazrConfig, paged: Option<PagedPool>) {
    if let Some(pool) = paged {
        config.inference.paged_attention = true;
        config.inference.block_size = pool.block_size;
        config.inference.kv_pool_blocks = pool.blocks;
        config.inference.max_batch_size = 2;
    }
}

/// Free blocks in `exec`'s shared KV pool.
pub(crate) fn free_pool_blocks(exec: &Executor<CpuRuntime>) -> Result<usize> {
    use boostr::inference::BlockAllocator;

    let shared = exec
        .shared_allocator()
        .ok_or_else(|| anyhow!("the executor has no shared allocator"))?;
    let free = shared
        .lock()
        .map_err(|e| anyhow!("lock poisoned: {e}"))?
        .free_block_count();
    Ok(free)
}

/// The id of the one-token piece `piece` in `exec`'s vocab.
pub(crate) fn piece_id(exec: &Executor<CpuRuntime>, piece: &[u8]) -> Result<u32> {
    (0..exec.vocab_size() as u32)
        .find(|&id| crate::tokenizer::token_piece_bytes(exec.tokenizer(), id) == piece)
        .ok_or_else(|| anyhow!("no token decodes to {:?}", String::from_utf8_lossy(piece)))
}

/// The schema of the end-to-end constraint tests: two required properties.
pub(crate) fn two_field_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {"n": {"type": "integer"}, "s": {"type": "string"}},
        "required": ["n", "s"],
        "additionalProperties": false
    })
}

/// Logit biases that steer a [`full_vocab_executor`] through one short
/// instance of [`two_field_schema`].
///
/// The mask decides which tokens are open. Among them, EOS outranks all,
/// and a closing quote outranks every other byte inside a string.
pub(crate) fn two_field_bias(exec: &Executor<CpuRuntime>) -> Result<HashMap<u32, f32>> {
    let eos = exec
        .tokenizer()
        .eos_token_id()
        .ok_or_else(|| anyhow!("llama3 states no end-of-sequence token"))?;
    let mut bias = HashMap::from([(eos, 30.0)]);
    let ranks = [
        ("\"", 25.0),
        ("}", 22.0),
        (",", 20.0),
        (":", 18.0),
        ("7", 15.0),
        ("n", 12.0),
        ("s", 11.0),
        ("{", 10.0),
    ];
    for (piece, value) in ranks {
        bias.insert(piece_id(exec, piece.as_bytes())?, value);
    }
    Ok(bias)
}

/// Check that `text` is an instance of [`two_field_schema`].
pub(crate) fn check_two_field_instance(text: &str) -> Result<()> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| anyhow!("{text:?} is not JSON: {e}"))?;
    let fields = value
        .as_object()
        .ok_or_else(|| anyhow!("{text:?} is not a JSON object"))?;
    let n_ok = fields.get("n").is_some_and(serde_json::Value::is_i64);
    let s_ok = fields.get("s").is_some_and(serde_json::Value::is_string);
    if fields.len() != 2 || !n_ok || !s_ok {
        return Err(anyhow!("{text:?} breaks the schema"));
    }
    Ok(())
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
    seeded_on(seed, paged, 16, false)
}

/// A [`seeded_executor`] with no paged pool and a context of `ctx` tokens.
pub(crate) fn seeded_executor_ctx(seed: u64, ctx: usize) -> Result<Executor<CpuRuntime>> {
    seeded_on(seed, None, ctx, false)
}

/// A [`seeded_executor`] with a context of `ctx` tokens and the prefix cache
/// on or off.
pub(crate) fn seeded_paged(
    seed: u64,
    pool: PagedPool,
    ctx: usize,
    prefix_cache: bool,
) -> Result<Executor<CpuRuntime>> {
    seeded_on(seed, Some(pool), ctx, prefix_cache)
}

/// A seeded target with a context of `ctx` tokens and a seeded draft attached.
///
/// The draft runs contiguous KV with the same context, whatever `paged`
/// sets for the target. Equal seeds give a draft with the target's weights.
/// The test sets `inference.speculative` itself.
pub(crate) fn seeded_executor_pair(
    target_seed: u64,
    draft_seed: u64,
    ctx: usize,
    paged: Option<PagedPool>,
) -> Result<Executor<CpuRuntime>> {
    let draft = seeded_on(draft_seed, None, ctx, false)?;
    seeded_on(target_seed, paged, ctx, false)?.with_draft(draft)
}

/// The tiny llama with seeded random weights over every llama3 token id,
/// with a context of [`FULL_VOCAB_CTX`] tokens.
///
/// Text encodes into its vocab, so a server test can send a text prompt.
/// Its logits depend on the prompt, and its hidden states are nonzero.
#[cfg(all(feature = "server", not(any(feature = "cuda", feature = "wgpu"))))]
pub(crate) fn seeded_full_vocab_executor(seed: u64) -> Result<Executor<CpuRuntime>> {
    seeded_full_vocab_on(seed, None, 1)
}

/// A [`seeded_full_vocab_executor`] of `layers` layers, so a pipeline test
/// can split it across stages.
#[cfg(all(
    feature = "cli",
    feature = "distributed",
    feature = "server",
    not(any(feature = "cuda", feature = "wgpu"))
))]
pub(crate) fn seeded_full_vocab_layers(seed: u64, layers: usize) -> Result<Executor<CpuRuntime>> {
    seeded_full_vocab_on(seed, None, layers)
}

/// Layers `layers` of the [`seeded_full_vocab_layers`] model of `total`
/// layers, loaded as a swarm stage holds them: a `LlamaStage` built from the
/// tensors of its own span only.
#[cfg(all(
    feature = "cli",
    feature = "distributed",
    feature = "server",
    not(any(feature = "cuda", feature = "wgpu"))
))]
pub(crate) fn seeded_full_vocab_stage(
    seed: u64,
    total: usize,
    layers: std::ops::Range<usize>,
) -> Result<LoadedModel<CpuRuntime>> {
    use boostr::model::llama::LayerSpan;
    let (_, vocab) = full_vocab_tokenizer()?;
    let device = CpuDevice::new();
    let config = full_vocab_config(vocab, total)?;
    let span = LayerSpan::for_model(&config, layers)?;
    let full = seeded_map(seed, vocab, total, &device)?;
    let mut map = VarMap::new();
    for (name, t) in full.all_tensors() {
        if span.reads(&config, name) {
            map.insert(name.to_string(), t.clone());
        }
    }
    let mut vb = VarBuilder::new(&mut map, &device);
    Ok(LoadedModel::<CpuRuntime>::load_layers(
        &config, &mut vb, span,
    )?)
}

/// The weight-free executor a swarm leader serves a
/// [`seeded_full_vocab_layers`] model of `layers` layers with: its config and
/// tokenizer, as `blazr swarm` builds it.
#[cfg(all(
    feature = "cli",
    feature = "distributed",
    feature = "server",
    not(any(feature = "cuda", feature = "wgpu"))
))]
pub(crate) fn full_vocab_front(layers: usize) -> Result<Executor<CpuRuntime>> {
    let (tokenizer, vocab) = full_vocab_tokenizer()?;
    let config = full_vocab_config(vocab, layers)?;
    let mut blazr_config = BlazrConfig::from_universal_with_dtype(config, "f32");
    blazr_config.inference.prefix_cache = false;
    Executor::tokenizer_only(blazr_config, tokenizer, CpuDevice::new(), FULL_VOCAB_CTX)
}

/// The config of a full-vocab tiny llama of `layers` layers.
#[cfg(all(feature = "server", not(any(feature = "cuda", feature = "wgpu"))))]
fn full_vocab_config(vocab: usize, layers: usize) -> Result<boostr::model::UniversalConfig> {
    let yaml = full_vocab_yaml(vocab).replace("num_layers: 1", &format!("num_layers: {layers}"));
    Ok(serde_saphyr::from_str(&yaml)?)
}

/// A [`seeded_full_vocab_executor`] over the paged KV pool `pool`, with the
/// prefix cache off.
///
/// A pool smaller than [`FULL_VOCAB_CTX`] tokens runs dry while a long
/// generation decodes. The generation then ends with an error.
#[cfg(all(feature = "server", not(any(feature = "cuda", feature = "wgpu"))))]
pub(crate) fn seeded_full_vocab_paged(seed: u64, pool: PagedPool) -> Result<Executor<CpuRuntime>> {
    seeded_full_vocab_on(seed, Some(pool), 1)
}

#[cfg(all(feature = "server", not(any(feature = "cuda", feature = "wgpu"))))]
fn seeded_full_vocab_on(
    seed: u64,
    paged: Option<PagedPool>,
    layers: usize,
) -> Result<Executor<CpuRuntime>> {
    let (tokenizer, vocab) = full_vocab_tokenizer()?;
    let device = CpuDevice::new();
    let config = full_vocab_config(vocab, layers)?;
    let model = seeded_model(seed, vocab, layers, &config, &device)?;
    let mut blazr_config = BlazrConfig::from_universal_with_dtype(config, "f32");
    enable_paged(&mut blazr_config, paged);
    blazr_config.inference.prefix_cache = false;
    Executor::new(model, blazr_config, tokenizer, device, FULL_VOCAB_CTX)
}

/// The model of `config`, `layers` layers deep, with weights drawn from `seed`.
fn seeded_model(
    seed: u64,
    vocab: usize,
    layers: usize,
    config: &boostr::model::UniversalConfig,
    device: &CpuDevice,
) -> Result<LoadedModel<CpuRuntime>> {
    let mut map = seeded_map(seed, vocab, layers, device)?;
    let mut vb = VarBuilder::new(&mut map, device);
    Ok(LoadedModel::<CpuRuntime>::load(config, &mut vb)?)
}

/// The checkpoint tensors of a [`seeded_model`].
fn seeded_map(
    seed: u64,
    vocab: usize,
    layers: usize,
    device: &CpuDevice,
) -> Result<VarMap<CpuRuntime>> {
    let mut weights = SeededWeights {
        state: seed,
        vocab,
        layers,
        map: VarMap::new(),
    };
    weights.fill(device)?;
    Ok(weights.map)
}

fn seeded_on(
    seed: u64,
    paged: Option<PagedPool>,
    ctx: usize,
    prefix_cache: bool,
) -> Result<Executor<CpuRuntime>> {
    let device = CpuDevice::new();
    let yaml = TINY_LLAMA_YAML.replace("max_seq_len: 16", &format!("max_seq_len: {ctx}"));
    let config: boostr::model::UniversalConfig = serde_saphyr::from_str(&yaml)?;
    let model = seeded_model(seed, VOCAB, 1, &config, &device)?;

    let mut blazr_config = BlazrConfig::from_universal_with_dtype(config, "f32");
    enable_paged(&mut blazr_config, paged);
    blazr_config.inference.prefix_cache = prefix_cache;
    let tokenizer = crate::tokenizer::from_pretrained("llama3")?;
    Executor::new(model, blazr_config, tokenizer, device, ctx)
}

/// Checkpoint tensors for [`TINY_LLAMA_YAML`], drawn from a linear
/// congruential generator.
struct SeededWeights {
    state: u64,
    /// Rows of the embedding and the LM head.
    vocab: usize,
    /// Decoder layers, drawn in order between the embedding and the final norm.
    layers: usize,
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

    /// The norms, attention and MLP weights of the layer named `p`.
    fn layer(&mut self, device: &CpuDevice, p: &str) -> Result<()> {
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
        Ok(())
    }

    fn fill(&mut self, device: &CpuDevice) -> Result<()> {
        self.put(
            device,
            "model.embed_tokens.weight",
            &[self.vocab, HIDDEN],
            -1.0,
            1.0,
        )?;
        for layer in 0..self.layers {
            self.layer(device, &format!("model.layers.{layer}"))?;
        }
        self.put(device, "model.norm.weight", &[HIDDEN], 0.5, 1.5)?;
        let vocab = self.vocab;
        self.linear(device, "lm_head.weight", vocab, HIDDEN)
    }
}

/// How long a wait in a test hook can block before the test calls it
/// stuck. It guards against a hang, never decides a pass.
pub(crate) const WAIT_LIMIT: Duration = Duration::from_secs(5);

/// Block this thread until `counter` passes `seen`. Returns false when
/// [`WAIT_LIMIT`] runs out first.
pub(crate) fn wait_past(counter: &AtomicUsize, seen: usize) -> bool {
    let deadline = Instant::now() + WAIT_LIMIT;
    while counter.load(Ordering::SeqCst) <= seen {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

/// A task on the runtime that counts up every millisecond.
pub(crate) fn ticker(ticks: &Arc<AtomicUsize>) -> tokio::task::JoinHandle<()> {
    let ticks = Arc::clone(ticks);
    tokio::spawn(async move {
        loop {
            ticks.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
}
