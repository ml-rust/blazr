//! Regular (non-quantized) SafeTensors loading

use std::ops::Range;
use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
use boostr::model::{LoadedModel, UniversalConfig};
use boostr::ops::TensorOps;
use boostr::VarBuilder;
use boostr::{DType, Runtime};

use crate::config::{BlazrConfig, DtypeRequest};

use super::super::layer_scope::{build_model, span_of};
use super::checkpoint_map::{load_checkpoint_map, TensorLoadReport};
use super::config::{checkpoint_compute_dtype, config_from_detected, load_or_create_config};
use super::detect_arch::{detect_architecture_from_loader, detect_awq, detect_gptq};

/// Load a regular (non-AWQ) SafeTensors model.
///
/// `request` is the user's compute dtype, see `config::compute_dtype`.
/// Every float weight takes the resolved dtype.
pub fn load_safetensors_regular<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    request: Option<&DtypeRequest>,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let (model, config, _) = load_safetensors_with_report(path, device, request)?;
    Ok((model, config))
}

/// [`load_safetensors_regular`], plus what the load read from the files.
///
/// The report is `None` for a GPTQ or AWQ checkpoint, which loads through
/// its own reader.
pub fn load_safetensors_with_report<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    request: Option<&DtypeRequest>,
) -> Result<(LoadedModel<R>, BlazrConfig, Option<TensorLoadReport>)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    load_scoped(path.as_ref(), device, request, None)
}

/// Layers `layers` of the SafeTensors checkpoint at `path`, as a pipeline
/// stage holds them, plus what the load read from the files.
///
/// Only the tensors of those layers are read, with the embedding on the
/// stage that starts at layer 0 and the final norm and LM head on the
/// stage that ends at the last layer. The model is a `LlamaStage`. GPTQ
/// and AWQ checkpoints filter the same way and report `None`.
///
/// # Errors
///
/// A model a layer range cannot hold, such as one with SSM or Gated
/// DeltaNet state, or a range outside the model. Both fail before any
/// weight is read. Otherwise as [`load_safetensors_with_report`].
pub fn load_safetensors_layers<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    request: Option<&DtypeRequest>,
    layers: Range<usize>,
) -> Result<(LoadedModel<R>, BlazrConfig, Option<TensorLoadReport>)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    load_scoped(path.as_ref(), device, request, Some(layers))
}

/// The whole model at `path`, or layers `layers` of it.
fn load_scoped<R: Runtime<DType = DType>>(
    path: &Path,
    device: &R::Device,
    request: Option<&DtypeRequest>,
    layers: Option<Range<usize>>,
) -> Result<(LoadedModel<R>, BlazrConfig, Option<TensorLoadReport>)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let config_dir = config_dir_of(path);

    // Use boostr's unified SafeTensorsLoader
    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    if detect_gptq(&loader, config_dir) {
        let (model, config) = super::gptq::load_gptq_scoped(path, device, request, layers)?;
        return Ok((model, config, None));
    }
    if detect_awq(&loader, config_dir) {
        let (model, config) = super::awq::load_awq_scoped(path, device, request, layers)?;
        return Ok((model, config, None));
    }

    if loader.is_sharded() {
        tracing::info!(
            "Loading sharded model with {} shards, total size: {:.2} GB",
            loader.num_shards(),
            loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        tracing::info!(
            "Loading single-file model, size: {:.2} GB",
            loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    // Resolve the compute dtype before any weight loads. Tensor names are
    // read only when no config.json describes the model.
    let dtype = checkpoint_compute_dtype::<R>(config_dir, request);
    let config = resolve_config(&loader, config_dir, dtype)?;
    let span = span_of(&config.model, layers)?;

    let (mut var_map, report) =
        load_checkpoint_map::<R>(&mut loader, &config.model, device, dtype, span.as_ref())?;

    // The model takes every tensor it reads out of the map.
    let mut vb = VarBuilder::new(&mut var_map, device);
    let model = build_model(&config.model, &mut vb, span)?;

    Ok((model, config, Some(report)))
}

/// Load a tensor-parallel model from SafeTensors.
/// Loads all tensors then uses `LoadedModel::load_tp` to shard weights per-rank.
pub fn load_safetensors_tp<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    comm: std::sync::Arc<dyn boostr::runtime::Communicator>,
    request: Option<&DtypeRequest>,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let path = path.as_ref();
    let config_dir = config_dir_of(path);

    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    let dtype = checkpoint_compute_dtype::<R>(config_dir, request);
    let config = resolve_config(&loader, config_dir, dtype)?;

    // Check the head split before any weight loads.
    if let Some(attention) = &config.model.attention {
        let heads = crate::engine::tensor_parallel::TpHeads {
            num_heads: attention.num_heads,
            num_kv_heads: attention.num_kv_heads.unwrap_or(attention.num_heads),
        };
        crate::engine::tensor_parallel::validate_tp_config(comm.world_size(), Some(heads))
            .map_err(|e| anyhow!("{e}"))?;
    }

    tracing::info!(
        "Loading tensor-parallel model (rank={}, world_size={})",
        comm.rank(),
        comm.world_size()
    );

    let (mut var_map, _) =
        load_checkpoint_map::<R>(&mut loader, &config.model, device, dtype, None)?;
    let mut vb = VarBuilder::new(&mut var_map, device);

    let model = LoadedModel::load_tp(&config.model, &mut vb, comm)
        .map_err(|e| anyhow!("Failed to load TP model: {}", e))?;

    Ok((model, config))
}

/// Load a model from SafeTensors with explicit configuration
pub fn load_safetensors_with_config<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    config: &UniversalConfig,
    device: &R::Device,
) -> Result<LoadedModel<R>>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let path = path.as_ref();

    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    if loader.is_sharded() {
        tracing::info!(
            "Loading sharded model with {} shards, total size: {:.2} GB",
            loader.num_shards(),
            loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    let dtype = checkpoint_compute_dtype::<R>(config_dir_of(path), None);
    let (mut var_map, _) = load_checkpoint_map::<R>(&mut loader, config, device, dtype, None)?;
    let mut vb = VarBuilder::new(&mut var_map, device);

    let model =
        LoadedModel::load(config, &mut vb).map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok(model)
}

/// The directory that holds `path`'s `config.json`: `path` itself, or the
/// parent of a file.
pub(super) fn config_dir_of(path: &Path) -> Option<&Path> {
    if path.is_file() {
        path.parent()
    } else {
        Some(path)
    }
}

/// The config of the checkpoint `loader` reads, from `config.json` in
/// `config_dir` when present, else from its tensor names.
pub(super) fn resolve_config(
    loader: &SafeTensorsLoader,
    config_dir: Option<&Path>,
    dtype: DType,
) -> Result<BlazrConfig> {
    match config_dir {
        Some(dir) => load_or_create_config(dir, || detect_architecture_from_loader(loader), dtype),
        None => config_from_detected(&detect_architecture_from_loader(loader)?, dtype),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::inference::LayeredKvCache;
    use boostr::{CpuDevice, CpuRuntime, Tensor};
    use std::collections::HashMap;

    const LAYERS: usize = 3;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const VOCAB: usize = 16;

    const CONFIG: &str = r#"{"architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "vocab_size": 16, "hidden_size": 16, "num_hidden_layers": 3,
        "max_position_embeddings": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 32, "rms_norm_eps": 1e-5, "tie_word_embeddings": false,
        "torch_dtype": "float32"}"#;

    /// A seeded three-layer Llama as `config.json` and `model.safetensors`
    /// in `dir`. Returns every tensor name.
    fn write_checkpoint(dir: &Path, device: &CpuDevice) -> Vec<String> {
        let mut state = 0x5eed_u64;
        let mut tensor = |shape: &[usize], lo: f32, hi: f32| {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    lo + (hi - lo) * ((state >> 40) as f32 / (1u64 << 24) as f32)
                })
                .collect();
            Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
        };
        let (kv, d) = (HIDDEN / 2, 0.25);
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".into(),
            tensor(&[VOCAB, HIDDEN], -1.0, 1.0),
        );
        for l in 0..LAYERS {
            let p = format!("model.layers.{l}");
            for (name, shape) in [
                ("self_attn.q_proj.weight", [HIDDEN, HIDDEN]),
                ("self_attn.k_proj.weight", [kv, HIDDEN]),
                ("self_attn.v_proj.weight", [kv, HIDDEN]),
                ("self_attn.o_proj.weight", [HIDDEN, HIDDEN]),
                ("mlp.gate_proj.weight", [INTER, HIDDEN]),
                ("mlp.up_proj.weight", [INTER, HIDDEN]),
                ("mlp.down_proj.weight", [HIDDEN, INTER]),
            ] {
                tensors.insert(format!("{p}.{name}"), tensor(&shape, -d, d));
            }
            for norm in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                tensors.insert(format!("{p}.{norm}"), tensor(&[HIDDEN], 0.5, 1.5));
            }
        }
        tensors.insert("model.norm.weight".into(), tensor(&[HIDDEN], 0.5, 1.5));
        tensors.insert("lm_head.weight".into(), tensor(&[VOCAB, HIDDEN], -d, d));
        std::fs::write(dir.join("config.json"), CONFIG).unwrap();
        boostr::format::safetensors::save_safetensors(
            dir.join("model.safetensors"),
            &tensors,
            None,
        )
        .unwrap();
        tensors.into_keys().collect()
    }

    fn cache(
        model: &LoadedModel<CpuRuntime>,
        layers: usize,
        device: &CpuDevice,
    ) -> LayeredKvCache<CpuRuntime> {
        let (heads, dim) = (model.num_kv_heads().unwrap(), model.head_dim().unwrap());
        LayeredKvCache::new_positional(layers, 1, heads, 8, 32, dim, DType::F32, device).unwrap()
    }

    /// Three stages range-loaded from one checkpoint each read only their
    /// own layers' tensors, and together give the whole model's logits.
    #[test]
    fn range_loaded_stages_read_their_layers_and_match_the_whole_model() {
        let device = CpuDevice::new();
        let dir = tempfile::tempdir().unwrap();
        let names = write_checkpoint(dir.path(), &device);
        let (full, _, full_report) =
            load_safetensors_with_report::<CpuRuntime, _>(dir.path(), &device, None).unwrap();
        assert_eq!(full_report.unwrap().uploaded.len(), names.len());

        let ids = Tensor::<CpuRuntime>::from_slice(&[3i64, 1, 4, 1, 5], &[1, 5], &device).unwrap();
        let logits = full
            .forward_with_kv_cache(&ids, &mut cache(&full, LAYERS, &device), 0)
            .unwrap();
        let all: Vec<f32> = logits.contiguous().unwrap().to_vec();
        let want = &all[all.len() - VOCAB..];

        let mut stages = Vec::new();
        for l in 0..LAYERS {
            let (stage, config, report) =
                load_safetensors_layers::<CpuRuntime, _>(dir.path(), &device, None, l..l + 1)
                    .unwrap();
            assert_eq!(config.num_layers(), LAYERS);
            let report = report.unwrap();
            assert_eq!(report.uploaded.len() + report.skipped.len(), names.len());
            for name in &report.uploaded {
                let own = name.starts_with(&format!("model.layers.{l}."))
                    || (l == 0 && name == "model.embed_tokens.weight")
                    || (l == LAYERS - 1
                        && (name == "model.norm.weight" || name == "lm_head.weight"));
                assert!(own, "stage {l} read {name}");
            }
            assert_eq!(stage.layer_span().map(|s| s.layers()), Some(l..l + 1));
            stages.push(stage);
        }

        let mut hidden = stages[0].forward_embed(&ids).unwrap();
        let mut mlp = None;
        for (l, stage) in stages.iter().enumerate() {
            (hidden, mlp) = stage
                .forward_layers_range(hidden, mlp, &mut cache(stage, 1, &device), l, l + 1, 0)
                .unwrap();
        }
        let logits = stages[LAYERS - 1].forward_head(hidden, mlp).unwrap();
        let got: Vec<f32> = logits.contiguous().unwrap().to_vec();
        let got = &got[got.len() - VOCAB..];
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "logit {i}: {g} vs {w}"
            );
        }
        let err = stages[1].forward_with_kv_cache(&ids, &mut cache(&full, LAYERS, &device), 0);
        assert!(err
            .unwrap_err()
            .to_string()
            .contains("runs the whole model"));
    }

    /// A range outside the model fails before any tensor is read.
    #[test]
    fn a_range_past_the_model_is_refused() {
        let device = CpuDevice::new();
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path(), &device);
        let err = load_safetensors_layers::<CpuRuntime, _>(dir.path(), &device, None, 2..5)
            .err()
            .unwrap();
        assert!(err.to_string().contains("not a non-empty range"), "{err}");
    }
}
