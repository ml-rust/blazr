//! A SafeTensors checkpoint read into a `VarMap` in the layout and dtype
//! the model loads, and the report of what was read.

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
use boostr::model::UniversalConfig;
use boostr::ops::TensorOps;
use boostr::{DType, Runtime, VarMap};

use super::tensors::TensorPolicy;

/// What a checkpoint load read from the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLoadReport {
    /// Every tensor read and uploaded to the device, in load order, under
    /// its file name.
    pub uploaded: Vec<String>,
    /// Every tensor left unread because the model never uses it, such as
    /// the MTP head and vision tower of a Qwen3.5 checkpoint.
    pub skipped: Vec<String>,
    /// The dtype every float weight holds when the model loads.
    pub compute_dtype: DType,
}

/// Every tensor of `loader` that the model of `config` reads, in the
/// layout and at the `compute` dtype it loads with. A sharded checkpoint
/// reads shard by shard. See `TensorPolicy` for the dtype order.
///
/// # Errors
///
/// A tensor does not read or cast, or the layout edits fail.
pub(super) fn load_checkpoint_map<R: Runtime<DType = DType>>(
    loader: &mut SafeTensorsLoader,
    config: &UniversalConfig,
    device: &R::Device,
    compute: DType,
) -> Result<(VarMap<R>, TensorLoadReport)>
where
    R::Client: TensorOps<R>,
{
    let policy = TensorPolicy::new(config, compute);
    let client = R::default_client(device);
    let mut map = VarMap::<R>::new();
    let mut report = TensorLoadReport {
        uploaded: Vec::new(),
        skipped: Vec::new(),
        compute_dtype: compute,
    };
    let names = loader.tensor_names();
    let total = names.len();
    tracing::info!("Loading {total} tensors to device...");
    for (idx, name) in names.into_iter().enumerate() {
        if !policy.reads(&name) {
            report.skipped.push(name);
            continue;
        }
        if idx % 50 == 0 {
            tracing::debug!("Loading tensor {}/{total}: {name}", idx + 1);
        }
        let tensor = loader
            .load_tensor::<R>(&name, device)
            .map_err(|e| anyhow!("Failed to load tensor '{name}': {e}"))?;
        let tensor = policy
            .admit(&client, tensor)
            .map_err(|e| anyhow!("tensor '{name}' did not cast: {e}"))?;
        report.uploaded.push(name.clone());
        map.insert(name, tensor);
    }
    tracing::info!(
        "Loaded {} tensors, left {} the model does not read",
        report.uploaded.len(),
        report.skipped.len()
    );
    policy.finish(&mut map, config, device)?;
    Ok((map, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::format::hf_qwen35::is_dropped;
    use boostr::format::restructure_hf_checkpoint;
    use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor, TypeConversionOps};
    use std::collections::HashMap;

    /// A two-layer Qwen3.5 text config: layer 0 Gated DeltaNet with two
    /// value heads per key head, layer 1 full attention.
    const TINY_CONFIG: &str = r#"{
        "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
        "vocab_size": 16, "hidden_size": 8, "num_hidden_layers": 2,
        "max_position_embeddings": 32, "intermediate_size": 16,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": true,
        "layer_types": ["linear_attention", "full_attention"],
        "full_attention_interval": 2,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 4,
        "linear_value_head_dim": 4, "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "attn_output_gate": true, "attention_bias": false, "hidden_act": "silu",
        "mlp_only_layers": [],
        "rope_parameters": {"mrope_interleaved": true, "mrope_section": [1, 1, 0],
            "partial_rotary_factor": 0.5, "rope_theta": 10000, "rope_type": "default"}
    }"#;

    const DROPPED: [&str; 2] = ["mtp.fc.weight", "model.visual.patch_embed.proj.weight"];

    fn tiny_config() -> UniversalConfig {
        super::super::hf_qwen35::qwen35_config_of(TINY_CONFIG)
            .unwrap()
            .unwrap()
    }

    /// Every tensor of the tiny checkpoint: Hugging Face name, shape, and
    /// whether the file stores it in F32 (`A_log`, as real checkpoints do).
    fn entries() -> Vec<(String, Vec<usize>, bool)> {
        let lm = "model.language_model";
        let mut out = vec![
            (format!("{lm}.embed_tokens.weight"), vec![16, 8], false),
            (format!("{lm}.norm.weight"), vec![8], false),
        ];
        for layer in 0..2 {
            for norm in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                out.push((format!("{lm}.layers.{layer}.{norm}"), vec![8], false));
            }
        }
        let gdn = format!("{lm}.layers.0.linear_attn");
        for (name, shape, f32) in [
            ("in_proj_qkv.weight", vec![32, 8], false),
            ("conv1d.weight", vec![32, 1, 4], false),
            ("in_proj_z.weight", vec![16, 8], false),
            ("in_proj_a.weight", vec![4, 8], false),
            ("in_proj_b.weight", vec![4, 8], false),
            ("A_log", vec![4], true),
            ("dt_bias", vec![4], false),
            ("norm.weight", vec![4], false),
            ("out_proj.weight", vec![8, 16], false),
        ] {
            out.push((format!("{gdn}.{name}"), shape, f32));
        }
        let attn = format!("{lm}.layers.1.self_attn");
        for (name, shape) in [
            ("q_proj.weight", vec![32, 8]),
            ("q_norm.weight", vec![8]),
            ("k_norm.weight", vec![8]),
        ] {
            out.push((format!("{attn}.{name}"), shape, false));
        }
        out.push((DROPPED[0].into(), vec![8, 16], false));
        out.push((DROPPED[1].into(), vec![4, 4], false));
        out
    }

    /// Deterministic values in `[-0.75, 0.75)`, seeded from `name`.
    fn values(name: &str, n: usize) -> Vec<f32> {
        let mut s = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 1.5
            })
            .collect()
    }

    /// `v` as a BF16 tensor.
    #[cfg(feature = "f16")]
    fn narrow(v: Vec<f32>, shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        let v: Vec<half::bf16> = v.into_iter().map(half::bf16::from_f32).collect();
        Tensor::<CpuRuntime>::from_slice(&v, shape, device).unwrap()
    }

    /// `v` as an F32 tensor: a build without `f16` has no BF16.
    #[cfg(not(feature = "f16"))]
    fn narrow(v: Vec<f32>, shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(&v, shape, device).unwrap()
    }

    /// The tiny checkpoint, BF16 except `A_log`, as `model.safetensors` in
    /// `dir`.
    fn write_checkpoint(dir: &std::path::Path, device: &CpuDevice) {
        let mut tensors = HashMap::new();
        for (name, shape, f32) in entries() {
            let v = values(&name, shape.iter().product());
            let t = if f32 {
                Tensor::<CpuRuntime>::from_slice(&v, &shape, device).unwrap()
            } else {
                narrow(v, &shape, device)
            };
            tensors.insert(name, t);
        }
        let path = dir.join("model.safetensors");
        boostr::format::safetensors::save_safetensors(path, &tensors, None).unwrap();
    }

    /// `t` widened to F32, as bits.
    fn bits(client: &CpuClient, t: &Tensor<CpuRuntime>) -> Vec<u32> {
        let wide = if t.dtype() == DType::F32 {
            t.clone()
        } else {
            client.cast(t, DType::F32).unwrap()
        };
        wide.to_vec::<f32>().iter().map(|v| v.to_bits()).collect()
    }

    /// The tiny checkpoint through [`load_checkpoint_map`] at `compute`
    /// equals boostr's layout edits on the file values widened to F32, cast
    /// to `compute` once.
    fn check_edit_order(compute: DType) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path(), &device);
        let config = tiny_config();

        let mut loader = SafeTensorsLoader::open(dir.path()).unwrap();
        let (map, report) =
            load_checkpoint_map::<CpuRuntime>(&mut loader, &config, &device, compute).unwrap();
        assert_eq!(report.compute_dtype, compute);

        let mut wide = VarMap::<CpuRuntime>::new();
        let mut file = SafeTensorsLoader::open(dir.path()).unwrap();
        for name in file.tensor_names() {
            if is_dropped(&name) {
                continue;
            }
            let t = file.load_tensor::<CpuRuntime>(&name, &device).unwrap();
            let t = if t.dtype() == DType::F32 {
                t
            } else {
                client.cast(&t, DType::F32).unwrap()
            };
            wide.insert(name, t);
        }
        restructure_hf_checkpoint(&mut wide, &config, &device).unwrap();

        let mut want: Vec<&str> = wide.names().collect();
        let mut got: Vec<&str> = map.names().collect();
        want.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, want);
        for name in want {
            let t = map.get_tensor(name).unwrap();
            assert_eq!(t.dtype(), compute, "{name}");
            let f32_edit = wide.get_tensor(name).unwrap();
            let once = if compute == DType::F32 {
                f32_edit.clone()
            } else {
                client.cast(f32_edit, compute).unwrap()
            };
            assert_eq!(bits(&client, t), bits(&client, &once), "{name}");
        }
    }

    #[test]
    fn f32_model_takes_the_edits_unrounded() {
        check_edit_order(DType::F32);
    }

    /// BF16 norms round once after `1 + w`, and the F32 `A_log` rounds
    /// once after `-exp`.
    #[cfg(feature = "f16")]
    #[test]
    fn bf16_model_rounds_each_edit_once() {
        check_edit_order(DType::BF16);
    }

    /// `TensorPolicy` casts every tensor of rank 2 or more as it loads, which
    /// is exact only while the layout edits never do arithmetic on one. Every
    /// value of a loaded matrix must be a value the file holds for a matrix.
    /// An arithmetic edit on a matrix writes new values and fails this test.
    #[test]
    fn layout_edits_do_no_arithmetic_on_matrices() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path(), &device);

        let mut file = SafeTensorsLoader::open(dir.path()).unwrap();
        let mut file_values = std::collections::HashSet::new();
        for name in file.tensor_names() {
            let t = file.load_tensor::<CpuRuntime>(&name, &device).unwrap();
            if t.shape().len() >= 2 {
                file_values.extend(bits(&client, &t));
            }
        }

        let mut loader = SafeTensorsLoader::open(dir.path()).unwrap();
        let (map, _) =
            load_checkpoint_map::<CpuRuntime>(&mut loader, &tiny_config(), &device, DType::F32)
                .unwrap();
        for (name, t) in map.all_tensors() {
            if t.shape().len() >= 2 {
                assert!(
                    bits(&client, t).iter().all(|b| file_values.contains(b)),
                    "{name} holds a value the file does not"
                );
            }
        }
    }

    #[test]
    fn dropped_tensors_are_never_read() {
        let device = CpuDevice::new();
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path(), &device);
        let mut loader = SafeTensorsLoader::open(dir.path()).unwrap();
        let (map, report) =
            load_checkpoint_map::<CpuRuntime>(&mut loader, &tiny_config(), &device, DType::F32)
                .unwrap();
        let mut skipped = report.skipped.clone();
        skipped.sort_unstable();
        let mut expected = DROPPED.map(String::from).to_vec();
        expected.sort_unstable();
        assert_eq!(skipped, expected);
        assert!(report.uploaded.iter().all(|n| !is_dropped(n)));
        assert_eq!(
            report.uploaded.len() + report.skipped.len(),
            entries().len()
        );
        assert!(map.names().all(|n| !is_dropped(n)));
        assert!(map.is_hf_restructured());
    }
}
