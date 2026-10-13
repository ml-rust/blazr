//! Which tensors of a SafeTensors checkpoint a load reads, the dtype each
//! takes, and the order of the layout edits and the compute-dtype cast.

use anyhow::{anyhow, Result};

use boostr::format::hf_qwen35::is_dropped;
use boostr::format::restructure_hf_checkpoint;
use boostr::model::UniversalConfig;
use boostr::ops::TensorOps;
use boostr::{DType, Runtime, Tensor, VarMap};

/// Which tensors of a checkpoint a load reads, and the dtype each takes
/// before and after `restructure_hf_checkpoint`.
///
/// Most architectures cast every float tensor to the compute dtype as it
/// loads. A Qwen3.5 checkpoint differs in two ways:
///
/// - Its MTP and vision tensors are never read.
/// - Its layout edits `1 + w` and `-exp(A_log)` compute in F32 and round
///   once to the dtype the tensor holds. Those edits touch 1-D tensors
///   only. Each 1-D tensor therefore loads at the compute dtype when that
///   holds the file value exactly (the same dtype, or F32 and wider), else
///   at F32. It takes the compute dtype after the edits. A BF16 tensor in
///   a BF16 model rounds once, and an F32 model takes the edit unrounded.
///   Matrices only move rows and columns, so they cast as they load. The
///   test `layout_edits_do_no_arithmetic_on_matrices` in `checkpoint_map`
///   fails if boostr ever does arithmetic on a matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TensorPolicy {
    qwen35: bool,
    compute: DType,
}

impl TensorPolicy {
    pub(super) fn new(config: &UniversalConfig, compute: DType) -> Self {
        Self {
            qwen35: config.model_type == "qwen35",
            compute,
        }
    }

    /// `false` for a tensor the model never reads. The load leaves it on
    /// disk.
    pub(super) fn reads(&self, name: &str) -> bool {
        !(self.qwen35 && is_dropped(name))
    }

    /// The dtype a float tensor of `rank` dimensions, loaded at `loaded`,
    /// holds until the layout edits run.
    fn load_dtype(&self, loaded: DType, rank: usize) -> DType {
        let exact =
            loaded == self.compute || self.compute.size_in_bytes() >= DType::F32.size_in_bytes();
        if !self.qwen35 || rank >= 2 || exact {
            self.compute
        } else {
            DType::F32
        }
    }

    /// `tensor`, fresh from the file, in the dtype it holds until the
    /// layout edits run.
    pub(super) fn admit<R: Runtime<DType = DType>>(
        &self,
        client: &R::Client,
        tensor: Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        R::Client: TensorOps<R>,
    {
        let dtype = self.load_dtype(tensor.dtype(), tensor.shape().len());
        cast_float(client, tensor, dtype)
    }

    /// Put `map` into the layout the model reads. For Qwen3.5, then cast
    /// every float tensor still off the compute dtype to it.
    pub(super) fn finish<R: Runtime<DType = DType>>(
        &self,
        map: &mut VarMap<R>,
        config: &UniversalConfig,
        device: &R::Device,
    ) -> Result<()>
    where
        R::Client: TensorOps<R>,
    {
        restructure_hf_checkpoint(map, config, device).map_err(|e| anyhow!("{e}"))?;
        if !self.qwen35 {
            return Ok(());
        }
        let client = R::default_client(device);
        let stale: Vec<String> = map
            .all_tensors()
            .filter(|(_, t)| is_float(t.dtype()) && t.dtype() != self.compute)
            .map(|(name, _)| name.to_string())
            .collect();
        for name in stale {
            let tensor = map.take_tensor(&name).map_err(|e| anyhow!("{e}"))?;
            let tensor = cast_float(&client, tensor, self.compute)
                .map_err(|e| anyhow!("tensor '{name}' did not cast to {:?}: {e}", self.compute))?;
            map.insert(name, tensor);
        }
        Ok(())
    }
}

fn is_float(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

/// `tensor` at `dtype` when it holds F32, F16 or BF16. Any other tensor
/// passes unchanged.
pub(super) fn cast_float<R: Runtime<DType = DType>>(
    client: &R::Client,
    tensor: Tensor<R>,
    dtype: DType,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R>,
{
    if !is_float(tensor.dtype()) || tensor.dtype() == dtype {
        return Ok(tensor);
    }
    boostr::TypeConversionOps::cast(client, &tensor, dtype).map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::{CpuDevice, CpuRuntime};

    const MTP: &str = "mtp.fc.weight";
    const VISION: &str = "model.visual.patch_embed.proj.weight";

    #[test]
    fn other_architectures_read_every_tensor_at_the_compute_dtype() {
        let policy = TensorPolicy {
            qwen35: false,
            compute: DType::BF16,
        };
        assert!(policy.reads(MTP) && policy.reads(VISION));
        assert_eq!(policy.load_dtype(DType::F32, 1), DType::BF16);
        assert_eq!(policy.load_dtype(DType::F16, 2), DType::BF16);
    }

    #[test]
    fn qwen35_skips_mtp_and_vision() {
        let policy = TensorPolicy {
            qwen35: true,
            compute: DType::F32,
        };
        for name in [
            MTP,
            VISION,
            "model.mtp.fc.weight",
            "mtp.layers.0.mlp.up_proj.weight",
        ] {
            assert!(!policy.reads(name), "{name}");
        }
        for name in [
            "model.language_model.norm.weight",
            "model.language_model.layers.0.linear_attn.A_log",
            "model.layers.0.input_layernorm.weight",
            "lm_head.weight",
        ] {
            assert!(policy.reads(name), "{name}");
        }
    }

    #[test]
    fn qwen35_vectors_wait_in_f32_unless_the_compute_dtype_holds_them() {
        let bf16 = TensorPolicy {
            qwen35: true,
            compute: DType::BF16,
        };
        assert_eq!(bf16.load_dtype(DType::BF16, 1), DType::BF16);
        assert_eq!(bf16.load_dtype(DType::F32, 1), DType::F32);
        assert_eq!(bf16.load_dtype(DType::F16, 1), DType::F32);
        assert_eq!(bf16.load_dtype(DType::F32, 2), DType::BF16);
        assert_eq!(bf16.load_dtype(DType::F32, 3), DType::BF16);
        let f32 = TensorPolicy {
            qwen35: true,
            compute: DType::F32,
        };
        assert_eq!(f32.load_dtype(DType::BF16, 1), DType::F32);
        assert_eq!(f32.load_dtype(DType::BF16, 2), DType::F32);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn float_weights_take_the_compute_dtype() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let values = [1.5f32, -2.25, 0.125, 3.0];
        let bf16: Vec<half::bf16> = values.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let weight = Tensor::<CpuRuntime>::from_slice(&bf16, &[2, 2], &device).unwrap();
        let up = cast_float(&client, weight, DType::F32).unwrap();
        assert_eq!(up.dtype(), DType::F32);
        assert_eq!(up.to_vec::<f32>(), values);

        let f32_weight = Tensor::<CpuRuntime>::from_slice(&values, &[4], &device).unwrap();
        let down = cast_float(&client, f32_weight, DType::BF16).unwrap();
        assert_eq!(down.dtype(), DType::BF16);
    }

    #[test]
    fn integer_tensors_keep_their_dtype() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let ids = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3], &[3], &device).unwrap();
        assert_eq!(
            cast_float(&client, ids, DType::F32).unwrap().dtype(),
            DType::I32
        );
    }
}
