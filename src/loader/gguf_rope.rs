//! RoPE scaling from a GGUF file, in llama.cpp's encoding.
//!
//! llama.cpp stores linear and YaRN scaling as `{arch}.rope.scaling.*` keys.
//! It stores llama3 scaling as the `rope_freqs.weight` tensor instead: one
//! divisor per rotated dimension pair, with no key naming the scheme.

use anyhow::{anyhow, Result};
use boostr::format::Gguf;
use boostr::model::RopeScalingConfig;

/// The tensor llama.cpp writes for per-dimension frequency divisors.
const ROPE_FREQS_TENSOR: &str = "rope_freqs.weight";

/// The RoPE scaling a GGUF file declares, or `None` when it declares none.
///
/// # Errors
///
/// A divisor tensor (`rope_freqs.weight`, or a `longrope` file's
/// `rope_factors_short.weight` and `rope_factors_long.weight`) is missing,
/// unreadable, not F32, or holds a non-finite or non-positive divisor. The
/// scaling type is outside `none`, `linear`, `yarn` and `longrope`.
pub(crate) fn rope_scaling_from_gguf(
    gguf: &mut Gguf,
    arch: &str,
) -> Result<Option<RopeScalingConfig>> {
    if gguf.tensor_info(ROPE_FREQS_TENSOR).is_ok() {
        let divisors = f32_tensor(gguf, ROPE_FREQS_TENSOR)?;
        let mut config = scaling_config("rope_freqs", 1.0, None);
        config.long_factor = Some(divisors);
        return Ok(Some(config));
    }
    let key = |name: &str| format!("{arch}.rope.scaling.{name}");
    let original = gguf
        .metadata()
        .get_u32(&key("original_context_length"))
        .map(|v| v as usize);
    if gguf.metadata().get_string(&key("type")) == Some("longrope") {
        // Phi-3 stores its two divisor lists as tensors.
        let mut config = scaling_config("longrope", 1.0, original);
        config.short_factor = Some(f32_tensor(gguf, "rope_factors_short.weight")?);
        config.long_factor = Some(f32_tensor(gguf, "rope_factors_long.weight")?);
        return Ok(Some(config));
    }
    let metadata = gguf.metadata();
    let Some(scaling_type) = metadata.get_string(&key("type")) else {
        return Ok(None);
    };
    let factor = metadata.get_f32(&key("factor"));
    Ok(match (scaling_type, factor) {
        ("linear", Some(factor)) => Some(scaling_config("linear", factor, original)),
        ("yarn", Some(factor)) => {
            let mut yarn = scaling_config("yarn", factor, original);
            yarn.attention_factor = metadata.get_f32(&key("yarn_attn_factor"));
            yarn.beta_fast = metadata.get_f32(&key("yarn_beta_fast"));
            yarn.beta_slow = metadata.get_f32(&key("yarn_beta_slow"));
            Some(yarn)
        }
        ("none", _) => None,
        (other, factor) => {
            return Err(anyhow!(
                "GGUF {} is '{other}' with factor {factor:?}. blazr reads 'linear' and 'yarn' \
                 with a factor, 'longrope', and 'none'.",
                key("type")
            ));
        }
    })
}

/// A scaling config with every optional field unset.
fn scaling_config(scaling_type: &str, factor: f32, original: Option<usize>) -> RopeScalingConfig {
    RopeScalingConfig {
        scaling_type: scaling_type.to_string(),
        factor,
        original_max_position_embeddings: original,
        low_freq_factor: None,
        high_freq_factor: None,
        attention_factor: None,
        beta_fast: None,
        beta_slow: None,
        short_factor: None,
        long_factor: None,
    }
}

/// A 1-D F32 tensor of positive finite divisors, read whole.
fn f32_tensor(gguf: &mut Gguf, name: &str) -> Result<Vec<f32>> {
    let info = gguf
        .tensor_info(name)
        .map_err(|e| anyhow!("Failed to read GGUF tensor {name}: {e}"))?;
    if info.ggml_type != boostr::format::gguf::GgmlType::F32 {
        return Err(anyhow!(
            "GGUF tensor {name} is {:?}. llama.cpp writes it as F32.",
            info.ggml_type
        ));
    }
    let bytes = gguf
        .read_tensor_bytes(name)
        .map_err(|e| anyhow!("Failed to read GGUF tensor {name}: {e}"))?;
    let divisors: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if let Some(bad) = divisors.iter().find(|d| !d.is_finite() || **d <= 0.0) {
        return Err(anyhow!(
            "GGUF tensor {name} holds the divisor {bad}. Each divisor is a positive finite number."
        ));
    }
    Ok(divisors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    /// A GGUF v3 file with string/f32 keys and at most one F32 vector tensor.
    fn gguf(keys: &[(&str, Result<&str, f32>)], tensor: Option<(&str, &[f32])>) -> Gguf {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensor.is_some() as u64).to_le_bytes());
        out.extend_from_slice(&(keys.len() as u64).to_le_bytes());
        for (key, value) in keys {
            put_str(&mut out, key);
            match value {
                Ok(s) => {
                    out.extend_from_slice(&8u32.to_le_bytes());
                    put_str(&mut out, s);
                }
                Err(f) => {
                    out.extend_from_slice(&6u32.to_le_bytes());
                    out.extend_from_slice(&f.to_le_bytes());
                }
            }
        }
        if let Some((name, values)) = tensor {
            put_str(&mut out, name);
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes());
        }
        out.resize(out.len().div_ceil(32) * 32, 0);
        if let Some((_, values)) = tensor {
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        Gguf::from_bytes(out).expect("hand-built GGUF parses")
    }

    #[test]
    fn rope_freqs_tensor_becomes_per_dimension_divisors() {
        let mut file = gguf(&[], Some(("rope_freqs.weight", &[1.0, 2.5, 32.0])));
        let config = rope_scaling_from_gguf(&mut file, "llama").unwrap().unwrap();
        assert_eq!(config.scaling_type, "rope_freqs");
        assert_eq!(config.long_factor.as_deref(), Some(&[1.0, 2.5, 32.0][..]));
    }

    #[test]
    fn linear_keys_become_linear_scaling() {
        let mut file = gguf(
            &[
                ("llama.rope.scaling.type", Ok("linear")),
                ("llama.rope.scaling.factor", Err(4.0)),
            ],
            None,
        );
        let config = rope_scaling_from_gguf(&mut file, "llama").unwrap().unwrap();
        assert_eq!(config.scaling_type, "linear");
        assert_eq!(config.factor, 4.0);
    }

    #[test]
    fn no_keys_and_none_mean_no_scaling() {
        assert!(rope_scaling_from_gguf(&mut gguf(&[], None), "llama")
            .unwrap()
            .is_none());
        let mut none = gguf(&[("llama.rope.scaling.type", Ok("none"))], None);
        assert!(rope_scaling_from_gguf(&mut none, "llama")
            .unwrap()
            .is_none());
    }

    #[test]
    fn unknown_type_and_bad_divisors_error() {
        let mut odd = gguf(&[("llama.rope.scaling.type", Ok("mystery"))], None);
        let err = rope_scaling_from_gguf(&mut odd, "llama").unwrap_err();
        assert!(err.to_string().contains("mystery"), "{err}");
        let mut zero = gguf(&[], Some(("rope_freqs.weight", &[1.0, 0.0])));
        assert!(rope_scaling_from_gguf(&mut zero, "llama").is_err());
    }
}
