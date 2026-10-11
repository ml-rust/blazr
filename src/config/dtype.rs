//! The compute dtype: the one dtype for float weights, activations and the
//! KV cache.
//!
//! Precedence, first set wins:
//!
//! 1. `--dtype` on `run`, `generate`, `chat`, `bench` and `serve`
//! 2. `inference.dtype` in `~/.blazr/config.yaml`
//! 3. `torch_dtype` in the model's `config.json`
//! 4. The backend default, `f32`
//!
//! The loader resolves the dtype once, before any weight loads. It casts
//! every float weight to it and records it in `inference.dtype`. The
//! executor sizes the KV cache from that record, so weights, activations
//! and cache agree.
//!
//! A runtime that computes in f32 only (wgpu) runs f32 and logs a warning
//! when a flag or key asks for another dtype.

use anyhow::{anyhow, Result};

use boostr::{DType, Runtime};

use super::parse_dtype;

/// The dtype every runtime computes, used when no source sets one.
pub const BACKEND_DEFAULT_DTYPE: DType = DType::F32;

/// Where a requested compute dtype comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypeSource {
    /// The `--dtype` flag.
    Flag,
    /// `inference.dtype` in `~/.blazr/config.yaml`.
    File,
    /// The target model of speculative decoding. Its draft model shares
    /// the target's compute dtype, as both run in one decode loop.
    Target,
}

impl DtypeSource {
    /// The flag or key, as a user writes it.
    pub fn key(self) -> &'static str {
        match self {
            Self::Flag => "--dtype",
            Self::File => "inference.dtype in ~/.blazr/config.yaml",
            Self::Target => "the speculative target model",
        }
    }
}

/// A compute dtype the user asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtypeRequest {
    pub dtype: DType,
    pub source: DtypeSource,
}

/// The short name blazr writes for a compute dtype: `f32`, `bf16` or `f16`.
///
/// # Errors
///
/// An error for a dtype that is not a float compute dtype.
pub fn dtype_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::BF16 => Ok("bf16"),
        DType::F16 => Ok("f16"),
        other => Err(anyhow!(
            "{other:?} is not a compute dtype. Use f32, bf16 or f16"
        )),
    }
}

/// Parse a dtype that `key` sets.
///
/// # Errors
///
/// An error naming `key` and the accepted values, or the missing `f16`
/// build feature.
pub fn parse_dtype_for(value: &str, key: &str) -> Result<DType> {
    parse_dtype(value).map_err(|e| anyhow!("{key} is '{value}': {e}. Use f32, bf16 or f16"))
}

/// The dtype a model `config.json` `torch_dtype` names. `None` for an
/// absent or unknown value.
pub fn torch_dtype(value: &str) -> Option<DType> {
    match value {
        "float32" => Some(DType::F32),
        "bfloat16" => Some(DType::BF16),
        "float16" => Some(DType::F16),
        _ => None,
    }
}

/// The compute dtype for runtime `R`: `request`, then `model`, then
/// [`BACKEND_DEFAULT_DTYPE`].
///
/// A runtime that cannot compute the chosen dtype runs f32. A request it
/// overrides logs a warning that names the flag or key.
pub fn compute_dtype<R: Runtime<DType = DType>>(
    request: Option<&DtypeRequest>,
    model: Option<DType>,
) -> DType {
    // A build without the `f16` feature has no half dtype. A request
    // never holds one, as its parse fails first. A model dtype falls back.
    let model = model.filter(|&d| d == DType::F32 || cfg!(feature = "f16"));
    let chosen = request
        .map(|r| r.dtype)
        .or(model)
        .unwrap_or(BACKEND_DEFAULT_DTYPE);
    if R::computes_float_dtype(chosen) {
        return chosen;
    }
    match request {
        Some(r) => tracing::warn!(
            "{} asks for {}. The {} runtime computes in f32 only, so blazr runs f32",
            r.source.key(),
            dtype_name(r.dtype).unwrap_or("that dtype"),
            R::name()
        ),
        None => tracing::info!(
            "the {} runtime computes in f32 only. The model runs in f32",
            R::name()
        ),
    }
    DType::F32
}

/// The compute dtype of a format that fixes its own, such as GGUF (f32)
/// or AWQ and GPTQ (f16). A request for another dtype logs a warning.
pub fn fixed_format_dtype(format: &str, fixed: DType, request: Option<&DtypeRequest>) -> DType {
    if let Some(r) = request.filter(|r| r.dtype != fixed) {
        tracing::warn!(
            "{} asks for {}. {format} weights compute in {}, so blazr runs {}",
            r.source.key(),
            dtype_name(r.dtype).unwrap_or("that dtype"),
            dtype_name(fixed).unwrap_or("their own dtype"),
            dtype_name(fixed).unwrap_or("their own dtype"),
        );
    }
    fixed
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::CpuRuntime;

    fn file(dtype: DType) -> DtypeRequest {
        DtypeRequest {
            dtype,
            source: DtypeSource::File,
        }
    }

    #[test]
    fn request_then_model_then_backend_default() {
        let f32_request = file(DType::F32);
        assert_eq!(
            compute_dtype::<CpuRuntime>(Some(&f32_request), Some(DType::BF16)),
            DType::F32
        );
        assert_eq!(
            compute_dtype::<CpuRuntime>(None, Some(DType::BF16)),
            DType::BF16
        );
        assert_eq!(compute_dtype::<CpuRuntime>(None, None), DType::F32);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn cpu_computes_every_half_dtype() {
        for dtype in [DType::BF16, DType::F16] {
            assert_eq!(compute_dtype::<CpuRuntime>(Some(&file(dtype)), None), dtype);
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_runs_f32_whatever_the_request() {
        use boostr::runtime::wgpu::WgpuRuntime;
        assert_eq!(
            compute_dtype::<WgpuRuntime>(Some(&file(DType::BF16)), Some(DType::F16)),
            DType::F32
        );
        assert_eq!(
            compute_dtype::<WgpuRuntime>(None, Some(DType::BF16)),
            DType::F32
        );
    }

    #[test]
    fn fixed_formats_keep_their_dtype() {
        assert_eq!(
            fixed_format_dtype("GGUF", DType::F32, Some(&file(DType::BF16))),
            DType::F32
        );
        assert_eq!(fixed_format_dtype("AWQ", DType::F16, None), DType::F16);
    }

    #[test]
    fn torch_dtype_names() {
        assert_eq!(torch_dtype("float32"), Some(DType::F32));
        assert_eq!(torch_dtype("bfloat16"), Some(DType::BF16));
        assert_eq!(torch_dtype("float16"), Some(DType::F16));
        assert_eq!(torch_dtype("int8"), None);
    }

    #[test]
    fn parse_names_the_key() {
        assert_eq!(parse_dtype_for("f32", "--dtype").unwrap(), DType::F32);
        let err = parse_dtype_for("fp8", "--dtype").unwrap_err().to_string();
        assert!(err.contains("--dtype"), "{err}");
        assert!(err.contains("fp8"), "{err}");
    }

    #[test]
    fn names_round_trip() {
        for name in ["f32", "bf16", "f16"] {
            if let Ok(dtype) = parse_dtype(name) {
                assert_eq!(dtype_name(dtype).unwrap(), name);
            }
        }
        assert!(dtype_name(DType::I32).is_err());
    }
}
