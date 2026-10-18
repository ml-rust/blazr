//! Host reads of model outputs as `f32`, whatever dtype the model computes in.
//!
//! A BF16 or F16 model returns half-width tensors. Each read here casts on
//! the device first, so the host copy is always one `f32` per element.

use boostr::error::{Error, Result};
use boostr::{DType, Runtime, Tensor, TypeConversionOps};

/// `tensor` on the host as `f32` values, in row-major order.
///
/// A tensor of another dtype casts on its device first.
///
/// # Errors
///
/// An error when the cast, the contiguous copy, or the host copy fails.
pub(crate) fn to_host_f32<R>(tensor: &Tensor<R>) -> Result<Vec<f32>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
{
    let wide = if tensor.dtype() == DType::F32 {
        tensor.clone()
    } else {
        R::default_client(tensor.device()).cast(tensor, DType::F32)?
    };
    // A contiguous view with an offset reads in place, with no device copy.
    let wide = if wide.is_contiguous() {
        wide
    } else {
        wide.contiguous()?
    };
    Ok(wide.try_to_vec::<f32>()?)
}

/// The last position's row of `[batch, seq, vocab]` logits, on the host.
///
/// Only that row casts and copies, `vocab` values. The batch must be one row.
///
/// # Errors
///
/// An error when the shape has no position, holds more than one row, or the
/// cast or the copy fails.
pub(crate) fn last_logits_row<R>(logits: &Tensor<R>) -> Result<Vec<f32>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
{
    let shape = logits.shape();
    let &[.., seq_len, vocab_size] = shape else {
        return Err(Error::InferenceError {
            reason: format!("sampling needs [batch, seq, vocab] logits, got shape {shape:?}"),
        });
    };
    if seq_len == 0 {
        return Err(Error::InferenceError {
            reason: format!("sampling needs at least one logits position, got shape {shape:?}"),
        });
    }
    let row = to_host_f32(&logits.narrow(-2, seq_len - 1, 1)?)?;
    if row.len() != vocab_size {
        return Err(Error::InferenceError {
            reason: format!(
                "sampling needs one logits row of {vocab_size} values, got {} from shape {shape:?}",
                row.len()
            ),
        });
    }
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::{CpuDevice, CpuRuntime};

    /// `[1, 2, 3]` logits in `dtype`, from 16-bit patterns.
    #[cfg(feature = "f16")]
    fn half_logits(bits: &[u16], dtype: DType) -> Result<Tensor<CpuRuntime>> {
        let bytes: &[u8] = bytemuck::cast_slice(bits);
        Ok(Tensor::from_bytes(
            bytes,
            &[1, 2, 3],
            dtype,
            &CpuDevice::new(),
        )?)
    }

    /// A BF16 last row reads as `f32` values, not as raw half-width bytes.
    ///
    /// Sabotage check: drop the cast in `to_host_f32` and the read is an
    /// error, since numr refuses a 4-byte read of a 2-byte dtype.
    #[cfg(feature = "f16")]
    #[test]
    fn bf16_last_row_reads_as_f32() -> Result<()> {
        // 1.0, 2.0, 0.0 | 50.0, 0.0, 20.0
        let logits = half_logits(&[0x3f80, 0x4000, 0, 0x4248, 0, 0x41a0], DType::BF16)?;
        assert_eq!(last_logits_row(&logits)?, vec![50.0, 0.0, 20.0]);
        Ok(())
    }

    /// An F16 last row reads as `f32` values.
    #[cfg(feature = "f16")]
    #[test]
    fn f16_last_row_reads_as_f32() -> Result<()> {
        // 1.0, 2.0, 0.0 | -2.0, 0.5, 1.0
        let logits = half_logits(&[0x3c00, 0x4000, 0, 0xc000, 0x3800, 0x3c00], DType::F16)?;
        assert_eq!(last_logits_row(&logits)?, vec![-2.0, 0.5, 1.0]);
        Ok(())
    }

    /// An F32 last row reads in place from its offset.
    #[test]
    fn f32_last_row_reads_the_last_position() -> Result<()> {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let logits = Tensor::<CpuRuntime>::from_slice(&values, &[1, 2, 3], &CpuDevice::new())?;
        assert_eq!(last_logits_row(&logits)?, vec![4.0, 5.0, 6.0]);
        Ok(())
    }

    /// A batch of two rows is an error, not a silent read of both.
    #[test]
    fn two_row_batch_is_an_error() -> Result<()> {
        let logits = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[2, 1, 2], &CpuDevice::new())?;
        assert!(last_logits_row(&logits).is_err());
        Ok(())
    }

    /// A whole half-width tensor reads as `f32` values.
    #[cfg(feature = "f16")]
    #[test]
    fn bf16_tensor_reads_whole_as_f32() -> Result<()> {
        let logits = half_logits(&[0x3f80, 0x4000, 0, 0x4248, 0, 0x41a0], DType::BF16)?;
        assert_eq!(to_host_f32(&logits)?, vec![1.0, 2.0, 0.0, 50.0, 0.0, 20.0]);
        Ok(())
    }
}
