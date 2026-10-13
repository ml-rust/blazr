//! `blazr_embed`: one vector for a text.

use std::ffi::c_char;

use super::error::{guard, BlazrStatus, FfiError};
use super::handle::{self, with_executor, BlazrModel, EngineClient};
use super::string::{into_c_array, read_str, write_out};
use crate::engine::pooling::{l2_normalize, pool_mean};
use crate::engine::Executor;
use boostr::{DType, Runtime};

/// Embed `text`: the mean of the model's last hidden states over its
/// tokens, scaled to unit length.
///
/// On success `*out_values` and `*out_len` hold one vector of the model's
/// hidden size. Free it with `blazr_floats_free(*out_values, *out_len)`.
/// A model with no hidden-state output returns `BLAZR_STATUS_UNSUPPORTED`.
///
/// # Safety
///
/// `text` is a NUL-terminated string. `out_values` and `out_len` are
/// writable.
#[no_mangle]
pub unsafe extern "C" fn blazr_embed(
    model: *mut BlazrModel,
    text: *const c_char,
    out_values: *mut *mut f32,
    out_len: *mut usize,
) -> BlazrStatus {
    guard(|| {
        if out_values.is_null() || out_len.is_null() {
            return Err(FfiError::invalid(
                "out_values or out_len is null. Pass pointers to write to",
            ));
        }
        let inner = handle::lookup(model)?;
        // SAFETY: forwarded from the caller.
        let text = unsafe { read_str(text, "text") }?;
        let vector = with_executor!(&inner.executor, executor => embed(executor, text))?;
        let (ptr, len) = into_c_array(vector);
        // SAFETY: both pointers checked non-null above.
        unsafe {
            write_out(out_values, ptr, "out_values")?;
            write_out(out_len, len, "out_len")?;
        }
        Ok(BlazrStatus::Ok)
    })
}

fn embed<R>(executor: &Executor<R>, text: &str) -> Result<Vec<f32>, FfiError>
where
    R: Runtime<DType = DType>,
    R::Client: EngineClient<R>,
{
    // Raw caller text, as the server's embeddings route reads it: no special
    // token in it becomes a control token.
    let ids = executor
        .tokenizer()
        .encode_with(text, &splintr::SpecialMode::Ordinary)
        .map_err(|e| FfiError::invalid(format!("text did not tokenize: {e}")))?;
    if ids.is_empty() {
        return Err(FfiError::invalid(
            "text encodes to no tokens. Pass a non-empty text",
        ));
    }
    let hidden = executor
        .get_embeddings(&ids)
        .map_err(|e| FfiError::new(BlazrStatus::Unsupported, format!("{e:#}")))?;
    if hidden.is_empty() || hidden.len() % ids.len() != 0 {
        return Err(FfiError::new(
            BlazrStatus::Generate,
            format!(
                "the model returned {} hidden values for {} tokens. Expected a whole number of rows",
                hidden.len(),
                ids.len()
            ),
        ));
    }
    let hidden_size = hidden.len() / ids.len();
    let mut vector = pool_mean(&hidden, ids.len(), hidden_size);
    l2_normalize(&mut vector);
    Ok(vector)
}
