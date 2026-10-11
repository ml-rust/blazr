//! `blazr_tokenize`: text to the model's token ids.

use std::ffi::c_char;

use super::error::{guard, BlazrStatus, FfiError};
use super::handle::{self, with_executor, BlazrModel};
use super::string::{into_c_array, read_str, write_out};

/// Encode `text` with the model's tokenizer, the same encode that
/// `blazr_generate` runs on its prompt.
///
/// On success `*out_ids` and `*out_len` hold the ids. Free them with
/// `blazr_tokens_free(*out_ids, *out_len)`.
///
/// # Safety
///
/// `text` is a NUL-terminated string. `out_ids` and `out_len` are writable.
#[no_mangle]
pub unsafe extern "C" fn blazr_tokenize(
    model: *mut BlazrModel,
    text: *const c_char,
    out_ids: *mut *mut u32,
    out_len: *mut usize,
) -> BlazrStatus {
    guard(|| {
        if out_ids.is_null() || out_len.is_null() {
            return Err(FfiError::invalid(
                "out_ids or out_len is null. Pass pointers to write to",
            ));
        }
        let inner = handle::lookup(model)?;
        // SAFETY: forwarded from the caller.
        let text = unsafe { read_str(text, "text") }?;
        let ids = with_executor!(&inner.executor, executor => executor.tokenizer().encode(text));
        let (ptr, len) = into_c_array(ids);
        // SAFETY: both pointers checked non-null above.
        unsafe {
            write_out(out_ids, ptr, "out_ids")?;
            write_out(out_len, len, "out_len")?;
        }
        Ok(BlazrStatus::Ok)
    })
}
