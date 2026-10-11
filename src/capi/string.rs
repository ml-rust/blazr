//! UTF-8 strings and buffers across the C boundary.
//!
//! blazr allocates every string and array it hands out. The caller returns
//! each one to its matching free function, never to `free()`.

use std::ffi::{c_char, CStr, CString};

use super::error::FfiError;

/// The library version, such as `0.1.3`. A static string: do not free it.
#[no_mangle]
pub extern "C" fn blazr_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Free a string that blazr returned. A null pointer is a no-op.
///
/// # Safety
///
/// `text` is null or a string from blazr that has not been freed yet.
#[no_mangle]
pub unsafe extern "C" fn blazr_string_free(text: *mut c_char) {
    if !text.is_null() {
        // SAFETY: the caller passes a pointer from `CString::into_raw`.
        drop(unsafe { CString::from_raw(text) });
    }
}

/// Free a token id array from `blazr_tokenize`. A null pointer is a no-op.
///
/// # Safety
///
/// `ids` and `len` are the exact pair `blazr_tokenize` returned.
#[no_mangle]
pub unsafe extern "C" fn blazr_tokens_free(ids: *mut u32, len: usize) {
    if !ids.is_null() {
        // SAFETY: the caller passes the pair from `into_c_array`.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ids, len)) });
    }
}

/// Free a float array from `blazr_embed`. A null pointer is a no-op.
///
/// # Safety
///
/// `values` and `len` are the exact pair `blazr_embed` returned.
#[no_mangle]
pub unsafe extern "C" fn blazr_floats_free(values: *mut f32, len: usize) {
    if !values.is_null() {
        // SAFETY: the caller passes the pair from `into_c_array`.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(values, len)) });
    }
}

/// Read a NUL-terminated UTF-8 argument. `name` names it in the error.
///
/// # Safety
///
/// `text` is null or points to a NUL-terminated string.
pub(crate) unsafe fn read_str<'a>(text: *const c_char, name: &str) -> Result<&'a str, FfiError> {
    if text.is_null() {
        return Err(FfiError::invalid(format!(
            "{name} is null. Pass a UTF-8 string"
        )));
    }
    // SAFETY: the caller guarantees a NUL-terminated string.
    unsafe { CStr::from_ptr(text) }
        .to_str()
        .map_err(|e| FfiError::invalid(format!("{name} is not UTF-8: {e}")))
}

/// Like [`read_str`], but a null pointer reads as `None`.
///
/// # Safety
///
/// `text` is null or points to a NUL-terminated string.
pub(crate) unsafe fn read_optional_str<'a>(
    text: *const c_char,
    name: &str,
) -> Result<Option<&'a str>, FfiError> {
    if text.is_null() {
        Ok(None)
    } else {
        // SAFETY: forwarded from the caller.
        unsafe { read_str(text, name) }.map(Some)
    }
}

/// Hand `text` to C as an owned string for [`blazr_string_free`].
pub(crate) fn into_c_string(text: String) -> Result<*mut c_char, FfiError> {
    CString::new(text).map(CString::into_raw).map_err(|e| {
        let at = e.nul_position();
        let message =
            format!("the generated text holds a NUL byte at offset {at}. Use the token callback");
        FfiError::new(super::error::BlazrStatus::Generate, message)
    })
}

/// Hand `values` to C as an owned array plus its length.
pub(crate) fn into_c_array<T>(values: Vec<T>) -> (*mut T, usize) {
    let boxed = values.into_boxed_slice();
    let len = boxed.len();
    (Box::into_raw(boxed).cast::<T>(), len)
}

/// Write `value` through an out pointer that the caller passed.
///
/// # Safety
///
/// `out` is null or valid for one write of `T`.
pub(crate) unsafe fn write_out<T>(out: *mut T, value: T, name: &str) -> Result<(), FfiError> {
    if out.is_null() {
        return Err(FfiError::invalid(format!(
            "{name} is null. Pass a pointer to write to"
        )));
    }
    // SAFETY: checked non-null, the caller guarantees it is writable.
    unsafe { out.write(value) };
    Ok(())
}
