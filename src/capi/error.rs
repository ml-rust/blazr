//! Status codes, the per-thread error message, and the panic guard.
//!
//! Every exported function returns a [`BlazrStatus`]. A failing call stores
//! its message for [`blazr_last_error`] on the calling thread. No panic
//! crosses the C boundary: [`guard`] turns one into [`BlazrStatus::Panic`].

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::fmt::Display;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// The result of every exported call. Zero and above mean success.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlazrStatus {
    /// The call succeeded.
    Ok = 0,
    /// Generation stopped early: the callback returned nonzero, or
    /// `blazr_cancel` ran. The text so far is valid.
    Cancelled = 1,
    /// A pointer is null, or a string is not UTF-8, or a value is out of range.
    InvalidArgument = -1,
    /// The model handle is unknown or already unloaded.
    InvalidHandle = -2,
    /// The model did not load.
    Load = -3,
    /// Tokenizing, generating or embedding failed.
    Generate = -4,
    /// This library build or this model does not support the request.
    Unsupported = -5,
    /// blazr panicked. The handle stays valid, but the call made no result.
    Panic = -6,
}

/// A failed call: its status and the message for [`blazr_last_error`].
#[derive(Debug)]
pub(crate) struct FfiError {
    pub status: BlazrStatus,
    pub message: String,
}

impl FfiError {
    pub fn new(status: BlazrStatus, message: impl Display) -> Self {
        Self {
            status,
            message: message.to_string(),
        }
    }

    pub fn invalid(message: impl Display) -> Self {
        Self::new(BlazrStatus::InvalidArgument, message)
    }
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(message: &str) {
    // A NUL byte would end the C string early. Replace it.
    let clean = message.replace('\0', "\u{FFFD}");
    let text = CString::new(clean).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = text);
}

/// Run `body`, store any error for this thread, and return the status.
///
/// A panic inside `body` stops here and returns [`BlazrStatus::Panic`].
pub(crate) fn guard(body: impl FnOnce() -> Result<BlazrStatus, FfiError>) -> BlazrStatus {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            set_last_error(&error.message);
            error.status
        }
        Err(payload) => {
            let what = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with no message".to_string());
            set_last_error(&format!("blazr panicked: {what}"));
            BlazrStatus::Panic
        }
    }
}

/// The message of the last failed call on this thread, as UTF-8.
///
/// The pointer stays valid until the next failed call on this thread. Do not
/// free it. Before any failure it points to an empty string.
#[no_mangle]
pub extern "C" fn blazr_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}
