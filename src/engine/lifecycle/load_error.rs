//! Why an executor lookup failed: [`ModelNotFound`] when the name resolves
//! to no model, and [`is_out_of_memory`] for a load the device could not
//! hold. The server maps each to its own HTTP status.

use std::error::Error;
use std::fmt;

use boostr::NumrError;

/// No model has the name: it is neither a path nor an entry of the model
/// directory. Carried through `anyhow`, so a caller can `downcast_ref` it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelNotFound {
    /// The requested model name.
    pub name: String,
}

impl ModelNotFound {
    /// The error for the model `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl fmt::Display for ModelNotFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model not found: {}", self.name)
    }
}

impl Error for ModelNotFound {}

/// Whether `error` or a cause in its chain is a typed device or host
/// out-of-memory error. A cause that a load stringified is not seen, so
/// `false` means "not recognized", not "not out of memory".
pub fn is_out_of_memory(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause.downcast_ref::<NumrError>().is_some_and(numr_oom)
            || cause
                .downcast_ref::<Box<NumrError>>()
                .is_some_and(|e| numr_oom(e))
            || matches!(
                cause.downcast_ref::<boostr::error::Error>(),
                Some(boostr::error::Error::Numr(e)) if numr_oom(e)
            )
    })
}

/// Whether `error` is an out-of-memory error, or an allocation error that
/// wraps one.
fn numr_oom(error: &NumrError) -> bool {
    match error {
        NumrError::OutOfMemory { .. } => true,
        NumrError::AllocFailed { source, .. } => numr_oom(source),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_found_names_the_model_once_and_survives_anyhow() {
        let error = ModelNotFound::new("m");
        assert_eq!(error.to_string(), "Model not found: m");
        let wrapped = anyhow::Error::from(error.clone()).context("while serving");
        assert_eq!(wrapped.downcast_ref::<ModelNotFound>(), Some(&error));
    }

    #[test]
    fn out_of_memory_is_found_in_the_chain() {
        let oom = NumrError::OutOfMemory { size: 1 << 30 };
        assert!(is_out_of_memory(&anyhow::Error::from(oom)));

        let alloc = NumrError::AllocFailed {
            shape: vec![4, 4],
            dtype: "f32".into(),
            device: "cpu".into(),
            source: Box::new(NumrError::OutOfMemory { size: 64 }),
        };
        let boostr_error = boostr::error::Error::Numr(alloc);
        let wrapped = anyhow::Error::from(boostr_error).context("model 'm' did not load");
        assert!(is_out_of_memory(&wrapped));
    }

    #[test]
    fn other_errors_are_not_out_of_memory() {
        assert!(!is_out_of_memory(&anyhow::anyhow!("out of memory")));
        assert!(!is_out_of_memory(&anyhow::Error::from(ModelNotFound::new(
            "m"
        ))));
        assert!(!is_out_of_memory(&anyhow::Error::from(
            NumrError::DeviceMismatch
        )));
    }
}
