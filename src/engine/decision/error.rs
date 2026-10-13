//! Why a decision request got no answers.

use std::fmt;

/// Why a decision request got no answers. Each kind maps to one HTTP status
/// on `/v1/systemone`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionFailure {
    /// No model of that name exists (404).
    ModelNotFound(String),
    /// The model is not a decision model (400).
    NotDecisionModel(String),
    /// The request is malformed or its family refuses it (422).
    InvalidRequest(String),
    /// The request is over a row or token limit (413).
    TooLarge(String),
    /// The queue holds too many rows (503, retry later).
    Busy(String),
    /// The model is not serving decisions right now (503).
    Unavailable(String),
    /// The server failed (500).
    Internal(String),
}

impl DecisionFailure {
    /// The caller-facing message.
    pub fn message(&self) -> &str {
        match self {
            Self::ModelNotFound(m)
            | Self::NotDecisionModel(m)
            | Self::InvalidRequest(m)
            | Self::TooLarge(m)
            | Self::Busy(m)
            | Self::Unavailable(m)
            | Self::Internal(m) => m,
        }
    }
}

impl fmt::Display for DecisionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for DecisionFailure {}

/// A LoRA variant failure on `/v1/systemone`: an unknown model or adapter
/// is a 404, a base that takes no adapter is not a decision model, a
/// malformed selection or an adapter that does not load or fit is a 422,
/// and a base that does not load is a 500.
impl From<crate::engine::LoraError> for DecisionFailure {
    fn from(error: crate::engine::LoraError) -> Self {
        use crate::engine::LoraError;
        match error {
            LoraError::ModelNotFound(m) | LoraError::AdapterNotFound(m) => Self::ModelNotFound(m),
            LoraError::Unsupported(m) => Self::NotDecisionModel(m),
            LoraError::Invalid(m) | LoraError::Adapter(m) => Self::InvalidRequest(m),
            LoraError::BaseLoad(m) => Self::Internal(m),
        }
    }
}

/// An unknown model on `/v1/systemone` is a 404.
impl From<crate::engine::ModelNotFound> for DecisionFailure {
    fn from(error: crate::engine::ModelNotFound) -> Self {
        Self::ModelNotFound(error.to_string())
    }
}

impl DecisionFailure {
    /// The failure of an executor lookup of the model `name`: a
    /// [`LoraError`](crate::engine::LoraError) or
    /// [`ModelNotFound`](crate::engine::ModelNotFound) by its kind, any
    /// other error a load that failed (500).
    pub fn of_lookup(name: &str, error: anyhow::Error) -> Self {
        let error = match error.downcast::<crate::engine::LoraError>() {
            Ok(lora) => return Self::from(lora),
            Err(error) => error,
        };
        match error.downcast::<crate::engine::ModelNotFound>() {
            Ok(missing) => Self::from(missing),
            Err(error) => Self::Internal(format!("model '{name}' did not load: {error:#}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{LoraError, ModelNotFound};

    #[test]
    fn lookups_take_the_route_statuses() {
        let missing = DecisionFailure::of_lookup("m", ModelNotFound::new("m").into());
        assert_eq!(
            missing,
            DecisionFailure::ModelNotFound("Model not found: m".into())
        );
        let lora = DecisionFailure::of_lookup("m", LoraError::Invalid("bad".into()).into());
        assert_eq!(lora, DecisionFailure::InvalidRequest("bad".into()));
        let load = DecisionFailure::of_lookup("m", anyhow::anyhow!("weights do not read"));
        assert_eq!(
            load,
            DecisionFailure::Internal("model 'm' did not load: weights do not read".into())
        );
    }

    #[test]
    fn lora_failures_take_the_route_statuses() {
        let cases = [
            (LoraError::ModelNotFound("m".into()), "ModelNotFound"),
            (LoraError::AdapterNotFound("m".into()), "ModelNotFound"),
            (LoraError::Unsupported("m".into()), "NotDecisionModel"),
            (LoraError::Invalid("m".into()), "InvalidRequest"),
            (LoraError::Adapter("m".into()), "InvalidRequest"),
            (LoraError::BaseLoad("m".into()), "Internal"),
        ];
        for (error, kind) in cases {
            let failure = DecisionFailure::from(error);
            assert!(format!("{failure:?}").starts_with(kind), "{failure:?}");
            assert_eq!(failure.message(), "m");
        }
    }
}
