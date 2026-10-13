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
