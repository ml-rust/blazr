//! [`LoraError`]: why an adapter variant does not register, build or
//! resolve. The server maps each kind to its own HTTP status.

/// A failure of the adapter variant path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraError {
    /// No model has the name.
    ModelNotFound(String),
    /// No adapter of the name is registered for the model.
    AdapterNotFound(String),
    /// A malformed adapter name or argument, or a request whose `model`
    /// and `lora_adapter` name different adapters.
    Invalid(String),
    /// The base model takes no adapter: not a `qwen35` model, a variant
    /// itself, or served by data-parallel replicas.
    Unsupported(String),
    /// The adapter files do not load, or the adapter does not fit the base
    /// model.
    Adapter(String),
    /// The base model did not load.
    BaseLoad(String),
}

impl LoraError {
    /// The refusal of a variant of `model`, which runs on `replicas`
    /// data-parallel replicas. A variant would run on the first replica's
    /// weights alone, outside the replica router and its batching.
    pub fn data_parallel(model: &str, replicas: usize) -> Self {
        Self::Unsupported(format!(
            "model {model:?} runs on {replicas} data-parallel replicas, and LoRA variants are \
             not replicated. Serve it with inference.data_parallel 1 to apply adapters"
        ))
    }

    /// The message, without the kind.
    pub fn message(&self) -> &str {
        match self {
            Self::ModelNotFound(m)
            | Self::AdapterNotFound(m)
            | Self::Invalid(m)
            | Self::Unsupported(m)
            | Self::Adapter(m)
            | Self::BaseLoad(m) => m,
        }
    }
}

impl std::fmt::Display for LoraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for LoraError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_the_message_and_survives_anyhow() {
        let error = LoraError::AdapterNotFound("no adapter 'x'".into());
        assert_eq!(error.to_string(), "no adapter 'x'");
        let wrapped = anyhow::Error::from(error.clone());
        assert_eq!(wrapped.downcast_ref::<LoraError>(), Some(&error));
    }
}
