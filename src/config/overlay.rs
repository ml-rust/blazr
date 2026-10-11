//! The `inference` section of `~/.blazr/config.yaml`, as an overlay.
//!
//! Each model's `config.json` sets its own inference values. The overlay
//! holds only the keys the user file sets, and writes them over every model.

use std::collections::BTreeSet;

use anyhow::{anyhow, Result};
use serde_json::{Map, Value};

use boostr::DType;

use super::{dtype_name, parse_dtype, InferenceConfig};

/// The key of the compute dtype request. It is never written over a
/// model's config. The loader reads it before any weight loads.
const DTYPE_KEY: &str = "dtype";

/// The inference keys the user file sets, with their parsed values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InferenceOverlay {
    values: Map<String, Value>,
}

impl InferenceOverlay {
    /// Keep the values of `keys` from a parsed `inference` section.
    pub fn from_section(section: &InferenceConfig, keys: &BTreeSet<String>) -> Result<Self> {
        let Value::Object(all) = serde_json::to_value(section)? else {
            return Err(anyhow!("the inference section did not serialize to a map"));
        };
        let values = all
            .into_iter()
            .filter(|(key, _)| keys.contains(key))
            .collect();
        Ok(Self { values })
    }

    /// The parsed value of `key`, when the user file sets it.
    pub fn value(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// The keys the user file sets, in struct field order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.values.keys().map(String::as_str)
    }

    /// `tensor_parallel_size`, when the user file sets it.
    pub fn tensor_parallel_size(&self) -> Option<usize> {
        self.value("tensor_parallel_size")
            .and_then(Value::as_u64)
            .and_then(|size| usize::try_from(size).ok())
    }

    /// `data_parallel`, when the user file sets it.
    pub fn data_parallel(&self) -> Option<usize> {
        self.value("data_parallel")
            .and_then(Value::as_u64)
            .and_then(|replicas| usize::try_from(replicas).ok())
    }

    /// `dtype`, when the user file sets it. The file parse checks the
    /// value, so an unknown name never reaches here.
    pub fn dtype(&self) -> Option<DType> {
        self.value(DTYPE_KEY)
            .and_then(Value::as_str)
            .and_then(|name| parse_dtype(name).ok())
    }

    /// Record the file's checked `dtype`.
    ///
    /// # Errors
    ///
    /// An error for a dtype that is not f32, bf16 or f16.
    pub fn set_dtype(&mut self, dtype: DType) -> Result<()> {
        self.values
            .insert(DTYPE_KEY.to_string(), Value::from(dtype_name(dtype)?));
        Ok(())
    }

    /// Whether the user file sets `graphs`.
    pub fn graphs(&self) -> Option<bool> {
        self.value("graphs").and_then(Value::as_bool)
    }

    /// Write each set key over `target`. Keys the file omits keep the
    /// model's value. `dtype` is a request the loader already resolved, so
    /// the loader's value stays.
    pub fn apply(&self, target: &mut InferenceConfig) -> Result<()> {
        if self.values.is_empty() {
            return Ok(());
        }
        let Value::Object(mut merged) = serde_json::to_value(&*target)? else {
            return Err(anyhow!(
                "the model inference config did not serialize to a map"
            ));
        };
        for (key, value) in self.values.iter().filter(|(key, _)| *key != DTYPE_KEY) {
            merged.insert(key.clone(), value.clone());
        }
        let dtype = std::mem::take(&mut target.dtype);
        let mut applied: InferenceConfig = serde_json::from_value(Value::Object(merged))
            .map_err(|e| anyhow!("the inference section of the user config did not apply: {e}"))?;
        applied.dtype = dtype;
        *target = applied;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overlay(section: InferenceConfig, keys: &[&str]) -> InferenceOverlay {
        let keys = keys.iter().map(|k| k.to_string()).collect();
        InferenceOverlay::from_section(&section, &keys).unwrap()
    }

    #[test]
    fn set_keys_replace_and_omitted_keys_keep_the_model_value() {
        let section = InferenceConfig {
            max_batch_size: 8,
            block_size: 32,
            ..Default::default()
        };
        // The file sets max_batch_size only. Its block_size is the default.
        let overlay = overlay(section, &["max_batch_size"]);
        let mut model = InferenceConfig {
            block_size: 64,
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        overlay.apply(&mut model).unwrap();
        assert_eq!(model.max_batch_size, 8);
        assert_eq!(model.block_size, 64);
        assert_eq!(model.dtype, "bf16");
    }

    #[test]
    fn empty_overlay_changes_nothing() {
        let mut model = InferenceConfig {
            paged_attention: true,
            ..Default::default()
        };
        InferenceOverlay::default().apply(&mut model).unwrap();
        assert!(model.paged_attention);
        assert_eq!(InferenceOverlay::default().tensor_parallel_size(), None);
    }

    #[test]
    fn reads_tensor_parallel_size_and_graphs_only_when_set() {
        let section = InferenceConfig {
            tensor_parallel_size: 4,
            graphs: true,
            ..Default::default()
        };
        assert_eq!(overlay(section.clone(), &[]).tensor_parallel_size(), None);
        let set = overlay(section, &["tensor_parallel_size", "graphs"]);
        assert_eq!(set.tensor_parallel_size(), Some(4));
        assert_eq!(set.graphs(), Some(true));
    }

    #[test]
    fn dtype_is_read_and_never_written_over_the_model() {
        let mut overlay = InferenceOverlay::default();
        assert_eq!(overlay.dtype(), None);
        overlay.set_dtype(DType::F32).unwrap();
        assert_eq!(overlay.dtype(), Some(DType::F32));
        assert_eq!(overlay.keys().collect::<Vec<_>>(), ["dtype"]);
        let mut model = InferenceConfig {
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        overlay.apply(&mut model).unwrap();
        assert_eq!(model.dtype, "bf16");
        assert!(overlay.set_dtype(DType::I32).is_err());
    }

    #[test]
    fn reads_data_parallel_only_when_set() {
        let section = InferenceConfig {
            data_parallel: 3,
            ..Default::default()
        };
        assert_eq!(overlay(section.clone(), &[]).data_parallel(), None);
        assert_eq!(
            overlay(section, &["data_parallel"]).data_parallel(),
            Some(3)
        );
    }
}
