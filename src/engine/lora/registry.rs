//! [`LoraRegistry`]: the registered adapter variants of one scheduler.
//!
//! A registration outlives its executor. The scheduler evicts and unloads
//! variant executors like any loaded model. The next request for a
//! registered variant builds it again from its base. Only `DELETE
//! /v1/lora` removes a registration.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// One registered variant: an adapter applied to a base model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantInfo {
    /// The variant id: `<base>:<adapter>`.
    pub id: String,
    /// The base model name.
    pub base: String,
    /// The adapter name.
    pub adapter: String,
    /// The adapter directory or `.safetensors` file.
    pub path: PathBuf,
    /// The adapter's `r` from `adapter_config.json`.
    pub rank: usize,
    /// How many base projections the adapter adapts.
    pub adapted_modules: usize,
}

/// The registered variants, by variant id.
#[derive(Debug, Default)]
pub struct LoraRegistry {
    variants: RwLock<BTreeMap<String, VariantInfo>>,
}

impl LoraRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// A poisoned lock still guards whole entries: each insert and remove
    /// is one map call, so a panicking holder leaves no half-written entry.
    fn read(&self) -> RwLockReadGuard<'_, BTreeMap<String, VariantInfo>> {
        self.variants.read().unwrap_or_else(|p| p.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, BTreeMap<String, VariantInfo>> {
        self.variants.write().unwrap_or_else(|p| p.into_inner())
    }

    /// Register `info`, replacing a variant of the same id. Returns the
    /// replaced one.
    pub fn insert(&self, info: VariantInfo) -> Option<VariantInfo> {
        self.write().insert(info.id.clone(), info)
    }

    /// Unregister the variant `id`. Returns it, or `None` when it was not
    /// registered.
    pub fn remove(&self, id: &str) -> Option<VariantInfo> {
        self.write().remove(id)
    }

    /// The variant `id`, when registered.
    pub fn get(&self, id: &str) -> Option<VariantInfo> {
        self.read().get(id).cloned()
    }

    /// The adapter name of the variant `id`, when registered.
    pub fn adapter_of(&self, id: &str) -> Option<String> {
        self.read().get(id).map(|v| v.adapter.clone())
    }

    /// Whether `base` has a registered variant.
    pub fn has_variants(&self, base: &str) -> bool {
        self.read().values().any(|v| v.base == base)
    }

    /// The variants of `base`, by id.
    pub fn of_base(&self, base: &str) -> Vec<VariantInfo> {
        self.read()
            .values()
            .filter(|v| v.base == base)
            .cloned()
            .collect()
    }

    /// Every registered variant, by id.
    pub fn all(&self) -> Vec<VariantInfo> {
        self.read().values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(base: &str, adapter: &str, rank: usize) -> VariantInfo {
        VariantInfo {
            id: super::super::variant_id(base, adapter),
            base: base.to_string(),
            adapter: adapter.to_string(),
            path: PathBuf::from(format!("/adapters/{adapter}")),
            rank,
            adapted_modules: 3,
        }
    }

    #[test]
    fn register_replace_list_and_remove() {
        let registry = LoraRegistry::new();
        assert!(registry.get("m:kev").is_none());
        assert!(!registry.has_variants("m"));

        assert!(registry.insert(info("m", "kev", 16)).is_none());
        assert!(registry.insert(info("m", "jev", 8)).is_none());
        assert!(registry.insert(info("other", "kev", 4)).is_none());
        assert_eq!(registry.adapter_of("m:kev").as_deref(), Some("kev"));
        assert!(registry.has_variants("m"));

        let replaced = registry.insert(info("m", "kev", 32));
        assert_eq!(replaced.map(|v| v.rank), Some(16));
        assert_eq!(registry.get("m:kev").map(|v| v.rank), Some(32));

        let ids = |v: Vec<VariantInfo>| v.into_iter().map(|v| v.id).collect::<Vec<_>>();
        assert_eq!(ids(registry.of_base("m")), ["m:jev", "m:kev"]);
        assert_eq!(ids(registry.all()), ["m:jev", "m:kev", "other:kev"]);

        assert_eq!(registry.remove("m:kev").map(|v| v.rank), Some(32));
        assert!(registry.remove("m:kev").is_none());
        assert_eq!(ids(registry.of_base("m")), ["m:jev"]);
    }
}
