//! Named registry of loaded PEFT LoRA adapters.
//!
//! boostr's [`PeftAdapter`] reads and checks the adapter files. This module
//! names the loaded adapters and holds them for the executor and the
//! `/v1/lora` routes.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use boostr::format::PeftAdapter;
use boostr::Runtime;

/// Thread-safe registry of named LoRA adapters.
///
/// Held as `Arc<LoraAdapterRegistry<R>>` inside the `Executor` so that adapters
/// can be loaded and unloaded concurrently with ongoing inference.
pub struct LoraAdapterRegistry<R: Runtime> {
    adapters: RwLock<HashMap<String, Arc<PeftAdapter<R>>>>,
}

impl<R: Runtime> LoraAdapterRegistry<R> {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            adapters: RwLock::new(HashMap::new()),
        }
    }

    /// Insert a loaded adapter under `name`.
    ///
    /// If an adapter with the same name already exists it is replaced.
    ///
    /// # Lock poisoning
    ///
    /// Every method here recovers a poisoned lock via `into_inner()` rather
    /// than panicking. The guarded value is a plain `HashMap` of adapters: a
    /// panic elsewhere while holding the lock cannot leave it in a state that
    /// is unsafe to read, only one where an insert or removal may not have
    /// landed. Panicking instead would take a serving process down for the
    /// rest of its life on every subsequent registry access.
    pub fn insert(&self, name: &str, adapter: PeftAdapter<R>) {
        let mut guard = self.adapters.write().unwrap_or_else(|p| p.into_inner());
        guard.insert(name.to_string(), Arc::new(adapter));
    }

    /// Remove an adapter by name. Returns `true` if the adapter existed.
    pub fn remove(&self, name: &str) -> bool {
        let mut guard = self.adapters.write().unwrap_or_else(|p| p.into_inner());
        guard.remove(name).is_some()
    }

    /// Retrieve an adapter by name.
    pub fn get(&self, name: &str) -> Option<Arc<PeftAdapter<R>>> {
        let guard = self.adapters.read().unwrap_or_else(|p| p.into_inner());
        guard.get(name).cloned()
    }

    /// List names of all currently loaded adapters.
    pub fn list(&self) -> Vec<String> {
        let guard = self.adapters.read().unwrap_or_else(|p| p.into_inner());
        guard.keys().cloned().collect()
    }
}

impl<R: Runtime> Default for LoraAdapterRegistry<R> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::format::PeftConfig;
    use boostr::CpuRuntime;
    use std::collections::BTreeMap;

    fn adapter(r: usize) -> PeftAdapter<CpuRuntime> {
        let text = format!(r#"{{"r": {r}, "lora_alpha": 16}}"#);
        PeftAdapter {
            config: PeftConfig::from_json_str(&text).unwrap(),
            layers: BTreeMap::new(),
        }
    }

    #[test]
    fn insert_get_replace_remove() {
        let registry = LoraAdapterRegistry::<CpuRuntime>::new();
        assert!(registry.get("style").is_none());

        registry.insert("style", adapter(8));
        assert_eq!(registry.get("style").unwrap().config.r, 8);
        assert_eq!(registry.list(), vec!["style".to_string()]);

        registry.insert("style", adapter(4));
        assert_eq!(registry.get("style").unwrap().config.r, 4);
        assert_eq!(registry.list().len(), 1);

        assert!(registry.remove("style"));
        assert!(!registry.remove("style"));
        assert!(registry.list().is_empty());
    }
}
