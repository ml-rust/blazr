//! The user config that a running server reads, swapped whole on reload.

use std::sync::{Arc, PoisonError, RwLock};

use super::FileConfig;

/// The current user config. A reload replaces the whole value at once, so
/// a reader never sees part of an old file and part of a new one.
#[derive(Debug, Default)]
pub struct LiveConfig {
    current: RwLock<Arc<FileConfig>>,
}

impl LiveConfig {
    /// Start from `config`.
    pub fn new(config: FileConfig) -> Self {
        Self {
            current: RwLock::new(Arc::new(config)),
        }
    }

    /// The current config. Hold the `Arc` for one request's reads.
    pub fn snapshot(&self) -> Arc<FileConfig> {
        Arc::clone(&self.current.read().unwrap_or_else(PoisonError::into_inner))
    }

    /// Replace the current config and return the one it replaces.
    pub fn replace(&self, config: FileConfig) -> Arc<FileConfig> {
        let mut guard = self.current.write().unwrap_or_else(PoisonError::into_inner);
        std::mem::replace(&mut *guard, Arc::new(config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_swaps_the_whole_config() {
        let live = LiveConfig::default();
        let before = live.snapshot();
        let mut next = FileConfig::default();
        next.generation.temperature = 0.25;
        next.num_ctx = Some(64);
        let old = live.replace(next);
        assert_eq!(old.generation.temperature, before.generation.temperature);
        let now = live.snapshot();
        assert_eq!(now.generation.temperature, 0.25);
        assert_eq!(now.num_ctx, Some(64));
        // A snapshot taken before the swap keeps the old values.
        assert_eq!(before.num_ctx, None);
    }
}
