//! Config file watcher for hot-reload
//!
//! Polls `~/.blazr/config.yaml` for changes and updates shared state.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use crate::config::UserConfig;

/// Poll interval for config file changes
const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Spawn a background task that watches the config file for changes.
///
/// When the file's modification time changes, it reloads and updates the shared config.
pub fn spawn_config_watcher(user_config: Arc<RwLock<UserConfig>>) {
    tokio::spawn(async move {
        let config_path = match UserConfig::config_path() {
            Some(p) => p,
            None => return,
        };

        let mut last_modified = file_modified(&config_path);

        loop {
            tokio::time::sleep(POLL_INTERVAL).await;

            let current_modified = file_modified(&config_path);
            if current_modified != last_modified {
                last_modified = current_modified;

                let new_config = UserConfig::load();
                tracing::info!("Reloaded config from {}", config_path.display());

                let mut guard = user_config.write().await;
                *guard = new_config;
            }
        }
    });
}

/// Get file modification time as an opaque comparable value
fn file_modified(path: &std::path::Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).ok().and_then(|m| m.modified().ok())
}
