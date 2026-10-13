//! Hot reload of `~/.blazr/config.yaml` while `blazr serve` runs.
//!
//! The watcher polls the file's modification time. A change parses the
//! whole file. A file that does not parse leaves the running config as it
//! is. A file that parses replaces the running config in one swap.
//!
//! `generation` keys and `server.reasoning_format` apply to the next
//! request. Every other key needs a restart: the watcher logs one warning
//! per changed key.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crate::config::{reload, FileConfig, LiveConfig};

/// Poll interval for config file changes
const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Spawn a task that reloads `path` into `live` on each change.
pub fn spawn_config_watcher(live: Arc<LiveConfig>, path: PathBuf) {
    tokio::spawn(async move {
        let mut last_modified = file_modified(&path);
        loop {
            tokio::time::sleep(POLL_INTERVAL).await;
            let current_modified = file_modified(&path);
            if current_modified != last_modified {
                last_modified = current_modified;
                reload_into(&live, &path);
            }
        }
    });
}

/// Parse `path` and swap it into `live`, or keep `live` on an error.
fn reload_into(live: &LiveConfig, path: &Path) {
    let parsed = match FileConfig::read(path) {
        Ok(parsed) => parsed,
        Err(e) => {
            tracing::error!("config reload skipped, the running config stays: {e:#}");
            return;
        }
    };
    for key in &parsed.unknown_keys {
        tracing::warn!(
            "{}: unknown key '{}' is ignored. Check its spelling in docs/configuration.md",
            path.display(),
            key
        );
    }
    let old = live.replace(parsed.config);
    let new = live.snapshot();
    for key in reload::restart_changes(&old, &new) {
        tracing::warn!(
            "{}: {} changed. Restart blazr serve to apply it",
            path.display(),
            key
        );
    }
    tracing::info!(
        "Reloaded {}. generation keys and server.reasoning_format apply now",
        path.display()
    );
}

/// Get file modification time as an opaque comparable value
fn file_modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_good_file_swaps_in_and_a_bad_file_keeps_the_old_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        let live = LiveConfig::default();

        std::fs::write(
            &path,
            "generation:\n  temperature: 0.4\nserver:\n  port: 9100\n",
        )
        .unwrap();
        reload_into(&live, &path);
        assert_eq!(live.snapshot().generation.temperature, 0.4);
        assert_eq!(live.snapshot().server.port, 9100);

        std::fs::write(
            &path,
            "generation:\n  temperature: 0.9\nserver:\n  port: bad\n",
        )
        .unwrap();
        reload_into(&live, &path);
        let kept = live.snapshot();
        assert_eq!(kept.generation.temperature, 0.4);
        assert_eq!(kept.server.port, 9100);

        std::fs::remove_file(&path).unwrap();
        reload_into(&live, &path);
        assert_eq!(live.snapshot().server.port, 8080);
    }
}
