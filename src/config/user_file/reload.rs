//! Which user config keys apply on a hot reload, and which need a restart.
//!
//! Live: every `generation` key and `server.reasoning_format`. Requests
//! read them when they omit a value. Every other key sets up the server,
//! the scheduler or a model at start, so a change needs a restart.

use serde::Serialize;
use serde_json::{Map, Value};

use super::FileConfig;

/// Whether a dotted key applies without a restart.
pub fn is_live_key(key: &str) -> bool {
    key == "generation" || key.starts_with("generation.") || key == "server.reasoning_format"
}

/// The restart-only keys whose value differs between `old` and `new`.
pub fn restart_changes(old: &FileConfig, new: &FileConfig) -> Vec<String> {
    let old = flatten(old);
    let new = flatten(new);
    let mut keys: Vec<&String> = old.keys().chain(new.keys()).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter(|key| !is_live_key(key))
        .filter(|key| old.get(*key) != new.get(*key))
        .cloned()
        .collect()
}

/// One entry per user key and per section key, such as `server.port`.
/// An inference key the file omits is absent.
fn flatten(config: &FileConfig) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert("default_model".into(), to_value(&config.default_model));
    out.insert("device".into(), to_value(&config.device));
    out.insert("num_ctx".into(), to_value(&config.num_ctx));
    out.insert("model_dir".into(), to_value(&config.model_dir));
    section(&mut out, "server", to_value(&config.server));
    section(&mut out, "generation", to_value(&config.generation));
    for key in config.inference.keys() {
        if let Some(value) = config.inference.value(key) {
            out.insert(format!("inference.{key}"), value.clone());
        }
    }
    out
}

fn section(out: &mut Map<String, Value>, name: &str, value: Value) {
    if let Value::Object(map) = value {
        for (key, value) in map {
            out.insert(format!("{name}.{key}"), value);
        }
    }
}

fn to_value<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn file(text: &str) -> FileConfig {
        FileConfig::parse(text, Path::new("config.yaml"))
            .unwrap()
            .config
    }

    #[test]
    fn live_keys_need_no_restart() {
        let old = file("generation:\n  temperature: 0.2\n");
        let new = file("generation:\n  temperature: 0.9\nserver:\n  reasoning_format: none\n");
        assert!(restart_changes(&old, &new).is_empty());
    }

    #[test]
    fn restart_keys_are_named_one_each() {
        let old = file("num_ctx: 1024\nserver:\n  port: 9000\ninference:\n  max_batch_size: 2\n");
        let new = file(
            "num_ctx: 2048\nserver:\n  port: 9001\n  max_body_size: 5\n\
             inference:\n  max_batch_size: 2\n  prefix_cache: true\n",
        );
        assert_eq!(
            restart_changes(&old, &new),
            [
                "inference.prefix_cache",
                "num_ctx",
                "server.max_body_size",
                "server.port",
            ]
        );
    }

    #[test]
    fn removing_an_inference_key_is_a_change() {
        let old = file("inference:\n  graphs: true\n");
        let new = file("");
        assert_eq!(restart_changes(&old, &new), ["inference.graphs"]);
    }

    #[test]
    fn unchanged_files_report_nothing() {
        let text = "default_model: m\nserver:\n  slo:\n    ttft_p50_ms: 5\n";
        assert!(restart_changes(&file(text), &file(text)).is_empty());
    }
}
