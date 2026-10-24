//! Ollama model names: the `:latest` tag and the not-found message.

use crate::server::handlers::AppState;

/// The tag Ollama clients add to an untagged name.
const LATEST: &str = ":latest";

/// Whether `name` is a loaded model, a model on disk, or a registered
/// LoRA variant.
pub(in crate::server) async fn is_known(state: &AppState, name: &str) -> bool {
    state.scheduler.is_loaded(name).await
        || state.scheduler.model_path(name).is_some()
        || state.scheduler.variant(name).is_some()
}

/// The model `name` runs on.
///
/// A trailing `:latest` is cut when `name` itself is unknown and the cut
/// name is known. Every other name passes through unchanged, so a
/// `<model>:<adapter>` LoRA variant still selects its adapter.
pub(in crate::server) async fn resolve(state: &AppState, name: &str) -> String {
    let Some(stripped) = name.strip_suffix(LATEST) else {
        return name.to_string();
    };
    let name_known = is_known(state, name).await;
    let stripped_known = !name_known && is_known(state, stripped).await;
    pick(name, name_known, stripped_known)
}

/// `name`, or `name` without `:latest` when only the cut name is known.
fn pick(name: &str, name_known: bool, stripped_known: bool) -> String {
    match name.strip_suffix(LATEST) {
        Some(stripped) if !name_known && stripped_known => stripped.to_string(),
        _ => name.to_string(),
    }
}

/// The Ollama message of an unknown model, under the name the client sent.
pub(in crate::server) fn not_found(name: &str) -> String {
    format!("model '{name}' not found")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::handlers::test_state;

    #[test]
    fn latest_is_cut_only_when_the_cut_name_is_the_known_one() {
        assert_eq!(pick("llama3:latest", false, true), "llama3");
        assert_eq!(pick("llama3:latest", true, true), "llama3:latest");
        assert_eq!(pick("llama3:latest", false, false), "llama3:latest");
        assert_eq!(pick("base:adapter", false, true), "base:adapter");
        assert_eq!(pick("llama3", false, false), "llama3");
    }

    #[tokio::test]
    async fn resolve_reads_the_model_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("llama3")).unwrap();
        std::fs::write(dir.path().join("llama3/config.json"), "{}").unwrap();
        let state = test_state(dir.path());
        assert_eq!(resolve(&state, "llama3:latest").await, "llama3");
        assert_eq!(resolve(&state, "llama3").await, "llama3");
        assert_eq!(resolve(&state, "llama3:adapter").await, "llama3:adapter");
        assert_eq!(resolve(&state, "ghost:latest").await, "ghost:latest");
    }

    #[test]
    fn not_found_names_the_requested_model() {
        assert_eq!(not_found("x:latest"), "model 'x:latest' not found");
    }
}
