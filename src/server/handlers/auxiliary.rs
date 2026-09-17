//! Auxiliary (non-text) models: vision embedders, ASR and TTS bundles.
//!
//! These are resident from server startup (`--vision-model`, `--asr-model`,
//! `--tts-model`) but were previously invisible to every model-listing
//! endpoint, which read only the text-model scheduler. This module gives
//! them one merge point so `/v1/models`, `/api/tags` and `/api/ps` can list
//! them alongside text models.
//!
//! The `*_paths` maps in [`AppState`] are the source of truth for listing:
//! `register_*` always inserts a name into the bundle map and its path into
//! the matching paths map together, so the paths map alone carries every
//! registered name.

use std::path::PathBuf;

use super::state::AppState;

/// Which registry an [`AuxiliaryModel`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxiliaryKind {
    Tts,
    Asr,
    Vision,
}

impl AuxiliaryKind {
    /// Ollama-style `details.family` value for this kind.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tts => "tts",
            Self::Asr => "asr",
            Self::Vision => "vision",
        }
    }
}

/// A non-text model resident in the server, for the model-listing endpoints.
#[derive(Debug, Clone)]
pub struct AuxiliaryModel {
    pub name: String,
    pub kind: AuxiliaryKind,
    /// Directory it was loaded from. Always `Some` for a model registered
    /// through `register_vision_embedder`/`register_asr_model`/
    /// `register_tts_model`, which require a path.
    pub path: Option<PathBuf>,
}

impl AppState {
    /// All auxiliary (vision, ASR, TTS) models currently registered, sorted
    /// by name.
    pub async fn auxiliary_models(&self) -> Vec<AuxiliaryModel> {
        let mut models = Vec::new();

        for (name, path) in self.vision_paths.read().await.iter() {
            models.push(AuxiliaryModel {
                name: name.clone(),
                kind: AuxiliaryKind::Vision,
                path: Some(path.clone()),
            });
        }

        #[cfg(feature = "audio")]
        {
            for (name, path) in self.asr_paths.read().await.iter() {
                models.push(AuxiliaryModel {
                    name: name.clone(),
                    kind: AuxiliaryKind::Asr,
                    path: Some(path.clone()),
                });
            }
            for (name, path) in self.tts_paths.read().await.iter() {
                models.push(AuxiliaryModel {
                    name: name.clone(),
                    kind: AuxiliaryKind::Tts,
                    path: Some(path.clone()),
                });
            }
        }

        models.sort_by(|a, b| a.name.cmp(&b.name));
        models
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Scheduler;
    use std::sync::Arc;

    #[cfg(not(feature = "cuda"))]
    type TestRuntime = boostr::CpuRuntime;
    #[cfg(feature = "cuda")]
    type TestRuntime = boostr::CudaRuntime;

    fn test_state() -> AppState {
        #[cfg(not(feature = "cuda"))]
        let device = boostr::CpuDevice::new();
        #[cfg(feature = "cuda")]
        let device = boostr::CudaDevice::new(0);
        let scheduler = Arc::new(Scheduler::<TestRuntime>::new(
            std::env::temp_dir().join("blazr-auxiliary-test-models"),
            device,
        ));
        let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        AppState::new(scheduler, metrics_handle)
    }

    #[tokio::test]
    async fn empty_state_has_no_auxiliary_models() {
        let state = test_state();
        assert!(state.auxiliary_models().await.is_empty());
    }

    #[tokio::test]
    async fn vision_path_is_reported_as_a_vision_model() {
        let state = test_state();
        state
            .vision_paths
            .write()
            .await
            .insert("clip".to_string(), PathBuf::from("/models/clip"));

        let models = state.auxiliary_models().await;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "clip");
        assert_eq!(models[0].kind, AuxiliaryKind::Vision);
        assert_eq!(models[0].path, Some(PathBuf::from("/models/clip")));
    }

    #[tokio::test]
    async fn auxiliary_models_are_sorted_by_name_across_kinds() {
        let state = test_state();
        state
            .vision_paths
            .write()
            .await
            .insert("zeta".to_string(), PathBuf::from("/models/zeta"));
        state
            .vision_paths
            .write()
            .await
            .insert("alpha".to_string(), PathBuf::from("/models/alpha"));

        let models = state.auxiliary_models().await;
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "zeta"]);
    }

    #[cfg(feature = "audio")]
    #[tokio::test]
    async fn asr_and_tts_paths_are_both_reported() {
        let state = test_state();
        state
            .asr_paths
            .write()
            .await
            .insert("whisper".to_string(), PathBuf::from("/models/whisper"));
        state
            .tts_paths
            .write()
            .await
            .insert("kokoro".to_string(), PathBuf::from("/models/kokoro"));

        let models = state.auxiliary_models().await;
        assert_eq!(models.len(), 2);
        assert!(models
            .iter()
            .any(|m| m.name == "whisper" && m.kind == AuxiliaryKind::Asr));
        assert!(models
            .iter()
            .any(|m| m.name == "kokoro" && m.kind == AuxiliaryKind::Tts));
    }
}
