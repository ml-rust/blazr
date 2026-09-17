//! Shared application state: model registries, inflight-token budget, and
//! per-request slot bookkeeping.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::config::UserConfig;
use crate::engine::{RequestScheduler, Scheduler, SlotManager};
use crate::server::metrics;

#[cfg(feature = "cuda")]
pub type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
pub type ServerRuntime = boostr::CpuRuntime;

/// Standalone vision embedder (SigLIP/CLIP) keyed by model name.
pub type VisionEmbedder = boostr::model::vision::ImageEmbedder<ServerRuntime>;

/// Standalone Whisper ASR bundle keyed by model name.
#[cfg(feature = "audio")]
pub type AsrBundle = boostr_audio::WhisperBundle<ServerRuntime>;

/// Standalone TTS bundle keyed by model name.
#[cfg(feature = "audio")]
pub type TtsBundle = boostr_audio::TtsBundle;

/// Shared application state
pub struct AppState {
    pub scheduler: Arc<Scheduler<ServerRuntime>>,
    pub metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    pub user_config: Arc<RwLock<UserConfig>>,
    /// Current in-flight token count (prompt + estimated decode tokens)
    pub inflight_tokens: AtomicUsize,
    /// Maximum in-flight token budget (0 = unlimited)
    pub max_inflight_tokens: usize,
    /// Inference slot manager
    pub slot_manager: SlotManager,
    /// Request scheduler for continuous batching (None = legacy single-request mode)
    pub request_scheduler: Option<Arc<RequestScheduler>>,
    /// Standalone vision embedders (SigLIP/CLIP) keyed by model name, used by
    /// `/v1/embeddings` for image inputs. Populated at server startup from CLI
    /// or config.
    pub vision_embedders: Arc<RwLock<HashMap<String, Arc<VisionEmbedder>>>>,
    /// Source directory each vision embedder was loaded from, keyed by model
    /// name. Recorded at registration time so `/v1/models`, `/api/tags` and
    /// `/api/ps` can report it without the loader itself keeping a path.
    pub(super) vision_paths: Arc<RwLock<HashMap<String, PathBuf>>>,
    /// Standalone Whisper ASR bundles keyed by model name, used by
    /// `/v1/audio/transcriptions`. Populated at server startup from `--asr-model`.
    #[cfg(feature = "audio")]
    pub asr_models: Arc<RwLock<HashMap<String, Arc<AsrBundle>>>>,
    /// Source directory each ASR bundle was loaded from, keyed by model name.
    #[cfg(feature = "audio")]
    pub(super) asr_paths: Arc<RwLock<HashMap<String, PathBuf>>>,
    /// Standalone TTS bundles keyed by model name, used by `/v1/audio/speech`.
    /// Populated at server startup from `--tts-model`.
    #[cfg(feature = "audio")]
    pub tts_models: Arc<RwLock<HashMap<String, Arc<TtsBundle>>>>,
    /// Source directory each TTS bundle was loaded from, keyed by model name.
    #[cfg(feature = "audio")]
    pub(super) tts_paths: Arc<RwLock<HashMap<String, PathBuf>>>,
    /// Directory searched for TTS voice files when the request specifies a
    /// bare voice ID (e.g. `af_alloy`). Resolved once at startup in this
    /// priority order: `--voice-dir` CLI flag, `$BLAZR_VOICE_DIR`, bundled
    /// `assets/kokoro_voices/` shipped with the binary.
    pub voice_dir: Option<PathBuf>,
}

impl AppState {
    pub fn new(
        scheduler: Arc<Scheduler<ServerRuntime>>,
        metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    ) -> Self {
        Self {
            scheduler,
            metrics_handle,
            user_config: Arc::new(RwLock::new(UserConfig::load())),
            inflight_tokens: AtomicUsize::new(0),
            max_inflight_tokens: 0,
            slot_manager: SlotManager::new(0), // unlimited by default
            request_scheduler: None,
            vision_embedders: Arc::new(RwLock::new(HashMap::new())),
            vision_paths: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "audio")]
            asr_models: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "audio")]
            asr_paths: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "audio")]
            tts_models: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "audio")]
            tts_paths: Arc::new(RwLock::new(HashMap::new())),
            voice_dir: None,
        }
    }

    /// Attach a voice directory (see [`AppState::voice_dir`]). Fluent builder
    /// for server startup.
    pub fn with_voice_dir(mut self, dir: Option<PathBuf>) -> Self {
        self.voice_dir = dir;
        self
    }

    /// Register a vision embedder under `model_name`, recording the
    /// directory it was loaded from for the model-listing endpoints.
    pub async fn register_vision_embedder(
        &self,
        model_name: String,
        embedder: Arc<VisionEmbedder>,
        path: PathBuf,
    ) {
        self.vision_embedders
            .write()
            .await
            .insert(model_name.clone(), embedder);
        self.vision_paths.write().await.insert(model_name, path);
    }

    /// Look up a vision embedder by model name.
    pub async fn vision_embedder(&self, model_name: &str) -> Option<Arc<VisionEmbedder>> {
        self.vision_embedders.read().await.get(model_name).cloned()
    }

    /// Register a Whisper ASR bundle under `model_name`, recording the
    /// directory it was loaded from for the model-listing endpoints.
    #[cfg(feature = "audio")]
    pub async fn register_asr_model(
        &self,
        model_name: String,
        bundle: Arc<AsrBundle>,
        path: PathBuf,
    ) {
        self.asr_models
            .write()
            .await
            .insert(model_name.clone(), bundle);
        self.asr_paths.write().await.insert(model_name, path);
    }

    /// Look up a Whisper ASR bundle by model name.
    #[cfg(feature = "audio")]
    pub async fn asr_model(&self, model_name: &str) -> Option<Arc<AsrBundle>> {
        self.asr_models.read().await.get(model_name).cloned()
    }

    /// Register a TTS bundle under `model_name`, recording the directory it
    /// was loaded from for the model-listing endpoints.
    #[cfg(feature = "audio")]
    pub async fn register_tts_model(
        &self,
        model_name: String,
        bundle: Arc<TtsBundle>,
        path: PathBuf,
    ) {
        self.tts_models
            .write()
            .await
            .insert(model_name.clone(), bundle);
        self.tts_paths.write().await.insert(model_name, path);
    }

    /// Look up a TTS bundle by model name.
    #[cfg(feature = "audio")]
    pub async fn tts_model(&self, model_name: &str) -> Option<Arc<TtsBundle>> {
        self.tts_models.read().await.get(model_name).cloned()
    }

    pub fn with_max_inflight_tokens(mut self, max: usize) -> Self {
        self.max_inflight_tokens = max;
        self
    }

    pub fn with_request_scheduler(mut self, rs: Arc<RequestScheduler>) -> Self {
        self.request_scheduler = Some(rs);
        self
    }

    /// Try to admit a request with the given token budget.
    /// Returns `false` (and does not increment) if the budget would be exceeded.
    pub fn try_admit(&self, tokens: usize) -> bool {
        if self.max_inflight_tokens == 0 {
            self.inflight_tokens.fetch_add(tokens, Ordering::Relaxed);
            metrics::adjust_inflight_tokens(tokens as f64);
            return true;
        }
        loop {
            let current = self.inflight_tokens.load(Ordering::Relaxed);
            if current + tokens > self.max_inflight_tokens {
                return false;
            }
            if self
                .inflight_tokens
                .compare_exchange_weak(
                    current,
                    current + tokens,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                metrics::adjust_inflight_tokens(tokens as f64);
                return true;
            }
        }
    }

    /// Release tokens back to the budget after a request completes
    pub fn release(&self, tokens: usize) {
        self.inflight_tokens.fetch_sub(tokens, Ordering::Relaxed);
        metrics::adjust_inflight_tokens(-(tokens as f64));
    }
}
