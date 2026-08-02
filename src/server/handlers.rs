//! HTTP request handlers
//!
//! Core handlers: health, models, tokenize/detokenize.
//! Completion and chat handlers are in separate modules.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use tokio::sync::RwLock;

use super::generation::{encode_with_allowed_special, error_response, policy_error_response};
use super::metrics;
use super::tools::{request_msg_to_chat_msg, ChatRequestMessage};
use crate::config::UserConfig;
use crate::engine::{RequestScheduler, Scheduler, SlotManager};

#[cfg(feature = "cuda")]
pub type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
pub type ServerRuntime = boostr::CpuRuntime;

/// Standalone vision embedder (SigLIP/CLIP) keyed by model name.
pub type VisionEmbedder = boostr::model::vision::ImageEmbedder<ServerRuntime>;

/// Standalone Whisper ASR bundle keyed by model name.
pub type AsrBundle = boostr::model::audio::WhisperBundle<ServerRuntime>;

/// Standalone TTS bundle keyed by model name.
pub type TtsBundle = boostr::model::audio::TtsBundle;

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
    /// Standalone Whisper ASR bundles keyed by model name, used by
    /// `/v1/audio/transcriptions`. Populated at server startup from `--asr-model`.
    pub asr_models: Arc<RwLock<HashMap<String, Arc<AsrBundle>>>>,
    /// Standalone TTS bundles keyed by model name, used by `/v1/audio/speech`.
    /// Populated at server startup from `--tts-model`.
    pub tts_models: Arc<RwLock<HashMap<String, Arc<TtsBundle>>>>,
    /// Directory searched for TTS voice files when the request specifies a
    /// bare voice ID (e.g. `af_alloy`). Resolved once at startup in this
    /// priority order: `--voice-dir` CLI flag, `$BLAZR_VOICE_DIR`, bundled
    /// `assets/kokoro_voices/` shipped with the binary.
    pub voice_dir: Option<std::path::PathBuf>,
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
            asr_models: Arc::new(RwLock::new(HashMap::new())),
            tts_models: Arc::new(RwLock::new(HashMap::new())),
            voice_dir: None,
        }
    }

    /// Attach a voice directory (see [`AppState::voice_dir`]). Fluent builder
    /// for server startup.
    pub fn with_voice_dir(mut self, dir: Option<std::path::PathBuf>) -> Self {
        self.voice_dir = dir;
        self
    }

    /// Register a vision embedder under `model_name`.
    pub async fn register_vision_embedder(
        &self,
        model_name: String,
        embedder: Arc<VisionEmbedder>,
    ) {
        self.vision_embedders
            .write()
            .await
            .insert(model_name, embedder);
    }

    /// Look up a vision embedder by model name.
    pub async fn vision_embedder(&self, model_name: &str) -> Option<Arc<VisionEmbedder>> {
        self.vision_embedders.read().await.get(model_name).cloned()
    }

    /// Register a Whisper ASR bundle under `model_name`.
    pub async fn register_asr_model(&self, model_name: String, bundle: Arc<AsrBundle>) {
        self.asr_models.write().await.insert(model_name, bundle);
    }

    /// Look up a Whisper ASR bundle by model name.
    pub async fn asr_model(&self, model_name: &str) -> Option<Arc<AsrBundle>> {
        self.asr_models.read().await.get(model_name).cloned()
    }

    /// Register a TTS bundle under `model_name`.
    pub async fn register_tts_model(&self, model_name: String, bundle: Arc<TtsBundle>) {
        self.tts_models.write().await.insert(model_name, bundle);
    }

    /// Look up a TTS bundle by model name.
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

/// Health check endpoint — returns status and loaded model info
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let loaded = state
        .scheduler
        .list_loaded()
        .await
        .into_iter()
        .map(|m| m.name)
        .collect::<Vec<_>>();

    let gpu_memory = get_gpu_memory_info();

    let response = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        loaded_models: loaded,
        gpu_memory,
    };
    (StatusCode::OK, Json(response)).into_response()
}

/// Query GPU memory info (CUDA only)
fn get_gpu_memory_info() -> Option<GpuMemoryInfo> {
    #[cfg(feature = "cuda")]
    {
        use boostr::Runtime;
        let device = <boostr::CudaRuntime as Runtime>::default_device();
        if let Ok((free, total)) = device.memory_info() {
            let used = total.saturating_sub(free);
            return Some(GpuMemoryInfo {
                used_bytes: used,
                free_bytes: free,
                total_bytes: total,
                used_gb: used as f64 / (1024.0 * 1024.0 * 1024.0),
                total_gb: total as f64 / (1024.0 * 1024.0 * 1024.0),
            });
        }
    }
    None
}

/// List available models
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.scheduler.list_available() {
        Ok(models) => {
            let response = ModelsResponse {
                object: "list".to_string(),
                data: models
                    .iter()
                    .map(|m| ModelInfo {
                        id: m.name.clone(),
                        object: "model".to_string(),
                        created: 0,
                        owned_by: "local".to_string(),
                    })
                    .collect(),
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Get single model details
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Response {
    match state.scheduler.list_available() {
        Ok(models) => {
            if let Some(m) = models.iter().find(|m| m.name == model_id) {
                let info = ModelInfo {
                    id: m.name.clone(),
                    object: "model".to_string(),
                    created: 0,
                    owned_by: "local".to_string(),
                };
                (StatusCode::OK, Json(info)).into_response()
            } else {
                error_response(
                    StatusCode::NOT_FOUND,
                    &format!("Model '{}' not found", model_id),
                    "invalid_request_error",
                )
            }
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &e.to_string(),
            "server_error",
        ),
    }
}

/// Tokenize text endpoint
pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenizeRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "invalid_request_error",
            );
        }
    };

    // Untrusted text with no chat template around it: nothing in it becomes a
    // control token unless the caller explicitly asks for that token by name.
    let tokens = match encode_with_allowed_special(
        executor.tokenizer(),
        &request.content,
        request.allowed_special.as_deref(),
    ) {
        Ok(tokens) => tokens,
        Err(e) => return policy_error_response(&e),
    };
    let response = TokenizeResponse {
        tokens: tokens.iter().map(|&t| t as i64).collect(),
    };
    (StatusCode::OK, Json(response)).into_response()
}

/// Detokenize token IDs endpoint
pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DetokenizeRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "invalid_request_error",
            );
        }
    };

    let token_ids: Vec<u32> = request.tokens.iter().map(|&t| t as u32).collect();
    match executor.tokenizer().decode(&token_ids) {
        Ok(text) => {
            let response = DetokenizeResponse { content: text };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Detokenization failed: {}", e),
            "server_error",
        ),
    }
}

// ── Types ──

#[derive(Serialize)]
pub struct GpuMemoryInfo {
    pub used_bytes: u64,
    pub free_bytes: u64,
    pub total_bytes: u64,
    pub used_gb: f64,
    pub total_gb: f64,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub loaded_models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory: Option<GpuMemoryInfo>,
}

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Deserialize)]
pub struct TokenizeRequest {
    pub model: String,
    pub content: String,
    /// Special tokens the caller permits `content` to spell out, tiktoken-style.
    ///
    /// Absent (the default) means none: the text is encoded as ordinary
    /// content, so a `content` of `"<|im_start|>"` yields the tokens of that
    /// literal string rather than the real control-token id. Naming a token
    /// here opts that one token back in; any *other* special token in the text
    /// is then refused rather than silently promoted.
    #[serde(default)]
    pub allowed_special: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<i64>,
}

#[derive(Deserialize)]
pub struct DetokenizeRequest {
    pub model: String,
    pub tokens: Vec<i64>,
}

#[derive(Serialize)]
pub struct DetokenizeResponse {
    pub content: String,
}

// ── Apply template ──

/// Apply chat template to messages without running inference (like llama.cpp's /apply-template)
pub async fn apply_template(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ApplyTemplateRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "invalid_request_error",
            );
        }
    };

    let msgs: Vec<crate::model::chat_template::ChatMessage> = request
        .messages
        .iter()
        .map(request_msg_to_chat_msg)
        .collect();
    let template = if let Some(ref tpl_name) = request.template {
        crate::model::chat_template::ChatTemplate::from_name(tpl_name)
    } else {
        executor.chat_template().clone()
    };
    let prompt = template.apply(&msgs);
    let response = ApplyTemplateResponse { prompt };
    (StatusCode::OK, Json(response)).into_response()
}

#[derive(Deserialize)]
pub struct ApplyTemplateRequest {
    pub model: String,
    pub messages: Vec<ChatRequestMessage>,
    #[serde(default)]
    pub template: Option<String>,
}

#[derive(Serialize)]
pub struct ApplyTemplateResponse {
    pub prompt: String,
}

// ── Slot management ──

/// Create a new inference slot
pub async fn create_slot(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateSlotRequest>,
) -> Response {
    match state.slot_manager.allocate(&request.model).await {
        Ok(id) => {
            let response = SlotResponse {
                id,
                model: request.model,
                status: "active".to_string(),
            };
            (StatusCode::CREATED, Json(response)).into_response()
        }
        Err(e) => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &e,
            "slot_allocation_failed",
        ),
    }
}

/// List all active inference slots
pub async fn list_slots(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let slots = state.slot_manager.list().await;
    let response: Vec<SlotListEntry> = slots
        .into_iter()
        .map(|s| SlotListEntry {
            id: s.id,
            model: s.model,
            total_tokens: s.total_tokens,
            idle_seconds: s.last_accessed.elapsed().as_secs(),
        })
        .collect();
    (StatusCode::OK, Json(response)).into_response()
}

/// Free an inference slot
pub async fn delete_slot(
    State(state): State<Arc<AppState>>,
    Path(slot_id): Path<String>,
) -> Response {
    if state.slot_manager.free(&slot_id).await {
        StatusCode::NO_CONTENT.into_response()
    } else {
        error_response(
            StatusCode::NOT_FOUND,
            &format!("Slot '{}' not found", slot_id),
            "slot_not_found",
        )
    }
}

#[derive(Deserialize)]
pub struct CreateSlotRequest {
    pub model: String,
}

#[derive(Serialize)]
pub struct SlotResponse {
    pub id: String,
    pub model: String,
    pub status: String,
}

#[derive(Serialize)]
pub struct SlotListEntry {
    pub id: String,
    pub model: String,
    pub total_tokens: usize,
    pub idle_seconds: u64,
}
