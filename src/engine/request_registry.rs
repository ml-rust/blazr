//! The shape a scheduler presents to the HTTP handlers, and the per-request
//! bookkeeping every scheduler keeps behind it.
//!
//! [`GenerationScheduler`] is the one way a handler reaches whichever
//! scheduler the server was started with. [`RequestRegistry`] owns the
//! sequence ids, the generation config, the token history and the channels a
//! request streams through, so a scheduler only has to own its own admission
//! policy.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use tokio::sync::{mpsc, oneshot};

use boostr::inference::memory::BlockAllocatorStats;
use boostr::inference::scheduler::SequenceId;

use super::types::GeneratedToken;
use crate::config::GenerationConfig;

/// Tokens a request buffers before the driver waits on its reader.
const TOKEN_CHANNEL_CAPACITY: usize = 32;

/// Handle returned to callers when submitting a request.
pub struct RequestHandle {
    /// Sequence ID for tracking
    pub seq_id: SequenceId,
    /// Receive generated tokens
    pub token_rx: mpsc::Receiver<Result<GeneratedToken>>,
    /// Notified when generation is complete
    pub done_rx: oneshot::Receiver<()>,
}

/// The scheduler interface the HTTP handlers use.
///
/// One server holds at most one of these, so a handler routes through it
/// without naming the model architecture behind it.
pub trait GenerationScheduler: Send + Sync {
    /// Queue a request and hand back the channel its tokens arrive on.
    ///
    /// # Errors
    ///
    /// An error when the request cannot be queued.
    fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        gen_config: GenerationConfig,
    ) -> Result<RequestHandle>;

    /// KV block usage for the metrics endpoint, or `None` when this scheduler
    /// holds no block pool.
    fn block_stats(&self) -> Option<BlockAllocatorStats> {
        None
    }
}

/// Per-request state stored beside whichever scheduler queued it.
struct RequestMeta {
    gen_config: GenerationConfig,
    token_tx: mpsc::Sender<Result<GeneratedToken>>,
    done_tx: Option<oneshot::Sender<()>>,
    prompt_tokens: Vec<u32>,
    token_history: Vec<u32>,
}

/// The requests a scheduler is carrying, keyed by sequence id.
pub struct RequestRegistry {
    meta: Mutex<HashMap<SequenceId, RequestMeta>>,
    next_seq_id: AtomicU64,
}

impl Default for RequestRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestRegistry {
    /// An empty registry whose first sequence id is 1.
    pub fn new() -> Self {
        Self {
            meta: Mutex::new(HashMap::new()),
            next_seq_id: AtomicU64::new(1),
        }
    }

    /// Reserve the next sequence id without storing anything under it.
    pub fn next_id(&self) -> SequenceId {
        self.next_seq_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Store `seq_id`'s state and hand back the caller's handle.
    ///
    /// # Errors
    ///
    /// An error when the registry lock is poisoned.
    pub fn insert(
        &self,
        seq_id: SequenceId,
        prompt_tokens: Vec<u32>,
        gen_config: GenerationConfig,
    ) -> Result<RequestHandle> {
        let (token_tx, token_rx) = mpsc::channel(TOKEN_CHANNEL_CAPACITY);
        let (done_tx, done_rx) = oneshot::channel();
        let meta = RequestMeta {
            gen_config,
            token_tx,
            done_tx: Some(done_tx),
            prompt_tokens: prompt_tokens.clone(),
            token_history: prompt_tokens,
        };
        self.lock_meta()?.insert(seq_id, meta);
        Ok(RequestHandle {
            seq_id,
            token_rx,
            done_rx,
        })
    }

    /// Drop `seq_id`'s state without signalling its caller.
    ///
    /// # Errors
    ///
    /// An error when the registry lock is poisoned.
    pub fn remove(&self, seq_id: SequenceId) -> Result<()> {
        self.lock_meta()?.remove(&seq_id);
        Ok(())
    }

    /// Signal `seq_id`'s caller that generation ended, then drop its state.
    ///
    /// # Errors
    ///
    /// An error when the registry lock is poisoned.
    pub fn finish(&self, seq_id: SequenceId) -> Result<()> {
        let removed = self.lock_meta()?.remove(&seq_id);
        if let Some(mut meta) = removed {
            if let Some(done_tx) = meta.done_tx.take() {
                let _ = done_tx.send(());
            }
        }
        Ok(())
    }

    /// Whether `seq_id` is still carried here.
    pub fn contains(&self, seq_id: SequenceId) -> bool {
        self.meta
            .lock()
            .map(|m| m.contains_key(&seq_id))
            .unwrap_or(false)
    }

    /// The generation config `seq_id` was submitted with.
    pub fn gen_config(&self, seq_id: SequenceId) -> Option<GenerationConfig> {
        let meta = self.meta.lock().ok()?;
        meta.get(&seq_id).map(|m| m.gen_config.clone())
    }

    /// The channel `seq_id`'s tokens stream out on.
    pub fn token_sender(&self, seq_id: SequenceId) -> Option<mpsc::Sender<Result<GeneratedToken>>> {
        let meta = self.meta.lock().ok()?;
        meta.get(&seq_id).map(|m| m.token_tx.clone())
    }

    /// The prompt ids `seq_id` was submitted with.
    pub fn prompt_tokens(&self, seq_id: SequenceId) -> Option<Vec<u32>> {
        let meta = self.meta.lock().ok()?;
        meta.get(&seq_id).map(|m| m.prompt_tokens.clone())
    }

    /// `seq_id`'s prompt ids followed by every id it has produced.
    pub fn token_history(&self, seq_id: SequenceId) -> Option<Vec<u32>> {
        let meta = self.meta.lock().ok()?;
        meta.get(&seq_id).map(|m| m.token_history.clone())
    }

    /// Record `token` as `seq_id`'s newest id.
    pub fn push_history(&self, seq_id: SequenceId, token: u32) {
        if let Ok(mut meta) = self.meta.lock() {
            if let Some(m) = meta.get_mut(&seq_id) {
                m.token_history.push(token);
            }
        }
    }

    /// Send `token` to `seq_id`'s caller.
    ///
    /// Returns `false` when the request is gone or its receiver was dropped,
    /// which is the signal to stop producing for it.
    pub async fn send_token(&self, seq_id: SequenceId, token: GeneratedToken) -> bool {
        let Some(tx) = self.token_sender(seq_id) else {
            return false;
        };
        tx.send(Ok(token)).await.is_ok()
    }

    /// Report `error` to `seq_id`'s caller, if it is still listening.
    pub async fn send_error(&self, seq_id: SequenceId, error: anyhow::Error) {
        if let Some(tx) = self.token_sender(seq_id) {
            let _ = tx.send(Err(error)).await;
        }
    }

    /// The metadata map, or an error naming the poisoned lock.
    fn lock_meta(&self) -> Result<std::sync::MutexGuard<'_, HashMap<SequenceId, RequestMeta>>> {
        self.meta
            .lock()
            .map_err(|e| anyhow!("request registry lock is poisoned: {e}; restart the server"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GenerationConfig {
        GenerationConfig {
            max_tokens: 4,
            ..GenerationConfig::default()
        }
    }

    #[test]
    fn ids_are_unique_and_start_at_one() {
        let registry = RequestRegistry::new();
        assert_eq!(registry.next_id(), 1);
        assert_eq!(registry.next_id(), 2);
    }

    #[test]
    fn insert_stores_prompt_config_and_history() {
        let registry = RequestRegistry::new();
        let id = registry.next_id();
        let handle = registry
            .insert(id, vec![7, 8, 9], config())
            .expect("insert");
        assert_eq!(handle.seq_id, id);
        assert!(registry.contains(id));
        assert_eq!(registry.prompt_tokens(id), Some(vec![7, 8, 9]));
        assert_eq!(registry.token_history(id), Some(vec![7, 8, 9]));
        assert_eq!(registry.gen_config(id).map(|c| c.max_tokens), Some(4));

        registry.push_history(id, 10);
        assert_eq!(registry.token_history(id), Some(vec![7, 8, 9, 10]));
        assert_eq!(registry.prompt_tokens(id), Some(vec![7, 8, 9]));
    }

    #[tokio::test]
    async fn finish_signals_the_caller_and_drops_the_state() {
        let registry = RequestRegistry::new();
        let id = registry.next_id();
        let handle = registry.insert(id, vec![1], config()).expect("insert");
        registry.finish(id).expect("finish");
        assert!(!registry.contains(id));
        assert!(handle.done_rx.await.is_ok());
        assert!(registry.token_sender(id).is_none());
    }

    #[tokio::test]
    async fn send_token_fails_once_the_receiver_is_dropped() {
        let registry = RequestRegistry::new();
        let id = registry.next_id();
        let handle = registry.insert(id, vec![1], config()).expect("insert");
        let token = GeneratedToken {
            token_id: 5,
            text: "a".to_string(),
            logprob: None,
            top_logprobs: None,
            finish_reason: None,
        };
        assert!(registry.send_token(id, token.clone()).await);
        drop(handle);
        assert!(!registry.send_token(id, token).await);
    }

    #[tokio::test]
    async fn send_token_fails_for_an_unknown_request() {
        let registry = RequestRegistry::new();
        let token = GeneratedToken {
            token_id: 5,
            text: String::new(),
            logprob: None,
            top_logprobs: None,
            finish_reason: None,
        };
        assert!(!registry.send_token(42, token).await);
    }
}
