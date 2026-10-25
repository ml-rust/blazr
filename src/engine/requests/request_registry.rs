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
use std::sync::{Mutex, Weak};

use anyhow::{anyhow, Result};
use tokio::sync::{mpsc, oneshot};

use boostr::inference::memory::BlockAllocatorStats;
use boostr::inference::scheduler::SequenceId;

use crate::config::GenerationConfig;
use crate::engine::types::{FinishReason, GeneratedToken};
use crate::engine::{StopCut, StopMatcher, WorkHold};
use crate::tokenizer::{AnyTokenizer, TokenTextStream};

/// Tokens a request buffers before the driver waits on its reader.
const TOKEN_CHANNEL_CAPACITY: usize = 32;

/// Handle returned to callers when submitting a request.
///
/// Dropping the handle cancels its request through the scheduler that
/// queued it. A request that already finished is not touched.
pub struct RequestHandle {
    /// Sequence ID for tracking
    pub seq_id: SequenceId,
    /// Receive generated tokens
    pub token_rx: mpsc::Receiver<Result<GeneratedToken>>,
    /// Notified when generation is complete
    pub done_rx: oneshot::Receiver<()>,
    /// The scheduler `Drop` cancels through. `None` when no scheduler
    /// carries the request.
    canceller: Option<Weak<dyn CancelRequest>>,
}

impl RequestHandle {
    /// Cancel the request through `canceller` when this handle drops.
    pub fn cancel_on_drop(mut self, canceller: Weak<dyn CancelRequest>) -> Self {
        self.canceller = Some(canceller);
        self
    }
}

impl Drop for RequestHandle {
    /// A dropped handle means the caller stopped listening, so the request
    /// stops producing. The cancel never awaits, and it holds each lock
    /// only for one short update.
    fn drop(&mut self) {
        let Some(canceller) = self.canceller.take().and_then(|weak| weak.upgrade()) else {
            return;
        };
        canceller.cancel_request(self.seq_id);
    }
}

/// A scheduler a dropped [`RequestHandle`] cancels through.
pub trait CancelRequest: Send + Sync {
    /// Stop `seq_id` and free what it holds.
    ///
    /// `Drop` calls this, so it must not block on an await or on a lock
    /// another task holds for long. A second call, or a call for a
    /// finished request, must be a no-op.
    fn cancel_request(&self, seq_id: SequenceId);
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

    /// Requests this scheduler holds that wait for a slot or for
    /// token-budget admission.
    fn queue_depth(&self) -> usize {
        0
    }

    /// Why this scheduler cannot run `gen_config` as asked, or `None`.
    ///
    /// A handler answers a refusal with 400 before it submits, so a request
    /// never runs with a field silently dropped. `submit` refuses the same
    /// configs.
    fn refusal(&self, _gen_config: &GenerationConfig) -> Option<String> {
        None
    }

    /// Why this scheduler can no longer serve any request, or `None` while
    /// it can. `GET /health` answers 503 with the reason while one is set.
    fn fault(&self) -> Option<String> {
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
    /// The request's text decoder, built at its first token.
    text: Option<TokenTextStream>,
    /// Cuts the text at the request's first stop sequence.
    stop: StopMatcher,
    /// The submitter's hold. It drops with this state, once the scheduler
    /// has freed what the request held.
    _hold: Option<WorkHold>,
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
    /// The state takes the caller's [`WorkHold`], so a guard behind it
    /// stays held until the request leaves the scheduler, even when the
    /// handle drops first.
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
            stop: StopMatcher::new(&gen_config.stop_sequences),
            gen_config,
            token_tx,
            done_tx: Some(done_tx),
            prompt_tokens: prompt_tokens.clone(),
            token_history: prompt_tokens,
            text: None,
            _hold: WorkHold::current(),
        };
        self.lock_meta()?.insert(seq_id, meta);
        Ok(RequestHandle {
            seq_id,
            token_rx,
            done_rx,
            canceller: None,
        })
    }

    /// Drop `seq_id`'s state without signalling its caller. Returns
    /// whether the request was still carried.
    ///
    /// # Errors
    ///
    /// An error when the registry lock is poisoned.
    pub fn remove(&self, seq_id: SequenceId) -> Result<bool> {
        Ok(self.lock_meta()?.remove(&seq_id).is_some())
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

    /// The text `token_id` adds to `seq_id`'s response, cut at its stops.
    ///
    /// The request's own streaming decoder renders the token, primed with
    /// the prompt. `finish` is the token's own finish reason. The request's
    /// stop sequences then apply as [`StopMatcher::cut_token`] applies them:
    /// a match finishes with `Stop`. The caller must end the sequence when
    /// the returned finish is set. Empty, with no finish, when the request
    /// is gone.
    pub fn token_text(
        &self,
        seq_id: SequenceId,
        tokenizer: &AnyTokenizer,
        token_id: u32,
        finish: Option<FinishReason>,
    ) -> StopCut {
        let gone = StopCut {
            text: String::new(),
            finish: None,
        };
        let Ok(mut meta) = self.meta.lock() else {
            return gone;
        };
        let Some(m) = meta.get_mut(&seq_id) else {
            return gone;
        };
        let prompt = &m.prompt_tokens;
        let piece = m
            .text
            .get_or_insert_with(|| TokenTextStream::new(tokenizer, prompt))
            .next_text(
                tokenizer,
                token_id,
                finish == Some(FinishReason::Eos),
                finish.is_some(),
            );
        m.stop.cut_token(&piece, finish)
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
        let mut handle = registry.insert(id, vec![1], config()).expect("insert");
        registry.finish(id).expect("finish");
        assert!(!registry.contains(id));
        assert!((&mut handle.done_rx).await.is_ok());
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

    /// llama3 ids 0 to 3 are the one-byte pieces `!`, `"`, `#` and `$`.
    #[test]
    fn token_text_holds_a_stop_prefix_and_ends_on_the_match() -> Result<()> {
        let tokenizer = crate::tokenizer::from_pretrained("llama3")?;
        let registry = RequestRegistry::new();
        let stops = |stops: &[&str]| GenerationConfig {
            stop_sequences: stops.iter().map(|s| s.to_string()).collect(),
            ..config()
        };
        let cut = |text: &str, finish| StopCut {
            text: text.to_string(),
            finish,
        };

        let id = registry.next_id();
        let _handle = registry.insert(id, vec![5], stops(&["\"#"]))?;
        assert_eq!(registry.token_text(id, &tokenizer, 0, None), cut("!", None));
        assert_eq!(registry.token_text(id, &tokenizer, 1, None), cut("", None));
        assert_eq!(
            registry.token_text(id, &tokenizer, 2, None),
            cut("", Some(FinishReason::Stop))
        );

        let id = registry.next_id();
        let _handle = registry.insert(id, vec![5], stops(&["!x"]))?;
        assert_eq!(registry.token_text(id, &tokenizer, 0, None), cut("", None));
        assert_eq!(
            registry.token_text(id, &tokenizer, 3, Some(FinishReason::Length)),
            cut("!$", Some(FinishReason::Length))
        );
        assert_eq!(registry.token_text(99, &tokenizer, 0, None), cut("", None));
        Ok(())
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
