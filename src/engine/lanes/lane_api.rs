//! Executor entry points for the `qwen35` lane session.
//!
//! Every call takes the session lock, does one admit, one prefill chunk, one
//! step or one release, and drops the lock before returning. No caller holds
//! it across a request's token stream or across a whole chunked prefill, so
//! two requests interleave their steps on the same session instead of
//! queueing behind each other.

#![cfg(feature = "cuda")]

use anyhow::{anyhow, Result};

use boostr::inference::lane_decode::LaneDecodeMode;
use boostr::CudaRuntime;

use crate::config::GenerationConfig;

use super::lane_session::{LaneAdmitted, LaneToken, Qwen35LaneSession};
use crate::engine::executor::Executor;

impl Executor<CudaRuntime> {
    /// Tokens one lane holds at most: the context this executor was opened
    /// with, capped by the model's own maximum.
    pub fn qwen35_lane_capacity(&self) -> usize {
        self.num_ctx().min(self.config().max_seq_len())
    }

    /// Open a session of `lanes` lanes stepping `mode`, replacing any session
    /// of a different shape.
    ///
    /// A session that already holds `lanes` lanes at this capacity in this
    /// mode is kept, so reopening it admits nothing and captures nothing.
    ///
    /// # Errors
    ///
    /// An error when the model is not `qwen35`, or when the allocation,
    /// warmup or capture fails.
    pub async fn qwen35_lane_open(&self, lanes: usize, mode: LaneDecodeMode) -> Result<()> {
        let capacity = self.qwen35_lane_capacity();
        let mut guard = self.qwen35_lane_session.lock().await;
        let keep = guard
            .as_ref()
            .is_some_and(|s| s.batch() == lanes && s.capacity() == capacity && s.mode() == mode);
        if keep {
            return Ok(());
        }
        // Drop the old session first so its caches are freed before the new
        // ones are allocated.
        *guard = None;
        *guard = Some(Qwen35LaneSession::open(self, lanes, capacity, mode)?);
        Ok(())
    }

    /// Drop the session and everything it allocated.
    pub async fn qwen35_lane_close(&self) {
        *self.qwen35_lane_session.lock().await = None;
    }

    /// Prefill `prompt_tokens` into a free lane and return that lane with the
    /// token its prefill produced.
    ///
    /// # Errors
    ///
    /// An error when no session is open, when every lane is taken, or when
    /// the prefill or the install fails.
    pub async fn qwen35_lane_admit(
        &self,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<LaneAdmitted> {
        let mut guard = self.qwen35_lane_session.lock().await;
        let session = guard.as_mut().ok_or_else(Self::no_lane_session)?;
        session.admit(self, prompt_tokens, gen_config)
    }

    /// Start a chunked prefill of `prompt_tokens` under `gen_config`, feeding
    /// no token yet.
    ///
    /// Drive it with [`qwen35_lane_prefill_chunk`](Self::qwen35_lane_prefill_chunk)
    /// until that hands back the admitted lane, and step the occupied lanes
    /// between the chunks.
    ///
    /// # Errors
    ///
    /// An error when no session is open, when a chunked prefill is already
    /// running, when every lane is taken, or when the prompt does not fit a
    /// lane.
    pub async fn qwen35_lane_admit_begin(
        &self,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<()> {
        let mut guard = self.qwen35_lane_session.lock().await;
        let session = guard.as_mut().ok_or_else(Self::no_lane_session)?;
        session.begin_admit(self, prompt_tokens, gen_config)
    }

    /// Feed the next `chunk_size` prompt tokens of the running chunked
    /// prefill: `None` while tokens remain, the admitted lane once the last
    /// chunk is installed.
    ///
    /// # Errors
    ///
    /// An error when no session is open, when no chunked prefill is running,
    /// when `chunk_size` is 0, or when the forward or the install fails.
    pub async fn qwen35_lane_prefill_chunk(
        &self,
        chunk_size: usize,
    ) -> Result<Option<LaneAdmitted>> {
        let mut guard = self.qwen35_lane_session.lock().await;
        let session = guard.as_mut().ok_or_else(Self::no_lane_session)?;
        session.prefill_chunk(self, chunk_size)
    }

    /// Drop a running chunked prefill and everything it allocated.
    pub async fn qwen35_lane_admit_cancel(&self) {
        let mut guard = self.qwen35_lane_session.lock().await;
        if let Some(session) = guard.as_mut() {
            session.cancel_admit();
        }
    }

    /// One decode step for every occupied lane, one [`LaneToken`] per lane.
    ///
    /// # Errors
    ///
    /// An error when no session is open, or when the step fails.
    pub async fn qwen35_lane_step(&self) -> Result<Vec<LaneToken>> {
        let mut guard = self.qwen35_lane_session.lock().await;
        let session = guard.as_mut().ok_or_else(Self::no_lane_session)?;
        session.step(self)
    }

    /// Give lane `lane` back for the next sequence.
    ///
    /// # Errors
    ///
    /// An error when no session is open, when `lane` is outside the batch, or
    /// when the clear fails.
    pub async fn qwen35_lane_release(&self, lane: usize) -> Result<()> {
        let mut guard = self.qwen35_lane_session.lock().await;
        let session = guard.as_mut().ok_or_else(Self::no_lane_session)?;
        session.release(self, lane)
    }

    /// How many lanes are free for a new sequence.
    ///
    /// # Errors
    ///
    /// An error when no session is open.
    pub async fn qwen35_lane_free(&self) -> Result<usize> {
        let guard = self.qwen35_lane_session.lock().await;
        let session = guard.as_ref().ok_or_else(Self::no_lane_session)?;
        Ok(session.free_lanes())
    }

    /// Which lanes hold a sequence.
    ///
    /// # Errors
    ///
    /// An error when no session is open.
    pub async fn qwen35_lane_occupied(&self) -> Result<Vec<usize>> {
        let guard = self.qwen35_lane_session.lock().await;
        let session = guard.as_ref().ok_or_else(Self::no_lane_session)?;
        Ok(session.occupied_lanes())
    }

    /// The error a lane call returns before a session is opened.
    fn no_lane_session() -> anyhow::Error {
        anyhow!(
            "no qwen35 lane session is open on this executor; call \
             qwen35_lane_open(lanes, mode) once before admitting or stepping"
        )
    }
}
