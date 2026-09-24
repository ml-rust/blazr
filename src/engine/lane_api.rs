//! Executor entry points for the `qwen35` lane session.
//!
//! Every call takes the session lock, does one admit, one step or one
//! release, and drops the lock before returning. No caller holds it across a
//! request's token stream, so two requests interleave their steps on the same
//! session instead of queueing behind each other.

#![cfg(feature = "cuda")]

use anyhow::{anyhow, Result};

use boostr::inference::lane_decode::LaneDecodeMode;
use boostr::CudaRuntime;

use crate::config::GenerationConfig;

use super::executor::Executor;
use super::lane_session::{LaneAdmitted, LaneToken, Qwen35LaneSession};

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
