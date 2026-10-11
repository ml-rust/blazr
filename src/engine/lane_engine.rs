//! Driver behind [`LaneScheduler`]: one task that fills the `qwen35` decode
//! lanes and steps them together.
//!
//! One iteration admits every waiting request a free lane can take, runs one
//! [`Executor::qwen35_lane_step`] for all occupied lanes, and forwards each
//! lane's token to that request's channel. A request ends when its lane
//! reports a finish reason or its receiver is gone; either way the lane is
//! released in the same iteration and the next waiting request takes it.
//!
//! With no lane occupied the task waits on the scheduler's wakeup channel, so
//! an idle server launches nothing.
//!
//! `inference.prefill_chunk_size` above 0 feeds an admitted prompt in slices
//! of that many tokens and runs one decode step for the already occupied
//! lanes between the slices, so a long prompt no longer holds back the token
//! streams of the requests already decoding. At 0 the whole prompt goes
//! through one prefill.
//!
//! Logprobs are not produced on this path: a lane's logits are consumed by its
//! own sampler and not kept.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::inference::lane_decode::LaneDecodeMode;
use boostr::inference::scheduler::SequenceId;
use boostr::CudaRuntime;

use crate::config::GenerationConfig;

use super::executor::Executor;
use super::lane_scheduler::LaneScheduler;
use super::lane_session::{LaneAdmitted, LaneToken};
use super::types::{FinishReason, GeneratedToken};

/// Fills and steps the lanes of one `qwen35` lane session.
pub struct LaneEngine {
    executor: Arc<Executor<CudaRuntime>>,
    scheduler: Arc<LaneScheduler>,
    /// The decode path the session steps.
    mode: LaneDecodeMode,
    /// Prompt tokens one prefill chunk feeds; 0 prefills a whole prompt in
    /// one call.
    chunk_size: usize,
    /// Request occupying each lane, `None` when the lane is free.
    active: Vec<Option<SequenceId>>,
    /// How many lanes are occupied.
    live: usize,
}

impl LaneEngine {
    /// Open a session of `scheduler.lanes()` lanes on `executor`.
    ///
    /// The decode path is the captured graph when the model supports graph
    /// mode, and the eager batched forward otherwise.
    ///
    /// # Errors
    ///
    /// An error when the model is not `qwen35`, or when the allocation,
    /// warmup or capture fails.
    pub async fn open(
        executor: Arc<Executor<CudaRuntime>>,
        scheduler: Arc<LaneScheduler>,
    ) -> Result<Self> {
        let lanes = scheduler.lanes();
        let mode = if executor.model().supports_gdn_graph_mode() {
            LaneDecodeMode::Graph
        } else {
            LaneDecodeMode::Eager
        };
        let chunk_size = executor.config().inference.prefill_chunk_size;
        executor.qwen35_lane_open(lanes, mode).await?;
        Ok(Self {
            executor,
            scheduler,
            mode,
            chunk_size,
            active: vec![None; lanes],
            live: 0,
        })
    }

    /// The decode path this engine steps.
    pub fn mode(&self) -> LaneDecodeMode {
        self.mode
    }

    /// Admit, step and forward until the scheduler's wakeup channel closes.
    ///
    /// Spawn this as a background task once per server.
    pub async fn run(mut self) {
        let Some(mut notify_rx) = self.scheduler.take_notify_rx() else {
            tracing::error!(
                "the lane scheduler's wakeup receiver was already taken; only one LaneEngine may \
                 drive one LaneScheduler"
            );
            return;
        };

        tracing::info!(
            lanes = self.active.len(),
            mode = ?self.mode,
            "lane decode loop started"
        );

        loop {
            self.admit_waiting().await;

            if self.live == 0 {
                match notify_rx.recv().await {
                    Some(()) => continue,
                    None => {
                        tracing::info!("lane scheduler closed its wakeup channel, shutting down");
                        break;
                    }
                }
            }

            // Wakeups that arrived while stepping need no action: the next
            // iteration admits from the queue before it steps again.
            while notify_rx.try_recv().is_ok() {}

            match self.executor.qwen35_lane_step().await {
                Ok(tokens) => self.forward_step(tokens).await,
                Err(e) => self.fail_live_lanes(&e).await,
            }
        }

        self.executor.qwen35_lane_close().await;
    }

    /// Put waiting requests into free lanes until one runs out, or until
    /// the SLO policy holds lane filling.
    async fn admit_waiting(&mut self) {
        while self.live < self.active.len() && self.live < self.scheduler.admission_limit(self.live)
        {
            let Some(seq_id) = self.scheduler.next_waiting() else {
                break;
            };
            let registry = self.scheduler.registry();
            let (Some(prompt_tokens), Some(mut gen_config)) =
                (registry.prompt_tokens(seq_id), registry.gen_config(seq_id))
            else {
                // The caller went away between submit and admit.
                continue;
            };

            // A lane holds a fixed number of tokens, so the request decodes at
            // most what its prompt leaves free in that lane. It decodes at
            // least one, so the lane always reaches a finish reason and comes
            // back.
            let room = self
                .executor
                .qwen35_lane_capacity()
                .saturating_sub(prompt_tokens.len());
            gen_config.max_tokens = gen_config.max_tokens.clamp(1, room.max(1));

            let outcome = if self.chunk_size > 0 {
                self.admit_chunked(&prompt_tokens, &gen_config).await
            } else {
                self.executor
                    .qwen35_lane_admit(&prompt_tokens, &gen_config)
                    .await
            };

            match outcome {
                Ok(admitted) => self.attach(seq_id, admitted).await,
                Err(e) => {
                    let reason = anyhow!("admitting a request into a qwen35 lane failed: {e}");
                    self.scheduler.registry().send_error(seq_id, reason).await;
                    let _ = self.scheduler.registry().finish(seq_id);
                }
            }
        }
    }

    /// Prefill one prompt in `chunk_size` slices, running one decode step for
    /// the already occupied lanes between the slices.
    ///
    /// A step between chunks forwards its tokens and releases its finished
    /// lanes exactly as the driver loop's own step does, so the requests
    /// already decoding keep producing tokens for the whole prefill. A failed
    /// step ends the requests in the live lanes and the prefill carries on.
    ///
    /// # Errors
    ///
    /// An error when the prompt does not fit a lane, or when a chunk's
    /// forward or the install fails. The part-way prefill is dropped either
    /// way.
    async fn admit_chunked(
        &mut self,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<LaneAdmitted> {
        self.executor
            .qwen35_lane_admit_begin(prompt_tokens, gen_config)
            .await?;

        loop {
            match self
                .executor
                .qwen35_lane_prefill_chunk(self.chunk_size)
                .await
            {
                Ok(Some(admitted)) => return Ok(admitted),
                Ok(None) => {}
                Err(e) => {
                    self.executor.qwen35_lane_admit_cancel().await;
                    return Err(e);
                }
            }

            if self.live == 0 {
                continue;
            }
            match self.executor.qwen35_lane_step().await {
                Ok(tokens) => self.forward_step(tokens).await,
                Err(e) => self.fail_live_lanes(&e).await,
            }
        }
    }

    /// Record an admitted lane against its request and forward the token the
    /// prefill produced, releasing the lane when that token ends the request.
    async fn attach(&mut self, seq_id: SequenceId, admitted: LaneAdmitted) {
        let lanes = self.active.len();
        if admitted.lane >= lanes {
            let reason = anyhow!(
                "an admit returned lane {}, outside the {lanes} lanes this engine tracks; \
                 restart the lane engine to match the session it drives",
                admitted.lane
            );
            self.scheduler.registry().send_error(seq_id, reason).await;
            let _ = self.scheduler.registry().finish(seq_id);
            return;
        }
        if let Some(slot) = self.active.get_mut(admitted.lane) {
            *slot = Some(seq_id);
        }
        self.live += 1;
        let keep = self
            .deliver(seq_id, admitted.first_token, admitted.finish_reason)
            .await;
        if !keep {
            self.end_lane(admitted.lane).await;
        }
    }

    /// Forward one step's tokens, ending every lane whose request is done.
    async fn forward_step(&mut self, tokens: Vec<LaneToken>) {
        for token in tokens {
            let Some(seq_id) = self.active.get(token.lane).copied().flatten() else {
                continue;
            };
            let keep = self
                .deliver(seq_id, token.token_id, token.finish_reason)
                .await;
            if !keep {
                self.end_lane(token.lane).await;
            }
        }
    }

    /// Send one token to its caller and record it.
    ///
    /// Returns `false` when this request is over: the lane reported a finish
    /// reason, or the caller's receiver is gone.
    async fn deliver(
        &self,
        seq_id: SequenceId,
        token_id: u32,
        finish_reason: Option<FinishReason>,
    ) -> bool {
        let text = if finish_reason == Some(FinishReason::Eos) {
            String::new()
        } else {
            self.executor
                .tokenizer()
                .decode(&[token_id])
                .unwrap_or_default()
        };
        let registry = self.scheduler.registry();
        registry.push_history(seq_id, token_id);
        let delivered = registry
            .send_token(
                seq_id,
                GeneratedToken {
                    token_id,
                    text,
                    logprob: None,
                    top_logprobs: None,
                    finish_reason,
                },
            )
            .await;
        delivered && finish_reason.is_none()
    }

    /// Finish the request in `lane` and give the lane back.
    async fn end_lane(&mut self, lane: usize) {
        if let Some(Some(seq_id)) = self.active.get(lane).copied() {
            if let Err(e) = self.scheduler.registry().finish(seq_id) {
                tracing::error!("releasing request {seq_id} from lane {lane}: {e}");
            }
        }
        if let Some(slot) = self.active.get_mut(lane) {
            if slot.take().is_some() {
                self.live = self.live.saturating_sub(1);
            }
        }
        if let Err(e) = self.executor.qwen35_lane_release(lane).await {
            tracing::error!("releasing qwen35 lane {lane}: {e}");
        }
    }

    /// Report `error` to every request in a lane and clear all of them.
    async fn fail_live_lanes(&mut self, error: &anyhow::Error) {
        tracing::error!("qwen35 lane step failed: {error}");
        for lane in 0..self.active.len() {
            let Some(Some(seq_id)) = self.active.get(lane).copied() else {
                continue;
            };
            let reason = anyhow!("qwen35 lane step failed: {error}");
            self.scheduler.registry().send_error(seq_id, reason).await;
            self.end_lane(lane).await;
        }
    }
}
