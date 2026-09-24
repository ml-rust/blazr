//! Admitting a prompt into a `qwen35` lane, in one prefill or in chunks.
//!
//! Both routes run the single-sequence prefill into a scratch one-lane KV
//! cache and GDN state, then install that state at position 0 of a free lane.
//! [`Qwen35LaneSession::admit`] feeds the whole prompt in one call.
//! [`Qwen35LaneSession::begin_admit`] followed by
//! [`Qwen35LaneSession::prefill_chunk`] feeds it a slice at a time, so the
//! caller steps the lanes that are already decoding between the slices.
//!
//! A chunk enters the prefill at rope position `fed`, the number of prompt
//! tokens already in the scratch cache, which for a text-only prompt is also
//! that cache's length. The attention layers offset their causal mask by the
//! keys already cached and each GDN layer carries its conv window and
//! delta-rule state forward, so a prompt fed in chunks leaves the same state
//! as the same prompt fed at once and the lane decodes the same ids either
//! way.
//!
//! One chunked prefill runs at a time per session. A failed chunk drops it, so
//! a failed prefill never leaves the session holding a prompt it cannot
//! finish.

#![cfg(feature = "cuda")]

use anyhow::{anyhow, Result};

use boostr::inference::lane_decode::LaneDecodeSession;
use boostr::inference::{LayeredGdnState, LayeredKvCache};
use boostr::{CudaRuntime, Tensor};

use crate::config::GenerationConfig;

use super::executor::Executor;
use super::lane_session::{LaneAdmitted, LaneRequest, Qwen35LaneSession};

/// A prompt part-way through a chunked prefill.
pub(super) struct PendingPrefill {
    /// The whole prompt; each chunk is a slice of it.
    prompt_tokens: Vec<u32>,
    /// Sampling policy the lane decodes under once the prompt is installed.
    gen_config: GenerationConfig,
    /// Scratch one-lane KV cache the chunks append to.
    kv: LayeredKvCache<CudaRuntime>,
    /// Scratch one-lane GDN state the chunks carry forward.
    gdn: LayeredGdnState<CudaRuntime>,
    /// Prompt tokens already fed, which is the next chunk's rope position.
    fed: usize,
}

impl Qwen35LaneSession {
    /// Prefill `prompt_tokens` in one call and install it into a free lane.
    ///
    /// The prefill logits are sampled under `gen_config`, so the returned
    /// first token is this lane's own.
    ///
    /// # Errors
    ///
    /// An error when `prompt_tokens` is empty, when the prompt does not leave
    /// room for a token inside `capacity`, when every lane is taken, or when
    /// the prefill or the install fails.
    pub fn admit(
        &mut self,
        exec: &Executor<CudaRuntime>,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<LaneAdmitted> {
        self.check_admit(prompt_tokens.len())?;
        let input = exec.create_input_tensor(prompt_tokens)?;
        let (mut kv, mut gdn) = exec.qwen35_caches(prompt_tokens.len(), self.capacity())?;
        let logits = exec.qwen35_prefill(&input, None, &mut kv, &mut gdn)?;
        self.install_prefilled(exec, prompt_tokens, gen_config, kv, gdn, &logits)
    }

    /// Start a chunked prefill of `prompt_tokens` under `gen_config`.
    ///
    /// Allocates the scratch caches the chunks accumulate into and feeds no
    /// token. Drive it with [`prefill_chunk`](Self::prefill_chunk) until that
    /// returns the admitted lane.
    ///
    /// # Errors
    ///
    /// An error when a chunked prefill is already running on this session, or
    /// on any condition [`admit`](Self::admit) rejects a prompt for.
    pub fn begin_admit(
        &mut self,
        exec: &Executor<CudaRuntime>,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<()> {
        if let Some(pending) = self.pending.as_ref() {
            return Err(anyhow!(
                "a chunked prefill of {} prompt tokens is already running on this qwen35 lane \
                 session, {} of them fed; drive it to its end with prefill_chunk before \
                 beginning another",
                pending.prompt_tokens.len(),
                pending.fed
            ));
        }
        self.check_admit(prompt_tokens.len())?;
        let (kv, gdn) = exec.qwen35_caches(prompt_tokens.len(), self.capacity())?;
        self.pending = Some(PendingPrefill {
            prompt_tokens: prompt_tokens.to_vec(),
            gen_config: gen_config.clone(),
            kv,
            gdn,
            fed: 0,
        });
        Ok(())
    }

    /// Feed the next `chunk_size` prompt tokens of the running chunked
    /// prefill.
    ///
    /// Returns `None` while prompt tokens remain, and the admitted lane once
    /// the last chunk is fed and installed. A `chunk_size` past the remaining
    /// tokens feeds what is left.
    ///
    /// # Errors
    ///
    /// An error when `chunk_size` is 0, when no chunked prefill is running, or
    /// when the forward or the install fails. A failed forward or install
    /// drops the prefill; a `chunk_size` of 0 leaves it where it is.
    pub fn prefill_chunk(
        &mut self,
        exec: &Executor<CudaRuntime>,
        chunk_size: usize,
    ) -> Result<Option<LaneAdmitted>> {
        if chunk_size == 0 {
            return Err(anyhow!(
                "a chunked prefill needs a chunk of at least one token, got chunk_size=0; set \
                 inference.prefill_chunk_size above 0, or admit the prompt through admit()"
            ));
        }
        let Some(mut pending) = self.pending.take() else {
            return Err(anyhow!(
                "no chunked prefill is running on this qwen35 lane session; call \
                 begin_admit(prompt_tokens, gen_config) before prefill_chunk({chunk_size})"
            ));
        };

        let total = pending.prompt_tokens.len();
        let end = (pending.fed + chunk_size).min(total);
        let chunk = &pending.prompt_tokens[pending.fed..end];
        let input = exec.create_input_tensor(chunk)?;
        let logits =
            exec.qwen35_prefill_at(&input, pending.fed, &mut pending.kv, &mut pending.gdn)?;
        pending.fed = end;

        if pending.fed < total {
            self.pending = Some(pending);
            return Ok(None);
        }

        let PendingPrefill {
            prompt_tokens,
            gen_config,
            kv,
            gdn,
            ..
        } = pending;
        self.install_prefilled(exec, &prompt_tokens, &gen_config, kv, gdn, &logits)
            .map(Some)
    }

    /// Drop a running chunked prefill and everything it allocated.
    pub fn cancel_admit(&mut self) {
        self.pending = None;
    }

    /// Reject a prompt this session cannot admit.
    fn check_admit(&self, prompt_len: usize) -> Result<()> {
        let capacity = self.capacity();
        if prompt_len == 0 {
            return Err(anyhow!(
                "a qwen35 lane takes at least one prompt token, got 0; encode a non-empty \
                 prompt before admitting it"
            ));
        }
        if prompt_len >= capacity {
            return Err(anyhow!(
                "prompt of {prompt_len} tokens leaves no room inside the lane capacity of \
                 {capacity}; shorten the prompt or open the session with a larger context"
            ));
        }
        if self.free_lanes() == 0 {
            return Err(anyhow!(
                "every one of the {} qwen35 lanes is taken; release a finished sequence's lane \
                 before admitting another",
                self.batch()
            ));
        }
        Ok(())
    }

    /// Install a finished prefill into a free lane and sample its first
    /// token.
    ///
    /// `kv` and `gdn` are the scratch one-lane caches the prefill wrote and
    /// are dropped once their contents are copied into the lane.
    /// `prefill_logits` is the last chunk's logits; the sampler reads their
    /// last position.
    fn install_prefilled(
        &mut self,
        exec: &Executor<CudaRuntime>,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
        kv: LayeredKvCache<CudaRuntime>,
        gdn: LayeredGdnState<CudaRuntime>,
        prefill_logits: &Tensor<CudaRuntime>,
    ) -> Result<LaneAdmitted> {
        let client = LaneDecodeSession::client(exec.device());
        let rope_pos = Executor::<CudaRuntime>::qwen35_first_rope_pos(None, &kv);
        let lane = self
            .decode
            .install(&client, &kv, &gdn, rope_pos)
            .map_err(|e| anyhow!("installing a prefilled sequence into a qwen35 lane: {e}"))?;
        drop(kv);
        drop(gdn);

        let mut request = LaneRequest {
            gen_config: gen_config.clone(),
            sampler: exec.sampler_state(prompt_tokens, gen_config)?,
            generated: 0,
        };
        let scratch = self.decode.scratch(lane)?;
        let first_token = Self::sample_lane(exec, prefill_logits, scratch, &mut request)?;
        let finish_reason = exec.decode_finish_reason(first_token, 0, gen_config.max_tokens);

        self.decode.set_next_token(lane, first_token)?;
        let lanes = self.requests.len();
        let Some(slot) = self.requests.get_mut(lane) else {
            return Err(anyhow!(
                "the install returned lane {lane}, outside the {lanes} lanes this qwen35 session \
                 holds; reopen the session before admitting again"
            ));
        };
        *slot = Some(request);
        self.decode.commit_next_tokens(&client)?;

        Ok(LaneAdmitted {
            lane,
            first_token,
            finish_reason,
        })
    }
}
