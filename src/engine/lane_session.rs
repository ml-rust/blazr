//! Per-request state over a `qwen35` lane decode session.
//!
//! boostr's [`LaneDecodeSession`] owns the machinery: the batched KV cache and
//! GDN state, the lane table, the decode driver and the next-token mirror. This
//! layer owns which request is in which lane and how that lane samples.
//!
//! [`Qwen35LaneSession::admit`] prefills one prompt into a scratch one-lane
//! cache through the single-sequence prefill, then installs it into a free
//! lane. [`Qwen35LaneSession::step`] runs one decode step for every occupied
//! lane, samples each lane's logits row by that lane's own
//! [`GenerationConfig`], and hands back the token each lane produced.
//! [`Qwen35LaneSession::release`] gives a lane back for the next sequence.
//!
//! A lane holds its sequence from cache position 0 with its own end, so a
//! lane produces the tokens it would produce alone, whatever the other lanes
//! hold. Sampling is per lane, so one lane can decode greedily in the same
//! step another lane samples.

#![cfg(feature = "cuda")]

use anyhow::{anyhow, Result};

use boostr::inference::lane_decode::{LaneDecodeMode, LaneDecodeSession, LaneDriverSpec};
use boostr::{CudaRuntime, Tensor};

use crate::config::GenerationConfig;

use super::executor::Executor;
use super::executor_generate_qwen35::SamplerState;
use super::types::FinishReason;

/// One admitted sequence's decode state, held beside its lane.
struct LaneRequest {
    /// Sampling policy this lane decodes under.
    gen_config: GenerationConfig,
    /// Token history for the repetition penalties, mirostat state and the
    /// grammar DFAs, exactly as the single-sequence loop carries them.
    sampler: SamplerState<CudaRuntime>,
    /// Tokens this lane has produced, which is also the next step's index.
    generated: usize,
}

/// A token one lane produced in one step.
pub struct LaneToken {
    /// The lane that produced it.
    pub lane: usize,
    /// The sampled token id.
    pub token_id: u32,
    /// Why this lane stops, when it does.
    pub finish_reason: Option<FinishReason>,
}

/// The lane a sequence took, and the token its prefill produced.
pub struct LaneAdmitted {
    /// The lane the sequence now occupies.
    pub lane: usize,
    /// The token sampled from the prefill logits.
    pub first_token: u32,
    /// Why this lane stops already, when it does.
    pub finish_reason: Option<FinishReason>,
}

/// A fixed set of `qwen35` decode lanes and the requests occupying them.
pub struct Qwen35LaneSession {
    /// The caches, the lane table, the decode path and the token mirror.
    decode: LaneDecodeSession,
    /// Per-lane request state, `None` when the lane is free.
    requests: Vec<Option<LaneRequest>>,
}

impl Qwen35LaneSession {
    /// Allocate a session of `lanes` lanes, each holding at most `capacity`
    /// tokens, stepping `mode`.
    ///
    /// # Errors
    ///
    /// An error when `lanes` or `capacity` is 0, when the model is not
    /// `qwen35`, or when the driver build fails.
    pub fn open(
        exec: &Executor<CudaRuntime>,
        lanes: usize,
        capacity: usize,
        mode: LaneDecodeMode,
    ) -> Result<Self> {
        if lanes == 0 || capacity == 0 {
            return Err(anyhow!(
                "a qwen35 lane session needs at least one lane and one token of capacity, got \
                 lanes={lanes} capacity={capacity}; open it with the number of sequences to \
                 decode at once and the context length"
            ));
        }
        let (kv, gdn) = exec.qwen35_caches_at(lanes, capacity, capacity)?;
        let decode = LaneDecodeSession::open(LaneDriverSpec {
            model: exec.qwen35_backbone()?,
            device: exec.device(),
            kv,
            gdn,
            lanes,
            capacity,
            mode,
        })
        .map_err(|e| anyhow!("opening a {lanes}-lane qwen35 decode session failed: {e}"))?;
        exec.note_graph_captures(decode.graph_captures());
        Ok(Self {
            decode,
            requests: (0..lanes).map(|_| None).collect(),
        })
    }

    /// Lane count one step serves.
    pub fn batch(&self) -> usize {
        self.decode.batch()
    }

    /// Tokens one lane holds at most.
    pub fn capacity(&self) -> usize {
        self.decode.capacity()
    }

    /// Which decode path this session steps.
    pub fn mode(&self) -> LaneDecodeMode {
        self.decode.mode()
    }

    /// How many lanes are free for a new sequence.
    pub fn free_lanes(&self) -> usize {
        self.decode.free_lanes()
    }

    /// Which lanes hold a sequence.
    pub fn occupied_lanes(&self) -> Vec<usize> {
        (0..self.requests.len())
            .filter(|&lane| self.requests[lane].is_some())
            .collect()
    }

    /// Prefill `prompt_tokens` and install it into a free lane.
    ///
    /// The prompt goes through the single-sequence prefill into a scratch
    /// one-lane cache, which is then installed at position 0 of a free lane.
    /// The prefill logits are sampled under `gen_config`, so the returned
    /// first token is this lane's own.
    ///
    /// # Errors
    ///
    /// An error when `prompt_tokens` is empty, when the prompt does not
    /// leave room for a token inside `capacity`, when every lane is taken,
    /// or when the prefill or the install fails.
    pub fn admit(
        &mut self,
        exec: &Executor<CudaRuntime>,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<LaneAdmitted> {
        let prompt_len = prompt_tokens.len();
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

        let client = LaneDecodeSession::client(exec.device());
        let input = exec.create_input_tensor(prompt_tokens)?;
        let (mut scratch_kv, mut scratch_gdn) = exec.qwen35_caches(prompt_len, capacity)?;
        let prefill_logits =
            exec.qwen35_prefill(&input, None, &mut scratch_kv, &mut scratch_gdn)?;
        let rope_pos = Executor::<CudaRuntime>::qwen35_first_rope_pos(None, &scratch_kv);

        let lane = self
            .decode
            .install(&client, &scratch_kv, &scratch_gdn, rope_pos)
            .map_err(|e| anyhow!("installing a prefilled sequence into a qwen35 lane: {e}"))?;
        drop(scratch_kv);
        drop(scratch_gdn);

        let mut request = LaneRequest {
            gen_config: gen_config.clone(),
            sampler: exec.sampler_state(prompt_tokens, gen_config)?,
            generated: 0,
        };
        let first_token = Self::sample_lane(exec, &prefill_logits, &mut request)?;
        let finish_reason = exec.decode_finish_reason(first_token, 0, gen_config.max_tokens);

        self.decode.set_next_token(lane, first_token)?;
        self.requests[lane] = Some(request);
        self.decode.commit_next_tokens(&client)?;

        Ok(LaneAdmitted {
            lane,
            first_token,
            finish_reason,
        })
    }

    /// One decode step for every occupied lane.
    ///
    /// Drives the graph replay or the eager batched forward, samples each
    /// occupied lane's logits row under that lane's own `GenerationConfig`,
    /// writes the sampled tokens back as the next step's inputs, and returns
    /// one [`LaneToken`] per occupied lane, in lane order. With no lane
    /// occupied it launches nothing and returns an empty vector.
    ///
    /// # Errors
    ///
    /// An error when the launch fails, when a lane's rope position runs past
    /// the model's rope table, or when a sample fails.
    pub fn step(&mut self, exec: &Executor<CudaRuntime>) -> Result<Vec<LaneToken>> {
        if self.requests.iter().all(Option::is_none) {
            return Ok(Vec::new());
        }
        let client = LaneDecodeSession::client(exec.device());
        let model = exec.qwen35_backbone()?;
        let logits = self.decode.launch(&client, model)?;

        let mut produced = Vec::with_capacity(self.requests.len());
        for lane in 0..self.requests.len() {
            let Some(request) = self.requests[lane].as_mut() else {
                continue;
            };
            let row = self.decode.logits_row(&logits, lane)?;
            let step_index = request.generated;
            let max_tokens = request.gen_config.max_tokens;
            let token_id = Self::sample_lane(exec, &row, request)?;
            let finish_reason = exec.decode_finish_reason(token_id, step_index, max_tokens);
            self.decode.set_next_token(lane, token_id)?;
            produced.push(LaneToken {
                lane,
                token_id,
                finish_reason,
            });
        }
        self.decode.commit_next_tokens(&client)?;
        Ok(produced)
    }

    /// Give lane `lane` back: clear its GDN state and drop its request
    /// state.
    ///
    /// # Errors
    ///
    /// An error when `lane` is outside the batch, or the clear fails.
    pub fn release(&mut self, exec: &Executor<CudaRuntime>, lane: usize) -> Result<()> {
        if lane >= self.requests.len() {
            return Err(anyhow!(
                "lane {lane} is outside the {} lanes this qwen35 session holds; release a lane \
                 admit handed out",
                self.batch()
            ));
        }
        let client = LaneDecodeSession::client(exec.device());
        self.decode.release(&client, lane)?;
        self.requests[lane] = None;
        Ok(())
    }

    /// Sample one lane's next token from `logits` through the executor's own
    /// sampler, then record it in the lane's history and grammar state.
    ///
    /// `logits` is that lane's row alone, `[1, 1, vocab]` from a decode step
    /// or `[1, prompt_len, vocab]` from a prefill; the sampler reads the last
    /// position either way.
    fn sample_lane(
        exec: &Executor<CudaRuntime>,
        logits: &Tensor<CudaRuntime>,
        request: &mut LaneRequest,
    ) -> Result<u32> {
        let (token_gpu, mirostat_id) = exec.sample_token_dispatch_with_grammar(
            logits,
            &request.sampler.token_history,
            &request.gen_config,
            &mut request.sampler.mirostat,
            request.sampler.device_grammar.as_ref(),
        )?;
        let token_id = match mirostat_id {
            Some(id) => id,
            None => {
                let event = token_gpu
                    .record_event()
                    .map_err(|e| anyhow!("Record event failed: {}", e))?;
                Executor::<CudaRuntime>::read_token_id(&token_gpu, event)?
            }
        };
        request.sampler.token_history.push(token_id);
        exec.advance_grammar_state(
            &mut request.sampler.grammar_dfa,
            &mut request.sampler.device_grammar,
            token_id,
        );
        request.generated += 1;
        Ok(token_id)
    }
}
