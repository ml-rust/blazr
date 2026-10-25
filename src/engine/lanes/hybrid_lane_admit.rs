//! Admission and the decode step of [`HybridLaneEngine`].
//!
//! A waiting request prefills into fresh caches of its own, in one forward
//! or in chunks with decode steps between them, then takes a free lane with
//! the token its prefill logits give. A step feeds every lane its last
//! token through one row-batched forward and samples each lane's row.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::inference::scheduler::SequenceId;
use boostr::inference::{LayeredKvCache, LayeredSsmState};
use boostr::model::{HybridRow, ModelClient};
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use super::hybrid_lane_engine::{HybridLane, HybridLaneEngine};
use crate::config::GenerationConfig;
use crate::engine::types::PrefillCancel;

/// A prefilled request: its caches and the logits of its last prompt token.
type Prefilled<R> = (HybridLane<R>, Tensor<R>);

impl<R: Runtime<DType = DType>> HybridLaneEngine<R>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + RoPEPackedOps<R>
        + boostr::quant::traits::DequantOps<R>,
{
    /// Put waiting requests into free lanes until the lanes run out, or
    /// until the SLO policy holds lane filling.
    pub(super) async fn admit_waiting(&mut self) {
        while self.live() < self.lanes.len()
            && self.live() < self.scheduler.admission_limit(self.live())
        {
            let Some(seq_id) = self.scheduler.next_waiting() else {
                break;
            };
            let registry = self.scheduler.registry();
            let (Some(prompt), Some(mut gen_config)) =
                (registry.prompt_tokens(seq_id), registry.gen_config(seq_id))
            else {
                // The caller left between submit and admit.
                continue;
            };
            // The same budget as the single-sequence loop, so a request ends
            // at the same token on either path.
            gen_config.max_tokens = crate::engine::token_budget(
                gen_config.max_tokens,
                self.executor.config().max_seq_len(),
                prompt.len(),
            );
            if prompt.is_empty() || gen_config.max_tokens == 0 {
                // The single-sequence loop yields no token here either.
                let _ = registry.finish(seq_id);
                continue;
            }
            match self.prefill(seq_id, &prompt, gen_config.max_tokens).await {
                Ok(Some(prefilled)) => self.install(prefilled, &prompt, gen_config).await,
                // The caller left mid-prefill: its request is gone already.
                Ok(None) => {}
                Err(e) => {
                    let reason = anyhow!("the prefill of this request failed: {e:#}");
                    self.refuse(seq_id, reason).await;
                }
            }
        }
    }

    /// Prefill `prompt` into fresh caches, in chunks of `chunk_size` with
    /// one decode step for the occupied lanes between chunks.
    ///
    /// A cancel read before each chunk stops the prefill once the request
    /// left the registry. The caches drop and the call returns `None`.
    ///
    /// # Errors
    ///
    /// An error when the caches do not allocate or a forward fails.
    async fn prefill(
        &mut self,
        seq_id: SequenceId,
        prompt: &[u32],
        max_tokens: usize,
    ) -> Result<Option<Prefilled<R>>> {
        let (mut kv, mut ssm) = self.lane_caches(prompt.len(), max_tokens)?;
        let scheduler = Arc::clone(&self.scheduler);
        let gone = move || !scheduler.registry().contains(seq_id);
        let cancel = PrefillCancel::when(&gone);
        let step = if self.chunk_size == 0 {
            prompt.len()
        } else {
            self.chunk_size
        };
        let (mut fed, mut logits) = (0, None);
        while fed < prompt.len() {
            if cancel.is_cancelled() {
                tracing::debug!(seq_id, fed, "hybrid lane prefill stopped: the caller left");
                return Ok(None);
            }
            let end = (fed + step).min(prompt.len());
            logits = Some(self.prefill_forward(&prompt[fed..end], fed, &mut kv, ssm.as_mut())?);
            fed = end;
            #[cfg(test)]
            if let Some(hook) = self.after_chunk.as_mut() {
                hook(fed);
            }
            if fed < prompt.len() && self.live() > 0 {
                self.step().await;
            }
        }
        let logits = logits.ok_or_else(|| anyhow!("a prefill of no prompt tokens ran"))?;
        let lane = HybridLane {
            seq_id,
            kv,
            ssm,
            position: prompt.len(),
            next_token: 0,
        };
        Ok(Some((lane, logits)))
    }

    /// Start the request's sampler, sample its first token from the prefill
    /// logits, put it into a free lane and stream the token.
    async fn install(&mut self, prefilled: Prefilled<R>, prompt: &[u32], cfg: GenerationConfig) {
        let (mut lane, logits) = prefilled;
        let seq_id = lane.seq_id;
        let sampled = self
            .seqs
            .begin(&self.executor, seq_id, prompt, cfg)
            .and_then(|()| self.seqs.sample(&self.executor, seq_id, &logits));
        let sampled = match sampled {
            Ok(sampled) => sampled,
            Err(e) => {
                let reason = anyhow!("sampling the first token of this request failed: {e:#}");
                return self.refuse(seq_id, reason).await;
            }
        };
        let Some(slot) = self.lanes.iter().position(Option::is_none) else {
            let reason = anyhow!(
                "every one of the {} hybrid lanes is taken; admission must stop at a full set",
                self.lanes.len()
            );
            return self.refuse(seq_id, reason).await;
        };
        lane.next_token = sampled.id;
        self.lanes[slot] = Some(lane);
        if !self.deliver(seq_id, sampled).await {
            self.end_lane(slot);
        }
    }

    /// One decode step for every occupied lane, each token streamed.
    ///
    /// A lane whose request left the registry is freed first and takes no
    /// row. A failed forward ends every lane's request with the error. A
    /// failed sample ends only its own request.
    pub(super) async fn step(&mut self) {
        self.free_gone_lanes();
        let (occupied, logits) = match self.step_forward() {
            Ok(Some(step)) => step,
            Ok(None) => return,
            Err(e) => return self.fail_live_lanes(&e).await,
        };
        for (row, &lane) in occupied.iter().enumerate() {
            let Some(seq_id) = self.lanes[lane].as_ref().map(|l| l.seq_id) else {
                continue;
            };
            let sampled = logits
                .narrow(0, row, 1)
                .map_err(|e| anyhow!("row {row} of the step logits: {e}"))
                .and_then(|row_logits| self.seqs.sample(&self.executor, seq_id, &row_logits));
            let sampled = match sampled {
                Ok(sampled) => sampled,
                Err(e) => {
                    let reason = anyhow!("sampling this request's token failed: {e:#}");
                    self.scheduler.registry().send_error(seq_id, reason).await;
                    self.end_lane(lane);
                    continue;
                }
            };
            if let Some(state) = self.lanes[lane].as_mut() {
                state.next_token = sampled.id;
            }
            if !self.deliver(seq_id, sampled).await {
                self.end_lane(lane);
            }
        }
    }

    /// Free every lane whose request left the registry.
    fn free_gone_lanes(&mut self) {
        for lane in 0..self.lanes.len() {
            let gone = self.lanes[lane]
                .as_ref()
                .is_some_and(|l| !self.scheduler.registry().contains(l.seq_id));
            if gone {
                self.end_lane(lane);
            }
        }
    }

    /// The row-batched forward of one step: the occupied lanes in lane
    /// order, and their `[N, 1, vocab]` logits. `None` with no lane occupied.
    fn step_forward(&mut self) -> Result<Option<(Vec<usize>, Tensor<R>)>> {
        let occupied: Vec<usize> = (0..self.lanes.len())
            .filter(|&lane| self.lanes[lane].is_some())
            .collect();
        if occupied.is_empty() {
            return Ok(None);
        }
        let tokens: Vec<i64> = self
            .lanes
            .iter()
            .flatten()
            .map(|lane| i64::from(lane.next_token))
            .collect();
        let input = Tensor::from_slice(&tokens, &[tokens.len(), 1], self.executor.device())?;
        let mut rows: Vec<HybridRow<'_, R>> = self
            .lanes
            .iter_mut()
            .flatten()
            .map(|lane| HybridRow {
                kv_cache: &mut lane.kv,
                ssm_state: lane.ssm.as_mut(),
                position: lane.position,
            })
            .collect();
        let logits = self
            .executor
            .model()?
            .forward_hybrid_rows(&input, &mut rows)
            .map_err(|e| anyhow!("the decode step over {} lanes failed: {e}", rows.len()))?;
        drop(rows);
        for lane in self.lanes.iter_mut().flatten() {
            lane.position += 1;
        }
        #[cfg(test)]
        {
            self.forwards += 1;
            self.widest_step = self.widest_step.max(occupied.len());
        }
        Ok(Some((occupied, logits)))
    }

    /// Fresh caches for one request of `prompt_len` tokens that decodes at
    /// most `max_tokens`: a KV cache for the attention layers, and zeroed
    /// state for the SSM layers when the model has any.
    fn lane_caches(
        &self,
        prompt_len: usize,
        max_tokens: usize,
    ) -> Result<(LayeredKvCache<R>, Option<LayeredSsmState<R>>)> {
        if self
            .executor
            .model()?
            .num_ssm_layers()
            .is_some_and(|n| n > 0)
        {
            let max_seq_len = self.executor.config().max_seq_len();
            let capacity = (prompt_len + max_tokens).min(max_seq_len);
            let (kv, ssm) = self.executor.hybrid_caches(capacity, max_seq_len)?;
            return Ok((kv, Some(ssm)));
        }
        let (kv, _) = self.executor.contiguous_kv_cache(prompt_len, max_tokens)?;
        Ok((kv, None))
    }

    /// One prefill forward of `chunk` at position `at`.
    fn prefill_forward(
        &mut self,
        chunk: &[u32],
        at: usize,
        kv: &mut LayeredKvCache<R>,
        ssm: Option<&mut LayeredSsmState<R>>,
    ) -> Result<Tensor<R>> {
        let input = self.executor.create_input_tensor(chunk)?;
        let model = self.executor.model()?;
        let logits = match ssm {
            Some(state) => model.forward_hybrid(&input, kv, state, at),
            None => model.forward_with_kv_cache(&input, kv, at),
        }
        .map_err(|e| {
            anyhow!(
                "prefill of positions {at}..{} failed: {e}",
                at + chunk.len()
            )
        })?;
        #[cfg(test)]
        {
            self.forwards += 1;
        }
        Ok(logits)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use boostr::{CpuDevice, CpuRuntime};

    use crate::engine::executor::Executor;
    use crate::engine::executors::generation::executor_generate_hybrid::tests::{
        config, load, write_checkpoint,
    };
    use crate::engine::lanes::lane_scheduler::LaneScheduler;
    use crate::engine::request_registry::RequestHandle;
    use crate::engine::requests::batch_test_rig::{alone_tokens, greedy, SEED};
    use crate::engine::test_support::seeded_executor;

    /// Mamba2, Mamba2, then MLA + MoE: oxidizr's `nano-start` layout.
    pub(in crate::engine) const HYBRID: &[bool] = &[true, true, false];
    /// Two MLA + MoE layers and no SSM layer.
    pub(in crate::engine) const MLA: &[bool] = &[false, false];
    /// Ticks a test allows before it calls the engine stuck.
    const TICK_LIMIT: usize = 64;

    /// An engine over a tiny seeded oxidizr checkpoint, with its scheduler.
    pub(in crate::engine) struct Rig {
        pub executor: Arc<Executor<CpuRuntime>>,
        pub scheduler: Arc<LaneScheduler>,
        pub engine: HybridLaneEngine<CpuRuntime>,
        _dir: tempfile::TempDir,
    }

    /// `lanes` lanes over the model `ssm` lays out, prefilling in chunks of
    /// `chunk` tokens.
    pub(in crate::engine) fn rig(ssm: &[bool], lanes: usize, chunk: usize) -> Result<Rig> {
        let dir = tempfile::tempdir()?;
        write_checkpoint(dir.path(), &config(ssm, true), true);
        let (model, blazr_config) = load(dir.path());
        let tokenizer = crate::tokenizer::from_pretrained("llama3")?;
        let executor = Executor::new(model, blazr_config, tokenizer, CpuDevice::new(), 32)?;
        let executor = Arc::new(executor);
        let scheduler = LaneScheduler::new(lanes)?;
        let engine = HybridLaneEngine::new(Arc::clone(&executor), Arc::clone(&scheduler))?
            .with_prefill_chunk_size(chunk);
        Ok(Rig {
            executor,
            scheduler,
            engine,
            _dir: dir,
        })
    }

    /// The ids `prompt` produces alone through the single-sequence loop.
    pub(in crate::engine) async fn alone(
        rig: &Rig,
        prompt: &[u32],
        cfg: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        let tokens = alone_tokens(&rig.executor, prompt, cfg).await?;
        Ok(tokens.into_iter().map(|t| t.token_id).collect())
    }

    /// Tick until no lane is occupied and no request waits.
    pub(in crate::engine) async fn drain(rig: &mut Rig) -> Result<()> {
        for _ in 0..TICK_LIMIT {
            if rig.engine.live() == 0 && rig.scheduler.waiting_len() == 0 {
                return Ok(());
            }
            rig.engine.tick().await;
        }
        Err(anyhow!(
            "the hybrid lanes still had work after {TICK_LIMIT} ticks"
        ))
    }

    /// The ids `handle` received so far. A streamed error fails the call.
    pub(in crate::engine) fn ids(handle: &mut RequestHandle) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        while let Ok(item) = handle.token_rx.try_recv() {
            ids.push(item?.token_id);
        }
        Ok(ids)
    }

    /// Three requests of different lengths in two lanes: two share every
    /// step, the third takes the lane the shortest frees. Each produces
    /// its ids alone, so no state of the finished request leaks into it.
    async fn three_requests_in_two_lanes(ssm: &[bool]) -> Result<()> {
        let mut rig = rig(ssm, 2, 2)?;
        let seeded = GenerationConfig {
            max_tokens: 5,
            temperature: 1.0,
            seed: Some(7),
            ..GenerationConfig::default()
        };
        let requests: [(&[u32], GenerationConfig); 3] = [
            (&[3, 10, 17, 1, 8], greedy(6)),
            (&[5, 2], greedy(3)),
            (&[11, 4, 19], seeded),
        ];
        let mut expected = Vec::new();
        for (prompt, cfg) in &requests {
            expected.push(alone(&rig, prompt, cfg).await?);
        }

        let mut handles = Vec::new();
        for (prompt, cfg) in requests {
            handles.push(rig.scheduler.submit(prompt.to_vec(), cfg)?);
        }
        drain(&mut rig).await?;

        for (handle, want) in handles.iter_mut().zip(&expected) {
            assert_eq!(&ids(handle)?, want, "ssm {ssm:?} request {}", handle.seq_id);
            assert!(!rig.scheduler.registry().contains(handle.seq_id));
            assert!(!rig.engine.seqs.contains(handle.seq_id));
        }
        assert_eq!(rig.engine.widest_step, 2, "two lanes share a step");
        Ok(())
    }

    #[tokio::test]
    async fn hybrid_requests_of_different_lengths_match_their_runs_alone() -> Result<()> {
        three_requests_in_two_lanes(HYBRID).await
    }

    #[tokio::test]
    async fn mla_requests_of_different_lengths_match_their_runs_alone() -> Result<()> {
        three_requests_in_two_lanes(MLA).await
    }

    /// One lane serves three requests in turn. The second and third run
    /// the same prompt after other requests left the lane, and both give
    /// the ids of that prompt alone.
    #[tokio::test]
    async fn a_reused_lane_starts_each_request_from_fresh_state() -> Result<()> {
        let mut rig = rig(HYBRID, 1, 0)?;
        let prompt = [9, 3, 14];
        let expected = alone(&rig, &prompt, &greedy(5)).await?;

        let mut first = rig.scheduler.submit(vec![20, 1, 6, 6, 2], greedy(7))?;
        drain(&mut rig).await?;
        assert_eq!(ids(&mut first)?.len(), 7);
        for _ in 0..2 {
            let mut again = rig.scheduler.submit(prompt.to_vec(), greedy(5))?;
            drain(&mut rig).await?;
            assert_eq!(ids(&mut again)?, expected);
        }
        Ok(())
    }

    /// A hybrid model never reaches the paged batch: its build fails and
    /// names the hybrid lanes.
    #[test]
    fn the_paged_batch_refuses_a_hybrid_model() -> Result<()> {
        let rig = rig(MLA, 2, 0)?;
        let paged = crate::engine::requests::batch_test_rig::rig(
            crate::engine::requests::batch_test_rig::POOL,
        )?;
        let err = crate::engine::BatchEngine::new(rig.executor, paged.scheduler)
            .err()
            .map(|e| e.to_string());
        assert!(
            err.as_deref().is_some_and(|m| m.contains("hybrid lanes")),
            "{err:?}"
        );
        Ok(())
    }

    #[test]
    fn a_model_without_hybrid_rows_is_refused() -> Result<()> {
        let executor = Arc::new(seeded_executor(SEED, None)?);
        let err = HybridLaneEngine::new(executor, LaneScheduler::new(2)?)
            .err()
            .map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|m| m.contains("does not decode in hybrid lanes")),
            "{err:?}"
        );
        Ok(())
    }
}
