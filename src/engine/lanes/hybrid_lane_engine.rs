//! Driver behind [`LaneScheduler`] for `hybrid` models, MLA models
//! included: decode lanes that each hold one request's KV cache and SSM
//! state, stepped together.
//!
//! One iteration admits every waiting request a free lane can take, then
//! runs one decode step for all occupied lanes. A step is one
//! [`LoadedModel::forward_hybrid_rows`](boostr::model::LoadedModel::forward_hybrid_rows)
//! over every lane: the embedding, norms, feed-forward halves and LM head
//! run once over all lanes, and each attention, MLA or Mamba2 mixer runs
//! over its lane's own cache or state. Lanes of different lengths share a
//! step, so a lane produces the tokens its request produces alone.
//!
//! A lane's caches are built at its admission and dropped when its request
//! ends, so no state of one request reaches the next request in that lane.
//! Each lane samples through its own [`BatchSeqs`] state, so grammars,
//! mirostat, the penalties and logprobs apply as on the single-sequence
//! path.
//!
//! `inference.prefill_chunk_size` above 0 feeds an admitted prompt in slices
//! of that many tokens and runs one decode step for the occupied lanes
//! between the slices. A [`PrefillCancel`] read before each slice stops the
//! prefill of a request whose caller left and drops its caches. At 0 the
//! whole prompt goes through one forward, and the cancel is read once.
//!
//! With no lane occupied the loop waits on the scheduler's wakeup channel,
//! so an idle server runs nothing.
//!
//! [`PrefillCancel`]: crate::engine::types::PrefillCancel

use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::inference::scheduler::SequenceId;
use boostr::inference::{LayeredKvCache, LayeredSsmState};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, TypeConversionOps, UnaryOps,
};

use super::lane_scheduler::LaneScheduler;
use crate::engine::executor::Executor;
use crate::engine::seq_sampler::{BatchSeqs, SampledToken};
use crate::engine::types::GeneratedToken;

/// One admitted request's decode state.
pub(in crate::engine) struct HybridLane<R: Runtime<DType = DType>> {
    /// The request this lane serves.
    pub(super) seq_id: SequenceId,
    /// KV cache of the attention layers.
    pub(super) kv: LayeredKvCache<R>,
    /// State of the SSM layers. `None` for a model with no SSM layer.
    pub(super) ssm: Option<LayeredSsmState<R>>,
    /// Tokens the caches hold: the position of the token the next step feeds.
    pub(super) position: usize,
    /// The token the next step feeds: the one this lane sampled last.
    pub(super) next_token: u32,
}

/// Fills and steps the decode lanes of a `hybrid` model.
pub struct HybridLaneEngine<R: Runtime<DType = DType>> {
    pub(super) executor: Arc<Executor<R>>,
    pub(super) scheduler: Arc<LaneScheduler>,
    /// Prompt tokens one prefill forward feeds; 0 feeds a whole prompt.
    pub(super) chunk_size: usize,
    /// Each lane's request state, `None` when the lane is free.
    pub(super) lanes: Vec<Option<HybridLane<R>>>,
    /// Sampling state of every request in a lane.
    pub(super) seqs: BatchSeqs<R>,
    /// Runs after each prefill forward with the prompt tokens fed so far,
    /// so a test can act between chunks.
    #[cfg(test)]
    pub(crate) after_chunk: Option<Box<dyn FnMut(usize) + Send>>,
    /// Prefill and decode forwards this engine ran, for the tests.
    #[cfg(test)]
    pub(crate) forwards: usize,
    /// The most lanes one decode step served, for the tests.
    #[cfg(test)]
    pub(crate) widest_step: usize,
}

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
    /// An engine with `scheduler.lanes()` lanes over `executor`.
    ///
    /// # Errors
    ///
    /// An error when the model is not a `hybrid` model, or when the sampling
    /// scratch does not allocate.
    pub fn new(executor: Arc<Executor<R>>, scheduler: Arc<LaneScheduler>) -> Result<Self> {
        let model = executor.model()?;
        if !model.supports_hybrid_rows() {
            return Err(anyhow!(
                "model type '{}' does not decode in hybrid lanes; they serve hybrid and MLA \
                 models. Serve it with the paged batch or with max_batch_size = 1",
                model.model_type()
            ));
        }
        let seqs = BatchSeqs::new(executor.vocab_size(), executor.device())?;
        let chunk_size = executor.config().inference.prefill_chunk_size;
        Ok(Self {
            lanes: (0..scheduler.lanes()).map(|_| None).collect(),
            executor,
            scheduler,
            chunk_size,
            seqs,
            #[cfg(test)]
            after_chunk: None,
            #[cfg(test)]
            forwards: 0,
            #[cfg(test)]
            widest_step: 0,
        })
    }

    /// Override `inference.prefill_chunk_size` for this engine. A value of
    /// `0` feeds a whole prompt in one forward.
    pub fn with_prefill_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Admit, step and stream until the scheduler's wakeup channel closes.
    ///
    /// A forward blocks its thread, so run this on a thread of its own, as
    /// [`crate::engine::EngineThread`] does.
    pub async fn run(self) {
        self.run_until(std::future::pending()).await;
    }

    /// [`Self::run`] until `stop` resolves. The loop reads `stop` while it
    /// waits for work and between steps.
    pub async fn run_until(mut self, stop: impl std::future::Future<Output = ()>) {
        use futures::FutureExt;

        let mut stop = std::pin::pin!(stop);
        let Some(mut notify_rx) = self.scheduler.take_notify_rx() else {
            tracing::error!(
                "the lane scheduler's wakeup receiver was already taken; only one engine can \
                 drive one LaneScheduler"
            );
            return;
        };
        tracing::info!(lanes = self.lanes.len(), "hybrid lane decode loop started");

        loop {
            self.admit_waiting().await;
            if self.live() == 0 {
                // `select` polls `stop` first.
                let wakeup = std::pin::pin!(notify_rx.recv());
                match futures::future::select(stop.as_mut(), wakeup).await {
                    futures::future::Either::Left(((), _)) => break,
                    futures::future::Either::Right((Some(()), _)) => continue,
                    futures::future::Either::Right((None, _)) => {
                        tracing::info!("lane scheduler closed its wakeup channel, shutting down");
                        break;
                    }
                }
            }
            // The next iteration admits from the queue, so these need no action.
            while notify_rx.try_recv().is_ok() {}
            self.step().await;
            if stop.as_mut().now_or_never().is_some() {
                break;
            }
        }
        tracing::info!("hybrid lane decode loop stopped");
    }

    /// One iteration of the loop without the wait: admit, then step.
    #[cfg(test)]
    pub(crate) async fn tick(&mut self) {
        self.admit_waiting().await;
        self.step().await;
    }

    /// How many lanes hold a request.
    pub(super) fn live(&self) -> usize {
        self.lanes.iter().filter(|lane| lane.is_some()).count()
    }

    /// Send one sampled token to its caller and record it.
    ///
    /// The request's stop sequences cut the text as on every other path.
    /// Returns `false` when the request is over: the token finished it, or
    /// the caller's receiver is gone.
    pub(super) async fn deliver(&self, seq_id: SequenceId, sampled: SampledToken) -> bool {
        let registry = self.scheduler.registry();
        let cut = registry.token_text(
            seq_id,
            self.executor.tokenizer(),
            sampled.id,
            sampled.finish,
        );
        registry.push_history(seq_id, sampled.id);
        let token = GeneratedToken {
            token_id: sampled.id,
            text: cut.text,
            logprob: sampled.logprob,
            top_logprobs: sampled.top_logprobs,
            finish_reason: cut.finish,
        };
        registry.send_token(seq_id, token).await && cut.finish.is_none()
    }

    /// End the request in `lane` and free the lane with its caches.
    ///
    /// The request leaves the registry here, so the hold of its slot drops.
    pub(super) fn end_lane(&mut self, lane: usize) {
        let Some(ended) = self.lanes.get_mut(lane).and_then(Option::take) else {
            return;
        };
        self.seqs.end_seq(ended.seq_id);
        if let Err(e) = self.scheduler.registry().finish(ended.seq_id) {
            tracing::error!(
                "releasing request {} from hybrid lane {lane}: {e}",
                ended.seq_id
            );
        }
    }

    /// End a request that holds no lane, telling its caller `error`.
    pub(super) async fn refuse(&mut self, seq_id: SequenceId, error: anyhow::Error) {
        let registry = self.scheduler.registry();
        registry.send_error(seq_id, error).await;
        self.seqs.end_seq(seq_id);
        if let Err(e) = registry.finish(seq_id) {
            tracing::error!("releasing request {seq_id} after its error: {e}");
        }
    }

    /// Report `error` to every request in a lane and free every lane.
    pub(super) async fn fail_live_lanes(&mut self, error: &anyhow::Error) {
        tracing::error!("hybrid lane step failed: {error:#}");
        for lane in 0..self.lanes.len() {
            let Some(seq_id) = self.lanes[lane].as_ref().map(|l| l.seq_id) else {
                continue;
            };
            let reason = anyhow!("the decode step of this request's lane failed: {error:#}");
            self.scheduler.registry().send_error(seq_id, reason).await;
            self.end_lane(lane);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::super::hybrid_lane_admit::tests::{alone, drain, ids, rig, HYBRID, MLA};
    use super::*;
    use crate::engine::request_registry::{GenerationScheduler, RequestHandle};
    use crate::engine::requests::batch_test_rig::greedy;
    use crate::engine::test_support::WAIT_LIMIT;
    use crate::engine::{EngineThread, FailRequests, ThreadedScheduler, WorkHold};

    /// A request decodes in one lane while a nine-token prompt prefills in
    /// chunks of two. Its handle drops after the first chunk, so no second
    /// chunk runs, its caches and sampler drop, and the other lane decodes
    /// on to its ids alone.
    #[tokio::test]
    async fn a_cancel_mid_prefill_stops_at_the_next_chunk_and_frees_the_request() -> Result<()> {
        let mut rig = rig(HYBRID, 2, 2)?;
        let running_prompt = [4, 7];
        let expected = alone(&rig, &running_prompt, &greedy(8)).await?;
        let mut running = rig.scheduler.submit(running_prompt.to_vec(), greedy(8))?;
        rig.engine.tick().await;
        assert_eq!(rig.engine.live(), 1);

        let cancelled = rig.scheduler.submit((1..=9).collect(), greedy(4))?;
        let seq_id = cancelled.seq_id;
        let chunks = Arc::new(AtomicUsize::new(0));
        let (seen, mut slot) = (Arc::clone(&chunks), Some(cancelled));
        rig.engine.after_chunk = Some(Box::new(move |_: usize| {
            seen.fetch_add(1, Ordering::SeqCst);
            drop(slot.take());
        }));
        rig.engine.tick().await;
        rig.engine.after_chunk = None;

        assert_eq!(chunks.load(Ordering::SeqCst), 1, "one chunk ran");
        assert!(!rig.scheduler.registry().contains(seq_id));
        assert!(!rig.engine.seqs.contains(seq_id));
        assert_eq!(rig.engine.live(), 1, "the cancelled request took no lane");
        drain(&mut rig).await?;
        assert_eq!(ids(&mut running)?, expected);
        Ok(())
    }

    /// Two requests decode together. One handle drops mid-decode: the next
    /// step frees its lane before the forward, and the other decodes on.
    #[tokio::test]
    async fn a_cancel_mid_decode_frees_the_lane_and_its_state() -> Result<()> {
        let mut rig = rig(MLA, 2, 0)?;
        let kept_prompt = [2, 9, 16];
        let expected = alone(&rig, &kept_prompt, &greedy(8)).await?;
        let dropped: RequestHandle = rig.scheduler.submit(vec![5, 6], greedy(8))?;
        let mut kept = rig.scheduler.submit(kept_prompt.to_vec(), greedy(8))?;
        rig.engine.tick().await;
        assert_eq!(rig.engine.live(), 2);

        let seq_id = dropped.seq_id;
        drop(dropped);
        let before = rig.engine.forwards;
        rig.engine.tick().await;

        assert_eq!(rig.engine.live(), 1);
        assert!(!rig.scheduler.registry().contains(seq_id));
        assert!(!rig.engine.seqs.contains(seq_id));
        assert_eq!(
            rig.engine.forwards,
            before + 1,
            "one step for the kept lane"
        );
        drain(&mut rig).await?;
        assert_eq!(ids(&mut kept)?, expected);
        Ok(())
    }

    /// Every item `handle` receives until its channel closes.
    async fn until_closed(handle: &mut RequestHandle) -> Result<Vec<Result<GeneratedToken>>> {
        let mut items = Vec::new();
        while let Some(item) = tokio::time::timeout(WAIT_LIMIT, handle.token_rx.recv()).await? {
            items.push(item);
        }
        Ok(items)
    }

    /// The first prefill chunk waits until both requests are queued, then
    /// panics. Each request ends with one error item, its hold drops, and
    /// the scheduler refuses the next submit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn a_panicking_engine_fails_every_request_and_refuses_new_ones() -> Result<()> {
        let rig = rig(HYBRID, 2, 1)?;
        let (scheduler, mut engine) = (rig.scheduler, rig.engine);
        let (go, gate) = std::sync::mpsc::channel::<()>();
        engine.after_chunk = Some(Box::new(move |_: usize| {
            let _ = gate.recv_timeout(WAIT_LIMIT);
            panic!("injected engine panic");
        }));
        let owner: Arc<dyn FailRequests> = scheduler.clone();
        let thread = EngineThread::spawn("blazr-hybrid-lanes", owner, move |stop| {
            engine.run_until(stop)
        })?;
        let threaded = Arc::new(ThreadedScheduler::new(scheduler.clone(), thread));

        let probe = Arc::new(());
        let hold = WorkHold::new(Arc::clone(&probe));
        let (mut first, mut second) = WorkHold::within(Some(hold), async {
            let first = threaded.submit((1..=6).collect(), greedy(4))?;
            let second = threaded.submit(vec![7, 8, 9], greedy(4))?;
            anyhow::Ok((first, second))
        })
        .await?;
        assert_eq!(Arc::strong_count(&probe), 2, "the requests hold the guard");
        go.send(()).map_err(|e| anyhow!("{e}"))?;

        for handle in [&mut first, &mut second] {
            let items = until_closed(handle).await?;
            let Some(Err(e)) = items.last() else {
                return Err(anyhow!("the stream closed without an error item"));
            };
            let message = format!("{e:#}");
            assert!(message.contains("blazr-hybrid-lanes panicked"), "{message}");
            assert!(message.contains("injected engine panic"), "{message}");
            assert_eq!(items.iter().filter(|i| i.is_err()).count(), 1);
        }
        assert_eq!(Arc::strong_count(&probe), 1, "every hold dropped");
        assert!(!scheduler.registry().contains(first.seq_id));
        assert_eq!(scheduler.waiting_len(), 0);

        let fault = threaded.fault().ok_or_else(|| anyhow!("no fault is set"))?;
        assert!(fault.contains("injected engine panic"), "{fault}");
        let refused = threaded.submit(vec![1, 2], greedy(2)).err();
        assert!(refused.is_some_and(|e| e.to_string().contains("the engine stopped")));
        let joined = tokio::task::spawn_blocking(move || drop(threaded));
        tokio::time::timeout(WAIT_LIMIT, joined).await??;
        Ok(())
    }
}
