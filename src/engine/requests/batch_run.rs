//! The continuous batching loop of [`BatchEngine`]: wait for work, then
//! schedule and process one batch per tick.

use std::future::Future;

use anyhow::anyhow;
use futures::FutureExt;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, TypeConversionOps, UnaryOps,
};

use super::batch_engine::BatchEngine;

impl<R: Runtime<DType = DType>> BatchEngine<R>
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
    /// Run the continuous batching loop until the scheduler's wakeup
    /// channel closes.
    ///
    /// A forward pass blocks its thread, so run this on a thread of its
    /// own, as [`crate::engine::EngineThread`] does.
    pub async fn run(self) {
        self.run_until(std::future::pending()).await;
    }

    /// [`Self::run`] until `stop` resolves. The loop reads `stop` while it
    /// waits for work and between ticks, so a batch in flight finishes its
    /// tick first.
    pub async fn run_until(mut self, stop: impl Future<Output = ()>) {
        let mut stop = std::pin::pin!(stop);
        let mut notify_rx = match self.scheduler.take_notify_rx() {
            Some(rx) => rx,
            None => {
                tracing::error!("BatchEngine: notify_rx already taken");
                return;
            }
        };

        tracing::info!("BatchEngine: continuous batching loop started");

        loop {
            // Wait for new work or process existing work. A wakeup goes back
            // to the check, so a cancel that leaves no work sleeps again.
            if !self.has_live_work() {
                // `select` polls `stop` first.
                let wakeup = std::pin::pin!(notify_rx.recv());
                let next = match futures::future::select(stop.as_mut(), wakeup).await {
                    futures::future::Either::Left(((), _)) => {
                        tracing::info!("BatchEngine: stop requested, shutting down");
                        break;
                    }
                    futures::future::Either::Right((next, _)) => next,
                };
                match next {
                    Some(()) => continue,
                    None => {
                        tracing::info!("BatchEngine: notify channel closed, shutting down");
                        break;
                    }
                }
            }

            // Drain any additional notifications
            while notify_rx.try_recv().is_ok() {}

            self.tick().await;

            if stop.as_mut().now_or_never().is_some() {
                tracing::info!("BatchEngine: stop requested, shutting down");
                break;
            }
        }
    }

    /// Drop every cancelled sequence, then report whether work remains.
    pub(super) fn has_live_work(&mut self) -> bool {
        if let Err(e) = self.scheduler.reap_cancelled() {
            tracing::error!("BatchEngine: dropping cancelled requests failed: {e:#}");
        }
        let scheduler = &self.scheduler;
        self.seqs.end_gone(|seq_id| scheduler.is_live(seq_id));
        self.scheduler.has_work()
    }

    /// Schedule and process one batch, then drop the state of every
    /// sequence that ended.
    pub(super) async fn tick(&mut self) {
        match self.scheduler.schedule_batch() {
            Ok(Some(batch)) => {
                if let Err(e) = self.process_batch(&batch).await {
                    tracing::error!("BatchEngine: batch processing error: {:#}", e);
                    // End every sequence in the failed batch and tell its caller.
                    for seq_id in batch.all_sequences() {
                        self.seqs.end_seq(seq_id);
                        if let Some(tx) = self.scheduler.get_token_sender(seq_id) {
                            let err = anyhow!(
                                "the batch this request ran in failed: {e:#}. Retry the request."
                            );
                            // A caller that already left needs no error.
                            let _ = tx.send(Err(err)).await;
                        }
                        if let Err(e) = self.scheduler.finish_sequence(seq_id) {
                            tracing::error!(
                                seq_id,
                                "BatchEngine: ending a sequence of a failed batch failed: {e:#}"
                            );
                        }
                    }
                }
            }
            Ok(None) => {
                // No work available, yield and wait
                tokio::task::yield_now().await;
            }
            Err(e) => {
                tracing::error!("BatchEngine: scheduling error: {}", e);
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        }

        // Periodic cleanup. A cancelled request leaves the registry, so its
        // sampler state ends here.
        self.scheduler.cleanup_finished();
        let scheduler = &self.scheduler;
        self.seqs.end_gone(|seq_id| scheduler.is_live(seq_id));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;

    use super::super::batch_test_rig::{rig, Rig, POOL, SEED};
    use super::*;
    use crate::engine::test_support::seeded_executor;

    /// With paged attention off no request can decode, so the build fails
    /// instead of every request.
    #[test]
    fn an_executor_without_the_paged_pool_is_refused() -> Result<()> {
        let Rig { scheduler, .. } = rig(POOL)?;
        let executor = Arc::new(seeded_executor(SEED, None)?);
        let err = BatchEngine::new(executor, scheduler)
            .err()
            .map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|m| m.contains("inference.paged_attention is off")),
            "{err:?}"
        );
        Ok(())
    }
}
