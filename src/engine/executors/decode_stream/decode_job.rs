//! One single-sequence generation, owned, ready to move to its decode thread.
//!
//! The job builds its stream on the thread that pumps it. The stream's
//! prefill reads the job's channel before each prompt chunk, so a reader
//! that drops its receiver stops the prefill at the next chunk boundary.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use tokio::sync::mpsc;

use boostr::{DType, Runtime};

use crate::config::GenerationConfig;

use super::super::executor::{EngineClient, Executor};
use super::super::types::PrefillCancel;
use super::decode_output::{forward_tokens, DecodeOutput};

/// A test hook that runs on the decode thread after each engine item.
#[cfg(test)]
pub(crate) type StepHook = Arc<dyn Fn(usize) + Send + Sync>;

/// A test hook that runs on the decode thread before each prefill chunk.
#[cfg(test)]
pub(crate) type PrefillHook = Arc<dyn Fn() + Send + Sync>;

/// The inputs a generation path takes beside the prompt ids.
pub(in crate::engine::executors) enum DecodeMedia {
    /// Text only: `Executor::generate`.
    Text,
    /// Images and audio: `Executor::generate_multimodal`.
    Multimodal {
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<f32>>,
    },
    /// Images through the vision tower: `Executor::generate_with_images`.
    Images { images: Vec<Vec<u8>> },
}

/// One generation, owned, ready to move to its decode thread.
pub(crate) struct DecodeJob<R: Runtime<DType = DType>> {
    pub(in crate::engine::executors) executor: Arc<Executor<R>>,
    pub(in crate::engine::executors) prompt: Vec<u32>,
    pub(in crate::engine::executors) config: GenerationConfig,
    pub(in crate::engine::executors) media: DecodeMedia,
    /// How long a send waits for room. `None` waits without limit.
    pub(in crate::engine::executors) idle: Option<Duration>,
    #[cfg(test)]
    pub(in crate::engine::executors) on_step: Option<StepHook>,
    #[cfg(test)]
    pub(in crate::engine::executors) on_prefill: Option<PrefillHook>,
}

impl<R> DecodeJob<R>
where
    R: Runtime<DType = DType>,
    R::Client: EngineClient<R>,
{
    /// A text generation of `prompt`.
    pub(crate) fn text(
        executor: Arc<Executor<R>>,
        prompt: Vec<u32>,
        config: GenerationConfig,
    ) -> Self {
        Self::new(executor, prompt, config, DecodeMedia::Text)
    }

    /// A generation of `prompt` with `images` and `audio` through the
    /// model's own encoders.
    pub(crate) fn multimodal(
        executor: Arc<Executor<R>>,
        prompt: Vec<u32>,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<f32>>,
        config: GenerationConfig,
    ) -> Self {
        let media = DecodeMedia::Multimodal { images, audio };
        Self::new(executor, prompt, config, media)
    }

    /// A generation of `prompt` with `images` through the vision tower.
    pub(crate) fn images(
        executor: Arc<Executor<R>>,
        prompt: Vec<u32>,
        images: Vec<Vec<u8>>,
        config: GenerationConfig,
    ) -> Self {
        Self::new(executor, prompt, config, DecodeMedia::Images { images })
    }

    fn new(
        executor: Arc<Executor<R>>,
        prompt: Vec<u32>,
        config: GenerationConfig,
        media: DecodeMedia,
    ) -> Self {
        let secs = executor.config().inference.decode_idle_timeout_secs;
        Self {
            executor,
            prompt,
            config,
            media,
            idle: (secs > 0).then(|| Duration::from_secs(secs)),
            #[cfg(test)]
            on_step: None,
            #[cfg(test)]
            on_prefill: None,
        }
    }

    /// Run `hook` on the decode thread after each engine item.
    #[cfg(test)]
    pub(crate) fn with_step_hook(mut self, hook: StepHook) -> Self {
        self.on_step = Some(hook);
        self
    }

    /// Run `hook` on the decode thread before each prefill chunk.
    #[cfg(test)]
    pub(crate) fn with_prefill_hook(mut self, hook: PrefillHook) -> Self {
        self.on_prefill = Some(hook);
        self
    }

    /// Replace the idle timeout the executor config set.
    #[cfg(test)]
    pub(crate) fn with_idle_timeout(mut self, idle: Option<Duration>) -> Self {
        self.idle = idle;
        self
    }

    /// Build the stream and forward it through `out` to `tx`, on the thread
    /// that calls this.
    ///
    /// The prefill stops at its next chunk once `tx`'s receiver is gone.
    pub(in crate::engine::executors) async fn pump<O: DecodeOutput>(
        self,
        out: &mut O,
        tx: &mpsc::Sender<O::Item>,
    ) {
        #[cfg(test)]
        let hook = self.on_step.clone();
        let on_step = move |step: usize| {
            #[cfg(test)]
            if let Some(hook) = &hook {
                hook(step);
            }
            let _ = step;
        };
        #[cfg(test)]
        let before_chunk = self.on_prefill.clone();
        let left = move || {
            #[cfg(test)]
            if let Some(hook) = &before_chunk {
                hook();
            }
            tx.is_closed()
        };
        let cancel = PrefillCancel::when(&left);
        let Self {
            executor,
            prompt,
            config,
            media,
            idle,
            ..
        } = self;
        let stream = match &media {
            DecodeMedia::Text => executor
                .generate_cancellable(&prompt, &config, cancel)
                .boxed_local(),
            DecodeMedia::Multimodal { images, audio } => executor
                .generate_multimodal_cancellable(&prompt, images, audio, &config, cancel)
                .boxed_local(),
            DecodeMedia::Images { images } => executor
                .generate_with_images_cancellable(&prompt, images, &config, cancel)
                .boxed_local(),
        };
        forward_tokens(stream, out, tx, idle, on_step).await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use anyhow::Result;

    use super::*;
    use crate::engine::blocking_stream::{stream_on_decode_thread, DECODE_CHANNEL_CAPACITY};
    use crate::engine::decode_stream::decode_output::RawTokens;
    use crate::engine::requests::batch_test_rig::{greedy, POOL, SEED};
    use crate::engine::test_support::{free_pool_blocks, seeded_executor, WAIT_LIMIT};

    /// Chunks of two over a nine-token prompt. The receiver drops before
    /// the second chunk, so one chunk runs, nothing is sent, and the KV
    /// lease returns its blocks.
    #[tokio::test]
    async fn a_receiver_dropped_mid_prefill_stops_it_at_the_next_chunk() -> Result<()> {
        let mut exec = seeded_executor(SEED, Some(POOL))?;
        exec.config.inference.prefill_chunk_size = 2;
        let exec = Arc::new(exec);
        let (tx, rx) = mpsc::channel(DECODE_CHANNEL_CAPACITY);
        let reads = Arc::new(AtomicUsize::new(0));
        let hook: PrefillHook = {
            let (reads, held) = (Arc::clone(&reads), Mutex::new(Some(rx)));
            Arc::new(move || {
                if reads.fetch_add(1, Ordering::SeqCst) == 1 {
                    if let Ok(mut held) = held.lock() {
                        drop(held.take());
                    }
                }
            })
        };
        let steps = Arc::new(AtomicUsize::new(0));
        let on_step: StepHook = {
            let steps = Arc::clone(&steps);
            Arc::new(move |_| {
                steps.fetch_add(1, Ordering::SeqCst);
            })
        };
        let job = DecodeJob::text(Arc::clone(&exec), (1..=9).collect(), greedy(4))
            .with_prefill_hook(hook)
            .with_step_hook(on_step);
        tokio::time::timeout(WAIT_LIMIT, stream_on_decode_thread(job, RawTokens, tx)).await?;

        assert_eq!(reads.load(Ordering::SeqCst), 2, "one of five chunks ran");
        assert_eq!(steps.load(Ordering::SeqCst), 0, "no token was produced");
        assert_eq!(free_pool_blocks(&exec)?, POOL.blocks);
        Ok(())
    }
}
