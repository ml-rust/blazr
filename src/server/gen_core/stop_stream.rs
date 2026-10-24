//! Streaming generation with stop sequence checking.
//!
//! Each `stream_*_with_stop_sequences` function runs one executor stream on
//! a decode thread and forwards its tokens to the handler's channel. The
//! `StopForwarder` cuts the text at the first stop sequence and records
//! the streaming metrics, on that thread. Each function returns once the
//! thread exits, so the permit the calling task holds outlives the decode.

use std::sync::Arc;

use super::stop_forwarder::StopForwarder;
use super::streaming::StreamToken;
use crate::config::GenerationConfig;
use crate::engine::blocking_stream::{stream_on_decode_thread, DecodeJob};
use crate::runtime::ServerRuntime;

/// Stream generation tokens with stop sequence checking.
///
/// Takes the prompt ids the handler already validated — stop sequences are
/// matched against the *generated* text, so nothing here needs the prompt text.
pub async fn stream_with_stop_sequences(
    executor: Arc<crate::engine::Executor<ServerRuntime>>,
    prompt_tokens: Vec<u32>,
    gen_config: GenerationConfig,
    tx: tokio::sync::mpsc::Sender<StreamToken>,
) {
    let out = StopForwarder::new(prompt_tokens.len(), &gen_config, "Generation");
    let job = DecodeJob::text(executor, prompt_tokens, gen_config);
    stream_on_decode_thread(job, out, tx).await;
}

/// Same as `stream_with_stop_sequences` over `generate_multimodal`, with the
/// decoded images and audio segments of the llava-style path.
pub async fn stream_multimodal_with_stop_sequences(
    executor: Arc<crate::engine::Executor<ServerRuntime>>,
    prompt_tokens: Vec<u32>,
    images: Vec<Vec<u8>>,
    audio_segments: Vec<Vec<f32>>,
    gen_config: GenerationConfig,
    tx: tokio::sync::mpsc::Sender<StreamToken>,
) {
    let out = StopForwarder::new(prompt_tokens.len(), &gen_config, "Multimodal generation");
    let job = DecodeJob::multimodal(executor, prompt_tokens, images, audio_segments, gen_config);
    stream_on_decode_thread(job, out, tx).await;
}

/// Same as `stream_with_stop_sequences` over `generate_with_images`: the
/// prompt ids hold one image placeholder per entry of `images`.
pub async fn stream_images_with_stop_sequences(
    executor: Arc<crate::engine::Executor<ServerRuntime>>,
    prompt_tokens: Vec<u32>,
    images: Vec<Vec<u8>>,
    gen_config: GenerationConfig,
    tx: tokio::sync::mpsc::Sender<StreamToken>,
) {
    let out = StopForwarder::new(prompt_tokens.len(), &gen_config, "Image generation");
    let job = DecodeJob::images(executor, prompt_tokens, images, gen_config);
    stream_on_decode_thread(job, out, tx).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::batch_test_rig::greedy;
    use crate::engine::test_support::seeded_executor;

    /// A panic on the decode thread reaches the client as an error item,
    /// after the text that went out before it.
    #[tokio::test]
    async fn a_decode_panic_reaches_the_client_as_an_error() -> anyhow::Result<()> {
        let exec = Arc::new(seeded_executor(1, None)?);
        let cfg = greedy(4);
        let job =
            DecodeJob::text(exec, vec![1, 2, 3], cfg.clone()).with_step_hook(Arc::new(|step| {
                if step == 1 {
                    panic!("injected decode panic");
                }
            }));
        let (tx, mut rx) = tokio::sync::mpsc::channel(32);
        stream_on_decode_thread(job, StopForwarder::new(3, &cfg, "Test"), tx).await;
        let mut items = Vec::new();
        while let Some(item) = rx.recv().await {
            items.push(item);
        }
        assert_eq!(items.len(), 2, "one token, then the error");
        assert!(items[0].error.is_none());
        let error = items[1].error.as_deref().unwrap_or_default();
        assert!(error.contains("injected decode panic"), "{error:?}");
        Ok(())
    }
}
