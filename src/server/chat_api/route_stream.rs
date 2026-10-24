//! Generation on the path a request's media picks: one result, or a stream.

use std::future::Future;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::chat_vision::{ImageRoute, RequestMedia};
use crate::config::GenerationConfig;
use crate::engine::GenerationResult;
use crate::server::generation::{
    stream_images_with_stop_sequences, stream_multimodal_with_stop_sequences,
};
use crate::server::handlers::{BudgetPermit, DecodeSlot, Route};
use crate::server::scheduled::generate_text_routed;
use crate::server::scheduled_stream::stream_routed;
use crate::server::streaming::StreamToken;

/// Generate one result for `prompt_tokens` on the generator `route` names.
///
/// The vision tower and multimodal paths run on `placement`'s executor.
/// A text request takes `generate_text_routed`, so it can use the batching scheduler.
pub(in crate::server) async fn generate_for_media(
    placement: &Route,
    route: ImageRoute,
    prompt_tokens: &[u32],
    media: &RequestMedia,
    gen_config: &GenerationConfig,
) -> anyhow::Result<GenerationResult> {
    let executor = &placement.executor;
    match route {
        ImageRoute::Tower => {
            executor
                .generate_images_text(prompt_tokens, &media.images, gen_config)
                .await
        }
        ImageRoute::Multimodal => {
            executor
                .generate_multimodal_text(prompt_tokens, &media.images, &media.audio, gen_config)
                .await
        }
        ImageRoute::Text => generate_text_routed(placement, prompt_tokens, gen_config).await,
    }
}

/// Spawn the stream of `prompt_tokens` and return its receiving end.
///
/// The task holds `permit` and a [`DecodeSlot`] until the stream ends.
/// It then runs `after`, still holding both.
pub(in crate::server) fn spawn_media_stream<F>(
    permit: BudgetPermit,
    placement: Route,
    route: ImageRoute,
    prompt_tokens: Vec<u32>,
    media: RequestMedia,
    gen_config: GenerationConfig,
    after: F,
) -> ReceiverStream<StreamToken>
where
    F: Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<StreamToken>(32);
    let slot = DecodeSlot::enter();
    tokio::spawn(async move {
        let _guards = (permit, slot);
        stream_for_media(&placement, route, prompt_tokens, media, gen_config, tx).await;
        after.await;
    });
    ReceiverStream::new(rx)
}

/// Stream `prompt_tokens` to `tx` on the generator `route` names.
///
/// The vision tower and multimodal paths run on `placement`'s executor.
/// A text request takes `stream_routed`, so it can use the batching scheduler.
pub(in crate::server) async fn stream_for_media(
    placement: &Route,
    route: ImageRoute,
    prompt_tokens: Vec<u32>,
    media: RequestMedia,
    gen_config: GenerationConfig,
    tx: mpsc::Sender<StreamToken>,
) {
    let executor = std::sync::Arc::clone(&placement.executor);
    match route {
        ImageRoute::Tower => {
            stream_images_with_stop_sequences(executor, prompt_tokens, media.images, gen_config, tx)
                .await
        }
        ImageRoute::Multimodal => {
            stream_multimodal_with_stop_sequences(
                executor,
                prompt_tokens,
                media.images,
                media.audio,
                gen_config,
                tx,
            )
            .await
        }
        ImageRoute::Text => stream_routed(placement, prompt_tokens, gen_config, tx).await,
    }
}
