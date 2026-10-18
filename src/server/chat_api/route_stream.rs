//! Streaming generation on the path a request's media picks.

use tokio::sync::mpsc;

use super::chat_vision::{ImageRoute, RequestMedia};
use crate::config::GenerationConfig;
use crate::server::generation::{
    stream_images_with_stop_sequences, stream_multimodal_with_stop_sequences,
};
use crate::server::handlers::Route;
use crate::server::scheduled_stream::stream_routed;
use crate::server::streaming::StreamToken;

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
