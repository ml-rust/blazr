//! Generation on the path a request's media picks: one result, or a stream.

use std::future::Future;

use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use super::chat_vision::{ImageRoute, RequestMedia};
use crate::config::GenerationConfig;
use crate::engine::{GenerationResult, WorkHold};
use crate::server::generation::{
    stream_images_with_stop_sequences, stream_multimodal_with_stop_sequences,
};
use crate::server::handlers::{BudgetPermit, DecodeSlot, Route};
use crate::server::routed_stream::stream_routed;
use crate::server::scheduled::generate_text_routed;
use crate::server::streaming::{StreamToken, TokenStream};

/// Items one choice's channel buffers.
const CHOICE_CHANNEL_CAPACITY: usize = 32;

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
) -> TokenStream
where
    F: Future<Output = ()> + Send + 'static,
{
    let configs = vec![gen_config];
    let mut streams = spawn_media_streams(
        permit,
        placement,
        route,
        prompt_tokens,
        media,
        configs,
        after,
    );
    streams
        .pop()
        .unwrap_or_else(|| Box::pin(futures::stream::empty()))
}

/// Spawn one stream of `prompt_tokens` per entry of `configs` and return
/// their receiving ends, in order.
///
/// One task runs the generations one after another, as the non-streaming
/// `n` path does, so the request never decodes more than one sequence at a
/// time under its one token reservation. The first generation starts at
/// once. Each later one starts when its stream is first polled. A reader
/// that drops a stream unpolled, on an error or a disconnect, starts no
/// later generation.
///
/// The task holds `permit` and a [`DecodeSlot`] until the last stream
/// ends. It then runs `after`, still holding both.
pub(in crate::server) fn spawn_media_streams<F>(
    permit: BudgetPermit,
    placement: Route,
    route: ImageRoute,
    mut prompt_tokens: Vec<u32>,
    mut media: RequestMedia,
    configs: Vec<GenerationConfig>,
    after: F,
) -> Vec<TokenStream>
where
    F: Future<Output = ()> + Send + 'static,
{
    let mut streams: Vec<TokenStream> = Vec::with_capacity(configs.len());
    let mut jobs = Vec::with_capacity(configs.len());
    for (i, config) in configs.into_iter().enumerate() {
        let (tx, rx) = mpsc::channel::<StreamToken>(CHOICE_CHANNEL_CAPACITY);
        if i == 0 {
            streams.push(Box::pin(ReceiverStream::new(rx)));
            jobs.push((config, tx, None));
        } else {
            let (start_tx, start_rx) = oneshot::channel();
            streams.push(Box::pin(start_on_first_poll(start_tx, rx)));
            jobs.push((config, tx, Some(start_rx)));
        }
    }
    let slot = DecodeSlot::enter();
    // The task carries the request's slot, so a client that leaves frees it
    // only once the decode stops.
    tokio::spawn(WorkHold::carry(async move {
        let _guards = (permit, slot);
        let count = jobs.len();
        for (i, (config, tx, start)) in jobs.into_iter().enumerate() {
            if let Some(start) = start {
                if start.await.is_err() {
                    break;
                }
            }
            let last = i + 1 == count;
            let (prompt, media) = if last {
                (
                    std::mem::take(&mut prompt_tokens),
                    std::mem::take(&mut media),
                )
            } else {
                (prompt_tokens.clone(), media.clone())
            };
            stream_for_media(&placement, route, prompt, media, config, tx).await;
        }
        after.await;
    }));
    streams
}

/// The items of `rx`. The first poll sends on `start`.
fn start_on_first_poll(
    start: oneshot::Sender<()>,
    mut rx: mpsc::Receiver<StreamToken>,
) -> impl futures::Stream<Item = StreamToken> + Send {
    let mut start = Some(start);
    futures::stream::poll_fn(move |cx| {
        if let Some(start) = start.take() {
            let _ = start.send(());
        }
        rx.poll_recv(cx)
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::Poll;

    use futures::StreamExt;

    #[tokio::test]
    async fn a_gated_stream_starts_its_producer_on_its_first_poll() {
        let (start_tx, mut start_rx) = oneshot::channel();
        let (tx, rx) = mpsc::channel(1);
        let mut stream = Box::pin(start_on_first_poll(start_tx, rx));
        assert!(start_rx.try_recv().is_err(), "nothing polled yet");

        let first = futures::poll!(stream.next());
        assert!(matches!(first, Poll::Pending));
        assert!(start_rx.try_recv().is_ok(), "the first poll starts it");
        drop(tx);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn a_gated_stream_dropped_unpolled_starts_nothing() {
        let (start_tx, start_rx) = oneshot::channel();
        let (_tx, rx) = mpsc::channel(1);
        drop(start_on_first_poll(start_tx, rx));
        assert!(start_rx.await.is_err());
    }
}
