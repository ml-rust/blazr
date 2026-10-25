//! Streaming text generation on a request's route.
//!
//! A route with a batching scheduler submits the prompt there and forwards
//! each token as it arrives. A route without one streams from its executor.

use std::sync::Arc;

use tokio::sync::mpsc;

use super::scheduled::batching_scheduler;
use super::scheduled_stream::stream_via_scheduler;
use crate::config::GenerationConfig;
use crate::server::generation::stream_with_stop_sequences;
use crate::server::handlers::Route;
use crate::server::streaming::StreamToken;

/// Stream `prompt_tokens` on `route` to `tx`.
///
/// The route picks the path by `batching_scheduler`, as `generate_text_routed`
/// does, so a streamed and a non-streamed request take the same one.
pub async fn stream_routed(
    route: &Route,
    prompt_tokens: Vec<u32>,
    gen_config: GenerationConfig,
    tx: mpsc::Sender<StreamToken>,
) {
    match batching_scheduler(route) {
        Some(scheduler) => stream_via_scheduler(scheduler, prompt_tokens, gen_config, tx).await,
        None => {
            let executor = Arc::clone(&route.executor);
            stream_with_stop_sequences(executor, prompt_tokens, gen_config, tx).await
        }
    }
}
