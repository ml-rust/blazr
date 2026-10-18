//! Streaming generation with stop sequence checking.
//!
//! Each `stream_*_with_stop_sequences` function runs one executor stream and
//! forwards its tokens to the handler's channel. The forwarder cuts the text at
//! the first stop sequence and records the streaming metrics.

use std::sync::Arc;

use futures::StreamExt;

use super::streaming::StreamToken;
use crate::config::GenerationConfig;
use crate::engine::{FinishReason, StopMatcher};
use crate::runtime::ServerRuntime;
use crate::server::metrics;

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
    let stream = executor.generate(&prompt_tokens, &gen_config);
    forward_with_stop_sequences(stream, &gen_config.stop_sequences, "Generation", tx).await;
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
    let stream =
        executor.generate_multimodal(&prompt_tokens, &images, &audio_segments, &gen_config);
    forward_with_stop_sequences(
        stream,
        &gen_config.stop_sequences,
        "Multimodal generation",
        tx,
    )
    .await;
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
    let stream = executor.generate_with_images(&prompt_tokens, &images, &gen_config);
    forward_with_stop_sequences(stream, &gen_config.stop_sequences, "Image generation", tx).await;
}

/// Forward `stream` to `tx`, cutting the text at the first stop sequence.
/// `label` names the path in the error log.
async fn forward_with_stop_sequences<S>(
    stream: S,
    stop_sequences: &[String],
    label: &str,
    tx: tokio::sync::mpsc::Sender<StreamToken>,
) where
    S: futures::Stream<Item = anyhow::Result<crate::engine::GeneratedToken>>,
{
    let mut stream = std::pin::pin!(stream);
    let mut matcher = StopMatcher::new(stop_sequences);
    let mut ended = false;
    let start = std::time::Instant::now();
    let mut last_token_time = start;
    let mut token_count = 0usize;

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                token_count += 1;
                // The first token closes the TTFT. Each later one is an ITL sample.
                if token_count == 1 {
                    metrics::record_ttft(start.elapsed().as_secs_f64());
                } else {
                    metrics::record_itl(last_token_time.elapsed().as_secs_f64());
                }
                last_token_time = std::time::Instant::now();

                let step = matcher.push(&token.text);
                if step.stopped {
                    let _ = tx
                        .send(StreamToken {
                            text: step.emit,
                            finish_reason: Some(FinishReason::Stop),
                            error: None,
                        })
                        .await;
                    ended = true;
                    break;
                }

                if let Some(reason) = token.finish_reason {
                    // The last token ends the text: held-back text is not a stop.
                    let mut text = step.emit;
                    text.push_str(&matcher.finish());
                    let _ = tx
                        .send(StreamToken {
                            text,
                            finish_reason: Some(reason),
                            error: None,
                        })
                        .await;
                    ended = true;
                    break;
                }

                // A token held back by the matcher sends nothing.
                if step.emit.is_empty() && !token.text.is_empty() {
                    continue;
                }
                if tx
                    .send(StreamToken {
                        text: step.emit,
                        finish_reason: None,
                        error: None,
                    })
                    .await
                    .is_err()
                {
                    ended = true;
                    break;
                }
            }
            Err(e) => {
                tracing::error!("{} error during streaming: {}", label, e);
                let _ = tx
                    .send(StreamToken {
                        text: String::new(),
                        finish_reason: None,
                        error: Some(e.to_string()),
                    })
                    .await;
                return;
            }
        }
    }
    if !ended {
        // The stream ended without a final token: flush the held-back text.
        let tail = matcher.finish();
        if !tail.is_empty() {
            let _ = tx
                .send(StreamToken {
                    text: tail,
                    finish_reason: None,
                    error: None,
                })
                .await;
        }
    }
    if token_count > 0 {
        metrics::record_e2e(start.elapsed().as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(
        text: &str,
        finish_reason: Option<FinishReason>,
    ) -> anyhow::Result<crate::engine::GeneratedToken> {
        Ok(crate::engine::GeneratedToken {
            token_id: 0,
            text: text.to_string(),
            logprob: None,
            top_logprobs: None,
            finish_reason,
        })
    }

    /// Run the forwarder over `tokens` and collect what the client receives.
    async fn forward(
        tokens: Vec<anyhow::Result<crate::engine::GeneratedToken>>,
        stops: &[&str],
    ) -> Vec<(String, Option<FinishReason>)> {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        forward_with_stop_sequences(futures::stream::iter(tokens), &stops, "Test", tx).await;
        let mut out = Vec::new();
        while let Some(t) = rx.recv().await {
            out.push((t.text, t.finish_reason));
        }
        out
    }

    #[tokio::test]
    async fn split_stop_never_reaches_the_client() {
        let out = forward(
            vec![tok("hi E", None), tok("N", None), tok("D!", None)],
            &["END"],
        )
        .await;
        let text: String = out.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(text, "hi ");
        assert_eq!(out.last().and_then(|(_, r)| *r), Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn no_stops_forward_every_token_unchanged() {
        let out = forward(
            vec![
                tok("a", None),
                tok("", None),
                tok("b", Some(FinishReason::Eos)),
            ],
            &[],
        )
        .await;
        assert_eq!(
            out,
            vec![
                ("a".to_string(), None),
                (String::new(), None),
                ("b".to_string(), Some(FinishReason::Eos)),
            ]
        );
    }

    #[tokio::test]
    async fn final_token_flushes_held_back_text() {
        let out = forward(
            vec![tok("x EN", None), tok("", Some(FinishReason::Length))],
            &["END"],
        )
        .await;
        assert_eq!(
            out,
            vec![
                ("x ".to_string(), None),
                ("EN".to_string(), Some(FinishReason::Length)),
            ]
        );
    }

    #[tokio::test]
    async fn stream_end_without_final_token_flushes_held_back_text() {
        let out = forward(vec![tok("x EN", None)], &["END"]).await;
        let text: String = out.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(text, "x EN");
    }
}
