//! Streaming generation with stop sequence checking.
//!
//! Each `stream_*_with_stop_sequences` function runs one executor stream and
//! forwards its tokens to the handler's channel. The forwarder cuts the text at
//! the first stop sequence and records the streaming metrics.

use std::sync::Arc;

use futures::StreamExt;

use super::stream_metrics::StreamMetrics;
use super::streaming::StreamToken;
use crate::config::GenerationConfig;
use crate::engine::{FinishReason, StopMatcher};
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
    let stream = executor.generate(&prompt_tokens, &gen_config);
    forward_with_stop_sequences(stream, prompt_tokens.len(), &gen_config, "Generation", tx).await;
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
        prompt_tokens.len(),
        &gen_config,
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
    forward_with_stop_sequences(
        stream,
        prompt_tokens.len(),
        &gen_config,
        "Image generation",
        tx,
    )
    .await;
}

/// Forward `stream` to `tx`, cutting the text at the first stop sequence.
/// `label` names the path in the error log.
///
/// With `gen_config.logprobs` set, each item carries its engine token's
/// logprobs. A token the matcher holds back then sends an empty-text item,
/// so its logprob goes out when it arrives. Without logprobs it sends nothing.
///
/// A stream that ends with no finish token closes with `Length`, as the
/// non-streaming collector reads it.
async fn forward_with_stop_sequences<S>(
    stream: S,
    prompt_len: usize,
    gen_config: &GenerationConfig,
    label: &str,
    tx: tokio::sync::mpsc::Sender<StreamToken>,
) where
    S: futures::Stream<Item = anyhow::Result<crate::engine::GeneratedToken>>,
{
    let mut stream = std::pin::pin!(stream);
    let mut matcher = StopMatcher::new(&gen_config.stop_sequences);
    let mut ended = false;
    let mut metrics = StreamMetrics::start(prompt_len);

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                metrics.token();

                // A match ends with `Stop`. The last token carries the
                // held-back text, which is not a stop.
                let cut = matcher.cut_token(&token.text, token.finish_reason);
                let held = cut.text.is_empty() && !token.text.is_empty();
                let item = StreamToken::text(cut.text, cut.finish)
                    .with_logprobs(&token, gen_config.logprobs);
                if cut.finish.is_some() {
                    let _ = tx.send(item).await;
                    ended = true;
                    break;
                }

                if held && item.logprob.is_none() {
                    continue;
                }
                if tx.send(item).await.is_err() {
                    ended = true;
                    break;
                }
            }
            Err(e) => {
                tracing::error!("{} error during streaming: {}", label, e);
                let _ = tx.send(StreamToken::error(e.to_string())).await;
                return;
            }
        }
    }
    if !ended {
        // The stream ended without a final token: flush the held-back text.
        let tail = matcher.finish();
        let _ = tx
            .send(StreamToken::text(tail, Some(FinishReason::Length)))
            .await;
    }
    metrics.finish();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::FinishReason;

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

    /// Run the forwarder over `tokens` and collect every item it sends.
    async fn forward_items(
        tokens: Vec<anyhow::Result<crate::engine::GeneratedToken>>,
        stops: &[&str],
        logprobs: bool,
    ) -> Vec<StreamToken> {
        let gen_config = GenerationConfig {
            stop_sequences: stops.iter().map(|s| s.to_string()).collect(),
            logprobs,
            ..GenerationConfig::default()
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let stream = futures::stream::iter(tokens);
        forward_with_stop_sequences(stream, 3, &gen_config, "Test", tx).await;
        let mut out = Vec::new();
        while let Some(t) = rx.recv().await {
            out.push(t);
        }
        out
    }

    /// Run the forwarder over `tokens` and collect what the client receives.
    async fn forward(
        tokens: Vec<anyhow::Result<crate::engine::GeneratedToken>>,
        stops: &[&str],
    ) -> Vec<(String, Option<FinishReason>)> {
        forward_items(tokens, stops, false)
            .await
            .into_iter()
            .map(|t| (t.text, t.finish_reason))
            .collect()
    }

    fn with_logprob(id: u32, text: &str) -> anyhow::Result<crate::engine::GeneratedToken> {
        let mut token = tok(text, None)?;
        token.token_id = id;
        token.logprob = Some(-(id as f32));
        Ok(token)
    }

    /// `E` and `N` start the stop `END`, so the matcher holds their text.
    /// Their items still go out, empty, each with its own logprob.
    #[tokio::test]
    async fn held_back_tokens_send_their_logprobs_when_asked() {
        let tokens = || {
            vec![
                with_logprob(1, "a"),
                with_logprob(2, "E"),
                with_logprob(3, "N"),
                with_logprob(4, "x"),
            ]
        };
        let out = forward_items(tokens(), &["END"], true).await;
        let got: Vec<(String, Option<u32>, Option<f32>)> = out
            .iter()
            .map(|t| {
                let lp = t.logprob.as_ref();
                (
                    t.text.clone(),
                    lp.map(|l| l.token_id),
                    lp.map(|l| l.logprob),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                ("a".to_string(), Some(1), Some(-1.0)),
                (String::new(), Some(2), Some(-2.0)),
                (String::new(), Some(3), Some(-3.0)),
                ("ENx".to_string(), Some(4), Some(-4.0)),
                (String::new(), None, None),
            ]
        );
        assert_eq!(
            out.last().and_then(|t| t.finish_reason),
            Some(FinishReason::Length)
        );

        let plain = forward_items(tokens(), &["END"], false).await;
        assert!(plain
            .iter()
            .all(|t| t.logprob.is_none() && t.top_logprobs.is_none()));
        let texts: Vec<&str> = plain.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            texts,
            ["a", "ENx", ""],
            "held tokens send nothing without logprobs"
        );
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
        assert_eq!(out.last().and_then(|(_, r)| *r), Some(FinishReason::Length));
    }

    type CpuExecutor = crate::engine::Executor<boostr::CpuRuntime>;

    /// What the client receives when `exec` streams `prompt` through the forwarder.
    async fn forward_engine(
        exec: &CpuExecutor,
        prompt: &[u32],
        cfg: &GenerationConfig,
    ) -> Vec<(String, Option<FinishReason>)> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        forward_with_stop_sequences(exec.generate(prompt, cfg), prompt.len(), cfg, "Test", tx)
            .await;
        let mut out = Vec::new();
        while let Some(t) = rx.recv().await {
            out.push((t.text, t.finish_reason));
        }
        out
    }

    #[tokio::test]
    async fn speculative_streams_match_plain_streams() -> anyhow::Result<()> {
        use crate::config::SpeculativeDecodingConfig;
        use crate::engine::batch_test_rig::greedy;
        use crate::engine::test_support::seeded_executor_ctx;

        let plain = seeded_executor_ctx(0x5bec, 64)?;
        let mut spec = seeded_executor_ctx(0x5bec, 64)?;
        spec.config.inference.speculative = Some(SpeculativeDecodingConfig {
            enabled: true,
            num_draft_tokens: Some(5),
            ..SpeculativeDecodingConfig::default()
        });
        let probe = GenerationConfig {
            logprobs: true,
            top_logprobs: 2,
            ..greedy(16)
        };
        let prompts: [&[u32]; 3] = [
            &[3, 7, 11, 3, 7, 11, 3, 7, 11],
            &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
            &[4, 4, 4, 4, 4, 4],
        ];
        let mut compared = 0;
        for prompt in prompts {
            // Skip a prompt whose greedy run has a top-2 tie the two paths can break apart.
            let result = plain.generate_text(prompt, &probe).await?;
            let tokens = result.token_logprobs.unwrap_or_default();
            let tied = tokens.iter().any(|t| match t.top_logprobs.as_deref() {
                Some([a, b, ..]) => (a.logprob - b.logprob).abs() < 1e-4,
                _ => true,
            });
            if tied {
                continue;
            }
            // Every id under 32 is one ASCII character, so a byte range is a token range.
            let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
            if text.len() != tokens.len() || !text.is_ascii() {
                return Err(anyhow::anyhow!(
                    "{text:?} is not one ASCII character per token"
                ));
            }
            let stop = GenerationConfig {
                stop_sequences: vec![text[3..5].to_string()],
                ..greedy(16)
            };
            for cfg in [greedy(16), stop] {
                let want = forward_engine(&plain, prompt, &cfg).await;
                assert_eq!(
                    forward_engine(&spec, prompt, &cfg).await,
                    want,
                    "{prompt:?}"
                );
            }
            compared += 1;
        }
        assert!(compared > 0, "every prompt has a near-tie");
        Ok(())
    }
}
